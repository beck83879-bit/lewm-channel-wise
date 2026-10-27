"""Selected-coordinate LeWM prediction with causal, prefix-local routing."""

import torch
from torch.nn import functional as F

from module import ARPredictor
from .channel_routing import ChannelRouting
from .source_ops import SourceSparseOps


def selected_bank(linear, active_indices, *, rows=False, cols=False, modulation=0):
    """Map LeWM's parameter names and six-way AdaLN to the source bank layout."""
    weights, biases = [], []
    for indices in active_indices:
        selected_rows = (torch.cat([indices + i * linear.in_features
                                    for i in range(modulation)]) if modulation
                         else indices if rows else None)
        weight = linear.weight
        if selected_rows is not None:
            weight = weight.index_select(0, selected_rows)
        if cols:
            weight = weight.index_select(1, indices)
        weights.append(weight)
        if linear.bias is not None:
            biases.append(linear.bias if selected_rows is None
                          else linear.bias.index_select(0, selected_rows))
    return torch.stack(weights), torch.stack(biases) if biases else None


def batched_selected_norm(x, norm, indices):
    if norm.weight is None:
        return F.layer_norm(x, (indices.size(-1),), eps=norm.eps)
    weight = norm.weight[indices]
    bias = norm.bias[indices] if norm.bias is not None else None
    return torch.vmap(F.layer_norm, in_dims=(0, None, 0, 0 if bias is not None else None, None))(
        x, (indices.size(-1),), weight, bias, norm.eps)


def batched_active_block(block, x, condition, selected_groups, active_indices):
    """Source gather/batched-linear/scatter execution, with LeWM's two branches."""
    indices = active_indices.index_select(0, selected_groups)
    state_indices = indices[:, None, :].expand(-1, x.size(1), -1)
    h = x.gather(-1, state_indices)
    active_condition = condition.gather(-1, state_indices)
    sa = block.attn
    mlp = block.mlp.net
    banks = {
        "mod": selected_bank(block.adaLN_modulation[-1], active_indices,
                             cols=True, modulation=6),
        "sa_qkv": selected_bank(sa.to_qkv, active_indices, cols=True),
        "sa_out": selected_bank(sa.to_out[0], active_indices, rows=True),
        "mlp_up": selected_bank(mlp[1], active_indices, cols=True),
        "mlp_down": selected_bank(mlp[4], active_indices, rows=True),
    }
    linear = lambda tensor, name: SourceSparseOps._batched_linear(
        tensor, banks[name], selected_groups)
    mod = linear(F.silu(active_condition), "mod")
    shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
    sa_input = batched_selected_norm(h, block.norm1, indices) * (1 + scale_sa) + shift_sa
    sa_input = batched_selected_norm(sa_input, sa.norm, indices)
    q, k, v = linear(sa_input, "sa_qkv").chunk(3, dim=-1)
    mask = torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device).tril()
    sa_result = SourceSparseOps._active_attention(
        q, k, v, sa.heads, sa.scale, sa.dropout, block.training, mask)
    h = h + gate_sa * sa.to_out[1](linear(sa_result, "sa_out"))
    mlp_input = batched_selected_norm(h, block.norm2, indices) * (1 + scale_mlp) + shift_mlp
    mlp_input = batched_selected_norm(mlp_input, mlp[0], indices)
    intermediate = mlp[3](mlp[2](linear(mlp_input, "mlp_up")))
    updated = h + gate_mlp * mlp[5](linear(intermediate, "mlp_down"))
    return x.scatter(-1, state_indices, updated)


class ChannelwiseARPredictor(ARPredictor):
    """Reuse upstream parameters; choose one shared+dynamic subnetwork per prefix.

    Pooling all history for every prediction would reveal future observations to
    earlier targets. Each output is therefore computed on its own visible prefix.
    There is no flow-time variable in LeWM; that router input is identically zero.
    """

    def __init__(self, *, routing=None, **kwargs):
        super().__init__(**kwargs)
        config = dict(routing or {})
        self.routing_enabled = bool(config.pop("enabled", False))
        self.sparse_compute = bool(config.pop("sparse_compute", False))
        self.routing = None
        self._last_trace = []
        if self.routing_enabled:
            if not self.sparse_compute:
                raise ValueError("This port only implements selected-submatrix routing")
            if config.get("selected_weight_mode", "binary") != "binary":
                raise ValueError("Use the source experiment's binary selection weights")
            config["selected_weight_mode"] = "binary"
            with torch.random.fork_rng(devices=[]):
                self.routing = ChannelRouting(
                    kwargs["hidden_dim"], kwargs["depth"], **config,
                )
            r = self.routing
            if r.mode not in {"learned", "random"}:
                raise ValueError("Dense control must disable routing")
            if r.top_k != 1 or r.groups != 4 or r.shared_dim != 3 * r.group_size:
                raise ValueError("Require 3 shared groups and 1 selected of 4 dynamic groups")
            indices = [torch.cat((torch.arange(r.shared_dim), torch.arange(
                r.shared_dim + i * r.group_size, r.shared_dim + (i + 1) * r.group_size,
            ))) for i in range(r.groups)]
            self.register_buffer("active_indices", torch.stack(indices), persistent=False)

    def set_routing_step(self, step):
        if self.routing is not None:
            self.routing.set_step(step)

    def get_last_route_trace(self):
        return self._last_trace

    def _prefix(self, embeddings, actions, overrides, record):
        transformer = self.transformer
        length = embeddings.size(1)
        history = transformer.input_proj(embeddings)
        x = transformer.input_proj(self.dropout(embeddings + self.pos_embedding[:, :length]))
        condition = transformer.cond_proj(actions)
        summary = history.mean(1)
        delta = ((history[:, -1] - history[:, 0]) / (length - 1)
                 if length > 1 else torch.zeros_like(summary))
        action_summary = condition.mean(1)
        recording = self.routing.begin_forward(overrides, record)
        for layer_index, block in enumerate(transformer.layers):
            inputs = torch.cat((x.mean(1), summary, delta, action_summary,
                                torch.zeros_like(summary)), dim=-1)
            weights = self.routing.route(
                layer_index, inputs, record=recording,
                override=None if overrides is None else overrides[layer_index],
            )
            selected = weights[:, 0, self.routing.shared_dim:]
            selected = selected.reshape(x.size(0), self.routing.groups,
                                        self.routing.group_size).sum(-1).argmax(-1)
            x = batched_active_block(block, x, condition, selected, self.active_indices)
        indices = self.active_indices.index_select(0, selected)
        h = x[:, -1:].gather(-1, indices[:, None, :])
        h = batched_selected_norm(h, transformer.norm, indices)
        output = SourceSparseOps._batched_linear(
            h, selected_bank(transformer.output_proj, self.active_indices, cols=True), selected)
        return output, self.routing.get_last_route_trace() if recording else None

    def forward(self, x, c, *, routing_overrides=None, routing_record=None):
        if x.shape[:2] != c.shape[:2] or not 1 <= x.size(1) <= self.pos_embedding.size(1):
            raise ValueError("History and action lengths must match the configured history")
        if self.routing is None:
            return super().forward(x, c)
        outputs, traces = [], []
        batch = x.size(0)
        for position in range(x.size(1)):
            override = None
            if routing_overrides is not None:
                override = [{key: value[position * batch:(position + 1) * batch]
                             for key, value in layer.items()} for layer in routing_overrides]
            result, trace = self._prefix(x[:, :position + 1], c[:, :position + 1],
                                         override, routing_record)
            outputs.append(result)
            if trace is not None:
                traces.append(trace)
        if traces:
            self._last_trace = [{key: torch.cat([prefix[layer][key] for prefix in traces])
                                 for key in traces[0][layer]}
                                for layer in range(len(self.transformer.layers))]
        return torch.cat(outputs, dim=1)
