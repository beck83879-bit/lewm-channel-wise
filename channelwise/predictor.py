"""Selected-coordinate LeWM prediction with causal, prefix-local routing."""

import torch
from torch import nn
from torch.nn import functional as F

from module import ARPredictor
from .channel_routing import ChannelRouting


def selected_norm(x, norm, indices):
    weight = norm.weight.index_select(0, indices) if norm.weight is not None else None
    bias = norm.bias.index_select(0, indices) if norm.bias is not None else None
    return F.layer_norm(x, (indices.numel(),), weight, bias, norm.eps)


def active_block(block, x, condition, indices, full_width):
    """Preserve LeWM's two branches, including their inner LayerNorms."""
    rows = torch.cat([indices + i * full_width for i in range(6)])
    modulation = block.adaLN_modulation[-1]
    mod = F.linear(
        F.silu(condition),
        modulation.weight.index_select(0, rows).index_select(1, indices),
        modulation.bias.index_select(0, rows),
    )
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mod.chunk(6, dim=-1)
    attention = block.attn
    h = selected_norm(x, block.norm1, indices) * (1 + scale_a) + shift_a
    h = selected_norm(h, attention.norm, indices)
    qkv = F.linear(h, attention.to_qkv.weight.index_select(1, indices)).chunk(3, -1)
    q, k, v = [value.reshape(*value.shape[:2], attention.heads, -1).transpose(1, 2)
               for value in qkv]
    h = F.scaled_dot_product_attention(
        q, k, v, dropout_p=attention.dropout if block.training else 0.0,
        is_causal=True,
    ).transpose(1, 2).flatten(2)
    if not isinstance(attention.to_out, nn.Sequential):
        raise ValueError("Selected-coordinate attention requires an output projection")
    projection, dropout = attention.to_out
    h = dropout(F.linear(h, projection.weight.index_select(0, indices),
                         projection.bias.index_select(0, indices)))
    x = x + gate_a * h
    h = selected_norm(x, block.norm2, indices) * (1 + scale_m) + shift_m
    norm, up, activation, dropout1, down, dropout2 = block.mlp.net
    h = selected_norm(h, norm, indices)
    h = dropout1(activation(F.linear(h, up.weight.index_select(1, indices), up.bias)))
    h = dropout2(F.linear(h, down.weight.index_select(0, indices),
                          down.bias.index_select(0, indices)))
    return x + gate_m * h


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
            updated = x.clone()
            for group in range(self.routing.groups):
                members = (selected == group).nonzero(as_tuple=True)[0]
                if not members.numel():
                    continue
                indices = self.active_indices[group]
                previous = x.index_select(0, members)
                active = previous.index_select(-1, indices)
                active_condition = condition.index_select(0, members).index_select(-1, indices)
                result = active_block(block, active, active_condition, indices, x.size(-1))
                written = previous.scatter(-1, indices[None, None].expand_as(result), result)
                updated = updated.index_copy(0, members, written)
            x = updated
        output = x.new_zeros((x.size(0), 1, transformer.output_proj.out_features))
        for group in range(self.routing.groups):
            members = (selected == group).nonzero(as_tuple=True)[0]
            if not members.numel():
                continue
            indices = self.active_indices[group]
            h = x.index_select(0, members)[:, -1:].index_select(-1, indices)
            h = selected_norm(h, transformer.norm, indices)
            h = F.linear(h, transformer.output_proj.weight.index_select(1, indices),
                         transformer.output_proj.bias)
            output = output.index_copy(0, members, h)
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
