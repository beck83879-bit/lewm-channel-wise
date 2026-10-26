"""Prediction-conditioned hidden-dimension routing for Flow-JEPA.

The predictor may use routed residual writes or select submatrices before
computing each block's attention and MLP. One clip-level choice is shared by
the three branches within a block.
"""

import torch
import torch.distributed as dist
from torch import nn


class ChannelRouting(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        mode="learned",
        shared_dim=128,
        group_size=16,
        top_k=8,
        router_hidden_dim=128,
        warmup_steps=100,
        balance_ema=0.99,
        balance_min_factor=0.25,
        balance_max_factor=1.5,
        bias_lr=0.01,
        bias_clip=2.0,
        balance_enabled=True,
        selected_weight_mode="sigmoid",
    ):
        super().__init__()
        if mode not in {"learned", "random", "dense"}:
            raise ValueError(f"Unknown channel routing mode: {mode}")
        if selected_weight_mode not in {"sigmoid", "binary"}:
            raise ValueError("selected_weight_mode must be sigmoid or binary")
        if not 0 <= shared_dim < dim:
            raise ValueError("shared_dim must be in [0, hidden_dim)")
        if group_size < 1 or (dim - shared_dim) % group_size:
            raise ValueError("Dynamic dimensions must divide into full groups")
        groups = (dim - shared_dim) // group_size
        if not 1 <= top_k <= groups:
            raise ValueError(f"top_k must lie in [1, {groups}]")
        if not 0 <= balance_ema < 1:
            raise ValueError("balance_ema must lie in [0, 1)")
        if not 0 <= balance_min_factor <= balance_max_factor:
            raise ValueError("Invalid load bounds")
        if bias_lr < 0 or bias_clip < 0 or warmup_steps < 0:
            raise ValueError("Bias controls and warmup must be nonnegative")
        self.dim = int(dim)
        self.depth = int(depth)
        self.mode = mode
        self.shared_dim = int(shared_dim)
        self.group_size = int(group_size)
        self.groups = int(groups)
        self.top_k = int(top_k)
        self.warmup_steps = int(warmup_steps)
        self.balance_ema = float(balance_ema)
        self.balance_min_factor = float(balance_min_factor)
        self.balance_max_factor = float(balance_max_factor)
        self.bias_lr = float(bias_lr)
        self.bias_clip = float(bias_clip)
        self.balance_enabled = bool(balance_enabled)
        self.selected_weight_mode = selected_weight_mode
        self.routers = nn.ModuleList()
        if mode == "learned":
            for _ in range(depth):
                # Inputs: pooled noisy state, visible history, history change,
                # action condition, and flow time. No clean future target.
                router = nn.Sequential(
                    nn.LayerNorm(5 * dim),
                    nn.Linear(5 * dim, router_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(router_hidden_dim, groups),
                )
                nn.init.normal_(router[-1].weight, mean=0.0, std=0.01)
                nn.init.zeros_(router[-1].bias)
                self.routers.append(router)
        initial_load = 1.0 if mode == "dense" else top_k / groups
        self.register_buffer("usage_ema", torch.full((depth, groups), initial_load))
        self.register_buffer("selection_bias", torch.zeros(depth, groups))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        # Plain transient tensors deliberately avoid DDP's buffer broadcast:
        # accumulated microbatch counts must stay rank-local until after_step.
        # Allocate on the active device when the first training batch arrives.
        self.pending_counts = None
        self.pending_samples = None
        self._step_value = 0
        self._last_trace = []

    @property
    def load_ema(self):
        return self.usage_ema

    def set_step(self, step):
        self._step_value = int(step)
        self.step.fill_(self._step_value)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._step_value = int(self.step.item())
        self.pending_counts = None
        self.pending_samples = None
        self._last_trace = []

    def begin_forward(self, overrides=None, record=None):
        """Only the ordinary differentiable training forward records load."""
        should_record = (
            self.training and torch.is_grad_enabled()
            and overrides is None and record is not False
        )
        if overrides is not None and len(overrides) != self.depth:
            raise ValueError("A routing replay must freeze every block")
        if should_record:
            self._last_trace = []
        return should_record

    def get_last_route_trace(self):
        """Detached records; later probes do not mutate the returned records."""
        return self._last_trace

    def rescore(self, block, inputs):
        """Credit supervision trains the router without changing its inputs."""
        if self.mode != "learned":
            raise RuntimeError("Only learned routing has differentiable scores")
        return self.routers[block](inputs.detach())

    def weights_from_scores(self, scores, mask):
        if self.selected_weight_mode == "binary":
            return mask.to(scores.dtype)
        if self.mode == "learned":
            return mask.to(scores.dtype) * (2.0 * scores.sigmoid())
        return mask.to(scores.dtype)

    def route(self, block, inputs, record=False, override=None):
        """Return B x 1 x D residual write coefficients for one block."""
        batch = inputs.size(0)
        if override is not None:
            mask = override["mask"]
            group_weights = override["weights"]
            if mask.shape != (batch, self.groups) or group_weights.shape != mask.shape:
                raise ValueError("Routing replay dimensions do not match this batch")
            # Preserve the captured autocast dtype. Casting to the pooled
            # input dtype could otherwise change all replayed residual writes.
            group_weights = group_weights.to(device=inputs.device)
        else:
            if self.mode == "learned":
                scores = self.routers[block](inputs)
            else:
                scores = inputs.new_zeros((batch, self.groups))
            if self.mode == "dense":
                mask = torch.ones_like(scores, dtype=torch.bool)
            else:
                warmup = self.training and self._step_value < self.warmup_steps
                if self.mode == "random" or warmup:
                    priorities = torch.rand_like(scores)
                else:
                    priorities = scores.float() + self.selection_bias[block]
                selected = priorities.topk(self.top_k, dim=-1).indices
                mask = torch.zeros_like(scores, dtype=torch.bool)
                mask.scatter_(1, selected, True)
            group_weights = self.weights_from_scores(scores, mask)
            if record:
                self._last_trace.append({
                    "inputs": inputs.detach(),
                    "scores": scores.detach(),
                    "mask": mask.detach(),
                    "weights": group_weights.detach(),
                })
                with torch.no_grad():
                    if self.pending_counts is None:
                        self.pending_counts = torch.zeros_like(self.usage_ema)
                        self.pending_samples = self.usage_ema.new_zeros(self.depth)
                    self.pending_counts[block].add_(mask.float().sum(0))
                    self.pending_samples[block].add_(batch)
        shared = group_weights.new_ones((batch, self.shared_dim))
        dynamic = group_weights.repeat_interleave(self.group_size, dim=-1)
        return torch.cat((shared, dynamic), dim=-1).unsqueeze(1)

    def make_swap(self, trace, block=None):
        """Sample a same-budget, per-clip selected/unselected group swap."""
        if self.mode == "dense" or self.top_k == self.groups:
            raise ValueError("A counterfactual swap requires inactive groups")
        if len(trace) != self.depth:
            raise ValueError("No complete training routing trace is available")
        if block is None:
            block = int(torch.randint(self.depth, ()).item())
        mask = trace[block]["mask"]
        removed = torch.multinomial(mask.float(), 1).squeeze(1)
        added = torch.multinomial((~mask).float(), 1).squeeze(1)
        return block, removed, added

    def make_overrides(self, trace, swap=None):
        """Freeze all routing weights; optionally replace one group per clip."""
        if len(trace) != self.depth:
            raise ValueError("A routing replay needs one record per block")
        overrides = [
            {"mask": item["mask"].detach().clone(),
             "weights": item["weights"].detach().clone()}
            for item in trace
        ]
        if swap is not None:
            block, removed, added = swap
            record = overrides[block]
            rows = torch.arange(record["mask"].size(0), device=record["mask"].device)
            record["mask"][rows, removed] = False
            record["mask"][rows, added] = True
            # Both kept and replacement weights use the original frozen
            # scores. No score/bias recomputation on altered intermediate h.
            record["weights"] = self.weights_from_scores(
                trace[block]["scores"], record["mask"]
            ).detach()
        return overrides

    @torch.no_grad()
    def after_step(self):
        """Commit all accumulated microbatches once per optimizer update.

        Every DDP rank calls this method in the same order. Probes and
        validation never add samples and never perform distributed collectives.
        """
        if not self.training:
            return {}
        if self.pending_counts is None:
            self.pending_counts = torch.zeros_like(self.usage_ema)
            self.pending_samples = self.usage_ema.new_zeros(self.depth)
        packed = torch.cat((self.pending_counts, self.pending_samples[:, None]), dim=1)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        counts, samples = packed[:, :-1], packed[:, -1]
        valid = samples > 0
        observed = counts / samples[:, None].clamp_min(1)
        updated = self.balance_ema * self.usage_ema + (1 - self.balance_ema) * observed
        self.usage_ema.copy_(torch.where(valid[:, None], updated, self.usage_ema))
        if self.mode == "learned" and self.balance_enabled:
            expected = self.top_k / self.groups
            lower = min(1.0, self.balance_min_factor * expected)
            upper = min(1.0, self.balance_max_factor * expected)
            change = (lower - self.usage_ema).clamp_min(0) - (self.usage_ema - upper).clamp_min(0)
            self.selection_bias.add_(self.bias_lr * change * valid[:, None])
            self.selection_bias.clamp_(-self.bias_clip, self.bias_clip)
        self.pending_counts.zero_()
        self.pending_samples.zero_()
        self.step.add_(1)
        self._step_value += 1
        probabilities = self.usage_ema / self.usage_ema.sum(-1, keepdim=True).clamp_min(1e-8)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
        entropy = entropy / torch.tensor(float(self.groups), device=entropy.device).log().clamp_min(1e-8)
        return {
            "usage_min": self.usage_ema.min(),
            "usage_max": self.usage_ema.max(),
            "usage_entropy": entropy.mean(),
            "bias_absmax": self.selection_bias.abs().max(),
        }
