"""Unmodified sparse primitives from Flow-JEPA f5763aa, FlowTrajectoryBlock.

Only the import paths and enclosing class name differ from the source.
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from .sparse_linear import selected_linear, selected_linear_grouped


class SourceSparseOps:
    @staticmethod
    def _active_attention(q, k, v, heads, scale, dropout, training, mask):
        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=heads)
            for t in (q, k, v)
        )
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if mask is not None:
            scores = scores.masked_fill(
                ~mask.bool()[None, None], -torch.finfo(scores.dtype).max
            )
        weights = F.dropout(scores.softmax(dim=-1), p=dropout, training=training)
        return rearrange(torch.matmul(weights, v), "b h n d -> b n (h d)")

    @staticmethod
    def _batched_linear(inputs, bank, selected_groups, use_triton=False,
                        grouped_order=None):
        if use_triton:
            if grouped_order is not None:
                return selected_linear_grouped(inputs, bank, *grouped_order)
            return selected_linear(inputs, bank, selected_groups)
        weights, biases = bank
        weights = weights.index_select(0, selected_groups)
        output = torch.bmm(inputs, weights.transpose(1, 2))
        if biases is not None:
            output = output + biases.index_select(0, selected_groups)[:, None, :].to(output.dtype)
        return output
