"""Optional inference-only selected-group matrix multiplication.

Each clip reads exactly one group's [out, active] weight matrix. No
per-sample copy of candidate matrices or inactive-group multiplication.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _grouped_linear(X, W, Bias, Order, Offsets, Counts, Y,
                        N: tl.constexpr, K: tl.constexpr, M: tl.constexpr,
                        HAS_BIAS: tl.constexpr,
                        BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
                        BLOCK_K: tl.constexpr):
        group = tl.program_id(2)
        first = tl.program_id(0) * BLOCK_N
        count = tl.load(Counts + group)
        if first < count * N:
            row = first + tl.arange(0, BLOCK_N)
            valid = row < count * N
            offset = tl.load(Offsets + group)
            batch = tl.load(Order + offset + row // N, mask=valid, other=0)
            token = row % N
            m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
            k = tl.arange(0, BLOCK_K)
            accum = tl.full((BLOCK_N, BLOCK_M), 0, tl.float32)
            for tile in range(tl.cdiv(K, BLOCK_K)):
                kk = tile * BLOCK_K + k
                a = tl.load(X + batch[:, None] * N * K + token[:, None] * K
                            + kk[None, :], valid[:, None] & (kk[None, :] < K),
                            other=0).to(tl.bfloat16)
                w = tl.load(W + group * M * K + m[None, :] * K + kk[:, None],
                            (m[None, :] < M) & (kk[:, None] < K), other=0)
                accum += tl.dot(a, w)
            if HAS_BIAS:
                bias = tl.load(Bias + group * M + m, m < M, other=0)
                accum += bias[None, :]
            tl.store(Y + batch[:, None] * N * M + token[:, None] * M
                     + m[None, :], accum,
                     valid[:, None] & (m[None, :] < M))

    @triton.jit
    def _selected_linear(X, W, Bias, Groups, Y,
                         N: tl.constexpr, K: tl.constexpr, M: tl.constexpr,
                         HAS_BIAS: tl.constexpr,
                         BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
                         BLOCK_K: tl.constexpr):
        n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        batch = tl.program_id(2)
        group = tl.load(Groups + batch)
        k = tl.arange(0, BLOCK_K)
        accum = tl.full((BLOCK_N, BLOCK_M), 0, tl.float32)
        for tile in range(tl.cdiv(K, BLOCK_K)):
            kk = tile * BLOCK_K + k
            a = tl.load(X + batch * N * K + n[:, None] * K + kk[None, :],
                        (n[:, None] < N) & (kk[None, :] < K), other=0).to(tl.bfloat16)
            w = tl.load(W + group * M * K + m[None, :] * K + kk[:, None],
                        (m[None, :] < M) & (kk[:, None] < K), other=0)
            accum += tl.dot(a, w)
        if HAS_BIAS:
            bias = tl.load(Bias + group * M + m, m < M, other=0)
            accum += bias[None, :]
        tl.store(Y + batch * N * M + n[:, None] * M + m[None, :],
                 accum, (n[:, None] < N) & (m[None, :] < M))


def selected_linear(inputs, bank, groups):
    if triton is None:
        raise RuntimeError("Triton is required for selected_linear")
    weights, biases = bank
    batch, tokens, width = inputs.shape
    output_dim = weights.size(1)
    assert inputs.is_cuda and weights.is_cuda and groups.is_cuda
    assert weights.dtype == torch.bfloat16 and weights.shape[2] == width
    assert groups.dtype == torch.long and groups.numel() == batch
    result = torch.empty((batch, tokens, output_dim),
                         dtype=torch.bfloat16, device=inputs.device)
    _selected_linear[(triton.cdiv(tokens, 16), triton.cdiv(output_dim, 64), batch)](
        inputs.contiguous(), weights.contiguous(),
        biases.contiguous() if biases is not None else weights,
        groups.contiguous(), result, tokens, width, output_dim,
        biases is not None, 16, 64, 64, num_warps=4,
    )
    return result


def selected_linear_grouped(inputs, bank, order, offsets, counts):
    if triton is None:
        raise RuntimeError("Triton is required for grouped sparse inference")
    weights, biases = bank
    batch, tokens, width = inputs.shape
    output_dim = weights.size(1)
    result = torch.empty((batch, tokens, output_dim),
                         dtype=torch.bfloat16, device=inputs.device)
    _grouped_linear[(triton.cdiv(batch * tokens, 32),
                     triton.cdiv(output_dim, 64), weights.size(0))](
        inputs.contiguous(), weights.contiguous(),
        biases.contiguous() if biases is not None else weights,
        order.contiguous(), offsets.contiguous(), counts.contiguous(), result,
        tokens, width, output_dim, biases is not None, 32, 64, 64,
        num_warps=4,
    )
    return result
