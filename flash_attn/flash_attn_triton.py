import torch
import triton
import triton.language as tl

from flash_attn.flash_attn_torch_simu import reference_attention


@triton.jit
def triton_flash_attention_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        B, H, L,
        D: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    q_block_id = tl.program_id(2)

    program_offset = (batch_id * H + head_id) * L * D
    q_offset = program_offset + q_block_id * BLOCK_SIZE_Q * D

    q_block_ptr = tl.make_block_ptr(
        Q_ptr + q_offset,
        (L, D),
        (D, 1),
        (0, 0),
        (BLOCK_SIZE_Q, D),
        (0, 1)
    )

    k_block_ptr = tl.make_block_ptr(
        K_ptr + program_offset,
        (L, D),
        (D, 1),
        (0, 0),
        (BLOCK_SIZE_KV, D),
        (0, 1)
    )

    v_block_ptr = tl.make_block_ptr(
        V_ptr + program_offset,
        (L, D),
        (D, 1),
        (0, 0),
        (BLOCK_SIZE_KV, D),
        (0, 1)
    )

    scale = 1.0 / (D ** 0.5)
    q_chunk = tl.load(q_block_ptr).view(BLOCK_SIZE_Q, D)

    max_score = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32) - float('inf')
    lse_accum = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
    out_chunk = tl.zeros((BLOCK_SIZE_Q, D), dtype=tl.float32)

    for _ in range(0, L, BLOCK_SIZE_Q):
        k_chunk = tl.load(k_block_ptr)
        v_chunk = tl.load(v_block_ptr)
        att = tl.dot(q_chunk, k_chunk.T) * scale

        chunk_max = tl.max(att, 1)
        max_score_new = tl.maximum(max_score, chunk_max)
        exp_scores = tl.exp(att - max_score_new[:, None])

        alpha = tl.exp(max_score - max_score_new)
        lse_accum = alpha * lse_accum + exp_scores.sum(1)
        out_chunk = alpha[:, None] * out_chunk + tl.dot(exp_scores, v_chunk)

        max_score = max_score_new
        k_block_ptr = k_block_ptr.advance((BLOCK_SIZE_KV, 0))
        v_block_ptr = v_block_ptr.advance((BLOCK_SIZE_KV, 0))

    out_chunk = out_chunk / lse_accum[:, None]

    o_block_ptr = tl.make_block_ptr(
        O_ptr + q_offset,
        (L, D),
        (D, 1),
        (0, 0),
        (BLOCK_SIZE_Q, D),
        (0, 1)
    )

    tl.store(o_block_ptr, out_chunk)


def triton_flash_attention(q, k, v, q_chunk_size=32, kv_chunk_size=32):
    B, H, L, D = q.shape

    output = torch.empty_like(q)

    grid = (B, H, L // q_chunk_size)
    triton_flash_attention_kernel[grid](
        q, k, v, output,
        B, H, L, D,
        BLOCK_SIZE_Q=q_chunk_size,
        BLOCK_SIZE_KV=kv_chunk_size
    )
    return output


def run_test(B=2, L=128, H=4, D=64, atol=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Testing FlashAttention on device: {device}")

    torch.manual_seed(42)
    q = torch.randn(B, H, L, D, device=device)
    k = torch.randn(B, H, L, D, device=device)
    v = torch.randn(B, H, L, D, device=device)

    out_flash = triton_flash_attention(q, k, v)
    out_ref = reference_attention(q, k, v)

    max_diff = (out_flash - out_ref).abs().max().item()
    print(f"Max difference vs reference: {max_diff:.6f}")
    if max_diff > atol:
        print("❌ Test FAILED: difference too large.")
    else:
        print("✅ Test PASSED: output is close to reference.")


if __name__ == '__main__':
    run_test()
