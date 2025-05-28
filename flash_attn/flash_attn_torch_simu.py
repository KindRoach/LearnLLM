import torch
import torch.nn.functional as F


def flash_attention_no_einsum(q, k, v, q_chunk_size=32, kv_chunk_size=32):
    B, H, L, D = q.shape
    scale = 1.0 / (D ** 0.5)
    device = q.device

    output = torch.zeros_like(q)

    for qs in range(0, L, q_chunk_size):
        qe = min(qs + q_chunk_size, L)
        q_chunk = q[:, :, qs:qe]  # [B, H, Cq, D]

        max_score = torch.full((B, H, qe - qs), float('-inf'), device=device)
        lse_accum = torch.zeros((B, H, qe - qs), device=device)
        out_chunk = torch.zeros((B, H, qe - qs, D), device=device)

        for ks in range(0, L, kv_chunk_size):
            ke = min(ks + kv_chunk_size, L)
            k_chunk = k[:, :, ks:ke]  # [B, H, Ck, D]
            v_chunk = v[:, :, ks:ke]  # [B, H, Ck, D]

            attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) * scale  # [B, H, Cq, Ck]

            block_max = attn_scores.max(dim=-1).values
            max_score_new = torch.maximum(max_score, block_max)
            exp_scores = torch.exp(attn_scores - max_score_new.unsqueeze(-1))

            alpha = torch.exp(max_score - max_score_new)
            lse_accum = alpha * lse_accum + exp_scores.sum(dim=-1)
            out_chunk = alpha.unsqueeze(-1) * out_chunk + torch.matmul(exp_scores, v_chunk)

            max_score = max_score_new

        out_chunk = out_chunk / lse_accum.unsqueeze(-1)
        output[:, :, qs:qe] = out_chunk

    return output


def reference_attention(q, k, v) -> torch.Tensor:
    """
    Naive scaled dot-product attention.
    q, k, v: [B, H, L, D]
    """
    scale = 1.0 / (q.shape[-1] ** 0.5)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L, L]
    attn_probs = F.softmax(attn_scores, dim=-1)
    out = torch.matmul(attn_probs, v)  # [B, H, L, D]
    return out


def run_test(B=2, L=128, H=4, D=64, atol=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Testing FlashAttention on device: {device}")

    torch.manual_seed(42)
    q = torch.randn(B, H, L, D, device=device)
    k = torch.randn(B, H, L, D, device=device)
    v = torch.randn(B, H, L, D, device=device)

    out_flash = flash_attention_no_einsum(q, k, v)
    out_ref = reference_attention(q, k, v)

    max_diff = (out_flash - out_ref).abs().max().item()
    print(f"Max difference vs reference: {max_diff:.6f}")
    if max_diff > atol:
        print("❌ Test FAILED: difference too large.")
    else:
        print("✅ Test PASSED: output is close to reference.")


if __name__ == '__main__':
    run_test()
