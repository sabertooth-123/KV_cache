import torch
from attention import scaled_dot_product_attention
from kv_cache import KVCache

torch.manual_seed(0)

D_MODEL = 32
VOCAB_SIZE = 20

W_Q = torch.randn(D_MODEL, D_MODEL) * 0.02
W_K = torch.randn(D_MODEL, D_MODEL) * 0.02
W_V = torch.randn(D_MODEL, D_MODEL) * 0.02
W_OUT = torch.randn(D_MODEL, VOCAB_SIZE) * 0.02
EMBED = torch.randn(VOCAB_SIZE, D_MODEL) * 0.02  # toy embedding table


def project(x):
    return x @ W_Q, x @ W_K, x @ W_V


def to_logits(attn_out):
    return attn_out @ W_OUT  # (batch, seq, vocab)


def generate_no_cache(prompt_ids: torch.Tensor, num_new_tokens: int):
    """Full recompute every step. prompt_ids: (batch, prompt_len) long tensor."""
    ids = prompt_ids.clone()
    all_step_logits = []
    for _ in range(num_new_tokens):
        emb = EMBED[ids]                         # (batch, seq_len_so_far, d_model)
        q, k, v = project(emb)
        out, _ = scaled_dot_product_attention(q, k, v, causal=True)
        logits = to_logits(out)                  # (batch, seq_len_so_far, vocab)
        next_logits = logits[:, -1, :]           # only the last position matters
        next_id = next_logits.argmax(dim=-1, keepdim=True)
        all_step_logits.append(next_logits)
        ids = torch.cat([ids, next_id], dim=1)
    return ids, all_step_logits


def generate_with_cache(prompt_ids: torch.Tensor, num_new_tokens: int):
    cache = KVCache(num_layers=1)
    ids = prompt_ids.clone()
    all_step_logits = []

    # prefill
    emb = EMBED[prompt_ids]
    q, k, v = project(emb)
    k_all, v_all = cache.update(0, k, v)
    out, _ = scaled_dot_product_attention(q, k_all, v_all, causal=True)
    logits = to_logits(out)
    next_logits = logits[:, -1, :]
    next_id = next_logits.argmax(dim=-1, keepdim=True)
    all_step_logits.append(next_logits)
    ids = torch.cat([ids, next_id], dim=1)

    # decode remaining tokens one at a time
    for _ in range(num_new_tokens - 1):
        emb = EMBED[ids[:, -1:]]                 # only the newest token
        q, k, v = project(emb)
        k_all, v_all = cache.update(0, k, v)
        out, _ = scaled_dot_product_attention(q, k_all, v_all, causal=False)
        logits = to_logits(out)
        next_logits = logits[:, -1, :]
        next_id = next_logits.argmax(dim=-1, keepdim=True)
        all_step_logits.append(next_logits)
        ids = torch.cat([ids, next_id], dim=1)

    return ids, all_step_logits


def test_cache_matches_no_cache_short_prompt():
    prompt_ids = torch.tensor([[3, 7, 1]])
    ids_a, logits_a = generate_no_cache(prompt_ids, num_new_tokens=5)
    ids_b, logits_b = generate_with_cache(prompt_ids, num_new_tokens=5)

    assert torch.equal(ids_a, ids_b), f"generated tokens differ: {ids_a} vs {ids_b}"
    for step, (la, lb) in enumerate(zip(logits_a, logits_b)):
        assert torch.allclose(la, lb, atol=1e-5, rtol=1e-4), (
            f"logits differ at step {step}: max abs diff {(la - lb).abs().max().item()}"
        )
    print("PASS: 3-token prompt, 5 new tokens")


def test_cache_matches_no_cache_longer_prompt():
    prompt_ids = torch.tensor([[0, 5, 9, 2, 14, 6]])
    ids_a, logits_a = generate_no_cache(prompt_ids, num_new_tokens=8)
    ids_b, logits_b = generate_with_cache(prompt_ids, num_new_tokens=8)

    assert torch.equal(ids_a, ids_b)
    for la, lb in zip(logits_a, logits_b):
        assert torch.allclose(la, lb, atol=1e-5, rtol=1e-4)
    print("PASS: 6-token prompt, 8 new tokens")


if __name__ == "__main__":
    test_cache_matches_no_cache_short_prompt()
    test_cache_matches_no_cache_longer_prompt()
    print("\nAll Phase 4 correctness tests passed.")