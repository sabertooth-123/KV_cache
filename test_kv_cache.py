import torch
from attention import scaled_dot_product_attention, scaled_dot_product_attention_gqa
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


def test_kv_cache_supports_gqa_shape():
    """Prove KVCache also works correctly for the real (batch, num_kv_heads, seq, d_k)
    shape used by GQA/MQA, not just the toy single-head (batch, seq, d_k) shape above.
    Previously KVCache hardcoded dim=1 for concatenation, which only happened to be
    correct for the 3D toy shape -- the real GQA/MQA experiment (mha_gqa_mqa.py) never
    actually routed through this class, so this generalization was never proven."""
    torch.manual_seed(2)
    BATCH, NUM_QUERY_HEADS, NUM_KV_HEADS, HEAD_DIM = 1, 4, 2, 8
    PROMPT_LEN, NUM_NEW_TOKENS = 3, 4
    TOTAL_LEN = PROMPT_LEN + NUM_NEW_TOKENS

    all_k = torch.randn(BATCH, NUM_KV_HEADS, TOTAL_LEN, HEAD_DIM)
    all_v = torch.randn(BATCH, NUM_KV_HEADS, TOTAL_LEN, HEAD_DIM)
    all_q = torch.randn(BATCH, NUM_QUERY_HEADS, TOTAL_LEN, HEAD_DIM)

    # no-cache: full recompute over the growing sequence, every step
    no_cache_outputs = []
    for t in range(PROMPT_LEN, TOTAL_LEN):
        q = all_q[:, :, :t + 1, :]
        k = all_k[:, :, :t + 1, :]
        v = all_v[:, :, :t + 1, :]
        out, _ = scaled_dot_product_attention_gqa(q, k, v, NUM_QUERY_HEADS, NUM_KV_HEADS, causal=True)
        no_cache_outputs.append(out[:, :, -1, :])

    # with-cache: prefill once, then feed one new token at a time through KVCache
    cache = KVCache(num_layers=1)
    k_all, v_all = cache.update(0, all_k[:, :, :PROMPT_LEN, :], all_v[:, :, :PROMPT_LEN, :])
    q_prompt = all_q[:, :, :PROMPT_LEN, :]
    scaled_dot_product_attention_gqa(q_prompt, k_all, v_all, NUM_QUERY_HEADS, NUM_KV_HEADS, causal=True)

    with_cache_outputs = []
    for t in range(PROMPT_LEN, TOTAL_LEN):
        k_all, v_all = cache.update(0, all_k[:, :, t:t + 1, :], all_v[:, :, t:t + 1, :])
        q_new = all_q[:, :, t:t + 1, :]
        out, _ = scaled_dot_product_attention_gqa(q_new, k_all, v_all, NUM_QUERY_HEADS, NUM_KV_HEADS, causal=False)
        with_cache_outputs.append(out[:, :, -1, :])

    for step, (no_cache_out, with_cache_out) in enumerate(zip(no_cache_outputs, with_cache_outputs)):
        assert torch.allclose(no_cache_out, with_cache_out, atol=1e-5, rtol=1e-4), (
            f"GQA-shape cache mismatch at step {step}: "
            f"max diff {(no_cache_out - with_cache_out).abs().max().item()}"
        )
    assert cache.seq_len(0) == TOTAL_LEN, f"expected seq_len {TOTAL_LEN}, got {cache.seq_len(0)}"
    print("PASS: KVCache correctly handles (batch, num_kv_heads, seq, d_k) GQA shape")


if __name__ == "__main__":
    test_cache_matches_no_cache_short_prompt()
    test_cache_matches_no_cache_longer_prompt()
    test_kv_cache_supports_gqa_shape()
    print("\nAll Phase 4 correctness tests passed.")