"""Minimal scaled dot-product attention, single head, no KV cache."""
import math
import torch


def scaled_dot_product_attention(Q, K, V, causal=False):
    """
    Q: (batch, seq_len_q, d_k)
    K: (batch, seq_len_k, d_k)
    V: (batch, seq_len_k, d_v)
    causal: mask out positions j > i (only valid when seq_len_q == seq_len_k)

    Returns:
        output: (batch, seq_len_q, d_v)
        weights: (batch, seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (batch, seq_len_q, seq_len_k)

    if causal:
        seq_len_q, seq_len_k = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=scores.device),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    output = weights @ V
    return output, weights

def scaled_dot_product_attention_gqa(Q, K, V, num_query_heads, num_kv_heads, causal=False):
    """
    Q: (batch, num_query_heads, seq_q, d_k)
    K, V: (batch, num_kv_heads, seq_k, d_k)   -- the actual STORED cache, small when num_kv_heads < num_query_heads
    num_query_heads must be divisible by num_kv_heads.
    MHA: num_kv_heads == num_query_heads.  MQA: num_kv_heads == 1.
    """
    assert num_query_heads % num_kv_heads == 0
    n_rep = num_query_heads // num_kv_heads

    if n_rep > 1:
        # Broadcast (not copy, until the reshape forces it) each KV head across its group
        # of query heads. This is the standard "repeat_kv" trick real GQA implementations use.
        K = K[:, :, None, :, :].expand(-1, num_kv_heads, n_rep, K.shape[-2], K.shape[-1])
        K = K.reshape(K.shape[0], num_query_heads, K.shape[-2], K.shape[-1])
        V = V[:, :, None, :, :].expand(-1, num_kv_heads, n_rep, V.shape[-2], V.shape[-1])
        V = V.reshape(V.shape[0], num_query_heads, V.shape[-2], V.shape[-1])

    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (batch, num_query_heads, seq_q, seq_k)

    if causal:
        seq_len_q, seq_len_k = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=scores.device),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    output = weights @ V
    return output, weights

