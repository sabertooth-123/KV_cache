import torch


class KVCache:
    """Holds per-layer K,V tensors and grows them as new tokens arrive.

    Supports both the toy (batch, seq, d_k) shape (single implicit head,
    Phases 2-4) and the real (batch, num_kv_heads, seq, d_k) shape used for
    MHA/GQA/MQA (Phase 7+) -- sequence is always the second-to-last axis in
    both conventions, so concatenating on dim=-2 works for either shape
    without the caller needing to say which one they're using. Previously
    this hardcoded dim=1, which only happened to be correct for the 3D case;
    the real GQA/MQA experiments (mha_gqa_mqa.py) never actually went through
    this class as a result, so it was never proven against the shape it was
    conceptually supposed to also support.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.k_cache: list = [None] * num_layers
        self.v_cache: list = [None] * num_layers

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """
        k_new, v_new: (batch, seq_len_new, d_k) OR (batch, num_kv_heads, seq_len_new, d_k).
        seq_len_new = prompt_len during prefill, or 1 during decode.
        Returns the FULL k, v so far for this layer (old + new).
        """
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = k_new
            self.v_cache[layer_idx] = v_new
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], k_new], dim=-2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], v_new], dim=-2)
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def seq_len(self, layer_idx: int) -> int:
        if self.k_cache[layer_idx] is None:
            return 0
        return self.k_cache[layer_idx].shape[-2]


