import torch


class KVCache:
    """Holds per-layer K,V tensors and grows them as new tokens arrive."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.k_cache: list = [None] * num_layers
        self.v_cache: list = [None] * num_layers

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """
        k_new, v_new: (batch, seq_len_new, d_k) for this layer.
        seq_len_new = prompt_len during prefill, or 1 during decode.
        Returns the FULL k, v so far for this layer (old + new).
        """
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = k_new
            self.v_cache[layer_idx] = v_new
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], k_new], dim=1)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], v_new], dim=1)
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def seq_len(self, layer_idx: int) -> int:
        if self.k_cache[layer_idx] is None:
            return 0
        return self.k_cache[layer_idx].shape[1]


