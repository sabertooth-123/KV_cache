import time
import torch
from kv_cache import KVCache  # your Phase 3/7 dynamic (torch.cat-based) cache

BATCH = 1
NUM_KV_HEADS = 2
HEAD_DIM = 64
NUM_LAYERS = 24
MAX_SEQ_LEN = 4096
DEVICE = "cuda"
DTYPE = torch.float16


class StaticKVCache:
    """Same concept -- pre-allocated, in-place writes -- but one tensor per layer
    (a list), not one big stacked tensor. Matches the pattern your own KVCache and
    HF's DynamicCache both already use."""

    def __init__(self, num_layers, batch, num_kv_heads, max_seq_len, head_dim, dtype, device):
        self.k_cache = [
            torch.zeros(batch, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(batch, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

    def update(self, layer_idx, k_new, v_new, start_pos):
        end_pos = start_pos + k_new.shape[2]
        self.k_cache[layer_idx][:, :, start_pos:end_pos, :] = k_new
        self.v_cache[layer_idx][:, :, start_pos:end_pos, :] = v_new
        return self.k_cache[layer_idx][:, :, :end_pos, :], self.v_cache[layer_idx][:, :, :end_pos, :]

def time_dynamic_cache(num_steps):
    cache = KVCache(num_layers=NUM_LAYERS)
    step_times = []
    for step in range(num_steps):
        k_new = torch.randn(BATCH, NUM_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        v_new = torch.randn(BATCH, NUM_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for layer_idx in range(NUM_LAYERS):
            cache.update(layer_idx, k_new, v_new)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
    return step_times


def time_static_cache(num_steps):
    cache = StaticKVCache(NUM_LAYERS, BATCH, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, DTYPE, DEVICE)
    step_times = []
    current_len = 0
    for step in range(num_steps):
        k_new = torch.randn(BATCH, NUM_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        v_new = torch.randn(BATCH, NUM_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for layer_idx in range(NUM_LAYERS):
            cache.update(layer_idx, k_new, v_new, current_len)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        current_len += 1
    return step_times


NUM_STEPS = 2000  # need many steps to see the per-step growth clearly

_ = time_dynamic_cache(20)   # warm-up, discarded
_ = time_static_cache(20)

dynamic_times = time_dynamic_cache(NUM_STEPS)
static_times = time_static_cache(NUM_STEPS)


def summarize(name, times):
    early = sum(times[:50]) / 50 * 1000
    late = sum(times[-50:]) / 50 * 1000
    total_ms = sum(times) * 1000
    print(f"{name}: early-step mean={early:.4f}ms  late-step mean={late:.4f}ms  total over {len(times)} steps={total_ms:.2f}ms")


summarize("Dynamic (torch.cat)", dynamic_times)
summarize("Static (pre-allocated)", static_times)