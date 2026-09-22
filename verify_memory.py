import torch
from memory import ModelConfig, kv_cache_bytes, format_bytes

assert torch.cuda.is_available(), "CUDA GPU required for this experiment"
device = "cuda"

# Force CUDA context init BEFORE baseline measurement, so context overhead
# doesn't leak into our before/after delta.
_ = torch.zeros(1, device=device)
torch.cuda.synchronize()

config = ModelConfig(num_layers=12, num_kv_heads=12, head_dim=64)  # GPT-2 small
batch_size, seq_len, dtype_str = 1, 1024, "fp16"
torch_dtype = torch.float16

predicted_bytes = kv_cache_bytes(config, batch_size, seq_len, dtype_str)
print("Predicted:", format_bytes(predicted_bytes))

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
allocated_before = torch.cuda.memory_allocated()

# Shape mirrors a real KV cache: (layers, batch, kv_heads, seq_len, head_dim)
k_cache = torch.zeros(
    config.num_layers, batch_size, config.num_kv_heads, seq_len, config.head_dim,
    dtype=torch_dtype, device=device,
)
v_cache = torch.zeros(
    config.num_layers, batch_size, config.num_kv_heads, seq_len, config.head_dim,
    dtype=torch_dtype, device=device,
)
torch.cuda.synchronize()

allocated_after = torch.cuda.memory_allocated()
reserved_after = torch.cuda.memory_reserved()
actual_allocated = allocated_after - allocated_before

print("Actual (allocated delta):", format_bytes(actual_allocated))
print("Actual (reserved, total):", format_bytes(reserved_after))

diff = actual_allocated - predicted_bytes
pct = 100 * diff / predicted_bytes
print(f"\nallocated vs predicted: {diff:+,} bytes ({pct:+.2f}%)")