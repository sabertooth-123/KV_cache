import time
import torch
from attention import scaled_dot_product_attention_gqa

DEVICE = "cuda"
BATCH = 1
SEQ_LEN = 2048
NUM_QUERY_HEADS = 16
HEAD_DIM = 64
NUM_LAYERS = 24        # scale the single-layer measurement to a full-model-equivalent size
BYTES_PER_ELEMENT = 2  # fp16
NUM_KV_HEADS_SWEEP = [16, 8, 4, 2, 1]   # MHA -> GQA(g=2) -> GQA(g=4) -> GQA(g=8) -> MQA
NUM_TRIALS = 10


def make_qkv(num_kv_heads):
    Q = torch.randn(BATCH, NUM_QUERY_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    K = torch.randn(BATCH, num_kv_heads, SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    V = torch.randn(BATCH, num_kv_heads, SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    return Q, K, V


def stored_cache_bytes_single_layer(num_kv_heads):
    return 2 * BATCH * SEQ_LEN * num_kv_heads * HEAD_DIM * BYTES_PER_ELEMENT


def time_attention(Q, K, V, num_kv_heads, trials=NUM_TRIALS):
    for _ in range(3):  # warm-up, discarded
        _ = scaled_dot_product_attention_gqa(Q, K, V, NUM_QUERY_HEADS, num_kv_heads, causal=True)
    torch.cuda.synchronize()

    times = []
    with torch.inference_mode():
        for _ in range(trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = scaled_dot_product_attention_gqa(Q, K, V, NUM_QUERY_HEADS, num_kv_heads, causal=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return times


print(f"{'H_KV':>6}  {'single-layer cache':>20}  {'x24 layers':>12}  {'mean latency':>13}  {'std':>9}")
for num_kv_heads in NUM_KV_HEADS_SWEEP:
    Q, K, V = make_qkv(num_kv_heads)
    single_layer_bytes = stored_cache_bytes_single_layer(num_kv_heads)
    full_model_mb = single_layer_bytes * NUM_LAYERS / 1024**2

    times = time_attention(Q, K, V, num_kv_heads)
    mean_ms = sum(times) / len(times) * 1000
    std_ms = (sum((t * 1000 - mean_ms) ** 2 for t in times) / len(times)) ** 0.5

    print(f"{num_kv_heads:>6}  {single_layer_bytes/1024:>17.1f}KB  {full_model_mb:>10.1f}MB  {mean_ms:>10.3f}ms  {std_ms:>7.3f}ms")

    del Q, K, V
    torch.cuda.empty_cache()