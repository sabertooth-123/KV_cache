import time
import torch
from attention import scaled_dot_product_attention

DEVICE = "cuda"
BATCH = 1
NUM_HEADS = 16
SEQ_LEN = 2048
HEAD_DIM = 64
NUM_TRIALS = 10

torch.manual_seed(0)


def quantize_int8(x: torch.Tensor):
    """Naive per-tensor symmetric quantization -- coarser than real systems use (they go per-channel/per-token)."""
    scale = x.abs().max() / 127.0
    x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale


def dequantize_int8(x_int8: torch.Tensor, scale: torch.Tensor, target_dtype) -> torch.Tensor:
    return x_int8.to(target_dtype) * scale


def time_call(fn, trials=NUM_TRIALS):
    for _ in range(3):  # warm-up, discarded
        fn()
    torch.cuda.synchronize()
    times = []
    with torch.inference_mode():
        for _ in range(trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return times


# --- ground-truth reference, FP32 ---
Q_fp32 = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float32)
K_fp32 = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float32)
V_fp32 = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float32)

with torch.inference_mode():
    reference_output, _ = scaled_dot_product_attention(Q_fp32, K_fp32, V_fp32, causal=True)

results = {}


def record(name, bytes_per_element, call_fn):
    cache_mb = 2 * BATCH * NUM_HEADS * SEQ_LEN * HEAD_DIM * bytes_per_element / 1024**2
    times = time_call(call_fn)
    with torch.inference_mode():
        out, _ = call_fn()
    error = (out.float() - reference_output).abs().max().item()
    results[name] = (cache_mb, times, error)


# --- INT8 (now measured FIRST -- worst position for any warm-up/clock-ramp advantage) ---
K_int8, K_scale = quantize_int8(K_fp32)
V_int8, V_scale = quantize_int8(V_fp32)
Q16_for_int8 = Q_fp32.half()

def int8_call():
    K_deq = dequantize_int8(K_int8, K_scale, torch.float16)
    V_deq = dequantize_int8(V_int8, V_scale, torch.float16)
    return scaled_dot_product_attention(Q16_for_int8, K_deq, V_deq, causal=True)

record("int8", 1, int8_call)

# --- BF16 ---
Qb, Kb, Vb = Q_fp32.bfloat16(), K_fp32.bfloat16(), V_fp32.bfloat16()
record("bf16", 2, lambda: scaled_dot_product_attention(Qb, Kb, Vb, causal=True))

# --- FP16 ---
Q16, K16, V16 = Q_fp32.half(), K_fp32.half(), V_fp32.half()
record("fp16", 2, lambda: scaled_dot_product_attention(Q16, K16, V16, causal=True))

# --- FP32 (now measured LAST -- best position for any warm-up/clock-ramp advantage) ---
record("fp32", 4, lambda: scaled_dot_product_attention(Q_fp32, K_fp32, V_fp32, causal=True))

print(f"{'dtype':>6}  {'cache size':>12}  {'mean latency':>13}  {'std':>9}  {'max abs error vs fp32':>22}")
for name, (cache_mb, times, error) in results.items():
    mean_ms = sum(times) / len(times) * 1000
    std_ms = (sum((t * 1000 - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
    print(f"{name:>6}  {cache_mb:>10.2f}MB  {mean_ms:>10.3f}ms  {std_ms:>7.3f}ms  {error:>20.6f}")