import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"
NUM_NEW_TOKENS = 16
NUM_TRIALS = 5
CONTEXT_LENGTHS = [128, 256, 512, 1024,2048,4096,8192]  
config = AutoConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
model.eval()
print(f"attn_implementation in use: {model.config._attn_implementation!r}")


def random_prompt(seq_len: int) -> torch.Tensor:
    return torch.randint(low=0, high=config.vocab_size, size=(1, seq_len), device=DEVICE)


def run_no_cache(input_ids: torch.Tensor, num_new_tokens: int):
    generated = input_ids.clone()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(generated, use_cache=False)
    next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    generated = torch.cat([generated, next_id], dim=1)

    decode_times = []
    with torch.inference_mode():
        for _ in range(num_new_tokens - 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(generated, use_cache=False)
            next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - t0)
            generated = torch.cat([generated, next_id], dim=1)
    return ttft, decode_times


def run_with_cache(input_ids: torch.Tensor, num_new_tokens: int):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)
    past = outputs.past_key_values
    next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    decode_times = []
    next_input = next_id
    with torch.inference_mode():
        for _ in range(num_new_tokens - 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(next_input, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            next_input = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - t0)
    return ttft, decode_times


def benchmark_at_length(seq_len: int):
    warm_prompt = random_prompt(seq_len)
    run_no_cache(warm_prompt, 3)
    run_with_cache(warm_prompt, 3)

    no_cache_ttft, no_cache_decode, no_cache_peak = [], [], []
    with_cache_ttft, with_cache_decode, with_cache_peak = [], [], []

    for _ in range(NUM_TRIALS):
        prompt = random_prompt(seq_len)

        torch.cuda.reset_peak_memory_stats()
        ttft, decode_times = run_no_cache(prompt, NUM_NEW_TOKENS)
        no_cache_ttft.append(ttft)
        no_cache_decode.extend(decode_times)
        no_cache_peak.append(torch.cuda.max_memory_allocated())

        torch.cuda.reset_peak_memory_stats()
        ttft, decode_times = run_with_cache(prompt, NUM_NEW_TOKENS)
        with_cache_ttft.append(ttft)
        with_cache_decode.extend(decode_times)
        with_cache_peak.append(torch.cuda.max_memory_allocated())

    def summarize(name, values):
        arr = np.array(values)
        print(f"  {name}: mean={arr.mean()*1000:.2f}ms  std={arr.std()*1000:.2f}ms")

    print(f"\n=== context_length = {seq_len} ===")
    summarize("no-cache TTFT", no_cache_ttft)
    summarize("no-cache decode/token", no_cache_decode)
    print(f"  no-cache peak mem: {np.mean(no_cache_peak)/1024**2:.1f} MB")
    summarize("with-cache TTFT", with_cache_ttft)
    summarize("with-cache decode/token", with_cache_decode)
    print(f"  with-cache peak mem: {np.mean(with_cache_peak)/1024**2:.1f} MB")

# global warm-up: stabilize CUDA context, allocator, GPU clocks before any real measurement
_ = model(random_prompt(64), use_cache=True)
torch.cuda.synchronize()

for length in CONTEXT_LENGTHS:
    benchmark_at_length(length)