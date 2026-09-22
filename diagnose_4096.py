import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"
SEQ_LEN = 4096
NUM_NEW_TOKENS = 16
NUM_TRIALS = 5

config = AutoConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
model.eval()


def random_prompt(seq_len):
    return torch.randint(low=0, high=config.vocab_size, size=(1, seq_len), device=DEVICE)


def run_no_cache(input_ids, num_new_tokens):
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


# global warm-up
_ = model(random_prompt(64), use_cache=True)
torch.cuda.synchronize()

# warm-up AT THIS SHAPE, discarded
run_no_cache(random_prompt(SEQ_LEN), 3)

for trial in range(NUM_TRIALS):
    free_before, total = torch.cuda.mem_get_info()
    print(f"\n--- Trial {trial+1} --- free VRAM before: {free_before/1024**2:.1f} MB / {total/1024**2:.1f} MB")

    torch.cuda.reset_peak_memory_stats()
    prompt = random_prompt(SEQ_LEN)
    ttft, decode_times = run_no_cache(prompt, NUM_NEW_TOKENS)
    peak_mem = torch.cuda.max_memory_allocated()

    print(f"  TTFT: {ttft*1000:.2f}ms")
    print(f"  decode times (ms): {[f'{t*1000:.1f}' for t in decode_times]}")
    print(f"  peak mem this trial: {peak_mem/1024**2:.1f} MB")