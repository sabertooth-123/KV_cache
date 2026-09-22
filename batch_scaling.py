import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"
SEQ_LEN = 512          # fixed context length -- isolate batch size as the only variable
NUM_NEW_TOKENS = 16
NUM_TRIALS = 5
BATCH_SIZES = [1, 2, 4, 8, 16, 32]

config = AutoConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
model.eval()

_ = model(torch.randint(0, config.vocab_size, (1, 64), device=DEVICE), use_cache=True)
torch.cuda.synchronize()


def random_prompt(batch_size, seq_len):
    return torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), device=DEVICE)


def run_with_cache(input_ids, num_new_tokens):
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


print(f"{'batch':>6}  {'peak mem':>10}  {'decode/step':>13}  {'std':>9}  {'aggregate tok/s':>16}")
for batch_size in BATCH_SIZES:
    try:
        warm_prompt = random_prompt(batch_size, SEQ_LEN)
        run_with_cache(warm_prompt, 3)

        decode_all = []
        peak_mems = []
        for _ in range(NUM_TRIALS):
            prompt = random_prompt(batch_size, SEQ_LEN)
            torch.cuda.reset_peak_memory_stats()
            ttft, decode_times = run_with_cache(prompt, NUM_NEW_TOKENS)
            decode_all.extend(decode_times)
            peak_mems.append(torch.cuda.max_memory_allocated())

        mean_ms = sum(decode_all) / len(decode_all) * 1000
        std_ms = (sum((t * 1000 - mean_ms) ** 2 for t in decode_all) / len(decode_all)) ** 0.5
        peak_mb = sum(peak_mems) / len(peak_mems) / 1024**2
        aggregate_tok_s = batch_size / (mean_ms / 1000)

        print(f"{batch_size:>6}  {peak_mb:>8.1f}MB  {mean_ms:>11.3f}ms  {std_ms:>7.3f}ms  {aggregate_tok_s:>14.1f}")
    except torch.cuda.OutOfMemoryError:
        print(f"{batch_size:>6}  OOM -- stopping sweep here")
        break