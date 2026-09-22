import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"
CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384]
NUM_NEW_TOKENS = 8  # small -- we only care about prefill's peak, plus a couple decode steps

config = AutoConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
model.eval()

_ = model(torch.randint(0, config.vocab_size, (1, 64), device=DEVICE), use_cache=True)
torch.cuda.synchronize()


def random_prompt(seq_len):
    return torch.randint(low=0, high=config.vocab_size, size=(1, seq_len), device=DEVICE)


def run_with_cache(input_ids, num_new_tokens):
    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)
        past = outputs.past_key_values
        next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for _ in range(num_new_tokens - 1):
            outputs = model(next_id, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)


print(f"{'context':>8}  {'predicted cache-only':>20}  {'actual peak':>14}")
for seq_len in CONTEXT_LENGTHS:
    try:
        torch.cuda.empty_cache()  # avoid Phase 6's reserved-memory-buildup-across-shapes issue
        torch.cuda.reset_peak_memory_stats()
        prompt = random_prompt(seq_len)
        run_with_cache(prompt, NUM_NEW_TOKENS)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2

        predicted_bytes = 2 * 24 * 1 * seq_len * 2 * 64 * 2  # L=24, B=1, H_KV=2, D=64, fp16
        predicted_mb = predicted_bytes / 1024**2

        print(f"{seq_len:>8}  {predicted_mb:>18.3f}MB  {peak_mb:>12.1f}MB")
    except torch.cuda.OutOfMemoryError:
        print(f"{seq_len:>8}  OOM -- this is the real ceiling on this hardware")
        break