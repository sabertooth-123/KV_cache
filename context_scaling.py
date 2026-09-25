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

# IMPORTANT: we never pinned attn_implementation, so the model is running on whatever
# transformers picked by default (commonly SDPA, which CAN use a memory-efficient/
# flash-attention-style kernel that never materializes the full S x S attention-score
# matrix). The earlier "the O(S^2) transient attention matrix explains the gap" story
# was never actually verified against which kernel ran -- print it so it's checkable,
# rather than assumed, going forward.
print(f"attn_implementation in use: {model.config._attn_implementation!r}")

_ = model(torch.randint(0, config.vocab_size, (1, 64), device=DEVICE), use_cache=True)
torch.cuda.synchronize()


def random_prompt(seq_len):
    return torch.randint(low=0, high=config.vocab_size, size=(1, seq_len), device=DEVICE)


def run_with_cache(input_ids, num_new_tokens):
    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)
        prefill_logits_shape = outputs.logits.shape  # capture before it's overwritten below
        past = outputs.past_key_values
        next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for _ in range(num_new_tokens - 1):
            outputs = model(next_id, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return prefill_logits_shape


print(f"{'context':>8}  {'predicted cache-only':>20}  {'prefill logits':>16}  {'actual peak':>14}")
for seq_len in CONTEXT_LENGTHS:
    try:
        torch.cuda.empty_cache()  # avoid reserved-memory buildup accumulating across shapes
        torch.cuda.reset_peak_memory_stats()
        prompt = random_prompt(seq_len)
        prefill_logits_shape = run_with_cache(prompt, NUM_NEW_TOKENS)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2

        predicted_bytes = 2 * 24 * 1 * seq_len * 2 * 64 * 2  # L=24, B=1, H_KV=2, D=64, fp16
        predicted_mb = predicted_bytes / 1024**2

        # Prefill computes logits for EVERY position by default (shape (1, seq_len, vocab_size)),
        # not just the last one -- at Qwen2.5's ~152K vocab this is a real, previously-unmeasured
        # memory cost that scales LINEARLY with seq_len, separate from the O(S^2) attention-score
        # matrix. Reported here so the two candidate explanations for the predicted-vs-actual gap
        # can actually be compared, instead of only ever attributing it to the attention matrix.
        logits_elements = 1
        for d in prefill_logits_shape:
            logits_elements *= d
        logits_mb = logits_elements * 2 / 1024**2  # fp16, 2 bytes/element

        print(f"{seq_len:>8}  {predicted_mb:>18.3f}MB  {logits_mb:>14.1f}MB  {peak_mb:>12.1f}MB")
    except torch.cuda.OutOfMemoryError:
        print(f"{seq_len:>8}  OOM (torch.cuda.OutOfMemoryError) -- this is the real ceiling on this hardware")
        break
    except RuntimeError as e:
        # At the true hardware edge, CUDA can raise a raw driver-level "out of memory"
        # RuntimeError instead of torch's own catchable OutOfMemoryError -- observed
        # directly on this card at the length right after 8192. Only swallow it if it's
        # actually an OOM; anything else is a real bug and should still crash loudly.
        if "out of memory" in str(e).lower():
            print(f"{seq_len:>8}  OOM (raw CUDA RuntimeError, not torch.cuda.OutOfMemoryError) -- real ceiling")
            break
        raise