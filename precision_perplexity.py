import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"
NUM_NEW_TOKENS = 30

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
model.eval()

prompt = "The history of artificial intelligence began with early philosophical debates about"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)


def quantize_dequantize_int8(x: torch.Tensor) -> torch.Tensor:
    scale = x.abs().max() / 127.0
    x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8.to(x.dtype) * scale


def generate_fp16_reference(input_ids, num_new_tokens):
    """Normal greedy decode, full FP16 cache. Records the log-prob of each chosen token."""
    generated = input_ids.clone()
    logprobs = []
    with torch.inference_mode():
        outputs = model(generated, use_cache=True)
        past = outputs.past_key_values
        for _ in range(num_new_tokens):
            logits = outputs.logits[:, -1, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            next_id = logits.argmax(dim=-1, keepdim=True)
            logprobs.append(log_probs[0, next_id.item()].item())
            generated = torch.cat([generated, next_id], dim=1)
            outputs = model(next_id, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
    return generated, logprobs


def evaluate_under_int8_cache(full_sequence, prompt_len):
    """Replay the SAME token sequence, teacher-forced, but quantize+dequantize
    every layer's stored K,V after every single step -- simulating a real INT8 cache."""
    logprobs = []
    disagreements = 0
    prompt_ids = full_sequence[:, :prompt_len]

    with torch.inference_mode():
        outputs = model(prompt_ids, use_cache=True)
        past = outputs.past_key_values
        for layer in past.layers:
            layer.keys = quantize_dequantize_int8(layer.keys)
            layer.values = quantize_dequantize_int8(layer.values)

        for step in range(full_sequence.shape[1] - prompt_len):
            logits = outputs.logits[:, -1, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)

            actual_next_id = full_sequence[:, prompt_len + step: prompt_len + step + 1]
            argmax_id = logits.argmax(dim=-1, keepdim=True)
            if argmax_id.item() != actual_next_id.item():
                disagreements += 1
            logprobs.append(log_probs[0, actual_next_id.item()].item())

            outputs = model(actual_next_id, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            for layer in past.layers:
                layer.keys = quantize_dequantize_int8(layer.keys)
                layer.values = quantize_dequantize_int8(layer.values)

    return logprobs, disagreements

def quantize_int8_per_token(x: torch.Tensor):
    """x: (batch, heads, seq_len, head_dim). One scale per token (per position)."""
    scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = scale.clamp(min=1e-8)
    x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale


def evaluate_under_int8_cache_v2(full_sequence, prompt_len):
    """Realistic INT8 cache: quantize each token exactly once, per-token scale,
    never re-quantize already-stored tokens."""
    logprobs = []
    disagreements = 0
    prompt_ids = full_sequence[:, :prompt_len]
    num_layers = model.config.num_hidden_layers
    int8_store = [None] * num_layers  # [k_int8, k_scale, v_int8, v_scale] per layer

    with torch.inference_mode():
        outputs = model(prompt_ids, use_cache=True)
        past = outputs.past_key_values

        for i, layer in enumerate(past.layers):
            k_int8, k_scale = quantize_int8_per_token(layer.keys)
            v_int8, v_scale = quantize_int8_per_token(layer.values)
            int8_store[i] = [k_int8, k_scale, v_int8, v_scale]
            layer.keys = k_int8.to(layer.keys.dtype) * k_scale
            layer.values = v_int8.to(layer.values.dtype) * v_scale

        for step in range(full_sequence.shape[1] - prompt_len):
            logits = outputs.logits[:, -1, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)

            actual_next_id = full_sequence[:, prompt_len + step: prompt_len + step + 1]
            argmax_id = logits.argmax(dim=-1, keepdim=True)
            if argmax_id.item() != actual_next_id.item():
                disagreements += 1
            logprobs.append(log_probs[0, actual_next_id.item()].item())

            outputs = model(actual_next_id, past_key_values=past, use_cache=True)
            past = outputs.past_key_values

            for i, layer in enumerate(past.layers):
                new_k_int8, new_k_scale = quantize_int8_per_token(layer.keys[:, :, -1:, :])
                new_v_int8, new_v_scale = quantize_int8_per_token(layer.values[:, :, -1:, :])

                k_int8, k_scale, v_int8, v_scale = int8_store[i]
                k_int8 = torch.cat([k_int8, new_k_int8], dim=2)
                k_scale = torch.cat([k_scale, new_k_scale], dim=2)
                v_int8 = torch.cat([v_int8, new_v_int8], dim=2)
                v_scale = torch.cat([v_scale, new_v_scale], dim=2)
                int8_store[i] = [k_int8, k_scale, v_int8, v_scale]

                layer.keys = k_int8.to(layer.keys.dtype) * k_scale
                layer.values = v_int8.to(layer.values.dtype) * v_scale

    return logprobs, disagreements


full_sequence, fp16_logprobs = generate_fp16_reference(input_ids, NUM_NEW_TOKENS)
prompt_len = input_ids.shape[1]
int8_logprobs, disagreements = evaluate_under_int8_cache(full_sequence, prompt_len)

int8_v2_logprobs, disagreements_v2 = evaluate_under_int8_cache_v2(full_sequence, prompt_len)
int8_v2_ppl = math.exp(-sum(int8_v2_logprobs) / len(int8_v2_logprobs))

print(f"\nREALISTIC INT8 (per-token scale, quantize-once) perplexity: {int8_v2_ppl:.4f}")
print(f"Argmax disagreements (v2), out of {NUM_NEW_TOKENS} steps: {disagreements_v2}")
print("\nPer-step log-prob (fp16 vs naive-int8 vs realistic-int8):")
for i, (a, b, c) in enumerate(zip(fp16_logprobs, int8_logprobs, int8_v2_logprobs)):
    print(f"  step {i:2d}: fp16={a:8.4f}  naive_int8={b:8.4f}  v2_int8={c:8.4f}")

fp16_ppl = math.exp(-sum(fp16_logprobs) / len(fp16_logprobs))
int8_ppl = math.exp(-sum(int8_logprobs) / len(int8_logprobs))

print("Generated continuation:", tokenizer.decode(full_sequence[0, prompt_len:], skip_special_tokens=True))
print(f"\nFP16 cache perplexity on its own greedy continuation: {fp16_ppl:.4f}")
print(f"INT8-simulated cache perplexity on the SAME continuation: {int8_ppl:.4f}")
print(f"Argmax disagreements (INT8 vs actual token), out of {NUM_NEW_TOKENS} steps: {disagreements}")

print("\nPer-step log-prob (fp16 vs int8-simulated):")
for i, (a, b) in enumerate(zip(fp16_logprobs, int8_logprobs)):
    print(f"  step {i:2d}: fp16={a:8.4f}  int8={b:8.4f}  diff={b - a:+.4f}")