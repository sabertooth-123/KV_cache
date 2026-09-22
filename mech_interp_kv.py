import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, attn_implementation="eager"
).to(DEVICE)
model.eval()

prompt = "The history of artificial intelligence began with early philosophical debates about"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
NUM_NEW_TOKENS = 10

# --- generate, capturing attention weights along the way ---
generated = input_ids.clone()
with torch.inference_mode():
    outputs = model(generated, use_cache=True, output_attentions=True)
    past = outputs.past_key_values
    for _ in range(NUM_NEW_TOKENS):
        next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
        outputs = model(next_id, past_key_values=past, use_cache=True, output_attentions=True)
        past = outputs.past_key_values

print("Full sequence:", tokenizer.decode(generated[0], skip_special_tokens=True))
seq_len_so_far = generated.shape[1]

# --- 1. attention weights at the final decode step, last layer, averaged over query heads ---
final_attn = outputs.attentions[-1]            # (1, num_query_heads, 1, seq_len_so_far)
avg_attn = final_attn[0].mean(dim=0)[0]        # (seq_len_so_far,)
print("\n[1] Attention distribution over cached positions (last layer, avg over heads):")
for pos, weight in enumerate(avg_attn.tolist()):
    tok = tokenizer.decode(generated[0, pos:pos + 1])
    print(f"  pos {pos:3d} ({tok!r}): {weight:.4f}")

# --- 2. K-vector L2 norms per cached position, last layer, averaged over KV heads ---
last_layer_k = past.layers[-1].keys[0]                    # (num_kv_heads, seq_len, head_dim)
k_norms = last_layer_k.float().norm(dim=-1).mean(dim=0)   # (seq_len,)
print("\n[2] K-vector L2 norm per cached position (last layer, avg over KV heads):")
for pos, norm in enumerate(k_norms.tolist()):
    tok = tokenizer.decode(generated[0, pos:pos + 1])
    print(f"  pos {pos:3d} ({tok!r}): {norm:.2f}")

# --- 3. causal ablation: zero one cached position at a time, measure the effect ---
baseline_logits = outputs.logits[:, -1, :].float()
baseline_probs = F.softmax(baseline_logits, dim=-1)
baseline_top1 = baseline_logits.argmax(dim=-1).item()
last_token = generated[:, -1:]

print("\n[3] Causal ablation: zeroing each position, effect on the final prediction")
print(f"{'pos':>4}  {'token':>12}  {'attn weight':>12}  {'KL div':>10}  {'top1 changed':>13}")

for ablate_pos in range(seq_len_so_far - 1):
    with torch.inference_mode():
        trial_outputs = model(generated[:, :-1], use_cache=True)
        trial_past = trial_outputs.past_key_values
        for layer in trial_past.layers:
            layer.keys[:, :, ablate_pos, :] = 0
            layer.values[:, :, ablate_pos, :] = 0
        ablated_outputs = model(last_token, past_key_values=trial_past, use_cache=True)
        ablated_logits = ablated_outputs.logits[:, -1, :].float()
        ablated_probs = F.softmax(ablated_logits, dim=-1)

    kl_div = F.kl_div(ablated_probs.log(), baseline_probs, reduction="sum").item()
    top1_changed = ablated_logits.argmax(dim=-1).item() != baseline_top1

    tok = tokenizer.decode(generated[0, ablate_pos:ablate_pos + 1])
    print(f"{ablate_pos:>4}  {tok!r:>12}  {avg_attn[ablate_pos].item():>12.4f}  {kl_div:>10.4f}  {str(top1_changed):>13}")