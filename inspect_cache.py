import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("cuda")
model.eval()

input_ids = tokenizer("The history of artificial intelligence began with", return_tensors="pt").input_ids.to("cuda")

with torch.inference_mode():
    outputs = model(input_ids, use_cache=True)

past = outputs.past_key_values
print("type of past_key_values:", type(past))
print("dir (non-dunder):", [a for a in dir(past) if not a.startswith("_")])

layer0 = past.layers[0]
print("DynamicLayer attributes:", [a for a in dir(layer0) if not a.startswith("_")])

for attr in ["keys", "values", "key_cache", "value_cache", "key_states", "value_states"]:
    if hasattr(layer0, attr):
        val = getattr(layer0, attr)
        print(f"\nlayer0.{attr} -> type {type(val)}")
        if hasattr(val, "shape"):
            print(f"  shape: {val.shape}, dtype: {val.dtype}")