# KV Cache Lab

A from-scratch implementation of attention and KV caching in PyTorch, plus a series of controlled systems experiments measuring how KV-cache design affects memory, latency, and model behavior  built and benchmarked entirely on a single RTX 4060 (8GB VRAM), no cloud GPUs.

This isn't a new KV-cache algorithm or a research contribution. It's a from-scratch implementation, a set of controlled benchmarks, and one exploratory mechanistic-interpretability experiment with the debugging dead-ends left in rather than edited out, because most of the actual learning happened there.

**blog:** *(https://evergreen-learning-5e5.notion.site/What-I-learned-building-a-KV-cache-from-scratch-3e2cb2d4f8b480729e12f7518c03854f?pvs=74)*

## Key findings

- The KV-cache memory formula (`M = 2 × L × B × S × H_KV × D × bytes`) was validated to the exact byte against a real GPU allocation.
- With-cache decoding is 2x-18x faster than no-cache as context grows from 512 to 2048 tokens; prefill time is nearly identical either way, since caching only ever helps decode.
- GQA (14 query heads, 2 KV heads on Qwen2.5-0.5B) gives a **7x KV-cache memory reduction** vs. standard MHA, but only a **~16% latency improvement** — reducing KV heads doesn't reduce attention FLOPs, only memory bandwidth.
- A naive INT8 KV-cache implementation cost **8.3% perplexity degradation**; fixing the quantization scheme (per-token scale, quantize-once instead of repeated re-quantization) roughly halved it to **4.35%**.
- Real measured INT8 memory reduction was **1.94x**, not the assumed 2.00x, once per-token scale metadata is accounted for.
- Batch size scaling: memory grows exactly linearly (`958.3MB fixed + 160.2MB × batch`); throughput scales almost perfectly up to batch 4, then hits diminishing returns as the GPU shifts from memory-bandwidth-bound to compute-bound.
- The honest max context length on this 8GB card is **~4096-6000 tokens** before hitting real memory pressure — not the ~8192-12288 that merely avoids an outright crash.
- A causal ablation experiment found a cached token position with unremarkable attention weight and K-vector norm that was, by a wide margin, the most causally important position in the sequence when removed — attention weight didn't predict it; only intervening did.

## Repository structure

| File | What it does |
|---|---|
| `attention.py` | Scaled dot-product attention from scratch, including GQA/MQA support |
| `kv_cache.py` | From-scratch `KVCache` class |
| `memory.py` | KV-cache memory calculator and max-context-under-budget tool |
| `verify_memory.py` | Validates the memory formula against real GPU allocation |
| `test_kv_cache.py` | Correctness tests: cache vs. no-cache equivalence, GQA shape support |
| `cache_vs_no_cache.py` | Latency/memory benchmark across context lengths |
| `context_scaling.py` | Memory scaling and real-ceiling-finding experiment |
| `diagnose_4096.py` | Diagnostic script from the VRAM-pressure investigation |
| `mha_gqa_mqa.py` | MHA/GQA/MQA memory and latency comparison |
| `cache_precision.py` | FP32/FP16/BF16/INT8 synthetic precision benchmark |
| `precision_perplexity.py` | Real-model INT8 quantization quality experiment |
| `batch_scaling.py` | Batch size vs. memory/throughput experiment |
| `static_cache_experiment.py` | Dynamic (`torch.cat`) vs. static (pre-allocated) cache comparison |
| `mech_interp_kv.py` | Causal ablation / attention-weight / K-norm interpretability experiment |
| `inspect_cache.py` | Utility for inspecting HF's `DynamicCache` internals |
| `plot.py` | Generates the benchmark plots |
| `*.png` | Generated benchmark plots |

## Running it

```bash
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers numpy matplotlib
python test_kv_cache.py

Requires a CUDA-enabled PyTorch build (not the CPU-only default from pip install torch) for any of the benchmark scripts
test_kv_cache.py and the correctness tests will run on CPU fine.
