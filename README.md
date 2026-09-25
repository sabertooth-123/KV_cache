# KV Cache Lab

A from-scratch implementation of attention and KV caching using PyTorch.

This project explores how KV caching affects memory, speed, and model behavior. Everything was implemented and tested on a single RTX 4060 with 8GB VRAM.

This is not a new KV cache algorithm. It is a learning project focused on implementation, experiments, debugging, and understanding how transformer inference works.

**Blog:**(https://evergreen-learning-5e5.notion.site/What-I-learned-building-a-KV-cache-from-scratch-3e2cb2d4f8b480729e12f7518c03854f?pvs=74)

## What I Implemented

1. Multi Head Attention (MHA)
2. Grouped Query Attention (GQA)
3. Multi Query Attention (MQA)
4. Dynamic and static KV caches
5. INT8 KV cache quantization
6. Batch size and context length experiments
7. KV cache memory calculations
8. Causal zero ablation experiments

## Key Findings

**1. Dynamic vs Static Cache**
Dynamic caching becomes more expensive as the sequence grows because new tokens are continuously added to the cache.
Static caching keeps a fixed memory layout and provides more consistent update times in my tests.
However, static caching was not always faster. The result depends on the sequence length and implementation.

**2. GQA and MQA**
GQA and MQA reduce the number of key and value heads.
For Qwen2.5 0.5B, the configuration has 14 query heads and 2 KV heads. This gives a theoretical 7× reduction in KV cache memory compared with an equivalent MHA configuration.
A separate synthetic experiment showed around 16% latency improvement across the tested configurations.
This experiment used random tensors, not Qwen's actual model weights.

**3. INT8 Quantization**
A basic INT8 quantization method caused 8.3% degradation on one 30-token continuation.
After using per-position scaling and quantizing each token only once, the degradation decreased to 4.35%.
These results come from one prompt and one short continuation. They are not a general perplexity benchmark.

**4. Context Length and Memory**
Longer contexts increase memory usage because the KV cache grows with sequence length.
On my RTX 4060 with 8GB VRAM, memory pressure became noticeable around 4,096 to 6,000 tokens in the tested workload.
The practical limit depends on the model, temporary activations, memory allocation, and attention implementation.

**5. Batch Size Scaling**
Memory usage increased approximately linearly with batch size.
Throughput improved at smaller batch sizes but showed diminishing returns at larger batch sizes.
The exact hardware bottleneck was not directly profiled.

**6. Causal Zero Ablation**
I tested the effect of setting the key and value vectors of a cached position to zero across layers.
This experiment found a position whose zero-ablation caused a large change in the final output distribution, even though its attention weight and key vector norm were not unusual.
This result was observed for one model, one prompt, and one specific ablation method. It is not a general claim about attention.

## Running the Project

```bash
python -m venv venv
venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers numpy matplotlib

python test_kv_cache.py

Requirements
Python 3.10 or newer
PyTorch
Transformers
NumPy
Matplotlib

CUDA-enabled PyTorch is required for GPU benchmarks.
Correctness tests can also run on the CPU.

What I Learned

This project helped me understand:

How attention works at the tensor level
How KV caching reduces repeated computation
How tensor dimensions affect cache correctness
How quantization affects memory and output quality
How GPU memory limits transformer inference
Why benchmarking requires careful experiment design
How debugging can change the interpretation of results
