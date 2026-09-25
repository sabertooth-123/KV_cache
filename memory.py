from dataclasses import dataclass

BYTES_PER_DTYPE = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
}


@dataclass
class ModelConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int


def kv_cache_bytes(config: ModelConfig, batch_size: int, seq_len: int, dtype: str) -> int:
    """M = 2 * L * B * S * H_KV * D * bytes_per_element"""
    bytes_per_element = BYTES_PER_DTYPE[dtype]
    return (
        2
        * config.num_layers
        * batch_size
        * seq_len
        * config.num_kv_heads
        * config.head_dim
        * bytes_per_element
    )


def format_bytes(num_bytes: int) -> str:
    mb = num_bytes / (1024 ** 2)
    gb = num_bytes / (1024 ** 3)
    return f"{num_bytes:,} bytes = {mb:.2f} MB = {gb:.4f} GB"

def max_context_length(config: ModelConfig, batch_size: int, dtype: str, budget_bytes: int) -> int:
    """Estimates the maximum context length based on persistent KV-cache storage alone.
    It does not account for model weights, temporary activations, logits, allocator
    behavior, or other transient memory requirements during prefill. The actual usable
    context limit must be measured for the selected model and attention implementation.
    """
    bytes_per_element = BYTES_PER_DTYPE[dtype]
    per_token_bytes = 2 * config.num_layers * batch_size * config.num_kv_heads * config.head_dim * bytes_per_element
    return budget_bytes // per_token_bytes

if __name__ == "__main__":
    gpt2_small = ModelConfig(num_layers=12, num_kv_heads=12, head_dim=64)
    result = kv_cache_bytes(gpt2_small, batch_size=1, seq_len=1024, dtype="fp16")
    print("GPT-2 small, batch=1, seq_len=1024, fp16:")
    print(format_bytes(result))
    assert result == 37_748_736, f"expected 37,748,736 bytes, got {result}"
    print("matches hand-derived value")