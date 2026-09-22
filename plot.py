import matplotlib.pyplot as plt

COLOR_CACHE = "#0072B2"     # blue -- with cache
COLOR_NO_CACHE = "#D55E00"  # vermillion -- no cache

context_lengths = [128, 256, 512, 1024, 2048]
no_cache_decode = [27.76, 28.84, 50.08, 157.16, 490.58]
cache_decode_5 = [25.43, 23.84, 24.47, 25.41, 26.62]
no_cache_ttft = [28.64, 28.06, 40.17, 123.51, 427.68]
cache_ttft_5 = [31.01, 29.33, 44.71, 126.44, 417.62]
no_cache_peak = [1163.9, 1236.9, 1385.8, 1684.3, 2279.3]
cache_peak_5 = [1120.7, 1159.8, 1239.9, 1400.1, 1721.3]

# With-cache measured reliably out to 4096. No-cache did NOT (VRAM-limited, unstable
# -- see the diagnostic run). We plot cache's real 4096 point, and only ANNOTATE
# no-cache's failure there rather than plot a number we don't trust.
cache_context_full = context_lengths + [4096]
cache_decode_full = cache_decode_5 + [60.30]
cache_ttft_full = cache_ttft_5 + [1453.27]
cache_peak_full = cache_peak_5 + [3297.8]

# --- Plot 1: decode latency (the headline result) ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(context_lengths, no_cache_decode, marker="o", color=COLOR_NO_CACHE, label="No cache")
ax.plot(cache_context_full, cache_decode_full, marker="o", color=COLOR_CACHE, label="With cache")
ax.axvline(4096, color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax.text(4096, ax.get_ylim()[1] if ax.get_ylim()[1] else 500, "no-cache: exceeds practical\nVRAM here (unstable, excluded)",
        rotation=90, va="top", ha="right", fontsize=8, color="gray")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("Decode latency per token (ms, log scale)")
ax.set_title("Decode Latency: No Cache vs KV Cache (Qwen2.5-0.5B-Instruct)")
ax.legend()
ax.grid(True, which="both", alpha=0.2)
fig.tight_layout()
fig.savefig("decode_latency_vs_context.png", dpi=150)

# --- Plot 2: TTFT / prefill ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(context_lengths, no_cache_ttft, marker="o", color=COLOR_NO_CACHE, label="No cache")
ax.plot(cache_context_full, cache_ttft_full, marker="o", color=COLOR_CACHE, label="With cache")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("Time to first token (ms, log scale)")
ax.set_title("Prefill (TTFT): No Cache vs KV Cache -- nearly identical, as expected")
ax.legend()
ax.grid(True, which="both", alpha=0.2)
fig.tight_layout()
fig.savefig("ttft_vs_context.png", dpi=150)

# --- Plot 3: peak memory ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(context_lengths, no_cache_peak, marker="o", color=COLOR_NO_CACHE, label="No cache")
ax.plot(cache_context_full, cache_peak_full, marker="o", color=COLOR_CACHE, label="With cache")
ax.set_xscale("log", base=2)
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("Peak GPU memory allocated (MB)")
ax.set_title("Peak GPU Memory vs Context Length")
ax.legend()
ax.grid(True, which="both", alpha=0.2)
fig.tight_layout()
fig.savefig("peak_memory_vs_context.png", dpi=150)

# --- Plot 4: throughput ---
no_cache_throughput = [1000 / t for t in no_cache_decode]
cache_throughput_full = [1000 / t for t in cache_decode_full]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(context_lengths, no_cache_throughput, marker="o", color=COLOR_NO_CACHE, label="No cache")
ax.plot(cache_context_full, cache_throughput_full, marker="o", color=COLOR_CACHE, label="With cache")
ax.set_xscale("log", base=2)
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("Throughput (tokens/sec)")
ax.set_title("Decode Throughput vs Context Length")
ax.legend()
ax.grid(True, which="both", alpha=0.2)
fig.tight_layout()
fig.savefig("throughput_vs_context.png", dpi=150)

plt.show()