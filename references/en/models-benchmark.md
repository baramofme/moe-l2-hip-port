# moe-l2 Supported Models — Measured Summary (updated 2026-08-11)

> All figures are reproducible measurements from the full moe-l2 pipeline (proxy + L2 cache + host-buffer GPU direct compute + selective pin/on-demand pin).
> **2026-08-10: main path upgraded to selective pin** (router-table-driven top-K pin + GPU cache prefetch, bins-v0.4.0); re-measured data on three GPUs (2080 Ti / 4090 / 5090) is in each model's report.
> Test conditions: 64-128 tokens per request, n_predict=128, c=2048-8192, `GGML_OP_OFFLOAD_MIN_BATCH=1`, `GGML_CUDA_EXPERT_CACHE=1`.
> Links to the detailed per-model reports are at the end of this document.

## Summary table (by model size)

| Model | Parameters | File | Quantization | Experts (active) | GPU VRAM | Host RSS | Speed | Verified on |
|------|------|------|------|------------|----------|----------|------|----------|
| DeepSeek-V2-Lite | 16B MoE | 6 GB | Q2_K | 64 (top-6) | **1.6-4.9 GB** | **2.1 GB** | **145.63 t/s** (4090) / 87.25 (2080 Ti) / 135.57 (5090) | RTX 4090 / 2080 Ti / 3080 Ti / 5090 |
| Qwen3.6-35B-A3B | 32B MoE | 11 GB | UD-IQ2_M | 256 (top-8) | **2.1-3.1 GB** | **2.3 GB** | **74.99 t/s** (4090) / 47.24 (2080 Ti) / 76.41 (5090) | RTX 4090 / 2080 Ti / 3080 Ti / 5090 |
| DeepSeek-V4-Flash | **157B MoE** | **85 GB** (3 shards) | UD-IQ2_M | 256 (top-6) | **16.5-16.7 GB** | **17.5-26.8 GB** (on-demand/selective pin) | **35.96 t/s** (4090) | RTX 4090 |
| Mixtral-8x7B | 47B MoE | ~16 GB | Q4_K_M | 8 (top-2) | 2.2-2.9 GB | — | 3.7 t/s* | RTX 4090 (cache test conditions) |
| Qwen3-235B-A22B | 235B MoE | 85.7 GB | Q2_K | 128 (top-8) | **13.9 GB** | **54.7 GB** (selective pin) | **~3.9 t/s** (steady state, 4090) | RTX 4090 |

\* Mixtral figures are from the bare llama-server cache-benefit test (experts computed on CPU, not the host-buffer GPU direct-compute main path); reference only.
\*\* 4090/5090/2080 Ti data measured on the **2026-08-10 selective pin/on-demand main path** (bins-v0.4.0, full pipeline `moe-l2 start --gpu`); 3080 Ti still uses the v3.1 multi-arch conditions (bins-v0.3.0).

## VRAM savings (host-buffer expert GPU direct compute + selective/on-demand pin)

| Model | Standard full load | moe-l2 | Savings | Speed retained |
|------|-------------|--------|------|----------|
| DeepSeek-V2-Lite | 23.3 GB VRAM, 65 t/s | **4.9 GB, 145.63 t/s** (4090, 2026-08-10) | **79%** | 224% |
| Qwen3.6-35B-A3B | OOM on 8 GB GPU | **3.1 GB, 74.99 t/s** (4090, 2026-08-10) | — | beats pre-lazy by 46.5 |
| DeepSeek-V4-Flash | OOM on 10-11 GB GPUs | **16.5 GB VRAM / 17.5 GB RSS, 35.96 t/s** (4090) | — | works with both on-demand and selective pin |

## Full-pipeline measurements (v3.1 fixed-expert-count eviction, RSS capped)

**Qwen3.6-A3B / DS-V2-Lite (same RTX 3080 10 GB GPU, bare vs full pipeline)**

| Model | Bare server | v3.1 full pipeline | Eviction impact | RSS | VRAM |
|------|-----------|------------|----------|-----|------|
| Qwen3.6-35B-A3B | 8.77 t/s | 9.40 t/s | **+7%** | 4.5 GB | 2.3 GB |
| DS-V2-Lite | 8.66 t/s | 10.21 t/s | **+18%** | 5.6 GB | 3.3 GB |

## Multi-architecture measurements (3 GPUs × 2 models, bins-v0.2.0+, CUDA 12.8)

| GPU | Architecture | DS-V2-Lite | Qwen3.6-A3B | VRAM |
|-----|------|-----------|-------------|------|
| RTX 2080 Ti | sm_75 | 87.25 t/s | 47.24 t/s | 1.0-2.4 GB |
| RTX 3080 Ti | sm_86 | 12.25 t/s | 13.28 t/s | 1.1-2.2 GB |
| RTX 5090 | sm_120a | 135.57 t/s | 76.41 t/s | 1.3-2.5 GB |
| RTX 4090 | sm_89 | 145.63 t/s | 74.99 t/s | 3.1-4.9 GB |

\* 4090/5090/2080 Ti measured on the 2026-08-10 bins-v0.4.0 full pipeline (selective pin); 3080 Ti measured with the v3.1 multi-arch package (bins-v0.3.0).

## Key conclusions

1. **VRAM is architecture-independent**: DS 1.6-4.9 GB, Qwen 2.1-3.1 GB, consistent across GPU generations — plenty of headroom on 8 GB cards
2. **v3.1 eviction does not slow things down**: same-GPU comparison shows the full pipeline is actually faster (Qwen +7%, DS +18% — L2 hot-expert prefetch benefit > eviction overhead)
3. **on-demand pin main path (08-07)**: Qwen 50.2 / DS 37.9 / V4 10.1 t/s (4090) — V4 improved 5× from 1.7-2.0; Qwen/DS reached and exceeded the pre-lazy host-buffer records
4. **selective pin + GPU cache prefetch (08-10, v0.4.0)**: router-table-driven top-K pin — V4 RSS **84.4 → 26.8 GB** at **34.67 t/s** (on-demand fallback 17.5 GB / 35.96 t/s; the earlier 10.1 t/s was the official unmodified binary — the moe-l2 optimized build was already ~30-35 t/s); GPU cache prefetch improves cold start +84% (10.7 → 19.7 t/s)
5. **V4 at 0.7-2.2 t/s on the 2080 Ti is the GPU's compute ceiling** (IQ2_M 157B, not an offload cost); 35.96 t/s on the 4090 is already near compute-bound
6. **Dynamic pin set (08-09, low-memory mode)**: register only active experts + LRU-evict cold experts — V4 RSS **84 GB → 17-24 GB** (`MOE_L2_LRU_MAX_EXPERTS` 2000≈17 GB / 12000≈24 GB), speed 4-5 t/s (V4 routing is extremely scattered: a 30-turn session touches ~29 GB of distinct experts, and the first touch of a new expert pays a ~2 ms page-fault disk read); Qwen/DS have small working sets and are unaffected. `MOE_L2_PIN_LAYERS=0-2,14-20,36-37` permanently pins the general/sparse layers (~5.4 GB for free). Trade-off: whole-pin is fastest (30.9 t/s) but needs 82 GB RAM; dynamic pin keeps memory bounded but halves V4 speed
7. **Concurrent shared cache (08-12)**: 4-way concurrency (`--parallel 4`) sharing the A3 cache / selective pin — Qwen (2080 Ti) total throughput 95.02 (same-domain) / 88.28 (cross-domain) t/s, DS 198.59 / 188.25, V4 (4090) 89.66 / 88.10 — **all 2.3-2.5× a single session, cross-domain only -5-7%, no pooling needed**; VRAM grows only by KV slots (+2.9 GB / 4-way), RSS +0.2 GB with no leaks

## Concurrency (2026-08-12, two machines, three models)

| Model (GPU) | Single-session t/s | 4-way same-domain throughput | 4-way cross-domain throughput | vs single session | Cross/same |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (2080 Ti) | 38.4 | **95.02** | **88.28** | 2.3-2.5× | -7% |
| DS-V2-Lite (2080 Ti) | 78.3 | **198.59** | **188.25** | 2.4-2.5× | -5% |
| DeepSeek-V4-Flash (4090, 256 experts) | 35.4-35.8 | **89.66** | **88.10** | 2.5× | -2% |
| Qwen3.6-35B-A3B (4090) | 67.4-68.7 | **174.56** | **144.47** | 2.6× / 2.1× | -17% |
| DS-V2-Lite (4090) | 143.4-143.7 | **369.48** | **354.73** | 2.6× / 2.5× | -4% |

- Conclusion: **concurrency does not break, sharing holds naturally** — under 4-way concurrency (even cross-domain) the active expert sets overlap heavily → no A3 cache contention; total throughput = 2.3-2.5× a single session, so "one AI PC, multiple users at once" is verified in practice
- VRAM: +2.9 GB for 4-way concurrency (all independent KV slots — expected cost, not a leak); RSS +0.2 GB (pages of experts outside the selective pin table do not accumulate)
- ⚠️ Occasional cublas crash on the 4090 at the first cross-domain concurrency (recovers on restart; not consistently reproducible; recorded)
- Full report: [concurrent-cache-sharing-20260812.md](concurrent-cache-sharing-20260812.md)

## Detailed reports

- [DeepSeek-V4-Flash verification report (157B, dual-GPU full pipeline)](deepseek-v4-flash-verify-20260805.md)
- [Qwen3-235B-A22B Q2_K benchmark (235B, three rounds on 4090)](qwen3-235b-a22b-q2k-benchmark.md)
- [Multi-arch three-GPU verification (2080 Ti / 3080 Ti / 5090)](multi-arch-three-gpu-benchmark.md)
- [Qwen3.6-A3B IQ2_M benchmark](qwen3.6-a3b-iq2m-benchmark.md)
- [DeepSeek-V2-Lite Q2_K benchmark](deepseek-v2-lite-q2k-benchmark.md)
- [Cache / sched layer benchmark (DS / Qwen / Mixtral three-model benefit matrix)](cache-sched-layer-benchmark.md)
- [Host-buffer design decisions (CN/EN)](design-decisions.md)
