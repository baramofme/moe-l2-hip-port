# Qwen3-235B-A22B Q2_K Benchmark Report (updated 2026-08-11)

## Latest results (2026-08-11): three rounds of RTX 4090 measurements (bins-v0.8.0 / selective pin)

> The moe-l2-optimized llama-server (moe_l2 0.8.0) fully runs **Qwen3-235B-A22B Q2_K (85.7 GB, 93 layers × 128 experts top-8)** on an RTX 4090 24 GB: 24 GB VRAM + 55 GB RAM, ~3.9 t/s steady state. That is a **4× improvement** over the first successful run on 08-03 (old implementation without A3 cache, 1 t/s).

### Test environment

| Item | Value |
|---|---|
| GPU | AutoDL bjb1, RTX 4090 24 GB (driver 580.76.05, CUDA 13.0) |
| Binary | moe-l2-optimized llama-server (/root/moe_l2/bin/, moe_l2 0.8.0) |
| Model | Qwen3-235B-A22B-GGUF Q2_K, 85,691,002,226 bytes (85.7 GB), two shards (49.9 GB + 35.8 GB), byte-level verification passed |
| Flags | `-ngl 99 -c 2048`, `GGML_OP_OFFLOAD_MIN_BATCH=1`, `GGML_CUDA_EXPERT_CACHE=1` (A3 cache 2048 slots) |
| Methodology | 128 tokens per scenario; prompt tokens are the full input of that request (including carried history) |

### Three rounds of experiments

1. **Bare, full**: llama-server started directly, no selective pin (all 128 experts/layer resident)
2. **Bare with selective pin**: `MOE_L2_ROUTER_FILE=router_qwen235b.map` (93 layers × top-60, covering 98.5% of expert activations)
3. **Full-pipeline selective pin**: `moe-l2 start --gpu --router-map router_qwen235b.map` (cli → proxy → llama-server)

### Measured data

| Metric | Bare full | Bare pin | Full-pipeline pin |
|---|---|---|---|
| Short conversation (21 prompts) | 4.07 t/s | 2.81 t/s | 2.79 t/s |
| Follow-up steady state (3 rounds) | 4.09-4.13 t/s | 3.76-3.90 t/s | 3.79-3.93 t/s |
| Long conversation (1216 prompts) | 3.57 t/s | 3.22 t/s | 3.03 t/s |
| VRAM peak | 12,817 MiB | 12,775 MiB | 13,909 MiB |
| **RSS peak** | **80,788 MB** | **54,216 MB (↓33%)** | **54,659 MB** |
| Load time | 72.8s | 33.1s (↓54%) | 48.1s (incl. proxy pipeline) |

### Key conclusions (2026-08-11)

1. **The 235B runs fully on a 4090**: 24 GB VRAM + 55 GB RAM = running an 85.7 GB model at ~3.9 t/s steady state; the full pipeline (`moe-l2 start --gpu`) matches bare pin (proxy overhead <2%)
2. **Selective pin benefit**: trades speed for 33% less RAM (80.8 → 54.2 GB) + half the load time (72.8 → 33.1 s). Speed cost by scenario: first token in short conversations -31% (on-demand cold start for experts outside the table), long conversations -10%, follow-up steady state -5~8%
3. **RAM tiers**: a 32 GB machine cannot run the 235B (54.7 GB > 32 GB); **64 GB RAM + 24 GB VRAM can**
4. **Expert routing is highly concentrated**: top-60 experts/layer cover 98.5% of activations — router-table-driven selective pin pays off significantly for very large MoE models (33% RAM saved for only a first-token cost)

---

## Reproducibility

- Scripts: `bench235b_full.py` (bare full), `bench235b_pin.py` (bare pin), `bench235b_fullchain.py` (full pipeline), `gen_router_235b.py` (router table generation)
- Raw data: `all_results.txt` (full data for the three rounds), `expert235b.log` (60,683 lines of EXPERT log), `router_qwen235b.map` (93×top-60)
- Model: Qwen3-235B-A22B Q2_K, two shards (byte-level verification passed)
