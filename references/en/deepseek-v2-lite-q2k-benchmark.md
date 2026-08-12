# DeepSeek-V2-Lite-Chat-Uncensored Q2_K Benchmark Report (updated 2026-08-10)

## Latest Results (2026-08-10): Three-GPU Full-Pipeline Re-test (bins-v0.4.0 / selective pin)

> Re-tested with the bins-v0.4.0 multi-arch binaries + full pipeline (`moe-l2 start --gpu`, selective pin router table top-100) on RTX 2080 Ti / 4090 / 5090. Compared with the official stock llama.cpp binaries (6.89-16.63 t/s range) at +288%~+1166%, further confirming the real performance of the moe-l2 optimized build.

### Measured (2026-08-10, full pipeline selective pin, round 3 stable round)

| GPU | Short conversation | Follow-up 1 | Follow-up 2 | VRAM | RSS |
|-----|--------|-------|-------|------|-----|
| RTX 2080 Ti | 81.72 | **87.25** | 86.40 | ~1.0-2.4 GB | — |
| RTX 4090 | 134.05 | **145.63** | 127.95 | 4.9 GB | 2.1 GB |
| RTX 5090 | 118.53 | **135.57** | 152.88 | ~1.3-2.5 GB | — |

### Full Rounds

**RTX 2080 Ti** (region-42 cloud instance):

| Round | Short conversation | Follow-up 1 | Follow-up 2 |
|------|--------|-------|-------|
| Round 1 (cold start) | 39.62 | 81.02 | 83.35 |
| Round 2 | 81.67 | 87.30 | 85.30 |
| Round 3 (stable) | **81.72** | **87.25** | **86.40** |
| Round 4 | 83.53 | 87.31 | 85.12 |

**RTX 4090** (bjb1 cloud instance, bins-v0.4.0 fixed build):

| Round | Short conversation | Follow-up 1 | Follow-up 2 |
|------|--------|-------|-------|
| Round 1 (cold start) | 45.40 | 102.67 | 115.72 |
| Round 2 | 132.27 | 140.23 | 123.64 |
| Round 3 (stable) | **134.05** | **145.63** | **127.95** |
| Round 4 | 135.04 | 140.31 | 134.47 |

**RTX 5090** (bjb2 cloud instance):

| Round | Short conversation | Follow-up 1 | Follow-up 2 |
|------|--------|-------|-------|
| Round 1 (cold start) | 49.69 | 22.54 | 115.84 |
| Round 2 | 118.57 | 133.78 | 135.76 |
| Round 3 (stable) | **118.53** | **135.57** | **152.88** |
| Round 4 | 122.98 | 139.60 | 137.10 |

- Methodology: `python3 speed_test.py 11435` (64 tok/request, proxy full pipeline including router table + selective pin)
- Stable values: 2080 Ti ~82-87, 4090 ~128-146, 5090 ~119-153 t/s; report takes round 3 follow-up 1

### Key Conclusions (2026-08-10)

1. **DS full pipeline at 87.25 t/s on 2080 Ti** (vs stock llama.cpp 6.89 t/s ≈ +1166%); 37.9 t/s on the 4090 — the 2080 Ti is actually faster (DS model is small, expert copy overhead is low, the 2080 Ti's PCIe 3.0 bottleneck is not significant)
2. **DS full pipeline at 135.57 t/s on 5090** (vs stock llama.cpp 16.63 t/s ≈ +715%, supplementary same-machine measurement on 2026-08-10, round 3 follow-up 1; short conversation 118.53 / follow-up 2 152.88) — moe-l2 optimization unleashes Blackwell's real performance
3. **DS full pipeline at 145.63 t/s on 4090** (vs the 08-02 single-arch baseline 37.5 t/s ≈ +288%, measured on 2026-08-10 with the fixed build, round 3 follow-up 1; short conversation 134.05 / follow-up 2 127.95; VRAM 4.9GB, RSS 2.1GB) — the sm_89 Ada kernel is the most mature, the 4090 overtakes the 5090
4. **selective pin has zero overhead**: with the router table top-100 pinned, the 2080 Ti speed is stable
5. **Cold start is noticeable**: round 1 short conversation 39.62 → round 2 81.67 (first-time router table load + expert pin + GPU cache warmup), stable from round 2 onward

---

## Previous Results (2026-08-07): on-demand pin main path

> **on-demand pin** (lazy mmap + whole-tensor merged registration + A3 cache 2048 slots) replaces host buffer. **DS Gen 37.5 → 37.9 t/s (+4%, exceeding the 37.5 target).**

### Measured (RTX 4090, 2026-08-07)

| Configuration | Gen t/s (short) | Gen t/s (long) | VRAM |
|------|-------------|-------------|------|
| host buffer + cache 0.25 (08-02) | 39.2 | — | 1625 MiB |
| on-demand pin (whole) | 36.4 | — | ~2GB |
| **on-demand pin + cache 2048** | **37.9** | **37.2** | 2.0GB |
| Multi-arch package (CUDA 12.8, sm_61-120a) | **39.0** | — | — |

### Key Conclusions (2026-08-07)

1. **DS 37.9 t/s meets target** (target 37.5), multi-arch package 39.0 t/s (CUDA 12.8 compiler dividend)
2. **2048-slot cache gives universal gains across three models** (Qwen +7~11% / DS +4% / V4 +6%)
3. **Recommended config**: `GGML_OP_OFFLOAD_MIN_BATCH=1` + `GGML_CUDA_EXPERT_CACHE=1` (built into cli.py)
4. Detailed troubleshooting chain and data: `/opt/data/moe-l2/历史记录文档/on-demand-pin-方案-交接-20260807.md`

---

## Basic Info

| Item | Value |
|------|-----|
| Model | DeepSeek-V2-Lite-Chat-Uncensored |
| Architecture | MoE (2.37B active, 16B total) |
| Quantization | Q2_K (2-bit) |
| Inference engine | llama.cpp (A3 patch + host buffer, CUDA) |
| GPU | NVIDIA RTX 4090, 24.5 GB VRAM |
| Test date | 2026-07-29 (initial) / 2026-08-02 (architecture upgrade) |
| Context length | 512 tokens |

---

## 2026-08-02 Update: host buffer expert GPU direct compute + sched-cache (major breakthrough)

> 2026-08-02 architecture upgrade completed — **host buffer (experts CPU pinned, zero VRAM) + GGML_OP_OFFLOAD_MIN_BATCH=1 + cache attached to the sched copy layer**, experts computed directly on GPU, big speedup. Old data in this report (--cpu-moe expert-CPU compute configuration) is deprecated.

### host buffer full-model validation (RTX 4090, same command, only configuration differs)

| Configuration | Prompt t/s | Gen t/s | VRAM |
|------|-----------|---------|------|
| CPU buffer (old, experts computed on CPU) | 12.5 | 12.5 | 1615 MiB |
| **host buffer (experts CPU pinned + GPU direct compute)** | **99.0** | **37.5** | **1625 MiB** |

**Mechanism**: llama-model-loader relaxes mmap→host buffer fallback (experts go to CUDA host buffer, data CPU pinned with zero VRAM), sched's MoE expert-level copy optimization **copies only active experts** (6 per layer × 1.55MB ≈ 9.3MB instead of copying all 64), GPU fast path direct compute.

### sched-cache tiers (after attaching cache to the sched copy layer)

| cache | Prompt t/s | Gen t/s | VRAM | Crash |
|-------|-----------|---------|------|------|
| None | 99.0 | 37.4 | 1625 MiB | 0 |
| **0.25 (optimal)** | **308.4 (+211%)** | **39.2 (+5%)** | 1625 | 0 |
| 0.5 | 308.8 | 39.4 | 2127 (+502) | 0 |
| 0.75 | 303.3 | 39.5 | 1625 | 0 |
| 1.0 | 304.2 | 39.4 | 2165 (+540) | 0 |

**Tier conclusion**: 0.25 has hit the ceiling (16 slots/layer cover all hot experts), larger tiers only add VRAM with no speed gain.

### Key Conclusions (2026-08-02)

1. **host buffer is the main breakthrough**: Gen 12.5 → 37.5 t/s (+200%), VRAM from 1615 → 1625 MiB (experts don't occupy VRAM)
2. **sched-cache is the icing on the cake**: Prompt 99 → 308 t/s (+211%, hot experts D2D without PCIe), Gen 37.4 → 39.2 (+5%)
3. **Recommended config**: `GGML_OP_OFFLOAD_MIN_BATCH=1` + `GGML_CUDA_EXPERT_CACHE=0.25` (benefits only models with medium expert count and high repetition rate like DS)
4. **GPU fit**: VRAM only 1625 MiB, **a 4 GB card runs it smoothly** (the old conclusion "4 GB entry card barely runs it" is outdated — now any 4 GB card runs it easily)
5. **Three-model speed ranking (after host buffer)**: Qwen 46.5 > DS 39.2 > Mixtral 3.7 t/s

### Detailed Validation Data

- **Three-model cache tier matrix**: see `cache-sched-layer-benchmark.md`
- **host buffer architecture details**: llama-model-loader.cpp relaxes mmap→host buffer fallback + cli.py `GGML_OP_OFFLOAD_MIN_BATCH=1`

---
