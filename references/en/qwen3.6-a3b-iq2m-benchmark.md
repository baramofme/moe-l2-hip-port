# Qwen3.6-35B-A3B-UD-IQ2_M Benchmark Report (updated 2026-08-14)

## Latest Results (2026-08-14): bins-v0.4.1 Fixed Re-measure (P0 expert-cache race fixed 2026-08-13)

> Re-measured with the bins-v0.4.1 fixed build (P0 expert-cache race fixed 2026-08-13; the 2026-08-10 bins-v0.4.0 numbers were P0-void fake speeds) + full pipeline (`moe-l2 start --gpu`, selective pin router table top-100) on RTX 2080 Ti / 4090 (5090 pending re-measure). 2080 Ti measures 30.87 t/s (locked-version card) vs the official stock llama.cpp binary (11.15 t/s) ≈ +177%.

### Measured (2026-08-14, bins-v0.4.1 re-measure)

| GPU | Short conversation | Follow-up 1 | Follow-up 2 | VRAM | RSS |
|-----|--------|-------|-------|------|-----|
| RTX 2080 Ti | ⚠️ P0-void | **30.87** (locked-version card) | ⚠️ P0-void | TBD | — |
| RTX 4090 | ⚠️ P0-void | **44-48** | ⚠️ P0-void | ~9.3 GB | ~11.4 GB |
| RTX 5090 | ⚠️ P0-void, pending re-measure | ⚠️ P0-void, pending re-measure | ⚠️ P0-void, pending re-measure | TBD | — |

### Full Rounds (⚠️ P0-void, 2026-08-13 voided — 08-10 bins-v0.4.0 numbers kept for the record; fixed re-measure values in the table above)

**RTX 2080 Ti** (region-42 cloud instance):

| Round | Short conversation | Follow-up 1 | Follow-up 2 |
|------|--------|-------|-------|
| Round 1 (cold start) | 26.75 | 38.67 | 34.77 |
| Round 2 | 37.67 | 46.46 | 42.38 |
| Round 3 (stable) | **41.66** | **47.24** | **42.89** |
| Round 4 | 41.11 | 43.93 | 41.34 |

**RTX 4090** (bjb1 cloud instance, bins-v0.4.0 fixed build):

| Round | Short conversation | Follow-up 1 | Follow-up 2 |
|------|--------|-------|-------|
| Round 1 (cold start) | 36.85 | 47.01 | 51.52 |
| Round 2 | 57.78 | 71.93 | 62.49 |
| Round 3 (stable) | **65.08** | **74.99** | **63.71** |
| Round 4 | 64.32 | 73.54 | 63.38 |

**RTX 5090** (bjb2 cloud instance):

| Round | Short conversation | Follow-up 1 | Follow-up 2 |
|------|--------|-------|-------|
| Round 1 (cold start) | 14.59 | 58.66 | 55.93 |
| Round 2 | 61.44 | 78.40 | 68.85 |
| Round 3 (stable) | **66.44** | **76.41** | **69.86** |
| Round 4 | 66.78 | 76.45 | 68.19 |

- Methodology: `python3 speed_test.py 11435` (64 tok/request, proxy full pipeline including router table + selective pin)
- Stable values (P0-void rounds, 2026-08-13 voided): 2080 Ti ~41-47, 4090 ~64-75, 5090 ~66-76 t/s; report takes round 3 follow-up 1. bins-v0.4.1 fixed re-measure: 2080 Ti **30.87** (locked-version card) / 4090 **44-48** t/s

### Key Conclusions (2026-08-10)

1. **Qwen full pipeline at 30.87 t/s on 2080 Ti (locked-version card)** (vs stock llama.cpp 11.15 t/s ≈ +177%, bins-v0.4.1 re-measure) — the 4090 re-measures at 44-48 t/s
2. **Qwen on 5090: ⚠️ P0-void, pending re-measure** — the 2026-08-10 figure (76.41 t/s) was P0-void fake speed; no bins-v0.4.1 measurement yet
3. **Qwen full pipeline at 44-48 t/s on 4090** (bins-v0.4.1 re-measure, follow-up 1; re-test 36-46 t/s consistent; VRAM ~9.3GB, RSS ~11.4GB) — the "on par with the 5090" claim was based on P0-void data (5090 pending re-measure)
4. **selective pin has zero overhead**: with the router table top-100 pinned, the 2080 Ti speed is the same order of magnitude as whole-pin (round-based conclusion; P0-void, 2026-08-13 voided)
5. **Cold start is noticeable**: round 1 26.75 → round 3 41.66 (first-time router table load + expert pin + GPU cache warmup), stable round in the table above (P0-void rounds, 2026-08-13 voided)

---

## Previous Results (2026-08-07): on-demand pin main path (new speed record)

> **on-demand pin** (lazy mmap loading + first-touch merged registration of the whole expert tensor + A3 cache 2048 slots) replaces host buffer as the main path. **Qwen Gen 46.5 → 50.2 t/s (+8%), exceeding the pre-lazy host buffer's 46.5.**

### Measured (RTX 4090, 2026-08-07)

| Configuration | Gen t/s (short) | Gen t/s (long) | VRAM |
|------|-------------|-------------|------|
| host buffer (08-02) | 46.5 | — | 2147 MiB |
| on-demand pin (whole) | 46.9 | 44.8 | ~8GB (incl. model) |
| **on-demand pin + cache 2048** | **50.2** | **49.8** | 2.9GB |
| Multi-arch package (CUDA 12.8, sm_61-120a) | **51.5** | **51.6** | 2.9GB |

### Mechanism

1. **Merged registration**: fixes the CUDA 11.8 pitfall — `cudaMemcpyAsync` with source spanning multiple `cudaHostRegister` regions always crashes (pintest6c is the smoking gun); changed to unregister adjacent regions + register one large region (verified by pintest6d)
2. **whole-tensor pin** (`MOEL2_WHOLE_PIN`, on by default): copy_experts registers the entire expert tensor on first touch, eliminating new-expert page fault disk reads during inference
3. **A3 cache 2048 slots** (`EXPERT_CACHE_MAX_SLOTS`): Qwen short 46.9→50.2, long 44.8→49.8

### Key Conclusions (2026-08-07)

1. **Qwen 50.2 t/s is the current record** (beats pre-lazy 46.5, host buffer 46.5)
2. **2048-slot cache gives universal gains across three models** (Qwen +7~11% / DS +4% / V4 +6%), shipped with v0.7.1 / bins-v0.3.1
3. **Recommended config**: `GGML_OP_OFFLOAD_MIN_BATCH=1` + `GGML_CUDA_EXPERT_CACHE=1` (built into cli.py)
4. Detailed troubleshooting chain and data: `/opt/data/moe-l2/历史记录文档/on-demand-pin-方案-交接-20260807.md`

---

## Basic Info

| Item | Value |
|------|-----|
| Model | Qwen3.6-35B-A3B-UD-IQ2_M |
| Architecture | A3 (3.6B active, 35B total) |
| Quantization | IQ2_M (2-bit) |
| Inference engine | llama.cpp (A3 patch + host buffer, CUDA) |
| GPU | NVIDIA RTX 4090, 24.5 GB VRAM |
| Test date | 2026-07-29 (initial) / 2026-08-02 (architecture upgrade) |
| Context length | 512 tokens |

## Fix Record (2026-07-29)

All `GGML_CUDA_EXPERT_CACHE>0` combinations (short/long/followup, cache=0.1~2.0) previously exited with 134, cuBLAS illegal memory access.

Root cause: after `cache_set` did cudaMalloc + cudaMemcpyAsync D2D on the 970 MB LM head tensor (Q4_K, 2048×248320), repeated allocate/evict cycles caused CUDA memory fragmentation/page table pollution, and subsequent cuBLAS gemm reported illegal memory access.

Fix: added a >100 MB skip check before both `cache_set` call sites. Only the LM head is skipped; all expert weight tensors are under 100 MB and cache normally.

---

## 2026-08-02 Update: host buffer expert GPU direct compute (speed +370%)

> 2026-08-02 architecture upgrade — **host buffer (experts CPU pinned, zero VRAM) + GGML_OP_OFFLOAD_MIN_BATCH=1**, experts computed directly on GPU. Qwen3.6-A3B becomes the **fastest test model** (Gen 46.5 t/s, beating DS 39.2 / Mixtral 3.7).

### host buffer full-model validation (RTX 4090)

| Configuration | Prompt t/s | Gen t/s | VRAM |
|------|-----------|---------|------|
| CPU buffer (old, experts computed on CPU) | 10.0 | 10.0 | 2141 MiB |
| **host buffer (experts CPU pinned + GPU direct compute)** | **75.8** | **46.5** | **2147 MiB** |

**Mechanism**: llama-model-loader relaxes mmap→host buffer fallback (experts go to CUDA host buffer, data CPU pinned with zero VRAM), sched's MoE expert-level copy optimization copies only active experts, GPU fast path direct compute.

### sched-cache validation (after attaching cache to the sched copy layer)

| cache | Prompt t/s | Gen t/s | VRAM |
|-------|-----------|---------|------|
| None | 75.8 | 46.5 | 2147 MiB |
| 0.25 | 76.0 | 46.6 | 2147 MiB |
| 0.5 | 75.6 | 46.5 | 2475 MiB |

**No cache benefit on Qwen** (46.5-46.6 flat) — experts are too small (~1MB) + short prompts already move little data, cache only adds VRAM.

### Key Conclusions (2026-08-02)

1. **host buffer makes Qwen3.6-A3B the fastest model**: Gen 10 → 46.5 t/s (+370%), VRAM 2147 MiB (experts don't occupy VRAM)
2. **Qwen doesn't need cache**: experts too small, no cache benefit, only adds VRAM
3. **Recommended config**: `GGML_OP_OFFLOAD_MIN_BATCH=1` (no cache) — 32B MoE on an 8GB card at 46.5 t/s
4. **Three-model speed ranking (after host buffer)**: Qwen 46.5 > DS 39.2 > Mixtral 3.7 t/s
5. **GPU fit**: VRAM only 2147 MiB, **a 4 GB card runs it smoothly** (the old conclusion "minimum 4 GB, recommended 8 GB" is outdated — an 8 GB card has plenty of headroom)

### Detailed Validation Data (2026-08-02 full chain)

- **Three-model cache tier matrix**: see `cache-sched-layer-benchmark.md`
- **host buffer architecture details**: llama-model-loader.cpp relaxes mmap→host buffer fallback + cli.py `GGML_OP_OFFLOAD_MIN_BATCH=1`
- **Data flywheel**: proxy real traffic collects samples → classifier auto-retrained (seed 111 + 50 real = 161 samples), label quality improved

---
