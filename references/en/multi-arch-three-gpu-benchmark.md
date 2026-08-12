# Multi-Architecture Three-GPU Verification Report (bins-v0.2.0) — 2026-08-03 (re-test updated 2026-08-10)

## 2026-08-10 Re-test: 2080 Ti Full Pipeline (bins-v0.4.0 / selective pin)

> Re-tested both models with the bins-v0.4.0 multi-arch binaries + full pipeline (`moe-l2 start --gpu`, selective pin router table top-100) on RTX 2080 Ti (region-42 cloud instance). **The 2080 Ti rows are fully refreshed** (stock llama.cpp binary → moe-l2 optimized multi-arch binary):

| Model | Old (08-03, stock binary) | New (08-10, full pipeline selective pin) | Improvement |
|------|------------------------|-----------------------------------|------|
| DS-V2-Lite (Q2_K) | 6.89 t/s | **87.25 t/s** (follow-up 1, round 3) | +1166% |
| Qwen3.6-35B-A3B (UD-IQ2_M) | 11.15 t/s | **47.24 t/s** (follow-up 1, round 3) | +324% |

- Full 4-round data in [qwen3.6-a3b-iq2m-benchmark.md](qwen3.6-a3b-iq2m-benchmark.md) / [deepseek-v2-lite-q2k-benchmark.md](deepseek-v2-lite-q2k-benchmark.md) (2026-08-10 sections)
- Conclusion: the previous 6.89/11.15 on the 2080 Ti were the **official stock llama.cpp binary** (sm_75 single-arch, no moe-l2 optimizations); the moe-l2 optimized multi-arch binary unleashes real performance on the 2080 Ti, DS +1166%, Qwen +324%
- The 08-03 data below is kept as historical baseline (bins-v0.2.0 / host-buffer main path / stock binary)

## 2026-08-10 Addition: RTX 5090 Full Pipeline Measurement (bins-v0.4.0 / selective pin)

> Measured both models with the bins-v0.4.0 multi-arch binaries + full pipeline (`moe-l2 start --gpu`, selective pin router table top-100) on RTX 5090 (bjb2 cloud instance, round 3 stable round):

| Model | Old (08-03, stock binary) | New (08-10, full pipeline selective pin) | Improvement |
|------|------------------------|-----------------------------------|------|
| DS-V2-Lite (Q2_K) | 16.63 t/s | **135.57 t/s** (follow-up 1, round 3) | +715% |
| Qwen3.6-35B-A3B (UD-IQ2_M) | 9.71 t/s | **76.41 t/s** (follow-up 1, round 3) | +687% |

- DS full rounds: round 1 49.69/22.54/115.84 → round 2 118.57/133.78/135.76 → round 3 **118.53/135.57/152.88** → round 4 122.98/139.60/137.10 (short conversation/follow-up 1/follow-up 2)
- Qwen full rounds: round 1 14.59/58.66/55.93 → round 2 61.44/78.40/68.85 → round 3 **66.44/76.41/69.86** → round 4 66.78/76.45/68.19
- Conclusion: the moe-l2 optimized build unleashes real performance on the 5090 (sm_120a), DS +715%, Qwen +687%; the 5090's Blackwell architecture performs impressively under moe-l2 optimization (DS 135.57 t/s beats the 4090's old baseline of 39.0)

## 2026-08-10 Addition: RTX 4090 Full Pipeline Measurement (bins-v0.4.0 fixed build / selective pin)

> Measured with the bins-v0.4.0 fixed build (downloaded from the GitHub release, includes libmtmd/libllama) on RTX 4090 (bjb1 cloud instance, round 3 stable round), collecting VRAM/memory at the same time:

| Model | Old (08-02 single-arch baseline) | New (08-10, full pipeline selective pin) | Improvement | VRAM | RSS |
|------|------------------------|-----------------------------------|------|------|-----|
| DS-V2-Lite (Q2_K) | 37.5 t/s | **145.63 t/s** (follow-up 1, round 3) | +288% | 4.9 GB | 2.1 GB |
| Qwen3.6-35B-A3B (UD-IQ2_M) | 46.8 t/s | **74.99 t/s** (follow-up 1, round 3) | +60% | 3.1 GB | 2.3 GB |

- DS full rounds: round 1 45.40/102.67/115.72 → round 2 132.27/140.23/123.64 → round 3 **134.05/145.63/127.95** → round 4 135.04/140.31/134.47
- Qwen full rounds: round 1 36.85/47.01/51.52 → round 2 57.78/71.93/62.49 → round 3 **65.08/74.99/63.71** → round 4 64.32/73.54/63.38
- VRAM sampling: DS peak 5065 MiB / Qwen peak 3199 MiB (-c 8192 including KV cache); RSS is for the single llama-server process
- Conclusion: the 4090 (sm_89 Ada) kernel optimizations are the most mature — DS 145.63 t/s beats the 5090 (135.57); Qwen 74.99 t/s is on par with the 5090 (76.41); VRAM usage of 3.1-4.9GB easily fits in an 8GB card

---

## Basic Info

| Item | Value |
|------|-----|
| Binary | llama-multi-v1 (`llama_bins.tar.gz`, bins-v0.2.0, 1.6 GB) |
| Architectures | sm_61 (GTX 1080) / sm_75 (RTX 20) / sm_86 (RTX 30) / sm_89 (RTX 40) / sm_120a (RTX 50) |
| Build | CUDA 12.8.2 (apt, nvidia.cn repo), llama.cpp 76f46ad + A3 patch + host-buffer |
| Inference engine | llama-server (host-buffer expert GPU direct compute + GGML_OP_OFFLOAD_MIN_BATCH=1) |
| Test methodology | c=512 (long conversation c=2048), n_predict=128, ngl 99, temp 0.7 |

---

## Three GPUs × Two Models, host-buffer Main Path (multi-arch build measurements)

### DS-V2-Lite (16B MoE, Q2_K, 6 GB)

| GPU | Architecture | Single-round gen t/s | Short conversation | Follow-up | Long conversation | VRAM |
|-----|------|------------|--------|------|--------|------|
| RTX 2080 Ti | sm_75 | 6.89 | 6.94 | 7.13 | 7.09 | ~1.0-1.3 GB |
| RTX 3080 Ti | sm_86 | 12.25 | 12.26 | - | - | ~1.1 GB |
| RTX 5090 | sm_120a | 16.63 | 16.42 | 18.48 | 16.17 | ~1.3-1.8 GB |
| RTX 4090 (single-arch baseline) | sm_89 | 37.5 | - | - | - | 1.6 GB |

### Qwen3.6-35B-A3B (32B MoE, UD-IQ2_M, 11 GB)

| GPU | Architecture | Single-round gen t/s | Short conversation | Follow-up | Long conversation | VRAM |
|-----|------|------------|--------|------|--------|------|
| RTX 2080 Ti | sm_75 | 11.15 | 11.15 | 11.32 | 11.27 | ~2.1-2.4 GB |
| RTX 3080 Ti | sm_86 | 13.28 | 13.24 | 13.37 | 13.48 | ~2.1-2.2 GB |
| RTX 5090 | sm_120a | 9.71 | 9.55 | 10.67 | 13.21 | ~2.4-2.5 GB |
| RTX 4090 (single-arch baseline) | sm_89 | 46.8 | - | - | - | 2.1 GB |

> Note: the 2080 Ti / 3080 Ti Qwen data comes from single-arch build-a3 (CUDA 11.8) tests; the 5090 is measured with the multi-arch build (CUDA 12.8). All DS rows are measured with the multi-arch build (2080 Ti / 3080 Ti / 5090).

---

## Multi-arch vs Single-arch Comparison (same machine)

| GPU | Single-arch (CUDA 11.8) | Multi-arch (CUDA 12.8) | Improvement |
|-----|-------------------|--------------------|------|
| RTX 3080 Ti (DS) | 7.88 t/s | 12.25 t/s | **+55%** |

The CUDA 12.8 compiler generates better code; the multi-arch package is 55% faster than the old single-arch on the 3080 Ti.

---

## Observations

1. **SM120a validated**: the multi-arch package loads and infers both DS + Qwen normally on RTX 5090 (Blackwell) (all 8 scenario groups passed)
2. **⚠️ 50-series kernel efficiency needs improvement**: the 5090 doesn't crush the 3080 Ti (DS +36%, Qwen -27%). llama.cpp 76f46ad's FP16 kernels for Blackwell (sm_120a, FP4-optimized architecture) aren't mature; rebuilding the multi-arch package with a newer llama.cpp should improve 50-series speed
3. **VRAM usage is architecture-independent**: DS 1.0-1.8 GB, Qwen 2.1-2.5 GB, consistent across all GPU generations (host-buffer property), 3-6x headroom for 8 GB cards
4. **Gen speed is conversation-scenario-independent**: per-card scenario variation <10% (per-token compute is identical; context only affects prefill and KV VRAM)
5. Long-prompt prefill is fast: 5090 DS 555 / Qwen 334 t/s (high batch-prefill efficiency)

---

## Qwen / DS-V2 Same-Card Comparison: Bare Server vs v3.1 Full Pipeline (2026-08-05 addition)

**⚠️ Comparison methodology warning**: the 3080 Ti data in the multi-arch main table (Qwen 13.28 / DS 12.25 t/s) comes from a **3080 Ti 12GB** (10240 cores) with the **bare llama-server** methodology; the local v3.1 full-pipeline test ran on an **RTX 3080 10GB** (8704 cores, ~17% lower compute, memory bandwidth 760 vs 912 GB/s) — **comparing across cards would misread "card differences" as "eviction slowdown"**. To check whether the eviction mechanism slows things down, you must compare bare vs full pipeline on the same card.

**RTX 3080 10GB (region-41, SM86) same-card measurement**, methodology: 128 tokens, n_predict=128, c=2048, `MOE_L2_EVICT_MB=3000 INTERVAL=4 LRU=1 MAX_EXPERTS=4000`, L2 4GB, values taken after warmup:

| Model | Bare server (no eviction) | v3.1 full pipeline | Eviction impact | server RSS (full pipeline) | VRAM |
|------|---------------------|-------------|----------|---------------------|------|
| Qwen3.6-35B-A3B | 8.77 t/s | 9.40-9.41 t/s | **+7%** | 4.5GB | 2.3GB |
| DS-V2-Lite | 8.66 t/s | 10.21-10.23 t/s | **+18%** | 5.6GB | 3.3GB |

**Conclusions**:
1. **v3.1 eviction doesn't slow things down** — same-card full pipeline is actually faster (L2 hot-expert preload + gate domain prediction benefits > eviction overhead), Qwen +7%, DS +18%
2. The gap vs the main table's 3080 Ti (Qwen -29% / DS -17%) is **card difference** (3080 Ti 10240 cores vs 3080 8704 cores + bandwidth difference), not the cost of the eviction mechanism
3. Qwen full pipeline 9.4 t/s is on par with the 2080 Ti's original full-pipeline measurement of 9.26-9.40 t/s (both cards are fast enough for Qwen; the bottleneck is expert transfer)

---

## Verification Environment

| GPU | Driver | CUDA runtime | CPU cores | Notes |
|-----|------|-------------|--------|------|
| RTX 2080 Ti 11GB | 580.105.08 | 12.8 (bundled cuda-libs) | 96 | driver symlink fix needed at boot |
| RTX 3080 Ti 12GB | 595.71.05 | 12.8 | 80 | - |
| RTX 5090 32GB | 580.76.05 | 12.8 | 208 | - |

Model source: DS/Qwen pulled directly from the same region as the 2080 Ti; md5 identical to the 2080 Ti / 3080 Ti copies (Qwen e3d23428...).
