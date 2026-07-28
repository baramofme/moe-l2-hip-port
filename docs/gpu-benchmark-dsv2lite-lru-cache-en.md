# MoE L2 GPU Benchmark — Phase 3: DS-V2-Lite LRU Expert Cache (Baseline Comparison)

## Test Configuration

| Item | Value |
|------|-------|
| Model | DeepSeek-V2-Lite-Chat-Uncensored.Q2_K.gguf |
| Size | 6.0 GB |
| Architecture | 27 layers, 64 experts/layer, top-2 active/token |
| Expert size | ~96 MB (Q2_K) |
| Binary | `/root/llama.cpp/build/bin/llama-cli` (GGML_CUDA_EXPERT_CACHE compiled in) |
| GPU | RTX 4090 24GB (AutoDL) |
| Prompt | "The capital of France is" |
| Generation | 128 tokens (`--single-turn`) |
| Batch size | 512 |
| Test date | 2026-07-28 |

---

## Full Results

| Mode | Prompt t/s | Gen t/s | Notes |
|------|-----------|---------|-------|
| **CPU baseline** (`-ngl 0`) | 12.3 | **4.5** | Pure CPU inference |
| **Full GPU** (`-ngl 99`) | 17.4 | **8.7** | Full model offloaded to GPU |
| **CPU-moe** (`-ngl 99 --cpu-moe`) | 19.7 | **8.4** | Non-expert layers on GPU, experts on CPU |
| LRU cache **0.25** | 16.9 | **8.1** | 64×0.25=16 slots/layer |
| LRU cache **0.5** | 19.7 | **8.1** | 64×0.5=32 slots/layer |
| LRU cache **0.75** | 18.0 | **8.1** | 64×0.75=48 slots/layer |
| LRU cache **1.0** | 17.8 | **8.4** | 64×1.0=64 slots/layer (full cache) |

---

## Summary

| Metric | Value |
|--------|-------|
| Gen speed range | 8.1–8.4 t/s (all cache modes) |
| CPU-moe vs full GPU gap | 8.4 vs 8.7 t/s (~3% — experts too small for H2D bottleneck) |
| CPU vs GPU speedup | 4.5 → 8.7 t/s (**1.93×**) |
| LRU cache vs CPU-moe | **No difference** (all fractions 8.1–8.4 t/s) |

---

## Key Findings

| Finding | Detail |
|---------|--------|
| **1. GPU acceleration works** | CPU 4.5 → GPU 8.7 t/s, 1.93× |
| **2. CPU-moe ≈ full GPU on this model** | Experts too small (96 MB), H2D copy 60 µs, kernel <10 µs |
| **3. LRU cache provides no benefit here** | All fractions equal; cache lookup/eviction overhead offsets gains |
| **4. Expected effective on large-expert models** | Mixtral (8 experts, ~2 GB each) — reducing H2D copies yields measurable speedup |

**Core finding**: DS-V2-Lite Q2_K's expert size (~96 MB) is too small for LRU cache overhead to pay off. The mechanism is better suited for **MoE architectures with large experts (>500 MB) and fewer layers** (e.g. Mixtral 8×7B).

## Next Steps

- [ ] Test LRU cache on Qwen3.6-35B-A3B (IQ2_M) — ~1 MB/expert, 256 experts/layer, top-16 active
- [ ] Test on Mixtral 8×7B (Q4_K_M) — ~2 GB/expert, 8 experts/layer, top-2 active (expected best case)
