# MoE L2 GPU Benchmark — Phase 3: Qwen3.6-35B-A3B LRU Expert Cache (Baseline Comparison)

## Test Configuration

| Item | Value |
|------|-------|
| Model | Qwen3.6-35B-A3B-UD-IQ2_M.gguf |
| Size | 11 GB |
| Architecture | ~244 layers, 256 experts/layer, top-8×2 active/token |
| Expert size | ~1 MB (IQ2_M) |
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
| **CPU baseline** (`-ngl 0`) | 6.1 | **1.8** | Pure CPU inference |
| **Full GPU** (`-ngl 99`) | 9.1 | **6.5** | Full model offloaded to GPU |
| **CPU-moe** (`-ngl 99 --cpu-moe`) | 9.1 | **6.3** | Non-expert layers on GPU, experts on CPU |
| LRU cache **0.25** | 10.2 | **6.2** | 256×0.25=64 slots/layer |
| LRU cache **0.5** | 9.0 | **6.0** | 256×0.5=128 slots/layer |
| LRU cache **0.75** | 9.5 | **6.0** | 256×0.75=192 slots/layer |
| LRU cache **1.0** | 8.6 | **5.8** | 256×1.0=256 slots/layer (full cache) |

---

## Summary

| Metric | Value |
|--------|-------|
| Gen speed range | 5.8–6.5 t/s |
| CPU-moe vs full GPU gap | 6.3 vs 6.5 t/s (~3% — experts too small) |
| CPU vs GPU speedup | 1.8 → 6.5 t/s (**3.6×**) |
| LRU cache vs CPU-moe | **No benefit; 1.0 is slowest** (5.8 vs 6.3) |

---

## Key Findings

| Finding | Detail |
|---------|--------|
| **1. GPU acceleration works well** | CPU 1.8 → GPU 6.5 t/s, 3.6× |
| **2. CPU-moe ≈ full GPU on this model** | Experts too small (~1 MB), H2D not bottleneck |
| **3. LRU cache provides no benefit** | All fractions ≤ CPU-moe; 1.0 slowest (5.8 t/s) |
| **4. Consistent with DS-V2-Lite findings** | Both models show experts too small for cache to help |

**Core finding**: Qwen3.6 IQ2_M experts (~1 MB) are too small. LRU cache is expected to benefit **models with experts > 500 MB** (e.g. Mixtral 8×7B Q4_K_M, ~2 GB/expert).

## Cross-Model Comparison

| Metric | DS-V2-Lite (Q2_K) | Qwen3.6 (IQ2_M) |
|--------|-------------------|-----------------|
| Model size | 6.0 GB | 11 GB |
| CPU Gen | 4.5 t/s | 1.8 t/s |
| Full GPU Gen | 8.7 t/s | 6.5 t/s |
| CPU-moe Gen | 8.4 t/s | 6.3 t/s |
| Best LRU cache | 8.4 t/s (1.0) | 6.2 t/s (0.25) |
| CPU→GPU speedup | 1.93× | 3.6× |
| Expert size | ~96 MB | ~1 MB |
| **Cache effective?** | ❌ No | ❌ No |

## Next Steps

- [ ] Test on Mixtral 8×7B (Q4_K_M) — ~2 GB/expert, expected cache benefit
