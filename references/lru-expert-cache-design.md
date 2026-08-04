# LRU Expert GPU Cache — Design

> Drafted: 2026-07-25
> Background: per-expert H2D launch achieves 8.6 t/s + 1.2 GiB VRAM on RTX 4090 (vs 13.8 t/s + 23.3 GiB fully offloaded). Of the extra 44 ms/token overhead, 66% (~29 ms) is PCIe H2D copying. An LRU cache keeps recently active experts resident in VRAM so hits read from HBM (1.5 TB/s) instead of PCIe (12 GB/s).

---

## 1. Problem

Per-expert on-demand transfer copies the activated experts from CPU to GPU every step:

```
→ 8 experts × 28 layers × 1.55 MB ≈ 347 MB/token
→ PCIe Gen4 x16 ~12 GB/s → ~29 ms/token
```

This is 66% of the on-demand overhead. **The bottleneck is bus bandwidth, not scheduling.**

## 2. LRU Cache Design

### Core idea
Maintain a fixed-size ring cache pool in VRAM holding recently used expert weights. When a later token selects the same expert, read it from cache — skip PCIe entirely.

### Cache structure

| Property | Value | Note |
|----------|-------|------|
| Data structure | Global static LRU pool, survives across compute calls | Not a scope-local allocator (doesn't persist across calls) |
| Key | `(tensor pointer, expert_id)` | Pointer uniquely identifies one layer's expert set |
| Value | `char *gpu_buf` (~1.55 MB) | Quantized weights for that expert |
| Slot count | **32** (~50 MB) | VRAM 1.2 → 1.25 GiB (+4%) |
| Eviction | LRU (counter + min-scan) | Evict oldest access when full |

### Projected hit-rate impact

DeepSeek-V2-Lite: 64 experts/layer, 6-8 activated per token. Autoregressive generation shows high overlap between adjacent tokens' expert sets.

| Hit rate | Copies/layer | Saved/token | Expected t/s |
|:--------:|:------------:|:-----------:|:------------:|
| 0% (no cache) | 8 | 0 ms | 8.6 |
| 40% | 4.8 | ~12 ms | ~11.5 |
| 60% | 3.2 | ~18 ms | ~13.0 |
| 80% | 1.6 | ~23 ms | ~15.0 |

**60%+ hit rate approaches full-offload speed.**

## 3. Pre-conditions to Verify

### 3.1 Actual expert-selection overlap (verify BEFORE building)
Print the selected expert IDs for a few tokens first. If real hit rate < 20%, building the cache wastes 1-2 days. **Data first, then cache.**

### 3.2 Prefill phase strategy
During prefill all 64 experts activate and would thrash the cache. Recommended:
- **Prefill: cache disabled**, use raw H2D (prefill happens once anyway)
- **Decode: cache enabled**, check + fill + evict per step

### 3.3 Warmup
Not needed. The cache fills naturally: first miss copies in, subsequent tokens hit. First 2-3 tokens have low hit rate, then stabilizes.

## 4. Risks & Mitigations

| Risk | Prob | Mitigation |
|------|:----:|------------|
| Actual hit rate < 20% | Med | Print expert sequence first; don't build the full cache without data |
| `cudaMalloc`/`cudaFree` slow on eviction | Low | 32 slots only evict after 32 distinct experts; overwrite data instead of free/realloc |
| Prefill pollutes cache | Low | Disable cache during prefill |
| Multi-stream race | Low | Cache query/update serialized on CPU side; doesn't touch CUDA stream logic |

## 5. Cache Capacity Policy — Fraction-of-experts (not fixed slots)

Fixed slot counts are unfair across models with very different expert counts (Mixtral 8, DeepSeek 64, Qwen 256). Use a **fraction of per-layer expert count** instead:

```
0        0.25     0.5      0.75     1
├─────────┼─────────┼─────────┼─────────┤
off      min      balanced  max     all
```

| Fraction | 8-expert model | 64-expert model | Effect |
|----------|---------------|-----------------|--------|
| `0` | 0 slots | 0 slots | No cache, original behavior |
| `0.25` | 2 slots | 16 slots | Hottest experts only, minimal VRAM |
| `0.5` | 4 slots | 32 slots | Balanced |
| `0.75` | 6 slots | 48 slots | High hit rate |
| `1` | 8 slots | 64 slots | Full cache, fastest, most VRAM |

```cpp
int slots = ceil(fraction * n_expert_per_layer);
slots = clamp(slots, 1, EXPERT_CACHE_MAX_SLOTS);
```

- **8-expert model** (Mixtral): fixed 16 slots → 8 wasted, 100% VRAM waste
- **64-expert model** (DeepSeek): fixed 16 slots → only caches 25%, caps hit rate
- **Fraction policy**: always `fraction × expert_count`, fair across models

Even at fraction=0.25, LRU keeps the **hottest** experts resident — autoregressive adjacency means real hit rate far exceeds the fraction intuition.

### Configuration (two compatible forms)

```
# CLI flag (recommended for users)
./llama-server -m model.gguf --expert-cache 0.5

# Environment variable (debugging)
GGML_CUDA_EXPERT_CACHE=0.5 ./llama-server ...
```

Default unset = cache off; existing users unaffected.

## 6. Decision Review

**Measured transfer-vs-compute breakdown (DS-V2-Lite Q2_K, CPU baseline)**: per-expert memcpy swap-in ~1,150 µs (peak ~1,800 µs) vs gemv compute ~2,045 µs — **swap is ~36% of the per-expert step cost**. Even on CPU DDR4→DDR4 (no PCIe involved), moving experts is a third of the latency; on GPU with PCIe (12 GB/s vs 60 GB/s DDR4 bandwidth) the transfer share is even larger. This is the concrete justification for an LRU cache: eliminate repeat transfers, keep the hot experts resident.

H2D pipelining (double-buffer + multi-stream overlap) was **already measured and rejected**: DS-V2-Lite Q2_K single expert is only 1.55 MB — 60 µs over PCIe Gen4, while the MMVQ kernel runs in <10 µs. Overlap yields zero benefit.

LRU cache is the next rational step — invest **~50 MB VRAM** (1.2 → 1.25 GiB, +4%) for a shot at **60%+ hit rate ≈ near-double speed**. Even 40% gives 8.6 → 11.5 t/s, far better than the 0-benefit pipeline.

**Measure first. Then cache.**
