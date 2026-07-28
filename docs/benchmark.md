<![CDATA[# Benchmark Results

## Setup

| Parameter | Value |
|-----------|-------|
| GPU | RTX 4090 24GB (AutoDL) |
| Model | Qwen3.6-35B-A3B-UD-IQ2_M.gguf |
| Quantization | IQ2_M (2-bit) |
| Experts/layer | 256, top-8×2 per token |
| Expert size | ~1.01 MB |
| Test domains | math, codegen, debug, logic, general_qa, chinese_tech, creative_write, translate |

## Phase 1: Domain-Expert Affinity

**Key finding: Domain-expert affinity exists at the per-layer level, not globally.**

| Metric | Value |
|--------|-------|
| Global shared experts (all 8 domains) | 256/256 (Jaccard=1.0) |
| Per-layer domain-unique rate | 60-78% (L35 lowest Jaccard=0.480) |
| Universal experts/layer (cross-domain active) | 7-24 |
| Most discriminative expert | Expert 5 (CV=0.877, prefers math) |
| Cross-domain backbone experts (Top-15 ∩) | 10 IDs: [41, 72, 89, 95, 112, 127, 191, 217, 221, 231] |
| Activation Gini coefficient | -0.256 ~ -0.275 (highly uniform) |

## Phase 1.5: LRU Cache Simulation

Simulated 19200 expert accesses across 5 domains with per-layer LRU caches.

| Slots/layer | Pure LRU | +Universal Pin | +Domain Pin | Notes |
|-------------|----------|----------------|-------------|-------|
| 16 | 31.2% | 74.5% | 77.0% | Pin helps most at small capacity |
| 32 | 47.9% | 74.6% | 77.1% | Biggest jump 32→48 |
| 48 | 61.0% | 78.3% | 79.0% | |
| 64 | 80.7% | 82.9% | 83.0% | Diminishing returns |
| 96 | 84.4% | 84.4% | 84.4% | Ceiling — cache fits all domain experts |
| 128 | 84.4% | 84.4% | 84.4% | No improvement |

**Throughput with L0 pool enabled:** 5.5 t/s (vs 5.3 t/s without pool — ~4% overhead)

## Phase 2: Pool Mechanism (CPU Verification)

| Metric | Value |
|--------|-------|
| Pool hit latency (memcpy) | ~1150 µs |
| Pool miss latency (mmap) | ~6500 µs |
| Hit/miss ratio improvement | ~5.6× |
| Minimum viable pool size | 16 MB (~10 expert activations) |
| Pool size impact on generation | None (5.5-5.6 t/s across 64/128/256/512 MB) |
| Swap ratio with pool | ~36% |

## Key Takeaways

1. **84.4% hit rate ceiling** — remaining 15.6% misses are domain-switch cold starts, not capacity issue
2. **Single-domain long conversations → asymptotic ~100%** after warmup
3. **96 slots/layer ≈ 3.9 GB L0 cache** — achievable on 8GB cards
4. **Domain pin is most valuable at small cache** (16-32 slots) — irrelevant at 64+
5. **Bandwidth savings**: 96 slots → ~3.0 GB saved vs no cache (19.4 GB)

Raw data: 48 log files across 8 domains × 3 phases (short/followup/longtail), 224,000+ lines.

### Test data

The full benchmark reports and experimental data are in the repo:

| Document | Contents |
|----------|----------|
| [`docs/expert-clustering-report.md`](docs/expert-clustering-report.md) | Domain-expert affinity verification (DS-V2-Lite & Qwen3.6) |
| [`docs/gpu-benchmark-dsv2lite.md`](docs/gpu-benchmark-dsv2lite.md) | DS-V2-Lite GPU benchmark: 8 domains × 3 phases, VRAM & speed |
| [`docs/gpu-benchmark-dsv2lite-lru-cache.md`](docs/gpu-benchmark-dsv2lite-lru-cache.md) | DS-V2-Lite LRU Expert Cache: baseline comparison across 4 cache fractions |
| [`docs/gpu-benchmark-qwen36.md`](docs/gpu-benchmark-qwen36.md) | Qwen3.6-35B-A3B GPU benchmark: short/followup/longtail per domain |
| [`docs/gpu-benchmark-qwen36-lru-cache.md`](docs/gpu-benchmark-qwen36-lru-cache.md) | Qwen3.6-35B-A3B LRU Expert Cache: baseline comparison across 4 cache fractions |
| [`docs/gpu-benchmark-intermediate.md`](docs/gpu-benchmark-intermediate.md) | Intermediate GPU results during development |
| [`docs/d2h-swap-conclusion.md`](docs/d2h-swap-conclusion.md) | Runtime D2H swap experiment — why it can't save VRAM |

## Reproducing

```bash
# LRU simulator (5-domain trace)
python3 /opt/data/lru_trace_sim.py

# Expert log collection
LLAMA_EXPERT_LOG=1 ./llama-cli -m model.gguf -p "your prompt" -n 128
```

Raw data: 48 log files across 8 domains × 3 phases (short/followup/longtail), 224,000+ lines.
]]>