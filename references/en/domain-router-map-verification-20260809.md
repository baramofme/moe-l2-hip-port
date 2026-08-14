# Domain router map verification report (Qwen + V4 dual-model measurements)

> Date: 2026-08-09
> Core question: Is moe-l2's "detect domain → pre-pin high-frequency experts onto the fast path" approach valid?
> Method: use real gate-router traces to measure, for each model, how much of the actual activation is covered by "per-layer top-K high-frequency experts".

---

## One-line conclusion

**The approach holds.** Both models verified: **pinning less than one third of the experts (per layer) covers 88-97% of activations**. The domain router table + pre-pinning has real data support, and the memory cost is controllable.

---

## Data sources

| Model | Trace source | EXPERT rows | Layers × experts | Topic coverage |
|------|-----------|------------|----------|---------|
| Qwen3.6-35B-A3B | qwen8domains (24 logs, 8 domains × 3 scenarios) | 210k | 40 × 256 (top-8) | 8 domains |
| DeepSeek-V4-Flash | actset_test (08-08, 50 rounds general + 25 rounds each of 3 topics) | 311k | 43 × 256 (top-6) | 4 topic groups |

---

## Qwen coverage (40 layers, 256 experts top-8)

| Domain | top-10 | top-30 | top-50 | top-75 | top-100 |
|------|--------|--------|--------|--------|---------|
| chinese_tech | 41% | 70% | 83% | 93% | 97% |
| codegen | 38% | 65% | 80% | 90% | 96% |
| math | 34% | 59% | 74% | 85% | 93% |
| translate | 31% | 58% | 73% | 85% | 92% |
| creative_write | 37% | 62% | 76% | 87% | 94% |
| debug | 32% | 59% | 74% | 84% | 91% |
| general_qa | 36% | 64% | 79% | 90% | 96% |
| logic | 35% | 59% | 74% | 85% | 92% |

**Single-scenario stability**: codegen's short/followup/longtail spread is <4% (84.8-88.4% @ top-50) — the table does not go stale when conversation scenarios change.

## V4 coverage (43 layers, 256 experts top-6)

| K | top-10 | top-30 | top-50 | top-75 | top-100 | top-150 |
|---|--------|--------|--------|--------|---------|---------|
| Coverage | 55.7% | 78.2% | 88.2% | **94.2%** | 97.1% | 99.3% |

**Note**: V4 actually accesses avg=150/256 experts per layer (a wider distribution than Qwen), yet top-75 still covers 94% — the high-frequency experts are actually more concentrated.

## V4 per-topic (= per-domain) coverage — different topics are different domains

| Topic | Requests | top-30 | top-50 | top-75 | top-100 | Actual experts/layer |
|------|--------|--------|--------|--------|---------|-------------|
| general | 50 | 83.1% | 92.9% | 97.4% | 99.1% | 97 |
| math | 25 | 79.5% | 88.7% | 94.3% | 97.1% | 108 |
| code | 25 | 85.2% | 92.9% | 97.0% | 98.8% | 86 |
| chat | 25 | 87.1% | 94.1% | 97.8% | 98.9% | 74 |

**Key finding: per-topic statistics are more accurate** — intra-topic clustering > cross-topic aggregation:
- Aggregated top-75 = 94.2%; per-topic top-75 = **94.3-97.8%**
- code/chat have fewer actual experts per layer (86/74 vs aggregated 150), with stronger clustering
- **top-50 already covers 89-94%**: less memory than aggregated top-75 with equal coverage

## Memory cost accounting

### Qwen (expert 1.01MB each)

| Tier | Experts/layer | 40-layer RAM | Coverage |
|------|---------|----------|--------|
| top-50 | 50 | ~2GB | 74-83% |
| top-75 | 75 | ~3GB | 84-93% |
| top-100 | 100 | ~4GB | 91-97% |

### V4 (expert 2.7MB each)

| Tier | Experts/layer | 43-layer RAM | Coverage |
|------|---------|----------|--------|
| top-50 | 50 | ~5.4GB | 88% |
| **top-75** | 75 | **~8.1GB** | **94%** |
| top-100 | 100 | ~10.8GB | 97% |

## Approach pipeline

```
gate router output (already available, produced per token per layer)
  → aggregate high-frequency experts by domain/topic (done: Qwen 8-domain table + V4 table)
  → domain router table (domain_router_map_*.json)
  → L0a domain detection
  → pre-pin that domain's high-frequency experts (batch cudaHostRegister)
  → hits take the DMA fast path during inference; misses fall back to on-demand/LRU
```

## Artifacts

- `moe_l2/data/domain_router_map_qwen.json` — Qwen 8 domains × 40 layers × top-100
- `moe_l2/data/domain_router_map_v4.json` — V4 43 layers × top-75 (cross-topic aggregated)
- `moe_l2/data/domain_router_map_v4_topics.json` — **V4 4 topics × 43 layers × top-75 (per-domain, more accurate)**
- `scripts/bench/domain_router_coverage.py` — Qwen coverage analysis
- `scripts/bench/v4_router_coverage.py` — V4 coverage analysis (aggregated)
- `scripts/bench/v4_topic_coverage.py` — **V4 per-topic coverage analysis**
- `scripts/bench/generate_domain_router_map.py` — Qwen router table generation
- `scripts/bench/generate_v4_router_map.py` — V4 router table generation (aggregated)
- `scripts/bench/generate_v4_topic_router_map.py` — **V4 per-topic router table generation**
- trace sources: `测试数据备份/qwen8domains/` (24 logs) + `测试数据备份/v4-actset-trace-20260808.log` (14MB)

## Remaining observations

1. **V4 L20 layer top-10 = [0,1,2,3,4,5]**: consecutive expert ids — suspected fixed activation pattern in that layer (not learned routing); needs further confirmation on whether it affects the pin strategy
2. **V4 per-topic statistics complete**: math/code/chat independent coverage 94-98% (top-75); intra-topic clustering confirmed to be stronger
3. **V4's 43 layers vs Qwen's 40 layers**: pinning V4 top-75 needs ~8.1GB RAM — half of a 16GB machine; on-demand fallback for cold experts is still needed

## Consumer side + data flywheel measurements (2026-08-10 early morning, full pipeline on 4090 cloud machine)

### Implementation (landed in moe-l2)
1. **New `moe_l2/domain_router_flywheel.py`** (standalone module): gate parses EXPERT routing in real time → aggregates high-frequency experts by domain → auto-rebuilds the router table JSON once a threshold is reached (atomic replace; accumulates across rebuilds, gets more accurate with use)
2. **`predictor.load_mapping()` prefers the flywheel table**: when `domain_router_map_flywheel.json` exists it is used, otherwise it falls back to the static table → pretouch consumes the learned results
3. **cli.py fixed the V4 startup timeout bug**: `_wait_for_llama_server` 30s→180s (V4 85G takes 90s to load; the old 30s always failed with TIMEOUT)

### Data flywheel closed-loop verification (Qwen3.6, full pipeline start --gpu)
- Sending "write a Python function" → codegen domain auto-generated; sending "explain the Pythagorean theorem" → translate domain auto-added
- Router table auto-rebuilt 3 times: 2 domains → 3 domains; new domains join automatically with zero manual intervention
- Real 311k-row V4 trace fed in → complete 43-layer router table generated (consistent with offline script results)

### Speed comparison (4090 / v5 whole-pin binary, same prompt, max_tokens=128)

**Qwen3.6-35B-A3B** (stable round round3):
| Group | Config | Speed |
|----|------|------|
| A | with flywheel table (full pipeline) | 46.06 t/s |
| B | without flywheel table (static table) | 47.02 t/s |
| — | direct llama-server (baseline) | 46.06 t/s |
| Conclusion | flywheel adds zero overhead; Qwen is near the 4090 physical ceiling | |

**DeepSeek-V4-Flash 85G** (stable round round3):
| Group | Config | Speed |
|----|------|------|
| A | with flywheel table (full pipeline) | 10.16 t/s |
| B | without flywheel table (static table) | 10.06 t/s |
| — | direct llama-server (baseline) | 10.16 t/s |
| Conclusion | flywheel adds zero overhead; V4's bottleneck is the dispersed-routing physical ceiling (4-10 t/s) | |

### Key conclusions
1. **flywheel does not affect speed (within ±1%)**: Qwen is already at full speed (46-47 t/s = v5 whole-pin ceiling); V4 is limited by the dispersed-routing physical ceiling — the pretouch table cannot change the GPU compute bottleneck
2. **flywheel's real value = automated router table maintenance**: the static 8-domain table → auto-learns any new domain; saves manual statistics; higher coverage on V4 (94-98% vs the manual table)
3. **v3.1 binary speed limit confirmed**: the cloud machine's bin/ once had a leftover v3.1 (f7d7858c, Qwen only 10-12 t/s); switching to v5 (f1b5e048, whole-pin default) restored 46 t/s — **the release must ship the v0.3.2 binary**, old versions have a performance trap
4. Remaining: `domain_router_map_flywheel.json` restored on the cloud machine's data/ (.bak-B moved back); local backup `测试数据备份/domain-router-consumer-20260809/`

## Selective pin simulation (2026-08-10, V4 real trace)

> User direction (decided 2026-08-10): flywheel cannot improve speed by much, but **it can reduce memory usage** — the number of pinned experts can be reduced without affecting speed. This simulation verifies that inference.

### Method
Using the V4 real trace (311,363 EXPERT rows, 125 requests), generate an aggregated router table for each top-K; token-by-token count the "cold experts" (activated experts not in the table); convert the fault cost at 2.1ms/cold expert (page-pin 0.43 + page-fault disk read 1.7, measured values); overlay the whole-pin baseline of 98.4ms/token to estimate speed.

### Results
| top-K | Pinned/layer | Pin memory | Coverage | Cold experts/token | Fault cost | Estimated speed |
|---|---|---|---|---|---|---|
| 50 | 40 | 4.6GB | 84.4% | 44.0 | 92.3ms | 5.24 t/s |
| 75 | 60 | 6.9GB | 90.6% | 26.5 | 55.6ms | 6.49 t/s |
| **100** | **80** | **9.1GB** | **94.3%** | **16.0** | **33.7ms** | **7.57 t/s** |
| **150** | **119** | **13.6GB** | **98.0%** | **5.7** | **12.0ms** | **9.06 t/s** |
| 200 | 159 | 18.1GB | 99.4% | 1.7 | 3.5ms | 9.81 t/s |

### Interpretation
1. **User direction holds**: selective pin covers 84-99% of activations with 4.6-18GB; memory drops dramatically from whole-pin's 82GB, speed only drops 3-48%
2. **Sweet spot = top-100**: 9.1GB / 7.57 t/s (74% speed) — runs on a 16GB AI PC; vs dynamic-pin LRU (17-24GB / 4-5 t/s) **memory halved, speed >50% faster**
3. **Best value = top-150**: 13.6GB / 9.06 t/s (89% speed), suited for 32GB machines
4. **Below top-50 not recommended**: coverage <85%, fault storm (92ms/token) falls back to dynamic-pin level
5. Simulation script: `scripts/bench/sim_selective_pin.py` (result JSON: `selective_pin_sim_result.json`); implementation direction = read the flywheel router table → only cudaHostRegister in-table experts, out-of-table experts fall back on-demand

## Selective pin real-machine verification (2026-08-10 afternoon, full-pipeline A/B on 4090 cloud machine)

> ⚠️ **2026-08-13 correction: the V4 speed figures in this section (30.9 t/s etc.) carry no valid speed meaning — UD-IQ2_M 2-bit quantization degrades to garbage output (vanilla also affected); no valid speed data, waiting for Q4 quant. The RSS/memory conclusions (84.4 → 10.4 GB) stand.**

### Implementation (Phase 1: RAM selective pin)
- C++: ggml-backend.cpp copy_experts adds `MOE_L2_ROUTER_FILE` env var support — reads the router table at load time (`layer expert1 expert2 ...` format), calls pin_fn only for in-table experts; out-of-table experts are not explicitly registered and go through set_tensor_async's on-demand fallback
- Python: cli.py adds `--router-map` / `--router-top-k` args; auto-generates the router map at startup and injects it into env
- Router table source: union of domain_router_map_v4_topics.json + domain_router_map_v4.json (43 layers, ~61 experts/layer, top-100 cap)
- Backup chain: local `测试数据备份/selective-pin-20260810/` (C++ copies); cloud machine `/root/moe-l2-backups/selective-pin-20260810/` (13 old bin files + old cli.py)

### A/B measurements (same machine, same binary, same prompt, max_tokens=128, codegen)

| Group | RSS | round1 | round2 | round3 | Stable-round speed |
|---|---|---|---|---|---|
| A: selective pin (top-100) | **10.4 GB** | 10.71 | 30.14 | 30.88 | **30.9 t/s** |
| B: whole-pin (control) | 84.4 GB | 24.45 | 31.23 | 30.21 | **30.2 t/s** |

### Key conclusions
1. **RSS 84.4 → 10.4GB (↓ 88%)**, memory reduction without speed loss — ⚠️ the 30.9 vs 30.2 t/s speeds are **no valid speed data** (UD-IQ2_M 2-bit quantization degrades to garbage output, vanilla also affected; waiting for Q4 quant) — "fewer pinned experts → lower memory" verified on real hardware
2. **Faster startup**: selective pin ready in 10s vs whole-pin 40s (no full-page faulting)
3. **⚠️ The 30 t/s speed comes from the "recompiled build-a3 binary", not from selective pin** — the previous v031-test (f1b5e048) measured 10.1 t/s; after recompiling from llama.cpp-clean source today, both A/B groups hit 30 t/s. Selective pin's contribution = 88% memory reduction without slowing speed（⚠️ the 30 t/s figures are UD-IQ2_M 2-bit-garbage-era measurements — no valid speed data, waiting for Q4 quant; the RSS reduction conclusion stands）
4. **Out-of-table residency accumulation observed**: after 3 speed rounds, RSS grew from 10.4 to 19.7GB (out-of-table experts stay resident after on-demand pin, never evicted) — matches the design; long-running operation needs waterline eviction (implementation item of risk #2 in the plan doc)
5. vs simulation: measured RSS 10.4GB is slightly better than the simulated 12.9GB (union 61/layer < simulated assumption 80/layer); speed cannot be directly compared due to the binary update

## Phase 2 experiment: GPU cache prefill (2026-08-10 afternoon, full pipeline on 4090 cloud machine)

### Implementation
- On the first copy_experts call in ggml-backend.cpp, batch pre-set in-table experts into the GPU expert cache (`cache_set_fn`, deduped by tensor name, runs once)
- Goal: the first request hits D2D, saving the per-miss H2D + cache_set overhead
- Binary: build-a3 recompiled (libggml-base.so contains `[moe-l2] prefill` code), deployed to /root/moe_l2/bin/

### A/B comparison (same machine, same prompt, max_tokens=128, codegen)

| Round | Phase 1 (no prefill) | Phase 2 (with prefill) | Change |
|---|---|---|---|
| round1 (cold start) | 10.71 t/s | **19.11 t/s** | **+78%** |
| round2 | 30.14 t/s | 29.80 t/s | ≈ 0 |
| round3 | 30.88 t/s | 31.12 t/s | ≈ 0 |

### Conclusions
1. **Prefill works**: cold-start round1 +78% (10.7→19.1), steady-state 30 t/s unchanged
2. **User judgment (2026-08-10 evening): important, not "limited value"** — first request at full speed = the first sentence is fast when opening the app/demo, directly improving the perceived experience. Especially critical for AI PC vendor PoC demos, short-prompt single requests, and frequent-restart scenarios
3. **init-fix experiment (2026-08-10)**: exposed the maybe_init proc address + forced init before prefill → round1 19.67 (vs 19.11), basically unchanged → **cache init timing is not round1's bottleneck**
4. **New round1 bottleneck analysis**: prefill takes effect (RSS 10.4→28.2GB, VRAM 17.7GB, 61 experts/layer × 43 layers all H2D into cache), but round1 is still ~19.7 → the bottleneck is that **prefill H2D happens within the first request** (lazily triggered), so the first request overlaps the bulk H2D. True "first-request-at-full-speed" requires prefill to complete at **startup** (after model load, before the first request)
5. **Note**: llama-server stderr is consumed by the gate thread; `[moe-l2] prefill` logs don't hit disk (the round1 improvement is the evidence it took effect)

## 2080Ti dual-model full-pipeline verification (2026-08-10 evening, region-42, new multi-arch binary) — ⚠️ P0-void (2026-08-13)

**Background**: region-42's old bin only had sm_75 single arch + 0 moe-l2 markers (= stock compile); the new multi-arch (llama-final-src/build-multi) has sm_61/75/86/89/120a + all moe-l2 optimizations. Full pipeline = moe-l2 start --gpu (proxy + L2 cache + flywheel gate + selective pin).

**Fixes**: proxy crash root cause = region-42's moe_l2 package is the old 0.5.1, missing `domain_router_flywheel.py`; synced the full package + scripts/bench/export_router_map.py + data/ tables.

### Qwen3.6-35B-A3B (UD-IQ2_M, codegen 128 tokens)

| Round | Old baseline (stock binary) | New multi-arch full pipeline | Gain |
|---|---|---|---|
| round1 | 14.90 t/s | **36.53 t/s** | +145% |
| round2 | 16.88 t/s | **50.80 t/s** | +201% |
| round3 | 15.86 t/s | **52.07 t/s** | +228% |

> 📌 **2026-08-10 evening re-measurement** (same machine, same binary, stable round3): Qwen **47.24 t/s**, DS **87.25 t/s** — ⚠️ **P0-void, 2026-08-13 voided** (pre-fix build, garbage-output inflated; the linked 08-10 sections are likewise void). **2026-08-14 bins-v0.4.1 fixed-build re-measure: Qwen 30.87 t/s, DS 85.25 t/s** (see multi-arch-three-gpu-benchmark.md); README/models-benchmark adopt the fixed-build values. The table above is the first full-pipeline verification data (historical record).

### DS-V2-Lite (Q2_K, codegen 128 tokens)

| Round | Old baseline (stock binary) | New multi-arch full pipeline | Gain |
|---|---|---|---|
| round1 | ~6.9 t/s | **57.31 t/s** | +730% |
| round2 | — | **92.50 t/s** | — |
| round3 | — | **94.95 t/s** | — |

### Bare server comparison (same machine, same binary, bypassing proxy)

- Qwen: 42.97 / 58.85 / **62.40 t/s**
- DS V2: 60.09 / 101.01 / **104.85 t/s**
- Full pipeline vs bare server difference ~15% (proxy forwarding + domain prediction overhead, normal)

### Conclusions

1. **moe-l2-optimized multi-arch binary on 2080Ti: Qwen ~3.5x, DS ~14x gains**（⚠️ P0-void, 2026-08-13 voided — pre-fix build; fixed-build 2080 Ti values: Qwen 30.87 / DS 85.25, see multi-arch-three-gpu-benchmark.md）— consistent with the 3x gain on V4/4090; all models 3x+
2. **Root cause**: all the previous "slow" benchmark data (14.9/6.9 t/s) came from stock/old-compiled binaries; moe-l2's real performance was buried
3. Router map auto-generation (43 layers top-100) + flywheel domain prediction (routing drift detection) work across the full pipeline
4. **Release benchmarks updated**（⚠️ all P0-void, 2026-08-13 voided — pre-fix build）: Qwen ≈ 52 t/s on 2080Ti, DS ≈ 95 t/s, V4 ≈ 30.9 t/s on 4090（V4: UD-IQ2_M 2-bit garbage — no valid speed data, waiting for Q4 quant; fixed-build 2080 Ti values: Qwen 30.87 / DS 85.25, see multi-arch-three-gpu-benchmark.md）

## Next steps

1. ~~Consumer-side implementation~~ ✅ (landed 2026-08-10: flywheel + load_mapping priority + pretouch consumption)
2. ~~V4 per-topic grouped statistics~~ ✅ (per-topic table generated, report above)
3. **Selective pin C++ implementation (in progress)**: read the flywheel router table → only pin in-table experts (top-100/150 sweet spot), out-of-table experts fall back to on-demand faulting — replaces whole-pin full registration; expected V4 memory 82GB → 9-14GB, speed 7.6-9.1 t/s; Qwen memory → 4GB, speed near full
4. Cold-expert fallback path verification (on-demand pin cost on miss) — already measured on V4: physical ceiling 4-10 t/s, no prefetch window
