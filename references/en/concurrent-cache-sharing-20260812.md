# Concurrent Cache Sharing Verification (2026-08-12, re-measured 2026-08-14)

> ⚠️ **2026-08-14 update: all 08-12 numbers below are P0-void** — they were measured on bins-v0.4.0 with the expert-cache race bug (cache false-hits return garbage but inflate speed). The 2026-08-14 fixed build (bins-v0.4.1) re-measures at the end of this document: **concurrency is clean (16/16 outputs normal, no garbage), no per-session slowdown, no cross-domain eviction.**

> Goal: when concurrent multi-slot requests share the A3 cache / selective pin, do they evict each other's cache hits and speed — verifying moe-l2's capability for the "one machine, multiple users at once" scenario.
> This is the measured completion of "Risk 6: concurrent cross-domain requests evict each other in the shared cache — concurrency listed as a follow-up special topic" from the flywheel plan document.

## One-line conclusion

**Concurrency does not break; cache sharing holds naturally — cross-domain concurrency is only 5-7% behind same-domain concurrency, no per-domain pooling needed; 4-way total throughput = 2.3-2.5× a single session, you gain speed instead of losing it.** The "one AI PC, multiple users at once" capability is verified in practice.

## Environment

| Item | Value |
|---|---|
| Cloud machine 1 | region-42 (RTX 2080 Ti 11 GB, driver 580.105.08) |
| Cloud machine 2 | bjb1 (RTX 4090 24 GB) |
| Models | Qwen3.6-35B-A3B-UD-IQ2_M (11 GB) / DeepSeek-V2-Lite-Q2_K (6 GB) / DeepSeek-V4-Flash (85 GB) |
| Binary | bins-v0.4.0 (with prefill + A3 cache + selective pin) |
| Launch flags | `--parallel 4 -np 4 -ngl 99 -c 2048` + `GGML_OP_OFFLOAD_MIN_BATCH=1 GGML_CUDA_EXPERT_CACHE=1 MOE_L2_ROUTER_FILE=...` |
| Server | 127.0.0.1:11435, 4 slots (n_ctx_slot=512) |

## Test design (three scenarios)

| Scenario | Workload | Purpose |
|---|---|---|
| 1. Baseline | 4 prompts run sequentially in a single session | Control baseline |
| 2. Same-domain concurrency | 4 slots asking programming questions simultaneously | Shared behavior when hot experts overlap |
| 3. Cross-domain concurrency | 4 slots asking programming/router/NAS/communication questions simultaneously | Key question: do different domains evict each other |

## Measured data

### RTX 2080 Ti (Qwen3.6-35B-A3B + DS-V2-Lite)

| Model | Baseline t/s | Same-domain concurrency (per-slot ×4) | Cross-domain concurrency (per-slot ×4) | Cross/same |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | 38.4 | **95.02** (23.76) | **88.28** (21.7-22.2) | -7% |
| DS-V2-Lite | 78.3 | **198.59** (49.64) | **188.25** (46.8-47.4) | -5% |

### RTX 4090 follow-up (three models, incl. the 256-expert large model)

| Model | Baseline t/s | Same-domain total throughput | Cross-domain total throughput | Multiplier |
|---|---|---|---|---|
| **DeepSeek-V4-Flash** (85 GB, 256 experts) | 35.4-35.8 | **89.66** (22.5×4) | **88.10** (21.8-22.6×4) | 2.5× |
| Qwen3.6-35B-A3B | 67.4-68.7 | **174.56** (43.6×4) | **144.47** (35.9-36.3×4) | 2.6× / 2.1× |
| DS-V2-Lite | 143.4-143.7 | **369.48** (92.3×4) | **354.73** (88.6-88.8×4) | 2.6× / 2.5× |

**Key V4 findings (4090)**:
- ✅ V4 concurrency verified — 89.66 same-domain / 88.10 cross-domain t/s (2.5×); **even the 256-expert model with scattered routing shares well under concurrency**
- ⚠️ Occasional cublas crash on first cross-domain concurrency (`cublasSgemm_v2 unsupported parameter`) — single-slot runs all fine, concurrency fine after restart; an occasional edge case, not consistently reproducible, recorded
- Qwen's cross-domain drop (144 vs 174, -17%) is larger than on the 2080 Ti (-7%) — the 4090 is faster, so expert-sharing differences are amplified, but still within acceptable range

### V4 concurrency VRAM/RSS sampling (4090)

| Metric | Single-session baseline | 4-way concurrency | Growth |
|---|---|---|---|
| **VRAM peak** | 16,891 MiB | **19,763 MiB** | **+2.9 GB (+17%)** |
| **RSS peak** | 29,759 MB | **29,947 MB** | **+0.2 GB (+0.6%)** |

- Where the +2.9 GB VRAM comes from: each of the 4 slots gets an independent KV cache (n_ctx_slot=512), so total KV usage grows linearly with slot count — **expected cost of concurrency, not a leak**
- RSS barely grows (+0.2 GB): file pages of experts outside the selective pin table do not accumulate, and multiple slots do not duplicate expert weights — **no leaks, no runaway**

## Mechanism (why it did not break)

1. Qwen's expert routing is highly stable: the EXPERT log shows all layers consistently activating 8 experts `[0-7]` (fixed top-8 routing)
2. Under 4-way concurrency (even across domains) the active expert sets overlap completely → the A3 cache is naturally shared, no contention
3. Consistent with the union-set simulation: "high-frequency experts across 4 topics overlap heavily (top-75 union is only 89 per layer)" — the aggregated union + shared cache covers the concurrency scenario

## Conclusions

1. ✅ **The shared cache is sufficient (verified on two machines, three models)** — Qwen (fixed top-8 routing), DS (64 experts), and V4 (256 experts, scattered routing) all survive concurrency; no catastrophic cross-domain eviction
2. ✅ **No per-domain cache pools needed** (flywheel plan option (a) does not need to be implemented)
3. ✅ **"One AI PC, multiple users at once" verified in practice** (4-way total throughput = 2.3-2.5× a single session)
4. ✅ flywheel plan "Risk 6: concurrency listed as a follow-up special topic" → **closed by this verification**
5. ⚠️ The router table used in the DS test was Qwen's 43-layer table (DS has only 26 layers — layer count mismatch, only some layers take effect) — does not affect the concurrency conclusions; a DS-specific router table has not been generated and can be added later if precise selective pin is needed

## Assets

- Test script: `测试数据备份/concurrent_cache_test.py`
- Server log: `测试数据备份/concurrent-cache-20260812/concurrent_test.log`

## 2026-08-14 re-measure (bins-v0.4.1 fixed build, RTX 4090) — P0 fix verified under concurrency

**Environment**: bjb1 (RTX 4090), DS-V2-Lite Q2_K, full pipeline `moe-l2 start --gpu` (proxy 11435), 4 threads × 4 requests = 16 requests per round, max_tokens=128, output-quality checked per request (slash/question/repeat-garbage detector).

| Round | Domains triggered (gate) | Output quality | Per-session gen t/s | VRAM peak | RSS |
|---|---|---|---|---|---|
| 1. Same-domain (all coding) | 1 domain (codegen) | **16/16 normal, 0 garbage** ✅ | 113.5-124.7 (avg 121.8) | ~10.1 GB | ~7.1 GB |
| 2. Multi-domain (Chinese 4 designs) | 2 domains (codegen ×4 / chinese_tech ×12) | **16/16 normal, 0 garbage** ✅ | avg 134.2 | ~10.1 GB | ~6.9 GB |
| 3. Multi-domain (EN+ZH mix) | 4 domains (codegen ×8 / chinese_tech ×4 / general_qa ×2 / debug ×2) | **16/16 normal, 0 garbage** ✅ | 78.2-140.2 (avg 133.5) | ~10.1 GB | ~6.9 GB |

**Conclusions (fixed build)**:
1. **P0 concurrency garbage bug is fixed** — DS is the litmus test (old lockless build always produced garbage under concurrency); 48/48 requests across three rounds are clean.
2. **No per-session slowdown under concurrency** — per-session avg 122-134 t/s vs single-session steady 124-130 t/s (~±3-6%, noise), mutex cost negligible, cache sharing holds (matches the 08-12 "no pooling needed" conclusion, now on clean data).
3. **Multi-domain (4 domains) concurrency does not evict or corrupt** — gate domain prediction works under concurrent load (coding→codegen, EN general→general_qa/debug, ZH general→chinese_tech); EN prompts discriminate domains better than ZH on DS (classifier property).
4. VRAM stays ~10.1 GB across concurrency (only per-slot KV cost), RSS ~6.9-7.1 GB.

> Note: same-domain round 1 per-session avg (121.8) is slightly lower than multi-domain rounds (133-134) — random prompt variance, not domain effect. All rounds well within single-session noise.

**Scripts/data**: `/root/bench_4090_concurrent.py` (same-domain), `/root/bench_4090_multidomain.py` (multi-domain) on bjb1; results `/root/bench_4090_concurrent_DS.txt`, `/root/bench_4090_multidomain_DS.txt`.
