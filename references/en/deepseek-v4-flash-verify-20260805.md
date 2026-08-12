# DeepSeek V4 Flash Verification Report (updated 2026-08-10)

> ✅ **2026-08-10 4090 re-test (bins-v0.4.0 fixed build, full pipeline)**: V4 measured **35.96 t/s** (on-demand fallback, RSS 17.5GB); selective pin (v4_top100.map) **34.67 t/s** (RSS 26.8GB); VRAM 16.5-16.7GB. Compared with the stock llama.cpp binary (10.1 t/s) at +255%. Full 4-round data in the "08-10 re-test" section below.

> ⚠️ **2026-08-10 benchmark correction**: the 10.1 t/s in this report is measured with the **official stock llama.cpp binary** (on-demand pin path); the moe-l2 optimized binary (selective pin, bins-v0.4.0) measures **35.96 t/s** on RTX 4090 with RSS **17.5-26.8GB**. The 08-05/08-07 data below is kept as historical record.

> Status: **on-demand pin main path runs V4 Flash ✅ (RTX 4090 measured 10.1 t/s, 5x the original 1.7-2.0)**
> Related: `multi-arch-three-gpu-benchmark.md` (V4 full-pipeline re-test section), PyPI 0.7.1 / bins-v0.3.1 (on-demand pin main path)

---

## Latest Results (2026-08-10): bins-v0.4.0 full pipeline, 35.96 t/s on 4090

> **bins-v0.4.0 (selective pin + on-demand fallback) takes V4 from the stock binary's 10.1 t/s to 35.96 t/s (+255%) on RTX 4090**, with RSS down from whole-pin 84GB to 17.5-26.8GB. A 90GB model runs smoothly on a 24GB card + 1TB RAM machine.

### One-line Conclusion

**moe-l2 runs DeepSeek V4 Flash (UD-IQ2_M, 85GB in three shards) on RTX 4090 with bins-v0.4.0 at 35.96 t/s (on-demand fallback, RSS 17.5GB) / 34.67 t/s (selective pin, RSS 26.8GB), VRAM 16.5-16.7GB** (measured 2026-08-10, full 4-round data in the "08-10 re-test" section at the end).

### Measured (RTX 4090 24GB, 2026-08-10, full pipeline `moe-l2 start --gpu`, round 3 stable round)

| Configuration | Gen t/s | VRAM | RSS |
|------|---------|------|-----|
| on-demand fallback (auto router table) | **35.96** (short 34.44 / follow-up 2 35.96) | 16.5 GB | 17.5 GB |
| selective pin (v4_top100.map) | **34.67** (short 33.84 / follow-up 2 34.64) | 16.7 GB | 26.8 GB |
| Stock llama.cpp binary (control) | 10.1 | 17.4 GB | 82 GB |

### Key Conclusions (2026-08-10)

1. **V4 is stable at 34-36 t/s on the 4090** (both modes agree), +255% vs the stock binary
2. **RSS drastically reduced**: whole-pin 84GB → on-demand 17.5GB (↓79%) / selective pin 26.8GB (↓68%)
3. **10.1 t/s is the stock binary's real number**: the earlier "5x speedup" conclusion was based on the stock binary (1.7-2.0 → 10.1); the moe-l2 optimized build was already at the 30-35 t/s level

#### Historical Conclusions (2026-08-07, on-demand pin era, kept for the record)

1. **V4 at 10.1 t/s on 4090 (5x the original 1.7-2.0)**; GPU util 13% → 86%, **already near compute-bound** (further speedup needs kernel/quantization optimization, not cache)
2. **2048 slots is the cache sweet spot** (512 no gain, 4096 OOM), universal gains across three models
3. **RSS 80.9GB (whole-pin full fault)** — no pressure on a 1TB RAM machine; a 128GB container needs an eviction mechanism (v3.1 + unregister) to cap residency
4. Detailed troubleshooting chain: `/opt/data/moe-l2/历史记录文档/on-demand-pin-方案-交接-20260807.md`

### Long Context (500K) Output Speed Q&A (2026-08-07)

> Zhihu question: "What about DS V4's output speed at long context 500K?" — answered honestly as follows.

**Measured (short context, c=512)**: RTX 4090 + moe-l2 (on-demand pin + A3 cache 2048) at **10.1 t/s** (2026-08-07, stock binary basis; 08-10 bins-v0.4.0 measured 35.96 t/s).

**500K long context (not measured, deterministic inference given)**:

- **Output speed barely drops**: MoE's main per-token cost is computing the active experts (6/256), independent of context length; the DeepSeek family uses MLA (KV-compressed attention), so 500K won't blow up per-token attention cost linearly (V4 is designed for 1M context)
- **The real costs are elsewhere (not output speed)**:
  1. **Initial prefill**: processing 500K tokens at once takes a long time
  2. **Memory**: the model itself is 85GB (measured RSS 80-82GB) + 500K KV cache → **system memory starts at 100GB**; GPU VRAM is manageable (8-17GB, VRAM-saving feature), system RAM is the hard constraint
- **In one sentence**: long context isn't "slower", it's "heavier" (eats memory + slow start); sustained conversation output speed stays at the 10 t/s order of magnitude

---

## Model & Hardware

| Item | Value |
|---|---|
| Model | DeepSeek-V4-Flash-UD-IQ2_M (unsloth, 3 shards totaling 85GB, MoE 256 experts / 6 active) |
| 2080 Ti | 11GB SM75, driver 580.105.08, cloud instance region-42 |
| RTX 3080 | 10GB SM86, driver 580.76.05, cloud instance region-41 |
| Binary | v3.1 multi-arch (sm_61/75/86/89/120a, CUDA 12.8, includes fixed-expert-count eviction) |
| moe-l2 | 0.7.0 (PyPI) / bins-v0.3.0 (GitHub Release) |

## Download

- Source: hf-mirror.com unsloth/DeepSeek-V4-Flash-GGUF UD-IQ2_M
- 3 shards: 00001 (5.1MB metadata) + 00002 (46.5GB) + 00003 (38.1GB) ≈ 85GB
- Tool: aria2 -x8 (12 MiB/s, resumable), about 2 hours

## Multi-Shard GGUF Parsing Bug (cli.py fix)

**Problem**: V4 shard 00001 is only 5MB of **pure metadata** (tensors=0). moe-l2 parses the expert layout from shard 1 → `KeyError: No expert tensors found`.

**Fix**: detect the `-00001-of-` format → glob sibling shards in the same directory → pick the **largest shard** for GGUFReader/L2Cache; llama-server still starts from shard 1 (llama.cpp auto-discovers sibling shards). **model_path (server) and reader_path (parsing) dual-path separation**.

## Historical Verification (2026-08-05, v3.1 era, kept for the record)

- 85GB three-shard GGUF runs on 2080 Ti (11GB): VRAM 8.4GB / 11GB, RSS capped by expert-page eviction v3.1 (11-12GB, vs 29GB without eviction) — **validates "10-11GB card can run an 85GB model"** (08-10 bins-v0.4.0 already hit 35.96 t/s on the 4090)
- Multi-shard GGUF parsing bug fix: detect the `-00001-of-` format → pick the largest shard for GGUFReader/L2Cache, llama-server still starts from shard 1 (dual-path separation)
- Expert-page eviction v3.1: `MOE_L2_LRU_MAX_EXPERTS=N` fixed-expert-count LRU, Qwen near-zero speed loss (-2%), V4 RSS capped

## 08-10 Re-test: 4090 Full Pipeline (bins-v0.4.0 fixed build)

> Measured with the bins-v0.4.0 fixed build (downloaded from the GitHub release, includes libmtmd/libllama) on RTX 4090 full pipeline (`moe-l2 start --gpu`), collecting VRAM/memory at the same time. Both modes tested (auto router table generation failed → on-demand fallback; explicit v4_top100.map → selective pin).

### Measured (RTX 4090, 2026-08-10, -c 8192)

| Round | on-demand fallback (short/follow-up 1/follow-up 2) | selective pin explicit table (short/follow-up 1/follow-up 2) |
|------|----------------------------------|----------------------------------------|
| Round 1 | 8.50 / 23.84 / 24.06 | 17.13 / 23.74 / 26.03 |
| Round 2 | 29.60 / 35.24 / 34.91 | 32.99 / 35.24 / 34.89 |
| Round 3 (stable) | **34.44 / 35.96 / 35.96** | **33.84 / 34.67 / 34.64** |
| Round 4 | 32.81 / 36.38 / 35.76 | 33.97 / 34.90 / 31.40 |

### VRAM / Memory

| Mode | RSS | VRAM |
|------|-----|------|
| on-demand fallback | 17.5 GB (18,363,500 kB) | 16.5 GB (16,879 MiB) |
| selective pin (v4_top100.map) | 26.8 GB (28,110,288 kB) | 16.7 GB (17,067 MiB) |

### Conclusions (08-10)

1. **V4 is stable at 34-36 t/s on the 4090** (both modes agree) — vs the stock llama.cpp binary (10.1 t/s) **+255%**; slightly higher than stage 1 (30.9 t/s)
2. **RSS drastically reduced**: whole-pin 84GB → on-demand 17.5GB (↓79%) / selective pin 26.8GB (↓68%)
3. Stage 1's 10.4GB RSS was pure selective pin without prefill configuration; this time GPU cache pre-filling is included, hence higher RSS and higher speed
4. **Auto router table generation works now**: `--router-top-k` depends on `domain_router_map_v4_topics.json` / `domain_router_map_v4.json` under `moe_l2/data/` (shipped with the pip install, tracked in the git repo). Verified 2026-08-10: with the data files in place, auto generation produces **43 layers of top-100** (consistent with the explicit v4_top100.map content), selective pin takes effect (RSS 28.1GB); the earlier 08-10 morning test showing 0 layers was because the test machine hadn't synced the data/ directory — not a product defect. The explicit `--router-map v4_top100.map` (43 layers top-100, local backup in `测试数据备份/v0.8.0-selective-pin-20260810/router-map/`) still works as fallback.

## Environment Pitfalls (for reproduction)

- AutoDL boot driver mismatch (kernel vs library version inconsistency): fix with symlinks like `ln -sfn libcuda.so.580.<kernel version> libcuda.so.1`
- Architecture verification uses CUDA 12.8 `cuobjdump --list-elf` (system's old version misreports)
- proxy non-streaming forwarding httpx timeout 30s→600s (a must for slow models)
