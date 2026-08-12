# moe-l2 Technical White Paper

**Run 100B+ parameter MoE models on 8GB GPUs**

- Version: v1.0 (2026-08-02)
- Author: moe-l2 project team (yalun753)
- Audience: technical decision-makers, architects, AI infrastructure teams
- Companion docs: README (quick start), references/ (raw benchmark reports), [design-decisions.md](design-decisions.md) (design evolution history: why host-buffer in the end)

---

## 1. Executive Summary

moe-l2 is an MoE (Mixture of Experts) inference acceleration solution for consumer NVIDIA GPUs. Its core capability is **running models that would normally need 16-24 GB of VRAM in 1.6-3.4 GB** (1.6-2.1 GB for short contexts, 2.4-3.4 GB for 8K long contexts), at roughly 58% of full-GPU speed.

Core numbers (measured on RTX 4090, 2026-08-07, on-demand pin main path):

| Metric | Full-GPU form | moe-l2 | Change |
|------|------------|--------|------|
| VRAM usage | 23.3 GB | **1.6-2.9 GB** | **-93%** |
| Generation speed (DS-V2-Lite) | 65 t/s | **145.63 t/s** | 224% (beats full GPU) |
| Generation speed (Qwen3.6-A3B) | — | **74.99 t/s** | beats pre-lazy 46.5 |
| V4-Flash (157B/85GB, 4090) | OOM | **35.96 t/s** | RSS 17.5-26.8GB (on-demand/selective pin) |
| Model/VRAM ratio | 0.26× | **3.9×** | +15x |

> **Long-context supplementary measurements (2026-08-09, v5 default whole-pin, RTX 4090, -c 8192, 3200-token long generation)**: Qwen3.6-A3B **45.1 t/s @ 2.4 GB (2477 MiB)**, DS-V2-Lite **34.1 t/s @ 3.4 GB (3441 MiB)** — 8K context + long generation is closer to real usage; short-context short-generation reaches the 74.99 / 145.63 t/s in the table above (measured 2026-08-10 with bins-v0.4.0). Raw sampling data and full generated text are in the demo asset package (rec_data.csv / rec_full.txt).

> **Multi-arch full compatibility (2026-08-10, bins-v0.4.0)**: one binary supports all consumer NVIDIA cards (GTX 1080 sm_61 → RTX 50 series sm_120a, compiled with CUDA 12.8). 2080 Ti full-pipeline measurements (bins-v0.4.0 + selective pin): DS-V2-Lite **87.25 t/s**, Qwen3.6-A3B **47.24 t/s** (2026-08-10, moe-l2-optimized multi-arch binary; the earlier 6.89/11.15 were official stock-binary numbers). Full report: [multi-arch-three-gpu-benchmark.md](multi-arch-three-gpu-benchmark.md). v0.4.0 includes selective pin + GPU cache prefill + flywheel router tables.

One-sentence technical essence: **each MoE inference step activates only a few experts (top-2~8 out of hundreds); keeping all of them resident in VRAM is wasteful. moe-l2 keeps experts in CPU memory (zero VRAM); the GPU fetches only the activated experts each step, and hot experts stay cached on the GPU to avoid PCIe round trips.**

Implementation path: no inference-engine rewrite — two patches on top of llama.cpp (host-buffer expert residency + scheduler copy optimization); users just `pip install`.

---

## 2. Problem and Opportunity

### 2.1 MoE is the mainstream architecture for large models, but inference is gated by VRAM

Mixture of Experts splits the model into "shared layers + hundreds of experts"; each inference step the gating network (router) activates a few of them. DeepSeek-V2 (236B), Qwen3-235B, and Mixtral all use this architecture.

MoE's advantage is **large total parameters, small activated parameters** — hundreds of billions of total parameters, with only a small subset of experts (top-2~8) activated per step.

But real-world deployment is gated by VRAM:

- Model weights (even Q4 quantized) must be loaded whole to run: Qwen3-30B Q4 ≈ 16+ GB, DeepSeek-V2 Q4 is on the order of a hundred GB
- Mainstream consumer VRAM: 8 GB (RTX 4060/3060), 12 GB (4060 Ti), 16 GB (4070 Ti Super)
- **8 GB GPUs can't even load Qwen3-30B**, let alone the 100B+ DeepSeek-V2 / Qwen3-235B

The result: MoE capabilities are locked behind the "VRAM gate" on high-end servers, and consumer hardware can only run 7B-class small models.

### 2.2 Shortcomings of existing approaches

| Approach | How | Shortcomings |
|------|------|------|
| llama.cpp CPU offload | if it doesn't fit in VRAM, put it in system memory | slow (PCIe bandwidth bottleneck), 3-5x speed loss |
| MoE-Infinity / ExpertFlow | complete engineering systems for server clusters | too heavy; requires clusters/training; not for individual users |
| GPU-CPU collaborative inference | CPU does cache-miss inference | CPU inference is slow, wastes resources |
| Buy more VRAM | 24 GB+ GPUs | costs 8000+ RMB, and 24 GB still isn't enough for a 236B model |

### 2.3 moe-l2's opportunity

**Target users**: 8 GB GPU users (one of the largest Steam GPU share segments), individual developers, home server (NAS) users, AI PC vendors.

**Value proposition**:

1. **93% VRAM reduction**: 23.3 GB → 1.6 GB (short context) / 3.4 GB (8K long context); 4 GB cards run 16B MoE, 8 GB cards run 32B MoE
2. **Usable speed**: 75-146 t/s generation (4090, measured 2026-08-10 with bins-v0.4.0; 47-87 t/s on 2080 Ti), far beyond reading speed, smooth conversational experience
3. **Zero migration**: OpenAI-compatible API; curl / Open WebUI / LangChain connect directly; no client code changes
4. **Plug and play**: pip install + one command; no llama.cpp compilation

---

## 3. Architecture Principles

### 3.1 Core insight

MoE inference memory-access characteristics:

- The model splits into two tiers: **dense layers** (attention + shared layers, computed every step) and **expert layers** (hundreds of experts, only top-2~8 activated per step)
- Full-residency approach: dense + all experts in VRAM → VRAM requirement = full model size
- moe-l2 approach: **dense layers resident in VRAM (small), experts resident in CPU memory (large but zero VRAM), only activated experts are moved into the GPU each step**

Analogy: an MoE model is like a huge library where every book (expert) is heavy. The traditional approach moves the entire library into the office (VRAM) before working; moe-l2 keeps only "the few commonly used books" at hand and fetches from the warehouse (memory) on demand, returning them when done. The bookshelf (VRAM) only holds a catalog (dense layers) and the recently used books.

### 3.2 System layering

```
User input
   │
   ▼
┌──────────────────────────────────────────┐
│  moe-l2 scheduler (Python process, port 11435) │
│                                          │
│  ├─ L0a domain predictor ── keyword / TF-IDF /   │
│  │    semantic embedding, three-tier fallback, <1ms │
│  │    classification                        │
│  │    output: domain label (codegen/math/...) │
│  ├─ L2 hot cache manager ── mmap shared-memory LRU │
│  │    preloads "that domain's common experts" to memory │
│  ├─ data flywheel ── accumulates samples from real │
│  │    traffic, auto-retrains classifier (more accurate │
│  │    with use)                                │
│  └─ request forwarding ── OpenAI-compatible passthrough │
└───────────────┬──────────────────────────┘
                │ POST /v1/chat/completions
                ▼
┌──────────────────────────────────────────┐
│  llama-server (port 11436, CUDA GPU)     │
│                                          │
│  1. experts resident in CPU pinned memory │
│     (host buffer, zero VRAM)             │
│  2. scheduler copies only activated       │
│     experts → GPU each step              │
│  3. GPU cuBLAS computes experts directly  │
│     (fast path)                          │
│  4. optional sched-cache: hot experts D2D │
│     (no PCIe round trip)                 │
└──────────────────────────────────────────┘
```

### 3.3 Key mechanism 1: host-buffer expert residency (zero VRAM)

**Design** (moe-l2's llama.cpp patch):

- Modify `llama-model-loader.cpp` to relax mmap → **CUDA host buffer fallback**: expert weights load into CPU pinned memory (readable directly by the GPU via PCIe DMA), **using zero VRAM**
- Dense layers (attention and other non-expert parts) stay resident in VRAM as usual
- Result: VRAM only holds dense layers + KV cache + temporary buffers

**Important distinction**: this step is moe-l2's design (loader patch); "copy only activated experts per step" is llama.cpp's scheduler built-in MoE expert-level copy optimization — moe-l2 ensures this fast path via the `GGML_OP_OFFLOAD_MIN_BATCH=1` env var.

**Data path** (per inference step):

```
CPU pinned (expert weights, zero VRAM)
   │  sched copy optimization: copy only the top-k experts
   │  activated this step (e.g. 6 × 1.55 MB ≈ 9.3 MB)
   │  instead of all 64 experts
   ▼
GPU VRAM (temporary expert buffer)
   │  cuBLAS computes MUL_MAT_ID directly (fast path)
   ▼
output
```

**Why fast**: expert weights stay resident in CPU memory (mmap lazy + on-demand pin registration); each step copies only the activated top-k experts (e.g. DS's 6 per layer × 1.55 MB ≈ 9.3 MB) instead of all 64 experts — copy volume is proportional to the activated count, not the model size. Measured gains (2026-08-10, bins-v0.4.0, 4090): DS 37.9 → 145.63 t/s (+284%), Qwen 50.2 → 74.99 t/s (+49%).

### 3.4 Key mechanism 2: A3 LRU expert cache (no moving hot experts)

When a user converses continuously within one domain (e.g. continuous coding), the experts activated per step are highly repetitive. moe-l2's A3 LRU cache maintains a fixed-size expert cache pool in GPU VRAM:

- **Hit**: hot expert already in VRAM → D2D copy (no PCIe round trip) → microsecond-level
- **Miss**: CPU → GPU copy + cache write-back

Cache mount point: llama.cpp scheduler's `copy_experts` (expert input copy layer) — lower-level than the compute layer, intercepting before experts enter the GPU.

**Benefit pattern** (three models measured): **cache benefit = expert size × hit rate**

| Model | Expert size | Activation | cache benefit |
|------|---------|---------|-----------|
| DeepSeek-V2-Lite | 1.55 MB | top-6 | **Prompt +211%, Gen +5%** |
| Qwen3.6-A3B | ~1 MB | top-8 | none (experts too small, moving them is cheap) |
| Mixtral-8x7B | 252 MB | top-2 | none (hit rate too low, and slots use lots of VRAM) |

**Theoretical basis** (LRU simulation, Phase 1.5):

- 5-domain trace: 96 slots/layer = 84.4% hit rate (near theoretical ceiling)
- 8-domain validation: pure LRU at 96 slots = 80.4%, 128 slots = 85.2%; Domain Pin needs ≥64 slots for gains
- Single-domain long conversation: once all experts are loaded, hit rate asymptotically approaches 100%
- Remaining miss source: **domain-switch cold start** (the new domain's experts have never appeared), not capacity shortage

### 3.5 Scheduling layer: domain prediction + data flywheel

moe-l2's differentiation is **domain-aware preloading**:

1. **L0a domain predictor**: classifies user prompts into domains (codegen / math / debug / chinese_tech / translate etc., 8 domains), three-tier fallback — keyword (<1ms, zero deps) → TF-IDF linear classifier (236.7 KB, 5-fold CV 59.3%) → semantic embedding (all-MiniLM-L6-v2, 10-30ms, optional)
2. **L2 hot cache**: after predicting the domain, preloads that domain's commonly used experts from SSD to memory (mmap shared-memory LRU, per-layer independent deque, async preload with 2 workers)
3. **Data flywheel**: the proxy layer accumulates each real request (prompt + real expert routing) into the sample library; every 20 samples it auto-retrains the classifier — **gets more accurate with use** (measured: seed 59.3% → seed+real samples 78.1%, +18.8pp)

### 3.6 Architecture evolution path (three steps)

| Stage | Approach | VRAM | Speed | Status |
|------|------|------|------|------|
| 1 | A3 patch (experts forced CPU-resident) | 23.3 → 1.2 GB (-95%) | 8.6 t/s | ✅ verified |
| 2 | host buffer (experts CPU-pinned + GPU direct compute) | 1.6 GB | **37.5 t/s** (+200%) | ✅ verified |
| 3 | sched-cache (hot experts D2D) | 1.6 GB | Prompt 99→308 (+211%), Gen 39.2 (+5%) | ✅ verified (per-model enable) |
| 4 | **selective pin (current mainline as of 08-10)** | 1.6-2.9 GB (V4 16.5-16.7GB VRAM) | **Qwen 74.99 / DS 145.63 / V4 34.67-35.96 t/s** | ✅ verified (current mainline) |

---

## 4. Measured Data

> Test environment: NVIDIA RTX 4090 (24.5 GB), CUDA driver 580.105.08, 512-token context, llama.cpp (host buffer patch), 2026-08-02. Full reports in references/.

### 4.1 Core results: host buffer → on-demand pin → selective pin evolution

| Model | Form | Prompt t/s | Gen t/s | VRAM |
|------|------|-----------|---------|------|
| DS-V2-Lite (16B MoE, 64 experts) | CPU buffer (old, experts computed on CPU) | 12.5 | 12.5 | 1615 MiB |
| **DS-V2-Lite** | **host buffer (experts computed directly on GPU)** | **99.0** | **37.5** | **1625 MiB** |
| **DS-V2-Lite** | **selective pin (08-10)** | — | **145.63 / 127.95** | **4.9 GB / 2.1 GB RSS** |
| Qwen3.6-A3B (32B MoE, 256 experts) | CPU buffer (old) | 10.0 | 10.0 | 2141 MiB |
| **Qwen3.6-A3B** | **host buffer** | **75.8** | **46.8** | **2147 MiB** |
| **Qwen3.6-A3B** | **selective pin (08-10)** | — | **74.99 / 63.71** | **3.1 GB / 2.3 GB RSS** |
| V4-Flash (157B/85GB) | lazy no pin (08-05) | — | 1.7-2.0 | 9.1 GB |
| **V4-Flash** | **selective pin / on-demand (08-10)** | — | **34.67 / 35.96** | **26.8 / 17.5 GB RSS** |

- DS-V2-Lite: speed **+284%** (37.9 → 145.63 t/s, bins-v0.4.0 08-10)
- Qwen3.6-A3B: speed **+49%** (50.2 → 74.99 t/s, beats pre-lazy 46.5)
- V4-Flash: **RSS 84.4 → 17.5-26.8GB (↓68~79%)**, speed 34.67-35.96 t/s (the earlier 10.1 t/s was the official stock binary; moe-l2's optimized build was always ~30-35 t/s), near compute-bound on 4090

### 4.2 sched-cache tier matrix (DS-V2-Lite, cache mounted at sched copy layer)

| cache | Prompt t/s | Gen t/s | VRAM | Crash |
|-------|-----------|---------|------|------|
| none | 99.0 | 37.4 | 1625 MiB | 0 |
| **0.25 (recommended)** | **308.4 (+211%)** | **39.2 (+5%)** | 1625 MiB | 0 |
| 0.5 | 308.8 | 39.4 | 2127 MiB | 0 |
| 0.75 | 303.3 | 39.5 | 1625 MiB | 0 |
| 1.0 | 304.2 | 39.4 | 2165 MiB | 0 |

Conclusion: **0.25 is at the ceiling** (16 slots/layer covers all hot experts); larger tiers only add VRAM with no speed gain.

### 4.3 Full-GPU vs moe-l2 comparison (DS-V2-Lite Q2_K, selective pin main path)

| Metric | Standard full GPU | moe-l2 (selective pin, 08-10) | Change |
|------|-----------|----------------------|------|
| VRAM usage | 23.3 GB | **1.6-4.9 GB** | **-79%** |
| Gen speed | 65 t/s | **145.63 t/s** | 224% (beats full GPU) |
| Model/VRAM ratio | 0.26× | **3.9×** | +15x |

> Early A3-patch form (experts computed on CPU) data: VRAM 23.3 → 1.2 GiB (-95%), Gen 13.8 → 8.6 t/s, Prompt 23.4 → 18.5 — VRAM compression achieved but with a large speed loss. After the 2026-08-02 host buffer upgrade, speed went from 8.6 to 37.5 t/s (+335%) while VRAM stayed at 1.6 GB. Comparing the two paths shows: **VRAM compression comes from "experts on CPU", speed recovery comes from "experts computed directly on GPU"** — both are indispensable.

### 4.4 Multi-model × multi-cache-tier regression (2026-08-02)

| Model | Experts | cache tiers | Conversation types | Status |
|------|------|-----------|---------|------|
| Qwen3.6-A3B IQ2_M | 256 | 0/0.1/0.5/1.0/2.0 | short/long/followup | 15/15 PASS |
| DS-V2-Lite Q2_K | 64 | 0/0.1/0.5/1.0/2.0 | short/long/followup | 15/15 PASS |

All 30 combinations passed with zero crashes. The A3 GPU LRU cache runs stably across 5 cache ratios × 3 conversation scenarios.

### 4.5 GPU compatibility matrix

| GPU | VRAM | What it can run (moe-l2) |
|------|------|------------------|
| GTX 1650 / MX series | 4 GB | DS-V2-Lite (16B MoE) ✅ |
| **RTX 3050 / 2060 / 4060 / 3060** | **6-8 GB** | **Qwen3.6-A3B (32B MoE) ✅ core target segment** |
| RTX 3060 12GB / 4060 Ti | 12 GB | larger MoE (target 50B+) |
| RTX 4070 / 4090 | 16-24 GB | everything; large cache can be enabled |

The 8 GB card is the core target: previously it could only run 7B dense models; with moe-l2 it runs 32B MoE.

### 4.6 Prediction accuracy measurements

| Metric | Value |
|------|-----|
| Keyword prediction hit rate | 100% (28/28 tests) |
| Keyword latency | sub-ms |
| Semantic fallback latency | 10-30 ms (CPU) |
| TF-IDF classifier (seed) | 5-fold CV 59.3% |
| **+real samples (data flywheel)** | **78.1% (+18.8pp)** |

---

## 5. Comparison

### 5.1 vs full-GPU form (internal baseline)

| Dimension | Full GPU | moe-l2 | Implication |
|------|--------|--------|---------|
| VRAM | 23.3 GB | 1.6-4.9 GB | 8 GB cards can run it; VRAM is the hard constraint |
| Speed | 65 t/s | 145.63 t/s (DS, 08-10) | beats full GPU, far beyond the fluidity threshold |
| Hardware cost | 24 GB card (8000+ RMB) | 4-8 GB card (500-2000 RMB) | cost drops an order of magnitude |

### 5.2 vs Palm-Infra / mollm (Tencent, official data)

mollm is the MoE inference engine open-sourced by Tencent's YouTu Palm team: Apple Silicon / ARM Linux platforms, SSD expert offload + LRU cache + cross-layer prefetch — the same idea as ours (expert offload + tiered caching) but a different platform. Official measured data (README):

| Model | Expert cache | Decode | Peak memory | Hit rate |
|------|-----------|--------|---------|--------|
| Qwen3.5-122B-A10B W4 | 1 GiB | 12.38 t/s | 5.90 GiB | 47.9% |
| Qwen3.5-122B-A10B W4 | 10 GiB | 16.19 t/s | 14.64 GiB | 83.5% |
| Qwen3.5-122B-A10B W4 | 16 GiB | **16.53 t/s** | **20.60 GiB** | 88.6% |
| DeepSeek-V4-Flash (157GB) | 10 GiB | 4.73 t/s | 24.32 GiB | — |

**Comparison conclusion (honest framing — different platforms, we don't claim superiority)**:

| Dimension | mollm (Tencent) | moe-l2 |
|------|-------------|--------|
| Platform | Apple Silicon / ARM Linux | **Linux x86_64 + NVIDIA GPU** |
| Expert storage | SSD → RAM | CPU RAM → GPU VRAM |
| Compute location | CPU (NEON-optimized kernels) | **GPU (cuBLAS)** |
| Speed (122B-class) | 16.53 t/s @ 20.6 GiB | Qwen3.6-A3B 74.99 t/s @ 3.1 GiB (08-10) |
| Installation | source compile (CMake + C++) | **pip install** |
| Model support | Qwen series | **any llama.cpp MoE** (DeepSeek/Qwen/Mixtral) |
| Target users | mobile / edge | **desktop / home servers** |

**Insight**: two teams independently validated the "expert offload + LRU cache + prefetch" route. mollm runs a 122B model with a 1 GB cache + 5.9 GiB total memory (12.38 t/s); moe-l2 runs a 16B MoE in 1.6-4.9 GB VRAM (145.63 t/s, 08-10) — both prove **consumer hardware can run large MoE models** on their respective platforms.

### 5.3 Relationship with the ecosystem

| Project | Relationship |
|------|------|
| llama.cpp | underlying inference engine; moe-l2 patches it for enhancement (host buffer + cache), **zero migration** |
| ollama | another inference entry point; moe-l2 can serve as its front-end proxy |
| vLLM | server scenarios (multi-GPU/high concurrency); moe-l2 consumer scenarios; no conflict |
| GGUF | standard model format; moe-l2 reads its metadata for expert mapping |

moe-l2 doesn't reinvent the wheel; it operates at the "scheduling" layer — domain prediction, expert preloading, cache policy.

---

## 6. Deployment and Usage

### 6.1 One-line install

```bash
curl -fsSL https://raw.githubusercontent.com/yalun753/moe-l2/main/scripts/install.sh | bash
```

The installer automatically: detects the system (Linux x86_64 + NVIDIA) → installs the PyPI package → downloads prebuilt CUDA binaries → optionally downloads a demo model (11.5 GB, resumable) → environment self-check.

### 6.2 Manual install

```bash
pip install moe-l2                   # pure keyword prediction (zero extra deps)
pip install moe-l2[predictor]        # hybrid mode (+ semantic embedding)
moe-l2 download-bins                 # prebuilt host-buffer CUDA binaries
moe-l2 model download --model qwen3.6-35b   # optional demo model
moe-l2 start --model model.gguf --gpu
```

### 6.3 Compatibility

- OpenAI-compatible API (`/v1/chat/completions`); curl / Open WebUI / LangChain connect directly
- Platform: Linux x86_64 + NVIDIA (CUDA) — macOS / Windows / ARM Linux not yet supported
- Prebuilt binaries distributed from GitHub Releases (`bins-v0.2.1`, 1.9 GB multi-arch fully compatible package: sm_61/75/86/89/120a, GTX 1080 → RTX 50 series all supported by one binary; cuda-libs includes libnccl.so.2)

---

## 7. Project Status and Roadmap

### Completed

- ✅ Domain predictor (keyword + TF-IDF + semantic three-tier fallback, data flywheel auto-retraining)
- ✅ L2 memory hot cache (mmap LRU, thread-safe, async preload)
- ✅ Transparent proxy (HTTP/SSE forwarding, OpenAI-compatible)
- ✅ CLI (doctor / model / download-bins / collect / start / stats)
- ✅ **host buffer: experts computed directly on GPU** (DS +200%, Qwen +370%)
- ✅ **sched-cache** (DS Prompt +211%, per-model toggle)
- ✅ One-click installer + PyPI release (v0.5.1)
- ✅ Mode A expert routing collection (collect) + GGUF mapping embedding
- ✅ **Lightweight domain classifier + data flywheel auto-retraining (2026-08-02)**: TF-IDF + LinearSVC three-tier fallback prediction (keyword → TF-IDF → semantic), KB-scale model; Mode B collects real traffic samples during use and auto-retrains once enough accumulate; measured accuracy 59.3% → 78.1% (+18.8pp), flywheel closed loop verified

### In progress

- (None — all planned items completed, see above)

### Verified (demonstration-grade, not in the main table)

- ✅ **Qwen3-235B-A22B ultimate validation (2026-08-02)**: 8 GB cards can run a 235B MoE (Q2_K 81.7 GB), proving the "small VRAM can run big models" ceiling holds. ~1 t/s is the compute physical limit of 22B activated parameters on a single 4090 (SM 82% saturated, not bandwidth-bound); demonstration/technical-verification grade, not a performance selling point; verification records archived in internal docs (历史记录文档/235B-专家路由数据侦察报告.md), not in the official Benchmark main table

### Roadmap

| Stage | Content | Priority |
|------|------|--------|
| P0 | One-click installer ✅ / Benchmark ✅ (235B demo validation archived) / White paper ✅ | Current |
| P1 | Windows + NVIDIA port, AI PC vendor PoC, community issue/PR ✅ (llama.cpp #26448 + ollama #17557) | Late productization |
| P2 | Mode B collect-while-using ✅ / multi-model generalization ✅ (DS/Qwen/Mixtral/235B + sm_61→sm_120a) / smooth domain switching ✅ | Done |

---

## Appendix A: Glossary

| Term | Meaning |
|------|------|
| MoE | Mixture of Experts: multiple expert subnetworks + gated routing |
| A3 | moe-l2's expert scheduling scheme codename (activate experts on demand, three-tier storage) |
| host buffer | CUDA host-side pinned memory; the GPU can read it directly via PCIe DMA, uses no VRAM |
| sched-cache | expert cache mounted at llama.cpp's scheduler copy layer (hot experts D2D, no PCIe) |
| L2 cache | expert hot cache at the memory (RAM) layer (mmap shared-memory LRU) |
| D2D | Device-to-Device, copy within VRAM (no PCIe round trip) |
| t/s | tokens per second |
| GGUF | llama.cpp's model format (metadata + quantized weights) |

## Appendix B: References

- [moe-l2 README (bilingual)](../README.md) / [README_zh.md](../README_zh.md)
- [qwen3.6-a3b-iq2m-benchmark.md](qwen3.6-a3b-iq2m-benchmark.md)
- [deepseek-v2-lite-q2k-benchmark.md](deepseek-v2-lite-q2k-benchmark.md)
- [cache-sched-layer-benchmark.md](cache-sched-layer-benchmark.md)
- [TencentYoutuResearch/Palm-Infra](https://github.com/TencentYoutuResearch/Palm-Infra) (mollm official README measured data)

---

*moe-l2 · Apache 2.0 License · GitHub: [yalun753/moe-l2](https://github.com/yalun753/moe-l2)*
