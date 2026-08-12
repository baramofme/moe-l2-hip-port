# moe-l2 Design Decisions

> Why did we end up with the host-buffer approach? This document records the full history of approaches we explored since July 2026.
> Each entry follows the format: goal → measured result → why it was dropped or kept.
> This covers high-level trade-offs only — no implementation details.

---

## 0. Background & Core Insight

**Goal**: Run 16GB+ MoE models (Qwen3-30B-A3B, Qwen3.6-35B-A3B, DeepSeek-V2, etc.) on consumer GPUs with 8GB VRAM or less, at usable speeds.

**Core insight**: MoE models only activate a small subset of experts per step (top-2 ~ top-8 out of hundreds). Keeping every expert weight resident in VRAM is wasteful — most experts are idle most of the time. If expert weights live in CPU memory (zero VRAM) and the GPU only fetches the *activated* experts for each step, huge models can run in tiny VRAM.

Around the question "how to feed experts from CPU memory to the GPU efficiently", we went through three phases and more than a dozen technical routes.

---

## 1. Phase 1: Architecture Selection (2026-07-21)

| Approach | Idea | Outcome |
|----------|------|---------|
| **A: Reuse community solutions** | Use MoE-Infinity / Mixtral-offloading directly | ❌ Dropped: heavy hardware dependencies, narrow architecture support, no GGUF support. Several days of setup went nowhere |
| **B: Self-built tiered scheduling** | CPU mmap shared memory + domain predictor + on-demand expert fetching | ✅ **Selected**: controllable, verifiable step by step |
| **C: Dual-track in parallel** | Run A and B in parallel, keep whichever works | ❌ Dropped: a 4-core CPU machine can't sustain two development tracks |

---

## 2. Phase 2: Implementation Paths for Tiered Scheduling (2026-07-23)

After selecting approach B, we explored three paths for controlling expert memory:

### Path 1: Plain mmap, no --mlock (zero code changes)
- **Idea**: rely on OS demand paging — inference only touches pages of activated experts
- **Measured**: 13+ min load, 23.3GB peak RSS, 31GB RAM + 13GB swap, no valid output
- **Why dropped**: llama.cpp's warmup phase touches *all* experts (n_expert_used = full set), pulling in every page at once; graph construction/allocation also touches all expert tensor headers. OS page granularity (4KB) is far coarser than "one expert", so memory can't be controlled without restructuring
- **Conclusion**: pure configuration is a dead end; C++ changes are mandatory

### Path 2: Split expert tensors at load time (3-4 days of work)
- **Idea**: split each 3D expert tensor into N independent 2D tensors, one mmap per expert
- **Why shelved**: touches three core paths (model loading / graph building / inference scheduling), and breaks GGUF community naming conventions. Path 3 achieves the same with far less change

### Path 3: Dynamic memory management inside the inference loop (final direction)
Eight concrete variants were evaluated under Path 3:

| Variant | Idea | Outcome |
|---------|------|---------|
| A: Release inside MUL_MAT_ID | madvise-release pages right after an expert is computed | ✅ Implemented (~20 lines of C), verified runnable |
| B: willneed/dontneed | Operate around graph_compute | Merged into G |
| C: Callback at backend_sched split boundary | Insert release at split boundaries | ❌ Not recommended: splits ≠ layer boundaries, big architecture change |
| D: Thread callback in ggml_cplan | Insert callback after each op | ❌ Dead end: modern llama.cpp uses backend_sched, bypassing this path |
| E: Reorder experts in GGUF file | Python rewrite to make experts contiguous | ❌ No longer needed: measured reality shows Q2_K-quantized experts are **already contiguous** — reordering was pointless |
| F: mincore check + madvise | Check which pages are resident, then advise | Merged into A (A is more direct) |
| G: Per-layer subgraph build & execute | Change decode to build+compute+release layer by layer | ⭐ Pushed hard, later replaced by host-buffer (see Phase 3) |
| H: MAP_FIXED virtual address sharding | One mmap per expert | ❌ Not recommended: 1728 VMAs + excessive syscalls |

**Key finding (changed the course)**: Q2_K-quantized expert data is *contiguous per expert* in memory (not row-interleaved). This made per-expert page control possible and is the prerequisite for everything that followed.

---

## 3. Phase 3: From "Saving Memory" to "GPU-side Experts" (late July – 2026-08-02)

Phases 1-2 solved *memory control* but exposed a new problem: **memory is saved, but speed is not**. So we iterated on "where should experts actually compute".

### Approach F: --cpu-moe (experts computed on CPU)
- **Idea**: use llama.cpp's native `--n-cpu-moe` to keep MoE experts on the CPU; GPU only runs non-expert layers. Zero C++ changes
- **Measured (RTX 4090)**: DeepSeek-V2-Lite ~12.5 t/s; Qwen3.6-35B-A3B ~9 t/s; Mixtral 2.7 t/s (90% of 8x7B parameters are experts — CPU can't keep up)
- **Why dropped**: experts carry ~90% of MoE compute; CPU-computed experts cap out at single-digit to low-teen t/s. **It runs, but the experience is unusable.** Mixtral tests also proved: large-expert models (1.7GB/expert) *require* GPU-side experts to have any speed-up headroom

### Approach G: Per-layer scheduling (deepseek2.cpp rework)
- **Idea**: rework decode to build+compute+release per layer, keeping only that layer's active expert pages
- **Why dropped**: superseded by host-buffer — host-buffer achieves the same with far less change (CPU-pinned experts + scheduler copies on demand) without touching the decode core

### A2: GPU-side on-demand expert packing
- **Idea**: pack only activated experts and transfer them to the GPU, avoiding the VRAM cost of full upload
- **Measured (Qwen3.6-35B)**: 10.76 t/s, 2.2GB VRAM — faster than F but far from the 40-100 t/s target
- **Blockers**: DS regression (outputs broke after the change), Qwen slot bug (output '20,20' never reached the end). Per-fetch overhead ate the GPU compute advantage
- **Conclusion**: right direction, but "pack on demand" is high engineering complexity and the benefit is eaten by transfer overhead

### ✅ host-buffer: the final approach (CPU-resident experts + GPU compute)
- **Idea**: expert weights stay in CPU memory (pinned, zero VRAM); the scheduler copies only the **activated** experts to VRAM per step; hot experts stay in an LRU cache on the GPU to avoid PCIe round-trips
- **Measured (RTX 4090, 2026-08-02)**:
  - DeepSeek-V2-Lite: 12.5 → **37.5 t/s** (+200%), 1.6 GB VRAM
  - Qwen3.6-35B-A3B: 10 → **46.8 t/s** (+370%), 2.1 GB VRAM
- **Why it won**:
  1. Experts compute on the **GPU** (not CPU) — avoids Approach F's speed ceiling
  2. Transfers at *activated-expert* granularity instead of full model — avoids the VRAM wall
  3. Changes are concentrated in the scheduler copy layer (sched copy hook), not the decode core — avoids Approach G's engineering complexity
- **Companion optimizations**:
  - **sched-cache** (2026-08-02): expert cache in the scheduler copy layer. DS prompt 99 → 308 t/s (+211% at cache=0.25), VRAM unchanged; Qwen experts too small to benefit — toggled per model
  - **Multi-arch binaries** (bins-v0.2.1): one binary for GTX 1080 (sm_61) through RTX 50-series (sm_120a); three-GPU measured: 2080 Ti 6.89 / 3080 Ti 12.25 / 5090 16.63 t/s

---

## 4. Technical Boundaries (honest disclosure)

### Framework behavior note: the truth about the 22GB intermediate buffer (llama.cpp ecosystem understanding)

> This is an analysis of llama.cpp framework behavior, not moe-l2 implementation detail — published for ecosystem reference.

**Phenomenon**: DS-V2-Lite (6GB Q2_K) reaches 23.3GB peak RSS during CPU-only inference — far beyond the model size itself.

**Common misconception**: the 22GB was blamed on warmup loading all experts (n_expert_used=64).

**Truth** (corrected 2026-08-28):
1. `cparams.warmup` is deprecated; real inference always uses `n_expert_used=6` (the normal activation count) — warmup is irrelevant
2. The 22GB breakdown: prefill phase (large n_tokens, 512-2048) attention matrices (kq/kq_mask, ~14GB) + MoE intermediates (~1.6GB) + KV cache (~1.5GB) + weight-file page-ins (~6GB) + ggml-alloc chunk padding
3. **Root cause is ggml-alloc's one-shot allocation strategy**: the first prefill mallocs all intermediate buffers (~22GB anonymous pages) for the large n_tokens; on later decode (n_tokens=1), the `needs_realloc` check finds new tensors ≤ already-allocated sizes → decides no realloc is needed → **the prefill peak buffer persists forever, never shrinks**
4. Decode of 1 token actually needs only ~5-10MB of intermediate buffers per layer — the 22GB is a prefill-peak residue, not decode's real requirement

**Ecosystem significance**: any llama.cpp user running a long-prompt prefill keeps the prefill peak RSS permanently, even if all subsequent steps are single-token decodes. This is an inherent behavior of ggml-alloc's reuse strategy ("allocate once, reuse forever") — not a bug, but worth knowing, especially on memory-constrained devices (8GB mini-PCs / Raspberry Pi) where the prefill peak can decide whether the model runs at all.

### 235B ultimate validation (Qwen3-235B-A22B, 2026-08-02)
- **Verified**: an 81.7GB 235B MoE (Q2_K) runs on an 8GB card — proving the "small VRAM runs big models" ceiling holds
- **Speed**: ~1 t/s — the physical compute limit of 22B activated parameters on a single RTX 4090 (SM 82% saturated; bottleneck is GPU compute itself, not transfer/cache/scheduling)
- **Positioning**: demonstration/technical validation, not a performance selling point. All three speed-up routes were tested and closed:
  - Predictive preloading: only 42% adjacent-token expert overlap ceiling, negative preload benefit
  - UVA direct read (cudaHostRegister): 5.8 GB/s direct vs 25.1 GB/s cudaMemcpy (4x slower); cuBLAS on UVA pointers 11x slower
  - Cache expansion: 512-slot cap gives 0 hits during generation; 9.3GB VRAM only reaches 50% hit rate
- **Conclusion**: 235B is proof of *what can run*, not what's *nice to run*. The official benchmark tables only include speed data usable on consumer cards

---

## 5. One-Line Summary

> From "saving memory" (Approach F / Path 1 — runs but slow) to "GPU-side experts" (A2 — right direction, but transfer overhead), the host-buffer approach finally solved **both** VRAM and speed with **"CPU-resident experts + scheduler copies only activated experts + hot-expert cache"**. This is the answer converged on after a dozen+ measured routes — not a shot in the dark.
