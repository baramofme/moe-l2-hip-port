# External Evidence: Why Weight Prefetching Fails on Large MoEs

> Archive date: 2026-08-08. All data below is from public GitHub discussions and is independently verifiable.

## Context: llama.cpp PR #21067 (`--prefetch-weights`)

- PR: https://github.com/ggml-org/llama.cpp/pull/21067
- Author am17an, opened 2026-03-27, **draft status as of 2026-08-08**, still being updated as of 07-31
- Adds `--prefetch-weights`: prefetch the next layer's weights (CPU→GPU) and overlap the transfer with the current layer's compute. CUDA-only, and **requires `--no-mmap`** — without pinning, transfers are implicitly serialized
- Works well for dense models; the author acknowledges the MoE limitation: "prefetching cannot do this as it does not know which experts will be selected in the next layer"

## Independent measurement: noonghunna (2026-07-31)

Setup: Laguna-S-2.1 (117.6B, 256 experts, top-10, 48 layers) IQ4_NL, 2× RTX 3090 (PCIe, no NVLink), 28 expert layers offloaded to CPU (`--override-tensor` + `--no-mmap`), 262K context, within-binary A/B (`-pw 0/1`).

| metric | `-pw 0` (baseline) | `-pw 1` (prefetch) | delta |
|---|---:|---:|---:|
| prefill 10K | 902.5 t/s | 1035.3 t/s | +14.7% |
| prefill 90K | 734.2 t/s | 835.7 t/s | +13.8% |
| **TTFT, short prompt** | **1085 ms** | **1604 ms** | **+47.8%** |
| decode | 38.40 t/s | 36.55 t/s | −4.8% |

### Root cause (confirmed by a control experiment)

- Prefetch runs one split ahead of the router, so it cannot know which experts will be selected and must copy **whole** expert tensors — forfeiting the selective-expert copy from #15346
- Bytes moved on the short-prompt graph: 13,668 MiB → 28,128 MiB (2.06×)
- Control: disabling the selective copy on the `-pw 0` path reproduces the regression (30,774 MiB, TTFT 1725 ms) — it is a byte-volume problem, not event overhead / allocation / stream sync
- Short vs long prompt asymmetry: at 256 experts / top-10, selective copy moves 44% of the weight set at 34 tokens but 84% at ubatch 2048; prefetch therefore forfeits a 56% saving on short prompts and only 16% on long ones

### Proposed mitigation (MoE batch gate)

Disable prefetch when MoE batch < min_batch (512 default). Recovers TTFT (1096 ms) and keeps the prefill win (~1030 t/s), decode 37.00.

### Caveat the gate does not fix

The scheduler reserves double-buffer memory for the worst-case graph at `n_ubatch` while prefetch is active, permanently costing ~15% of VRAM that would otherwise hold resident expert weights. On memory-constrained MoE setups the steady-state decode cost can outweigh the prefill gain — `--prefetch-weights` is best left opt-in, not default, for MoE.

## What this means for moe-l2

moe-l2's design avoids both failure modes:

1. **No TTFT regression**: moe-l2 never prefetches whole expert tensors — experts are pinned on first touch (cudaHostRegister, whole-tensor merged registration) and only activated experts are copied per layer, keeping the selective-copy advantage
2. **No double-buffer VRAM cost**: the low-memory mode (dynamic pin set) caps resident memory instead of reserving worst-case buffers (V4-Flash 157B: 84 GB → 17-24 GB RSS on 16 GB-RAM machines)
3. **Decode-side prefetch has no gain** — three independent measurements agree: noonghunna −4.8%; moe-l2 double-buffered H2D pipeline 5.0 vs 5.5 t/s; thecodacus (#25859) +64% on prefill but only at large ubatch
4. Consensus: on MoE, prefetch helps prefill at large batch only; decode gains nothing and pays TTFT/VRAM costs. "Guess the next layer's experts" has no prediction headroom on large scattered-routing models (moe-l2 V4 measurement: a 30-turn session touched ~29 GB of distinct experts; WILLNEED prefetch ineffective)

## Related threads

- llama.cpp PR #21067 (weight prefetch, draft): https://github.com/ggml-org/llama.cpp/pull/21067
- llama.cpp issue #25859 (n-cpu-moe H2D bottleneck): https://github.com/ggml-org/llama.cpp/issues/25859
- llama.cpp issue #26448 (moe-l2 feature request): https://github.com/ggml-org/llama.cpp/issues/26448
- llama.cpp issue #25257 (SSD streaming, 0 comments, stale): https://github.com/ggml-org/llama.cpp/issues/25257
- llama.cpp issue #20757 (two-tier GPU+RAM expert cache, self-closed, discussion continues): https://github.com/ggml-org/llama.cpp/issues/20757
- ollama issue #8861 (hierarchical memory management for MoE): https://github.com/ollama/ollama/issues/8861
- ollama issue #17557 (moe-l2 feature request): https://github.com/ollama/ollama/issues/17557
- ollama issue #4161 (LRU cache for GPU VRAM, 0 comments): https://github.com/ollama/ollama/issues/4161
- Paper: Fast Inference of Mixture-of-Experts Language Models with Offloading (Eliseev & Mazur, 2023-12): https://arxiv.org/abs/2312.17238
