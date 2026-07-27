# Why Runtime D2H Swap (Approach B) Can't Save VRAM

> 2026-07-25 — Verified conclusion

## Problem

In `ggml_cuda_mul_mat_id`, perform D2H copy on an already-allocated CUDA buffer expert tensor, change `src0->data` pointer to CPU memory, and set `experts_on_host=true`. The goal: make inference read expert weights from CPU memory, freeing GPU VRAM.

## Measured Results

| Metric | Value |
|--------|-------|
| Approach B runtime VRAM | **7,591 MiB** (same as baseline) |
| Env-var version VRAM | 2,233 MiB (previously measured) |
| Conclusion | Runtime swap **does not save VRAM** |

## Root Cause

**The tensor's data pointer belongs to user-space operations, but the GPU buffer belongs to the CUDA backend buffer manager.**

- `src0->data = host_buf` only changes the tensor's read source (making subsequent computation read from CPU memory)
- The original GPU buffer allocation (by `ggml_backend_cuda_buffer_type`) **is never freed**
- The CUDA backend's buffer manager has no "partial free" API — individual allocations can't be reclaimed from inference code

In effect:
```
GPU VRAM:   [ original weight copy ] ← never freed, but computation no longer uses it
CPU Memory: [ new D2H copy ]         ← computation actually reads from this
                  ↑
             Double memory waste
```

## Comparison with Env-Var (Compile-Time Intercept)

`GGML_CUDA_FORCE_CPU_EXPERTS=1` saves VRAM for a fundamentally different reason: it blocks expert tensor CUDA buffer allocation at **model load / op scheduling time**, not at runtime. Weights never enter GPU VRAM in the first place.

| Approach | Mechanism | VRAM Saved |
|----------|-----------|------------|
| Compile-time intercept (env var) | Prevent allocation | ✅ 2.2 GiB |
| Runtime D2H swap (Approach B) | Allocate then move | ❌ 7.6 GiB |

## Future Direction

To save VRAM on MoE models, interception must happen at the **buffer allocation layer**, not at the computation layer. Two viable paths:

1. **Compile-time intercept (existing)** — Prevent expert tensors from entering CUDA buffers in `buft_for_tensor` / `load_tensors`
2. **LRU expert cache** — Manage expert weight lifecycle (swap in/out) between GPU and CPU at the buffer layer

Approach A3 (per-expert H2D launch) serves as an acceleration mechanism independent of VRAM savings — the two can be combined.
