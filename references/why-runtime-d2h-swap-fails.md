# Why Runtime D2H Swap Does Not Save VRAM

> Verified conclusion, 2026-07-25

---

## Problem

Attempt: at inference time, copy expert tensors from GPU (device) back to host memory (D2H), repoint the tensor's data pointer to the CPU copy, and continue computing from CPU memory — freeing the GPU VRAM that the experts previously occupied.

## Measured Result

| Metric | Value |
|--------|-------|
| Runtime D2H swap VRAM | **7,591 MiB** — identical to baseline |
| Compile-time interception VRAM | 2,233 MiB (earlier measurement) |
| Conclusion | **Runtime swap does not save VRAM** |

## Root Cause

**A tensor's data pointer is user-controlled, but the GPU buffer itself belongs to the CUDA backend buffer manager.**

- Repointing `src0->data` to host memory only changes where *future compute reads from*
- The original GPU buffer allocation was **never released**
- The CUDA backend buffer manager has no "partial release" interface — you cannot return a single allocation from inference code

Net effect — double memory:

```
GPU VRAM: [ original weight copy ]  ← never freed, but no longer used for compute
CPU RAM : [ new D2H copy ]          ← what compute actually reads
```

## Why Compile-Time Interception Works

`GGML_CUDA_FORCE_CPU_EXPERTS=1` saves VRAM because it does **not** move data at runtime — it prevents the expert tensors from being allocated into CUDA buffers in the first place, at model load / op scheduling time. The weights start on CPU; they never occupy VRAM.

| Approach | Mechanism | VRAM saved |
|----------|-----------|------------|
| Compile-time interception | Prevent allocation | ✅ 2.2 GiB |
| Runtime D2H swap | Allocate then move away | ❌ 7.6 GiB |

## Takeaway

To save VRAM on MoE models, intercept at the **buffer allocation layer**, not the compute layer:

1. **Compile-time interception (existing)** — prevent expert tensors from entering CUDA buffers during load
2. **LRU expert cache** — manage expert weight lifecycles between GPU/CPU at the buffer layer (swap in/out)

Per-expert H2D launch (A3-style) is an orthogonal *speed* mechanism and can be stacked with either.
