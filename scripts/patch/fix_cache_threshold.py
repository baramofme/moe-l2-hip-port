#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════╗
# ║  INTERNAL PATCH SCRIPT                               ║
# ║  Targets a specific llama.cpp fork + commit.         ║
# ║  `/root/llama.cpp/...` path is hardcoded.            ║
# ║  Adjust path and verify code context before use.    ║
# ╚══════════════════════════════════════════════════════╝
# ⚠️ DEPRECATED (2026-07-29): 一次性补丁脚本（fix 已固化进源码）。
#
"""Apply permanent fix: skip cache_set for tensors > 100 MB.

Two locations in ggml-cuda.cu:
1. cublas_impl path (L~1474) — clean up probe fprintfs, keep size check
2. mul_mat_id path (L~2082) — add size check
"""

import re

path = "/root/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu"
with open(path, "r") as f:
    content = f.read()

# ── Fix 1: Clean up probe code in cublas_impl path ──
# Replace the verbose probe block with clean code
old_probe = """                {   // -- PROBE: skip cache_set for huge tensors > 100 MB --
                    const size_t cache_sz = (size_t)ggml_nelements(src0) * sizeof(cuda_t);
                    fprintf(stderr, \"[CACHE-SET] entering size=%zu ne=%zu cuda_t_sz=%zu\\n\",
                        cache_sz, (size_t)ggml_nelements(src0), sizeof(cuda_t));
                    if (cache_sz <= 100 * 1024 * 1024) {
                        const void * cached = ggml_cuda_expert_cache_set(src0->data,
                            cache_sz, src0_alloc.get(), main_stream);
                        cudaError_t err = cudaGetLastError();
                        fprintf(stderr, \"[CACHE-SET] done cached=%p cuda_err=%d(%s)\\n\",
                            cached, err, cudaGetErrorString(err));
                    } else {
                        fprintf(stderr, \"[CACHE-SET] SKIPPED (too large: %zu MB)\\n\", cache_sz / (1024*1024));
                    }
                }"""

new_clean = """                {   // Skip cache_set for huge tensors (> 100 MB) to avoid cuBLAS illegal memory access
                    const size_t cache_sz = (size_t)ggml_nelements(src0) * sizeof(cuda_t);
                    if (cache_sz <= 100 * 1024 * 1024) {
                        ggml_cuda_expert_cache_set(src0->data, cache_sz, src0_alloc.get(), main_stream);
                    }
                }"""

assert old_probe in content, "Fix 1: Could not find the probe code block!"
content = content.replace(old_probe, new_clean, 1)
print("Fix 1 applied: cublas_impl path cleaned up")

# ── Fix 2: Add size protection to mul_mat_id cache_set ──
# The cache_set at L~2082 uses nb02 as size. Expert tensors are small,
# but add protection for safety.
old_mm_id = """                    // Write back to cache for next token
                    const void * cached = ggml_cuda_expert_cache_set(
                        cpu_src, nb02, temp_gpu.ptr, ctx.stream());
                    gpu_ptr = cached ? cached : temp_gpu.ptr;"""

new_mm_id = """                    // Write back to cache for next token (skip if > 100MB)
                    const void * cached = NULL;
                    if (nb02 <= 100ul * 1024 * 1024) {
                        cached = ggml_cuda_expert_cache_set(
                            cpu_src, nb02, temp_gpu.ptr, ctx.stream());
                    }
                    gpu_ptr = cached ? cached : temp_gpu.ptr;"""

assert old_mm_id in content, "Fix 2: Could not find the mul_mat_id cache_set!"
content = content.replace(old_mm_id, new_mm_id, 1)
print("Fix 2 applied: mul_mat_id path protected")

# Verify no stale probe stuff left
fprintf_count = content.count("fprintf(stderr, \"[CACHE-SET]")
if fprintf_count == 0:
    print("Verified: no stale CACHE-SET probes remain")
else:
    print(f"WARNING: {fprintf_count} stale CACHE-SET probes remain!")

with open(path, "w") as f:
    f.write(content)

print("File written successfully.")
