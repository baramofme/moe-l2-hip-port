# llama.cpp expert_cache cuBLAS illegal memory access 修复记录

> **历史记录（2026-07-29）**：本文档记录 A3 LRU expert cache 在**旧架构**下对超大 tensor（LM head 970 MB）反复缓存导致的 CUDA 崩溃修复。
> **2026-08-02 起架构已升级**（host-buffer + cache 挂 sched 拷贝层），该崩溃路径已不再存在；文档保留供其他模型排查 cache crash 参考。
> 文末 benchmark 数据（4.5-5.3 / 6.8-7.9 t/s）为旧架构形态，当前性能见 [deepseek-v2-lite-q2k-benchmark.md](deepseek-v2-lite-q2k-benchmark.md) 与 [qwen3.6-a3b-iq2m-benchmark.md](qwen3.6-a3b-iq2m-benchmark.md)。

## 背景

Qwen3.6-35B-A3B-UD-IQ2_M 在 `GGML_CUDA_EXPERT_CACHE>0` 时崩溃（exit 134），stderr 打印：
```
CUDA error 77: an illegal memory access was encountered
```
触发点：最后一个 `force-cublas` 调用（cuBLAS gemm），tensor type=12（Q4_K），ne0=2048 ne1=248320（LM head 输出投影），共 1074 次 force-cublas 调用中最后一次崩溃。

## 根因

`cache_set` 对 **970 MB 的 LM head tensor**（Q4_K, 2048×248320）执行了：

```cpp
ggml_cuda_expert_cache_set(src0->data, cache_sz, src0_alloc.get(), main_stream);
```

内部做 cudaMalloc 970 MB + cudaMemcpyAsync D2D，分配后缓存未命中再淘汰，每次触发一次 970 MB 的分配→释放→cuBLAS 读取同一地址。这个 LM head 本身每 1-2 token 就被淘汰（因为是单 token 计算），反复大块分配导致 CUDA 内存碎片或页表污染，最终 cuBLAS sync 时报 illegal memory access。

重复拷贝 LM head 也毫无价值——它不包含专家权重，唯一作用是下次 token 正好命中 LM head 时省一次 D2D 拷贝，但 LM head 太特么大了。

## 修复

在两处 `cache_set` 调用前加大小阈值检查，**跳过 >100 MB 的 tensor**：

### 位置 1：cublas_impl（L1474）

```cpp
{   // Skip cache_set for huge tensors (> 100 MB) to avoid cuBLAS illegal memory access
    const size_t cache_sz = (size_t)ggml_nelements(src0) * sizeof(cuda_t);
    if (cache_sz <= 100 * 1024 * 1024) {
        ggml_cuda_expert_cache_set(src0->data, cache_sz, src0_alloc.get(), main_stream);
    }
}
```

### 位置 2：mul_mat_id 回写路径（L2075）

```cpp
// Write back to cache for next token (skip if > 100MB)
const void * cached = NULL;
if (nb02 <= 100ul * 1024 * 1024) {
    cached = ggml_cuda_expert_cache_set(
        cpu_src, nb02, temp_gpu.ptr, ctx.stream());
}
gpu_ptr = cached ? cached : temp_gpu.ptr;
```

## 验证

### 单次烟雾测试
```
model=Qwen3.6-35B-A3B-UD-IQ2_M
cache=0.5  env GGML_CUDA_EXPERT_CACHE=0.5
Prompt: 8.7 t/s | Generation: 5.2 t/s  EXIT 0
```

修复后 cache_set 调用分布（cache=0.5 单次推理）：
- 2148 次 cache_set 调用
- 其中 4 次 LM head（970 MB）→ **SKIPPED**
- 1070 次小 tensor（专家权重等）→ 正常缓存，累计分配 ~17.45 GB

### 完整 benchmark（30 项，全部 PASS）

| 模型 | Cache 级别 | Conv 类型 | 状态 | VRAM 峰值 | 生成速度 |
|------|-----------|----------|------|----------|---------|
| Qwen3.6-A3B IQ2_M | 0/0.1/0.5/1.0/2.0 | short/long/followup | **15/15 PASS** | 3.4→6.6 GB | 4.5-5.3 t/s |
| DS-V2-Lite Q2_K | 0/0.1/0.5/1.0/2.0 | short/long/followup | **15/15 PASS** | 1.7→3.4 GB | 6.8-7.9 t/s |

**0 failures across all 30 combos.**

## 为什么 100 MB 是安全阈值

- 只跳过了 LM head（970 MB）这一个 tensor
- 所有专家权重 tensor 都小于 100 MB，正常缓存
- LM head 是唯一一个既巨大又频繁淘汰的 tensor——缓存它收益极低，副作用极高
- 其他模型如果也有异常巨大的单 tensor，也可以用同样逻辑跳过

## 后续建议

如果遇到其他模型的 expert_cache crash，第一反应看 `ggml_nelements * sizeof(cuda_t)` 最大的 tensor 是否超过 100 MB。cudaMalloc 大块 + 频繁释放 → 碎片化是 CUDA illegal memory access 的常见根因。
