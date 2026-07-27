# 运行时 D2H swap（方案 B）为什么不能节省显存

> 2026-07-25 验证结论

## 问题

在 `ggml_cuda_mul_mat_id` 中，对已分配在 CUDA buffer 上的 expert tensor 执行 D2H 拷贝 + 改 `src0->data` 指针指向 CPU 内存 + `experts_on_host=true`。试图让 expert 权重在推理时从 CPU 读取，释放 GPU 显存。

## 实测结果

| 指标 | 值 |
|------|-----|
| 方案 B 运行时 VRAM | **7,591 MiB**（与 baseline 一致） |
| env var 版 VRAM | 2,233 MiB（此前实测） |
| 结论 | 运行时 swap **不省显存** |

## 根因

**tensor 的 data 指针归用户操作，但 GPU buffer 归 CUDA backend buffer 管理器。**

- `src0->data = host_buf` 只改了 tensor 的读取位置（让后续计算走 CPU 内存）
- 原始 GPU buffer 分配（由 `ggml_backend_cuda_buffer_type` 分配）**从未被释放**
- CUDA backend 的 buffer 管理器没有"部分释放"接口，无法从推理代码侧主动归还单块分配

相当于：
```
GPU 显存: [ 原始权重副本 ] ← 从未释放，但计算已不再用它
CPU 内存: [ 新 D2H 副本 ] ← 计算实际读取这个
                  ↑
             两倍内存浪费
```

## 对比 env var（编译期拦截）

`GGML_CUDA_FORCE_CPU_EXPERTS=1` 能省显存的真正原因：它不是在运行时搬数据，而是在 **模型加载 / op 调度阶段**就阻止了 expert tensor 的 CUDA buffer 分配。权重从一开始就在 CPU 上，GPU 显存里根本没它们。

| 方案 | 机制 | VRAM 节省 |
|------|------|-----------|
| 编译期拦截（env var） | 阻止分配 | ✅ 2.2 GiB |
| 运行时 D2H swap（方案 B） | 分配后搬走 | ❌ 7.6 GiB |

## 后续方向

要节省 MoE 模型的显存，需要在 **buffer 分配层** 做拦截，而非在计算层做 swap。两条可行路径：

1. **编译期拦截（已有）** — 在 `buft_for_tensor` / `load_tensors` 中阻止 expert tensor 进入 CUDA buffer
2. **LRU expert cache** — 在 buffer 层管理 GPU/CPU 之间的 expert 权重生命周期（换入换出）

方案 A3（逐 expert H2D launch）作为加速手段独立于显存节省，两者可以叠加使用。
