# Phase 3 GPU LRU Expert Cache — 基准测试 & 逐层改造方案

> 存档日期：2026-07-28
> 测试环境：AutoDL RTX 4090 24GB / CUDA 12.2 / llama.cpp

## 已完成的测试

### 测试模型

| 模型 | 大小 | Experts/层 | 层数 | Expert 大小 |
|------|------|-----------|------|------------|
| DeepSeek-V2-Lite Q2_K | 6.0 GB | 64 | 27 | ~17 MB |
| Qwen3.6-35B-A3B IQ2_M | 11 GB | 256 | 244 | ~1 MB |

### 测试命令

全部使用 `--gpu-layers 99 --cpu-moe --single-turn`，128 tokens，同一 prompt。

### 结果摘要

#### DS-V2-Lite Q2_K

| 配置 | Prompt t/s | Gen t/s | 备注 |
|------|-----------|--------|------|
| CPU baseline (-ngl 0) | 12.3 | 4.5 | 无 GPU |
| Full GPU (-ngl 99, 无 --cpu-moe) | 17.4 | 8.7 | 最快 |
| CPU-moe (-ngl 99, --cpu-moe) | 19.7 | 8.4 | ≈Full GPU |
| LRU 0.25 | 16.9 | 8.1 | |
| LRU 0.5 | 19.7 | 8.1 | |
| LRU 0.75 | 18.0 | 8.1 | |
| LRU 1.0 | 17.8 | 8.4 | |

#### Qwen3.6-35B-A3B IQ2_M

| 配置 | Prompt t/s | Gen t/s |
|------|-----------|--------|
| CPU baseline | 6.1 | 1.8 |
| Full GPU | 9.1 | 6.5 |
| CPU-moe | 9.1 | 6.3 |
| LRU 0.25 | 10.2 | 6.2 |
| LRU 0.5 | 9.0 | 6.0 |
| LRU 0.75 | 9.5 | 6.0 |
| LRU 1.0 | 8.6 | 5.8 |

### 关键发现

1. **LRU cache 对速度无提升** — 所有分数下的 gen t/s 都与 CPU-moe 持平或略低，未超过 full GPU
2. **显存无明显变化** — 控制台 nvidia-smi 看不到 cache 分配（逐层改造后有更好的控制台可视效果）
3. **根本原因：全局单例设计缺陷**
   - 全局 16~64 个 slots 被所有层共用
   - 层 0 刚加载的 expert 到层 1 时被踢掉
   - 命中率 ≈ 0%，cache 退化
   - 每个 expert 都走了完整的 H2D 拷贝 + LRU 查询 evict 开销

### 根因总结

- 设计时估算不足：以为 64 个 slot 够，但 MoE 有 27 层，每层 64 个 expert
- 全局 single expert cache pool 的过度共享导致了灾难性的缓存抢占

## 改造方案：逐层独立 Cache（per-layer slab）

### 设计变更

| 当前 | 改后 |
|------|------|
| 全局 `ExpertCacheSlab slabs[MAX_SLOTS]` | `std::vector<ExpertCacheSlab> per_layer_slabs[n_layers]` |
| `n_slots = ceil(fraction × experts_per_layer)` 全局 | 每层独立计算，且使用 `expert_size_bytes` 预分配 |
| lookup 参数 `(expert_id, cpu_src, ...)` | 加 `layer_id` 参数 |
| `maybe_init(device, n_slots, 0)` | `maybe_init(device, n_slots, expert_size_bytes, n_layers)` |

### 每层 slots 的计算

| 模型 | 层数 | 每层 expert | 0.25 slots/层 | 每 slot 大小 | 总 cache 显存 |
|------|------|------------|---------------|-------------|--------------|
| DS-V2-Lite Q2_K | 27 | 64 | 16 | ~17 MB | ~7.3 GB |
| Qwen3.6 IQ2_M | 244 | 256 | 64 | ~1 MB | ~15.6 GB |

### 预期效果

- 每层 cache 由本层独享，不会被其他层踢掉
- 连续 token 如果激活了同一层的 expert → 命中
- 每层的 Top-K routing 通常是同一批 expert → 命中率从 0% 提升到 60%+
- 同时修复 `expert_size_bytes = 0` 的问题，预分配真实显存

### 主要改动文件

| 文件 | 改动 |
|------|------|
| `ggml/src/ggml-cuda/expert-cache.cuh` | API 加 layer_id 参数 |
| `ggml/src/ggml-cuda/expert-cache.cu` | 重构为 `std::vector<ExpertCacheSlab>`，每层独立 slab 和 mutex |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 需要从 llm_graph 层面传 layer_id |

### 耗时评估

| 步骤 | 时间 |
|------|------|
| 理清 layer 信息传递路径 | ~20 min |
| 改 .cuh/.cu 做 per-layer slabs | ~30 min |
| 把 layer_id 从构图层穿到 CUDA 层 | ~30~60 min |
| 修复 expert_size_bytes 预分配 | ~10 min |
| 编译 + 云上跑一轮基准测试 | ~30 min |
| 调试 | ~30 min |

**乐观：2h。带 debug：4h。**

### 前置条件

- AutoDL 云 GPU（RTX 4090）
- 已编译通过的 llama.cpp 分支，源码已包含 Phase 3 整合
- 已测试的 GGUF 模型 DS-V2-Lite Q2_K 和 Qwen3.6-35B-A3B IQ2_M

## 结论

**决定：暂时搁置。**

当前 H2D 拷贝开销 ~60 µs（17 MB），cache 查询 + evict 也有几十 µs。即使修复 per-layer 架构，在 Expert 内存占用较小的情况下速度收益可能有限。建议将来有时间时再改造，或者等有其他优化方向时一起做。
