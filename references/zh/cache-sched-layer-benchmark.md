# moe-l2 cache 挂 sched 拷贝层 benchmark（2026-08-02）

## 背景

A3 LRU expert cache 原挂在 mul_mat_id **计算层**，但专家拷贝发生在 sched **输入拷贝层**（更早更底层），计算层永远看不到 CPU 指针 → cache 是死代码。本次把 cache 挂到 sched 拷贝层：`ggml-backend.cpp` 的 `copy_experts` 拷之前先查 cache，命中 → D2D 免 PCIe，miss → 原 CPU 拷贝 + 写回。

## 实现

- `expert-cache.cuh/cu`：新增 `ggml_cuda_expert_cache_copy_if_hit()`（命中 D2D 拷入目标 GPU buffer）
- `ggml-backend.cpp`：`copy_experts` 单专家分组走 cache（proc-address 跨 DSO 调用；多专家分组回退原路径）
- `ggml-cuda.cu`：`maybe_init` 提前到 mul_mat_id 函数入口（快路径也初始化 cache）+ 注册表暴露 cache 函数

## 三模型验证（RTX 4090，host buffer + OFFLOAD_MIN_BATCH=1）

### DS-V2-Lite Q2_K（专家 1.55MB，top-6）

| cache | Prompt t/s | Gen t/s | VRAM | 崩溃 |
|-------|-----------|---------|------|------|
| 无 | 99.0 | 37.4 | 1625 MiB | 0 |
| **0.25（最优）** | **308.4（+211%）** | **39.2（+5%）** | 1625 | 0 |
| 0.5 | 308.8 | 39.4 | 2127（+502） | 0 |
| 0.75 | 303.3 | 39.5 | 1625 | 0 |
| 1.0 | 304.2 | 39.4 | 2165（+540） | 0 |

### Qwen3.6-35B-A3B IQ2_M（专家 ~1MB，top-8）

| cache | Prompt t/s | Gen t/s | VRAM |
|-------|-----------|---------|------|
| 无 | 75.8 | 46.5 | 2147 MiB |
| 0.25 | 76.0 | 46.6 | 2147 MiB |
| 0.5 | 75.6 | 46.5 | 2475 MiB |

### Mixtral-8x7B Q4_K_M（专家 ~252MB，top-2）

| cache | Prompt t/s | Gen t/s | VRAM |
|-------|-----------|---------|------|
| 无 | 15.0 | 3.7 | 2243 MiB |
| 0.25 | 15.1 | 3.7 | 2903（+660） |
| 0.5 | 15.1 | 3.7 | 2903（+660） |

## 收益规律

**cache 收益 = 专家大小 × 命中率**：

| 模型 | 专家大小 | 激活方式 | 收益 |
|------|---------|---------|------|
| DS-V2-Lite | 1.55 MB | top-6 | **Prompt +211%，Gen +5%** |
| Qwen3.6-A3B | ~1 MB | top-8 | 无（专家太小，搬运本来就不贵） |
| Mixtral-8x7B | 252 MB | top-2 | 无（top-2 命中率太低，且槽占 VRAM 大） |

## 档位结论

DS 上 cache=0.25 已到顶（16 slots/层覆盖全部热专家），0.5/0.75/1.0 全部持平（303-309 / 39.4-39.5），更大档位只加 VRAM（+500MiB）无速度收益。

## 推荐配置（按模型）

| 模型 | 推荐 | 理由 |
|------|------|------|
| DS-V2-Lite | `GGML_CUDA_EXPERT_CACHE=0.25` | Prompt +211%，VRAM 零增加 |
| Qwen3.6-A3B | 不开 cache | 专家太小，无收益 |
| Mixtral-8x7B | 不开 cache | top-2 命中率低，白占 VRAM |

## 踩坑记录

1. **cache 槽按单专家分配**：连续多专家分组（first_id≠last_id）不可 cache，必须单专家才走（否则 size 越界 → cudaMemcpyAsync invalid argument 崩溃）
2. **maybe_init 位置**：原在 A3 慢速管线，host buffer 快路径永不触发 → 提前到函数入口（无条件执行）
3. **跨 DSO 调用**：ggml-backend（libggml-base.so）不能直接 include CUDA 头，用 proc-address 注册表暴露（`ggml_cuda_expert_cache_copy_if_hit` / `set`）

## 代码位置

- 云机：`/root/llama.cpp-clean/ggml/src/ggml-backend.cpp`、`ggml-cuda/expert-cache.cuh/cu`、`ggml-cuda/ggml-cuda.cu`
- 备份：云机 `/root/moe-l2-backups/sched-cache-fix-20260802/`，本地 `测试数据备份/a3on-fix-20260802/`

---

## 2026-08-07 更新：cache 上限 2048 槽（三模型通用增益）

> 08-02 结论"Qwen/Mixtral 无收益不开"在 **EXPERT_CACHE_MAX_SLOTS 512 → 2048** 后被推翻：容量不足是原结论的主因（小模型专家少、512 槽够用所以之前测不出差异；V4 每 token 激活专家 >512 直接 LRU 抖动）。

### 三模型实测（RTX 4090，2026-08-07，cache=1.0）

| 模型 | 无 cache | + cache 2048 | 提升 | GPU util 变化 |
|------|---------|-------------|------|--------------|
| Qwen3.6-A3B | 46.9 / 44.8 | **50.2 / 49.8** | +7% / +11% | — |
| DS-V2-Lite | 36.4 | **37.9 / 37.2** | +4% | — |
| V4-Flash | 9.5 | **10.1** | +6% | 13% → 86%（近计算 bound） |

### 结论（2026-08-07）

1. **2048 槽是 V4 平衡点**：512 槽无提升（每 token 激活专家 >512）、4096 槽 OOM（17.6GB cache + 基础 8.4GB > 24GB）
2. **推荐配置**：`GGML_CUDA_EXPERT_CACHE=1`（cli.py 已内置），所有模型开
3. V4 GPU util 86% 已近计算 bound——cache 已把拷贝瓶颈解决，再提速需优化 kernel/量化
4. 详细排错链与数据：`/opt/data/moe-l2/历史记录文档/on-demand-pin-方案-交接-20260807.md`
