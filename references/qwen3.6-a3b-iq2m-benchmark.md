# Qwen3.6-35B-A3B-UD-IQ2_M benchmark 报告（更新至 2026-08-07）

## 最新成果（2026-08-07）：on-demand pin 主路径（速度再破纪录）

> **on-demand pin**（lazy mmap 加载 + 首次触碰合并注册整个专家 tensor + A3 cache 2048 槽）取代 host buffer 成为主路径。**Qwen Gen 46.5 → 50.2 t/s（+8%），超过 pre-lazy host buffer 的 46.5。**

### 实测（RTX 4090，2026-08-07）

| 形态 | Gen t/s（短） | Gen t/s（长） | VRAM |
|------|-------------|-------------|------|
| host buffer（08-02） | 46.5 | — | 2147 MiB |
| on-demand pin（whole） | 46.9 | 44.8 | ~8GB（含模型） |
| **on-demand pin + cache 2048** | **50.2** | **49.8** | 2.9GB |
| 多架构包（CUDA 12.8，sm_61-120a） | **51.5** | **51.6** | 2.9GB |

### 机制

1. **合并注册**：修复 CUDA 11.8 坑——`cudaMemcpyAsync` 源跨多个 `cudaHostRegister` 区间必崩（pintest6c 铁证）；改为 unregister 相邻区间 + register 单大区间（pintest6d 验证）
2. **whole-tensor pin**（`MOEL2_WHOLE_PIN`，默认开）：copy_experts 首次触碰时注册整个专家 tensor，消除推理时新专家页 fault 读盘
3. **A3 cache 2048 槽**（`EXPERT_CACHE_MAX_SLOTS`）：Qwen 短 46.9→50.2、长 44.8→49.8

### 关键结论（2026-08-07）

1. **Qwen 50.2 t/s 为当前最高纪录**（超 pre-lazy 46.5、host buffer 46.5）
2. **2048 槽 cache 三模型通用增益**（Qwen +7~11% / DS +4% / V4 +6%），已随 v0.7.1 / bins-v0.3.1 发布
3. **推荐配置**：`GGML_OP_OFFLOAD_MIN_BATCH=1` + `GGML_CUDA_EXPERT_CACHE=1`（cli.py 已内置）
4. 详细排错链与数据：`/opt/data/moe-l2/历史记录文档/on-demand-pin-方案-交接-20260807.md`

---

## 基本信息

| 项目 | 值 |
|------|-----|
| 模型 | Qwen3.6-35B-A3B-UD-IQ2_M |
| 架构 | A3 (3.6B active, 35B total) |
| 量化 | IQ2_M (2-bit) |
| 推理引擎 | llama.cpp (A3 patch + host buffer, CUDA) |
| GPU | NVIDIA RTX 4090, 24.5 GB VRAM |
| 测试日期 | 2026-07-29（初版）/ 2026-08-02（架构升级） |
| context 长度 | 512 tokens |

## 修复记录（2026-07-29）

所有 `GGML_CUDA_EXPERT_CACHE>0` 组合（short/long/followup, cache=0.1~2.0）曾 exit 134，cuBLAS illegal memory access。

根因：`cache_set` 对 970 MB 的 LM head tensor (Q4_K, 2048×248320) 做 cudaMalloc + cudaMemcpyAsync D2D 后，反复分配淘汰导致 CUDA 内存碎片/页表污染，后续 cuBLAS gemm 报 illegal memory access。

修复：在两处 `cache_set` 前加 >100 MB 跳过检查。仅跳过 LM head，专家权重 tensor 全部小于 100 MB，正常缓存。

---

## 2026-08-02 更新：host buffer 专家 GPU 直算（速度 +370%）

> 2026-08-02 架构升级——**host buffer（专家 CPU pinned 不占 VRAM）+ GGML_OP_OFFLOAD_MIN_BATCH=1**，专家走 GPU 直算。Qwen3.6-A3B 成为**当前最快的测试模型**（Gen 46.5 t/s，超过 DS 39.2 / Mixtral 3.7）。

### host buffer 全模型验证（RTX 4090）

| 形态 | Prompt t/s | Gen t/s | VRAM |
|------|-----------|---------|------|
| CPU buffer（旧，专家 CPU 算） | 10.0 | 10.0 | 2141 MiB |
| **host buffer（专家 CPU pinned + GPU 直算）** | **75.8** | **46.5** | **2147 MiB** |

**机制**：llama-model-loader 放开 mmap→host buffer 回退（专家走 CUDA host buffer，数据在 CPU pinned 零 VRAM），sched 的 MoE 专家级拷贝优化只拷激活专家，GPU 快路径直算。

### sched-cache 验证（cache 挂 sched 拷贝层后）

| cache | Prompt t/s | Gen t/s | VRAM |
|-------|-----------|---------|------|
| 无 | 75.8 | 46.5 | 2147 MiB |
| 0.25 | 76.0 | 46.6 | 2147 MiB |
| 0.5 | 75.6 | 46.5 | 2475 MiB |

**Qwen 上 cache 无收益**（46.5-46.6 持平）——专家太小（~1MB）+ 短 prompt 搬运本来就少，cache 只加 VRAM。

### 关键结论（2026-08-02）

1. **host buffer 让 Qwen3.6-A3B 成为最快模型**：Gen 10 → 46.5 t/s（+370%），VRAM 2147 MiB（专家不占显存）
2. **Qwen 不需要开 cache**：专家太小，cache 无收益只加 VRAM
3. **推荐配置**：`GGML_OP_OFFLOAD_MIN_BATCH=1`（不开 cache）——8GB 卡跑 32B MoE，46.5 t/s
4. **三模型速度排序（host buffer 后）**：Qwen 46.5 > DS 39.2 > Mixtral 3.7 t/s
5. **显卡适配**：VRAM 仅 2147 MiB，**4 GB 卡即可流畅运行**（旧结论"最低 4 GB、推荐 8 GB"已过时——8 GB 卡绰绰有余）

### 详细验证数据（2026-08-02 完整链路）

- **三模型 cache 档位矩阵**：见 `cache-sched-layer-benchmark.md`
- **host buffer 架构细节**：llama-model-loader.cpp 放开 mmap→host buffer 回退 + cli.py `GGML_OP_OFFLOAD_MIN_BATCH=1`
- **数据飞轮**：proxy 真实流量攒样本 → 自动重训分类器（种子 111 + 真实 50 = 161 samples），标签质量提升

---