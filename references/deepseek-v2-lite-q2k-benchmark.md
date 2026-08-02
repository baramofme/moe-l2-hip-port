# DeepSeek-V2-Lite-Chat-Uncensored Q2_K benchmark 报告

## 基本信息

| 项目 | 值 |
|------|-----|
| 模型 | DeepSeek-V2-Lite-Chat-Uncensored |
| 架构 | MoE (2.37B active, 16B total) |
| 量化 | Q2_K (2-bit) |
| 推理引擎 | llama.cpp (A3 patch, CUDA) |
| GPU | NVIDIA RTX 4090, 24.5 GB VRAM |
| 测试日期 | 2026-07-29 |
| context 长度 | 512 tokens |

## 测试结果

### 测试条件

- 5 个 cache 级别：0, 0.1, 0.5, 1.0, 2.0
- 3 种对话类型：short（第一次短回）、long（第一次长回）、followup（第二次回复）
- 共计 15 项测试
- `--cpu-moe -ngl 99 -c 512 --no-warmup`

### 结果表

| Cache | 类型 | 状态 | VRAM 峰值 | Prompt t/s | Gen t/s |
|-------|------|------|-----------|-----------|--------|
| 0     | short | ✅ PASS | 1,737 MiB | 13.5 | 7.0 |
| 0     | long | ✅ PASS | 1,737 MiB | 15.2 | 6.8 |
| 0     | followup | ✅ PASS | 1,737 MiB | 95.9 | 7.1 |
| 0.1   | short | ✅ PASS | 2,089 MiB | 16.6 | 7.1 |
| 0.1   | long | ✅ PASS | 2,089 MiB | 16.7 | 7.1 |
| 0.1   | followup | ✅ PASS | 2,089 MiB | 97.3 | 7.1 |
| 0.5   | short | ✅ PASS | 2,701 MiB | 13.5 | 7.0 |
| 0.5   | long | ✅ PASS | 2,701 MiB | 16.6 | 7.2 |
| 0.5   | followup | ✅ PASS | 2,693 MiB | 102.6 | 7.0 |
| 1.0   | short | ✅ PASS | 3,391 MiB | 13.6 | 7.3 |
| 1.0   | long | ✅ PASS | 3,391 MiB | 16.6 | 7.2 |
| 1.0   | followup | ✅ PASS | 3,391 MiB | 103.0 | 7.3 |
| 2.0   | short | ✅ PASS | 3,391 MiB | 13.4 | 7.2 |
| 2.0   | long | ✅ PASS | 3,391 MiB | 17.0 | 7.5 |
| 2.0   | followup | ✅ PASS | 3,391 MiB | 97.0 | 7.9 |

**15/15 PASS, 0 failures.**

### 汇总分析

#### VRAM 消耗

| Cache 比值 | VRAM 峰值 | 增量 |
|-----------|----------|------|
| 0 (无 cache) | 1,737 MiB | — |
| 0.1 | 2,089 MiB | +352 MiB |
| 0.5 | 2,701 MiB | +964 MiB |
| 1.0 | 3,391 MiB | +1,654 MiB |
| 2.0 | 3,391 MiB | +1,654 MiB (上限) |

模型裸显存：~1.7 GB（cache=0）
最大 VRAM（cache=1.0+）：~3.4 GB

DS-V2-Lite 裸显存极低（仅 1.7 GB），即使缓存全开也仅 3.4 GB。这是一个可以在 **4 GB VRAM 显卡**上流畅运行的 MoE 模型。

#### 生成速度

- short/long 首次回复：6.8~7.9 t/s，非常稳定
- followup 首次 prompt（缓存命中）：95~103 t/s
- DS-V2-Lite 的生成速度显著快于 Qwen3.6-A3B（~7 vs ~5 t/s），因为 Q2_K 精度比 IQ2_M 更适合 GPU 并行计算

#### 与 Qwen3.6-A3B 对比

| 指标 | DS-V2-Lite Q2_K | Qwen3.6-A3B IQ2_M |
|------|-----------------|-------------------|
| 裸显存 | 1.7 GB | 3.3 GB |
| 最大 VRAM | 3.4 GB | 6.6 GB |
| 生成速度 | 6.8~7.9 t/s | 4.5~5.3 t/s |
| Prompt 速度（无缓存） | 13~17 t/s | 7~9 t/s |
| Prompt 速度（缓存命中） | 95~103 t/s | 65~87 t/s |

DS-V2-Lite 在一切指标上都优于 Qwen3.6-A3B——VRAM 更低、生成更快、缓存命中更猛。核心原因是 DS-V2-Lite 的 Q2_K 量化格式在 GPU cuBLAS 上更高效，而 IQ2_M 的压缩比虽然更高，但计算路径更重。

## 原始速度 vs A3 速度对比

原始模式（不传 `--cpu-moe`，**默认 mmap**）：A3 调度开启 + **专家计算在 GPU cuBLAS** 上完成。专家权重通过 mmap 留在 CPU RAM，GPU 仅放非专家层（~1.3 GB），非「全量加载」。
A3 模式（传 `--cpu-moe`）：A3 调度开启 + **专家计算在 CPU** 上完成。

> **注意**：原始的 mmap 默认模式≠`--no-mmap` 全量加载。`--no-mmap` 会把全部模型权重塞进 GPU VRAM（含专家），DS-V2-Lite Q2_K 约 **~6.6 GB**。A3 对比的目标是那种全量加载场景。

### 生成速度对比

| 指标 | 原始 (mmap 默认) | A3 (CPU专家) | 差距 |
|------|------------|-------------|------|
| 短 prompt (short) | 15.9 / 7.4 t/s | 13.5 / 7.0 t/s | 原始快 6% |
| 长 prompt (long) | 43.9 / 8.4 t/s | 15.2 / 6.8 t/s | 原始 Prompt 快 2.9x |
| VRAM 占用 | ~1.3 GB | ~1.7 GB | 原始略低 |

### 关键发现

1. **生成速度**：原始比 A3 快 6%，差距比 Qwen3.6-A3B 小（Qwen 差 23%）。因为 DS-V2-Lite 的 expert（Q2_K，~96 MB）在 CPU 上计算的退化较小。
2. **长 prompt 差距显著**：原始 43.9 t/s vs A3 15.2 t/s。原因同 Qwen，每步 CPU→GPU 搬运开销累加。
3. **VRAM 几乎一样**：1.3 GB vs 1.7 GB，差 400 MB。这个模型 VRAM 本身极低，两种模式都随便跑。

### 实际选型建议

| 场景 | 推荐模式 | 理由 |
|------|---------|------|
| 任何 4 GB+ 显卡 | **原始**（无 `--cpu-moe`） | 更快，VRAM 才 1.3 GB |
| 需要 followup 缓存命中加速 | **A3 + cache** | A3 模式下 cache 对 followup prompt 提升巨大（15→97 t/s） |

**核心结论**：DS-V2-Lite 在几乎所有场景下都应该用原始模式。只有在需要 followup prompt 加速时才开 `--cpu-moe` + cache。

## 显卡适配

### 在 RTX 4090 (24.5 GB VRAM) 上运行

**实在太轻了。** 实测在 RTX 4090 上，所有 cache 级别均顺畅运行：

| Cache | VRAM 占用 | 4090 占用率 |
|-------|----------|------------|
| 0     | 1.7 GB   | **7%**   |
| 0.5   | 2.7 GB   | **11%**  |
| 1.0   | 3.4 GB   | **14%**  |

最大 VRAM 仅 3.4 GB，4090 的 24.5 GB 还有 **86% 空闲**。生成速度 6.8~7.9 t/s，比 Qwen3.6-A3B 快约 50%，每秒输出 14~16 个汉字。

### 可以在哪些显卡上跑

| 显卡 | VRAM | 适用 cache 级别 |
|------|------|---------------|
| RTX 4090 / 4080 | 16~24 GB | ✅ 全部（推荐 0.5~1.0） |
| RTX 4070 / 4060 Ti | 8~12 GB | ✅ 无压力（推荐 0.5~1.0） |
| RTX 4060 / 3060 | 8~12 GB | ✅ 推荐 cache=0.5~1.0 |
| RTX 3050 / 2060 | 6~8 GB | ✅ 推荐 cache=0.5 |
| GTX 1650 / MX系列 | 4 GB | ⚠️ cache=0~0.1 勉强可跑 |

**结论：这个模型 4 GB 显卡就能跑，8 GB 显卡可以闭眼开 cache。如果真的想体验 A3 MoE 在入门卡上的效果，DS-V2-Lite Q2_K 是目前最佳测试对象。**

## Cache 比值含义

`GGML_CUDA_EXPERT_CACHE` 是一个 **cache slot 分配比率**，不是显存绝对值。

| 比值 | 缓存槽位 | 槽位容量 |
|------|---------|---------|
| **0** | 0 slots | 完全关闭 cache。每 token 所有 expert 都要从 CPU → GPU 加载 |
| **0.1** | ~6 slots | 只保留少量 slot，仅缓存最近几个 token 调过的 expert（DS-V2-Lite 共 64 个 expert） |
| **0.5** | ~32 slots | 缓存一半 expert，能覆盖一轮推理的热点 expert |
| **1.0** | 64 slots | 恰够存所有 expert，理论上 0 淘汰 |
| **2.0** | 128 slots (上限) | 双倍 slots，LRU 深度更大，但 VRAM 不再涨（已达上限） |

### 实际效果

- **0 → 0.1**：followup prompt 速度从 ~15 t/s → ~97 t/s（加极少缓存就覆盖了首次回复的 expert 调用记录）
- **0.1 → 0.5**：VRAM +612 MB，生成速度几乎不变（瓶颈在 expert compute 本身，不在 cache hit rate）
- **0.5 → 1.0/2.0**：VRAM +690 MB，生成速度依然不变——DS-V2-Lite 的 expert（~96 MB）太小，cache 开销 ≈ 直接算

结论：**0.1 就给足 followup 加速收益了**，DS-V2-Lite 的 expert 太小，更高 cache 比值对生成速度无增益。

## 结论

1. **DS-V2-Lite 是当前 A3 体系下最轻量、最高效的测试模型**
2. **4 GB VRAM 即可流畅运行** — 甚至可以在大多数 4 GB 入门卡上使用 expert cache
3. **推荐 cache 比率** — 0.5~1.0 足够，VRAM 占用仅 2.7~3.4 GB，不需要更高
4. **后续优化方向** — 测试更多 quantization 格式（Q3_K, Q4_K）看速度 vs 精度 tradeoff

---

## 2026-08-02 更新：host buffer 专家 GPU 直算 + sched-cache（重大突破）

> 本报告旧数据基于 `--cpu-moe`（专家 CPU 计算）形态。2026-08-02 完成架构升级——**host buffer（专家 CPU pinned 不占 VRAM）+ GGML_OP_OFFLOAD_MIN_BATCH=1 + cache 挂 sched 拷贝层**，专家走 GPU 直算，速度大幅提升。

### host buffer 全模型验证（RTX 4090，同一命令只差形态）

| 形态 | Prompt t/s | Gen t/s | VRAM |
|------|-----------|---------|------|
| CPU buffer（旧，专家 CPU 算） | 12.5 | 12.5 | 1615 MiB |
| **host buffer（专家 CPU pinned + GPU 直算）** | **99.0** | **37.5** | **1625 MiB** |

**机制**：llama-model-loader 放开 mmap→host buffer 回退（专家走 CUDA host buffer，数据在 CPU pinned 零 VRAM），sched 的 MoE 专家级拷贝优化**只拷激活专家**（每层 6 个 × 1.55MB ≈ 9.3MB 而非 64 个全拷），GPU 快路径直算。

### sched-cache 档位（cache 挂 sched 拷贝层后）

| cache | Prompt t/s | Gen t/s | VRAM | 崩溃 |
|-------|-----------|---------|------|------|
| 无 | 99.0 | 37.4 | 1625 MiB | 0 |
| **0.25（最优）** | **308.4（+211%）** | **39.2（+5%）** | 1625 | 0 |
| 0.5 | 308.8 | 39.4 | 2127（+502） | 0 |
| 0.75 | 303.3 | 39.5 | 1625 | 0 |
| 1.0 | 304.2 | 39.4 | 2165（+540） | 0 |

**档位结论**：0.25 已到顶（16 slots/层覆盖全部热专家），更大档位只加 VRAM 无速度收益。

### 关键结论（2026-08-02）

1. **host buffer 是主要突破**：Gen 12.5 → 37.5 t/s（+200%），VRAM 从 1615 → 1625 MiB（专家不占显存）
2. **sched-cache 锦上添花**：Prompt 99 → 308 t/s（+211%，热专家 D2D 免 PCIe），Gen 37.4 → 39.2（+5%）
3. **推荐配置**：`GGML_OP_OFFLOAD_MIN_BATCH=1` + `GGML_CUDA_EXPERT_CACHE=0.25`（仅 DS 这类中专家高重复率模型受益）
4. **旧结论修正**：此前"推荐 cache 0.5~1.0"已过时——host buffer 形态下 **0.25 最优**，且基础速度已 3 倍于旧形态
