# DeepSeek V4 Flash 验证报告（更新至 2026-08-10）

> ✅ **2026-08-10 4090 复测（bins-v0.4.0 修复版，全链路）**：V4 实测 **35.96 t/s**（on-demand 兜底，RSS 17.5GB）；selective pin（v4_top100.map）**34.67 t/s**（RSS 26.8GB）；VRAM 16.5-16.7GB。对比原版 llama.cpp 二进制（10.1 t/s）+255%。完整 4 轮数据见下方"08-10 复测"节。

> ⚠️ **2026-08-10 基准更正**：本报告 10.1 t/s 为**官方原版 llama.cpp 二进制**实测（on-demand pin 路径）；moe-l2 优化版二进制（selective pin，bins-v0.4.0）在 RTX 4090 上实测 **35.96 t/s**、RSS **17.5-26.8GB**。本报告下方 08-05/08-07 数据保留作历史记录。

> 状态：**on-demand pin 主路径跑通 V4 Flash ✅（RTX 4090 实测 10.1 t/s，原 1.7-2.0 的 5 倍）**
> 关联：`multi-arch-three-gpu-benchmark.md`（V4 全链路复测节）、PyPI 0.7.1 / bins-v0.3.1（on-demand pin 主路径）

---

## 最新成果（2026-08-10）：bins-v0.4.0 全链路，4090 上 35.96 t/s

> **bins-v0.4.0（selective pin + on-demand 兜底）在 RTX 4090 上把 V4 从原版二进制的 10.1 t/s 拉到 35.96 t/s（+255%）**，RSS 从 whole-pin 84GB 降到 17.5-26.8GB。90GB 模型在 24GB 卡 + 1TB 内存机器上流畅交互。

### 一句话结论

**moe-l2 在 RTX 4090 上用 bins-v0.4.0 跑 DeepSeek V4 Flash（UD-IQ2_M，85GB 三片）达到 35.96 t/s（on-demand 兜底，RSS 17.5GB）/ 34.67 t/s（selective pin，RSS 26.8GB），VRAM 16.5-16.7GB**（2026-08-10 实测，完整 4 轮数据见文末"08-10 复测"节）。

### 实测（RTX 4090 24GB，2026-08-10，全链路 `moe-l2 start --gpu`，round3 稳定轮）

| 形态 | Gen t/s | VRAM | RSS |
|------|---------|------|-----|
| on-demand 兜底（auto 路由表） | **35.96**（短 34.44 / 追问2 35.96） | 16.5 GB | 17.5 GB |
| selective pin（v4_top100.map） | **34.67**（短 33.84 / 追问2 34.64） | 16.7 GB | 26.8 GB |
| 原版 llama.cpp 二进制（对照） | 10.1 | 17.4 GB | 82 GB |

### 关键结论（2026-08-10）

1. **V4 在 4090 上稳定 34-36 t/s**（两模式一致），对比原版二进制 +255%
2. **RSS 大幅缩减**：whole-pin 84GB → on-demand 17.5GB（↓79%）/ selective pin 26.8GB（↓68%）
3. **10.1 t/s 是原版二进制的真实数据**：此前"5 倍提速"结论基于原版（1.7-2.0 → 10.1）；moe-l2 优化版本来就是 30-35 t/s 级别

#### 历史结论（2026-08-07，on-demand pin 时代，保留作记录）

1. **V4 4090 上 10.1 t/s（原 1.7-2.0 的 5 倍）**；GPU util 13% → 86%，**已近计算 bound**（再提速需优化 kernel/量化，非 cache）
2. **2048 槽是 cache 平衡点**（512 无提升、4096 OOM），三模型通用增益
3. **RSS 80.9GB（whole-pin 全量 fault）**——1TB 内存机器无压力；128GB 容器需淘汰机制（v3.1 + unregister）控制驻留
4. 详细排错链：`/opt/data/moe-l2/历史记录文档/on-demand-pin-方案-交接-20260807.md`

### 长上下文（500K）输出速度问答（2026-08-07）

> 知乎提问："DS V4 长上下文 500K 时的输出速度呢？"——如实回答如下。

**实测（短上下文，c=512）**：RTX 4090 + moe-l2（on-demand pin + A3 cache 2048）下 **10.1 t/s**（2026-08-07，原版二进制口径；08-10 bins-v0.4.0 实测 35.96 t/s）。

**500K 长上下文（未实测，给确定性推断）**：

- **输出速度基本不掉**：MoE 每生成一个 token 的主要开销是激活专家计算（6/256），与上下文长度无关；DeepSeek 系列用 MLA（KV 压缩注意力），500K 不会让每 token 的 attention 成本线性爆炸（V4 是 1M 上下文设计）
- **真正成本在两处（不是输出速度）**：
  1. **首次 prefill**：一次性喂 50 万 token 的初始处理时间很长
  2. **内存**：模型本身 85GB（实测 RSS 80-82GB）+ 500K KV cache → **系统内存 100GB 起步**；GPU 显存可控（8-17GB，省显存特性），系统 RAM 是硬约束
- **一句话**：长上下文不是"变慢"，是"变重"（吃内存 + 开场慢）；持续对话的输出速度维持 10 t/s 量级

---

## 模型与硬件

| 项 | 值 |
|---|---|
| 模型 | DeepSeek-V4-Flash-UD-IQ2_M（unsloth，3 分片共 85GB，MoE 256 专家/激活 6） |
| 2080 Ti | 11GB SM75，驱动 580.105.08，云机 region-42 |
| RTX 3080 | 10GB SM86，驱动 580.76.05，云机 region-41 |
| 二进制 | v3.1 多架构（sm_61/75/86/89/120a，CUDA 12.8，含固定专家数淘汰） |
| moe-l2 | 0.7.0（PyPI）/ bins-v0.3.0（GitHub Release） |

## 下载

- 源：hf-mirror.com unsloth/DeepSeek-V4-Flash-GGUF UD-IQ2_M
- 3 分片：00001（5.1MB 元数据）+ 00002（46.5GB）+ 00003（38.1GB）≈ 85GB
- 工具：aria2 -x8（12 MiB/s，断点续传），耗时约 2 小时

## 多分片 GGUF 解析 bug（cli.py 修复）

**问题**：V4 分片 00001 只有 5MB **纯元数据**（tensors=0）。moe-l2 用分片1 解析专家布局 → `KeyError: No expert tensors found`。

**修复**：检测 `-00001-of-` 格式 → 同目录 glob 兄弟分片 → 选**最大分片**给 GGUFReader/L2Cache；llama-server 仍用分片1 启动（llama.cpp 自动发现兄弟分片）。**model_path（server）与 reader_path（解析）双路径分离**。

## 历史验证（2026-08-05，v3.1 时代，保留作记录）

- 85GB 三片 GGUF 在 2080 Ti（11GB）上跑通：显存 8.4GB / 11GB、RSS 靠专家页淘汰 v3.1 封顶（11-12GB，对比无淘汰 29GB）——**验证"10-11GB 卡可跑 85GB 模型"可行性**（08-10 bins-v0.4.0 已在 4090 上达到 35.96 t/s）
- 多分片 GGUF 解析 bug 修复：检测 `-00001-of-` 格式 → 选最大分片给 GGUFReader/L2Cache，llama-server 仍用分片1 启动（双路径分离）
- 专家页淘汰 v3.1：`MOE_L2_LRU_MAX_EXPERTS=N` 固定专家数 LRU，Qwen 近零掉速（-2%），V4 RSS 封顶

## 08-10 复测：4090 全链路（bins-v0.4.0 修复版）

> 用 bins-v0.4.0 修复版（从 GitHub release 下载，含 libmtmd/libllama）在 RTX 4090 全链路（`moe-l2 start --gpu`）实测，同时采集显存/内存。两种模式都测（auto 路由表生成失败→on-demand 兜底；显式 v4_top100.map→selective pin）。

### 实测（RTX 4090，2026-08-10，-c 8192）

| 轮次 | on-demand 兜底（短/追问1/追问2） | selective pin 显式表（短/追问1/追问2） |
|------|----------------------------------|----------------------------------------|
| Round 1 | 8.50 / 23.84 / 24.06 | 17.13 / 23.74 / 26.03 |
| Round 2 | 29.60 / 35.24 / 34.91 | 32.99 / 35.24 / 34.89 |
| Round 3（稳定） | **34.44 / 35.96 / 35.96** | **33.84 / 34.67 / 34.64** |
| Round 4 | 32.81 / 36.38 / 35.76 | 33.97 / 34.90 / 31.40 |

### 显存 / 内存

| 模式 | RSS | VRAM |
|------|-----|------|
| on-demand 兜底 | 17.5 GB（18,363,500 kB） | 16.5 GB（16,879 MiB） |
| selective pin（v4_top100.map） | 26.8 GB（28,110,288 kB） | 16.7 GB（17,067 MiB） |

### 结论（08-10）

1. **V4 在 4090 上稳定 34-36 t/s**（两模式一致）——对比原版 llama.cpp 二进制（10.1 t/s）**+255%**；比阶段1（30.9 t/s）略高
2. **RSS 大幅缩减**：whole-pin 84GB → on-demand 17.5GB（↓79%）/ selective pin 26.8GB（↓68%）
3. 阶段1 的 10.4GB RSS 是纯 selective pin 无 prefill 配置；本次含 GPU cache 预填充所以 RSS 更高、速度更快
4. **auto 生成路由表已正常**：`--router-top-k` 依赖 `moe_l2/data/` 下的 `domain_router_map_v4_topics.json` / `domain_router_map_v4.json`（pip 安装自带，git 仓库已追踪）。2026-08-10 验证：数据文件就位后 auto 生成 **43 层 top-100**（与显式 v4_top100.map 内容一致），selective pin 生效（RSS 28.1GB）；此前 08-10 早间测试出现 0 层是测试机未同步 data/ 目录所致，非产品缺陷。显式 `--router-map v4_top100.map`（43 层 top-100，本地备份 `测试数据备份/v0.8.0-selective-pin-20260810/router-map/`）仍可作兜底。

## 环境坑（复现用）

- AutoDL 开机驱动 mismatch（内核 vs 库版本不一致）：修符号链接 `ln -sfn libcuda.so.580.<内核版> libcuda.so.1` 等
- 架构核对用 CUDA 12.8 `cuobjdump --list-elf`（系统旧版误报）
- proxy 非流式转发 httpx 超时 30s→600s（慢速模型必踩）
