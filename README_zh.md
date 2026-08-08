# moe-l2

[English](README.md) | [**中文**](README_zh.md)

[![PyPI version](https://img.shields.io/pypi/v/moe-l2)](https://pypi.org/project/moe-l2/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

**MoE 专家卸载（expert offload）低显存方案 — 8GB 显卡也能跑 100B+ MoE 大模型（DeepSeek、Qwen、Mixtral），省 93% 显存，一行 pip 搞定。**

> ⭐ **觉得有用？点个 Star** —— 让更多需要的人发现它。[★ 去 GitHub 点赞](https://github.com/yalun753/moe-l2)

| 你的显卡 | 正常能跑 | **用了 moe-l2** | **实测速度**（RTX 4090） |
|----------|---------|-----------------|----------------------|
| 4 GB | — | DeepSeek-V2-Lite (16B MoE) ✅ | **37.9 t/s** |
| **8 GB** | 7B 稠密模型 | **Qwen3.6-A3B (32B MoE) ✅** | **50.2 t/s** |
| 10-11 GB | — | **DeepSeek-V4-Flash（157B MoE，85 GB 文件）✅** | **10.1 t/s** |

> 速度 = RTX 4090 实测（2026-08-07，on-demand pin + A3 cache 2048，多架构包）；2080 Ti：Qwen 24.5 t/s、DS 6.89 t/s、V4 0.89-1.07 t/s。详见 [models-benchmark.md](references/models-benchmark.md)。

**DeepSeek-V4-Flash（157B 参数 / 85GB 文件，256 专家、激活 6）也能跑**——RTX 4090 实测 **10.1 t/s**（on-demand pin + A3 cache，显存 17.4GB）；2080 Ti（11GB）和 RTX 3080（10GB）实测：显存 8.3-9.1GB、RSS 靠专家页淘汰 v3.1 封顶、速度 0.89-2.22 t/s（卡算力极限）。完整报告：[deepseek-v4-flash-verify-20260805.md](references/deepseek-v4-flash-verify-20260805.md) · **全部已测模型汇总：[models-benchmark.md](references/models-benchmark.md)**

**一行安装（Linux x86_64 + NVIDIA 显卡）：**

```bash
curl -fsSL https://raw.githubusercontent.com/yalun753/moe-l2/main/scripts/install.sh | bash
```

安装脚本自动检测显卡/驱动/Python → 从 PyPI 装 moe-l2 → 下载预编译 CUDA 二进制 → 可选下载演示模型（Qwen3.6-35B-A3B，约 11.5GB，断点续传）→ 自检。

**手动安装：**

```bash
pip install moe-l2
moe-l2 download-bins
moe-l2 model download --model qwen3.6-35b   # 可选：下载演示模型（约 11.5GB）
moe-l2 start --model model.gguf --gpu
```

常用命令：

```bash
moe-l2 doctor                        # 环境自检（GPU/CUDA/Python/磁盘）
moe-l2 model list                    # 查看可下载的模型
moe-l2 model download --model <name> # 下载模型（断点续传，走 hf-mirror）
```

连接 `localhost:11435`，现有工具（curl、Open WebUI、LangChain）无需改配置。

---

## moe-l2 能帮你解决什么

MoE 模型有几十上百个"专家"，但每步推理只激活其中几个。标准推理栈会把所有 expert 全塞进显存——80-95% 的内存被闲置权重白白占用。

**moe-l2 只让活跃 expert 驻留 GPU，其余留在内存或硬盘，需要时再换入。**

### 实测数据

基于 **RTX 4090** 实测（2026-08-07，on-demand pin 主路径：lazy mmap 惰性加载 + 首次触碰合并注册整个专家 tensor + A3 cache 2048 槽）：

| 模式 | GPU 显存 | 速度 | 意味着什么 |
|------|----------|------|-----------|
| 标准（全 expert 在 GPU） | 23.3 GB | 65 t/s | 需要 24 GB 显卡 |
| **moe-l2**（on-demand pin 专家，GPU 计算） | **1.6-2.9 GB** | **DS 37.9 t/s · Qwen 50.2 t/s** | **4 GB 卡也能跑** |
| **节省** | **93% 显存** | 全 GPU 速度的 ~58% | 腾出 ~20 GB 做别的 |

不开 moe-l2，8 GB 显卡**根本无法加载这个模型**——直接 OOM。开了之后 32B MoE 只占 ~2.9 GB（on-demand pin 专家，GPU 计算），还剩 5 GB 干别的。

> 我们在 RTX 4090 上对 **Qwen3.6-A3B**（32B MoE）和 **DeepSeek-V2-Lite**（16B MoE，64 expert）做了全量测试。**2026-08-07 主路径升级为 on-demand pin**（mmap 惰性加载 + 首次触碰合并注册整个专家 tensor + A3 cache 2048 槽）：专家驻留 CPU RAM（零显存），调度器每步只把**激活的专家**拷到 GPU 直算，热专家缓存在 GPU 显存。DS-V2-Lite **12.5 → 37.9 t/s**（+200%），Qwen3.6-A3B **10 → 50.2 t/s**（+400%，超 pre-lazy 46.5）。完整报告：[Qwen3.6](references/qwen3.6-a3b-iq2m-benchmark.md) · [DS-V2-Lite](references/deepseek-v2-lite-q2k-benchmark.md) · [cache-sched-layer](references/cache-sched-layer-benchmark.md) · [models-benchmark](references/models-benchmark.md) · **为什么是 host-buffer？完整方案演进史：[design-decisions.md](references/design-decisions.md) / [design-decisions_EN.md (English)](references/design-decisions_EN.md)**

### 低内存模式：动态 pin 集合（2026-08-09）

默认 on-demand pin 首次触碰时注册**整个专家 tensor**——最快（DeepSeek-V4-Flash 10.1 t/s），但 85GB 模型会把 ~82GB 专家页钉在内存。内存受限机器用 **动态 pin 集合**：只注册实际激活的专家（逐专家、按连续组注册），LRU 淘汰器把冷专家 unregister + madvise 释放。RTX 4090 / DeepSeek-V4-Flash-UD-IQ2_M（85GB）实测：**RSS 84GB → 17-24GB**（由 `MOE_L2_LRU_MAX_EXPERTS` 调节），速度 4-5 t/s（新专家首次触碰要付一次缺页读盘；V4 路由极分散——30 轮会话会触及 ~29GB 不同专家）。小 MoE 模型（Qwen3.6-A3B / DS-V2-Lite）工作集小，**保持满速**。

开启（32GB 内存机器示例）：

```bash
LLAMA_EXPERT_LOG=1 MOE_L2_LRU=1 MOE_L2_LRU_MAX_EXPERTS=12000 \
MOE_L2_EVICT_MB=20000 MOE_L2_EVICT_INTERVAL=4 \
MOE_L2_PIN_LAYERS=0-2,14-20,36-37 GGML_OP_OFFLOAD_MIN_BATCH=1 \
llama-server -m model.gguf -ngl 99 -c 2048 --no-webui
```

`MOE_L2_PIN_LAYERS` 把通用层/稀疏层永久 pin（V4 上 L0-L2 + 稀疏层 = ~5.4GB 免费午餐）。调 `MOE_L2_LRU_MAX_EXPERTS`：2000 ≈ 17GB RSS（紧，较慢）/ 12000 ≈ 24GB RSS（V4 约 5.3 t/s）/ 不设 = 关闭淘汰。**权衡总结**：whole-pin = 最快（V4 10.1 t/s）但 82GB 内存；动态 pin = 17-24GB 内存但 V4 4-5 t/s（小模型不受影响）。

### 多架构二进制（bins-v0.3.2，2026-08-09）

**一个二进制兼容所有 NVIDIA 消费卡**——GTX 1080（sm_61）到 RTX 50 系（sm_120a）。CUDA 12.8 编译，无需按显卡单独编译，`moe-l2 download-bins` 自动拉取。bins-v0.3.2 含 **on-demand pin 主路径** + **动态 pin 集合（低内存模式）** + 专家页淘汰 v3.1（`MOE_L2_LRU_MAX_EXPERTS=N`）+ 分层 pin（`MOE_L2_PIN_LAYERS`）+ A3 cache 2048 槽 + cuda-libs（无 libnccl，单卡不需要）。

| 显卡 | 架构 | DS-V2-Lite 生成 | Qwen3.6-A3B 生成 | 显存 |
|------|------|----------------|-----------------|------|
| RTX 2080 Ti | sm_75（Turing） | 6.89 t/s | 11.15 t/s | ~1.0-2.4 GB |
| RTX 3080 Ti | sm_86（Ampere） | 12.25 t/s | 13.28 t/s | ~1.1-2.2 GB |
| RTX 5090 | sm_120a（Blackwell） | 16.63 t/s | 9.71 t/s | ~1.3-2.5 GB |
| RTX 4090* | sm_89（Ada） | 39.0 t/s | 51.5 t/s | 1.6-2.9 GB |

\* 4090 为多架构包实测（2026-08-07，on-demand pin + cache 2048，CUDA 12.8）；2080 Ti / 3080 Ti / 5090 为 v3.1 多架构包（bins-v0.3.0）实测。Qwen 单轮 24.5 t/s（2080 Ti，bins-v0.3.2，旧 host-buffer 11.15 翻倍）。

> 2080 Ti（SM75）、3080 Ti（SM86）、5090（SM120a）已用多架构包实测；3080 Ti 比旧 CUDA 11.8 单架构版**快 55%**（12.25 vs 7.88 t/s）。注意：llama.cpp 76f46ad 对 SM120a（50 系）内核优化还不成熟——5090 比 3080 Ti 只快 36%（DS）甚至慢 27%（Qwen），换新版 llama.cpp 重编后 50 系速度有望提升。完整报告：[multi-arch-three-gpu-benchmark.md](references/multi-arch-three-gpu-benchmark.md)

### 可视化演示（RTX 4090，2026-08-02）

| Qwen3.6-35B-A3B（32B MoE）— 标准 vs moe-l2 | DeepSeek-V2-Lite（16B MoE）— 8GB 卡 vs 24GB 卡 |
|---|---|
| ![Qwen 显存对比](examples/demo-assets/fig1-qwen-vram.png) | ![DS 显存对比](examples/demo-assets/fig2-ds-vram.png) |

一句话总结：**显存省 93% · 速度保留 58% · 模型/显存比 3.1×** —— 8 GB 卡跑出原本 24 GB 卡的效果（图为 2026-08-02 host-buffer 构建实测；2026-08-07 on-demand pin 最新：DS 37.9 t/s @ 2.0 GB / Qwen 50.2 t/s @ 2.9 GB）：

![moe-l2 汇总](examples/demo-assets/fig3-summary.png)

实机录屏：Qwen3.6-35B-A3B 生成 **3200 tokens，全程显存钉在 ~2.4 GB**（41.6 t/s）—— 曲线全程平直，远低于 8 GB 红线：

[`examples/demo-assets/demo-vram-animation.mp4`](examples/demo-assets/demo-vram-animation.mp4)（45 秒，1280×720）· 原始采样：[`examples/demo-assets/rec_data.csv`](examples/demo-assets/rec_data.csv) · 生成全文：[`examples/demo-assets/rec_full.txt`](examples/demo-assets/rec_full.txt)

---

## 系统架构

```
                         ┌─────────────────────────┐
   你的 prompt ──────────▶│  moe-l2 代理 (:11435)    │
                         │                          │
                         │  ┌─────────────────────┐ │
                         │  │ 领域预测器           │ │
                         │  │ (关键词+TF-IDF+语义) │ │
                         │  └────────┬────────────┘ │
                         │           │ 预测领域       │
                         │           ▼               │
                         │  ┌─────────────────────┐ │
                         │  │ L2 缓存 (RAM)       │ │
                         │  │ LRU · mmap · 异步   │ │
                         │  │ 从硬盘预加载         │ │
                         │  └────────┬────────────┘ │
                         │           │ 转发           │
                         └───────────┼───────────────┘
                                     ▼
                         ┌─────────────────────────┐
                         │  llama-server (:11436)   │
                         │  on-demand pin 专家：    │
                         │  lazy mmap 零显存，首次  │
                         │  触碰即 pinned，GPU 经   │
                         │  PCIe DMA 直读；热专家   │
                         │  驻留 VRAM (A3 LRU)，   │
                         │  冷页淘汰 RSS 封顶       │
                         └─────────────────────────┘
```

### 数据驻留层级（2026-08-07 架构）

```
L0 ─ CPU 路由器    门控路由 + 领域分类（你的 CPU）
 ↑
L1 ─ GPU 显存      激活 expert 计算 + KV cache（你的显卡，只放激活权重）
 ↑
L2 ─ CPU RAM       全部专家权重 lazy mmap 驻留，首次触碰即 pinned（零显存）
 ↑
L3 ─ SSD 冷存储    GGUF 文件 mmap，冷专家页按需读入 + v3.1 淘汰（RSS 封顶）
```

专家权重整体 lazy mmap 驻留 CPU RAM（零显存），首次激活即 pinned，GPU 经 PCIe DMA 直读；热专家驻留 VRAM（A3 LRU 2048 槽），冷页 v3.1 淘汰保持 RSS 封顶——这就是为什么 32B MoE 只占 ~2.9 GB 显存、85GB V4 也能在 11GB 卡上跑。

---

## 适用场景

### ✅ 适合
- **单人聊天** — 用 4-12 GB 显卡跑大 MoE 模型
- **测试和实验** — 在预算硬件上玩 MoE 架构
- **家庭实验室 / 边缘部署** — 每一 GB 显存都珍贵
- **研究 expert 缓存**、分层调度、领域感知预加载

### ❌ 不适合
- 高并发 API 服务（频繁换 expert 产生 I/O 瓶颈）
- 对延迟敏感的应用（SSD 缓存缺失导致速度波动）
- 机械硬盘作存储 — **需要 NVMe 固态**

---

## 快速开始

### 1. 安装

```bash
pip install moe-l2
```

### 2. 下载 GPU 二进制（仅 `--gpu` 模式需要）

```bash
moe-l2 download-bins
```
从 GitHub Release 拉取预编译的 CUDA llama-server（bins-v0.3.2，约 1.9 GB 多架构全兼容包，含 cuda-libs）。

### 3. 启动

**GPU 模式（推荐，on-demand pin 专家 GPU 直算，省 93% 显存）：**
```bash
moe-l2 start --model /path/to/model.gguf --gpu
```

**纯代理模式**（不省显存，仅 expert 缓存）：
```bash
moe-l2 start --model /path/to/model.gguf --l2-size 4GB
```

代理启动在 `localhost:11435`，直接当普通 OpenAI 兼容接口用就行（GPU 模式下后端 llama-server 监听 11436）。

### 4. 看统计

```bash
moe-l2 stats
# → 命中率: 85% · 槽位: 320/960 · 当前领域: codegen
```

---

## CLI 参考

| 命令 | 说明 |
|------|------|
| `moe-l2 start --model <路径> --l2-size <大小>` | 启动代理 + 缓存 |
| `moe-l2 start --model <路径> --gpu` | 使用 GPU 加速的 llama-server 启动 |
| `moe-l2 stats --port <端口>` | 查看实时缓存统计 |
| `moe-l2 download-bins [--release TAG]` | 从 GitHub 下载预编译 GPU 二进制 |
| `moe-l2 collect --model <路径>` | 采集 MoE 路由数据 → `~/.moe-l2/maps/domain_expert_map.json` |
| `moe-l2 stop --port <端口>` | 停止代理 |

可选参数：
- `--model auto`：自动扫描 `/opt/data/models/*.gguf`
- `--l2-size 4GB` / `--l2-size 512MB`：目标缓存大小
- `--port 11435`（默认）
- `--gpu`：启用 GPU 模式（需要 CUDA + NVIDIA 显卡）

> **GPU 二进制**：不在 git 中追踪（`llama_bins.tar.gz`，bins-v0.3.2 约 1.9 GB 多架构包，sm_61/75/86/89/120a 一个二进制兼容所有 NVIDIA 消费卡，含 cuda-libs），运行时通过 `moe-l2 download-bins` 获取。

---

## 工作原理（简版）

1. 你的 prompt 到达 moe-l2 代理
2. 领域预测器分类（代码生成 → 数学 → 中文技术 ……）
3. 专家权重 lazy mmap 驻留 CPU RAM（**零显存**），首次激活即 pinned（on-demand pin）——不再整体塞进 GPU
4. GPU 经 PCIe DMA 直读激活专家（cuBLAS），热专家驻留 VRAM（A3 LRU 2048 槽），冷页 v3.1 淘汰
5. 可选 sched-cache（`GGML_CUDA_EXPERT_CACHE=0.25`）：命中热专家走 D2D 免 PCIe，DS 类模型 Prompt +211%

---

## 平台要求

- **仅 Linux x86_64** — 预编译二进制目标为 Linux AMD64（CUDA `.so` + `llama-server`）
- macOS、Windows、ARM Linux **暂不支持**
- **强烈建议使用 NVMe 固态硬盘**
- `--gpu` 模式需要 NVIDIA 显卡（CUDA 后端）

---

## 更多数据

| 指标 | 标准 | moe-l2 |
|------|------|--------|
| Prompt 处理（DS-V2-Lite） | 110 t/s | 99 t/s · **308 t/s**（sched-cache=0.25） |
| 生成速度（DS-V2-Lite） | 65 t/s | 37.9 t/s · 39.2 t/s（sched-cache=0.25） |
| 生成速度（Qwen3.6-A3B） | — | 50.2 t/s |
| 显存占用 | 23.3 GB | **2.0 GB** |
| 模型大小 / 显存比 | 0.26× | **3.1×** |

速度取舍是可预期的：专家驻留 CPU RAM（mmap 惰性 + on-demand pin，零显存），调度器每步只把激活的专家拷到 GPU。2026-08-07 on-demand pin 主路径：DS-V2-Lite 生成 37.9 t/s（+200%）、Qwen3.6-A3B 50.2 t/s（+400%，超 pre-lazy 46.5）；DS 开 sched-cache=0.25 后 prompt 处理 308 t/s（+211%）。

---

## 相关工作

### AirLLM（lyogavin/airllm，~29k stars）

AirLLM 是通用型超大模型分层加载方案，以 **Transformer 整层**为调度粒度：推理全程仅在显存保留单一层权重，其余落盘交换，实现极致低显存门槛（4GB 跑 70B）。但存在三个短板：① 每生成一个 Token 都要反复读写磁盘加载/释放整层权重，IO 开销巨大，对话生成速度极低；② 无 MoE 专属路由预测与专家热缓存（2026-07 才开始逐专家流式加载，Kimi K3），重复提问持续触发大量磁盘读取；③ 基于 Hugging Face Transformers 原生开发，无内置 OpenAI 服务接口，难以直接对接 Open WebUI、LangChain 等工具链。

| 维度 | AirLLM | moe-l2 |
|------|--------|--------|
| 调度最小单元 | Transformer 完整网络层 | **MoE 独立专家（稀疏最优）** |
| 目标模型 | 全模型兼容（稠密 + MoE） | **深度优化 MoE（DeepSeek/Qwen/Mixtral）** |
| 底层权重格式 | Hugging Face 原生权重 | **GGUF（llama.cpp 生态）** |
| 运行平台 | Windows/macOS/Linux 全兼容 | Linux x86_64 + NVIDIA 显卡 |
| MoE 场景内存 | 整层落盘交换，无热缓存 | **85GB V4：VRAM 8.3GB + RSS 11-12GB 封顶（实测）** |
| MoE 场景速度 | 逐层反复磁盘交换，适合批量离线 | 热专家缓存减少磁盘 IO，支持实时对话（Qwen 全链路 9.3 t/s 实测） |
| 服务接口 | 仅 Python 代码调用，无内置 Web 服务 | **内置 OpenAI 兼容代理（11435），开箱即用** |
| 显卡适配 | 原生 transformers，CUDA 适配繁琐 | **download-bins 多架构内核，10 系~50 系 N 卡全覆盖** |
| 超大分片模型 | 无针对性适配 | **原生修复多分片元数据解析 BUG，85GB 3 分片 V4 稳定** |

**选择建议**：选 moe-l2——本地跑 DeepSeek/Qwen 等 MoE 日常聊天、8G~12G 老消费 N 卡兼顾显存与速度、需要 OpenAI API 对接工具链、使用多分片超大 GGUF。选 AirLLM——需要运行稠密大模型、使用 Windows/macOS/AMD 或 CPU-only 环境（moe-l2 当前仅支持 Linux + NVIDIA）、仅一次性批量生成、只能用原生 HF 权重。

---

## 项目状态

- ✅ 领域预测器（关键词 + 可选语义）
- ✅ L2 缓存（mmap LRU、线程安全、异步预加载）
- ✅ 透明代理（HTTP/SSE 转发）
- ✅ CLI（start/stats/collect/embed-map/download-bins，自动模型检测，GPU 模式）
- ✅ host-buffer 专家 GPU 直算（2026-08-02）：DS-V2-Lite 12.5 → 37.5 t/s、Qwen3.6-A3B 10 → 46.8 t/s，VRAM 1.6 / 2.1 GB — 专家驻留 CPU pinned 零显存，调度器只拷激活专家
- ✅ **on-demand pin 主路径（2026-08-07）**：mmap 惰性加载 + 首次触碰合并注册整个专家 tensor + A3 cache 2048 槽 → Qwen **50.2** / DS **37.9** / V4 **10.1** t/s（4090），V4 从 1.7-2.0 提升 5 倍；修复 CUDA 11.8 跨 register 区间拷贝崩溃
- ✅ cache 挂 sched 拷贝层（2026-08-02）：DS 类模型 Prompt 99 → 308 t/s（+211%，cache=0.25，VRAM 不变）；Qwen/Mixtral 无收益不开
- ✅ **DeepSeek-V4-Flash（157B MoE）验证通过（2026-08-05）**：85GB 三片 GGUF 在 2080 Ti（11GB）和 RTX 3080（10GB）上跑通——VRAM 8.3-9.1GB、RSS 靠专家页淘汰 v3.1 封顶（`MOE_L2_LRU_MAX_EXPERTS` 固定专家数 LRU）、多分片 GGUF 解析修复已随 0.7.0 发布。速度 0.89-2.22 t/s（卡算力极限）。[完整报告](references/deepseek-v4-flash-verify-20260805.md)
- ✅ PyPI 包（`moe-l2`）

---

## 许可证

**Apache 2.0。** 详见 [LICENSE](LICENSE)。