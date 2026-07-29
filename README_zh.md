# moe-l2

English | [**中文**](README_zh.md)

[![PyPI version](https://img.shields.io/pypi/v/moe-l2)](https://pypi.org/project/moe-l2/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

**8GB 显卡也能跑 32B MoE 大模型 — 省 91% 显存，一行 pip 搞定。**

| 你的显卡 | 正常能跑 | **用了 moe-l2** |
|----------|---------|-----------------|
| 4 GB | — | DeepSeek-V2-Lite (16B MoE) ✅ |
| **8 GB** | 7B 稠密模型 | **Qwen2.5-32B-A3B (32B MoE) ✅** |
| 12 GB | 13B 稠密模型 | DeepSeek-V2 (236B MoE) ✅ |
| 24 GB | 34B 稠密模型 | DeepSeek-V2 (236B MoE) ✅ |

```bash
pip install moe-l2
moe-l2 download-bins
moe-l2 start --model model.gguf --l2-size 4GB
```

连接 `localhost:11435`，现有工具（curl、Open WebUI、LangChain）无需改配置。

---

## moe-l2 能帮你解决什么

MoE 模型有几十上百个"专家"，但每步推理只激活其中几个。标准推理栈会把所有 expert 全塞进显存——80-95% 的内存被闲置权重白白占用。

**moe-l2 只让活跃 expert 驻留 GPU，其余留在内存或硬盘，需要时再换入。**

### 实测数据

基于 **DeepSeek-V2-Lite**（160 亿参数，64 expert，top-6，Q2_K 量化）：

| 模式 | GPU 显存 | 速度 | 意味着什么 |
|------|----------|------|-----------|
| 标准（全 expert 在 GPU） | 23.3 GB | 65 t/s | 需要 24 GB 显卡 |
| **moe-l2**（热缓存 expert） | **2.7 GB** | **~7 t/s** 生成 · **103 t/s** prompt | **4 GB 卡也能跑** |
| **节省** | **88% 显存** | 速度的 11% | 腾出 ~20 GB 做别的 |

不开 moe-l2，8 GB 显卡**根本无法加载这个模型**——直接 OOM。开了之后只用 2.7 GB（cache=0.5），还剩 5.3 GB 干别的。

> 我们在 RTX 4090 上对 **Qwen3.6-A3B**（32B MoE）和 **DeepSeek-V2-Lite**（16B MoE，64 expert）做了全量测试。GPU LRU expert 缓存（Phase 3）现已 **稳定运行**——7 种缓存级别 × 3 种对话类型全部通过，0 崩溃。生成速度 ~5-7 t/s（受 CPU expert compute 瓶颈限制），但 followup 对话的 prompt 处理因缓存命中提速 **10 倍**（~80-103 t/s）。完整报告：[Qwen3.6](references/qwen3.6-a3b-iq2m-benchmark.md) · [DS-V2-Lite](references/deepseek-v2-lite-q2k-benchmark.md)

---

## 系统架构

```
                         ┌─────────────────────────┐
  你的 prompt ──────────▶│  moe-l2 代理 (:11435)    │
                         │                          │
                         │  ┌─────────────────────┐ │
                         │  │ 领域预测器           │ │
                         │  │ (关键词 + 语义兜底)  │ │
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
                         │  llama.cpp / ollama      │
                         │  (:11434, CUDA GPU)      │
                         │  只放活跃 expert          │
                         └─────────────────────────┘
```

### 四级存储模型

```
L0 ─ CPU 路由器    门控路由 + 领域分类（你的 CPU）
 ↑
L1 ─ GPU 显存      活跃推理 — expert + KV cache（你的显卡）
 ↑
L2 ─ RAM 热缓存    基于领域的 LRU 缓存，mmap 共享内存（本项目的核心）
 ↑
L3 ─ SSD 冷存储    完整 expert 权重，按需从硬盘加载
```

越靠近 GPU 越快但空间越小，越远越慢但越便宜。调度器把活跃数据留在高速层，其余推到下层。

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
从 GitHub Release 拉取预编译的 CUDA llama-server（约 530 MB）。

### 3. 启动

**CPU 模式**（仅 expert 缓存，不省显存）：
```bash
moe-l2 start --model /path/to/model.gguf --l2-size 4GB
```

**GPU 模式**（A3 补丁版 llama-server，省 91% 显存）：
```bash
moe-l2 start --model /path/to/model.gguf --l2-size 4GB --gpu
```

代理启动在 `localhost:11435`，直接当普通 OpenAI/Ollama 用就行。

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
| `moe-l2 stop --port <端口>` | 停止代理 |

可选参数：
- `--model auto`：自动扫描 `/opt/data/models/*.gguf`
- `--l2-size 4GB` / `--l2-size 512MB`：目标缓存大小
- `--port 11435`（默认）
- `--gpu`：启用 GPU 模式（需要 CUDA + NVIDIA 显卡）

> **GPU 二进制**：不在 git 中追踪（~530 MB），运行时通过 `moe-l2 download-bins` 获取。

---

## 工作原理（简版）

1. 你的 prompt 到达 moe-l2 代理
2. 领域预测器分类（代码生成 → 数学 → 中文技术 ……）
3. L2 缓存从 SSD 预加载预测的 expert 到共享内存（`/dev/shm/`）
4. 请求转发到 llama.cpp/ollama — 热 expert 从 RAM 加载（~1150 µs）而非冷 SSD（~6500 µs）
5. 同一会话缓存命中率超过 85%

---

## 平台要求

- **仅 Linux x86_64** — 预编译二进制目标为 Linux AMD64（CUDA `.so` + `llama-server`）
- macOS、Windows、ARM Linux **暂不支持**
- **强烈建议使用 NVMe 固态硬盘**
- `--gpu` 模式需要 NVIDIA 显卡（下一阶段支持 GPU LRU expert 缓存）

---

## 更多数据

| 指标 | 标准 | moe-l2 |
|------|------|--------|
| Prompt 处理 | 110 t/s | 110 t/s |
| 生成速度 | 65 t/s | ~5-7 t/s |
| 显存占用 | 23.3 GB | 2.7 GB |
| 模型大小 / 显存比 | 0.26× | **2.2×** |

速度取舍是可预期的：expert 通过 PCIe 从系统内存加载。Phase 3 GPU LRU 缓存稳定运行，生成速度受 CPU expert compute 瓶颈限制（~5-7 t/s），但 followup prompt 处理因缓存命中提升至 ~80-103 t/s。

---

## 相关工作

[TencentYoutuResearch/Palm-Infra](https://github.com/TencentYoutuResearch/Palm-Infra) / **mollm** 是腾讯的 C++ 推理引擎，在 Apple Silicon / ARM Linux 上通过 SSD expert 卸载运行 MoE 模型（122B MoE + 16 GB 峰值 RSS，16.22 t/s）。

两者核心思路相同（expert 缓存 + 分层存储），但面向不同用户：

| 维度 | mollm（腾讯） | moe-l2 |
|------|-------------|--------|
| 平台 | Apple Silicon / ARM Linux | **Linux x86_64 + GPU（NVIDIA）** |
| 安装 | 源码编译（CMake + C++） | **pip install moe-l2** |
| 模型兼容性 | 仅 Qwen 系列 | **任何 llama.cpp 支持的 MoE** |
| 后端 | 自研 C++ 引擎 | **llama.cpp 代理 — 零迁移** |
| GPU 加速 | 仅 CPU（NEON） | **CUDA + GPU 显存** |
| 目标用户 | 移动 / 边缘设备开发者 | **桌面家庭用户** |

---

## 项目状态

- ✅ 领域预测器（关键词 + 可选语义）
- ✅ L2 缓存（mmap LRU、线程安全、异步预加载）
- ✅ 透明代理（HTTP/SSE 转发）
- ✅ CLI（start/stats，自动模型检测，GPU 模式）
- ✅ GPU 模式已验证（RTX 4090，DS-V2-Lite，~1.6 GiB 显存，95% 节省）
- ✅ PyPI 包（`moe-l2`）
- ✅ GPU LRU expert 缓存（已验证 Qwen3.6 + DS-V2-Lite，7 级别 × 3 类型，0 崩溃）

---

## 许可证

**Apache 2.0。** 详见 [LICENSE](LICENSE)。
