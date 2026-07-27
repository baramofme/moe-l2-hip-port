# moe-l2

**8GB 显卡也能跑 32B MoE 大模型。**  
moe-l2 最高可砍掉 91% 的显存占用 — 只有活跃的 expert 驻留 GPU，其余保留在 CPU 内存。

完整英文说明见 [README.md](README.md)。

| 你的显卡 | 正常能跑 | **用了 moe-l2** |
|----------|---------|-----------------|
| 4 GB | — | DeepSeek-V2-Lite (16B MoE) ✅ |
| 8 GB | 7B 稠密模型 | Qwen2.5-32B-A3B (32B MoE) ✅ |
| 12 GB | 13B 稠密模型 | DeepSeek-V2 (236B MoE) ✅ |
| 24 GB | 34B 稠密模型 | DeepSeek-V2 (236B MoE) ✅ |

```bash
pip install moe-l2
moe-l2 start --model model.gguf --l2-size 4GB
```

## 为什么需要这个东西

MoE（混合专家）模型动辄几十上百个"专家"，但每步推理只激活其中几个。标准推理栈会把所有 expert 权重全塞进 GPU 显存，80-95% 的内存被闲置权重白白占用。

moe-l2 会先判断你的输入属于哪个领域（代码、数学、中文技术等），把相关的 expert 预加载到基于 mmap 的 LRU 缓存中，其余留在 CPU/系统内存。GPU 里只放当前需要的 expert 集合。

## 基准测试

基于 **DeepSeek-V2-Lite**（160 亿参数，64 expert，top-6）Q2_K 量化：

| 模式 | GPU 显存 | 生成速度 |
|------|----------|---------|
| 标准（全部 expert 在 GPU） | 23.3 GB | 65 t/s |
| **moe-l2**（热缓存 expert） | **2.2 GB** | **8.6 t/s** |
| **节省** | **91% 显存** | 原始速度的 13% |

原本需要 24 GB 显卡才能跑的模型，moe-l2 把它压到 **2.2 GB** — 省下 22 GB 做其他事，或者让 8 GB / 4 GB 的卡也能跑。

## 平台要求

**仅 Linux x86_64。** 预编译的 GPU 二进制（CUDA `.so` 文件 + `llama-server`）编译目标为 Linux AMD64。暂不支持 macOS、Windows 和 ARM Linux。

## 快速开始

```bash
pip install moe-l2
moe-l2 start --model model.gguf --l2-size 4GB
```

代理启动在 `localhost:11435` — 你的所有现有工具（curl、Open WebUI、LangChain）无需任何修改即可使用。

## 工作原理

```
你的客户端 → moe-l2 代理 (:11435) → ollama/llama.cpp (:11434)
               ├── 领域预测器（关键词 + 可选语义）
               ├── L2 缓存（LRU、mmap /dev/shm/、异步预加载）
               └── 透明转发（SSE 流式）
```

1. 用户发送 prompt
2. 领域预测器进行分类（代码生成、数学、中文技术……）
3. L2 缓存将预测的 expert 预加载到共享内存
4. 请求转发到后端 — expert 已在缓存中
5. 同一会话中缓存命中率通常超过 85%

## 系统架构

moe-l2 使用四层分级调度模型，将 expert 存储与 GPU 内存解耦：

```
┌───────────────────────────────────────────────────┐
│  L0 — CPU Router                                  │
│  Gate network 路由 + 领域分类                      │
├───────────────────────────────────────────────────┤
│  L1 — GPU VRAM                                    │
│  活跃推理：expert + KV cache                       │
├───────────────────────────────────────────────────┤
│  L2 — RAM 热缓存（本项目）                         │
│  基于领域的 expert 预加载（LRU、mmap）              │
├───────────────────────────────────────────────────┤
│  L3 — SSD 冷存储                                  │
│  完整 expert 权重，按需加载                         │
└───────────────────────────────────────────────────┘
```

每层就是一个存储层。离 GPU 越近越快但空间越小，越远越慢但越便宜。调度器把活跃数据留在高速层，其余推到下层。

## 功能特性

- **零配置** — 装完就能用，指定模型完事
- **透明代理** — 客户端无需改配置，兼容任何 OpenAI 工具
- **双模式预测** — 关键词模式（零额外依赖）或关键词 + 语义嵌入混合模式
- **LRU 淘汰** — 缓存大小可配置（`--l2-size 512MB` 到 `16GB`）
- **GPU 模式** — A3 补丁版 llama-server 支持 expert 卸载，RTX 4090 已验证
- **库 API** — `from moe_l2 import predict, L2Cache` 供嵌入式使用

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

> **GPU 二进制**：仓库不追踪 500MB+ 的 .so 文件。预编译二进制通过 GitHub Release 分发。运行 `moe-l2 download-bins` 即可获取。

## 更多数据

| 指标 | 标准 | moe-l2 |
|------|------|--------|
| Prompt 处理 | 110 t/s | 110 t/s |
| 生成速度 | 65 t/s | 8.6 t/s |
| 显存占用 | 23.3 GB | 2.2 GB |
| 模型大小 / 显存比 | 0.26× | **2.7×** |

速度取舍是可以预期的：expert 通过 PCIe 从系统内存加载。这是有意为之的设计选择，适合优先考虑内存效率而非峰值吞吐的场景 — 家庭实验室、边缘部署、预算有限的硬件。

> **注：** 8.6 t/s 是当前 Phase 2 的实测值（expert 在 CPU 侧，每步通过 PCIe 加载）。下一阶段优化 — GPU LRU expert 缓存 — 将热 expert 保留在显存中，目标 **40+ t/s**，消除缓存命中时的 PCIe 瓶颈。

## 相关工作

[TencentYoutuResearch/Palm-Infra](https://github.com/TencentYoutuResearch/Palm-Infra) / **mollm** 是腾讯的 C++ 推理引擎，在 Apple Silicon / ARM Linux 上通过 SSD expert 卸载运行 MoE 模型 — 122B MoE 模型配合 16 GB 峰值 RSS 达到 16.22 t/s。

moe-l2 和 mollm 核心思路相同（expert 缓存 + 分层存储），但面向不同用户：

| 维度 | mollm（腾讯） | moe-l2 |
|------|-------------|--------|
| 平台 | Apple Silicon / ARM Linux | **Linux x86_64 + GPU（NVIDIA）** |
| 安装 | 源码编译（CMake + C++） | **pip install moe-l2** |
| 模型兼容性 | 仅 Qwen 系列 | **任何 llama.cpp 支持的 MoE**（DeepSeek、Qwen、Mixtral……） |
| 后端 | 自研 C++ 引擎 | **llama.cpp 代理** — 零迁移成本 |
| GPU 加速 | 仅 CPU（NEON） | **CUDA + GPU 显存** |
| 目标用户 | 移动 / 边缘设备开发者 | **桌面家庭用户** |

对于已经在用 llama.cpp 的桌面用户，moe-l2 一行命令即可添加 expert 缓存，无需更换引擎。

## 项目状态

- ✅ 领域预测器（关键词 + 可选语义）
- ✅ L2 缓存（mmap LRU、线程安全、异步预加载）
- ✅ 透明代理（HTTP/SSE 转发）
- ✅ CLI（start/stats 带自动模型检测、--gpu 模式）
- ✅ GPU 模式已验证（RTX 4090，DS-V2-Lite，~1.6 GiB 显存）
- ✅ PyPI 包（moe-l2）
- 🔲 GPU LRU expert 缓存（热 expert 留在显存，减少 PCIe 传输）— 下一阶段

## 许可证

**Apache 2.0。** 详见 [LICENSE](LICENSE)。
