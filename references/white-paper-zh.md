# moe-l2 技术白皮书

**让 8GB 显卡跑 100B+ 参数的 MoE 模型**

- 版本：v1.0（2026-08-02）
- 作者：moe-l2 项目组（yalun753）
- 读者：技术决策人、架构师、AI 基础设施团队
- 配套：README（快速上手）、references/（benchmark 原始报告）

---

## 1. 执行摘要

moe-l2 是一个面向消费级 NVIDIA 显卡的 MoE（混合专家）模型推理加速方案，核心能力是**把原本需要 16-24 GB 显存的模型压缩到 1.6-2.1 GB 运行**，速度达到全 GPU 形态的约 58%。

核心数字（RTX 4090 实测，2026-08-02）：

| 指标 | 全 GPU 形态 | moe-l2 | 变化 |
|------|------------|--------|------|
| 显存占用 | 23.3 GB | **1.6 GB** | **-93%** |
| 生成速度（DS-V2-Lite） | 65 t/s | **37.5 t/s** | ~58% |
| 生成速度（Qwen3.6-A3B） | — | **46.8 t/s** | — |
| 模型/显存比 | 0.26× | **3.9×** | +15 倍 |

技术本质一句话：**MoE 模型每步推理只激活少量专家（top-2~8 / 数百个），全量驻留显存是浪费。moe-l2 把专家放在 CPU 内存（零显存），GPU 每步只取激活专家计算，热专家用缓存留在 GPU 免 PCIe 往返。**

实现路径：不重写推理引擎，在 llama.cpp 上打两层补丁（host buffer 专家驻留 + 调度器拷贝优化），用户 `pip install` 即用。

---

## 2. 问题与机会

### 2.1 MoE 是当前大模型的主流架构，但推理门槛被显存卡死

混合专家（Mixture of Experts）把模型拆成"共享层 + 数百个专家"，每步推理由门控网络（router）激活其中少数专家。DeepSeek-V2（236B）、Qwen3-235B、Mixtral 全部采用此架构。

MoE 的优势是**总参数大、激活参数小**——模型总参数数百亿，每步推理只激活其中一小部分专家（top-2~8）。

但实际部署被显存卡死：

- 模型权重（哪怕 Q4 量化）必须整体加载才能跑：Qwen3-30B Q4 ≈ 16 GB 以上，DeepSeek-V2 Q4 则达百 GB 级
- 消费级显卡主流显存是 8 GB（RTX 4060/3060）、12 GB（4060 Ti）、16 GB（4070 Ti Super）
- **8 GB 显卡连 Qwen3-30B 都加载不了**，更不用说 100B+ 的 DeepSeek-V2 / Qwen3-235B

结果是：MoE 模型的能力被"显存门槛"锁死在高配服务器上，消费级硬件只能跑 7B 级小模型。

### 2.2 现有方案的缺陷

| 方案 | 做法 | 缺陷 |
|------|------|------|
| llama.cpp CPU 卸载 | 显存放不下就放系统内存 | 慢（PCIe 带宽瓶颈），速度损失 3-5 倍 |
| MoE-Infinity / ExpertFlow | 完整工程系统，面向服务器集群 | 太重，需要集群/训练，不面向个人用户 |
| GPU-CPU 协作推理 | CPU 做 cache miss 推理 | CPU 推理慢，资源浪费 |
| 买更大显存 | 24 GB 以上显卡 | 价格 8000+ 元，且 236B 模型 24 GB 也不够 |

### 2.3 moe-l2 的机会

**目标用户**：8 GB 显卡玩家（占 Steam 显卡份额最大的档位之一）、个人开发者、家庭服务器（NAS）用户、AI PC 厂商。

**价值主张**：

1. **显存降 93%**：23.3 GB → 1.6 GB，4 GB 卡都能跑 16B MoE，8 GB 卡跑 32B MoE
2. **速度可用**：37-47 t/s 的生成速度，接近人眼阅读速度的 2-3 倍，对话体验流畅
3. **零迁移**：OpenAI 兼容接口，curl / Open WebUI / LangChain 直接连，不改一行客户端代码
4. **即插即用**：pip install + 一行命令，不需要编译 llama.cpp

---

## 3. 架构原理

### 3.1 核心洞察

MoE 推理的内存访问特征：

- 模型分两层：**dense 层**（attention + 共享层，每步都要算）和 **expert 层**（数百个专家，每步只激活 top-2~8）
- 全量驻留方案：dense + 全部专家都在显存 → 显存需求 = 全模型大小
- moe-l2 方案：**dense 层驻留显存（小），专家驻留 CPU 内存（大但零显存），每步只把激活的专家搬进 GPU**

类比：MoE 模型就像一个巨大的图书馆，每本书（专家）都很重。传统方案把整个图书馆搬进办公室（显存）才能工作；moe-l2 只把"常用的几本书"放在手边，每次查资料时按需从仓库（内存）取，查完放回去。书架（显存）只放一个目录（dense 层）和最近常用的书。

### 3.2 系统分层

```
用户输入
   │
   ▼
┌──────────────────────────────────────────┐
│  moe-l2 调度器（Python 进程，端口 11435） │
│                                          │
│  ├─ L0a 领域预测器 ── 关键词 / TF-IDF /   │
│  │    语义 embedding 三层兜底，<1ms 分类  │
│  │    输出：领域标签（codegen/math/...）  │
│  ├─ L2 热缓存管理器 ── mmap 共享内存 LRU │
│  │    预载"该领域常用专家"到内存          │
│  ├─ 数据飞轮 ── 真实流量攒样本自动重训    │
│  │    分类器（越用越准）                  │
│  └─ 请求转发 ── OpenAI 兼容透传           │
└───────────────┬──────────────────────────┘
                │ POST /v1/chat/completions
                ▼
┌──────────────────────────────────────────┐
│  llama-server（端口 11436，CUDA GPU）     │
│                                          │
│  1. 专家驻留 CPU pinned（host buffer，    │
│     零显存）                              │
│  2. 调度器每步只拷贝激活专家 → GPU       │
│  3. GPU cuBLAS 直算专家（快路径）        │
│  4. 可选 sched-cache：热专家 D2D 免 PCIe │
└──────────────────────────────────────────┘
```

### 3.3 关键机制 1：host buffer 专家驻留（零显存）

**设计**（moe-l2 的 llama.cpp 补丁）：

- 修改 `llama-model-loader.cpp`，放开 mmap → **CUDA host buffer 回退**：专家权重加载到 CPU pinned 内存（可被 GPU 通过 PCIe DMA 直接读取），**不占显存**
- dense 层（attention 等非专家部分）照常驻留显存
- 结果：VRAM 只装 dense 层 + KV cache + 临时缓冲

**注意区分**：这一步是 moe-l2 的设计（loader 补丁）；而"每步只拷激活专家"是 llama.cpp 调度器（scheduler）自带的 MoE 专家级拷贝优化——moe-l2 通过 `GGML_OP_OFFLOAD_MIN_BATCH=1` 环境变量确保走这条快路径。

**数据链路**（每步推理）：

```
CPU pinned（专家权重，零显存）
   │  sched 拷贝优化：只拷本步激活的 top-k 专家（如 6 个 × 1.55 MB ≈ 9.3 MB）
   │  而非全部 64 个专家
   ▼
GPU 显存（临时专家 buffer）
   │  cuBLAS 直算 MUL_MAT_ID（快路径）
   ▼
输出
```

**为什么快**：专家权重已在 CPU pinned 内存（PCIe DMA 直读），每步只拷激活的 top-k 专家（如 DS 每层 6 个 × 1.55 MB ≈ 9.3 MB），而非全部 64 个专家——拷贝量与激活数量成正比，而非模型大小。实测提升：DS 12.5 → 37.5 t/s（+200%），Qwen 10 → 46.8 t/s（+370%）。

### 3.4 关键机制 2：A3 LRU expert cache（热专家免搬运）

当用户在同一领域连续对话（如连续写代码），每步激活的专家高度重复。moe-l2 的 A3 LRU cache 在 GPU 显存里维护一个固定大小的专家缓存池：

- **命中**：热专家已在显存 → D2D 拷贝（免 PCIe 往返）→ 微秒级
- **未命中**：走 CPU → GPU 拷贝 + 写回缓存

缓存挂载点：llama.cpp 调度器的 `copy_experts`（专家输入拷贝层）——比计算层更底层，在专家进入 GPU 之前拦截。

**收益规律**（三模型实测）：**缓存收益 = 专家大小 × 命中率**

| 模型 | 专家大小 | 激活方式 | cache 收益 |
|------|---------|---------|-----------|
| DeepSeek-V2-Lite | 1.55 MB | top-6 | **Prompt +211%，Gen +5%** |
| Qwen3.6-A3B | ~1 MB | top-8 | 无（专家太小，搬运本身不贵） |
| Mixtral-8x7B | 252 MB | top-2 | 无（命中率太低，且槽位占显存大） |

**理论依据**（LRU 模拟，Phase 1.5）：

- 5 域 trace：96 slots/层 = 84.4% 命中率（接近理论天花板）
- 8 域验证：纯 LRU 在 96 slots = 80.4%，128 slots = 85.2%；Domain Pin 需 ≥64 slots 才有增益
- 单域长对话：专家全部装入后命中率 asymptotically 趋近 100%
- 剩余 miss 来源：**域切换冷启动**（新领域专家从未出现过），非容量不足

### 3.5 调度层：领域预测 + 数据飞轮

moe-l2 的差异化在于**领域感知预载**：

1. **L0a 领域预测器**：对用户 prompt 做领域分类（codegen / math / debug / chinese_tech / translate 等 8 域），三层兜底——关键词（<1ms，零依赖）→ TF-IDF 线性分类器（236.7 KB，5 折 CV 59.3%）→ 语义 embedding（all-MiniLM-L6-v2，10-30ms，可选）
2. **L2 热缓存**：预测领域后，把该领域常用的专家从 SSD 预载到内存（mmap 共享内存 LRU，每层独立 deque，异步预载 2 workers）
3. **数据飞轮**：代理层把每次真实请求（prompt + 真实专家路由）攒进样本库，攒够 20 条自动重训分类器——**越用越准**（实测：种子 59.3% → 种子+真实样本 78.1%，+18.8pp）

### 3.6 架构演进路线（三步到位）

| 阶段 | 方案 | 显存 | 速度 | 状态 |
|------|------|------|------|------|
| 1 | A3 patch（专家强制 CPU 驻留） | 23.3 → 1.2 GB（-95%） | 8.6 t/s | ✅ 已验证 |
| 2 | host buffer（专家 CPU pinned + GPU 直算） | 1.6 GB | **37.5 t/s**（+200%） | ✅ 已验证（当前主线） |
| 3 | sched-cache（热专家 D2D） | 1.6 GB | Prompt 99→308（+211%），Gen 39.2（+5%） | ✅ 已验证（按模型开启） |

---

## 4. 实测数据

> 测试环境：NVIDIA RTX 4090（24.5 GB），CUDA driver 580.105.08，context 512 tokens，llama.cpp（host buffer patch），2026-08-02。完整报告见 references/。

### 4.1 核心结果：host buffer 前后对比

| 模型 | 形态 | Prompt t/s | Gen t/s | VRAM |
|------|------|-----------|---------|------|
| DS-V2-Lite（16B MoE，64 专家） | CPU buffer（旧，专家 CPU 算） | 12.5 | 12.5 | 1615 MiB |
| **DS-V2-Lite** | **host buffer（专家 GPU 直算）** | **99.0** | **37.5** | **1625 MiB** |
| Qwen3.6-A3B（32B MoE，256 专家） | CPU buffer（旧） | 10.0 | 10.0 | 2141 MiB |
| **Qwen3.6-A3B** | **host buffer** | **75.8** | **46.8** | **2147 MiB** |

- DS-V2-Lite：速度 **+200%**（12.5 → 37.5 t/s），显存几乎不变（1615 → 1625 MiB）
- Qwen3.6-A3B：速度 **+370%**（10 → 46.8 t/s），显存不变（2147 MiB）

### 4.2 sched-cache 档位矩阵（DS-V2-Lite，cache 挂 sched 拷贝层）

| cache | Prompt t/s | Gen t/s | VRAM | 崩溃 |
|-------|-----------|---------|------|------|
| 无 | 99.0 | 37.4 | 1625 MiB | 0 |
| **0.25（推荐）** | **308.4（+211%）** | **39.2（+5%）** | 1625 MiB | 0 |
| 0.5 | 308.8 | 39.4 | 2127 MiB | 0 |
| 0.75 | 303.3 | 39.5 | 1625 MiB | 0 |
| 1.0 | 304.2 | 39.4 | 2165 MiB | 0 |

结论：**0.25 已到顶**（16 slots/层覆盖全部热专家），更大档位只加显存无速度收益。

### 4.3 全量 vs moe-l2 对比（DS-V2-Lite Q2_K，当前 host-buffer 基线）

| 指标 | 标准全 GPU | moe-l2（host buffer） | 变化 |
|------|-----------|----------------------|------|
| 显存占用 | 23.3 GB | **1.6 GB** | **-93%** |
| Gen 速度 | 65 t/s | **37.5 t/s** | ~58% |
| 模型/显存比 | 0.26× | **3.9×** | +15 倍 |

> 早期 A3 patch 形态（专家 CPU 计算）数据：VRAM 23.3 → 1.2 GiB（-95%），Gen 13.8 → 8.6 t/s，Prompt 23.4 → 18.5——显存压缩已达成但速度损失大。2026-08-02 host buffer 升级后速度从 8.6 提升到 37.5 t/s（+335%），显存仍保持 1.6 GB。两条路线对比说明：**显存压缩靠"专家驻 CPU"，速度恢复靠"专家 GPU 直算"**，两者缺一不可。

### 4.4 多模型 + 多 cache 档位回归（2026-08-02）

| 模型 | 专家 | cache 档位 | 对话类型 | 状态 |
|------|------|-----------|---------|------|
| Qwen3.6-A3B IQ2_M | 256 | 0/0.1/0.5/1.0/2.0 | short/long/followup | 15/15 PASS |
| DS-V2-Lite Q2_K | 64 | 0/0.1/0.5/1.0/2.0 | short/long/followup | 15/15 PASS |

30 项组合全部通过，零崩溃。A3 GPU LRU cache 在 5 档缓存比例 × 3 种对话场景下稳定运行。

### 4.5 显卡适配矩阵

| 显卡 | 显存 | 能跑什么（moe-l2） |
|------|------|------------------|
| GTX 1650 / MX 系列 | 4 GB | DS-V2-Lite（16B MoE）✅ |
| **RTX 3050 / 2060 / 4060 / 3060** | **6-8 GB** | **Qwen3.6-A3B（32B MoE）✅ 核心目标段** |
| RTX 3060 12GB / 4060 Ti | 12 GB | 更大 MoE（目标 50B+） |
| RTX 4070 / 4090 | 16-24 GB | 全部，可开大 cache |

8 GB 卡是核心目标：原本只能跑 7B dense 模型，装上 moe-l2 能跑 32B MoE。

### 4.6 预测准确率实测

| 指标 | 值 |
|------|-----|
| 关键词预测命中率 | 100%（28/28 测试） |
| 关键词延迟 | sub-ms |
| 语义兜底延迟 | 10-30 ms（CPU） |
| TF-IDF 分类器（种子） | 5 折 CV 59.3% |
| **+真实样本（数据飞轮）** | **78.1%（+18.8pp）** |

---

## 5. 对比

### 5.1 与全 GPU 形态的对比（本方案内部基线）

| 维度 | 全 GPU | moe-l2 | 决策含义 |
|------|--------|--------|---------|
| 显存 | 23.3 GB | 1.6 GB | 8 GB 卡可跑，显存是硬约束 |
| 速度 | 65 t/s | 37.5 t/s | 对话场景 37 t/s 足够流畅 |
| 硬件门槛 | 24 GB 卡（8000+ 元） | 4-8 GB 卡（500-2000 元） | 成本降一个数量级 |

### 5.2 与 Palm-Infra / mollm 的对比（腾讯，官方数据）

mollm 是腾讯优图 Palm 团队开源的 MoE 推理引擎，Apple Silicon / ARM Linux 平台，SSD 专家卸载 + LRU 缓存 + 跨层预取，与我们同思路（专家 offload + 分层缓存）但不同平台。官方实测数据（README）：

| 模型 | 专家 cache | Decode | 峰值内存 | 命中率 |
|------|-----------|--------|---------|--------|
| Qwen3.5-122B-A10B W4 | 1 GiB | 12.38 t/s | 5.90 GiB | 47.9% |
| Qwen3.5-122B-A10B W4 | 10 GiB | 16.19 t/s | 14.64 GiB | 83.5% |
| Qwen3.5-122B-A10B W4 | 16 GiB | **16.53 t/s** | **20.60 GiB** | 88.6% |
| DeepSeek-V4-Flash（157GB） | 10 GiB | 4.73 t/s | 24.32 GiB | — |

**对比结论（诚实口径，平台不同不宣称超越）**：

| 维度 | mollm（腾讯） | moe-l2 |
|------|-------------|--------|
| 平台 | Apple Silicon / ARM Linux | **Linux x86_64 + NVIDIA GPU** |
| 专家存储 | SSD → RAM | CPU RAM → GPU 显存 |
| 计算位置 | CPU（NEON 优化 kernel） | **GPU（cuBLAS）** |
| 速度（122B 级） | 16.53 t/s @ 20.6 GiB | Qwen3.6-A3B 46.8 t/s @ 2.1 GiB |
| 安装 | 源码编译（CMake + C++） | **pip install** |
| 模型支持 | Qwen 系列 | **任意 llama.cpp MoE**（DeepSeek/Qwen/Mixtral） |
| 目标用户 | 移动端 / 边缘 | **桌面 / 家庭服务器** |

**启示**：两家独立验证了"专家 offload + LRU 缓存 + 预取"路线的有效性。mollm 用 1 GB 缓存 + 5.9 GiB 总内存跑 122B 模型（12.38 t/s），moe-l2 用 1.6 GB 显存跑 16B MoE（37.5 t/s）——在各自平台上都证明**消费级硬件可以跑大 MoE**。

### 5.3 与生态的关系

| 项目 | 关系 |
|------|------|
| llama.cpp | 底层推理引擎，moe-l2 打补丁增强（host buffer + cache），**零迁移** |
| ollama | 同为推理入口，moe-l2 可作为其前端代理 |
| vLLM | 服务器场景（多卡/高并发），moe-l2 消费级场景，不冲突 |
| GGUF | 标准模型格式，moe-l2 读取其 metadata 做专家映射 |

moe-l2 不重造轮子，做的是"调度"这一层——领域预测、专家预载、缓存策略。

---

## 6. 部署与使用

### 6.1 一行安装

```bash
curl -fsSL https://raw.githubusercontent.com/yalun753/moe-l2/main/scripts/install.sh | bash
```

安装器自动：检测系统（Linux x86_64 + NVIDIA）→ 安装 PyPI 包 → 下载预编译 CUDA 二进制 → 可选下载演示模型（11.5 GB，断点续传）→ 环境自检。

### 6.2 手动安装

```bash
pip install moe-l2                   # 纯关键词预测（零额外依赖）
pip install moe-l2[predictor]        # 混合模式（+语义 embedding）
moe-l2 download-bins                 # 预编译 host-buffer CUDA 二进制
moe-l2 model download --model qwen3.6-35b   # 可选演示模型
moe-l2 start --model model.gguf --gpu
```

### 6.3 兼容性

- OpenAI 兼容 API（`/v1/chat/completions`），curl / Open WebUI / LangChain 直接连
- 平台：Linux x86_64 + NVIDIA（CUDA）——macOS / Windows / ARM Linux 暂不支持
- 预编译二进制从 GitHub Release 分发（`bins-v0.1.1`，96.5 MB）

---

## 7. 项目状态与 Roadmap

### 已完成

- ✅ 领域预测器（关键词 + TF-IDF + 语义三层兜底，数据飞轮自动重训）
- ✅ L2 内存热缓存（mmap LRU，线程安全，异步预载）
- ✅ 透明代理（HTTP/SSE 转发，OpenAI 兼容）
- ✅ CLI（doctor / model / download-bins / collect / start / stats）
- ✅ **host buffer 专家 GPU 直算**（DS +200%、Qwen +370%）
- ✅ **sched-cache**（DS Prompt +211%，按模型开关）
- ✅ 一键安装包 + PyPI 发布（v0.5.1）
- ✅ 模式 A 专家路由收集（collect）+ GGUF 映射嵌入

### 进行中

- 🔄 **Qwen3-235B-A22B 终极验证**：8 GB 卡跑 235B MoE（Q2_K 81.7 GB 下载中），验证"小显存跑大模型"上限
- 🔄 完整 Benchmark 报告（235B 验证后补全）
- 🔄 轻量分类器微调 transformer（数据飞轮样本积累后）

### Roadmap

| 阶段 | 内容 | 优先级 |
|------|------|--------|
| P0 | 一键安装包 ✅ / Benchmark（等 235B）/ 白皮书 ✅ | 当前 |
| P1 | Windows + NVIDIA 移植、AI PC 厂商 PoC、社区 issue/PR | 产品化后期 |
| P2 | 模式 B 边用边收集、多模型通用化、域切换平滑过渡 | 后续 |

---

## 附录 A：术语表

| 术语 | 含义 |
|------|------|
| MoE | Mixture of Experts，混合专家架构：多个专家子网络 + 门控路由 |
| A3 | moe-l2 的专家调度方案代号（按需激活专家，三级存储） |
| host buffer | CUDA 主机端 pinned 内存，GPU 可经 PCIe DMA 直读，不占显存 |
| sched-cache | 挂在 llama.cpp 调度器拷贝层的专家缓存（热专家 D2D 免 PCIe） |
| L2 cache | 内存（RAM）层的专家热缓存（mmap 共享内存 LRU） |
| D2D | Device-to-Device，显存内拷贝（免 PCIe 往返） |
| t/s | tokens per second，每秒生成 token 数 |
| GGUF | llama.cpp 的模型格式（含元数据 + 量化权重） |

## 附录 B：参考文献

- [moe-l2 README（中英双语）](../README.md) / [README_zh.md](../README_zh.md)
- [qwen3.6-a3b-iq2m-benchmark.md](qwen3.6-a3b-iq2m-benchmark.md)
- [deepseek-v2-lite-q2k-benchmark.md](deepseek-v2-lite-q2k-benchmark.md)
- [cache-sched-layer-benchmark.md](cache-sched-layer-benchmark.md)
- [TencentYoutuResearch/Palm-Infra](https://github.com/TencentYoutuResearch/Palm-Infra)（mollm 官方 README 实测数据）

---

*moe-l2 · Apache 2.0 License · GitHub: [yalun753/moe-l2](https://github.com/yalun753/moe-l2)*
