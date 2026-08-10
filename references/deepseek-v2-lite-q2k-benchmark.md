# DeepSeek-V2-Lite-Chat-Uncensored Q2_K benchmark 报告（更新至 2026-08-10）

## 最新成果（2026-08-10）：2080 Ti 全链路复测（bins-v0.4.0 / selective pin）

> 在 RTX 2080 Ti（region-42 云机）上用 bins-v0.4.0 多架构二进制 + 全链路（`moe-l2 start --gpu`，selective pin 路由表 top-100）复测：**DS Gen 87.25 t/s（追问场景），短对话 81.72 t/s**。相比官方原版 llama.cpp 二进制（~7-10 t/s 级别）+700~1000%，进一步确认 moe-l2 优化版在 2080 Ti 上的真实性能。

### 实测（RTX 2080 Ti，2026-08-10，全链路 selective pin）

| 轮次 | 短对话 | 追问1 | 追问2 |
|------|--------|-------|-------|
| Round 1（冷启动） | 39.62 | 81.02 | 83.35 |
| Round 2 | 81.67 | 87.30 | 85.30 |
| Round 3（稳定） | **81.72** | **87.25** | **86.40** |
| Round 4 | 83.53 | 87.31 | 85.12 |

- 口径：`python3 speed_test.py 11435`（64 tok/请求，proxy 全链路含路由表 + selective pin）
- 稳定值：短对话 ~82-84、追问 ~85-87 t/s；报告取 **87.25 t/s**（round3 追问1）

### 关键结论（2026-08-10）

1. **2080 Ti 上 DS 全链路 87.25 t/s**（vs 原版 llama.cpp 6.89 t/s ≈ +1166%）；4090 上为 37.9 t/s——2080 Ti 反而更快（DS 模型小、专家拷贝开销低，2080 Ti PCIe 3.0 瓶颈不明显）
2. **selective pin 零拖累**：路由表 top-100 pin 后 2080 Ti 速度稳定
3. **冷启动明显**：round1 短对话 39.62 → round2 81.67（首次加载路由表 + 专家 pin + GPU cache 预热），round2 起即达稳定

---

## 上一版成果（2026-08-07）：on-demand pin 主路径

> **on-demand pin**（lazy mmap + whole-tensor 合并注册 + A3 cache 2048 槽）取代 host buffer。**DS Gen 37.5 → 37.9 t/s（+4%，超过 37.5 目标）。**

### 实测（RTX 4090，2026-08-07）

| 形态 | Gen t/s（短） | Gen t/s（长） | VRAM |
|------|-------------|-------------|------|
| host buffer + cache 0.25（08-02） | 39.2 | — | 1625 MiB |
| on-demand pin（whole） | 36.4 | — | ~2GB |
| **on-demand pin + cache 2048** | **37.9** | **37.2** | 2.0GB |
| 多架构包（CUDA 12.8，sm_61-120a） | **39.0** | — | — |

### 关键结论（2026-08-07）

1. **DS 37.9 t/s 达标**（目标 37.5），多架构包 39.0 t/s（CUDA 12.8 编译器红利）
2. **2048 槽 cache 三模型通用增益**（Qwen +7~11% / DS +4% / V4 +6%）
3. **推荐配置**：`GGML_OP_OFFLOAD_MIN_BATCH=1` + `GGML_CUDA_EXPERT_CACHE=1`（cli.py 已内置）
4. 详细排错链与数据：`/opt/data/moe-l2/历史记录文档/on-demand-pin-方案-交接-20260807.md`

---

## 基本信息

| 项目 | 值 |
|------|-----|
| 模型 | DeepSeek-V2-Lite-Chat-Uncensored |
| 架构 | MoE (2.37B active, 16B total) |
| 量化 | Q2_K (2-bit) |
| 推理引擎 | llama.cpp (A3 patch + host buffer, CUDA) |
| GPU | NVIDIA RTX 4090, 24.5 GB VRAM |
| 测试日期 | 2026-07-29（初版）/ 2026-08-02（架构升级） |
| context 长度 | 512 tokens |

---

## 2026-08-02 更新：host buffer 专家 GPU 直算 + sched-cache（重大突破）

> 2026-08-02 完成架构升级——**host buffer（专家 CPU pinned 不占 VRAM）+ GGML_OP_OFFLOAD_MIN_BATCH=1 + cache 挂 sched 拷贝层**，专家走 GPU 直算，速度大幅提升。本报告旧数据（--cpu-moe 专家 CPU 计算形态）已废弃。

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
4. **显卡适配**：VRAM 仅 1625 MiB，**4 GB 卡即可流畅运行**（旧结论"4 GB 入门卡勉强可跑"已过时——现在 4 GB 卡随便跑）
5. **三模型速度排序（host buffer 后）**：Qwen 46.5 > DS 39.2 > Mixtral 3.7 t/s

### 详细验证数据

- **三模型 cache 档位矩阵**：见 `cache-sched-layer-benchmark.md`
- **host buffer 架构细节**：llama-model-loader.cpp 放开 mmap→host buffer 回退 + cli.py `GGML_OP_OFFLOAD_MIN_BATCH=1`

---