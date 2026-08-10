# moe-l2 支持的模型实测汇总（2026-08-07 更新）

> 全部为 moe-l2 完整链路（proxy + L2 cache + host-buffer GPU 直算 + v3.1 专家页淘汰）实测数据，可复现。
> **2026-08-07 主路径升级为 on-demand pin**（mmap 惰性加载 + 首次触碰合并注册整个专家 tensor + A3 cache 2048 槽），4090 实测显著提速。
> 测试口径：每请求 64-128 token、n_predict=128、c=512-2048、`GGML_OP_OFFLOAD_MIN_BATCH=1`、`GGML_CUDA_EXPERT_CACHE=1`。
> 各模型详细报告链接见文末。

## 总表（按模型规模）

| 模型 | 参数 | 文件 | 量化 | 专家 (激活) | GPU 显存 | Host RSS | 速度 | 在哪验证 |
|------|------|------|------|------------|----------|----------|------|----------|
| DeepSeek-V2-Lite | 16B MoE | 6 GB | Q2_K | 64 (top-6) | **1.6-2.0 GB** | — | **37.9 t/s** (4090) | RTX 4090 / 2080 Ti / 3080 Ti / 5090 |
| Qwen3.6-35B-A3B | 32B MoE | 11 GB | UD-IQ2_M | 256 (top-8) | **2.1-2.9 GB** | — | **50.2 t/s** (4090) | RTX 4090 / 2080 Ti / 3080 Ti / 5090 |
| DeepSeek-V4-Flash | **157B MoE** | **85 GB** (3 分片) | UD-IQ2_M | 256 (top-6) | **8.4-17.4 GB** | **10.4 GB**（selective pin） | **30.9 t/s** (4090) / 0.89-2.22 (2080 Ti/3080) | RTX 4090 / RTX 2080 Ti / RTX 3080 |
| Mixtral-8x7B | 47B MoE | ~16 GB | Q4_K_M | 8 (top-2) | 2.2-2.9 GB | — | 3.7 t/s* | RTX 4090（cache 测试口径） |
| Qwen3-235B-A22B | 235B MoE | 81.7 GB | Q2_K | 256 (top-8) | 验证中 | — | TBD | — |

\* Mixtral 为裸 llama-server cache 收益测试口径（专家 CPU 计算，非 host-buffer GPU 直算主路径），仅作参考。
\*\* 4090 数据为 **on-demand pin 主路径**（whole-pin + A3 cache 2048 槽，2026-08-07 实测）；2080 Ti/3080 仍为 v3.1 全链路口径（多架构包，待 v0.3.1 重编后复测）。

## 显存节省（host-buffer 专家 GPU 直算 + on-demand pin）

| 模型 | 标准全量加载 | moe-l2 | 节省 | 速度保留 |
|------|-------------|--------|------|----------|
| DeepSeek-V2-Lite | 23.3 GB VRAM, 65 t/s | **2.0 GB, 37.9 t/s** | **91%** | 58% |
| Qwen3.6-35B-A3B | 8GB 卡 OOM | **2.9 GB, 50.2 t/s** | — | 超 pre-lazy 46.5 |
| DeepSeek-V4-Flash | 10-11GB 卡 OOM | **10.4 GB RSS, 30.9 t/s**（4090） | — | selective pin 零拖累 |

## 全链路实测（v3.1 固定专家数淘汰，RSS 封顶）

**DeepSeek-V4-Flash（157B / 85GB，2080 Ti + RTX 3080 双卡）**

| 场景 | RTX 3080 10GB | RTX 2080 Ti 11GB |
|------|---------------|------------------|
| 短对话 | 2.11 t/s | 0.89 t/s |
| 追问 r1-r4 | 2.18-2.22 t/s | 1.02-1.07 t/s |
| 长对话 (~1500 tok) | 1.78 t/s | 0.72 t/s |
| Server RSS | 18.9 GB 封顶 | 11-12 GB 封顶 |
| VRAM | 9.07 / 10 GB | 8.3-8.4 / 11 GB |

**Qwen3.6-A3B / DS-V2-Lite（RTX 3080 10GB 同卡，裸 vs 全链路）**

| 模型 | 裸 server | v3.1 全链路 | 淘汰影响 | RSS | VRAM |
|------|-----------|------------|----------|-----|------|
| Qwen3.6-35B-A3B | 8.77 t/s | 9.40 t/s | **+7%** | 4.5 GB | 2.3 GB |
| DS-V2-Lite | 8.66 t/s | 10.21 t/s | **+18%** | 5.6 GB | 3.3 GB |

## 多架构实测（三卡 × 双模型，bins-v0.2.0+，CUDA 12.8）

| GPU | 架构 | DS-V2-Lite | Qwen3.6-A3B | VRAM |
|-----|------|-----------|-------------|------|
| RTX 2080 Ti | sm_75 | 87.25 t/s | 47.24 t/s | 1.0-2.4 GB |
| RTX 3080 Ti | sm_86 | 12.25 t/s | 13.28 t/s | 1.1-2.2 GB |
| RTX 5090 | sm_120a | 135.57 t/s | 76.41 t/s | 1.3-2.5 GB |
| RTX 4090 | sm_89 | 145.63 t/s | 74.99 t/s | 3.1-4.9 GB |

\* 4090 为单架构基线（CUDA 11.8，08-02），非多架构包实测。

## 关键结论

1. **显存与架构无关**：DS 1.6-2.0 GB、Qwen 2.1-2.9 GB，各代卡一致——8GB 卡余量 3-6 倍
2. **v3.1 淘汰不掉速**：同卡对比全链路反而更快（Qwen +7%、DS +18%，L2 热专家预加载收益 > 淘汰开销）
3. **on-demand pin 主路径（08-07）**：Qwen 50.2 / DS 37.9 / V4 10.1 t/s（4090）——V4 从 1.7-2.0 提升 5 倍；Qwen/DS 达到并超过 pre-lazy host-buffer 记录
4. **selective pin + GPU 预填充（08-10，v0.4.0）**：路由表驱动 top-K pin——V4 RSS **84.4 → 10.4GB（↓88%）** 且 **30.9 t/s 零拖累**（此前 10.1 t/s 为官方原版二进制，moe-l2 优化版本来就是 ~30 t/s）；GPU cache 预填充冷启动 +84%（10.7 → 19.7 t/s）
5. **V4 在 2080 Ti / 3080 上 0.7-2.2 t/s = 卡算力极限**（IQ2_M 157B，非 offload 代价）；4090 上 30.9 t/s 已近计算 bound
6. **动态 pin 集合（08-09，低内存模式）**：只注册激活专家 + LRU 淘汰冷专家——V4 RSS **84GB → 17-24GB**（`MOE_L2_LRU_MAX_EXPERTS` 2000≈17GB / 12000≈24GB），速度 4-5 t/s（V4 路由极分散，30 轮会话触及 ~29GB 不同专家，新专家首次触碰付缺页读盘 ~2ms）；Qwen/DS 工作集小不受影响。`MOE_L2_PIN_LAYERS=0-2,14-20,36-37` 永久 pin 通用/稀疏层（~5.4GB 免费）。权衡：whole-pin 最快（30.9 t/s）但 82GB 内存；动态 pin 内存可控但 V4 掉速一半

## 详细报告

- [DeepSeek-V4-Flash 验证报告（157B，双卡全链路）](deepseek-v4-flash-verify-20260805.md)
- [多架构三卡验证（2080 Ti / 3080 Ti / 5090）](multi-arch-three-gpu-benchmark.md)
- [Qwen3.6-A3B IQ2_M 基准](qwen3.6-a3b-iq2m-benchmark.md)
- [DeepSeek-V2-Lite Q2_K 基准](deepseek-v2-lite-q2k-benchmark.md)
- [Cache / sched 层基准（DS / Qwen / Mixtral 三模型收益矩阵）](cache-sched-layer-benchmark.md)
- [Host-buffer 设计决策（中英）](design-decisions.md)
