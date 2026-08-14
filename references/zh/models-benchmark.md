# moe-l2 支持的模型实测汇总（2026-08-11 更新）

> 全部为 moe-l2 完整链路（proxy + L2 cache + host-buffer GPU 直算 + selective pin/on-demand pin）实测数据，可复现。
> **2026-08-10 主路径升级为 selective pin**（路由表驱动 top-K pin + GPU cache 预填充，bins-v0.4.0）；三卡复测（2080 Ti / 4090 / 5090）数据见各模型报告。
> 测试口径：每请求 64-128 token、n_predict=128、c=2048-8192、`GGML_OP_OFFLOAD_MIN_BATCH=1`、`GGML_CUDA_EXPERT_CACHE=1`。
> 各模型详细报告链接见文末。

## 总表（按模型规模）

| 模型 | 参数 | 文件 | 量化 | 专家 (激活) | GPU 显存 | Host RSS | 速度 | 在哪验证 |
|------|------|------|------|------------|----------|----------|------|----------|
| DeepSeek-V2-Lite | 16B MoE | 6 GB | Q2_K | 64 (top-6) | **~10 GB** | **~6.7 GB** | **133.2 t/s** (4090，修复版) / 85.25 (2080 Ti，修复版) / ⚠️ 135.57 (5090，P0 假速度，待重测) | RTX 4090 / 2080 Ti / 3080 Ti / 5090 |
| Qwen3.6-35B-A3B | 32B MoE | 11 GB | UD-IQ2_M | 256 (top-8) | **~9.3 GB** | **~11.4 GB** | **44-48 t/s** (4090，修复版) / 30.87 (2080 Ti，有锁版) / ⚠️ 76.41 (5090，P0 假速度，待重测) | RTX 4090 / 2080 Ti / 3080 Ti / 5090 |
| DeepSeek-V4-Flash | **157B MoE** | **85 GB** (3 分片) | UD-IQ2_M | 256 (top-6) | **16.5-16.7 GB** | **17.5-26.8 GB**（on-demand/selective pin） | ⚠️ 35.96 t/s（UD-IQ2_M 2bit 量化退化乱码，无有效速度数据，等 Q4 量化版） | RTX 4090 |
| Mixtral-8x7B | 47B MoE | ~16 GB | Q4_K_M | 8 (top-2) | 2.2-2.9 GB | — | 3.7 t/s* | RTX 4090（cache 测试口径） |
| Qwen3-235B-A22B | 235B MoE | 85.7 GB | Q2_K | 128 (top-8) | **13.9 GB** | **54.7 GB**（selective pin） | **~3.9 t/s**（稳态，4090） | RTX 4090 |

\* Mixtral 为裸 llama-server cache 收益测试口径（专家 CPU 计算，非 host-buffer GPU 直算主路径），仅作参考。
\*\* 4090/2080 Ti 数据为 **2026-08-14 bins-v0.4.1 修复版全链路实测**（selective pin，`moe-l2 start --gpu`）；5090 为旧版 P0 假速度待重测；3080 Ti 仍为 v3.1 多架构口径（bins-v0.3.0）。

## 显存节省（host-buffer 专家 GPU 直算 + selective/on-demand pin）

| 模型 | 标准全量加载 | moe-l2 | 节省 | 速度保留 |
|------|-------------|--------|------|----------|
| DeepSeek-V2-Lite | 23.3 GB VRAM, 65 t/s | **~10 GB, 133.2 t/s**（4090，2026-08-14 修复版） | **57%** | 205% |
| Qwen3.6-35B-A3B | 8GB 卡 OOM | **~9.3 GB, 44-48 t/s**（4090，2026-08-14 修复版） | — | ≈ pre-lazy 46.5 |
| DeepSeek-V4-Flash | 10-11GB 卡 OOM | **16.5 GB VRAM / 17.5 GB RSS**（4090；⚠️ 35.96 t/s 为 UD-IQ2_M 2bit 量化退化乱码，无有效速度数据，等 Q4 量化版） | — | on-demand/selective pin 均可 |

## 全链路实测（v3.1 固定专家数淘汰，RSS 封顶）

**Qwen3.6-A3B / DS-V2-Lite（RTX 3080 10GB 同卡，裸 vs 全链路）**

| 模型 | 裸 server | v3.1 全链路 | 淘汰影响 | RSS | VRAM |
|------|-----------|------------|----------|-----|------|
| Qwen3.6-35B-A3B | 8.77 t/s | 9.40 t/s | **+7%** | 4.5 GB | 2.3 GB |
| DS-V2-Lite | 8.66 t/s | 10.21 t/s | **+18%** | 5.6 GB | 3.3 GB |

## 多架构实测（三卡 × 双模型，bins-v0.2.0+，CUDA 12.8）

| GPU | 架构 | DS-V2-Lite | Qwen3.6-A3B | VRAM |
|-----|------|-----------|-------------|------|
| RTX 2080 Ti | sm_75 | 85.25 t/s | 30.87 t/s（有锁版） | 待补测 |
| RTX 3080 Ti | sm_86 | 12.25 t/s | 13.28 t/s | 1.1-2.2 GB |
| RTX 5090 | sm_120a | ⚠️ 135.57 t/s（P0 假速度，待重测） | ⚠️ 76.41 t/s（P0 假速度，待重测） | 1.3-2.5 GB |
| RTX 4090 | sm_89 | 133.2 t/s | 44-48 t/s | ~9.3-10 GB |

\* 4090/2080 Ti 为 2026-08-14 bins-v0.4.1 修复版全链路实测（selective pin）；5090 旧数据为 P0 假速度待重测；3080 Ti 为 v3.1 多架构包（bins-v0.3.0）实测。

## 关键结论

1. **显存实测更新（08-14 修复版）**：4090 上 DS ~10 GB、Qwen ~9.3 GB（真实占用，非 P0 假数据）；2080 Ti 显存待补测——「8GB 卡余量充足」结论需重新评估
2. **v3.1 淘汰不掉速**：同卡对比全链路反而更快（Qwen +7%、DS +18%，L2 热专家预加载收益 > 淘汰开销）
3. **on-demand pin 主路径（08-07）**：Qwen 50.2 / DS 37.9 / V4 10.1 t/s（4090）——V4 从 1.7-2.0 提升 5 倍；Qwen/DS 达到并超过 pre-lazy host-buffer 记录
4. **selective pin + GPU 预填充（08-10，v0.4.0）**：路由表驱动 top-K pin——V4 RSS **84.4 → 26.8GB**（⚠️ 34.67 / 35.96 t/s 为 UD-IQ2_M 2bit 量化退化乱码，无有效速度数据，等 Q4 量化版；此前 10.1 t/s 为官方原版二进制）；GPU cache 预填充冷启动 +84%（10.7 → 19.7 t/s）
5. **V4 在 2080 Ti 上 0.7-2.2 t/s = 卡算力极限**（IQ2_M 157B，非 offload 代价）；4090 上 35.96 t/s ⚠️ 为 UD-IQ2_M 2bit 量化退化乱码数据，无有效速度，等 Q4 量化版
6. **动态 pin 集合（08-09，低内存模式）**：只注册激活专家 + LRU 淘汰冷专家——V4 RSS **84GB → 17-24GB**（`MOE_L2_LRU_MAX_EXPERTS` 2000≈17GB / 12000≈24GB），速度 4-5 t/s（V4 路由极分散，30 轮会话触及 ~29GB 不同专家，新专家首次触碰付缺页读盘 ~2ms）；Qwen/DS 工作集小不受影响。`MOE_L2_PIN_LAYERS=0-2,14-20,36-37` 永久 pin 通用/稀疏层（~5.4GB 免费）。权衡：whole-pin 最快（30.9 t/s，⚠️ UD-IQ2_M 2bit 量化退化乱码，无有效速度数据，等 Q4 量化版）但 82GB 内存；动态 pin 内存可控但 V4 掉速一半
7. **并发共享 cache（08-12）**：4 路并发（`--parallel 4`）共享 A3 cache / selective pin——Qwen（2080 Ti）总吞吐 95.02（同领域）/ 88.28（跨领域）t/s，DS 198.59 / 188.25，V4（4090）89.66 / 88.10——**全部 = 单会话 2.3-2.5 倍，跨领域仅 -5-7%，无需分池**；显存只增 KV slot（+2.9GB/4 路），内存 +0.2GB 无泄漏

## 并发能力（2026-08-12，双机三模型）

| 模型（GPU） | 单会话 t/s | 4 路同领域总吞吐 | 4 路跨领域总吞吐 | vs 单会话 | 跨/同 |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B（2080 Ti） | 38.4 | **95.02** | **88.28** | 2.3-2.5× | -7% |
| DS-V2-Lite（2080 Ti） | 78.3 | **198.59** | **188.25** | 2.4-2.5× | -5% |
| DeepSeek-V4-Flash（4090，256 专家） | 35.4-35.8 | **89.66** | **88.10** | 2.5× | -2% |
| Qwen3.6-35B-A3B（4090） | 67.4-68.7 | **174.56** | **144.47** | 2.6× / 2.1× | -17% |
| DS-V2-Lite（4090） | 143.4-143.7 | **369.48** | **354.73** | 2.6× / 2.5× | -4% |

- 结论：**并发不崩、共享天然成立**——4 路并发（哪怕跨领域）激活专家集合高度重叠 → A3 cache 无竞争；总吞吐 = 单会话 2.3-2.5 倍，「一台 AI PC 多人同时用」实测成立
- 显存：4 路并发 +2.9GB（全是独立 KV slot，预期成本非泄漏）；RSS +0.2GB（selective pin 表外专家页不累积）
- ⚠️ 4090 首次跨领域并发偶发 cublas 崩溃（重启即恢复，非稳定复现，已记录）
- ⚠️ 本节为 2026-08-12 并发测试（P0 bug 修复前，bins-v0.4.0）实测：单会话速度与总吞吐受 P0 假速度影响（虚高），需修复版（bins-v0.4.1）重测；「2.3-2.5×」「跨/同」等相对比值仅供参考
- 完整报告：[concurrent-cache-sharing-20260812.md](concurrent-cache-sharing-20260812.md)

## 详细报告

- [DeepSeek-V4-Flash 验证报告（157B，双卡全链路）](deepseek-v4-flash-verify-20260805.md)
- [Qwen3-235B-A22B Q2_K 基准（235B，4090 三轮实测）](qwen3-235b-a22b-q2k-benchmark.md)
- [多架构三卡验证（2080 Ti / 3080 Ti / 5090）](multi-arch-three-gpu-benchmark.md)
- [Qwen3.6-A3B IQ2_M 基准](qwen3.6-a3b-iq2m-benchmark.md)
- [DeepSeek-V2-Lite Q2_K 基准](deepseek-v2-lite-q2k-benchmark.md)
- [Cache / sched 层基准（DS / Qwen / Mixtral 三模型收益矩阵）](cache-sched-layer-benchmark.md)
- [Host-buffer 设计决策（中英）](design-decisions.md)
