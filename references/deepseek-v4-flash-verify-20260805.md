# DeepSeek V4 Flash 验证报告（2026-08-05）

> 状态：moe-l2 完整链路跑通 DeepSeek V4 Flash ✅（2080 Ti + RTX 3080 双卡实测）
> 关联：`multi-arch-three-gpu-benchmark.md`（V4 全链路复测节）、PyPI 0.7.0 / bins-v0.3.0（v3.1 专家页淘汰）

---

## 一句话结论

**moe-l2 在 2080 Ti（11G 显存）上成功加载并推理 DeepSeek V4 Flash（UD-IQ2_M，85GB 三片）**——显存 8.3GB、速度 ~2 t/s、RSS 10→29GB 后趋稳；同套调度器在 RTX 3080（10G 显存）上全链路 2.11-2.22 t/s。跑通过程修复了**多分片 GGUF 解析 bug**，并完成了**专家页淘汰 v3.1**（固定专家数 LRU，RSS 封顶）。

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

## 裸 llama-server 实测（2080 Ti，无 proxy）

| 指标 | 值 |
|---|---|
| 加载后 RSS | 10.4 GB |
| 加载后显存 | 8.2 GB |
| 首请求速度（128 token） | 0.92 t/s（冷盘） |
| 热身后速度（64 token） | 2.0-2.06 t/s |
| 首请求后 RSS | 28.9 GB（+18.5GB 专家页驻留） |
| 显存（推理后） | 8.4 GB 稳定，无泄漏 |

速度观察：首请求 0.92 → 热身后 2.0 t/s（专家页进 RAM 后读盘开销消失）——印证 L2 热缓存价值方向。

## moe-l2 完整链路（proxy + L2 cache + v3.1 淘汰）

### 2080 Ti（SM75）

| 场景 | 速度 | RSS |
|---|---|---|
| 短对话 | 0.89 t/s | 12.4GB |
| 追问 r1-r4 | 1.02-1.07 t/s | 11.2-12.2GB |
| 长对话（~1500 token prefill） | 0.72 t/s | 11.9GB |

- 显存 8.4GB / 11GB（余量 2.7GB）
- RSS 11-12GB 封顶（v3.1 淘汰生效，对比无淘汰 29GB）
- L2 cache：569 slots/layer × 24 层，expert 7MB/个

### RTX 3080（SM86，v3.1 多架构产物）

| 场景 | 3080 v3.1 | 2080 Ti 原测 | 提升 |
|---|---|---|---|
| 短对话 | **2.11 t/s** | 0.89 | +137% |
| 追问 r1 | 2.22 | 1.02 | +118% |
| 追问 r2 | 2.22 | 1.03 | +116% |
| 追问 r3 | 2.22 | 1.05 | +111% |
| 追问 r4 | 2.18 | 1.07 | +104% |
| 长对话 | **1.78 t/s** | 0.72 | +147% |

- server RSS 18.9GB（v3.1 淘汰生效；裸 server 无淘汰 42GB）
- proxy (L2) RSS 8.2GB
- **VRAM 9.07GB / 10GB**（余量 1GB，较紧；长对话留意 KV 增长）

## 专家页淘汰 v3.1（固定专家数 LRU）

- 机制：`MOE_L2_LRU_MAX_EXPERTS=N`——最多驻留 N 个最热专家，只淘汰最冷溢出（不读 RSS、不全清）
- 效果：Qwen 实测几乎零掉速（-2% vs v2 -24% / v3 -45%）；V4 双卡 RSS 封顶（2080 Ti 11-12GB、3080 18.9GB）
- 关键坑：_exps 专家轴在 ne[2] 非 ne[0]（用 model.hparams.n_expert 定位）；淘汰实现细节见 `expert-page-eviction-cpp-20260805.md`（skill 侧）与历史记录文档

## 速度定位

- 0.72-2.22 t/s = **卡算力极限**（V4 激活 ~10B 参数，2080 Ti 算力天花板 ~2 t/s），非淘汰机制开销
- 同卡裸跑对照：3080 裸 server Qwen 8.77 / DS 8.66 t/s，全链路反而 +7%/+18%（L2 热专家预加载收益 > 淘汰开销）
- 换 4090 预期 5-8 t/s（待 deepseek4 CUDA bug 修复后重测）

## 结论

1. **85GB 级 MoE 大模型可在消费级显卡（10-11G）跑通**——显存 8.3-9.1GB，分层调度 + 专家页淘汰生效
2. **速度受限于卡算力**，不是调度器开销；调度器在 2080 Ti 上不拖后腿（与裸跑一致）
3. **多分片修复对 50B+ 大模型普适**（未来 100G+ 分片只会更多）

## 环境坑（复现用）

- AutoDL 开机驱动 mismatch（内核 vs 库版本不一致）：修符号链接 `ln -sfn libcuda.so.580.<内核版> libcuda.so.1` 等
- 架构核对用 CUDA 12.8 `cuobjdump --list-elf`（系统旧版误报）
- proxy 非流式转发 httpx 超时 30s→600s（慢速模型必踩）

---

## 2026-08-07 更新：on-demand pin 主路径（4090 上 5 倍提速）

> **on-demand pin**（lazy mmap + whole-tensor 合并注册 + A3 cache 2048 槽）在 RTX 4090 上把 V4 从 **1.7-2.0 t/s 拉到 10.1 t/s（5 倍）**。90GB 模型在 24GB 卡 + 1TB 内存机器上可交互使用。

### 实测（RTX 4090 24GB，2026-08-07，裸 server 口径）

| 形态 | Gen t/s | VRAM | RSS |
|------|---------|------|-----|
| lazy 无 pin（裸 server，08-05 测） | 1.71-1.98 | 9.1 GB | — |
| on-demand pin（whole） | 9.5 | 8.4 GB | 80.9 GB |
| **on-demand pin + cache 2048** | **10.1** | 17.4 GB | 82 GB |
| cache 512 槽 | 9.29 | ~9GB | —（每 token 激活专家 >512，命中率≈0） |
| cache 4096 槽 | OOM | — | —（17.6GB cache + 基础 8.4GB > 24GB） |

### 关键结论（2026-08-07）

1. **V4 4090 上 10.1 t/s（原 1.7-2.0 的 5 倍）**；GPU util 13% → 86%，**已近计算 bound**（再提速需优化 kernel/量化，非 cache）
2. **2048 槽是 cache 平衡点**（512 无提升、4096 OOM），三模型通用增益
3. **RSS 80.9GB（whole-pin 全量 fault）**——1TB 内存机器无压力；128GB 容器需淘汰机制（v3.1 + unregister）控制驻留
4. 2080 Ti / 3080 上仍为 v3.1 全链路 0.89-2.22（卡算力极限），待 v0.3.1 多架构包复测
5. 详细排错链：`/opt/data/moe-l2/历史记录文档/on-demand-pin-方案-交接-20260807.md`
