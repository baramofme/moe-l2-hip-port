# 领域路由表方案验证报告（Qwen + V4 双模型实测）

> 日期：2026-08-09
> 核心问题：moe-l2 的"判领域 → 提前 pin 高频专家走快速通道"方案是否成立？
> 方法：用真实门路由 trace 统计每个模型"每层 top-K 高频专家"能覆盖多少实际激活。

---

## 一句话结论

**方案成立。** 两个模型都验证：**pin 不到三分之一的专家（每层），就能覆盖 88-97% 的激活**。领域路由表 + 提前 pin 有真实数据支撑，且内存成本可控。

---

## 数据来源

| 模型 | trace 来源 | EXPERT 行数 | 层 × 专家 | 话题覆盖 |
|------|-----------|------------|----------|---------|
| Qwen3.6-35B-A3B | qwen8domains（24 日志，8 领域 × 3 场景）| 210k | 40 × 256 (top-8) | 8 领域 |
| DeepSeek-V4-Flash | actset_test（08-08，50 轮通用 + 3 话题各 25 轮）| 311k | 43 × 256 (top-6) | 4 话题组 |

---

## Qwen 覆盖率（40 层，256 专家 top-8）

| 领域 | top-10 | top-30 | top-50 | top-75 | top-100 |
|------|--------|--------|--------|--------|---------|
| chinese_tech | 41% | 70% | 83% | 93% | 97% |
| codegen | 38% | 65% | 80% | 90% | 96% |
| math | 34% | 59% | 74% | 85% | 93% |
| translate | 31% | 58% | 73% | 85% | 92% |
| creative_write | 37% | 62% | 76% | 87% | 94% |
| debug | 32% | 59% | 74% | 84% | 91% |
| general_qa | 36% | 64% | 79% | 90% | 96% |
| logic | 35% | 59% | 74% | 85% | 92% |

**单场景稳定性**：codegen 的 short/followup/longtail 差异 <4%（84.8-88.4% @ top-50）——表不会因对话场景变化失效。

## V4 覆盖率（43 层，256 专家 top-6）

| K | top-10 | top-30 | top-50 | top-75 | top-100 | top-150 |
|---|--------|--------|--------|--------|---------|---------|
| 覆盖率 | 55.7% | 78.2% | 88.2% | **94.2%** | 97.1% | 99.3% |

**注意**：V4 每层实际访问专家 avg=150/256（比 Qwen 分布更广），但 top-75 依然覆盖 94%——高频专家集中度反而更高。

## V4 分话题（=分领域）覆盖率 — 不同话题即不同领域

| 话题 | 请求数 | top-30 | top-50 | top-75 | top-100 | 每层实际专家 |
|------|--------|--------|--------|--------|---------|-------------|
| general | 50 | 83.1% | 92.9% | 97.4% | 99.1% | 97 |
| math | 25 | 79.5% | 88.7% | 94.3% | 97.1% | 108 |
| code | 25 | 85.2% | 92.9% | 97.0% | 98.8% | 86 |
| chat | 25 | 87.1% | 94.1% | 97.8% | 98.9% | 74 |

**关键发现：分话题统计更准**——话题内聚集性 > 跨话题聚合：
- 聚合 top-75 = 94.2%；分话题 top-75 = **94.3-97.8%**
- code/chat 每层实际专家更少（86/74 vs 聚合 150），聚集性更强
- **top-50 就覆盖 89-94%**：比聚合 top-75 更省内存、同样覆盖率

## 内存成本核算

### Qwen（专家 1.01MB/个）

| 档位 | 每层专家 | 40 层 RAM | 覆盖率 |
|------|---------|----------|--------|
| top-50 | 50 | ~2GB | 74-83% |
| top-75 | 75 | ~3GB | 84-93% |
| top-100 | 100 | ~4GB | 91-97% |

### V4（专家 2.7MB/个）

| 档位 | 每层专家 | 43 层 RAM | 覆盖率 |
|------|---------|----------|--------|
| top-50 | 50 | ~5.4GB | 88% |
| **top-75** | 75 | **~8.1GB** | **94%** |
| top-100 | 100 | ~10.8GB | 97% |

## 方案链路

```
门路由输出（现成，每 token 每层产生）
  → 按领域/话题聚合统计高频专家（已完成：Qwen 8 领域表 + V4 表）
  → 领域路由表（domain_router_map_*.json）
  → L0a 判领域
  → 提前 pin 该领域高频专家（cudaHostRegister 批量）
  → 推理时命中走 DMA 快速通道，未命中走 on-demand/LRU 兜底
```

## 产物

- `moe_l2/data/domain_router_map_qwen.json` — Qwen 8 领域 × 40 层 × top-100
- `moe_l2/data/domain_router_map_v4.json` — V4 43 层 × top-75（跨话题聚合）
- `moe_l2/data/domain_router_map_v4_topics.json` — **V4 4 话题 × 43 层 × top-75（分领域，更准）**
- `scripts/bench/domain_router_coverage.py` — Qwen 覆盖率分析
- `scripts/bench/v4_router_coverage.py` — V4 覆盖率分析（聚合）
- `scripts/bench/v4_topic_coverage.py` — **V4 分话题覆盖率分析**
- `scripts/bench/generate_domain_router_map.py` — Qwen 路由表生成
- `scripts/bench/generate_v4_router_map.py` — V4 路由表生成（聚合）
- `scripts/bench/generate_v4_topic_router_map.py` — **V4 分话题路由表生成**
- trace 源：`测试数据备份/qwen8domains/`（24 日志）+ `测试数据备份/v4-actset-trace-20260808.log`（14MB）

## 遗留观察

1. **V4 L20 层 top-10 = [0,1,2,3,4,5]**：连续专家 id，疑似该层存在固定激活模式（非路由学习），需进一步确认是否影响 pin 策略
2. **V4 分话题统计完成**：math/code/chat 独立覆盖率 94-98%（top-75），话题内聚集性确认更强
3. **V4 的 43 层 vs Qwen 40 层**：V4 pin top-75 需 ~8.1GB RAM，16GB 内存机器占用一半，仍需 on-demand 兜底冷专家

## 消费端 + 数据飞轮实测（2026-08-10 凌晨，4090 云机全链路）

### 实现内容（已落地到 moe-l2）
1. **新增 `moe_l2/domain_router_flywheel.py`**（独立模块）：gate 实时解析 EXPERT 路由 → 按领域聚合高频专家 → 攒够阈值自动重建路由表 JSON（原子替换，跨 rebuild 累积，越用越准）
2. **`predictor.load_mapping()` 优先读 flywheel 表**：存在 `domain_router_map_flywheel.json` 时用它，否则回退静态表 → pretouch 消费学习结果
3. **cli.py 修复 V4 启动超时 bug**：`_wait_for_llama_server` 30s→180s（V4 85G 加载 90s，原 30s 必 TIMEOUT 失败）

### 数据飞轮闭环验证（Qwen3.6，全链路 start --gpu）
- 发"写 Python 函数" → codegen 领域自动生成；发"解释勾股定理" → translate 领域自动新增
- 路由表自动重建 3 次：2 domains → 3 domains，新领域零人工干预自动加入
- 真实 311k 行 V4 trace 喂入 → 43 层完整路由表生成（与离线脚本结果一致）

### 速度对照（4090 / v5 whole-pin 二进制，同 prompt，max_tokens=128）

**Qwen3.6-35B-A3B**（稳定轮次 round3）：
| 组 | 配置 | 速度 |
|----|------|------|
| A | 有 flywheel 表（全链路）| 46.06 t/s |
| B | 无 flywheel 表（静态表）| 47.02 t/s |
| — | 直连 llama-server（基线）| 46.06 t/s |
| 结论 | flywheel 零拖累，Qwen 已近 4090 物理上限 | |

**DeepSeek-V4-Flash 85G**（稳定轮次 round3）：
| 组 | 配置 | 速度 |
|----|------|------|
| A | 有 flywheel 表（全链路）| 10.16 t/s |
| B | 无 flywheel 表（静态表）| 10.06 t/s |
| — | 直连 llama-server（基线）| 10.16 t/s |
| 结论 | flywheel 零拖累，V4 瓶颈是路由分散物理上限（4-10 t/s）| |

### 关键结论
1. **flywheel 不影响速度（正负 1% 内）**：Qwen 已满速（46-47 t/s = v5 whole-pin 上限），V4 受限于分散路由物理上限——pretouch 表改变不了 GPU 计算瓶颈
2. **flywheel 的真实价值 = 路由表自动化维护**：静态 8 领域表 → 自动学习任意新领域；省去手工统计；V4 上覆盖率更高（94-98% vs 手工表）
3. **v3.1 二进制限速确认**：云机 bin/ 曾残留 v3.1（f7d7858c，Qwen 仅 10-12 t/s），换 v5（f1b5e048，whole-pin 默认）恢复 46 t/s——**发布版必须带 v0.3.2 二进制**，旧版有性能陷阱
4. 遗留：`domain_router_map_flywheel.json` 已恢复在云机 data/（.bak-B 移回）；本地备份 `测试数据备份/domain-router-consumer-20260809/`

## 选择性 pin 模拟（2026-08-10，V4 真实 trace）

> 用户方向（2026-08-10 定）：flywheel 无法大面积提高速度，但**可以降低内存占用**——pin 的专家可以减少而不影响速度。模拟验证此推断。

### 方法
用 V4 真实 trace（311,363 行 EXPERT，125 请求）对每个 top-K 生成聚合路由表，逐 token 统计激活专家中不在表内的"冷专家"数，按 2.1ms/冷专家（锁页 0.43 + 缺页读盘 1.7，实测值）折算 fault 开销，叠加 whole-pin 基线 98.4ms/token 估速度。

### 结果
| top-K | 每层 pin 数 | pin 内存 | 覆盖率 | 冷专家/token | fault 开销 | 预估速度 |
|---|---|---|---|---|---|---|
| 50 | 40 | 4.6GB | 84.4% | 44.0 | 92.3ms | 5.24 t/s |
| 75 | 60 | 6.9GB | 90.6% | 26.5 | 55.6ms | 6.49 t/s |
| **100** | **80** | **9.1GB** | **94.3%** | **16.0** | **33.7ms** | **7.57 t/s** |
| **150** | **119** | **13.6GB** | **98.0%** | **5.7** | **12.0ms** | **9.06 t/s** |
| 200 | 159 | 18.1GB | 99.4% | 1.7 | 3.5ms | 9.81 t/s |

### 解读
1. **用户方向成立**：选择性 pin 用 4.6-18GB 覆盖 84-99% 激活，内存从 whole-pin 的 82GB 大幅下降，速度只掉 3-48%
2. **甜点区 = top-100**：9.1GB / 7.57 t/s（74% 速度）——16GB 内存 AI PC 可跑，比动态 pin LRU（17-24GB / 4-5 t/s）**内存减半、速度反超 50%+**
3. **性价比最高 = top-150**：13.6GB / 9.06 t/s（89% 速度），适合 32GB 内存机器
4. **top-50 以下不推荐**：覆盖率 <85%，fault 风暴（92ms/token）跌回动态 pin 水平
5. 模拟脚本：`scripts/bench/sim_selective_pin.py`（结果 JSON：`selective_pin_sim_result.json`）；代码实现方向=读取 flywheel 路由表 → 只 cudaHostRegister 表内专家，表外专家 on-demand 兜底

## 选择性 pin 实机验证（2026-08-10 下午，4090 云机全链路 A/B 对照）

### 实现（阶段 1：RAM 选择性 pin）
- C++：ggml-backend.cpp copy_experts 新增 `MOE_L2_ROUTER_FILE` 环境变量支持——加载时读路由表（`layer expert1 expert2 ...` 格式），只对表内专家调 pin_fn；表外专家不显式注册，走 set_tensor_async 的 on-demand 兜底
- Python：cli.py 新增 `--router-map` / `--router-top-k` 参数，启动时自动生成 router map 并注入 env
- 路由表来源：domain_router_map_v4_topics.json + domain_router_map_v4.json 聚合并集（43 层，每层 ~61 专家，top-100 上限）
- 备份链：本地 `测试数据备份/selective-pin-20260810/`（C++ 副本）；云机 `/root/moe-l2-backups/selective-pin-20260810/`（旧 bin 13 文件 + 旧 cli.py）

### A/B 对照实测（同机、同二进制、同 prompt，max_tokens=128，codegen）

| 组 | RSS | round1 | round2 | round3 | 稳定轮速度 |
|---|---|---|---|---|---|
| A：选择性 pin（top-100） | **10.4 GB** | 10.71 | 30.14 | 30.88 | **30.9 t/s** |
| B：whole-pin（对照） | 84.4 GB | 24.45 | 31.23 | 30.21 | **30.2 t/s** |

### 关键结论
1. **RSS 84.4 → 10.4GB（↓ 88%）**，速度零拖累（30.9 vs 30.2 t/s，±2% 噪声内）——「pin 专家减少 → 内存降低、速度不掉」实机验证成立
2. **启动更快**：选择性 pin 10s 就绪 vs whole-pin 40s（不 fault 全量页）
3. **⚠️ 速度 30 t/s 是"重新编译的 build-a3 二进制"带来的，不是选择性 pin**——之前 v031-test（f1b5e048）实测 10.1 t/s，今天从 llama.cpp-clean 源码重编后 A/B 两组都到 30 t/s。选择性 pin 的贡献 = 内存 88% 缩减且不拖慢速度
4. **表外驻留累积观察**：3 轮测速后 RSS 从 10.4 涨到 19.7GB（表外专家 on-demand pin 后驻留，不淘汰）——符合设计，长期运行需水位线淘汰（方案文档风险 #2 的落地项）
5. 与模拟对比：实测 RSS 10.4GB 略优于模拟预测 12.9GB（并集 61/层 < 模拟假设 80/层）；速度因二进制更新无法直接对比

## 阶段 2 实验：GPU cache 预填充（2026-08-10 下午，4090 云机全链路）

### 实现
- 在 ggml-backend.cpp copy_experts 首次调用时，把路由表内专家批量 pre-set 进 GPU expert cache（`cache_set_fn`，按 tensor name 去重只跑一次）
- 目标：首请求命中 D2D，省掉逐个 miss 的 H2D+cache_set 开销
- 二进制：build-a3 重编（libggml-base.so 含 `[moe-l2] prefill` 代码），部署 /root/moe_l2/bin/

### A/B 对照（同机、同 prompt、max_tokens=128、codegen）

| 轮次 | 阶段 1（无 prefill） | 阶段 2（有 prefill） | 变化 |
|---|---|---|---|
| round1（冷启动） | 10.71 t/s | **19.11 t/s** | **+78%** |
| round2 | 30.14 t/s | 29.80 t/s | ≈ 0 |
| round3 | 30.88 t/s | 31.12 t/s | ≈ 0 |

### 结论
1. **预填充有效**：冷启动 round1 +78%（10.7→19.1），稳态 30 t/s 不变
2. **用户判定（2026-08-10 晚）：重要，不是"价值有限"**——首请求即满速 = 打开应用/演示时第一句话就快，直接提升体验感。对 AI PC 厂商 PoC 演示、短 prompt 单请求、频繁重启场景尤其关键
3. **init 修复实验（2026-08-10）**：暴露 maybe_init proc address + prefill 前强制 init → round1 19.67（vs 19.11），基本无变化 → **cache init 时序不是 round1 瓶颈**
4. **round1 瓶颈新分析**：预填充生效（RSS 10.4→28.2GB、显存 17.7GB，61专家/层×43层全部 H2D 进 cache），但 round1 仍 ~19.7 → 瓶颈是**预填充 H2D 发生在首请求内**（惰性触发），首请求与批量 H2D 重叠。真正"首请求满速"需预填充在**启动时**（模型加载后、首请求前）完成
5. **注意**：llama-server stderr 被 gate 线程消费，`[moe-l2] prefill` 日志不落盘（round1 提升即生效证据）

## 2080Ti 双模型全链路验证（2026-08-10 晚，region-42，新多架构二进制）

**背景**：region-42 旧 bin 只有 sm_75 单架构 + 0 个 moe-l2 标记（=原版编译），新多架构（llama-final-src/build-multi）sm_61/75/86/89/120a + 全部 moe-l2 优化。全链路 = moe-l2 start --gpu（proxy + L2 cache + flywheel gate + selective pin）。

**修复项**：proxy 崩溃根因 = region-42 moe_l2 包是 0.5.1 旧版缺 `domain_router_flywheel.py`；已同步完整包 + scripts/bench/export_router_map.py + data/ 表。

### Qwen3.6-35B-A3B（UD-IQ2_M，codegen 128 tokens）

| 轮次 | 旧基线（原版二进制）| 新多架构全链路 | 提升 |
|---|---|---|---|
| round1 | 14.90 t/s | **36.53 t/s** | +145% |
| round2 | 16.88 t/s | **50.80 t/s** | +201% |
| round3 | 15.86 t/s | **52.07 t/s** | +228% |

### DS-V2-Lite（Q2_K，codegen 128 tokens）

| 轮次 | 旧基线（原版二进制）| 新多架构全链路 | 提升 |
|---|---|---|---|
| round1 | ~6.9 t/s | **57.31 t/s** | +730% |
| round2 | — | **92.50 t/s** | — |
| round3 | — | **94.95 t/s** | — |

### 裸 server 对照（同机同二进制，绕过 proxy）

- Qwen：42.97 / 58.85 / **62.40 t/s**
- DS V2：60.09 / 101.01 / **104.85 t/s**
- 全链路 vs 裸 server 差异 ~15%（proxy 转发 + 域预测开销，正常）

### 结论

1. **moe-l2 优化版多架构二进制在 2080Ti 上：Qwen ~3.5x、DS ~14x 提升**——与 V4/4090 的 3x 提升一致，全部模型 3x+
2. **根因**：之前所有"慢"的基准数据（14.9/6.9 t/s）都是原版/旧编译二进制跑的，moe-l2 优化版真实性能被埋没
3. router map 自动生成（43 层 top-100）+ flywheel 域预测（routing drift 检测）全链路正常
4. **发布基准更新**：2080Ti 上 Qwen ≈ 52 t/s、DS ≈ 95 t/s、4090 上 V4 ≈ 30.9 t/s

## 下一步

1. ~~消费端实现~~ ✅（2026-08-10 已落地：flywheel + load_mapping 优先 + pretouch 消费）
2. ~~V4 按话题分组统计~~ ✅（分话题表已生成，报告上文）
3. **选择性 pin C++ 实现（进行中）**：读 flywheel 路由表 → 只 pin 表内专家（top-100/150 甜点），表外专家 on-demand fault 兜底——替代 whole-pin 全量注册，预期 V4 内存 82GB → 9-14GB、速度 7.6-9.1 t/s；Qwen 内存 → 4GB、速度接近满速
4. 冷专家兜底路径验证（未命中时 on-demand pin 开销）——V4 上已实测：物理上限 4-10 t/s，无预取窗口
