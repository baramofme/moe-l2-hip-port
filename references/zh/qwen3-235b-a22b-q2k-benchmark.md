# Qwen3-235B-A22B Q2_K benchmark 报告（2026-08-11 更新）

## 最新成果（2026-08-11）：RTX 4090 三轮实测（bins-v0.8.0 / selective pin）

> moe-l2 优化版 llama-server（moe_l2 0.8.0）在 RTX 4090 24GB 上完整跑通 **Qwen3-235B-A22B Q2_K（85.7GB，93 层 × 128 专家 top-8）**：24GB 显存 + 55GB 内存，稳态 ~3.9 t/s。相比 08-03 首次跑通（旧实现无 A3 cache，1 t/s）**4 倍提升**。

### 测试环境

| 项 | 值 |
|---|---|
| GPU | AutoDL bjb1，RTX 4090 24GB（驱动 580.76.05，CUDA 13.0） |
| 二进制 | moe-l2 优化版 llama-server（/root/moe_l2/bin/，moe_l2 0.8.0） |
| 模型 | Qwen3-235B-A22B-GGUF Q2_K，85,691,002,226 字节（85.7GB），两分片（49.9GB + 35.8GB），字节级校验通过 |
| 参数 | `-ngl 99 -c 2048`，`GGML_OP_OFFLOAD_MIN_BATCH=1`，`GGML_CUDA_EXPERT_CACHE=1`（A3 cache 2048 slots） |
| 口径 | 每场景 128 tokens；prompt tokens 为该请求完整输入（含携带历史） |

### 三轮实验

1. **裸测全量**：llama-server 直启，无 selective pin（全部 128 专家/层可驻留）
2. **裸测 selective pin**：`MOE_L2_ROUTER_FILE=router_qwen235b.map`（93 层 × top-60，覆盖 98.5% 专家激活）
3. **全链路 selective pin**：`moe-l2 start --gpu --router-map router_qwen235b.map`（cli → proxy → llama-server）

### 实测数据

| 指标 | 裸测全量 | 裸测 pin | 全链路 pin |
|---|---|---|---|
| 短对话（21 prompt） | 4.07 t/s | 2.81 t/s | 2.79 t/s |
| 追问稳态（3 轮） | 4.09-4.13 t/s | 3.76-3.90 t/s | 3.79-3.93 t/s |
| 长对话（1216 prompt） | 3.57 t/s | 3.22 t/s | 3.03 t/s |
| 显存峰值 | 12,817 MiB | 12,775 MiB | 13,909 MiB |
| **内存峰值 RSS** | **80,788 MB** | **54,216 MB（↓33%）** | **54,659 MB** |
| 加载时间 | 72.8s | 33.1s（↓54%） | 48.1s（含 proxy 链路） |

### 关键结论（2026-08-11）

1. **235B 在 4090 上完整可用**：24GB 显存 + 55GB 内存 = 跑 85.7GB 模型，稳态 ~3.9 t/s；全链路（moe-l2 start --gpu）与裸测 pin 一致（proxy 开销 <2%）
2. **selective pin 收益**：用速度换 33% 内存（80.8→54.2GB）+ 加载时间减半（72.8→33.1s）。速度损耗分场景：短对话首token -31%（表外专家 on-demand 冷启动）、长对话 -10%、追问稳态 -5~8%
3. **内存档位**：32GB 内存机器跑不了 235B（54.7GB > 32GB）；**64GB 内存 + 24GB 显存可跑**
4. **专家路由高度集中**：top-60 专家/层覆盖 98.5% 激活——路由表驱动的 selective pin 对超大 MoE 模型收益显著（换 33% 内存只付首token损耗）

---

## 数据可复现性

- 脚本：`bench235b_full.py`（裸测全量）、`bench235b_pin.py`（裸测 pin）、`bench235b_fullchain.py`（全链路）、`gen_router_235b.py`（路由表生成）
- 原始数据：`all_results.txt`（三轮完整数据）、`expert235b.log`（60,683 行 EXPERT 日志）、`router_qwen235b.map`（93×top-60）
- 模型：Qwen3-235B-A22B Q2_K 两分片（字节级校验通过）
