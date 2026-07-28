# MoE L2 GPU 实测 — 阶段3：Qwen3.6-35B-A3B LRU Expert Cache（基准线对比）

## 测试配置

| 项目 | 值 |
|------|-----|
| 模型 | Qwen3.6-35B-A3B-UD-IQ2_M.gguf |
| 大小 | 11 GB |
| 架构 | 244 层（预估），每层 256 expert，top-8×2 active/token |
| 每 expert | ~1 MB（IQ2_M） |
| binary | `/root/llama.cpp/build/bin/llama-cli`（GGML_CUDA_EXPERT_CACHE 编译注入） |
| GPU | RTX 4090 24GB (AutoDL) |
| 提示词 | "The capital of France is" |
| 生成长度 | 128 tokens（`--single-turn`） |
| 批大小 | 512 |
| 测试时间 | 2026-07-28 |

---

## 完整数据

| 模式 | Prompt t/s | Gen t/s | 说明 |
|-----|-----------|---------|------|
| **CPU 基线**（`-ngl 0`） | 6.1 | **1.8** | 纯 CPU 推理 |
| **全量 GPU**（`-ngl 99`） | 9.1 | **6.5** | 全模型 offload 到 GPU |
| **CPU-moe**（`-ngl 99 --cpu-moe`） | 9.1 | **6.3** | 非 expert 层在 GPU，expert 在 CPU |
| LRU cache **0.25** | 10.2 | **6.2** | 每层 256×0.25=64 slot |
| LRU cache **0.5** | 9.0 | **6.0** | 每层 256×0.5=128 slot |
| LRU cache **0.75** | 9.5 | **6.0** | 每层 256×0.75=192 slot |
| LRU cache **1.0** | 8.6 | **5.8** | 每层 256×1.0=256 slot（全缓存） |

---

## 汇总

| 指标 | 值 |
|------|-----|
| Gen 速度范围 | 5.8–6.5 t/s |
| CPU-moe vs 全量 GPU 差距 | 6.3 vs 6.5 t/s（~3%——expert 过小） |
| CPU vs GPU 加速 | 1.8 → 6.5 t/s（**3.6×**） |
| LRU cache vs CPU-moe | **无收益，1.0 反而最慢**（5.8 vs 6.3） |

---

## 关键结论

| 结论 | 详情 |
|------|------|
| **1. GPU 加速效果显著** | CPU 1.8 → GPU 6.5 t/s，3.6× |
| **2. CPU-moe 在该模型近似全量 GPU** | expert 太小（~1 MB），H2D 拷贝非瓶颈 |
| **3. LRU cache 在此模型上无收益** | 所有分数均不优于 CPU-moe；1.0 最慢（5.8 t/s），cache 维护开销 > H2D 收益 |
| **4. DS-V2-Lite 与 Qwen3.6 结论一致** | 均因 expert 过小导致 cache 不生效 |

**核心发现**：Qwen3.6 IQ2_M 的 expert 大小（~1 MB）同样太小。LRU cache 预期在 **expert > 500 MB** 的模型（如 Mixtral 8×7B Q4_K_M，每 expert ~2 GB）上才能体现价值。

## 两模型对比

| 指标 | DS-V2-Lite (Q2_K) | Qwen3.6 (IQ2_M) |
|------|-------------------|-----------------|
| 模型大小 | 6.0 GB | 11 GB |
| CPU Gen | 4.5 t/s | 1.8 t/s |
| 全量 GPU Gen | 8.7 t/s | 6.5 t/s |
| CPU-moe Gen | 8.4 t/s | 6.3 t/s |
| LRU cache 最佳 | 8.4 t/s（1.0） | 6.2 t/s（0.25） |
| CPU→GPU 加速比 | 1.93× | 3.6× |
| Expert 大小 | ~96 MB | ~1 MB |
| **Cache 有效性** | ❌ 无效 | ❌ 无效 |

## 下一步

- [ ] 用 Mixtral 8×7B (Q4_K_M) 测试——每 expert ~2 GB，预期 cache 生效
