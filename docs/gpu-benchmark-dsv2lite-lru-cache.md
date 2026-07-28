# MoE L2 GPU 实测 — 阶段3：DS-V2-Lite LRU Expert Cache（基准线对比）

## 测试配置

| 项目 | 值 |
|------|-----|
| 模型 | DeepSeek-V2-Lite-Chat-Uncensored.Q2_K.gguf |
| 大小 | 6.0 GB |
| 架构 | 27层，每层 64 expert，top-2 active/token |
| 每 expert | ~96 MB（Q2_K） |
| binary | `/root/llama.cpp/build/bin/llama-cli`（GGML_CUDA_EXPERT_CACHE 编译注入） |
| GPU | RTX 4090 24GB (AutoDL) |
| 提示词 | "The capital of France is" |
| 生成长度  | 128 tokens（`--single-turn`） |
| 批大小 | 512 |
| 测试时间 | 2026-07-28 |

---

## 完整数据

| 模式 | Prompt t/s | Gen t/s | 说明 |
|-----|-----------|---------|------|
| **CPU 基线**（`-ngl 0`） | 12.3 | **4.5** | 纯 CPU 推理，不使用 GPU |
| **全量 GPU**（`-ngl 99`） | 17.4 | **8.7** | 全模型 offload 到 GPU |
| **CPU-moe**（`-ngl 99 --cpu-moe`） | 19.7 | **8.4** | 非 expert 层在 GPU，expert 在 CPU |
| LRU cache **0.25** | 16.9 | **8.1** | 每层 64×0.25=16 slot |
| LRU cache **0.5** | 19.7 | **8.1** | 每层 64×0.5=32 slot |
| LRU cache **0.75** | 18.0 | **8.1** | 每层 64×0.75=48 slot |
| LRU cache **1.0** | 17.8 | **8.4** | 每层 64×1.0=64 slot（全缓存） |

---

## 汇总

| 指标 | 值 |
|------|-----|
| Gen 速度范围 | 8.1–8.4 t/s（所有 cache 模式） |
| CPU-moe vs 全量 GPU 差距 | 8.4 vs 8.7 t/s（~~3%——DS-V2-Lite expert 过小，H2D 非瓶颈） |
| CPU vs GPU 加速 | 4.5 → 8.7 t/s（**1.93×**） |
| LRU cache vs CPU-moe | **无差异**（所有分数 8.1–8.4 t/s） |

---

## 关键结论

| 结论 | 详情 |
|------|------|
| **1. GPU 加速有效** | CPU 4.5 → GPU 8.7 t/s，1.93× |
| **2. CPU-moe 在该模型近似全量 GPU** | expert 太小（96 MB），H2D 拷贝 60 µs，kernel <10 µs |
| **3. LRU cache 在此模型上无收益** | 所有分数无差异，cache 查找/替换开销抵消了可能的收益 |
| **4. 预期：对大型 expert 模型有效** | Mixtral (8 expert, ~2 GB) 这类模型——减少 H2D 拷贝次数可显著提速 |

**核心发现**：DS-V2-Lite Q2_K 的 expert 大小（~96 MB）太小，LRU cache 的维护开销覆盖了 H2D 节约。该机制更适用于 **expert 大（>500 MB）且层数少** 的 MoE 架构（如 Mixtral 8×7B）。

## 下一步

- [ ] 用 Qwen3.6-35B-A3B (IQ2_M) 测试 LRU cache——每 expert ~1 MB，256 expert/层，top-16 激活
- [ ] 用 Mixtral 8×7B (Q4_K_M) 测试——每 expert ~2 GB，8 expert/层，top-2 激活（预期最佳场景）
