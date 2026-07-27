# MoE L2 GPU 实测 — 阶段2：DS-V2-Lite（完整报告）

## 测试配置

| 项目 | 值 |
|------|-----|
| 模型 | DeepSeek-V2-Lite-Chat-Uncensored.Q2_K.gguf |
| 大小 | 6.0 GB |
| binary | `/root/llama-cli-force-cpu-experts-20260901`（CPU 专家 offload 版） |
| GPU | RTX 4090 24GB (AutoDL) |
| KV cache | q8_0 |
| 测试时间 | 2026-07-27 |
| 测试次数 | 8 领域 × 3 阶段 = **24/24** ✅ |

---

## 完整测试数据

### codegen（编程生成）
| 阶段 | Prompt t/s | Gen t/s | VRAM (MiB) |
|------|-----------|---------|------------|
| short (-n 128, -c 512) | 18.5 | 7.9 | — |
| followup (-n 8, -c 1536) | 45.4 | 9.5 | — |
| longtail (-n 512, -c 1024) | 82.3 | 8.8 | 1363 |

### debug（代码调试审查）
| 阶段 | Prompt t/s | Gen t/s | VRAM (MiB) |
|------|-----------|---------|------------|
| short | 61.7 | 8.3 | — |
| followup | 47.6 | 9.5 | — |
| longtail | 83.8 | 8.4 | 1363 |

### math（数学推理）
| 阶段 | Prompt t/s | Gen t/s | VRAM (MiB) |
|------|-----------|---------|------------|
| short | 31.7 | 8.7 | — |
| followup | 35.9 | 8.3 | — |
| longtail | 66.2 | 8.6 | 1363 |

### logic（逻辑谜题）
| 阶段 | Prompt t/s | Gen t/s | VRAM (MiB) |
|------|-----------|---------|------------|
| short | 43.9 | 8.1 | — |
| followup | 40.6 | 9.3 | — |
| longtail | 69.1 | 8.2 | 1363 |

### general_qa（通用问答）
| 阶段 | Prompt t/s | Gen t/s | VRAM (MiB) |
|------|-----------|---------|------------|
| short | 18.5 | 8.2 | — |
| followup | 38.0 | 8.7 | — |
| longtail | 40.5 | 7.9 | 1359 |

### chinese_tech（中文技术）
| 阶段 | Prompt t/s | Gen t/s | VRAM (MiB) |
|------|-----------|---------|------------|
| short | 19.3 | 8.5 | — |
| followup | 47.0 | 9.0 | — |
| longtail | 42.7 | 7.9 | 1359 |

### creative_write（创意写作）
| 阶段 | Prompt t/s | Gen t/s | VRAM (MiB) |
|------|-----------|---------|------------|
| short | 34.1 | 8.1 | — |
| followup | 38.2 | 9.2 | — |
| longtail | 42.9 | 8.2 | 1359 |

### translate（中英翻译）
| 阶段 | Prompt t/s | Gen t/s | VRAM (MiB) |
|------|-----------|---------|------------|
| short | 36.0 | 8.2 | — |
| followup | 72.2 | 8.8 | — |
| longtail | 69.9 | 7.9 | 1363 |

---

## 汇总统计

| 指标 | DS-V2-Lite (Q2_K) |
|------|-------------------|
| Gen 速度范围 | 7.9–9.5 t/s |
| Gen 速度均值 | ~8.4 t/s |
| Prompt 速度范围 | 18.5–83.8 t/s |
| VRAM（所有 longtail 平均） | ~1361 MiB（1359–1363） |
| VRAM 压缩比 | 6.0 GB / 1.36 GB ≈ **4.41×** |

### 域间差异分析

| 领域 | 平均 Gen t/s | vs 全量均值 |
|------|-------------|------------|
| codegen | 8.73 | +3.9% |
| debug | 8.73 | +3.9% |
| math | 8.53 | +1.5% |
| logic | 8.53 | +1.5% |
| general_qa | 8.27 | -1.5% |
| chinese_tech | 8.47 | +0.8% |
| creative_write | 8.50 | +1.2% |
| translate | 8.30 | -1.2% |

域间差异 **<5%**，与 Qwen3.6 结论一致——Gen 速度是 domain-independent。

---

## 完整对比：DS-V2-Lite vs Qwen3.6-35B-A3B

| 指标 | Qwen3.6 (IQ2_M, 10.7GB) | DS-V2-Lite (Q2_K, 6.0GB) |
|------|------------------------|--------------------------|
| 模型大小 | 10.7 GB | 6.0 GB |
| Gen 速度范围 | 5.0–6.6 t/s | 7.9–9.5 t/s |
| **Gen 速度均值** | **~6.0 t/s** | **~8.4 t/s** (+40%) |
| Prompt 速度范围 | 10–119 t/s | 18–84 t/s |
| **VRAM 均值** | **~2242 MiB** | **~1361 MiB** (-39%) |
| VRAM 压缩比 | 10.7/2.24 ≈ **4.78×** | 6.0/1.36 ≈ **4.41×** |
| 域间 Gen 差异 | <3% | <5% |
| 输出质量 | ✅ 可用（IQ2_M 保留基本语义） | ❌ Q2_K 严重退化（大量重复/幻觉） |
| 适用场景 | 日常推理、技术问答 | 仅测速/显存实验，不建议实际使用 |

### 核心发现

1. **DS-V2-Lite 未被 binary 正确 CPU-offload** — 1.36GB VRAM vs 6.0GB 模型大小 = 4.41× 压缩比，与 Qwen3.6 的 4.78× 在同一量级，说明 force-cpu 二进制对该架构也生效了部分 offload
2. **Gen 速度比 Qwen3.6 快 40%** — 主要因为模型小（6GB vs 10.7GB），但 Q2_K 输出质量太差抵消了这个优势
3. **VRAM 稳定 1.36GB** — 极低，8GB 消费级显卡可以轻松运行
4. **Q2_K 量化对大模型经验损失严重** — 所有 longtail 测试都出现严重重复循环，DS-V2-Lite 在 Q2_K 下实际不可用
