# MoE-L2 GPU 显存实测 — 中间结果

**模型**: Qwen3.6-35B-A3B-UD-IQ2_M (10.7 GB)  
**硬件**: RTX 4090 24GB (AutoDL)  
**配置**: Force-CPU-Experts (默认已压缩 VRAM)  
**binary**: `/root/llama-cli-force-cpu-experts-20260901`  
**日期**: 2026-07-26

---

## 汇总表

| 领域 | 短对话 Prompt | 短对话 Gen | 追问 Prompt | 追问 Gen | 长尾 Prompt | 长尾 Gen | VRAM (MiB) |
|------|:-----------:|:---------:|:---------:|:-------:|:---------:|:-------:|:---------:|
| **codegen** 编程生成 | 10.1 | 6.2 | 119.1 | 5.3 | 39.5 | 6.1 | 2231~2241 |
| **debug** 代码调试审查 | 40.4 | 6.0 | 69.5 | 5.7 | 47.3 | 6.2 | 2235~2243 |
| **math** 数学推理 | 44.6 | 6.1 | 70.2 | 5.8 | 46.8 | 6.3 | 2243 |

**进展**: 3/8 领域 × 3 阶段 = 9/24 次测试完成

---

## 逐条原始数据

### codegen (编程生成)

**短对话** (-n 128, -c 512)
```
[ Prompt: 10.1 t/s | Generation: 6.2 t/s ]
VRAM: 2231 MiB
```

**追问** (-n 8, -c 1536) — 长上下文 + 短回答
```
[ Prompt: 119.1 t/s | Generation: 5.3 t/s ]
VRAM: ~2235 MiB (太快未捕获)
```

**长追尾** (-n 512, -c 1024)
```
[ Prompt: 39.5 t/s | Generation: 6.1 t/s ]
VRAM: 2241 MiB
```

### debug (代码调试审查)

**短对话**
```
[ Prompt: 40.4 t/s | Generation: 6.0 t/s ]
VRAM: 2235 MiB
```

**追问**
```
[ Prompt: 69.5 t/s | Generation: 5.7 t/s ]
VRAM: ~2235 MiB (太快未捕获)
```

**长追尾**
```
[ Prompt: 47.3 t/s | Generation: 6.2 t/s ]
VRAM: 2243 MiB
```

### math (数学推理)

**短对话**
```
[ Prompt: 44.6 t/s | Generation: 6.1 t/s ]
VRAM: ~2235 MiB (nvidia-smi未捕获到模型加载后)
```

**追问**
```
[ Prompt: 70.2 t/s | Generation: 5.8 t/s ]
VRAM: ~2235 MiB (太快未捕获)
```

**长追尾**
```
[ Prompt: 46.8 t/s | Generation: 6.3 t/s ]
VRAM: 2243 MiB
```

---

## 初步观察

### VRAM
- 所有测试稳定 **2231~2243 MiB** (~2.2 GiB)
- 短对话 vs 长追尾 VRAM 差异极小 (< 12 MiB)，模型权重占绝对主导
- 上下文长度对 VRAM 影响可忽略

### Generation 速度
- 稳定 **5.3 ~ 6.3 t/s** 区间
- 追问 Gen 略慢 (5.3~5.8)，可能因长上下文增加 KV cache 访存开销
- 领域间差异不大

### Prompt 处理速度
- **短问题** (10~40 tokens): 10~45 t/s
- **长上下文** (300~700 tokens): 69~119 t/s — 更长 prompt 并行化更充分
- 追问和长尾的 Prompt 速度明显高于短对话

---

## 待完成

剩余 5 个领域 × 3 阶段 = 15 次测试:

| 领域 | 状态 |
|------|------|
| logic 逻辑谜题 | ⏳ 未跑 |
| general_qa 通用问答 | ⏳ 未跑 |
| chinese_tech 中文技术 | ⏳ 未跑 |
| creative_write 创意写作 | ⏳ 未跑 |
| translate 中英翻译 | ⏳ 未跑 |

Prompt 文件已上传至云 GPU `/root/prompts/`。测试模式已验证，直接接着跑就行。
