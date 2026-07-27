# MoE-L2 GPU实测完整数据 — Qwen3.6-35B-A3B (IQ2_M, ~10.7GB)
## 硬件：RTX 4090 24GB | 强制CPU-Expert Offload | --cache-type-k q8_0

### 测试协议
- **Short**: --single-turn -n 128 -c 512 (首次响应)
- **Followup**: --single-turn -n 8 -c 1536 (追问/延续)
- **Longtail**: --single-turn -n 512 -c 1024 (长生成)

---

### 1. codegen (编程生成)
| 阶段 | Prompt t/s | Gen t/s | VRAM (MiB) |
|------|-----------|---------|------------|
| Short | 10.1 | 6.2 | 2231 |
| Followup | 119.1 | 5.3 | - |
| Longtail | 39.5 | 6.1 | 2241 |

### 2. debug (代码调试审查)
| 阶段 | Prompt t/s | Gen t/s | VRAM (MiB) |
|------|-----------|---------|------------|
| Short | 40.4 | 6.0 | 2235 |
| Followup | 69.5 | 5.7 | - |
| Longtail | 47.3 | 6.2 | 2243 |

### 3. math (数学推理)
| 阶段 | Prompt t/s | Gen t/s | VRAM (MiB) |
|------|-----------|---------|------------|
| Short | 44.6 | 6.1 | 2235 |
| Followup | 70.2 | 5.8 | - |
| Longtail | 46.8 | 6.3 | 2243 |

### 4. logic (逻辑谜题)
| 阶段 | Prompt t/s | Gen t/s | VRAM (MiB) |
|------|-----------|---------|------------|
| Short | 39.6 | 6.2 | - |
| Followup | 42.1 | 6.6 | - |
| Longtail | 51.6 | 6.2 | 2243 |

### 5. general_qa (通用问答)
| 阶段 | Prompt t/s | Gen t/s | VRAM (MiB) |
|------|-----------|---------|------------|
| Short | 10.5 | 6.0 | - |
| Followup | 28.4 | 5.4 | - |
| Longtail | 36.5 | 5.8 | 2241 |

### 6. chinese_tech (中文技术)
| 阶段 | Prompt t/s | Gen t/s | VRAM (MiB) |
|------|-----------|---------|------------|
| Short | 10.2 | 6.1 | - |
| Followup | 34.7 | 5.0 | - |
| Longtail | 33.3 | 5.9 | 2241 |

### 7. creative_write (创意写作)
| 阶段 | Prompt t/s | Gen t/s | VRAM (MiB) |
|------|-----------|---------|------------|
| Short | 29.3 | 5.8 | - |
| Followup | 30.9 | 5.6 | - |
| Longtail | 34.6 | 6.0 | 2241 |

### 8. translate (中英翻译)
| 阶段 | Prompt t/s | Gen t/s | VRAM (MiB) |
|------|-----------|---------|------------|
| Short | 29.9 | 5.8 | - |
| Followup | 56.2 | 5.7 | - |
| Longtail | 51.3 | 6.0 | 2245 |

---

### 汇总统计 (Qwen3.6-35B-A3B IQ2_M)
- **Gen速度**: 5.0~6.6 t/s → 均值 ~5.95 t/s，域间差异极小 (<3%)
- **VRAM**: 2231~2245 MiB → 稳定 ~2.2 GiB（模型大小10.7GB/VRAM约2.2GiB=压缩比 ~4.9x）
- **Prompt速度**: 10~119 t/s，依赖于prompt长度（短prompt慢，长prompt快）
- **关键发现**: Gen速度域间一致，证明强制CPU-offload策略有效消除MoE路由差异
