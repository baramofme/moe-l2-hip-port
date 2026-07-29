# moe-l2 Demos

| Demo | What it shows | Run it |
|------|--------------|--------|
| [demo_a3_compression.sh](demo_a3_compression.sh) | OG vs A3 VRAM 对比 | `bash examples/demo_a3_compression.sh` |

---

## A3 VRAM Compression Demo

在同一张 GPU 上跑同一个 MoE 模型，对比两种模式的显存和速度。

### 两种模式

| 模式 | 参数 | 行为 |
|------|------|------|
| **OG**（原始模式） | `--no-mmap -ngl 99` | 把整个模型加载到 GPU 显存 |
| **A3**（Expert Cache） | `--cpu-moe --expert-cache 0.25 -ngl 99` | 非专家层上 GPU，专家层放 CPU RAM。GPU 只缓存最近用过的 25% 专家，未命中的专家实时从 CPU 换入 |

### 怎么跑

```bash
# 需要：
#   1. A3-patched llama-batched 二进制
#   2. DeepSeek-V2-Lite 或类似 MoE 的 GGUF 模型
#   3. NVIDIA GPU + CUDA

# 修改脚本顶部的 MODEL 和 LLAMA_CLI 路径
vim examples/demo_a3_compression.sh

# 运行
bash examples/demo_a3_compression.sh
```

### 输出示例

```
==============================================
  RESULTS SUMMARY
==============================================

                       OG (full GPU)     A3 (expert cache)
  ────────────  ────────────────────  ────────────────────
  VRAM used                  6635 MB               1175 MB
  Speed                   126.64 t/s              8.22 t/s
  Savings                          -       5460 MB (5.64x)
```

### 结果怎么看

- **VRAM used**：模型加载后占用的 GPU 显存（减去空闲值）。OG 模式把 6.4 GB 的 DS-V2-Lite Q2_K 全量加载，A3 模式只有 ~1.2 GB。
- **Speed**：生成速度。A3 模式因为专家从 CPU RAM 经 PCIe 换入，比全显存慢 10-15 倍。单用户聊天场景可接受，批量场景建议加大 `--expert-cache` 值（如 0.5、0.75）。
- **Savings**：A3 省了多少显存和压缩比。省下来的显存可以同时跑第二个模型或做其他 GPU 任务。

### 显存监控原理

脚本用 `nvidia-smi --loop-ms=200` 在后台每 200ms 采样一次显存用量，推理结束后找采样序列的最大值作为峰值显存。之所以不用跑完再查，是因为 CUDA 进程退出后显存会被立即释放，跑完再查永远是 0。

### 调整参数

- `N_GPU_LAYERS=99` — 卸载所有层到 GPU。显存紧张时可减到 50-60。
- `--expert-cache 0.25` — 缓存 25% 的专家。显存够用时可提到 0.5 或 0.75 提速度。
- 去掉 `--no-mmap`（OG 模式）会变成按需加载，显存占用更低但推理也慢。
- 模型换成 Qwen3.6-A3B IQ2_M（~3 GB）在 8 GB 显卡上也能跑。
