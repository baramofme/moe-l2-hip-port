# moe-l2 Demos

| Demo | What it shows | Run it |
|------|--------------|--------|
| [demo_a3_compression.sh](demo_a3_compression.sh) | OG vs host-buffer vs sched-cache VRAM/速度对比 | `bash examples/demo_a3_compression.sh` |

---

## A3 VRAM Compression Demo

在同一张 GPU 上跑同一个 MoE 模型，对比三种模式的显存和速度。

### 三种模式

| 模式 | 参数 | 行为 |
|------|------|------|
| **OG**（原始模式） | `--no-mmap -ngl 99` | 把整个模型加载到 GPU 显存 |
| **moe-l2**（host-buffer） | 默认 mmap + `GGML_OP_OFFLOAD_MIN_BATCH=1` | 专家驻留 CPU pinned 内存（零显存），调度器每步只把**激活的专家**拷到 GPU，GPU 快路径直算 |
| **cache**（+sched-cache） | 上述 + `GGML_CUDA_EXPERT_CACHE=0.25` | 热专家留在 GPU（D2D 拷贝，免 PCIe），未命中才从 host buffer 换入 |

### 怎么跑

```bash
# 需要：
#   1. A3-patched + host-buffer llama-batched 二进制（moe-l2 bin bundle）
#   2. DeepSeek-V2-Lite 或类似 MoE 的 GGUF 模型
#   3. NVIDIA GPU + CUDA

# 修改脚本顶部的 MODEL 和 LLAMA_CLI 路径
vim examples/demo_a3_compression.sh

# 运行
bash examples/demo_a3_compression.sh
```

### 输出示例（RTX 4090 / DS-V2-Lite Q2_K，2026-08-02 实测）

```
==============================================
  RESULTS SUMMARY
==============================================

                        OG (full GPU)     host-buffer     host-buffer+cache
  ────────────  ────────────────────  ────────────────────  ────────────────────
  VRAM used                6635 MB             1625 MB              1625 MB
  Speed                    65 t/s             37.5 t/s             39.2 t/s
  Savings                          -       5010 MB (4.08x)                -
```

Prompt 处理：OG 110 t/s、host-buffer 99 t/s、cache 模式 **308 t/s（+211%）**。

### 结果怎么看

- **VRAM used**：模型加载后占用的 GPU 显存（减去空闲值）。OG 模式把 ~6.6 GB 的 DS-V2-Lite Q2_K 全量加载，host-buffer 模式只有 ~1.6 GB（专家不占显存）。
- **Speed**：host-buffer 模式因为专家从 CPU pinned 内存经 PCIe 拷入（只拷激活专家），约为全显存的 58%。单用户聊天场景完全流畅（37.5 t/s ≈ 每秒 75 个汉字）。
- **cache 模式**：对 DS 这类中专家高重复率模型，sched-cache 让 prompt 处理 +211%（99 → 308 t/s），生成 +5%（37.5 → 39.2 t/s），VRAM 零增加。

### 显存监控原理

脚本用 `nvidia-smi --loop-ms=200` 在后台每 200ms 采样一次显存用量，推理结束后找采样序列的最大值作为峰值显存。之所以不用跑完再查，是因为 CUDA 进程退出后显存会被立即释放，跑完再查永远是 0。

### 调整参数

- `N_GPU_LAYERS=99` — 卸载所有层到 GPU。显存紧张时可减到 50-60。
- `CACHE_RATIO=0.25` — sched-cache 比例。DS 上 0.25 已到顶（16 slots/层覆盖全部热专家），更大档位只加 VRAM 无速度收益。
- **Qwen3.6-A3B 不需要开 cache**（专家太小，无收益只加 VRAM）；Mixtral-8x7B 也不需要（top-2 命中率低）。cache 只对 DS 这类中专家（~1.5MB）高重复率模型有用。
- 模型换成 Qwen3.6-A3B IQ2_M（~3 GB）在 8 GB 显卡上也能跑。

### 三模型推荐配置（2026-08-02 实测）

| 模型 | 推荐 | 理由 |
|------|------|------|
| DS-V2-Lite | `OFFLOAD_MIN_BATCH=1` + `GGML_CUDA_EXPERT_CACHE=0.25` | Prompt +211%，VRAM 零增加 |
| Qwen3.6-A3B | `OFFLOAD_MIN_BATCH=1`（不开 cache） | 专家太小，无收益 |
| Mixtral-8x7B | `OFFLOAD_MIN_BATCH=1`（不开 cache） | top-2 命中率低，白占 VRAM |
