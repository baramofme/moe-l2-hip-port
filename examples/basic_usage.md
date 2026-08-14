# moe-l2 — Example: Basic Usage

## Prerequisites

- Linux x86_64 + NVIDIA GPU + CUDA（GPU 模式需要）
- 一个 MoE GGUF 模型（DeepSeek-V2-Lite / Qwen3.6-A3B / Mixtral 等）
- Python 3.10+

## 安装

```bash
pip install moe-l2
moe-l2 download-bins   # 拉取预编译 CUDA llama-server（含 host-buffer A3 patch）
```

## 启动（GPU 模式，推荐）

```bash
moe-l2 start --model /path/to/model.gguf --gpu
```

输出：
```
moe-l2 0.5.0 — starting L2 scheduler + proxy
  model:   /path/to/model.gguf
  expert:  1 MB each
  L2 size: 4GB total → 384 slots/layer
  layers:  32
  slots:   384/layer = 12288 total

  [GPU mode] Starting bundled llama-server (A3 tiered scheduling)...
  Waiting for llama-server to start... READY
  backend:  127.0.0.1:11436 (llama-server + CUDA + A3)
Starting proxy on 127.0.0.1:11435
```

要点：
- 代理监听 `localhost:11435`，后端 llama-server 在 `11436`
- 内部自动设置 `GGML_OP_OFFLOAD_MIN_BATCH=1`：专家驻留 CPU pinned 内存（host buffer，零显存），每步只把激活的专家拷到 GPU 直算
- 实测（RTX 4090，2026-08-14 bins-v0.4.1 修复版 selective pin）：DS-V2-Lite 133.2 t/s @ ~10 GB、Qwen3.6-A3B 44-48 t/s @ ~9.3 GB（旧 145.63/74.99 @ 4.9/3.1GB 为 P0 假速度作废）
- 多架构二进制（bins-v0.4.1）：GTX 1080 → RTX 50 系一个包全支持；2080 Ti 全链路实测 DS-V2-Lite **85.25** / Qwen3.6-A3B **30.87 t/s（有锁版）**（bins-v0.4.1 修复版，2026-08-14；旧 87.25/47.24 为 P0 假速度作废），见 [multi-arch-three-gpu-benchmark.md](../references/en/multi-arch-three-gpu-benchmark.md)

## 使用

```bash
# OpenAI 兼容端点（llama-server 后端）
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Write a Python web server"}]}'

# 流式响应
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "hi"}], "stream": true}'

# ollama 兼容端点
curl http://localhost:11435/api/chat \
  -d '{"model":"default","messages":[{"role":"user","content":"hello"}]}'
```

## 采集路由数据（数据飞轮种子）

```bash
# 采集 8 个领域的 expert 路由日志 → ~/.moe-l2/maps/domain_expert_map.json
moe-l2 collect --model /path/to/model.gguf

# 把映射内嵌进 GGUF（可选，免外部文件）
moe-l2 embed-map --model model.gguf --output model-embedded.gguf
```

## 查看统计

```bash
moe-l2 stats
```

## 常见参数

| 参数 | 说明 |
|------|------|
| `--model auto` | 自动扫描 `/opt/data/models/*.gguf` |
| `--l2-size 4GB` | 目标 L2 缓存大小（默认 4GB） |
| `--port 11435` | 代理端口（默认） |
| `--gpu` | GPU 模式（需要 CUDA + NVIDIA 显卡 + download-bins） |

## Docker

```yaml
# docker-compose.yml
services:
  moe-l2:
    image: moe-l2
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "11435:11435"
    volumes:
      - ./models:/models
      - ~/.moe-l2:/root/.moe-l2
```
