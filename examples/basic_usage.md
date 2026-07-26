# moe-l2 — Example: Basic Usage

## Prerequisites

- [ollama](https://ollama.com) installed and running
- A MoE model pulled (e.g., `ollama pull qwen3:30b`)

## Start the Scheduler

```bash
# After pip install moe-l2:
moe-l2 start --model qwen3:30b --l2-size 8GB
```

Output:
```
moe-l2 0.1.0 — starting L2 scheduler
  model:   qwen3:30b
  L2 size: 8GB
  port:    11435
L2 cache initialized: 40 layers × 96 slots
Starting proxy on 127.0.0.1:11435 → ollama
Connect your client to http://127.0.0.1:11435
```

## Use It

```bash
# Point your client to localhost:11435 instead of 11434
curl http://localhost:11435/api/generate \
  -d '{"model": "qwen3:30b", "prompt": "Write a Python web server"}'

# Or configure your app to use the moe-l2 proxy
```

## Check Stats

```bash
moe-l2 stats
```

## Docker

```yaml
# docker-compose.yml
services:
  ollama:
    image: ollama/ollama
    volumes:
      - ./models:/root/.ollama

  moe-l2:
    image: moe-l2
    environment:
      - OLLAMA_HOST=http://ollama:11434
    ports:
      - "11435:11435"
```
