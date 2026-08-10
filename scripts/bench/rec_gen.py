#!/usr/bin/env python3
"""Qwen 3200 tokens 长生成，流式记录 token 进度到 /tmp/rec_gen.log，完成写 rec_done.flag + rec_full.txt。"""
import json, time, urllib.request

BASE = "http://127.0.0.1:11435/v1/chat/completions"
PROMPT = ("请写一篇关于 MoE 大模型推理优化的科普文章，包含五段："
          "1) MoE 为什么省算力 2) 显存瓶颈在哪 3) 专家卸载原理 4) 实际效果对比 5) 未来方向。"
          "每段 300 字以上，共 3200 字左右。")

body = json.dumps({
    "model": "x",
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": 3500,
    "stream": True,
}).encode()

t0 = time.time()
toks = 0
full = []
gen_log = open("/tmp/rec_gen.log", "w")

req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as resp:
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            d = json.loads(payload)
            delta = d.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content") or delta.get("reasoning_content") or ""
            if piece:
                toks += 1
                full.append(piece)
                gen_log.write(f"{time.time()-t0:.1f} {toks}\n")
                gen_log.flush()
        except Exception:
            pass

gen_log.close()
elapsed = time.time() - t0
with open("/tmp/rec_full.txt", "w") as f:
    f.write("".join(full))
with open("/tmp/rec_done.flag", "w") as f:
    f.write(f"tokens={toks} elapsed={elapsed:.1f}s tps={toks/elapsed:.2f}\n")
print(f"DONE tokens={toks} elapsed={elapsed:.1f}s tps={toks/elapsed:.2f}")
