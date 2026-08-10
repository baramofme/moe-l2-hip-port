#!/usr/bin/env python3
"""云机实时采样：每秒记录显存 + token 进度（先启动本脚本，1.5s 后再启动生成）。"""
import os, subprocess, time, sys

# 读 token 进度（生成脚本写 /tmp/rec_gen.log: "<elapsed> <tokens>")
def read_tokens():
    try:
        with open("/tmp/rec_gen.log") as f:
            parts = f.read().strip().split()
            return int(parts[1]) if len(parts) >= 2 else 0
    except Exception:
        return 0

DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 300
t0 = time.time()
with open("/tmp/rec_data.csv", "w") as f:
    f.write("t,vram_mib,tokens\n")
    while time.time() - t0 < DURATION:
        try:
            v = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            ).stdout.strip().split("\n")[0]
        except Exception:
            v = "0"
        toks = read_tokens()
        f.write(f"{time.time()-t0:.1f},{v},{toks}\n")
        f.flush()
        time.sleep(1.0)
print("SAMPLING_DONE")
