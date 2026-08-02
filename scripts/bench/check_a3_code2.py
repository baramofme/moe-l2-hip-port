# ╔══════════════════════════════════════════════════════╗
# ║  INTERNAL EXPERIMENT SCRIPT                          ║
# ║  This is an internal research script.                ║
# ║  Paths (model files, SSH hosts, llama.cpp binary)    ║
# ║  are hardcoded to the author's environment.          ║
# ║  You MUST edit paths before running it.              ║
# ╚══════════════════════════════════════════════════════╝
# ⚠️ DEPRECATED (2026-07-29): 一次性代码检查脚本，SSH 端口已失效。
#
import pexpect, time, sys, os
pw = os.environ['AUTODL_PASSWORD']

# Kill everything
c = pexpect.spawn('/usr/bin/ssh', ['-oStrictHostKeyChecking=no', '-p39657', 'root@connect.bjb1.seetacloud.com',
    'pkill -9 -f llama-cli 2>/dev/null; sleep 2; python3 -c "import ctypes; lib=ctypes.CDLL(\"libcuda.so.1\"); lib.cuInit(0); lib.cuDeviceReset(0)" 2>&1; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits'], encoding='utf-8', timeout=20)
c.expect('password:', timeout=10)
c.sendline(pw)
c.expect(pexpect.EOF, timeout=20)
print(f"VRAM: {c.before.strip()[-10:]}MiB")

# Read the A3 code around line 1969
c2 = pexpect.spawn('/usr/bin/ssh', ['-oStrictHostKeyChecking=no', '-p39657', 'root@connect.bjb1.seetacloud.com',
    'sed -n "1940,2100p" /root/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu'], encoding='utf-8', timeout=15)
c2.expect('password:', timeout=10)
c2.sendline(pw)
c2.expect(pexpect.EOF, timeout=15)
print("=== A3 code (lines 1940-2100) ===")
print(c2.before or '')
