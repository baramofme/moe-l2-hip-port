#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════╗
# ║  INTERNAL DEPLOY SCRIPT                              ║
# ║  Requires SSH access to AutoDL GPU instance.         ║
# ║  HOST/PORT/USER are hardcoded. Use:                  ║
# ║    python3 reset_and_bench.py <password>              ║
# ║  Edit HOST/PORT/USER before use.                     ║
# ╚══════════════════════════════════════════════════════╝
# ⚠️ DEPRECATED (2026-07-29): 一次性部署脚本，SSH 端口已失效。
#
"""One-shot: reset GPU, run benchmark, report results."""
import pexpect, sys, base64, time

password = sys.argv[1]

# Step 1: reset GPU via reset_gpu.py
c = pexpect.spawn('/usr/bin/ssh', ['-oStrictHostKeyChecking=no', '-p39657',
    'root@connect.bjb1.seetacloud.com', 'python3 /root/reset_gpu.py'],
    encoding='utf-8', timeout=20)
c.expect('password:', timeout=10)
c.sendline(password)
c.expect(pexpect.EOF, timeout=20)
print("=== RESET ===")
print(c.before[-500:] if c.before else '')

time.sleep(2)

# Step 2: check VRAM
c = pexpect.spawn('/usr/bin/ssh', ['-oStrictHostKeyChecking=no', '-p39657',
    'root@connect.bjb1.seetacloud.com',
    "nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits"],
    encoding='utf-8', timeout=15)
c.expect('password:', timeout=10)
c.sendline(password)
c.expect(pexpect.EOF, timeout=15)
print("=== VRAM ===")
print(c.before[-200:] if c.before else '')

# Step 3: Upload bench v2 script
with open('/opt/data/run_bench_v2.sh', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

c = pexpect.spawn('/usr/bin/ssh', ['-oStrictHostKeyChecking=no', '-p39657',
    'root@connect.bjb1.seetacloud.com',
    f'echo {b64} | base64 -d > /root/run_bench_v2.sh && chmod +x /root/run_bench_v2.sh'],
    encoding='utf-8', timeout=20)
c.expect('password:', timeout=10)
c.sendline(password)
c.expect(pexpect.EOF, timeout=20)
print("=== UPLOAD ===")
print(c.before[-200:] if c.before else '')

# Step 4: Run just DS-V2-Lite (first model, 5 tests) with verbose output
c = pexpect.spawn('/usr/bin/ssh', ['-oStrictHostKeyChecking=no', '-p39657',
    'root@connect.bjb1.seetacloud.com',
    'bash /root/run_bench_v2.sh 2>&1 | tee /tmp/bench_final.log'],
    encoding='utf-8', timeout=600)
c.expect('password:', timeout=10)
c.sendline(password)

# Read output incrementally
output_lines = []
while True:
    try:
        c.expect('\n', timeout=5)
        line = c.before.strip()
        if line:
            print(f"  {line}")
            output_lines.append(line)
    except pexpect.TIMEOUT:
        break
    except pexpect.EOF:
        break

print(f"\n=== DONE (exit: {c.exitstatus}) ===")
print(f"Total lines: {len(output_lines)}")
