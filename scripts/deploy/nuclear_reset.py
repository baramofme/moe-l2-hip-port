#!/usr/bin/env python3
"""Nuclear option: unload/reload nvidia modules to force GPU memory cleanup."""
import pexpect, sys

password = sys.argv[1]

# Check modules
c = pexpect.spawn('/usr/bin/ssh', ['-oStrictHostKeyChecking=no', '-p39657',
    'root@connect.bjb1.seetacloud.com',
    "lsmod | grep nvidia"],
    encoding='utf-8', timeout=15)
c.expect('password:', timeout=10)
c.sendline(password)
c.expect(pexpect.EOF, timeout=15)
print("=== NVIDIA MODULES ===")
print(c.before or '')

# Unload in reverse dependency order
cmds = [
    "rmmod nvidia_uvm",
    "sleep 1",
    "nvidia-smi 2>&1 | head -3",
]

for cmd in cmds:
    c = pexpect.spawn('/usr/bin/ssh', ['-oStrictHostKeyChecking=no', '-p39657',
        'root@connect.bjb1.seetacloud.com', cmd],
        encoding='utf-8', timeout=15)
    c.expect('password:', timeout=10)
    c.sendline(password)
    c.expect(pexpect.EOF, timeout=15)
    out = (c.before or '').strip()
    if out:
        print(f"  {cmd[:50]}: {out[:200]}")
    time.sleep(0.5)

import time
