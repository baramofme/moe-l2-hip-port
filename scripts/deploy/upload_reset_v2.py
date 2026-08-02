#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════╗
# ║  INTERNAL DEPLOY SCRIPT                              ║
# ║  Requires SSH access to AutoDL GPU instance.         ║
# ║  HOST/PORT/USER are hardcoded. Edit before use.      ║
# ║  Also edits local file path (/opt/data/reset_gpu.py). ║
# ╚══════════════════════════════════════════════════════╝
# ⚠️ DEPRECATED (2026-07-29): 一次性部署脚本，SSH 端口已失效。
#
"""SCP upload using paramiko-style approach, but with shell pipe.
Encode file as base64, SSH in, decode on remote side."""
import pexpect, sys, base64

with open('/opt/data/reset_gpu.py', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

cmd = f'echo {b64} | base64 -d > /root/reset_gpu.py && echo DONE'

child = pexpect.spawn(
    '/usr/bin/ssh',
    ['-o', 'StrictHostKeyChecking=no', '-p', '39657',
     'root@connect.bjb1.seetacloud.com', cmd],
    encoding='utf-8',
    timeout=30
)
child.expect('password:', timeout=10)
child.sendline(sys.argv[1])
child.expect(pexpect.EOF, timeout=30)
out = child.before[-300:] if child.before else ''
print(out)
print("exit:", child.exitstatus)
