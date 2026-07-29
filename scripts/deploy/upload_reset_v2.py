#!/usr/bin/env python3
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
