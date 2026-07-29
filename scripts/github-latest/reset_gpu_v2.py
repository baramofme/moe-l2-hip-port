#!/usr/bin/env python3
"""Reset GPU via cudaDeviceReset on remote server and check VRAM."""
import pexpect, sys, base64

password = sys.argv[1]

code = '''
import ctypes, time, subprocess
cudart = ctypes.CDLL("libcudart.so")
cudart.cudaSetDevice(0)
r1 = cudart.cudaDeviceReset()
time.sleep(1)
r2 = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
print(f"cudaDeviceReset: {r1}")
print(f"nvidia-smi: {r2.stdout.strip()}")
'''

b64 = base64.b64encode(code.encode()).decode()
cmd = f'echo {b64} | base64 -d | python3'

child = pexpect.spawn('/usr/bin/ssh',
    ['-oStrictHostKeyChecking=no', '-p39657', 'root@connect.bjb1.seetacloud.com', cmd],
    encoding='utf-8', timeout=20)
child.expect('password:', timeout=10)
child.sendline(password)
child.expect(pexpect.EOF, timeout=20)
print(child.before or '')
print("exit:", child.exitstatus)
