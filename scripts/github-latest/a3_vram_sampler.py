import pexpect, time, threading
pw = 'VhcV9y0QIyE+'

# 1) Clean + reset
c = pexpect.spawn('/usr/bin/ssh', ['-oStrictHostKeyChecking=no', '-p39657', 'root@connect.bjb1.seetacloud.com',
    'pkill -9 -f llama-cli 2>/dev/null; python3 -c "import ctypes; lib=ctypes.CDLL(\"libcuda.so.1\"); lib.cuInit(0); lib.cuDeviceReset(0)" 2>&1; echo DONE'
], encoding='utf-8', timeout=20)
c.expect('password:', timeout=10)
c.sendline(pw)
c.expect('DONE', timeout=20)
c.close()

# 2) Launch model in background with env var
launch = 'nohup bash -c "GGML_CUDA_FORCE_CPU_EXPERTS=1 /root/llama.cpp/build/bin/llama-cli -m /root/autodl-tmp/DeepSeek-V2-Lite-Chat-Uncensored.Q2_K.gguf -p Hello -n 50 --cpu-moe --expert-cache 0 -ngl 99 --no-warmup -np 1 2>/tmp/a3_out_final.txt" >/dev/null 2>&1 & echo PID=$!'
c2 = pexpect.spawn('/usr/bin/ssh', ['-oStrictHostKeyChecking=no', '-p39657', 'root@connect.bjb1.seetacloud.com',
    launch], encoding='utf-8', timeout=20)
c2.expect('password:', timeout=10)
c2.sendline(pw)
c2.expect(pexpect.EOF, timeout=20)
pid_line = (c2.before or '').strip()
print("Launch result:", pid_line)
c2.close()

# Wait a moment for model to start loading
time.sleep(3)

# 3) Sample VRAM in a separate SSH session
def sample_vram():
    samples = []
    for i in range(30):
        try:
            c = pexpect.spawn('/usr/bin/ssh', ['-oStrictHostKeyChecking=no', '-p39657', 'root@connect.bjb1.seetacloud.com',
                'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits'], encoding='utf-8', timeout=10)
            c.expect('password:', timeout=5)
            c.sendline(pw)
            c.expect(pexpect.EOF, timeout=10)
            v = c.before.strip()
            c.close()
            if v and v.isdigit():
                samples.append(int(v))
        except:
            pass
        time.sleep(0.5)
    return samples

samples = sample_vram()
print(f"VRAM samples: {samples}")
if samples:
    print(f"MIN: {min(samples)} MiB, MAX: {max(samples)} MiB")
    from collections import Counter
    counts = Counter(samples)
    print("Distribution:", counts.most_common(5))

# 4) Wait for model to finish
time.sleep(5)

# 5) Check if still running
c3 = pexpect.spawn('/usr/bin/ssh', ['-oStrictHostKeyChecking=no', '-p39657', 'root@connect.bjb1.seetacloud.com',
    'pgrep llama-cli > /dev/null && echo RUNNING || echo DONE'], encoding='utf-8', timeout=15)
c3.expect('password:', timeout=10)
c3.sendline(pw)
c3.expect(pexpect.EOF, timeout=15)
status = c3.before.strip()
print(f"Status: {status}")

# If still running, wait more
if 'RUNNING' in status:
    time.sleep(15)
    c3b = pexpect.spawn('/usr/bin/ssh', ['-oStrictHostKeyChecking=no', '-p39657', 'root@connect.bjb1.seetacloud.com',
        'pgrep llama-cli > /dev/null && echo RUNNING || echo DONE'], encoding='utf-8', timeout=15)
    c3b.expect('password:', timeout=10)
    c3b.sendline(pw)
    c3b.expect(pexpect.EOF, timeout=15)
    print(f"Status after wait: {c3b.before.strip()}")
    c3b.close()

# 6) Get output and VRAM
c4 = pexpect.spawn('/usr/bin/ssh', ['-oStrictHostKeyChecking=no', '-p39657', 'root@connect.bjb1.seetacloud.com',
    'cat /tmp/a3_out_final.txt; echo "===VRAM==="; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits'], encoding='utf-8', timeout=15)
c4.expect('password:', timeout=10)
c4.sendline(pw)
c4.expect(pexpect.EOF, timeout=15)
print("=== Results ===")
print(c4.before or '')
c4.close()
