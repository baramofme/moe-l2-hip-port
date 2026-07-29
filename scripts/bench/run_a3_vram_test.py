import pexpect, time
pw = 'VhcV9y0QIyE+'

# Connect via SSH and run the whole thing
c = pexpect.spawn('/usr/bin/ssh', ['-oStrictHostKeyChecking=no', '-p39657', 'root@connect.bjb1.seetacloud.com'],
    encoding='utf-8', timeout=180, maxread=100000)
c.expect('password:', timeout=10)
c.sendline(pw)
c.expect('root@', timeout=10)

# Clean
c.sendline('pkill -9 -f llama-cli 2>/dev/null; python3 -c "import ctypes; lib=ctypes.CDLL(\"libcuda.so.1\"); lib.cuInit(0); lib.cuDeviceReset(0)" 2>&1')
c.expect('root@', timeout=10)

# Run VRAM sampling script
c.sendline('''VRAM_LOG=/tmp/vram_a3_log.txt; >$VRAM_LOG; \
GGML_CUDA_FORCE_CPU_EXPERTS=1 /root/llama.cpp/build/bin/llama-cli \
  -m /root/autodl-tmp/DeepSeek-V2-Lite-Chat-Uncensored.Q2_K.gguf \
  -p "Hello" -n 50 --cpu-moe --expert-cache 0 -ngl 99 \
  --no-warmup -np 1 2>/tmp/a3_run_out.txt & \
PID=$!; \
while kill -0 $PID 2>/dev/null; do \
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $VRAM_LOG; \
  sleep 0.3; \
done; \
wait $PID; EC=$?; \
echo "EXIT=$EC"; \
PEAK=$(sort -n $VRAM_LOG | tail -1); \
echo "PEAK=$PEAK"; \
echo "ALL_SAMPLES:"; \
cat $VRAM_LOG | sort -n | uniq -c | sort -rn | head -10; \
echo "OUTPUT:"; \
cat /tmp/a3_run_out.txt''')

try:
    idx = c.expect(['root@', pexpect.TIMEOUT], timeout=180)
    out = c.before or ''
    # Show last 3000 chars
    print(out[-3000:])
except pexpect.EOF:
    print("EOF:", (c.before or '')[-3000:])
except:
    print("Exception, partial:", (c.before or '')[-2000:])

c.sendline('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits')
try:
    c.expect('root@', timeout=5)
    print("FINAL VRAM:", (c.before or '').strip()[-20:])
except:
    pass
c.close()
