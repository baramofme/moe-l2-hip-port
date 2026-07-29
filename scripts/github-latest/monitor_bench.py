#!/tmp/ssh_venv/bin/python3
"""Monitor benchmark progress on cloud server."""
import pexpect
import time
import sys

HOST = "connect.bjb1.seetacloud.com"
PORT = 39657
USER = "root"
KEY = "/opt/data/home/.ssh/hermes_autodl"

def ssh_cmd(cmd, timeout=15):
    child = pexpect.spawn(
        f'ssh -i {KEY} -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -p {PORT} {USER}@{HOST}',
        encoding='utf-8', timeout=timeout,
    )
    try:
        child.expect(['[#$] ', pexpect.EOF, pexpect.TIMEOUT], timeout=20)
        child.sendline(cmd)
        child.expect(['[#$] ', pexpect.EOF, pexpect.TIMEOUT], timeout=timeout)
        out = child.before or ''
        child.close()
        return out
    except:
        try: child.close()
        except: pass
        return ''

# Wait a few seconds for the benchmark to start
time.sleep(5)

print("Monitoring benchmark (PID=13090)...")
print("=" * 60)

for i in range(20):  # up to ~10 minutes
    time.sleep(25)
    
    # Check alive
    alive = ssh_cmd('kill -0 13090 2>/dev/null && echo "ALIVE" || echo "DONE"', timeout=10)
    is_alive = "ALIVE" in alive
    
    # Get last result line
    results = ssh_cmd('tail -1 /tmp/bench_results_v4.txt 2>/dev/null', timeout=10)
    
    # Get GPU status
    vram = ssh_cmd('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null', timeout=10)
    
    # Get which test is running
    proc_info = ssh_cmd('ps aux | grep llama-cli | grep -v grep | head -1 | awk \'{print $NF}\'', timeout=10)
    
    timestamp = time.strftime('%H:%M:%S')
    status = f"[{timestamp}] VRAM={vram.strip()}MiB"
    if proc_info.strip():
        status += f" Running: ...--expert-cache {proc_info.strip()}"
    else:
        status += " (between tests)"
    
    print(f"  {status}", flush=True)
    
    if results.strip():
        print(f"  -> {results.strip()[:120]}", flush=True)
    
    if not is_alive:
        print("\nBenchmark finished! Getting final results...", flush=True)
        final = ssh_cmd('cat /tmp/bench_results_v4.txt', timeout=15)
        print("\n=== FINAL RESULTS ===")
        print(final)
        
        log = ssh_cmd('tail -30 /tmp/bench_run_v4.log 2>/dev/null', timeout=10)
        print("\n=== LOG TAIL ===")
        print(log)
        break
else:
    print("\nMonitoring timeout (10 min). Checking final state...")
    final = ssh_cmd('tail -20 /tmp/bench_results_v4.txt 2>/dev/null', timeout=10)
    print(f"Partial results:\n{final}")
