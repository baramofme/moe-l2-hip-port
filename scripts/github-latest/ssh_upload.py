#!/tmp/ssh_venv/bin/python3
"""Upload run_bench_v4.sh to cloud server via SSH (key auth) and execute it."""

import pexpect
import base64
import sys
import time

HOST = "connect.bjb1.seetacloud.com"
PORT = 39657
USER = "root"
KEY = "/opt/data/home/.ssh/hermes_autodl"

def ssh_run(cmd, timeout=120):
    child = pexpect.spawn(
        f'ssh -i {KEY} -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -p {PORT} {USER}@{HOST}',
        encoding='utf-8',
        timeout=timeout,
    )
    try:
        child.expect(['[#$] ', pexpect.EOF, pexpect.TIMEOUT], timeout=30)
        
        child.sendline(cmd)
        child.expect(['[#$] ', pexpect.EOF, pexpect.TIMEOUT], timeout=timeout)
        output = child.before or ''
        
        child.sendline('echo "EXIT_CODE=$?"')
        child.expect(['[#$] ', pexpect.EOF, pexpect.TIMEOUT], timeout=10)
        more = child.before or ''
        exit_code = 0
        for line in more.split('\n'):
            if 'EXIT_CODE=' in line:
                exit_code = int(line.split('=')[1].strip())
        
        child.close()
        return output + '\n' + more, exit_code
    except Exception as e:
        try:
            child.close()
        except:
            pass
        return str(e), -1

def upload_file(local_bytes, remote_path):
    b64 = base64.b64encode(local_bytes).decode()
    chunk_size = 800
    chunks = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]
    
    out, rc = ssh_run(f'echo -n > {remote_path}.b64')
    print(f"  Init: rc={rc}")
    
    for i, chunk in enumerate(chunks):
        out, rc = ssh_run(f'echo -n "{chunk}" >> {remote_path}.b64')
        if rc != 0:
            print(f"  Chunk {i}/{len(chunks)} FAILED: rc={rc}")
            return False
    
    out, rc = ssh_run(f'base64 -d {remote_path}.b64 > {remote_path} && chmod +x {remote_path} && rm -f {remote_path}.b64 && ls -la {remote_path}')
    print(f"  Decode: rc={rc}")
    return rc == 0

def main():
    with open('/opt/data/run_bench_v4.sh', 'rb') as f:
        script_data = f.read()
    print(f"Script size: {len(script_data)} bytes")
    
    # Upload
    print("Uploading run_bench_v4.sh...", flush=True)
    ok = upload_file(script_data, '/root/run_bench_v4.sh')
    if not ok:
        print("FAILED to upload", flush=True)
        sys.exit(1)
    print("Upload OK", flush=True)
    
    # Verify
    out, rc = ssh_run('head -5 /root/run_bench_v4.sh')
    print(f"Verify (first 5 lines): {out[:200]}", flush=True)
    
    # Check GPU pre-state
    out, rc = ssh_run('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null')
    print(f"GPU before: {out.strip()} MiB", flush=True)
    
    # Launch benchmark in background
    print("\nLaunching benchmark (nohup)...", flush=True)
    out, rc = ssh_run(
        'nohup bash /root/run_bench_v4.sh > /tmp/bench_run_v4.log 2>&1 & echo "PID=$!"',
        timeout=15
    )
    print(f"Launch: {out[:200]}", flush=True)
    
    # Get PID
    pid = None
    for line in out.split('\n'):
        if 'PID=' in line:
            pid = line.split('=')[1].strip()
    print(f"PID={pid}", flush=True)
    
    # Wait and monitor
    print("\nMonitoring (checking every 30s)...", flush=True)
    for i in range(20):
        time.sleep(30)
        
        # Check if alive
        if pid:
            out, rc = ssh_run(f'kill -0 {pid} 2>/dev/null && echo "ALIVE" || echo "DONE"', timeout=10)
        else:
            out, rc = ssh_run('pgrep -f run_bench_v4 && echo "ALIVE" || echo "DONE"', timeout=10)
        
        is_alive = "ALIVE" in out
        
        # Progress snapshot
        out2, _ = ssh_run('tail -3 /tmp/bench_results_v4.txt 2>/dev/null || echo "NO_RESULTS"', timeout=10)
        out3, _ = ssh_run('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null', timeout=10)
        
        print(f"  [{i*30}s] Alive={is_alive} VRAM={out3.strip()}MiB Progress: {out2[:150]}", flush=True)
        
        if not is_alive:
            print("Benchmark finished!", flush=True)
            break
    
    # Get final results
    print("\n=== FINAL RESULTS ===", flush=True)
    out, rc = ssh_run('cat /tmp/bench_results_v4.txt', timeout=15)
    print(out, flush=True)
    
    # Get log tail
    print("\n=== LOG TAIL ===", flush=True)
    out, rc = ssh_run('tail -30 /tmp/bench_run_v4.log 2>/dev/null', timeout=10)
    print(out, flush=True)

if __name__ == '__main__':
    main()
