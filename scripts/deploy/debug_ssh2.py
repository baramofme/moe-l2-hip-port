#!/tmp/ssh_venv/bin/python3
"""Debug SSH connection v2."""
import pexpect
import sys

HOST = "connect.bjb1.seetacloud.com"
PORT = 39657
USER = "root"
PASSWORD = "SEhHehEHGy4X"

child = pexpect.spawn(
    f'ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -p {PORT} {USER}@{HOST}',
    encoding='utf-8',
    timeout=30,
)

try:
    i = child.expect(['password:', '(yes/no)', pexpect.EOF, pexpect.TIMEOUT], timeout=30)
    print(f"MATCHED index={i}")
    print(f"BEFORE len={len(child.before)}", flush=True)
    
    if i == 0:
        child.sendline(PASSWORD)
        
        # Wait for shell prompt - try several patterns
        j = child.expect(['[#$] ', '\\$ ', 'root@', pexpect.EOF, pexpect.TIMEOUT], timeout=20)
        print(f"AFTER PWD: index={j}")
        print(f"BEFORE: {repr(child.before[:500])}")
        print(f"AFTER: {repr(child.after)}")
        
        if j < 3:  # got a prompt
            child.sendline('id')
            k = child.expect(['[#$] ', '\\$ ', pexpect.EOF, pexpect.TIMEOUT], timeout=10)
            print(f"\nID CMD: index={k}")
            print(f"OUTPUT: {repr(child.before[:1000])}")
            
            child.sendline('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits')
            k = child.expect(['[#$] ', '\\$ ', pexpect.EOF, pexpect.TIMEOUT], timeout=10)
            print(f"\nNVIDIA CMD: index={k}")
            print(f"OUTPUT: {repr(child.before[:1000])}")
        else:
            print(f"Login may have failed. EOF/TIMEOUT")
    elif i == 1:
        print("Got yes/no prompt")
        child.sendline('yes')
        child.expect(['password:', pexpect.EOF, pexpect.TIMEOUT], timeout=30)
        child.sendline(PASSWORD)
        child.expect(['[#$] ', pexpect.EOF, pexpect.TIMEOUT], timeout=20)
        print(f"After yes+password: {repr(child.before[:500])}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    child.close()
