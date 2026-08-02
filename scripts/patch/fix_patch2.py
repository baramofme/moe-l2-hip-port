#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════╗
# ║  INTERNAL PATCH SCRIPT                               ║
# ║  Targets a specific llama.cpp fork + commit.         ║
# ║  `/root/llama.cpp/...` path is hardcoded.            ║
# ║  Adjust path and verify code context before use.    ║
# ╚══════════════════════════════════════════════════════╝
# ⚠️ DEPRECATED (2026-07-29): 一次性补丁脚本（fix 已固化进源码）。
#
"""Clean up duplicate A3 cache blocks in ggml-cuda.cu."""
path = '/root/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu'

with open(path, 'r') as f:
    content = f.read()

# Remove the corrupted sed block (contains literal \n and bad quoting)
# Pattern: starts with "// A3 cache: force cuBLAS" followed by corrupted lines
import re

# Remove the corrupted first block (has literal \\n in it)
content = re.sub(
    r'    // A3 cache: force cuBLAS for quantized MUL_MAT so convert\+cache triggers\\\\n    \\{\\\\n        static const int a3_on =.*?\\{\\\\n            return;\n        \\}\\\\n    }\n\n',
    '',
    content,
    flags=re.DOTALL
)

with open(path, 'w') as f:
    f.write(content)

print("Cleaned. Verifying:")
import subprocess
r = subprocess.run(['sed', '-n', '1848,1875p', path], capture_output=True, text=True)
print(r.stdout)
