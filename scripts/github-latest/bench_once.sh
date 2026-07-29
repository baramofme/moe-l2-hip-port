#!/usr/bin/env bash
set -euo pipefail

MODEL=/root/autodl-tmp/DeepSeek-V2-Lite-Chat-Uncensored.Q2_K.gguf
BIN=/root/llama.cpp/build/bin/llama-cli
ARGS="-m $MODEL -p 'Hello' -n 50 --cpu-moe -ngl 99 -c 512 --single-turn"

test_run() {
    local label="$1" env_vars="$2"
    local peak_file=$(mktemp)
    echo 0 > "$peak_file"

    (
        while true; do
            cur=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
            saved=$(cat "$peak_file")
            [ "$cur" -gt "$saved" ] && echo "$cur" > "$peak_file"
            sleep 0.2
        done
    ) &
    local samp=$!

    echo "=== $label ==="
    env $env_vars timeout 40 $BIN $ARGS 2>&1 | grep -E 'Prompt:|Generation:|total'

    kill $samp 2>/dev/null; wait $samp 2>/dev/null; sleep 1
    echo "Peak VRAM: $(cat "$peak_file") MiB"
    echo ""
    rm -f "$peak_file"
}

test_run "no cache" "GGML_CUDA_FORCE_CPU_EXPERTS=1"
test_run "cache=1" "GGML_CUDA_FORCE_CPU_EXPERTS=1 GGML_CUDA_EXPERT_CACHE=1"

echo "DONE"
