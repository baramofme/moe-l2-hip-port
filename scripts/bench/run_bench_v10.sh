#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════╗
# ║  INTERNAL EXPERIMENT SCRIPT                          ║
# ║  Paths (model files, llama.cpp binary) are           ║
# ║  hardcoded to the author's environment.              ║
# ║  Edit MODEL_*, LLAMA_CLI, RESULTS_FILE before use.   ║
# ╚══════════════════════════════════════════════════════╝
#
# MoE host-buffer + sched-cache Benchmark v10 — 2026-08-02 架构
# Tests Qwen3.6 IQ2_M + DS-V2-Lite Q2_K across cache levels.
# host-buffer 形态：默认 mmap + GGML_OP_OFFLOAD_MIN_BATCH=1（专家 GPU 直算），
# GGML_CUDA_EXPERT_CACHE 挂在 sched 拷贝层（0.25 对 DS 最优）。

set -euo pipefail

LLAMA_CLI="/root/llama.cpp/build/bin/llama-cli"
MODEL_QWEN="/root/autodl-tmp/Qwen3.6-35B-A3B-UD-IQ2_M.gguf"
MODEL_DS="/root/autodl-tmp/DeepSeek-V2-Lite-Chat-Uncensored.Q2_K.gguf"
RESULTS_FILE="/tmp/bench_results_v10.txt"
FOLLOWUP_FILE="/tmp/conv_followup.txt"

# Ensure followup file exists
if [ ! -f "$FOLLOWUP_FILE" ]; then
    echo "What is the capital of France?" > /tmp/conv_followup.txt
    echo "Paris" >> /tmp/conv_followup.txt
    echo "And what is the population of France?" >> /tmp/conv_followup.txt
fi

reset_cuda() {
    python3 -c "
import ctypes
try:
    lib = ctypes.CDLL('libcuda.so.1')
    if lib.cuInit(0) == 0:
        lib.cuDeviceReset(0)
except: pass
" 2>/dev/null || true
    sleep 2
}

get_vram() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

vram_peak=0
peak_file=$(mktemp)
sampler_pid=""

start_vram_sampler() {
    echo 0 > "$peak_file"
    (
        while true; do
            cur=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
            [ -n "$cur" ] && [ "$cur" -gt "$(cat "$peak_file" 2>/dev/null || echo 0)" ] && echo "$cur" > "$peak_file"
            sleep 0.3
        done
    ) &
    sampler_pid=$!
}

stop_vram_sampler() {
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
    sleep 1
}

run_test() {
    local label="$1"
    local model_path="$2"
    local cache_ratio="$3"
    local prompt_type="$4"  # short, long, followup

    local n_tokens=""
    local prompt_args=""
    case "$prompt_type" in
        short)    n_tokens=50;  prompt_args="-p Hello" ;;
        long)     n_tokens=200; prompt_args="-f /tmp/long_prompt.txt" ;;
        followup) n_tokens=50;  prompt_args="-f $FOLLOWUP_FILE" ;;
    esac

    reset_cuda
    local pre_vram=$(get_vram)

    # Build env — host-buffer 专家 GPU 直算（2026-08-02 架构）
    local env_vars="GGML_OP_OFFLOAD_MIN_BATCH=1 GGML_CUDA_EXPERT_CACHE=$cache_ratio"

    start_vram_sampler

    echo ">>> [$label|$prompt_type|cache=$cache_ratio]" >&2

    local output
    output=$(timeout 40 env $env_vars $LLAMA_CLI -m "$model_path" \
        $prompt_args -n $n_tokens -ngl 99 -c 512 --no-warmup --single-turn \
        2>&1) || true

    stop_vram_sampler
    local peak_vram=$(cat "$peak_file" 2>/dev/null || echo 0)
    local post_vram=$(get_vram)
    local exit_code=$?

    # Parse speeds
    local prompt_speed=$(echo "$output" | grep -oP 'Prompt: \K[\d.]+' | head -1)
    local gen_speed=$(echo "$output" | grep -oP 'Generation: \K[\d.]+' | head -1)

    # Determine status
    if [ "$exit_code" != "0" ] && [ -z "$prompt_speed" ]; then
        local status="EXIT_${exit_code}"
    else
        local status="PASS"
    fi

    # Write result line
    echo "$label | $prompt_type | $cache_ratio | $status | $pre_vram | $peak_vram | $post_vram | ${prompt_speed:-N/A} | ${gen_speed:-N/A}" >> "$RESULTS_FILE"
    echo "  → $status | VRAM $pre_vram→$peak_vram MiB | Prompt=${prompt_speed:-N/A} t/s | Gen=${gen_speed:-N/A} t/s" >&2
}

# Prepare long prompt
echo "The history of artificial intelligence spans several decades" > /tmp/long_prompt.txt
for i in $(seq 1 10); do
    cat /tmp/long_prompt.txt >> /tmp/long_prompt_tmp.txt 2>/dev/null || true
done

echo "====== MoE LRU Cache Benchmark v10 ======" > "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)" >> "$RESULTS_FILE"
echo "Arch: host-buffer (OFFLOAD_MIN_BATCH=1) + sched-cache" >> "$RESULTS_FILE"
echo "Binary: $(stat --format='%Y' $LLAMA_CLI 2>/dev/null)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "Model | Type | Cache | Status | VRAM_pre | VRAM_peak | VRAM_post | Prompt_t/s | Gen_t/s" >> "$RESULTS_FILE"
echo "------|------|-------|--------|----------|-----------|-----------|------------|--------" >> "$RESULTS_FILE"

# ── Qwen3.6 IQ2_M ──
echo "" >> "$RESULTS_FILE"
echo "========== Qwen3.6-A3B IQ2_M ==========" >> "$RESULTS_FILE"
echo "========== Qwen3.6-A3B IQ2_M ==========" >&2
for cache in 0 0.25 0.5 0.75 1.0; do
    run_test "Qwen" "$MODEL_QWEN" "$cache" "short"
    run_test "Qwen" "$MODEL_QWEN" "$cache" "long"
    run_test "Qwen" "$MODEL_QWEN" "$cache" "followup"
done

# ── DS-V2-Lite Q2_K ──
echo "" >> "$RESULTS_FILE"
echo "========== DeepSeek-V2-Lite Q2_K ==========" >> "$RESULTS_FILE"
echo "========== DeepSeek-V2-Lite Q2_K ==========" >&2
for cache in 0 0.25 0.5 0.75 1.0; do
    run_test "DS-V2-Lite" "$MODEL_DS" "$cache" "short"
    run_test "DS-V2-Lite" "$MODEL_DS" "$cache" "long"
    run_test "DS-V2-Lite" "$MODEL_DS" "$cache" "followup"
done

echo "" >> "$RESULTS_FILE"
echo "====== DONE ======" >> "$RESULTS_FILE"
echo "====== DONE ======" >&2
echo "Results: $RESULTS_FILE" >&2
