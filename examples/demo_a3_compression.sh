#!/usr/bin/env bash
# ============================================================
# MoE host-buffer VRAM Compression Demo
#
# Demonstrates the same MoE model running in three modes:
#   1. OG mode     — full model on GPU (--no-mmap, no cache)
#   2. moe-l2 mode — host-buffer experts + GPU compute
#                    (default mmap + GGML_OP_OFFLOAD_MIN_BATCH=1)
#   3. cache mode  — mode 2 + sched-cache (GGML_CUDA_EXPERT_CACHE=0.25)
#
# 2026-08-02 architecture: experts live in CPU pinned memory
# (CUDA host buffer, zero VRAM), the scheduler copies ONLY the
# activated experts to GPU each step, GPU computes them on the
# fast path. sched-cache keeps hot experts on GPU (D2D, no PCIe).
#
# Edit MODEL and LLAMA_CLI paths below before running.
# ============================================================
set -uo pipefail

# ---- CONFIG ----
MODEL="/root/autodl-tmp/DeepSeek-V2-Lite-Chat-Uncensored.Q2_K.gguf"
LLAMA_CLI="/root/autodl-tmp/build-a3/bin/llama-batched"
N_GPU_LAYERS=99            # offload all layers to GPU
CACHE_RATIO=0.25           # sched-cache ratio (DS: 0.25 optimal)

PROMPT="Explain the difference between Mixture of Experts (MoE) and dense transformer models. Focus on inference memory usage."

# ---- Pre-flight ----
if [ ! -f "$MODEL" ]; then echo "ERROR: model not found at $MODEL"; exit 1; fi
if [ ! -x "$LLAMA_CLI" ]; then echo "ERROR: llama-batched not found at $LLAMA_CLI"; exit 1; fi

echo "=============================================="
echo "  MoE host-buffer VRAM Compression Demo"
echo "=============================================="
echo "Model : $(basename $MODEL)"
echo "Binary: $(basename $LLAMA_CLI)"
echo "GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "VRAM  : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
echo ""

# ---- Helpers ----
gpu_mem() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
}

run_test() {
    local mode_label="$1" mode="$2" env_args="$3" extra_args="$4" outfile="$5"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $mode_label"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Baseline VRAM (idle, no model loaded)
    BASE_MEM=$(gpu_mem)
    echo "  [idle VRAM: ${BASE_MEM} MB]"

    # Start nvidia-smi memory monitor in background (every 200ms)
    VRAM_LOG="/tmp/vram_${mode}.txt"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --loop-ms=200 2>/dev/null > "$VRAM_LOG" &
    MON_PID=$!

    # Run inference (foreground) — monitor will capture peak VRAM during run
    echo "  [running inference + VRAM monitor...]"
    env $env_args $LLAMA_CLI \
        -m "$MODEL" \
        -p "$PROMPT" \
        -n 128 \
        -ngl $N_GPU_LAYERS \
        $extra_args \
        --temp 0.7 \
        > "$outfile" 2>&1

    # Stop monitor — process just exited, readings after this point are stale
    kill $MON_PID 2>/dev/null || true
    wait $MON_PID 2>/dev/null || true

    # Find peak VRAM from monitor log (skip header reads)
    PEAK_MEM=0
    while IFS= read -r line; do
        line="${line//[!0-9]/}"
        [ -z "$line" ] && continue
        [ "$line" -gt "$PEAK_MEM" ] && PEAK_MEM=$line
    done < "$VRAM_LOG"
    [ "$PEAK_MEM" -eq 0 ] && PEAK_MEM=$(tail -1 "$VRAM_LOG" 2>/dev/null || echo "0")
    # The last reading before kill is still during the run, so usable

    rm -f "$VRAM_LOG"

    # Parse speed from output
    TPS=$(grep -oP 'speed: \K[\d.]+(?= t/s)' "$outfile" | tail -1)
    TOKENS=$(grep -oP '\d+(?= tokens in)' "$outfile" | tail -1 || echo "128")
    DURATION=$(grep -oP 'in \K[\d.]+(?= s, speed:)' "$outfile" | tail -1)
    MEM_USED=$(( PEAK_MEM - BASE_MEM ))

    echo ""
    echo "  ⏱  Duration:  ${DURATION:-N/A}s"
    echo "  ⚡ Tokens/s:  ${TPS:-N/A}"
    echo "  🎯 VRAM used: ${MEM_USED} MB (delta: ${PEAK_MEM} - ${BASE_MEM})"
    echo ""

    # Store in global vars for summary
    declare -g "${mode}_DURATION=${DURATION:-0}"
    declare -g "${mode}_MEM=${MEM_USED:-0}"
    declare -g "${mode}_TPS=${TPS:-N/A}"
    declare -g "${mode}_TOKENS=${TOKENS:-128}"
}

# ============ MODE 1: ORIGINAL (no expert cache) ============
run_test \
    "MODE 1: OG — Full model on GPU" \
    "OG" \
    "" \
    "--no-mmap" \
    "/tmp/og_output.txt"

sleep 2

# ============ MODE 2: moe-l2 host-buffer ============
run_test \
    "MODE 2: moe-l2 — host-buffer experts + GPU compute" \
    "HB" \
    "GGML_OP_OFFLOAD_MIN_BATCH=1" \
    "" \
    "/tmp/hb_output.txt"

sleep 2

# ============ MODE 3: host-buffer + sched-cache ============
run_test \
    "MODE 3: cache — host-buffer + sched-cache (${CACHE_RATIO})" \
    "CACHE" \
    "GGML_OP_OFFLOAD_MIN_BATCH=1 GGML_CUDA_EXPERT_CACHE=${CACHE_RATIO}" \
    "" \
    "/tmp/cache_output.txt"

# ============ RESULTS ============
echo "=============================================="
echo "  RESULTS SUMMARY"
echo "=============================================="
echo ""
printf "  %-12s  %20s  %20s  %20s\n" "" "OG (full GPU)" "host-buffer" "host-buffer+cache"
printf "  %-12s  %20s  %20s  %20s\n" "────────────" "────────────────────" "────────────────────" "────────────────────"
printf "  %-12s  %20s  %20s  %20s\n" "VRAM used" "${OG_MEM:-N/A} MB" "${HB_MEM:-N/A} MB" "${CACHE_MEM:-N/A} MB"
printf "  %-12s  %20s  %20s  %20s\n" "Speed" "${OG_TPS:-N/A} t/s" "${HB_TPS:-N/A} t/s" "${CACHE_TPS:-N/A} t/s"

if [ -n "${HB_MEM:-}" ] && [ "${HB_MEM:-0}" -gt 0 ] && [ -n "${OG_MEM:-}" ] && [ "${OG_MEM:-0}" -gt 0 ]; then
    COMPRESSION=$(echo "scale=2; $OG_MEM / $HB_MEM" | bc -l 2>/dev/null || echo "N/A")
    SAVING=$(( OG_MEM - HB_MEM ))
    printf "  %-12s  %20s  %20s  %20s\n" "Savings" "-" "${SAVING} MB (${COMPRESSION}x)" "-"
fi
echo ""
echo "  OG mode:    --no-mmap loads full model weights to GPU."
echo "  moe-l2:     default mmap + GGML_OP_OFFLOAD_MIN_BATCH=1:"
echo "              experts live in CPU pinned memory (zero VRAM),"
echo "              scheduler copies only activated experts to GPU."
echo "  cache mode: + GGML_CUDA_EXPERT_CACHE=0.25 keeps hot experts"
echo "              on GPU (D2D copy, no PCIe) — DS prompt +211%."
echo ""
echo "  Reference (RTX 4090, DS-V2-Lite Q2_K, 2026-08-02):"
echo "    OG  23.3 GB  65 t/s"
echo "    HB  1.6 GB  37.5 t/s gen (99 prompt)"
echo "    cache 1.6 GB 39.2 t/s gen (308 prompt, +211%)"
echo "=============================================="
