#!/usr/bin/env bash
set -x

PIDS=()

cleanup() {
    # SIGTERM to entire process group (shell + vllm serve + their worker children)
    kill -- -$$ 2>/dev/null
    # Wait up to 15s for graceful shutdown
    for i in $(seq 1 15); do
        all_dead=true
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                all_dead=false
                break
            fi
        done
        if $all_dead; then
            break
        fi
        sleep 1
    done
    # Force kill anything still alive in the process group
    kill -9 -- -$$ 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT SIGTERM SIGINT SIGHUP

# /apdcephfs_gy6/share_304153846/hunyuan/arlenxu/dev/models/
# Qwen3-30B-A3B Qwen3.5-27B Llama-3.1-70B-Instruct OpenSWE-72B
MODEL_PATH="${MODEL_PATH:-/dockerdata/OpenSWE-72B}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
TP_SIZE="${TP_SIZE:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}" # hermes llama3_json
NUM_GPUS="${NUM_GPUS:-8}"

ARGS=()
if [ "${ENFORCE_EAGER}" = "true" ]; then
    ARGS+=(--enforce-eager)
fi

DP_SIZE="${DP_SIZE:-2}"
HALF=$((NUM_GPUS / DP_SIZE))
GPUS_INSTANCE_0=$(seq -s, 0 $((HALF - 1)))
GPUS_INSTANCE_1=$(seq -s, "${HALF}" $((NUM_GPUS - 1)))

CUDA_VISIBLE_DEVICES="${GPUS_INSTANCE_0}" \
vllm serve "${MODEL_PATH}" \
    "${ARGS[@]}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --enable-prompt-tokens-details \
    --host 0.0.0.0 \
    --port 8000 &
PIDS+=($!)

CUDA_VISIBLE_DEVICES="${GPUS_INSTANCE_1}" \
vllm serve "${MODEL_PATH}" \
    "${ARGS[@]}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --enable-prompt-tokens-details \
    --host 0.0.0.0 \
    --port 8001 &
PIDS+=($!)

wait
