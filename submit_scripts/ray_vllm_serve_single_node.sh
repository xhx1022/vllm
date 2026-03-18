#!/usr/bin/env bash
set -x
# Qwen3-30B-A3B Qwen3.5-27B Llama-3.1-70B-Instruct
MODEL_PATH="${MODEL_PATH:-/apdcephfs_gy6/share_304153846/hunyuan/arlenxu/dev/models/Qwen3-30B-A3B}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
DP_SIZE="${DP_SIZE:-4}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}" # hermes llama3_json
# ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-deepep_low_latency}"

ARGS=()
if [ "${ENFORCE_EAGER}" = "true" ]; then
    ARGS+=(--enforce-eager)
fi

vllm serve "${MODEL_PATH}" \
    "${ARGS[@]}" \
    --data-parallel-size "${DP_SIZE}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER}" \

    # --chat-template examples/tool_chat_template_llama3.1_json.jinja \
    # --tokenizer-mode slow
