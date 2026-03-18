#!/usr/bin/env bash
set -x

HOST="${BENCH_HOST:-http://127.0.0.1:8000}"
MODEL_PATH="${MODEL_PATH:-/dockerdata/OpenSWE-72B}"
DATASET_NAME="${DATASET_NAME:-sharegpt}"
DATASET_PATH="${DATASET_PATH:-/apdcephfs_gy6/share_304153846/hunyuan/arlenxu/dev/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
REQUEST_RATE="${REQUEST_RATE:-inf}"

vllm bench serve \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --base-url "${HOST}" \
    --model "${MODEL_PATH}" \
    --dataset-name "${DATASET_NAME}" \
    --dataset-path "${DATASET_PATH}" \
    --num-prompts "${NUM_PROMPTS}" \
    --request-rate "${REQUEST_RATE}"

