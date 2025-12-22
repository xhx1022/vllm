#!/usr/bin/env bash

set -e
set -o pipefail

VLLM_PATH=$(pip show vllm | awk -F': ' '/^Location:/ {print $2"/vllm"}')

echo "vLLM path: ${VLLM_PATH}"

echo "[1/4] Installing vLLM in editable mode with precompiled extensions..."
VLLM_USE_PRECOMPILED=1 pip install --editable .

echo "[2/4] Copying vllm .so files..."
cp "${VLLM_PATH}"/*.so vllm/

echo "[3/4] Copying vllm_flash_attn .so files..."
cp ${VLLM_PATH}/vllm_flash_attn/*.so vllm/vllm_flash_attn/

echo "[4/4] Copying flash_attn_interface.py..."
cp ${VLLM_PATH}/vllm_flash_attn/flash_attn_interface.py \
   vllm/vllm_flash_attn/flash_attn_interface.py

echo "All steps completed successfully."
