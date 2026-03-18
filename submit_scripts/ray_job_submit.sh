#!/usr/bin/env bash
set -x

# 所有节点 link + 启动 vllm
# bash submit_scripts/ray_job_submit.sh python submit_scripts/serve_multi_nodes.py bash submit_scripts/vllm_serve_multi_nodes.sh

# 单机 link + 执行命令（原有用法不变）
# bash submit_scripts/ray_job_submit.sh bash submit_scripts/vllm_serve_multi_nodes.sh

ray job submit --no-wait --runtime-env=submit_scripts/runtime_env.yaml -- \
    python submit_scripts/entrypoint.py "$@"
