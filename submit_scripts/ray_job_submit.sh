#!/usr/bin/env bash
set -x

ray job submit --no-wait --runtime-env=submit_scripts/runtime_env.yaml -- \
    python submit_scripts/entrypoint.py "$@"
