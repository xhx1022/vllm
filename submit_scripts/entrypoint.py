# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ray environment setup: symlinks compiled .so and missing subdirs from remote NFS.

Used as:
- worker_process_setup_hook: "submit_scripts.entrypoint.setup"
- driver entrypoint: python submit_scripts/entrypoint.py <cmd> [args...]
"""

import os
import sys

REMOTE_ROOT = "/apdcephfs_gy6/share_304153846/hunyuan/arlenxu/dev/projects/xhx/vllm"
REMOTE_VLLM = os.path.join(REMOTE_ROOT, "vllm")


def _symlink_missing(remote_dir, local_dir):
    for entry in os.listdir(remote_dir):
        dst = os.path.join(local_dir, entry)
        if not os.path.exists(dst):
            os.symlink(os.path.join(remote_dir, entry), dst)


def setup():
    candidates = [
        os.path.join(os.getcwd(), "vllm"),
        os.path.join(os.getcwd(), "third_party", "vllm", "vllm"),
    ]
    local_vllm = next((p for p in candidates if os.path.isdir(p)), None)
    if local_vllm is None:
        return

    _symlink_missing(REMOTE_VLLM, local_vllm)

    flash_attn_dir = os.path.join(local_vllm, "vllm_flash_attn")
    if os.path.isdir(flash_attn_dir):
        _symlink_missing(os.path.join(REMOTE_VLLM, "vllm_flash_attn"), flash_attn_dir)

    machete_gen = os.path.join(
        os.getcwd(), "csrc", "quantization", "machete", "generated"
    )
    if os.path.isdir(machete_gen):
        _symlink_missing(
            os.path.join(REMOTE_ROOT, "csrc", "quantization", "machete", "generated"),
            machete_gen,
        )


if __name__ == "__main__":
    setup()
    os.execvp(sys.argv[1], sys.argv[1:])
