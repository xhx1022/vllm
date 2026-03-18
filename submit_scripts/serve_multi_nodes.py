"""Dispatch vLLM serving script to every node in the Ray cluster.

Each node will first run symlink setup (entrypoint.setup), then execute the
given shell command (typically vllm_serve_multi_nodes.sh).

Usage:
    python submit_scripts/serve_multi_nodes.py bash submit_scripts/vllm_serve_multi_nodes.sh
"""

import ctypes
import ctypes.util
import os
import signal
import subprocess
import sys

import ray

from submit_scripts.entrypoint import setup


def _set_pdeathsig():
    """Request SIGTERM when parent process dies (Linux only).

    Acts as a safety net: if the Ray worker is killed with SIGKILL, the child
    subprocess automatically receives SIGTERM instead of becoming an orphan.
    """
    try:
        libc_name = ctypes.util.find_library("c") or "libc.so.6"
        libc = ctypes.CDLL(libc_name, use_errno=True)
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG = 1
    except Exception:
        pass


@ray.remote(max_concurrency=2)
class _NodeRunner:
    def __init__(self):
        self.proc = None

    def run(self, cmd):
        setup()
        self.proc = subprocess.Popen(
            cmd, start_new_session=True, preexec_fn=_set_pdeathsig
        )
        rc = self.proc.wait()
        if rc not in (0, -signal.SIGTERM, -signal.SIGKILL):
            raise RuntimeError(f"[{os.uname().nodename}] rc={rc}")

    def stop(self):
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def _stop_all(runners):
    futs = []
    for r in runners:
        try:
            futs.append(r.stop.remote())
        except Exception:
            pass
    for f in futs:
        try:
            ray.get(f, timeout=30)
        except Exception:
            pass


def main():
    ray.init(address="auto")
    nodes = [n for n in ray.nodes() if n["Alive"]]
    print(f"Dispatching to {len(nodes)} node(s)")

    runners = []
    tasks = []
    for n in nodes:
        runner = _NodeRunner.options(
            resources={f"node:{n['NodeManagerAddress']}": 0.01}
        ).remote()
        runners.append(runner)
        tasks.append(runner.run.remote(sys.argv[1:]))

    def _shutdown(signum, frame):
        _stop_all(runners)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        ray.get(tasks)
    except (KeyboardInterrupt, SystemExit):
        _stop_all(runners)
        raise
    except Exception:
        _stop_all(runners)
        raise


if __name__ == "__main__":
    main()
