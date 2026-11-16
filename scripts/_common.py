"""Shared by the scripts: put the repo root on sys.path, run from the repo root, check that the GPU is idle."""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)


def gpu_busy():
    """Return a reason string if another job seems to be using the GPU, else None.

    Utilization is averaged over a short window and lingers right after a previous run ends, so the GPU counts as
    busy only if it looks busy on 5 checks 2 s apart."""
    q = lambda *a: subprocess.run(
        ["nvidia-smi", *a], capture_output=True, text=True
    ).stdout.strip()
    reason = None
    for _ in range(5):
        apps = [
            l
            for l in q(
                "--query-compute-apps=pid,used_memory", "--format=csv,noheader"
            ).splitlines()
            if l.strip()
        ]
        used = int(
            q("--query-gpu=memory.used", "--format=csv,noheader,nounits").splitlines()[
                0
            ]
        )
        util = int(
            q(
                "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"
            ).splitlines()[0]
        )
        if not (apps or used > 1500 or util > 30):
            return None
        reason = f"compute processes {apps}, {used} MiB used, {util}% utilization"
        time.sleep(2)
    return reason
