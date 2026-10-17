"""Stage bookkeeping for scripts/gpu_run.sh: `status.py set <stage> <ok|fail|skip> <seconds> [note]`
and `status.py env` (environment / provenance snapshot). Written by an LLM agent (Claude Code)."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, ROOT, read_json, write_json  # noqa: E402

PATH = os.path.join(DATA, "status.json")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"error: {e}"


def main():
    cmd = sys.argv[1]
    if cmd == "set":
        st = read_json(PATH, {}) or {}
        stage, state, secs = sys.argv[2], sys.argv[3], float(sys.argv[4])
        note = " ".join(sys.argv[5:])
        st[stage] = dict(state=state, seconds=secs, note=note, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write_json(PATH, st)
    elif cmd == "env":
        sys.path.insert(0, ROOT)
        from arcmlp.eval.local_eval import provenance
        env = dict(provenance=provenance(ROOT), nvidia_smi=sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
                   cuda_version=sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"), nproc=os.cpu_count(),
                   mem_total_gb=(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 2 ** 30,
                   platform=platform.platform(), python=sys.version.split()[0],
                   profile=os.environ.get("PROFILE"), device=os.environ.get("ARCMLP_DEVICE"), argv=os.environ.get("GPU_RUN_ARGS"))
        try:
            from arcmlp.reference.device import device_info
            env["device_info"] = device_info(os.environ.get("ARCMLP_DEVICE", "cpu"))
        except Exception as e:
            env["device_info"] = dict(error=str(e))
        write_json(os.path.join(DATA, "env.json"), env)
        print(json.dumps(env, default=str))


if __name__ == "__main__":
    main()
