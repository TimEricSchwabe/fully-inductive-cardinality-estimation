#!/usr/bin/env python3
"""Submit an experiment by running run-experiment.sh.

The shell script SSHes to a cluster, git pulls, and submits an sbatch job.
This wrapper captures the returned Slurm job ID from the script's stdout.

Configure the shell script via environment variables:
  REMOTE_HOST, REMOTE_REPO, BRANCH, SBATCH_SCRIPT, SBATCH_NODE
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SUBMIT_SCRIPT = SCRIPT_DIR / "run-experiment.sh"

SBATCH_JOB_RE = re.compile(r"Submitted\s+batch\s+job\s+(\d+)")
RAW_JOB_RE = re.compile(r"^(\d+)(?:;.*)?$")


class SubmitError(RuntimeError):
    pass


def extract_job_id(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise SubmitError("Submit script produced no stdout; could not extract job_id.")

    for candidate in reversed(lines):
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict) and "job_id" in payload:
                return str(payload["job_id"])
        except json.JSONDecodeError:
            pass

    for candidate in reversed(lines):
        m = RAW_JOB_RE.match(candidate)
        if m:
            return m.group(1)

    for candidate in reversed(lines):
        m = SBATCH_JOB_RE.search(candidate)
        if m:
            return m.group(1)

    raise SubmitError(
        "Could not extract job_id from submit script stdout. "
        "Expected raw job id, 'Submitted batch job <id>', or JSON with a job_id field.\n"
        f"Stdout was:\n{stdout}"
    )


def run_submit_script() -> tuple[str, str, str]:
    if not SUBMIT_SCRIPT.exists():
        raise SubmitError(f"Script does not exist: {SUBMIT_SCRIPT}")

    cmd = ["bash", str(SUBMIT_SCRIPT)]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)

    if proc.returncode != 0:
        raise SubmitError(
            f"Submit script failed with exit code {proc.returncode}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stdout:\n{proc.stdout}\n"
            f"Stderr:\n{proc.stderr}"
        )

    job_id = extract_job_id(proc.stdout)
    return job_id, proc.stdout, proc.stderr


def main() -> int:
    try:
        job_id, raw_stdout, raw_stderr = run_submit_script()
    except SubmitError as e:
        print(str(e), file=sys.stderr)
        return 1

    payload = {
        "job_id": job_id,
        "submit_script": str(SUBMIT_SCRIPT),
        "submit_stdout": raw_stdout,
        "submit_stderr": raw_stderr,
    }

    print(f"Submitted Slurm job {job_id}", file=sys.stderr)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
