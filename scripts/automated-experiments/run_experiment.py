#!/usr/bin/env python3
"""End-to-end experiment orchestrator.

Commits + pushes pending changes, submits the Slurm job, watches until
completion, and prints the best metrics extracted from CSV files on the server.

Usage:
  python run_experiment.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_commit_and_push() -> None:
    """Stage all changes, commit (skip if nothing to commit), and push."""
    subprocess.run(["git", "add", "-u"], cwd=REPO_ROOT, check=True)

    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        subprocess.run(
            ["git", "commit", "-m", "experiment"],
            cwd=REPO_ROOT,
            check=True,
        )

    subprocess.run(["git", "push"], cwd=REPO_ROOT, check=True)


def main() -> int:
    from submit_experiment import run_submit_script, SubmitError
    from watch_experiment import wait_for_completion

    try:
        git_commit_and_push()
    except subprocess.CalledProcessError as e:
        print(f"Git commit/push failed: {e}", file=sys.stderr)
        return 1
    print("Committed and pushed.", file=sys.stderr)

    try:
        job_id, _, _ = run_submit_script()
    except SubmitError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"Submitted Slurm job {job_id}", file=sys.stderr)

    print(f"Watching job_id={job_id} ...", file=sys.stderr)
    payload = wait_for_completion(job_id=job_id)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
