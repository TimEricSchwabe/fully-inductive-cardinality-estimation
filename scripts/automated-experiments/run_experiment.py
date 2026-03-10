#!/usr/bin/env python3
"""End-to-end experiment orchestrator.

Generates a unique W&B run ID, patches it into configs/train_full.yaml,
commits + pushes the change, submits the Slurm job, watches until completion,
and prints the best metrics.

Usage:
  python run_experiment.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "train_full.yaml"


def generate_run_id() -> str:
    return uuid.uuid4().hex[:8]


def patch_wandb_run_id(config_path: Path, run_id: str) -> None:
    """Set wandb_run_id in the YAML config via string replacement (preserves anchors/aliases)."""
    text = config_path.read_text()

    if re.search(r"^wandb_run_id:", text, re.MULTILINE):
        text = re.sub(r"^(wandb_run_id:).*$", rf'\1 "{run_id}"', text, flags=re.MULTILINE)
    else:
        text = re.sub(
            r"^(wandb_run_name:.*)$",
            rf'\1\nwandb_run_id: "{run_id}"',
            text,
            count=1,
            flags=re.MULTILINE,
        )

    config_path.write_text(text)


def git_commit_and_push(config_path: Path, run_id: str) -> None:
    rel = config_path.relative_to(REPO_ROOT)
    subprocess.run(["git", "add", str(rel)], cwd=REPO_ROOT, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Set wandb_run_id={run_id}"],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(["git", "push"], cwd=REPO_ROOT, check=True)


def main() -> int:
    # Lazy imports so the module is importable without these deps at top level.
    from submit_experiment import run_submit_script, SubmitError
    from watch_experiment import wait_for_completion

    run_id = generate_run_id()
    print(f"Generated wandb_run_id: {run_id}", file=sys.stderr)

    patch_wandb_run_id(CONFIG_PATH, run_id)
    print(f"Patched {CONFIG_PATH.relative_to(REPO_ROOT)}", file=sys.stderr)

    try:
        git_commit_and_push(CONFIG_PATH, run_id)
    except subprocess.CalledProcessError as e:
        print(f"Git commit/push failed: {e}", file=sys.stderr)
        return 1
    print("Committed and pushed config change.", file=sys.stderr)

    try:
        job_id, _, _ = run_submit_script()
    except SubmitError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"Submitted Slurm job {job_id}", file=sys.stderr)

    print(f"Watching job_id={job_id} wandb_run_id={run_id} ...", file=sys.stderr)
    payload = wait_for_completion(job_id=job_id, run_id=run_id)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
