#!/usr/bin/env python3
"""Watch a Slurm job until completion, then extract metrics from CSV files.

Usage:
  python watch_experiment.py <job_id>

Polls Slurm via SSH until the job reaches a terminal state. On success,
extracts the run directory from the Slurm log and reads the best metrics
from the CSV files written by train.py.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

REMOTE_HOST = "coma"
REMOTE_PROJECT_DIR = "fully-inductive-cardinality-estimation"

POLL_PENDING = 45
POLL_RUNNING = 15

METRICS_TO_EXTRACT = [
    "test/overall/loss",
    "test/overall/q_error",
    "test/overall/median_q_error",
    "test/per_graph/dbpedia100k/loss",
    "test/per_graph/dbpedia100k/q_error",
    "test/per_graph/dbpedia100k/median_q_error",
    "test/per_graph/wn18rr_v4_ind/loss",
    "test/per_graph/wn18rr_v4_ind/q_error",
    "test/per_graph/wn18rr_v4_ind/median_q_error",
]


@dataclass
class SlurmStatus:
    state: str
    expected_start: Optional[str] = None
    exit_code: Optional[str] = None
    raw: Optional[str] = None


TERMINAL_SLURM_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}

PENDING_SLURM_STATES = {"PENDING", "PD", "CONFIGURING", "CF", "RESV_DEL_HOLD", "REQUEUE_HOLD"}
RUNNING_SLURM_STATES = {"RUNNING", "R", "COMPLETING", "CG", "STAGE_OUT", "S", "SUSPENDED"}


class WatchError(RuntimeError):
    pass


def run_local(cmd: list[str], timeout: int = 60) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise WatchError(
            f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def ssh(remote_cmd: str, timeout: int = 60) -> str:
    return run_local(["ssh", REMOTE_HOST, remote_cmd], timeout=timeout)


def get_slurm_status(job_id: str) -> SlurmStatus:
    quoted_job = shlex.quote(job_id)

    squeue_out = ssh(f"squeue -h -j {quoted_job} -o '%T|%S' || true")
    if squeue_out:
        first = squeue_out.splitlines()[0].strip()
        parts = first.split("|", 1)
        state = parts[0].strip() if parts else "UNKNOWN"
        expected_start = parts[1].strip() if len(parts) > 1 and parts[1].strip() not in {"N/A", "Unknown"} else None
        return SlurmStatus(state=state, expected_start=expected_start, raw=first)

    sacct_cmd = (
        "sacct -n -P -j {job} --format=JobIDRaw,State,ExitCode "
        "| awk -F'|' '$1==\"{job}\" {{print $2 \"|\" $3; exit}}' || true"
    ).format(job=job_id)
    sacct_out = ssh(sacct_cmd)
    if sacct_out:
        parts = sacct_out.split("|", 1)
        state = parts[0].strip() if parts else "UNKNOWN"
        exit_code = parts[1].strip() if len(parts) > 1 else None
        return SlurmStatus(state=state, exit_code=exit_code, raw=sacct_out)

    return SlurmStatus(state="UNKNOWN")


# ---------------------------------------------------------------------------
# CSV-based metric extraction
# ---------------------------------------------------------------------------

def extract_run_dir(job_id: str) -> Optional[str]:
    """Parse the Slurm log on the server to find the run directory."""
    quoted = shlex.quote(job_id)
    try:
        out = ssh(
            f"grep -m1 'Run directory:' {REMOTE_PROJECT_DIR}/slurm-{quoted}.out || true"
        )
    except WatchError:
        return None
    if not out:
        return None
    m = re.search(r"Run directory:\s*(\S+)", out)
    if not m:
        return None
    run_dir = m.group(1)
    if not run_dir.startswith("/"):
        run_dir = f"{REMOTE_PROJECT_DIR}/{run_dir}"
    return run_dir


def _read_remote_csv(path: str) -> list[dict]:
    """SSH cat a CSV file and parse it into a list of dicts."""
    quoted = shlex.quote(path)
    try:
        content = ssh(f"cat {quoted}")
    except WatchError:
        return []
    if not content:
        return []
    reader = csv.DictReader(io.StringIO(content))
    return list(reader)


def extract_best_metrics(run_dir: str) -> Dict[str, float]:
    """Read CSV files from the run directory and compute the minimum for each metric."""
    log_dir = f"{run_dir}/logs"

    overall_rows = _read_remote_csv(f"{log_dir}/test_training_metrics.csv")
    per_graph_rows = _read_remote_csv(f"{log_dir}/test_metrics_per_graph.csv")

    best: Dict[str, float] = {}

    for key in METRICS_TO_EXTRACT:
        parts = key.split("/")

        if parts[1] == "overall":
            col = parts[2]
            for row in overall_rows:
                val = row.get(col)
                if val is not None and val != "":
                    best[key] = min(best.get(key, float("inf")), float(val))

        elif parts[1] == "per_graph":
            dataset_name = parts[2]
            col = parts[3]
            for row in per_graph_rows:
                if row.get("dataset_name") != dataset_name:
                    continue
                val = row.get(col)
                if val is not None and val != "":
                    best[key] = min(best.get(key, float("inf")), float(val))

    return best


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------

def wait_for_completion(*, job_id: str) -> Dict[str, Any]:
    while True:
        slurm = get_slurm_status(job_id)

        line = f"slurm={slurm.state}"
        if slurm.expected_start:
            line += f" expected_start={slurm.expected_start}"
        print(line, file=sys.stderr)

        state_upper = (slurm.state or "UNKNOWN").upper()

        if state_upper in PENDING_SLURM_STATES:
            time.sleep(POLL_PENDING)
            continue

        if state_upper in RUNNING_SLURM_STATES:
            time.sleep(POLL_RUNNING)
            continue

        if state_upper == "COMPLETED":
            run_dir = extract_run_dir(job_id)
            metrics = extract_best_metrics(run_dir) if run_dir else {}
            return {
                "status": "success",
                "message": "Slurm job completed.",
                "job_id": job_id,
                "slurm": asdict(slurm),
                "run_dir": run_dir,
                "best_metrics": metrics,
            }

        if state_upper in TERMINAL_SLURM_STATES:
            return {
                "status": "failed",
                "message": f"Slurm reached terminal state '{state_upper}'.",
                "job_id": job_id,
                "slurm": asdict(slurm),
                "best_metrics": {},
            }

        time.sleep(POLL_RUNNING)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <job_id>", file=sys.stderr)
        return 1

    job_id = sys.argv[1]
    payload = wait_for_completion(job_id=job_id)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
