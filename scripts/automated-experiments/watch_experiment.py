#!/usr/bin/env python3
"""Watch a Slurm job and the corresponding W&B run until terminal completion.

Usage:
  python watch_experiment.py <job_id> <wandb_run_id>

Design:
  - Slurm is used as the source of truth before the W&B run exists or if the job
    fails before wandb.init() completes.
  - W&B is used as the source of truth for experiment completion and result retrieval
    once the run exists.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import wandb

REMOTE_HOST = "coma"
WANDB_ENTITY = "fice"
WANDB_PROJECT = "fice"

POLL_PENDING = 45
POLL_RUNNING = 15
GRACE_AFTER_SLURM_COMPLETED = 60

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

TERMINAL_WANDB_STATES = {"finished", "failed", "crashed", "killed", "preempted"}


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


def get_wandb_run(run_id: str, timeout: int = 30):
    api = wandb.Api(timeout=timeout)
    api.flush()
    try:
        return api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    except Exception:
        return None


def extract_best_metrics(run: Any) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for row in run.scan_history(page_size=500):
        for key in METRICS_TO_EXTRACT:
            val = row.get(key)
            if val is not None:
                best[key] = min(best.get(key, float("inf")), val)
    return best


def wait_for_completion(
    *,
    job_id: str,
    run_id: str,
) -> Dict[str, Any]:
    slurm_completed_seen_at: Optional[float] = None

    while True:
        slurm = get_slurm_status(job_id)
        run = get_wandb_run(run_id)

        line = f"slurm={slurm.state}"
        if slurm.expected_start:
            line += f" expected_start={slurm.expected_start}"
        if run is not None:
            line += f" wandb={getattr(run, 'state', 'unknown')}"
        print(line, file=sys.stderr)

        if run is not None:
            run_state = (getattr(run, "state", "") or "").lower()
            if run_state == "running" or run_state == "preempting":
                time.sleep(POLL_RUNNING)
                continue

            if run_state in TERMINAL_WANDB_STATES:
                status = "success" if run_state == "finished" else "failed"
                metrics = extract_best_metrics(run) if status == "success" else {}
                return {
                    "status": status,
                    "message": f"W&B run reached terminal state '{run_state}'.",
                    "job_id": job_id,
                    "slurm": asdict(slurm),
                    "wandb_state": run_state,
                    "wandb_url": getattr(run, "url", None),
                    "best_metrics": metrics,
                }

        state_upper = (slurm.state or "UNKNOWN").upper()

        if state_upper in PENDING_SLURM_STATES:
            time.sleep(POLL_PENDING)
            continue

        if state_upper in RUNNING_SLURM_STATES:
            time.sleep(POLL_RUNNING)
            continue

        if state_upper == "COMPLETED":
            if slurm_completed_seen_at is None:
                slurm_completed_seen_at = time.time()

            if run is not None:
                time.sleep(min(POLL_RUNNING, 10))
                continue

            if time.time() - slurm_completed_seen_at < GRACE_AFTER_SLURM_COMPLETED:
                time.sleep(min(POLL_RUNNING, 10))
                continue

            return {
                "status": "failed",
                "message": "Slurm completed but no W&B run appeared within the grace period.",
                "job_id": job_id,
                "slurm": asdict(slurm),
                "best_metrics": {},
            }

        if state_upper in TERMINAL_SLURM_STATES:
            metrics = {}
            if run is not None:
                run_state = (getattr(run, "state", "") or "").lower()
                if run_state in TERMINAL_WANDB_STATES and run_state == "finished":
                    metrics = extract_best_metrics(run)
                return {
                    "status": "success" if run_state == "finished" else "failed",
                    "message": f"Slurm state '{state_upper}' and W&B state '{run_state}' are terminal.",
                    "job_id": job_id,
                    "slurm": asdict(slurm),
                    "wandb_state": run_state,
                    "wandb_url": getattr(run, "url", None),
                    "best_metrics": metrics,
                }

            return {
                "status": "failed",
                "message": f"Slurm reached terminal state '{state_upper}' before a terminal W&B success was observed.",
                "job_id": job_id,
                "slurm": asdict(slurm),
                "best_metrics": {},
            }

        time.sleep(POLL_RUNNING)


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <job_id> <wandb_run_id>", file=sys.stderr)
        return 1

    job_id = sys.argv[1]
    wandb_run_id = sys.argv[2]

    payload = wait_for_completion(job_id=job_id, run_id=wandb_run_id)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
