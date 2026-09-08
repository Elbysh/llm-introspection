#!/usr/bin/env python3
"""Read-only experiment monitor. Run on Ruche after authenticating with SSH."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time


TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                   "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def read_progress(directory):
    try:
        return json.loads((directory / "progress.json").read_text())
    except FileNotFoundError:
        return {}


def run_command(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if result.returncode:
        return None, result.stderr.strip()[:300]
    return result.stdout.strip(), None


def scheduler_status(job_id):
    if not job_id:
        return "", "No Slurm job ID (use --job-id while a job is queued)."
    output, queue_error = run_command(["squeue", "-h", "-j", job_id, "-o", "%i|%T|%M|%l|%R"])
    if output:
        fields = output.splitlines()[0].split("|", 4)
        if len(fields) == 5:
            job, state, elapsed, limit, reason = fields
            return state, f"Job {job}: {state} | elapsed {elapsed} / limit {limit} | {reason}"
        return "", output
    output, error = run_command([
        "sacct", "-n", "-P", "-j", job_id, "--format=JobIDRaw,State%40,Elapsed,Timelimit,ExitCode",
    ])
    if error:
        return "", f"Slurm unavailable: {queue_error or error}"
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) >= 5 and fields[0] == job_id:
            job, state, elapsed, limit, exit_code = fields[:5]
            return state.split()[0].rstrip("+"), (
                f"Job {job}: {state} | elapsed {elapsed} / limit {limit} | exit {exit_code}")
    return "", f"Job {job_id} is not in squeue; accounting may not be available yet."


def duration(seconds):
    if seconds is None:
        return "not available yet"
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s"


def tail_log(path, count):
    if count == 0 or not path.exists():
        return []
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 32768))
        lines = stream.read().decode(errors="replace").splitlines()
    return lines[-count:]


def render(directory, progress, scheduler, job_id=None, log_lines=5):
    lines = [f"Position detection — {directory}", scheduler]
    if not progress:
        lines.append("Waiting for progress.json (written after prompt preparation).")
    else:
        done, total = progress["forward_evaluations"], progress["total_forward_evaluations"]
        fraction = min(1, max(0, done / total)) if total else 0
        fill = int(fraction * 32)
        lines.append(f"[{'#' * fill}{'-' * (32-fill)}] {100*fraction:.2f}% | {done:,}/{total:,} forward evaluations")
        lines.append(f"Saved conditions: {progress['completed_conditions']:,}/{progress['total_conditions']:,}"
                     f" | saved controls: {progress['completed_controls']:,}")
        age = max(0, time.time() - progress["updated_at"])
        lines.append(f"Last reported phase: {progress['phase']} | update age: {duration(age)}")
        if job_id and str(progress.get("job_id")) != job_id:
            lines.append("This snapshot belongs to a different job; waiting for this job to start/resume.")
        elif age > max(60, 2 * progress.get("log_interval_seconds", 30)) and progress["phase"] in {"running", "loading_model"}:
            lines.append("No recent update: check Slurm state and its .out/.err logs; this alone does not prove a failure.")
        lines.append(f"Session rate: {progress['forward_evaluations_per_second']:.2f} forward/s"
                     f" | approximate remaining time: {duration(progress['eta_seconds'])}")
        lines.append("Current: " + json.dumps(progress.get("current", {}), sort_keys=True))
        if progress.get("error"):
            lines.append("Error: " + progress["error"])
    recent = tail_log(directory / "run.log", log_lines)
    if recent:
        lines.extend(["", "Recent run.log:", *recent])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Experiment directory containing manifest.json")
    parser.add_argument("--job-id", help="Slurm job ID; inferred from progress.json once the run starts")
    parser.add_argument("--interval", type=float, default=10, help="Refresh interval in seconds")
    parser.add_argument("--tail", type=int, default=5, help="Number of recent run.log lines")
    parser.add_argument("--once", action="store_true", help="Print a single snapshot and exit")
    args = parser.parse_args()
    if not 1 <= args.interval <= 3600 or args.tail < 0:
        parser.error("Interval must be 1–3600 seconds and --tail must be nonnegative.")
    if args.job_id and not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", args.job_id):
        parser.error("Job ID must be a numeric Slurm ID, optionally with an array-task suffix.")
    try:
        while True:
            progress = read_progress(args.output_dir)
            job_id = args.job_id or progress.get("job_id")
            state, scheduler = scheduler_status(job_id)
            if sys.stdout.isatty() and not args.once:
                print("\033[2J\033[H", end="")
            print(render(args.output_dir, progress, scheduler, job_id, args.tail), flush=True)
            same_job = not job_id or str(progress.get("job_id")) == job_id
            finished = same_job and progress.get("phase") in {"completed", "failed", "interrupted", "incomplete"}
            if args.once:
                return 0
            if state in TERMINAL_STATES or finished:
                return 0 if same_job and progress.get("phase") == "completed" else 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped; the experiment is unaffected.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
