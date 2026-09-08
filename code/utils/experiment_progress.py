"""Throttled run logging and resume-aware forward-evaluation progress."""

import json
import logging
import os
from pathlib import Path
import sys
import time

from tqdm import tqdm


def saved_progress(directory, manifest):
    """Count durable checkpoints, not the last possibly interrupted heartbeat."""
    config = manifest["config"]
    conditions = controls = evaluations = 0
    for prompt in manifest["prompts"]:
        pid = prompt["prompt_id"]
        if ((directory / "controls" / f"{pid}.json").exists()
                and (directory / "activations" / pid / "control.pt").exists()):
            controls += 1
            evaluations += 1
        for concept in config["concepts"]:
            for layer in config["layers"]:
                for alpha_index in range(len(config["alphas"])):
                    name = f"{concept}_layer{layer}_alpha{alpha_index}"
                    if ((directory / "conditions" / pid / f"{name}.json").exists()
                            and (directory / "activations" / pid / f"{name}.pt").exists()):
                        conditions += 1
                        evaluations += 2 + 32 - layer
    return controls, conditions, evaluations


class RunProgress:
    def __init__(self, directory, manifest, interval=30, show_bar=True):
        self.directory = Path(directory)
        self.interval = interval
        self.controls, self.conditions, self.evaluations = saved_progress(self.directory, manifest)
        config = manifest["config"]
        self.expected_conditions = (len(manifest["prompts"]) * len(config["concepts"])
                                    * len(config["layers"]) * len(config["alphas"]))
        self.total = manifest["forward_passes"]["total"]
        self.initial_evaluations = self.evaluations
        self.started = time.time()
        self.running_started = None
        self.last_write = 0
        self.phase = "loading_model"
        self.current = {}
        self.error = None
        self.logger = logging.getLogger(f"position_detection.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (logging.FileHandler(self.directory / "run.log"), logging.StreamHandler(sys.stdout)):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.show_bar = show_bar
        self.bar = None

    def __enter__(self):
        self.logger.info("Starting job=%s; reusing %s conditions and %s controls",
                         os.environ.get("SLURM_JOB_ID", "local"), self.conditions, self.controls)
        self.write(force=True)
        return self

    def start_running(self):
        self.running_started = time.monotonic()
        self.phase = "running"
        self.bar = tqdm(total=self.total, initial=self.evaluations, unit="forward",
                        desc="Experiment", dynamic_ncols=True, mininterval=1,
                        disable=not (self.show_bar and sys.stderr.isatty()))
        self.write(force=True)

    def set_condition(self, **fields):
        self.current = fields

    def advance(self, stage):
        self.current["stage"] = stage
        self.evaluations += 1
        self.bar.update(1)
        self.write()

    def condition_saved(self):
        self.conditions += 1
        self.current["stage"] = "condition saved"
        self.write()

    def control_saved(self):
        self.controls += 1
        self.advance("control saved")

    def write(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_write < self.interval:
            return
        self.last_write = now
        session_evaluations = self.evaluations - self.initial_evaluations
        elapsed = now - self.running_started if self.running_started is not None else 0
        rate = session_evaluations / elapsed if elapsed > 0 else 0
        eta = max(0, self.total - self.evaluations) / rate if rate > 0 else None
        if self.evaluations == self.total:
            eta = 0
        value = {
            "phase": self.phase, "job_id": os.environ.get("SLURM_JOB_ID"), "pid": os.getpid(),
            "started_at": self.started, "updated_at": time.time(),
            "log_interval_seconds": self.interval,
            "completed_conditions": self.conditions, "total_conditions": self.expected_conditions,
            "completed_controls": self.controls, "forward_evaluations": self.evaluations,
            "total_forward_evaluations": self.total, "reused_forward_evaluations": self.initial_evaluations,
            "session_forward_evaluations": session_evaluations,
            "forward_evaluations_per_second": rate, "eta_seconds": eta,
            "current": self.current, "error": self.error,
        }
        path = self.directory / "progress.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
        self.logger.info("%s | forwards %s/%s (%.1f%%) | saved conditions %s/%s | %.2f forward/s | %s",
                         self.phase, self.evaluations, self.total, 100*self.evaluations/self.total,
                         self.conditions, self.expected_conditions, rate,
                         json.dumps(self.current, sort_keys=True))

    def __exit__(self, exc_type, exc, traceback):
        if exc is None:
            self.phase = ("completed" if self.conditions == self.expected_conditions
                          and self.evaluations == self.total else "incomplete")
        else:
            self.phase = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            self.error = f"{exc_type.__name__}: {exc}"
            self.logger.error(self.error, exc_info=(exc_type, exc, traceback))
        try:
            self.write(force=True)
        finally:
            if self.bar is not None:
                self.bar.close()
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
        return False
