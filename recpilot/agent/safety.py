"""Subprocess execution with a hard wall-clock timeout."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from recpilot.paths import REPO_ROOT


class RunnerTimeout(Exception):
    def __init__(self, seconds: float, tail: str):
        super().__init__(f"runner exceeded {seconds:.0f}s")
        self.tail = tail


class RunnerError(Exception):
    def __init__(self, returncode: int, tail: str):
        super().__init__(f"runner exited {returncode}")
        self.returncode = returncode
        self.tail = tail


def run_in_subprocess(
    run_dir: Path,
    timeout_s: float,
    synthetic: bool = False,
) -> str:
    cmd = [sys.executable, "-m", "recpilot.harness.runner", "--run_dir", str(run_dir)]
    if synthetic:
        cmd.append("--synthetic")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    ).rstrip(os.pathsep)
    log_path = run_dir / "runner.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            tail = _tail(log_path)
            raise RunnerTimeout(timeout_s, tail) from None
    tail = _tail(log_path)
    if proc.returncode != 0:
        raise RunnerError(proc.returncode or 1, tail)
    return tail


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()


def _tail(path: Path, n: int = 4000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(errors="replace")
    return data[-n:]
