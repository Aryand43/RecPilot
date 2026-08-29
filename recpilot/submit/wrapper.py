"""Wrap official submit.py helpers. row_id order == data.load()[split] order."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from recpilot.eval.wrapper import validate_scores
from recpilot.paths import ensure_kit_on_path

ensure_kit_on_path()
from submit import read_submission, write_submission  # noqa: E402


def write_scores(path: Path | str, rows: Sequence, scores: Sequence[float] | np.ndarray) -> Path:
    path = Path(path)
    arr = validate_scores(scores, len(rows))
    write_submission(str(path), rows, arr)
    return path


def check_submission(path: Path | str, rows: Sequence) -> list[float]:
    return read_submission(str(path), rows)
