"""Repo / starter-kit paths. The kit is imported, never copied or edited."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KIT_DIR = REPO_ROOT / "kuairand-starter-kit"
DEFAULT_DATA_DIR = REPO_ROOT / "KuaiRand-Pure" / "data"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
BASELINE_SCORES = KIT_DIR / "baseline_scores.json"


def ensure_kit_on_path() -> Path:
    """Put the official starter kit on sys.path so `data` / `evaluate` / `submit` import."""
    p = str(KIT_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    return KIT_DIR
