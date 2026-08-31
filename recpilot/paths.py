"""Repo / starter-kit paths. The kit is imported, never copied or edited."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KIT_DIR = REPO_ROOT / "kuairand-starter-kit"
DEFAULT_DATA_DIR = REPO_ROOT / "KuaiRand-Pure" / "data"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
BASELINE_SCORES = KIT_DIR / "baseline_scores.json"


def load_dotenv(path: Path | None = None) -> None:
    """Load gitignored .env into os.environ without overwriting existing vars."""
    src = path or (REPO_ROOT / ".env")
    if not src.exists():
        return
    for raw in src.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def ensure_kit_on_path() -> Path:
    """Put the official starter kit on sys.path so `data` / `evaluate` / `submit` import."""
    p = str(KIT_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    return KIT_DIR
