#!/usr/bin/env python3
"""NDCG@10 and Recall@50 alongside the official GAUC / nDCG@5 pair.

The problem statement's Limits row mentions NDCG@10 / Recall@50 while the
Benchmarks section and the Starter Kit pin GAUC / nDCG@5. The kit's `evaluate.py`
is the authority and is never modified; these are reported *in addition*, so the
submission carries whichever pair the portal asks for.

Recall@50 is near 1.0 for every model on this split because users average ~5.6
logged impressions, far under 50 — it is reported for completeness, not as a
discriminating metric.

Reads a submission CSV plus the split's labels. Scoring predictions after a run
has converged does not feed any selection decision.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recpilot.harness.dataio import load_kit  # noqa: E402
from recpilot.paths import ensure_kit_on_path  # noqa: E402

ensure_kit_on_path()
from evaluate import evaluate  # noqa: E402


def supplemental(users, labels, scores, ndcg_k: int = 10, recall_k: int = 50) -> dict:
    """Per-user NDCG@k and Recall@k, averaged over users that have a positive."""
    by_user = defaultdict(list)
    for u, y, s in zip(users, labels, scores):
        by_user[u].append((s, y))
    ndcgs, recalls, cand = [], [], []
    for _, rows in by_user.items():
        rows.sort(key=lambda t: -t[0])
        lab = [y for _, y in rows]
        pos = sum(lab)
        cand.append(len(lab))
        if pos == 0:
            continue                       # undefined for both; excluded like the kit's GAUC
        dcg = sum(l / math.log2(i + 2) for i, l in enumerate(lab[:ndcg_k]))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(pos, ndcg_k)))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        recalls.append(sum(lab[:recall_k]) / pos)
    return {
        f"NDCG@{ndcg_k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        f"Recall@{recall_k}": sum(recalls) / len(recalls) if recalls else 0.0,
        "evaluated_users": len(ndcgs),
        f"users_with_at_most_{recall_k}_candidates": (
            sum(1 for c in cand if c <= recall_k) / len(cand) if cand else 0.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--split", default="test", choices=("valid", "test"))
    ap.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_kit(args.data_dir)[args.split]
    scores = []
    with open(args.submission) as fh:
        reader = csv.DictReader(fh)
        for n, rec in enumerate(reader):
            if int(rec["row_id"]) != n:
                raise SystemExit(f"row_id {rec['row_id']} out of order at line {n}")
            scores.append(float(rec["score"]))
    if len(scores) != len(rows):
        raise SystemExit(f"{len(scores)} scores vs {len(rows)} rows in split {args.split}")

    users = [r[1] for r in rows]
    labels = [int(r[6]) for r in rows]
    official = evaluate(users, labels, scores)
    extra = supplemental(users, labels, scores)
    out = {"split": args.split, "submission": args.submission,
           "official": {k: official[k] for k in ("GAUC", "nDCG@5", "primary")},
           "supplemental": extra}
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
