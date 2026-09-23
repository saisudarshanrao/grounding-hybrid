"""Bridge to GASP's pinned pipeline code (third_party/GASP/pipeline), used unmodified.

We load GASP's canonical cases and reuse its analysis helpers (feature derivation,
source-level split, classifier, paired bootstrap), so every comparison runs under the
exact protocol that produced the published numbers.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GASP_PIPELINE = ROOT / "third_party" / "GASP" / "pipeline"
if not GASP_PIPELINE.exists():
    raise ImportError("GASP not found. Run: python scripts/setup_gasp.py")
sys.path.insert(0, str(GASP_PIPELINE))

import analyze_gasp  # noqa: E402
import gasp_canonical  # noqa: E402

GASP_FEATS = analyze_gasp.GASP_FEATS      # six context-sensitivity features
BASE_FEATS = analyze_gasp.BASE_FEATS      # mean_surprisal, n_tok


def load_cases(canon_dir):
    """GASP's canonical cases for one run, in file order."""
    with open(Path(canon_dir) / "cases.jsonl", encoding="utf-8") as f:
        return [gasp_canonical.from_record(json.loads(line)) for line in f]


def load_sentences(canon_dir):
    """GASP's sentence.csv with its derived features (max_drop, ..., max_jsd), row order kept.

    Also adds prior_logprob: mean token log-prob WITHOUT context (our S2), recovered from
    GASP's own columns as -mean_surprisal - gap, so it needs no extra forward pass.
    """
    df = analyze_gasp.add_features(pd.read_csv(Path(canon_dir) / "sentence.csv"))
    df["prior_logprob"] = -df["mean_surprisal"] - df["gap"]
    return df
