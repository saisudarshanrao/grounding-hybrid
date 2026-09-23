"""Part B check on the DEV split only: does coverage-aware reading fix truncated contexts?

For each run with coverage-aware features (results/features/<TAG>/features_chunked.npz), builds
an out-of-fold S3 score (grouped 5-fold CV inside dev) from each Lookback variant and reports
its AUC overall, on truncated vs fully visible contexts, and per task (RAGBench domain):
  window1        GASP's window only (today's Lookback baseline)
  max / mean     ratios combined over all windows
  window1+max    both, concatenated

Usage:
    python scripts/analyze_coverage.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grounding_hybrid.gasp_bridge import analyze_gasp  # noqa: E402
from grounding_hybrid.gating import _folds, s3_scores  # noqa: E402
from grounding_hybrid.signals import load_signals  # noqa: E402

VARIANTS = {"window1": ("lookback",), "max": ("lookback_max",), "mean": ("lookback_mean",),
            "window1+max": ("lookback", "lookback_max")}


def oof(dev, cols, k=5):
    s = np.zeros(len(dev))
    for tr, va in _folds(dev, k):
        _, s[va] = s3_scores(dev.iloc[tr], dev.iloc[va], cols)
    return s


def auc(g, col):
    return roc_auc_score(g["label"], g[col]) if g["label"].nunique() == 2 and g["label"].sum() >= 10 else np.nan


def main():
    pd.set_option("display.width", 180)
    for canon in sorted((ROOT / "results" / "gasp_repro" / "canon_results").glob("*_K5")):
        feat = ROOT / "results" / "features" / canon.name / "features_chunked.npz"
        if not feat.exists():
            continue
        dev = None
        for name, keys in VARIANTS.items():
            df, lb_cols = load_signals(canon, feat, lb_keys=keys)
            d, _test = analyze_gasp.source_split(df, seed=0)
            if dev is None:
                dev = d[["case_id", "sent_idx", "label", "source_id", "task", "ctx_kept"]].copy()
            dev[name] = oof(d, lb_cols)
        groups = [("ALL dev", dev), ("truncated (ctx_kept<1)", dev[dev.ctx_kept < 1]),
                  ("fully visible", dev[dev.ctx_kept >= 1])] + list(dev.groupby("task"))
        rows = [dict(group=g, n=len(x), halluc=x["label"].mean(), **{v: auc(x, v) for v in VARIANTS})
                for g, x in groups]
        print(f"\n# {canon.name}: dev-only S3 AUC by Lookback variant")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
