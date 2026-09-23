"""Hypothesis check on the DEV split only: does GASP's signal fail when the model already knows?

Bins dev sentences by S2 (no-context prior log-prob) into terciles and reports, per bin, the
hallucination rate and the ranking AUC of GASP's sensitivity features and of perplexity (each
feature's direction fixed once on all dev rows, as analyze_gasp.py does). The test split is not
touched, so this can guide the method design without leaking.

When a run's Lookback features exist (results/features/<TAG>/features.npz), S3 is added as an
out-of-fold score (grouped 5-fold CV inside dev) and the per-task (RAGBench domain) table is shown.

Usage:
    python scripts/analyze_prior.py                     # all GASP runs under results/gasp_repro
    python scripts/analyze_prior.py --canon_root <dir>
"""
import argparse
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

FEATS = ["gap", "max_drop", "jsd_noctx", "min_drop", "mean_surprisal"]


def binned_auc(dev, feats, bins=3):
    dev = dev.copy()
    dev["prior_bin"] = pd.qcut(dev["prior_logprob"], bins, labels=["low", "mid", "high"][:bins])
    signs = {f: (1.0 if roc_auc_score(dev["label"], dev[f].fillna(dev[f].median())) >= 0.5 else -1.0)
             for f in feats}
    rows = []
    for b, g in dev.groupby("prior_bin", observed=True):
        r = dict(prior_bin=b, n=len(g), halluc_rate=g["label"].mean(),
                 prior_range=f"[{g['prior_logprob'].min():.2f}, {g['prior_logprob'].max():.2f}]")
        for f in feats:
            gg = g.dropna(subset=[f])
            r[f] = roc_auc_score(gg["label"], signs[f] * gg[f]) if gg["label"].nunique() == 2 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def s3_oof(dev, lb_cols, k=5):
    """Out-of-fold Lookback score for every dev row (each fold's model never saw that fold)."""
    s = np.zeros(len(dev))
    for tr, va in _folds(dev, k):
        _, s[va] = s3_scores(dev.iloc[tr], dev.iloc[va], lb_cols)
    return s


def by_task(dev, feats):
    rows = []
    for t, g in dev.groupby("task"):
        r = dict(task=t, n=len(g), halluc_rate=g["label"].mean(), ctx_kept=g["ctx_kept"].mean())
        for f in feats:
            ok = g["label"].nunique() == 2 and g["label"].sum() >= 10
            r[f] = roc_auc_score(g["label"], -g[f] if f in ("gap", "max_drop", "jsd_noctx") else g[f]) if ok else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_root", default=str(ROOT / "results" / "gasp_repro" / "canon_results"))
    ap.add_argument("--features_root", default=str(ROOT / "results" / "features"))
    args = ap.parse_args()
    pd.set_option("display.width", 180)
    fmt = lambda x: f"{x:.3f}"  # noqa: E731
    for canon in sorted(Path(args.canon_root).glob("*_K5")):
        feat = Path(args.features_root) / canon.name / "features.npz"
        df, lb_cols = load_signals(canon, feat if feat.exists() else None)
        dev, _test = analyze_gasp.source_split(df, seed=0)
        feats = list(FEATS)
        if lb_cols:
            dev = dev.copy()
            dev["S3_lookback"] = s3_oof(dev, lb_cols)
            feats.append("S3_lookback")
        print(f"\n# {canon.name}  (dev only: {len(dev)} sentences, {int(dev['label'].sum())} hallucinated, "
              f"S3 {'on' if lb_cols else 'missing'})")
        print(binned_auc(dev, feats).to_string(index=False, float_format=fmt))
        if dev["task"].nunique() > 1:
            print("by task (direction: low sensitivity / high surprisal / high S3 score = suspicious)")
            print(by_task(dev, ["gap", "mean_surprisal"] + (["S3_lookback"] if lb_cols else []))
                  .to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
