"""Part B check on the DEV split only: does coverage-aware reading recover truncated evidence?

For each run with coverage-aware features, builds an out-of-fold S3 score (grouped 5-fold CV inside
dev) from each Lookback variant and reports its AUC overall, on truncated vs fully visible
contexts, and per task:
  window1   the first window only (what a single truncated pass sees)
  max/mean  ratios combined over all windows (coverage-aware reading)
  full      the 1800-token single-pass features (features.npz): for the controlled-truncation
            runs this sees the whole context, so it is the upper bound
For the truncated rows it adds a source-level paired bootstrap CI for (max or mean) - window1 and
the share of the window1 -> full gap that the windowed reading recovers.

Usage:
    python scripts/analyze_coverage.py                                   # RAGBench, 1800-token windows
    python scripts/analyze_coverage.py --suffix chunked_ctx512 --datasets ragtruth tofueval
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

VARIANTS = {"window1": ("lookback",), "max": ("lookback_max",), "mean": ("lookback_mean",)}


def oof(dev, cols, k=5):
    s = np.zeros(len(dev))
    for tr, va in _folds(dev, k):
        _, s[va] = s3_scores(dev.iloc[tr], dev.iloc[va], cols)
    return s


def auc(g, col):
    return roc_auc_score(g["label"], g[col]) if g["label"].nunique() == 2 and g["label"].sum() >= 10 else np.nan


def paired_ci(d, a, b, n=2000, seed=0):
    """Source-level paired bootstrap of AUC(a) - AUC(b) on the rows of d."""
    rng, srcs = np.random.default_rng(seed), d["source_id"].unique()
    pos = {s: np.where(d["source_id"].values == s)[0] for s in srcs}
    diffs = []
    for _ in range(n):
        idx = np.concatenate([pos[s] for s in rng.choice(srcs, len(srcs), replace=True)])
        y = d["label"].values[idx]
        if y.min() != y.max():
            diffs.append(roc_auc_score(y, d[a].values[idx]) - roc_auc_score(y, d[b].values[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="chunked", help="features_<suffix>.npz to read")
    ap.add_argument("--datasets", nargs="*", default=None, help="only these datasets")
    args = ap.parse_args()
    pd.set_option("display.width", 180)
    fmt = lambda x: f"{x:.3f}"  # noqa: E731
    for canon in sorted((ROOT / "results" / "gasp_repro" / "canon_results").glob("*_K5")):
        if args.datasets and not any(f"_{d}_" in canon.name for d in args.datasets):
            continue
        feat = ROOT / "results" / "features" / canon.name / f"features_{args.suffix}.npz"
        if not feat.exists():
            continue
        z = np.load(feat)
        full = ROOT / "results" / "features" / canon.name / "features.npz"
        todo = list(VARIANTS.items()) + ([("full", ("lookback",))] if full.exists() else [])
        dev = None
        for name, keys in todo:
            df, lb_cols = load_signals(canon, full if name == "full" else feat, lb_keys=keys)
            d, _test = analyze_gasp.source_split(df, seed=0)
            if dev is None:
                dev = d[["case_id", "sent_idx", "label", "source_id", "task"]].copy()
                nwin = dict(zip(zip(z["case_id"], z["sent_idx"]), z["n_windows"]))
                dev["truncated"] = [nwin[(c, s)] > 1 for c, s in zip(dev["case_id"], dev["sent_idx"])]
            dev[name] = oof(d, lb_cols)
        cols = [c for c in ["window1", "max", "mean", "full"] if c in dev]
        groups = [("ALL dev", dev), ("truncated (>1 window)", dev[dev.truncated]),
                  ("fully visible", dev[~dev.truncated])] + list(dev.groupby("task"))
        rows = [dict(group=g, n=len(x), halluc=x["label"].mean(), **{v: auc(x, v) for v in cols})
                for g, x in groups]
        print(f"\n# {canon.name} [{args.suffix}]: dev-only S3 AUC by Lookback variant")
        print(pd.DataFrame(rows).to_string(index=False, float_format=fmt))
        tr = dev[dev.truncated]
        if tr["label"].nunique() == 2:
            for v in ("max", "mean"):
                m, lo, hi = paired_ci(tr, v, "window1")
                gap = auc(tr, "full") - auc(tr, "window1") if "full" in dev else np.nan
                rec = (auc(tr, v) - auc(tr, "window1")) / gap if gap == gap and abs(gap) > 1e-3 else np.nan
                print(f"  truncated rows ({len(tr)} / {tr['source_id'].nunique()} sources): {v} - window1 = "
                      f"{m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}] {'SIGNIFICANT' if lo > 0 or hi < 0 else 'n.s.'}"
                      + (f" | recovers {rec:.0%} of the window1 -> full gap ({gap:+.3f})" if rec == rec else ""))


if __name__ == "__main__":
    main()
