"""Part B check on the DEV split only: does coverage-aware reading recover truncated evidence?

For each run with coverage-aware features, builds an out-of-fold S3 score (grouped 5-fold CV inside
dev) from each Lookback variant and reports its AUC overall, on truncated vs fully visible
contexts, and per task:
  window1   the first window only (what a single truncated pass sees)
  max/mean  ratios combined over all windows (coverage-aware reading)
  full      the 1800-token single-pass features (features.npz): for the controlled-truncation
            runs this sees the whole context, so it is the upper bound
  len       log context length (characters, GASP's audit) alone. A control: Lookback's context
            term is a mean over the context tokens, so max/mean/full can pick up the context's
            length, which window1 cannot see on truncated rows
  w1+len    window1 plus log context length
Two checks come first: on fully visible rows window1 and the 1800-token features must be identical
(both passes read the same tokens), and "full" is an upper bound only where the 1800-token window
kept the whole context (GASP's audit ctx_kept). For the truncated rows it then adds source-level
paired bootstrap CIs, all from one set of draws: the gain (max or mean) - window1 with the share of
the window1 -> full gap it recovers, the gain over w1+len, and, in the controlled-truncation runs
(where full reads more than window1), the truncation loss full - window1 and what is still missing
(full - max or mean) with a CI for the share of the loss recovered.

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


def boot_aucs(d, cols, n=2000, seed=0):
    """Source-level paired bootstrap on the rows of d: the AUC of every column, one row per draw."""
    rng, srcs = np.random.default_rng(seed), d["source_id"].unique()
    pos = {s: np.where(d["source_id"].values == s)[0] for s in srcs}
    draws = []
    for _ in range(n):
        idx = np.concatenate([pos[s] for s in rng.choice(srcs, len(srcs), replace=True)])
        y = d["label"].values[idx]
        if y.min() != y.max():
            draws.append([roc_auc_score(y, d[c].values[idx]) for c in cols])
    return pd.DataFrame(draws, columns=cols)


def ci(diffs):
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return f"{np.mean(diffs):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}] {'SIGNIFICANT' if lo > 0 or hi < 0 else 'n.s.'}"


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
        zi = {k: i for i, k in enumerate(zip(z["case_id"], z["sent_idx"]))}
        full = ROOT / "results" / "features" / canon.name / "features.npz"
        todo = list(VARIANTS.items()) + ([("full", ("lookback",))] if full.exists() else [])
        dev = None
        for name, keys in todo:
            df, lb_cols = load_signals(canon, full if name == "full" else feat, lb_keys=keys)
            d, _test = analyze_gasp.source_split(df, seed=0)
            if dev is None:
                dev = d[["case_id", "sent_idx", "label", "source_id", "task", "ctx_kept"]].copy()
                nwin = z["n_windows"]
                dev["truncated"] = [nwin[zi[k]] > 1 for k in zip(dev["case_id"], dev["sent_idx"])]
                chars = pd.read_csv(canon / "audit.csv").set_index("case_id")["ctx_chars"]
                d["log_len"] = np.log1p(d["case_id"].map(chars).to_numpy(float))
                dev["len"], dev["w1+len"] = oof(d, ["log_len"]), oof(d, lb_cols + ["log_len"])
            assert (d["case_id"].values == dev["case_id"].values).all() and \
                (d["sent_idx"].values == dev["sent_idx"].values).all(), "dev rows differ between variants"
            dev[name] = oof(d, lb_cols)
        cols = [c for c in ["window1", "max", "mean", "full", "len", "w1+len"] if c in dev]
        groups = [("ALL dev", dev), ("truncated (>1 window)", dev[dev.truncated]),
                  ("fully visible", dev[~dev.truncated])] + list(dev.groupby("task"))
        rows = [dict(group=g, n=len(x), halluc=x["label"].mean(), **{v: auc(x, v) for v in cols})
                for g, x in groups]
        print(f"\n# {canon.name} [{args.suffix}]: dev-only S3 AUC by Lookback variant")
        print(pd.DataFrame(rows).to_string(index=False, float_format=fmt))
        tr = dev[dev.truncated]
        # "full" is a longer reading only in the controlled-truncation runs; in 1800-token runs it is window1
        longer = "full" in dev and not np.allclose(tr["full"], tr["window1"])
        if full.exists():
            zf = np.load(full)
            zfi = {k: i for i, k in enumerate(zip(zf["case_id"], zf["sent_idx"]))}
            vis = list(zip(dev["case_id"][~dev.truncated], dev["sent_idx"][~dev.truncated]))
            if vis:
                w1 = z["lookback"][[zi[k] for k in vis]].astype(np.float32)
                f1800 = zf["lookback"][[zfi[k] for k in vis]].astype(np.float32)
                print(f"  check: fully visible rows, window1 vs 1800-token features: max |diff| "
                      f"{np.abs(w1 - f1800).max():.4f} (~0 expected: both passes read the same tokens)")
        if longer:
            print(f"  check: truncated rows the 1800-token window also cut (full is no upper bound there): "
                  f"{int((tr['ctx_kept'] < 1).sum())} of {len(tr)}")
        if tr["label"].nunique() == 2:
            b = boot_aucs(tr, cols)
            head = f"  truncated rows ({len(tr)} / {tr['source_id'].nunique()} sources): "
            gap = auc(tr, "full") - auc(tr, "window1") if "full" in dev else np.nan
            for v in ("max", "mean"):
                rec = (auc(tr, v) - auc(tr, "window1")) / gap if gap == gap and abs(gap) > 1e-3 else np.nan
                print(head + f"{v} - window1 = {ci(b[v] - b['window1'])}"
                      + (f" | recovers {rec:.0%} of the window1 -> full gap ({gap:+.3f})" if rec == rec else ""))
            if longer:
                loss = b["full"] - b["window1"]
                print(head + f"full - window1 = {ci(loss)}  <- truncation loss")
                for v in ("max", "mean"):
                    share = ""
                    if np.percentile(loss, 2.5) > 0:   # a share of the loss only means something if there is one
                        lo, hi = np.nanpercentile((b[v] - b["window1"]) / loss, [2.5, 97.5])
                        share = f" | share recovered 95% CI [{lo:.0%}, {hi:.0%}]"
                    print(head + f"full - {v} = {ci(b['full'] - b[v])}  <- still missing" + share)
            for v in ("max", "mean"):
                print(head + f"{v} - w1+len = {ci(b[v] - b['w1+len'])}  <- gain beyond context length")


if __name__ == "__main__":
    main()
