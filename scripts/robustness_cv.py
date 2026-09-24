"""Robustness of the key DEV comparisons to the cross-validation split.

Every dev result so far uses one grouped 5-fold split (GroupKFold by source). Here the sources are
assigned to the 5 folds at random, for 10 seeds, and each detector is re-scored out of fold. The
dev/test split itself is NOT changed (that would move test sources into dev).

Comparisons (sentence level):
  Lookback - GASP+base     Lookback [cv] vs GASP's own classifier (C=1)
  Lookback - ReDeEP        Lookback [cv] vs ReDeEP [cv] (regression on ECS + PKS)
  B - Lookback             max over windows vs window 1, on the rows that need >1 window: TechQA
                           (real truncation) and the 512-token controlled-truncation runs
Reports, per run: mean and range of each AUC difference over the seeds and how many seeds favour it.

Usage:
    python -W ignore scripts/robustness_cv.py            # -> results/gating/robustness_cv.{csv,txt}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grounding_hybrid.gasp_bridge import BASE_FEATS, GASP_FEATS, analyze_gasp  # noqa: E402
from grounding_hybrid.signals import load_signals  # noqa: E402
from rq2_where import fit_score  # noqa: E402

SEEDS = range(10)


def seeded_folds(df, k, seed):
    srcs = df["source_id"].unique()
    fold_of = dict(zip(np.random.default_rng(seed).permutation(srcs), np.arange(len(srcs)) % k))
    f = df["source_id"].map(fold_of).values
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def oof(dev, cols, tune, seed):
    s = np.full(len(dev), np.nan)
    for tr, va in seeded_folds(dev, 5, seed):
        s[va] = fit_score(dev.iloc[tr], dev.iloc[va], cols, tune)
    return s


def dev_frame(canon, npz=None, keys=("lookback",)):
    df, cols = load_signals(canon, npz, lb_keys=keys) if npz else (load_signals(canon)[0], [])
    dev, _ = analyze_gasp.source_split(df, seed=0)
    return dev.reset_index(drop=True), cols


def runs():
    """(label, canon dir, {detector: (npz, keys, tuned)}, rows-mask spec)."""
    canon_root = ROOT / "results" / "gasp_repro" / "canon_results"
    for canon in sorted(canon_root.glob("*_K5")):
        f = ROOT / "results" / "features" / canon.name
        short = canon.name.replace("-Instruct", "").replace("_K5", "")
        if (f / "features_redeep.npz").exists():
            yield short, canon, {"GASP+base": (None, GASP_FEATS + BASE_FEATS, False),
                                 "Lookback": (f / "features_redeep.npz", ("lookback",), True),
                                 "ReDeEP": (f / "features_redeep.npz", ("ecs", "pks"), True)}, None
        if (f / "features_chunked_redeep.npz").exists():
            z = f / "features_chunked_redeep.npz"
            yield short, canon, {"GASP+base": (None, GASP_FEATS + BASE_FEATS, False),
                                 "Lookback": (z, ("lookback",), True), "ReDeEP": (z, ("ecs", "pks"), True),
                                 "B": (z, ("lookback_max",), True)}, z
        if (f / "features_chunked_ctx512.npz").exists():
            z = f / "features_chunked_ctx512.npz"
            yield short + " [512 trunc]", canon, {"Lookback": (z, ("lookback",), True),
                                                   "B": (z, ("lookback_max",), True)}, z


def main():
    rows = []
    for label, canon, dets, trunc_npz in runs():
        frames = {}
        for name, (npz, keys, tune) in dets.items():
            if npz is None:
                frames[name] = (dev_frame(canon)[0], list(keys), tune)
            else:
                d, cols = dev_frame(canon, npz, keys)
                frames[name] = (d, cols, tune)
        ref = next(iter(frames.values()))[0]
        mask = np.ones(len(ref), bool)
        if trunc_npz is not None:
            z = np.load(trunc_npz)
            nwin = dict(zip(zip(z["case_id"], z["sent_idx"]), z["n_windows"]))
            mask = np.array([nwin[(c, s)] > 1 for c, s in zip(ref["case_id"], ref["sent_idx"])])
        y = ref["label"].values[mask]
        for seed in SEEDS:
            auc = {}
            for name, (d, cols, tune) in frames.items():
                assert (d["case_id"].values == ref["case_id"].values).all()
                auc[name] = roc_auc_score(y, oof(d, cols, tune, seed)[mask])
            rows.append(dict(run=label, rows="truncated" if trunc_npz is not None else "all", seed=seed, **auc))
        print(f"done {label}", flush=True)
    res = pd.DataFrame(rows)
    out = ROOT / "results" / "gating" / "robustness_cv"
    res.to_csv(out.with_suffix(".csv"), index=False)
    lines = [f"# DEV AUC over {len(SEEDS)} random grouped 5-fold splits: mean [min, max]; diff = paired per seed\n"]
    for (run, rw), g in res.groupby(["run", "rows"], sort=False):
        parts = [f"{run} ({rw} rows)"]
        for det in ["GASP+base", "Lookback", "ReDeEP", "B"]:
            if det in g and g[det].notna().all():
                parts.append(f"{det} {g[det].mean():.3f} [{g[det].min():.3f}, {g[det].max():.3f}]")
        lines.append("  " + " | ".join(parts))
        for a, b in [("Lookback", "GASP+base"), ("Lookback", "ReDeEP"), ("B", "Lookback")]:
            if a in g and b in g and g[a].notna().all() and g[b].notna().all():
                d = g[a] - g[b]
                lines.append(f"      {a} - {b}: {d.mean():+.3f} [{d.min():+.3f}, {d.max():+.3f}], "
                             f"positive in {(d > 0).sum()}/{len(d)} seeds")
    txt = "\n".join(lines)
    out.with_suffix(".txt").write_text(txt + "\n")
    print(txt)
    print(f"\nsaved {out}.csv / .txt")


if __name__ == "__main__":
    main()
