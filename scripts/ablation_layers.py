"""Ablation on the DEV split: which Lookback layers and heads carry the signal?

Lookback features are one ratio per (layer, head). Each variant is scored out of fold (grouped
5-fold CV by source; C chosen inside the training folds) with a subset of them:
  all              every layer x head (the Lookback [cv] detector)
  early/mid/late   the first, middle or last third of the layers, all heads
  layer means      one feature per layer: the ratio averaged over its heads
  top-k heads      the k heads most correlated with the label on the TRAINING rows of each fold
  per layer        each layer alone (all its heads), to find where the signal peaks
Runs: the six main runs (window 1 = GASP's view) and, for B, TechQA's max-over-windows features.

Usage:
    python -W ignore scripts/ablation_layers.py      # -> results/gating/ablation_layers.{csv,txt}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grounding_hybrid.gasp_bridge import analyze_gasp  # noqa: E402
from grounding_hybrid.gating import _folds  # noqa: E402
from grounding_hybrid.signals import load_signals  # noqa: E402
from rq2_where import fit_score  # noqa: E402

TOPK = (4, 16, 64)


def oof(dev, cols, k_top=None):
    s = np.full(len(dev), np.nan)
    for tr, va in _folds(dev, 5):
        train = dev.iloc[tr]
        use = cols
        if k_top:
            x = train[cols].to_numpy(float)
            y = train["label"].to_numpy(float)
            xc, yc = x - x.mean(0), y - y.mean()
            r = np.abs((xc * yc[:, None]).sum(0) / (np.sqrt((xc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12))
            use = [cols[i] for i in np.argsort(-r)[:k_top]]
        s[va] = fit_score(train, dev.iloc[va], use, True)
    return s


def main():
    specs = []
    for canon in sorted((ROOT / "results" / "gasp_repro" / "canon_results").glob("*_K5")):
        f = ROOT / "results" / "features" / canon.name
        short = canon.name.replace("-Instruct", "").replace("_K5", "")
        if (f / "features.npz").exists():
            specs.append((short, canon, f / "features.npz", "lookback"))
        if (f / "features_chunked_redeep.npz").exists():
            specs.append((short + " B", canon, f / "features_chunked_redeep.npz", "lookback_max"))
    rows, curves = [], []
    for label, canon, npz, key in specs:
        z = np.load(npz)
        L, H = z[key].shape[1:]
        df, cols = load_signals(canon, npz, lb_keys=(key,))
        dev, _ = analyze_gasp.source_split(df, seed=0)
        dev = dev.reset_index(drop=True)
        grid = np.array(cols).reshape(L, H)                     # lb_{l*H+h}
        for l in range(L):
            dev[f"lm_{l}"] = dev[list(grid[l])].mean(1)
        third = np.array_split(np.arange(L), 3)
        variants = {"all": (cols, None), "early third": (list(grid[third[0]].ravel()), None),
                    "middle third": (list(grid[third[1]].ravel()), None),
                    "late third": (list(grid[third[2]].ravel()), None),
                    "layer means": ([f"lm_{l}" for l in range(L)], None)}
        variants.update({f"top-{k} heads": (cols, k) for k in TOPK})
        y = dev["label"].values
        for name, (c, k) in variants.items():
            rows.append(dict(run=label, variant=name, n_feats=k or len(c), auc=roc_auc_score(y, oof(dev, c, k))))
        for l in range(L):
            curves.append(dict(run=label, layer=l, rel_depth=l / (L - 1), auc=roc_auc_score(y, oof(dev, list(grid[l])))))
        print(f"done {label}", flush=True)
    res, cur = pd.DataFrame(rows), pd.DataFrame(curves)
    out = ROOT / "results" / "gating" / "ablation_layers"
    res.to_csv(out.with_suffix(".csv"), index=False)
    cur.to_csv(str(out) + "_per_layer.csv", index=False)
    pd.set_option("display.width", 250)
    t = res.pivot_table(index="variant", columns="run", values="auc").reindex(
        ["all", "early third", "middle third", "late third", "layer means"] + [f"top-{k} heads" for k in TOPK])
    best = cur.loc[cur.groupby("run")["auc"].idxmax()].set_index("run")
    txt = ("# DEV AUC (out-of-fold) by Lookback feature subset\n" + t.round(3).to_string()
           + "\n\n# Best single layer per run (all its heads)\n"
           + best[["layer", "rel_depth", "auc"]].round(3).to_string())
    out.with_suffix(".txt").write_text(txt + "\n")
    print(txt)
    print(f"\nsaved {out}.csv / .txt and {out.name}_per_layer.csv")


if __name__ == "__main__":
    main()
