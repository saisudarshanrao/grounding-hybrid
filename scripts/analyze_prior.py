"""Hypothesis check on the DEV split only: does GASP's signal fail when the model already knows?

Bins dev sentences by S2 (no-context prior log-prob) into terciles and reports, per bin, the
hallucination rate and the ranking AUC of GASP's sensitivity features and of perplexity (each
feature's direction fixed once on all dev rows, as analyze_gasp.py does). The test split is not
touched, so this can guide the method design without leaking.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_root", default=str(ROOT / "results" / "gasp_repro" / "canon_results"))
    args = ap.parse_args()
    pd.set_option("display.width", 160)
    for canon in sorted(Path(args.canon_root).glob("*_K5")):
        df, _ = load_signals(canon)
        dev, _test = analyze_gasp.source_split(df, seed=0)
        print(f"\n# {canon.name}  (dev only: {len(dev)} sentences, {int(dev['label'].sum())} hallucinated)")
        print(binned_auc(dev, FEATS).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
