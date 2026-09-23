"""Compare gating variants by grouped cross-validation INSIDE the dev split (test untouched).

For every GASP run (model x dataset) it takes GASP's own dev split (source_split, seed 0) and
reports, per method, the mean and spread of the per-fold ROC-AUC (span or response level), and the mean
per-fold difference to GASP+base. Methods that need S3 run only when the run's Lookback
features exist (results/features/<TAG>/features.npz); otherwise they show n/a.

Usage:
    python scripts/eval_gating.py                      # every run found
    python scripts/eval_gating.py --runs ragbench      # runs whose name contains "ragbench"
    python scripts/eval_gating.py --level response     # one row per answer
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grounding_hybrid.gasp_bridge import GASP_FEATS, analyze_gasp  # noqa: E402
from grounding_hybrid.gating import HardGate, Linear, dev_cv  # noqa: E402
from grounding_hybrid.signals import (BASE_FEATS, POS_FEATS, S1_FEATS, S2_FEATS, load_signals,  # noqa: E402
                                     to_response_level)

S1, S2, B = S1_FEATS, S2_FEATS, BASE_FEATS
ALL = S1 + S2 + ["s3"] + B
REF = "GASP+base [reference]"
METHODS = {   # name: (constructor, needs S3)
    "base (perplexity+length)": (lambda: Linear(B), False),
    REF: (lambda: Linear(GASP_FEATS + B), False),
    "S1+S2+base": (lambda: Linear(S1 + S2 + B), False),
    "(b) soft gate prior, no S3": (lambda: Linear(S1 + S2 + B, gate="prior_logprob", gated_cols=S1), False),
    "(a) hard split prior, no S3": (lambda: HardGate("prior_logprob", S1 + S2 + B, S1 + S2 + B), False),
    "position+base [control]": (lambda: Linear(POS_FEATS + B), False),
    "S3 Lookback alone": (lambda: Linear(["s3"]), True),
    "S3+position+base [control]": (lambda: Linear(["s3"] + POS_FEATS + B), True),
    "(c) combined S1+S2+S3+base": (lambda: Linear(ALL), True),
    "(b) soft gate: prior": (lambda: Linear(ALL, gate="prior_logprob", gated_cols=S1 + ["s3"]), True),
    "(b) soft gate: coverage": (lambda: Linear(ALL + ["ctx_kept"], gate="ctx_kept", gated_cols=S1 + ["s3"]), True),
    "(a) hard gate prior: S1 lo, S3 hi": (lambda: HardGate("prior_logprob", S1 + B, ["s3"] + S2 + B), True),
    "(a) hard split prior, all signals": (lambda: HardGate("prior_logprob", ALL, ALL), True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_root", default=str(ROOT / "results" / "gasp_repro" / "canon_results"))
    ap.add_argument("--features_root", default=str(ROOT / "results" / "features"))
    ap.add_argument("--runs", default="", help="only runs whose folder name contains this")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--level", choices=["span", "response"], default="span")
    ap.add_argument("--out", default=None, help="default: results/gating/dev_cv_<level>.json")
    args = ap.parse_args()

    results, summary = {}, {}
    for canon in sorted(Path(args.canon_root).glob("*_K5")):
        if args.runs not in canon.name:
            continue
        feat = Path(args.features_root) / canon.name / "features.npz"
        df, lb_cols = load_signals(canon, feat if feat.exists() else None)
        if args.level == "response":
            df = to_response_level(df, lb_cols)
        dev, _test = analyze_gasp.source_split(df, seed=0)          # test is never used here
        t0 = time.time()
        aucs = dev_cv(dev, METHODS, lb_cols, k=args.folds)
        ref = np.array(aucs[REF])
        rows = []
        for name, a in aucs.items():
            a = np.array(a)
            rows.append(dict(method=name, auc=np.nanmean(a) if np.isfinite(a).any() else np.nan,
                             sd=np.nanstd(a) if np.isfinite(a).any() else np.nan,
                             d_vs_gasp=np.nanmean(a - ref) if np.isfinite(a).any() else np.nan,
                             folds_better=int(np.sum(a > ref))))
        table = pd.DataFrame(rows)
        run = canon.name.replace("-Instruct", "").replace("_K5", "")
        print(f"\n# {run} [{args.level}]  dev: {len(dev)} rows / {dev['source_id'].nunique()} sources, "
              f"{args.folds}-fold grouped CV, S3 {'on' if lb_cols else 'MISSING'}  ({time.time() - t0:.0f}s)")
        print(table.to_string(index=False, float_format=lambda x: f"{x:+.3f}" if abs(x) < 0.3 else f"{x:.3f}"))
        results[run] = {r["method"]: r for r in rows}
        summary[run] = table.set_index("method")["auc"]

    if summary:
        print(f"\n# Summary: mean dev-CV {args.level}-level AUC (rows = methods)")
        print(pd.DataFrame(summary).round(3).to_string())
        out = Path(args.out or ROOT / "results" / "gating" / f"dev_cv_{args.level}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(out, "w"), indent=1, default=float)
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
