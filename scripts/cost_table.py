"""Compute cost per detector (RQ3), from the timings the runs already saved.

Seconds per case on one Kaggle T4, per scorer and dataset:
  GASP        results/gasp_repro/run_log*.jsonl: 2 + K forward passes per case (full context, no context,
              K leave-one-chunk-out), fp16 with SDPA attention. Wall time of the whole run, so it also
              includes model / data loading and GASP's analysis (a small overestimate).
  Lookback    features.npz run: one fp32 eager pass per case (attention weights needed).
  +ReDeEP     features_redeep.npz run: the same pass plus the logit lens (2 x layers x vocabulary
              projections over the answer tokens) and the ECS gathers.
  B windows   features_chunked*.npz runs: one fp32 eager pass per context window.
Feature-run timings cover the extraction loop only (model loading excluded). Classifiers cost
milliseconds on a CPU and are left out.

Usage:
    python scripts/cost_table.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATS = {"features": "Lookback (1 pass, fp32 eager)", "features_redeep": "Lookback + ReDeEP (1 pass + logit lens)",
         "features_chunked": "B: windows of 1800 (RAGBench)", "features_chunked_ctx512": "B: windows of 512 (controlled)",
         "features_chunked_redeep": "B windows 1800 + ReDeEP (TechQA)"}


def main():
    rows = []
    for log in sorted((ROOT / "results" / "gasp_repro").glob("run_log*.jsonl")):
        for line in open(log):
            r = json.loads(line)
            if r.get("max_cases"):
                continue
            n = sum(1 for _ in open(ROOT / "results" / "gasp_repro" / "canon_results" / r["tag"] / "cases.jsonl"))
            rows.append(dict(tag=r["tag"], method="GASP (7 passes, fp16 SDPA)", cases=n, windows=None,
                             s_per_case=r["minutes"] * 60 / n))
    for meta in sorted((ROOT / "results" / "features").glob("*/meta*.json")):
        stem = meta.stem.replace("meta", "features")
        if stem not in FEATS:
            continue
        m = json.load(open(meta))
        win = None
        if "chunked" in stem:
            import numpy as np
            z = np.load(meta.parent / f"{stem}.npz")
            per_case = pd.Series(z["n_windows"], index=z["case_id"]).groupby(level=0).first()
            win = float(per_case.mean())
        rows.append(dict(tag=m["tag"], method=FEATS[stem], cases=m["n_cases"], windows=win,
                         s_per_case=m["minutes"] * 60 / m["n_cases"]))
    df = pd.DataFrame(rows)
    df["scorer"] = df["tag"].str.split("_").str[0]
    df["dataset"] = df["tag"].str.split("_").str[1]
    pd.set_option("display.width", 200)
    print("# Seconds per case on one T4 (mean windows per case for B)\n")
    print(df.pivot_table(index="method", columns=["scorer", "dataset"], values="s_per_case")
          .round(2).to_string())
    w = df.dropna(subset=["windows"])
    if len(w):
        print("\n# Mean windows per case (B)\n")
        print(w.pivot_table(index="method", columns=["scorer", "dataset"], values="windows").round(2).to_string())
    out = ROOT / "results" / "gating" / "cost_table.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
