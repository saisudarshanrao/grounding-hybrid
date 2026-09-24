"""Step 7c (DEV only): does frequency-aware attention (FA; arXiv 2602.18145) beat Lookback Lens?

Pre-registered rule (CLAUDE.md, "Step 7c record"): out-of-fold AUC on all dev rows (S3-style classifier as
analyze_coverage.oof); if FA [cv] - Lookback [cv] pooled over the 6 runs (RAGTruth, TofuEval, RAGBench x 2
scorers; sources resampled per dataset, same draw for both scorers) has its 95% CI above 0, FA replaces
Lookback as the strongest baseline; otherwise FA is reported as an additional baseline. Descriptive:
FA+Lookback [cv] and per-run paired CIs.

Usage:
    python -W ignore scripts/freq_check.py        # -> results/gating/freq_check.txt
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_coverage import oof, paired_ci  # noqa: E402
from grounding_hybrid.gasp_bridge import analyze_gasp  # noqa: E402
from grounding_hybrid.signals import load_signals  # noqa: E402

MODELS = ["Qwen2.5-1.5B-Instruct", "SmolLM2-1.7B-Instruct"]
DATASETS = ["ragtruth", "tofueval", "ragbench"]
VARIANTS = {"Lookback": ("lookback",), "FA": ("freq",), "FA+Lookback": ("freq", "lookback")}


def run_frame(model, ds):
    canon = ROOT / "results" / "gasp_repro" / "canon_results" / f"{model}_{ds}_K5"
    npz = ROOT / "results" / "features" / f"{model}_{ds}_K5" / "features_freq.npz"
    out = None
    for name, keys in VARIANTS.items():
        df, cols = load_signals(canon, npz, lb_keys=keys)
        d, _ = analyze_gasp.source_split(df, seed=0)
        d = d.reset_index(drop=True)
        if out is None:
            out = d[["case_id", "sent_idx", "label", "source_id"]].copy()
        out[name] = oof(d, cols)
    return out


def pooled(runs, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    point = np.mean([roc_auc_score(r["label"], r[a]) - roc_auc_score(r["label"], r[b]) for r in runs.values()])
    pos = {k: {s: np.where(r["source_id"].values == s)[0] for s in r["source_id"].unique()} for k, r in runs.items()}
    diffs = []
    for _ in range(n):
        dd = []
        for ds in DATASETS:
            keys = [k for k in runs if k[1] == ds]
            srcs = sorted(set().union(*[set(runs[k]["source_id"]) for k in keys]))
            draw = rng.choice(srcs, len(srcs), replace=True)
            for k in keys:
                idx = np.concatenate([pos[k][s] for s in draw if s in pos[k]])
                y = runs[k]["label"].values[idx]
                if y.min() != y.max():
                    dd.append(roc_auc_score(y, runs[k][a].values[idx]) - roc_auc_score(y, runs[k][b].values[idx]))
        diffs.append(np.mean(dd))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    lines = []
    say = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731
    runs = {}
    say("# Step 7c: frequency-aware attention (FA) vs Lookback, DEV out-of-fold AUC, all rows")
    for ds in DATASETS:
        for model in MODELS:
            r = run_frame(model, ds)
            runs[(model, ds)] = r
            auc = {v: roc_auc_score(r["label"], r[v]) for v in VARIANTS}
            m, lo, hi = paired_ci(r, "FA", "Lookback")
            say(f"  {model.split('-')[0]:8s} {ds:9s} n={len(r):5d} src={r['source_id'].nunique():4d} | "
                + "  ".join(f"{v} {a:.3f}" for v, a in auc.items())
                + f" | FA-Lookback {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    m, lo, hi = pooled(runs, "FA", "Lookback")
    say(f"\nRULE: pooled FA - Lookback over 6 runs = {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> "
        + ("FA replaces Lookback as the strongest baseline" if lo > 0 else "FA reported as an additional baseline"))
    m, lo, hi = pooled(runs, "FA+Lookback", "Lookback")
    say(f"Descriptive: pooled FA+Lookback - Lookback = {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    (ROOT / "results" / "gating" / "freq_check.txt").write_text("\n".join(lines) + "\n")
    print("saved results/gating/freq_check.txt")


if __name__ == "__main__":
    main()
