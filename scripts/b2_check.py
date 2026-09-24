"""Step 6 (DEV only): does B2 = [window-1 ratios, max-over-windows ratios] beat B = max-over-windows?

Pre-registered rule (CLAUDE.md, "Step 6 record"), on rows whose context needs more than one window, with
out-of-fold S3 scores exactly as analyze_coverage.py:
  (i)  pooled over the 4 real long-context runs (TechQA, ExpertQA-long x 2 scorers), AUC(B2) - AUC(B) > 0
       with the 95% source-level bootstrap CI above 0 (sources resampled per dataset, same draw for both
       scorers), and
  (ii) no run among those 4 and the 4 controlled-truncation runs (512-token windows) has B2 - B < -0.01.
Also descriptive: B2 - B on the "all relevant evidence inside window 1" group (evidence_position.py), the
case where B was slightly worse than window 1 on ExpertQA-long.

Usage:
    HF_DATASETS_CACHE=<scratch> python -W ignore scripts/b2_check.py     # -> results/gating/b2_check.txt
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

CANON = ROOT / "results" / "gasp_repro" / "canon_results"
FEAT = ROOT / "results" / "features"
MODELS = ["Qwen2.5-1.5B-Instruct", "SmolLM2-1.7B-Instruct"]
REAL = [("techqa", "features_chunked_redeep.npz"), ("expertqalong", "features_chunked_redeep.npz")]
CONTROLLED = [("ragtruth", "features_chunked_ctx512.npz"), ("tofueval", "features_chunked_ctx512.npz")]


def run_frame(model, ds, fname):
    canon, npz = CANON / f"{model}_{ds}_K5", FEAT / f"{model}_{ds}_K5" / fname
    z = np.load(npz)
    nwin = dict(zip(zip(z["case_id"], z["sent_idx"]), z["n_windows"]))
    out = None
    for name, keys in (("B", ("lookback_max",)), ("B2", ("lookback", "lookback_max"))):
        df, cols = load_signals(canon, npz, lb_keys=keys)
        d, _ = analyze_gasp.source_split(df, seed=0)
        d = d.reset_index(drop=True)
        if out is None:
            out = d[["case_id", "sent_idx", "label", "source_id"]].copy()
        out[name] = oof(d, cols)
    trunc = np.array([nwin[(c, s)] > 1 for c, s in zip(out["case_id"], out["sent_idx"])])
    return canon, out[trunc].reset_index(drop=True)


def pooled(runs, n=2000, seed=0):
    """Mean over runs of AUC(B2) - AUC(B); sources resampled per dataset, same draw for both scorers."""
    rng = np.random.default_rng(seed)
    point = np.mean([roc_auc_score(r["label"], r["B2"]) - roc_auc_score(r["label"], r["B"]) for r in runs.values()])
    by_ds = {}
    for (model, ds), r in runs.items():
        by_ds.setdefault(ds, []).append(r)
    pos = {k: {s: np.where(r["source_id"].values == s)[0] for s in r["source_id"].unique()} for k, r in runs.items()}
    diffs = []
    for _ in range(n):
        ds_d = []
        for ds in by_ds:
            keys = [k for k in runs if k[1] == ds]
            srcs = sorted(set().union(*[set(runs[k]["source_id"]) for k in keys]))
            draw = rng.choice(srcs, len(srcs), replace=True)
            for k in keys:
                idx = np.concatenate([pos[k][s] for s in draw if s in pos[k]])
                y = runs[k]["label"].values[idx]
                if y.min() != y.max():
                    ds_d.append(roc_auc_score(y, runs[k]["B2"].values[idx]) - roc_auc_score(y, runs[k]["B"].values[idx]))
        diffs.append(np.mean(ds_d))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    lines = []
    say = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731
    real, per_run = {}, []
    say("# Step 6: B2 = [window 1, max over windows] vs B = max over windows (DEV, rows with >1 window)")
    for group, sets in (("real", REAL), ("controlled 512", CONTROLLED)):
        for ds, fname in sets:
            for model in MODELS:
                canon, r = run_frame(model, ds, fname)
                aB, aB2 = roc_auc_score(r["label"], r["B"]), roc_auc_score(r["label"], r["B2"])
                m, lo, hi = paired_ci(r, "B2", "B")
                per_run.append(aB2 - aB)
                say(f"  {group:15s} {model.split('-')[0]:8s} {ds:13s} n={len(r):5d} src={r['source_id'].nunique():4d} | "
                    f"B {aB:.3f}  B2 {aB2:.3f}  B2-B {aB2 - aB:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                if group == "real":
                    real[(model, ds)] = r
    m, lo, hi = pooled(real)
    rule_i = lo > 0
    rule_ii = min(per_run) >= -0.01
    say()
    say(f"(i)  pooled over the 4 real runs: B2 - B = {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'MET' if rule_i else 'not met'}")
    say(f"(ii) worst single run B2 - B = {min(per_run):+.3f} (must be >= -0.010) -> {'MET' if rule_ii else 'not met'}")
    say(f"DECISION: {'B2 becomes the method' if rule_i and rule_ii else 'B stays the method; B2 reported as a tested variant'}")

    # descriptive: the weakness B2 targets (all relevant evidence inside window 1)
    try:
        import evidence_position as ep
        from transformers import AutoTokenizer
        say("\n# Descriptive: by where the relevant evidence lies (RAGBench GPT-4 annotations)")
        for ds, (domain, min_chars) in ep.SETS.items():
            rows = ep.ragbench_rows(domain, min_chars)
            for model in MODELS:
                if (model, ds) not in real:
                    continue
                canon = CANON / f"{model}_{ds}_K5"
                ev = ep.case_evidence(canon, rows, AutoTokenizer.from_pretrained(
                    {"Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
                     "SmolLM2-1.7B-Instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct"}[model]))
                r = real[(model, ds)].copy()
                r["g"] = [ep.group(ev.get(c, {}).get("relevant", np.nan)) for c in r["case_id"]]
                for g in ("in window 1", "all out"):
                    x = r[r["g"] == g]
                    if x["label"].sum() >= 10 and (1 - x["label"]).sum() >= 10:
                        m, lo, hi = paired_ci(x, "B2", "B")
                        say(f"  {model.split('-')[0]:8s} {ds:13s} {g:12s} n={len(x):5d} | B {roc_auc_score(x['label'], x['B']):.3f}"
                            f"  B2 {roc_auc_score(x['label'], x['B2']):.3f}  B2-B {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    except Exception as e:  # the rule above does not depend on this part
        say(f"(descriptive part skipped: {type(e).__name__}: {e})")
    (ROOT / "results" / "gating" / "b2_check.txt").write_text("\n".join(lines) + "\n")
    print("saved results/gating/b2_check.txt")


if __name__ == "__main__":
    main()
