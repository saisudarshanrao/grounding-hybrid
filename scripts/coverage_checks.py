"""Two follow-up checks for coverage-aware reading (part B), on the DEV split only.

(a) Pooled test. The per-run CIs in analyze_coverage.py are wide (TofuEval has 30 dev sources), so
    this pools the four runs (2 models x 2 datasets): the average over runs of AUC(max) - AUC(window1)
    on truncated rows, with a source-level bootstrap. Sources are resampled per dataset and the same
    draw is used for both models, since they score the same responses.
(b) Mechanism check. In analyze_coverage.py each variant trains its own S3 classifier, so part of a
    gain can come from a better-trained classifier, not from reading the cut-off evidence. Here ONE
    classifier per model is trained on fully visible rows (one window: window1 = max = mean = full)
    and applied unchanged to the truncated rows with each variant's features, so only the features
    differ. RAGTruth: trained on its own fully visible dev rows. TofuEval has none (every context is
    cut at 512 tokens), so it uses the same model's RAGTruth classifier (cross-dataset; absolute AUCs
    are lower, the comparison between variants is still like for like).

Usage:
    python -W ignore scripts/coverage_checks.py --suffix chunked_ctx512
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analyze_coverage import VARIANTS, oof, paired_ci  # noqa: E402
from grounding_hybrid.gasp_bridge import analyze_gasp  # noqa: E402
from grounding_hybrid.gating import s3_scores  # noqa: E402
from grounding_hybrid.signals import load_signals  # noqa: E402

NAMES = ["window1", "max", "mean", "full"]


def load_dev(canon, suffix):
    """{variant: dev rows with that variant's Lookback columns} + truncated flag, same rows for all."""
    feat = ROOT / "results" / "features" / canon.name / f"features_{suffix}.npz"
    full = ROOT / "results" / "features" / canon.name / "features.npz"
    z = np.load(feat)
    nwin = dict(zip(zip(z["case_id"], z["sent_idx"]), z["n_windows"]))
    out = {}
    for name, keys in list(VARIANTS.items()) + [("full", ("lookback",))]:
        df, cols = load_signals(canon, full if name == "full" else feat, lb_keys=keys)
        d, _test = analyze_gasp.source_split(df, seed=0)
        d = d.reset_index(drop=True)
        d["truncated"] = [nwin[(c, s)] > 1 for c, s in zip(d["case_id"], d["sent_idx"])]
        out[name] = d
    ref = out["window1"]
    for d in out.values():
        assert (d["case_id"].values == ref["case_id"].values).all()
    return out, cols


def pooled(runs, a="max", b="window1", n=2000, seed=0):
    """Mean over runs of AUC(a) - AUC(b) on truncated rows; bootstrap resamples sources per dataset."""
    rng = np.random.default_rng(seed)
    by_ds = {}
    for (model, ds), r in runs.items():
        by_ds.setdefault(ds, []).append((model, r))
    point = np.mean([roc_auc_score(r["label"], r[a]) - roc_auc_score(r["label"], r[b]) for r in runs.values()])
    gap = np.mean([roc_auc_score(r["label"], r["full"]) - roc_auc_score(r["label"], r[b]) for r in runs.values()])
    pos = {k: {s: np.where(r["source_id"].values == s)[0] for s in r["source_id"].unique()} for k, r in runs.items()}
    diffs = []
    for _ in range(n):
        ds_diffs = []
        for ds, members in by_ds.items():
            srcs = sorted(set().union(*[set(r["source_id"]) for _, r in members]))
            draw = rng.choice(srcs, len(srcs), replace=True)
            for model, r in members:
                p = pos[(model, ds)]
                idx = np.concatenate([p[s] for s in draw if s in p])
                y = r["label"].values[idx]
                if y.min() != y.max():
                    ds_diffs.append(roc_auc_score(y, r[a].values[idx]) - roc_auc_score(y, r[b].values[idx]))
        diffs.append(np.mean(ds_diffs))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(diff=float(point), lo=float(lo), hi=float(hi), p_le0=float(np.mean(np.array(diffs) <= 0)),
                gap=float(gap), recovered=float(point / gap) if abs(gap) > 1e-3 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="chunked_ctx512")
    ap.add_argument("--datasets", nargs="*", default=["ragtruth", "tofueval"])
    args = ap.parse_args()
    canon_root = ROOT / "results" / "gasp_repro" / "canon_results"
    report, lines = {"suffix": args.suffix}, []
    say = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731

    data = {}
    for canon in sorted(canon_root.glob("*_K5")):
        model, ds = canon.name.split("_")[0], canon.name.split("_")[1]
        if ds in args.datasets and (ROOT / "results" / "features" / canon.name / f"features_{args.suffix}.npz").exists():
            data[(model, ds)] = load_dev(canon, args.suffix)

    # (a) pooled test on the same per-variant out-of-fold S3 scores as analyze_coverage.py
    say(f"(a) POOLED TEST [{args.suffix}], truncated dev rows, per-variant S3 classifier (as analyze_coverage.py)")
    runs = {}
    for key, (dev, cols) in data.items():
        r = dev["window1"][["label", "source_id", "truncated"]].copy()
        for name in NAMES:
            r[name] = oof(dev[name], cols)
        runs[key] = r[r["truncated"]].reset_index(drop=True)
    groups = {"ALL 4 runs": list(runs)}
    groups.update({f"dataset {d}": [k for k in runs if k[1] == d] for d in args.datasets})
    groups.update({f"model {m}": [k for k in runs if k[0] == m] for m in sorted({k[0] for k in runs})})
    report["pooled"] = {}
    for g, keys in groups.items():
        for v in ("max", "mean"):
            res = pooled({k: runs[k] for k in keys}, a=v)
            report["pooled"][f"{g} | {v}"] = res
            say(f"  {g:32s} {v:4s} - window1 = {res['diff']:+.3f}  95% CI [{res['lo']:+.3f}, {res['hi']:+.3f}]"
                f"  {'SIGNIFICANT' if res['lo'] > 0 else 'n.s.'}  (boot P(diff<=0) = {res['p_le0']:.3f})"
                + (f"  recovers {res['recovered']:.0%} of the gap ({res['gap']:+.3f})" if res["recovered"] is not None else ""))

    # (b) mechanism check: one fixed classifier per model, trained on fully visible RAGTruth dev rows
    say()
    say("(b) MECHANISM CHECK: one S3 classifier per model, trained on fully visible RAGTruth dev rows,")
    say("    applied unchanged to truncated rows with each variant's features (only the features differ)")
    report["fixed_classifier"] = {}
    for model in sorted({k[0] for k in data}):
        if (model, "ragtruth") not in data:
            continue
        rt, cols = data[(model, "ragtruth")]
        train = rt["window1"][~rt["window1"]["truncated"]].reset_index(drop=True)
        vis = ~rt["window1"]["truncated"]
        drift = max(float(np.abs(rt[n].loc[vis, cols].to_numpy(float) - train[cols].to_numpy(float)).max())
                    for n in ("max", "mean", "full"))
        say(f"  {model}: trained on {len(train)} fully visible rows / {train['source_id'].nunique()} sources "
            f"(max |feature diff| between variants on those rows: {drift:.4f})")
        for ds in args.datasets:
            if (model, ds) not in data:
                continue
            dev, _ = data[(model, ds)]
            tr_mask = dev["window1"]["truncated"].values
            shared = set(train["source_id"]) & set(dev["window1"].loc[tr_mask, "source_id"])
            fit = train[~train["source_id"].isin(shared)].reset_index(drop=True) if shared else train
            r = dev["window1"].loc[tr_mask, ["label", "source_id"]].reset_index(drop=True)
            for name in NAMES:
                _, r[name] = s3_scores(fit, dev[name].loc[tr_mask].reset_index(drop=True), cols)
            aucs = {n: float(roc_auc_score(r["label"], r[n])) for n in NAMES}
            gap = aucs["full"] - aucs["window1"]
            res = dict(n=len(r), sources=int(r["source_id"].nunique()), dropped_shared_sources=len(shared), auc=aucs)
            txt = "  ".join(f"{n} {aucs[n]:.3f}" for n in NAMES)
            say(f"  {model} {ds} truncated rows ({len(r)} / {r['source_id'].nunique()} sources"
                + (f", {len(shared)} shared sources dropped from training" if shared else "") + f"): {txt}")
            for v in ("max", "mean"):
                m, lo, hi = paired_ci(r, v, "window1")
                rec = (aucs[v] - aucs["window1"]) / gap if abs(gap) > 1e-3 else None
                res[v] = dict(diff=m, lo=lo, hi=hi, recovered=rec)
                say(f"      {v:4s} - window1 = {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}] "
                    f"{'SIGNIFICANT' if lo > 0 or hi < 0 else 'n.s.'}"
                    + (f" | recovers {rec:.0%} of the window1 -> full gap ({gap:+.3f})" if rec is not None else ""))
            report["fixed_classifier"][f"{model} {ds}"] = res
        say()

    out = ROOT / "results" / "gating" / f"dev_coverage_checks_{args.suffix}"
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    json.dump(report, open(out.with_suffix(".json"), "w"), indent=1)
    print(f"saved {out}.txt/.json")


if __name__ == "__main__":
    main()
