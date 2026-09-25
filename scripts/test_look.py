"""Step 8: the single, logged TEST look for every frozen detector (CLAUDE.md, "FREEZE record").

Every detector is fit on GASP's whole DEV split (C chosen by grouped 5-fold CV on dev; GASP+base and
perplexity+length use GASP's C = 1; ReDeEP's training-free form selects heads / layers / beta on dev) and
scored once on GASP's TEST split (source_split, seed 0). Missing GASP features are filled with dev medians so
that every detector is scored on the same rows (paired comparisons). Differences use GASP's source-level paired
bootstrap (2000 resamples); pooled differences resample sources per dataset, same draw for both scorers.

Tables (sentence level; response level for the main table):
  main       RAGTruth, TofuEval, RAGBench, TechQA, ExpertQA-long x 2 scorers: every frozen detector
  truncated  rows whose context needs >1 window: window 1 (Lookback) vs B, plus GASP / FA / ReDeEP there
  controlled 512-token windows on RAGTruth / TofuEval: window 1 vs B vs the full 1800-token view (upper bound)
  evidence   (descriptive) B - window 1 on truncated test rows by where RAGBench's annotated evidence lies

--eval_on devhalf fits on half of the dev sources and scores the other half: a dry run that never touches the
test split (used to check the code before the one real run).

Usage:
    python -W ignore scripts/test_look.py --eval_on devhalf     # dry run, dev only
    python -W ignore scripts/test_look.py --eval_on test        # THE test look (log it in CLAUDE.md)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_redeep import ReDeEP  # noqa: E402
from grounding_hybrid.gasp_bridge import BASE_FEATS, GASP_FEATS, analyze_gasp  # noqa: E402
from grounding_hybrid.gating import _folds  # noqa: E402
from grounding_hybrid.signals import load_signals  # noqa: E402

CANON = ROOT / "results" / "gasp_repro" / "canon_results"
FEAT = ROOT / "results" / "features"
MODELS = ["Qwen2.5-1.5B-Instruct", "SmolLM2-1.7B-Instruct"]
CS = np.logspace(-4, 1, 11)
MAIN = ["ragtruth", "tofueval", "ragbench", "techqa", "expertqalong"]
LONG = ["techqa", "expertqalong"]


# ---------------------------------------------------------------- data
def blocks(model, ds):
    """{block name: (npz, key)} for the features that exist for this run; 'lb' is window 1 (GASP's view)."""
    f = FEAT / f"{model}_{ds}_K5"
    out = {}
    if ds in LONG:
        cr = f / "features_chunked_redeep.npz"
        out.update(lb=(cr, "lookback"), lbmax=(cr, "lookback_max"), ecs=(cr, "ecs"), pks=(cr, "pks"))
    else:
        out.update(lb=(f / "features.npz", "lookback"), ecs=(f / "features_redeep.npz", "ecs"),
                   pks=(f / "features_redeep.npz", "pks"))
        if (f / "features_chunked.npz").exists():
            out["lbmax"] = (f / "features_chunked.npz", "lookback_max")
    if (f / "features_freq.npz").exists():
        out["fa"] = (f / "features_freq.npz", "freq")
    return out


def frame(model, ds):
    """Sentence frame (sentence.csv order) with GASP / base / prior columns and every feature block."""
    canon = CANON / f"{model}_{ds}_K5"
    df, _ = load_signals(canon)
    cols = {}
    for name, (npz, key) in blocks(model, ds).items():
        d, c = load_signals(canon, npz, lb_keys=(key,))
        assert (d["case_id"].values == df["case_id"].values).all()
        new = [f"{name}_{i}" for i in range(len(c))]
        df = pd.concat([df, d[c].set_axis(new, axis=1)], axis=1)
        cols[name] = new
    return df, cols


def windows(model, ds, fname):
    z = np.load(FEAT / f"{model}_{ds}_K5" / fname)
    return dict(zip(zip(z["case_id"], z["sent_idx"]), z["n_windows"]))


def to_response(df, cols):
    feats = sum(cols.values(), [])
    agg = {**{f: "max" for f in GASP_FEATS + BASE_FEATS}, "prior_logprob": "min",
           **{c: "mean" for c in feats}, "source_id": "first", "label": "max", "trunc": "max"}
    return df.groupby("case_id").agg(agg).reset_index()   # sorted, as analyze_gasp.py: same split


def split(df, eval_on):
    dev, test = analyze_gasp.source_split(df, seed=0)
    if eval_on == "test":
        return dev.reset_index(drop=True), test.reset_index(drop=True)
    srcs = np.sort(dev["source_id"].unique())
    half = set(np.random.default_rng(0).permutation(srcs)[: len(srcs) // 2])
    a = dev[dev["source_id"].isin(half)].reset_index(drop=True)
    b = dev[~dev["source_id"].isin(half)].reset_index(drop=True)
    return a, b


# ---------------------------------------------------------------- detectors
def fit_linear(train, test, cols, tune=True):
    med = train[cols].median()
    xtr, xte = train[cols].fillna(med), test[cols].fillna(med)
    sc = StandardScaler().fit(xtr)
    xtr, xte, y = sc.transform(xtr), sc.transform(xte), train["label"].values
    c = (LogisticRegressionCV(Cs=CS, cv=_folds(train, 5), scoring="roc_auc", class_weight="balanced",
                              max_iter=3000).fit(xtr, y).C_[0] if tune else 1.0)
    return LogisticRegression(C=c, class_weight="balanced", max_iter=3000).fit(xtr, y).decision_function(xte)


def detectors(cols):
    d = {"Perplexity+length": lambda tr, te: fit_linear(tr, te, BASE_FEATS, False),
         "GASP+base": lambda tr, te: fit_linear(tr, te, GASP_FEATS + BASE_FEATS, False),
         "ReDeEP": lambda tr, te: ReDeEP(cols["ecs"], cols["pks"]).fit(tr).score(te),
         "ReDeEP [cv]": lambda tr, te: fit_linear(tr, te, cols["ecs"] + cols["pks"]),
         "Lookback [cv]": lambda tr, te: fit_linear(tr, te, cols["lb"])}
    if "fa" in cols:
        d["Frequency-aware [cv]"] = lambda tr, te: fit_linear(tr, te, cols["fa"])
    if "lbmax" in cols:
        d["B (ours) [cv]"] = lambda tr, te: fit_linear(tr, te, cols["lbmax"])
    return d


# ---------------------------------------------------------------- statistics
def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def paired(test, sa, sb):
    r = analyze_gasp.paired_bootstrap_diff(test, "label", pd.Series(sa, index=test.index), pd.Series(sb, index=test.index))
    return r if r is None else dict(mean=r["mean"], lo=r["lo"], hi=r["hi"], sig=r["sig"])


def pooled(items, n=2000, seed=0):
    """items: list of (dataset, test frame, a scores, b scores); mean AUC(a) - AUC(b) over items."""
    rng = np.random.default_rng(seed)
    point = float(np.mean([auc(t["label"], a) - auc(t["label"], b) for _, t, a, b in items]))
    by_ds = {}
    for it in items:
        by_ds.setdefault(it[0], []).append(it)
    diffs = []
    for _ in range(n):
        dd = []
        for ds, its in by_ds.items():
            srcs = sorted(set().union(*[set(t["source_id"]) for _, t, _, _ in its]))
            draw = rng.choice(srcs, len(srcs), replace=True)
            for _, t, a, b in its:
                pos = {s: np.where(t["source_id"].values == s)[0] for s in t["source_id"].unique()}
                idx = np.concatenate([pos[s] for s in draw if s in pos])
                y = t["label"].values[idx]
                if y.min() != y.max():
                    dd.append(roc_auc_score(y, a[idx]) - roc_auc_score(y, b[idx]))
        diffs.append(np.mean(dd))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(mean=point, lo=float(lo), hi=float(hi), sig=bool(lo > 0 or hi < 0))


def fmt(r):
    return "n/a" if r is None else f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]{' SIG' if r['sig'] else ''}"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_on", choices=["devhalf", "test"], required=True)
    ap.add_argument("--no_evidence", action="store_true", help="skip the evidence-position part")
    args = ap.parse_args()
    out_dir = ROOT / "results" / ("test" if args.eval_on == "test" else "test_dryrun")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, cmp, lines = [], {}, []
    say = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731
    say(f"# FROZEN DETECTORS, eval_on={args.eval_on} (fit on {'dev' if args.eval_on == 'test' else 'dev half A'}, "
        f"scored on {'TEST' if args.eval_on == 'test' else 'dev half B'})")
    keep = {}                          # (model, ds, level) -> (eval frame, {detector: scores})
    for ds in MAIN:
        for model in MODELS:
            df, cols = frame(model, ds)
            wfile = ("features_chunked_redeep.npz" if ds in LONG else
                     "features_chunked.npz" if (FEAT / f"{model}_{ds}_K5" / "features_chunked.npz").exists() else None)
            nwin = windows(model, ds, wfile) if wfile else None
            df["trunc"] = [nwin[(c, s)] > 1 for c, s in zip(df["case_id"], df["sent_idx"])] if nwin else False
            for level in ("span", "response"):
                d = df if level == "span" else to_response(df, cols)
                tr, te = split(d, args.eval_on)
                scores = {name: np.asarray(f(tr, te), float) for name, f in detectors(cols).items()}
                keep[(model, ds, level)] = (te, scores)
                for name, s in scores.items():
                    rows.append(dict(model=model, dataset=ds, level=level, rows="all", detector=name, n=len(te),
                                     sources=int(te["source_id"].nunique()), auc=auc(te["label"], s)))
                    if te["trunc"].any():
                        m = te["trunc"].values.astype(bool)
                        rows.append(dict(model=model, dataset=ds, level=level, rows="truncated", detector=name,
                                         n=int(m.sum()), sources=int(te.loc[m, "source_id"].nunique()),
                                         auc=auc(te["label"].values[m], s[m])))
            print(f"done {model} {ds}", flush=True)
    res = pd.DataFrame(rows)

    # paired comparisons (sentence level unless noted)
    for (model, ds, level), (te, sc) in keep.items():
        k = f"{model.split('-')[0]} {ds} {level}"
        cmp[f"{k} | Lookback - GASP+base"] = paired(te, sc["Lookback [cv]"], sc["GASP+base"])
        cmp[f"{k} | Lookback - ReDeEP [cv]"] = paired(te, sc["Lookback [cv]"], sc["ReDeEP [cv]"])
        if "Frequency-aware [cv]" in sc:
            cmp[f"{k} | FA - Lookback"] = paired(te, sc["Frequency-aware [cv]"], sc["Lookback [cv]"])
        if "B (ours) [cv]" in sc:
            cmp[f"{k} | B - Lookback (all rows)"] = paired(te, sc["B (ours) [cv]"], sc["Lookback [cv]"])
            m = te["trunc"].values.astype(bool)
            if m.any():
                t = te[m]
                cmp[f"{k} | B - Lookback (truncated rows)"] = paired(t, sc["B (ours) [cv]"][m], sc["Lookback [cv]"][m])
                cmp[f"{k} | B - GASP+base (truncated rows)"] = paired(t, sc["B (ours) [cv]"][m], sc["GASP+base"][m])
    real = [(ds, keep[(m, ds, "span")][0][keep[(m, ds, "span")][0]["trunc"].values.astype(bool)],
             keep[(m, ds, "span")][1]["B (ours) [cv]"][keep[(m, ds, "span")][0]["trunc"].values.astype(bool)],
             keep[(m, ds, "span")][1]["Lookback [cv]"][keep[(m, ds, "span")][0]["trunc"].values.astype(bool)])
            for ds in LONG for m in MODELS]
    real = [(ds, t.reset_index(drop=True), a, b) for ds, t, a, b in real]
    cmp["POOLED real long-context (TechQA, ExpertQA-long x 2) truncated rows | B - Lookback"] = pooled(real)
    main3 = [(ds, keep[(m, ds, "span")][0], keep[(m, ds, "span")][1]["Lookback [cv]"],
              keep[(m, ds, "span")][1]["GASP+base"]) for ds in MAIN for m in MODELS]
    cmp["POOLED all 5 datasets x 2 | Lookback - GASP+base"] = pooled(main3)
    fa3 = [(ds, keep[(m, ds, "span")][0], keep[(m, ds, "span")][1]["Frequency-aware [cv]"],
            keep[(m, ds, "span")][1]["Lookback [cv]"]) for ds in MAIN for m in MODELS
           if "Frequency-aware [cv]" in keep[(m, ds, "span")][1]]
    if fa3:
        cmp[f"POOLED {len(fa3)} runs with FA | FA - Lookback"] = pooled(fa3)

    # controlled truncation (512-token windows): window 1 vs B vs full view
    ctrl, ctrl_items = [], []
    for ds in ("ragtruth", "tofueval"):
        for model in MODELS:
            canon = CANON / f"{model}_{ds}_K5"
            f512 = FEAT / f"{model}_{ds}_K5" / "features_chunked_ctx512.npz"
            nwin = windows(model, ds, "features_chunked_ctx512.npz")
            sc = {}
            for name, npz, key in (("window 1", f512, "lookback"), ("B", f512, "lookback_max"),
                                   ("full view", FEAT / f"{model}_{ds}_K5" / "features.npz", "lookback")):
                d, c = load_signals(canon, npz, lb_keys=(key,))
                tr, te = split(d, args.eval_on)
                sc[name] = np.asarray(fit_linear(tr, te, c), float)
            m = np.array([nwin[(c_, s_)] > 1 for c_, s_ in zip(te["case_id"], te["sent_idx"])])
            t = te[m].reset_index(drop=True)
            a = {k2: auc(t["label"], v[m]) for k2, v in sc.items()}
            gap = a["full view"] - a["window 1"]
            ctrl.append(dict(model=model, dataset=ds, n=int(m.sum()), sources=int(t["source_id"].nunique()), **a,
                             recovered=(a["B"] - a["window 1"]) / gap if abs(gap) > 1e-3 else None,
                             diff=paired(t, sc["B"][m], sc["window 1"][m])))
            ctrl_items.append((ds, t, sc["B"][m], sc["window 1"][m]))
    cmp["POOLED controlled 512 (4 runs) | B - window 1"] = pooled(ctrl_items)

    # descriptive: B - window 1 by evidence location (RAGBench annotations), long-context sets
    ev = []
    if not args.no_evidence:
        try:
            import evidence_position as ep
            from transformers import AutoTokenizer
            for ds, (domain, min_chars) in ep.SETS.items():
                rb = ep.ragbench_rows(domain, min_chars)
                for model in MODELS:
                    te, sc = keep[(model, ds, "span")]
                    tok = AutoTokenizer.from_pretrained({"Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
                                                         "SmolLM2-1.7B-Instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct"}[model])
                    e = ep.case_evidence(CANON / f"{model}_{ds}_K5", rb, tok)
                    g = np.array([ep.group(e.get(c, {}).get("relevant", np.nan)) for c in te["case_id"]])
                    m = te["trunc"].values.astype(bool)
                    for grp in ("all out", "partly out", "in window 1"):
                        mm = m & (g == grp)
                        t = te[mm].reset_index(drop=True)
                        if t["label"].sum() >= 10 and (1 - t["label"]).sum() >= 10:
                            ev.append(dict(model=model, dataset=ds, group=grp, n=int(mm.sum()),
                                           window1=auc(t["label"], sc["Lookback [cv]"][mm]),
                                           B=auc(t["label"], sc["B (ours) [cv]"][mm]),
                                           diff=paired(t, sc["B (ours) [cv]"][mm], sc["Lookback [cv]"][mm])))
        except Exception as exc:
            say(f"(evidence part skipped: {type(exc).__name__}: {exc})")

    # ---------------------------------------------------------------- report
    pd.set_option("display.width", 250)
    for level in ("span", "response"):
        for rw in ("all", "truncated"):
            t = res[(res.level == level) & (res.rows == rw)]
            if len(t):
                say(f"\n## {level}-level AUC, {rw} rows")
                say(t.pivot_table(index=["dataset", "model"], columns="detector", values="auc").round(3).to_string())
    say("\n## Controlled truncation (512-token windows), truncated rows")
    for r in ctrl:
        say(f"  {r['model'].split('-')[0]:8s} {r['dataset']:9s} n={r['n']:5d} src={r['sources']:4d} | window 1 {r['window 1']:.3f}"
            f"  B {r['B']:.3f}  full {r['full view']:.3f} | B - window 1 {fmt(r['diff'])}"
            + (f" | recovers {r['recovered']:.0%}" if r["recovered"] is not None else ""))
    say("\n## Paired differences (GASP's source-level bootstrap; pooled: sources resampled per dataset)")
    for k, r in cmp.items():
        say(f"  {k}: {fmt(r)}")
    if ev:
        say("\n## Descriptive: B - window 1 on truncated rows by where the relevant evidence lies")
        for r in ev:
            say(f"  {r['model'].split('-')[0]:8s} {r['dataset']:13s} {r['group']:12s} n={r['n']:5d} | window 1 {r['window1']:.3f}"
                f"  B {r['B']:.3f} | {fmt(r['diff'])}")
    res.to_csv(out_dir / "test_look.csv", index=False)
    json.dump(dict(eval_on=args.eval_on, comparisons=cmp, controlled=ctrl, evidence=ev),
              open(out_dir / "test_look.json", "w"), indent=1, default=float)
    (out_dir / "test_look.txt").write_text("\n".join(lines) + "\n")
    print(f"\nsaved {out_dir}/test_look.{{txt,csv,json}}")


if __name__ == "__main__":
    main()
