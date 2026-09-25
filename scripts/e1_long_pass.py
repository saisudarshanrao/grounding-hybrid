"""Step E1 (CLAUDE.md, "E1 record"): B vs one long single pass (L) on the long-context sets.

L = the frozen Lookback features from ONE forward pass over as much of the context as the scorer's positions allow
(scripts/extract_long.py): Qwen2.5-1.5B reads every context whole, SmolLM2-1.7B cuts contexts beyond ~8k tokens.
All three detectors are the frozen one (test_look.fit_linear: standardized logistic regression, balanced classes,
C by grouped 5-fold CV on the training part); only the features differ:
  window 1   Lookback on GASP's retained context (features_chunked_redeep.npz, "lookback")
  B          max over 1800-token windows (same file, "lookback_max")
  L          one long pass (features_long.npz, "lookback")

Primary (the E1 rule), on sentences whose context needs more than one 1800-token window, pooled over TechQA and
ExpertQA-long x 2 scorers (sources resampled per dataset, same draw for both scorers, 2000 resamples):
    D = AUC(B) - AUC(L)
    lower bound > 0         -> "B beats one long pass"
    interval contains 0     -> "B matches one long pass with fixed memory"
    upper bound < 0         -> "one long pass beats B"
The primary test uses every such row, including SmolLM2 rows whose context its 8k limit cuts; those rows are also
reported on their own, and D without them is shown (descriptive).
Secondary (descriptive, no rule): L - window 1 and B - window 1 on the same rows; per dataset and scorer; L's
seconds and peak GPU memory by sequence length (meta_long.json).

--eval_on dev    out-of-fold scores from grouped 5-fold CV inside GASP's dev split: the E1 decision
--eval_on test   fit on the whole dev split, scored once on GASP's test split: test look P4. Run it ONCE, only
                 after the dev decision is recorded in CLAUDE.md, and log it there.

Usage:
    python -W ignore scripts/e1_long_pass.py --eval_on dev
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grounding_hybrid.gasp_bridge import analyze_gasp  # noqa: E402
from grounding_hybrid.gating import _folds  # noqa: E402
from grounding_hybrid.signals import load_signals  # noqa: E402
from test_look import CANON, FEAT, LONG, MODELS, auc, fit_linear, fmt, paired, pooled  # noqa: E402

FEATS = {"lb": ("features_chunked_redeep.npz", "lookback"),     # window 1
         "B": ("features_chunked_redeep.npz", "lookback_max"),
         "L": ("features_long.npz", "lookback")}
NAMES = {"lb": "window 1", "B": "B", "L": "L"}
SEQ_BINS = [0, 2048, 4096, 8192, 16384, 32768]


def frame(model, ds):
    """Sentence frame (sentence.csv order) with the three feature blocks, trunc (>1 window) and cut (L cut)."""
    canon = CANON / f"{model}_{ds}_K5"
    f = FEAT / canon.name
    df, cols = None, {}
    for name, (fname, key) in FEATS.items():
        d, c = load_signals(canon, f / fname, lb_keys=(key,))
        new = [f"{name}_{i}" for i in range(len(c))]
        part = d[c].set_axis(new, axis=1)
        if df is None:
            df = d.drop(columns=c)
        assert (d["case_id"].values == df["case_id"].values).all()
        df = pd.concat([df, part], axis=1)
        cols[name] = new
    z = np.load(f / "features_chunked_redeep.npz")
    zl = np.load(f / "features_long.npz")
    nwin = dict(zip(zip(z["case_id"], z["sent_idx"]), z["n_windows"]))
    cut = dict(zip(zip(zl["case_id"], zl["sent_idx"]), zl["ctx_cut"]))
    keys = list(zip(df["case_id"], df["sent_idx"]))
    df["trunc"] = [bool(nwin[k] > 1) for k in keys]
    df["cut"] = [bool(cut[k]) for k in keys]
    return df, cols


def score(df, cols, eval_on):
    """(evaluated rows, {block: scores}): out-of-fold on dev, or dev-fit scores on test."""
    dev, test = analyze_gasp.source_split(df, seed=0)
    dev = dev.reset_index(drop=True)
    if eval_on == "test":
        te = test.reset_index(drop=True)
        return te, {n: np.asarray(fit_linear(dev, te, c), float) for n, c in cols.items()}
    s = {n: np.zeros(len(dev)) for n in cols}
    for tr, va in _folds(dev, 5):
        for n, c in cols.items():
            s[n][va] = fit_linear(dev.iloc[tr], dev.iloc[va], c)
    return dev, s


def verdict(r):
    if r["lo"] > 0:
        return "B beats one long pass"
    if r["hi"] < 0:
        return "one long pass beats B"
    return "B matches one long pass with fixed memory"


def cost_by_length(model, ds):
    meta = json.load(open(FEAT / f"{model}_{ds}_K5" / "meta_long.json"))
    pc = pd.DataFrame(meta["per_case"])
    pc["bin"] = pd.cut(pc["seq_len"], SEQ_BINS)
    g = pc.groupby("bin", observed=True)
    return pd.DataFrame(dict(cases=g.size(), seconds_median=g["seconds"].median(),
                             peak_gb_median=g["peak_gb"].median(), peak_gb_max=g["peak_gb"].max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_on", choices=["dev", "test"], required=True)
    args = ap.parse_args()
    out_dir = ROOT / "results" / ("e1" if args.eval_on == "dev" else "test")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "e1_dev" if args.eval_on == "dev" else "e1_test_look_P4"
    lines, report = [], dict(eval_on=args.eval_on)
    say = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731
    say(f"# E1: B vs one long pass (L), eval_on={args.eval_on} "
        + ("(grouped 5-fold out-of-fold inside dev)" if args.eval_on == "dev" else "(fit on dev, scored on TEST)"))

    runs, dump = {}, []
    for ds in LONG:
        for model in MODELS:
            df, cols = frame(model, ds)
            ev, sc = score(df, cols, args.eval_on)
            runs[(model, ds)] = (ev, sc)
            dump.append(ev[["case_id", "sent_idx", "source_id", "label", "trunc", "cut"]]
                        .assign(model=model, dataset=ds, **{NAMES[n]: s for n, s in sc.items()}))
            print(f"done {model} {ds}", flush=True)

    # per run: AUC of each block on all / truncated / truncated-not-cut / cut rows
    rows = []
    for (model, ds), (ev, sc) in runs.items():
        tr, ct = ev["trunc"].values, ev["cut"].values
        for rw, m in (("all", np.ones(len(ev), bool)), ("truncated", tr), ("truncated, not cut", tr & ~ct),
                      ("cut by L", ct)):
            y = ev["label"].values[m]
            if m.sum() and y.min() != y.max():
                rows.append(dict(dataset=ds, model=model.split("-")[0], rows=rw, n=int(m.sum()),
                                 sources=int(ev.loc[m, "source_id"].nunique()), halluc=round(float(y.mean()), 3),
                                 **{NAMES[n]: round(auc(y, s[m]), 3) for n, s in sc.items()}))
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    say("\n## Sentence-level AUC by rows")
    say(table.to_string(index=False))

    def items(a, b, keep):
        out = []
        for (model, ds), (ev, sc) in runs.items():
            m = keep(ev)
            out.append((ds, ev[m].reset_index(drop=True), sc[a][m], sc[b][m]))
        return out

    trunc = lambda ev: ev["trunc"].values  # noqa: E731
    trunc_not_cut = lambda ev: ev["trunc"].values & ~ev["cut"].values  # noqa: E731

    # primary
    D = pooled(items("B", "L", trunc))
    report["primary"] = dict(D=D, verdict=verdict(D))
    say("\n## PRIMARY (E1 rule): pooled truncated rows, TechQA + ExpertQA-long x 2 scorers, 2000 resamples")
    say(f"  D = AUC(B) - AUC(L) = {fmt(D)}  ->  {verdict(D)}")

    # secondary
    sec = {"L - window 1 (truncated rows)": pooled(items("L", "lb", trunc)),
           "B - window 1 (truncated rows)": pooled(items("B", "lb", trunc)),
           "B - L (truncated rows, SmolLM2 cut rows removed)": pooled(items("B", "L", trunc_not_cut))}
    for (model, ds), (ev, sc) in runs.items():
        k = f"{model.split('-')[0]} {ds}"
        for label, keep in (("truncated", trunc), ("cut by L", lambda e: e["cut"].values)):
            m = keep(ev)
            t = ev[m]
            if m.sum() and t["label"].nunique() == 2:
                sec[f"{k} | B - L ({label})"] = paired(t, sc["B"][m], sc["L"][m])
                sec[f"{k} | L - window 1 ({label})"] = paired(t, sc["L"][m], sc["lb"][m])
    report["secondary"] = sec
    say("\n## Secondary (descriptive)")
    for k, r in sec.items():
        say(f"  {k}: {fmt(r)}")

    # cost of L by sequence length
    say("\n## L cost by sequence length (per response; peak = GPU memory during the pass)")
    report["cost"] = {}
    for (model, ds) in runs:
        c = cost_by_length(model, ds)
        report["cost"][f"{model} {ds}"] = c.reset_index().astype({"bin": str}).to_dict(orient="records")
        say(f"  {model.split('-')[0]} {ds}")
        say("    " + c.round(2).to_string().replace("\n", "\n    "))

    report["table"] = rows
    json.dump(report, open(out_dir / f"{stem}.json", "w"), indent=1, default=float)
    (out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    pd.concat(dump, ignore_index=True).to_csv(out_dir / f"{stem}_scores.csv.gz", index=False)
    print(f"\nsaved {out_dir}/{stem}.{{txt,json}} and {stem}_scores.csv.gz")


if __name__ == "__main__":
    main()
