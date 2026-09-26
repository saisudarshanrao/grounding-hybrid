"""Step E3 (CLAUDE.md, "E3 record"): when the evidence is moved out of window 1 on purpose, does B recover it?

Responses whose original context fits one window (RAGTruth, RAGBench; both scorers) are read with their context moved
behind 1808 tokens of other cases' contexts (scripts/extract_displaced.py). Three readings, the frozen classifier
(test_look.fit_linear) on each:
  orig   Lookback on the original context (features.npz): the upper bound
  w1d    Lookback on the displaced context's window 1 (distractor text only; features_displaced.npz, "lookback")
  Bd     B on the displaced context (same file, "lookback_max")
GASP's source split is taken on the FULL sentence table and then restricted to the included responses, so dev and
test are exactly GASP's.

Primary (the E3 rule), pooled over RAGTruth + RAGBench x 2 scorers (sources resampled per dataset, same draw for both
scorers, 2000 resamples):
    D3 = AUC(Bd) - AUC(w1d)
    lower bound > 0      -> "B recovers evidence moved out of window 1"
    interval contains 0  -> "not shown"
    upper bound < 0      -> "B loses to window 1"
Secondary (descriptive): share of the loss recovered, (Bd - w1d) / (orig - w1d); Bd - orig; orig - w1d; per dataset,
scorer and task.

--eval_on dev    out-of-fold scores from grouped 5-fold CV inside GASP's dev split: the E3 decision
--eval_on test   fit on dev, scored once on GASP's test split: test look P6. Run it ONCE, only after the dev decision
                 is recorded in CLAUDE.md, and log it there.

Usage:
    python -W ignore scripts/e3_displace.py --eval_on dev
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
from test_look import CANON, FEAT, MODELS, auc, fit_linear, fmt, paired, pooled  # noqa: E402

DATASETS = ("ragtruth", "ragbench")
NAMES = {"orig": "original", "w1d": "window 1 (displaced)", "Bd": "B (displaced)"}


def frame(model, ds):
    """(dev, test) sentence frames of the included responses, GASP's split, with the three feature blocks."""
    canon = CANON / f"{model}_{ds}_K5"
    df, c = load_signals(canon, FEAT / canon.name / "features.npz", lb_keys=("lookback",))
    cols = {"orig": [f"orig_{i}" for i in range(len(c))]}
    df = df.rename(columns=dict(zip(c, cols["orig"])))
    z = np.load(FEAT / canon.name / "features_displaced.npz")
    L = len(z["case_id"])
    disp = pd.DataFrame({"case_id": z["case_id"], "sent_idx": z["sent_idx"]})
    for name, key in (("w1d", "lookback"), ("Bd", "lookback_max")):
        a = z[key].astype(np.float32).reshape(L, -1)
        cols[name] = [f"{name}_{i}" for i in range(a.shape[1])]
        disp = pd.concat([disp, pd.DataFrame(a, columns=cols[name])], axis=1)
    inc = set(z["case_id"])
    dev, test = analyze_gasp.source_split(df, seed=0)          # on the FULL table: GASP's exact split
    out = []
    for part in (dev, test):
        part = part[part["case_id"].isin(inc)]
        merged = part.merge(disp, on=["case_id", "sent_idx"], how="left", sort=False)
        assert len(merged) == len(part) and (merged["case_id"].values == part["case_id"].values).all()
        if merged[cols["w1d"][0]].isna().any():
            raise ValueError(f"{model} {ds}: displaced features miss sentences of included responses")
        out.append(merged.reset_index(drop=True))
    return out[0], out[1], cols


def score(dev, test, cols, eval_on):
    if eval_on == "test":
        return test, {n: np.asarray(fit_linear(dev, test, c), float) for n, c in cols.items()}
    s = {n: np.zeros(len(dev)) for n in cols}
    for tr, va in _folds(dev, 5):
        for n, c in cols.items():
            s[n][va] = fit_linear(dev.iloc[tr], dev.iloc[va], c)
    return dev, s


def verdict(r):
    if r["lo"] > 0:
        return "B recovers evidence moved out of window 1"
    if r["hi"] < 0:
        return "B loses to window 1"
    return "not shown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_on", choices=["dev", "test"], required=True)
    args = ap.parse_args()
    out_dir = ROOT / "results" / ("e3" if args.eval_on == "dev" else "test")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "e3_dev" if args.eval_on == "dev" else "e3_test_look_P6"
    lines, report = [], dict(eval_on=args.eval_on)
    say = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731
    say(f"# E3: evidence displaced out of window 1, eval_on={args.eval_on} "
        + ("(grouped 5-fold out-of-fold inside dev)" if args.eval_on == "dev" else "(fit on dev, scored on TEST)"))

    runs, dump = {}, []
    for ds in DATASETS:
        for model in MODELS:
            dev, test, cols = frame(model, ds)
            ev, sc = score(dev, test, cols, args.eval_on)
            runs[(model, ds)] = (ev, sc)
            dump.append(ev[["case_id", "sent_idx", "source_id", "label", "task"]]
                        .assign(model=model, dataset=ds, **{NAMES[n]: s for n, s in sc.items()}))
            print(f"done {model} {ds}", flush=True)

    rows = []
    for (model, ds), (ev, sc) in runs.items():
        groups = [("all", np.ones(len(ev), bool))] + [(f"task {t}", (ev["task"] == t).values)
                                                      for t in sorted(ev["task"].unique())]
        for g, m in groups:
            y = ev["label"].values[m]
            if m.sum() and y.min() != y.max():
                a = {n: auc(y, s[m]) for n, s in sc.items()}
                loss = a["orig"] - a["w1d"]
                rows.append(dict(dataset=ds, model=model.split("-")[0], rows=g, n=int(m.sum()),
                                 sources=int(ev.loc[m, "source_id"].nunique()), halluc=round(float(y.mean()), 3),
                                 **{NAMES[n]: round(v, 3) for n, v in a.items()},
                                 recovered=round((a["Bd"] - a["w1d"]) / loss, 2) if abs(loss) > 1e-3 else None))
    pd.set_option("display.width", 220)
    say("\n## Sentence-level AUC (recovered = share of the original -> displaced-window-1 loss that B recovers)")
    say(pd.DataFrame(rows).to_string(index=False))

    def items(a, b):
        return [(ds, ev, sc[a], sc[b]) for (model, ds), (ev, sc) in runs.items()]

    D3 = pooled(items("Bd", "w1d"))
    report["primary"] = dict(D3=D3, verdict=verdict(D3))
    say("\n## PRIMARY (E3 rule): pooled over RAGTruth + RAGBench x 2 scorers, 2000 resamples")
    say(f"  D3 = AUC(Bd) - AUC(w1d) = {fmt(D3)}  ->  {verdict(D3)}")

    sec = {"orig - w1d (the loss)": pooled(items("orig", "w1d")),
           "Bd - orig": pooled(items("Bd", "orig"))}
    loss, gain = sec["orig - w1d (the loss)"]["mean"], D3["mean"]
    sec["recovered share (pooled point estimate)"] = dict(mean=gain / loss if abs(loss) > 1e-3 else float("nan"),
                                                          lo=float("nan"), hi=float("nan"), sig=False)
    for (model, ds), (ev, sc) in runs.items():
        k = f"{model.split('-')[0]} {ds}"
        sec[f"{k} | Bd - w1d"] = paired(ev, sc["Bd"], sc["w1d"])
        sec[f"{k} | orig - w1d"] = paired(ev, sc["orig"], sc["w1d"])
        sec[f"{k} | Bd - orig"] = paired(ev, sc["Bd"], sc["orig"])
    report["secondary"] = sec
    say("\n## Secondary (descriptive)")
    for k, r in sec.items():
        say(f"  {k}: " + (f"{r['mean']:.2f}" if k.startswith("recovered") else fmt(r)))

    report["table"] = rows
    json.dump(report, open(out_dir / f"{stem}.json", "w"), indent=1, default=float)
    (out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    pd.concat(dump, ignore_index=True).to_csv(out_dir / f"{stem}_scores.csv.gz", index=False)
    print(f"\nsaved {out_dir}/{stem}.{{txt,json}} and {stem}_scores.csv.gz")


if __name__ == "__main__":
    main()
