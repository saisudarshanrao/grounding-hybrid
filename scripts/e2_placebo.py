"""Step E2 (CLAUDE.md, "E2 record"): does B's gain on TechQA need the real context beyond window 1?

Bp = the placebo reading (scripts/extract_placebo.py): B with windows 2..K taken from a donor case of the same split
and another source. All three detectors are the frozen one (test_look.fit_linear); only the features differ:
  window 1   Lookback on GASP's retained context (features_chunked_redeep.npz, "lookback")
  B          max over the case's own 1800-token windows (same file, "lookback_max")
  Bp         max over window 1 + the donor's windows 2..K (features_placebo.npz, "lookback_placebo_max")

Primary (the E2 rule), TechQA sentences whose context needs more than one window, pooled over the 2 scorers (sources
resampled with the same draw for both scorers, 2000 resamples):
    D2 = AUC(B) - AUC(Bp)
    lower bound > 0       -> "B's gain needs the real context beyond window 1"
    interval contains 0   -> "the placebo matches B: B's gain is not shown to come from the context beyond window 1"
    upper bound < 0       -> "the placebo beats B"
Secondary (descriptive): Bp - window 1; B - window 1; share of B's gain the placebo keeps; per scorer; a fixed-
classifier check (B's classifier applied to Bp features); B - Bp and Bp - window 1 by where RAGBench's annotated
evidence lies.

--eval_on dev    out-of-fold scores from grouped 5-fold CV inside GASP's dev split: the E2 decision
--eval_on test   fit on the whole dev split, scored once on GASP's test split: test look P5. Run it ONCE, only
                 after the dev decision is recorded in CLAUDE.md, and log it there.

Usage:
    python -W ignore scripts/e2_placebo.py --eval_on dev
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

DS = "techqa"
FEATS = {"lb": ("features_chunked_redeep.npz", "lookback"),      # window 1
         "B": ("features_chunked_redeep.npz", "lookback_max"),
         "Bp": ("features_placebo.npz", "lookback_placebo_max")}
NAMES = {"lb": "window 1", "B": "B", "Bp": "Bp (placebo)", "Bfix": "B classifier on Bp"}


def frame(model):
    """Sentence frame (sentence.csv order) with the three feature blocks and trunc (> 1 window)."""
    canon = CANON / f"{model}_{DS}_K5"
    f = FEAT / canon.name
    df, cols = None, {}
    for name, (fname, key) in FEATS.items():
        d, c = load_signals(canon, f / fname, lb_keys=(key,))
        new = [f"{name}_{i}" for i in range(len(c))]
        if df is None:
            df = d.drop(columns=c)
        assert (d["case_id"].values == df["case_id"].values).all()
        df = pd.concat([df, d[c].set_axis(new, axis=1)], axis=1)
        cols[name] = new
    z = np.load(f / "features_chunked_redeep.npz")
    zp = np.load(f / "features_placebo.npz")
    nwin = dict(zip(zip(z["case_id"], z["sent_idx"]), z["n_windows"]))
    pwin = dict(zip(zip(zp["case_id"], zp["sent_idx"]), zp["n_windows"]))
    keys = list(zip(df["case_id"], df["sent_idx"]))
    assert all(nwin[k] == pwin[k] for k in keys), "placebo and B disagree on the number of windows"
    df["trunc"] = [bool(nwin[k] > 1) for k in keys]
    return df, cols


def with_features(df, src, dst):
    """Copy of df whose dst columns hold the src columns' values (a classifier trained on dst reads src)."""
    out = df.copy()
    out[dst] = df[src].to_numpy()
    return out


def score(df, cols, eval_on):
    """(evaluated rows, {block: scores}) incl. Bfix: out-of-fold on dev, or dev-fit scores on test."""
    dev, test = analyze_gasp.source_split(df, seed=0)
    dev = dev.reset_index(drop=True)
    if eval_on == "test":
        te = test.reset_index(drop=True)
        s = {n: np.asarray(fit_linear(dev, te, c), float) for n, c in cols.items()}
        s["Bfix"] = np.asarray(fit_linear(dev, with_features(te, cols["Bp"], cols["B"]), cols["B"]), float)
        return te, s
    s = {n: np.zeros(len(dev)) for n in list(cols) + ["Bfix"]}
    for tr, va in _folds(dev, 5):
        for n, c in cols.items():
            s[n][va] = fit_linear(dev.iloc[tr], dev.iloc[va], c)
        s["Bfix"][va] = fit_linear(dev.iloc[tr], with_features(dev.iloc[va], cols["Bp"], cols["B"]), cols["B"])
    return dev, s


def verdict(r):
    if r["lo"] > 0:
        return "B's gain needs the real context beyond window 1"
    if r["hi"] < 0:
        return "the placebo beats B"
    return "the placebo matches B: B's gain is not shown to come from the context beyond window 1"


def evidence_groups(runs, lines):
    """Descriptive: B - Bp and Bp - window 1 on truncated rows by where RAGBench's annotated evidence lies."""
    import evidence_position as ep
    from transformers import AutoTokenizer
    domain, min_chars = ep.SETS[DS]
    rb = ep.ragbench_rows(domain, min_chars)
    out = []
    for model, (ev, sc) in runs.items():
        tok = AutoTokenizer.from_pretrained({"Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
                                             "SmolLM2-1.7B-Instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct"}[model])
        e = ep.case_evidence(CANON / f"{model}_{DS}_K5", rb, tok)
        g = np.array([ep.group(e.get(c, {}).get("relevant", np.nan)) for c in ev["case_id"]])
        m = ev["trunc"].values.astype(bool)
        for grp in ("all out", "partly out", "in window 1"):
            mm = m & (g == grp)
            t = ev[mm].reset_index(drop=True)
            if t["label"].sum() >= 10 and (1 - t["label"]).sum() >= 10:
                r = dict(model=model, group=grp, n=int(mm.sum()),
                         **{NAMES[k]: auc(t["label"], sc[k][mm]) for k in ("lb", "B", "Bp")},
                         B_minus_Bp=paired(t, sc["B"][mm], sc["Bp"][mm]),
                         Bp_minus_w1=paired(t, sc["Bp"][mm], sc["lb"][mm]))
                out.append(r)
                lines.append(f"  {model.split('-')[0]:8s} {grp:12s} n={r['n']:5d} | window 1 {r['window 1']:.3f}  "
                             f"B {r['B']:.3f}  Bp {r['Bp (placebo)']:.3f} | B - Bp {fmt(r['B_minus_Bp'])} | "
                             f"Bp - window 1 {fmt(r['Bp_minus_w1'])}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_on", choices=["dev", "test"], required=True)
    ap.add_argument("--no_evidence", action="store_true", help="skip the evidence-position part")
    args = ap.parse_args()
    out_dir = ROOT / "results" / ("e2" if args.eval_on == "dev" else "test")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "e2_dev" if args.eval_on == "dev" else "e2_test_look_P5"
    lines, report = [], dict(eval_on=args.eval_on)
    say = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731
    say(f"# E2: placebo windows on TechQA, eval_on={args.eval_on} "
        + ("(grouped 5-fold out-of-fold inside dev)" if args.eval_on == "dev" else "(fit on dev, scored on TEST)"))

    runs, dump = {}, []
    for model in MODELS:
        df, cols = frame(model)
        ev, sc = score(df, cols, args.eval_on)
        runs[model] = (ev, sc)
        dump.append(ev[["case_id", "sent_idx", "source_id", "label", "trunc"]]
                    .assign(model=model, dataset=DS, **{NAMES[n]: s for n, s in sc.items()}))
        print(f"done {model}", flush=True)

    rows = []
    for model, (ev, sc) in runs.items():
        for rw, m in (("all", np.ones(len(ev), bool)), ("truncated", ev["trunc"].values)):
            y = ev["label"].values[m]
            a = {NAMES[n]: round(auc(y, s[m]), 3) for n, s in sc.items()}
            gain = a["B"] - a["window 1"]
            rows.append(dict(model=model.split("-")[0], rows=rw, n=int(m.sum()),
                             sources=int(ev.loc[m, "source_id"].nunique()), halluc=round(float(y.mean()), 3), **a,
                             placebo_keeps=round((a["Bp (placebo)"] - a["window 1"]) / gain, 2) if abs(gain) > 1e-3 else None))
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    say("\n## TechQA sentence-level AUC by rows (placebo_keeps = share of B's gain over window 1 kept by Bp)")
    say(table.to_string(index=False))

    def items(a, b):
        out = []
        for model, (ev, sc) in runs.items():
            m = ev["trunc"].values
            out.append((DS, ev[m].reset_index(drop=True), sc[a][m], sc[b][m]))
        return out

    D2 = pooled(items("B", "Bp"))
    report["primary"] = dict(D2=D2, verdict=verdict(D2))
    say("\n## PRIMARY (E2 rule): pooled truncated TechQA rows, 2 scorers, 2000 resamples")
    say(f"  D2 = AUC(B) - AUC(Bp) = {fmt(D2)}  ->  {verdict(D2)}")

    sec = {"Bp - window 1 (truncated rows)": pooled(items("Bp", "lb")),
           "B - window 1 (truncated rows)": pooled(items("B", "lb")),
           "B classifier on Bp - window 1 (truncated rows)": pooled(items("Bfix", "lb")),
           "B - B classifier on Bp (truncated rows)": pooled(items("B", "Bfix"))}
    for model, (ev, sc) in runs.items():
        m = ev["trunc"].values
        t = ev[m]
        k = model.split("-")[0]
        sec[f"{k} | B - Bp"] = paired(t, sc["B"][m], sc["Bp"][m])
        sec[f"{k} | Bp - window 1"] = paired(t, sc["Bp"][m], sc["lb"][m])
        sec[f"{k} | B - window 1"] = paired(t, sc["B"][m], sc["lb"][m])
    report["secondary"] = sec
    say("\n## Secondary (descriptive)")
    for k, r in sec.items():
        say(f"  {k}: {fmt(r)}")

    if not args.no_evidence:
        say("\n## Descriptive: by where RAGBench's annotated evidence lies (truncated rows)")
        try:
            report["evidence"] = evidence_groups(runs, lines)
            for ln in lines[-len(report["evidence"]):]:
                print(ln, flush=True)
        except Exception as exc:
            say(f"  (evidence part skipped: {type(exc).__name__}: {exc})")

    report["table"] = rows
    json.dump(report, open(out_dir / f"{stem}.json", "w"), indent=1, default=float)
    (out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    pd.concat(dump, ignore_index=True).to_csv(out_dir / f"{stem}_scores.csv.gz", index=False)
    print(f"\nsaved {out_dir}/{stem}.{{txt,json}} and {stem}_scores.csv.gz")


if __name__ == "__main__":
    main()
