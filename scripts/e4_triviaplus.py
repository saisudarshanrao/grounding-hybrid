"""Step E4 (CLAUDE.md, "E4 record"): does B's gain over Lookback replicate on TRIVIA+-long (human sentence labels)?

Detectors are the frozen ones (test_look.fit_linear: standardized logistic regression, balanced classes, C by grouped
5-fold CV on the training part, GASP's own classifier for GASP+base / perplexity); only the features differ:
  Lookback (window 1)   features_chunked_redeep.npz "lookback"
  B                     features_chunked_redeep.npz "lookback_max" (max over 1800/256 windows)
  L (one long pass)     features_long.npz "lookback"
  ReDeEP                features_chunked_redeep.npz "ecs" + "pks" (window 1)
  GASP+base, perplexity sentence.csv

Primary (the E4 rule), on TRIVIA+-long sentences whose context needs more than one window, pooled over the 2 scorers
(sources resampled, same draw for both scorers, 2000 resamples):
    D4 = AUC(B) - AUC(Lookback)
    lower bound > 0       -> "B's gain replicates on TRIVIA+"
    interval contains 0   -> "not shown on TRIVIA+"
    upper bound < 0       -> "B loses to Lookback on TRIVIA+"
Secondary (descriptive, no rule): per scorer; per source benchmark; GASP+base, ReDeEP and L next to B (B - L as in E1);
response level; AUC by the share of the article inside window 1.

--eval_on dev      out-of-fold scores from grouped 5-fold CV inside GASP's dev split: the E4 decision
--eval_on test     fit on the whole dev split, scored once on GASP's test split: test look P7. Run it ONCE, only after
                   the dev decision is recorded in CLAUDE.md, and log it there.
--eval_on dryrun   the test branch on two halves of dev (fit on one half, score the other): a code check only.
--canon_root / --feat_root / --dataset point the script at other run folders (dry runs in the scratchpad).

Usage:
    python -W ignore scripts/e4_triviaplus.py --eval_on dev
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

from eval_redeep import ReDeEP  # noqa: E402
from grounding_hybrid.gasp_bridge import BASE_FEATS, GASP_FEATS, analyze_gasp  # noqa: E402
from grounding_hybrid.gating import _folds  # noqa: E402
from grounding_hybrid.signals import load_signals  # noqa: E402
from test_look import CANON, FEAT, MODELS, auc, fit_linear, fmt, paired, pooled  # noqa: E402

BLOCKS = {"lb": ("features_chunked_redeep.npz", "lookback"), "B": ("features_chunked_redeep.npz", "lookback_max"),
          "ecs": ("features_chunked_redeep.npz", "ecs"), "pks": ("features_chunked_redeep.npz", "pks"),
          "L": ("features_long.npz", "lookback")}
DETECTORS = {
    "Perplexity+length": lambda tr, te, c: fit_linear(tr, te, BASE_FEATS, False),
    "GASP+base": lambda tr, te, c: fit_linear(tr, te, GASP_FEATS + BASE_FEATS, False),
    "ReDeEP": lambda tr, te, c: ReDeEP(c["ecs"], c["pks"]).fit(tr).score(te),
    "ReDeEP [cv]": lambda tr, te, c: fit_linear(tr, te, c["ecs"] + c["pks"]),
    "Lookback [cv]": lambda tr, te, c: fit_linear(tr, te, c["lb"]),
    "L (one pass) [cv]": lambda tr, te, c: fit_linear(tr, te, c["L"]),
    "B (ours) [cv]": lambda tr, te, c: fit_linear(tr, te, c["B"]),
}
KEPT_BINS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]


def frame(canon_root, feat_root, model, ds):
    """Sentence frame (sentence.csv order) with GASP / base columns, every feature block, trunc and cut flags."""
    canon, f = Path(canon_root) / f"{model}_{ds}_K5", Path(feat_root) / f"{model}_{ds}_K5"
    df, _ = load_signals(canon)
    cols = {}
    for name, (fname, key) in BLOCKS.items():
        d, c = load_signals(canon, f / fname, lb_keys=(key,))
        assert (d["case_id"].values == df["case_id"].values).all()
        new = [f"{name}_{i}" for i in range(len(c))]
        df = pd.concat([df, d[c].set_axis(new, axis=1)], axis=1)
        cols[name] = new
    z, zl = np.load(f / "features_chunked_redeep.npz"), np.load(f / "features_long.npz")
    nwin = dict(zip(zip(z["case_id"], z["sent_idx"]), z["n_windows"]))
    cut = dict(zip(zip(zl["case_id"], zl["sent_idx"]), zl["ctx_cut"]))
    keys = list(zip(df["case_id"], df["sent_idx"]))
    df["trunc"] = [bool(nwin[k] > 1) for k in keys]
    df["cut"] = [bool(cut[k]) for k in keys]
    return df, cols


def score(df, cols, eval_on):
    """(evaluated rows, {detector: scores}): out-of-fold inside dev, dev-fit scores on test, or dev halves."""
    dev, test = analyze_gasp.source_split(df, seed=0)          # on the FULL table: GASP's exact split
    dev = dev.reset_index(drop=True)
    if eval_on in ("test", "dryrun"):
        if eval_on == "test":
            tr, te = dev, test.reset_index(drop=True)
        else:
            srcs = np.sort(dev["source_id"].unique())
            half = set(np.random.default_rng(0).permutation(srcs)[: len(srcs) // 2])
            tr = dev[dev["source_id"].isin(half)].reset_index(drop=True)
            te = dev[~dev["source_id"].isin(half)].reset_index(drop=True)
        return te, {n: np.asarray(fn(tr, te, cols), float) for n, fn in DETECTORS.items()}
    s = {n: np.zeros(len(dev)) for n in DETECTORS}
    for tr, va in _folds(dev, 5):
        for n, fn in DETECTORS.items():
            s[n][va] = fn(dev.iloc[tr], dev.iloc[va], cols)
    return dev, s


def verdict(r):
    if r["lo"] > 0:
        return "B's gain replicates on TRIVIA+"
    if r["hi"] < 0:
        return "B loses to Lookback on TRIVIA+"
    return "not shown on TRIVIA+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_on", choices=["dev", "test", "dryrun"], required=True)
    ap.add_argument("--canon_root", default=str(CANON))
    ap.add_argument("--feat_root", default=str(FEAT))
    ap.add_argument("--dataset", default="triviapluslong")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / ("test" if args.eval_on == "test" else "e4")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = {"dev": "e4_dev", "test": "e4_test_look_P7", "dryrun": "e4_dryrun"}[args.eval_on]
    lines, report = [], dict(eval_on=args.eval_on, dataset=args.dataset)
    say = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731
    say(f"# E4: B vs Lookback on {args.dataset}, eval_on={args.eval_on} " + {
        "dev": "(grouped 5-fold out-of-fold inside dev)", "test": "(fit on dev, scored on TEST)",
        "dryrun": "(fit on one half of dev, scored on the other: code check only)"}[args.eval_on])

    runs, dump = {}, []
    for model in MODELS:
        df, cols = frame(args.canon_root, args.feat_root, model, args.dataset)
        ev, sc = score(df, cols, args.eval_on)
        runs[model] = (ev, sc)
        dump.append(ev[["case_id", "sent_idx", "source_id", "task", "label", "trunc", "cut", "ctx_kept"]]
                    .assign(model=model, **sc))
        print(f"done {model}", flush=True)

    say("\n## Power (evaluated rows)")
    report["power"] = {}
    for model, (ev, _) in runs.items():
        t = ev[ev["trunc"]]
        p = dict(sentences=int(len(ev)), truncated=int(len(t)), sources=int(ev["source_id"].nunique()),
                 halluc=round(float(ev["label"].mean()), 3))
        report["power"][model] = p
        say(f"  {model}: {p}")

    trunc = lambda ev: ev["trunc"].values  # noqa: E731

    def items(a, b, keep):
        out = []
        for model, (ev, sc) in runs.items():
            m = keep(ev)
            out.append((args.dataset, ev[m].reset_index(drop=True), sc[a][m], sc[b][m]))
        return out

    D = pooled(items("B (ours) [cv]", "Lookback [cv]", trunc))
    report["primary"] = dict(D4=D, verdict=verdict(D))
    say("\n## PRIMARY (E4 rule): truncated rows, pooled over 2 scorers, 2000 resamples")
    say(f"  D4 = AUC(B) - AUC(Lookback) = {fmt(D)}  ->  {verdict(D)}")

    say("\n## Sentence-level AUC per scorer (truncated rows)")
    rows = []
    for model, (ev, sc) in runs.items():
        m = trunc(ev)
        y = ev["label"].values[m]
        rows.append(dict(model=model.split("-")[0], n=int(m.sum()), **{n: round(auc(y, s[m]), 3) for n, s in sc.items()}))
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    say(table.to_string(index=False))
    report["table"] = rows

    sec = {"B - L (truncated rows), pooled": pooled(items("B (ours) [cv]", "L (one pass) [cv]", trunc)),
           "Lookback - GASP+base (truncated rows), pooled": pooled(items("Lookback [cv]", "GASP+base", trunc))}
    for model, (ev, sc) in runs.items():
        k = model.split("-")[0]
        m = trunc(ev)
        t = ev[m]
        sec[f"{k} | B - Lookback"] = paired(t, sc["B (ours) [cv]"][m], sc["Lookback [cv]"][m])
        sec[f"{k} | B - L"] = paired(t, sc["B (ours) [cv]"][m], sc["L (one pass) [cv]"][m])
        for task in sorted(t["task"].unique()):
            mt = m & (ev["task"].values == task)
            tt = ev[mt]
            if tt["label"].nunique() == 2 and tt["source_id"].nunique() >= 5:
                sec[f"{k} | {task} | B - Lookback (n={int(mt.sum())})"] = paired(
                    tt, sc["B (ours) [cv]"][mt], sc["Lookback [cv]"][mt])
    report["secondary"] = sec
    say("\n## Secondary (descriptive)")
    for k, r in sec.items():
        say(f"  {k}: {fmt(r)}")

    say("\n## AUC by the share of the article inside window 1 (truncated rows)")
    report["by_kept"] = []
    for model, (ev, sc) in runs.items():
        m = trunc(ev)
        t = ev[m].assign(bin=pd.cut(ev.loc[m, "ctx_kept"], KEPT_BINS))
        for b, g in t.groupby("bin", observed=True):
            idx = g.index.values
            if g["label"].nunique() == 2:
                r = dict(model=model.split("-")[0], kept=str(b), n=len(g),
                         Lookback=round(auc(g["label"], sc["Lookback [cv]"][idx]), 3),
                         B=round(auc(g["label"], sc["B (ours) [cv]"][idx]), 3))
                report["by_kept"].append(r)
                say(f"  {r}")

    say("\n## Response level (max over sentences of each detector's score)")
    report["response"] = []
    for model, (ev, sc) in runs.items():
        g = pd.DataFrame({"case_id": ev["case_id"], "label": ev["label"], **sc}).groupby("case_id").max()
        r = dict(model=model.split("-")[0], responses=len(g), **{n: round(auc(g["label"], g[n]), 3) for n in sc})
        report["response"].append(r)
        say(f"  {r}")

    json.dump(report, open(out_dir / f"{stem}.json", "w"), indent=1, default=float)
    (out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    pd.concat(dump, ignore_index=True).to_csv(out_dir / f"{stem}_scores.csv.gz", index=False)
    print(f"\nsaved {out_dir}/{stem}.{{txt,json}} and {stem}_scores.csv.gz")


if __name__ == "__main__":
    main()
