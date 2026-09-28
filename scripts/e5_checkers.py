"""Step E5 (CLAUDE.md, "E5 record"): B against two trained, off-the-shelf checkers (scripts/score_checkers.py).

  B              max over 1800/256 windows of the Lookback features, the frozen detector (test_look.fit_linear)
  MiniCheck      1 - P(supported) from MiniCheck-Flan-T5-Large, as released (no fitting)
  LettuceDetect  max token hallucination probability of LettuceDetect-large, as released (no fitting)

Primary (the E5 rule), on sentences needing more than one window of the real long-context sets TechQA, ExpertQA-long and
TRIVIA+-long, pooled over sets x 2 scorer runs (sources resampled per dataset, same draw for both runs, 2000 resamples),
each checker separately:
    D5m = AUC(B) - AUC(MiniCheck),  D5l = AUC(B) - AUC(LettuceDetect)
    lower bound > 0       -> "B beats <checker> on long contexts"
    interval contains 0   -> "B matches <checker> on long contexts"
    upper bound < 0       -> "<checker> beats B on long contexts"
Secondary (descriptive): every dataset (short sets too, where B = Lookback), response level (max over sentences), cost.
LettuceDetect was trained on RAGTruth: its RAGTruth numbers are shown as in-domain and never pooled.

--eval_on dev      B out-of-fold (grouped 5-fold inside dev), checkers on the same dev rows: the E5 decision
--eval_on test     B fit on all of dev, everything scored on GASP's test split: test look P8 (ONCE, after the decision)
--eval_on dryrun   the test branch on two halves of dev: a code check only

Usage:
    python -W ignore scripts/e5_checkers.py --eval_on dev
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
from test_look import CANON, FEAT, MODELS, auc, fit_linear, fmt, pooled  # noqa: E402

LONG3 = ["techqa", "expertqalong", "triviapluslong"]
SHORT = ["ragtruth", "tofueval", "ragbench"]
CHECKERS = {"MiniCheck": "minicheck", "LettuceDetect": "lettuce"}


def lookback_blocks(ds):
    """(Lookback npz, key), (B npz, key), windows npz for a dataset (B = Lookback where every context fits)."""
    if ds in LONG3:
        f = "features_chunked_redeep.npz"
        return (f, "lookback"), (f, "lookback_max"), f
    if ds == "ragbench":
        return ("features.npz", "lookback"), ("features_chunked.npz", "lookback_max"), "features_chunked.npz"
    return ("features.npz", "lookback"), ("features.npz", "lookback"), None


def frame(canon_root, feat_root, model, ds):
    canon, f = Path(canon_root) / f"{model}_{ds}_K5", Path(feat_root) / f"{model}_{ds}_K5"
    df, _ = load_signals(canon)
    (lbf, lbk), (bf, bk), winf = lookback_blocks(ds)
    cols = {}
    for name, (fname, key) in (("lb", (lbf, lbk)), ("B", (bf, bk))):
        d, c = load_signals(canon, f / fname, lb_keys=(key,))
        assert (d["case_id"].values == df["case_id"].values).all()
        new = [f"{name}_{i}" for i in range(len(c))]
        df = pd.concat([df, d[c].set_axis(new, axis=1)], axis=1)
        cols[name] = new
    z = np.load(f / "checkers.npz")
    ck = pd.DataFrame({"case_id": z["case_id"], "sent_idx": z["sent_idx"], "minicheck": z["minicheck"],
                       "lettuce": z["lettuce"]})
    j = df[["case_id", "sent_idx"]].merge(ck, on=["case_id", "sent_idx"], how="left", sort=False)
    assert len(j) == len(df) and j[["minicheck", "lettuce"]].notna().all().all(), f"checker scores missing: {f}"
    df["minicheck"], df["lettuce"] = j["minicheck"].values, j["lettuce"].values
    if winf:
        w = np.load(f / winf)
        nwin = dict(zip(zip(w["case_id"], w["sent_idx"]), w["n_windows"]))
        df["trunc"] = [bool(nwin[k] > 1) for k in zip(df["case_id"], df["sent_idx"])]
    else:
        df["trunc"] = False
    return df, cols


def score(df, cols, eval_on):
    """(evaluated rows, {detector: scores}): B / Lookback fitted as the frozen detector, checkers as released."""
    dev, test = analyze_gasp.source_split(df, seed=0)
    dev = dev.reset_index(drop=True)
    if eval_on in ("test", "dryrun"):
        if eval_on == "test":
            tr, te = dev, test.reset_index(drop=True)
        else:
            srcs = np.sort(dev["source_id"].unique())
            half = set(np.random.default_rng(0).permutation(srcs)[: len(srcs) // 2])
            tr = dev[dev["source_id"].isin(half)].reset_index(drop=True)
            te = dev[~dev["source_id"].isin(half)].reset_index(drop=True)
        s = {"Lookback": fit_linear(tr, te, cols["lb"]), "B": fit_linear(tr, te, cols["B"])}
        ev = te
    else:
        s = {n: np.zeros(len(dev)) for n in ("Lookback", "B")}
        for tr, va in _folds(dev, 5):
            s["Lookback"][va] = fit_linear(dev.iloc[tr], dev.iloc[va], cols["lb"])
            s["B"][va] = fit_linear(dev.iloc[tr], dev.iloc[va], cols["B"])
        ev = dev
    s = {k: np.asarray(v, float) for k, v in s.items()}
    for name, col in CHECKERS.items():
        s[name] = ev[col].values.astype(float)
    return ev, s


def verdict(r, checker):
    if r["lo"] > 0:
        return f"B beats {checker} on long contexts"
    if r["hi"] < 0:
        return f"{checker} beats B on long contexts"
    return f"B matches {checker} on long contexts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_on", choices=["dev", "test", "dryrun"], required=True)
    ap.add_argument("--canon_root", default=str(CANON))
    ap.add_argument("--feat_root", default=str(FEAT))
    ap.add_argument("--datasets", nargs="*", default=SHORT + LONG3)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / ("test" if args.eval_on == "test" else "e5")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = {"dev": "e5_dev", "test": "e5_test_look_P8", "dryrun": "e5_dryrun"}[args.eval_on]
    lines, report = [], dict(eval_on=args.eval_on)
    say = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731
    say(f"# E5: B vs MiniCheck and LettuceDetect, eval_on={args.eval_on}")

    runs = {}
    for ds in args.datasets:
        for model in MODELS:
            if not (Path(args.feat_root) / f"{model}_{ds}_K5" / "checkers.npz").exists():
                say(f"  [skip] {model} {ds}: no checkers.npz")
                continue
            df, cols = frame(args.canon_root, args.feat_root, model, ds)
            runs[(model, ds)] = score(df, cols, args.eval_on)
            print(f"done {model} {ds}", flush=True)

    long_items = lambda a, b: [(ds, ev[ev["trunc"].values].reset_index(drop=True),  # noqa: E731
                                sc[a][ev["trunc"].values], sc[b][ev["trunc"].values])
                               for (model, ds), (ev, sc) in runs.items() if ds in LONG3]
    report["primary"] = {}
    say("\n## PRIMARY (E5 rule): truncated rows of TechQA, ExpertQA-long, TRIVIA+-long, pooled over sets x 2 scorers")
    for checker in CHECKERS:
        D = pooled(long_items("B", checker))
        report["primary"][checker] = dict(D=D, verdict=verdict(D, checker))
        say(f"  AUC(B) - AUC({checker}) = {fmt(D)}  ->  {verdict(D, checker)}")

    say("\n## Sentence-level AUC per run (truncated rows on the long sets, all rows on the short sets)")
    rows = []
    for (model, ds), (ev, sc) in runs.items():
        m = ev["trunc"].values if ds in LONG3 else np.ones(len(ev), bool)
        y = ev["label"].values[m]
        r = dict(dataset=ds, model=model.split("-")[0], rows=int(m.sum()),
                 **{n: round(auc(y, s[m]), 3) for n, s in sc.items()})
        if ds == "ragtruth":
            r["note"] = "LettuceDetect in-domain (trained on RAGTruth)"
        rows.append(r)
    pd.set_option("display.width", 220)
    say(pd.DataFrame(rows).to_string(index=False))
    report["table"] = rows

    say("\n## Response level (max over sentences)")
    report["response"] = []
    for (model, ds), (ev, sc) in runs.items():
        g = pd.DataFrame({"case_id": ev["case_id"], "label": ev["label"], **sc}).groupby("case_id").max()
        r = dict(dataset=ds, model=model.split("-")[0], responses=len(g),
                 **{n: round(auc(g["label"], g[n]), 3) for n in sc})
        report["response"].append(r)
        say(f"  {r}")

    say("\n## Cost (seconds per case on the scoring device; meta_checkers.json)")
    report["cost"] = {}
    for (model, ds) in runs:
        m = json.load(open(Path(args.feat_root) / f"{model}_{ds}_K5" / "meta_checkers.json"))
        report["cost"][f"{model} {ds}"] = {k: m.get(k) for k in ("device", "minicheck_s_median", "lettuce_s_median",
                                                                  "lettuce_multi_chunk_contexts",
                                                                  "lettuce_truncated_contexts")}
        say(f"  {model.split('-')[0]} {ds}: {report['cost'][f'{model} {ds}']}")

    json.dump(report, open(out_dir / f"{stem}.json", "w"), indent=1, default=float)
    (out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    print(f"\nsaved {out_dir}/{stem}.{{txt,json}}")


if __name__ == "__main__":
    main()
