"""RQ2 on the DEV split: which signal works where.

For every run (scorer x dataset, plus the TechQA long-context set) each detector gets one
out-of-fold score per dev sentence (grouped 5-fold CV by source; classifiers and C chosen inside
the training folds only), and its ROC-AUC is reported overall and within subgroups:
  task        RAGTruth task / RAGBench domain
  prior       tertiles of S2 = no-context log-prob of the sentence (low / mid / high: "already known")
  position    tertiles of the sentence's relative position in the answer
  coverage    share of the context inside the 1800-token window (all / partly / mostly cut)
Detectors: perplexity+length (base), S2 prior alone, GASP+base (S1), Lookback (S3), ReDeEP, frequency-aware
attention (arXiv 2602.18145) where extracted, and B (Lookback max over windows) where coverage-aware features exist. A subgroup is scored only with at
least 10 hallucinated and 10 grounded sentences.

Usage:
    python -W ignore scripts/rq2_where.py        # -> results/gating/rq2_where.{csv,txt}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grounding_hybrid.gasp_bridge import BASE_FEATS, GASP_FEATS, analyze_gasp  # noqa: E402
from grounding_hybrid.gating import _folds  # noqa: E402
from grounding_hybrid.signals import load_signals  # noqa: E402

CS = np.logspace(-4, 1, 11)
MIN_CLASS = 10


def fit_score(train, test, cols, tune):
    med = train[cols].median()
    xtr, xte = train[cols].fillna(med), test[cols].fillna(med)
    sc = StandardScaler().fit(xtr)
    xtr, xte, y = sc.transform(xtr), sc.transform(xte), train["label"].values
    c = (LogisticRegressionCV(Cs=CS, cv=_folds(train, 3), scoring="roc_auc", class_weight="balanced",
                              max_iter=2000).fit(xtr, y).C_[0] if tune else 1.0)
    return LogisticRegression(C=c, class_weight="balanced", max_iter=2000).fit(xtr, y).decision_function(xte)


def oof(dev, cols, tune):
    s = np.full(len(dev), np.nan)
    for tr, va in _folds(dev, 5):
        s[va] = fit_score(dev.iloc[tr], dev.iloc[va], cols, tune)
    return s


def feature_sets(tag):
    """(name, npz file, keys, tuned) for the detectors whose features exist for this run."""
    f = ROOT / "results" / "features" / tag
    out = []
    lb = f / "features.npz" if (f / "features.npz").exists() else f / "features_chunked_redeep.npz"
    if lb.exists():
        out.append(("Lookback (S3)", lb, ("lookback",), True))
    for name in ("features_chunked.npz", "features_chunked_redeep.npz"):
        if (f / name).exists():
            out.append(("B: Lookback max over windows", f / name, ("lookback_max",), True))
            break
    rd = f / "features_redeep.npz" if (f / "features_redeep.npz").exists() else f / "features_chunked_redeep.npz"
    if rd.exists():
        out.append(("ReDeEP [cv]", rd, ("ecs", "pks"), True))
    if (f / "features_freq.npz").exists():
        out.append(("Frequency-aware [cv]", f / "features_freq.npz", ("freq",), True))
    return out


def tertile(x):
    return pd.qcut(x.rank(method="first"), 3, labels=["low", "mid", "high"])


def main():
    rows = []
    for canon in sorted((ROOT / "results" / "gasp_repro" / "canon_results").glob("*_K5")):
        tag = canon.name
        sets = feature_sets(tag)
        if not sets:
            continue
        base_df, _ = load_signals(canon)
        dev, _test = analyze_gasp.source_split(base_df, seed=0)
        dev = dev.reset_index(drop=True)
        scores = {"Perplexity+length": oof(dev, BASE_FEATS, False),
                  "S2 prior alone": oof(dev, ["prior_logprob"], False),
                  "GASP+base (S1)": oof(dev, GASP_FEATS + BASE_FEATS, False)}
        for name, npz, keys, tune in sets:
            df, cols = load_signals(canon, npz, lb_keys=keys)
            d, _ = analyze_gasp.source_split(df, seed=0)
            d = d.reset_index(drop=True)
            assert (d["case_id"].values == dev["case_id"].values).all()
            scores[name] = oof(d, cols, tune)
        groups = {"all": pd.Series("all", index=dev.index), "task": dev["task"].astype(str),
                  "prior": tertile(dev["prior_logprob"]), "position": tertile(dev["rel_pos"]),
                  "coverage": pd.cut(dev["ctx_kept"], [-1, 0.5, 0.999, 2], labels=["<50% kept", "50-99% kept", "all kept"])}
        for gname, g in groups.items():
            for level, idx in dev.groupby(g.astype(str), observed=True).groups.items():
                y = dev.loc[idx, "label"]
                if y.sum() < MIN_CLASS or (1 - y).sum() < MIN_CLASS:
                    continue
                for det, s in scores.items():
                    rows.append(dict(run=tag.replace("-Instruct", "").replace("_K5", ""), by=gname, group=level,
                                     n=len(idx), halluc=float(y.mean()), detector=det,
                                     auc=roc_auc_score(y, s[np.asarray(idx)])))
        print(f"done {tag}", flush=True)
    res = pd.DataFrame(rows)
    out = ROOT / "results" / "gating" / "rq2_where"
    res.to_csv(out.with_suffix(".csv"), index=False)
    pd.set_option("display.width", 250)
    lines = []
    for by in ["all", "task", "prior", "position", "coverage"]:
        t = res[res.by == by].pivot_table(index=["run", "group", "n", "halluc"], columns="detector", values="auc")
        lines.append(f"\n# DEV AUC by {by} (out-of-fold, grouped 5-fold CV)\n" + t.round(3).to_string())
    txt = "\n".join(lines)
    out.with_suffix(".txt").write_text(txt + "\n")
    print(txt)
    print(f"\nsaved {out}.csv / .txt")


if __name__ == "__main__":
    main()
