"""Evaluate extracted features under GASP's exact protocol (same split, classifier, bootstrap).

Joins features.npz onto GASP's sentence.csv by (case_id, sent_idx) while keeping the csv's
row order, so GASP's own source_split yields the identical dev/test split as the published
analysis. The GASP rows therefore reproduce analysis_span.txt / analysis_response.txt
exactly, and every new method is scored on the same test sources.

Response level follows GASP (one row per case, label = any hallucinated sentence): GASP and
base features are max-aggregated as in analyze_gasp.py, the prior by its minimum (the least
known sentence), and Lookback ratios are averaged over the response's sentences (Lookback
Lens scores a span by its mean ratio).

Classifiers: rows without a tag use GASP's clf_auc unchanged (standardized logistic regression,
C=1), which is the faithful Lookback Lens setup too. Rows tagged [cv] choose C on the dev split
only (5-fold CV grouped by source); needed once hundreds of attention features are added.
[stack] adds the Lookback classifier's out-of-fold score as ONE feature to GASP+base (late
fusion), so hundreds of attention features cannot dilute GASP's eight.

Usage:
    python scripts/eval_features.py --canon_dir results/gasp_repro/canon_results/<TAG>
    python scripts/eval_features.py --canon_dir ... --level response
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grounding_hybrid.gasp_bridge import BASE_FEATS, GASP_FEATS, analyze_gasp, load_sentences  # noqa: E402

RAW = GASP_FEATS + BASE_FEATS + ["prior_logprob"]
CS = np.logspace(-4, 1, 11)


def clf_auc_cv(dev, test, feats, y):
    """GASP's classifier with its L2 strength chosen on dev only (5-fold CV grouped by source).

    GASP's clf_auc fixes C=1, which suits its 8 features but overfits hundreds of attention
    features. Same model family, same split; the test set is never seen during tuning.
    """
    d = dev.dropna(subset=feats)
    t = test.dropna(subset=feats)
    if d[y].nunique() < 2 or t[y].nunique() < 2 or len(d) < 10:
        return np.nan, np.nan, None
    sc = StandardScaler().fit(d[feats])
    folds = list(GroupKFold(5).split(d, d[y], d["source_id"]))
    m = LogisticRegressionCV(Cs=CS, cv=folds, scoring="roc_auc", class_weight="balanced",
                             max_iter=2000).fit(sc.transform(d[feats]), d[y])
    p = m.predict_proba(sc.transform(t[feats]))[:, 1]
    return roc_auc_score(t[y], p), average_precision_score(t[y], p), pd.Series(p, index=t.index)


def stacked_score(dev, test, feats, y):
    """One score per row from a [cv] classifier: out-of-fold on dev, dev-fit model on test.

    Lets a high-dimensional signal enter a small classifier as a single feature (late
    fusion) without leaking dev labels into its own dev scores or test labels anywhere.
    """
    sc = StandardScaler().fit(dev[feats])
    xd, xt = sc.transform(dev[feats]), sc.transform(test[feats])
    folds = list(GroupKFold(5).split(dev, dev[y], dev["source_id"]))
    c = LogisticRegressionCV(Cs=CS, cv=folds, scoring="roc_auc", class_weight="balanced",
                             max_iter=2000).fit(xd, dev[y]).C_[0]
    oof = np.zeros(len(dev))
    for tr, va in folds:
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=2000).fit(xd[tr], dev[y].values[tr])
        oof[va] = m.decision_function(xd[va])
    m = LogisticRegression(C=c, class_weight="balanced", max_iter=2000).fit(xd, dev[y])
    return pd.Series(oof, index=dev.index), pd.Series(m.decision_function(xt), index=test.index)


def load_joined(canon_dir, features_file):
    df = load_sentences(canon_dir)
    z = np.load(features_file)
    lb = z["lookback"].astype(np.float32).reshape(len(z["case_id"]), -1)
    lb_cols = [f"lb_{i}" for i in range(lb.shape[1])]
    feats = pd.DataFrame(lb, columns=lb_cols)
    feats["case_id"], feats["sent_idx"] = z["case_id"], z["sent_idx"]
    joined = df.merge(feats, on=["case_id", "sent_idx"], how="left", sort=False)
    assert len(joined) == len(df) and (joined["case_id"].values == df["case_id"].values).all()
    missing = int(joined[lb_cols[0]].isna().sum())
    return joined, lb_cols, missing


def evaluate(df, lb_cols, level, seed=0):
    if level == "response":
        agg = {**{f: "max" for f in RAW if f != "prior_logprob"}, "prior_logprob": "min",
               **{c: "mean" for c in lb_cols}, "source_id": "first", "label": "max"}
        df = df.groupby("case_id").agg(agg).reset_index()   # sorted, as analyze_gasp.py: same split
    dev, test = analyze_gasp.source_split(df, seed=seed)
    res = dict(level=level, dev_rows=len(dev), test_rows=len(test),
               test_sources=int(test["source_id"].nunique()), test_pos=int(test["label"].sum()),
               raw={}, clf={}, cmp={})
    for f in RAW:
        auc, ap = analyze_gasp.raw_auc(dev, test, f, "label")
        res["raw"][f] = [auc, ap]
    gasp_base = GASP_FEATS + BASE_FEATS
    dev["lookback_score"], test["lookback_score"] = stacked_score(dev, test, lb_cols, "label")
    sets = {"base_combined": (BASE_FEATS, analyze_gasp.clf_auc), "gasp+base": (gasp_base, analyze_gasp.clf_auc),
            "lookback": (lb_cols, analyze_gasp.clf_auc), "lookback+base": (lb_cols + BASE_FEATS, analyze_gasp.clf_auc),
            "gasp+lookback+base": (gasp_base + lb_cols, analyze_gasp.clf_auc),
            "gasp+base [cv]": (gasp_base, clf_auc_cv), "lookback [cv]": (lb_cols, clf_auc_cv),
            "gasp+lookback+base [cv]": (gasp_base + lb_cols, clf_auc_cv),
            "gasp+base+lookback [stack]": (gasp_base + ["lookback_score"], analyze_gasp.clf_auc)}
    scores = {}
    for name, (cols, fit) in sets.items():
        auc, ap, p = fit(dev, test, cols, "label")
        res["clf"][name], scores[name] = [auc, ap], p
    for a, b in [("gasp+base", "base_combined"), ("lookback", "base_combined"),
                 ("lookback [cv]", "base_combined"), ("gasp+lookback+base [cv]", "gasp+base [cv]"),
                 ("gasp+base+lookback [stack]", "gasp+base")]:
        res["cmp"][f"{a} - {b}"] = analyze_gasp.paired_bootstrap_diff(test, "label", scores.get(a), scores.get(b))
    return res


def report(res, title):
    print(f"# {title}  [{res['level']}]")
    print(f"dev {res['dev_rows']} rows | test {res['test_rows']} rows / {res['test_sources']} sources "
          f"| test positives {res['test_pos']}")
    print("## raw single-feature AUC (direction fixed on dev)")
    for f, (auc, ap) in res["raw"].items():
        print(f"  {f:27s} AUC {auc:.3f}  AUPRC {ap:.3f}" if auc == auc else f"  {f:27s} n/a")
    print("## trained classifiers (dev-fit; untagged = GASP's clf_auc, [cv] = C tuned on dev, [stack] = late fusion)")
    for f, (auc, ap) in res["clf"].items():
        print(f"  {f:27s} AUC {auc:.3f}  AUPRC {ap:.3f}" if auc == auc else f"  {f:27s} n/a")
    print("## paired source-level bootstrap (2000 resamples, two-sided 95% CI)")
    for k, r in res["cmp"].items():
        print(f"  {k}: " + ("insufficient data" if r is None else
              f"dAUC {r['mean']:+.3f}  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  -> "
              + ("DIFFERENT" if r["sig"] else "no difference found")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_dir", required=True)
    ap.add_argument("--features", default=None, help="default: results/features/<TAG>/features.npz")
    ap.add_argument("--level", choices=["span", "response", "both"], default="both")
    ap.add_argument("--out", default=None, help="json output (default: next to the features)")
    args = ap.parse_args()

    tag = Path(args.canon_dir).name
    feat_file = Path(args.features or ROOT / "results" / "features" / tag / "features.npz")
    df, lb_cols, missing = load_joined(args.canon_dir, feat_file)
    if missing:
        sys.exit(f"{missing} of {len(df)} GASP sentences have no features: extraction incomplete")
    out = {}
    for level in (["span", "response"] if args.level == "both" else [args.level]):
        out[level] = evaluate(df, lb_cols, level)
        report(out[level], tag)
        print()
    out_file = Path(args.out or feat_file.parent / "eval.json")
    json.dump(out, open(out_file, "w"), indent=1, default=float)
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
