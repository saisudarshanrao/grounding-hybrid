"""ReDeEP baseline next to Lookback Lens and GASP, under GASP's protocol.

Reads features_redeep.npz (lookback [layers, heads], ecs [layers, heads], pks [layers] per GASP
sentence) and scores:
  ReDeEP              ReDeEP's own detector (Sun et al., ICLR 2025): H = alpha * sum of PKS over the
                      selected "knowledge FFN" layers - beta * sum of ECS over the selected "copying"
                      heads. Heads / layers are the ones whose score correlates most with the label
                      (ECS negatively, PKS positively) on TRAINING data only; how many (N heads, M
                      layers) and beta/alpha are chosen by an inner grouped CV on the training part.
  ReDeEP [cv]         the "multivariate regression" form: logistic regression on all ECS + PKS
                      features, C tuned by grouped CV on the training part (as Lookback [cv]).
  Lookback [cv], GASP+base, Lookback+ReDeEP [cv] for the one-protocol comparison (RQ3).
  With coverage-aware features (features_chunked_redeep.npz, e.g. TechQA) also Lookback-max [cv], the
  frozen part-B method (Lookback ratios max over all context windows), alone and with ReDeEP.

Two modes, following the dev/test rule in CLAUDE.md:
  default   grouped 5-fold CV INSIDE the dev split (design choices, no test look)
  --test    dev-fit, scored once on GASP's test split, paired source-level bootstrap (log the look)

Usage:
    python -W ignore scripts/eval_redeep.py                        # every run with features_redeep.npz
    python -W ignore scripts/eval_redeep.py --runs ragtruth --level response
    python -W ignore scripts/eval_redeep.py --test
"""
import argparse
import itertools
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

from grounding_hybrid.gasp_bridge import BASE_FEATS, GASP_FEATS, analyze_gasp, load_sentences  # noqa: E402
from grounding_hybrid.gating import _folds  # noqa: E402

CS = np.logspace(-4, 1, 11)
N_HEADS = (1, 2, 4, 8, 16, 32)
M_LAYERS = (1, 2, 3, 5, 8)
BETAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, np.inf)   # beta / alpha; 0 = PKS only, inf = ECS only


def load(canon, feat, partial=False):
    df = load_sentences(canon)
    z = np.load(feat)
    n = len(z["case_id"])
    blocks = {"lb": z["lookback"], "ecs": z["ecs"], "pks": z["pks"]}
    if "lookback_max" in z.files:
        blocks["lbmax"] = z["lookback_max"]
    cols = {k: [f"{k}_{i}" for i in range(v.reshape(n, -1).shape[1])] for k, v in blocks.items()}
    feats = pd.DataFrame(np.concatenate([v.astype(np.float32).reshape(n, -1) for v in blocks.values()], axis=1),
                         columns=sum(cols.values(), []))
    feats["case_id"], feats["sent_idx"] = z["case_id"], z["sent_idx"]
    joined = df.merge(feats, on=["case_id", "sent_idx"], how="left", sort=False)
    assert len(joined) == len(df) and (joined["case_id"].values == df["case_id"].values).all()
    if joined[cols["ecs"][0]].isna().any():
        if not partial:
            raise ValueError(f"{feat} does not cover every GASP sentence")
        joined = joined.dropna(subset=cols["ecs"][:1]).reset_index(drop=True)   # smoke tests only
    return joined, cols


def to_response(df, cols):
    agg = {**{f: "max" for f in GASP_FEATS + BASE_FEATS}, **{c: "mean" for c in sum(cols.values(), [])},
           "source_id": "first", "label": "max"}
    return df.groupby("case_id").agg(agg).reset_index()   # sorted, as analyze_gasp.py: same split


def _auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


class ReDeEP:
    """alpha * sum PKS[selected layers] - beta * sum ECS[selected heads]; selection on training rows."""

    def __init__(self, ecs_cols, pks_cols, k=3):
        self.ecs, self.pks, self.k = ecs_cols, pks_cols, k

    @staticmethod
    def _corr(x, y):
        x = x - x.mean(0)
        y = y - y.mean()
        return (x * y[:, None]).sum(0) / (np.sqrt((x ** 2).sum(0) * (y ** 2).sum()) + 1e-12)

    def _arrays(self, df):
        return df[self.ecs].to_numpy(float), df[self.pks].to_numpy(float), df["label"].to_numpy(float)

    def _rank(self, e, p, y):
        return np.argsort(self._corr(e, y)), np.argsort(-self._corr(p, y))   # ECS most negative, PKS most positive

    @staticmethod
    def _combine(e_sum, p_sum, beta):
        return -e_sum if beta == np.inf else p_sum - beta * e_sum

    def fit(self, df):
        e, p, y = self._arrays(df)
        grid = list(itertools.product(N_HEADS, M_LAYERS, BETAS))
        res = np.zeros(len(grid))
        for tr, va in _folds(df, self.k):
            h, l = self._rank(e[tr], p[tr], y[tr])
            ce, cp = np.cumsum(e[va][:, h], 1), np.cumsum(p[va][:, l], 1)   # sums over the top-n heads / layers
            res += [_auc(y[va], self._combine(ce[:, n - 1], cp[:, m - 1], b)) for n, m, b in grid]
        self.n, self.m, self.beta = grid[int(np.nanargmax(res))]
        self.heads, self.layers = self._rank(e, p, y)
        return self

    def score(self, df):
        e, p, _ = self._arrays(df)
        return self._combine(e[:, self.heads[:self.n]].sum(1), p[:, self.layers[:self.m]].sum(1), self.beta)

    def desc(self):
        b = "ECS only" if self.beta == np.inf else ("PKS only" if self.beta == 0 else f"beta/alpha {self.beta:g}")
        return f"{self.n} heads, {self.m} layers, {b}"


class LinearCV:
    """Standardized logistic regression, C chosen by grouped CV on the training rows."""

    def __init__(self, cols, fixed_c=None):
        self.cols, self.fixed_c = cols, fixed_c

    def fit(self, df):
        d = df.dropna(subset=self.cols)
        self.sc = StandardScaler().fit(d[self.cols])
        x, y = self.sc.transform(d[self.cols]), d["label"].values
        c = self.fixed_c or LogisticRegressionCV(Cs=CS, cv=_folds(d, 5), scoring="roc_auc", class_weight="balanced",
                                                 max_iter=2000).fit(x, y).C_[0]
        self.m = LogisticRegression(C=c, class_weight="balanced", max_iter=2000).fit(x, y)
        return self

    def score(self, df):
        d = df.dropna(subset=self.cols)
        return pd.Series(self.m.decision_function(self.sc.transform(d[self.cols])), index=d.index)


def methods(cols):
    g = GASP_FEATS + BASE_FEATS
    return {"GASP+base [reference]": lambda: LinearCV(g, fixed_c=1.0),
            "Lookback [cv]": lambda: LinearCV(cols["lb"]),
            "ReDeEP": lambda: ReDeEP(cols["ecs"], cols["pks"]),
            "ReDeEP [cv]": lambda: LinearCV(cols["ecs"] + cols["pks"]),
            "ECS only [cv]": lambda: LinearCV(cols["ecs"]),
            "PKS only [cv]": lambda: LinearCV(cols["pks"]),
            "Lookback+ReDeEP [cv]": lambda: LinearCV(cols["lb"] + cols["ecs"] + cols["pks"]),
            **({"Lookback-max [cv] (B)": lambda: LinearCV(cols["lbmax"]),
                "Lookback-max+ReDeEP [cv]": lambda: LinearCV(cols["lbmax"] + cols["ecs"] + cols["pks"])}
               if "lbmax" in cols else {})}


def as_series(s, df):
    return s if isinstance(s, pd.Series) else pd.Series(s, index=df.index)


def dev_cv(dev, cols, k=5):
    out = {}
    for name, make in methods(cols).items():
        aucs, chosen = [], []
        for tr, va in _folds(dev, k):
            m = make().fit(dev.iloc[tr])
            s = as_series(m.score(dev.iloc[va]), dev.iloc[va])
            aucs.append(_auc(dev.loc[s.index, "label"], s))
            if isinstance(m, ReDeEP):
                chosen.append(m.desc())
        out[name] = dict(auc=float(np.nanmean(aucs)), sd=float(np.nanstd(aucs)), folds=aucs, chosen=chosen)
    return out


def test_eval(dev, test, cols):
    scores, out = {}, {}
    for name, make in methods(cols).items():
        m = make().fit(dev)
        s = as_series(m.score(test), test)
        scores[name] = s
        out[name] = dict(auc=float(_auc(test.loc[s.index, "label"], s)))
        if isinstance(m, ReDeEP):
            out[name]["chosen"] = m.desc()
    cmp = {}
    pairs = [("ReDeEP", "GASP+base [reference]"), ("ReDeEP [cv]", "GASP+base [reference]"),
             ("Lookback [cv]", "ReDeEP"), ("Lookback [cv]", "ReDeEP [cv]"), ("Lookback+ReDeEP [cv]", "Lookback [cv]")]
    if "lbmax" in cols:
        pairs += [("Lookback-max [cv] (B)", "Lookback [cv]"), ("Lookback-max [cv] (B)", "GASP+base [reference]")]
    for a, b in pairs:
        common = scores[a].index.intersection(scores[b].index)
        cmp[f"{a} - {b}"] = analyze_gasp.paired_bootstrap_diff(test, "label", scores[a].loc[common],
                                                               scores[b].loc[common])
    return out, cmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="only runs whose folder name contains this")
    ap.add_argument("--level", choices=["span", "response"], default="span")
    ap.add_argument("--features_name", default="features_redeep.npz")
    ap.add_argument("--features_root", default=str(ROOT / "results" / "features"))
    ap.add_argument("--partial", action="store_true", help="smoke test: keep only covered sentences")
    ap.add_argument("--test", action="store_true", help="score ONCE on the test split (log it in CLAUDE.md)")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    results, table = {}, {}
    for canon in sorted((ROOT / "results" / "gasp_repro" / "canon_results").glob("*_K5")):
        feat = Path(args.features_root) / canon.name / args.features_name
        if args.runs not in canon.name or not feat.exists():
            continue
        df, cols = load(canon, feat, args.partial)
        if args.level == "response":
            df = to_response(df, cols)
        dev, test = analyze_gasp.source_split(df, seed=0)
        dev, test = dev.reset_index(drop=True), test.reset_index(drop=True)
        tag = canon.name.replace("-Instruct", "").replace("_K5", "")
        if args.test:
            res, cmp = test_eval(dev, test, cols)
            results[canon.name] = dict(test=res, cmp=cmp)
            print(f"\n# {tag} [{args.level}] TEST ({len(test)} rows / {test['source_id'].nunique()} sources)")
            for k, v in res.items():
                print(f"  {k:24s} AUC {v['auc']:.3f}" + (f"   ({v['chosen']})" if "chosen" in v else ""))
            for k, r in cmp.items():
                print(f"  {k}: " + ("insufficient data" if r is None else
                      f"dAUC {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] {'DIFFERENT' if r['sig'] else 'n.s.'}"))
        else:
            res = dev_cv(dev, cols)
            results[canon.name] = res
            print(f"\n# {tag} [{args.level}] DEV grouped 5-fold CV ({len(dev)} rows / {dev['source_id'].nunique()} sources)")
            for k, v in res.items():
                print(f"  {k:24s} AUC {v['auc']:.3f} +/- {v['sd']:.3f}" + (f"   chosen: {'; '.join(v['chosen'])}" if v["chosen"] else ""))
        table[tag] = {k: v["auc"] for k, v in (results[canon.name]["test"] if args.test else results[canon.name]).items()}
    if table:
        print("\n# Summary: " + ("TEST" if args.test else "DEV CV") + f" {args.level}-level AUC")
        print(pd.DataFrame(table).to_string(float_format=lambda x: f"{x:.3f}"))
    extra = (f"_{args.runs}" if args.runs else "") + ("" if args.features_name == "features_redeep.npz"
                                                      else "_" + Path(args.features_name).stem)
    out = ROOT / "results" / "gating" / f"redeep{extra}_{'test' if args.test else 'dev_cv'}_{args.level}.json"
    json.dump(results, open(out, "w"), indent=1, default=float)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
