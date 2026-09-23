"""Gated combination of the three signals, designed and compared on the DEV split only.

All methods are standardized logistic regressions (GASP's classifier family), so differences
come from the gating, not from model capacity:

  combined   (c) one classifier on S1 + S2 + S3 + base features, no gate.
  soft gate  (b) learned gate: the classifier also sees S1 and S3 multiplied by the standardized
             gate variable, so each signal's weight can rise or fall with it.
  hard gate  (a) rule: rows with gate variable >= tau are scored by the "high" expert, the rest by
             the "low" expert, each trained on its own region; tau is chosen among training-part
             quantiles by an inner grouped CV.

Gate variables: prior_logprob (S2; the brief's hypothesis) and ctx_kept (evidence coverage; the
dev diagnostic's alternative, known at inference time). S3 enters as ONE score: a Lookback
classifier fit strictly inside each training part (out-of-fold on it, fitted model on the rest),
so the rows being evaluated never inform their own S3 score.

Missing S1 values (no chunk inside the window) are filled with training-part medians for every
method alike; GASP's analysis drops those rows instead, so absolute numbers can differ slightly.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

S3_CS = np.logspace(-4, 0, 5)


def _lr(c=1.0):
    return LogisticRegression(C=c, class_weight="balanced", max_iter=2000)


def _folds(df, k):
    k = min(k, df["source_id"].nunique())
    return list(GroupKFold(k).split(df, df["label"], df["source_id"]))


def s3_scores(train, other, lb_cols, k=3):
    """Lookback score: out-of-fold on `train`, and from a train-fit model on `other`."""
    sc = StandardScaler().fit(train[lb_cols])
    xt, xo, y = sc.transform(train[lb_cols]), sc.transform(other[lb_cols]), train["label"].values
    folds = _folds(train, k)
    c = LogisticRegressionCV(Cs=S3_CS, cv=folds, scoring="roc_auc", class_weight="balanced",
                             max_iter=2000).fit(xt, y).C_[0]
    oof = np.zeros(len(train))
    for tr, va in folds:
        oof[va] = _lr(c).fit(xt[tr], y[tr]).decision_function(xt[va])
    return oof, _lr(c).fit(xt, y).decision_function(xo)


class Linear:
    """Logistic regression on fixed columns; with `gate`, adds gated_cols x standardized gate."""

    def __init__(self, cols, gate=None, gated_cols=()):
        self.cols, self.gate, self.gated_cols = list(cols), gate, list(gated_cols)

    def _x(self, df):
        x = df[self.cols].to_numpy(float)
        if self.gate:
            g = (df[self.gate].to_numpy(float) - self.g_mu) / self.g_sd
            x = np.hstack([x, df[self.gated_cols].to_numpy(float) * g[:, None]])
        return x

    def fit(self, df):
        if self.gate:
            self.g_mu, self.g_sd = df[self.gate].mean(), df[self.gate].std()
            if not self.g_sd > 1e-6:
                raise ValueError(f"gate variable {self.gate} is constant here")
        x = self._x(df)
        self.sc = StandardScaler().fit(x)
        self.m = _lr().fit(self.sc.transform(x), df["label"])
        return self

    def score(self, df):
        return self.m.decision_function(self.sc.transform(self._x(df)))


class HardGate:
    """Two experts split by a threshold on the gate variable; tau picked by inner grouped CV."""

    def __init__(self, gate, lo_cols, hi_cols, quantiles=(0.33, 0.5, 0.67), k=3):
        self.gate, self.lo_cols, self.hi_cols = gate, list(lo_cols), list(hi_cols)
        self.quantiles, self.k = quantiles, k

    def _fit_tau(self, df, tau):
        hi = df[self.gate] >= tau
        if min(hi.sum(), (~hi).sum()) < 20 or df[hi]["label"].nunique() < 2 or df[~hi]["label"].nunique() < 2:
            return None
        return tau, Linear(self.lo_cols).fit(df[~hi]), Linear(self.hi_cols).fit(df[hi])

    @staticmethod
    def _score(fitted, df):
        tau, lo, hi = fitted
        is_hi = (df["gate_value"] >= tau).to_numpy()
        s = np.empty(len(df))
        if (~is_hi).any():
            s[~is_hi] = lo.score(df[~is_hi])
        if is_hi.any():
            s[is_hi] = hi.score(df[is_hi])
        return s

    def fit(self, df):
        df = df.assign(gate_value=df[self.gate])
        best = (-1.0, None)
        for q in self.quantiles:
            tau = float(df[self.gate].quantile(q))
            aucs = []
            for tr, va in _folds(df, self.k):
                f = self._fit_tau(df.iloc[tr], tau)
                if f is None or df.iloc[va]["label"].nunique() < 2:
                    aucs = []
                    break
                aucs.append(roc_auc_score(df.iloc[va]["label"], self._score(f, df.iloc[va])))
            if aucs and np.mean(aucs) > best[0]:
                best = (float(np.mean(aucs)), tau)
        if best[1] is None:
            raise ValueError("no usable threshold (a region too small or single-class)")
        self.tau = best[1]
        self.fitted = self._fit_tau(df, self.tau)
        return self

    def score(self, df):
        return self._score(self.fitted, df.assign(gate_value=df[self.gate]))


def dev_cv(dev, methods, lb_cols=(), k=5):
    """Grouped k-fold CV inside the dev split. Returns {method: per-fold AUCs} (NaN if skipped)."""
    lb_cols = list(lb_cols)
    out = {name: [] for name in methods}
    fill_cols = [c for c in dev.columns if dev[c].dtype.kind == "f" and c not in lb_cols]
    for tr, va in _folds(dev, k):
        d_tr, d_va = dev.iloc[tr].copy(), dev.iloc[va].copy()
        med = d_tr[fill_cols].median()
        d_tr[fill_cols], d_va[fill_cols] = d_tr[fill_cols].fillna(med), d_va[fill_cols].fillna(med)
        if lb_cols:
            d_tr["s3"], d_va["s3"] = s3_scores(d_tr, d_va, lb_cols)
        for name, (make, needs_s3) in methods.items():
            if needs_s3 and not lb_cols:
                out[name].append(np.nan)
                continue
            try:
                out[name].append(roc_auc_score(d_va["label"], make().fit(d_tr).score(d_va)))
            except ValueError:
                out[name].append(np.nan)
    return out
