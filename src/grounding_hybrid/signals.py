"""The three signals of the prior-aware detector, one row per GASP sentence.

  S1  signed context sensitivity: GASP's features plus their sign. A positive chunk drop means
      the sentence relies on that chunk; a negative one means removing the chunk makes the
      sentence MORE likely, i.e. the chunk contradicts it (min_drop, neg_drop_mass).
  S2  prior confidence: mean token log-prob WITHOUT context (prior_logprob). High = the model
      would say this anyway, so context removal barely moves it and S1 loses its signal.
  S3  evidence reading: Lookback Lens attention ratios (lb_*), reduced to one score by a
      dev-trained classifier when a scalar is needed.

Row order is sentence.csv's, so GASP's source_split still gives its exact dev/test split.
"""
import json

import numpy as np
import pandas as pd

from grounding_hybrid.gasp_bridge import BASE_FEATS, GASP_FEATS, load_sentences

S1_FEATS = GASP_FEATS + ["min_drop", "neg_drop_mass"]
S2_FEATS = ["prior_logprob"]


def _signed(js):
    v = np.array([x for x in json.loads(js) if x is not None], dtype=float)
    if v.size == 0:
        return np.nan, np.nan
    return float(v.min()), float(-v[v < 0].sum())


def load_signals(canon_dir, features_file=None):
    """sentence.csv + S1 sign features (+ Lookback columns when a features.npz is given)."""
    df = load_sentences(canon_dir)
    df["min_drop"], df["neg_drop_mass"] = zip(*df["chunk_drops"].map(_signed))
    lb_cols = []
    if features_file is not None:
        z = np.load(features_file)
        lb = z["lookback"].astype(np.float32).reshape(len(z["case_id"]), -1)
        lb_cols = [f"lb_{i}" for i in range(lb.shape[1])]
        feats = pd.DataFrame(lb, columns=lb_cols)
        feats["case_id"], feats["sent_idx"] = z["case_id"], z["sent_idx"]
        joined = df.merge(feats, on=["case_id", "sent_idx"], how="left", sort=False)
        assert len(joined) == len(df) and (joined["case_id"].values == df["case_id"].values).all()
        if joined[lb_cols[0]].isna().any():
            raise ValueError("features do not cover every GASP sentence")
        df = joined
    return df, lb_cols


__all__ = ["BASE_FEATS", "S1_FEATS", "S2_FEATS", "load_signals"]
