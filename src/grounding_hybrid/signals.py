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
from pathlib import Path

import numpy as np
import pandas as pd

from grounding_hybrid.gasp_bridge import BASE_FEATS, GASP_FEATS, load_sentences

S1_FEATS = GASP_FEATS + ["min_drop", "neg_drop_mass"]
S2_FEATS = ["prior_logprob"]
POS_FEATS = ["rel_pos", "sent_idx"]


def _signed(js):
    v = np.array([x for x in json.loads(js) if x is not None], dtype=float)
    if v.size == 0:
        return np.nan, np.nan
    return float(v.min()), float(-v[v < 0].sum())


def load_signals(canon_dir, features_file=None, lb_keys=("lookback",)):
    """sentence.csv + S1 sign features (+ Lookback columns when a features.npz is given).

    lb_keys picks the Lookback arrays to use (concatenated): "lookback" is GASP's window only;
    with coverage-aware features also "lookback_max" / "lookback_mean" over all windows.

    Also adds, per case: task (RAGBench domain / RAGTruth task) and ctx_kept, the fraction of the
    context inside the scorer's window (GASP's audit), which is known at inference time; and per
    sentence rel_pos, its relative position in the answer (a control: Lookback ratios drift with
    position, and in RAGBench later sentences are hallucinated more often).
    """
    canon_dir = Path(canon_dir)
    df = load_sentences(canon_dir)
    df["min_drop"], df["neg_drop_mass"] = zip(*df["chunk_drops"].map(_signed))
    meta = pd.read_csv(canon_dir / "response.csv")[["case_id", "task"]].merge(
        pd.read_csv(canon_dir / "audit.csv")[["case_id", "ctx_ret_frac"]], on="case_id")
    meta = meta.rename(columns={"ctx_ret_frac": "ctx_kept"})
    joined = df.merge(meta, on="case_id", how="left", sort=False)
    assert (joined["case_id"].values == df["case_id"].values).all()
    df = joined
    df["rel_pos"] = df["sent_idx"] / (df.groupby("case_id")["sent_idx"].transform("max") + 1)
    lb_cols = []
    if features_file is not None:
        z = np.load(features_file)
        lb = np.concatenate([z[k].astype(np.float32).reshape(len(z["case_id"]), -1) for k in lb_keys], axis=1)
        lb_cols = [f"lb_{i}" for i in range(lb.shape[1])]
        feats = pd.DataFrame(lb, columns=lb_cols)
        feats["case_id"], feats["sent_idx"] = z["case_id"], z["sent_idx"]
        joined = df.merge(feats, on=["case_id", "sent_idx"], how="left", sort=False)
        assert len(joined) == len(df) and (joined["case_id"].values == df["case_id"].values).all()
        if joined[lb_cols[0]].isna().any():
            raise ValueError("features do not cover every GASP sentence")
        df = joined
    return df, lb_cols


def to_response_level(df, lb_cols=()):
    """One row per case (label = any hallucinated sentence), as GASP's response-level analysis:
    S1 and base features by max, the prior by min (least-known sentence), Lookback ratios and
    relative position by mean, sent_idx by max (answer length in sentences). Sorted by case_id
    like analyze_gasp.py, so its source_split gives the same response-level split."""
    agg = {**{f: "max" for f in S1_FEATS + BASE_FEATS}, "prior_logprob": "min", "rel_pos": "mean",
           "sent_idx": "max", "ctx_kept": "first", "task": "first", "source_id": "first", "label": "max",
           **{c: "mean" for c in lb_cols}}
    return df.groupby("case_id").agg(agg).reset_index()


__all__ = ["BASE_FEATS", "POS_FEATS", "S1_FEATS", "S2_FEATS", "load_signals", "to_response_level"]
