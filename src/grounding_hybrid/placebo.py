"""Donor choice for the placebo reading of step E2 (CLAUDE.md, "E2 record").

Each case whose context needs K >= 2 windows gets one donor: a case of the same GASP split (source_split, seed 0)
with a different source_id and at least K windows, drawn uniformly with np.random.default_rng(0), one draw per
case in cases.jsonl order, among eligible donors sorted by case_id. If no case has K windows, the eligible case
with the most windows is used (its windows are then reused in order; flagged as a fallback). Only the donor's
context text is used, never its labels.
"""
import numpy as np

from grounding_hybrid.gasp_bridge import analyze_gasp, load_sentences


def case_splits(canon_dir):
    """{case_id: 'dev' | 'test'} under GASP's source_split (seed 0); cases GASP skipped are absent."""
    dev, test = analyze_gasp.source_split(load_sentences(canon_dir), seed=0)
    out = {c: "dev" for c in dev["case_id"].unique()}
    out.update({c: "test" for c in test["case_id"].unique()})
    return out


def donor_map(cases, n_windows, split_of, seed=0):
    """{case_id: (donor case_id, fallback)} for every case with >= 2 windows that GASP scored."""
    rng = np.random.default_rng(seed)
    pool = sorted((c for c in cases if c.case_id in split_of), key=lambda c: c.case_id)
    out = {}
    for c in cases:
        k = n_windows[c.case_id]
        if k < 2 or c.case_id not in split_of:
            continue
        same = [d for d in pool if d.source_id != c.source_id and split_of[d.case_id] == split_of[c.case_id]
                and n_windows[d.case_id] >= 2]
        elig = [d for d in same if n_windows[d.case_id] >= k]
        if elig:
            out[c.case_id] = (elig[rng.integers(len(elig))].case_id, False)
        else:
            out[c.case_id] = (max(same, key=lambda d: (n_windows[d.case_id], d.case_id)).case_id, True)
    return out
