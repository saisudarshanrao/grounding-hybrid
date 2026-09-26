"""Evidence displacement for step E3 (CLAUDE.md, "E3 record").

A response is included when its original context fits one window (<= max_ctx tokens), so the original reading sees
all of it. Its displaced context is a prefix + "\\n\\n" + the original context. The prefix is the contexts of other
cases of the same dataset (same GASP split, different source_id) in a random order (np.random.default_rng(seed): one
permutation per included response, responses in cases.jsonl order), joined with "\\n\\n" and cut at exactly
PREFIX_TOKENS tokens, so window 1 (the first max_ctx tokens) holds only distractor text. Question, answer, sentence
map and labels are unchanged; only the context differs.
"""
import dataclasses

import numpy as np

PREFIX_TOKENS = 1808
SEP = "\n\n"


def included(cases, n_tokens, split_of, max_ctx=1800):
    """Cases GASP scored whose original context fits one window."""
    return [c for c in cases if c.case_id in split_of and n_tokens[c.case_id] <= max_ctx]


def build_prefixes(cases, keep, split_of, tok, seed=0, n_prefix=PREFIX_TOKENS):
    """{case_id: (prefix text, [donor case_ids used])} for the included cases `keep` (cases.jsonl order)."""
    rng = np.random.default_rng(seed)
    pool = sorted((c for c in cases if c.case_id in split_of), key=lambda c: c.case_id)
    out = {}
    for c in keep:
        elig = [d for d in pool if d.source_id != c.source_id and split_of[d.case_id] == split_of[c.case_id]]
        text, used = "", []
        for i in rng.permutation(len(elig)):
            d = elig[i]
            text = text + (SEP if text else "") + d.context
            used.append(d.case_id)
            if len(tok(text, add_special_tokens=False).input_ids) >= n_prefix:
                break
        offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        if len(offs) < n_prefix:
            raise ValueError(f"not enough distractor text for {c.case_id}")
        out[c.case_id] = (text[: offs[n_prefix - 1][1]], used)
    return out


def displaced(case, prefix):
    """The case with its context moved behind `prefix` (an empty prefix returns an identical copy)."""
    return dataclasses.replace(case, context=prefix + SEP + case.context if prefix else case.context)
