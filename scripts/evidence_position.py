"""Step 5b diagnostic (DEV only): does B help where the evidence lies beyond the first window?

RAGBench annotates, per response, which document sentences are relevant to the question
(all_relevant_sentence_keys) and which the response used (all_utilized_sentence_keys). For every case
of a long-context set (TechQA, ExpertQA-long) this locates those sentences in the joined context GASP
scored and compares their start with the end of window 1 (the scorer's first 1800 context tokens), then
splits the truncated dev rows by where the case's evidence lies:
  in window 1   every annotated sentence starts inside window 1
  partly out    some start beyond it
  all out       all start beyond it
  none          no sentence annotated
and reports, per group, the out-of-fold S3 AUC of window 1 vs B (max over windows) with the same
source-level paired bootstrap as analyze_coverage.py. B is not changed here (no new variant).

Usage:
    python -W ignore scripts/evidence_position.py            # -> results/gating/evidence_position.txt
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_coverage import oof, paired_ci  # noqa: E402
from grounding_hybrid.gasp_bridge import analyze_gasp  # noqa: E402
from grounding_hybrid.signals import load_signals  # noqa: E402

SETS = {"techqa": ("techqa", 0), "expertqalong": ("expertqa", 9000)}
MAX_CTX = 1800
WS = re.compile(r"\s+")


def normalized(text):
    """Whitespace-collapsed text and, for each of its characters, the offset in the original."""
    out, pos, i = [], [], 0
    for m in WS.finditer(text):
        for k in range(i, m.start()):
            out.append(text[k]); pos.append(k)
        out.append(" "); pos.append(m.start())
        i = m.end()
    for k in range(i, len(text)):
        out.append(text[k]); pos.append(k)
    return "".join(out), pos


def sentence_offsets(docs, doc_sents):
    """Key -> start offset in "\\n".join(docs) of every annotated document sentence found."""
    offs, start = {}, 0
    for doc, sents in zip(docs, doc_sents):
        norm, pos = normalized(doc)
        cursor = 0
        for key, sent in sents:
            s = WS.sub(" ", str(sent)).strip()
            if not s:
                continue
            j = norm.find(s, cursor)
            if j < 0:
                j = norm.find(s)
            if j >= 0:
                offs[str(key)] = start + pos[j]
                cursor = j + len(s)
        start += len(doc) + 1
    return offs


def ragbench_rows(domain, min_chars):
    from datasets import load_dataset
    rows = {}
    for split in ("train", "validation", "test"):
        for e in load_dataset("rungalileo/ragbench", domain, split=split):
            docs = [str(d) for d in (e.get("documents") or [])]
            ctx = "\n".join(docs)
            if not docs or len(ctx) < min_chars:
                continue
            key = (hashlib.md5(ctx.encode("utf-8")).hexdigest()[:12], (e.get("question") or "").strip())
            rows.setdefault(key, []).append(e)
    return rows


def case_evidence(canon, rows, tok):
    """Per case: share of relevant / utilized sentences starting beyond window 1."""
    out = {}
    for line in open(canon / "cases.jsonl"):
        c = json.loads(line)
        md5 = c["source_id"].split("_", 1)[1]
        cands = rows.get((md5, c["query"].strip()), [])
        if len(cands) > 1:   # same context and question: pick the response that matches the answer
            want = WS.sub(" ", c["answer"]).strip()[:200]
            cands = [e for e in cands if WS.sub(" ", " ".join(t for _, t in e["response_sentences"])).strip()[:200] == want] or cands
        if not cands:
            continue
        e = cands[0]
        coffs = tok(c["context"], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        end1 = coffs[MAX_CTX - 1][1] if len(coffs) >= MAX_CTX else len(c["context"])
        offs = sentence_offsets([str(d) for d in e["documents"]], e["documents_sentences"])
        rec = {}
        for kind in ("relevant", "utilized"):
            keys = [str(k) for k in (e.get(f"all_{kind}_sentence_keys") or []) if str(k) in offs]
            rec[kind] = np.nan if not keys else float(np.mean([offs[k] >= end1 for k in keys]))
        out[c["case_id"]] = rec
    return out


def group(share):
    if share != share:
        return "none annotated"
    return "in window 1" if share == 0 else ("all out" if share == 1 else "partly out")


def main():
    from transformers import AutoTokenizer
    lines = []
    say = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731
    for ds, (domain, min_chars) in SETS.items():
        rows = ragbench_rows(domain, min_chars)
        for canon in sorted((ROOT / "results" / "gasp_repro" / "canon_results").glob(f"*_{ds}_K5")):
            model = {"Qwen2.5": "Qwen/Qwen2.5-1.5B-Instruct", "SmolLM2": "HuggingFaceTB/SmolLM2-1.7B-Instruct"}[
                canon.name.split("-")[0]]
            ev = case_evidence(canon, rows, AutoTokenizer.from_pretrained(model))
            npz = ROOT / "results" / "features" / canon.name / "features_chunked_redeep.npz"
            z = np.load(npz)
            nwin = dict(zip(zip(z["case_id"], z["sent_idx"]), z["n_windows"]))
            dev = None
            for name, key in (("window1", "lookback"), ("max", "lookback_max")):
                df, cols = load_signals(canon, npz, lb_keys=(key,))
                d, _ = analyze_gasp.source_split(df, seed=0)
                d = d.reset_index(drop=True)
                if dev is None:
                    dev = d[["case_id", "sent_idx", "label", "source_id"]].copy()
                dev[name] = oof(d, cols)
            dev = dev[[nwin[(c, s)] > 1 for c, s in zip(dev["case_id"], dev["sent_idx"])]]
            n_cases = dev["case_id"].nunique()
            matched = sum(c in ev for c in dev["case_id"].unique())
            say(f"\n# {canon.name}: truncated dev rows {len(dev)} / {n_cases} cases ({matched} matched to RAGBench)")
            for kind in ("relevant", "utilized"):
                dev["g"] = [group(ev.get(c, {}).get(kind, np.nan)) for c in dev["case_id"]]
                share = dev.drop_duplicates("case_id")["g"].value_counts(normalize=True)
                say(f"  by {kind} evidence: cases " + ", ".join(f"{k} {v:.0%}" for k, v in share.items()))
                for g, x in dev.groupby("g"):
                    if x["label"].sum() < 10 or (1 - x["label"]).sum() < 10:
                        say(f"    {g:15s} n={len(x):5d}  (too few of one class)")
                        continue
                    from sklearn.metrics import roc_auc_score
                    a1, am = roc_auc_score(x["label"], x["window1"]), roc_auc_score(x["label"], x["max"])
                    m, lo, hi = paired_ci(x, "max", "window1")
                    say(f"    {g:15s} n={len(x):5d} src={x['source_id'].nunique():4d} halluc {x['label'].mean():.2f} | "
                        f"window1 {a1:.3f}  B {am:.3f}  B-window1 {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                        f"{' SIG' if lo > 0 or hi < 0 else ''}")
    (ROOT / "results" / "gating" / "evidence_position.txt").write_text("\n".join(lines) + "\n")
    print("\nsaved results/gating/evidence_position.txt")


if __name__ == "__main__":
    main()
