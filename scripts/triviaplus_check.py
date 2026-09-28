"""E4 checks for TRIVIA+-long (E4 record in CLAUDE.md); any failure is a bug.

--cutoff fixes C, the smallest multiple of 1,000 characters at which every article with at least C characters
exceeds the 1,800-token window for both scorers' tokenizers. It reads article lengths only and runs before any
detector.

--canon_dir + --model checks one GASP run folder of TRIVIA+-long:
  (a) conversion: each case matches its TRIVIA+ response (article, question, rebuilt answer, response label);
      every TRIVIA+ sentence overlaps at least one GASP sentence; each GASP sentence's label (GASP's sentence.csv)
      equals the OR of the labels of the TRIVIA+ sentences it overlaps; sentence counts are reported, not required
      to match (GASP re-splits the rebuilt answer with its own splitter);
  (b) every article exceeds 1,800 tokens for the scorer's tokenizer;
  (d) source_ids group responses by article (one id per article text) and no source is in both dev and test.

    python scripts/triviaplus_check.py --cutoff
    python scripts/triviaplus_check.py --canon_dir results/gasp_repro/canon_results/<TAG> --model <model> \
        --min_ctx_chars <C> --out <json>
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gasp_triviaplus import DEFAULT_PARQUET, balanced, load_pool, run_gasp  # noqa: E402

WINDOW = 1800


def n_tokens(tok, texts):
    return [len(tok(t, add_special_tokens=False).input_ids) for t in texts]


def cutoff(parquet, models, n_per_class, seed):
    from transformers import AutoTokenizer
    pool = load_pool(parquet, 0)
    arts = {}
    for r in pool:
        arts.setdefault(r["source_id"], r["context"])
    ids = sorted(arts)
    chars = {i: len(arts[i]) for i in ids}
    toks = {m: dict(zip(ids, n_tokens(AutoTokenizer.from_pretrained(m), [arts[i] for i in ids]))) for m in models}
    over = {m: sum(v > WINDOW for v in toks[m].values()) for m in models}
    c = 1000
    while True:
        kept = [i for i in ids if chars[i] >= c]
        if all(toks[m][i] > WINDOW for m in models for i in kept):
            break
        c += 1000
    kept = set(i for i in ids if chars[i] >= c)
    resp = [r for r in pool if r["source_id"] in kept]
    lab = Counter(r["resp_label"] for r in resp)
    sample = balanced(resp, n_per_class, seed)
    srcs = sorted({r["source_id"] for r in sample})
    out = dict(C=c, responses_all=len(pool), articles_all=len(ids),
               articles_over_window_all={m.split("/")[-1]: over[m] for m in models},
               responses_kept=len(resp), articles_kept=len(kept), kept_by_label={str(k): v for k, v in lab.items()},
               kept_by_benchmark=dict(Counter(r["task"] for r in resp)),
               balanced_per_class=len(sample) // 2, sample_responses=len(sample), sample_sources=len(srcs),
               sample_sentences=sum(len(r["sents"]) for r in sample),
               sample_unfaithful_sentences=sum(sum(r["labels"]) for r in sample),
               min_tokens_kept={m.split("/")[-1]: min(toks[m][i] for i in kept) for m in models})
    print(json.dumps(out, indent=1))
    return out


def check_run(canon_dir, model, parquet, min_chars, n_per_class, seed):
    import pandas as pd
    from transformers import AutoTokenizer
    from grounding_hybrid.gasp_bridge import analyze_gasp, load_cases, load_sentences
    sample = balanced(load_pool(parquet, min_chars), n_per_class, seed)
    cases = load_cases(canon_dir)
    sent = pd.read_csv(Path(canon_dir) / "sentence.csv")
    slab = {(r.case_id, int(r.sent_idx)): int(r.label) for r in sent.itertuples(index=False)}
    fails, n_sent_trivia, n_sent_gasp, uncovered = [], 0, 0, 0
    for c in cases:
        r = sample[int(c.case_id.split("::")[-1])]
        answer, _ = run_gasp._spans_from_sentences(r["sents"], r["labels"])
        if (c.source_id, c.context, c.query, c.answer) != (r["source_id"], r["context"], r["query"], answer.strip()):
            fails.append(f"(a) {c.case_id}: case does not match its TRIVIA+ response")
            continue
        ranges, st = [], 0
        for s in r["sents"]:
            ranges.append((st, st + len(s)))
            st += len(s) + 1
        n_sent_trivia += len(ranges)
        n_sent_gasp += len(c.sent_spans)
        for a, b in ranges:
            if not any(not (e <= a or s >= b) for s, e in c.sent_spans):
                uncovered += 1
        for j, (s, e) in enumerate(c.sent_spans):
            want = int(any(lab for (a, b), lab in zip(ranges, r["labels"]) if not (b <= s or a >= e)))
            got = slab.get((c.case_id, j))
            if got is not None and got != want:
                fails.append(f"(a) {c.case_id} sentence {j}: GASP label {got}, TRIVIA+ {want}")
    if uncovered:
        fails.append(f"(a) {uncovered} TRIVIA+ sentences overlap no GASP sentence")
    tok = AutoTokenizer.from_pretrained(model)
    short = [c.case_id for c, n in zip(cases, n_tokens(tok, [c.context for c in cases])) if n <= WINDOW]
    if short:
        fails.append(f"(b) {len(short)} articles fit one window, e.g. {short[:3]}")
    by_ctx, by_src = defaultdict(set), defaultdict(set)
    for c in cases:
        by_ctx[c.context].add(c.source_id)
        by_src[c.source_id].add(c.context)
    if any(len(v) > 1 for v in by_ctx.values()) or any(len(v) > 1 for v in by_src.values()):
        fails.append("(d) source_id does not match article one to one")
    dev, test = analyze_gasp.source_split(load_sentences(canon_dir), seed=0)
    if set(dev["source_id"]) & set(test["source_id"]):
        fails.append("(d) a source is in both dev and test")
    out = dict(canon_dir=str(canon_dir), model=model, cases=len(cases), trivia_sentences=n_sent_trivia,
               gasp_sentences=n_sent_gasp, gasp_rows=len(sent), dev_sources=int(dev["source_id"].nunique()),
               test_sources=int(test["source_id"].nunique()), failures=fails, passed=not fails)
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--cutoff", action="store_true")
    ap.add_argument("--canon_dir")
    ap.add_argument("--model")
    ap.add_argument("--min_ctx_chars", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="unused; accepted for run_features' check interface")
    ap.add_argument("--out")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "reproduce.yaml"))
    n_per_class, seed = cfg["params"]["n_per_class"], cfg["params"]["seed"]
    if args.cutoff:
        out = cutoff(args.parquet, cfg["models"], n_per_class, seed)
    else:
        out = check_run(args.canon_dir, args.model, args.parquet, args.min_ctx_chars, n_per_class, seed)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    sys.exit(0 if out.get("passed", True) else 1)


if __name__ == "__main__":
    main()
