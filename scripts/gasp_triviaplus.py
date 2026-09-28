"""Run GASP's own pipeline (run_gasp.py, unmodified) on TRIVIA+-long (step E4).

TRIVIA+ (Amazon Science, ACL 2026; github.com/amazon-science/hallucination-benchmark-trivialplus) holds 3,224
responses by Mixtral 8x7B, Claude and Gemma-7B to DROP, MS MARCO, NQ, TriviaQA and CovidQA questions, with human
sentence labels. Licence CC BY-NC-ND 4.0, research use only: the parquet and every converted copy (the GASP run
folders) stay local or in the Kaggle session and are never committed or shared.

TRIVIA+-long keeps the responses whose `article` has at least --min_ctx_chars characters, pooled over the official
splits (GASP's source split is used instead). The cutoff is the smallest multiple of 1,000 at which every kept
article exceeds the 1,800-token window for both scorers (E4 record in CLAUDE.md; triviaplus_check.py --cutoff).

Conversion, as gasp_longctx.py does for RAGBench: context = article, query = question, the answer is rebuilt from
answer_sentence_list by GASP's _spans_from_sentences, a sentence is hallucinated if its majority vote is
contradicts or not-mentioned (the dataset's own mapping), the response label is response_level_label_binary,
source_id = md5 of the article (every response to one article sits in one split), and GASP's balanced sampling
(n_per_class, seed). build_case re-splits the answer with GASP's splitter; labels follow the hallucinated spans.

It patches run_gasp.load_ragbench_cases in memory and calls run_gasp.main(), so scoring and every output file are
GASP's code. Called by reproduce_gasp.py for "triviapluslong"; takes run_gasp.py's arguments plus --parquet and
--min_ctx_chars.

    python scripts/gasp_triviaplus.py --parquet results/data/triviaplus/triviaplus_dataset.parquet \
        --min_ctx_chars 9000 --model Qwen/Qwen2.5-1.5B-Instruct --dataset ragbench \
        --tag Qwen2.5-1.5B-Instruct_triviapluslong_K5 --outroot results/gasp_repro/canon_results
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "GASP" / "pipeline"))

import run_gasp  # noqa: E402

DEFAULT_PARQUET = os.environ.get("TRIVIAPLUS_PARQUET",
                                 str(ROOT / "results" / "data" / "triviaplus" / "triviaplus_dataset.parquet"))
UNFAITHFUL = {"contradicts", "not-mentioned"}          # TRIVIA+'s binary mapping (its DATA_DETAILS.md)
FAITHFUL = {"supports", "supported", "supplementary"}


def norm_label(x):
    return str(x).strip().lower().replace("_", "-").replace(" ", "-")


def load_pool(parquet=DEFAULT_PARQUET, min_ctx_chars=0):
    """TRIVIA+ responses as GASP-style records (all official splits), articles with >= min_ctx_chars characters.
    An unknown sentence label raises: the mapping must cover every value in the file."""
    import pandas as pd
    df = pd.read_parquet(parquet)
    pool = []
    for e in df.itertuples(index=False):
        ctx = str(e.article if e.article is not None else "").strip()
        sents = [str(s) for s in list(e.answer_sentence_list)]
        labs = [norm_label(v) for v in list(e.sentence_level_majority_vote)]
        unknown = set(labs) - UNFAITHFUL - FAITHFUL
        if unknown:
            raise ValueError(f"unknown TRIVIA+ sentence labels: {sorted(unknown)}")
        keep = [(s.strip(), lab) for s, lab in zip(sents, labs) if s.strip()]
        if not ctx or not keep or len(sents) != len(labs) or len(ctx) < min_ctx_chars:
            continue
        pool.append(dict(task=str(e.source).strip().lower().replace("_", ""),     # "ms_marco" = "msmarco"
                         split=str(e.split), generator=str(e.model),
                         query=str(e.question or "").strip(), context=ctx,
                         sents=[s for s, _ in keep], labels=[1 if lab in UNFAITHFUL else 0 for _, lab in keep],
                         resp_label=int(e.response_level_label_binary),
                         source_id="triviaplus_" + hashlib.md5(ctx.encode("utf-8")).hexdigest()[:12]))
    return pool


def balanced(pool, n_per_class, seed):
    """GASP's balanced response-level sampling (load_ragbench_cases): one permutation, n per class."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool))
    pos, neg = [], []
    for i in idx:
        r = pool[int(i)]
        b = pos if r["resp_label"] == 1 else neg
        if len(b) < n_per_class:
            b.append(r)
        if len(pos) >= n_per_class and len(neg) >= n_per_class:
            break
    n = min(len(pos), len(neg))
    return pos[:n] + neg[:n]


def load_cases(parquet, n_per_class, k_chunks, seed, min_ctx_chars):
    pool = load_pool(parquet, min_ctx_chars)
    sample = balanced(pool, n_per_class, seed)
    print(f"      TRIVIA+ (all splits, article >= {min_ctx_chars} chars) balanced: "
          f"{len(sample) // 2}/class from {len(pool)} pooled")
    cases = []
    for j, r in enumerate(sample):
        answer, hspans = run_gasp._spans_from_sentences(r["sents"], r["labels"])
        case = run_gasp.build_case(case_id=f"{r['source_id']}::{j}", source_id=r["source_id"], answer_id=str(j),
                                   dataset="triviaplus", task=r["task"], query=r["query"], context=r["context"],
                                   answer=answer, k_chunks=k_chunks, halluc_spans=hspans)
        if case is not None:
            cases.append(case)
    return cases


def pop_arg(argv, name, default=None):
    if name not in argv:
        return default
    i = argv.index(name)
    val = argv[i + 1]
    del argv[i:i + 2]
    return val


def main():
    argv = sys.argv[1:]
    parquet = pop_arg(argv, "--parquet", DEFAULT_PARQUET)
    min_chars = int(pop_arg(argv, "--min_ctx_chars", "0"))
    run_gasp.load_ragbench_cases = lambda n, k, s: load_cases(parquet, n, k, s, min_chars)
    sys.argv = [str(Path(run_gasp.__file__))] + argv
    run_gasp.main()


if __name__ == "__main__":
    main()
