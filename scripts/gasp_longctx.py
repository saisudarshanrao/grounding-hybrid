"""Run GASP's own pipeline (run_gasp.py, unmodified) on one long-context RAGBench domain.

GASP's RAGBench sample pools six domains from the test split only, so it holds just 49 TechQA
cases, the domain where most of the context falls outside the 1800-token window. To test
coverage-aware reading on real truncation, this draws a larger sample of one domain: the same
loader logic as GASP's load_ragbench_cases (same fields, filters, labels, source_id and balanced
n_per_class sampling with the same seed), but restricted to --domain and pooled over the train,
validation and test splits of RAGBench (GPT-4 labels, as in GASP's sample; only our detectors are
trained on them, on GASP's source-level dev split of this sample).

--min_ctx_chars keeps only responses whose joined context has at least that many characters (a
tokenizer-free cutoff, so both scorers get the same cases): ExpertQA-long uses 9000, the smallest
round cutoff at which every context exceeds the 1800-token window for both Qwen2.5 and SmolLM2
(fixed on 24 Sep 2026 from context lengths only, before any detector result).

It patches run_gasp.load_ragbench_cases in memory and then calls run_gasp.main(), so scoring,
chunking, sentence spans and every output file are GASP's code. Called by reproduce_gasp.py for
the datasets in LONGCTX_DOMAINS; takes run_gasp.py's arguments plus --domain [--min_ctx_chars].

    python scripts/gasp_longctx.py --domain techqa --model Qwen/Qwen2.5-1.5B-Instruct \
        --dataset ragbench --tag Qwen2.5-1.5B-Instruct_techqa_K5 --outroot results/gasp_repro/canon_results
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "GASP" / "pipeline"))

import run_gasp  # noqa: E402

SPLITS = ("train", "validation", "test")


def load_domain_cases(domain, n_per_class, k_chunks, seed, min_ctx_chars=0):
    """GASP's load_ragbench_cases for one domain, pooled over all RAGBench splits."""
    from datasets import load_dataset
    pool = []
    for split in SPLITS:
        ds = load_dataset("rungalileo/ragbench", domain, split=split)
        for e in ds:
            docs = e.get("documents") or []
            rsents = e.get("response_sentences") or []
            if not docs or not (e.get("response") or "").strip() or not rsents:
                continue
            unsup = set(e.get("unsupported_response_sentence_keys") or [])
            ctx = "\n".join(str(d) for d in docs)
            if len(ctx) < min_ctx_chars:
                continue
            sid = domain + "_" + hashlib.md5(ctx.encode("utf-8")).hexdigest()[:12]
            pool.append(dict(domain=domain, query=(e.get("question") or "").strip(), context=ctx,
                             sents=[str(t) for _, t in rsents],
                             labels=[1 if str(k) in unsup else 0 for k, _ in rsents],
                             resp_label=0 if e.get("adherence_score") is True else 1, source_id=sid))
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
    sample = pos[:n] + neg[:n]
    print(f"      RAGBench {domain} (all splits, context >= {min_ctx_chars} chars) balanced: "
          f"{n}/class from {len(pool)} pooled")
    cases = []
    for j, r in enumerate(sample):
        answer, hspans = run_gasp._spans_from_sentences(r["sents"], r["labels"])
        case = run_gasp.build_case(case_id=f"{r['source_id']}::{j}", source_id=r["source_id"], answer_id=str(j),
                                   dataset="ragbench", task=r["domain"], query=r["query"], context=r["context"],
                                   answer=answer, k_chunks=k_chunks, halluc_spans=hspans)
        if case is not None:
            cases.append(case)
    return cases


def main():
    argv = sys.argv[1:]
    if "--domain" not in argv:
        sys.exit("--domain is required")
    i = argv.index("--domain")
    domain = argv[i + 1]
    del argv[i:i + 2]
    min_chars = 0
    if "--min_ctx_chars" in argv:
        j = argv.index("--min_ctx_chars")
        min_chars = int(argv[j + 1])
        del argv[j:j + 2]
    run_gasp.load_ragbench_cases = lambda n, k, s: load_domain_cases(domain, n, k, s, min_chars)
    sys.argv = [str(Path(run_gasp.__file__))] + argv
    run_gasp.main()


if __name__ == "__main__":
    main()
