"""Step E5 checks (CLAUDE.md, "E5 record"), run before the full scoring; exit code 1 if any check fails.

  (a) each checker reproduces its README example: MiniCheck-Flan-T5-Large gives P(supported) 0.98059 / 0.00712 for the
      two README claims (within 1e-3); LettuceDetect flags the README's hallucinated sentence (a span overlapping
      characters 31-71 with confidence >= 0.5) and nothing before it (the README shows the base model; we use large)
  (b) every GASP sentence of the first n cases gets a finite score from each checker
  (c) every LettuceDetect answer token maps to exactly one GASP sentence (its start offset lies inside one span, GASP's
      token-to-sentence rule; spans are disjoint) and every GASP sentence gets >= 1 token
  (d) logged: contexts that needed more than one LettuceDetect chunk and contexts whose prompt was truncated
"GASP sentence" is a sentence GASP scores, i.e. a row of a run's sentence.csv (CLARIFICATION 28 Sep ~17:25 IST, before
any E5 result: cases.jsonl also lists spans GASP skips, e.g. a lone "-" of fewer than 3 scorer tokens; such spans can
hold no answer token and were wrongly counted in smoke #2). They are reported, not failed.

    python scripts/checkers_check.py --canon_dirs <Qwen TAG dir> <SmolLM2 TAG dir> --n 20 --out <json>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_checkers import MINICHECK, Lettuce, load_cases, sentence_max  # noqa: E402

DOC = "A group of students gather in the school library to study for their upcoming final exams."
CLAIMS = ["The students are preparing for an examination.", "The students are on vacation."]
MC_REF = [0.9805923700332642, 0.007121330592781305]
L_CTX = ["France is a country in Europe. The capital of France is Paris. The population of France is 67 million."]
L_Q = "What is the capital of France? What is the population of France?"
L_ANS = "The capital of France is Paris. The population of France is 69 million."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_dirs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--out")
    args = ap.parse_args()
    from minicheck.minicheck import MiniCheck
    mc = MiniCheck(model_name=MINICHECK, cache_dir=args.cache_dir)
    lt = Lettuce(args.device)
    rep, fails = {}, []

    labels, probs, _, _ = mc.score(docs=[DOC, DOC], claims=CLAIMS)
    diff = float(np.max(np.abs(np.array(probs, float) - np.array(MC_REF))))
    spans = lt.det.predict(L_CTX, L_ANS, L_Q, output_format="spans")
    hit = [s for s in spans if s["end"] > 31 and s["start"] < 71 and s["confidence"] >= 0.5]
    early = [s for s in spans if s["start"] < 31]
    rep["a"] = dict(minicheck_labels=[int(x) for x in labels], minicheck_probs=[float(x) for x in probs],
                    minicheck_absdiff_max=diff, lettuce_spans=spans,
                    **{"pass": bool(diff <= 1e-3 and list(labels) == [1, 0] and hit and not early)})
    if not rep["a"]["pass"]:
        fails.append("(a) README examples not reproduced")

    cases, scored = {}, set()
    for cd in args.canon_dirs:
        for c in load_cases(cd):
            cases.setdefault(c["case_id"], c)
        s = pd.read_csv(Path(cd) / "sentence.csv", usecols=["case_id", "sent_idx"])
        scored |= set(zip(s["case_id"], s["sent_idx"].astype(int)))
    ids = sorted(cases)[: args.n]
    nan_mc = nan_l = cross = empty = multi = trunc = n_sent = n_spans = skipped_empty = 0
    for cid in ids:
        c = cases[cid]
        sp = [tuple(x) for x in c["sent_spans"]]
        texts = [c["answer"][s:e].strip() for s, e in sp]
        _, sup, _, _ = mc.score(docs=[c["context"]] * len(texts), claims=texts)
        toks, n_chunks, tr = lt.token_probs(c["context"], c["query"], c["answer"])
        lett = sentence_max(toks, sp)
        keep = [(cid, j) in scored for j in range(len(sp))]          # the GASP sentences (sentence.csv rows)
        n_spans += len(sp)
        n_sent += sum(keep)
        nan_mc += int(np.sum(~np.isfinite(np.asarray(sup, float)[keep])))
        nan_l += int(np.sum(~np.isfinite(np.asarray(lett, float)[keep])))
        for ts, te, _ in toks:
            if sum(1 for s, e in sp if s <= ts < e) != 1:        # start in no span (or, impossibly, in two)
                cross += 1
        no_tok = [not any(s <= ts < e for ts, _, _ in toks) for s, e in sp]
        empty += sum(k and t for k, t in zip(keep, no_tok))
        skipped_empty += sum((not k) and t for k, t in zip(keep, no_tok))
        multi += int(n_chunks > 1)
        trunc += int(tr)
    rep["b"] = dict(cases=len(ids), sentences=n_sent, spans_listed=n_spans, nan_minicheck=nan_mc, nan_lettuce=nan_l,
                    **{"pass": nan_mc == 0 and nan_l == 0})
    rep["c"] = dict(tokens_not_in_one_sentence=cross, sentences_without_tokens=empty,
                    skipped_spans_without_tokens=skipped_empty, **{"pass": cross == 0 and empty == 0})
    rep["d"] = dict(multi_chunk_contexts=multi, truncated_contexts=trunc)
    for k in ("b", "c"):
        if not rep[k]["pass"]:
            fails.append(f"({k}) failed: {rep[k]}")
    rep["failures"], rep["verdict"] = fails, "PASS" if not fails else "FAIL"
    print(json.dumps(rep, indent=1, default=str), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(args.out, "w"), indent=1, default=str)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
