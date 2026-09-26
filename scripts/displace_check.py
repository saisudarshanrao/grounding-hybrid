"""Step E3 checks, written before the full run (CLAUDE.md, "E3 record"). Exit code 1 if any check fails.

  (a) for EVERY included response: the displaced context's window 1 ends before the original context starts
      (char offsets), and the prefix has 1808 tokens
  (b) with an empty prefix, the displaced reading equals the frozen reading of the original case (lookback,
      lookback_max, log-probs; max |diff| <= 1e-6), on the first --n included responses
  (d) every donor of every included response has a different source_id and the same split
((c), alignment, is checked by extract_displaced.py itself.)

Usage:
    python scripts/displace_check.py --canon_dir results/gasp_repro/canon_results/Qwen2.5-1.5B-Instruct_ragtruth_K5 \
        --model Qwen/Qwen2.5-1.5B-Instruct --n 2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grounding_hybrid.displace import PREFIX_TOKENS, build_prefixes, displaced, included  # noqa: E402
from grounding_hybrid.extractor import SharedExtractor  # noqa: E402
from grounding_hybrid.gasp_bridge import load_cases  # noqa: E402
from grounding_hybrid.placebo import case_splits  # noqa: E402

TOL = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="write the report here (json)")
    args = ap.parse_args()
    cases = load_cases(args.canon_dir)
    by_id = {c.case_id: c for c in cases}
    ex = SharedExtractor(args.model, device=args.device)
    split_of = case_splits(args.canon_dir)
    n_tokens = {c.case_id: len(ex.tok(c.context, add_special_tokens=False).input_ids) for c in cases}
    keep = included(cases, n_tokens, split_of, ex.max_ctx)
    prefixes = build_prefixes(cases, keep, split_of, ex.tok)
    rep = dict(canon=Path(args.canon_dir).name, model=args.model, responses=len(cases), included=len(keep))

    # (a) and (d) over every included response
    after, ntok_ok, donors_ok, ntoks = 0, 0, 0, []
    for c in keep:
        prefix, used = prefixes[c.case_id]
        after += ex.windows(displaced(c, prefix))[0][1] <= len(prefix)
        nt = len(ex.tok(prefix, add_special_tokens=False).input_ids)
        ntoks.append(nt)
        ntok_ok += nt == PREFIX_TOKENS
        donors_ok += all(by_id[d].source_id != c.source_id and split_of[d] == split_of[c.case_id] for d in used)
    rep["a"] = dict(original_after_window1=after, prefix_tokens_ok=ntok_ok, prefix_tokens_min=min(ntoks),
                    prefix_tokens_max=max(ntoks), **{"pass": bool(after == len(keep) and ntok_ok == len(keep))})
    rep["d"] = dict(donor_rule_ok=donors_ok, **{"pass": bool(donors_ok == len(keep))})
    print("(a)", json.dumps(rep["a"]), "\n(d)", json.dumps(rep["d"]), flush=True)

    # (b) empty prefix = the frozen reading of the original case
    d_lb, d_max, d_lp, sents = 0.0, 0.0, 0.0, 0
    for c in keep[: args.n]:
        ref = {r["sent_idx"]: r for r in ex.extract(c, chunked=True)}
        for r in ex.extract(displaced(c, ""), chunked=True):
            e = ref[r["sent_idx"]]
            sents += 1
            d_lb = max(d_lb, float(np.abs(r["lookback"] - e["lookback"]).max()))
            d_max = max(d_max, float(np.abs(r["lookback_max"] - e["lookback_max"]).max()))
            d_lp = max(d_lp, abs(r["logprob_full"] - e["logprob_full"]))
    rep["b"] = dict(responses=min(args.n, len(keep)), sentences=sents, lookback_absdiff_max=d_lb,
                    lookback_max_absdiff_max=d_max, logprob_absdiff_max=d_lp,
                    **{"pass": bool(sents > 0 and max(d_lb, d_max, d_lp) <= TOL)})
    print("(b)", json.dumps(rep["b"]), flush=True)

    ok = rep["a"]["pass"] and rep["b"]["pass"] and rep["d"]["pass"]
    rep["verdict"] = "PASS" if ok else "FAIL"
    print(json.dumps(rep, indent=1), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(args.out, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
