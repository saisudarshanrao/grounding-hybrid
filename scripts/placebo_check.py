"""Step E2 checks, written before the full run (CLAUDE.md, "E2 record"). Exit code 1 if any check fails.

  (a) with the donor set to the case itself, the placebo reading equals B from the frozen extractor on the same
      cases (lookback_placebo_max = lookback_max and window 1 = lookback, max |diff| <= 1e-6; same code path, so
      0.0 is expected), with the same number of windows and log-probs
  (c) the donor map for the whole run keeps the rule: different source_id, same split, >= K windows (or a logged
      fallback), and it is deterministic (built twice, identical)
((b), alignment, is checked by extract_placebo.py itself.)

Usage:
    python scripts/placebo_check.py --canon_dir results/gasp_repro/canon_results/Qwen2.5-1.5B-Instruct_techqa_K5 \
        --model Qwen/Qwen2.5-1.5B-Instruct --n 2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grounding_hybrid.extractor import SharedExtractor  # noqa: E402
from grounding_hybrid.gasp_bridge import load_cases  # noqa: E402
from grounding_hybrid.placebo import case_splits, donor_map  # noqa: E402

TOL = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="write the report here (json)")
    args = ap.parse_args()
    all_cases = load_cases(args.canon_dir)
    cases = all_cases[: args.n]
    ex = SharedExtractor(args.model, device=args.device)
    rep = dict(canon=Path(args.canon_dir).name, model=args.model, n=len(cases))

    # (c) donor rule over the whole run, and determinism
    by_id = {c.case_id: c for c in all_cases}
    split_of = case_splits(args.canon_dir)
    n_windows = {c.case_id: len(ex.windows(c)) for c in all_cases}
    donors = donor_map(all_cases, n_windows, split_of)
    same = donors == donor_map(all_cases, n_windows, split_of)
    breaks = [c for c, (d, fb) in donors.items() if by_id[d].source_id == by_id[c].source_id
              or split_of[d] != split_of[c] or (not fb and n_windows[d] < n_windows[c])]
    need = sum(1 for c in all_cases if n_windows[c.case_id] >= 2 and c.case_id in split_of)
    rep["c"] = dict(cases=len(all_cases), with_donor=len(donors), needing_donor=need,
                    fallback=sum(fb for _, fb in donors.values()), rule_breaks=len(breaks), deterministic=same)
    rep["c"]["pass"] = bool(same and not breaks and len(donors) == need)
    print("(c)", json.dumps(rep["c"]), flush=True)

    # (a) self-donor placebo = B from the frozen extractor
    print(f"(a) self-donor placebo vs B on {len(cases)} cases", flush=True)
    d_w1, d_max, d_lp, sents, win_ok = 0.0, 0.0, 0.0, 0, True
    for c in cases:
        ref = {r["sent_idx"]: r for r in ex.extract(c, chunked=True)}
        for r in ex.extract_placebo(c, c):
            e = ref[r["sent_idx"]]
            sents += 1
            win_ok &= r["n_windows"] == e["n_windows"]
            d_w1 = max(d_w1, float(np.abs(r["lookback"] - e["lookback"]).max()))
            d_max = max(d_max, float(np.abs(r["lookback_placebo_max"] - e["lookback_max"]).max()))
            d_lp = max(d_lp, abs(r["logprob_full"] - e["logprob_full"]))
    rep["a"] = dict(sentences=sents, windows_match=bool(win_ok), window1_absdiff_max=d_w1,
                    placebo_self_vs_B_absdiff_max=d_max, logprob_absdiff_max=d_lp)
    rep["a"]["pass"] = bool(sents > 0 and win_ok and d_w1 <= TOL and d_max <= TOL and d_lp <= TOL)
    print("(a)", json.dumps(rep["a"]), flush=True)

    ok = rep["a"]["pass"] and rep["c"]["pass"]
    rep["verdict"] = "PASS" if ok else "FAIL"
    print(json.dumps(rep, indent=1), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(args.out, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
