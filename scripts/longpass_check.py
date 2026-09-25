"""Step E1 checks, written before the full run (CLAUDE.md, "E1 record"). Exit code 1 if any check fails.

  (a) answer rows: with eager attention, the recomputed answer-row attention equals the model's own attention
      weights in the same forward pass (max |diff| <= 1e-5)
  (b) window 1: Lookback features from the new path (SDPA, answer rows only, context cut at 1800 tokens) equal the
      frozen eager extractor's on the same cases (max |diff| <= 1e-3, mean <= 1e-4), with equal token counts and
      answer log-probs within 1e-3
  plus a sample of real long passes (context as long as the model allows): sequence length, seconds, peak memory.

Usage:
    python scripts/longpass_check.py --canon_dir results/gasp_repro/canon_results/Qwen2.5-1.5B-Instruct_techqa_K5 \
        --model Qwen/Qwen2.5-1.5B-Instruct --n 3 [--no_long]
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grounding_hybrid.extractor import SharedExtractor  # noqa: E402
from grounding_hybrid.gasp_bridge import load_cases  # noqa: E402
from grounding_hybrid.longpass import LongPass  # noqa: E402

TOL_A, TOL_B_MAX, TOL_B_MEAN, TOL_LP = 1e-5, 1e-3, 1e-4, 1e-3


def cleanup():
    """Release a model the caller has already dropped (set to None)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20, help="cases for check (b) and the long-pass sample")
    ap.add_argument("--n_a", type=int, default=2, help="cases for check (a)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_long", action="store_true", help="skip the long-pass sample (slow on a Mac)")
    ap.add_argument("--out", default=None, help="write the report here (json)")
    args = ap.parse_args()
    cases = load_cases(args.canon_dir)[: args.n]
    rep = dict(canon=Path(args.canon_dir).name, model=args.model, n=len(cases))

    # (b) reference: the frozen eager extractor, window 1
    print(f"(b) eager reference on {len(cases)} cases", flush=True)
    ex = SharedExtractor(args.model, device=args.device)
    ref = {(c.case_id, r["sent_idx"]): r for c in cases for r in ex.extract(c)}
    device = ex.device
    ex = None
    cleanup()

    print("(b) new path, context cut at 1800 tokens", flush=True)
    lp = LongPass(args.model, device=device, attn="sdpa")
    diffs, lpd, ntok_ok, covered = [], [], True, 0
    for c in cases:
        rows, _ = lp.extract(c, limit=1800)
        for r in rows:
            e = ref.get((c.case_id, r["sent_idx"]))
            if e is None:
                continue
            covered += 1
            ntok_ok &= e["n_tok"] == r["n_tok"]
            diffs.append(np.abs(e["lookback"] - r["lookback"]))
            lpd.append(abs(e["logprob_full"] - r["logprob_full"]))
    d = np.stack(diffs)
    rep["b"] = dict(sentences=covered, reference_sentences=len(ref), n_tok_match=bool(ntok_ok),
                    lookback_absdiff_max=float(d.max()), lookback_absdiff_mean=float(d.mean()),
                    logprob_absdiff_max=float(max(lpd)))
    rep["b"]["pass"] = bool(covered == len(ref) and ntok_ok and d.max() <= TOL_B_MAX and d.mean() <= TOL_B_MEAN
                            and max(lpd) <= TOL_LP)

    if not args.no_long:   # real long passes on the same cases
        print("long-pass sample", flush=True)
        runs = []
        for c in cases:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            rows, info = lp.extract(c)
            if not rows:
                continue
            info.update(seconds=round(time.time() - t0, 2),
                        peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
                        finite=bool(all(np.isfinite(r["lookback"]).all() and np.isfinite(r["logprob_full"]) for r in rows)))
            runs.append(info)
        rep["long_sample"] = dict(cases=len(runs), max_seq_len=max(r["seq_len"] for r in runs),
                                  cut=sum(r["ctx_cut"] for r in runs), all_finite=all(r["finite"] for r in runs),
                                  max_seconds=max(r["seconds"] for r in runs),
                                  max_peak_gb=max((r["peak_gb"] or 0) for r in runs))
    lp = None
    cleanup()

    # (a) recomputed answer rows vs the eager weights of the same pass
    print("(a) answer rows vs eager weights", flush=True)
    la = LongPass(args.model, device=device, attn="eager")
    worst = 0.0
    for c in cases[: args.n_a]:
        _, info = la.extract(c, limit=1800, check=True)
        worst = max(worst, info.get("answer_rows_vs_eager_absdiff_max", 0.0))
    la = None
    cleanup()
    rep["a"] = dict(cases=min(args.n_a, len(cases)), answer_rows_absdiff_max=worst, **{"pass": bool(worst <= TOL_A)})

    ok = rep["a"]["pass"] and rep["b"]["pass"] and rep.get("long_sample", {}).get("all_finite", True)
    rep["verdict"] = "PASS" if ok else "FAIL"
    print(json.dumps(rep, indent=1), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(args.out, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
