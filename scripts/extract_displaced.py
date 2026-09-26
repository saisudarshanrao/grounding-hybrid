"""Step E3: Lookback readings with the evidence displaced out of window 1, for one GASP run (CLAUDE.md, "E3 record").

For each response whose original context fits one window, the context is moved behind a 1808-token prefix of other
cases' contexts (src/grounding_hybrid/displace.py) and read by the frozen extractor with chunked=True:
  lookback      window 1 of the displaced context (distractor text only): w1d
  lookback_max  B over the displaced context (1800/256 windows, max per layer x head): Bd
The original reading (orig) is the frozen features.npz. Rows are keyed by (case_id, sent_idx) and cover the included
responses only; the run is gated on alignment (every GASP sentence of those responses covered, token counts equal,
no NaN), on the displacement (the original starts after window 1 ends, prefix of 1808 tokens) and on the donor rule
(other source, same split).

Output: <outroot>/<TAG>/features_displaced.npz (+ meta_displaced.json); with --max_cases N (first N included
responses), features_displaced_firstN.npz.

Usage:
    python scripts/extract_displaced.py --canon_dir results/gasp_repro/canon_results/Qwen2.5-1.5B-Instruct_ragtruth_K5 \
        --model Qwen/Qwen2.5-1.5B-Instruct [--max_cases 3]
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grounding_hybrid.displace import PREFIX_TOKENS, build_prefixes, displaced, included  # noqa: E402
from grounding_hybrid.gasp_bridge import load_cases, load_sentences  # noqa: E402
from grounding_hybrid.placebo import case_splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outroot", default=str(ROOT / "results" / "features"))
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    canon = Path(args.canon_dir)
    tag = canon.name
    if not tag.startswith(args.model.split("/")[-1] + "_"):
        sys.exit(f"{tag} was not scored by {args.model}: sentence rows would not align")
    out_dir = Path(args.outroot) / tag
    suffix = "_displaced" + (f"_first{args.max_cases}" if args.max_cases else "")
    out_file = out_dir / f"features{suffix}.npz"
    if out_file.exists():
        print(f"[skip] {out_file} exists")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from grounding_hybrid.extractor import SharedExtractor

    ex = SharedExtractor(args.model, device=args.device)
    cases = load_cases(canon)
    by_id = {c.case_id: c for c in cases}
    split_of = case_splits(canon)
    n_tokens = {c.case_id: len(ex.tok(c.context, add_special_tokens=False).input_ids) for c in cases}
    keep = included(cases, n_tokens, split_of, ex.max_ctx)
    prefixes = build_prefixes(cases, keep, split_of, ex.tok)       # over ALL included cases: a smoke uses the same
    todo = keep[: args.max_cases] if args.max_cases else keep
    print(f"{tag}: {len(keep)} of {len(cases)} responses fit one window; reading {len(todo)} displaced, "
          f"{args.model} on {ex.device}", flush=True)

    t0, rows, secs, checks = time.time(), [], [], []
    for i, case in enumerate(todo):
        tc = time.time()
        prefix, used = prefixes[case.case_id]
        dcase = displaced(case, prefix)
        w1_end = ex.windows(dcase)[0][1]
        checks.append(dict(case_id=case.case_id, prefix_tokens=len(ex.tok(prefix, add_special_tokens=False).input_ids),
                           window1_end=w1_end, original_start=len(prefix) + 2, donors=len(used),
                           donor_rule=all(by_id[d].source_id != case.source_id and split_of[d] == split_of[case.case_id]
                                          for d in used)))
        for r in ex.extract(dcase, chunked=True):
            rows.append(dict(case_id=case.case_id, **r))
        secs.append(time.time() - tc)
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            print(f"  {i + 1}/{len(todo)} responses, {(time.time() - t0) / (i + 1):.2f} s/response", flush=True)
    minutes = (time.time() - t0) / 60

    arrays = dict(
        case_id=np.array([r["case_id"] for r in rows]),
        sent_idx=np.array([r["sent_idx"] for r in rows], dtype=np.int32),
        n_tok=np.array([r["n_tok"] for r in rows], dtype=np.int32),
        logprob_full=np.array([r["logprob_full"] for r in rows], dtype=np.float32),
        lookback=np.stack([r["lookback"] for r in rows]).astype(np.float16),
        lookback_max=np.stack([r["lookback_max"] for r in rows]).astype(np.float16),
        n_windows=np.array([r["n_windows"] for r in rows], dtype=np.int32),
    )
    np.savez_compressed(out_file, **arrays)

    sent = load_sentences(canon)
    sent = sent[sent["case_id"].isin({c.case_id for c in todo})]
    key = {(r["case_id"], r["sent_idx"]): r for r in rows}
    hit = [key.get((c, s)) for c, s in zip(sent["case_id"], sent["sent_idx"])]
    covered = sum(h is not None for h in hit)
    ntok_ok = all(h is None or h["n_tok"] == n for h, n in zip(hit, sent["n_tok"]))
    nan_rows = sum(1 for r in rows if not (np.isfinite(r["lookback"]).all() and np.isfinite(r["lookback_max"]).all()))
    align = dict(gasp_rows=len(sent), covered=covered, extra=len(rows) - covered, n_tok_match=ntok_ok, nan_rows=nan_rows)
    disp = dict(responses=len(checks),
                original_after_window1=sum(c["window1_end"] <= len(prefixes[c["case_id"]][0]) for c in checks),
                prefix_tokens_ok=sum(c["prefix_tokens"] == PREFIX_TOKENS for c in checks),
                donor_rule_ok=sum(c["donor_rule"] for c in checks),
                windows=dict(zip(*np.unique(arrays["n_windows"], return_counts=True))) if rows else {})
    disp["windows"] = {int(k): int(v) for k, v in disp["windows"].items()}
    print("alignment vs GASP sentence.csv:", align, flush=True)
    print("displacement:", disp, flush=True)
    meta = dict(tag=tag, model=args.model, device=ex.device, dtype="float32", attention="eager",
                max_ctx_tokens=ex.max_ctx, overlap=ex.overlap, prefix_tokens=PREFIX_TOKENS, seed=0,
                n_included=len(keep), n_cases=len(todo), n_sentences=len(rows), n_layers=ex.n_layers, n_heads=ex.n_heads,
                features={"lookback": "window 1 of the displaced context (distractor text only)",
                          "lookback_max": "B: max over the displaced context's windows"},
                minutes=round(minutes, 2), seconds_median=float(np.median(secs)) if secs else None,
                alignment=align, displacement=disp, per_case=checks,
                env=dict(python=platform.python_version(), torch=torch.__version__))
    json.dump(meta, open(out_dir / f"meta{suffix}.json", "w"), indent=1)
    print(f"saved {len(rows)} sentences to {out_file} in {minutes:.1f} min")
    bad = (nan_rows or covered < len(sent) or not ntok_ok or disp["original_after_window1"] < len(checks)
           or disp["prefix_tokens_ok"] < len(checks) or disp["donor_rule_ok"] < len(checks))
    if bad:
        sys.exit(f"alignment or displacement check FAILED for {tag}: {align} {disp}")


if __name__ == "__main__":
    main()
