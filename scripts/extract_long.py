"""Step E1: Lookback features from ONE long pass per response (baseline L), for one GASP run.

The prompt holds as much of the context as the scorer's position limit allows (Qwen2.5-1.5B: 32,768 positions,
SmolLM2-1.7B: 8,192), minus template, question and answer; see src/grounding_hybrid/longpass.py. Rows are keyed
by (case_id, sent_idx) like every other feature file, and the run is gated on the same alignment checks
(every GASP sentence covered, equal token counts, no NaN). Answer log-probs are compared with GASP's only for
contexts that fit GASP's 1800-token window, the one case where the two passes read the same text.

Output: <outroot>/<TAG>/features_long.npz (+ meta_long.json); with --max_cases N, features_long_firstN.npz.
Per sentence: lookback [layers, heads], logprob_full, n_tok, and its response's n_ctx_tokens, ctx_read_tokens,
ctx_cut (context longer than the model allows) and seq_len. Per response, meta_long.json keeps seconds and peak
GPU memory.

Usage:
    python scripts/extract_long.py --canon_dir results/gasp_repro/canon_results/Qwen2.5-1.5B-Instruct_techqa_K5 \
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

from grounding_hybrid.gasp_bridge import load_cases, load_sentences  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outroot", default=str(ROOT / "results" / "features"))
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    canon = Path(args.canon_dir)
    tag = canon.name
    if not tag.startswith(args.model.split("/")[-1] + "_"):
        sys.exit(f"{tag} was not scored by {args.model}: sentence rows would not align")
    out_dir = Path(args.outroot) / tag
    suffix = "_long" + (f"_first{args.max_cases}" if args.max_cases else "")
    out_file = out_dir / f"features{suffix}.npz"
    if out_file.exists():
        print(f"[skip] {out_file} exists")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from grounding_hybrid.longpass import LongPass

    cases = load_cases(canon)
    if args.max_cases:
        cases = cases[: args.max_cases]
    lp = LongPass(args.model, device=args.device, dtype=args.dtype)
    print(f"{tag}: {len(cases)} cases, {args.model} on {lp.device}, {lp.n_layers} layers x {lp.n_heads} heads, "
          f"{args.dtype}, up to {lp.max_pos} positions", flush=True)

    cuda = torch.cuda.is_available()
    t0, rows, per_case = time.time(), [], []
    for i, case in enumerate(cases):
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        tc = time.time()
        out, info = lp.extract(case)
        if out:
            info.update(case_id=case.case_id, seconds=round(time.time() - tc, 3),
                        peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3) if cuda else None)
            per_case.append(info)
            rows += [dict(case_id=case.case_id, **r, **{k: info[k] for k in
                                                         ("n_ctx_tokens", "ctx_read_tokens", "ctx_cut", "seq_len")})
                     for r in out]
        if (i + 1) % 50 == 0 or i + 1 == len(cases):
            print(f"  {i + 1}/{len(cases)} cases, {(time.time() - t0) / (i + 1):.2f} s/case", flush=True)
    minutes = (time.time() - t0) / 60

    arrays = dict(
        case_id=np.array([r["case_id"] for r in rows]),
        sent_idx=np.array([r["sent_idx"] for r in rows], dtype=np.int32),
        n_tok=np.array([r["n_tok"] for r in rows], dtype=np.int32),
        logprob_full=np.array([r["logprob_full"] for r in rows], dtype=np.float32),
        lookback=np.stack([r["lookback"] for r in rows]).astype(np.float16),
        n_ctx_tokens=np.array([r["n_ctx_tokens"] for r in rows], dtype=np.int32),
        ctx_read_tokens=np.array([r["ctx_read_tokens"] for r in rows], dtype=np.int32),
        ctx_cut=np.array([r["ctx_cut"] for r in rows], dtype=bool),
        seq_len=np.array([r["seq_len"] for r in rows], dtype=np.int32),
    )
    np.savez_compressed(out_file, **arrays)

    # alignment with GASP's sentence.csv: same rows and token counts, no NaN
    sent = load_sentences(canon)
    sent = sent[sent["case_id"].isin({c.case_id for c in cases})]
    key = {(r["case_id"], r["sent_idx"]): r for r in rows}
    hit = [key.get((c, s)) for c, s in zip(sent["case_id"], sent["sent_idx"])]
    covered = sum(h is not None for h in hit)
    ntok_ok = all(h is None or h["n_tok"] == n for h, n in zip(hit, sent["n_tok"]))
    nan_rows = sum(1 for r in rows if not (np.isfinite(r["logprob_full"]) and np.isfinite(r["lookback"]).all()))
    same_text = np.array([abs(h["logprob_full"] + m) for h, m in zip(hit, sent["mean_surprisal"])
                          if h is not None and h["n_ctx_tokens"] <= 1800])
    check = dict(gasp_rows=len(sent), covered=covered, extra=len(rows) - covered, n_tok_match=ntok_ok,
                 nan_rows=nan_rows, fits_1800_sentences=int(same_text.size),
                 logprob_absdiff_mean_fits_1800=float(same_text.mean()) if same_text.size else None)
    secs = np.array([c["seconds"] for c in per_case])
    peaks = np.array([c["peak_gb"] for c in per_case if c["peak_gb"] is not None])
    cost = dict(responses=len(per_case), cut=int(sum(c["ctx_cut"] for c in per_case)),
                seq_len_median=int(np.median([c["seq_len"] for c in per_case])),
                seq_len_max=int(max(c["seq_len"] for c in per_case)),
                seconds_median=float(np.median(secs)), seconds_max=float(secs.max()),
                peak_gb_median=float(np.median(peaks)) if peaks.size else None,
                peak_gb_max=float(peaks.max()) if peaks.size else None)
    print("alignment vs GASP sentence.csv:", check, flush=True)
    print("cost:", cost, flush=True)
    meta = dict(tag=tag, model=args.model, device=lp.device, dtype=args.dtype, attention="sdpa (memory-efficient) + "
                "answer-row recompute", max_positions=lp.max_pos, n_cases=len(cases), n_sentences=len(rows),
                n_layers=lp.n_layers, n_heads=lp.n_heads,
                features={"lookback": "A_ctx/(A_ctx+A_new) per layer x head, mean over sentence tokens, ONE pass over "
                                      "as much context as the model's positions allow",
                          "logprob_full": "mean token log-prob with that context"},
                minutes=round(minutes, 2), alignment=check, cost=cost, per_case=per_case,
                env=dict(python=platform.python_version(), torch=torch.__version__))
    json.dump(meta, open(out_dir / f"meta{suffix}.json", "w"), indent=1)
    print(f"saved {len(rows)} sentences to {out_file} in {minutes:.1f} min")
    if nan_rows or covered < len(sent) or not ntok_ok:
        sys.exit(f"alignment FAILED for {tag}: {check}")


if __name__ == "__main__":
    main()
