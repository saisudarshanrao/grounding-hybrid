"""Step E2: placebo reading (Bp) for one GASP run: B's windows 2..K replaced by a donor's (CLAUDE.md, "E2 record").

Window 1 is the case's own (GASP's retained context, = Lookback); windows 2..K come from a donor case of the same
split with a different source_id (src/grounding_hybrid/placebo.py), each read with the case's own question and
answer; Bp = max over the K windows per layer x head, exactly as B. Rows are keyed by (case_id, sent_idx) and the
run is gated on the usual alignment checks (every GASP sentence covered, equal token counts, no NaN, window-1
log-probs within ~0.001 of GASP's) and on the donor rule (different source, same split, >= K windows or a logged
fallback).

Output: <outroot>/<TAG>/features_placebo.npz (+ meta_placebo.json); with --max_cases N, features_placebo_firstN.npz.
Per sentence: lookback (window 1), lookback_placebo_max, logprob_full, n_tok, n_windows, donor_case_id, fallback.

Usage:
    python scripts/extract_placebo.py --canon_dir results/gasp_repro/canon_results/Qwen2.5-1.5B-Instruct_techqa_K5 \
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
from grounding_hybrid.placebo import case_splits, donor_map  # noqa: E402


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
    suffix = "_placebo" + (f"_first{args.max_cases}" if args.max_cases else "")
    out_file = out_dir / f"features{suffix}.npz"
    if out_file.exists():
        print(f"[skip] {out_file} exists")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from grounding_hybrid.extractor import SharedExtractor

    ex = SharedExtractor(args.model, device=args.device)
    all_cases = load_cases(canon)
    by_id = {c.case_id: c for c in all_cases}
    split_of = case_splits(canon)
    n_windows = {c.case_id: len(ex.windows(c)) for c in all_cases}
    donors = donor_map(all_cases, n_windows, split_of)            # over ALL cases, so a smoke uses the same donors
    cases = all_cases[: args.max_cases] if args.max_cases else all_cases
    bad = [c for c, (d, fb) in donors.items() if by_id[d].source_id == by_id[c].source_id
           or split_of[d] != split_of[c] or (not fb and n_windows[d] < n_windows[c])]
    print(f"{tag}: {len(cases)} cases, {args.model} on {ex.device}; donors for {len(donors)} of {len(all_cases)} "
          f"cases ({sum(fb for _, fb in donors.values())} fallback, {len(bad)} breaking the rule)", flush=True)

    t0, rows, secs = time.time(), [], []
    for i, case in enumerate(cases):
        tc = time.time()
        d, fb = donors.get(case.case_id, (case.case_id, False))   # K = 1: no donor, Bp = B = Lookback
        for r in ex.extract_placebo(case, by_id[d]):
            rows.append(dict(case_id=case.case_id, donor_case_id=d if d != case.case_id else "", fallback=fb, **r))
        secs.append(time.time() - tc)
        if (i + 1) % 50 == 0 or i + 1 == len(cases):
            print(f"  {i + 1}/{len(cases)} cases, {(time.time() - t0) / (i + 1):.2f} s/case", flush=True)
    minutes = (time.time() - t0) / 60

    arrays = dict(
        case_id=np.array([r["case_id"] for r in rows]),
        sent_idx=np.array([r["sent_idx"] for r in rows], dtype=np.int32),
        n_tok=np.array([r["n_tok"] for r in rows], dtype=np.int32),
        logprob_full=np.array([r["logprob_full"] for r in rows], dtype=np.float32),
        lookback=np.stack([r["lookback"] for r in rows]).astype(np.float16),
        lookback_placebo_max=np.stack([r["lookback_placebo_max"] for r in rows]).astype(np.float16),
        n_windows=np.array([r["n_windows"] for r in rows], dtype=np.int32),
        donor_case_id=np.array([r["donor_case_id"] for r in rows]),
        fallback=np.array([r["fallback"] for r in rows], dtype=bool),
    )
    np.savez_compressed(out_file, **arrays)

    # alignment with GASP's sentence.csv: same rows and token counts, no NaN, window-1 log-probs as GASP's
    sent = load_sentences(canon)
    sent = sent[sent["case_id"].isin({c.case_id for c in cases})]
    key = {(r["case_id"], r["sent_idx"]): r for r in rows}
    hit = [key.get((c, s)) for c, s in zip(sent["case_id"], sent["sent_idx"])]
    covered = sum(h is not None for h in hit)
    ntok_ok = all(h is None or h["n_tok"] == n for h, n in zip(hit, sent["n_tok"]))
    nan_rows = sum(1 for r in rows if not (np.isfinite(r["logprob_full"]) and np.isfinite(r["lookback"]).all()
                                           and np.isfinite(r["lookback_placebo_max"]).all()))
    dlp = np.array([abs(h["logprob_full"] + m) for h, m in zip(hit, sent["mean_surprisal"]) if h is not None])
    check = dict(gasp_rows=len(sent), covered=covered, extra=len(rows) - covered, n_tok_match=ntok_ok,
                 nan_rows=nan_rows, logprob_absdiff_mean=float(dlp.mean()) if dlp.size else None)
    used = {c.case_id: donors[c.case_id] for c in cases if c.case_id in donors}
    donor_check = dict(cases_with_donor=len(used), fallback=sum(fb for _, fb in used.values()),
                       rule_breaks=len([c for c in bad if c in used]),
                       distinct_donors=len({d for d, _ in used.values()}))
    print("alignment vs GASP sentence.csv:", check, flush=True)
    print("donors:", donor_check, flush=True)
    meta = dict(tag=tag, model=args.model, device=ex.device, dtype="float32", attention="eager",
                max_ctx_tokens=ex.max_ctx, overlap=ex.overlap, n_cases=len(cases), n_sentences=len(rows),
                n_layers=ex.n_layers, n_heads=ex.n_heads, donor_seed=0,
                features={"lookback": "A_ctx/(A_ctx+A_new) per layer x head on window 1 (GASP's retained context)",
                          "lookback_placebo_max": "max over window 1 + the donor's windows 2..K (K = own windows)"},
                minutes=round(minutes, 2), seconds_median=float(np.median(secs)), alignment=check, donors=donor_check,
                donor_map={c: dict(donor=d, fallback=fb, k=n_windows[c], k_donor=n_windows[d]) for c, (d, fb) in used.items()},
                env=dict(python=platform.python_version(), torch=torch.__version__))
    json.dump(meta, open(out_dir / f"meta{suffix}.json", "w"), indent=1)
    print(f"saved {len(rows)} sentences to {out_file} in {minutes:.1f} min")
    if nan_rows or covered < len(sent) or not ntok_ok or donor_check["rule_breaks"]:
        sys.exit(f"alignment or donor check FAILED for {tag}: {check} {donor_check}")


if __name__ == "__main__":
    main()
