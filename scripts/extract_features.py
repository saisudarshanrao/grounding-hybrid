"""Extract per-sentence features for one GASP run with the shared single-pass extractor.

Reads GASP's canonical cases (cases.jsonl) from a finished GASP run and writes compact
per-sentence features keyed by (case_id, sent_idx), the key of GASP's sentence.csv. Use the
same model as that GASP run: which sentences get features depends on the tokenizer.
Resumable: an existing output file is skipped.

Usage:
    python scripts/extract_features.py \
        --canon_dir results/gasp_repro/canon_results/Qwen2.5-1.5B-Instruct_ragtruth_K5 \
        --model Qwen/Qwen2.5-1.5B-Instruct
    python scripts/extract_features.py ... --max_cases 3     # quick Mac test

Output: <outroot>/<TAG>/features.npz (+ meta.json), where TAG is the GASP run's folder name;
features_chunked[_ctxN].npz with --chunked [--max_ctx_tokens N]; features_redeep.npz with --redeep
(adds ReDeEP's ecs [layers, heads] and pks [layers]; lookback is recomputed and must equal
features.npz, a second alignment check). With a smaller window than GASP's
(controlled truncation) the log-prob alignment check is expected to differ; coverage, token counts
and NaN checks still gate the run.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grounding_hybrid.gasp_bridge import load_cases, load_sentences  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_dir", required=True, help="a GASP run folder with cases.jsonl + sentence.csv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--outroot", default=str(ROOT / "results" / "features"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "reproduce.yaml"))
    ap.add_argument("--max_cases", type=int, default=0, help="cap cases (0 = all); for quick tests")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32", help="float16 makes Qwen2.5 eager attention overflow")
    ap.add_argument("--chunked", action="store_true", help="coverage-aware reading of the whole context")
    ap.add_argument("--max_ctx_tokens", type=int, default=0, help="override GASP's window (controlled truncation)")
    ap.add_argument("--overlap", type=int, default=256, help="token overlap between windows (chunked)")
    ap.add_argument("--redeep", action="store_true", help="also extract ReDeEP's ECS and PKS scores")
    args = ap.parse_args()

    canon = Path(args.canon_dir)
    tag = canon.name
    if not tag.startswith(args.model.split("/")[-1] + "_"):
        sys.exit(f"{tag} was not scored by {args.model}: sentence rows would not align")
    out_dir = Path(args.outroot) / tag
    suffix = (("_chunked" if args.chunked else "") + ("_redeep" if args.redeep else "")
              + (f"_ctx{args.max_ctx_tokens}" if args.max_ctx_tokens else "")
              + (f"_first{args.max_cases}" if args.max_cases else ""))
    out_file = out_dir / f"features{suffix}.npz"
    if out_file.exists():
        print(f"[skip] {out_file} exists")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from grounding_hybrid.extractor import SharedExtractor

    p = yaml.safe_load(open(args.config))["params"]
    cases = load_cases(canon)
    if args.max_cases:
        cases = cases[: args.max_cases]
    ex = SharedExtractor(args.model, device=args.device, dtype=args.dtype, overlap=args.overlap, redeep=args.redeep,
                         max_ctx_tokens=args.max_ctx_tokens or p["max_ctx_tokens"], max_ans_tokens=p["max_ans_tokens"])
    print(f"{tag}: {len(cases)} cases, {args.model} on {ex.device}, "
          f"{ex.n_layers} layers x {ex.n_heads} heads, {ex.dtype}", flush=True)

    t0 = time.time()
    rows = []
    for i, case in enumerate(cases):
        for r in ex.extract(case, chunked=args.chunked):
            rows.append(dict(case_id=case.case_id, **r))
        if (i + 1) % 50 == 0 or i + 1 == len(cases):
            print(f"  {i + 1}/{len(cases)} cases, {(time.time() - t0) / (i + 1):.2f} s/case", flush=True)
    minutes = (time.time() - t0) / 60

    arrays = dict(
        case_id=np.array([r["case_id"] for r in rows]),
        sent_idx=np.array([r["sent_idx"] for r in rows], dtype=np.int32),
        n_tok=np.array([r["n_tok"] for r in rows], dtype=np.int32),
        logprob_full=np.array([r["logprob_full"] for r in rows], dtype=np.float32),
        lookback=np.stack([r["lookback"] for r in rows]).astype(np.float16),
    )
    if args.chunked:
        arrays.update(lookback_max=np.stack([r["lookback_max"] for r in rows]).astype(np.float16),
                      lookback_mean=np.stack([r["lookback_mean"] for r in rows]).astype(np.float16),
                      n_windows=np.array([r["n_windows"] for r in rows], dtype=np.int32))
    if args.redeep:
        arrays.update(ecs=np.stack([r["ecs"] for r in rows]).astype(np.float16),
                      pks=np.stack([r["pks"] for r in rows]).astype(np.float32))
    np.savez_compressed(out_file, **arrays)

    # alignment check against GASP's own sentence.csv (same key, same tokens)
    sent = load_sentences(canon)
    sent = sent[sent["case_id"].isin({c.case_id for c in cases})]
    key = {(r["case_id"], r["sent_idx"]): r for r in rows}
    hit = [key.get((c, s)) for c, s in zip(sent["case_id"], sent["sent_idx"])]
    covered = sum(h is not None for h in hit)
    ntok_ok = all(h is None or h["n_tok"] == n for h, n in zip(hit, sent["n_tok"]))
    diff = np.array([abs(h["logprob_full"] + m) for h, m in zip(hit, sent["mean_surprisal"]) if h is not None])
    lb_keys = ["lookback"] + (["lookback_max", "lookback_mean"] if args.chunked else []) + (["ecs", "pks"] if args.redeep else [])
    nan_rows = sum(1 for r in rows if not np.isfinite(r["logprob_full"])
                   or not all(np.isfinite(r[k]).all() for k in lb_keys))
    check = dict(gasp_rows=len(sent), covered=covered, extra=len(rows) - covered, n_tok_match=ntok_ok,
                 nan_rows=nan_rows,
                 logprob_absdiff_mean=float(diff.mean()) if diff.size else None,
                 logprob_absdiff_max=float(diff.max()) if diff.size else None)
    ref = ROOT / "results" / "features" / tag / "features.npz"   # --redeep: lookback must equal Week 2's
    if args.redeep and not args.max_ctx_tokens and ref.exists():
        z = np.load(ref)
        idx = {(c, s): i for i, (c, s) in enumerate(zip(z["case_id"], z["sent_idx"]))}
        pairs = [(i, idx[(r["case_id"], r["sent_idx"])]) for i, r in enumerate(rows) if (r["case_id"], r["sent_idx"]) in idx]
        if pairs:
            a, b = zip(*pairs)
            check["lookback_vs_features_npz_absdiff_max"] = float(np.abs(
                arrays["lookback"][list(a)].astype(np.float32) - z["lookback"][list(b)].astype(np.float32)).max())
    print("alignment vs GASP sentence.csv:", check, flush=True)

    meta = dict(tag=tag, model=args.model, device=ex.device, dtype=ex.dtype, chunked=args.chunked,
                max_ctx_tokens=ex.max_ctx, overlap=ex.overlap, n_cases=len(cases),
                n_sentences=len(rows), n_layers=ex.n_layers, n_heads=ex.n_heads,
                features={"lookback": "A_ctx/(A_ctx+A_new) per layer x head, mean over sentence tokens",
                          "logprob_full": "mean full-context token log-prob (= -GASP mean_surprisal)",
                          **({"ecs": "ReDeEP external context score per layer x head (top-10% context tokens)",
                              "pks": "ReDeEP parametric knowledge score per layer (JSD before/after FFN)"}
                             if args.redeep else {})},
                params=p, minutes=round(minutes, 2), alignment=check,
                env=dict(python=platform.python_version(), torch=torch.__version__))
    json.dump(meta, open(out_dir / f"meta{suffix}.json", "w"), indent=1)
    print(f"saved {len(rows)} sentences to {out_file} in {minutes:.1f} min")
    if nan_rows or covered < len(sent) or not ntok_ok:
        sys.exit(f"alignment FAILED for {tag}: {check}")


if __name__ == "__main__":
    main()
