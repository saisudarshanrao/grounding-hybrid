"""Step E5 (CLAUDE.md, "E5 record" + clarification): score every GASP sentence with two trained, off-the-shelf
checkers, as released.

  MiniCheck-Flan-T5-Large (lytang/MiniCheck-Flan-T5-Large): P(supported) for (GASP's context as one document, the
      sentence text), with MiniCheck's own chunking and max over chunks. Hallucination score: 1 - P(supported).
  LettuceDetect-large (KRLabsOrg/lettucedect-large-modernbert-en-v1): token-level hallucination probabilities for
      (context passages, question, GASP's answer) with LettuceDetect's own passage grouping, chunking (4,096 tokens,
      its default), input preparation and max over chunks. Passages: GASP's context split on newlines (empty lines
      dropped). A sentence's score is the max probability over the answer tokens that start inside it (character
      offsets, GASP's token-to-sentence rule).

The checkers do not depend on the scorer, and both scorer runs of a dataset share cases and GASP sentence spans, so a
dataset is scored once (union of the runs' cases) and written to every run's feature folder, aligned with that run's
sentence.csv: <outroot>/<TAG>/checkers.npz (case_id, sent_idx, minicheck, lettuce) and meta_checkers.json (versions,
seconds, chunking and truncation counts).

Runs in its own environment (ModernBERT needs transformers >= 4.48; the frozen pipeline stays on 4.44.2).

    python scripts/score_checkers.py --canon_dirs results/gasp_repro/canon_results/<Qwen TAG> \
        results/gasp_repro/canon_results/<SmolLM2 TAG> --outroot results/features [--max_cases 20]
"""
import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MINICHECK = "flan-t5-large"                                    # lytang/MiniCheck-Flan-T5-Large
LETTUCE = "KRLabsOrg/lettucedect-large-modernbert-en-v1"


def load_cases(canon_dir):
    with open(Path(canon_dir) / "cases.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def passages(context):
    return [p for p in context.split("\n") if p.strip()] or [context]


class Lettuce:
    """LettuceDetect as released, with character offsets kept so tokens can be mapped to GASP's sentences."""

    def __init__(self, device=None):
        from lettucedetect.models.inference import HallucinationDetector
        self.det = HallucinationDetector(method="transformer", model_path=LETTUCE,
                                         **({"device": device} if device else {})).detector
        from lettucedetect.datasets.hallucination_dataset import HallucinationDataset
        from lettucedetect.detectors.prompt_utils import PromptUtils
        self.prep, self.fmt = HallucinationDataset.prepare_tokenized_input, PromptUtils.format_context

    @torch.no_grad()
    def token_probs(self, context, question, answer):
        """[(start, end, prob)] per answer token (answer coordinates, max over chunks), n_chunks, truncated."""
        d = self.det
        groups = d._group_passages_into_chunks(passages(context), question, answer)
        best, spans, truncated = None, None, False
        for g in groups:
            prompt = self.fmt(g, question, d.lang)
            encoding, _, offsets, start = self.prep(d.tokenizer, prompt, answer, d.max_length)
            n_prompt = len(d.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            truncated |= n_prompt > start - 2        # [CLS] prompt [SEP] answer [SEP]: the prompt lost tokens
            enc = {k: v.to(d.device) for k, v in encoding.items() if k in ("input_ids", "attention_mask")}
            p = torch.softmax(d.model(**enc).logits, dim=-1)[0, :, 1].float().cpu().numpy()
            probs, offs = p[start:], offsets[start:].tolist()
            if best is None:
                best, spans = probs.copy(), offs
            else:
                n = min(len(best), len(probs))                   # released code: max per token index
                best[:n] = np.maximum(best[:n], probs[:n])
        return [(s, e, float(pr)) for (s, e), pr in zip(spans, best) if e > s], len(groups), truncated


def sentence_max(tokens, sent_spans):
    out = []
    for s, e in sent_spans:
        v = [pr for ts, te, pr in tokens if s <= ts < e]
        out.append(max(v) if v else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_dirs", nargs="+", required=True, help="the scorer runs of ONE dataset")
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache_dir", default=None, help="MiniCheck checkpoint folder")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")

    cases = {}
    for cd in args.canon_dirs:
        for c in load_cases(cd):
            cases.setdefault(c["case_id"], c)
    ids = sorted(cases)[: args.max_cases or None]
    print(f"{len(ids)} cases from {len(args.canon_dirs)} runs, device {device}", flush=True)

    from minicheck.minicheck import MiniCheck
    mc = MiniCheck(model_name=MINICHECK, cache_dir=args.cache_dir)
    lt = Lettuce(device)

    scores, stats = {}, dict(minicheck_s=[], lettuce_s=[], lettuce_chunks=[], lettuce_truncated=0)
    for k, cid in enumerate(ids):
        c = cases[cid]
        spans = [tuple(x) for x in c["sent_spans"]]
        texts = [c["answer"][s:e].strip() for s, e in spans]
        t0 = time.time()
        _, sup, _, _ = mc.score(docs=[c["context"]] * len(texts), claims=texts)
        t1 = time.time()
        toks, n_chunks, trunc = lt.token_probs(c["context"], c["query"], c["answer"])
        t2 = time.time()
        lett = sentence_max(toks, spans)
        for j in range(len(spans)):
            scores[(cid, j)] = (1.0 - float(sup[j]), lett[j])
        stats["minicheck_s"].append(t1 - t0)
        stats["lettuce_s"].append(t2 - t1)
        stats["lettuce_chunks"].append(n_chunks)
        stats["lettuce_truncated"] += int(trunc)
        if (k + 1) % 25 == 0 or k + 1 == len(ids):
            print(f"  {k + 1}/{len(ids)} cases, MiniCheck {np.mean(stats['minicheck_s']):.2f} s, "
                  f"LettuceDetect {np.mean(stats['lettuce_s']):.2f} s per case", flush=True)

    meta_common = dict(minicheck=MINICHECK, lettuce=LETTUCE, device=device, cases=len(ids),
                       versions={p: version(p) for p in ("torch", "transformers", "lettucedetect", "minicheck")},
                       minicheck_s_median=float(np.median(stats["minicheck_s"])),
                       lettuce_s_median=float(np.median(stats["lettuce_s"])),
                       lettuce_multi_chunk_contexts=int(sum(n > 1 for n in stats["lettuce_chunks"])),
                       lettuce_max_chunks=int(max(stats["lettuce_chunks"])),
                       lettuce_truncated_contexts=int(stats["lettuce_truncated"]))
    for cd in args.canon_dirs:
        sent = pd.read_csv(Path(cd) / "sentence.csv", usecols=["case_id", "sent_idx"])
        sent = sent[sent["case_id"].isin(set(ids))]
        vals = np.array([scores.get((a, int(b)), (np.nan, np.nan)) for a, b in zip(sent["case_id"], sent["sent_idx"])])
        out = Path(args.outroot) / Path(cd).name
        out.mkdir(parents=True, exist_ok=True)
        suffix = f"_first{args.max_cases}" if args.max_cases else ""
        np.savez(out / f"checkers{suffix}.npz", case_id=np.array(sent["case_id"].tolist(), dtype=str),
                 sent_idx=sent["sent_idx"].values.astype(int),
                 minicheck=vals[:, 0], lettuce=vals[:, 1])
        meta = dict(meta_common, gasp_rows=int(len(sent)),
                    nan_minicheck=int(np.isnan(vals[:, 0]).sum()), nan_lettuce=int(np.isnan(vals[:, 1]).sum()))
        json.dump(meta, open(out / f"meta_checkers{suffix}.json", "w"), indent=1)
        print(f"saved {len(sent)} sentences to {out}/checkers{suffix}.npz: {meta}", flush=True)


if __name__ == "__main__":
    main()
