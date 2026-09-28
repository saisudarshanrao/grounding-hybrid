"""Step E5 (CLAUDE.md, "E5 record" + clarifications): score every GASP sentence with two trained, off-the-shelf
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
seconds, chunking and truncation counts, and the E5 checks (b)-(d) on every row written).

Checks on every row (exit code 1 after writing if any fails): (b) every GASP sentence (a row of the run's sentence.csv)
has a finite score from each checker; (c) every LettuceDetect answer token starts inside exactly one GASP sentence span
(spans are disjoint) and every GASP sentence gets >= 1 token (a sentence without tokens is the only source of a NaN
LettuceDetect score); (d) logged: contexts needing more than one LettuceDetect chunk, and contexts with a truncated
passage.

MiniCheck scores only the sentences GASP scores (it checks each sentence on its own, so no written row changes);
LettuceDetect reads the whole answer as always.

Shards: --shard i/n scores every n-th case (sorted case ids, offset i) and writes checkers.part<i>of<n>.npz; --merge n
joins the n parts into checkers.npz (every sentence.csv row exactly once; the parts must have been aligned to a
sentence.csv with the same sha256) without loading any model, so parts from several notebook versions can be merged
on the Mac.

Runs in its own environment (ModernBERT needs transformers >= 4.48; the frozen pipeline stays on 4.44.2).

    python scripts/score_checkers.py --canon_dirs results/gasp_repro/canon_results/<Qwen TAG> \
        results/gasp_repro/canon_results/<SmolLM2 TAG> --outroot results/features [--max_cases 20] [--shard 0/2]
    python scripts/score_checkers.py --canon_dirs <same> --outroot results/features --merge 2
"""
import argparse
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

MINICHECK = "flan-t5-large"                                    # lytang/MiniCheck-Flan-T5-Large
LETTUCE = "KRLabsOrg/lettucedect-large-modernbert-en-v1"
CHECKS = ("nan_minicheck", "nan_lettuce", "sentences_without_tokens", "tokens_not_in_one_span")


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

    def token_probs(self, context, question, answer):
        """[(start, end, prob)] per answer token (answer coordinates, max over chunks), n_chunks, truncated."""
        import torch
        d = self.det
        groups = d._group_passages_into_chunks(passages(context), question, answer)
        best, spans, truncated = None, None, False
        for g in groups:
            prompt = self.fmt(g, question, d.lang)
            encoding, _, offsets, start = self.prep(d.tokenizer, prompt, answer, d.max_length)
            n_prompt = len(d.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            truncated |= n_prompt > start - 2        # [CLS] prompt [SEP] answer [SEP]: the prompt lost tokens
            enc = {k: v.to(d.device) for k, v in encoding.items() if k in ("input_ids", "attention_mask")}
            with torch.no_grad():
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


def names(max_cases, i=0, n=1):
    suffix = f"_first{max_cases}" if max_cases else ""
    part = f".part{i}of{n}" if n > 1 else ""
    return f"checkers{suffix}{part}.npz", f"meta_checkers{suffix}{part}.json"


def summarize(pc):
    """Cost and check (d) numbers from the per-case lists (identical for one run and for merged parts)."""
    return dict(cases=len(pc["case_id"]),
                minicheck_s_median=float(np.median(pc["minicheck_s"])),
                lettuce_s_median=float(np.median(pc["lettuce_s"])),
                minicheck_s_total=float(np.sum(pc["minicheck_s"])), lettuce_s_total=float(np.sum(pc["lettuce_s"])),
                lettuce_multi_chunk_contexts=int(sum(n > 1 for n in pc["lettuce_chunks"])),
                lettuce_max_chunks=int(max(pc["lettuce_chunks"])),
                lettuce_truncated_contexts=int(sum(pc["lettuce_truncated"])),
                tokens_not_in_one_span=int(sum(pc["tokens_not_in_one_span"])))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_run(out, npz_name, meta_name, sent, mc, lt, meta):
    """Save one run's rows (sentence.csv order) and its meta with the row checks (b), (c); True if they pass."""
    meta = dict(meta, gasp_rows=int(len(sent)), nan_minicheck=int(np.isnan(mc).sum()),
                nan_lettuce=int(np.isnan(lt).sum()))
    meta["sentences_without_tokens"] = meta["nan_lettuce"]
    meta["checks_pass"] = all(meta[k] == 0 for k in CHECKS)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / npz_name, case_id=np.array(sent["case_id"].tolist(), dtype=str),
             sent_idx=sent["sent_idx"].values.astype(int), minicheck=mc, lettuce=lt)
    json.dump(meta, open(out / meta_name, "w"), indent=1)
    shown = {k: v for k, v in meta.items() if k != "per_case"}
    print(f"saved {len(sent)} sentences to {out}/{npz_name}: {shown}", flush=True)
    return meta["checks_pass"]


def merge(args):
    ok = True
    for cd in args.canon_dirs:
        out = Path(args.outroot) / Path(cd).name
        parts = [names(args.max_cases, i, args.merge) for i in range(args.merge)]
        zs = [np.load(out / f) for f, _ in parts]
        metas = [json.load(open(out / m)) for _, m in parts]
        assert [m["shard"] for m in metas] == [f"{i}/{args.merge}" for i in range(args.merge)], "missing shard"
        here = sha256(Path(cd) / "sentence.csv")
        assert all(m["sentence_csv_sha256"] == here for m in metas), f"parts were aligned to another sentence.csv: {cd}"
        pc = {k: sum((m["per_case"][k] for m in metas), []) for k in metas[0]["per_case"]}
        ids = pc["case_id"]
        assert len(ids) == len(set(ids)) == metas[0]["cases_all_shards"], "shards overlap or miss cases"
        df = pd.concat([pd.DataFrame({k: z[k] for k in ("case_id", "sent_idx", "minicheck", "lettuce")}) for z in zs])
        df["case_id"] = df["case_id"].astype(str)
        sent = pd.read_csv(Path(cd) / "sentence.csv", usecols=["case_id", "sent_idx"])
        sent = sent[sent["case_id"].isin(set(ids))].reset_index(drop=True)
        assert not df.duplicated(["case_id", "sent_idx"]).any() and len(df) == len(sent), "rows differ from GASP's"
        j = sent.merge(df, on=["case_id", "sent_idx"], how="left", validate="one_to_one", indicator=True)
        assert (j["_merge"] == "both").all(), "a GASP row has no checker row"
        meta = {k: v for k, v in metas[0].items() if k not in ("shard", "per_case", "gasp_rows", "cases_all_shards",
                                                                "sentence_csv_sha256")
                and k not in CHECKS and k not in summarize(pc) and k != "checks_pass"}
        meta.update(summarize(pc), merged_from=[m for _, m in parts], per_case=pc, sentence_csv_sha256=here,
                    part_devices=[m.get("device") for m in metas], part_versions=[m.get("versions") for m in metas])
        npz_name, meta_name = names(args.max_cases)
        ok &= write_run(out, npz_name, meta_name, sent, j["minicheck"].values.astype(float),
                        j["lettuce"].values.astype(float), meta)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_dirs", nargs="+", required=True, help="the scorer runs of ONE dataset")
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: score every n-th case, starting at i")
    ap.add_argument("--merge", type=int, default=0, help="join n shard files into checkers.npz (no scoring)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache_dir", default=None, help="MiniCheck checkpoint folder")
    args = ap.parse_args()
    if args.merge:
        sys.exit(0 if merge(args) else 1)

    import torch
    i, n = (int(x) for x in args.shard.split("/"))
    assert 0 <= i < n
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    cases, scored = {}, set()
    for cd in args.canon_dirs:
        for c in load_cases(cd):
            cases.setdefault(c["case_id"], c)
        s = pd.read_csv(Path(cd) / "sentence.csv", usecols=["case_id", "sent_idx"])
        scored |= set(zip(s["case_id"], s["sent_idx"].astype(int)))
    all_ids = sorted(cases)[: args.max_cases or None]
    ids = all_ids[i::n]
    print(f"{len(ids)} cases (shard {i}/{n} of {len(all_ids)}) from {len(args.canon_dirs)} runs, device {device}",
          flush=True)

    from minicheck.minicheck import MiniCheck
    mc = MiniCheck(model_name=MINICHECK, cache_dir=args.cache_dir)
    lt = Lettuce(device)

    scores = {}
    pc = dict(case_id=[], minicheck_s=[], lettuce_s=[], lettuce_chunks=[], lettuce_truncated=[],
              tokens_not_in_one_span=[])
    for k, cid in enumerate(ids):
        c = cases[cid]
        spans = [tuple(x) for x in c["sent_spans"]]
        texts = [c["answer"][s:e].strip() for s, e in spans]
        keep = [j for j in range(len(spans)) if (cid, j) in scored]   # MiniCheck checks each sentence on its own:
        t0 = time.time()                                               # skipping spans GASP never scores changes
        sup = mc.score(docs=[c["context"]] * len(keep), claims=[texts[j] for j in keep])[1] if keep else []
        mcs = dict(zip(keep, (1.0 - float(p) for p in sup)))          # no written row
        t1 = time.time()
        toks, n_chunks, trunc = lt.token_probs(c["context"], c["query"], c["answer"])
        t2 = time.time()
        lett = sentence_max(toks, spans)
        for j in range(len(spans)):
            scores[(cid, j)] = (mcs.get(j, float("nan")), lett[j])
        pc["case_id"].append(cid)
        pc["minicheck_s"].append(t1 - t0)
        pc["lettuce_s"].append(t2 - t1)
        pc["lettuce_chunks"].append(int(n_chunks))
        pc["lettuce_truncated"].append(int(trunc))
        pc["tokens_not_in_one_span"].append(sum(1 for ts, _, _ in toks if sum(s <= ts < e for s, e in spans) != 1))
        if (k + 1) % 25 == 0 or k + 1 == len(ids):
            print(f"  {k + 1}/{len(ids)} cases, MiniCheck {np.mean(pc['minicheck_s']):.2f} s, "
                  f"LettuceDetect {np.mean(pc['lettuce_s']):.2f} s per case", flush=True)

    meta = dict(minicheck=MINICHECK, lettuce=LETTUCE, device=device, shard=f"{i}/{n}", cases_all_shards=len(all_ids),
                versions={p: version(p) for p in ("torch", "transformers", "lettucedetect", "minicheck")},
                **summarize(pc), per_case=pc)
    npz_name, meta_name = names(args.max_cases, i, n)
    ok = True
    for cd in args.canon_dirs:
        sent = pd.read_csv(Path(cd) / "sentence.csv", usecols=["case_id", "sent_idx"])
        sent = sent[sent["case_id"].isin(set(ids))].reset_index(drop=True)
        vals = np.array([scores.get((a, int(b)), (np.nan, np.nan)) for a, b in zip(sent["case_id"], sent["sent_idx"])],
                        float).reshape(-1, 2)
        ok &= write_run(Path(args.outroot) / Path(cd).name, npz_name, meta_name, sent, vals[:, 0], vals[:, 1],
                        dict(meta, sentence_csv_sha256=sha256(Path(cd) / "sentence.csv")))
    if not ok:
        print("E5 ROW CHECK FAILED (see nan_* / sentences_without_tokens / tokens_not_in_one_span above)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
