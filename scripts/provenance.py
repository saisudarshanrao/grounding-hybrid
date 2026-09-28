"""Step 9: provenance of every saved result file (which Kaggle version, commit, and whether it lines up).

For each feature file under results/features/ this records the Kaggle version that made it, the
commit that version ran, the extractor settings from its meta json, the alignment check the
extractor wrote (every GASP sentence covered, same token counts, no NaN, full-context log-prob
within ~0.01 of GASP's), and two cross-file checks made here:
  rows  - same (case_id, sent_idx) rows, in the same order, as GASP's sentence.csv for that run
  lb=   - max |lookback - lookback of the base file| (features.npz; for the long sets, the
          chunked_redeep file): 0 means window 1 is exactly the same reading as the base file
Controlled-truncation files (ctx512) read a shorter window on purpose, so lb= is not expected to be 0.
The placebo file (features_placebo.npz, step E2) reads window 1 exactly as the base file, so its lb= must be 0.
The displaced file (features_displaced.npz, step E3) covers only the responses whose context fits one window (rows =
GASP's rows of those responses, in order) and reads distractor text in window 1 on purpose, so it has no lb=.
The long-pass file (features_long.npz, step E1) reads more context on purpose; its lb= is taken only on
responses whose whole context fits GASP's 1800 tokens (the same text as window 1, read with a different
attention kernel), where it must stay within the E1 check's 1e-3.
The checker file (checkers.npz, step E5) holds MiniCheck and LettuceDetect scores (as released) for GASP's rows; it has
no Lookback reading, so no lb=; align = no NaN score.

Usage:
    python scripts/provenance.py        # -> results/PROVENANCE.md
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "results" / "features"
CANON = ROOT / "results" / "gasp_repro" / "canon_results"
OUT = ROOT / "results" / "PROVENANCE.md"

# Kaggle notebook sudarshan1234/notebook83d38bc56a; logs at .../log?scriptVersionId=<id>
VERSIONS = {
    1: ("352192837", "0b67ac5", "GASP reproduction (MODE full)"),
    2: ("352219108", "1636251", "Lookback features (MODE features)"),
    3: ("352235698", "38e0f7c", "RAGBench 1800-token windows (MODE chunked)"),
    4: ("352321617", "f2f0eb0", "controlled truncation 512/128 (MODE trunc)"),
    5: ("352375851", "c9a5fdb", "ReDeEP ECS/PKS (MODE redeep)"),
    6: ("352382308", "8b11aaa", "TechQA GASP + windows + ReDeEP (MODE techqa)"),
    7: ("352456307", "cba5da1", "ExpertQA-long GASP + windows + ReDeEP (MODE expertqa)"),
    9: ("352495157", "64ab33e + runtime patch (= 88668d9)", "frequency-aware attention (MODE freq)"),
    10: ("352565636", "47b04f9", "frequency-aware attention, long sets (MODE longfreq)"),
    11: ("352758371", "561950f", "one long pass (baseline L, step E1), long sets (MODE longpass)"),
    12: ("352812136", "c93d4e0", "placebo reading (step E2), TechQA (MODE placebo)"),
    13: ("352890215", "32c5c30", "evidence displacement (step E3), RAGTruth + RAGBench (MODE displace)"),
    14: ("353556420", "a8a2c14", "TRIVIA+-long (step E4): GASP + windows + ReDeEP + one long pass (MODE triviaplus)"),
    15: ("353596833", "2b7e1d1", "checkers part a (step E5; MODE checkers-a): saved TechQA 1/4, ExpertQA-long 0/4, RAGTruth, "
                                 "TofuEval; 2 shards out of GPU memory, so the version ends 'failed'"),
    16: ("353598265", "2b7e1d1", "checkers part b (step E5; MODE checkers-b): saved TechQA 2/4, ExpertQA-long 2/4, TRIVIA+-long; "
                                 "3 jobs out of GPU memory, so the version ends 'failed'"),
    17: ("353651141", "0bb8455", "checkers rerun C (step E5, out-of-memory fallback): TechQA 0/8 + 4/8, ExpertQA-long 1/8 + 5/8"),
    18: ("353653036", "0bb8455", "checkers rerun D (step E5, out-of-memory fallback): TechQA 3/8 + 7/8, ExpertQA-long 3/8 + "
                                 "7/8, RAGBench 0/2 + 1/2"),
}
LONG = ("techqa", "expertqalong", "triviapluslong")
LONGPASS_VERSION = 11
PLACEBO_VERSION = 12
DISPLACE_VERSION = 13
TRIVIAPLUS_VERSION = 14
CHECKERS_VERSION = {"ragtruth": "15", "tofueval": "15", "triviapluslong": "16", "ragbench": "18",   # step E5: parts
                    "techqa": "15+16+17+18", "expertqalong": "15+16+17+18"}                     # merged on the Mac


def version_of(ds, name):
    if ds == "triviapluslong" and name != "checkers.npz":
        return TRIVIAPLUS_VERSION
    if name == "checkers.npz":
        return CHECKERS_VERSION[ds]
    if name == "canon":
        return {"techqa": 6, "expertqalong": 7}.get(ds, 1)
    if name == "features_freq.npz":
        return 10 if ds in LONG else 9
    if name == "features_long.npz":
        return LONGPASS_VERSION
    if name == "features_placebo.npz":
        return PLACEBO_VERSION
    if name == "features_displaced.npz":
        return DISPLACE_VERSION
    return {"features.npz": 2, "features_chunked.npz": 3, "features_chunked_ctx512.npz": 4,
            "features_redeep.npz": 5}.get(name, {"techqa": 6, "expertqalong": 7}.get(ds))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main():
    rows = []
    for run in sorted(FEAT.glob("*_K5")):
        ds = run.name.rsplit("_", 2)[1]
        sent = pd.read_csv(CANON / run.name / "sentence.csv", usecols=["case_id", "sent_idx"])
        base_name = "features_chunked_redeep.npz" if ds in LONG else "features.npz"
        base = np.load(run / base_name)["lookback"] if (run / base_name).exists() else None
        v = version_of(ds, "canon")
        rows.append({"run": run.name, "file": "canon_results/ (GASP scores)", "version": v,
                     "rows": f"{len(sent)} (reference)", "lb": "", "align": "", "sha": sha(CANON / run.name / "sentence.csv")})
        if (run / "checkers.npz").exists():            # step E5: checker scores, aligned with GASP's rows
            z = np.load(run / "checkers.npz")
            meta = json.loads((run / "meta_checkers.json").read_text())
            same = (len(z["case_id"]) == len(sent) and (z["case_id"] == sent["case_id"].values).all()
                    and (z["sent_idx"] == sent["sent_idx"].values).all())
            nan = int(np.isnan(z["minicheck"]).sum() + np.isnan(z["lettuce"]).sum())
            rows.append({"run": run.name, "file": "checkers.npz", "version": CHECKERS_VERSION[ds],
                         "windows": f"MiniCheck + LettuceDetect as released ({meta.get('lettuce_multi_chunk_contexts')} "
                                    f"multi-chunk contexts)",
                         "rows": f"{len(z['case_id'])} {'same' if same else 'DIFFERENT'}", "lb": "",
                         "align": f"{'ok' if nan == 0 else 'FAIL'}, {nan} NaN", "sha": sha(run / "checkers.npz")})
        for f in sorted(run.glob("features*.npz")):
            if f.name == "features_first5.npz":  # Mac check on 5 cases, not used in any result
                continue
            z = np.load(f)
            meta = json.loads((run / f.name.replace("features", "meta").replace(".npz", ".json")).read_text())
            a = meta["alignment"]
            disp = "displaced" in f.name       # step E3: only the responses whose context fits one window
            ref = sent[sent["case_id"].isin(set(z["case_id"]))] if disp else sent
            same = (len(z["case_id"]) == len(ref) and (z["case_id"] == ref["case_id"].values).all()
                    and (z["sent_idx"] == ref["sent_idx"].values).all())
            long = f.name.startswith("features_long")
            fit = z["n_ctx_tokens"] <= 1800 if long else np.ones(len(z["case_id"]), bool)
            lb = ("" if base is None or f.name == base_name or disp or not (same and fit.any()) else
                  f"{np.abs(z['lookback'][fit].astype(np.float32) - base[fit].astype(np.float32)).max():.1e}")
            ok = a["covered"] == a["gasp_rows"] and a["extra"] == 0 and a["n_tok_match"] and a.get("nan_rows", 0) == 0
            win = ("one long pass (up to %d positions)" % meta["max_positions"] if long else
                   f"{meta['max_ctx_tokens']}/{meta['overlap']}, windows 2..K from a donor" if "placebo" in f.name else
                   f"{meta['max_ctx_tokens']}/{meta['overlap']}, context behind {meta['prefix_tokens']} distractor tokens"
                   if disp else
                   f"{meta.get('max_ctx_tokens', 1800)}/{meta.get('overlap', '-')}" if meta.get("chunked") else "1800 (GASP view)")
            dlogp = a.get("logprob_absdiff_mean_fits_1800" if long else "logprob_absdiff_mean")
            dlogp = ("n/a (displaced context)" if disp else "n/a (no context fits)" if dlogp is None
                     else f"{dlogp:.4f}" + (" (fits 1800)" if long else ""))
            rows.append({"run": run.name, "file": f.name, "version": version_of(ds, f.name), "windows": win,
                         "rows": f"{len(z['case_id'])} {'same' if same else 'DIFFERENT'}"
                                 + (f" ({len(set(z['case_id']))} responses that fit)" if disp else ""),
                         "lb": lb,
                         "align": f"{'ok' if ok else 'FAIL'}, dlogp {dlogp}", "sha": sha(f)})
    lines = ["# Provenance of saved results (Step 9)", "",
             "Made by `scripts/provenance.py`. Every number in the paper comes from these files; none were re-extracted.",
             "Kaggle notebook `sudarshan1234/notebook83d38bc56a`; a version's log is at",
             "`https://www.kaggle.com/code/sudarshan1234/notebook83d38bc56a/log?scriptVersionId=<id>`.", "",
             "## Kaggle versions", "", "| Version | scriptVersionId | commit | what |", "|---|---|---|---|"]
    lines += [f"| {k} | {i} | {c} | {w} |" for k, (i, c, w) in VERSIONS.items()]
    lines += ["", "Version 8 was an accidental save, cancelled; it produced nothing used here.",
              "Version 9's runtime patch only copies the answer part of the attention matrix instead of keeping a view",
              "(memory); it does not change any value. The same change was pushed as 88668d9.",
              "Version 10 reran GASP on TechQA and ExpertQA-long: sentence.csv, response.csv and cases.jsonl are",
              "byte-identical to Versions 6 and 7 (GASP scoring is deterministic).",
              "Step E5's checker scores come in parts (shards of cases) from Versions 15-18, joined on the Mac by",
              "score_checkers.py --merge auto, which requires every case and every sentence.csv row exactly once and that",
              "every part was aligned to a sentence.csv with the local file's sha256; Versions 15 and 16 end 'failed' only",
              "because some of their jobs ran out of GPU memory (rerun in 17 and 18); their saved parts are complete.", "",
              "## Files", "",
              "rows = same (case_id, sent_idx) rows in the same order as GASP's sentence.csv; "
              "lb = max |lookback - base lookback| (0 = window 1 reads exactly what the base file read; "
              "ctx512 reads 512-token windows on purpose); align = extractor's own check "
              "(all GASP sentences covered, same token counts, no NaN) and mean |log-prob - GASP's|.", "",
              "| run | file | version | windows | rows | lb | align | sha256[:12] |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['run']} | {r['file']} | {r['version']} | {r.get('windows', '')} | {r['rows']} | {r['lb']} | "
              f"{r['align']} | {r['sha']} |" for r in rows]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    bad = [r for r in rows if "DIFFERENT" in r["rows"] or "FAIL" in r["align"]
           or (r["lb"] and "ctx512" not in r["file"]
               and float(r["lb"]) > (1e-3 if r["file"].startswith("features_long") else 1e-4))]
    print(f"\n{len(rows)} entries, {len(bad)} problems" + ("".join(f"\n  {r['run']} {r['file']}" for r in bad)))


if __name__ == "__main__":
    main()
