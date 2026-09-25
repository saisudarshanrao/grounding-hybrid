"""Step 9: provenance of every saved result file (which Kaggle version, commit, and whether it lines up).

For each feature file under results/features/ this records the Kaggle version that made it, the
commit that version ran, the extractor settings from its meta json, the alignment check the
extractor wrote (every GASP sentence covered, same token counts, no NaN, full-context log-prob
within ~0.01 of GASP's), and two cross-file checks made here:
  rows  - same (case_id, sent_idx) rows, in the same order, as GASP's sentence.csv for that run
  lb=   - max |lookback - lookback of the base file| (features.npz; for the long sets, the
          chunked_redeep file): 0 means window 1 is exactly the same reading as the base file
Controlled-truncation files (ctx512) read a shorter window on purpose, so lb= is not expected to be 0.

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
}
LONG = ("techqa", "expertqalong")


def version_of(ds, name):
    if name == "canon":
        return {"techqa": 6, "expertqalong": 7}.get(ds, 1)
    if name == "features_freq.npz":
        return 10 if ds in LONG else 9
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
        for f in sorted(run.glob("features*.npz")):
            if f.name == "features_first5.npz":  # Mac check on 5 cases, not used in any result
                continue
            z = np.load(f)
            meta = json.loads((run / f.name.replace("features", "meta").replace(".npz", ".json")).read_text())
            a = meta["alignment"]
            same = (len(z["case_id"]) == len(sent) and (z["case_id"] == sent["case_id"].values).all()
                    and (z["sent_idx"] == sent["sent_idx"].values).all())
            lb = "" if base is None or f.name == base_name else f"{np.abs(z['lookback'] - base).max():.1e}"
            ok = a["covered"] == a["gasp_rows"] and a["extra"] == 0 and a["n_tok_match"] and a.get("nan_rows", 0) == 0
            win = f"{meta.get('max_ctx_tokens', 1800)}/{meta.get('overlap', '-')}" if meta.get("chunked") else "1800 (GASP view)"
            rows.append({"run": run.name, "file": f.name, "version": version_of(ds, f.name), "windows": win,
                         "rows": f"{len(z['case_id'])} {'same' if same else 'DIFFERENT'}", "lb": lb,
                         "align": f"{'ok' if ok else 'FAIL'}, dlogp {a['logprob_absdiff_mean']:.4f}",
                         "sha": sha(f)})
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
              "byte-identical to Versions 6 and 7 (GASP scoring is deterministic).", "",
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
           or (r["lb"] and "ctx512" not in r["file"] and float(r["lb"]) > 1e-4)]
    print(f"\n{len(rows)} entries, {len(bad)} problems" + ("".join(f"\n  {r['run']} {r['file']}" for r in bad)))


if __name__ == "__main__":
    main()
