"""Appendix tables for the paper, as markdown, from saved results only (nothing is fit or scored here).

  A1  test sentence- and response-level AUC of every frozen detector (results/test/test_look.csv)
  A2  dev (grouped 5-fold CV inside dev, results/gating/rq2_where.csv) next to test, main detectors
  A3  controlled truncation per run, dev (results/gating/dev_coverage_trunc512.txt) next to test
  A4  compute cost, seconds per case on one T4 (results/gating/cost_table.csv)

Usage:
    python scripts/paper_tables.py        # -> results/paper_tables.md
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODELS = {"Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B", "SmolLM2-1.7B-Instruct": "SmolLM2-1.7B"}
DS = {"ragtruth": "RAGTruth", "tofueval": "TofuEval", "ragbench": "RAGBench", "techqa": "TechQA",
      "expertqalong": "ExpertQA-long"}
DETS = ["Perplexity+length", "GASP+base", "ReDeEP", "ReDeEP [cv]", "Frequency-aware [cv]", "Lookback [cv]",
        "B (ours) [cv]"]
DEV_NAME = {"Perplexity+length": "Perplexity+length", "GASP+base": "GASP+base (S1)", "ReDeEP [cv]": "ReDeEP [cv]",
            "Frequency-aware [cv]": "Frequency-aware [cv]", "Lookback [cv]": "Lookback (S3)",
            "B (ours) [cv]": "B: Lookback max over windows"}


def pair(a, b, fmt="{:.3f}", strip0=True):
    """'a / b'; AUCs lose the leading zero (.823), a missing value shows as a dash."""
    f = lambda v: "–" if pd.isna(v) else (fmt.format(v).replace("0.", ".", 1) if strip0 else fmt.format(v))  # noqa: E731
    return f"{f(a)} / {f(b)}"


def cell(det, a, b, fmt="{:.3f}"):
    """B has no own features where every context fits one window: it is Lookback there."""
    return "= Lookback" if det == "B (ours) [cv]" and pd.isna(a) and pd.isna(b) else pair(a, b, fmt)


def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def a1(test):
    out = []
    for level in ("span", "response"):
        t = test[(test.level == level) & (test.rows == "all")]
        p = t.pivot_table(index=["dataset", "model"], columns="detector", values="auc")
        rows = [[DS[ds]] + [cell(d, p[d].get((ds, "Qwen2.5-1.5B-Instruct")), p[d].get((ds, "SmolLM2-1.7B-Instruct")))
                            for d in DETS] for ds in DS]
        name = "sentence" if level == "span" else "response"
        out.append(f"**A1{'ab'[level == 'response']}. Test {name}-level AUC (Qwen / SmolLM2).** B equals Lookback "
                   f"where every context fits one window (RAGTruth, TofuEval).\n\n" + md(["", *DETS], rows))
    return "\n\n".join(out)


def a2(test):
    dev = pd.read_csv(RES / "gating" / "rq2_where.csv")
    dev = dev[dev.by == "all"]
    t = test[(test.level == "span") & (test.rows == "all")]
    dets = ["GASP+base", "ReDeEP [cv]", "Frequency-aware [cv]", "Lookback [cv]", "B (ours) [cv]"]
    rows = []
    for ds in DS:
        for model, short in MODELS.items():
            r = [DS[ds], short]
            for d in dets:
                dv = dev[(dev.run == f"{short}_{ds}") & (dev.detector == DEV_NAME[d])]["auc"]
                tv = t[(t.dataset == ds) & (t.model == model) & (t.detector == d)]["auc"]
                r.append(cell(d, dv.iloc[0] if len(dv) else float("nan"), tv.iloc[0] if len(tv) else float("nan")))
            rows.append(r)
    return ("**A2. Sentence-level AUC, dev (out-of-fold, grouped 5-fold CV inside dev) / test.**\n\n"
            + md(["Dataset", "Scorer", *dets], rows))


def a3():
    txt = (RES / "gating" / "dev_coverage_trunc512.txt").read_text()
    dev = {}
    for block in txt.split("\n# "):
        m = re.search(r"(\S+)_K5 \[chunked_ctx512\]", block)
        d = re.search(r"max - window1 = ([+-][\d.]+)\s+95% CI \[([+-][\d.]+), ([+-][\d.]+)\] (\S+) \| recovers "
                      r"(-?\d+)%", block)
        if m and d:
            dev[m.group(1)] = (f"{d.group(1)} [{d.group(2)}, {d.group(3)}]{' SIG' if d.group(4) == 'SIGNIFICANT' else ''}",
                               f"{d.group(5)}%")
    ctrl = json.load(open(RES / "test" / "test_look.json"))["controlled"]
    rows = []
    for r in ctrl:
        tag = f"{r['model']}_{r['dataset']}"
        c = r["diff"]
        test_d = f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]{' SIG' if c['sig'] else ''}"
        rec = "–" if r["recovered"] is None else f"{r['recovered']:.0%}"
        rows.append([DS[r["dataset"]], MODELS[r["model"]], *dev.get(tag, ("–", "–")), f"{r['sources']}", test_d, rec])
    return ("**A3. Controlled truncation (512-token windows), B − window 1 on truncated rows, dev / test.** "
            "\"recovers\" = share of the window-1 → full-view gap closed by B.\n\n"
            + md(["Dataset", "Scorer", "dev B − window 1", "dev recovers", "test sources", "test B − window 1",
                  "test recovers"], rows))


def a4():
    c = pd.read_csv(RES / "gating" / "cost_table.csv")
    p = c.pivot_table(index="method", columns=["dataset", "scorer"], values="s_per_case")
    rows = []
    for method in p.index:
        r = [method]
        for ds in DS:
            q = p.loc[method].get((ds, "Qwen2.5-1.5B-Instruct"), float("nan"))
            s = p.loc[method].get((ds, "SmolLM2-1.7B-Instruct"), float("nan"))
            r.append(pair(q, s, "{:.2f}", strip0=False))
        rows.append(r)
    return ("**A4. Seconds per case on one T4 (Qwen / SmolLM2).** GASP: whole-run wall time; feature runs: the "
            "extraction loop only.\n\n" + md(["Method", *DS.values()], rows))


def main():
    test = pd.read_csv(RES / "test" / "test_look.csv")
    text = "\n\n".join(["# Appendix tables (generated by scripts/paper_tables.py from saved results)",
                        a1(test), a2(test), a3(), a4()]) + "\n"
    (RES / "paper_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
