"""Appendix tables for the paper, as markdown, from saved results only (nothing is fit or scored here).

  A1  test sentence- and response-level AUC of every frozen detector (results/test/test_look.csv)
  A2  dev (grouped 5-fold CV inside dev, results/gating/rq2_where.csv) next to test, main detectors
  A3  controlled truncation per run, dev (results/gating/dev_coverage_trunc512.txt) next to test
  A4  compute cost, seconds per case on one T4 (results/gating/cost_table.csv)
  A5  step E1, B vs one long pass (L): dev (results/e1/e1_dev.json) next to test (results/test/e1_test_look_P4.json),
      SmolLM2's cut rows, and L's cost (meta_long.json)
  A6  step E2, B vs its placebo on TechQA: dev (results/e2/e2_dev.json) next to test (e2_test_look_P5.json)
  A7  step E3, evidence displaced out of window 1: dev (results/e3/e3_dev.json) next to test (e3_test_look_P6.json)

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


def a5():
    """Step E1: B vs one long pass (L) on truncated rows, dev (out-of-fold) next to test (P4), plus L's cost."""
    f_dev, f_test = RES / "e1" / "e1_dev.json", RES / "test" / "e1_test_look_P4.json"
    if not (f_dev.exists() and f_test.exists()):
        return None
    ev = {"dev": json.load(open(f_dev)), "test": json.load(open(f_test))}
    ci = lambda r: f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]" + ("*" if r["sig"] else "")  # noqa: E731
    a = lambda v: f"{v:.3f}".replace("0.", ".", 1)  # noqa: E731

    def row(e, ds, m, rows):
        return next(x for x in e["table"] if x["dataset"] == ds and x["model"] == m.split("-")[0] and x["rows"] == rows)

    pooled = []
    for name, key in (("B − L (P4, the pre-written rule)", None),
                      ("B − L without SmolLM2's cut rows", "B - L (truncated rows, SmolLM2 cut rows removed)"),
                      ("L − window 1", "L - window 1 (truncated rows)"),
                      ("B − window 1", "B - window 1 (truncated rows)")):
        pooled.append([name] + [ci(e["primary"]["D"] if key is None else e["secondary"][key]) for e in ev.values()])
    runs, cut, cost = [], [], []
    for ds in ("techqa", "expertqalong"):
        for m, short in MODELS.items():
            k = f"{m.split('-')[0]} {ds}"
            for split, e in ev.items():
                t = row(e, ds, m, "truncated")
                runs.append([DS[ds], short, split, f"{t['n']} / {t['sources']}",
                             f"{a(t['window 1'])} / {a(t['B'])} / {a(t['L'])}",
                             ci(e["secondary"][f"{k} | B - L (truncated)"]),
                             ci(e["secondary"][f"{k} | L - window 1 (truncated)"])])
                if m.startswith("SmolLM2"):
                    c = row(e, ds, m, "cut by L")
                    cut.append([DS[ds], split, f"{c['n']} / {c['sources']}",
                                f"{a(c['window 1'])} / {a(c['B'])} / {a(c['L'])}",
                                ci(e["secondary"][f"{k} | B - L (cut by L)"])])
            meta = json.load(open(RES / "features" / f"{m}_{ds}_K5" / "meta_long.json"))["cost"]
            cost.append([DS[ds], short, f"{meta['cut']} / {meta['responses']}",
                         f"{meta['seq_len_median']:,} / {meta['seq_len_max']:,}",
                         f"{meta['seconds_median']:.1f} / {meta['seconds_max']:.1f}",
                         f"{meta['peak_gb_median']:.1f} / {meta['peak_gb_max']:.1f}"])
    return ("**A5. One long pass (L) vs B on sentences whose context needs more than one 1800-token "
            "window.** Dev: out-of-fold scores from grouped 5-fold CV inside dev (the decision); test: fit on dev, "
            "scored once (P4). Window 1, B and L use the same classifier. * = 95% interval excludes 0.\n\n"
            + md(["Pooled (TechQA, ExpertQA-long × 2 scorers)", "dev", "test"], pooled) + "\n\n"
            + md(["Dataset", "Scorer", "split", "sentences / sources", "AUC window 1 / B / L", "B − L", "L − window 1"],
                 runs) + "\n\n"
            + "Sentences whose context SmolLM2's 8,192-position limit cuts (L reads about the first 8k tokens, B reads "
              "all of it):\n\n"
            + md(["Dataset", "split", "sentences / sources", "AUC window 1 / B / L", "B − L"], cut) + "\n\n"
            + "Cost of L per response (median / max; one T4, fp32, memory-efficient attention):\n\n"
            + md(["Dataset", "Scorer", "contexts cut", "sequence length", "seconds", "peak GPU memory (GB)"], cost))


def a6():
    """Step E2: B vs its placebo Bp on TechQA truncated rows, dev (out-of-fold) next to test (P5)."""
    f_dev, f_test = RES / "e2" / "e2_dev.json", RES / "test" / "e2_test_look_P5.json"
    if not (f_dev.exists() and f_test.exists()):
        return None
    ev = {"dev": json.load(open(f_dev)), "test": json.load(open(f_test))}
    ci = lambda r: f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]" + ("*" if r["sig"] else "")  # noqa: E731
    a = lambda v: f"{v:.3f}".replace("0.", ".", 1)  # noqa: E731
    pooled = [[name] + [ci(e["primary"]["D2"] if key is None else e["secondary"][key]) for e in ev.values()]
              for name, key in (("B − placebo (P5, the pre-written rule)", None),
                                ("placebo − window 1", "Bp - window 1 (truncated rows)"),
                                ("B − window 1", "B - window 1 (truncated rows)"),
                                ("B's classifier on placebo features − window 1",
                                 "B classifier on Bp - window 1 (truncated rows)"))]
    runs = []
    for m in ("Qwen2.5", "SmolLM2"):
        for split, e in ev.items():
            t = next(x for x in e["table"] if x["model"] == m and x["rows"] == "truncated")
            runs.append([m, split, f"{t['n']} / {t['sources']}",
                         f"{a(t['window 1'])} / {a(t['B'])} / {a(t['Bp (placebo)'])}",
                         ci(e["secondary"][f"{m} | B - Bp"]), f"{t['placebo_keeps']:.0%}"])
    groups = []
    for m in ("Qwen2.5", "SmolLM2"):
        for g in ("all out", "partly out", "in window 1"):
            cells = [m, {"all out": "all beyond window 1", "partly out": "partly beyond",
                         "in window 1": "all inside window 1"}[g]]
            for e in ev.values():
                r = next((x for x in e.get("evidence", []) if x["model"].startswith(m) and x["group"] == g), None)
                cells.append("–" if r is None else f"{r['n']}: {ci(r['B_minus_Bp'])}")
            groups.append(cells)
    return ("**A6. B vs its placebo (windows 2..K taken from another document of the same split) on TechQA sentences "
            "whose context needs more than one window.** Dev: out-of-fold (the decision); test: fit on dev, scored once "
            "(P5). * = 95% interval excludes 0.\n\n"
            + md(["Pooled over the 2 scorers", "dev", "test"], pooled) + "\n\n"
            + md(["Scorer", "split", "sentences / sources", "AUC window 1 / B / placebo", "B − placebo",
                  "share of B's gain the placebo keeps"], runs) + "\n\n"
            + "B − placebo by where RAGBench's annotated evidence lies (sentences: difference):\n\n"
            + md(["Scorer", "relevant evidence", "dev", "test"], groups))


def a7():
    """Step E3: evidence displaced out of window 1, dev (out-of-fold) next to test (P6)."""
    f_dev, f_test = RES / "e3" / "e3_dev.json", RES / "test" / "e3_test_look_P6.json"
    if not (f_dev.exists() and f_test.exists()):
        return None
    ev = {"dev": json.load(open(f_dev)), "test": json.load(open(f_test))}
    ci = lambda r: f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]" + ("*" if r["sig"] else "")  # noqa: E731
    a = lambda v: f"{v:.3f}".replace("0.", ".", 1)  # noqa: E731
    pooled = [[name] + [ci(e["primary"]["D3"] if key is None else e["secondary"][key]) for e in ev.values()]
              for name, key in (("B − window 1, displaced (P6, the pre-written rule)", None),
                                ("original − window 1, displaced (the loss)", "orig - w1d (the loss)"),
                                ("B, displaced − original", "Bd - orig"))]
    pooled.append(["share of the loss B recovers"]
                  + [f"{e['secondary']['recovered share (pooled point estimate)']['mean']:.0%}" for e in ev.values()])
    runs = []
    for ds in ("ragtruth", "ragbench"):
        for m in ("Qwen2.5", "SmolLM2"):
            for split, e in ev.items():
                t = next(x for x in e["table"] if x["dataset"] == ds and x["model"] == m and x["rows"] == "all")
                runs.append([DS[ds], m, split, f"{t['n']} / {t['sources']}",
                             f"{a(t['original'])} / {a(t['window 1 (displaced)'])} / {a(t['B (displaced)'])}",
                             ci(e["secondary"][f"{m} {ds} | Bd - w1d"]),
                             "–" if t["recovered"] is None else f"{t['recovered']:.0%}"])
    return ("**A7. Evidence displaced out of window 1: responses whose context fits one window, read with the context "
            "moved behind 1808 tokens of other responses' contexts.** Dev: out-of-fold (the decision); test: fit on "
            "dev, scored once (P6). Each reading has its own classifier. * = 95% interval excludes 0.\n\n"
            + md(["Pooled over RAGTruth, RAGBench × 2 scorers", "dev", "test"], pooled) + "\n\n"
            + md(["Dataset", "Scorer", "split", "sentences / sources", "AUC original / window 1 / B (displaced)",
                  "B − window 1, displaced", "loss recovered"], runs))


def main():
    test = pd.read_csv(RES / "test" / "test_look.csv")
    text = "\n\n".join(["# Appendix tables (generated by scripts/paper_tables.py from saved results)",
                        a1(test), a2(test), a3(), a4()] + [t for t in [a5(), a6(), a7()] if t]) + "\n"
    (RES / "paper_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
