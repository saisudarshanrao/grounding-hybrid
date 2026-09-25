"""Paper figures. Default: from the saved DEV results (no test-split numbers). --test: the two main figures
from the single logged test look (scripts/test_look.py outputs; nothing is refit here).

  fig1_coverage     (a) TechQA: out-of-fold AUC by share of the context inside the 1800-token window,
                        for GASP+base, Lookback (window 1) and B (max over windows)
                    (b) controlled truncation (512-token windows): window 1 vs B vs the full-context
                        upper bound, on the truncated rows of RAGTruth and TofuEval
  fig2_detectors    dev AUC of every detector on every dataset, both scorers (rq2_where.csv)
  fig3_layers       AUC of each layer alone vs relative depth (ablation_layers_per_layer.csv)
  fig4_prior        GASP+base and Lookback AUC by prior tertile (rq2_where.csv): the "already known"
                    premise does not hold
Out-of-fold scores use the same grouped 5-fold CV as every dev analysis (scripts/rq2_where.py).

  --test:
  fig1_coverage_test   (a) TechQA / ExpertQA-long test AUC by share of the context inside window 1 (GASP+base,
                           Lookback = window 1, B); (b) controlled truncation, test: window 1 vs B vs full view
  fig2_detectors_test  test AUC of every frozen detector, 5 datasets x 2 scorers (sentence level)

Usage:
    python -W ignore scripts/make_figures.py                   # dev figures -> results/figures/*.pdf and *.png
    python -W ignore scripts/make_figures.py --test            # test figures from results/test/
    python -W ignore scripts/make_figures.py --test results/test_dryrun   # same code on the dev-half dry run
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grounding_hybrid.gasp_bridge import BASE_FEATS, GASP_FEATS, analyze_gasp  # noqa: E402
from grounding_hybrid.signals import load_signals  # noqa: E402
from rq2_where import oof  # noqa: E402

OUT = ROOT / "results" / "figures"
GATING = ROOT / "results" / "gating"
CANON = ROOT / "results" / "gasp_repro" / "canon_results"
FEAT = ROOT / "results" / "features"
MODELS = {"Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B", "SmolLM2-1.7B-Instruct": "SmolLM2-1.7B"}
COL = {"Perplexity+length": "#bbbbbb", "GASP+base (S1)": "#e69f00", "ReDeEP [cv]": "#cc79a7",
       "Frequency-aware [cv]": "#56b4e9",
       "Lookback (S3)": "#0072b2", "B: Lookback max over windows": "#009e73", "S2 prior alone": "#999999"}
SHORT = {"Perplexity+length": "Perplexity", "GASP+base (S1)": "GASP", "ReDeEP [cv]": "ReDeEP",
         "Frequency-aware [cv]": "Freq-aware",
         "Lookback (S3)": "Lookback", "B: Lookback max over windows": "B (ours, new)", "S2 prior alone": "Prior"}
TEST_NAME = {"Perplexity+length": "Perplexity+length", "GASP+base": "GASP+base (S1)", "ReDeEP [cv]": "ReDeEP [cv]",
             "Frequency-aware [cv]": "Frequency-aware [cv]", "Lookback [cv]": "Lookback (S3)",
             "B (ours) [cv]": "B: Lookback max over windows"}
SPLIT = "test"                         # axis / title label of the --test figures
OURS = dict(hatch="////", edgecolor="k", lw=0.9)   # how B's bars stand out as this paper's new result
DS_NAME = {"ragtruth": "RAGTruth", "tofueval": "TofuEval", "ragbench": "RAGBench", "techqa": "TechQA",
           "expertqalong": "ExpertQA-long"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 200})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"saved {OUT / name}.pdf/.png")


def dev_of(canon, npz=None, keys=("lookback",)):
    df, cols = load_signals(canon, npz, lb_keys=keys) if npz else (load_signals(canon)[0], [])
    d, _ = analyze_gasp.source_split(df, seed=0)
    return d.reset_index(drop=True), cols


def fig1():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.4), gridspec_kw={"width_ratios": [1.1, 1]})
    for model, short in MODELS.items():
        canon = CANON / f"{model}_techqa_K5"
        npz = FEAT / f"{model}_techqa_K5" / "features_chunked_redeep.npz"
        base, _ = dev_of(canon)
        scores = {"GASP+base (S1)": oof(base, GASP_FEATS + BASE_FEATS, False)}
        for name, key in [("Lookback (S3)", "lookback"), ("B: Lookback max over windows", "lookback_max")]:
            d, cols = dev_of(canon, npz, (key,))
            scores[name] = oof(d, cols, True)
        bins = pd.qcut(base["ctx_kept"], 4)
        mids = [iv.mid for iv in bins.cat.categories]
        for name, s in scores.items():
            aucs = [roc_auc_score(base["label"][bins == iv], s[(bins == iv).values]) for iv in bins.cat.categories]
            a.plot(mids, aucs, marker="o", ms=3, color=COL[name], ls="-" if short.startswith("Qwen") else "--",
                   label=SHORT[name] if short.startswith("Qwen") else None)
    a.axhline(0.5, color="k", lw=0.5, ls=":")
    a.set_xlabel("share of the context inside the 1800-token window")
    a.set_ylabel("dev AUC (TechQA)")
    a.set_title("(a) real truncation: TechQA, 600 cases")
    a.set_ylim(0.48, 0.87)
    from matplotlib.lines import Line2D
    first = a.legend(ncol=3, frameon=False, loc="lower left")
    a.add_artist(first)
    a.legend([Line2D([], [], color="k", ls="-"), Line2D([], [], color="k", ls="--")],
             ["Qwen2.5-1.5B", "SmolLM2-1.7B"], ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.07))
    runs, vals = [], {"window 1 (truncated)": [], "B (max over windows)": [], "full context (upper bound)": []}
    for model, short in MODELS.items():
        for ds in ("ragtruth", "tofueval"):
            canon = CANON / f"{model}_{ds}_K5"
            npz = FEAT / f"{model}_{ds}_K5" / "features_chunked_ctx512.npz"
            z = np.load(npz)
            nwin = dict(zip(zip(z["case_id"], z["sent_idx"]), z["n_windows"]))
            s = {}
            for name, f, key in [("window 1 (truncated)", npz, "lookback"), ("B (max over windows)", npz, "lookback_max"),
                                 ("full context (upper bound)", FEAT / f"{model}_{ds}_K5" / "features.npz", "lookback")]:
                d, cols = dev_of(canon, f, (key,))
                s[name] = oof(d, cols, True)
            m = np.array([nwin[(c, k)] > 1 for c, k in zip(d["case_id"], d["sent_idx"])])
            for name in vals:
                vals[name].append(roc_auc_score(d["label"][m], s[name][m]))
            runs.append(f"{short.split('-')[0]}\n{'RAGTruth' if ds == 'ragtruth' else 'TofuEval'}")
    x = np.arange(len(runs))
    for i, (name, v) in enumerate(vals.items()):
        b.bar(x + (i - 1) * 0.27, v, 0.27, label=name,
              color=["#56b4e9", "#009e73", "#dddddd"][i], edgecolor="k", lw=0.3)
    b.set_xticks(x, runs)
    b.set_ylim(0.6, 0.95)
    b.set_ylabel("dev AUC, truncated rows")
    b.set_title("(b) controlled truncation: 512-token windows")
    b.legend(frameon=False, loc="upper left", ncol=3, fontsize=6, columnspacing=0.8, handlelength=1.2)
    save(fig, "fig1_coverage")


def fig2():
    r = pd.read_csv(GATING / "rq2_where.csv")
    r = r[r.by == "all"]
    dets = [d for d in ["Perplexity+length", "GASP+base (S1)", "ReDeEP [cv]", "Frequency-aware [cv]", "Lookback (S3)",
                        "B: Lookback max over windows"] if d in set(r["detector"])]
    w = 0.8 / len(dets)
    order = ["ragtruth", "tofueval", "ragbench", "techqa"]
    names = {"ragtruth": "RAGTruth", "tofueval": "TofuEval", "ragbench": "RAGBench", "techqa": "TechQA\n(long)"}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2), sharey=True)
    for ax, (model, short) in zip(axes, MODELS.items()):
        sub = r[r.run.str.startswith(short)]
        x = np.arange(len(order))
        for i, det in enumerate(dets):
            v = [sub[(sub.run == f"{short}_{ds}") & (sub.detector == det)]["auc"].mean() for ds in order]
            ax.bar(x + (i - (len(dets) - 1) / 2) * w, v, w, color=COL[det], edgecolor="k", lw=0.3, label=SHORT[det])
        ax.set_xticks(x, [names[d] for d in order])
        ax.set_ylim(0.45, 0.87)
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
        ax.set_title(short)
    axes[0].set_ylabel("dev AUC (sentence level)")
    axes[1].legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "fig2_detectors")


def fig3():
    c = pd.read_csv(GATING / "ablation_layers_per_layer.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.1), sharey=True)
    styles = {"ragtruth": "#0072b2", "tofueval": "#d55e00", "ragbench": "#e69f00", "techqa B": "#009e73"}
    for ax, short in zip(axes, MODELS.values()):
        for key, color in styles.items():
            sub = c[c.run == f"{short}_{key}"]
            if len(sub):
                ax.plot(sub.rel_depth, sub.auc, color=color, lw=0.6, alpha=0.3)
                ax.plot(sub.rel_depth, sub.auc.rolling(3, center=True, min_periods=1).mean(), color=color, lw=1.4,
                        label={"ragtruth": "RAGTruth", "tofueval": "TofuEval", "ragbench": "RAGBench",
                               "techqa B": "TechQA (B)"}[key])
        ax.set_xlabel("relative layer depth")
        ax.set_title(short)
    axes[0].set_ylabel("dev AUC, one layer's heads\n(3-layer running mean)")
    axes[1].legend(frameon=False, loc="lower right")
    save(fig, "fig3_layers")


def fig4():
    r = pd.read_csv(GATING / "rq2_where.csv")
    r = r[(r.by == "prior") & r.detector.isin(["GASP+base (S1)", "Lookback (S3)"])]
    order = ["ragtruth", "tofueval", "ragbench", "techqa"]
    names = {"ragtruth": "RAGTruth", "tofueval": "TofuEval", "ragbench": "RAGBench", "techqa": "TechQA"}
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.9), sharey=True)
    for ax, ds in zip(axes, order):
        for short, ls in zip(MODELS.values(), ["-", "--"]):
            for det in ["GASP+base (S1)", "Lookback (S3)"]:
                sub = r[(r.run == f"{short}_{ds}") & (r.detector == det)].set_index("group").reindex(["low", "mid", "high"])
                ax.plot(range(3), sub.auc, marker="o", ms=3, ls=ls, color=COL[det],
                        label=f"{SHORT[det]}, {short.split('-')[0]}")
        ax.set_xticks(range(3), ["low", "mid", "high"])
        ax.set_title(names[ds])
    axes[0].set_ylabel("dev AUC")
    fig.subplots_adjust(bottom=0.33)
    fig.supxlabel("prior tertile (no-context likelihood of the sentence)", fontsize=8, y=0.13)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    save(fig, "fig4_prior")


def fig1_test(tdir):
    """(a) test AUC by share of the context inside window 1; (b) controlled truncation on test."""
    sc = pd.read_csv(tdir / "test_scores.csv.gz")
    sc = sc[sc.level == "span"]
    ctrl = json.load(open(tdir / "test_look.json"))["controlled"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    for ax, ds in zip(axes[:2], ("techqa", "expertqalong")):
        for model, short in MODELS.items():
            t = sc[(sc.dataset == ds) & (sc.model == model)]
            bins = pd.qcut(t["ctx_kept"], 4, duplicates="drop")
            for det in ("GASP+base", "Lookback [cv]", "B (ours) [cv]"):
                name = TEST_NAME[det]
                aucs = [roc_auc_score(t["label"][bins == iv], t[det][bins == iv])
                        if t["label"][bins == iv].nunique() == 2 else np.nan for iv in bins.cat.categories]
                ours = det.startswith("B")
                ax.plot([iv.mid for iv in bins.cat.categories], aucs, marker="o", ms=4 if ours else 3,
                        lw=2.2 if ours else 1.0, color=COL[name], ls="-" if short.startswith("Qwen") else "--",
                        label=SHORT[name] if short.startswith("Qwen") else None, zorder=3 if ours else 2)
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
        ax.set_xlabel("share of context in window 1")
        ax.set_title(f"({'ab'[ds != 'techqa']}) {DS_NAME[ds]}, {SPLIT}")
        ax.set_ylim(0.4, 0.95)
    axes[0].set_ylabel(f"{SPLIT} AUC (sentence level)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=6)
    from matplotlib.lines import Line2D
    axes[1].legend([Line2D([], [], color="k", ls="-"), Line2D([], [], color="k", ls="--")],
                   ["Qwen2.5-1.5B", "SmolLM2-1.7B"], frameon=False, loc="lower left", fontsize=6)
    b = axes[2]
    keys = [("window 1", "window 1 (truncated)", "#56b4e9"), ("B", "B (ours, new)", "#009e73"),
            ("full view", "full context (upper bound)", "#dddddd")]
    x = np.arange(len(ctrl))
    for i, (k, lab, color) in enumerate(keys):
        style = OURS if k == "B" else dict(edgecolor="k", lw=0.3)
        b.bar(x + (i - 1) * 0.27, [r[k] for r in ctrl], 0.27, label=lab, color=color, **style)
    for xi, r in zip(x, ctrl):                      # B - window 1 on this run, as tested
        d = r["diff"]
        b.text(xi, max(r["B"], r["window 1"]) + 0.012, f"{d['mean']:+.3f}{'*' if d['sig'] else ''}",
               ha="center", fontsize=5.5, color="#006b4f")
    b.set_xticks(x, [f"{r['model'].split('-')[0].replace('2.5', '')}\n{DS_NAME[r['dataset']]}" for r in ctrl], fontsize=6)
    b.set_ylim(0.5, 1.0)
    b.set_ylabel(f"{SPLIT} AUC, truncated rows")
    b.set_title(f"(c) controlled truncation (512), {SPLIT}")
    fig.subplots_adjust(wspace=0.45)
    b.legend(frameon=False, loc="upper left", fontsize=6, handlelength=1.2)
    save(fig, "fig1_coverage_test")


def fig0_test(tdir):
    """This paper's new results at a glance: (a) B vs baselines on truncated sentences of the long-context sets,
    with B - Lookback written above B; (b) the three checks written before the test look (P1-P3)."""
    r = pd.read_csv(tdir / "test_look.csv")
    r = r[(r.level == "span") & (r.rows == "truncated")]
    cmp = json.load(open(tdir / "test_look.json"))["comparisons"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.6), gridspec_kw={"width_ratios": [1.5, 1]})
    runs = [(ds, m) for ds in ("techqa", "expertqalong") for m in MODELS]
    x, w = np.arange(len(runs)), 0.26
    for i, (det, lab) in enumerate([("GASP+base", "GASP"), ("Lookback [cv]", "Lookback (window 1)"),
                                    ("B (ours) [cv]", "B (ours, new)")]):
        v = [r[(r.dataset == ds) & (r.model == m) & (r.detector == det)]["auc"].iloc[0] for ds, m in runs]
        style = OURS if det.startswith("B") else dict(edgecolor="k", lw=0.3)
        a.bar(x + (i - 1) * w, v, w, color=COL[TEST_NAME[det]], label=lab, **style)
        if det.startswith("B"):
            for xi, vi, (ds, m) in zip(x, v, runs):
                d = cmp[f"{m.split('-')[0]} {ds} span | B - Lookback (truncated rows)"]
                a.text(xi + w, vi + 0.008, f"{d['mean']:+.3f}{'*' if d['sig'] else ''}", ha="center", fontsize=6,
                       color="#006b4f", fontweight="bold" if d["sig"] else "normal")
    a.set_xticks(x, [f"{DS_NAME[ds]}\n{MODELS[m].split('-')[0].replace('2.5', '')}" for ds, m in runs], fontsize=6.3)
    a.set_ylim(0.5, 0.9)
    a.set_ylabel(f"{SPLIT} AUC, truncated sentences")
    a.set_title("(a) Long contexts: B vs the baselines")
    a.legend(frameon=False, loc="upper left", ncol=3, fontsize=6, columnspacing=1.0, handlelength=1.6)
    dev_p1 = json.load(open(GATING / "dev_coverage_checks_chunked_ctx512.json"))["pooled"]["ALL 4 runs | max"]
    checks = [("P3  Lookback \u2212 GASP\n5 datasets \u00d7 2 scorers", cmp["POOLED all 5 datasets x 2 | Lookback - GASP+base"], None),
              ("P2  B \u2212 Lookback\nlong contexts, truncated", cmp["POOLED real long-context (TechQA, ExpertQA-long x 2) truncated rows | B - Lookback"], None),
              ("P1  B \u2212 window 1\ncontrolled 512 tokens", cmp["POOLED controlled 512 (4 runs) | B - window 1"],
               dict(mean=dev_p1["diff"], lo=dev_p1["lo"], hi=dev_p1["hi"]))]
    for yi, (lab, t, dv) in enumerate(checks):
        y = len(checks) - 1 - yi
        color = "#009e73" if "B" in lab.split("\n")[0] else COL["Lookback (S3)"]
        b.errorbar(t["mean"], y, xerr=[[t["mean"] - t["lo"]], [t["hi"] - t["mean"]]], fmt="o", color=color, ms=5,
                   capsize=2, lw=1.2)
        if dv is not None:
            b.errorbar(dv["mean"], y - 0.28, xerr=[[dv["mean"] - dv["lo"]], [dv["hi"] - dv["mean"]]], fmt="o",
                       mfc="white", color=color, ms=4, capsize=2, lw=0.8)
        b.text(0.125, y, "holds" if t["sig"] and t["mean"] > 0 else "does not hold", va="center", fontsize=6.5,
               color="#006b4f" if t["sig"] else "#a33")
    b.axvline(0, color="k", lw=0.6, ls=":")
    b.set_yticks(range(len(checks)), [c[0] for c in reversed(checks)], fontsize=6.5)
    b.set_xlim(-0.03, 0.16)
    b.set_ylim(-0.6, len(checks) - 0.5)
    b.set_xlabel("AUC difference, 95% interval (filled: test, open: dev)")
    b.set_title("(b) Checks written before the test look")
    fig.subplots_adjust(wspace=0.55)
    save(fig, "fig0_main_results_test")


def fig2_test(tdir):
    """Test AUC of every frozen detector; B only where a context needs more than one window."""
    r = pd.read_csv(tdir / "test_look.csv")
    r = r[(r.level == "span") & (r.rows == "all")]
    dets = [d for d in TEST_NAME if d in set(r["detector"])]
    order = [d for d in DS_NAME if d in set(r["dataset"])]
    w = 0.8 / len(dets)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2), sharey=True)
    for ax, (model, short) in zip(axes, MODELS.items()):
        sub = r[r.model == model].set_index(["dataset", "detector"])["auc"]
        x = np.arange(len(order))
        for i, det in enumerate(dets):
            v = [sub.get((ds, det), np.nan) for ds in order]
            style = OURS if det.startswith("B") else dict(edgecolor="k", lw=0.3)
            ax.bar(x + (i - (len(dets) - 1) / 2) * w, v, w, color=COL[TEST_NAME[det]], label=SHORT[TEST_NAME[det]],
                   **style)
        ax.set_xticks(x, [DS_NAME[d].replace("-", "-\n") for d in order], fontsize=6)
        ax.set_ylim(0.45, 0.9)
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
        ax.set_title(short)
    axes[0].set_ylabel(f"{SPLIT} AUC (sentence level)")
    axes[1].legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "fig2_detectors_test")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="?", const=str(ROOT / "results" / "test"), default=None,
                    help="draw the test figures from this test_look output folder")
    args = ap.parse_args()
    if args.test:
        tdir = Path(args.test)
        if tdir.name != "test":            # dry run: keep its figures apart from the real ones
            OUT, SPLIT = OUT / tdir.name, "dry run"
        fig0_test(tdir)
        fig2_test(tdir)
        fig1_test(tdir)
    else:
        fig2()
        fig3()
        fig4()
        fig1()
