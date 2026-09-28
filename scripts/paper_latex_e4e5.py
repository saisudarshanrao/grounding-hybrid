"""LaTeX snippets for the ACL paper (paper/acl/) from the saved E4 and E5 results; nothing is fit or scored here.

Reads results/e4/e4_dev.json, results/test/e4_test_look_P7.json, results/e5/e5_dev.json, results/test/e5_test_look_P8.json
and writes paper/acl/generated/:
  checks_rows.tex   the P7 and P8 rows of the paper's checks table (Table 1)
  e4_table.tex      pooled E4 comparison (single-column table body)
  e5_table.tex      pooled E5 comparison (single-column table body)
  a8.tex, a9.tex    appendix tables (full width)

Usage:
    python scripts/paper_latex_e4e5.py [--res results] [--out paper/acl/generated]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES = {"ragtruth": "RAGTruth", "tofueval": "TofuEval", "ragbench": "RAGBench", "techqa": "TechQA",
         "expertqalong": "ExpertQA-long", "triviapluslong": "TRIVIA+-long"}


def num(x, digits=3, sign=True):
    s = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    return s.replace("-", "$-$")


def ci(r, digits=3):
    return f"{num(r['mean'], digits)} [{num(r['lo'], digits)}, {num(r['hi'], digits)}]" + (r"\sig" if r["sig"] else "")


def auc(x):
    return "--" if x is None else f"{x:.3f}".replace("0.", ".", 1)


def verdict_word(r):
    return "holds" if r["lo"] > 0 else "reversed" if r["hi"] < 0 else "not shown"


def load(res, a, b):
    fa, fb = res / a, res / b
    return (json.load(open(fa)) if fa.exists() else None), (json.load(open(fb)) if fb.exists() else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "acl" / "generated"))
    args = ap.parse_args()
    res, out = Path(args.res), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    e4d, e4t = load(res, "e4/e4_dev.json", "test/e4_test_look_P7.json")
    e5d, e5t = load(res, "e5/e5_dev.json", "test/e5_test_look_P8.json")
    rows = []
    if e4d and e4t:
        d, t = e4d["primary"]["D4"], e4t["primary"]["D4"]
        rows.append(rf"P7 & B $-$ Lookback on TRIVIA+-long (human labels) & {ci(d)} & {ci(t)} & {verdict_word(t)} \\")
        (out / "e4_table.tex").write_text("\n".join([
            r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"TRIVIA+-long, pooled & Dev (out-of-fold) & Test (P7) \\",
            r"\midrule",
            rf"B $-$ Lookback & {ci(d)} & {ci(t)} \\",
            rf"B $-$ L & {ci(e4d['secondary']['B - L (truncated rows), pooled'])} & "
            rf"{ci(e4t['secondary']['B - L (truncated rows), pooled'])} \\",
            rf"Lookback $-$ GASP & {ci(e4d['secondary']['Lookback - GASP+base (truncated rows), pooled'])} & "
            rf"{ci(e4t['secondary']['Lookback - GASP+base (truncated rows), pooled'])} \\",
            r"\bottomrule", r"\end{tabular}"]) + "\n")
        body = []
        for split, e in (("dev", e4d), ("test", e4t)):
            for r in e["table"]:
                m = r["model"]
                full = next(k for k in e["power"] if k.startswith(m))
                body.append(rf"{m} & {split} & {r['n']} / {e['power'][full]['sources']} & "
                            rf"{auc(r['GASP+base'])} / {auc(r['Lookback [cv]'])} / {auc(r['L (one pass) [cv]'])} / "
                            rf"{auc(r['B (ours) [cv]'])} & {ci(e['secondary'][f'{m} | B - Lookback'])} & "
                            rf"{ci(e['secondary'][f'{m} | B - L'])} \\")
        (out / "a8.tex").write_text("\n".join([
            r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
            r"Scorer & Split & Sentences / sources & AUC GASP / Lookback / L / B & B $-$ Lookback & B $-$ L \\",
            r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    if e5d and e5t:
        cells = []
        for c in ("MiniCheck", "LettuceDetect"):
            d, t = e5d["primary"][c]["D"], e5t["primary"][c]["D"]
            cells.append((c, d, t))
            word = "B ahead" if t["lo"] > 0 else f"{c} ahead" if t["hi"] < 0 else "tie"
            rows.append(rf"P8 & B $-$ {c} on the long sets & {ci(d)} & {ci(t)} & {word} \\")
        (out / "e5_table.tex").write_text("\n".join(
            [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"Long sets, pooled & Dev & Test (P8) \\", r"\midrule"]
            + [rf"B $-$ {c} & {ci(d)} & {ci(t)} \\" for c, d, t in cells]
            + [r"\bottomrule", r"\end{tabular}"]) + "\n")
        body = []
        for split, e in (("dev", e5d), ("test", e5t)):
            for r in e["table"]:
                note = r"$^{\dagger}$" if r["dataset"] == "ragtruth" else ""
                body.append(rf"{NAMES.get(r['dataset'], r['dataset'])} & {r['model']} & {split} & {r['rows']} & "
                            rf"{auc(r['Lookback'])} / {auc(r['B'])} / {auc(r['MiniCheck'])} / "
                            rf"{auc(r['LettuceDetect'])}{note} \\")
        (out / "a9.tex").write_text("\n".join([
            r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
            r"Dataset & Scorer & Split & Sentences & AUC Lookback / B / MiniCheck / LettuceDetect \\",
            r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    (out / "checks_rows.tex").write_text("\n".join(rows) + "\n")
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
