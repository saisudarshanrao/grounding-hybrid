"""Reproduce the GASP baseline using GASP's own, unmodified pipeline scripts.

Usage:
    python scripts/reproduce_gasp.py --smoke          # 30 cases, one model: checks everything works
    python scripts/reproduce_gasp.py                  # full run: all models x all datasets
    python scripts/reproduce_gasp.py --models Qwen/Qwen2.5-1.5B-Instruct --datasets ragtruth
    python scripts/reproduce_gasp.py --dry-run        # print the commands without running them
    python scripts/reproduce_gasp.py --datasets techqa   # long-context RAGBench domain (gasp_longctx.py)

The run is resumable: a (model, dataset) pair whose sentence.csv already exists is skipped,
so if a Kaggle session times out, just run the same command again.

Outputs (under --outroot, default results/gasp_repro):
    canon_results/<TAG>/cases.jsonl, response.csv, sentence.csv, audit.csv   (from GASP)
    canon_results/<TAG>/analysis_span.txt, analysis_response.txt             (GASP's analysis)
    run_log.jsonl                                                            (one line per run)
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
GASP_PIPE = ROOT / "third_party" / "GASP" / "pipeline"
LONGCTX_DOMAINS = ("techqa",)   # one RAGBench domain, all splits, via scripts/gasp_longctx.py


def tag_for(model, dataset, k):
    return f"{model.split('/')[-1]}_{dataset}_K{k}"


def versions():
    info = {"python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info


def run(cmd, log_path=None, dry=False):
    print("$", " ".join(cmd), flush=True)
    if dry:
        return 0
    if log_path is None:
        return subprocess.run(cmd).returncode
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        f.write(proc.stdout)
    print(proc.stdout)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "reproduce.yaml"))
    ap.add_argument("--outroot", default=str(ROOT / "results" / "gasp_repro"))
    ap.add_argument("--models", nargs="*", help="override the model list")
    ap.add_argument("--datasets", nargs="*", help="override the dataset list")
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the pipeline")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max_cases", type=int, default=0, help="cap cases per run (quick tests)")
    args = ap.parse_args()

    if not GASP_PIPE.exists():
        sys.exit("GASP not found. Run: python scripts/setup_gasp.py")

    cfg = yaml.safe_load(open(args.config))
    p = cfg["params"]
    if args.smoke:
        models, datasets = [cfg["smoke"]["model"]], [cfg["smoke"]["dataset"]]
        max_cases = cfg["smoke"]["max_cases"]
        outroot = Path(args.outroot + "_smoke")
    else:
        models = args.models or cfg["models"]
        datasets = args.datasets or cfg["datasets"]
        max_cases = args.max_cases
        outroot = Path(args.outroot)

    canon = outroot / "canon_results"
    canon.mkdir(parents=True, exist_ok=True)
    env = versions()
    print("Environment:", env)

    for model in models:
        for ds in datasets:
            tag = tag_for(model, ds, p["k_chunks"])
            out_dir = canon / tag
            if (out_dir / "sentence.csv").exists():
                print(f"[skip] {tag}: already done")
                continue

            print(f"\n=== {tag} ===", flush=True)
            t0 = time.time()
            entry = ([str(ROOT / "scripts" / "gasp_longctx.py"), "--domain", ds, "--dataset", "ragbench"]
                     if ds in LONGCTX_DOMAINS else [str(GASP_PIPE / "run_gasp.py"), "--dataset", ds])
            cmd = [sys.executable] + entry + ["--model", model,
                   "--k_chunks", str(p["k_chunks"]),
                   "--max_ctx_tokens", str(p["max_ctx_tokens"]),
                   "--max_ans_tokens", str(p["max_ans_tokens"]),
                   "--n_per_class", str(p["n_per_class"]),
                   "--seed", str(p["seed"]),
                   "--tag", tag, "--outroot", str(canon),
                   "--datadir", str(GASP_PIPE / "tofueval_data")]
            if max_cases:
                cmd += ["--max_cases", str(max_cases)]
            rc = run(cmd, dry=args.dry_run)
            if rc != 0:
                print(f"[error] run_gasp failed for {tag} (exit {rc}); continuing")
                continue

            # GASP's own analysis: raw-feature AUCs, trained classifiers, bootstrap CIs
            for level in ("span", "response"):
                run([sys.executable, str(GASP_PIPE / "analyze_gasp.py"),
                     str(out_dir / "sentence.csv"), "--level", level],
                    log_path=None if args.dry_run else out_dir / f"analysis_{level}.txt",
                    dry=args.dry_run)

            if not args.dry_run:
                with open(outroot / "run_log.jsonl", "a") as f:
                    f.write(json.dumps({"tag": tag, "model": model, "dataset": ds,
                                        "minutes": round((time.time() - t0) / 60, 1),
                                        "max_cases": max_cases, "params": p, "env": env}) + "\n")

    print("\nDone. Compare RAGTruth span AUCs with 'reference_ragtruth_span_auc' in the config.")


if __name__ == "__main__":
    main()
