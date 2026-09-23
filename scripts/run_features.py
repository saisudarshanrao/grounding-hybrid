"""Extract and evaluate features for every (model, dataset), one GPU per model in parallel.

Reads finished GASP runs (cases.jsonl + sentence.csv per TAG) from --canon_root, e.g. the
Week 1 Kaggle output attached as notebook input, and writes <outroot>/<TAG>/features.npz,
meta.json, eval.json and eval.txt. Each model runs in its own process pinned to its own GPU
(both T4s busy); its datasets run in turn. Resumable: finished runs are skipped.

Usage:
    python scripts/run_features.py --canon_root /kaggle/input/<...>/canon_results \
        --outroot /kaggle/working/results/features
    python scripts/run_features.py ... --smoke      # 20 cases per run, extraction only
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def tag_for(model, dataset, k):
    return f"{model.split('/')[-1]}_{dataset}_K{k}"


def run_worker(model, args, cfg):
    """All datasets for one model, sequentially, on whatever GPU this process was given."""
    k = cfg["params"]["k_chunks"]
    for ds in cfg["datasets"]:
        tag = tag_for(model, ds, k)
        canon = Path(args.canon_root) / tag
        if not (canon / "cases.jsonl").exists():
            print(f"[missing] {canon}: no GASP run to extract from", flush=True)
            continue
        print(f"\n=== {tag} ===", flush=True)
        cmd = [sys.executable, str(ROOT / "scripts" / "extract_features.py"), "--canon_dir", str(canon),
               "--model", model, "--outroot", args.outroot]
        if args.smoke:
            cmd += ["--max_cases", "20"]
        if subprocess.run(cmd).returncode != 0:
            print(f"[error] extraction failed for {tag}", flush=True)
            continue
        out_dir = Path(args.outroot) / tag
        if args.smoke or (out_dir / "eval.json").exists():
            continue
        r = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts" / "eval_features.py"),
                            "--canon_dir", str(canon), "--features", str(out_dir / "features.npz")],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (out_dir / "eval.txt").write_text(r.stdout)
        print(r.stdout, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_root", required=True, help="folder holding the GASP run folders (TAGs)")
    ap.add_argument("--outroot", default=str(ROOT / "results" / "features"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "reproduce.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.worker:
        run_worker(args.worker, args, cfg)
        return

    import torch
    n_gpu = torch.cuda.device_count()
    models = cfg["models"]
    log_dir = Path(args.outroot) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(models)} models on {n_gpu} GPU(s): " + ", ".join(models), flush=True)

    t0, procs = time.time(), []
    for i, model in enumerate(models):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % n_gpu)) if n_gpu else dict(os.environ)
        cmd = [sys.executable, __file__, "--worker", model, "--canon_root", args.canon_root,
               "--outroot", args.outroot, "--config", args.config] + (["--smoke"] if args.smoke else [])
        log = open(log_dir / f"{model.split('/')[-1]}.log", "w")
        procs.append((model, log, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
        if n_gpu <= 1:                      # one device: run the models one after another
            procs[-1][2].wait()
    for model, log, p in procs:
        p.wait()
        log.close()
        print(f"\n##### {model} (exit {p.returncode}) #####", flush=True)
        print((log_dir / f"{model.split('/')[-1]}.log").read_text(), flush=True)
    print(f"All done in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
