"""Extract and evaluate features for every (model, dataset), one GPU per model in parallel.

Reads finished GASP runs (cases.jsonl + sentence.csv per TAG) from --canon_root, e.g. the
Week 1 Kaggle output attached as notebook input, and writes <outroot>/<TAG>/features.npz,
meta.json, eval.json and eval.txt. Each model runs in its own process pinned to its own GPU
(both T4s busy); its datasets run in turn. Worker output streams live, prefixed with the
model name, and is also saved to <outroot>/logs/<model>.log. Resumable: finished runs are skipped.

Usage:
    python scripts/run_features.py --canon_root /kaggle/input/<...>/canon_results \
        --outroot /kaggle/working/results/features
    python scripts/run_features.py ... --smoke      # 20 cases per run, extraction only
    python scripts/run_features.py ... --chunked --datasets ragbench --no_eval   # coverage-aware
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def tag_for(model, dataset, k):
    return f"{model.split('/')[-1]}_{dataset}_K{k}"


def run_worker(model, args, cfg):
    """All datasets for one model, sequentially, on whatever GPU this process was given."""
    k = cfg["params"]["k_chunks"]
    for ds in (args.datasets or cfg["datasets"]):
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
        if args.chunked:
            cmd += ["--chunked"]
        if subprocess.run(cmd).returncode != 0:
            print(f"[error] extraction failed for {tag}", flush=True)
            continue
        out_dir = Path(args.outroot) / tag
        if args.smoke or args.no_eval or args.chunked or (out_dir / "eval.json").exists():
            continue
        r = subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts" / "eval_features.py"),
                            "--canon_dir", str(canon), "--features", str(out_dir / "features.npz")],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (out_dir / "eval.txt").write_text(r.stdout)
        print(r.stdout, flush=True)


def _pump(proc, log_path, prefix):
    """Copy a worker's output to its log file and, prefixed, to this process's stdout (live)."""
    with open(log_path, "w") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(f"[{prefix}] {line}", end="", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_root", required=True, help="folder holding the GASP run folders (TAGs)")
    ap.add_argument("--outroot", default=str(ROOT / "results" / "features"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "reproduce.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--chunked", action="store_true", help="coverage-aware reading (extraction only)")
    ap.add_argument("--datasets", nargs="*", help="override the dataset list")
    ap.add_argument("--no_eval", action="store_true", help="extract only; evaluate on the Mac")
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
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % n_gpu), PYTHONUNBUFFERED="1") if n_gpu \
            else dict(os.environ, PYTHONUNBUFFERED="1")
        cmd = [sys.executable, __file__, "--worker", model, "--canon_root", args.canon_root,
               "--outroot", args.outroot, "--config", args.config]
        cmd += (["--smoke"] if args.smoke else []) + (["--chunked"] if args.chunked else [])
        cmd += (["--no_eval"] if args.no_eval else []) + (["--datasets"] + args.datasets if args.datasets else [])
        short = model.split("/")[-1]
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        pump = threading.Thread(target=_pump, args=(p, log_dir / f"{short}.log", short))
        pump.start()
        procs.append((model, p, pump))
        if n_gpu <= 1:                      # one device: run the models one after another
            p.wait()
            pump.join()
    for model, p, pump in procs:
        p.wait()
        pump.join()
        print(f"##### {model}: exit {p.returncode} #####", flush=True)
    print(f"All done in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
