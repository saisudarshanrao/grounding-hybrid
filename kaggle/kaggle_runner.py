"""Paste this whole file into ONE cell of a Kaggle notebook and run it.

Kaggle notebook settings (right-hand panel):
  - Accelerator: GPU T4 x2
  - Internet: On
  - Add-ons > Secrets: add a secret named GITHUB_TOKEN (a fine-grained GitHub token with
    read-only access to this one repository). Never paste the token into the code.
    Without the secret the clone still works, but only while the repository is public.

MODE (set below):
  "smoke" / "full"                 week 1: GASP reproduction (30 cases / everything)
  "features-smoke" / "features"    week 2: shared-extractor features (Lookback Lens) + evaluation,
                                   both models in parallel, one per T4. Needs the week 1 GASP output
                                   attached as input: Add Input > Your Work > Notebooks > the GASP
                                   notebook (its results/gasp_repro/canon_results is found automatically).
  "chunked-smoke" / "chunked"      week 2b: coverage-aware reading (whole context in 1800-token
                                   windows) on RAGBench, extraction only; evaluate on the Mac.
  "trunc-smoke" / "trunc"          week 2b: controlled truncation, 512-token windows on RAGTruth and
                                   TofuEval, extraction only; evaluate on the Mac.
  "redeep-smoke" / "redeep"        week 3: ReDeEP baseline scores (ECS per head, PKS per layer) plus
                                   Lookback, all datasets, extraction only; evaluate on the Mac.
  "techqa-smoke" / "techqa"        week 3b: real truncation. GASP's pipeline on a 600-case TechQA sample
                                   (RAGBench, all splits; both models in parallel, one per T4), then
                                   coverage-aware Lookback + ReDeEP features on it; evaluate on the Mac.
Run the smoke variant first. If the cell stops midway, run it again in the same session:
finished parts are skipped. Long runs: Save Version > Save & Run All, so a closed browser
does not stop them. Results go to /kaggle/working/results, kept as the notebook output.
"""
import glob
import os
import shutil
import subprocess

GITHUB_USER = "saisudarshanrao"
REPO_NAME = "grounding-hybrid"
MODE = "smoke"   # smoke, full, features-smoke, features, chunked-smoke, chunked, trunc-smoke, trunc, redeep-smoke, redeep, techqa-smoke, techqa

REPO_DIR = "/tmp/" + REPO_NAME                 # code lives in /tmp, which is NOT saved as output
OUTROOT = "/kaggle/working/results"            # results ARE saved as output


def sh(cmd, cwd=None, secret=False):
    print("$", "git clone <private repo>" if secret else cmd)
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        if secret:  # the exception text contains the command, i.e. the token: never show it
            raise RuntimeError(f"command failed (exit {e.returncode}); "
                               "check the GITHUB_TOKEN secret and its repo access") from None
        raise


# 1. Fresh clone of the latest code (token read from Kaggle Secrets, then removed from git config)
from kaggle_secrets import UserSecretsClient

if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
try:
    token = UserSecretsClient().get_secret("GITHUB_TOKEN").strip()
except Exception:
    token = None
    print("No GITHUB_TOKEN secret attached: cloning without it (works only if the repo is public)")
# GitHub's documented form: user name as the user, token as the password
auth = f"{GITHUB_USER}:{token}@" if token else ""
sh(f"git clone -q https://{auth}github.com/{GITHUB_USER}/{REPO_NAME}.git {REPO_DIR}", secret=bool(token))
sh(f"git remote set-url origin https://github.com/{GITHUB_USER}/{REPO_NAME}.git", cwd=REPO_DIR)
del token, auth
sh("git log -1 --oneline", cwd=REPO_DIR)   # shows exactly which commit this run used

# 2. Install pinned dependencies and fetch GASP + TofuEval
sh("pip install -q -r requirements.txt", cwd=REPO_DIR)
sh("python scripts/env_check.py", cwd=REPO_DIR)
sh("python scripts/setup_gasp.py", cwd=REPO_DIR)

# 3. Run (resumable within the session)
flag = "--smoke" if MODE.endswith("smoke") else ""
if MODE in ("smoke", "full"):
    sh(f"python scripts/reproduce_gasp.py {flag} --outroot {OUTROOT}/gasp_repro", cwd=REPO_DIR)
elif MODE in ("features-smoke", "features", "chunked-smoke", "chunked", "trunc-smoke", "trunc",
              "redeep-smoke", "redeep"):
    found = [d for d in sorted(glob.glob("/kaggle/input/**/canon_results", recursive=True)) if "_smoke" not in d]
    if not found:
        raise RuntimeError("attach the week 1 GASP notebook output as input (see the notes at the top)")
    print("GASP runs read from", found[0])
    extra = {"chunked": "--chunked --datasets ragbench",
             "trunc": "--chunked --max_ctx_tokens 512 --overlap 128 --datasets ragtruth tofueval",
             "redeep": "--redeep"}.get(
        MODE.replace("-smoke", ""), "")
    sh(f"python scripts/run_features.py {flag} {extra} --canon_root {found[0]} --outroot {OUTROOT}/features",
       cwd=REPO_DIR)
elif MODE in ("techqa-smoke", "techqa"):
    import torch
    import yaml
    models = yaml.safe_load(open(f"{REPO_DIR}/configs/reproduce.yaml"))["models"]
    cap = "--max_cases 20" if flag else ""
    procs = []                                 # GASP: one model per GPU, in parallel
    for i, m in enumerate(models):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % max(torch.cuda.device_count(), 1)), PYTHONUNBUFFERED="1")
        cmd = f"python scripts/reproduce_gasp.py --models {m} --datasets techqa {cap} --outroot {OUTROOT}/gasp_repro"
        print("$", cmd, f"(GPU {env['CUDA_VISIBLE_DEVICES']})", flush=True)
        procs.append(subprocess.Popen(cmd, shell=True, cwd=REPO_DIR, env=env))
    if any(p.wait() for p in procs):
        raise RuntimeError("a GASP run failed; see the log above")
    sh(f"python scripts/run_features.py {flag} --chunked --redeep --datasets techqa "
       f"--canon_root {OUTROOT}/gasp_repro/canon_results --outroot {OUTROOT}/features", cwd=REPO_DIR)
else:
    raise ValueError(f"unknown MODE {MODE!r}")

print("\nFinished. Download /kaggle/working/results from the notebook's Output tab.")
