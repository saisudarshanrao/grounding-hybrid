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
  "expertqa-smoke" / "expertqa"    step 5: the same on ExpertQA-long (RAGBench ExpertQA, contexts
                                   >= 9000 characters, 466 cases): a second real long-context set.
  "freq-smoke" / "freq"            step 7c: frequency-aware attention baseline (+ Lookback), RAGTruth,
                                   TofuEval, RAGBench, extraction only; evaluate on the Mac.
  "longfreq-smoke" / "longfreq"    step 7d: frequency-aware attention on TechQA + ExpertQA-long: GASP's
                                   deterministic sampling/scoring again (to rebuild the same cases), then
                                   one --freq pass per case (window 1); evaluate on the Mac.
  "longpass-smoke" / "longpass"    step E1: baseline L, one long pass per response over as much context as the
                                   scorer allows (SDPA + answer-row attention), TechQA + ExpertQA-long; GASP
                                   reruns to rebuild the cases. The smoke runs the E1 checks (a) and (b) first.
  "placebo-smoke" / "placebo"      step E2: placebo reading on TechQA, B with windows 2..K taken from a donor
                                   case (same split, other source); GASP reruns to rebuild the cases. The smoke
                                   runs the E2 checks (a) and (c) first.
  "displace-smoke" / "displace"    step E3: evidence displacement on RAGTruth + RAGBench (responses whose context
                                   fits one window, moved behind 1808 tokens of other cases' contexts), read with
                                   the frozen windowed extractor; uses the attached GASP runs (no rerun). The smoke
                                   runs the E3 checks (a), (b) and (d) first.
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
MODE = "smoke"   # smoke, full, features, chunked, trunc, redeep, techqa, expertqa, freq, longfreq, longpass, placebo, displace (+ -smoke)

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


def prefetch_models(attempts=4, timeout=600):
    """Download every scorer before any stage runs. A Hugging Face download can stall forever
    ("Read timed out ... Trying to resume download..."), which would hang a background run until
    Kaggle's 12 h limit; here a stalled attempt is killed after `timeout` s and resumed."""
    import sys
    import yaml
    code = ("import sys; from huggingface_hub import snapshot_download; "
            "print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model']))")
    for m in yaml.safe_load(open(f"{REPO_DIR}/configs/reproduce.yaml"))["models"]:
        for i in range(attempts):
            try:
                subprocess.run([sys.executable, "-c", code, m], check=True, timeout=timeout)
                print(f"model ready: {m}", flush=True)
                break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                print(f"download of {m} failed or stalled (attempt {i + 1}/{attempts}: {type(e).__name__}); "
                      "retrying", flush=True)
        else:
            raise RuntimeError(f"could not download {m} after {attempts} attempts")


prefetch_models()

# 3. Run (resumable within the session)
flag = "--smoke" if MODE.endswith("smoke") else ""
if MODE in ("smoke", "full"):
    sh(f"python scripts/reproduce_gasp.py {flag} --outroot {OUTROOT}/gasp_repro", cwd=REPO_DIR)
elif MODE in ("features-smoke", "features", "chunked-smoke", "chunked", "trunc-smoke", "trunc",
              "redeep-smoke", "redeep", "freq-smoke", "freq", "displace-smoke", "displace"):
    found = [d for d in sorted(glob.glob("/kaggle/input/**/canon_results", recursive=True)) if "_smoke" not in d]
    if not found:
        raise RuntimeError("attach the week 1 GASP notebook output as input (see the notes at the top)")
    print("GASP runs read from", found[0])
    extra = {"chunked": "--chunked --datasets ragbench",
             "trunc": "--chunked --max_ctx_tokens 512 --overlap 128 --datasets ragtruth tofueval",
             "redeep": "--redeep", "freq": "--freq", "displace": "--displace --datasets ragtruth ragbench"}.get(
        MODE.replace("-smoke", ""), "")
    sh(f"python scripts/run_features.py {flag} {extra} --canon_root {found[0]} --outroot {OUTROOT}/features",
       cwd=REPO_DIR)
elif MODE in ("techqa-smoke", "techqa", "expertqa-smoke", "expertqa", "longfreq-smoke", "longfreq",
              "longpass-smoke", "longpass", "placebo-smoke", "placebo"):
    lds = {"techqa": "techqa", "expertqa": "expertqalong", "longfreq": "techqa expertqalong",
           "longpass": "techqa expertqalong", "placebo": "techqa"}[MODE.replace("-smoke", "")]
    feat_flags = {"longfreq": "--freq", "longpass": "--long", "placebo": "--placebo"}.get(
        MODE.replace("-smoke", ""), "--chunked --redeep")
    import torch
    import yaml
    models = yaml.safe_load(open(f"{REPO_DIR}/configs/reproduce.yaml"))["models"]
    cap = "--max_cases 20" if flag else ""
    procs = []                                 # GASP: one model per GPU, in parallel
    for i, m in enumerate(models):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % max(torch.cuda.device_count(), 1)), PYTHONUNBUFFERED="1")
        cmd = f"python scripts/reproduce_gasp.py --models {m} --datasets {lds} {cap} --outroot {OUTROOT}/gasp_repro"
        print("$", cmd, f"(GPU {env['CUDA_VISIBLE_DEVICES']})", flush=True)
        procs.append(subprocess.Popen(cmd, shell=True, cwd=REPO_DIR, env=env))
    if any(p.wait() for p in procs):
        raise RuntimeError("a GASP run failed; see the log above")
    sh(f"python scripts/run_features.py {flag} {feat_flags} --datasets {lds} "
       f"--canon_root {OUTROOT}/gasp_repro/canon_results --outroot {OUTROOT}/features", cwd=REPO_DIR)
else:
    raise ValueError(f"unknown MODE {MODE!r}")

print("\nFinished. Download /kaggle/working/results from the notebook's Output tab.")
