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
  "triviaplus-smoke" / "triviaplus" step E4: TRIVIA+-long (human sentence labels; articles >= 9000 characters): the
                                   parquet is fetched into /tmp (licence CC BY-NC-ND 4.0: never saved as output),
                                   GASP's pipeline without its analysis (keeps test numbers unseen), then the frozen
                                   windowed features (+ ReDeEP) and the E1 long pass. The smoke runs the E4 checks
                                   (a), (b) and (d) on the 20-case GASP runs first.
  "checkers-smoke" / "checkers"    step E5: MiniCheck-Flan-T5-Large and LettuceDetect-large, as released, score every GASP
                                   sentence of the 6 datasets x 2 scorer runs found in the attached inputs (canon_results
                                   of the week-1 dataset and of the notebook versions with the long sets). Installs the
                                   checkers' own packages (transformers >= 4.48) after rebuilding any missing long set's
                                   GASP run (deterministic reruns, before the upgrade); one T4 per half of the datasets. The smoke runs the E5 checks (a)-(d) and
                                   scores 20 cases per dataset.
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
MODE = "smoke"   # smoke, full, features, chunked, trunc, redeep, techqa, expertqa, freq, longfreq, longpass, placebo, displace, triviaplus, checkers (+ -smoke)

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

TRIVIAPLUS_URL = ("https://raw.githubusercontent.com/amazon-science/hallucination-benchmark-trivialplus/main/"
                  "triviaplus_dataset.parquet")
TRIVIAPLUS_SHA256 = "fecd7a981778f6c09b35eb493f3fe76cddff81e268ae4a90711e4f2298fc383d"   # file used for E4's rule


def fetch_triviaplus():
    """E4: TRIVIA+ into /tmp, which is NOT saved as output (CC BY-NC-ND 4.0: research use, no sharing)."""
    import hashlib
    import urllib.request
    path = "/tmp/triviaplus/triviaplus_dataset.parquet"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        urllib.request.urlretrieve(TRIVIAPLUS_URL, path)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if digest != TRIVIAPLUS_SHA256:
        raise RuntimeError(f"TRIVIA+ parquet differs from the one E4 was designed on (sha256 {digest})")
    os.environ["TRIVIAPLUS_PARQUET"] = path
    print("TRIVIA+ parquet ready:", path, flush=True)

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
              "longpass-smoke", "longpass", "placebo-smoke", "placebo", "triviaplus-smoke", "triviaplus"):
    base = MODE.replace("-smoke", "")
    lds = {"techqa": "techqa", "expertqa": "expertqalong", "longfreq": "techqa expertqalong",
           "longpass": "techqa expertqalong", "placebo": "techqa", "triviaplus": "triviapluslong"}[base]
    feat_flags = {"longfreq": "--freq", "longpass": "--long", "placebo": "--placebo"}.get(base, "--chunked --redeep")
    if base == "triviaplus":
        fetch_triviaplus()
    import torch
    import yaml
    models = yaml.safe_load(open(f"{REPO_DIR}/configs/reproduce.yaml"))["models"]
    cap = "--max_cases 20" if flag else ""
    procs = []                                 # GASP: one model per GPU, in parallel
    for i, m in enumerate(models):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % max(torch.cuda.device_count(), 1)), PYTHONUNBUFFERED="1")
        cmd = f"python scripts/reproduce_gasp.py --models {m} --datasets {lds} {cap} --outroot {OUTROOT}/gasp_repro"
        cmd += " --no_analysis" if base == "triviaplus" else ""
        print("$", cmd, f"(GPU {env['CUDA_VISIBLE_DEVICES']})", flush=True)
        procs.append(subprocess.Popen(cmd, shell=True, cwd=REPO_DIR, env=env))
    if any(p.wait() for p in procs):
        raise RuntimeError("a GASP run failed; see the log above")
    if base == "triviaplus" and flag:          # E4 checks (a), (b), (d) on the 20-case GASP runs, before extraction
        for m in models:
            tag = f"{m.split('/')[-1]}_triviapluslong_K5"
            sh(f"python -W ignore scripts/triviaplus_check.py --canon_dir {OUTROOT}/gasp_repro/canon_results/{tag} "
               f"--model {m} --min_ctx_chars 9000 --out {OUTROOT}/features/{tag}/triviaplus_check.json", cwd=REPO_DIR)
    sh(f"python scripts/run_features.py {flag} {feat_flags} --datasets {lds} "
       f"--canon_root {OUTROOT}/gasp_repro/canon_results --outroot {OUTROOT}/features", cwd=REPO_DIR)
    if base == "triviaplus":                   # E4 also runs the E1 long pass (L), next to B
        sh(f"python scripts/run_features.py {flag} --long --datasets {lds} "
           f"--canon_root {OUTROOT}/gasp_repro/canon_results --outroot {OUTROOT}/features", cwd=REPO_DIR)
elif MODE in ("checkers-smoke", "checkers"):
    import sys
    import torch
    import yaml
    models = yaml.safe_load(open(f"{REPO_DIR}/configs/reproduce.yaml"))["models"]
    halves = [["techqa", "ragtruth", "tofueval"], ["expertqalong", "triviapluslong", "ragbench"]]   # one T4 each
    roots = [r for r in sorted(set(glob.glob("/kaggle/input/**/canon_results", recursive=True))) if "_smoke" not in r]
    print("canon roots:", roots, flush=True)

    def canon_dir(tag):
        hits = [f"{r}/{tag}" for r in roots if os.path.exists(f"{r}/{tag}/cases.jsonl")]
        return hits[0] if hits else None

    plan = {}
    for ds in sum(halves, []):
        dirs = [canon_dir(f"{m.split('/')[-1]}_{ds}_K5") for m in models]
        if all(dirs):
            plan[ds] = dirs
        else:
            print(f"[missing] {ds}: {dirs}", flush=True)
    print("datasets found:", {d: v for d, v in plan.items()}, flush=True)
    # long sets not among the inputs: rebuild their GASP runs here, BEFORE the package upgrade below (GASP needs the
    # pinned transformers). GASP is deterministic: reruns were byte-identical to the saved runs (Versions 10-12).
    rerun = [ds for ds in ("techqa", "expertqalong", "triviapluslong") if ds not in plan]
    if rerun:
        if "triviapluslong" in rerun:
            fetch_triviaplus()
        cap = "--max_cases 20" if flag else ""
        gprocs = []
        for i, m in enumerate(models):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % max(torch.cuda.device_count(), 1)), PYTHONUNBUFFERED="1")
            cmd = (f"python scripts/reproduce_gasp.py --models {m} --datasets {' '.join(rerun)} {cap} "
                   f"--outroot {OUTROOT}/gasp_repro --no_analysis")
            print("$", cmd, f"(GPU {env['CUDA_VISIBLE_DEVICES']})", flush=True)
            gprocs.append(subprocess.Popen(cmd, shell=True, cwd=REPO_DIR, env=env))
        if any(p.wait() for p in gprocs):
            raise RuntimeError("a GASP rerun failed; see the log above")
        roots.append(f"{OUTROOT}/gasp_repro/canon_results")
        for ds in rerun:
            dirs = [canon_dir(f"{m.split('/')[-1]}_{ds}_K5") for m in models]
            if all(dirs):
                plan[ds] = dirs
        print("datasets after the GASP reruns:", list(plan), flush=True)
    # E5's own environment: ModernBERT needs transformers >= 4.48 (the frozen pipeline is pinned at 4.44.2 and is not
    # used in this mode); MiniCheck pinned to the commit the E5 code was written against
    sh('pip install -q "transformers>=4.48.3,<5" "lettucedetect==0.2.3" accelerate sentencepiece '
       '"minicheck @ git+https://github.com/Liyan06/MiniCheck.git@b58b9fa69acbd1015ec970fa65dd752413a053d2"')
    os.environ["NLTK_DATA"] = "/tmp/nltk_data"
    sh(f"{sys.executable} -c \"import nltk; [nltk.download(p, download_dir='/tmp/nltk_data', quiet=True) "
       f"for p in ('punkt', 'punkt_tab')]\"")
    out = f"{OUTROOT}/features"
    procs = []
    for gpu, dss in enumerate(halves):
        cmds = []
        for ds in [d for d in dss if d in plan]:
            dirs = " ".join(plan[ds])
            if flag:
                cmds.append(f"python -W ignore scripts/checkers_check.py --canon_dirs {dirs} --n 20 "
                            f"--out {out}/checks/checkers_check_{ds}.json")
            cmds.append(f"python -W ignore scripts/score_checkers.py --canon_dirs {dirs} --outroot {out}"
                        + (" --max_cases 20" if flag else ""))
        if not cmds:
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu % max(torch.cuda.device_count(), 1)),
                   PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        script = " && ".join(cmds)
        print("$", script, f"(GPU {env['CUDA_VISIBLE_DEVICES']})", flush=True)
        procs.append(subprocess.Popen(script, shell=True, cwd=REPO_DIR, env=env))
    if any(p.wait() for p in procs):
        raise RuntimeError("a checkers worker failed; see the log above")
else:
    raise ValueError(f"unknown MODE {MODE!r}")

print("\nFinished. Download /kaggle/working/results from the notebook's Output tab.")
