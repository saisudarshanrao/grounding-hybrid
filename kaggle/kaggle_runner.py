"""Paste this whole file into ONE cell of a Kaggle notebook and run it.

Kaggle notebook settings (right-hand panel):
  - Accelerator: GPU T4 x2
  - Internet: On
  - Add-ons > Secrets: add a secret named GITHUB_TOKEN (a fine-grained GitHub token with
    read-only access to this one repository). Never paste the token into the code.

Set MODE below: "smoke" first (30 cases, a few minutes), then "full" once smoke succeeds.
If the cell stops midway, run it again in the same session: finished parts are skipped.
Results are written to /kaggle/working/results, which Kaggle keeps as the notebook output.
"""
import os
import shutil
import subprocess

GITHUB_USER = "saisudarshanrao"
REPO_NAME = "grounding-hybrid"
MODE = "smoke"          # "smoke" or "full"

REPO_DIR = "/tmp/" + REPO_NAME                 # code lives in /tmp, which is NOT saved as output
OUTROOT = "/kaggle/working/results/gasp_repro"  # results ARE saved as output


def sh(cmd, cwd=None, secret=False):
    print("$", "git clone <private repo>" if secret else cmd)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


# 1. Fresh clone of the latest code (token read from Kaggle Secrets, then removed from git config)
from kaggle_secrets import UserSecretsClient

if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
sh(f"git clone -q https://{token}@github.com/{GITHUB_USER}/{REPO_NAME}.git {REPO_DIR}", secret=True)
sh(f"git remote set-url origin https://github.com/{GITHUB_USER}/{REPO_NAME}.git", cwd=REPO_DIR)
del token
sh("git log -1 --oneline", cwd=REPO_DIR)   # shows exactly which commit this run used

# 2. Install pinned dependencies and fetch GASP + TofuEval
sh("pip install -q -r requirements.txt", cwd=REPO_DIR)
sh("python scripts/env_check.py", cwd=REPO_DIR)
sh("python scripts/setup_gasp.py", cwd=REPO_DIR)

# 3. Run (resumable within the session)
flag = "--smoke" if MODE == "smoke" else ""
sh(f"python scripts/reproduce_gasp.py {flag} --outroot {OUTROOT}", cwd=REPO_DIR)

print("\nFinished. Download /kaggle/working/results from the notebook's Output tab.")
