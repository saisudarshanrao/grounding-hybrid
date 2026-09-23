"""Fetch the GASP baseline (pinned commit) and the TofuEval label files it needs.

Run once per machine (Mac or Kaggle):
    python scripts/setup_gasp.py

Result:
    third_party/GASP/                         GASP code at the pinned commit
    third_party/GASP/pipeline/tofueval_data/  TofuEval MeetingBank dev/test CSVs
"""
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
THIRD = ROOT / "third_party"
GASP_DIR = THIRD / "GASP"
TOFU_TMP = THIRD / "_tofueval_tmp"
TOFU_FILES = ["meetingbank_factual_eval_dev.csv", "meetingbank_factual_eval_test.csv"]


def run(cmd, cwd=None):
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "reproduce.yaml"))
    g = cfg["gasp"]
    THIRD.mkdir(exist_ok=True)

    # 1. GASP at the pinned commit
    if not GASP_DIR.exists():
        run(["git", "clone", g["repo"], str(GASP_DIR)])
    run(["git", "fetch", "--all", "--quiet"], cwd=GASP_DIR)
    run(["git", "checkout", "--quiet", g["commit"]], cwd=GASP_DIR)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=GASP_DIR, text=True).strip()
    if head != g["commit"]:
        sys.exit(f"GASP is at {head}, expected {g['commit']}")
    print(f"GASP pinned at {head[:7]}")

    # 2. TofuEval label files, placed where GASP's run_gasp.py expects them
    dest = GASP_DIR / "pipeline" / "tofueval_data"
    dest.mkdir(exist_ok=True)
    if all((dest / f).exists() for f in TOFU_FILES):
        print("TofuEval files already present")
    else:
        if TOFU_TMP.exists():
            shutil.rmtree(TOFU_TMP)
        run(["git", "clone", "--depth", "1", g["tofueval_repo"], str(TOFU_TMP)])
        for f in TOFU_FILES:
            shutil.copy(TOFU_TMP / "factual_consistency" / f, dest / f)
        shutil.rmtree(TOFU_TMP)
        print(f"TofuEval files copied to {dest}")

    print("Setup complete.")


if __name__ == "__main__":
    main()
