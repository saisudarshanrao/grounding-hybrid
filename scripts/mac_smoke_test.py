"""Run GASP on one toy example, on your Mac (MPS) or any machine.

Purpose: (1) check that GASP works in your environment, and (2) see our research
problem with your own eyes. The answer below has three kinds of sentence:
  A. grounded, and the model already knows it   -> GASP should score it LOW (the failure case)
  B. grounded, and only knowable from the context -> GASP should score it HIGH
  C. unsupported by the context                  -> GASP should score it LOW (correct)
If A and C get similar low scores, GASP cannot tell them apart. That is the gap we fix.
The query asks for sentence A's fact directly, so the model can predict it without the
context (as in short-answer QA). If the query does not elicit A, A scores high instead.

Usage:
    python scripts/mac_smoke_test.py                       # Qwen2.5-0.5B, fast
    python scripts/mac_smoke_test.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "GASP" / "src"))  # use the pinned GASP version

import torch  # noqa: E402
from gasp import GASP  # noqa: E402

CONTEXT = (
    "Paris is the capital of France and its largest city. "
    "The Lumen Institute was founded in Paris in 2011 by the chemist Adele Morand. "
    "The institute studies light-sensitive polymers and employs 240 researchers. "
    "Its main building sits on the left bank of the Seine."
)
QUERY = "What is the capital of France, and what do you know about the Lumen Institute?"
ANSWER = (
    "Paris is the capital of France. "                            # A: grounded + model knows it
    "The Lumen Institute was founded in 2011 by Adele Morand. "   # B: grounded, context-only
    "It has won three Nobel Prizes for its polymer research."     # C: unsupported
)
LABELS = ["A grounded, already known", "B grounded, context-only", "C unsupported"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = ap.parse_args()

    if torch.cuda.is_available():
        device, dtype = "cuda", "float16"
    elif torch.backends.mps.is_available():
        device, dtype = "mps", "float32"   # float32 is the safest choice on MPS
    else:
        device, dtype = "cpu", "float32"
    print(f"Model {args.model} on {device} ({dtype})\n")

    det = GASP(args.model, k_chunks=4, device=device, dtype=dtype)
    result = det.detect(context=CONTEXT, answer=ANSWER, query=QUERY)

    print(f"{'sensitivity':>11} {'gap':>7}  sentence")
    for label, s in zip(LABELS, result):
        print(f"{s.sensitivity:+11.3f} {s.features['gap']:+7.3f}  [{label}] {s.text}")
    print("\nHigher = depends more on the context (more likely grounded).")
    print("Watch sentence A: it is grounded, but may score low because the model already knows it.")


if __name__ == "__main__":
    main()
