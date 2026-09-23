# grounding-hybrid

Prior-aware hybrid detection of RAG hallucinations with small open models.

**Goal.** Context-removal detectors such as GASP fail when the scorer model already knows the
answer (short-answer QA). We combine signed context sensitivity, the model's no-context prior,
and evidence-reading attention, gated by the prior, and compare against GASP, Lookback Lens,
ReDeEP and perplexity under one leakage-clean protocol. All signals are extracted from one
shared full-context pass plus GASP's perturbation passes.

## Structure

```
configs/reproduce.yaml     settings, pinned GASP commit, published reference numbers
scripts/env_check.py       prints the device (cuda / mps / cpu) and versions
scripts/setup_gasp.py      fetches GASP (pinned) + TofuEval label files into third_party/
scripts/reproduce_gasp.py  runs GASP's own pipeline for each model x dataset (resumable)
scripts/mac_smoke_test.py  runs GASP on one toy example; shows the known-answer failure
kaggle/kaggle_runner.py    paste into one Kaggle cell to run everything on a T4
src/grounding_hybrid/      our shared feature extractor (from week 2)
```

`third_party/` and `results/` are not committed; scripts recreate them.

## Workflow: Mac -> GitHub -> Kaggle -> Mac

1. **Edit and test on the Mac**, then push:
   ```bash
   git add -A && git commit -m "describe the change" && git push
   ```
2. **Run on Kaggle**: paste `kaggle/kaggle_runner.py` into a notebook cell and run it.
   It clones the latest commit, so every run uses exactly the code you pushed.
3. **Download results** from the notebook's Output tab to the Mac for analysis.

Final reported numbers always come from Kaggle runs; the Mac is for development and analysis.

## One-time Mac setup

```bash
# Apple Silicon Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install torch
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1     # add to ~/.zshrc to make it permanent

python scripts/env_check.py              # should say: Apple MPS available
python scripts/setup_gasp.py             # fetch GASP + TofuEval
python scripts/mac_smoke_test.py         # toy example with Qwen2.5-0.5B
```

## Week 1 checklist

- [ ] Mac: environment works, `mac_smoke_test.py` prints three sensitivity scores
- [ ] Kaggle: `MODE = "smoke"` run completes
- [ ] Kaggle: `MODE = "full"` run completes for both models on RAGTruth
- [ ] RAGTruth span AUCs within about +/-0.02 of `reference_ragtruth_span_auc` in the config
- [ ] RAGBench and TofuEval runs complete for Qwen2.5-1.5B (our target regime is RAGBench)

## Credits

Builds on GASP (Bouke, 2026, arXiv:2607.04223, MIT license), fetched unmodified at a pinned commit.
