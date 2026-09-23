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
scripts/extract_features.py  shared single-pass features (Lookback Lens) for one GASP run
scripts/eval_features.py   scores features under GASP's exact split/classifier/bootstrap
scripts/run_features.py    all models x datasets, one GPU per model in parallel
kaggle/kaggle_runner.py    paste into one Kaggle cell; MODE picks week 1 or week 2 runs
src/grounding_hybrid/      gasp_bridge.py (GASP protocol, unmodified), extractor.py (hooks)
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
# Apple Silicon Python environment (macOS's own python3 is 3.9, too old for current torch)
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv
echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> .venv/bin/activate   # set on every activate
source .venv/bin/activate
pip install torch
pip install -r requirements.txt

python scripts/env_check.py              # should say: Apple MPS available
python scripts/setup_gasp.py             # fetch GASP + TofuEval
python scripts/mac_smoke_test.py         # toy example with Qwen2.5-0.5B
```

## Week 1 checklist

- [x] Mac: environment works, `mac_smoke_test.py` prints three sensitivity scores
- [x] Kaggle: `MODE = "smoke"` run completes
- [x] Kaggle: `MODE = "full"` run completes for both models on RAGTruth
- [x] RAGTruth span AUCs within about +/-0.02 of `reference_ragtruth_span_auc` in the config (exact match, diff 0.000)
- [x] RAGBench and TofuEval runs complete for Qwen2.5-1.5B (our target regime is RAGBench)

## Week 2: shared extractor, Lookback Lens first

One full-context forward pass per case on exactly the tokens GASP scored; attention is
reduced to Lookback ratios inside forward hooks and never stored. Features join GASP's
`sentence.csv` by `(case_id, sent_idx)`, so GASP's own source-level split is reused unchanged.

```bash
# Mac check on a few cases (alignment vs GASP must show covered == gasp_rows, diff ~0.001)
python scripts/extract_features.py --canon_dir results/gasp_repro/canon_results/Qwen2.5-1.5B-Instruct_ragtruth_K5 \
    --model Qwen/Qwen2.5-1.5B-Instruct --max_cases 5
```

On Kaggle: attach the week 1 notebook output as input, set `MODE = "features-smoke"`, then `"features"`.

- [x] Extractor aligned with GASP on the Mac: every sentence covered, log-probs within ~0.001
- [x] Evaluation reproduces GASP's week 1 numbers exactly; random features score ~0.5
- [ ] Kaggle: `MODE = "features"` completes for both models x 3 datasets
- [ ] Lookback Lens span/response AUC vs GASP on RAGTruth, TofuEval, RAGBench

## Credits

Builds on GASP (Bouke, 2026, arXiv:2607.04223, MIT license), fetched unmodified at a pinned commit.
