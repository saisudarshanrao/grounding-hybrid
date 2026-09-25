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
scripts/analyze_prior.py   dev-only check: GASP / Lookback AUC by prior (S2) bins and domain
scripts/eval_gating.py     dev-only grouped CV of gating variants (span or response level)
scripts/analyze_coverage.py  dev-only: does coverage-aware (windowed) reading fix truncation?
scripts/coverage_checks.py   dev-only: pooled test + fixed-classifier mechanism check for windowed reading
scripts/eval_redeep.py     ReDeEP baseline vs Lookback vs GASP (dev CV, or one --test look)
scripts/gasp_longctx.py    GASP's unmodified pipeline on one long-context RAGBench domain (TechQA, all splits)
scripts/cost_table.py      seconds per case for every detector, from the saved run timings
scripts/evidence_position.py  dev-only: B's gain by where the evidence lies (RAGBench annotations)
scripts/b2_check.py        dev-only: pre-registered test of variant B2 (rejected)
scripts/freq_check.py      dev-only: frequency-aware attention baseline vs Lookback (pre-registered rule)
kaggle/kaggle_runner.py    paste into one Kaggle cell; MODE picks the run (GASP, features, chunked, trunc, redeep)
src/grounding_hybrid/      gasp_bridge.py (GASP protocol, unmodified), extractor.py (hooks),
                           signals.py (S1 signed sensitivity, S2 prior, S3 Lookback),
                           gating.py (combined / soft / hard gates, dev-only CV)
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
- [x] Kaggle: `MODE = "features"` completes for both models x 3 datasets
- [x] Lookback Lens span/response AUC vs GASP on RAGTruth, TofuEval, RAGBench (Lookback wins everywhere)
- [x] Dev-only gating comparison: no gate beats Lookback alone or the plain combination
- [x] Coverage-aware reading (`MODE = "chunked"`) on RAGBench: TechQA direction right but n.s. (31 dev sources)
- [x] Controlled truncation (`MODE = "trunc"`, 512-token windows, RAGTruth + TofuEval): max over windows
      recovers ~70% of the truncation loss (pooled, significant); fixed-classifier check confirms (dev only)

## Week 3: ReDeEP baseline

ReDeEP's ECS (per head) and PKS (per layer) come from the same single pass (`--redeep`); head/layer
selection and weights are chosen on training folds only.

```bash
python scripts/extract_features.py --canon_dir results/gasp_repro/canon_results/Qwen2.5-1.5B-Instruct_ragtruth_K5 \
    --model Qwen/Qwen2.5-1.5B-Instruct --redeep --max_cases 3       # quick Mac check
python -W ignore scripts/eval_redeep.py                              # dev CV, after the Kaggle run
```

On Kaggle: `MODE = "redeep-smoke"`, then `"redeep"`; download the output and evaluate on the Mac.

- [x] Kaggle: `MODE = "redeep"` completes for both models x 3 datasets (lookback equals Week 2's exactly)
- [x] Dev CV: Lookback > ReDeEP [cv] >= GASP ~ training-free ReDeEP (test look after the method freeze)

## Week 3b: coverage-aware reading on real truncation (TechQA)

GASP's RAGBench sample holds only 49 TechQA cases. `--datasets techqa` draws GASP's balanced
sample (300/class, seed 42) from TechQA alone, pooled over all RAGBench splits: 600 cases, 502
sources, 97% of contexts longer than the 1800-token window. GASP's own code scores it; then one
pass extracts coverage-aware Lookback + ReDeEP features.

```bash
python scripts/reproduce_gasp.py --models Qwen/Qwen2.5-0.5B-Instruct --datasets techqa --max_cases 3   # Mac check
python -W ignore scripts/analyze_coverage.py --suffix chunked_redeep --datasets techqa             # after Kaggle
```

On Kaggle: `MODE = "techqa-smoke"`, then `"techqa"` (no input needed: GASP runs inside).

- [x] Kaggle: `MODE = "techqa"` completes (GASP + features, both models; ~3 h)
- [x] Dev: max over windows beats window 1 by +0.06-0.07 (significant); best detector on TechQA

## Step 5: a second long-context set (ExpertQA-long)

RAGBench ExpertQA, all splits, contexts of at least 9000 characters (100% longer than the window for
both scorers): 466 cases. `--datasets expertqalong`; Kaggle `MODE = "expertqa-smoke"`, then `"expertqa"`.

- [x] Kaggle: `MODE = "expertqa"` completes
- [x] Dev: B vs window 1 on ExpertQA-long: positive but not significant (rule not met); B's gain is large where
      the evidence lies beyond window 1 on both sets (`scripts/evidence_position.py`)

## Step 7c: frequency-aware attention baseline

Frequency-aware attention (arXiv 2602.18145, authors' cutoff 0.45) from the same single pass (`--freq`); checked
against an authors-style reference implementation (difference 0.0).
On Kaggle: `MODE = "freq-smoke"`, then `"freq"`; then `python -W ignore scripts/freq_check.py` on the Mac.

- [x] Kaggle: `MODE = "freq"` completes (Version 9)
- [x] Dev: FA vs Lookback: on par (pooled +0.009, n.s.) -> additional baseline
- [ ] Kaggle: `MODE = "longfreq"` (FA on TechQA + ExpertQA-long, so every frozen baseline covers every dataset)

## Freeze (25 Sep 2026) and the single test look

Method, baselines, datasets and metrics are frozen (CLAUDE.md "FREEZE record"). `scripts/test_look.py` scores
every frozen detector once on GASP's test split.

`python scripts/cost_table.py` gives seconds per case for every detector (RQ3 cost).

## Credits

Builds on GASP (Bouke, 2026, arXiv:2607.04223, MIT license), fetched unmodified at a pinned commit.
