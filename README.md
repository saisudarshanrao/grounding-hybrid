# grounding-hybrid

Coverage-aware attention reading for detecting RAG hallucinations with small open models.

**Question.** Detectors that read a scorer model's internals (context removal like GASP, attention reading like
Lookback Lens) only see the part of the retrieved context that fits in the scorer's window. When the evidence lies
beyond it, they judge a faithful sentence against the wrong text. We measure how much this costs and test a simple
fix, under one leakage-clean protocol (GASP's source-level split) with 1.5-1.7B scorers on free T4 GPUs.

**Method (B).** Read the whole context in windows of 1800 tokens (overlap 256; window 1 is exactly GASP's
retained context), with the answer held fixed. For each (layer, head), compute Lookback Lens' ratio
A_ctx / (A_ctx + A_new) per window and keep the maximum over windows. A logistic regression on these features
(C tuned by grouped 5-fold CV on the dev split) gives the sentence score.

**Compared under the same protocol.** Perplexity + length; GASP (its own classifier); Lookback Lens; ReDeEP
(training-free and regression forms); frequency-aware attention (arXiv 2602.18145). Scorers: Qwen2.5-1.5B-Instruct
and SmolLM2-1.7B-Instruct. Datasets: RAGTruth, TofuEval, RAGBench (GASP's samples), TechQA (600 cases) and
ExpertQA-long (466 cases, contexts of at least 9000 characters) from RAGBench, and a controlled truncation study
(512-token windows on RAGTruth and TofuEval, the full view as upper bound).

**Protocol.** Every design choice was made on GASP's dev split (grouped CV by source). The method, baselines,
datasets and metrics were then frozen and every detector was scored once on GASP's test split
(`scripts/test_look.py`). Differences use GASP's paired source-level bootstrap (2000 resamples).

## Structure

```
configs/reproduce.yaml       settings, pinned GASP commit, published reference numbers
kaggle/kaggle_runner.py      one Kaggle cell; MODE picks the run (see "Reproduce" below)
src/grounding_hybrid/
  gasp_bridge.py             GASP's protocol, imported unmodified (split, classifier, bootstrap)
  extractor.py               one fp32 pass per case (or per window) with attention hooks: Lookback ratios,
                             windowed max/mean, ReDeEP ECS/PKS, frequency-aware attention
  signals.py                 joins features to GASP's sentence.csv by (case_id, sent_idx)
  gating.py                  grouped CV folds; prior-gating variants (tested and rejected)
  longpass.py                baseline L: one long pass with memory-efficient attention, recomputing only the
                             answer rows of attention (for contexts that do not fit eager attention)

scripts/ -- pipeline
  env_check.py               device and versions
  setup_gasp.py              fetches GASP (pinned) + TofuEval labels into third_party/
  reproduce_gasp.py          GASP's own pipeline per model x dataset (resumable)
  gasp_longctx.py            GASP's pipeline on one long-context RAGBench domain (TechQA, ExpertQA-long)
  extract_features.py        features for one GASP run (--chunked, --redeep, --freq, --max_ctx_tokens)
  run_features.py            all models x datasets, one GPU per model
  extract_long.py            baseline L features for one GASP run (one pass over as much context as the model allows)
  longpass_check.py          checks for baseline L: recomputed answer rows = eager weights; window 1 = frozen extractor

scripts/ -- the test look, figures, bookkeeping
  test_look.py               every frozen detector, fit on dev, scored once on test (--eval_on devhalf = dry run)
  make_figures.py            dev figures; --test draws the main figures from the test look's saved scores
  cost_table.py              seconds per case for every detector, from saved run timings
  provenance.py              which Kaggle run made each result file; row alignment and cross-run checks

scripts/ -- dev-only analyses (grouped CV on the dev split; never touch test)
  eval_features.py           Lookback vs GASP under GASP's split and classifier
  eval_redeep.py             ReDeEP vs Lookback vs GASP
  analyze_prior.py           detector AUC by the no-context prior (the "already known" premise)
  eval_gating.py             prior-gating variants (rejected)
  analyze_coverage.py        windowed reading vs window 1 (TechQA, controlled truncation)
  coverage_checks.py         pooled tests and a fixed-classifier mechanism check for windowed reading
  evidence_position.py       B's gain by where RAGBench's annotated evidence lies
  b2_check.py                variant B2 = [window 1, max] (rejected)
  freq_check.py              frequency-aware attention vs Lookback
  rq2_where.py               every detector by dataset, domain, prior and position
  robustness_cv.py           stability of the key comparisons over CV fold assignments
  ablation_layers.py         which layers and heads carry the Lookback signal
  e1_long_pass.py            B vs one long single pass (baseline L) on the long sets (--eval_on dev; test once after)
  mac_smoke_test.py          GASP on one toy example (Mac check)
```

`third_party/` and `results/` are not committed; the scripts recreate them.

## Reproduce

Heavy runs (GASP, feature extraction) run on Kaggle (GPU T4 x2, Internet on); analysis runs anywhere.
Paste `kaggle/kaggle_runner.py` into one notebook cell (or a short loader that fetches it from this repository),
set `MODE`, run the `-smoke` variant first, then the full one via Save Version > Save & Run All.
The runner clones this repository, so every run uses exactly the pushed code.

| MODE | Produces | Input needed |
|---|---|---|
| `full` | GASP on RAGTruth, TofuEval, RAGBench, both scorers (`results/gasp_repro/canon_results`) | - |
| `features` | Lookback features (`features.npz`) | `full` output |
| `chunked` | windowed Lookback on RAGBench (`features_chunked.npz`) | `full` output |
| `trunc` | controlled truncation, 512/128 windows (`features_chunked_ctx512.npz`) | `full` output |
| `redeep` | ReDeEP ECS/PKS + Lookback (`features_redeep.npz`) | `full` output |
| `freq` | frequency-aware attention + Lookback (`features_freq.npz`) | `full` output |
| `techqa` | GASP + windowed Lookback + ReDeEP on TechQA (`features_chunked_redeep.npz`) | - |
| `expertqa` | the same on ExpertQA-long | - |
| `longfreq` | frequency-aware attention on TechQA + ExpertQA-long | - |
| `longpass` | baseline L (one long pass) on TechQA + ExpertQA-long; the smoke runs its checks first | - |

"`full` output" = the `results/gasp_repro/canon_results` folder of a `full` run, attached to the notebook as input
(for example as a private Kaggle dataset); the runner finds it automatically. GASP's sampling and scoring are
deterministic: reruns give byte-identical `sentence.csv` files.

Download each run's `results/` into this folder, then:

```bash
python scripts/provenance.py                               # every file lines up with GASP's rows
python -W ignore scripts/test_look.py --eval_on devhalf    # dry run on dev halves (never touches test)
python -W ignore scripts/test_look.py --eval_on test       # the single test look -> results/test/
python -W ignore scripts/make_figures.py --test            # main figures from results/test/
python -W ignore scripts/make_figures.py                   # dev diagnostics (layers, prior)
python scripts/cost_table.py                               # compute cost per detector
```

## Setup (Mac, for development and analysis)

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv
echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> .venv/bin/activate
source .venv/bin/activate
pip install torch
pip install -r requirements.txt

python scripts/env_check.py              # device and versions
python scripts/setup_gasp.py             # fetch GASP + TofuEval
python scripts/mac_smoke_test.py         # toy example with Qwen2.5-0.5B
```

Quick feature check on a few cases (alignment must show `covered == gasp_rows`, log-prob difference ~0.001):

```bash
python scripts/extract_features.py --canon_dir results/gasp_repro/canon_results/Qwen2.5-1.5B-Instruct_ragtruth_K5 \
    --model Qwen/Qwen2.5-1.5B-Instruct --chunked --max_cases 5
```

## Credits

Builds on GASP (Bouke, 2026, arXiv:2607.04223, MIT license), fetched unmodified at a pinned commit.
