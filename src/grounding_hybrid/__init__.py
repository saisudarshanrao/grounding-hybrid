"""Prior-aware hybrid detection of RAG hallucinations (shared single-pass feature pipeline).

Week 1 reproduces the GASP baseline (see scripts/). Week 2 adds the shared extractor:
gasp_bridge (GASP's cases and analysis protocol, used unmodified) and extractor (one
full-context pass per case, attention reduced in hooks; Lookback Lens ratios first).
"""
__version__ = "0.0.2"
