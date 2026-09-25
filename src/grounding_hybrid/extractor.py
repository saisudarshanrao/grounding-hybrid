"""Shared single-pass feature extractor.

One full-context forward pass per case, on exactly the token sequence GASP scored (same
prompt, same context/answer truncation, same sentence-to-token map), so every feature lines
up with a row of GASP's sentence.csv. Attention is reduced inside forward hooks and the
weights are dropped at once; no attention matrix or hidden state is kept.

Per answer sentence (its tokens = answer tokens whose start offset lies inside it, as GASP):
  lookback      [layers, heads] Lookback Lens ratio A_ctx / (A_ctx + A_new), averaged over
                the sentence's tokens (Chuang et al., 2024). For the answer token at position
                p, A_ctx is its mean attention over the prompt (positions < P) and A_new its
                mean attention over the answer so far (positions P..p, itself included).
  logprob_full  mean full-context token log-prob. Must match -mean_surprisal in GASP's
                sentence.csv: this is the alignment check.

ReDeEP scores (redeep=True; Sun et al., ICLR 2025), averaged over the sentence's tokens:
  ecs           [layers, heads] External Context Score: cosine similarity between the answer
                token's last-layer hidden state and the mean last-layer hidden state of the
                top-10% retrieved-context tokens that head attends to most (the context span
                only, not the question or template).
  pks           [layers] Parametric Knowledge Score: Jensen-Shannon divergence between the
                logit-lens vocabulary distributions (final norm + unembedding) of the residual
                stream just before and just after the layer's FFN.
Head and layer selection (ReDeEP's copying heads / knowledge FFNs) happens later, on training
folds only, so every head and layer is kept here.

Frequency-aware attention (freq=True; Qi et al., 2026, arXiv 2602.18145, as in the authors' code):
  freq          [layers, heads, 2] for each answer token's attention row, separately over the prompt
                (positions < P) and over the answer so far (P..p): FFT, keep the bins with
                |fftfreq(N)| >= f_cutoff (0.45, the authors' default), inverse FFT (real part), L2
                norm; averaged over the sentence's tokens. [..., 0] = context part, [..., 1] = answer part.

Coverage-aware reading (chunked=True): the first window is exactly GASP's retained context, so
`lookback` is unchanged; when the context is longer than max_ctx_tokens, further windows of the
same size (overlapping by `overlap` tokens) cover the rest, one pass each with the same prompt
and answer. Per sentence, lookback_max / lookback_mean combine the windows elementwise, so
evidence beyond GASP's window is read too. Contexts that fit get identical values in all three.
ReDeEP and frequency features come from the first window only (the baselines see what GASP sees).

Runs in float32 by default on every device. Eager attention (needed to read the weights)
overflows in float16 for Qwen2.5: every feature came out NaN on a T4, while GASP's own fp16
passes use SDPA and are unaffected.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer: "   # GASP's prompt, no chat template
CTX_START = len("Context:\n")                                  # first context character in PROMPT


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class SharedExtractor:
    def __init__(self, model_id, device=None, dtype="float32", max_ctx_tokens=1800, max_ans_tokens=256,
                 overlap=256, redeep=False, ecs_topk=0.1, freq=False, f_cutoff=0.45):
        self.device = device or default_device()
        self.dtype = dtype
        self.model_id = model_id
        self.max_ctx, self.max_ans, self.overlap = max_ctx_tokens, max_ans_tokens, overlap
        self.redeep, self.ecs_topk = redeep, ecs_topk
        self.freq, self.f_cutoff, self._freq_on, self._fq_ctx, self._fq_new = freq, f_cutoff, False, None, None
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), attn_implementation="eager").to(self.device).eval()
        layers = self.model.model.layers
        self.n_layers, self.n_heads = len(layers), self.model.config.num_attention_heads
        self._P, self._A, self._ctx, self._lookback = None, None, None, None
        self._ecs_idx, self._mid, self._pks, self._xl, self._in_lens = None, None, None, None, False
        for i, layer in enumerate(layers):
            layer.self_attn.register_forward_hook(self._attention_hook(i))
            if redeep:
                layer.post_attention_layernorm.register_forward_pre_hook(self._mid_hook(i))
                layer.register_forward_hook(self._ffn_hook(i))
        if redeep:
            self.model.model.norm.register_forward_hook(self._final_hook)

    def _attention_hook(self, layer_idx):
        def hook(module, args, output):
            attn = output[1]                           # batch x heads x T x T, or None
            if attn is not None and self._P is not None:
                P = self._P
                w = attn[0, :, P:, :].float()          # heads x A x T: rows of answer tokens
                a_ctx = w[..., :P].mean(-1)
                # causal mask: row P+i is zero past column P+i, so this sums columns P..P+i
                a_new = w[..., P:].sum(-1) / torch.arange(1, w.shape[1] + 1, device=w.device)
                self._lookback[layer_idx] = a_ctx / (a_ctx + a_new)
                if self._freq_on:                      # frequency-aware attention (window 1)
                    self._fq_ctx[layer_idx] = self._hf_norm(w[..., :P])     # heads x A
                    # copy: a view would keep the whole T x T attention of every layer alive (GPU OOM)
                    self._fq_new[layer_idx] = w[..., P:].clone()            # heads x A x A (causal)
                if self._ctx is not None:              # ReDeEP: top-k% attended context tokens
                    c0, c1 = self._ctx
                    k = max(1, math.ceil(self.ecs_topk * (c1 - c0)))
                    self._ecs_idx[layer_idx] = w[..., c0:c1].topk(k, dim=-1).indices + c0
            return (output[0], None) + tuple(output[2:])   # drop the weights immediately
        return hook

    def _hf_norm(self, x):
        """L2 norm of the high-pass part (|fftfreq| >= f_cutoff) of x along its last axis."""
        n = x.shape[-1]
        keep = torch.fft.fftfreq(n, device=x.device).abs() >= self.f_cutoff
        return torch.fft.ifft(torch.fft.fft(x, dim=-1) * keep, dim=-1).real.norm(dim=-1)

    def _mid_hook(self, layer_idx):
        def hook(module, args):
            if self._ctx is not None:                  # residual stream before the FFN
                self._mid[layer_idx] = args[0][0, self._P:self._P + self._A].float()
        return hook

    def _ffn_hook(self, layer_idx):
        def hook(module, args, output):
            if self._ctx is not None:                  # residual stream after the FFN
                lp = self._lens(self._mid[layer_idx])
                lq = self._lens(output[0][0, self._P:self._P + self._A].float())
                lm = torch.logaddexp(lp, lq) - math.log(2.0)
                self._pks[layer_idx] = 0.5 * ((lp.exp() * (lp - lm)).sum(-1) + (lq.exp() * (lq - lm)).sum(-1))
                self._mid[layer_idx] = None
        return hook

    def _final_hook(self, module, args, output):
        if self._ctx is not None and not self._in_lens:
            self._xl = output[0].float()               # last-layer hidden states (after final norm), T x d

    def _lens(self, x):
        """Logit lens: log-softmax of unembed(final_norm(x)), rows of x = answer positions."""
        self._in_lens = True
        try:
            h = self.model.model.norm(x.to(self.model.dtype))
        finally:
            self._in_lens = False
        return torch.log_softmax(self.model.lm_head(h).float(), dim=-1)

    def windows(self, case):
        """Char spans of the context windows: the first is GASP's retained context."""
        coffs = self.tok(case.context, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        spans, start = [], 0
        while True:
            end = min(start + self.max_ctx, len(coffs))
            spans.append((0 if start == 0 else coffs[start][0], coffs[end - 1][1]))
            if end >= len(coffs):
                return spans
            start = end - self.overlap

    def encode(self, case):
        """Prompt ids, answer ids and sentence -> answer-token map, exactly as GASP's LM.score."""
        enc = self.tok(case.answer, return_offsets_mapping=True, add_special_tokens=False)
        aid = enc["input_ids"][: self.max_ans]
        offs = enc["offset_mapping"][: self.max_ans]
        if len(aid) < 8:
            return None
        coffs = self.tok(case.context, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        keep = self.max_ctx
        ctx_ret = coffs[keep - 1][1] if len(coffs) >= keep and keep > 0 else len(case.context)
        pid = self.tok(PROMPT.format(ctx=case.context[:ctx_ret], query=case.query)).input_ids
        sents = []
        for j, (s, e) in enumerate(case.sent_spans):
            tk = [t for t in range(len(aid)) if s <= offs[t][0] < e]
            if len(tk) >= 3:                           # GASP skips shorter sentences
                sents.append((j, tk))
        return pid, aid, sents

    def context_range(self, ctx, query, pid):
        """Token positions [c0, c1) of the retrieved context inside the prompt (ReDeEP's ECS)."""
        enc = self.tok(PROMPT.format(ctx=ctx, query=query), return_offsets_mapping=True)
        assert enc.input_ids == pid, "prompt tokenization changed"
        pos = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > s and CTX_START <= s < CTX_START + len(ctx)]
        return (pos[0], pos[-1] + 1) if pos else None

    def _pass(self, pid, aid, ctx_range=None, first=True):
        """One forward pass: answer-token log-probs and Lookback ratios (layers x heads x A);
        with ctx_range (ReDeEP) also ECS (layers x heads x A) and PKS (layers x A); with freq on the
        first window, frequency features (layers x heads x A x 2)."""
        P, A = len(pid), len(aid)
        ids = torch.tensor([pid + aid], device=self.device)
        self._P, self._A, self._lookback = P, A, [None] * self.n_layers
        self._freq_on = self.freq and first
        if self._freq_on:
            self._fq_ctx, self._fq_new = [None] * self.n_layers, [None] * self.n_layers
        self._ctx = ctx_range if self.redeep else None
        if self._ctx is not None:
            self._ecs_idx, self._mid, self._pks = [None] * self.n_layers, [None] * self.n_layers, [None] * self.n_layers
        try:
            logits = self.model(ids, output_attentions=True, use_cache=False).logits[0, P - 1:P - 1 + A]
        finally:
            self._P = None
        lp = torch.log_softmax(logits.float(), dim=-1)
        tlp = lp[torch.arange(A, device=self.device), ids[0, P:]].cpu().numpy()
        lb = torch.stack(self._lookback).cpu().numpy()
        self._lookback = None
        fq = None
        if self._freq_on:
            new = torch.stack(self._fq_new)                                # L x H x A x A
            fq_new = torch.stack([self._hf_norm(new[:, :, i, :i + 1]) for i in range(A)], dim=-1)
            fq = torch.stack([torch.stack(self._fq_ctx), fq_new], dim=-1).cpu().numpy()   # L x H x A x 2
            self._fq_ctx, self._fq_new, self._freq_on = None, None, False
        if self._ctx is None:
            return tlp, lb, None, None, fq
        X = self._xl                                   # T x d
        xa = F.normalize(X[P:P + A], dim=-1)           # A x d
        ecs = []
        for idx in self._ecs_idx:                      # heads x A x k
            H, _, k = idx.shape
            M = torch.zeros(H * A, X.shape[0], device=X.device)
            M.scatter_(1, idx.reshape(H * A, k), 1.0 / k)
            E = (M @ X).reshape(H, A, -1)              # mean hidden state of the attended tokens
            ecs.append((F.normalize(E, dim=-1) * xa).sum(-1))
        ecs = torch.stack(ecs).cpu().numpy()
        pks = torch.stack(self._pks).cpu().numpy()
        self._ctx, self._ecs_idx, self._mid, self._pks, self._xl = None, None, None, None, None
        return tlp, lb, ecs, pks, fq

    @torch.no_grad()
    def extract(self, case, chunked=False):
        """List of per-sentence feature dicts for one case (empty if GASP skipped it)."""
        enc = self.encode(case)
        if enc is None:
            return []
        pid, aid, sents = enc
        rng = None
        if self.redeep:
            coffs = self.tok(case.context, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            ctx_ret = coffs[self.max_ctx - 1][1] if len(coffs) >= self.max_ctx else len(case.context)
            rng = self.context_range(case.context[:ctx_ret], case.query, pid)
        tlp, lb, ecs, pks, fq = self._pass(pid, aid, rng)
        out = [dict(sent_idx=j, n_tok=len(tk), logprob_full=float(tlp[tk].mean()),
                    lookback=lb[:, :, tk].mean(-1)) for j, tk in sents]
        if fq is not None:
            for r, (_, tk) in zip(out, sents):
                r.update(freq=fq[:, :, tk, :].mean(2))
        if ecs is not None:
            for r, (_, tk) in zip(out, sents):
                r.update(ecs=ecs[:, :, tk].mean(-1), pks=pks[:, tk].mean(-1))
        elif self.redeep:                              # no context tokens: ReDeEP is undefined
            for r in out:
                r.update(ecs=np.full((self.n_layers, self.n_heads), np.nan), pks=np.full(self.n_layers, np.nan))
        if chunked:
            per_window = [[r["lookback"]] for r in out]
            spans = self.windows(case)
            for cs, ce in spans[1:]:
                pw = self.tok(PROMPT.format(ctx=case.context[cs:ce], query=case.query)).input_ids
                _, lbw, _, _, _ = self._pass(pw, aid, first=False)
                for k, (_, tk) in enumerate(sents):
                    per_window[k].append(lbw[:, :, tk].mean(-1))
            for r, ws in zip(out, per_window):
                ws = np.stack(ws)
                r.update(lookback_max=ws.max(0), lookback_mean=ws.mean(0), n_windows=len(spans))
        return out

    @torch.no_grad()
    def extract_placebo(self, case, donor):
        """Step E2 placebo reading: window 1 is the case's own (as extract), windows 2..K are the DONOR's windows
        2..K (K = the case's own number of windows; the donor's are reused in order if it has fewer), each read with
        the case's question and answer. With donor = case this is exactly B (extract with chunked=True)."""
        enc = self.encode(case)
        if enc is None:
            return []
        pid, aid, sents = enc
        tlp, lb, _, _, _ = self._pass(pid, aid)
        per_window = [[lb[:, :, tk].mean(-1)] for _, tk in sents]
        k_own = len(self.windows(case))
        dspans = self.windows(donor)[1:] if k_own > 1 else []
        for i in range(k_own - 1):
            cs, ce = dspans[i % len(dspans)]
            pw = self.tok(PROMPT.format(ctx=donor.context[cs:ce], query=case.query)).input_ids
            _, lbw, _, _, _ = self._pass(pw, aid, first=False)
            for k, (_, tk) in enumerate(sents):
                per_window[k].append(lbw[:, :, tk].mean(-1))
        return [dict(sent_idx=j, n_tok=len(tk), logprob_full=float(tlp[tk].mean()), lookback=ws[0],
                     lookback_placebo_max=np.stack(ws).max(0), n_windows=k_own)
                for (j, tk), ws in zip(sents, per_window)]
