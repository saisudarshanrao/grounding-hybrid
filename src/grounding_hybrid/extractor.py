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
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer: "   # GASP's prompt, no chat template


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class SharedExtractor:
    def __init__(self, model_id, device=None, max_ctx_tokens=1800, max_ans_tokens=256):
        self.device = device or default_device()
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32   # as GASP
        self.model_id = model_id
        self.max_ctx, self.max_ans = max_ctx_tokens, max_ans_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="eager").to(self.device).eval()
        layers = self.model.model.layers
        self.n_layers, self.n_heads = len(layers), self.model.config.num_attention_heads
        self._P, self._lookback = None, None
        for i, layer in enumerate(layers):
            layer.self_attn.register_forward_hook(self._attention_hook(i))

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
            return (output[0], None) + tuple(output[2:])   # drop the weights immediately
        return hook

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

    @torch.no_grad()
    def extract(self, case):
        """List of per-sentence feature dicts for one case (empty if GASP skipped it)."""
        enc = self.encode(case)
        if enc is None:
            return []
        pid, aid, sents = enc
        P, A = len(pid), len(aid)
        ids = torch.tensor([pid + aid], device=self.device)
        self._P, self._lookback = P, [None] * self.n_layers
        try:
            logits = self.model(ids, output_attentions=True, use_cache=False).logits[0, P - 1:P - 1 + A]
        finally:
            self._P = None
        lp = torch.log_softmax(logits.float(), dim=-1)
        tlp = lp[torch.arange(A, device=self.device), ids[0, P:]].cpu().numpy()
        lb = torch.stack(self._lookback).cpu().numpy()     # layers x heads x A
        self._lookback = None
        return [dict(sent_idx=j, n_tok=len(tk), logprob_full=float(tlp[tk].mean()),
                     lookback=lb[:, :, tk].mean(-1)) for j, tk in sents]
