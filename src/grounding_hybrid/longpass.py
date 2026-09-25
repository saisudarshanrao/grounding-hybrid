"""Long single pass (step E1, baseline L): Lookback features from ONE forward pass over as much of the context as
the scorer's position limit allows, keeping only the answer rows of attention in memory.

The frozen extractor (extractor.py) reads full attention weights: that needs eager attention and heads x T x T
memory per layer, so a 20k-token pass would not fit a 16 GB T4. Here the model runs with PyTorch's memory-efficient
SDPA attention. The library's apply_rotary_pos_emb is wrapped so the layer being computed leaves its post-RoPE
queries (answer rows only) and keys in TAP; a forward hook on the attention module then recomputes
softmax(Q K^T / sqrt(d)) for the answer rows only, with the causal mask and an fp32 softmax, exactly as eager
attention does. Memory per layer: heads x answer x T. The model's own computation is unchanged.

The features follow the frozen Lookback definition: for the answer token at position t = P + i, A_ctx is its mean
attention over the prompt (positions < P) and A_new its mean attention over the answer so far (P..t, itself
included); ratio = A_ctx / (A_ctx + A_new); the sentence feature is the mean over the sentence's tokens. Only the
amount of context in the prompt differs from window 1.
"""
import contextlib
import math

import numpy as np
import torch
import transformers.models.llama.modeling_llama as _llama
import transformers.models.qwen2.modeling_qwen2 as _qwen2
from transformers import AutoModelForCausalLM, AutoTokenizer

from grounding_hybrid.extractor import PROMPT, default_device


class _RopeTap:
    """While P is set, keeps the post-RoPE queries (answer rows, from position P on) and keys of the layer being run."""

    def __init__(self):
        self.P, self.q, self.k = None, None, None

    def install(self):
        for mod in (_qwen2, _llama):
            if getattr(mod.apply_rotary_pos_emb, "_tapped", False):
                continue
            orig = mod.apply_rotary_pos_emb

            def wrapped(q, k, *args, _orig=orig, **kwargs):
                qe, ke = _orig(q, k, *args, **kwargs)
                if self.P is not None:
                    self.q, self.k = qe[:, :, self.P:, :], ke
                return qe, ke

            wrapped._tapped = True
            mod.apply_rotary_pos_emb = wrapped


TAP = _RopeTap()
TAP.install()


def answer_rows(module, q, k, P):
    """softmax(q k^T / sqrt(d)) for the answer rows (causal, fp32): q 1 x H x A x d, k 1 x KV x T x d -> H x A x T."""
    k = _qwen2.repeat_kv(k, module.num_key_value_groups)
    A, T = q.shape[2], k.shape[2]
    s = torch.matmul(q.float(), k.float().transpose(2, 3)) / math.sqrt(module.head_dim)
    cols = torch.arange(T, device=s.device)
    rows = P + torch.arange(A, device=s.device)
    s.masked_fill_(cols[None, :] > rows[:, None], float("-inf"))
    return torch.softmax(s, dim=-1)[0]


def lookback_ratio(w, P):
    """Lookback ratio per head and answer token from answer-row attention w (H x A x T), as extractor.py."""
    A = w.shape[1]
    a_ctx = w[..., :P].mean(-1)
    a_new = w[..., P:].sum(-1) / torch.arange(1, A + 1, device=w.device)
    return a_ctx / (a_ctx + a_new)


def _efficient_sdpa(device):
    """On CUDA, force the memory-efficient SDPA kernel (an error, not a silent T x T fallback, if it cannot run)."""
    if str(device).startswith("cuda"):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    return contextlib.nullcontext()


class LongPass:
    """Lookback features from one pass over the context cut at `limit` tokens (None: all that fits the model)."""

    def __init__(self, model_id, device=None, dtype="float32", max_ans_tokens=256, attn="sdpa"):
        self.device = device or default_device()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), attn_implementation=attn).to(self.device).eval()
        layers = self.model.model.layers
        self.n_layers, self.n_heads = len(layers), self.model.config.num_attention_heads
        self.max_pos = self.model.config.max_position_embeddings
        self.max_ans, self.attn = max_ans_tokens, attn
        self._P, self._lb, self.check_diff = None, None, None
        for i, layer in enumerate(layers):
            layer.self_attn.register_forward_hook(self._hook(i))

    def _hook(self, layer_idx):
        def hook(module, args, output):
            if self._P is None or TAP.q is None:
                return
            w = answer_rows(module, TAP.q, TAP.k, self._P)
            TAP.q, TAP.k = None, None
            if self.check_diff is not None and output[1] is not None:   # check (a): eager weights of this pass
                eager = output[1][0, :, self._P:, :].float()
                self.check_diff = max(self.check_diff, float((w - eager).abs().max()))
            self._lb[layer_idx] = lookback_ratio(w, self._P)
        return hook

    def encode(self, case, limit=None):
        """Prompt ids with the context cut at `limit` tokens (GASP's rule for window 1), answer ids, sentence map."""
        enc = self.tok(case.answer, return_offsets_mapping=True, add_special_tokens=False)
        aid, offs = enc["input_ids"][: self.max_ans], enc["offset_mapping"][: self.max_ans]
        if len(aid) < 8:
            return None
        coffs = self.tok(case.context, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        if limit is None:   # as much context as the model's positions allow
            limit = self.max_pos - len(self.tok(PROMPT.format(ctx="", query=case.query)).input_ids) - len(aid) - 8
        while True:
            ctx_ret = coffs[limit - 1][1] if len(coffs) >= limit and limit > 0 else len(case.context)
            pid = self.tok(PROMPT.format(ctx=case.context[:ctx_ret], query=case.query)).input_ids
            if len(pid) + len(aid) <= self.max_pos:
                break
            limit -= len(pid) + len(aid) - self.max_pos
        sents = []
        for j, (s, e) in enumerate(case.sent_spans):
            tk = [t for t in range(len(aid)) if s <= offs[t][0] < e]
            if len(tk) >= 3:
                sents.append((j, tk))
        info = dict(n_ctx_tokens=len(coffs), ctx_read_tokens=min(limit, len(coffs)), ctx_cut=len(coffs) > limit,
                    seq_len=len(pid) + len(aid))
        return pid, aid, sents, info

    @torch.no_grad()
    def extract(self, case, limit=None, check=False):
        """Per-sentence dicts (sent_idx, n_tok, logprob_full, lookback [layers, heads]) and the case's info."""
        enc = self.encode(case, limit)
        if enc is None:
            return [], {}
        pid, aid, sents, info = enc
        P, A = len(pid), len(aid)
        ids = torch.tensor([pid + aid], device=self.device)
        self._P, self._lb = P, [None] * self.n_layers
        self.check_diff = 0.0 if check else None
        TAP.P = P
        try:
            with _efficient_sdpa(self.device):
                out = self.model.model(input_ids=ids, use_cache=False, output_attentions=check)
        finally:
            TAP.P, self._P = None, None
        h = out.last_hidden_state[0, P - 1:P - 1 + A]           # logits for the answer positions only
        lp = torch.log_softmax(self.model.lm_head(h).float(), dim=-1)
        tlp = lp[torch.arange(A, device=self.device), ids[0, P:]].cpu().numpy()
        lb = torch.stack(self._lb).cpu().numpy()                 # layers x heads x A
        self._lb = None
        if check:
            info["answer_rows_vs_eager_absdiff_max"] = self.check_diff
        return [dict(sent_idx=j, n_tok=len(tk), logprob_full=float(tlp[tk].mean()),
                     lookback=lb[:, :, tk].mean(-1)) for j, tk in sents], info
