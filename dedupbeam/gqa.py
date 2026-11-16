"""A GQA model (Qwen2.5) with the same interface as dedupbeam.opt.OPT, so PerBeamEngine runs it unchanged.

The engines treat E as "numbers per token per beam per layer for K (or V)", so here E = KV heads x head dim
(2 x 128 = 256), not the hidden size (2048). All three modes (ingpu, baseline, lbs) then work as-is.

Differences from OPT:
- RoPE: q and k are rotated before k is written to the KV. cos/sin come from the HF model's own rotary_emb for
  positions 0..max_pos-1.
- GQA: 16 query heads share 2 K/V heads (8 query heads per KV head). Attention runs bmm directly on the
  [token, beam, 2x128] KV, grouping query heads the way HF repeat_kv does (query head h uses KV head h // 8);
  the KV is never repeated.
- RMSNorm, SwiGLU MLP, biases on q/k/v but not on o_proj: these call the HF modules directly.
"""

import torch

from .common import DEV, read_prompt_ids

QWEN = "Qwen/Qwen2.5-3B"


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


class QwenGQA:
    noise = "token"  # vocab of 151,936: Gumbel noise is drawn per token (see PerBeamEngine._gumbel)

    def __init__(self, dtype=torch.float16, name=QWEN, max_pos=4096):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.dtype = dtype
        self.elem = torch.empty(0, dtype=dtype).element_size()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.hf = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(DEV).eval()
        cfg = self.hf.config
        self.mdl = self.hf.model
        self.layers = list(self.mdl.layers)
        self.L = cfg.num_hidden_layers
        self.hidden = cfg.hidden_size
        self.H = cfg.num_attention_heads
        self.KVH = cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or self.hidden // self.H
        self.G = self.H // self.KVH  # query heads per KV head
        self.E = self.KVH * self.D  # KV width
        self.V = self.hf.lm_head.out_features
        self.scale = self.D**-0.5
        self.plane_tok = self.E * self.elem
        self.layer_tok = 2 * self.plane_tok
        self.beam_tok = self.L * self.layer_tok
        self.n_params = sum(p.numel() for p in self.hf.parameters())
        with torch.inference_mode():
            pos = torch.arange(max_pos, device=DEV)[None]
            dummy = torch.zeros(1, 1, self.hidden, dtype=dtype, device=DEV)
            cos, sin = self.mdl.rotary_emb(dummy, pos)
        self.cos, self.sin = (
            cos[0],
            sin[0],
        )  # [max_pos, D], the same values the HF forward uses

    def config_summary(self):
        cfg = self.hf.config
        return dict(
            model=self.name,
            dtype=str(self.dtype).replace("torch.", ""),
            n_params=self.n_params,
            num_hidden_layers=self.L,
            hidden_size=self.hidden,
            num_attention_heads=self.H,
            num_key_value_heads=self.KVH,
            head_dim=self.D,
            intermediate_size=cfg.intermediate_size,
            vocab_size=self.V,
            rope_theta=getattr(cfg, "rope_theta", None),
            kv_bytes_per_token_per_layer=self.layer_tok,
            kv_bytes_per_token_all_layers=self.beam_tok,
        )

    def embed(self, tokens, pos):
        return self.mdl.embed_tokens(tokens)  # position enters through RoPE in layer()

    def layer(self, l, h, K, V, s):
        """Layer l. K, V: [capacity, B, KVH*D]; rows [0, s) are past tokens, this token (position s) goes to row s."""
        lay = self.layers[l]
        a = lay.self_attn
        B, H, KVH, D, G = h.shape[0], self.H, self.KVH, self.D, self.G
        x = lay.input_layernorm(h)
        q = a.q_proj(x).view(B, H, D)
        k = a.k_proj(x).view(B, KVH, D)
        v = a.v_proj(x)
        cos, sin = self.cos[s], self.sin[s]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        K[s] = k.reshape(B, KVH * D)
        V[s] = v
        n = s + 1
        Kt = (
            K[:n].view(n, B * KVH, D).permute(1, 2, 0)
        )  # [B*KVH, D, n], a view, no copy
        Vv = V[:n].view(n, B * KVH, D).permute(1, 0, 2)  # [B*KVH, n, D]
        w = torch.bmm(q.view(B * KVH, G, D), Kt) * self.scale  # [B*KVH, G, n]
        w = torch.softmax(w, dim=-1, dtype=torch.float32).to(h.dtype)
        o = torch.bmm(w, Vv).view(B, H * D)
        h = h + a.o_proj(o)
        return h + lay.mlp(lay.post_attention_layernorm(h))

    def head(self, h):
        return self.hf.lm_head(self.mdl.norm(h))


def load_prompt(model, n_tokens, path=None):
    """The same prompt text, cut with this model's tokenizer to the first n_tokens tokens (Qwen adds no BOS)."""
    return read_prompt_ids(model.tok, n_tokens, path)
