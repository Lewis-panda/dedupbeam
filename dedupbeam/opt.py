"""OPT decoder written against the Hugging Face weights, with an explicit per-layer KV interface.

The engines manage KV themselves (no `generate`, no HF Cache). A model class exposes:
    L, E (KV width per token per layer for K or V), V (vocab), dtype, elem, plane_tok, layer_tok, beam_tok,
    tok, embed(tokens, pos), layer(l, h, K, V, s), head(h).
`scripts/check_percopy.py` compares this forward with the HF forward.
"""

import torch

from .common import DEV, read_prompt_ids

MODEL_NAME = "facebook/opt-2.7b"


class OPT:
    def __init__(self, dtype=torch.float16, name=MODEL_NAME):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.dtype = dtype
        self.elem = torch.empty(0, dtype=dtype).element_size()

        self.tok = AutoTokenizer.from_pretrained(name)
        self.hf = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(DEV).eval()
        dec = self.hf.model.decoder
        cfg = self.hf.config
        assert (
            dec.project_in is None
            and dec.project_out is None
            and cfg.do_layer_norm_before
        )
        self.dec = dec
        self.layers = list(dec.layers)
        self.L = cfg.num_hidden_layers
        self.E = cfg.hidden_size
        self.H = cfg.num_attention_heads
        self.D = self.E // self.H
        self.scale = self.D**-0.5
        self.V = cfg.vocab_size
        self.plane_tok = (
            self.E * self.elem
        )  # bytes of K (or V) for one token, one beam, one layer
        self.layer_tok = 2 * self.plane_tok  # K+V: 10,240 bytes in FP16
        self.beam_tok = self.L * self.layer_tok  # all layers: 327,680 bytes in FP16

    def embed(self, tokens, pos):
        # OPT's learned position embedding has an offset of 2
        return (
            self.dec.embed_tokens.weight[tokens]
            + self.dec.embed_positions.weight[pos + 2]
        )

    def layer(self, l, h, K, V, s):
        """Layer l. K, V: [capacity, B, E]; rows [0, s) hold past tokens, this token's k, v go to row s."""
        lay = self.layers[l]
        a = lay.self_attn
        B = h.shape[0]
        x = lay.self_attn_layer_norm(h)
        q = a.q_proj(x) * self.scale
        K[s] = a.k_proj(x)
        V[s] = a.v_proj(x)
        n = s + 1
        Kv = (
            K[:n].view(n, B * self.H, self.D).permute(1, 0, 2)
        )  # [B*H, n, D], a view, no copy
        Vv = V[:n].view(n, B * self.H, self.D).permute(1, 0, 2)
        w = torch.bmm(Kv, q.view(B * self.H, self.D, 1))  # [B*H, n, 1]
        w = torch.softmax(w, dim=1, dtype=torch.float32).to(h.dtype)
        o = torch.bmm(w.transpose(1, 2), Vv).view(B, self.E)
        h = h + a.out_proj(o)
        x = lay.final_layer_norm(h)
        h = h + lay.fc2(lay.activation_fn(lay.fc1(x)))
        return h

    def head(self, h):
        return self.hf.lm_head(self.dec.final_layer_norm(h))


def load_prompt(model, n_tokens, path=None):
    """The fixed prompt text (data/prompt.txt), cut to the first n_tokens tokens (OPT prepends </s>)."""
    return read_prompt_ids(model.tok, n_tokens, path)
