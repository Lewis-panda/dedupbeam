"""Correctness of the GQA model class (Qwen2.5-3B) with the per-beam engine. Usage: python scripts/check_gqa.py [--skip-fp32]

0. Our forward vs the HF forward over the 128 prompt positions: FP32 vs FP32, FP16 vs FP16, FP16 vs FP32
   (max logit difference, argmax agreement, inf/nan).
1. Whole decode vs HF: 4 beams on the GPU, 96 generated tokens (2 prunes); the best beam is fed to the HF forward and
   every generated token's log-prob is compared (covers RoPE positions, KV writes/reads, KV copies at pruning).
2. Offloading (FP16): A. baseline vs ingpu bitwise; B. LBS vs grouped ingpu bitwise; D. bytes equal the paper's formula
   with 36 layers and 1,024 bytes per layer per token.
3. FP32: ingpu, baseline, LBS run independently give identical trees.
M keeps the same number of tokens as the OPT checks: M = M_opt * (Qwen KV per token / OPT KV per token).
"""

import gc, json, sys

import _common  # noqa: F401  (sets up paths)
import numpy as np
import torch

from dedupbeam import predict
from dedupbeam.common import DEV
from dedupbeam.gqa import QwenGQA, load_prompt
from dedupbeam.perbeam import PerBeamEngine

MiB = 2**20
OPT_BEAM_TOK = 327680
CASES = [  # (name, beams, step, prompt, generated, M used in the OPT checks)
    ("4 beams / step 32 / gen 64", 4, 32, 128, 64, 170 * MiB),
    ("4 beams / step 8 / gen 64", 4, 8, 128, 64, 200 * MiB),
    ("8 beams / step 16 / gen 128", 8, 16, 128, 128, 400 * MiB),
    ("10 beams / step 32 / gen 192", 10, 32, 128, 192, 450 * MiB),
]
quiet = lambda *a: None


def mine_logits(m, ids):
    P = len(ids)
    with torch.inference_mode():
        K = torch.empty(2 * m.L, P, 1, m.E, dtype=m.dtype, device=DEV)
        t = torch.tensor(ids, device=DEV)
        out = []
        for s in range(P):
            h = m.embed(t[s : s + 1], s)
            for l in range(m.L):
                h = m.layer(l, h, K[2 * l], K[2 * l + 1], s)
            out.append(m.head(h)[0].float())
    return torch.stack(out)


def hf_logits(m, ids):
    with torch.inference_mode():
        return m.hf(torch.tensor([ids], device=DEV)).logits[0].float()


def path_logprobs(tree, beam):
    """Walk the beam tree backwards from final slot `beam`; return the log-prob of every token on its path."""
    S = tree["parent"].shape[0]
    pos, out = beam, []
    for i in range(S - 1, -1, -1):
        out.append(tree["token_logprob"][i][pos])
        if i > 0:
            pos = int(tree["parent"][i][pos])
    return np.concatenate(out[::-1])


def decode_vs_hf(m, ids, label):
    N, T, gen = 4, 32, 96
    e = PerBeamEngine(m, "ingpu", N, T, ids, gen, 0, seed=0, log=quiet)
    r = e.run()
    tree = e.tree()
    e.close()
    b = r["best_beam"]
    seq = r["_tokens"][b].tolist()
    mine = path_logprobs(tree, b)
    with torch.inference_mode():
        lg = m.hf(torch.tensor([seq], device=DEV)).logits[0].float()
    lp = torch.log_softmax(lg, -1)
    P = len(ids)
    ref = (
        lp[P - 1 : P - 1 + gen]
        .gather(1, torch.tensor(seq[P:], device=DEV)[:, None])
        .squeeze(1)
        .double()
        .cpu()
        .numpy()
    )
    d = float(np.abs(mine - ref).max())
    print(
        f"[1 whole decode vs HF, {label}] best beam, {gen} tokens: max log-prob difference {d:.2e}, "
        f"score {mine.sum():.4f} vs HF {ref.sum():.4f}; starts {m.tok.decode(seq[P : P + 24])!r}"
    )
    return d


def run(m, mode, case, M, **kw):
    _, N, step, P, gen, _ = case
    e = PerBeamEngine(
        m, mode, N, step, load_prompt(m, P), gen, M, seed=0, log=quiet, **kw
    )
    r = e.run()
    r["_tree"] = e.tree()
    e.close()
    return r


def tree_same(a, b, tol=0.0):
    ta, tb = a["_tree"], b["_tree"]
    return bool(
        all(np.array_equal(ta[k], tb[k]) for k in ("parent", "tokens", "kept"))
        and np.abs(ta["score"] - tb["score"]).max() <= tol
    )


def offload_checks(m, strict):
    ok, out = True, []
    for case in CASES:
        name, N, step, P, gen, M_opt = case
        M = (
            M_opt * m.beam_tok // OPT_BEAM_TOK
        )  # beam_tok already scales with precision, so FP32 holds the same tokens
        ref = run(m, "ingpu", case, M)
        base = run(m, "baseline", case, M)
        lbs = run(m, "lbs", case, M)
        pb = predict.baseline_dm(N, M, P, gen, n_layer=m.L, layer_tok=m.layer_tok)
        pl = predict.lbs_dm(N, M, P, gen, step, beam_tok=m.beam_tok)
        d = base["h2d_bytes"] == pb and lbs["h2d_bytes"] == pl and pb > 0 and pl > 0
        if strict:  # FP16: pairs with equal batch sizes must match bit for bit
            gref = run(m, "ingpu", case, M, group_ref=True)
            a, b = tree_same(base, ref), tree_same(lbs, gref)
            free_run = tree_same(lbs, ref, float("inf"))
            print(
                f"== {name} (M = {M / MiB:.1f} MiB)\n   A baseline vs ingpu bitwise={a}; B LBS vs grouped ingpu bitwise={b}; "
                f"D bytes equal formula={d} (baseline {base['h2d_bytes']:,}, LBS {lbs['h2d_bytes']:,}, "
                f"up to {max(s['n_groups'] for s in lbs['steps'])} groups); (info) unforced LBS vs ingpu tree identical={free_run}"
            )
            c_ok = a and b and d
            out.append(dict(case=name, ok=c_ok, A=a, B=b, D=d, free_run_same=free_run))
        else:  # FP32: independent runs give the same tree
            t1, t2 = tree_same(base, ref, 1e-2), tree_same(lbs, ref, 1e-2)
            print(
                f"   E {name} (M = {M / MiB:.1f} MiB): baseline tree identical={t1}, LBS tree identical={t2}, bytes equal formula={d}"
            )
            c_ok = t1 and t2 and d
            out.append(dict(case=name, ok=c_ok, baseline_same=t1, lbs_same=t2, D=d))
        ok &= c_ok
    return ok, out


def main():
    m32 = QwenGQA(torch.float32)
    print("model config:", json.dumps(m32.config_summary()))
    ids = load_prompt(m32, 128)
    ref32 = hf_logits(m32, ids)
    my32 = mine_logits(m32, ids)
    d32 = (my32 - ref32).abs().max().item()
    a32 = (my32.argmax(-1) == ref32.argmax(-1)).float().mean().item()
    print(
        f"[0 FP32] our forward vs HF: max logit difference {d32:.2e}, argmax agreement {a32 * 100:.1f}%"
    )
    ref32c = ref32.cpu()
    dec32 = decode_vs_hf(m32, ids, "FP32")
    ok32, out32 = (True, [])
    if "--skip-fp32" not in sys.argv:
        print("\n===== FP32: independent runs =====")
        ok32, out32 = offload_checks(m32, strict=False)
    del m32, ref32, my32
    gc.collect()
    torch.cuda.empty_cache()

    m16 = QwenGQA(torch.float16)
    ref16 = hf_logits(m16, ids)
    my16 = mine_logits(m16, ids)
    d16 = (my16 - ref16).abs().max().item()
    a16 = (my16.argmax(-1) == ref16.argmax(-1)).float().mean().item()
    fin = bool(torch.isfinite(my16).all() and torch.isfinite(ref16).all())
    d16_32 = (my16.cpu() - ref32c).abs().max().item()
    a16_32 = (my16.cpu().argmax(-1) == ref32c.argmax(-1)).float().mean().item()
    print(
        f"[0 FP16] our forward vs HF FP16: max logit difference {d16:.4f}, argmax agreement {a16 * 100:.1f}%; no inf/nan={fin}"
    )
    print(
        f"[0 FP16 vs FP32] our FP16 vs HF FP32: max logit difference {d16_32:.4f}, argmax agreement {a16_32 * 100:.1f}%"
    )
    dec16 = decode_vs_hf(m16, ids, "FP16 (HF also FP16)")
    print("\n===== FP16: offloading =====")
    ok16, out16 = offload_checks(m16, strict=True)

    ok0 = (
        d32 < 1e-2
        and a32 == 1.0
        and fin
        and a16 >= 0.99
        and dec32 < 1e-2
        and dec16 < 0.25
    )
    ok = ok0 and ok16 and ok32
    print(
        f"\n0, 1 (vs HF): {'PASS' if ok0 else 'FAIL'}; FP16 offloading (A, B, D): {'PASS' if ok16 else 'FAIL'}; "
        f"FP32 (E): {'PASS' if ok32 else 'FAIL'}"
    )
    res = dict(
        ok=ok,
        fp32_logits_maxdiff=d32,
        fp32_argmax=a32,
        fp16_logits_maxdiff=d16,
        fp16_argmax=a16,
        fp16_finite=fin,
        fp16_vs_fp32_maxdiff=d16_32,
        fp16_vs_fp32_argmax=a16_32,
        decode_fp32_lp_maxdiff=dec32,
        decode_fp16_lp_maxdiff=dec16,
        fp16=out16,
        fp32=out32,
    )
    json.dump(res, open("results/checks/gqa.json", "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
