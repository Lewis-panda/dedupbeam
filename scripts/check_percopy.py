"""Correctness of the per-beam engine (OPT-2.7B). Usage: python scripts/check_percopy.py [--skip-fp32]

FP16 (the precision of the experiments):
  0. Our forward vs the HF forward over the 128 prompt positions: max logit difference, argmax agreement.
  A. baseline vs ingpu: tokens and scores bitwise identical (same batch size; only KV placement differs).
  B. LBS vs "ingpu decoded in the groups LBS uses": bitwise identical (same batch size; only the host round trip differs).
  C. LBS vs ingpu (different batch sizes): LBS is forced to take ingpu's tokens; per-token log-prob difference and
     pruning decisions are compared. (Unforced, FP16 rounding eventually flips one sampled token and the sequences
     diverge, so this is reported separately.)
  D. Host->GPU bytes equal the paper's formula (dedupbeam/predict.py).
  T. Beam trees (parent, tokens, scores, kept) identical for the pairs in A and B.
FP32 (to show the differences in C come only from FP16 rounding):
  E. ingpu / baseline / LBS run independently: identical tokens and beam trees, score difference < 1e-2.

M is set small on purpose so that 4-10 short beams overflow it.
"""

import json, sys

import _common  # noqa: F401  (sets up paths)
import numpy as np
import torch

from dedupbeam import predict
from dedupbeam.common import DEV
from dedupbeam.opt import OPT, load_prompt
from dedupbeam.perbeam import PerBeamEngine

MiB = 2**20
# (name, beams, step, prompt, generated, M in bytes for FP16)
CASES = [
    (
        "4 beams / step 32 / gen 64 / M=170 MiB (overflows in the first step; LBS groups of 2)",
        4,
        32,
        128,
        64,
        170 * MiB,
    ),
    (
        "4 beams / step 8 / gen 64 / M=200 MiB (overflows mid-way; 7 prunes)",
        4,
        8,
        128,
        64,
        200 * MiB,
    ),
    (
        "8 beams / step 16 / gen 128 / M=400 MiB (overflows mid-way; LBS groups of 4)",
        8,
        16,
        128,
        128,
        400 * MiB,
    ),
    (
        "10 beams / step 32 / gen 192 / M=450 MiB (LBS 2 groups of 5, then 3 groups of 4/3/3)",
        10,
        32,
        128,
        192,
        450 * MiB,
    ),
]
quiet = lambda *a: None


def check_vs_hf(m, ids):
    P = len(ids)
    with torch.inference_mode():
        ref = m.hf(torch.tensor([ids], device=DEV)).logits[0].float()
        K = torch.empty(2 * m.L, P, 1, m.E, dtype=m.dtype, device=DEV)
        mine = []
        t = torch.tensor(ids, device=DEV)
        for s in range(P):
            h = m.embed(t[s : s + 1], s)
            for l in range(m.L):
                h = m.layer(l, h, K[2 * l], K[2 * l + 1], s)
            mine.append(m.head(h)[0].float())
        mine = torch.stack(mine)
    diff = (mine - ref).abs().max().item()
    agree = (mine.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(
        f"[0 vs HF] {P} positions: max logit difference {diff:.4f}, argmax agreement {agree * 100:.1f}%"
    )
    return agree == 1.0 and diff < 0.25


def run(m, mode, case, M, **kw):
    _, N, step, P, gen, _ = case
    e = PerBeamEngine(
        m, mode, N, step, load_prompt(m, P), gen, M, seed=0, log=quiet, **kw
    )
    r = e.run()
    r["_tree"] = e.tree()
    rec = e.rec
    e.close()
    return r, rec


def same_tree(a, b, score_tol=0.0):
    """Same parents, tokens and kept flags; scores within score_tol."""
    ta, tb = a["_tree"], b["_tree"]
    return bool(
        np.array_equal(ta["parent"], tb["parent"])
        and np.array_equal(ta["tokens"], tb["tokens"])
        and np.array_equal(ta["kept"], tb["kept"])
        and np.abs(ta["score"] - tb["score"]).max() <= score_tol
        and np.abs(ta["token_logprob"] - tb["token_logprob"]).max() <= score_tol
    )


def same(a, b):
    tok = bool((a["_tokens"] == b["_tokens"]).all())
    sdiff = max(abs(x - y) for x, y in zip(a["scores"], b["scores"]))
    return tok, sdiff


def fp16_checks(m):
    ok = check_vs_hf(m, load_prompt(m, 128))
    out = []
    for case in CASES:
        name, N, step, P, gen, M = case
        ref, ref_rec = run(m, "ingpu", case, M)
        base, _ = run(m, "baseline", case, M)
        lbs, _ = run(m, "lbs", case, M)
        gref, _ = run(m, "ingpu", case, M, group_ref=True)
        forced, _ = run(m, "lbs", case, M, force=ref_rec)
        print(f"\n== {name}")
        a_tok, a_s = same(base, ref)
        a_tree = same_tree(base, ref)
        print(
            f"   A  baseline vs ingpu: tokens identical={a_tok}, max score difference={a_s:.1e}"
        )
        b_tok, b_s = same(lbs, gref)
        b_tree = same_tree(lbs, gref)
        print(
            f"   B  LBS vs ingpu in the same groups: tokens identical={b_tok}, max score difference={b_s:.1e}"
        )
        free_tree = same_tree(lbs, base, float("inf"))
        print(
            f"   T  beam trees: baseline vs ingpu identical={a_tree}; LBS vs grouped ingpu identical={b_tree}; "
            f"(info) unforced LBS vs baseline parents/tokens/kept identical={free_tree}"
        )
        fs = forced["force_stats"]
        c_tok, c_s = same(forced, ref)
        c_ok = (
            c_tok
            and fs["max_lp_diff"] < 0.25
            and fs["prune_differs"] == 0
            and c_s < 0.02 * gen
        )
        print(
            f"   C  LBS forced onto ingpu's tokens: max per-token log-prob difference={fs['max_lp_diff']:.4f}, "
            f"max final score difference={c_s:.4f}, steps with a different pruning choice={fs['prune_differs']}, "
            f"LBS would have sampled a different token {fs['own_differs']} of {fs['samples']} times"
        )
        f_tok, f_s = same(lbs, ref)
        print(
            f"      (info) unforced LBS vs ingpu: tokens identical={f_tok}, max score difference={f_s:.3f}"
        )
        pb, pl = predict.baseline_dm(N, M, P, gen), predict.lbs_dm(N, M, P, gen, step)
        d_ok = base["h2d_bytes"] == pb and lbs["h2d_bytes"] == pl and pb > 0 and pl > 0
        print(
            f"   D  host->GPU bytes: baseline {base['h2d_bytes']:,} measured / {pb:,} formula; "
            f"LBS {lbs['h2d_bytes']:,} / {pl:,}; equal={d_ok}"
        )
        case_ok = (
            a_tok
            and a_s == 0
            and b_tok
            and b_s == 0
            and c_ok
            and d_ok
            and a_tree
            and b_tree
        )
        ok &= case_ok
        out.append(
            dict(
                case=name,
                ok=case_ok,
                A_tokens_same=a_tok,
                A_score_diff=a_s,
                B_tokens_same=b_tok,
                B_score_diff=b_s,
                tree_baseline_eq_ingpu=a_tree,
                tree_lbs_eq_grouped_ingpu=b_tree,
                tree_lbs_eq_baseline_free_run=free_tree,
                C_max_lp_diff=fs["max_lp_diff"],
                C_score_diff=c_s,
                C_prune_differs=fs["prune_differs"],
                C_samples=fs["samples"],
                C_own_differs=fs["own_differs"],
                free_run_tokens_same=f_tok,
                baseline_h2d=base["h2d_bytes"],
                baseline_pred=pb,
                lbs_h2d=lbs["h2d_bytes"],
                lbs_pred=pl,
            )
        )
    return ok, out


def fp32_checks(m):
    ok, out = True, []
    for case in CASES:
        name, N, step, P, gen, M16 = case
        M = (
            2 * M16
        )  # FP32 KV is twice as large; doubling M keeps the same grouping as in FP16
        ref, _ = run(m, "ingpu", case, M)
        base, _ = run(m, "baseline", case, M)
        lbs, _ = run(m, "lbs", case, M)
        a_tok, a_s = same(base, ref)
        l_tok, l_s = same(lbs, ref)
        pb = 2 * predict.baseline_dm(N, M16, P, gen)
        pl = 2 * predict.lbs_dm(N, M16, P, gen, step)
        d_ok = base["h2d_bytes"] == pb and lbs["h2d_bytes"] == pl
        t_ok = same_tree(base, ref, 1e-2) and same_tree(lbs, ref, 1e-2)
        case_ok = a_tok and l_tok and a_s < 1e-2 and l_s < 1e-2 and d_ok and t_ok
        ok &= case_ok
        print(
            f"   E  {name.split(' (')[0]} (M doubled): baseline tokens identical={a_tok} (score diff {a_s:.1e}); "
            f"LBS tokens identical={l_tok} (score diff {l_s:.1e}); bytes equal formula={d_ok}; trees identical={t_ok}"
        )
        out.append(
            dict(
                case=name,
                ok=case_ok,
                baseline_tokens_same=a_tok,
                baseline_score_diff=a_s,
                lbs_tokens_same=l_tok,
                lbs_score_diff=l_s,
                bytes_match=d_ok,
                trees_same=t_ok,
            )
        )
    return ok, out


def main():
    print("===== FP16 =====")
    m = OPT(torch.float16)
    ok16, out16 = fp16_checks(m)
    ok32, out32 = None, []
    if "--skip-fp32" not in sys.argv:
        del m
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        print("\n===== FP32: independent runs =====")
        m = OPT(torch.float32)
        ok32, out32 = fp32_checks(m)
    ok = ok16 and ok32 is not False
    print(f"\nFP16 checks (0, A, B, C, D, T): {'PASS' if ok16 else 'FAIL'}")
    if ok32 is not None:
        print(f"FP32 checks (E): {'PASS' if ok32 else 'FAIL'}")
    json.dump(
        dict(ok=ok, fp16_ok=ok16, fp32_ok=ok32, fp16=out16, fp32=out32),
        open("results/checks/percopy.json", "w"),
        indent=1,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
