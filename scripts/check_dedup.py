"""Correctness of the deduplicated KV store (OPT-2.7B). Usage: python scripts/check_dedup.py [--skip-fp32]

FP16:
  A. dedup baseline (small M, offloads) vs dedup with all KV on the GPU: beam trees bitwise identical.
  B. dedup LBS (groups) vs the same flow with the "host" copy kept on the GPU: bitwise identical
     (the round trip through real host memory changes no bit).
  C. dedup on the GPU vs per-beam on the GPU, with dedup forced onto the per-beam tokens: per-token log-prob difference;
     and dedup LBS vs dedup on the GPU, forced the same way.
  D. Host->GPU bytes equal a GPU-free replay of the recorded beam tree (dedupbeam.segbook.simulate_*).
FP32:
  E. per-beam ingpu, dedup on the GPU, dedup baseline, dedup LBS run independently: identical tokens and beam trees.
M is given in rows (one row = all layers of one distinct token).
"""

import json, sys, gc

import _common  # noqa: F401  (sets up paths)
import numpy as np
import torch

from dedupbeam import segbook
from dedupbeam.dedup import DedupEngine
from dedupbeam.opt import OPT, load_prompt
from dedupbeam.perbeam import PerBeamEngine

# (name, beams, step, prompt, generated, M in rows)
CASES = [
    ("4 beams / step 8 / gen 64 / M=200 rows", 4, 8, 128, 64, 200),
    ("8 beams / step 16 / gen 128 / M=300 rows", 8, 16, 128, 128, 300),
    ("8 beams / step 32 / gen 256 / M=420 rows", 8, 32, 128, 256, 420),
    ("10 beams / step 32 / gen 192 / M=520 rows", 10, 32, 128, 192, 520),
]
quiet = lambda *a: None


def run_pc(m, case):
    _, N, T, P, gen, _ = case
    e = PerBeamEngine(m, "ingpu", N, T, load_prompt(m, P), gen, 0, seed=0, log=quiet)
    r = e.run()
    r["_tree"] = e.tree()
    rec = e.rec
    e.close()
    return r, rec


def run_dd(m, mode, case, rows, **kw):
    _, N, T, P, gen, _ = case
    e = DedupEngine(
        m,
        mode,
        N,
        T,
        load_prompt(m, P),
        gen,
        rows * m.beam_tok,
        seed=0,
        log=quiet,
        **kw,
    )
    r = e.run()
    r["_tree"] = e.tree()
    rec = e.rec
    e.close()
    return r, rec


def tree_same(a, b, tol=0.0):
    ta, tb = a["_tree"], b["_tree"]
    return bool(
        all(np.array_equal(ta[k], tb[k]) for k in ("parent", "tokens", "kept"))
        and np.abs(ta["score"] - tb["score"]).max() <= tol
        and np.abs(ta["token_logprob"] - tb["token_logprob"]).max() <= tol
    )


def fp16(m):
    ok, out = True, []
    for case in CASES:
        name, N, T, P, gen, rows = case
        big = N * (P + gen)
        pc, pc_rec = run_pc(m, case)
        din, din_rec = run_dd(m, "lbs", case, big)
        dbin, _ = run_dd(m, "baseline", case, big)
        db, _ = run_dd(m, "baseline", case, rows)
        dl, _ = run_dd(m, "lbs", case, rows)
        dlg, _ = run_dd(m, "lbs", case, rows, host_on_gpu=True)
        f1, _ = run_dd(m, "lbs", case, big, force=pc_rec)
        f2, _ = run_dd(m, "lbs", case, rows, force=din_rec)
        print(f"\n== {name}")
        a = tree_same(db, dbin) and tree_same(dbin, din)
        print(
            f"   A  dedup baseline vs dedup on the GPU: trees bitwise identical={a} "
            f"(baseline moved {db['h2d_bytes']:,} bytes; fewest GPU layers {min(s['L_in_gpu'] for s in db['steps'])})"
        )
        b = tree_same(dl, dlg)
        print(
            f"   B  dedup LBS vs the same flow with the host copy on the GPU: bitwise identical={b} "
            f"(LBS moved {dl['h2d_bytes']:,} bytes; up to {dl['max_groups']} groups)"
        )
        s1, s2 = f1["force_stats"], f2["force_stats"]
        c = (
            tree_same(f1, pc, 0.02 * gen)
            and tree_same(f2, din, 0.02 * gen)
            and s1["max_lp_diff"] < 0.25
            and s2["max_lp_diff"] < 0.25
            and s1["prune_differs"] == 0
            and s2["prune_differs"] == 0
        )
        print(
            f"   C  dedup on the GPU forced onto per-beam tokens: max log-prob difference={s1['max_lp_diff']:.4f}, "
            f"would have sampled differently {s1['own_differs']} of {s1['samples']} times, pruning differs in {s1['prune_differs']} steps"
        )
        print(
            f"      dedup LBS forced onto dedup-on-GPU tokens: max log-prob difference={s2['max_lp_diff']:.4f}, "
            f"differs {s2['own_differs']} times, pruning differs in {s2['prune_differs']} steps"
        )
        tb, tl = db["_tree"], dl["_tree"]
        sb = segbook.simulate_baseline(
            tb["parent"], tb["kept"], P, T, rows * m.beam_tok, m.layer_tok
        )
        sl = segbook.simulate_lbs(
            tl["parent"], tl["kept"], P, T, rows * m.beam_tok, m.beam_tok
        )[0]
        d = (
            db["h2d_bytes"] == sb
            and dl["h2d_bytes"] == sl
            and sb > 0
            and sl > 0
            and dl["max_groups"] > 1
        )
        print(
            f"   D  host->GPU bytes: baseline {db['h2d_bytes']:,} measured / {sb:,} replayed; "
            f"LBS {dl['h2d_bytes']:,} / {sl:,}; equal={d}"
        )
        print(
            f"      (info) unforced: dedup on GPU vs per-beam tree identical={tree_same(din, pc, float('inf'))}; "
            f"dedup LBS vs dedup on GPU identical={tree_same(dl, din, float('inf'))}"
        )
        case_ok = a and b and c and d
        ok &= case_ok
        out.append(
            dict(
                case=name,
                ok=case_ok,
                A=a,
                B=b,
                C=c,
                D=d,
                C_dedup_vs_percopy=s1,
                C_lbs_vs_ingpu=s2,
            )
        )
    return ok, out


def fp32(m):
    ok, out = True, []
    for case in CASES:
        name, N, T, P, gen, rows = case
        big = N * (P + gen)
        pc, _ = run_pc(m, case)
        din, _ = run_dd(m, "lbs", case, big)
        db, _ = run_dd(m, "baseline", case, rows)
        dl, _ = run_dd(m, "lbs", case, rows)
        t = [tree_same(x, pc, 1e-2) for x in (din, db, dl)]
        moved = db["h2d_bytes"] > 0 and dl["h2d_bytes"] > 0
        case_ok = all(t) and moved
        ok &= case_ok
        print(
            f"   E  {name}: same tree as per-beam on the GPU: dedup on GPU={t[0]}, dedup baseline={t[1]}, "
            f"dedup LBS={t[2]}; max score difference "
            f"{max(np.abs(x['_tree']['score'] - pc['_tree']['score']).max() for x in (din, db, dl)):.1e}"
        )
        out.append(dict(case=name, ok=case_ok, trees_same=t))
    return ok, out


def main():
    print("===== FP16 =====")
    m = OPT(torch.float16)
    ok16, out16 = fp16(m)
    ok32, out32 = None, []
    if "--skip-fp32" not in sys.argv:
        del m
        gc.collect()
        torch.cuda.empty_cache()
        print("\n===== FP32: independent runs =====")
        m = OPT(torch.float32)
        ok32, out32 = fp32(m)
    print(f"\nFP16 checks (A, B, C, D): {'PASS' if ok16 else 'FAIL'}")
    if ok32 is not None:
        print(f"FP32 checks (E): {'PASS' if ok32 else 'FAIL'}")
    json.dump(
        dict(fp16_ok=ok16, fp32_ok=ok32, fp16=out16, fp32=out32),
        open("results/checks/dedup.json", "w"),
        indent=1,
        default=str,
    )
    sys.exit(0 if (ok16 and ok32 is not False) else 1)


if __name__ == "__main__":
    main()
