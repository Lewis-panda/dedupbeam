"""Load the result JSON files and derive the numbers used by make_tables.py and make_figures.py."""

import glob, json, os

import _common  # noqa: F401  (sets up paths)

from dedupbeam import predict, segbook

GiB = 2**30
STEPS = (32, 64, 128)
M_MAIN, M_CTRL, M_SUPP = 2.2, 8, 1
PROMPT, GEN, N_DEDUP = 128, 1920, 32
METHODS = [  # (store, mode, label)
    ("percopy", "baseline", "Baseline, per-beam KV"),
    ("percopy", "lbs", "LBS, per-beam KV"),
    ("dedup", "baseline", "Baseline, dedup KV"),
    ("dedup", "lbs", "LBS, dedup KV"),
]
GPU = "RTX 5060 Ti 16 GB"


def _load(path):
    if not os.path.exists(path):
        return None
    r = json.load(open(path))
    return r if r.get("status") == "ok" else r


def ok(r):
    return r is not None and r.get("status") == "ok"


def run(exp, store, mode, beams, step, mem):
    return _load(f"results/{exp}/{store}_{mode}_N{beams}_step{step}_M{mem:g}.json")


def machine():
    return json.load(open("results/machine.json"))


def h2d_gib_per_s():
    return machine()["transfer"]["h2d_pinned_1 GiB"]["gib_per_s"]


def other_s(r):
    """Everything that is neither host->GPU transfer nor compute: GPU->host, GPU rearrangement, pruning."""
    return r["d2h_s"] + r["rearrange_s"] + r["prune_s"]


def ref_tree(step):
    import numpy as np

    p = f"results/dedup/trees/percopy_baseline_N{N_DEDUP}_step{step}_M{M_MAIN:g}.npz"
    return np.load(p) if os.path.exists(p) else None


def predicted_h2d(store, mode, step, mem):
    """Predicted host->GPU bytes. Per-beam: the paper's Eq. 1-5. Dedup: replay of the reference beam tree."""
    M = int(mem * GiB)
    if store == "percopy":
        if mode == "baseline":
            return predict.baseline_dm(N_DEDUP, M, PROMPT, GEN)
        return predict.lbs_dm(N_DEDUP, M, PROMPT, GEN, step)
    t = ref_tree(step)
    if t is None:
        return None
    if mode == "baseline":
        return segbook.simulate_baseline(
            t["parent"], t["kept"], PROMPT, step, M, predict.LAYER_TOK
        )
    return segbook.simulate_lbs(
        t["parent"], t["kept"], PROMPT, step, M, predict.BEAM_TOK
    )[0]


def dedup_runs(mem=M_MAIN):
    return {
        T: {(s, m): run("dedup", s, m, N_DEDUP, T, mem) for s, m, _ in METHODS}
        for T in STEPS
    }


def ratio(a, b):
    """a / b of total time (how many times faster b is than a)."""
    return a["total_s"] / b["total_s"] if ok(a) and ok(b) else None


def speedups(R):
    out = {}
    for T in STEPS:
        r = R[T]
        out[T] = dict(
            lbs_vs_base_percopy=ratio(
                r[("percopy", "baseline")], r[("percopy", "lbs")]
            ),
            lbs_vs_base_dedup=ratio(r[("dedup", "baseline")], r[("dedup", "lbs")]),
            dedup_vs_percopy_base=ratio(
                r[("percopy", "baseline")], r[("dedup", "baseline")]
            ),
            dedup_vs_percopy_lbs=ratio(r[("percopy", "lbs")], r[("dedup", "lbs")]),
        )
    return out


def percopy_kv_bytes(beams=N_DEDUP, beam_tok=predict.BEAM_TOK):
    return beams * (PROMPT + GEN) * beam_tok


def repro_runs():
    return {
        n: {
            m: run("repro", "percopy", m, n, 32, 8)
            for m in ("ingpu", "baseline", "lbs")
        }
        for n in (8, 16, 24, 32)
    }


def gqa_runs():
    out = {}
    for f in sorted(glob.glob("results/gqa/percopy_*_N*_step32_M8.json")):
        r = json.load(open(f))
        out[(r["mode"], r["n_beams"])] = r
    return out


def gqa_config():
    for r in gqa_runs().values():
        if ok(r) and "model_config" in r:
            return r["model_config"]
    return None


def microbench():
    p = "results/microbench_attention.json"
    return json.load(open(p)) if os.path.exists(p) else None


def checks():
    out = {}
    for name in ("percopy", "dedup", "gqa"):
        p = f"results/checks/{name}.json"
        if os.path.exists(p):
            out[name] = json.load(open(p))
    return out
