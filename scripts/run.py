"""One run of step-wise beam search. Writes a JSON summary and the beam tree.

Examples:
  python scripts/run.py --exp repro --store percopy --mode lbs --beams 32 --mem-gib 8
  python scripts/run.py --exp dedup --store dedup --mode baseline --step 64 --mem-gib 2.2 \
      --force-tree results/dedup/trees/percopy_baseline_N32_step64_M2.2.npz
  python scripts/run.py --exp gqa --model qwen --store percopy --mode ingpu --beams 112

Output: results/<exp>/<store>_<mode>_N<beams>_step<step>_M<mem>.json and results/<exp>/trees/<same>.npz.
An existing result with status "ok" is skipped unless --redo. Exit codes: 3 not enough host memory (not run),
4 GPU busy (not run), 5 GPU out of memory.
"""

import argparse, datetime, json, os, sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import _common  # noqa: E402  (sets up paths)

MODELS = {"opt": "facebook/opt-2.7b", "qwen": "Qwen/Qwen2.5-3B"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp",
        required=True,
        help="output folder under results/ (repro, dedup, gqa, ...)",
    )
    ap.add_argument("--model", default="opt", choices=sorted(MODELS))
    ap.add_argument(
        "--store",
        default="percopy",
        choices=["percopy", "dedup"],
        help="one KV copy per beam, or deduplicated",
    )
    ap.add_argument("--mode", required=True, choices=["ingpu", "baseline", "lbs"])
    ap.add_argument(
        "--beams",
        type=int,
        default=32,
        help="total beams (beam width 2: half survive each step)",
    )
    ap.add_argument("--step", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen", type=int, default=1920)
    ap.add_argument(
        "--mem-gib", type=float, default=8.0, help="GPU memory reserved for KV (M)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--force-tree", default=None, help="take every token from this beam tree (.npz)"
    )
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--no-gpu-check", action="store_true")
    a = ap.parse_args()
    if a.store == "dedup" and (a.mode == "ingpu" or a.model != "opt"):
        sys.exit("the dedup store supports --model opt with --mode baseline or lbs")
    name = f"{a.store}_{a.mode}_N{a.beams}_step{a.step}_M{a.mem_gib:g}"
    out = f"results/{a.exp}/{name}.json"
    tree_path = f"results/{a.exp}/trees/{name}.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if (
        os.path.exists(out)
        and not a.redo
        and json.load(open(out)).get("status") == "ok"
    ):
        print(f"[run] already done, skipping: {out}")
        return
    meta = dict(
        model=MODELS[a.model],
        store=a.store,
        mode=a.mode,
        n_beams=a.beams,
        step=a.step,
        prompt_len=a.prompt_len,
        n_gen=a.gen,
        mem_gib=a.mem_gib,
        seed=a.seed,
        forced_from=a.force_tree,
        started=datetime.datetime.now().isoformat(timespec="seconds"),
    )

    def finish(status, code, **kw):
        json.dump(dict(status=status, **meta, **kw), open(out, "w"), indent=1)
        print(f"[run] {status} -> {out}", flush=True)
        sys.exit(code)

    if not a.no_gpu_check:
        busy = _common.gpu_busy()
        if busy:
            finish("gpu_busy", 4, reason=busy)

    import torch
    from dedupbeam.common import GiB, mem_available_bytes
    from dedupbeam.dedup import DedupEngine
    from dedupbeam.perbeam import PerBeamEngine, load_force

    log = lambda *x: print(*x, flush=True)
    if a.model == "opt":
        from dedupbeam.opt import OPT as Model, load_prompt
    else:
        from dedupbeam.gqa import QwenGQA as Model, load_prompt
    m = Model(torch.float16)
    ids = load_prompt(m, a.prompt_len)
    force = load_force(a.force_tree) if a.force_tree else None
    full = a.beams * (a.prompt_len + a.gen) * m.beam_tok
    log(
        f"[run] {name}: per-beam KV of all beams at the end = {full / GiB:.2f} GiB; M = {a.mem_gib:g} GiB"
    )
    try:
        Engine = PerBeamEngine if a.store == "percopy" else DedupEngine
        e = Engine(
            m,
            a.mode,
            a.beams,
            a.step,
            ids,
            a.gen,
            int(a.mem_gib * GiB),
            a.seed,
            log=log,
            force=force,
        )
        r = e.run()
    except MemoryError as ex:
        log(f"[run] not run: {ex}")
        finish("skipped_host_memory", 3, reason=str(ex), kv_full_bytes=full)
    except torch.OutOfMemoryError as ex:
        log(f"[run] GPU out of memory: {str(ex)[:300]}")
        finish("gpu_oom", 5, reason=str(ex)[:500], kv_full_bytes=full)
    r.pop("_tokens")
    e.save_tree(
        tree_path, started=meta["started"], store=a.store, forced_from=a.force_tree
    )
    r["tree_file"] = tree_path
    r["kv_full_bytes"] = full
    r["kv_bytes_per_token"] = m.beam_tok
    if hasattr(m, "config_summary"):
        r["model_config"] = m.config_summary()
    r["mem_available_gib_during_run"] = mem_available_bytes() / GiB
    r.setdefault("max_groups", max(s["n_groups"] for s in r["steps"]))
    e.close()
    log(
        f"[run] total {r['total_s']:.1f} s; compute {r['compute_s']:.1f}, host->GPU {r['h2d_s']:.1f} "
        f"({r['h2d_bytes'] / GiB:.1f} GiB), GPU->host {r['d2h_s']:.1f}, rearrange {r['rearrange_s']:.1f}, "
        f"prune {r['prune_s']:.1f}; KV on GPU peak {r['kv_gpu_peak_bytes'] / GiB:.2f} GiB, up to {r['max_groups']} groups"
    )
    for k in (
        "store",
        "mode",
        "n_beams",
        "step",
        "prompt_len",
        "n_gen",
        "seed",
        "mem_gib",
    ):
        r.pop(k, None)
    finish("ok", 0, **r)


if __name__ == "__main__":
    main()
