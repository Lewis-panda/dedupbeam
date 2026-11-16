"""Step-wise beam search with one KV copy per beam (the layout the LBS paper assumes).

Three execution modes:
  ingpu     all KV on the GPU, all beams in one batch (reference).
  baseline  layer-wise offloading, token by token, all beams in one batch. L_in(s) = min(L, floor(M / KV_layer(s)))
            layers stay on the GPU (paper Eq. 1); every other layer is copied host->GPU in full for every token.
  lbs       Locality-aware Beam Scheduling, reorder only: when the KV of all beams exceeds M, beams are split into
            groups; each group's KV (all layers) is loaded once, the group decodes the whole step, then the next group.

GPU KV layout (identical in all modes so the arithmetic is identical):
    each layer's K and V is a plane of shape [capacity, beams, E] (time first). The first s tokens are one contiguous
    block (one memcpy to/from the host), and attention runs bmm directly on views of it.
"""

import os, time, json, math, hashlib
from concurrent.futures import ThreadPoolExecutor

import torch

from .common import (
    DEV,
    GiB,
    Timer,
    mem_available_bytes,
    pinned_empty,
    pinned_free,
    sample,
)
from .opt import MODEL_NAME


class PerBeamEngine:
    def __init__(
        self,
        model,
        mode,
        n_beams,
        step,
        prompt_ids,
        n_gen,
        mem_bytes,
        seed,
        log=print,
        group_ref=False,
        force=None,
    ):
        # group_ref: ingpu only. Keep all KV on the GPU but decode in the same groups LBS would use under mem_bytes
        #            (same batch sizes as LBS), for a bitwise comparison with LBS.
        # force    : the record (`rec`) of another run; every token uses that run's choice (to compare log-probs).
        assert mode in ("ingpu", "baseline", "lbs")
        assert n_beams % 2 == 0 and n_gen % step == 0
        m = self.m = model
        self.mode, self.N, self.step, self.seed, self.log = (
            mode,
            n_beams,
            step,
            seed,
            log,
        )
        self.P = len(prompt_ids)
        self.prompt_ids = prompt_ids
        self.n_gen = n_gen
        self.total = self.P + n_gen
        self.n_steps = n_gen // step
        N, L, E = self.N, m.L, m.E
        full = N * self.total * m.beam_tok
        self.M = full if mode == "ingpu" else int(mem_bytes)
        self.M_group = int(mem_bytes)
        self.group_ref = group_ref and mode == "ingpu"
        self.win = None  # group_ref: the beam range being decoded
        self.force = force
        self.force_stats = dict(
            samples=0, own_differs=0, max_lp_diff=0.0, prune_differs=0
        )
        self.rec = dict(tok=[], lp=[], keep=[], parent=[], score=[])
        self.parent = (
            [-1] * n_beams
        )  # which slot of the previous step each slot was copied from; -1 = prompt
        self.will_offload = full > self.M
        self.timer = Timer()
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.host_copy_bytes = 0
        self.gpu_copy_bytes = 0
        self.pinned_bytes = 0
        self.kv_gpu_peak = 0  # highest KV content held on the GPU (bytes)
        self.steps_log = []
        self.pool = ThreadPoolExecutor(8)

        # --- pinned host memory (check that it fits first) ---
        if self.will_offload:
            if mode == "lbs":
                need = N * self.total * m.beam_tok
            else:
                self.L_final = min(L, self.M // (N * (self.total - 1) * m.layer_tok))
                need = (L - self.L_final) * N * self.total * m.layer_tok
            avail = mem_available_bytes()
            self.log(
                f"[host] need {need / GiB:.2f} GiB pinned, MemAvailable {avail / GiB:.2f} GiB"
            )
            if need + 3 * GiB > avail:
                raise MemoryError(
                    f"not enough host memory: need {need / GiB:.2f} GiB pinned + 3 GiB headroom, "
                    f"only {avail / GiB:.2f} GiB available"
                )
            t0 = time.perf_counter()
            if mode == "lbs":
                # one block per beam: [token, plane (2L), E]; the first s tokens are contiguous
                self.host = [
                    pinned_empty((self.total, 2 * L, E), m.dtype) for _ in range(N)
                ]
            else:
                # one block per evicted layer, for K and V: [token, beam, E]
                self.hostK = {
                    l: pinned_empty((self.total, N, E), m.dtype)
                    for l in range(self.L_final, L)
                }
                self.hostV = {
                    l: pinned_empty((self.total, N, E), m.dtype)
                    for l in range(self.L_final, L)
                }
            self.pinned_bytes = need
            self.log(
                f"[host] pinned allocation done in {time.perf_counter() - t0:.1f} s"
            )

        # --- GPU space for KV (arena) ---
        self.c_fit = self.M // (
            N * m.beam_tok
        )  # longest length at which all beams and layers fit
        assert self.c_fit >= self.P, (
            "M too small: the prompt KV does not fit on the GPU"
        )
        if self.will_offload:
            elems = (
                self.M // m.elem + 2 * L * N * E
            )  # one spare row per plane for the token being written
            self.c = min(self.total, self.c_fit + 1)
        else:
            elems = 2 * L * self.total * N * E
            self.c = self.total
        self.arena = torch.empty(elems, dtype=m.dtype, device=DEV)
        self.B = N  # beams in the current arena layout
        self.Lg = L  # layers in the current arena layout
        self.offloaded = False
        if self.will_offload and mode == "baseline":
            self.scrK = torch.empty(self.total, N, E, dtype=m.dtype, device=DEV)
            self.scrV = torch.empty(self.total, N, E, dtype=m.dtype, device=DEV)
        if self.will_offload and mode == "lbs":
            self.R = max(256, step)
            self.stage = torch.empty(self.R, 2 * L, E, dtype=m.dtype, device=DEV)

        self.gum = None  # Gumbel noise for one step [N, step, vocab]
        # Noise generation: OPT uses "step" (one [N, step, vocab] draw per step). Models with a large vocabulary
        # (Qwen2.5: 151,936) use "token" (one [N, vocab] draw per token); a per-step draw needs 5 GB at 256 beams.
        self.noise = getattr(m, "noise", "step")
        self.tokens = torch.zeros(N, self.total, dtype=torch.long, device=DEV)
        self.scores = torch.zeros(N, dtype=torch.float64, device=DEV)
        self.logits = None

    def _gumbel(self, i, t):
        """Gumbel noise [N, vocab] for token t of step i. Depends only on (seed, i, t), never on the grouping."""
        if self.noise == "step":
            return self.gum[:, t]
        gen = torch.Generator(device=DEV).manual_seed(
            self.seed * 1000003 + i * 4099 + t
        )
        u = torch.rand(self.N, self.m.V, generator=gen, device=DEV)
        return u.log_().neg_().log_().neg_()

    # ---- arena views ----
    def plane(self, p):
        n = self.c * self.B * self.m.E
        v = self.arena[p * n : (p + 1) * n].view(self.c, self.B, self.m.E)
        return v if self.win is None else v[:, self.win[0] : self.win[1]]

    def arena4(self):
        L, E = self.m.L, self.m.E
        return self.arena[: 2 * L * self.c * self.B * E].view(2 * L, self.c, self.B, E)

    # ---- prompt: computed once, copied to every beam ----
    def prefill(self):
        m, P = self.m, self.P
        tmp = torch.empty(2 * m.L, P, 1, m.E, dtype=m.dtype, device=DEV)
        ids = torch.tensor(self.prompt_ids, device=DEV)
        for s in range(P):
            h = m.embed(ids[s : s + 1], s)
            for l in range(m.L):
                h = m.layer(l, h, tmp[2 * l], tmp[2 * l + 1], s)
        lg = m.head(h)
        for p in range(2 * m.L):
            self.plane(p)[:P] = tmp[p].expand(P, self.N, m.E)
        self.tokens[:, :P] = ids
        self.logits = lg.expand(self.N, -1).contiguous()

    # ---- token loop for one step (all beams, or one group gi) ----
    def _tokens(self, i, s0, gi=None):
        m, T = self.m, self.timer
        sl = slice(None) if gi is None else gi
        lg = self.logits if gi is None else self.logits[gi]
        base = self.mode == "baseline" and self.will_offload
        for t in range(self.step):
            s = s0 + t
            forced = None
            if self.force is not None:
                forced = self.force["tok"][i][sl, t]
            own, tok, lp = sample(lg, self._gumbel(i, t)[sl], forced)
            if self.force is not None:
                fs = self.force_stats
                fs["samples"] += tok.numel()
                fs["own_differs"] += int((own != tok).sum())
                d = (lp - self.force["lp"][i][sl, t]).abs().max().item()
                fs["max_lp_diff"] = max(fs["max_lp_diff"], d)
            self.tokens[sl, s] = tok
            self.scores[sl] += lp
            self.step_lp[sl, t] = lp
            h = m.embed(tok, s)
            L_in = m.L
            if base:
                L_in = min(m.L, self.M // (self.N * s * m.layer_tok))  # Eq. 1
                if L_in < self.Lg:
                    self._evict_to(L_in, s)
                self.kv_gpu_peak = max(
                    self.kv_gpu_peak,
                    (L_in + (L_in < m.L)) * self.N * (s + 1) * m.layer_tok,
                )
            for l in range(m.L):
                if l < L_in:
                    h = m.layer(l, h, self.plane(2 * l), self.plane(2 * l + 1), s)
                else:
                    T.switch("h2d")  # reload the whole layer (all beams)
                    self.scrK[:s].copy_(self.hostK[l][:s])
                    self.scrV[:s].copy_(self.hostV[l][:s])
                    self.h2d_bytes += self.N * s * m.layer_tok
                    T.switch("compute")
                    h = m.layer(l, h, self.scrK, self.scrV, s)
                    T.switch("d2h")  # write the new token's k, v back to the host
                    self.hostK[l][s].copy_(self.scrK[s])
                    self.hostV[l][s].copy_(self.scrV[s])
                    self.d2h_bytes += self.N * m.layer_tok
                    T.switch("compute")
            lg = m.head(h)
        if gi is None:
            self.logits = lg
        else:
            self.logits[gi] = lg

    # ---- baseline: when the GPU runs out, evict the last layers to the host and re-lay the rest ----
    def _evict_to(self, L_new, s):
        m, T, N = self.m, self.timer, self.N
        T.switch("d2h")
        for l in range(L_new, self.Lg):
            self.hostK[l][:s].copy_(self.plane(2 * l)[:s])
            self.hostV[l][:s].copy_(self.plane(2 * l + 1)[:s])
            self.d2h_bytes += N * s * m.layer_tok
        T.switch("rearrange")
        if L_new > 0:
            c_new = min(self.total, self.M // (L_new * N * m.layer_tok) + 1)
            row = N * m.E
            if c_new != self.c:
                for p in range(2 * L_new - 1, 0, -1):
                    src = self.arena[p * self.c * row : p * self.c * row + s * row]
                    dst = self.arena[p * c_new * row : p * c_new * row + s * row]
                    if (
                        p * (c_new - self.c) < s
                    ):  # old and new ranges overlap: go through scratch
                        scr = self.scrK.view(-1)[: s * row]
                        scr.copy_(src)
                        dst.copy_(scr)
                    else:
                        dst.copy_(src)
                    self.gpu_copy_bytes += s * row * 2
                self.c = c_new
        self.Lg = L_new
        T.switch("compute")

    # ---- LBS: the first time KV no longer fits, write all GPU KV to the host ----
    def _lbs_flush(self, s0):
        m, T = self.m, self.timer
        A = self.arena4()
        for b in range(self.N):
            for r0 in range(0, s0, self.R):
                n = min(self.R, s0 - r0)
                T.switch("rearrange")
                self.stage[:n].copy_(A[:, r0 : r0 + n, b].transpose(0, 1))
                T.switch("d2h")
                self.host[b][r0 : r0 + n].copy_(self.stage[:n])
                self.d2h_bytes += n * m.beam_tok
        self.offloaded = True
        T.switch("compute")

    def _groups(self, s_end):
        """At most floor(M / one beam's KV at the end of the step) beams per group; fewest groups, balanced sizes."""
        b_max = self.M_group // (s_end * self.m.beam_tok)
        assert b_max >= 1, "M cannot hold even one beam"
        n_groups = math.ceil(self.N / b_max)
        return [list(map(int, g)) for g in torch.arange(self.N).tensor_split(n_groups)]

    # ---- LBS: one step, group by group ----
    def _lbs_step(self, i, s0):
        m, T, N = self.m, self.timer, self.N
        s_end = s0 + self.step
        groups = self._groups(s_end)
        for g in groups:
            self.c, self.B = s_end, len(g)
            A = self.arena4()
            for j, b in enumerate(g):  # load this group's KV (all layers) once
                for r0 in range(0, s0, self.R):
                    n = min(self.R, s0 - r0)
                    T.switch("h2d")
                    self.stage[:n].copy_(self.host[b][r0 : r0 + n])
                    self.h2d_bytes += n * m.beam_tok
                    T.switch("rearrange")
                    A[:, r0 : r0 + n, j].copy_(self.stage[:n].transpose(0, 1))
            T.switch("compute")
            self._tokens(i, s0, torch.tensor(g, device=DEV))
            for j, b in enumerate(g):  # write this step's new KV back to the host
                T.switch("rearrange")
                self.stage[: self.step].copy_(A[:, s0:s_end, j].transpose(0, 1))
                T.switch("d2h")
                self.host[b][s0:s_end].copy_(self.stage[: self.step])
                self.d2h_bytes += self.step * m.beam_tok
            T.switch("compute")
        return len(groups)

    # ---- end of step: keep the best half by cumulative log-prob; copy each survivor into a dropped slot ----
    def _prune(self, i, s):
        m, T, N = self.m, self.timer, self.N
        T.switch("prune")
        order = torch.sort(
            self.scores.cpu(), descending=True, stable=True
        ).indices.tolist()
        keep, drop = sorted(order[: N // 2]), sorted(order[N // 2 :])
        if self.force is not None and keep != self.force["keep"][i]:
            self.force_stats["prune_differs"] += 1
            keep = self.force["keep"][i]
            drop = sorted(set(range(N)) - set(keep))
        self.rec["keep"].append(keep)
        self.parent = list(
            range(N)
        )  # survivors stay in place; dropped slot drop[j] becomes a copy of keep[j]
        for d, k in zip(drop, keep):
            self.parent[d] = k
        kt, dt = torch.tensor(keep, device=DEV), torch.tensor(drop, device=DEV)
        self.tokens[dt] = self.tokens[kt]
        self.scores[dt] = self.scores[kt]
        self.logits[dt] = self.logits[kt]
        if self.offloaded:  # LBS: all KV is on the host, copy there
            jobs = [(self.host[d][:s], self.host[k][:s]) for d, k in zip(drop, keep)]
            list(self.pool.map(lambda a: a[0].copy_(a[1]), jobs))
            self.host_copy_bytes += len(jobs) * s * m.beam_tok
        else:
            for p in range(2 * self.Lg):  # layers on the GPU
                v = self.plane(p)[:s]
                v.index_copy_(1, dt, v.index_select(1, kt))
            self.gpu_copy_bytes += self.Lg * (N // 2) * s * m.layer_tok
            if self.Lg < m.L:  # baseline: layers on the host
                jobs = []
                for l in range(self.Lg, m.L):
                    for hv in (self.hostK[l], self.hostV[l]):
                        jobs += [(hv[:s, d], hv[:s, k]) for d, k in zip(drop, keep)]
                list(self.pool.map(lambda a: a[0].copy_(a[1]), jobs))
                self.host_copy_bytes += (m.L - self.Lg) * (N // 2) * s * m.layer_tok
        T.switch("compute")

    def run(self):
        m, T = self.m, self.timer
        with torch.inference_mode():
            t0 = time.perf_counter()
            self.prefill()
            torch.cuda.synchronize()
            self.prefill_s = time.perf_counter() - t0
            torch.cuda.reset_peak_memory_stats()
            t_start = time.perf_counter()
            T.switch("compute")
            for i in range(self.n_steps):
                s0 = self.P + i * self.step
                s_end = s0 + self.step
                n_groups = 1
                if self.noise == "step":
                    gen = torch.Generator(device=DEV).manual_seed(
                        self.seed * 1000003 + i
                    )
                    u = torch.rand(self.N, self.step, m.V, generator=gen, device=DEV)
                    self.gum = u.log_().neg_().log_().neg_()  # Gumbel noise, in place
                    del u
                self.step_lp = torch.zeros(
                    self.N, self.step, dtype=torch.float64, device=DEV
                )
                too_big = self.N * s_end * m.beam_tok > self.M_group
                if self.mode == "lbs" and too_big:
                    if not self.offloaded:
                        self._lbs_flush(s0)
                    n_groups = self._lbs_step(i, s0)
                elif self.group_ref and too_big:
                    groups = self._groups(s_end)
                    n_groups = len(groups)
                    for g in groups:
                        self.win = (g[0], g[-1] + 1)
                        self._tokens(i, s0, torch.tensor(g, device=DEV))
                    self.win = None
                else:
                    self._tokens(i, s0)
                if not (self.mode == "baseline" and self.will_offload):
                    biggest = math.ceil(self.N / n_groups)  # beams in the largest group
                    self.kv_gpu_peak = max(
                        self.kv_gpu_peak, biggest * s_end * m.beam_tok
                    )
                self.rec["tok"].append(self.tokens[:, s0:s_end].clone())
                self.rec["lp"].append(self.step_lp)
                self.rec["parent"].append(list(self.parent))
                self.rec["score"].append(self.scores.clone())
                if (
                    i == self.n_steps - 1
                ):  # no pruning after the last step; record who would survive
                    order = torch.sort(
                        self.scores.cpu(), descending=True, stable=True
                    ).indices.tolist()
                    self.rec["keep"].append(sorted(order[: self.N // 2]))
                if i < self.n_steps - 1:
                    self._prune(i, s_end)
                T.switch("compute")
                rec = dict(
                    step=i,
                    s_end=s_end,
                    n_groups=n_groups,
                    L_in_gpu=self.Lg,
                    wall=time.perf_counter() - t_start,
                    h2d_bytes=self.h2d_bytes,
                    d2h_bytes=self.d2h_bytes,
                    **{k: round(v, 4) for k, v in T.t.items()},
                )
                self.steps_log.append(rec)
                if i % 5 == 4 or i == self.n_steps - 1:
                    self.log(
                        f"[{self.mode} N={self.N}] step {i + 1}/{self.n_steps} s={s_end} "
                        f"wall={rec['wall']:.1f}s h2d={self.h2d_bytes / GiB:.1f}GiB "
                        f"groups={n_groups} L_in={self.Lg} "
                        + " ".join(f"{k}={v:.1f}" for k, v in T.t.items())
                    )
            T.switch(None)
            self.total_s = time.perf_counter() - t_start
        return self.result()

    def result(self):
        tok = self.tokens.cpu()
        scores = self.scores.cpu().tolist()
        best = max(range(self.N), key=lambda b: scores[b])
        t = dict(self.timer.t)
        return dict(
            mode=self.mode,
            n_beams=self.N,
            step=self.step,
            prompt_len=self.P,
            n_gen=self.n_gen,
            mem_gib=self.M / GiB,
            seed=self.seed,
            will_offload=self.will_offload,
            total_s=self.total_s,
            prefill_s=self.prefill_s,
            compute_s=t.get("compute", 0.0),
            h2d_s=t.get("h2d", 0.0),
            d2h_s=t.get("d2h", 0.0),
            rearrange_s=t.get("rearrange", 0.0),
            prune_s=t.get("prune", 0.0),
            h2d_bytes=self.h2d_bytes,
            d2h_bytes=self.d2h_bytes,
            host_copy_bytes=self.host_copy_bytes,
            gpu_copy_bytes=self.gpu_copy_bytes,
            pinned_bytes=self.pinned_bytes,
            kv_gpu_peak_bytes=self.kv_gpu_peak,
            gpu_peak_alloc_bytes=torch.cuda.max_memory_allocated(),
            tokens_sha256=hashlib.sha256(tok.numpy().tobytes()).hexdigest(),
            scores=scores,
            best_beam=best,
            best_text_head=self.m.tok.decode(tok[best, self.P : self.P + 48].tolist()),
            steps=self.steps_log,
            force_stats=self.force_stats,
            _tokens=tok,
        )

    def tree(self):
        """Beam tree: for every step and slot, the parent slot, tokens, score, and whether it survived."""
        import numpy as np

        S, N = self.n_steps, self.N
        kept = np.zeros((S, N), dtype=bool)
        for i, k in enumerate(self.rec["keep"]):
            kept[i, k] = True
        return dict(
            parent=np.array(self.rec["parent"], dtype=np.int16),
            tokens=torch.stack(self.rec["tok"]).cpu().numpy().astype(np.int32),
            token_logprob=torch.stack(self.rec["lp"]).cpu().numpy(),
            score=torch.stack(self.rec["score"]).cpu().numpy(),
            kept=kept,
        )

    def save_tree(self, path, **extra):
        import numpy as np, transformers

        header = dict(
            model=getattr(self.m, "name", MODEL_NAME),
            dtype=str(self.m.dtype).replace("torch.", ""),
            mode=self.mode,
            n_beams=self.N,
            beam_width=2,
            beam_size=self.N // 2,
            step=self.step,
            n_steps=self.n_steps,
            prompt_len=self.P,
            n_gen=self.n_gen,
            seed=self.seed,
            temperature=1.0,
            top_k=None,
            top_p=None,
            sampling="Gumbel-max: token = argmax(log_softmax(logits) + noise); noise = -log(-log(u)), "
            + (
                "u = torch.rand(n_beams, step, vocab, generator=cuda Generator(seed * 1000003 + step index))"
                if self.noise == "step"
                else "u = torch.rand(n_beams, vocab, generator=cuda Generator(seed * 1000003 + step index * 4099 + token index))"
            ),
            scoring="cumulative log-probability (log_softmax at temperature 1, accumulated in float64)",
            pruning="after each step keep the n_beams/2 highest scores (ties: lower slot first); no pruning after the "
            "last step, where `kept` records who would survive under the same rule",
            kv_space_gib=self.M_group / GiB,
            prompt_ids=list(map(int, self.prompt_ids)),
            tokens_sha256=hashlib.sha256(
                self.tokens.cpu().numpy().tobytes()
            ).hexdigest(),
            torch=torch.__version__,
            transformers=transformers.__version__,
            gpu=torch.cuda.get_device_name(0),
            **extra,
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path, header=json.dumps(header, ensure_ascii=False), **self.tree()
        )
        return path

    def close(self):
        self.pool.shutdown()
        for t in getattr(self, "host", []):
            pinned_free(t)
        for d in (getattr(self, "hostK", {}), getattr(self, "hostV", {})):
            for t in d.values():
                pinned_free(t)
        for name in (
            "arena",
            "scrK",
            "scrV",
            "stage",
            "host",
            "hostK",
            "hostV",
            "gum",
            "step_lp",
            "tokens",
            "logits",
        ):
            if hasattr(self, name):
                delattr(self, name)
        import gc

        gc.collect()
        torch.cuda.empty_cache()


def load_force(path):
    """Build a `force` record from a beam-tree file (.npz): a later run then picks every token from this tree."""
    import numpy as np

    z = np.load(path)
    S, N = z["parent"].shape
    return dict(
        tok=[
            torch.from_numpy(z["tokens"][i].astype(np.int64)).to(DEV) for i in range(S)
        ],
        lp=[torch.from_numpy(z["token_logprob"][i]).to(DEV) for i in range(S)],
        keep=[[n for n in range(N) if z["kept"][i, n]] for i in range(S)],
    )
