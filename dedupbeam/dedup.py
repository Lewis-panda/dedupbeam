"""Baseline and LBS with a deduplicated KV store: content shared by several beams is stored once.

Storage: each layer's K and V is a pool of shape [rows, E]; one row = one distinct token (there is no beam axis).
         A segment (what one ancestor wrote in one step) has one copy on the GPU and at most one on the host.
Attention: one score matrix [head, beam, rows] over the whole pool; rows a beam cannot see get -inf (mask); one softmax;
         one weighted sum over the pool. Still two bmm per layer, reading the shared copy directly; nothing is
         expanded into per-beam copies. segbook.SegBook tracks which segment lives in which rows.
"""

import math, time, hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
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
from .perbeam import PerBeamEngine
from .segbook import SegBook


def layer_pool(m, l, h, K, V, n0, neg):
    """Layer l over a pool. K, V: [capacity, E]; rows [0, n0) exist, the B new tokens go to rows n0 .. n0+B-1.
    neg: [B, rows] mask (0 = visible, -inf = not this beam's)."""
    lay = m.layers[l]
    a = lay.self_attn
    B, H, D = h.shape[0], m.H, m.D
    x = lay.self_attn_layer_norm(h)
    q = a.q_proj(x) * m.scale
    n = n0 + B
    K[n0:n] = a.k_proj(x)
    V[n0:n] = a.v_proj(x)
    w = torch.bmm(
        q.view(B, H, D).transpose(0, 1), K[:n].view(n, H, D).permute(1, 2, 0)
    )  # [H, B, n]
    w += neg[:, :n]
    w = torch.softmax(w, dim=-1, dtype=torch.float32).to(h.dtype)
    o = (
        torch.bmm(w, V[:n].view(n, H, D).permute(1, 0, 2))
        .transpose(0, 1)
        .reshape(B, m.E)
    )
    h = h + a.out_proj(o)
    x = lay.final_layer_norm(h)
    return h + lay.fc2(lay.activation_fn(lay.fc1(x)))


def move_rows(t, src, dst, n):
    """Within one tensor, move rows [src, src+n) down to [dst, dst+n) (dst < src) in non-overlapping chunks."""
    shift = src - dst
    for off in range(0, n, shift):
        k = min(shift, n - off)
        t[dst + off : dst + off + k].copy_(t[src + off : src + off + k])


class DedupEngine:
    noise = "step"  # Gumbel noise is drawn once per step, as in PerBeamEngine for OPT

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
        force=None,
        host_on_gpu=False,
        max_unique_rows=32768,
    ):
        # host_on_gpu: for testing. The "host" copy lives on the GPU and everything else is unchanged; used to show
        #              that a round trip through real host memory leaves every bit intact.
        assert mode in ("baseline", "lbs")
        assert n_beams % 2 == 0 and n_gen % step == 0
        m = self.m = model
        self.mode, self.N, self.step, self.seed, self.log = (
            mode,
            n_beams,
            step,
            seed,
            log,
        )
        self.P, self.prompt_ids, self.n_gen = len(prompt_ids), prompt_ids, n_gen
        self.total = self.P + n_gen
        self.n_steps = n_gen // step
        self.M = self.M_group = int(mem_bytes)
        self.force = force
        self.force_stats = dict(
            samples=0, own_differs=0, max_lp_diff=0.0, prune_differs=0
        )
        self.rec = dict(tok=[], lp=[], keep=[], parent=[], score=[])
        self.parent = [-1] * n_beams
        self.timer = Timer()
        self.h2d_bytes = self.d2h_bytes = self.host_copy_bytes = self.gpu_copy_bytes = 0
        self.h2d_bytes_no_retain = 0
        self.kv_gpu_peak = 0
        self.unique_rows_peak = 0  # distinct rows after a step is written, before pruning (GPU + host, once each)
        self.unique_rows_peak_after_prune = 0
        self.steps_log = []
        self.pool = ThreadPoolExecutor(8)
        self.book = SegBook(n_beams, self.P, step)
        N, L, E, T = n_beams, m.L, m.E, step
        self.max_rows = min(max_unique_rows, N * self.total)
        self.host_dev_gpu = host_on_gpu

        def host_empty(shape):
            return (
                torch.empty(shape, dtype=m.dtype, device=DEV)
                if host_on_gpu
                else pinned_empty(shape, m.dtype)
            )

        need = self.max_rows * m.beam_tok
        if not host_on_gpu:
            avail = mem_available_bytes()
            log(
                f"[host] need {need / GiB:.2f} GiB pinned (up to {self.max_rows} distinct tokens), "
                f"MemAvailable {avail / GiB:.2f} GiB"
            )
            if need + 3 * GiB > avail:
                raise MemoryError(
                    f"not enough host memory: need {need / GiB:.2f} GiB pinned + 3 GiB headroom, "
                    f"only {avail / GiB:.2f} GiB available"
                )
        self.pinned_bytes = 0 if host_on_gpu else need
        if mode == "lbs":
            self.cap = self.M // m.beam_tok  # GPU pool rows
            assert self.cap >= self.P + T, (
                "M too small: the prompt plus one beam's step does not fit"
            )
            self.arena = torch.empty(2 * L * self.cap * E, dtype=m.dtype, device=DEV)
            self.host = host_empty(
                (self.max_rows, 2 * L, E)
            )  # one row = all layers of one distinct token
            self.R = max(256, T)
            self.stage = torch.empty(self.R, 2 * L, E, dtype=m.dtype, device=DEV)
            self.tmp = torch.empty(self.cap, E, dtype=m.dtype, device=DEV)
        else:
            assert self.M // (L * m.layer_tok) >= self.P, (
                "M too small: the prompt KV does not fit on the GPU"
            )
            self.arena = torch.empty(
                self.M // m.elem + 2 * L * N * E, dtype=m.dtype, device=DEV
            )
            self.Lg = L
            self.cap = self._cap(L)
            self.hostK = [host_empty((self.max_rows, E)) for _ in range(L)]
            self.hostV = [host_empty((self.max_rows, E)) for _ in range(L)]
            self.scrK = torch.empty(self.max_rows, E, dtype=m.dtype, device=DEV)
            self.scrV = torch.empty(self.max_rows, E, dtype=m.dtype, device=DEV)
            self.tmp = self.scrK
        self.gum = None
        self.tokens = torch.zeros(N, self.total, dtype=torch.long, device=DEV)
        self.scores = torch.zeros(N, dtype=torch.float64, device=DEV)
        self.logits = None

    # ---------- arena ----------
    def _cap(self, n_layers):
        return (
            self.M // (n_layers * self.m.layer_tok) + self.N
        )  # one spare token (N rows) for new writes

    def plane(self, p):
        n = self.cap * self.m.E
        return self.arena[p * n : (p + 1) * n].view(self.cap, self.m.E)

    def planes3(self):
        L, E = self.m.L, self.m.E
        return self.arena[: 2 * L * self.cap * E].view(2 * L, self.cap, E)

    # ---------- mask ----------
    def _mask(self, g, base):
        G, T, bk = len(g), self.step, self.book
        vis = np.zeros((G, base + T * G), dtype=bool)
        for j, b in enumerate(g):
            for s in bk.chain[b]:
                sg = bk.segs[s]
                vis[j, sg.gpu : sg.gpu + sg.len] = True
        vis[:, base:] = np.tile(
            np.eye(G, dtype=bool), (1, T)
        )  # new rows: token t of beam j sits at base + t*G + j
        neg = torch.zeros(vis.shape, dtype=self.m.dtype, device=DEV)
        neg.masked_fill_(~torch.from_numpy(vis).to(DEV), float("-inf"))
        return neg

    # ---------- GPU pool compaction (one plane) ----------
    def _gpu_compact_plane(self, pl, plan):
        for src, dst, n in plan["moves"]:
            self.tmp[:n].copy_(pl[src : src + n])
            pl[dst : dst + n].copy_(self.tmp[:n])
            self.gpu_copy_bytes += n * self.m.plane_tok
        if plan["strided"]:
            o = 0
            for dst, base, stride, idx, n in plan["strided"]:
                self.tmp[o : o + n].copy_(pl[base + idx : base + n * stride : stride])
                o += n
            d0 = plan["strided"][0][0]
            pl[d0 : d0 + o].copy_(self.tmp[:o])
            self.gpu_copy_bytes += o * self.m.plane_tok

    @staticmethod
    def _host_compact_plane(pl, plan):
        for src, dst, n in plan["moves"]:
            move_rows(pl, src, dst, n)
        if plan["strided"]:
            parts = [
                pl[base + idx : base + n * stride : stride]
                for _, base, stride, idx, n in plan["strided"]
            ]
            tmp = torch.cat(parts)
            d0 = plan["strided"][0][0]
            pl[d0 : d0 + tmp.shape[0]].copy_(tmp)

    # ---------- prompt ----------
    def prefill(self):
        m, P = self.m, self.P
        ids = torch.tensor(self.prompt_ids, device=DEV)
        neg = torch.zeros(1, P, dtype=m.dtype, device=DEV)
        for s in range(P):
            h = m.embed(ids[s : s + 1], s)
            for l in range(m.L):
                h = layer_pool(
                    m, l, h, self.plane(2 * l), self.plane(2 * l + 1), s, neg
                )
        self.tokens[:, :P] = ids
        self.logits = m.head(h).expand(self.N, -1).contiguous()

    # ---------- one step for one group of beams ----------
    def _tokens(self, i, g, base):
        m, T, tm, N = self.m, self.step, self.timer, self.N
        G = len(g)
        gi = torch.tensor(g, device=DEV)
        neg = self._mask(g, base)
        lg = self.logits[gi]
        base_mode = self.mode == "baseline"
        for t in range(T):
            s = self.P + i * T + t
            forced = self.force["tok"][i][gi, t] if self.force is not None else None
            own, tok, lp = sample(lg, self.gum[gi, t], forced)
            if self.force is not None:
                fs = self.force_stats
                fs["samples"] += tok.numel()
                fs["own_differs"] += int((own != tok).sum())
                fs["max_lp_diff"] = max(
                    fs["max_lp_diff"],
                    (lp - self.force["lp"][i][gi, t]).abs().max().item(),
                )
            self.tokens[gi, s] = tok
            self.scores[gi] += lp
            self.step_lp[gi, t] = lp
            h = m.embed(tok, s)
            n0 = base + t * G
            L_in = m.L
            if base_mode:
                L_in = min(
                    m.L, self.M // (n0 * m.layer_tok)
                )  # Eq. 1, with the layer size taken over distinct rows
                if L_in < self.Lg:
                    self._evict_to(L_in, n0)
                elif L_in > self.Lg:
                    self._restore_to(L_in, n0)
                assert n0 + G <= self.max_rows, (
                    "distinct tokens exceed the reserved capacity (max_unique_rows)"
                )
                self.kv_gpu_peak = max(
                    self.kv_gpu_peak, (L_in + (L_in < m.L)) * (n0 + G) * m.layer_tok
                )
            for l in range(m.L):
                if l < L_in:
                    h = layer_pool(
                        m, l, h, self.plane(2 * l), self.plane(2 * l + 1), n0, neg
                    )
                else:
                    tm.switch("h2d")  # reload the whole layer: distinct content only
                    self.scrK[:n0].copy_(self.hostK[l][:n0])
                    self.scrV[:n0].copy_(self.hostV[l][:n0])
                    self.h2d_bytes += n0 * m.layer_tok
                    tm.switch("compute")
                    h = layer_pool(m, l, h, self.scrK, self.scrV, n0, neg)
                    tm.switch("d2h")
                    self.hostK[l][n0 : n0 + G].copy_(self.scrK[n0 : n0 + G])
                    self.hostV[l][n0 : n0 + G].copy_(self.scrV[n0 : n0 + G])
                    self.d2h_bytes += G * m.layer_tok
                    tm.switch("compute")
            lg = m.head(h)
        self.logits[gi] = lg

    # ---------- baseline: the number of GPU-resident layers changes ----------
    def _evict_to(self, L_new, n0):
        m, tm = self.m, self.timer
        tm.switch("d2h")
        for l in range(L_new, self.Lg):
            self.hostK[l][:n0].copy_(self.plane(2 * l)[:n0])
            self.hostV[l][:n0].copy_(self.plane(2 * l + 1)[:n0])
            self.d2h_bytes += n0 * m.layer_tok
        tm.switch("rearrange")
        if L_new > 0:
            c_new, E = self._cap(L_new), m.E
            for p in range(2 * L_new - 1, 0, -1):  # more rows per layer: planes move up
                src = self.arena[p * self.cap * E : p * self.cap * E + n0 * E]
                dst = self.arena[p * c_new * E : p * c_new * E + n0 * E]
                if p * (c_new - self.cap) < n0:
                    scr = self.scrV.view(-1)[: n0 * E]
                    scr.copy_(src)
                    dst.copy_(scr)
                else:
                    dst.copy_(src)
                self.gpu_copy_bytes += n0 * m.plane_tok
            self.cap = c_new
        self.Lg = L_new
        tm.switch("compute")

    def _restore_to(self, L_new, n0):
        """Pruning shrank the distinct content and more layers fit: move them back from the host
        (counted as host->GPU transfer)."""
        m, tm = self.m, self.timer
        tm.switch("rearrange")
        c_new, E = self._cap(L_new), m.E
        if self.Lg > 0:
            for p in range(1, 2 * self.Lg):  # fewer rows per layer: planes move down
                src = self.arena[p * self.cap * E : p * self.cap * E + n0 * E]
                dst = self.arena[p * c_new * E : p * c_new * E + n0 * E]
                if p * (self.cap - c_new) < n0:
                    scr = self.scrV.view(-1)[: n0 * E]
                    scr.copy_(src)
                    dst.copy_(scr)
                else:
                    dst.copy_(src)
                self.gpu_copy_bytes += n0 * m.plane_tok
        self.cap = c_new
        tm.switch("h2d")
        for l in range(self.Lg, L_new):
            self.plane(2 * l)[:n0].copy_(self.hostK[l][:n0])
            self.plane(2 * l + 1)[:n0].copy_(self.hostV[l][:n0])
            self.h2d_bytes += n0 * m.layer_tok
        self.Lg = L_new
        tm.switch("compute")

    # ---------- LBS: one group ----------
    def _lbs_group(self, i, g, multi):
        m, tm, T, bk = self.m, self.timer, self.step, self.book
        info = bk.open_group(g)
        assert bk.host_rows <= self.max_rows, (
            "distinct tokens exceed the reserved capacity (max_unique_rows)"
        )
        A = self.planes3()
        for goff, hoff, n in info[
            "flush"
        ]:  # segments leaving the GPU that the host lacks: write them out
            for r in range(0, n, self.R):
                k = min(self.R, n - r)
                tm.switch("rearrange")
                self.stage[:k].copy_(A[:, goff + r : goff + r + k].transpose(0, 1))
                tm.switch("d2h")
                self.host[hoff + r : hoff + r + k].copy_(self.stage[:k])
                self.d2h_bytes += k * m.beam_tok
        tm.switch("rearrange")
        if info["moves"]:
            plan = dict(moves=info["moves"], strided=[])
            for p in range(2 * m.L):
                self._gpu_compact_plane(A[p], plan)
        for hoff, goff, n in info[
            "load"
        ]:  # segments this group needs that are not on the GPU: load them
            for r in range(0, n, self.R):
                k = min(self.R, n - r)
                tm.switch("h2d")
                self.stage[:k].copy_(self.host[hoff + r : hoff + r + k])
                self.h2d_bytes += k * m.beam_tok
                tm.switch("rearrange")
                A[:, goff + r : goff + r + k].copy_(self.stage[:k].transpose(0, 1))
        if multi:
            self.h2d_bytes_no_retain += info["needed_rows"] * m.beam_tok
        base, G = info["base"], len(g)
        assert base + T * G <= self.cap
        self.kv_gpu_peak = max(self.kv_gpu_peak, (base + T * G) * m.beam_tok)
        tm.switch("compute")
        self._tokens(i, g, base)
        for j, hoff in bk.close_group(
            g, multi
        ):  # several groups: new segments go back to the host
            tm.switch("rearrange")
            self.stage[:T].copy_(A[:, base + j : base + T * G : G].transpose(0, 1))
            tm.switch("d2h")
            self.host[hoff : hoff + T].copy_(self.stage[:T])
            self.d2h_bytes += T * m.beam_tok
        assert bk.host_rows <= self.max_rows, (
            "distinct tokens exceed the reserved capacity (max_unique_rows)"
        )
        tm.switch("compute")

    # ---------- end of step: prune; no KV copies, only drop unused segments and compact ----------
    def _prune(self, i):
        m, tm, N, bk = self.m, self.timer, self.N, self.book
        tm.switch("prune")
        order = torch.sort(
            self.scores.cpu(), descending=True, stable=True
        ).indices.tolist()
        keep, drop = sorted(order[: N // 2]), sorted(order[N // 2 :])
        if self.force is not None and keep != self.force["keep"][i]:
            self.force_stats["prune_differs"] += 1
            keep = self.force["keep"][i]
            drop = sorted(set(range(N)) - set(keep))
        self.rec["keep"].append(keep)
        self.parent = list(range(N))
        for d, k in zip(drop, keep):
            self.parent[d] = k
        kt, dt = torch.tensor(keep, device=DEV), torch.tensor(drop, device=DEV)
        self.tokens[dt] = self.tokens[kt]
        self.scores[dt] = self.scores[kt]
        self.logits[dt] = self.logits[kt]
        plans = bk.prune(keep, drop)
        gp = plans["gpu"]
        if self.mode == "lbs":
            if gp is not None:
                A = self.planes3()
                for p in range(2 * m.L):
                    self._gpu_compact_plane(A[p], gp)
            for src, dst, n in plans["host_moves"]:
                move_rows(self.host, src, dst, n)
                self.host_copy_bytes += n * m.beam_tok
        else:
            for p in range(2 * self.Lg):
                self._gpu_compact_plane(self.plane(p), gp)
            if self.Lg < m.L:
                host_planes = self.hostK[self.Lg :] + self.hostV[self.Lg :]
                list(
                    self.pool.map(
                        lambda pl: self._host_compact_plane(pl, gp), host_planes
                    )
                )
                moved = sum(n for _, _, n in gp["moves"]) + sum(
                    x[4] for x in gp["strided"]
                )
                self.host_copy_bytes += (m.L - self.Lg) * moved * m.layer_tok
        self.unique_rows_peak_after_prune = max(
            self.unique_rows_peak_after_prune, bk.rows(bk.alive())
        )
        tm.switch("compute")

    def run(self):
        m, tm, bk, N, T = self.m, self.timer, self.book, self.N, self.step
        with torch.inference_mode():
            t0 = time.perf_counter()
            self.prefill()
            torch.cuda.synchronize()
            self.prefill_s = time.perf_counter() - t0
            torch.cuda.reset_peak_memory_stats()
            t_start = time.perf_counter()
            tm.switch("compute")
            for i in range(self.n_steps):
                gen = torch.Generator(device=DEV).manual_seed(self.seed * 1000003 + i)
                u = torch.rand(N, T, m.V, generator=gen, device=DEV)
                self.gum = u.log_().neg_().log_().neg_()
                del u
                self.step_lp = torch.zeros(N, T, dtype=torch.float64, device=DEV)
                rows_start = bk.rows(bk.alive())
                n_buf = bk.n_merged_buffers()
                self.unique_rows_peak = max(self.unique_rows_peak, rows_start + N * T)
                if self.mode == "lbs":
                    groups = bk.plan_groups(self.cap)
                    for g in groups:
                        self._lbs_group(i, g, len(groups) > 1)
                else:
                    groups = [list(range(N))]
                    info = bk.open_group(groups[0])
                    self._tokens(i, groups[0], info["base"])
                    bk.close_group(groups[0], False)
                bk.end_step()
                s0 = self.P + i * T
                self.rec["tok"].append(self.tokens[:, s0 : s0 + T].clone())
                self.rec["lp"].append(self.step_lp)
                self.rec["parent"].append(list(self.parent))
                self.rec["score"].append(self.scores.clone())
                if i == self.n_steps - 1:
                    order = torch.sort(
                        self.scores.cpu(), descending=True, stable=True
                    ).indices.tolist()
                    self.rec["keep"].append(sorted(order[: N // 2]))
                else:
                    self._prune(i)
                tm.switch("compute")
                rec = dict(
                    step=i,
                    s_end=s0 + T,
                    n_groups=len(groups),
                    unique_rows_start=rows_start,
                    n_segments=len(bk.segs),
                    n_merged_buffers=n_buf,
                    L_in_gpu=getattr(self, "Lg", m.L),
                    wall=time.perf_counter() - t_start,
                    h2d_bytes=self.h2d_bytes,
                    d2h_bytes=self.d2h_bytes,
                    **{k: round(v, 4) for k, v in tm.t.items()},
                )
                self.steps_log.append(rec)
                if (i + 1) % max(1, self.n_steps // 12) == 0 or i == self.n_steps - 1:
                    self.log(
                        f"[dedup {self.mode} N={N}] step {i + 1}/{self.n_steps} s={s0 + T} wall={rec['wall']:.1f}s "
                        f"h2d={self.h2d_bytes / GiB:.1f}GiB distinct={rows_start * m.beam_tok / GiB:.2f}GiB "
                        f"groups={len(groups)} L_in={rec['L_in_gpu']} "
                        + " ".join(f"{k}={v:.1f}" for k, v in tm.t.items())
                    )
            tm.switch(None)
            self.total_s = time.perf_counter() - t_start
        return self.result()

    def result(self):
        tok = self.tokens.cpu()
        scores = self.scores.cpu().tolist()
        t = dict(self.timer.t)
        m = self.m
        return dict(
            store="dedup",
            mode=self.mode,
            n_beams=self.N,
            step=self.step,
            prompt_len=self.P,
            n_gen=self.n_gen,
            mem_gib=self.M / GiB,
            seed=self.seed,
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
            h2d_bytes_no_retain=self.h2d_bytes_no_retain,
            pinned_bytes=self.pinned_bytes,
            gpu_peak_alloc_bytes=torch.cuda.max_memory_allocated(),
            kv_gpu_peak_bytes=self.kv_gpu_peak,
            unique_kv_peak_bytes=self.unique_rows_peak * m.beam_tok,
            unique_kv_peak_after_prune_bytes=self.unique_rows_peak_after_prune
            * m.beam_tok,
            max_groups=max(s["n_groups"] for s in self.steps_log),
            max_merged_buffers=max(s["n_merged_buffers"] for s in self.steps_log),
            buffers_per_layer_used=1,
            tokens_sha256=hashlib.sha256(tok.numpy().tobytes()).hexdigest(),
            scores=scores,
            best_beam=max(range(self.N), key=lambda b: scores[b]),
            steps=self.steps_log,
            force_stats=self.force_stats,
            _tokens=tok,
        )

    tree = PerBeamEngine.tree
    save_tree = PerBeamEngine.save_tree

    def close(self):
        self.pool.shutdown()
        if not self.host_dev_gpu:
            for t in (
                ([self.host] if hasattr(self, "host") else [])
                + getattr(self, "hostK", [])
                + getattr(self, "hostV", [])
            ):
                pinned_free(t)
        for name in (
            "arena",
            "scrK",
            "scrV",
            "stage",
            "tmp",
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
