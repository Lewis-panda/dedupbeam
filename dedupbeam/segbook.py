"""Bookkeeping for the deduplicated KV store (pure Python, no GPU).

Terms:
- segment: the KV one candidate writes during one step (T tokens), or the prompt (P tokens). A segment never changes
  after it is written, so many beams can share it, and it can have one copy on the GPU and one on the host at once.
- chain[b]: the segments beam b has passed through, oldest first.
- pool: where segments live on the GPU (or host); one row = one distinct token.

SegBook only tracks which segment sits in which rows and which rows each action has to move; dedup.py moves the data.
The same bookkeeping replays a recorded beam tree without a GPU (simulate_*), which predicts the transfer volume.
"""


class Seg:
    __slots__ = ("len", "gpu", "host", "il")

    def __init__(self, n):
        self.len = n
        self.gpu = None  # first row in the GPU pool (len contiguous rows)
        self.host = None  # first row in the host pool
        self.il = None  # (base, stride, idx): a segment being written this step is interleaved on the GPU


def _coalesce(moves):
    out = []
    for src, dst, n in moves:
        if out and out[-1][0] + out[-1][2] == src and out[-1][1] + out[-1][2] == dst:
            out[-1] = (out[-1][0], out[-1][1], out[-1][2] + n)
        else:
            out.append((src, dst, n))
    return out


class SegBook:
    def __init__(self, n_beams, prompt_len, step):
        self.N, self.P, self.T = n_beams, prompt_len, step
        self.segs = {0: Seg(prompt_len)}
        self.segs[0].gpu = 0
        self.chain = [[0] for _ in range(n_beams)]
        self.gpu_rows = prompt_len  # rows laid out contiguously in the GPU pool (excluding this step's writes)
        self.host_rows = 0
        self.next_id = 1
        self.cur = {}  # beam -> this step's segment

    # ---- queries ----
    def alive(self):
        return {s for c in self.chain for s in c}

    def rows(self, ids):
        return sum(self.segs[s].len for s in ids)

    def union(self, beams):
        ids = set()
        for b in beams:
            ids.update(self.chain[b])
        return ids

    def n_merged_buffers(self):
        """How many buffers it would take if contiguous segments shared by the same beams were merged into one."""
        owners = {}
        for b, c in enumerate(self.chain):
            for s in c:
                owners.setdefault(s, []).append(b)
        return len({tuple(v) for v in owners.values()})

    # ---- grouping ----
    def plan_groups(self, budget_rows):
        """One group (in slot order) if everything fits. Otherwise walk the beams in beam-tree order (beams that share
        the most are adjacent) and start a new group when "distinct old segments + this step's new segments" would
        exceed the budget."""
        N, T = self.N, self.T
        if self.rows(self.alive()) + N * T <= budget_rows:
            return [list(range(N))]
        order = sorted(range(N), key=lambda b: self.chain[b])
        groups, g, u = [], [], set()
        for b in order:
            nu = u | set(self.chain[b])
            if g and self.rows(nu) + (len(g) + 1) * T > budget_rows:
                groups.append(g)
                g, u = [b], set(self.chain[b])
            else:
                g.append(b)
                u = nu
            assert self.rows(u) + len(g) * T <= budget_rows, (
                "M cannot hold even one beam"
            )
        groups.append(g)
        return groups

    # ---- GPU pool compaction: keep only keep_ids, packed to the front ----
    def gpu_compact(self, keep_ids):
        compact = sorted(
            (s for s in keep_ids if self.segs[s].gpu is not None),
            key=lambda s: self.segs[s].gpu,
        )
        inter = sorted(
            (s for s in keep_ids if self.segs[s].il is not None),
            key=lambda s: self.segs[s].il[2],
        )
        pos, moves, strided = 0, [], []
        for s in compact:
            sg = self.segs[s]
            if sg.gpu != pos:
                moves.append((sg.gpu, pos, sg.len))
            sg.gpu = pos
            pos += sg.len
        for s in inter:
            sg = self.segs[s]
            strided.append((pos, sg.il[0], sg.il[1], sg.il[2], sg.len))
            sg.gpu, sg.il = pos, None
            pos += sg.len
        keep = set(keep_ids)
        for s, sg in self.segs.items():
            if s not in keep:
                sg.gpu, sg.il = None, None
        self.gpu_rows = pos
        return dict(moves=_coalesce(moves), strided=strided, rows=pos)

    def host_compact(self, alive):
        on_host = sorted(
            (s for s in alive if self.segs[s].host is not None),
            key=lambda s: self.segs[s].host,
        )
        pos, moves = 0, []
        for s in on_host:
            sg = self.segs[s]
            if sg.host != pos:
                moves.append((sg.host, pos, sg.len))
            sg.host = pos
            pos += sg.len
        self.host_rows = pos
        return _coalesce(moves)

    # ---- start of a group: decide what to write back, what to keep, what to load ----
    def open_group(self, g):
        needed = self.union(g)
        alive = self.alive()
        flush = []  # (gpu row, host row, rows): alive segments leaving the GPU that the host does not have yet
        for s, sg in self.segs.items():
            if (
                sg.gpu is not None
                and s not in needed
                and s in alive
                and sg.host is None
            ):
                sg.host = self.host_rows
                self.host_rows += sg.len
                flush.append((sg.gpu, sg.host, sg.len))
        keep = [s for s in needed if self.segs[s].gpu is not None]
        plan = self.gpu_compact(keep)
        load = []  # (host row, gpu row, rows)
        for s in sorted(
            (s for s in needed if self.segs[s].gpu is None),
            key=lambda s: self.segs[s].host,
        ):
            sg = self.segs[s]
            assert sg.host is not None
            sg.gpu = self.gpu_rows
            load.append((sg.host, sg.gpu, sg.len))
            self.gpu_rows += sg.len
        base = self.gpu_rows
        for j, b in enumerate(g):
            sg = Seg(self.T)
            sg.il = (base, len(g), j)
            self.segs[self.next_id] = sg
            self.cur[b] = self.next_id
            self.next_id += 1
        return dict(
            flush=_coalesce(flush),
            moves=plan["moves"],
            load=_coalesce(load),
            base=base,
            needed_rows=self.rows(needed),
        )

    def close_group(self, g, write_back):
        """write_back=True (more than one group this step): new segments go to the host and leave the GPU.
        Returns [(j, host row)]."""
        out = []
        if write_back:
            for j, b in enumerate(g):
                sg = self.segs[self.cur[b]]
                sg.host = self.host_rows
                self.host_rows += sg.len
                sg.il = None
                out.append((j, sg.host))
        return out

    def end_step(self):
        for b in range(self.N):
            self.chain[b].append(self.cur[b])
        self.cur = {}

    def prune(self, keep, drop):
        """Survivors stay in place; dropped slot drop[j] becomes a copy of keep[j] (it points at the same segments;
        no KV is copied)."""
        for d, k in zip(drop, keep):
            self.chain[d] = list(self.chain[k])
        alive = self.alive()
        gpu_plan = None
        if any(
            sg.il is not None for sg in self.segs.values()
        ):  # single group this step: new segments still interleaved
            gpu_plan = self.gpu_compact(
                [
                    s
                    for s in alive
                    if self.segs[s].gpu is not None or self.segs[s].il is not None
                ]
            )
        host_moves = self.host_compact(alive)
        for s in [s for s in self.segs if s not in alive]:
            del self.segs[s]
        return dict(gpu=gpu_plan, host_moves=host_moves)


# ======================= GPU-free replay of a recorded beam tree =======================
def _replay(parent, kept, P, T):
    """Replay a beam tree, yielding (step index, SegBook); the caller drives the book between yields."""
    S, N = parent.shape
    book = SegBook(N, P, T)
    for i in range(S):
        yield i, book
        if i < S - 1:
            keep = [n for n in range(N) if kept[i, n]]
            drop = [n for n in range(N) if not kept[i, n]]
            book.prune(keep, drop)


def unique_rows_profile(parent, kept, P, T):
    """Distinct rows alive at the start of each step (excluding the N*T rows the step will write)."""
    out = []
    for i, book in _replay(parent, kept, P, T):
        out.append(book.rows(book.alive()))
        book.open_group(list(range(book.N)))
        book.close_group(list(range(book.N)), False)
        book.end_step()
    return out


def simulate_baseline(parent, kept, P, T, M, layer_tok, n_layer=32):
    """Dedup baseline: per token L_in = min(n_layer, floor(M / (distinct rows * bytes per row per layer))).
    Returns host->GPU bytes: layers reloaded every token + layers moved back once pruning frees room."""
    N = parent.shape[1]
    total, Lg = 0, n_layer
    for a in unique_rows_profile(parent, kept, P, T):
        for t in range(T):
            n0 = a + t * N
            L_in = min(n_layer, M // (n0 * layer_tok))
            if L_in > Lg:
                total += (L_in - Lg) * n0 * layer_tok
            Lg = L_in
            total += (n_layer - L_in) * n0 * layer_tok
    return total


def simulate_lbs(parent, kept, P, T, M, beam_tok):
    """Dedup LBS. Returns (host->GPU bytes, groups per step, host->GPU bytes if shared segments were not retained)."""
    budget = M // beam_tok
    h2d, no_retain, groups_per_step = 0, 0, []
    for i, book in _replay(parent, kept, P, T):
        groups = book.plan_groups(budget)
        groups_per_step.append(len(groups))
        for g in groups:
            info = book.open_group(g)
            h2d += sum(n for _, _, n in info["load"]) * beam_tok
            if len(groups) > 1:
                no_retain += info["needed_rows"] * beam_tok
            book.close_group(g, len(groups) > 1)
        book.end_step()
    return h2d, groups_per_step, no_retain
