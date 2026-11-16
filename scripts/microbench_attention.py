"""One decode attention layer at 32 beams, FP16, three ways. Writes results/microbench_attention.json.

(a) per-beam: K, V of shape [2048, 32 beams, 2560] (the per-beam store, as in PerBeamEngine).
(b) pool + mask: one shared pool K, V of shape [U, 2560] (U distinct tokens), scores [head, beam, U], masked rows
    get -inf, one softmax, one weighted sum (as in DedupEngine).
(c) n buffers: the same pool split into n contiguous buffers, one bmm per buffer for the scores and for the output,
    one softmax over the concatenated scores (the "one buffer per shared segment run" alternative).
The timed region is the attention core (scores, mask, softmax, weighted sum); projections are excluded.
Usage: python scripts/microbench_attention.py
"""

import json, time

import _common  # noqa: F401  (sets up paths)
import torch

DEV = "cuda"
H, D, E, B = 32, 80, 2560, 32


def bench(fn, n=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main():
    torch.manual_seed(0)
    q = torch.randn(B, E, device=DEV, dtype=torch.half)
    out = dict(
        beams=B,
        heads=H,
        head_dim=D,
        dtype="float16",
        gpu=torch.cuda.get_device_name(0),
        rows=[],
    )
    s = 2048
    Ka = torch.randn(s, B, E, device=DEV, dtype=torch.half)
    Va = torch.randn_like(Ka)

    def per_beam():
        Kv = Ka.view(s, B * H, D).permute(1, 0, 2)
        Vv = Va.view(s, B * H, D).permute(1, 0, 2)
        w = torch.bmm(Kv, q.view(B * H, D, 1))
        w = torch.softmax(w, dim=1, dtype=torch.float32).half()
        return torch.bmm(w.transpose(1, 2), Vv).view(B, E)

    out["per_beam_ms"] = bench(per_beam)
    out["per_beam_tokens"] = s
    out["per_beam_kv_gib"] = 2 * Ka.numel() * 2 / 2**30
    del Ka, Va
    for U in (2048, 5200, 9000, 14500):
        K = torch.randn(U, E, device=DEV, dtype=torch.half)
        V = torch.randn_like(K)
        neg = torch.zeros(B, U, device=DEV, dtype=torch.half)
        neg.masked_fill_(torch.rand(B, U, device=DEV) < 0.3, float("-inf"))

        def pool():
            Kt = K.view(U, H, D).permute(1, 2, 0)
            Vv = V.view(U, H, D).permute(1, 0, 2)
            w = torch.bmm(q.view(B, H, D).transpose(0, 1), Kt)
            w += neg
            w = torch.softmax(w, dim=-1, dtype=torch.float32).half()
            return torch.bmm(w, Vv).transpose(0, 1).reshape(B, E)

        row = dict(
            pool_rows=U, pool_kv_gib=2 * K.numel() * 2 / 2**30, pool_mask_ms=bench(pool)
        )
        for n_buf in (8, 63):
            cuts = torch.linspace(0, U, n_buf + 1).long().tolist()

            def buffers():
                qs = q.view(B, H, D).transpose(0, 1)
                ws = [
                    torch.bmm(qs, K[a:b].view(b - a, H, D).permute(1, 2, 0))
                    for a, b in zip(cuts, cuts[1:])
                ]
                w = torch.cat(ws, -1)
                w += neg
                w = torch.softmax(w, dim=-1, dtype=torch.float32).half()
                o = 0
                for a, b in zip(cuts, cuts[1:]):
                    o = o + torch.bmm(
                        w[:, :, a:b], V[a:b].view(b - a, H, D).permute(1, 0, 2)
                    )
                return o.transpose(0, 1).reshape(B, E)

            row[f"buffers_{n_buf}_ms"] = bench(buffers)
        out["rows"].append(row)
        print(row)
        del K, V, neg
    print(f"per-beam (2048 tokens x 32 beams): {out['per_beam_ms']:.3f} ms")
    json.dump(out, open("results/microbench_attention.json", "w"), indent=1)


if __name__ == "__main__":
    main()
