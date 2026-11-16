"""Host->GPU transfer volume predicted by the LBS paper's analytical model (Eq. 1-5). Units: bytes.

Defaults are OPT-2.7B in FP16: KV per token, per layer, per beam = 2 x 2560 x 2 = 10,240 bytes; all 32 layers =
327,680 bytes. Pass n_layer / layer_tok / beam_tok for another model (Qwen2.5-3B: 36 layers, 1,024 bytes per layer).
"""

import math

N_LAYER = 32
LAYER_TOK = 2 * 2560 * 2
BEAM_TOK = N_LAYER * LAYER_TOK
GiB = 2**30


def l_in_gpu(n_beams, s, mem, n_layer=N_LAYER, layer_tok=LAYER_TOK):  # Eq. 1
    return min(n_layer, mem // (n_beams * s * layer_tok))


def baseline_dm(
    n_beams, mem, prompt=128, n_gen=1920, n_layer=N_LAYER, layer_tok=LAYER_TOK
):  # Eq. 2, 3
    return sum(
        (n_layer - l_in_gpu(n_beams, s, mem, n_layer, layer_tok))
        * n_beams
        * s
        * layer_tok
        for s in range(prompt, prompt + n_gen)
    )


def lbs_dm(
    n_beams, mem, prompt=128, n_gen=1920, step=32, s_at="start", beam_tok=BEAM_TOK
):  # Eq. 4, 5
    """Sum over the steps whose KV, at the end of the step, exceeds `mem`.

    s_at="start": the paper's Eq. 5, s = i * N_step + S_prompt (KV that exists when the step starts; what is moved).
    s_at="end"  : s = (i+1) * N_step + S_prompt (also counts the KV the step itself produces; overestimates by ~2%).
    """
    total = 0
    for i in range(math.ceil(n_gen / step)):
        s0 = i * step + prompt
        s_end = s0 + step
        if n_beams * s_end * beam_tok > mem:
            total += n_beams * (s0 if s_at == "start" else s_end) * beam_tok
    return total


def first_offload_len(n_beams, mem, beam_tok=BEAM_TOK):
    """Longest length at which all KV still fits."""
    return mem // (n_beams * beam_tok)


if __name__ == "__main__":
    M = 8 * GiB
    print("beams  fits up to  baseline (GiB)  LBS Eq.5 (GiB)  ratio")
    for n in (8, 16, 24, 32):
        b, l = baseline_dm(n, M), lbs_dm(n, M)
        r = f"{b / l:.1f}" if l else "-"
        print(
            f"{n:<6d}{first_offload_len(n, M):>10d}{b / GiB:>16.1f}{l / GiB:>16.1f}{r:>7s}"
        )
