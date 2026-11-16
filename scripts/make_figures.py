"""Draw the figures in figures/ from the result JSON files. Standard library only.

Every figure is written as SVG; a PNG at 200 dpi is rendered from it with the first tool found among rsvg-convert,
cairosvg, and a headless Chrome/Chromium (set CHROME=/path/to/chrome to pick one). Without any of them only SVGs are
written. Usage: python scripts/make_figures.py
"""

import math, os, shutil, subprocess
from html import escape

import _data as D

FONT = "DejaVu Sans, Liberation Sans, Arial, Helvetica, sans-serif"
C = dict(
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    blue="#2a78d6",
    orange="#eb6834",
    aqua="#1baf7a",
    gray="#a3a29c",
    host="#e9e8e3",
)
# Fixed meaning across the figure set: blue = deduplicated KV, orange = one KV copy per beam.
DEDUP, PERBEAM = C["blue"], C["orange"]
GPU = "RTX 5060 Ti 16 GB"
DPI = 200
OUT = "figures"


def tw(s, size):
    """Approximate rendered width of text (DejaVu Sans averages ~0.58 em per character)."""
    return len(s) * size * 0.58


def fmt(x, d=1):
    return f"{x:,.{d}f}"


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.el = w, h, []

    def text(
        self,
        x,
        y,
        s,
        size=13,
        color=None,
        weight="normal",
        anchor="start",
        italic=False,
        halo=False,
    ):
        style = ' font-style="italic"' if italic else ""
        if halo:  # a surface-colored outline keeps a label legible where it crosses a line
            style += f' stroke="{C['surface']}" stroke-width="4" stroke-linejoin="round" paint-order="stroke"'
        self.el.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color or C["ink"]}" text-anchor="{anchor}"{style}>{escape(s)}</text>'
        )

    def rect(self, x, y, w, h, fill, rx=0, stroke=None, sw=1):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.el.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" rx="{rx}" fill="{fill}"{st}/>'
        )

    def line(self, x1, y1, x2, y2, color=None, width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.el.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color or C["axis"]}" stroke-width="{width}"{d}/>'
        )

    def circle(self, cx, cy, r, fill, stroke=None, sw=2):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.el.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"{st}/>'
        )

    def path(self, d, fill="none", stroke=None, sw=1):
        st = (
            f' stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"'
            if stroke
            else ""
        )
        self.el.append(f'<path d="{d}" fill="{fill}"{st}/>')

    def hbar(self, x, y, length, thick, fill, round_end=True):
        """Horizontal bar from baseline x; 4px rounded data end, square at the baseline."""
        if length <= 0:
            return
        r = min(4, length / 2, thick / 2) if round_end else 0
        x2, y2 = x + length, y + thick
        self.path(
            f"M{x:.1f},{y:.1f} H{x2 - r:.1f} Q{x2:.1f},{y:.1f} {x2:.1f},{y + r:.1f} V{y2 - r:.1f} "
            f"Q{x2:.1f},{y2:.1f} {x2 - r:.1f},{y2:.1f} H{x:.1f} Z",
            fill=fill,
        )

    def vbar(self, x, ybase, length, width, fill):
        """Vertical bar growing up from ybase; 4px rounded top, square at the baseline."""
        if length <= 0:
            return
        r = min(4, length / 2, width / 2)
        y = ybase - length
        self.path(
            f"M{x:.1f},{ybase:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + width - r:.1f} "
            f"Q{x + width:.1f},{y:.1f} {x + width:.1f},{y + r:.1f} V{ybase:.1f} Z",
            fill=fill,
        )

    def header(self, title, subtitle, legend=None, x=24):
        self.text(x, 36, title, 19, weight="bold")
        self.text(x, 60, subtitle, 13, C["ink2"])
        if legend:
            lx = x
            for color, label, kind in legend:
                if kind == "ring":
                    self.circle(lx + 6, 84, 4.5, C["surface"], stroke=C["ink2"], sw=2)
                elif kind == "dot":
                    self.circle(lx + 6, 84, 5, color)
                else:
                    self.rect(lx, 78, 12, 12, color, rx=3)
                self.text(lx + 18, 88.5, label, 12.5, C["ink2"])
                lx += 18 + tw(label, 12.5) + 22

    def footnote(self, lines, y=None, x=24):
        y = y or self.h - 14 - 16 * (len(lines) - 1)
        for i, s in enumerate(lines):
            self.text(x, y + 16 * i, s, 11.5, C["muted"])

    def save(self, name):
        os.makedirs(OUT, exist_ok=True)
        body = "\n".join(self.el)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">\n'
            f'<rect width="{self.w}" height="{self.h}" fill="{C["surface"]}"/>\n{body}\n</svg>\n'
        )
        p = os.path.join(OUT, name + ".svg")
        open(p, "w").write(svg)
        png = to_png(p, self.w, self.h)
        print(f"wrote {p}" + (f" and {png}" if png else " (no PNG renderer found)"))


def to_png(svg_path, w, h):
    png = svg_path[:-4] + ".png"
    scale = DPI / 96
    if shutil.which("rsvg-convert"):
        subprocess.run(
            [
                "rsvg-convert",
                "-d",
                str(DPI),
                "-p",
                str(DPI),
                "-z",
                f"{scale:.4f}",
                "-o",
                png,
                svg_path,
            ],
            check=True,
        )
        return png
    try:
        import cairosvg

        cairosvg.svg2png(url=svg_path, write_to=png, scale=scale)
        return png
    except ImportError:
        pass
    chrome = os.environ.get("CHROME") or next(
        (
            shutil.which(c)
            for c in (
                "chromium",
                "chromium-browser",
                "google-chrome",
                "google-chrome-stable",
                "chrome",
            )
            if shutil.which(c)
        ),
        None,
    )
    if not chrome:
        return None
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                    f"--force-device-scale-factor={scale:.6f}", f"--window-size={w},{h}",
                    f"--screenshot={os.path.abspath(png)}", "file://" + os.path.abspath(svg_path)],
                   check=True, capture_output=True, timeout=120)
    return png


def nice_ticks(vmax, n=5):
    raw = vmax / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return [
        i * step
        for i in range(int(vmax / step) + 2)
        if i * step <= vmax * 1.0001
    ]


def axis_label(v):
    return f"{v:,.0f}" if v >= 10 or v == int(v) else f"{v:g}"


SUB_DEDUP = f"OPT-2.7B FP16 · 32 beams · prompt 128 + 1,920 generated tokens · {GPU}"


# ---------------------------------------------------------------- 1. mechanism
def fig_mechanism():
    W, H = 1110, 560
    s = SVG(W, H)
    s.header(
        "DedupBeam keeps one GPU copy of each shared KV segment instead of one per beam",
        "Example with 4 beams: segments A (prompt) and 1 are shared by all beams, 2 by beams 1–2, 3 by beams 3–4.",
    )
    beams = [("beam 1", "A12"), ("beam 2", "A12"), ("beam 3", "A13"), ("beam 4", "A13")]
    segs = "A123"
    bw, gap = 54, 8
    rows = dict(host=130, pcie=196, gpu=262)
    for label, y in (
        ("Host memory", rows["host"] + 26),
        ("PCIe", rows["pcie"] + 22),
        ("GPU memory", rows["gpu"] + 26),
    ):
        s.text(24, y, label, 13.5, weight="bold")
    panels = [
        (
            235,
            "LBS as described in the paper",
            "Footnote 2: “each beam still maintains a separate copy”",
        ),
        (
            690,
            "DedupBeam (this repository)",
            "One copy per segment on the host and on the GPU",
        ),
    ]
    for px, title, note in panels:
        s.text(px, 108, title, 14.5, weight="bold")
        s.text(px, 124, note, 11.5, C["ink2"])
        for i, g in enumerate(segs):  # host: one copy each in both designs
            x = px + i * (bw + gap)
            s.rect(x, rows["host"] + 6, bw, 30, C["host"], rx=4)
            s.text(
                x + bw / 2,
                rows["host"] + 26,
                g,
                14,
                C["ink"],
                weight="bold",
                anchor="middle",
            )
        s.text(
            px + 4 * (bw + gap) + 6, rows["host"] + 26, "one copy each", 12, C["ink2"]
        )
        for i in range(4):  # PCIe arrows
            cx = px + i * (bw + gap) + bw / 2
            s.line(cx, rows["pcie"] - 22, cx, rows["pcie"] + 26, C["muted"], 1.5)
            s.path(
                f"M{cx - 5},{rows['pcie'] + 20} L{cx},{rows['pcie'] + 28} L{cx + 5},{rows['pcie'] + 20}",
                stroke=C["muted"],
                sw=1.5,
            )
    s.text(
        235 + 4 * (bw + gap) + 6, rows["pcie"] + 6, "each segment sent", 12, C["ink2"]
    )
    s.text(235 + 4 * (bw + gap) + 6, rows["pcie"] + 22, "once per group", 12, C["ink2"])
    s.text(
        690 + 4 * (bw + gap) + 6, rows["pcie"] + 6, "only segments not", 12, C["ink2"]
    )
    s.text(
        690 + 4 * (bw + gap) + 6, rows["pcie"] + 22, "already on the GPU", 12, C["ink2"]
    )
    # GPU, paper: one row of copies per beam
    px, y0, rh = 235, rows["gpu"] + 6, 30
    for r, (bname, chain) in enumerate(beams):
        y = y0 + r * (rh + 8)
        s.text(px - 8, y + 20, bname, 12, C["ink2"], anchor="end")
        for g in chain:
            i = segs.index(g)
            x = px + i * (bw + gap)
            s.rect(x, y, bw, rh, PERBEAM, rx=4)
            s.text(x + bw / 2, y + 20, g, 14, "#ffffff", weight="bold", anchor="middle")
    counts = {g: sum(g in c for _, c in beams) for g in segs}
    ycount = y0 + 4 * (rh + 8) + 18
    for i, g in enumerate(segs):
        s.text(
            px + i * (bw + gap) + bw / 2,
            ycount,
            f"×{counts[g]}",
            13,
            C["ink"],
            weight="bold",
            anchor="middle",
        )
    total = sum(counts.values())
    s.text(
        px,
        ycount + 30,
        f"GPU holds {total} segment copies",
        13.5,
        C["ink"],
        weight="bold",
    )
    # GPU, dedup: one copy each + who reads what
    px = 690
    for i, g in enumerate(segs):
        x = px + i * (bw + gap)
        s.rect(x, y0, bw, rh, DEDUP, rx=4)
        s.text(x + bw / 2, y0 + 20, g, 14, "#ffffff", weight="bold", anchor="middle")
    s.text(px + 4 * (bw + gap) + 6, y0 + 20, "stored once", 12, C["ink2"])
    s.text(
        px, y0 + rh + 22, "Attention mask (● read, ○ masked to −inf):", 11.5, C["ink2"]
    )
    for r, (bname, chain) in enumerate(beams):
        y = y0 + rh + 44 + r * 20
        s.text(px - 8, y + 4, bname, 12, C["ink2"], anchor="end")
        for i, g in enumerate(segs):
            cx = px + i * (bw + gap) + bw / 2
            if g in chain:
                s.circle(cx, y, 5, DEDUP)
            else:
                s.circle(cx, y, 4.5, C["surface"], stroke=C["axis"], sw=1.5)
    s.text(
        px,
        ycount + 30,
        f"GPU holds {len(segs)} segments ({total / len(segs):.0f}× fewer)",
        13.5,
        C["ink"],
        weight="bold",
    )
    s.footnote(
        [
            "In the experiments the shared prefix is long: at the end (step 32) all 32 beams of OPT-2.7B share their first 1,697 tokens (128 prompt + 1,569",
            "generated), so one copy per beam needs 20 GiB of KV while the distinct KV never exceeds 1.58 GiB.",
        ]
    )
    s.save("fig1_mechanism")


# ---------------------------------------------------------------- 2. KV peak
def fig_kv_peak():
    R = D.dedup_runs()
    peaks = {
        T: (R[T][("dedup", "baseline")] or {}).get("unique_kv_peak_bytes", 0) / D.GiB
        for T in D.STEPS
    }
    per = D.percopy_kv_bytes() / D.GiB
    W, H = 860, 520
    s = SVG(W, H)
    s.header(
        f"Deduplicated KV peaks at {min(peaks.values()):.2f}–{max(peaks.values()):.2f} GiB; one copy per beam needs {per:.0f} GiB",
        SUB_DEDUP,
        legend=[
            (PERBEAM, "One KV copy per beam", "box"),
            (DEDUP, "Deduplicated (distinct KV)", "box"),
        ],
    )
    x0, x1, yb, yt = 90, W - 150, H - 90, 118
    vmax = 22
    sy = lambda v: yb - (v / vmax) * (yb - yt)
    for v in nice_ticks(vmax, 4):
        s.line(x0, sy(v), x1, sy(v), C["grid"])
        s.text(x0 - 10, sy(v) + 4, axis_label(v), 12, C["muted"], anchor="end")
    s.text(24, yt - 8, "GiB", 12, C["muted"])
    s.line(x0, sy(D.M_MAIN), x1 + 6, sy(D.M_MAIN), C["ink"], 1.5, dash="5 4")
    gw = (x1 - x0) / 3
    for gi, T in enumerate(D.STEPS):
        cx = x0 + gw * gi + gw / 2
        for j, (v, col) in enumerate(((per, PERBEAM), (peaks[T], DEDUP))):
            bx = cx - 26 + j * 28
            s.vbar(bx, yb, yb - sy(v), 24, col)
            s.text(
                bx + 12,
                sy(v) - 7,
                f"{v:.2f}" if v < 10 else f"{v:.0f}",
                12.5,
                C["ink"],
                anchor="middle",
                halo=True,
            )
        s.text(cx, yb + 22, f"step {T}", 13, C["ink2"], anchor="middle")
    s.line(x0, yb, x1, yb, C["axis"])
    s.text(
        x1 + 10, sy(D.M_MAIN) + 4, f"M = {D.M_MAIN} GiB", 12.5, C["ink"], weight="bold"
    )
    s.text(x1 + 10, sy(D.M_MAIN) + 20, "GPU memory for KV", 11.5, C["ink2"])
    s.footnote(
        [
            "One copy per beam: 32 beams × 2,048 tokens × 320 KiB at the end of the search.",
            "Deduplicated: highest distinct KV over the run, measured after each step is written and before pruning (GPU + host, each segment once).",
        ]
    )
    s.save("fig2_kv_peak")


# ---------------------------------------------------------------- 3. end-to-end time
def fig_end_to_end():
    R = D.dedup_runs()
    dd = [R[T][(st, m)]["total_s"] for T in D.STEPS for st, m in (("dedup", "baseline"), ("dedup", "lbs")) if D.ok(R[T][(st, m)])]
    pb = [R[T][("percopy", "baseline")]["total_s"] for T in D.STEPS if D.ok(R[T][("percopy", "baseline")])]
    lx, x0, x1 = 24, 250, 870
    thick, pitch, facet_gap = 20, 30, 14
    top = 120
    y_end = top + len(D.STEPS) * (22 + len(D.METHODS) * pitch + facet_gap)
    W, H = 980, y_end + 70
    s = SVG(W, H)
    s.header(f"Deduplicated runs finish in {min(dd):.0f}–{max(dd):.0f} s; the per-beam baseline takes about {sum(pb) / len(pb):,.0f} s",
             f"{SUB_DEDUP} · M = {D.M_MAIN} GiB",
             legend=[(C["blue"], "Compute", "box"), (C["orange"], "Host→GPU transfer", "box"),
                     (C["aqua"], "Other (GPU→host, rearrange, prune)", "box")])
    vmax = max(r["total_s"] for T in D.STEPS for r in R[T].values() if D.ok(r))
    vmax = math.ceil(vmax / 200) * 200
    sx = lambda v: x0 + v / vmax * (x1 - x0)
    for v in nice_ticks(vmax, 5):  # grid first, so the bars sit on top of it
        s.line(sx(v), top + 18, sx(v), y_end - 10, C["grid"])
        s.text(sx(v), y_end + 6, axis_label(v), 12, C["muted"], anchor="middle")
    s.line(x0, top + 18, x0, y_end - 10, C["axis"])
    s.text((x0 + x1) / 2, y_end + 26, "Total time (s)", 12.5, C["muted"], anchor="middle")
    y = top
    for T in D.STEPS:
        s.text(lx, y + 12, f"Step {T}", 13.5, weight="bold")
        y += 22
        for st, m, label in D.METHODS:
            r = R[T][(st, m)]
            s.text(x0 - 10, y + 15, label, 12.5, C["ink2"], anchor="end")
            if not D.ok(r):
                s.text(x0 + 4, y + 15, "not run", 12, C["muted"])
                y += pitch
                continue
            parts = [(r["compute_s"], C["blue"]), (r["h2d_s"], C["orange"]), (D.other_s(r), C["aqua"])]
            parts = [(v, c) for v, c in parts if sx(v) - x0 >= 1]
            xx = x0
            for k, (v, col) in enumerate(parts):
                wpx = sx(v) - x0
                last = k == len(parts) - 1
                s.hbar(xx, y, wpx - (0 if last else 2), thick, col, round_end=last)
                xx += wpx
            s.text(sx(r["total_s"]) + 8, y + 15, f"{r['total_s']:.1f} s", 12.5, C["ink"])
            y += pitch
        y += facet_gap
    s.footnote(["Every run except the per-beam baseline takes its tokens from that run's beam tree, so all four do the same work."])
    s.save("fig3_end_to_end")


# ---------------------------------------------------------------- 4. speedups
def fig_speedups():
    R = D.dedup_runs()
    sp = D.speedups(R)
    a = [sp[T]["lbs_vs_base_percopy"] for T in D.STEPS]
    b = [sp[T]["lbs_vs_base_dedup"] for T in D.STEPS]
    c = [sp[T]["dedup_vs_percopy_lbs"] for T in D.STEPS]
    W, H = 920, 540
    s = SVG(W, H)
    s.header(
        f"With dedup KV, group-wise gains {min(b):.2f}–{max(b):.2f}×, not {min(a):.1f}–{max(a):.1f}×",
        f"{SUB_DEDUP} · M = {D.M_MAIN} GiB · ratios of end-to-end time",
        legend=[
            (PERBEAM, "Group-wise vs baseline, per-beam KV", "box"),
            (DEDUP, "Group-wise vs baseline, both dedup", "box"),
            (C["aqua"], "Dedup vs per-beam, group-wise", "box"),
        ],
    )
    x0, x1, yb, yt = 80, W - 40, H - 90, 120
    vmax = math.ceil(max(a + b + c) + 0.5)
    sy = lambda v: yb - v / vmax * (yb - yt)
    for v in range(0, vmax + 1):
        s.line(x0, sy(v), x1, sy(v), C["grid"])
        s.text(x0 - 10, sy(v) + 4, f"{v}×", 12, C["muted"], anchor="end")
    gw = (x1 - x0) / 3
    for gi, T in enumerate(D.STEPS):
        cx = x0 + gw * gi + gw / 2
        for j, (vals, col) in enumerate(((a, PERBEAM), (b, DEDUP), (c, C["aqua"]))):
            v = vals[gi]
            bx = cx - 40 + j * 28
            s.vbar(bx, yb, yb - sy(v), 24, col)
            s.text(bx + 12, sy(v) - 7, f"{v:.2f}×", 12, C["ink"], anchor="middle", halo=True)
        s.text(cx, yb + 22, f"step {T}", 13, C["ink2"], anchor="middle")
    s.line(x0, yb, x1, yb, C["axis"])
    s.line(x0, sy(1), x1, sy(1), C["ink"], 1.2, dash="5 4")
    s.text(x1, sy(1) - 6, "1× = no gain", 11.5, C["ink2"], anchor="end")
    s.footnote(
        [
            "A ratio is the slower run's total time divided by the faster one's; e.g. 3.58× means the group-wise schedule took 1/3.58 of the baseline's time."
        ]
    )
    s.save("fig4_speedups")


# ---------------------------------------------------------------- 5. transfer: predicted vs measured
def fig_transfer():
    W = 980
    rows = []
    for T in D.STEPS:
        for st, m, label in D.METHODS:
            r = D.run("dedup", st, m, D.N_DEDUP, T, D.M_MAIN)
            pred = D.predicted_h2d(st, m, T, D.M_MAIN)
            rows.append(
                (
                    T,
                    st,
                    label,
                    r["h2d_bytes"] / D.GiB if D.ok(r) else None,
                    None if pred is None else pred / D.GiB,
                )
            )
    H = 140 + len(rows) * 28 + 3 * 16 + 90
    s = SVG(W, H)
    exact = all(
        m is not None and p is not None and abs(m - p) < 1e-9 for _, _, _, m, p in rows
    )
    s.header(
        "Measured host→GPU transfer equals the prediction in every run"
        if exact
        else "Measured host→GPU transfer vs prediction",
        f"{SUB_DEDUP} · M = {D.M_MAIN} GiB",
        legend=[
            (C["ink2"], "Predicted", "ring"),
            (PERBEAM, "Measured, per-beam KV", "dot"),
            (DEDUP, "Measured, deduplicated", "dot"),
        ],
    )
    x0, x1 = 300, W - 170
    lo, hi = 0, 5  # 10^0 .. 10^5 GiB
    sx = lambda v: x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)
    y = 122
    top = y
    for T in D.STEPS:
        s.text(24, y + 14, f"Step {T}", 13.5, weight="bold")
        for _, st, label, meas, pred in [r for r in rows if r[0] == T]:
            s.text(x0 - 12, y + 14, label, 12.5, C["ink2"], anchor="end")
            if meas == 0 and (pred or 0) == 0:
                s.text(
                    x0 + 4,
                    y + 14,
                    "0 — all KV fits in M (predicted 0)",
                    12,
                    C["ink2"],
                    italic=True,
                )
            else:
                if pred:
                    s.circle(sx(pred), y + 10, 7, C["surface"], stroke=C["ink2"], sw=2)
                if meas:
                    s.circle(
                        sx(meas), y + 10, 4.5, PERBEAM if st == "percopy" else DEDUP
                    )
                s.text(x1 + 12, y + 14, f"{fmt(meas)} GiB", 12.5, C["ink"])
            y += 28
        y += 16
    for e in range(lo, hi + 1):
        v = 10**e
        s.line(sx(v), top, sx(v), y - 12, C["grid"])
        s.text(sx(v), y + 4, f"{v:,}", 12, C["muted"], anchor="middle")
    s.text(
        (x0 + x1) / 2,
        y + 24,
        "Host→GPU transfer over the whole search (GiB, log scale)",
        12.5,
        C["muted"],
        anchor="middle",
    )
    s.footnote(
        [
            "Predicted, per-beam KV: the LBS paper's Eq. 1–5. Predicted, deduplicated: GPU-free replay of the recorded beam tree (dedupbeam/segbook.py).",
            "Measured: bytes summed at every host→GPU copy in the engine.",
        ]
    )
    s.save("fig5_transfer_pred_vs_meas")


# ---------------------------------------------------------------- 6. attention microbenchmark
def fig_microbench():
    mb = D.microbench()
    if mb is None:
        print(
            "skip fig6: results/microbench_attention.json missing (run scripts/microbench_attention.py)"
        )
        return
    row = min(mb["rows"], key=lambda r: abs(r["pool_rows"] - 5200))
    bars = [
        (
            "One KV copy per beam",
            f"{mb['per_beam_tokens']:,} tokens × 32 beams",
            mb["per_beam_ms"],
            PERBEAM,
        ),
        (
            "Pool + mask",
            f"{row['pool_rows']:,} distinct tokens, 1 buffer",
            row["pool_mask_ms"],
            DEDUP,
        ),
        (
            "8 buffers",
            "same pool split into 8 bmm calls",
            row["buffers_8_ms"],
            C["gray"],
        ),
        (
            "63 buffers",
            "same pool split into 63 bmm calls",
            row["buffers_63_ms"],
            C["gray"],
        ),
    ]
    W, H = 900, 420
    s = SVG(W, H)
    s.header(f"One masked pool: {row['pool_mask_ms']:.2f} ms per layer vs {mb['per_beam_ms']:.2f} ms for per-beam copies",
             f"One decode attention layer · OPT-2.7B shapes (32 heads × 80) · 32 beams · FP16 · {GPU}")
    x0, x1 = 300, W - 120
    vmax = max(b[2] for b in bars) * 1.1
    sx = lambda v: x0 + v / vmax * (x1 - x0)
    y_end = 104 + 56 * len(bars)
    for v in nice_ticks(vmax, 5):  # grid first, bars on top
        s.line(sx(v), 100, sx(v), y_end - 6, C["grid"])
        s.text(sx(v), y_end + 10, f"{v:g}", 12, C["muted"], anchor="middle")
    s.line(x0, 100, x0, y_end - 6, C["axis"])
    s.text((x0 + x1) / 2, y_end + 28, "Time per layer (ms): scores, mask, softmax, weighted sum", 12.5, C["muted"], anchor="middle")
    y = 104
    for name, detail, v, col in bars:
        s.text(x0 - 12, y + 13, name, 13, C["ink"], anchor="end")
        s.text(x0 - 12, y + 29, detail, 11.5, C["muted"], anchor="end")
        s.hbar(x0, y + 4, sx(v) - x0, 22, col)
        s.text(sx(v) + 8, y + 20, f"{v:.2f} ms", 12.5, C["ink"], halo=True)
        y += 56
    s.footnote(
        [
            f"{row['pool_rows']:,} rows ≈ the peak distinct KV at step 32 (1.58 GiB). In this synthetic test 30% of the pool is masked for each beam."
        ]
    )
    s.save("fig6_attention_microbench")


# ---------------------------------------------------------------- 7. GQA
def fig_gqa():
    runs = D.gqa_runs()
    cfg = D.gqa_config() or {}
    btok = cfg.get("kv_bytes_per_token_all_layers", 36864)
    M = 8
    fit_max = int(M * D.GiB // (2048 * btok)) // 2 * 2
    g = lambda mode, n: runs.get((mode, n))
    b256, l256 = g("baseline", 256), g("lbs", 256)
    W, H = 1100, 560
    s = SVG(W, H)
    title = f"GQA: Qwen2.5-3B fits {fit_max} beams in 8 GiB"
    if D.ok(b256) and D.ok(l256):
        title += (
            f"; group-wise pays off only at 256 ({b256['total_s'] / l256['total_s']:.2f}×)"
        )
    s.header(
        title,
        f"Qwen2.5-3B FP16 · one KV copy per beam · step 32 · prompt 128 + 1,920 generated · M = {M} GiB · {GPU}",
        legend=[
            (DEDUP, "Fits in M: all KV on the GPU", "box"),
            (PERBEAM, "Exceeds M: baseline (layer-wise offloading)", "box"),
            (C["aqua"], "Exceeds M: group-wise schedule", "box"),
        ],
    )
    # left: KV size by beam count
    beams = [32, 64, 112, 128, 256, 512]
    x0, x1, yb, yt = 80, 480, H - 100, 130
    kv = {n: n * 2048 * btok / D.GiB for n in beams}
    vmax = 40
    sy = lambda v: yb - v / vmax * (yb - yt)
    s.text(24, 118, "KV for all beams at 2,048 tokens (GiB): blue fits in M, gray exceeds it", 13, weight="bold")
    for v in nice_ticks(vmax, 4):
        s.line(x0, sy(v), x1, sy(v), C["grid"])
        s.text(x0 - 10, sy(v) + 4, axis_label(v), 12, C["muted"], anchor="end")
    bw = (x1 - x0) / len(beams)
    for i, n in enumerate(beams):
        bx = x0 + i * bw + bw / 2 - 12
        col = DEDUP if kv[n] <= M else C["gray"]
        s.vbar(bx, yb, yb - sy(kv[n]), 24, col)
        s.text(bx + 12, sy(kv[n]) - 7, f"{kv[n]:.1f}", 12, C["ink"], anchor="middle", halo=True)
        s.text(bx + 12, yb + 20, str(n), 12.5, C["ink2"], anchor="middle")
    s.line(x0, yb, x1, yb, C["axis"])
    s.text((x0 + x1) / 2, yb + 40, "Total beams", 12.5, C["muted"], anchor="middle")
    s.line(x0, sy(M), x1, sy(M), C["ink"], 1.5, dash="5 4")
    s.text(x0 + 6, sy(M) - 6, f"M = {M} GiB", 12, C["ink"], weight="bold")
    # right: total time
    x0, x1 = 600, W - 40
    s.text(560, 118, "End-to-end time (s)", 13, weight="bold")
    groups = [(n, [("ingpu", DEDUP)]) for n in (32, 64, 112)] + [
        (n, [("baseline", PERBEAM), ("lbs", C["aqua"])]) for n in (128, 256)
    ]
    tmax = max((r["total_s"] for r in runs.values() if D.ok(r)), default=1)
    vmax = math.ceil(tmax * 1.15 / 100) * 100
    sy2 = lambda v: yb - v / vmax * (yb - yt)
    for v in nice_ticks(vmax, 5):
        s.line(x0, sy2(v), x1, sy2(v), C["grid"])
        s.text(x0 - 10, sy2(v) + 4, axis_label(v), 12, C["muted"], anchor="end")
    gw = (x1 - x0) / (len(groups) + 1)
    for i, (n, series) in enumerate(groups):
        cx = x0 + gw * i + gw / 2
        k = len(series)
        for j, (mode, col) in enumerate(series):
            r = g(mode, n)
            bx = cx - (k * 28 - 4) / 2 + j * 28
            if D.ok(r):
                s.vbar(bx, yb, yb - sy2(r["total_s"]), 24, col)
                s.text(
                    bx + 12,
                    sy2(r["total_s"]) - 7,
                    f"{r['total_s']:.0f}",
                    12,
                    C["ink"],
                    anchor="middle",
                    halo=True,
                )
        if k == 2 and D.ok(g("baseline", n)) and D.ok(g("lbs", n)):
            ratio = g("baseline", n)["total_s"] / g("lbs", n)["total_s"]
            top = sy2(max(g("baseline", n)["total_s"], g("lbs", n)["total_s"]))
            s.text(cx, top - 44, "Group-wise", 12.5, C["ink"], weight="bold", anchor="middle")
            s.text(
                cx,
                top - 26,
                f"{ratio:.2f}×",
                12.5,
                C["ink"],
                weight="bold",
                anchor="middle",
            )
        s.text(cx, yb + 20, str(n), 12.5, C["ink2"], anchor="middle")
    cx = x0 + gw * len(groups) + gw / 2
    s.text(cx, yb - 58, "512:", 12, C["ink2"], anchor="middle")
    s.text(cx, yb - 42, "not run", 12, C["ink2"], anchor="middle")
    s.text(cx, yb - 26, "(host RAM)", 11, C["muted"], anchor="middle")
    s.text(cx, yb + 20, "512", 12.5, C["ink2"], anchor="middle")
    s.line(x0, yb, x1, yb, C["axis"])
    s.text((x0 + x1) / 2, yb + 40, "Total beams", 12.5, C["muted"], anchor="middle")
    s.footnote(
        [
            f"Qwen2.5-3B KV is {btok:,} bytes per token (2 KV heads × 128 × 36 layers); OPT-2.7B's is 327,680. For comparison, OPT-2.7B with 32 beams",
            "needs 20 GiB of KV: per-beam baseline 517 s, per-beam group-wise 170 s, deduplicated 45 s (M = 8 GiB). Different models: compare scale, not accuracy.",
        ]
    )
    s.save("fig7_gqa")


if __name__ == "__main__":
    fig_mechanism()
    fig_kv_peak()
    fig_end_to_end()
    fig_speedups()
    fig_transfer()
    fig_microbench()
    fig_gqa()
