"""Draw pipekit's README assets as light and dark SVGs into ``docs/assets/``.

The SVGs are generated; edit this script and re-run it::

    uv run python scripts/make_readme_assets.py

It writes the icon (``icon.svg``), the logo (``logo-{light,dark}.svg``),
one operator and its four shapes (``operator-{light,dark}.svg``), the
workspace (``packages-{light,dark}.svg``), who builds on it
(``ecosystem-{light,dark}.svg``) and a bigger graph, the README's anomaly
detector (``graph-{light,dark}.svg``). Standard library only; the style follows the
stack's other READMEs (gaussx, radtrax): a rounded square with a gradient
and a white glyph, chips in a monospace face.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# The brand: teal to cyan, things flowing through pipes.
GRAD = ("#14B8A6", "#0891B2")

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "ink": "#1F2937",
        "muted": "#6B7280",
        "panel": "#F8FAFC",
        "panel_stroke": "#E5E7EB",
        "chip": "#FFFFFF",
        "chip_stroke": "#D1D5DB",
        "accent": "#0D9488",
        "accent_ink": "#0F766E",
        "edge": "#9CA3AF",
        "faint": "#CBD5E1",
    },
    "dark": {
        "ink": "#E5E7EB",
        "muted": "#9CA3AF",
        "panel": "#111827",
        "panel_stroke": "#374151",
        "chip": "#1F2937",
        "chip_stroke": "#4B5563",
        "accent": "#2DD4BF",
        "accent_ink": "#5EEAD4",
        "edge": "#6B7280",
        "faint": "#475569",
    },
}

W92 = 'fill="#FFFFFF" fill-opacity="0.92"'
W22 = 'fill="#FFFFFF" fill-opacity="0.22"'


# ── primitives ───────────────────────────────────────────────────────────────


def svg(width: float, height: float, body: str, label: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" '
        f'height="{height:g}" viewBox="0 0 {width:g} {height:g}" role="img" '
        f'aria-label="{escape(label)}">\n<title>{escape(label)}</title>\n'
        f"{body}\n</svg>\n"
    )


def text(
    x: float,
    y: float,
    s: str,
    *,
    size: float = 14,
    fill: str,
    weight: int = 400,
    anchor: str = "start",
    family: str = SANS,
) -> str:
    return (
        f'<text x="{x:g}" y="{y:g}" font-family="{family}" font-size="{size:g}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>'
    )


def rect(x, y, w, h, *, rx=8, fill="none", stroke="none", sw=1.5, dash="", op=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' fill-opacity="{op:g}"' if op != 1.0 else ""
    return (
        f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx:g}" '
        f'fill="{fill}"{o} stroke="{stroke}" stroke-width="{sw:g}"{d}/>'
    )


def path(d, *, stroke, sw=2.0, dash="", marker="", fill="none", op=1.0):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    mk = f' marker-end="url(#{marker})"' if marker else ""
    o = f' stroke-opacity="{op:g}"' if op != 1.0 else ""
    return (
        f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}" '
        f'stroke-linecap="round" stroke-linejoin="round"{da}{mk}{o}/>'
    )


def circle(cx, cy, r, attrs):
    return f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}" {attrs}/>'


def arrowhead(mid: str, color: str) -> str:
    return (
        f'<marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{color}"/></marker>'
    )


def gradient(gid: str) -> str:
    return (
        f'<linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{GRAD[0]}"/>'
        f'<stop offset="1" stop-color="{GRAD[1]}"/></linearGradient>'
    )


def mono_width(s: str, size: float) -> float:
    return 0.61 * size * len(s)


def chip(x, y, s, t, *, size=12.0, dashed=False, h=26.0) -> tuple[str, float]:
    w = mono_width(s, size) + 22
    body = rect(
        x,
        y,
        w,
        h,
        rx=6,
        fill=t["chip"],
        stroke=t["chip_stroke"],
        sw=1.2,
        dash="5 4" if dashed else "",
    )
    body += text(
        x + 11,
        y + h / 2 + size * 0.36,
        escape(s),
        size=size,
        fill=t["ink"],
        family=MONO,
    )
    return body, w


def chips_row(x0, y, items, t, *, right, size=12.0, gap=8.0):
    """Lay chips left to right, wrapping at ``right``; return (svg, bottom)."""
    out, x = [], x0
    for s in items:
        w = mono_width(s, size) + 22
        if x + w > right:
            x, y = x0, y + 34
        out.append(chip(x, y, s, t, size=size)[0])
        x += w + gap
    return "".join(out), y + 26


# ── the icon (128 x 128, the glyph in white on the gradient) ─────────────────


def glyph_chain(ox: float, oy: float) -> str:
    """Three operators in a row: ``a | b | c``."""
    out = [path(f"M{ox + 12} {oy + 64} L{ox + 22} {oy + 64}", stroke="#FFFFFF", sw=4)]
    for i, x in enumerate((22, 54, 86)):
        out.append(
            f'<rect x="{ox + x}" y="{oy + 50}" width="22" height="28" rx="6" {W92}/>'
        )
        if i < 2:
            out.append(
                path(
                    f"M{ox + x + 22} {oy + 64} L{ox + x + 32} {oy + 64}",
                    stroke="#FFFFFF",
                    sw=4,
                )
            )
    out.append(
        path(f"M{ox + 108} {oy + 64} L{ox + 116} {oy + 64}", stroke="#FFFFFF", sw=4)
    )
    for x in (49, 81):  # the pipes: |
        out.append(
            f'<rect x="{ox + x - 1.5}" y="{oy + 34}" width="3" height="60" rx="1.5" '
            f"{W22}/>"
        )
    return "".join(out)


def tile(ox: float, oy: float, gid: str) -> str:
    return rect(ox + 4, oy + 4, 120, 120, rx=26, fill=f"url(#{gid})") + glyph_chain(
        ox, oy
    )


def icon() -> str:
    return svg(
        128,
        128,
        f"<defs>{gradient('pk-icon')}</defs>" + tile(0, 0, "pk-icon"),
        "pipekit",
    )


def logo(theme: str) -> str:
    t = THEMES[theme]
    gid = f"pk-{theme}"
    body = f"<defs>{gradient(gid)}</defs>" + tile(6, 6, gid)
    body += (
        f'<text x="154" y="92" font-family="{SANS}" font-size="72" font-weight="700" '
        f'letter-spacing="-1.5" fill="{t["ink"]}">pipe'
        f'<tspan fill="{t["accent"]}">kit</tspan></text>'
    )
    return svg(470, 140, body, "pipekit")


# ── one operator, four shapes ────────────────────────────────────────────────


def block(x, y, w, h, label, t, *, accent=False, size=12.0):
    return rect(
        x,
        y,
        w,
        h,
        rx=8,
        fill=t["chip"],
        stroke=t["accent"] if accent else t["chip_stroke"],
        sw=1.6,
    ) + text(
        x + w / 2,
        y + h / 2 + size * 0.36,
        escape(label),
        size=size,
        fill=t["ink"],
        anchor="middle",
        family=MONO,
    )


def operator_figure(theme: str) -> str:
    t = THEMES[theme]
    W, H = 1200, 492
    b = [f"<defs>{arrowhead('oa', t['edge'])}{arrowhead('ob', t['accent'])}</defs>"]

    # Left: the contract.
    lx, ly, lw, lh = 20, 20, 400, 430
    b.append(rect(lx, ly, lw, lh, rx=18, fill=t["panel"], stroke=t["accent"], sw=1.8))
    b.append(text(lx + 22, ly + 36, "Operator", size=22, fill=t["ink"], weight=700))
    b.append(
        text(
            lx + 22,
            ly + 60,
            "one method: carrier in, carrier out",
            size=13,
            fill=t["muted"],
        )
    )
    cy = ly + 150
    b.append(block(lx + 22, cy - 22, 92, 44, "carrier", t))
    b.append(block(lx + 154, cy - 30, 92, 60, "_apply", t, accent=True, size=14))
    b.append(block(lx + 286, cy - 22, 92, 44, "carrier", t))
    for x0, x1 in ((lx + 116, lx + 150), (lx + 248, lx + 282)):
        b.append(
            path(f"M{x0} {cy} L{x1} {cy}", stroke=t["accent"], sw=2.4, marker="ob")
        )
    code = [
        ("class", "Standardize(Operator):"),
        ("    def", "__init__(self, *, mean, std):"),
        ("", "        self.mean, self.std = mean, std"),
        ("", ""),
        ("    def", "_apply(self, x):"),
        ("        return", "(x - self.mean) / self.std"),
    ]
    yy = cy + 66
    for kw, rest in code:
        b.append(
            f'<text x="{lx + 22}" y="{yy}" font-family="{MONO}" font-size="12.5" '
            f'fill="{t["ink"]}" xml:space="preserve"><tspan fill="{t["accent_ink"]}" '
            f'font-weight="700">{escape(kw)}</tspan> {escape(rest)}</text>'
        )
        yy += 20
    b.append(
        text(lx + 22, ly + lh - 46, "the carrier is yours:", size=13, fill=t["muted"])
    )
    b.append(
        text(
            lx + 22,
            ly + lh - 24,
            "ndarray · xarray · DataFrame · AnnData · dict …",
            size=12.5,
            fill=t["ink"],
            family=MONO,
        )
    )

    # Right: four shapes, each itself an Operator.
    gx, gy, gw, gh, gap = 450, 20, 355, 205, 20
    panels = [
        ("Sequential", "a | b | c"),
        ("Graph", "Graph(inputs=…, outputs=…)"),
        ("Wrap", "Retry(op) · Cache(op) · Tap(fn)"),
        ("State", "(x, s) → (x, s): Cycle, TrainingLoop"),
    ]
    for i, (title, code_line) in enumerate(panels):
        px = gx + (i % 2) * (gw + gap)
        py = gy + (i // 2) * (gh + gap)
        b.append(rect(px, py, gw, gh, rx=16, fill=t["panel"], stroke=t["panel_stroke"]))
        b.append(text(px + 18, py + 30, title, size=16, fill=t["ink"], weight=700))
        b.append(
            text(
                px + 18,
                py + gh - 18,
                escape(code_line),
                size=12,
                fill=t["muted"],
                family=MONO,
            )
        )
        mx, my = px + 18, py + 52  # drawing box: (gw - 36) x 110
        if i == 0:
            for k, s in enumerate("abc"):
                x = mx + 20 + k * 110
                b.append(block(x, my + 34, 72, 42, s, t, accent=True, size=14))
                if k < 2:
                    b.append(
                        path(
                            f"M{x + 74} {my + 55} L{x + 106} {my + 55}",
                            stroke=t["edge"],
                            sw=2,
                            marker="oa",
                        )
                    )
        elif i == 1:
            nodes = {
                "in": (mx + 30, my + 55),
                "f": (mx + 150, my + 20),
                "g": (mx + 150, my + 90),
            }
            nodes["h"] = (mx + 270, my + 55)
            for p, q in (("in", "f"), ("in", "g"), ("f", "h"), ("g", "h")):
                (x0, y0), (x1, y1) = nodes[p], nodes[q]
                x0 += 22 if p == "in" else 32
                x1 -= 32 if q != "h" else 34
                if q == "h":
                    y1 += -8 if p == "f" else 8
                b.append(
                    path(f"M{x0} {y0} L{x1} {y1}", stroke=t["edge"], sw=2, marker="oa")
                )
            b.append(
                circle(
                    *nodes["in"],
                    20,
                    f'fill="{t["chip"]}" stroke="{t["chip_stroke"]}" '
                    'stroke-width="1.6"',
                )
            )
            b.append(
                text(
                    nodes["in"][0],
                    nodes["in"][1] + 4,
                    "x",
                    size=13,
                    fill=t["ink"],
                    anchor="middle",
                    family=MONO,
                )
            )
            for k in ("f", "g", "h"):
                x, y = nodes[k]
                b.append(block(x - 30, y - 17, 60, 34, k, t, accent=True, size=13))
        elif i == 2:
            b.append(
                rect(
                    mx + 60,
                    my + 14,
                    200,
                    82,
                    rx=14,
                    fill="none",
                    stroke=t["accent"],
                    sw=1.6,
                    dash="6 5",
                )
            )
            b.append(
                text(
                    mx + 72,
                    my + 32,
                    "Retry",
                    size=12,
                    fill=t["accent_ink"],
                    weight=700,
                    family=MONO,
                )
            )
            b.append(block(mx + 120, my + 38, 80, 40, "op", t, accent=True, size=14))
            b.append(
                path(
                    f"M{mx} {my + 58} L{mx + 116} {my + 58}",
                    stroke=t["edge"],
                    sw=2,
                    marker="oa",
                )
            )
            b.append(
                path(
                    f"M{mx + 202} {my + 58} L{mx + 316} {my + 58}",
                    stroke=t["edge"],
                    sw=2,
                    marker="oa",
                )
            )
            # a tap off the output
            b.append(
                path(
                    f"M{mx + 290} {my + 58} L{mx + 290} {my + 100}",
                    stroke=t["edge"],
                    sw=2,
                    dash="3 4",
                )
            )
            b.append(
                text(mx + 296, my + 104, "Tap", size=11, fill=t["muted"], family=MONO)
            )
        else:
            b.append(block(mx + 110, my + 30, 100, 46, "step", t, accent=True, size=14))
            b.append(
                path(
                    f"M{mx} {my + 53} L{mx + 106} {my + 53}",
                    stroke=t["edge"],
                    sw=2,
                    marker="oa",
                )
            )
            b.append(
                path(
                    f"M{mx + 212} {my + 53} L{mx + 316} {my + 53}",
                    stroke=t["edge"],
                    sw=2,
                    marker="oa",
                )
            )
            b.append(
                path(
                    f"M{mx + 190} {my + 78} C{mx + 200} {my + 112} "
                    f"{mx + 120} {my + 112} {mx + 130} {my + 82}",
                    stroke=t["accent"],
                    sw=2.2,
                    marker="ob",
                )
            )
            b.append(
                text(
                    mx + 160,
                    my + 120,
                    "s",
                    size=13,
                    fill=t["accent_ink"],
                    weight=700,
                    anchor="middle",
                    family=MONO,
                )
            )
            b.append(
                text(
                    mx + 50,
                    my + 46,
                    "x",
                    size=13,
                    fill=t["ink"],
                    anchor="middle",
                    family=MONO,
                )
            )
            b.append(
                text(
                    mx + 264,
                    my + 46,
                    "x'",
                    size=13,
                    fill=t["ink"],
                    anchor="middle",
                    family=MONO,
                )
            )
    b.append(
        text(
            gx + gw + gap / 2,
            H - 12,
            "each shape is itself an Operator, so they nest",
            size=13,
            fill=t["muted"],
            anchor="middle",
        )
    )
    return svg(
        W,
        H,
        "".join(b),
        "A pipekit Operator takes a carrier and returns one through a single "
        "_apply method; it composes into a Sequential (a | b | c), a Graph with "
        "named inputs and outputs, wrappers such as Retry, Cache and Tap, and "
        "stateful steps such as Cycle and TrainingLoop, each itself an Operator",
    )


# ── the workspace ────────────────────────────────────────────────────────────


def packages_figure(theme: str) -> str:
    t = THEMES[theme]
    W = 960
    b = [f"<defs>{arrowhead('pa', t['edge'])}</defs>"]
    cards = [
        (
            "pipekit-cycle",
            "time stepping · data assimilation",
            ["Cycle", "DACycle", "ForwardModel"],
        ),
        (
            "pipekit-train",
            "training pipelines",
            ["TrainingLoop", "TrainingDataset", "HyperSweep"],
        ),
        ("pipekit-experiment", "registry · tracking", ["LocalModelRegistry", "Run"]),
        ("pipekit-evaluate", "benchmarks · scoring", ["Benchmark", "MetricScorer"]),
        (
            "pipekit-array",
            "Array API: NumPy, JAX, CuPy, torch, dask",
            ["array_namespace", "StackAlong"],
        ),
        (
            "pipekit-jax",
            "Equinox models · diffrax",
            ["JaxModelOp", "DiffraxForwardModel"],
        ),
    ]
    cw, ch, gap, x0, y0 = 296, 132, 16, 20, 20
    bottoms = []
    for i, (name, sub, items) in enumerate(cards):
        x = x0 + (i % 3) * (cw + gap)
        y = y0 + (i // 3) * (ch + gap)
        b.append(rect(x, y, cw, ch, rx=14, fill=t["panel"], stroke=t["accent"], sw=1.6))
        b.append(
            text(x + 16, y + 28, name, size=15, fill=t["ink"], weight=700, family=MONO)
        )
        b.append(text(x + 16, y + 48, escape(sub), size=12, fill=t["muted"]))
        c, _ = chips_row(x + 16, y + 64, items, t, right=x + cw - 12, size=11.5)
        b.append(c)
        bottoms.append(y + ch)
    # notes on the cross-package links
    ny = max(bottoms) + 26
    b.append(
        text(
            W / 2,
            ny,
            "pipekit-jax builds on pipekit-cycle; "
            "every other link between them is an opt-in extra",
            size=12.5,
            fill=t["muted"],
            anchor="middle",
        )
    )
    # the core
    cy = ny + 22
    core = [
        "Operator",
        "Sequential",
        "Graph",
        "Branch",
        "Retry",
        "Tap",
        "Cache",
        "ThreadMap",
        "StatefulOperator",
        "dumps / loads",
    ]
    band = len(b)
    b.append("")  # the band, drawn once its chips are laid out
    b.append(
        text(
            x0 + 18,
            cy + 30,
            "pipekit",
            size=18,
            fill=t["accent_ink"],
            weight=700,
            family=MONO,
        )
    )
    b.append(
        text(
            W - 38,
            cy + 30,
            "the core: pure Python, no dependencies",
            size=12.5,
            fill=t["muted"],
            anchor="end",
        )
    )
    c, bottom = chips_row(x0 + 18, cy + 46, core, t, right=W - 36)
    b.append(c)
    b[band] = rect(
        x0,
        cy,
        W - 40,
        bottom - cy + 18,
        rx=16,
        fill=t["accent"],
        op=0.08,
        stroke=t["accent"],
        sw=2,
    )
    # every card stands on the core
    for i in range(3):
        x = x0 + i * (cw + gap) + cw / 2
        b.append(
            path(f"M{x} {ny + 6} L{x} {cy - 4}", stroke=t["edge"], sw=2, marker="pa")
        )
    H = bottom + 18 + 20
    return svg(
        W,
        H,
        "".join(b),
        "The pipekit workspace: six sister packages (cycle, train, experiment, "
        "evaluate, array, jax) on the dependency-free pipekit core",
    )


# ── who builds on it ─────────────────────────────────────────────────────────


def ecosystem(theme: str) -> str:
    t = THEMES[theme]
    W, H = 1100, 520
    gid = f"eco-{theme}"
    b = [
        f"<defs>{gradient(gid)}{arrowhead('ea', t['edge'])}"
        f"{arrowhead('eb', t['accent'])}</defs>"
    ]

    # pipekit in the middle
    px, py, pw, ph = 390, 200, 320, 96
    b.append(rect(px, py, pw, ph, rx=18, fill=t["chip"], stroke=t["accent"], sw=2.6))
    b.append(
        f'<g transform="translate({px + 14} {py + 14}) scale(0.53)">'
        + tile(0, 0, gid)
        + "</g>"
    )
    b.append(
        f'<text x="{px + 96}" y="{py + 48}" font-family="{SANS}" font-size="28" '
        f'font-weight="700" fill="{t["ink"]}">pipe'
        f'<tspan fill="{t["accent"]}">kit</tspan></text>'
    )
    b.append(
        text(px + 96, py + 74, "operators for any carrier", size=13, fill=t["muted"])
    )

    def panel(x, y, w, h, title, sub, dashed=False):
        b.append(
            rect(
                x,
                y,
                w,
                h,
                rx=18,
                fill=t["panel"],
                stroke=t["edge"] if dashed else t["panel_stroke"],
                dash="6 5" if dashed else "",
            )
        )
        b.append(text(x + 20, y + 30, title, size=16, fill=t["ink"], weight=700))
        b.append(text(x + 20, y + 50, escape(sub), size=12.5, fill=t["muted"]))

    def node(x, y, w, name, sub, *, dashed=False):
        b.append(
            rect(
                x,
                y,
                w,
                52,
                rx=10,
                fill=t["chip"],
                stroke=t["edge"] if dashed else t["accent"],
                sw=1.6,
                dash="6 5" if dashed else "",
            )
        )
        b.append(
            text(
                x + 14,
                y + 22,
                escape(name),
                size=14,
                fill=t["ink"],
                weight=700,
                family=MONO,
            )
        )
        b.append(text(x + 14, y + 41, escape(sub), size=12, fill=t["muted"]))

    # Left: the GeoStack today.
    panel(20, 20, 340, 480, "In the GeoStack today", "packages that build on pipekit")
    geo = [
        ("geotoolz", "Operators on GeoTensors"),
        ("xrtoolz", "Operators on xarray"),
        ("vardax", "variational DA: cycle, train, jax"),
        ("plumax", "plume models: cycle"),
        ("radtrax[geo]", "retrievals on GeoTensors"),
    ]
    for i, (n, s) in enumerate(geo):
        y = 92 + i * 78
        node(40, y, 300, n, s)
        b.append(
            path(
                f"M340 {y + 26} L365 {y + 26} L365 {py + ph / 2} "
                f"L{px - 6} {py + ph / 2}",
                stroke=t["accent"],
                sw=2,
                marker="eb",
            )
        )

    # Right: any science with a carrier (illustrative).
    panel(
        740,
        20,
        340,
        480,
        "Any science with a carrier",
        "illustrative: bring your own",
        dashed=True,
    )
    other = [
        ("genomics", "AnnData, DataFrame"),
        ("astronomy", "astropy Table, FITS cube"),
        ("climate · ocean", "xarray Dataset"),
        ("imaging · microscopy", "ndarray, torch.Tensor"),
        ("simulation · ML", "dict of arrays, pytree"),
    ]
    for i, (n, s) in enumerate(other):
        y = 92 + i * 78
        node(760, y, 300, n, s, dashed=True)
        b.append(
            path(
                f"M760 {y + 26} L735 {y + 26} L735 {py + ph / 2} "
                f"L{px + pw + 6} {py + ph / 2}",
                stroke=t["edge"],
                sw=2,
                dash="6 5",
                marker="ea",
            )
        )

    # Below: what the core promises, kept between the two buses.
    for k, line in enumerate(
        (
            "carrier-agnostic core:",
            "domain packages narrow the carrier,",
            "and algorithm libraries can satisfy",
            "its Protocols without importing it",
        )
    ):
        b.append(
            text(
                px + pw / 2,
                py + ph + 52 + 22 * k,
                line,
                size=13,
                fill=t["muted"],
                anchor="middle",
            )
        )
    return svg(
        W,
        H,
        "".join(b),
        "pipekit in the middle: on the left the GeoStack packages that build on it "
        "today (geotoolz, xrtoolz, vardax, plumax, radtrax's geo extra); on the "
        "right, illustrative carriers from other sciences (genomics, astronomy, "
        "climate, imaging, simulation)",
    )


# ── a bigger graph: the anomaly detector ─────────────────────────────────────


def graph_figure(theme: str) -> str:
    t = THEMES[theme]
    W, H = 900, 1110
    b = [f"<defs>{arrowhead('ga', t['edge'])}{arrowhead('gb', t['accent'])}</defs>"]
    cx = 330  # the main column

    def pill(x, y, w, label, *, out=False):
        b.append(
            rect(
                x,
                y,
                w,
                36,
                rx=18,
                fill=t["chip"],
                stroke=t["accent"],
                sw=2.6 if out else 1.6,
            )
        )
        b.append(
            text(
                x + w / 2,
                y + 23,
                escape(label),
                size=13,
                fill=t["ink"],
                weight=700,
                anchor="middle",
                family=MONO,
            )
        )

    def op(x, y, w, h, name, sub=""):
        b.append(rect(x, y, w, h, rx=10, fill=t["panel"], stroke=t["accent"], sw=1.6))
        b.append(
            text(
                x + w / 2,
                y + (h / 2 + 5 if not sub else 24),
                escape(name),
                size=13.5,
                fill=t["ink"],
                weight=700,
                anchor="middle",
                family=MONO,
            )
        )
        if sub:
            b.append(
                text(
                    x + w / 2,
                    y + 44,
                    escape(sub),
                    size=11.5,
                    fill=t["muted"],
                    anchor="middle",
                )
            )

    def down(x, y0, y1, shape=""):
        b.append(path(f"M{x} {y0} L{x} {y1 - 4}", stroke=t["edge"], sw=2, marker="ga"))
        if shape:
            b.append(
                text(
                    x + 10,
                    (y0 + y1) / 2 + 4,
                    escape(shape),
                    size=11.5,
                    fill=t["muted"],
                    family=MONO,
                )
            )

    def to_output(y, x0, label):
        b.append(
            path(
                f"M{x0} {y} L{676} {y}",
                stroke=t["accent"],
                sw=2,
                dash="2 5",
                marker="gb",
            )
        )
        pill(680, y - 18, 150, label, out=True)

    # inputs
    pill(cx - 70, 20, 140, "sources")
    pill(30, 20, 150, "background")
    b.append(text(cx + 80, 43, "list[str]", size=11.5, fill=t["muted"], family=MONO))
    b.append(text(30, 74, "(C, H, W)", size=11.5, fill=t["muted"], family=MONO))

    # read: ThreadMap ⊃ Cache ⊃ Retry ⊃ ReadTile
    down(cx, 56, 96)
    nest = [
        ("ThreadMap", "8 workers"),
        ("Cache", "read once"),
        ("Retry", "3 attempts on OSError"),
    ]
    x0, y0, w0, h0 = cx - 160, 96, 320, 210
    for k, (name, note) in enumerate(nest):
        d = 16 * k
        b.append(
            rect(
                x0 + d,
                y0 + 34 * k,
                w0 - 2 * d,
                h0 - 44 * k,
                rx=14,
                fill="none",
                stroke=t["accent"],
                sw=1.4,
                dash="6 5",
            )
        )
        b.append(
            text(
                x0 + d + 12,
                y0 + 34 * k + 22,
                name,
                size=12.5,
                fill=t["accent_ink"],
                weight=700,
                family=MONO,
            )
        )
        b.append(
            text(
                x0 + w0 - d - 12,
                y0 + 34 * k + 22,
                escape(note),
                size=11.5,
                fill=t["muted"],
                anchor="end",
            )
        )
    op(cx - 70, y0 + 112, 140, 40, "ReadTile")
    b.append(
        text(cx + 172, y0 + h0 / 2 + 4, "one per source", size=11.5, fill=t["muted"])
    )

    # stack and check
    down(cx, y0 + h0, 346, "list[(C, H, W)]")
    op(cx - 80, 346, 160, 40, "StackAlong")
    down(cx, 386, 416, "(T, C, H, W)")
    op(cx - 80, 416, 160, 40, "AssertShape")

    # branch
    down(cx, 456, 490)
    bx, by, bw, bh = cx - 150, 490, 300, 116
    b.append(
        rect(
            bx,
            by,
            bw,
            bh,
            rx=14,
            fill=t["panel"],
            stroke=t["accent"],
            sw=1.6,
            dash="6 5",
        )
    )
    b.append(
        text(
            bx + 14,
            by + 24,
            "Branch",
            size=12.5,
            fill=t["accent_ink"],
            weight=700,
            family=MONO,
        )
    )
    b.append(
        text(
            bx + bw - 14,
            by + 24,
            "T &lt; 5 ?",
            size=12,
            fill=t["muted"],
            anchor="end",
            family=MONO,
        )
    )
    for k, (lab, how) in enumerate((("yes", "median"), ("no", "mean"))):
        xx = bx + 16 + k * 142
        op(xx, by + 40, 126, 58, "Composite", f"{how} over T")
        b.append(text(xx + 63, by + 36, lab, size=11, fill=t["muted"], anchor="middle"))
    to_output(by + bh / 2, bx + bw + 4, "composite")

    # two features from the composite
    fy = 690
    b.append(path(f"M{cx} {by + bh} L{cx} {650}", stroke=t["edge"], sw=2))
    b.append(text(cx + 10, 636, "(C, H, W)", size=11.5, fill=t["muted"], family=MONO))
    zx, rx_ = 150, 360
    for x in (zx + 100, rx_ + 100):
        b.append(
            path(
                f"M{cx} 650 L{x} 650 L{x} {fy - 4}", stroke=t["edge"], sw=2, marker="ga"
            )
        )
    op(zx, fy, 200, 56, "ZScore", "z = (x̄ − μ_b) / σ_b")  # noqa: RUF001 (maths in a label)
    op(rx_, fy, 200, 56, "NormalizedDifference", "r = (x_a − x_b)/(x_a + x_b)")  # noqa: RUF001 (maths in a label)
    # background into ZScore
    b.append(
        path(
            f"M105 56 L105 {fy + 28} L{zx - 4} {fy + 28}",
            stroke=t["edge"],
            sw=2,
            marker="ga",
        )
    )

    # score
    sy = 806
    for x in (zx + 100, rx_ + 100):
        b.append(path(f"M{x} {fy + 56} L{x} 780 L{cx} 780", stroke=t["edge"], sw=2))
    down(cx, 780, sy, "")
    b.append(text(zx + 110, 774, "(C, H, W)", size=11.5, fill=t["muted"], family=MONO))
    b.append(text(rx_ + 110, 774, "(H, W)", size=11.5, fill=t["muted"], family=MONO))
    op(cx - 110, sy, 220, 56, "Score", "s = maxᶜ |z| + w · |r − r̄|")  # noqa: RUF001 (maths in a label)
    to_output(sy + 28, cx + 114, "score")

    # threshold, tapped
    ty = 900
    down(cx, sy + 56, ty, "(H, W)")
    op(cx - 110, ty, 220, 56, "Threshold", "m = 1[s > k]")
    b.append(
        path(
            f"M{cx - 114} {ty + 28} L{120} {ty + 28}",
            stroke=t["edge"],
            sw=2,
            dash="3 4",
            marker="ga",
        )
    )
    b.append(
        rect(
            30, ty + 10, 86, 36, rx=8, fill="none", stroke=t["edge"], sw=1.4, dash="3 4"
        )
    )
    b.append(
        text(
            73,
            ty + 33,
            "Tap",
            size=12.5,
            fill=t["muted"],
            weight=700,
            anchor="middle",
            family=MONO,
        )
    )
    b.append(text(73, ty + 64, "log count", size=11, fill=t["muted"], anchor="middle"))
    to_output(ty + 28, cx + 114, "mask")

    # fanout
    fy2 = 998
    down(cx, ty + 56, fy2)
    b.append(
        rect(cx - 150, fy2, 300, 76, rx=14, fill=t["panel"], stroke=t["accent"], sw=1.6)
    )
    b.append(
        text(
            cx - 136,
            fy2 + 22,
            "Fanout",
            size=12.5,
            fill=t["accent_ink"],
            weight=700,
            family=MONO,
        )
    )
    for k, lab in enumerate(("flagged", "fraction")):
        c, _ = chip(cx - 136 + k * 140, fy2 + 36, lab, t, size=12)
        b.append(c)
    to_output(fy2 + 38, cx + 154, "summary")

    b.append(
        text(
            W - 20,
            H - 12,
            "dashed: wrappers and control · grey dotted: observers · "
            "teal dotted: outputs",
            size=11.5,
            fill=t["muted"],
            anchor="end",
        )
    )
    return svg(
        W,
        H,
        "".join(b),
        "An anomaly detector as one pipekit Graph: sources are read in parallel "
        "(ThreadMap around Cache around Retry around ReadTile), stacked and "
        "checked, composited by a Branch (median for few exposures, mean for "
        "many), turned into a z-score against a background input and a channel "
        "ratio, scored, thresholded with a Tap, and summarised by a Fanout; "
        "composite, score, mask and summary are its outputs",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {"icon.svg": icon()}
    for th in THEMES:
        out[f"logo-{th}.svg"] = logo(th)
        out[f"operator-{th}.svg"] = operator_figure(th)
        out[f"packages-{th}.svg"] = packages_figure(th)
        out[f"ecosystem-{th}.svg"] = ecosystem(th)
        out[f"graph-{th}.svg"] = graph_figure(th)
    for name, s in out.items():
        (OUT / name).write_text(s, encoding="utf-8")
        print(name)


if __name__ == "__main__":
    main()
