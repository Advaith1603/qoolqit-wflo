"""Build the two-slide contest deck (dev environment only; python-pptx is not a notebook dependency).

Every number comes from the JSON results record in wflo_rydberg.py; the S-vs-rung figure is the PNG output of the
executed notebook. Usage: python deck/build_deck.py <executed notebook .ipynb>
"""
import ast, base64, json, sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "wflo_rydberg.py").read_text()
a = src.index('FULL_RESULTS = json.loads(r"""') + len('FULL_RESULTS = json.loads(r"""')
REC = json.loads(src[a:src.index('""")', a)])

# The S-vs-rung figure, exported from the executed notebook.
nb = json.loads(Path(sys.argv[1]).read_text())
fig_png = next(base64.b64decode(o["data"]["image/png"]) for c in nb["cells"] if c["cell_type"] == "code"
               and "Solution quality as wake physics" in "".join(c["source"]) for o in c.get("outputs", [])
               if "image/png" in o.get("data", {}))
FIG = ROOT / "deck" / "s_vs_rung.png"
FIG.write_bytes(fig_png)

# Numbers used on the slides (all from the record).
inst, A, B = REC["instances"], REC["A"], REC["B"]
b0 = src.index("PUBLISHED_R0 = {")                                  # published numbers, as quoted in the notebook
PUB = ast.literal_eval(src[b0 + len("PUBLISHED_R0 = "):src.index("}", b0) + 1].replace("100 * 2080 / 2304", "0"))
vqe = [v for k, v in PUB.items() if k.startswith("VQE")]
S_of = lambda pct: (pct / 100 * inst["R0"]["optimum"] - inst["R0"]["P_random"]) / (inst["R0"]["optimum"] - inst["R0"]["P_random"])
r0_sel = A["R0"]["resampled"]
n_sel = round(r0_sel["frac_selected_optimal"] * 36)
r1_S = A["R1"]["exact"]["exp_norm_given_4"]
r3_popt = max(A["R3"]["exact"]["P_opt"], B["R3"]["exact"]["P_opt"], *(v["exact"]["P_opt"] for v in REC["R3_probes"].values()))
spam = REC["noise_A"]["results"]
budget_shots, ecrr = 2 * 500, 0.25
assert n_sel == 36 and round(r0_sel["mean_norm"], 2) == 1.0 and round(r1_S, 2) == 0.89 and r3_popt < 1e-3

INK, INK2, MUTED, LINE, SURF = "0B0B0B", "52514E", "898781", "E1E0D9", "FCFCFB"
BLUE, ORANGE, GREEN, TINT = "2A78D6", "EB6834", "1BAF7A", "F1F0EC"
rgb = lambda h: RGBColor.from_string(h)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def bg(slide):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = rgb(SURF)


def text(slide, x, y, w, h, runs, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name=None):
    """runs: str, or list of paragraphs, each a list of (text, dict(size, color, bold)) tuples."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    paras = [[(runs, {})]] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        for t, o in para:
            r = p.add_run()
            r.text = t
            r.font.name = "Calibri"
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.color.rgb = rgb(o.get("color", color))
    return tb


def card(slide, x, y, w, h, fill=TINT, name=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.08
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


# ---------------- Slide 1
s1 = prs.slides.add_slide(BLANK)
bg(s1)
text(s1, 0.5, 0.4, 12.3, 0.55, "Emulated Rydberg atoms solve native wakes, not long directional ones",
     size=22, bold=True, name="Headline")

ladder = [("R0", "published", "x = 1, r = 1.5, uniform wind", f"{inst['R0']['n_optima']} optima = king-graph MIS"),
          ("R1", "ours", "x = 3, r = 1, uniform wind", f"{inst['R1']['n_optima']} optimum (four corners)"),
          ("R2", "ours", "x = 1, r = 1.5, 75% from west", f"same {inst['R2']['n_optima']} optima as R0"),
          ("R3", "ours", "x = 3, r = 1, 75% from west", f"{inst['R3']['n_optima']} optima, not MIS")]
fills = ["E3EEFB", "EBEFF5", "F2EFEA", "FBE9E1"]           # native (blue tint) to non-native (orange tint)
for i, (rung, who, wake, opt) in enumerate(ladder):
    x = 0.5 + i * 3.1
    card(s1, x, 1.2, 2.85, 1.0, fill=fills[i], name=f"Ladder {rung}")
    text(s1, x + 0.18, 1.28, 2.55, 0.9,
         [[(rung, dict(size=18, bold=True)), (f"  {who}", dict(size=12, color=MUTED))],
          [(wake, dict(size=12, color=INK2))], [(opt, dict(size=12, color=INK2))]], name=f"Ladder {rung} text")
    if i < 3:
        arr = s1.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(x + 2.92), Inches(1.6), Inches(0.13), Inches(0.22))
        arr.fill.solid(); arr.fill.fore_color.rgb = rgb(MUTED); arr.line.fill.background()
text(s1, 0.5, 2.27, 12.3, 0.25, "Rydberg-native  →  increasingly non-native wake physics (wake length x, spread r, wind regime)",
     size=11, color=MUTED, name="Ladder caption")

pic = s1.shapes.add_picture(str(FIG), Inches(0.5), Inches(2.65), height=Inches(4.2))
pic.name = "S vs rung plot"

fmt = lambda v: f"{v:+.2f}".replace("-", "\u2212")              # true minus sign
NB = "\u00a0"                                                     # keep "S = value" on one line
stats = [(f"{n_sel}/36", BLUE, f"R0: the paper's selection rule picks an optimal layout in every run (S{NB}={NB}1.00). "
          f"Published VQE means: S{NB}{fmt(S_of(min(vqe)))}{NB}to{NB}{fmt(S_of(max(vqe)))}, about random."),
         (f"S = {r1_S:.2f}", BLUE, "R1: the same rung-blind program still has the optimum as its ground state; "
          "van der Waals tails mimic the long wake."),
         ("P(opt) ≈ 0", ORANGE, f"R3, every encoding below 0.001: scalable ones misrank the optimum; the one fit "
          "that ranks it well fails on dynamics. Both bottlenecks bind.")]
for i, (big, col, label) in enumerate(stats):
    y = 2.65 + i * 1.42
    card(s1, 7.0, y, 5.8, 1.27, name=f"Stat {i + 1}")
    text(s1, 7.22, y + 0.12, 2.25, 1.03, big, size=27, bold=True, color=col, anchor=MSO_ANCHOR.MIDDLE, name=f"Stat {i + 1} number")
    text(s1, 9.55, y + 0.12, 3.05, 1.03, label, size=12, color=INK2, anchor=MSO_ANCHOR.MIDDLE, name=f"Stat {i + 1} label")

text(s1, 0.5, 6.95, 12.3, 0.3,
     [[("Why it matters: ", dict(bold=True, color=INK)),
       ("it shows which wake models a neutral-atom array encodes natively, before spending QPU time.",
        dict(color=INK2))]], size=14, name="Why it matters")
text(s1, 0.5, 7.22, 12.3, 0.2, "Noiseless emulation (QoolQit 1.4.0, emu-sv). S = 0 random layout, 1 optimal. "
     "Published: Hancock et al., arXiv:2312.13123. R1–R3 are our instances.", size=10, color=MUTED, name="Source")

# ---------------- Slide 2
s2 = prs.slides.add_slide(BLANK)
bg(s2)
text(s2, 0.5, 0.4, 12.3, 0.55, "QoolQit in practice, and a two-job QPU ask", size=22, bold=True,
     name="Headline")
cols = [("What helped", GREEN, [
            ("One program, many devices", "Dimensionless Register → Drive → QuantumProgram, compiled to AnalogDevice; "
             "the same program passes Orion Beta's limits."),
            ("Exact, fast emulation", "emu-sv agrees with QuTiP to 4e-5; a 16-atom run takes 8.8 s on GPU."),
            ("Native MIS encoding", "Blockade on a square grid solves the published instance with no QUBO embedding."),
            ("Device limits in code", "AnalogDevice specs (spacing, radius, amplitude, duration) are readable, so every "
             "program is checked before it runs.")]),
        ("What we hit", ORANGE, [
            ("24-atom cap", "LocalEmulator + emu-sv cannot sample above 24 atoms (torch.multinomial 2²⁴): 5×5 dropped."),
            ("Radius measured from the origin", "max_radial_distance ignores the register centre; registers must be centred."),
            ("No QUBO tooling in 1.4.0", "QUBOInstance and AdiabaticDriveShaper are absent; we wrote our own L-BFGS embedding."),
            ("Exact spacing check", "A spacing of exactly 1.0 fails compile_to after rounding; rescale to 1 + 10⁻⁶.")])]
for j, (head, col, items) in enumerate(cols):
    x = 0.5 + j * 6.25
    dot = s2.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(1.27), Inches(0.22), Inches(0.22))
    dot.fill.solid(); dot.fill.fore_color.rgb = rgb(col); dot.line.fill.background()
    text(s2, x + 0.35, 1.2, 5.5, 0.4, head, size=20, bold=True, name=f"{head} header")
    for i, (t, d) in enumerate(items):
        y = 1.8 + i * 0.95
        card(s2, x, y, 6.0, 0.82, name=f"{head} {i + 1}")
        text(s2, x + 0.2, y + 0.1, 5.6, 0.65, [[(t, dict(bold=True, size=14, color=INK))], [(d, dict(size=12, color=INK2))]],
             anchor=MSO_ANCHOR.MIDDLE, name=f"{head} {i + 1} text")

ask_y = 5.75
card(s2, 0.5, ask_y, 12.3, 1.3, fill="E3EEFB", name="QPU ask")
text(s2, 0.75, ask_y + 0.15, 3.3, 1.0, [[("QPU ask", dict(size=14, bold=True, color=BLUE))],
                                        [(f"2 jobs × 500 shots", dict(size=20, bold=True))],
                                        [(f"≈ {budget_shots / ecrr / 60:.0f} min at {ecrr} Hz (Orion Beta)", dict(size=12, color=INK2))]],
     anchor=MSO_ANCHOR.MIDDLE, name="QPU ask numbers")
r0n, r1n = spam["R0"]["SPAM"], spam["R1"]["SPAM"]
text(s2, 4.3, ask_y + 0.15, 8.3, 1.0,
     [[("One Encoding A program, scored for R0 and R1.", dict(bold=True, color=INK))],
      [(f"R0: emulator-vs-QPU gap on a native instance (predicted P(opt) ≈ {r0n['P_opt'][0]:.2f} with SPAM errors).",
        dict(color=INK2))],
      [(f"R1: does the van der Waals tail effect survive? (four corners predicted most probable, "
        f"P\u00a0≈\u00a0{REC['qpu_budget']['R1_P_corners']:.2f} vs {REC['qpu_budget']['R1_runner_up']:.2f}).", dict(color=INK2))]],
     size=13, anchor=MSO_ANCHOR.MIDDLE, name="QPU ask detail")
text(s2, 0.5, 7.22, 12.3, 0.2, "Failure log: 16 entries in the notebook. Device figures: Pasqal, Technical Overview for "
     "Advanced Users, Orion Beta (2025). SPAM values: De Léséleuc et al. 2018.", size=10, color=MUTED, name="Source")

out = ROOT / "deck" / "qoolqit_wflo.pptx"
prs.save(out)
print("saved", out)
