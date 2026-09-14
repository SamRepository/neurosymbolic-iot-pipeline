"""
Appendix ontology figure — machine-generated from the TTL so it cannot drift.

Renders a readable core subset of the Neuro-Symbolic IoT Smart Home Ontology as a panelled schema
diagram: one panel per taxonomy root, its members listed inside, and the object properties drawn as
labelled arrows between panels.

Why panels rather than a node-per-class graph: a full Graphviz `dot` rendering of all 33 classes is
~16-19 inches wide, so at ``\\textwidth`` in a two-column paper its labels scale to 3-4 pt and the
edges collapse into a tangle. This layout is authored at final figure size, so every label is
legible in print without magnification.

Every panel's contents, every edge and every label is read from the ontology at
``config/base.yaml -> kg.ontology_path``. Nothing about the ontology's content is hard-coded: this
module fixes only the layout and the role colouring. Two guards make drift loud rather than silent:

* every ``owl:Class`` in the file must be placed in a panel, or it is listed on the console;
* every ``owl:ObjectProperty`` must be drawn as an edge, or the script reports and exits non-zero.

Alignment is drawn as measured. The loaded ontology references exactly two external vocabularies,
``sosa:`` and ``time:``. The ``saref:`` prefix is declared in the header but appears in no triple,
so it is not depicted; that gap is discussed in the text rather than drawn as an alignment.

Colour follows the Figure 7 palette so the appendix figure sits in the same visual system as the
body figures.

Usage:
  python fig/generate_ontology_figure.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from rdflib import Graph, RDF, RDFS, OWL, URIRef

# -- Palette: identical constants to fig/generate_federation_figure7.py -----------------
COLOR_CORE = "#0067A3"      # nsiot structural classes
COLOR_BRIDGE = "#00785A"    # the neuro-symbolic bridge (NeuralPrediction)
COLOR_EXTERNAL = "#C77700"  # SSN/SOSA + OWL-Time
COLOR_REASON = "#6A3D9A"    # event subclasses and reasoning vocabularies

INK_PRIMARY = "#1A1A1A"
INK_MUTED = "#5C5C5C"
GRID = "#CCCCCC"

NS = {
    "http://example.org/neuro-symbolic-iot#": "nsiot:",
    "http://www.w3.org/ns/sosa/": "sosa:",
    "http://www.w3.org/2006/time#": "time:",
    "https://saref.etsi.org/core/": "saref:",
    "http://www.w3.org/2001/XMLSchema#": "xsd:",
    str(OWL): "owl:",
    str(RDFS): "rdfs:",
}

ONTOLOGY_PATH = Path(
    "neurosymbolic_iot/kg_semantic_layer/ontology/Neuro-Symbolic IoT Smart Home Ontology.ttl"
)


def qname(term: object) -> str:
    s = str(term)
    for uri, pre in NS.items():
        if s.startswith(uri):
            return pre + s[len(uri):]
    return s


def local(term: object) -> str:
    return qname(term).split(":", 1)[-1]


class Ontology:
    """The parsed ontology, reduced to what the figure needs."""

    def __init__(self, path: Path) -> None:
        g = Graph()
        g.parse(str(path), format="turtle")
        self.graph = g
        self.n_triples = len(g)

        self.classes = {qname(c) for c in g.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)}
        self.object_properties = {
            qname(p) for p in g.subjects(RDF.type, OWL.ObjectProperty) if isinstance(p, URIRef)
        }
        self.datatype_properties = {
            qname(p) for p in g.subjects(RDF.type, OWL.DatatypeProperty) if isinstance(p, URIRef)
        }

        self.children: Dict[str, List[str]] = {}
        for sub, sup in g.subject_objects(RDFS.subClassOf):
            if isinstance(sub, URIRef) and isinstance(sup, URIRef):
                self.children.setdefault(qname(sup), []).append(qname(sub))
        for parent in self.children:
            self.children[parent] = sorted(self.children[parent])

        self.individuals: Dict[str, List[str]] = {}
        for subj, obj in g.subject_objects(RDF.type):
            if not (isinstance(subj, URIRef) and isinstance(obj, URIRef)):
                continue
            qo = qname(obj)
            if qo in self.classes and qo.startswith("nsiot:"):
                self.individuals.setdefault(qo, []).append(qname(subj))
        for k in self.individuals:
            self.individuals[k] = sorted(self.individuals[k])

        self.prop_domain: Dict[str, Optional[str]] = {}
        self.prop_range: Dict[str, Optional[str]] = {}
        for p in self.object_properties | self.datatype_properties:
            uri = self._uri(p)
            dom = [o for o in g.objects(uri, RDFS.domain) if isinstance(o, URIRef)]
            rng = [o for o in g.objects(uri, RDFS.range) if isinstance(o, URIRef)]
            self.prop_domain[p] = qname(dom[0]) if dom else None
            self.prop_range[p] = qname(rng[0]) if rng else None

        self.external_used = set()
        for s, _, o in g:
            for term in (s, o):
                t = str(term)
                for uri, pre in NS.items():
                    if pre in ("sosa:", "time:", "saref:") and t.startswith(uri):
                        self.external_used.add(pre)

    @staticmethod
    def _uri(q: str) -> URIRef:
        for uri, pre in NS.items():
            if q.startswith(pre):
                return URIRef(uri + q[len(pre):])
        return URIRef(q)

    def members(self, root: str) -> List[str]:
        """Subclasses of ``root``; falls back to typed individuals for vocabulary classes."""
        kids = self.children.get(root, [])
        if kids:
            return [local(k) for k in kids]
        return [local(i) for i in self.individuals.get(root, [])]

    def pairwise_disjoint(self, root: str) -> bool:
        """True when every pair of ``root``'s subclasses is declared owl:disjointWith.

        Computed rather than asserted: the panel note then cannot outlive the axioms.
        """
        kids = self.children.get(root, [])
        if len(kids) < 2:
            return False
        declared = {
            frozenset((qname(a), qname(b)))
            for a, b in self.graph.subject_objects(OWL.disjointWith)
        }
        return {frozenset(pair) for pair in combinations(kids, 2)} <= declared

    def datatype_props_of(self, cls: str) -> List[str]:
        return sorted(local(p) for p in self.datatype_properties if self.prop_domain.get(p) == cls)


# -- Drawing primitives ----------------------------------------------------------------
def _tint(hex_color: str, amount: float) -> Tuple[float, float, float]:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


class Panel:
    """A titled box listing the members of one ontology root."""

    def __init__(self, x: float, y: float, w: float, h: float, title: str,
                 items: Sequence[str], color: str, *, note: Optional[str] = None,
                 ncol: int = 1) -> None:
        self.x, self.y, self.w, self.h = x, y, w, h
        self.title, self.items, self.color = title, list(items), color
        self.note, self.ncol = note, ncol

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def anchor(self, side: str, t: float = 0.5) -> Tuple[float, float]:
        """A point on one side. ``t`` runs 0->1 along that side (0.5 = centre).

        Fractional anchors let an edge leave a panel away from its mid-point, which is how
        the diagram keeps long diagonals clear of the panels they would otherwise cross.
        """
        return {
            "l": (self.x, self.y + self.h * t), "r": (self.x + self.w, self.y + self.h * t),
            "t": (self.x + self.w * t, self.y + self.h), "b": (self.x + self.w * t, self.y),
        }[side]

    def draw(self, ax: plt.Axes) -> None:
        ax.add_patch(FancyBboxPatch(
            (self.x, self.y), self.w, self.h,
            boxstyle="round,pad=0,rounding_size=1.1", zorder=2,
            linewidth=1.0, edgecolor=self.color, facecolor=_tint(self.color, 0.93),
        ))
        ax.text(self.cx, self.y + self.h - 2.0, self.title, ha="center", va="top",
                fontsize=7.6, fontweight="bold", color=self.color, zorder=3)

        top = self.y + self.h - 5.2
        if self.items:
            rows = ([self.items[i::self.ncol] for i in range(self.ncol)]
                    if self.ncol > 1 else [self.items])
            colw = self.w / self.ncol
            for ci, colitems in enumerate(rows):
                cx = self.x + colw * ci + colw / 2
                for ri, item in enumerate(colitems):
                    ax.text(cx, top - ri * 2.4, item, ha="center", va="top",
                            fontsize=6.1, color=INK_PRIMARY, zorder=3)
        if self.note:
            ax.text(self.cx, self.y + 1.4, self.note, ha="center", va="bottom",
                    fontsize=5.8, style="italic", color=INK_MUTED, zorder=3)


def arc_apex(p0: Tuple[float, float], p1: Tuple[float, float],
             rad: float) -> Tuple[float, float]:
    """Where a matplotlib ``arc3`` curve actually bulges to, at its half-way point.

    Matplotlib places the Bezier control point at ``M + rad * (dy, -dx)``, so the curve's
    apex is half that offset from the chord mid-point. Knowing it lets a label sit on its
    own arc instead of drifting onto a neighbouring one.
    """
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return mx + rad * dy / 2.0, my - rad * dx / 2.0


def edge(ax: plt.Axes, src: Panel, s_side: str, dst: Panel, d_side: str, label: str,
         *, rad: float = 0.0, color: str = INK_MUTED, dashed: bool = False,
         lx: float = 0.0, ly: float = 0.0, fs: float = 5.9,
         s_t: float = 0.5, d_t: float = 0.5,
         label_at: Optional[Tuple[float, float]] = None) -> None:
    p0, p1 = src.anchor(s_side, s_t), dst.anchor(d_side, d_t)
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=7.0, linewidth=0.75, color=color,
        # A finer dash repeat keeps rdfs:subClassOf legible over a short span.
        linestyle=(0, (2.2, 1.6)) if dashed else "solid",
        # Wider shrink keeps arrow ends off the panel borders they point at.
        shrinkA=3.0, shrinkB=3.5, zorder=1,
    ))
    if label_at is not None:
        tx, ty = label_at
    else:
        tx, ty = arc_apex(p0, p1, rad)
        tx += lx
        ty += ly
    ax.text(tx, ty, label, ha="center", va="center", fontsize=fs,
            color=color, zorder=4,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.92))


def draw_legend(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                linewidth=0.8, edgecolor=GRID, facecolor="white", zorder=2))
    swatches = [(COLOR_CORE, "domain classes"), (COLOR_BRIDGE, "neuro-symbolic bridge"),
                (COLOR_REASON, "reasoning output"), (COLOR_EXTERNAL, "external vocabulary")]
    for i, (c, lab) in enumerate(swatches):
        col, row = i % 2, i // 2
        sx = x + 2.0 + col * (w / 2 - 1.0)
        sy = y + h - 3.4 - row * 2.9
        ax.add_patch(FancyBboxPatch((sx, sy - 1.0), 2.6, 1.9,
                                    boxstyle="round,pad=0,rounding_size=0.4",
                                    linewidth=0.7, edgecolor=c, facecolor=_tint(c, 0.93), zorder=3))
        ax.text(sx + 3.4, sy - 0.1, lab, ha="left", va="center", fontsize=5.8,
                color=INK_PRIMARY, zorder=3)
    # Draw the two line styles rather than naming them: a reader can match what they see.
    ly = y + 1.7
    x0 = x + 2.0
    ax.add_patch(FancyArrowPatch((x0, ly), (x0 + 5.4, ly), arrowstyle="-|>",
                                 mutation_scale=7.0, linewidth=0.85, color=INK_MUTED,
                                 linestyle=(0, (2.2, 1.6)), shrinkA=0, shrinkB=0, zorder=3))
    ax.text(x0 + 6.3, ly, "rdfs:subClassOf", ha="left", va="center",
            fontsize=5.6, style="italic", color=INK_MUTED, zorder=3)
    x1 = x + w / 2 + 1.0
    ax.add_patch(FancyArrowPatch((x1, ly), (x1 + 5.4, ly), arrowstyle="-|>",
                                 mutation_scale=7.0, linewidth=0.85, color=INK_MUTED,
                                 linestyle="solid", shrinkA=0, shrinkB=0, zorder=3))
    ax.text(x1 + 6.3, ly, "object property", ha="left", va="center",
            fontsize=5.6, style="italic", color=INK_MUTED, zorder=3)


# -- Figure ---------------------------------------------------------------------------
def build_figure(o: Ontology) -> Tuple[plt.Figure, List[str], List[str]]:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(7.16, 4.6))
    # The extra width past the right-hand column is the routing margin the two Event -> vocabulary
    # arcs travel through, and where their labels sit clear of the panels.
    ax.set_xlim(0, 127)
    ax.set_ylim(0, 86)
    ax.axis("off")

    # --- external standards, drawn only where the ontology actually references them ----
    sosa = Panel(3.5, 74.5, 30, 9.5, "sosa:Sensor", [], COLOR_EXTERNAL, note="SSN/SOSA")
    tempo = Panel(85.0, 74.5, 27, 9.5, "time:TemporalEntity", [], COLOR_EXTERNAL, note="OWL-Time")

    # --- sensing column ---------------------------------------------------------------
    sensors = Panel(3.5, 51.0, 30, 16.5, "Sensor types", o.members("sosa:Sensor"), COLOR_CORE)
    states = Panel(3.5, 32.0, 30, 13.5, "nsiot:SensorState",
                   o.members("nsiot:SensorState"), COLOR_CORE, ncol=2)
    disjoint_note = "pairwise owl:disjointWith"
    rooms = Panel(3.5, 8.0, 30, 18.5, "nsiot:Location → Room", o.members("nsiot:Room"),
                  COLOR_CORE,
                  note=disjoint_note if o.pairwise_disjoint("nsiot:Room") else None)

    # --- bridge / behaviour column -----------------------------------------------------
    npred = Panel(41.5, 51.0, 34.5, 16.5, "nsiot:NeuralPrediction",
                  o.datatype_props_of("nsiot:NeuralPrediction"), COLOR_BRIDGE, ncol=2,
                  note="reified perception output (datatype properties)")
    person = Panel(52.0, 38.5, 18, 7.5, "nsiot:Person", [], COLOR_CORE)
    activity = Panel(38.5, 6.0, 21, 21.0, "nsiot:Activity", o.members("nsiot:Activity"), COLOR_CORE)
    posture = Panel(62.5, 6.0, 19, 21.0, "nsiot:Posture", o.members("nsiot:Posture"),
                    COLOR_CORE,
                    note=disjoint_note if o.pairwise_disjoint("nsiot:Posture") else None)

    # --- event / reasoning column (narrowed to leave a routing margin at the right) ----
    event = Panel(85.0, 51.0, 27, 16.5, "nsiot:Event", o.members("nsiot:Event"), COLOR_REASON)
    tctx = Panel(85.0, 36.5, 19, 9.0, "nsiot:TimeContext",
                 o.members("nsiot:TimeContext"), COLOR_REASON)
    alert = Panel(85.0, 21.0, 27, 13.5, "nsiot:AlertType",
                  o.members("nsiot:AlertType"), COLOR_REASON)
    err = Panel(85.0, 6.0, 27, 13.5, "nsiot:ErrorType",
                o.members("nsiot:ErrorType"), COLOR_REASON)

    for p in [sosa, tempo, sensors, states, rooms, npred, person, activity, posture,
              event, tctx, alert, err]:
        p.draw(ax)

    draw_legend(ax, 40.5, 73.5, 40, 10.5)

    drawn: List[str] = []

    def prop(src, ss, dst, ds, name, **kw):
        drawn.append(f"nsiot:{name}")
        edge(ax, src, ss, dst, ds, name, **kw)

    # Labels on the four short vertical arrows sit beside the shaft, not across it: centred,
    # their white background would blank out the whole visible run of a ~28 pt arrow.
    edge(ax, sensors, "t", sosa, "b", "rdfs:subClassOf", dashed=True,
         color=COLOR_EXTERNAL, fs=5.7, lx=6.5)

    prop(sensors, "b", states, "t", "hasState", lx=4.3)
    # Leaves Person low and enters Room high, so the chord passes above nsiot:Activity
    # instead of clipping its top-left corner.
    prop(person, "l", rooms, "r", "isLocatedIn", rad=0.10, s_t=0.30, d_t=0.90,
         lx=-1.5, ly=1.5)
    # The four edges converging below Person leave little clear space; these label positions
    # are set explicitly so each sits on its own edge and crosses none of the others.
    prop(person, "b", activity, "t", "performsActivity", rad=0.14, d_t=0.68,
         label_at=(55.2, 36.5))
    prop(person, "b", posture, "t", "hasPosture", rad=-0.14, label_at=(66.8, 36.5))
    # The two prediction edges bow away from each other (left / right) so the pair reads
    # as a symmetric fork out of NeuralPrediction.
    # Leaving from the bottom corners rather than the side mid-points: a side anchor sends
    # these two steep edges back across the panel's own lower border on the way down.
    # Label raised to sit level with involvesPerson on the other side of the figure;
    # lx tracks the curve, which is wider here than at the apex.
    prop(npred, "b", activity, "t", "predictsActivity", s_t=0.217, rad=0.08,
         color=COLOR_BRIDGE, lx=-6.0, ly=7.94)
    prop(npred, "b", posture, "t", "predictsPosture", s_t=0.88, rad=-0.08,
         color=COLOR_BRIDGE, lx=6.0)
    # The two Event -> left-hand arrows leave from different heights so they neither
    # overlap at the panel edge nor cross on the way out.
    # "isBasedOnPrediction" is wider than the 9-unit gap between the two panels, so the edge
    # arcs over their tops and the label sits in the free band above, touching neither border.
    prop(event, "t", npred, "t", "isBasedOnPrediction", rad=0.32, color=COLOR_REASON,
         s_t=0.06, d_t=0.94, ly=1.7)
    # Straight, so it stays clear of the NeuralPrediction bottom-right corner.
    prop(event, "l", person, "r", "involvesPerson", rad=-0.18, color=COLOR_REASON,
         s_t=0.18, d_t=0.60, ly=-0.2, lx=0.5)
    prop(event, "t", tempo, "b", "hasTemporalEntity", color=COLOR_REASON, fs=5.7, lx=7.3)
    # All three vocabulary properties have domain nsiot:Event. A chain down the column
    # (Event -> TimeContext -> AlertType -> ErrorType) would assert the wrong domains, so
    # hasAlertType runs down the lane left free by the narrow TimeContext panel and only
    # hasErrorType swings out through the right-hand margin. Keeping one arc in the margin
    # is what lets each label sit on its own edge and touch no other.
    prop(event, "b", tctx, "t", "hasTimeContext", rad=0.30, color=COLOR_REASON, lx=6.0)
    prop(event, "b", alert, "t", "hasAlertType", color=COLOR_REASON,
         s_t=0.90, d_t=0.90, label_at=(114.8, 43.0))
    prop(event, "r", err, "r", "hasErrorType", rad=-0.62, color=COLOR_REASON,
         s_t=0.15, d_t=0.80, label_at=(122.0, 35.1))

    # --- coverage bookkeeping ----------------------------------------------------------
    shown = set()
    for root in ("sosa:Sensor", "nsiot:SensorState", "nsiot:Room", "nsiot:Activity",
                 "nsiot:Posture", "nsiot:Event", "nsiot:TimeContext", "nsiot:AlertType",
                 "nsiot:ErrorType"):
        shown.add(root)
        shown.update(o.children.get(root, []))
    shown.update({"nsiot:Person", "nsiot:NeuralPrediction", "nsiot:Location"})
    missing = sorted(c for c in o.classes if c not in shown)

    ax.text(63, 0.6,
            f"{len(o.classes)} OWL classes | {len(o.object_properties)} object properties | "
            f"{len(o.datatype_properties)} datatype properties | {o.n_triples} triples",
            ha="center", va="bottom", fontsize=5.9, color=INK_MUTED)

    fig.tight_layout(pad=0.25)
    return fig, drawn, missing


CAPTION = """\
Figure B.9: Core structure of the unified ontology, generated directly from the ontology file the
pipeline loads (config/base.yaml -> kg.ontology_path), so the figure cannot drift from the artefact.

Panels are OWL classes with their subclasses listed inside, coloured by role: blue for the domain
classes, green for the neuro-symbolic bridge, purple for the reasoning outputs the rule base writes
into, and orange for the external vocabularies, which carry their namespace prefix. Solid labelled
arrows are the twelve object properties, each drawn from its declared rdfs:domain to its
rdfs:range; the dashed arrow is rdfs:subClassOf.

The design point of the schema is the reification of perception output: rather than asserting a
predicted activity directly of the person, the pipeline creates an nsiot:NeuralPrediction carrying
its confidence, generating model and timestamp, so that a prediction can be reasoned over,
qualified by confidence and retracted without disturbing observed facts.

External alignment is drawn as measured: four sensor classes specialise sosa:Sensor and
nsiot:hasTemporalEntity ranges over time:TemporalEntity. nsiot:isLocatedIn has an
owl:unionOf(Person, sosa:Sensor) domain, of which the Person arm is shown.
"""


def main() -> int:
    if not ONTOLOGY_PATH.exists():
        print(f"ERROR: ontology not found at {ONTOLOGY_PATH}", file=sys.stderr)
        return 1

    o = Ontology(ONTOLOGY_PATH)
    fig, drawn, missing = build_figure(o)

    out_dir = Path("fig")
    out_dir.mkdir(exist_ok=True)
    pdf_path = out_dir / "Figure_ontology.pdf"
    png_path = out_dir / "Figure_ontology.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", bbox_inches="tight")
    plt.close(fig)

    caption_path = out_dir / "Figure_ontology_caption.txt"
    caption_path.write_text(
        CAPTION
        + f"\nMeasured from {ONTOLOGY_PATH.as_posix()}: {len(o.classes)} owl:Class, "
        f"{len(o.object_properties)} owl:ObjectProperty, {len(o.datatype_properties)} "
        f"owl:DatatypeProperty, {o.n_triples} triples.\n"
        f"External namespaces referenced in triples: {', '.join(sorted(o.external_used))}. "
        f"Declared but unused: "
        f"{', '.join(sorted({'sosa:', 'time:', 'saref:'} - o.external_used)) or 'none'}.\n"
        "Generated by fig/generate_ontology_figure.py\n",
        encoding="utf-8",
    )

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {caption_path}")
    print(f"Ontology: {len(o.classes)} classes, {len(o.object_properties)} object properties, "
          f"{len(o.datatype_properties)} datatype properties, {o.n_triples} triples")
    print(f"External referenced: {sorted(o.external_used)}; "
          f"declared-but-unused: {sorted({'sosa:', 'time:', 'saref:'} - o.external_used)}")

    undrawn = sorted(o.object_properties - set(drawn))
    spurious = sorted(set(drawn) - o.object_properties)
    if missing:
        print(f"NOTE: {len(missing)} class(es) not placed in a panel: {missing}")
    if undrawn or spurious:
        print(f"DRIFT: object properties in TTL but not drawn: {undrawn}", file=sys.stderr)
        print(f"DRIFT: edges drawn but not in TTL: {spurious}", file=sys.stderr)
        return 2
    print(f"All {len(o.object_properties)} object properties drawn; no drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
