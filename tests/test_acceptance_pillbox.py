# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The acceptance gate on a shape whose wall curves and whose ends do not.

The sphere gate solves a surface curved everywhere, and every other gate here
draws boxes and flat sheets the grid holds exactly. A cylinder is the shape that
puts the two next to each other, and the two ask opposite things of the
translation: the wall has to be grown before openEMS samples it, because a metal
edge is decided on one point and so arrives inscribed, and the ends have to stay
exactly where the grid holds them. A rule that did one of those to both surfaces
is wrong in a way no sphere and no box can show.

The mode read is the one with no variation along the axis. Its frequency is the
first zero of ``J0`` over the radius and depends on nothing else - so what is
being scored is one curved wall, in frequency units, and the flat ends are out of
the answer twice over: the grid holds them and the mode would not notice if it
did not.

What this gate scores
---------------------

**The answer, against the uncertainty the study computed.** A resonance off a
curved conductor is claimed to be good for at any cell this workbench's own mesh
policy would choose, and it is held at the coarsest as well as the finest.

**Where the wall ended up.** The same distance read as a length rather than a
frequency, and in cells - which is the unit the thing being removed is measured
in. A correction gone missing puts it back near half a cell; one of the wrong
size leaves its own fixed share.

**The same cavity at two heights.** One resonance, no reference, no error bar. A
closed form computed from the radius cannot notice the height leaking into the
answer, and this is the check that does.

**Where the lattice falls.** A cell size says how big the cells are and nothing
about where they fall. Every case here carries the same drawing and the same
cells and differs only in where the lines sit against it.

**What the probe is worth.** It is the one object in the arrangement that is not
geometry: openEMS builds a lumped port as two conducting plates joined by a
resistance, so it is metal standing in the field. Solving the same cavity with
the element doubled prices it on the same scale as everything else.

**The rest of the band.** One further case stands the same element off the axis
and off mid-height, where the four modes with an axial electric field are all
alive rather than sitting on a node - so one solve is scored against four exact
Bessel roots instead of one. Two of them have no variation along the axis and
are held to the claim above; one of *those* has azimuthal variation, which is a
placement of the curved wall the dominant line cannot see at all. The two with a
half-wave along the axis are held an order looser, and
:data:`AXIAL_MODE_ACCURACY` says why.

What has to be held still for the sequence to be about the cell
--------------------------------------------------------------

A cylinder standing on the mesh axis presents *the same circle* to the grid at
every height. Where a surface curved in two directions meets the lattice at many
phases at once and averages them, this one repeats one phase down its whole
length - so where the lines fall against the wall stays in the answer instead of
cancelling, and sliding the lattice through one cell moves the line as far as
refining the cell across the whole sequence does.

A cell size says how big the cells are and nothing about where they fall, so a
sequence of cell sizes alone varies the size and the phase together and an
exponent fitted through it measures neither. What is held still instead:

- **The wall stands at the same share of its own cell in every case.** Each mesh
  is slid until it does; :data:`~tests.pillbox.WALL_PHASE` says where and why
  that is a choice rather than a derivation, and
  ``test_pillbox_drawing`` asserts what each mesh achieved.
- **The probe is the same object at every cell.** openEMS builds a lumped element
  from its box snapped to the grid, so a probe left to whatever lines the grading
  left is rebuilt at a size the drawing never stated - and this one is metal
  standing in the field, so its size is in the answer. The adapter pins the box's
  faces, and an alignment case carries the probe along with the mesh it slides.

What the sequence then measures is the cell, at one alignment. What the alignment
is worth is measured beside it and reported on its own line, because a user
solving this cavity once gets whichever alignment the mesher gave them and needs
both halves to know what they have.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from Microwave.Solvers.openems import preflight, read, run, staircase
from Microwave.Solvers.openems.model import CONDUCTOR_KINDS, Problem
from Microwave.Solvers.openems.surface import parts
from tests import convergence, pillbox, polyhedron, resonance, solving
from tests.conftest import draw_cases
from tests.triangulated import bore_volume, bores_apart

pytestmark = pytest.mark.slow

#: Every case, as parameters. A property of one solved line is asserted at all of
#: them and solved at one, :data:`~tests.pillbox.NOMINAL` being the operating
#: point and the rest a study.
EVERY_CASE = solving.parameters(pillbox.CASES, pillbox.NOMINAL)

#: The cases that made both corrections, which is what the rounding is asserted
#: on. The one handed its flat faces on the grid line is the other half of a pair
#: and is scored as a difference from its partner instead.
CORRECTED = solving.parameters(
    [name for name, case in pillbox.CASES.items() if case.clearance], pillbox.NOMINAL
)

PROBE = os.path.join(os.path.dirname(__file__), "pillbox_probe.py")

#: How far apart, in mm, the polygons the cases were given may be before they
#: stop being one shape. A refinement study holds the drawing still and moves
#: only the mesh, so a sequence whose geometry drifted between points is not one.
POLYGON_ALIKE = 1e-6

#: How much of a cell the correction may leave the wall displaced by. What
#: remains once the rounding is removed is a real surface sampling the cell at
#: one phase rather than at all of them, and a correction of the wrong size would
#: show here first, as a displacement that is once again a fixed share of a cell.
WALL_LEFT_OF_THE_ROUNDING = 0.25

#: How far apart two heights of one cavity may answer, as a share of what a whole
#: cell of displacement is worth. The mode has no variation along the axis, so
#: the two are one resonance and anything between them is the discretisation
#: reaching a dimension the physics does not use.
HEIGHT_SHARE = 0.05

#: How far apart the same mesh may answer when it is slid under the drawing, as a
#: share of what a whole cell of displacement is worth. It is also the error bar
#: on every other figure read off a single solve, one solve being one alignment.
LATTICE_SHARE = 0.25

#: How far doubling the probe may move the line, on the same scale. The probe is
#: a conductor inside the resonator, so it pulls the line as well as damping it,
#: and what this holds is that the cavity is what is being measured.
PROBE_SHARE = 0.4

#: How deep the line has to stand above what the fit left behind. A fit through
#: noise leaves a residual the size of the depth it claims.
LINE_OVER_NOISE = 10.0

#: What a mode with no variation along the axis is claimed to be good for, as a
#: share of its own frequency.
#:
#: Written down rather than computed, because these lines come off a *single*
#: solve: the modes are read from one spectrum, so there is no sequence behind
#: them and nothing to run a refinement study on. What licenses the figure is the
#: study the dominant line does have - these are held to the interval that one
#: earns, at a second Bessel root and at an azimuthal order the dominant line has
#: not got.
FLAT_MODE_ACCURACY = 0.008

#: And what a mode with a half-wave along the axis is good for, which is looser
#: by more than an order at the coarsest cell this cavity is solved at.
#:
#: **Such a mode carries transverse electric field, and on a staircased wall that
#: is a different boundary from the one the axial field meets.** openEMS zeroes a
#: field edge whose own sample point reads metal, and the Yee cell puts the axial
#: and transverse components at different points - so on a curved wall the two
#: are asked about the conductor where they do not agree, and a mode constrained
#: by both is bounded by neither alone. A mode with no axial variation carries
#: only the axial field and meets one wall.
#:
#: Nothing in this is the flat ends: they are held on grid lines, where the two
#: samplings coincide, and a rectangular cavity read the same way gives its
#: dimensions back to within a tenth of a cell.
AXIAL_MODE_ACCURACY = 0.02

#: How much of the loss the declared fill has to carry against everything else
#: together - the probe's resistance, and whatever a staircase dissipates on its
#: own. At one it is simply the larger share, which is what makes the line's
#: position a statement about the shape.
FILL_DOMINATES = 1.0

#: How far apart the two members of the clearance pair have to answer, in the
#: same cells :data:`WALL_LEFT_OF_THE_ROUNDING` is stated in.
#:
#: Twice that rather than a figure of its own, which is the whole claim: a
#: corrected run is held inside a quarter of a cell, so a pair that separates a
#: wall which is a conducting boundary from one which is not has to stand clear
#: of that on both sides. Stated as a floor because how far apart is not one
#: number - openEMS zeroes an edge or it does not, so a face that failed costs a
#: whole cell of boundary, and how many faces fail is decided by the containment
#: segment rather than by the drawing.
CAP_IS_WORTH = 2.0 * WALL_LEFT_OF_THE_ROUNDING


@pytest.fixture(scope="module")
def envelopes(tmp_path_factory):
    """Draw every case under a real FreeCAD, once."""
    return draw_cases(PROBE, tmp_path_factory.mktemp("pillbox"), "PILLBOX_OUT")


def _envelope(name: str, envelopes):
    """One case as the kernel drew it, with nothing solved."""
    return Problem.from_dict(json.loads((envelopes[name] / "openems.json").read_text()))


def _one_solid(name: str, problem: Problem, material: str, what: str, surfaces: int):
    """The one triangulated solid of a material, arriving in the shape it should.

    ``surfaces`` is how many closed surfaces its triangulation is: the can is
    two, an outside and a bore, and the fill is one. Once they are triangles that
    is all that tells the two apart, and holding each to its own stops a reading
    pointed at the wrong solid - taking the smaller of a cylinder's surfaces
    takes the only one it has, so it answers plausibly.
    """
    found = [solid for solid in problem.solids if solid.material == material and solid.faces]
    assert len(found) == 1, (
        f"{name}: the cavity's {what} came through as {len(found)} triangulated "
        f"solids, so what surface openEMS was given for the {what} is not answerable here"
    )
    held = len(parts(found[0].faces))
    assert held == surfaces, (
        f"{name}: the cavity's {what} reached the engine as {held} closed surfaces "
        f"rather than {surfaces}, so it is not the {what} this reads it as"
    )
    return found[0]


def _wall_of(name: str, problem: Problem) -> float:
    """The radius of the can's inner triangulation, in mm, from its own faces.

    Read off the can and not the fill, because the conductor handed to the engine
    is made from the can, grown by half a cell on the way there: the metal does
    not start here, and ``line["displacement"]`` is what that leaves. The fill's
    outer triangulation is a second answer to the same drawn circle;
    ``test_the_can_and_the_fill_are_drawn_to_one_wall`` holds the two together.
    """
    can = _one_solid(name, problem, "Copper", "can", surfaces=2)
    return pillbox.radius_of(bore_volume(can.vertices, can.faces), pillbox.CASES[name].height)


def _fill_of(name: str, problem: Problem) -> float:
    """The radius the fill's own triangulation reaches, in mm, off its faces."""
    fill = _one_solid(name, problem, "Vacuum", "fill", surfaces=1)
    return pillbox.radius_of(bore_volume(fill.vertices, fill.faces), pillbox.CASES[name].height)


def _solve(name: str, envelopes, interpreter):
    """One case, read off the envelope the kernel drew and solved."""
    directory = envelopes[name]
    problem = _envelope(name, envelopes)
    preflight.refuse_if_blocked(preflight.check(problem))
    run.run(str(directory / "openems.json"), interpreter=interpreter)
    return (read.read(str(directory)), problem)


@pytest.fixture(scope="module")
def solved(envelopes, interpreter):
    """Each case, solved when something asks for it and once."""
    return solving.Cases(lambda name: _solve(name, envelopes, interpreter))


def _measure(name: str, solved):
    """One case as the line it answered with, and the mesh it answered on."""
    want = pillbox.frequency()
    result, problem = solved(name)
    line = resonance.fit_line(
        result.frequency,
        1.0 - np.abs(result.s(1, 1)) ** 2,
        about=want,
        window=pillbox.WINDOW,
        quality=1.0 / pillbox.loss_tangent(),
    )
    # The cell the wall was sampled on, off the finished grid rather than off
    # what the policy was asked for: the mesher answers to the drawing too,
    # and the two parting would leave every figure below against the wrong
    # length.
    line["cell"] = pillbox.wall_cell(problem.grid.x)
    # The radius openEMS was *given*, off the triangles in the envelope
    # rather than off the drawing. A chord cuts inside the arc it spans, so
    # the polygon is inscribed by an amount the triangulation sets and no cell
    # moves - a constant under every figure below, and the one kind of term a
    # refinement study cannot see, since the errors approach it instead of
    # zero and nothing says the geometry is what stalled.
    line["given"] = _wall_of(name, problem)
    line["want"] = pillbox.frequency(radius=line["given"])
    line["error"] = (line["centre"] - line["want"]) / line["want"]
    # What the case was drawn as, so a check comparing two of them names the
    # cavities rather than the strings it looked them up by.
    line["height"] = pillbox.CASES[name].height
    line["probe"] = pillbox.CASES[name].probe
    # The mode's whole geometry dependence is one Bessel root over the
    # radius, so a measured frequency is a measured radius - which is the
    # quantity a staircase displaces and the one it is legible in.
    line["displacement"] = pillbox.effective_radius(line["centre"]) - pillbox.RADIUS
    print(
        f"GATE pillbox {name}: cell {line['cell']:.4f} mm, "
        f"line {line['centre'] / 1e9:.4f} GHz against {want / 1e9:.4f} "
        f"({100 * line['error']:+.3f} %), "
        f"Q {line['quality']:.1f} of {1 / pillbox.loss_tangent():.1f}, "
        f"wall out {line['displacement'] / line['cell']:+.3f} cells"
    )
    return line


@pytest.fixture(scope="module")
def lines(solved):
    """Each case as it is asked for, measured once and kept."""
    return solving.Cases(lambda name: _measure(name, solved))


@pytest.fixture(scope="module")
def sequence(lines):
    """The tall cavity at every cell, coarsest first."""
    return [lines(name) for name in pillbox.SEQUENCE]


@pytest.fixture(scope="module")
def alignments(lines):
    """One mesh in every place it was solved.

    The drawing's own alignment is the sequence's coarsest point rather than a
    case of its own, so it leads these.
    """
    found = [lines(pillbox.SEQUENCE[0])]
    found += [lines(pillbox.phase_case(phase)) for phase in pillbox.LATTICE_PHASES]
    cells = {round(line["cell"], 9) for line in found}
    assert len(cells) == 1, (
        f"the alignments came out on different cells - {sorted(cells)} mm - so what "
        "separates their answers is the mesh as well as where it fell"
    )
    return found


def _lattice_spread(alignments) -> float:
    """How far apart the alignments answered, as a share of the frequency."""
    return float(np.ptp([line["centre"] for line in alignments]) / pillbox.frequency())


@pytest.fixture(scope="module")
def spectrum(solved):
    """The spectrum case as one fitted line per mode its probe can drive.

    Every other case reads the dominant line through an element on the axis at
    mid-height, which sits on a node of everything else in the band. This one
    stands the same element off both, where the four modes with an axial electric
    field are all alive - and each of those is an exact Bessel root, off one
    solve.
    """
    result, problem = solved(pillbox.SPECTRUM_CASE)
    absorbed = 1.0 - np.abs(result.s(1, 1)) ** 2
    cell = pillbox.wall_cell(problem.grid.x)
    found = {}
    for mode in pillbox.SPECTRUM_MODES:
        want = pillbox.mode_frequency(mode)
        line = resonance.fit_line(
            result.frequency,
            absorbed,
            about=want,
            window=pillbox.SPECTRUM_WINDOW,
            quality=1.0 / pillbox.loss_tangent(want),
        )
        line["cell"] = cell
        line["want"] = want
        line["error"] = (line["centre"] - want) / want
        found[mode] = line
        print(
            f"GATE pillbox {pillbox.mode_name(mode)}: {line['centre'] / 1e9:.4f} GHz against "
            f"{want / 1e9:.4f} ({100 * line['error']:+.3f} %), Q {line['quality']:.1f}, "
            f"driven at {abs(pillbox.axial_field(mode)):.3f} of its own crest"
        )
    return found


def _across_a_cap(fill, at):
    """Points spread over one cap of the bore, where the pinned line falls."""
    centre = [(low + high) / 2.0 for low, high in zip(fill.lower, fill.upper)]
    return polyhedron.across_a_disc(centre, (fill.upper[0] - fill.lower[0]) / 2.0, at)


@pytest.fixture(scope="module")
def caps(envelopes):
    """Every case's bore caps: where each is, what lies beyond it in the metal,
    and whether openEMS holds the line the mesher pinned to it.

    Asked of the vertices the *driver* hands over - placed at the origin, grown,
    and displaced or not - because how far into a face containment stays unsure
    moves with where the solid sits and how many faces the segment crosses on its
    way out. Asked at both clearances for every case, so what is compared is the
    displacement rather than which case happens to carry which.

    **A sibling of the run's own answer rather than a replay of it.** The segment
    each primitive is tested against ends at a point drawn from ``rand()`` when
    that primitive is built, so which faces of a shape read as air is fixed for a
    primitive and not for a shape - and the run built its own in another process.
    What survives that is the answer for a point strictly *inside*: every segment
    out of one crosses an odd number of faces wherever it ends. So the census
    asserts the displaced faces and prints the rest.
    """
    found = {}
    for name in sorted(envelopes):
        problem, _ = _envelope(name, envelopes).at_the_origin()
        metals = {m.name for m in problem.materials if m.kind in CONDUCTOR_KINDS}
        pieces = [s for s in problem.solids if s.is_mesh and not s.is_sheet]
        shell = [s for s in pieces if s.material in metals]
        fill = [s for s in pieces if s.material not in metals]
        assert len(shell) == 1 and len(fill) == 1, (
            f"{name}: the can came through as {len(shell)} conducting and {len(fill)} "
            "filled surfaces, so which faces close the cavity is not answerable here"
        )
        shell, fill = shell[0], fill[0]
        lines = tuple(problem.grid[dim] for dim in range(3))
        zlines = np.asarray(lines[2])
        surfaces = {
            clearance: staircase.grown(
                shell.vertices, shell.faces, lines, problem.grown_by, clearance
            )
            for clearance in (staircase.PINNED_CLEARANCE, 0.0)
        }
        here = []
        for at, outward in ((fill.upper[2], 1), (fill.lower[2], -1)):
            points = _across_a_cap(fill, at)
            # The mesher pins a line to the face, so the neighbour on the metal
            # side is where a boundary that failed at the face is built instead.
            index = int(np.searchsorted(zlines, at, side="right" if outward > 0 else "left"))
            here.append(
                {
                    "at": at,
                    "beyond": abs(float(zlines[index if outward > 0 else index - 1]) - at),
                    **{
                        key: all(polyhedron.holds(surfaces[clearance], shell.faces, points))
                        for key, clearance in (
                            ("held", staircase.PINNED_CLEARANCE),
                            ("held_as_drawn", 0.0),
                        )
                    },
                }
            )
        found[name] = tuple(here)
    return found


# ---------------------------------------------------------------------------
# What the reading has to be before any of it means anything
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(pillbox.CASES))
def test_the_can_and_the_fill_are_drawn_to_one_wall(name, envelopes):
    """The can's inner triangulation and the fill's outer one are one surface.

    Two solids meet at the cavity wall and each is triangulated on its own: the
    fill is a cylinder, the can is a larger cylinder with that one cut out of
    it. The conductor is made from the can, so that is the surface the gate
    scores and nothing it asserts is read off the fill - which leaves this free
    to fail.

    What it protects is the drawing rather than the score. Where the fill stops
    short of the can, a layer at the wall is claimed by neither solid, so the
    cavity is not filled with the material whose loss
    ``test_the_fill_is_what_damps_the_ring`` holds the ring's Q against. Where
    the fill overruns instead, the can's priority hides it.

    They are one surface because each solid is triangulated at a deflection
    scaled off its own size - the can being larger by a wall at each end - and
    the kernel answers a wide band of requests with one mesh. That stops holding
    as soon as a triangulation here is asked for finer than the band.

    Compared as surfaces and not as sizes, which is what
    ``triangulated.bores_apart`` is for: a volume is one number, and a bore drawn
    off centre encloses exactly what it did. The volume is asked after the
    corners rather than instead of them, one set of corners carrying more than
    one triangulation.

    Every case rather than the operating point alone, and so not through
    ``solving.parameters``: what puts a case behind the release marker is the
    solve it costs, and this reads the envelope and costs none.
    """
    problem = _envelope(name, envelopes)
    can = _one_solid(name, problem, "Copper", "can", surfaces=2)
    fill = _one_solid(name, problem, "Vacuum", "fill", surfaces=1)
    apart = bores_apart((can.vertices, can.faces), (fill.vertices, fill.faces))
    assert apart < POLYGON_ALIKE, (
        f"{name}: the can's bore and the fill stand {apart:.3e} mm apart at a "
        "corner, so the two solids are triangulated to different surfaces and a "
        "layer at the wall belongs to neither of them"
    )
    wall, held = _wall_of(name, problem), _fill_of(name, problem)
    assert abs(wall - held) < POLYGON_ALIKE, (
        f"{name}: the can is drawn to {wall:.9f} mm and the fill to {held:.9f} mm, "
        "so the two run through the same corners and enclose different volumes"
    )


@pytest.mark.parametrize("name", EVERY_CASE)
def test_a_line_was_found_rather_than_the_noise_floor(lines, name):
    """Deep against what the fit left behind, and shallow against what a share of
    the offered power can be.

    The second is the one that catches a fit that ran away rather than one that
    found nothing: turned loose on a flank instead of a line, ``curve_fit``
    answers with a centre outside the window, a Q near zero and a depth of tens -
    which passes any test that only asks for depth over residual, the residual
    of a curve through a monotone flank being tiny.
    """
    line = lines(name)
    assert line["depth"] <= 1.0, (
        f"{name}: the fit claims the cavity took in {line['depth']:.1f} times the "
        "power offered to it, so what it found is not a resonance"
    )
    assert line["depth"] > LINE_OVER_NOISE * line["residual"], (
        f"{name}: the line is {line['depth']:.4f} deep against a fit residual of "
        f"{line['residual']:.4f}, which is not a resonance"
    )


@pytest.mark.parametrize("name", EVERY_CASE)
def test_the_fill_is_what_damps_the_ring(lines, name):
    """The probe is a resistor inside the resonator, so it damps what it measures
    and pulls the line down as it grows. So does a coarse staircase, for reasons
    of its own - which is why this asks how the loss divides rather than how near
    Q comes to the fill's own number, a quantity the two spend together.

    Losses add as reciprocals, so what everything other than the fill carries is
    ``1/Q - tan(delta)``. It has to be positive, because a cavity cannot be less
    lossy than what fills it, and it has to be the smaller share.
    """
    fill = pillbox.loss_tangent()
    line = lines(name)
    rest = 1.0 / line["quality"] - fill
    assert rest > 0.0, (
        f"{name}: Q came back {line['quality']:.1f}, above the fill's own "
        f"{1 / fill:.1f} - a cavity cannot be less lossy than what fills it"
    )
    assert fill > FILL_DOMINATES * rest, (
        f"{name}: everything but the fill carries a loss of {rest:.2e} against "
        f"the fill's {fill:.2e}, so what damps the ring is not what was declared"
    )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_the_closed_form_is_inside_the_uncertainty_the_study_computed(sequence):
    """The gate on the answer, and two claims about it.

    The interval the study computed contains the closed form, which is what says
    refinement is heading where the drawing says it should; and that interval is
    still an interval, narrower than the run of answers it was computed from. A
    band wider than that has priced the sequence at more than it contains, and
    covering the answer with one costs nothing.

    **The second is a statement about the study's design rather than about its
    noise**, and where the procedure fitted a single free exponent to a sequence
    it could read, it is arithmetic. The band is then the safety factor times the
    error the fit puts on the finest grid, and the run of answers is the coarsest
    of those errors less the finest, so asking for one below the other is asking
    for::

        (coarsest cell / finest) ** order > 1 + safety

    - the lever arm and the rate, together. It is printed where it holds and not
    where it does not: an unreadable sequence is priced by a different formula
    entirely, a fixed-order fallback carries two terms rather than one, and the
    order reported is the one the *data* supported rather than the one the chosen
    fit used. Stating it in those cases would be arithmetic about a band that was
    not computed that way.
    """
    radii = [line["given"] for line in sequence]
    assert max(radii) - min(radii) < POLYGON_ALIKE, (
        f"the cases were given cylinders of {min(radii):.6f} to {max(radii):.6f} mm, "
        "so this sequence changes the shape as well as the cell and no single "
        "reference describes it"
    )
    want = float(np.mean([line["want"] for line in sequence]))
    drawn = pillbox.frequency()
    estimate = convergence.uncertainty_of(
        [line["cell"] for line in sequence], [line["centre"] for line in sequence]
    )
    off = (estimate.finest - want) / want
    cells = [line["cell"] for line in sequence]
    lever = max(cells) / min(cells)
    # The band is the safety factor times one term only where the sequence was
    # readable and a single free exponent described it. Anywhere else the
    # procedure builds it differently, and the lever-arm reading would be
    # arithmetic about a band that was not computed that way.
    reduces = estimate.readable and estimate.expansion == "free order"
    print(
        f"\nGATE pillbox band: {estimate.finest / 1e9:.4f} GHz against "
        f"{want / 1e9:.4f} for the polygon it was given, drawn {drawn / 1e9:.4f} "
        f"({100 * off:+.3f} %), uncertainty +/-{100 * estimate.uncertainty / want:.3f} % "
        f"at order {estimate.order:.2f} by the {estimate.expansion} expansion, "
        f"safety {estimate.safety:g}, scatter {estimate.scatter:.4g} against a data "
        f"range of {estimate.data_range:.4g}; refinement heads for "
        f"{estimate.limit / 1e9:.4f} GHz "
        f"({100 * (estimate.limit - want) / want:+.3f} %)"
        + (
            f"; a lever arm of {lever:.2f} at that order reaches "
            f"{lever**estimate.order:.2f} against the {1 + estimate.safety:g} the band asks"
            if reduces
            else ""
        )
    )
    assert estimate.covers(want), (
        f"{100 * off:+.3f} % from the closed form against an uncertainty of "
        f"{100 * estimate.uncertainty / want:.3f} %, so refinement is heading for "
        f"{estimate.limit / 1e9:.4f} GHz and the polygon resonates at "
        f"{want / 1e9:.4f} GHz"
    )
    spread = float(np.ptp([line["centre"] for line in sequence]))
    assert estimate.uncertainty < spread, (
        f"the band is +/-{estimate.uncertainty / 1e6:.3f} MHz on a sequence whose "
        f"answers span {spread / 1e6:.3f} MHz, so covering the closed form with it asks "
        "nothing of the run. "
        + (
            f"A lever arm of {lever:.2f} at an observed order of {estimate.order:.2f} "
            f"reaches {lever**estimate.order:.2f} against the {1 + estimate.safety:g} the "
            "band asks, so the study is either too short or converging too slowly and "
            "those two figures say which"
            if reduces
            else f"The {estimate.expansion!r} expansion was chosen at an observed order "
            f"of {estimate.order:.2f}, readable {estimate.readable}, so the band is not "
            "the safety factor times one term and what is short cannot be read off the "
            "lever arm"
        )
    )


@pytest.mark.parametrize("name", CORRECTED)
def test_the_rounding_openems_makes_has_been_taken_out_of_the_wall(lines, name):
    """What openEMS is handed is grown by half the cell it will be sampled on, so
    what it builds should land on the drawing rather than a rounding inside it.

    Asserted against the cell, because that is the size of the thing being
    removed. It is also the half of this gate that the flat ends are in: a rule
    that stepped them the way it steps the wall moves the ends off the lines that
    hold them, and a wall of the cavity that is not a conducting boundary shows
    up here as the bore reading larger than it was drawn.

    Of the cases that made both corrections. The one that was handed its flat
    faces on the grid line rather than displaced off it is the pair's other half
    and is scored as a difference from its partner, not against this bar - what
    it reads is the fault, and reading it is the point.
    """
    share = lines(name)["displacement"] / lines(name)["cell"]
    assert abs(share) < WALL_LEFT_OF_THE_ROUNDING, (
        f"{name}: the wall is {share:+.3f} cells from where it was drawn, which "
        "is the size of a rounding rather than what one leaves"
    )


def test_one_cavity_at_two_heights_reads_one_resonance(lines):
    """No reference and no error bar. The mode has no variation along the axis,
    so its frequency is the first zero of ``J0`` over the radius and the height
    is not in it - and a closed form computed from the radius cannot notice the
    height leaking into the answer.

    Both are solved at the coarsest cell, where a staircase is largest, and the
    mesher plans the same cross-section for each.
    """
    coarsest = min(pillbox.DIVISORS)
    tall, short = lines(f"tall-{coarsest}"), lines(f"short-{coarsest}")
    assert tall["height"] != short["height"], "both cases are the same cavity"
    apart = abs(short["centre"] - tall["centre"]) / tall["centre"]
    worth = pillbox.displacement_worth(tall["cell"])
    print(
        f"\nGATE pillbox height: {tall['height']:g} mm and {short['height']:g} mm answer "
        f"{100 * apart:.4f} % apart, against {100 * worth:.3f} % for a whole cell "
        "of displacement"
    )
    assert apart < HEIGHT_SHARE * worth, (
        f"the two heights answer {100 * apart:.4f} % apart, which is not small beside "
        f"the {100 * worth:.3f} % a cell of displacement is worth - so the height is "
        "in an answer the physics says it is not in"
    )


def test_where_the_lattice_falls_is_worth_a_fraction_of_a_cell(alignments):
    """The same mesh, slid under the drawing.

    openEMS decides a conducting boundary by sampling, so two grids of the same
    cell catch the wall at different points of it. Across the axis only: the mode
    does not use the third direction and the ends are flat, so there is nothing
    to slide along it.
    """
    spread = _lattice_spread(alignments)
    worth = pillbox.displacement_worth(alignments[0]["cell"])
    print(
        f"\nGATE pillbox lattice: {len(alignments)} alignments at "
        f"{alignments[0]['cell']:.4f} mm answer {100 * spread:.3f} % apart, against "
        f"{100 * worth:.3f} % for a whole cell of displacement"
    )
    assert spread < LATTICE_SHARE * worth, (
        f"sliding the grid under the drawing moved the line {100 * spread:.3f} %, which "
        f"is not a fraction of the {100 * worth:.3f} % a whole cell of displacement is "
        "worth - so the answer belongs to where the grid fell rather than to the drawing"
    )


def test_the_sequence_falls_with_every_refinement(sequence, alignments):
    """What licenses reading the sequence as a rate rather than as four answers.

    The alignment is held and the probe is the same object at every cell, so the
    cell is the only thing separating these solves - and a sequence that did not
    descend would be one where something else was still varying. Asked
    assumption-free: it says the errors are heading somewhere without saying how
    fast, which is what has to hold before an exponent is worth reading off them.

    The exponent is printed beside what an alignment reaches, because those are
    the two halves a reader needs. The first says how fast this drawing converges
    once the lattice is held; the second says what holding it was worth, and it
    is measured at the coarsest cell of the sequence, where sampling can displace
    a boundary furthest.
    """
    cells = [line["cell"] for line in sequence]
    errors = [abs(line["error"]) for line in sequence]
    spread = _lattice_spread(alignments)
    print(
        f"\nGATE pillbox sequence: {100 * errors[0]:.3f} % down to {100 * errors[-1]:.3f} % "
        f"across the cells solved, at an exponent of "
        f"{convergence.order_of(cells, errors):.2f}; sliding the coarsest mesh under the "
        f"drawing spans {100 * spread:.3f} %"
    )
    assert convergence.falls_with_every_refinement(cells, errors), (
        "the errors run "
        + ", ".join(f"{100 * error:.4f} %" for error in errors)
        + " down the cells "
        + ", ".join(f"{cell:.4f} mm" for cell in cells)
        + ", so refining made one of them worse. With the lattice held and the probe "
        "built at the size it was drawn, what is left to look at is whatever else the "
        "grid decides and the drawing does not"
    )


@pytest.mark.release
def test_every_line_the_spectrum_scores_was_found_where_it_was_looked_for(spectrum):
    """A fit is free to place its line anywhere, including outside the window it
    was given data from - and there it is fitting a flank rather than a line.

    It is the check the crowded part of this band needs: two of these sit close
    enough that a fit which wandered out of its own window would land on its
    neighbour's line and report a real resonance against the wrong root.
    """
    for mode, line in spectrum.items():
        reach = pillbox.SPECTRUM_WINDOW * line["want"]
        assert abs(line["centre"] - line["want"]) < reach, (
            f"{pillbox.mode_name(mode)} was looked for within {line['want'] / 1e9:.4f} GHz "
            f"+- {reach / 1e9:.4f} and the fit answered {line['centre'] / 1e9:.4f}, which is "
            "outside the data it was given"
        )
        assert line["depth"] <= 1.0, (
            f"{pillbox.mode_name(mode)}: the fit claims the cavity took in "
            f"{line['depth']:.1f} times the power offered to it"
        )


@pytest.mark.release
def test_the_whole_band_agrees_with_its_own_exact_root(spectrum):
    """Four modes off one solve, each an exact Bessel root, where every other
    case here reads one.

    Two of them have no variation along the axis and are held to the same claim
    the dominant line is - and one of *those* carries azimuthal variation, which
    is a placement of the curved wall the dominant line cannot see at all, its
    own field being the same all the way round.

    The two with a half-wave along the axis are held looser, and
    :data:`AXIAL_MODE_ACCURACY` says why: they carry transverse electric field,
    which on a staircased wall is zeroed at different points of the Yee cell from
    the axial field, so they are bounded by a wall the flat modes never meet.
    """
    for mode, line in spectrum.items():
        axial = mode[3] > 0
        bar = AXIAL_MODE_ACCURACY if axial else FLAT_MODE_ACCURACY
        assert abs(line["error"]) < bar, (
            f"{pillbox.mode_name(mode)}: {100 * line['error']:+.3f} % from its exact root "
            f"at a cell of {line['cell']:.4f} mm, against a claim of {100 * bar:g} %"
        )


@pytest.mark.release
def test_the_two_kinds_of_mode_miss_their_root_in_opposite_directions(spectrum):
    """The signature of the two walls, and the one thing in this band that needs
    no bar to be read.

    The modes carrying only the axial field land *above* their root and the ones
    carrying transverse field as well land *below* it, at every cell. Which is
    the more accurate is not the claim and does not hold - at some meshes an
    axial mode is nearer its root than a flat one is to its own, the flat ones
    carrying the wall's staircase and the lattice with it. What separates them is
    that they are bounded by different walls, and a difference in direction says
    that where a difference in size does not.
    """
    print(
        "\nGATE pillbox spectrum: "
        + ", ".join(
            f"{pillbox.mode_name(mode)} {100 * spectrum[mode]['error']:+.3f} %"
            for mode in pillbox.SPECTRUM_MODES
        )
    )
    for mode, line in spectrum.items():
        if mode[3] == 0:
            assert line["error"] > 0.0, (
                f"{pillbox.mode_name(mode)} carries no transverse field and reads "
                f"{100 * line['error']:+.3f} %, on the far side of its root from every "
                "other mode of its kind"
            )
        else:
            assert line["error"] < 0.0, (
                f"{pillbox.mode_name(mode)} carries transverse field and reads "
                f"{100 * line['error']:+.3f} %, on the far side of its root from every "
                "other mode of its kind"
            )


def test_the_cavity_is_what_is_being_measured_and_not_the_probe(lines):
    """The probe is not a circuit element beside the resonator: openEMS builds a
    lumped port with conducting caps, so it is metal standing in the field, and
    it snaps outward to whatever lines are nearest.

    Doubled rather than nudged, so what it moves stands above everything else
    that moves an answer here and can be read as a bound on it.
    """
    coarsest = min(pillbox.DIVISORS)
    plain, doubled = lines(f"tall-{coarsest}"), lines(f"probe-{coarsest}")
    assert doubled["probe"] > plain["probe"], "both cases carry the same element"
    apart = abs(doubled["centre"] - plain["centre"]) / plain["centre"]
    worth = pillbox.displacement_worth(plain["cell"])
    print(
        f"\nGATE pillbox probe: doubling it to {doubled['probe']:g} mm moved the line "
        f"{100 * apart:.3f} % and Q from {plain['quality']:.1f} to {doubled['quality']:.1f}, "
        f"against {100 * worth:.3f} % for a whole cell of displacement"
    )
    assert apart < PROBE_SHARE * worth, (
        f"doubling the probe moved the line {100 * apart:.3f} %, against the "
        f"{100 * worth:.3f} % a whole cell of the wall is worth - so what the gate reads "
        "is the fixture as much as the cavity"
    )


# ---------------------------------------------------------------------------
# Whether the wall the drawing states is built at all
# ---------------------------------------------------------------------------


def test_a_line_pinned_to_a_flat_face_is_in_the_metal_only_because_it_was_displaced(caps):
    """The mesher puts a grid line on a flat conductor face square to an axis,
    which is what keeps that wall where it was drawn. openEMS then decides whether
    the point on that line is metal by casting a segment from it and counting the
    faces crossed, and a point lying *on* a face gives that count nothing to be
    sure about.

    So what is asserted is the half that cannot go either way: displaced into the
    void, the pinned line is strictly inside the metal, and a segment out of an
    interior point crosses an odd number of faces wherever it ends. Handed over on
    the face, the answer belongs to that primitive's own segment - it differs
    between the two caps of one can and between cases drawing the same can on
    different cells, and it is not the answer the run's own process got. That is
    printed, and the pair is what reads it on Maxwell instead.
    """
    for name, here in sorted(caps.items()):
        print(
            f"\nGATE pillbox clearance {name}: "
            + ", ".join(
                f"cap at {cap['at']:.4f} mm {'holds' if cap['held'] else 'READS AS AIR'} the "
                f"pinned line and {'holds' if cap['held_as_drawn'] else 'does not hold'} it "
                f"as drawn, with a line {cap['beyond']:.4f} mm further into the metal"
                for cap in here
            )
        )
    assert all(cap["held"] for here in caps.values() for cap in here), (
        "a cap displaced by the clearance still reads as air at the line pinned to it, "
        "so the wall the drawing states is not built and the cavity is the wrong size"
    )


def test_without_the_clearance_the_cavity_openems_builds_is_a_cell_too_big(lines):
    """The pair, and what it prices.

    Both members carry the same drawing, the same cells, the same alignment and
    the same probe, and differ by a displacement of a thousandth of a cell on the
    flat faces alone. What that displacement buys is whether the line the mesher
    pinned to each face falls in metal - and where it does not, the field standing
    on the conductor's own plane is never zeroed and the structure openEMS built
    is not the one the drawing states.

    **The unit is borrowed and the fault is not where the unit says.** This line's
    frequency is a Bessel root over the radius and nothing else *for the cavity
    the drawing states*, so reading the shift in cells of radial displacement is a
    scale rather than an attribution: what the run was given is a different cavity
    - one end a cell out, and the rim where that end meets the wall unresolved -
    and no single dimension of the drawing describes it. The scale is the one
    every other figure in this gate is stated in, which is why it is used.
    """
    on_the_face, drawn = (lines(name) for name in pillbox.ON_THE_FACE)
    fell = (on_the_face["centre"] - drawn["centre"]) / drawn["centre"]
    apart = abs(fell) / pillbox.displacement_worth(drawn["cell"])
    print(
        f"\nGATE pillbox clearance: handing the flat faces over on the grid line moved "
        f"the line {100 * fell:+.3f} %, which is {apart:.3f} of what a cell of the wall "
        f"is worth, against {abs(drawn['displacement']) / drawn['cell']:.3f} left by the "
        f"corrected run - bought by displacing a face {staircase.PINNED_CLEARANCE:g} of a cell"
    )
    assert apart > CAP_IS_WORTH, (
        f"handing the caps over on the face moved the answer {apart:.3f} cells, so this "
        "pair does not separate a wall that conducts from one that does not and nothing "
        "here measures what the clearance is for"
    )
