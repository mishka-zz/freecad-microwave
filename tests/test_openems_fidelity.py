# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What holding a curved conductor to a share of its radius actually buys.

``lfs.SURFACE_FIDELITY`` is a share of the radius a surface curves through, and
a share of a radius is not a cell size: it does not move when the cells do. So
what it settles is a **displacement** - how far the conductor openEMS builds
lands from the one that was drawn - and a tolerance a user might state has to be
inverted through that displacement to reach a cell.

Inverting it needs both of these, and only the first is a measurement:

* **How far the surface lands out, in cells.** openEMS decides an electric edge
  on one sampled point, so a conductor arrives inscribed on the grid; the
  adapter answers for that on a wall by growing it half a cell before handing it
  over, and what is left afterwards is this. It is the method's own number - if
  it is the *method's* at all, which is what a second shape is here to settle.
* **What that costs the answer**, which is arithmetic off the drawing rather
  than a measurement: a sphere's resonance goes as one over its radius, a
  coaxial line's impedance as the logarithm of a ratio of two.

The walls are a cylinder and a sphere - singly and doubly curved, so their
surfaces meet the lattice differently, and both with an exact capacitance to be
read against. Both are priced by the same rule and neither needs a solver:
:mod:`tests.staircase_model` in the plane, :mod:`tests.staircase_solid` in
space.

A flat disc is here for the other kind of boundary a drawing carries, and it is
the shape that shows where this instrument stops. A sheet has no wall - it is a
conductor with no inside, and where its *rim* landed is the whole of what it
stores - but a flat conductor keeps its charge exactly there, at a singularity
no grid resolves better than a share of a cell. So a plate whose every edge lies
on a grid line already reads as a conductor of another size, by more than a
staircased rim reads as displaced, and the disc says what a drawing and a
discretisation come to together. What is asserted off it is that, and not a
correction.

A capacitance throughout, so the walls' figures are one measurement repeated. A
conductor in openEMS is only *electrically* perfect, so a capacitance and a
resonance read walls a fraction of a cell apart, and the cavity gate's
displacement is the second of those rather than another sample of this one.

**Every figure here reads low, and by a known mechanism.** A displacement is
inferred from a capacitance by attributing all of it to the conductor being
measured, and the return path around it recedes too - which enters the inference
weighted by the ratio of the two radii, so pushing the return further out moves
the answer up and costs a grid going as that ratio squared, or cubed in space.
The return radii below are where that trade was left. It biases the size and not
the sign, and nothing asserted here is a size.
"""

from __future__ import annotations

import numpy as np
import pytest

from Microwave.Solvers.openems.lfs import SURFACE_FIDELITY
from Microwave.Solvers.openems.sizing import connection
from Microwave.Solvers.openems.staircase import GROWN_BY
from tests import staircase_solid
from tests.staircase_model import recession as plane_recession
from tests.staircase_model import uniform

#: The conductor whose surface is measured, in mm. Only its ratio to the cell
#: matters, so this is one and the resolutions below are read as cells across it.
RADIUS = 1.0

#: The return path, and how far the grid reaches past it, in mm. Far enough out
#: that its own recession is a minor share of what is inferred, and no further:
#: the plane solves a grid going as the square of this and space one going as
#: the cube, against a bias that falls only as the ratio of the radii. The one
#: in space is nearer because that bias falls as the square of the ratio there.
PLANE_OUTER, PLANE_SPAN = 10.0, 10.6
SPACE_OUTER, SPACE_SPAN = 4.0, 4.3

#: Cells across the radius, which is the unit a fidelity is stated in. What the
#: range has to be is set by the alternative being ruled out: a displacement
#: refinement could reach would shrink in the proportion the cell does, so the
#: ratio between the ends is the size of the effect this looks for.
STEPS = (6.0, 9.0, 12.0)

#: Where the lattice sits against the drawing, in cells. A square lattice and a
#: shape centred on it make most of the plane a copy of the rest, so these are
#: the corners of what is left - no offset, half a cell along an axis, half
#: along a diagonal - and one point between the extremes. The third component is
#: the cylinder's own axis and it ignores it.
#:
#: They are here because one alignment is not a measurement. Where a curved
#: boundary falls between the lines is a free variable of every grid, and a
#: figure read at one place in it is a figure with an unstated error bar.
PHASES = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, 0.5, 0.0), (0.25, 0.25, 0.25))

#: The same free variable for a sheet, which has it in its own plane alone: the
#: mesher pins a line at the elevation a sheet lies on, so the third axis has no
#: alignment to sweep.
#:
#: The corners of what the symmetry leaves, as above, and the midpoint of each
#: side between them. Where a rim crosses the lattice varies across the side and
#: not only at its ends, so corners alone do not average it - and a mean that
#: still moves with the resolution cannot be told from one that follows the
#: cell, which is the thing this is swept to settle.
RIM_PHASES = (
    (0.0, 0.0),
    (0.25, 0.0),
    (0.5, 0.0),
    (0.25, 0.25),
    (0.5, 0.25),
    (0.5, 0.5),
)

#: How far two shapes may disagree about where their surface landed, as a share
#: of the correction that put it there. Beyond this the half is a number about
#: one drawing rather than about the rule, and no single share of a radius can
#: stand for how well a curve is followed.
SHAPES_AGREE = 0.5

#: How much of a fall the displacement may show across the sweep, as a share of
#: the fall that would say refinement reaches it.
#:
#: The alternative is the thing to measure against rather than a flat bar: a
#: displacement that were a share of the *radius* instead of a share of the cell
#: would shrink, read in cells, in exactly the proportion the cell does. So what
#: the sweep's own ends predict under that reading is the size of the effect,
#: and this is how near to it still counts as holding steady. It degenerates
#: correctly - a sweep of one resolution repeated predicts no fall at all, and
#: then nothing can pass.
HOLDS_ITS_SHARE = 0.5


def _cells_per_radius(fidelity: float) -> float:
    """How many cells a fidelity spends across the radius it is stated against.

    A fidelity is that share of the *diameter* a surface curves through, and
    what a thickness becomes on each axis is the mesher's own arithmetic rather
    than restated here - so this follows a change to either.
    """
    return RADIUS / min(connection(fidelity * 2.0 * RADIUS))


def _landed(handed: float, recede: float, cell: float) -> float:
    """How far from the drawing the surface settled, in cells.

    The conductor is handed over larger than it was drawn and the sampling then
    takes ``recede`` off what it was handed, so where it ends up is the one
    minus the other - positive where the growth carried it past the drawing.
    Read from the radius that was handed over rather than from the growth that
    produced it, so a conductor handed over as something else is measured as
    something else.
    """
    return (handed - recede - RADIUS) / cell


def _cylinder(cell: float, phase, grown: float) -> float:
    handed = RADIUS + grown * cell
    grid = [uniform(cell, PLANE_SPAN, offset) for offset in phase[:2]]
    recede = plane_recession(*grid, handed, PLANE_OUTER - grown * cell)
    return _landed(handed, recede, cell)


def _sphere(cell: float, phase, grown: float) -> float:
    handed = RADIUS + grown * cell
    grid = [uniform(cell, SPACE_SPAN, offset) for offset in phase]
    recede = staircase_solid.sphere_recession(*grid, handed, SPACE_OUTER - grown * cell)
    return _landed(handed, recede, cell)


def _disc(cell: float, phase, grown: float) -> float:
    """Where a sheet's rim landed - which is the whole of what a disc reads.

    The third axis takes no offset. A sheet reaches openEMS as a polygon at one
    elevation, found only where a sample's own coordinate is that elevation, and
    the mesher answers for that by pinning a line there - so the lattice's
    freedom against this drawing is in the sheet's plane and nowhere else.
    """
    handed = RADIUS + grown * cell
    grid = [uniform(cell, SPACE_SPAN, offset) for offset in (*phase, 0.0)]
    recede = staircase_solid.disc_recession(*grid, handed, SPACE_OUTER - grown * cell)
    return _landed(handed, recede, cell)


#: Each shape: what reaches the sampling rule for it, the alignments swept, and
#: how much it was grown by first. Each is handed over the way the adapter hands
#: it over - the walls grown, the rim as drawn - so what is read here is the
#: shipped arrangement rather than a candidate for it.
SHAPES = {
    "cylinder": (_cylinder, PHASES, GROWN_BY),
    "sphere": (_sphere, PHASES, GROWN_BY),
    "rim": (_disc, RIM_PHASES, 0.0),
}

#: The shapes whose answer is a wall the field is kept out of, rather than the
#: extent of a conductor that has no inside.
WALLS = ("cylinder", "sphere")

#: The conductors the grid holds exactly, and the cubic shield around them, in
#: mm. Every face of every one of them lands on a grid line at every resolution
#: swept, so no part of the arrangement is staircased.
EXACT_HALF, EXACT_SHIELD, EXACT_SPAN = 1.0, 3.0, 3.2

#: Those conductors: a flat one and a solid one, so that what a rim costs the
#: instrument can be told from what a body does.
EXACT = {"plate": staircase_solid.plate, "block": staircase_solid.block}


@pytest.fixture(scope="module")
def measured():
    """Where each shape's surface landed, at every resolution and alignment.

    In cells, because that is the unit the sampling rule works in and the one a
    figure carries between shapes. What it is worth in an answer is the
    drawing's business and is not here.
    """
    found = {}
    for shape, (where, phases, grown) in SHAPES.items():
        for steps in STEPS:
            cell = RADIUS / steps
            found[shape, steps] = np.array([where(cell, phase, grown) for phase in phases])
    return found


def _reads(drawn, half: float, cell: float) -> tuple[float, float]:
    """That conductor's capacitance, and how it moves with its own size.

    The derivative comes off the same grid the capacitance did and carries the
    same shield, which is what makes the pair self-contained: no closed form
    enters, and nothing has to be assumed about the return path, because whatever
    the shield does to one reading it does to the other.

    Stepped by a whole cell so that the larger shape lands on grid lines too - a
    step of any other length would staircase the thing the step is measuring.
    """
    grid = [uniform(cell, EXACT_SPAN) for _ in range(3)]
    here = staircase_solid.capacitance(*grid, drawn(half, EXACT_SHIELD))
    wider = staircase_solid.capacitance(*grid, drawn(half + cell, EXACT_SHIELD))
    return here, (wider - here) / cell


@pytest.fixture(scope="module")
def zero_point():
    """What the instrument says about a conductor it should say nothing about.

    A capacitance read on a finite grid is not the continuum one, and that
    difference divided by how the capacitance moves with the conductor's size is
    a length - the size the answer implies, against the size that was drawn. On a
    shape the grid holds exactly there is no staircase for it to be, so it is
    the discretisation, and it is in the same units as everything above.

    The continuum value is the sequence's own limit, fitted rather than looked
    up. So this needs no closed form at all, which is the point: it is a
    statement about the instrument that owes nothing to the shapes the
    instrument is used on.
    """
    found = {}
    for shape, drawn in EXACT.items():
        cells = np.array([EXACT_HALF / steps for steps in STEPS])
        pairs = [_reads(drawn, EXACT_HALF, cell) for cell in cells]
        caps = np.array([capacity for capacity, _ in pairs])
        slopes = np.array([slope for _, slope in pairs])
        limit = float(np.polyfit(cells, caps, 1)[1])
        found[shape] = float(np.mean((caps - limit) / slopes / cells))
    return found


@pytest.fixture(scope="module")
def calibration(measured):
    """Each shape as one displacement, over every resolution and alignment."""
    found = {}
    for shape in SHAPES:
        over = np.concatenate([measured[shape, steps] for steps in STEPS])
        found[shape] = (float(over.mean()), float(np.ptp(over)))
    spent = _cells_per_radius(SURFACE_FIDELITY)

    def said(shape, at=""):
        lands, spread = found[shape]
        return (
            f"{lands:+.3f} cells out{at} ({100 * lands / spent:.2f} % of the radius, "
            f"spread {100 * spread / spent:.2f} %)"
        )

    print(
        f"\nGATE fidelity: {SURFACE_FIDELITY:g} of a radius is {spent:.2f} cells "
        "across it, where a grown surface lands "
        + ", ".join(said(shape, f" on a {shape}") for shape in WALLS)
        + f"\nGATE fidelity: a sheet's rim, handed over as drawn, reads {said('rim')}"
    )
    return found


# ---------------------------------------------------------------------------
# What the instrument has to be before a figure off it means anything
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_capacitor_the_grid_holds_exactly_comes_out_as_arithmetic(axis):
    """The three-dimensional solve, against a case with no staircase in it.

    Two slabs normal to an axis, their faces on grid lines, hold a uniform field
    between them and nothing else - so the capacitance is the area over the gap
    exactly, and any error is the assembly rather than the geometry. It is the
    one arrangement here whose answer owes nothing to the sampling rule, which
    is what makes it the check the rest stands on: the conductance weights, the
    energy sum, and which nodes were found to be a conductor at all.

    Along each axis in turn, because the area an edge carries is read off the
    *other two* - so a weight fetched from the wrong axis answers correctly in
    one orientation and not in the others. Graded across the gap for the same
    reason at one remove: a weight read from a single spacing rather than from
    each line's own neighbours also answers correctly until the spacings differ.

    One gap and not two. The driven conductor is the one holding the origin and
    ground is the one holding the grid's far corner, so the slab at the other
    end is tied to neither - and a conductor at no fixed potential, with nothing
    beyond it to pull charge from, sits at the potential of what surrounds it
    and stores nothing.
    """
    driven, grounded = 0.5, 1.5
    graded = np.unique(
        np.concatenate(
            [
                np.linspace(-2.0, -driven, 13),
                np.linspace(-driven, driven, 9),
                np.linspace(driven, 2.0, 13),
            ]
        )
    )
    lines = [np.linspace(-1.0, 1.0, 21), np.linspace(-1.5, 1.5, 16), np.linspace(-1.2, 1.2, 19)]
    area = float(np.prod([line[-1] - line[0] for index, line in enumerate(lines) if index != axis]))
    lines[axis] = graded

    def slabs(*point):
        return (np.abs(point[axis]) <= driven) | (np.abs(point[axis]) >= grounded)

    measured = staircase_solid.capacitance(*lines, slabs)
    assert measured == pytest.approx(area / (grounded - driven), rel=1e-12, abs=0.0)


# ---------------------------------------------------------------------------
# The calibration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", WALLS)
def test_the_growth_removes_more_than_it_leaves_on_either_shape(calibration, shape):
    """That the shipped half is the right *size*, said of two shapes.

    Growing by half a cell removes the inscription and puts a little of it back
    the other way, the half being the mean of what a boundary meeting the
    lattice at every phase alike gives up, and a curved surface giving up less
    than that. What says the half is nearer right than wrong is that what it
    leaves is smaller than what it took away - and a correction fitted to one
    drawing would say that of one shape and not of the other.
    """
    lands, _ = calibration[shape]
    assert lands > 0.0, (
        f"a grown {shape} lands {lands:.3f} cells out, so the growth did not reach "
        "the drawing and the surface is still inscribed on the grid"
    )
    assert lands < GROWN_BY - lands, (
        f"a grown {shape} lands {lands:.3f} cells out where the growth removed "
        f"{GROWN_BY - lands:.3f}, so the correction leaves more than it takes"
    )


def test_the_two_shapes_agree_about_where_their_surface_landed(calibration):
    """The claim a fidelity is a setting at all rests on.

    A share of a radius is one number offered to every curve a user draws. It
    can only mean anything if what it buys is a property of the sampling rule
    rather than of the shape being sampled - so a cylinder and a sphere, which
    curve in one direction and in two, have to land in about the same place.

    About, and no better: this is what bounds any tolerance quoted off it.
    """
    cylinder, sphere = calibration["cylinder"][0], calibration["sphere"][0]
    apart = abs(sphere - cylinder)
    assert apart < SHAPES_AGREE * GROWN_BY, (
        f"a cylinder lands {cylinder:.3f} cells out and a sphere {sphere:.3f}, "
        f"{apart:.3f} apart against a correction of {GROWN_BY:g} - so where a curved "
        "surface ends up is about the shape rather than about the rule"
    )


def test_a_conductor_the_grid_holds_exactly_reads_as_displaced_anyway(zero_point):
    """The instrument's own reading, on shapes with no staircase in them at all.

    A flat conductor keeps its charge at its rim, where the density is singular,
    and a grid resolves that to a share of a cell however the rim is drawn. So a
    plate whose every edge lands on a grid line - nothing sampled, nothing
    rounded, nothing to correct - still reads as a conductor of a different size,
    and the size it reads is the finding rather than an error bar on one.

    Printed in the same units as everything above, because that is the only way
    to see that it is not small compared to them.
    """
    print(
        "\nGATE fidelity: a shape the grid holds exactly reads "
        + ", ".join(
            f"{lands:+.3f} cells out as a {shape}" for shape, lands in sorted(zero_point.items())
        )
    )
    assert zero_point["plate"] > 0.0, (
        f"a plate the grid holds exactly reads {zero_point['plate']:+.3f} cells out, so "
        "the discretisation this is here to expose has stopped being there"
    )


def test_and_a_flat_conductor_reads_further_out_than_a_solid_one(zero_point):
    """So it is not one number the whole instrument could be corrected by.

    A cube's charge sits on its faces and only its edges are singular; a plate is
    all rim. If the two read alike the reading would be a property of the grid
    and could be taken off every shape at once - they do not, and it cannot.
    """
    plate, block = zero_point["plate"], zero_point["block"]
    assert plate > block, (
        f"a plate reads {plate:+.3f} cells out and a cube {block:+.3f}, so what a "
        "discretised conductor gains does not depend on it having a rim"
    )


def test_which_is_more_than_a_rim_says_it_gave_up(calibration, zero_point):
    """Why this cannot answer whether a sheet's outline wants correcting.

    The two numbers are the same size and they are not separable: a rim that
    receded and a flat conductor that reads large are both a share of the cell,
    both live at the outline, and refining reduces neither in the units they are
    measured in. So a disc says what a *drawing plus a discretisation* comes to
    and not where the metal went, and whether an outline wants correcting stays
    open until something that solves Maxwell is asked it.

    Asserted rather than noted, so that an instrument which stopped confounding
    them - or a rim whose displacement grew past its own zero - says so here.
    """
    rim, zero = calibration["rim"][0], zero_point["plate"]
    assert abs(rim) < zero, (
        f"a rim reads {rim:+.3f} cells from its drawing where a plate the grid holds "
        f"exactly reads {zero:+.3f}, so the displacement is now the larger of the two "
        "and can be read off this after all"
    )


@pytest.mark.parametrize("shape", [*WALLS, "rim"])
def test_the_displacement_belongs_to_the_cell_and_not_to_the_resolution(measured, shape):
    """Why a fidelity cannot be traded away by refining the rest of the mesh.

    A displacement that is a share of the cell stays where it is, read in cells,
    as the cell shrinks - and then a fidelity stated as a share of a radius
    leaves a fixed share of that radius however fine the grid around it gets. A
    displacement refinement could reach would instead be a share of the
    *radius*, and read in cells it would shrink in exactly the proportion the
    cell does. That is the alternative, it is what the bar is set against, and
    the sweep's own ends are what predict its size.

    Scored on the mean over alignments and not on any one of them. Where a
    circle crosses the lattice lines is a different pattern at every radius
    measured in cells, so one alignment's name does not follow it from one
    resolution to the next and only the average over them is the same quantity
    twice.

    On the magnitude, so that a shape landing inside the drawing is scored the
    same way as one landing outside it. Which side it lands is asserted where
    each shape's own displacement is.

    It is also what would catch a closed form that is wrong by a fixed share of
    the size: that is exactly a displacement following the radius, so it fails
    here rather than passing quietly everywhere.
    """
    over = [abs(float(measured[shape, steps].mean())) for steps in STEPS]
    followed = min(STEPS) / max(STEPS)
    held = min(over) / max(over)
    assert held > 1.0 - HOLDS_ITS_SHARE * (1.0 - followed), (
        f"across the sweep a {shape}'s surface landed between {min(over):.3f} and "
        f"{max(over):.3f} cells out, a ratio of {held:.3f} where a displacement "
        f"following the cell would give {followed:.3f} - so refining the mesh reaches "
        "some of it and a share of a radius is slack rather than a floor"
    )
