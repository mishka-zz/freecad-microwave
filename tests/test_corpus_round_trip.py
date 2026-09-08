# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A shape and its round trip through a file are one shape, and get one answer.

A STEP file carries surfaces and the curves that trim them. It carries no
parametric history, no proxy, no FreeCAD type - a cone written out and read back
is not a ``Part::Cone`` any more, and neither is a cone that arrived from
somebody else's CAD system in the first place. So the geometry layer is required
to say the same thing about a shape and about its round trip, and anything that
branched on what *drew* a shape shows up here as a difference. Nothing has to be
enumerated for that to work, which is what makes it worth more than a test per
type: it catches the special case that has not been written yet, in whatever form
it takes.

What is compared is everything the layer produces.

**The verdict and the pieces**, which is the classification: meshed or refused,
how many primitives it became, whether each is an area or a volume, whether it
is held exactly or as triangles, and where each sits.

**The demand set** - every length the drawing carries, projected onto the three
axes. Not the demands themselves: STEP re-parametrises a surface, so a curvature
is sampled at a different number of points and a demand list that is one entry
longer is not a difference in what the grid is asked for. What a demand set means
is the sizing field it induces, ``min over demands of size + g|x - y|``, and what
that field means is the cells it asks for. So the two are compared in cells:
``mesh.py::_segment_lines`` takes the count across a gap from the integral of
``1/size`` over it, so the integral of the difference of two reciprocals is how
many cells one set asks for that the other does not. Below one, neither set asks
for as much as a cell more than the other anywhere - which is a statement about
the request and not about the grid, since the mesher rounds that count and a gap
sitting astride a whole number tips across it under any difference at all. It is
also a statement at one grading slope, the one ``GRADING`` names, and at that
slope only.

The lint in ``test_measured_not_typed.py`` is the cheap half of the same rule,
and it catches the forms this cannot: a branch on a type that no corpus specimen
happens to reach.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from Microwave.Solvers.openems.lfs import CHORD_TOLERANCE, SEPARATION_REACH, SURFACE_FIDELITY
from Microwave.Solvers.openems.sizing import DIMENSIONS
from tests import corpus
from tests.conftest import corpus_record as _record

#: The sizing field's slope, as ``ln(max cell ratio)``, and the ratio a mesh
#: policy ships with. So the comparison below is made at the slope a drawing is
#: meshed at, unless somebody sets that property to something else. The mesher
#: ramps a fraction shallower than this, ``mesh.py`` keeping headroom for the
#: scaling that lands a gap's cells on a whole count.
#:
#: **The comparison is made at this slope and says nothing at another one.** A
#: shallower ramp carries a demand's influence further before the field reaches
#: the cap, so a demand that moved is felt over more of the axis. That is not a
#: direction, and the figure is not monotone in the slope. A demand that another
#: one covers at a shallow ramp, and that the cap has swallowed at a steep one,
#: binds between the two and nowhere else - so a difference between two sets
#: there is worth nothing at either end and cells in the middle.
#:
#: Sweeping the slope does not rescue the statement. Over this corpus the figure
#: climbs as the ramp shallows, and two specimens cross the bound at ratios a
#: user may select - not because the layer read them differently, but because
#: the file handed back a re-approximated surface and a curvature read on one
#: asks for a different cell. ``MaxGrowthRatio`` takes any ratio above one, so
#: there is no shallowest one to make the comparison at instead.
GRADING = math.log(1.3)

#: How far past the shape the two fields are compared, as a share of its extent.
#: Beyond the shape as well as across it, because a demand is a point constraint
#: that ramps outward and a field that agrees only where the metal is would say
#: nothing about the grid around it.
#:
#: It is a window rather than the mesher's whole axis, so a demand's ramp is
#: followed until it is well into the coarse cells and not to where it meets the
#: cap. What that leaves out is difference, so the comparison is the weaker for
#: it rather than the stronger.
BEYOND = 0.25

#: Integration samples per cell inside one smooth piece of the field, and the
#: most any one piece gets. ``1/field`` carries a cusp a few cells wide at each
#: demand, so a sample every eighth of a cell resolves it however long the axis
#: is, where a fixed count over the axis steps across the cusp of a fine demand
#: on a large board. The cap bounds what one piece can ask for.
#:
#: Sampled rather than integrated in closed form, which is what the mesher
#: does. This compares two demand sets and must not be able to agree with the
#: mesher by sharing its arithmetic.
PER_CELL = 8
MOST_SAMPLES = 8192

#: How far apart two demand sets may be, in cells.
#:
#: A demand is a length arrived at through a chain of kernel queries, so
#: re-parametrising the surface it was taken from moves its last bits and moves
#: the point it was taken at. Two sets that differ that way are not asking for
#: different grids, and the way to say so is to measure the difference in the
#: thing a demand set produces: ``mesh.py::_segment_lines`` takes the cell count
#: across a gap from the integral of ``1/size`` over it, so the integral of
#: ``|1/mine - 1/theirs|`` is how many cells one set asks for and the other does
#: not.
#:
#: **It bounds what is asked for, not what is laid.** ``mesh.py::_cell_count``
#: rounds the count it is handed, so a gap whose count sits astride a whole
#: number tips across it under a difference of any size - which
#: ``mesh.py::_settle`` states in its own words about the same rounding, and
#: which ``tests/test_mesh_budget.py`` states about the same integral. The claim
#: here is the one that survives that: neither set asks for as much as one cell
#: more than the other, anywhere on the axis. Where the lines then land has its
#: own tests.
#:
#: **It is what this corpus does at one slope**, and it is not a property of
#: drawings. The figure is not smooth in the drawing any more than it is in the
#: slope: the finest cell a shape asks for is often a chord with no floor under
#: it, read at a station near a shallow trim, and one station more or less there
#: reads a length orders finer. A bore's radius nudged by a fraction of itself
#: moves the figure by orders, in either direction. So this bound is a
#: measurement of this corpus rather than one anybody derived, and a specimen
#: added tomorrow may not meet it.
#:
#: The figure measured is on the failure message.
CELLS = 1.0

#: How far the kernel's own measure of a shape may move across the round trip,
#: relative. A STEP writer re-approximates a trimmed surface rather than copying
#: it, so a volume or an area taken on the far side is a measure of a slightly
#: different surface.
#:
#: It is also what makes a measure *nothing*: below this share of what the
#: shape's own bounding box would hold, there is no measurement there to compare
#: and what the kernel answers is rounding either way. Against the box rather
#: than against a power of the largest extent, so that a plate is judged by the
#: volume a plate has.
SAME_SHAPE = 1e-3

#: How many of the cells a shape asks for have to span it before the grid is
#: following it rather than stepping over it. Asked only of the shapes that
#: arrive as triangles: one the grid holds exactly pins its own faces and needs
#: no cells across itself at all to be in the right place.
#:
#: Halfway between the two ends, in the ratio rather than the difference. A body
#: resolved by nothing but its own cross-section spans ``sqrt(DIMENSIONS)`` of
#: the cells it asks for, that being what the connection form spends a thickness
#: as; one held to :data:`SURFACE_FIDELITY` of its radius spans that over the
#: fraction. So this is the second with a factor of two in hand, and it is far
#: above the first.
FOLLOWED = math.sqrt(DIMENSIONS) / SURFACE_FIDELITY / 2

#: How far a piece's corner may move, relative to the shape's size. The same
#: re-approximation read as a length: a triangulation is laid on the surface
#: that came back, and the extent of those vertices is what the mesher is told.
PLACEMENT = 1e-3

#: How far a demand read off a witness pair may sit from the length that was
#: drawn, relative. A separation reaches the grid as the distance itself where
#: the pair points along one axis, so what stands between the drawing and the
#: demand is the kernel's own arithmetic in finding the pair and nothing else -
#: unlike a thickness, which is placed by a bisection that stops at a tolerance
#: of its own.
WITNESS = 1e-9


def envelope(demands, points: np.ndarray) -> np.ndarray:
    """The finest cell each point is held to by ``demands``, on one axis.

    A demand asking for cells of ``size`` over an interval holds down cells of
    ``size + g * d`` at distance ``d`` beyond it, and the field is the minimum
    over all of them, under the coarsest cell the grid will use.

    Written from the definition rather than taken from the mesher, so that two
    demand sets are compared through arithmetic that is not the thing under
    test. What the mesher builds is this and two things more: the other
    constraints a problem carries - a material's bulk size, a refinement region,
    the thirds rule - and a floor under the whole of it. Both are monotone in
    this, so two demand sets that agree here agree there; a difference this
    cannot see is one a floor or a finer neighbour would have hidden anyway.
    """
    finest = np.full(len(points), float(corpus.CELL_CAP))
    for lower, upper, size in demands:
        away = np.maximum(np.maximum(lower - points, points - upper), 0.0)
        finest = np.minimum(finest, size + GRADING * away)
    return finest


def _apart(mine, theirs, low: float, high: float) -> float:
    """How many cells one demand set asks for over this window that the other
    does not.

    The integrand is non-negative, so this bounds the difference over every gap
    inside the window at once as well as over the window itself.

    The field is a lower envelope of ramps, so ``1/field`` bends at each
    demand's own ends; the window is split there and each piece sampled against
    its own finest cell, which keeps a fine demand on a large board from being
    stepped over. A count fixed over the axis instead would step over exactly
    that.

    A piece's finest cell is read off its two ends and not searched for inside:
    every ramp is flat across its own demand and linear away from it, and the
    window is split at every one of those ends, so on a piece each ramp is
    linear and their minimum is concave - and a concave function over an
    interval is smallest at an end.

    Both fields are then evaluated once over the union of the pieces rather than
    piece by piece. A drawing carrying a thousand demands makes as many pieces,
    and a call per piece spends its whole time in the loop over demands.
    """
    edges = np.array(
        sorted(
            {low, high}
            | {
                float(edge)
                for demands in (mine, theirs)
                for lower, upper, _ in demands
                for edge in (lower, upper)
                if low < edge < high
            }
        )
    )
    at_edges = np.minimum(envelope(mine, edges), envelope(theirs, edges))
    finest = np.minimum(at_edges[:-1], at_edges[1:])
    counts = np.clip(PER_CELL * np.diff(edges) / finest, PER_CELL, MOST_SAMPLES).astype(int)
    points = np.unique(
        np.concatenate(
            [
                np.linspace(start, stop, count + 1)
                for start, stop, count in zip(edges[:-1], edges[1:], counts)
            ]
        )
    )
    gap = np.abs(1.0 / envelope(mine, points) - 1.0 / envelope(theirs, points))
    # Written out rather than taken from numpy, which named this function
    # differently in the version FreeCAD ships.
    return float(np.sum(np.diff(points) * (gap[:-1] + gap[1:]) / 2.0))


def _off_by(one: float, other: float) -> float:
    """How far two measurements are apart, relative to the larger."""
    scale = max(abs(one), abs(other))
    return 0.0 if scale == 0.0 else abs(one - other) / scale


def _extents(measured) -> tuple[float, float, float]:
    box = measured["bound_box"]
    return tuple(box[dim + 3] - box[dim] for dim in range(3))


def _size_of(measured) -> float:
    """The shape's largest extent, which the placement bound is stated against."""
    return max(_extents(measured))


def _boxful(measured) -> dict[str, float]:
    """What the shape's own bounding box holds, per measure.

    The yardstick a measure is called nothing against, and it has to be the
    shape's rather than a power of one of its extents: a plate the size of a
    board has a volume the size of a plate, and judging it by the board's would
    call a wall a thousand times too thick no measurement at all.
    """
    across, along, up = _extents(measured)
    return {
        "volume": across * along * up,
        "area": 2.0 * (across * along + along * up + up * across),
    }


def _moved(record) -> str:
    """What the kernel says the round trip changed about the shape, or nothing.

    The invariant is that one shape gets one answer, so it applies only where the
    file gave back the shape that went into it. That is not this workbench's
    judgement to make and is not made here: it is the kernel's own volume, area
    and solid count, before the geometry layer sees any of it.

    Both measures are compared **signed**, so a solid handed back wound the other
    way is a different shape rather than the same one - which is the whole of the
    difference between a box and everything outside it.
    """
    one, other = record["measured"], record["round_trip"]["measured"]
    holds = _boxful(one)
    faults = []
    for key in ("volume", "area"):
        if max(abs(one[key]), abs(other[key])) <= SAME_SHAPE * holds[key]:
            continue
        if _off_by(one[key], other[key]) > SAME_SHAPE:
            faults.append(f"{key} {one[key]:g} -> {other[key]:g}")
    if one["solids"] != other["solids"]:
        faults.append(f"solids {one['solids']} -> {other['solids']}")
    return "; ".join(faults)


def _both(artifacts, name):
    """One specimen's record and its round trip's, or a skip saying why not."""
    record = _record(artifacts, name)
    back = record["round_trip"]
    if back["status"] == "unavailable":
        pytest.skip(f"STEP would not carry {name}: {back['reason']}")
    changed = _moved(record)
    if changed:
        pytest.skip(f"the file gave back a different shape, so nothing to compare: {changed}")
    return record, back


#: The two records a criterion is asked of. Both, because a criterion the
#: drawing meets and its round trip does not is exactly what this file exists to
#: find, and comparing the two sides with each other cannot see it - they agree
#: while both are wrong. Nothing is solved twice for it: the probe already wrote
#: both sides, so this is the same measurement read a second time.
DRAWN = "as drawn"
FROM_A_FILE = "through a file"
SIDES = (DRAWN, FROM_A_FILE)

#: The same two as test ids, without the spaces that stop ``-k`` selecting them.
SIDE_IDS = ("drawn", "from-a-file")


def _side(artifacts, name, side):
    """One specimen's record, drawn or read back, or a skip saying why not."""
    record = _record(artifacts, name)
    if side == DRAWN:
        return record
    back = record["round_trip"]
    if back["status"] == "unavailable":
        pytest.skip(f"STEP would not carry {name}: {back['reason']}")
    changed = _moved(record)
    if changed:
        pytest.skip(f"the file gave back a different shape, so it is not this shape: {changed}")
    return back


def _ordered(pieces):
    """Pieces by where they are, since a file may hand the lumps back in any order."""
    return sorted(pieces, key=lambda piece: tuple(piece["lower"]))


def _retriangulated(record, back) -> str:
    """What the file changed about the triangles the mesher measures, said aloud.

    A chord is read off the triangulation and off nothing else, so a body the
    file tessellates differently is a body a different length was measured on.
    That is not a fault and it is not this workbench's doing: openEMS is handed
    those triangles too, so a shape through a file has always been solved as a
    slightly different shape. It is the coarsest of the ways a file moves a
    demand, and the one a reader cannot see from the demands, so a comparison
    that fails says whether it happened.
    """
    drawn, restored = _ordered(record["pieces"]), _ordered(back["pieces"])
    if len(drawn) != len(restored):
        return f", and the file gave back {len(restored)} pieces where the drawing has {len(drawn)}"
    changed = [
        f"{len(mine['faces'])} -> {len(theirs['faces'])}"
        for mine, theirs in zip(drawn, restored)
        if len(mine["faces"]) != len(theirs["faces"])
    ]
    if not changed:
        return ", off a triangulation the file left the same size"
    return ", off a triangulation the file changed: " + "; ".join(changed) + " triangles"


def _laid(record):
    """One record's face station counts, gathered by the body each was laid on,
    and a length each count was taken from.

    A specimen drawn beside a companion is measured with it, so an entry says
    which body it belongs to. Within a body the counts are a multiset: a file
    may hand faces back in any order, and none of what is compared is a face's
    place in a list.

    The counts are compared and the lengths are not. A length is a measurement
    and its last bits move across a file on almost every face, which is the
    whole reason the counts are worth asserting; the length is carried so that
    a failure can say which face sat on a step.
    """
    counts: dict[str, Counter] = {}
    sizes: dict[tuple[str, int, int], tuple[float, float]] = {}
    for label, across, along, width, height in record["lattice"]:
        counts.setdefault(label, Counter())[(across, along)] += 1
        sizes.setdefault((label, across, along), (width, height))
    return counts, sizes


def _lattice_moved(record, back) -> str:
    """What the file laid a different lattice on, said in full, or nothing.

    Where every length read off a face is read is decided by the lattice on it,
    so two sides that laid the same lattices measured one shape at one set of
    places. Two sides that did not measured two sets, and their demands then
    differ by where they were taken rather than by what the drawing asks for.

    A length each count was taken from is on the message, because a count is an
    integer read off one: a face that is a whole number of cells across sits on
    the step ``lfs.py::_steps`` takes, and the counts alone do not show that it
    did.
    """
    mine, my_sizes = _laid(record)
    theirs, their_sizes = _laid(back)
    for label in sorted(set(mine) | set(theirs)):
        one, other = mine.get(label, Counter()), theirs.get(label, Counter())
        gone, came = one - other, other - one
        if not gone and not came:
            continue
        return (
            f"on {label!r} the drawing lays {_faces_said(label, gone, my_sizes)} "
            f"and the file lays {_faces_said(label, came, their_sizes)}"
        )
    return ""


def _faces_said(label, counts, sizes) -> str:
    """A handful of face lattices, as a reader would say them."""
    if not counts:
        return "nothing of its own"
    return "; ".join(
        f"{across} by {along} stations on a face measuring "
        f"{sizes[(label, across, along)][0]:.12g} by "
        f"{sizes[(label, across, along)][1]:.12g} mm"
        for across, along in sorted(counts.elements())
    )


def _carried(artifacts):
    """Every artifact that got as far as being written to a file.

    Read off what the probe wrote rather than filtered against the corpus, so
    that a specimen missing from the run cannot be quietly left out of the two
    checks below - it is absent from both sides, and the corpus gate is where a
    run that shrank is caught.
    """
    return {
        name: record
        for name, record in artifacts.items()
        if record["status"] not in ("unavailable", "undrawable")
    }


def _finest(record) -> float:
    """The smallest cell anything on the drawing asked for.

    A demand does not say what measured it, so reading the finest is sound only
    where nothing else on the specimen could have supplied it - which is a
    property of the specimen and is stated at each place this is used.
    """
    return min(size for axis in record["demands"] for _, _, size in axis)


def _across(record):
    """The extent both shapes are compared over, per axis, padded."""
    boxes = [record["measured"]["bound_box"], record["round_trip"]["measured"]["bound_box"]]
    for dim in range(3):
        low = min(box[dim] for box in boxes)
        high = max(box[dim + 3] for box in boxes)
        pad = BEYOND * max(high - low, 1.0)
        yield low - pad, high + pad


@pytest.mark.slow
@pytest.mark.parametrize("specimen", corpus.specimens(), ids=lambda s: s.name)
class TestOneShapeGetsOneAnswer:
    """Each specimen against itself, written to STEP and read back.

    The checks here are of two kinds. Some compare the two sides with each
    other, which is what says the layer read the geometry and not what drew it.
    The rest ask a criterion of each side on its own, and ask it of both: two
    sides that agree with each other can be wrong together, and a criterion is
    the only thing here that can say so.
    """

    def test_the_verdict_survives_the_round_trip(self, specimen, artifacts):
        """Meshed stays meshed and refused stays refused.

        A refusal that becomes a mesh is the worse direction: the user drew
        something this workbench cannot solve, saved it, opened it, and was told
        nothing.
        """
        record, back = _both(artifacts, specimen.name)
        assert back["status"] == record["status"], (
            f"{specimen.name!r} ({specimen.subject}) is {record['status']} as drawn "
            f"and {back['status']} through a file, which is the same geometry: "
            f"{back.get('reason', record.get('reason', ''))}"
        )
        # Named, because two of anything agree with each other. A layer that
        # raised on every shape in the corpus would satisfy the line above and
        # leave every comparison below it skipping.
        assert record["status"] in ("meshed", "refused"), (
            f"{specimen.name!r} got neither verdict: {record.get('reason', '')}"
        )

    def test_the_pieces_survive_the_round_trip(self, specimen, artifacts):
        """As many primitives, each the same kind of thing.

        ``sheet_normal`` says an area rather than a volume, and whether a piece
        carries triangles says whether the grid holds it exactly. Each is a
        decision the layer makes about every shape, and each is a measurement
        rather than a type.

        ``thickened`` is the one neither of those can see: a surface offset into
        metal and a solid drawn with that metal both arrive as triangles
        bounding a volume. A file that closes an open surface would pass the
        other comparisons while changing what the run has to declare.
        """
        record, back = _both(artifacts, specimen.name)
        if record["status"] != "meshed":
            pytest.skip("nothing was meshed, so there are no pieces to compare")

        drawn, restored = _ordered(record["pieces"]), _ordered(back["pieces"])
        assert len(restored) == len(drawn), (
            f"{specimen.name!r} became {len(drawn)} primitives as drawn and "
            f"{len(restored)} through a file"
        )
        for mine, theirs in zip(drawn, restored):
            assert theirs["sheet_normal"] == mine["sheet_normal"], (
                f"{specimen.name!r} is an area on axis {mine['sheet_normal']} as "
                f"drawn and {theirs['sheet_normal']} through a file"
            )
            assert bool(theirs["faces"]) == bool(mine["faces"]), (
                f"{specimen.name!r} is held "
                f"{'as triangles' if mine['faces'] else 'exactly'} as drawn and "
                f"{'as triangles' if theirs['faces'] else 'exactly'} through a file"
            )
            assert theirs["thickened"] == mine["thickened"], (
                f"{specimen.name!r} was given {mine['thickened']:g} mm of thickness "
                f"as drawn and {theirs['thickened']:g} mm through a file"
            )

    def test_the_pieces_are_in_the_same_place(self, specimen, artifacts):
        """A round trip that moved a conductor is a round trip that changed the
        device, however alike the two verdicts read."""
        record, back = _both(artifacts, specimen.name)
        if record["status"] != "meshed":
            pytest.skip("nothing was meshed, so there is nothing placed")

        allowed = PLACEMENT * _size_of(record["measured"])
        for mine, theirs in zip(_ordered(record["pieces"]), _ordered(back["pieces"])):
            for corner in ("lower", "upper"):
                for dim in range(3):
                    assert abs(theirs[corner][dim] - mine[corner][dim]) <= allowed, (
                        f"{specimen.name!r} has its {corner} corner at "
                        f"{mine[corner][dim]:.9g} on axis {dim} as drawn and at "
                        f"{theirs[corner][dim]:.9g} through a file"
                    )

    def test_the_demands_survive_the_round_trip(self, specimen, artifacts):
        """The same grid is asked for, which is what a demand set is for.

        Compared as the cells the demands ask for and not as the demands
        themselves: re-parametrising a surface moves where it is sampled, so the
        two lists differ in length on the cone and on the text without either
        asking for anything the other does not.

        Nor as the field, which is the same argument one step short. A demand is
        the end of a chain of kernel queries and its last bits move whether or
        not anything about the drawing did, and the chain amplifies that: a
        chord is where a ray met a facet, so a vertex that moved carries into
        the answer divided by the cosine of the angle the ray made with it, and
        that angle is free to be shallow. Nothing in it asks for a different
        grid, and a comparison of fields cannot say so except under a tolerance
        chosen to let it through.

        Counting cells can. What the mesher does with a field is integrate
        ``1/size`` across each gap and take the count from it, so the same
        integral of the difference of the two reciprocals is the count one set
        asks for and the other does not. A layer that branched on how a shape
        was drawn asks for a different count, because a demand it stopped making
        leaves the field at the material's bulk size wherever that demand was
        the finest thing.

        It holds over every specimen, with no kind of shape left out. What made
        that possible is that both sides read the shape in the same places:
        ``TestTheRoundTripItself`` asserts that the file lays the same lattice on
        every face, and where a lattice moves the lengths are measured somewhere
        else and this comparison is between two samplings rather than about one
        drawing.
        """
        record, back = _both(artifacts, specimen.name)
        if record["status"] != "meshed":
            pytest.skip("nothing was meshed, so nothing was measured off it")
        if not any(record["demands"]) and not any(back["demands"]):
            pytest.skip("this shape is held exactly, so no length was measured off it")

        worst, worst_axis = 0.0, 0
        for dim, (low, high) in enumerate(_across(record)):
            apart = _apart(record["demands"][dim], back["demands"][dim], low, high)
            if apart > worst:
                worst, worst_axis = apart, dim

        assert worst < CELLS, (
            f"{specimen.name!r} ({specimen.subject}) asks for a different grid "
            f"through a file: the two demand sets are {worst:.3g} cells apart "
            f"on axis {worst_axis}{_retriangulated(record, back)}"
        )

    @pytest.mark.parametrize("side", SIDES, ids=SIDE_IDS)
    def test_a_shape_the_grid_does_not_hold_is_asked_for_finely_enough(
        self, specimen, artifacts, side
    ):
        """What makes the comparison above more than two empty sets agreeing.

        A piece that came back as triangles is one no rectilinear grid follows,
        so the lengths it carries have to be measured off it - and a demand set
        that is empty for one is a shape that reached the mesher as nothing but
        its material's bulk cell size. A shape the grid *does* hold is the other
        case, and it is not this one: it pins its own faces, nothing is measured
        off it, and the comparison above says so and skips.

        That a demand exists is not that it resolves anything. A conductor's own
        cross-section is a length it carries, and asking for a cell that fits
        inside it is satisfied by a couple of cells across a ball of metal -
        which conducts, and is not the shape that was drawn. So what is asserted
        is how many of the cells it asks for span it.

        Which demand supplies that is not fixed, and does not need to be: on a
        specimen with edges it is usually one of those, and on a smooth closed
        one there is nothing else it can be. The smooth ones are what this
        gates, and they are also the ones a rule about curvature can break
        without any other specimen noticing.
        """
        record = _side(artifacts, specimen.name, side)
        if record["status"] != "meshed":
            pytest.skip("nothing was meshed, so nothing was measured off it")
        if not any(piece["faces"] for piece in record["pieces"]):
            pytest.skip("this shape is held exactly, so its own faces are where the grid is")
        assert any(record["demands"]), (
            f"{specimen.name!r} ({specimen.subject}) {side} is meshed as triangles "
            "and asks the grid for nothing, so no length was read off it at all"
        )
        finest = _finest(record)
        across = _size_of(record["measured"]) / finest
        assert across >= FOLLOWED, (
            f"{specimen.name!r} ({specimen.subject}) {side} is meshed as triangles and "
            f"asks for cells of {finest:.4g}, which is {across:.3g} of them across the "
            f"whole shape - the grid steps over it rather than following it"
        )

    @pytest.mark.parametrize("side", SIDES, ids=SIDE_IDS)
    def test_a_body_is_asked_for_across_its_own_wall(self, specimen, artifacts, side):
        """Following a shape is not resolving the metal it is made of.

        A conductor is sampled at a point per Yee edge, so a wall the grid puts
        no cell inside is a wall the engine can open - and the demand that says
        so has to come from the wall itself. Curvature cannot supply it: on a
        shell the radius the surface carries is the shell's and not the wall's,
        and on a bar with planar faces there is no curvature at all.

        Stated as the criterion rather than as a figure: a cell whose body
        diagonal fits inside the wall, which is what the connection form spends
        a thickness as. The chord the wall is read from carries its own
        precision, so the comparison allows it.

        What is read is the finest demand the whole record carries, and a demand
        does not say what measured it. That is sound only where nothing else on
        the shape could have supplied it - which is why the specimen declaring a
        wall has no edges anywhere, and why a specimen that has both would need
        the source recorded before it could be gated here.
        """
        if specimen.wall is None:
            pytest.skip("this shape has no single thickness to state")
        if specimen.dielectric:
            pytest.skip("a dielectric is counted across its wall, not fitted inside it")
        record = _side(artifacts, specimen.name, side)
        if record["status"] != "meshed":
            pytest.skip("nothing was meshed, so nothing was measured off it")
        finest = _finest(record)
        fits = specimen.wall / math.sqrt(3)
        assert finest <= fits * (1.0 + CHORD_TOLERANCE), (
            f"{specimen.name!r} ({specimen.subject}) {side} has a wall of "
            f"{specimen.wall:.4g} and asks for cells of {finest:.4g}, where a "
            f"cell fitting inside that wall is {fits:.4g} - so nothing measured "
            "the wall and the grid is free to open it"
        )

    @pytest.mark.parametrize("side", SIDES, ids=SIDE_IDS)
    def test_a_gap_inside_one_object_is_asked_for_across(self, specimen, artifacts, side):
        """Two lumps drawn in one operation are still two conductors.

        A gap the grid puts no cell inside is a gap the mesh closes, and two
        conductors shorted together is a run that completes and answers about a
        device nobody drew. The measurement has to come from the lumps
        themselves: they are one object, so nothing outside is paired with them.

        A gap has a direction, and what a cell must fit inside is the gap
        along it - where a cross-section spends its length across all three
        axes. The witness pair between two lumps side by side points along one
        axis and asks for the gap exactly; the walk along the run reads the
        same gap through each sample's own normal, and an oblique reading
        spends a longer crossing across more axes.

        Bounded on both sides rather than asserted equal, because those
        readings straddle the gap: the finest of them cannot go below the
        allocation's own worst case over unit normals, which is the Holder
        bound ``SEPARATION_REACH`` is derived from, less the walk's own
        precision - and the witness itself asks for the gap on its axis, so
        the finest the record carries cannot sit above it. Both ends are
        arithmetic. The specimen declaring one has no sharp join anywhere and
        a radius whose fidelity demand is well coarser, so the gap is the only
        thing on it that can ask inside this band - and the band would go
        quietly vacuous the day that stopped being true, passing on whatever
        demand was finest instead.
        """
        if specimen.gap is None:
            pytest.skip("this shape has no two lumps to state a clearance between")
        if specimen.beside is not None:
            pytest.skip("this clearance is to a second object, which is the case below")
        record = _side(artifacts, specimen.name, side)
        if record["status"] != "meshed":
            pytest.skip("nothing was meshed, so nothing was measured off it")
        finest = _finest(record)
        floor = specimen.gap / SEPARATION_REACH * (1.0 - CHORD_TOLERANCE)
        assert floor <= finest <= specimen.gap * (1.0 + WITNESS), (
            f"{specimen.name!r} ({specimen.subject}) {side} has a gap of "
            f"{specimen.gap:.4g} between its lumps and asks for cells of "
            f"{finest:.4g}, where no reading of that gap can allocate below "
            f"{floor:.4g} - so either nothing measured the gap and the mesh is "
            "free to close it, or something else on the shape now asks for less "
            "and this specimen has stopped being the instrument it says it is"
        )

    @pytest.mark.parametrize("side", SIDES, ids=SIDE_IDS)
    def test_a_gap_to_a_second_drawn_object_is_asked_for_across(self, specimen, artifacts, side):
        """The same clearance, drawn the way a user assembles a device.

        A gap is a property of two bodies, and the pairing is over the bodies
        the mesher was handed - so which drawn object each came from is not
        something it can see. That is the claim rather than the assumption: a
        specimen is one object, the corpus drew them one at a time, and until
        there was a second object nothing here ran that pairing across one.

        Bounded to the same band for the reason the case above states it, and
        the specimen is drawn from the same two lumps its compound sibling is
        built from, so a difference between the two is the object count and
        nothing else.
        """
        if specimen.beside is None:
            pytest.skip("this specimen is one drawn object")
        record = _side(artifacts, specimen.name, side)
        if record["status"] != "meshed":
            pytest.skip("nothing was meshed, so nothing was measured off it")
        assert record["beside"], (
            f"{specimen.name!r} {side} declares a second object and the layer made "
            "nothing of it, so the gap has only one side"
        )
        finest = _finest(record)
        floor = specimen.gap / SEPARATION_REACH * (1.0 - CHORD_TOLERANCE)
        assert floor <= finest <= specimen.gap * (1.0 + WITNESS), (
            f"{specimen.name!r} ({specimen.subject}) {side} stands {specimen.gap:.4g} "
            f"from a second drawn object and asks for cells of {finest:.4g}, "
            f"where no reading of that gap can allocate below {floor:.4g} - so "
            "the pairing did not cross the two objects, and a mesh is free to "
            "short them together"
        )

    @pytest.mark.parametrize("side", SIDES, ids=SIDE_IDS)
    def test_a_dielectric_is_counted_across_its_own_wall(self, specimen, artifacts, side):
        """What a layer asks is a count, and a count is not a fit.

        openEMS averages a dielectric over the cell rather than sampling it at a
        point, so nothing about the layer staircases and no cell has to fit
        inside it. What under-resolves is the field varying across the layer,
        and the answer to that is several cells over the thickness - the pitch
        along the layer's own normal, which is the wall over the count.

        Bounded on both sides rather than asserted equal, because the demand a
        count arrives as is a cell size **per axis** and delivery is carried by
        the dominant axis alone. Its cell is the chord's run on it over the
        count - ``t max_j(|m_j|) / n`` - which is the wall over the count
        exactly where the normal lies on an axis, and ``sqrt(3)`` times finer
        where the normal points equally at all three: those two are the whole
        range any orientation can produce, and both ends are arithmetic rather
        than observation.

        It stays a narrow window. The specimens declaring a wall carry no sharp
        join and are asked nothing about their curvature, so the count is the
        only thing on either drawing that can ask for a cell this small; and the
        fault the rule exists to stop - a thickness read off the bounding box,
        which on the rolled one is the bore - lands an order outside the coarse
        end rather than just past it.
        """
        if not specimen.dielectric:
            pytest.skip("a conductor is asked that a cell fit inside it, not how many span it")
        record = _side(artifacts, specimen.name, side)
        if record["status"] != "meshed":
            pytest.skip("nothing was meshed, so nothing was measured off it")
        finest = _finest(record)
        coarse = specimen.wall / corpus.ELEMENTS_ACROSS
        fine = coarse / math.sqrt(DIMENSIONS)
        assert fine * (1.0 - CHORD_TOLERANCE) <= finest <= coarse * (1.0 + CHORD_TOLERANCE), (
            f"{specimen.name!r} ({specimen.subject}) {side} has a wall of "
            f"{specimen.wall:.4g} and asks for cells of {finest:.4g}, where "
            f"{corpus.ELEMENTS_ACROSS} across that wall is {coarse:.4g} on an "
            f"axis-facing layer and no orientation can ask for finer than "
            f"{fine:.4g} - so the layer is not being counted along its own "
            "normal at all"
        )


@pytest.mark.slow
class TestTheRoundTripItself:
    """What the comparison rests on, asserted rather than assumed."""

    def test_the_file_carried_every_shape(self, artifacts):
        """A specimen STEP will not write is one the class above skips silently.

        Not a judgement on this workbench - it is a file format's limit - but a
        skip nobody sees is a corpus that shrank, so it is named here.
        """
        lost = {
            name: record["round_trip"]["reason"]
            for name, record in _carried(artifacts).items()
            if record["round_trip"]["status"] == "unavailable"
        }
        assert not lost, f"STEP would not carry these: {lost}"

    def test_the_only_shape_the_file_changes_is_the_reversed_one(self, artifacts):
        """The skip that keeps the invariant honest, held to one specimen.

        A reversed solid is the kernel's way of writing the complement of a
        region, and it is not a thing STEP has: the exporter writes the faces and
        the importer reads them back wound the ordinary way, so what comes back
        is the box rather than everything outside it. Its volume changes sign,
        which is how the comparison above knows not to make it.
        """
        changed = {
            name: _moved(record)
            for name, record in _carried(artifacts).items()
            if record["round_trip"]["status"] != "unavailable" and _moved(record)
        }
        assert set(changed) == {"inside_out"}, changed

        record = _record(artifacts, "inside_out")
        assert record["measured"]["volume"] < 0.0 < record["round_trip"]["measured"]["volume"]
        assert record["status"] == "refused"
        assert record["round_trip"]["status"] == "meshed", (
            "a solid wound the ordinary way is meshed, so this is the shape "
            "changing and not the layer answering differently about one shape"
        )

    def test_the_file_lays_the_same_lattice_on_every_shape(self, artifacts):
        """What the demand comparison rests on, and the one thing it cannot see.

        Every length a drawing carries is read at a station, and the lattice is
        what decides where the stations are. Two sides that laid the same
        lattice measured one shape in one set of places; two sides that did not
        measured two sets, and their demands then differ by where they were
        taken rather than by what the drawing asks for. The comparison in the
        class above would report that as a shape asking for a different grid,
        which is the wrong sentence about the right fact.

        The lattice is a function of the shape because ``lfs.py::_steps`` takes
        each direction's count from a length in space. It is an integer taken
        from a real, so it steps, and a drawing sits on a step whenever a face is
        a round number of cells across - which is most drawings. That is what
        :data:`~Microwave.Solvers.openems.lfs.COUNT_SLACK` is under the step for,
        and this is the assertion that says so.
        """
        both = {
            name: record
            for name, record in _carried(artifacts).items()
            if record["status"] == "meshed" and record["round_trip"]["status"] == "meshed"
        }
        # Named, because two empty lists agree with each other. A record that
        # never carried a lattice would pass this without a face being read.
        laid = {name: len(record["lattice"]) for name, record in both.items()}
        assert sum(laid.values()), (
            f"no specimen laid a lattice at all, so nothing here was compared: {laid}"
        )

        moved = {
            name: _lattice_moved(record, record["round_trip"])
            for name, record in both.items()
            if _lattice_moved(record, record["round_trip"])
        }
        assert not moved, f"the file lays a different lattice on these: {moved}"


class TestTheFieldTheDemandsInduce:
    """The comparison's own arithmetic, on demands written here.

    :func:`envelope` is what decides whether two demand sets are the same demand
    set, so a mistake in it makes every case above agree.
    """

    def test_a_demand_holds_its_own_interval_at_its_own_size(self):
        assert envelope([(1.0, 3.0, 0.5)], np.array([1.0, 2.0, 3.0])) == pytest.approx(0.5)

    def test_it_ramps_away_at_the_grading_slope(self):
        got = envelope([(0.0, 0.0, 0.5)], np.array([2.0]))
        assert got[0] == pytest.approx(0.5 + 2.0 * GRADING, abs=0.0)

    def test_nothing_is_asked_beyond_the_cap(self):
        assert envelope([], np.array([0.0, 5.0])) == pytest.approx(corpus.CELL_CAP)
        far = envelope([(0.0, 0.0, 0.5)], np.array([1e6]))
        assert far[0] == pytest.approx(corpus.CELL_CAP)

    def test_the_finest_demand_at_a_point_is_the_one_that_holds(self):
        points = np.array([0.0, 10.0])
        got = envelope([(0.0, 0.0, 2.0), (10.0, 10.0, 0.1)], points)
        assert got[0] == pytest.approx(2.0)
        assert got[1] == pytest.approx(0.1)

    def test_a_demand_another_covers_changes_nothing(self):
        """Which is why two sets that pruned differently are still one set."""
        points = np.linspace(-5.0, 5.0, 21)
        alone = envelope([(0.0, 0.0, 0.2)], points)
        covered = envelope([(0.0, 0.0, 0.2), (1.0, 1.0, 0.2 + GRADING)], points)
        assert covered == pytest.approx(alone, abs=0.0)
