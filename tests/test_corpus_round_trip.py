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
is the sizing field it induces, ``min over demands of size + g|x - y|``, and two
sets that induce the same field ask for the same grid. That is the comparison,
and it is the same argument the sizing layer's own pruning rests on.

The lint in ``test_measured_not_typed.py`` is the cheap half of the same rule,
and it catches the forms this cannot: a branch on a type that no corpus specimen
happens to reach.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Microwave.Solvers.openems.lfs import CHORD_TOLERANCE, SURFACE_FIDELITY
from Microwave.Solvers.openems.sizing import DIMENSIONS
from tests import corpus
from tests.conftest import corpus_record as _record

#: The sizing field's slope, as ``ln(max cell ratio)``. The invariant holds at
#: any positive slope - it says two demand sets are the same demand set - and
#: this is the one the sizing layer's pruning assumes, so it is the slope at
#: which two sets that pruned differently still have to agree.
GRADING = math.log(1.3)

#: Points per axis the two fields are compared at, and how far past the shape
#: they reach as a share of its extent. Beyond the shape as well as across it,
#: because a demand is a point constraint that ramps outward and a field that
#: agrees only where the metal is would say nothing about the grid around it.
SAMPLES = 33
BEYOND = 0.25

#: How far apart the two fields may be, relative. A demand is a length arrived
#: at through a chain of kernel queries, so re-parametrising the surface it was
#: taken from moves its last bits and moves the point it was taken at. What the
#: field must not do is answer a different question, and this is what separates
#: the two: a grid holds ``ceil(length / size)`` cells across a gap, so a
#: relative move this small can change a line count only where the quotient
#: already sits within it of a whole number. The departure actually measured is
#: on the failure message.
FIELD = 1e-9

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


def _ordered(pieces):
    """Pieces by where they are, since a file may hand the lumps back in any order."""
    return sorted(pieces, key=lambda piece: tuple(piece["lower"]))


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
    """Each specimen against itself, written to STEP and read back."""

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

        Compared as the field the demands induce and not as the demands
        themselves: re-parametrising a surface moves where it is sampled, so the
        two lists differ in length on the cone and on the text without either
        asking for anything the other does not.
        """
        record, back = _both(artifacts, specimen.name)
        if record["status"] != "meshed":
            pytest.skip("nothing was meshed, so nothing was measured off it")
        if not any(record["demands"]) and not any(back["demands"]):
            pytest.skip("this shape is held exactly, so no length was measured off it")
        if record.get("skin"):
            pytest.skip("a skin is sampled on a surface this layer built - see the test below")

        worst, worst_axis, worst_point = 0.0, 0, 0.0
        for dim, (low, high) in enumerate(_across(record)):
            points = np.linspace(low, high, SAMPLES)
            mine = envelope(record["demands"][dim], points)
            theirs = envelope(back["demands"][dim], points)
            departure = np.abs(mine - theirs) / np.maximum(mine, theirs)
            if departure.max() > worst:
                worst = float(departure.max())
                worst_axis, worst_point = dim, float(points[departure.argmax()])

        assert worst <= FIELD, (
            f"{specimen.name!r} ({specimen.subject}) asks for a different grid "
            f"through a file: the sizing field is off by {worst:.3g} at "
            f"{worst_point:.6g} on axis {worst_axis}"
        )

    def test_a_skin_asks_for_the_same_lengths_through_the_round_trip(self, specimen, artifacts):
        """The finest and coarsest cell asked for on each axis, and not where.

        A skin's demands are read off a solid this layer *built*, and the two
        solids agree - the corner comparison above says so. What the round trip
        moves is where that solid gets sampled, the parameterisation being
        inherited from the surface it was offset from, so the field between
        samples differs while nothing asks for a different length. The finest
        demand is what costs, and it is what has to survive.
        """
        record, back = _both(artifacts, specimen.name)
        if not record.get("skin") or record["status"] != "meshed":
            pytest.skip("this specimen is not a conductor drawn as a surface")

        for dim in range(DIMENSIONS):
            mine = [size for _, _, size in record["demands"][dim]]
            theirs = [size for _, _, size in back["demands"][dim]]
            assert bool(mine) == bool(theirs), (
                f"{specimen.name!r} measures lengths on axis {dim} through a file "
                "and not off the drawing, or the other way about"
            )
            if not mine:
                continue
            assert min(theirs) == pytest.approx(min(mine), rel=SAME_SHAPE, abs=0.0)
            assert max(theirs) == pytest.approx(max(mine), rel=SAME_SHAPE, abs=0.0)

    def test_a_shape_the_grid_does_not_hold_is_asked_for_finely_enough(self, specimen, artifacts):
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
        record = _record(artifacts, specimen.name)
        if record["status"] != "meshed":
            pytest.skip("nothing was meshed, so nothing was measured off it")
        if not any(piece["faces"] for piece in record["pieces"]):
            pytest.skip("this shape is held exactly, so its own faces are where the grid is")
        assert any(record["demands"]), (
            f"{specimen.name!r} ({specimen.subject}) is meshed as triangles and "
            "asks the grid for nothing, so no length was read off it at all"
        )
        finest = min(size for axis in record["demands"] for _, _, size in axis)
        across = _size_of(record["measured"]) / finest
        assert across >= FOLLOWED, (
            f"{specimen.name!r} ({specimen.subject}) is meshed as triangles and asks "
            f"for cells of {finest:.4g}, which is {across:.3g} of them across the "
            f"whole shape - the grid steps over it rather than following it"
        )

    def test_a_body_is_asked_for_across_its_own_wall(self, specimen, artifacts):
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
        record = _record(artifacts, specimen.name)
        if record["status"] != "meshed":
            pytest.skip("nothing was meshed, so nothing was measured off it")
        finest = min(size for axis in record["demands"] for _, _, size in axis)
        fits = specimen.wall / math.sqrt(3)
        assert finest <= fits * (1.0 + CHORD_TOLERANCE), (
            f"{specimen.name!r} ({specimen.subject}) has a wall of "
            f"{specimen.wall:.4g} and asks for cells of {finest:.4g}, where a "
            f"cell fitting inside that wall is {fits:.4g} - so nothing measured "
            "the wall and the grid is free to open it"
        )

    def test_a_gap_inside_one_object_is_asked_for_across(self, specimen, artifacts):
        """Two lumps drawn in one operation are still two conductors.

        A gap the grid puts no cell inside is a gap the mesh closes, and two
        conductors shorted together is a run that completes and answers about a
        device nobody drew. The measurement has to come from the lumps
        themselves: they are one object, so nothing outside is paired with them.

        A gap has a direction, and a witness pair between two lumps side by side
        points along one axis - so what a cell must fit inside is the gap
        itself, where a cross-section spends its length across all three.

        Read as the finest demand the whole record carries, and asserted to
        *be* the gap rather than merely to clear it. The specimen declaring one
        has no sharp join anywhere and a radius whose fidelity demand is well
        coarser, so the gap is the only thing on it that can ask for a cell this
        small - and an inequality would go quietly vacuous the day that stopped
        being true, passing on whatever demand was finest instead.
        """
        if specimen.gap is None:
            pytest.skip("this shape has no two lumps to state a clearance between")
        record = _record(artifacts, specimen.name)
        if record["status"] != "meshed":
            pytest.skip("nothing was meshed, so nothing was measured off it")
        finest = min(size for axis in record["demands"] for _, _, size in axis)
        assert finest == pytest.approx(specimen.gap, rel=WITNESS, abs=0.0), (
            f"{specimen.name!r} ({specimen.subject}) has a gap of "
            f"{specimen.gap:.4g} between its lumps and asks for cells of "
            f"{finest:.4g} - so either nothing measured the gap and the mesh is "
            "free to close it, or something else on the shape now asks for less "
            "and this specimen has stopped being the instrument it says it is"
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
