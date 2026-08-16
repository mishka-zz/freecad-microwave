# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The geometry layer, run over shapes the real CAD kernel made.

The rest of the suite tests this layer against a hand-written ``Shape``, which is
a model of the kernel and holds only the behaviours somebody thought to put in
it. A defect living in the difference between that model and the kernel - a
tessellation answered from a view provider's display mesh, an edge whose
curvature is sampled at one point, a flat area whose boundary asks for nothing -
is invisible to every test written against the model. So these run the kernel.

What is asserted here is the weakest thing worth asserting, and it is
deliberately weak: every shape a user can draw gets an **honest verdict**. It is
meshed, or it is refused with a sentence naming the object and saying what is
wrong with it. Nothing crashes, and nothing quietly disappears - which is the
failure that matters, because a shape that vanishes leaves a run that completes,
looks clean, and answers about a different device.

Whether what came back is *correct* is a different question, asked by the oracles
that read the same artifacts.
"""

from __future__ import annotations

import math

import pytest

from Microwave.Solvers.openems.geometry import (
    BOX_TOLERANCE,
    FLATNESS,
    MAX_SHAPE_LOSS,
    MOST_OF_A_SKIN,
    NOT_A_REGION,
)
from tests import corpus
from tests.conftest import corpus_record as _record

pytestmark = pytest.mark.slow

#: How far outside its own bounding box a piece may sit. Nothing: a piece is
#: either the box the kernel measured, or the extent of vertices the kernel
#: placed on the shape, and neither can be outside the first.
TOUCHING = 0.0


def _volume(piece) -> float:
    """The volume a triangle set encloses, by the divergence theorem.

    ``sum(a . (b x c)) / 6`` over outward-wound triangles. Written here rather
    than imported so the answer does not come from the code being judged.
    """
    vertices = piece["vertices"]
    total = 0.0
    for one, two, three in piece["faces"]:
        a, b, c = vertices[one], vertices[two], vertices[three]
        cross = (
            b[1] * c[2] - b[2] * c[1],
            b[2] * c[0] - b[0] * c[2],
            b[0] * c[1] - b[1] * c[0],
        )
        total += a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]
    return abs(total) / 6.0


def _area(piece) -> float:
    """The area a triangle set covers: half the cross product, summed."""
    vertices = piece["vertices"]
    total = 0.0
    for one, two, three in piece["faces"]:
        a, b, c = vertices[one], vertices[two], vertices[three]
        u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        cross = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        total += 0.5 * math.sqrt(sum(component * component for component in cross))
    return total


def _extent_volume(piece) -> float:
    return math.prod(piece["upper"][dim] - piece["lower"][dim] for dim in range(3))


def _fills_its_box(measured) -> bool:
    """Whether the shape's own measure is its bounding box's.

    Volume against the box for anything carrying a solid, area against the
    rectangle for a flat one. Which of the two applies is the shape's topology
    and not the thinness of its box: a solid rolled down to a foil is still a
    solid, and its area counts both of its faces.
    """
    lower, upper = measured["bound_box"][:3], measured["bound_box"][3:]
    extents = [high - low for low, high in zip(lower, upper)]
    if measured["solids"]:
        box, drawn = math.prod(extents), measured["volume"]
    else:
        flat = [dim for dim in range(3) if extents[dim] <= FLATNESS]
        if len(flat) != 1:
            return False
        box = math.prod(e for dim, e in enumerate(extents) if dim != flat[0])
        drawn = measured["area"]
    return box > 0.0 and math.isclose(drawn, box, rel_tol=BOX_TOLERANCE)


@pytest.mark.parametrize("specimen", corpus.specimens(), ids=lambda s: s.name)
class TestEveryShapeGetsAnHonestVerdict:
    """The corpus gate. A shape is meshed or it is refused, and either way the
    user is told which object and why."""

    def test_the_kernel_could_draw_it(self, specimen, artifacts):
        """A specimen the kernel refuses to build tests nothing, and it would sit
        in the corpus looking like coverage. Caught here rather than counted as
        a pass."""
        record = _record(artifacts, specimen.name)
        assert record["status"] != "undrawable", (
            f"the corpus could not draw {specimen.name!r} ({specimen.subject}), so "
            f"nothing was exercised: {record.get('reason')}"
        )

    def test_it_does_not_crash_the_geometry_layer(self, specimen, artifacts):
        """An exception that is not a refusal reaches the user as a traceback
        with no object named in it."""
        record = _record(artifacts, specimen.name)
        assert record["status"] != "crashed", (
            f"{specimen.name!r} ({specimen.subject}) raised out of the geometry "
            f"layer rather than refusing:\n{record.get('traceback', record.get('reason'))}"
        )

    def test_the_verdict_is_the_one_required(self, specimen, artifacts):
        record = _record(artifacts, specimen.name)
        assert record["status"] == specimen.expect, (
            f"{specimen.name!r} ({specimen.subject}) is required to be "
            f"{specimen.expect} and was {record['status']}: {record.get('reason', '')}"
        )

    def test_a_refusal_names_the_object_and_says_why(self, specimen, artifacts):
        """A refusal the user cannot act on is barely better than a crash."""
        record = _record(artifacts, specimen.name)
        if record["status"] != "refused":
            pytest.skip("this specimen is meshed, so there is no refusal to read")
        reason = record["reason"]
        assert specimen.name in reason, (
            f"the refusal of {specimen.name!r} does not name it, so a user with "
            f"several objects cannot tell which one is meant: {reason}"
        )
        assert len(reason.split()) > 8, (
            f"the refusal of {specimen.name!r} says too little: {reason}"
        )

    def test_triangles_are_carried_by_exactly_the_shapes_that_need_them(self, specimen, artifacts):
        """A bounding box exists for every shape, which is the trap: taking one
        from a cone answers a question that had none, and openEMS would solve
        that box without complaint. The other way round is the cost rather than
        the fault - triangulating something a rectilinear grid holds exactly.

        Asked of every specimen rather than of a list of shapes somebody
        expected to be curved. A list is what a layer branching on what *drew* a
        shape passes: it holds for the primitives anybody thought to name and
        says nothing about the boolean, the sweep or the import that reaches the
        same geometry another way. Here the criterion is the shape's own measure
        against its box's, so a specimen added to the corpus is covered without
        this being touched.
        """
        record = _record(artifacts, specimen.name)
        if record["status"] != "meshed" or len(record["pieces"]) != 1:
            pytest.skip("a shape cut into several pieces has no one piece to compare")
        triangulated = bool(record["pieces"][0]["faces"])
        fills = _fills_its_box(record["measured"])
        assert triangulated != fills, (
            f"{specimen.name!r} came back "
            f"{'triangulated' if triangulated else 'as a plain box'} while its own "
            f"measure {'is' if fills else 'is not'} its bounding box's"
        )

    def test_it_claims_no_space_the_user_did_not_draw(self, specimen, artifacts):
        """Pieces stay inside the shape's own bounding box.

        This is the asymmetric half of placement, and the one that is always a
        fault. A polygonised curve is *inscribed*, so falling short of the box is
        ordinary and is bounded by the volume budget instead; reaching past it is
        a conductor occupying space nobody drew, which no tessellation does by
        accident.

        A skin is the one shape that reaches past on purpose, having been given a
        thickness the drawing does not carry. What it may not do is reach past by
        more than that thickness: an offset that ran further, or ran both ways,
        is metal nobody asked for, and the bound is the same statement as before
        with the thickness written into it.
        """
        record = _record(artifacts, specimen.name)
        if record["status"] != "meshed":
            pytest.skip("nothing was meshed, so there is nothing to place")
        pieces = record["pieces"]
        assert pieces, f"{specimen.name!r} meshed to no pieces at all, which is not a mesh"

        allowed = TOUCHING + (record.get("skin") or 0.0)
        drawn = record["measured"]["bound_box"]
        for dim in range(3):
            low = min(piece["lower"][dim] for piece in pieces)
            high = max(piece["upper"][dim] for piece in pieces)
            assert low >= drawn[dim] - allowed, (
                f"{specimen.name!r} is drawn from {drawn[dim]:g} on axis {dim} and "
                f"a piece starts at {low:g}, outside the shape"
            )
            assert high <= drawn[dim + 3] + allowed, (
                f"{specimen.name!r} is drawn to {drawn[dim + 3]:g} on axis {dim} and "
                f"a piece reaches {high:g}, outside the shape"
            )

    def test_it_keeps_the_measure_of_what_was_drawn(self, specimen, artifacts):
        """A solid's triangles enclose its volume, and a sheet's cover its area.

        This bounds the shortfall the test above deliberately allows. It is a
        second transcription of the layer's own budget for a solid, and worth
        little there; where it earns its place is a **sheet**, which the layer
        measures before collapsing its vertices onto the declared plane and this
        measures after - so a collapse that moved area has nowhere to hide.
        """
        record = _record(artifacts, specimen.name)
        if record["status"] != "meshed":
            pytest.skip("nothing was meshed, so there is nothing to measure")
        if record.get("skin"):
            # A skin keeps no *drawn* measure - its area is a surface's, what
            # gets solved is a volume, and the figure the kernel computes across
            # an open surface is not a region. What it does have to keep is the
            # sweep: offsetting a surface through a thickness encloses that
            # thickness times the area, give or take what the shape's own
            # curvature adds at the rim. That is the one measurement saying how
            # much metal was invented, so it is made here rather than skipped.
            pieces = record["pieces"]
            assert len(pieces) == 1, "a skin is offset into one solid"
            swept = record["measured"]["area"] * record["skin"]
            got = _volume(pieces[0])
            assert got == pytest.approx(swept, rel=MOST_OF_A_SKIN, abs=0.0), (
                f"{specimen.name!r} was drawn with area "
                f"{record['measured']['area']:g} and given a thickness of "
                f"{record['skin']:g}, so the metal should sweep {swept:g}; what "
                f"will be solved holds {got:g}"
            )
            return
        pieces = record["pieces"]
        if len(pieces) > 1:
            pytest.skip("pieces may overlap, so their measures do not simply add")

        piece = pieces[0]
        drawn = record["measured"]
        # The magnitude throughout: a solid wound the other way reports its
        # volume negative and fills exactly the space it was drawn in.
        if piece["sheet_normal"] is not None:
            got, want, what = _area(piece), abs(drawn["area"]), "area"
        elif piece["faces"]:
            got, want, what = _volume(piece), abs(drawn["volume"]), "volume"
        else:
            got = _extent_volume(piece)
            want, what = abs(drawn["volume"]), "volume"
            if want == 0.0:
                pytest.skip("a flat region held as a box has no volume to compare")

        assert want > 0.0
        assert abs(got - want) <= MAX_SHAPE_LOSS * want, (
            f"{specimen.name!r} was drawn with {what} {want:g} and what will be "
            f"solved has {what} {got:g}, off by {100 * abs(got - want) / want:.3g}%"
        )


class TestTheCorpusItself:
    def test_every_specimen_ran(self, artifacts):
        """A specimen silently missing from the run is a corpus that shrank."""
        missing = [s.name for s in corpus.specimens() if s.name not in artifacts]
        assert not missing, f"these specimens produced no artifact: {missing}"

    def test_a_region_and_a_rounding_are_far_apart_either_side_of_the_threshold(self, artifacts):
        """``NOT_A_REGION`` separates a figure the kernel computed across an
        open surface from the volume of a real solid, and both sides of it are
        measured here rather than declared beside the constant.

        A threshold that sat near either side would decide by itself which
        refusal a user gets, and neither the rounding nor the thinnest solid
        anybody draws is a quantity this repository controls. So what is
        asserted is the room, and a specimen that closes it fails here rather
        than changing a message quietly.
        """
        room = 1000.0
        roundings, regions = [], []
        for name, record in artifacts.items():
            measured = record.get("measured")
            if measured is None:
                continue
            span = math.prod(
                measured["bound_box"][dim + 3] - measured["bound_box"][dim] for dim in range(3)
            )
            if span <= 0.0:
                continue
            ratio = abs(measured["volume"]) / span
            assert ratio < NOT_A_REGION / room or ratio > NOT_A_REGION * room, (
                f"{name!r} fills {ratio:.3g} of its own bounding box, which is "
                "within a few decades of the threshold - so whether it is read as "
                "a region or as the kernel's rounding is decided by the constant "
                "rather than by the shape"
            )
            (roundings if ratio < NOT_A_REGION else regions).append(ratio)
        assert roundings, "no shape in the corpus reports a figure that is only rounding"
        assert regions, "no shape in the corpus encloses a region"

    def test_a_solid_in_several_pieces_becomes_several_primitives(self, artifacts):
        """One object holding lumps that touch, overlap or stand apart is emitted
        as one primitive each.

        A single surface drawn over all of them would have to be a manifold,
        which two lumps meeting at a vertex are not - and the engine needs no
        surface between them, resolving overlap by priority instead.
        """
        for name in ("two_disjoint", "touching_at_a_corner", "self_intersecting"):
            record = _record(artifacts, name)
            assert record["status"] == "meshed", record.get("reason")
            assert len(record["pieces"]) == record["measured"]["solids"], (
                f"{name!r} has {record['measured']['solids']} solids and became "
                f"{len(record['pieces'])} pieces"
            )

    def test_pieces_of_one_object_are_told_apart(self, artifacts):
        """Every piece carries its own label.

        A label is what a refusal names and what the sizing field reads a body
        back by, so two pieces sharing one would leave the second reachable by
        nothing - present in the geometry, absent from everything that sizes it.
        """
        for name in ("two_disjoint", "touching_at_a_corner", "self_intersecting"):
            record = _record(artifacts, name)
            labels = [piece["label"] for piece in record["pieces"]]
            assert len(set(labels)) == len(labels), f"{name!r} reuses a label: {labels}"

    def test_a_solid_wound_the_other_way_is_refused_rather_than_straightened(self, artifacts):
        """A reversed solid is not the shape it looks like.

        The kernel reads it as the complement: its volume is negative, the space
        it appears to occupy answers *outside*, and everywhere beyond answers
        inside. Taking its bounding box would build the one reading, and taking
        the magnitude of its volume would build the same one - but the drawing
        does not say which was meant, and the two differ at every point.
        """
        record = _record(artifacts, "inside_out")
        assert record["measured"]["volume"] < 0.0, "this specimen no longer reads negative"
        assert record["status"] == "refused", record.get("pieces")
        assert "inside out" in record["reason"]

    def test_a_sheet_on_no_axis_is_refused_for_being_on_no_axis(self, artifacts):
        """The refusal has to name the property that failed, and not a neighbouring
        one that holds.

        An outline of any shape is meshed, so telling the user that a sheet
        needs axis-aligned edges sends them to redraw something that would
        already have worked. What this shape genuinely lacks is a plane to be
        declared at: openEMS lays a zero-thickness conductor at one elevation on
        one axis, and one tilted across all three has no elevation to take.
        """
        record = _record(artifacts, "tilted_sheet")
        assert record["status"] == "refused", record.get("pieces")
        assert "flat on one of the three axes" in record["reason"], record["reason"]

    def test_and_an_open_surface_that_encloses_something_is_not_refused_for_that(self, artifacts):
        """The other side of the same branch, and the reason it cannot be read
        off the sign of a volume: a shell left open reports a figure across its
        faces, and that figure is a region rather than the kernel's rounding.
        Something is missing from the drawing, and it is not thickness."""
        record = _record(artifacts, "open_shell")
        assert record["status"] == "refused", record.get("pieces")
        assert "Check geometry" in record["reason"], record["reason"]
        assert "flat on one of the three axes" not in record["reason"], record["reason"]

    def test_a_solid_thinner_than_the_flatness_floor_is_still_a_solid(self, artifacts):
        """Its area counts both of its faces, so judging it as an area cannot
        agree with any single face's extent however thin it gets."""
        record = _record(artifacts, "near_flat")
        assert record["status"] == "meshed", record.get("reason")
        piece = record["pieces"][0]
        assert piece["sheet_normal"] is None, "a thin solid was classified as a sheet"
        drawn = record["measured"]["bound_box"]
        assert piece["upper"][2] - piece["lower"][2] == pytest.approx(
            drawn[5] - drawn[2], rel=1e-9, abs=0.0
        ), "the thickness was flattened away, so what is solved is a sheet"
