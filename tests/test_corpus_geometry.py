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
    CUT_TOLERANCE,
    FLATNESS,
    MAX_SHAPE_LOSS,
    MOST_OF_A_SKIN,
    NOT_A_REGION,
)
from Microwave.Solvers.openems.lfs import SHARP_DEGREES
from Microwave.Solvers.openems.sizing import DIMENSIONS, separation
from tests import corpus
from tests.conftest import corpus_record as _record
from tests.directions import rotated, scaled, widest_across

pytestmark = pytest.mark.slow

#: How far outside its own bounding box a piece may sit, as a share of the
#: shape's own size. A piece is either the box the kernel measured or the extent
#: of vertices the kernel placed on the shape, and the drawing puts nothing
#: between the two - but the kernel reaches them by different arithmetic, so on a
#: solid turned about a line lying on no axis the triangulation's widest vertex
#: lands past the box by the rounding a rotation of those coordinates leaves.
#:
#: A share rather than a length, so it follows a shape drawn at any scale.
TOUCHING = 1e-12


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


#: How closely the reported displacement has to match one measured facet by
#: facet.
#:
#: It is set by the kernel's own volume rather than by the arithmetic, the
#: identity the figure rests on being exact. What it has to absorb is the other
#: input: the volume of a surface the kernel re-fits is integrated to a tolerance
#: of its own, which enters the numerator as a fixed quantity of volume and so as
#: a fixed distance. It does not fall as the triangulation refines, which is why
#: it is a share here and cannot be a share of the departure.
#:
#: A smaller effect runs the other way: the denominator is the area the faces
#: were drawn with, the arc rather than the chord, which biases the figure low.
#:
#: A width rather than a bound, therefore. Every specimen inside it is one
#: measurement, and one that needed more would fail rather than be accommodated.
DEPARTURE_TOLERANCE = 0.06

#: Below this a reported displacement is the arithmetic of subtracting two
#: volumes rather than a departure, in mm. One nanometre: far under any length
#: this workbench meshes, and far over what subtracting two figures of a
#: millimetre-sized body leaves in double precision.
DEPARTURE_IS_NOTHING = 1e-9


#: How far a sheet's emitted outline may stand from the drawn one, as a share of
#: what the drawing bends through at that place - never of the sharpest radius
#: anywhere on the sheet, which would divide a departure sitting on the widest
#: curve by a radius from the narrowest.
#:
#: Set by the run rather than chosen: the worst sheet in the corpus is a glyph,
#: whose outline is a spline and so is stepped unevenly, and it stands well
#: inside this. That glyph is also the one specimen whose shape this repository
#: does not control - it is drawn in whatever outline font the machine has - so
#: the room between it and this carries a machine drawing a different letter.
#:
#: What it is here to catch misses by orders rather than by a factor: a request
#: scaled off the sheet grows with the sheet while the feature does not, runs out
#: of the band the kernel answers in, and the polygon saturates at a square.
OUTLINE_TOLERANCE = 0.01


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

    def test_every_station_is_placed_where_the_kernel_places_it(self, specimen, artifacts):
        """A face's lattice is placed against the face's own area and its own
        boundary, and the kernel is asked only where the two could disagree.
        This is the one place that route meets a real face.

        The stand-ins the rest of the suite uses answer their own boundary, so
        they agree with it by construction. A wire whose edges the kernel orders
        differently, an edge with no parameter curve, a seam a real surface
        carries and a stand-in does not - each of those is a face placed one way
        here and another way by the kernel, and nothing else would see it.
        """
        record = _record(artifacts, specimen.name)
        if record["status"] != "meshed":
            pytest.skip("this specimen is not meshed, so no lattice is laid on it")
        placed = record["stations"]
        assert placed["apart"] == 0, (
            f"{specimen.name!r} ({specimen.subject}) has stations the kernel and "
            f"the boundary disagree about, of {placed['walked']} laid on "
            f"{placed['faces']} faces"
        )
        assert placed["walked"] > 0, "no lattice was laid, so nothing was compared"

    def test_the_skip_costs_the_mesher_no_demand(self, specimen, artifacts):
        """The curvature reading skips a face whose surface says it is planar.
        This is the one place that skip meets a real kernel, and the only claim
        worth holding it to is that nothing the mesher acts on moves.

        Every stand-in in the suite answers zero for a face it was told is flat,
        so none of them can score this: they agree by construction. And what
        ``isPlanar`` answers is a distance from a fitted plane rather than a
        curvature, so a face may be called planar and still carry a radius - the
        question is whether the demand it would have raised was one the reach
        keeps.
        """
        record = _record(artifacts, specimen.name)
        if record["status"] != "meshed":
            pytest.skip("this specimen is not meshed, so no face is read for a curvature")
        flat = record["flatness"]
        assert flat["demands"] == flat["demands_asking_every_face"], (
            f"{specimen.name!r} ({specimen.subject}) loses "
            f"{flat['demands_asking_every_face'] - flat['demands']} curvature demands to the "
            f"skip, the first of them at {flat['demands_the_skip_drops'][:3]}"
        )

    def test_the_kernel_answered_every_station_the_measurement_put_to_it(self, specimen, artifacts):
        """Each site that reads a length off a shape absorbs a kernel refusal
        and carries on, because one station is not the face. What the run keeps
        of that is a count per source, and the report reads it.

        Held to empty rather than to a number. A refusal is not this workbench's
        arithmetic and there is no figure it should settle on: a station the
        kernel declines is a place nothing was read, and a source read nowhere
        is a face sized by whatever else reaches it. It is also the only place
        the record meets a kernel: every stand-in that scores it elsewhere
        declines exactly what it was built to decline, so a new specimen, a
        kernel version or a change to how a face is sampled is seen here and
        nowhere else. The round trip is scored beside the drawing, because STEP
        re-approximates a trimmed surface and so is where a face is likeliest
        to arrive carrying a place its own parameterisation cannot answer for.

        A specimen putting no station to the kernel passes this and says
        nothing, which is why the guard against that is over the corpus rather
        than here: a drawing whose every face is a plane and whose every body
        reached the mesher as a box is asked nothing at all, and there are
        several of those on purpose.
        """
        record = _record(artifacts, specimen.name)
        if record["status"] != "meshed":
            pytest.skip("this specimen is not meshed, so no station is put to the kernel")
        for side, read in (
            ("as drawn", record),
            ("through a STEP round trip", record["round_trip"]),
        ):
            if read.get("status") != "meshed":
                continue
            declined = {source: counts for source, counts in read["refused"].items() if counts[1]}
            assert not declined, (
                f"{specimen.name!r} ({specimen.subject}) {side} has stations the kernel "
                f"would not answer for: {sorted(declined.items())[:3]}"
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

        What it does not reach is the shape that comes back in **pieces**, which
        has no one measure to compare and is skipped below. A cut is judged by
        whether its pieces add back up to the shape, which is a different
        criterion asked in its own test.
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

        drawn = record["measured"]["bound_box"]
        span = max(drawn[dim + 3] - drawn[dim] for dim in range(3))
        allowed = TOUCHING * span + (record.get("skin") or 0.0)
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
            #
            # Summed over the pieces, because the solid built off the surface is
            # then held however the layer holds any: whole where it is curved,
            # and cut into the boxes it is made of where its faces are square to
            # the grid - which a wall creased at a right angle is.
            pieces = record["pieces"]
            # One solid all the same, and that is what the triangles say: a
            # single surface, or none because the whole of it was cut into
            # boxes. Several triangulated lumps would be an offset that broke
            # the surface up, and their volumes could still add to the sweep.
            assert len([piece for piece in pieces if piece["faces"]]) <= 1, (
                f"{specimen.name!r} was offset into several triangulated solids"
            )
            swept = record["measured"]["area"] * record["skin"]
            got = sum(
                _volume(piece) if piece["faces"] else _extent_volume(piece) for piece in pieces
            )
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

    def test_a_sheet_s_outline_is_followed_to_a_share_of_what_it_bends_through(
        self, specimen, artifacts
    ):
        """What reaches the engine is a polygon, and how far it stands from the
        curve is the only thing a flat sheet's triangulation decides.

        Scored against the radius the outline bends through where the polygon
        stands furthest off it, never against the sheet's own size: the two are
        the same length on a disc and part company on a clearance cut in a plane.
        """
        record = _record(artifacts, specimen.name)
        if record["status"] != "meshed":
            pytest.skip("nothing was emitted, so there is no outline to score")
        scored = 0
        for piece in record["pieces"]:
            share, departure = piece["outline_share"], piece["outline_departure"]
            if share is None:
                continue
            scored += 1
            # A chord across an arc stands off it by something, so a departure of
            # exactly nothing is the measurement having gone quiet.
            assert share > 0.0, (
                f"{specimen.name!r} has a curved outline and its emitted polygon "
                "measures no departure from it at all, which no polygon does"
            )
            assert share <= OUTLINE_TOLERANCE, (
                f"{specimen.name!r} stands {departure:g} mm off the outline it "
                f"was drawn with, which is {share:.2%} of what that outline bends "
                "through where it stands furthest off"
            )
        if not scored:
            pytest.skip("no sheet here has a curved outline")

    def test_what_the_run_says_it_departed_by_is_what_it_departed_by(self, specimen, artifacts):
        """The reported displacement, scored against measuring every facet.

        The report divides the volume the triangles lost by the area that curves
        and touches no facet; this compares it against asking the kernel how far
        each facet's middle stands from the surface. Two methods for one number,
        so it checks the identity the report rests on rather than running the
        same arithmetic twice.

        Equality only where the surface bends one way. Where it bends both, the
        volume lost on one wall is volume gained on the other and the figure is a
        net of the two. The report says so on those, and what is asserted here is
        that it understates rather than by how much.
        """
        record = _record(artifacts, specimen.name)
        if record["status"] != "meshed":
            pytest.skip("nothing was meshed, so nothing departed")
        scored = 0
        for piece in record["pieces"]:
            measured = piece.get("measured_displacement")
            departure = piece.get("reported_displacement")
            if measured is None or departure is None:
                continue
            truth, reported = measured["mean"], departure["displaced"]
            if truth <= 0.0:
                # Nothing moved, so there is no mean to be near.
                assert abs(reported) <= DEPARTURE_IS_NOTHING, (
                    f"{specimen.name!r} reports a displacement of {reported:g} mm "
                    "where no facet moved at all"
                )
                scored += 1
                continue
            if departure["turns_both_ways"]:
                assert abs(reported) <= truth * (1.0 + DEPARTURE_TOLERANCE), (
                    f"{specimen.name!r} bends both ways, so its reported "
                    f"{reported:g} mm is a net and cannot exceed the {truth:g} mm "
                    "measured facet by facet"
                )
            else:
                assert abs(reported) == pytest.approx(truth, rel=DEPARTURE_TOLERANCE, abs=0.0), (
                    f"{specimen.name!r} reports {reported:g} mm against {truth:g} mm "
                    "measured facet by facet, and its surface bends one way - so "
                    "the two are the same quantity and should agree"
                )
            scored += 1
        if not scored:
            pytest.skip("no triangulated piece small enough to measure facet by facet")

    def test_a_surface_that_bends_both_ways_is_the_only_one_that_understates(
        self, specimen, artifacts
    ):
        """The flag has to be the whole of the exception: it may not fire on a
        shape whose figure is exact, nor stay quiet on one whose figure is low.
        So the shapes that carry it and the shapes whose two figures disagree are
        asserted to be the same population.
        """
        record = _record(artifacts, specimen.name)
        if record["status"] != "meshed":
            pytest.skip("nothing was meshed, so nothing departed")
        for piece in record["pieces"]:
            measured = piece.get("measured_displacement")
            departure = piece.get("reported_displacement")
            if measured is None or departure is None or measured["mean"] <= 0.0:
                continue
            agrees = abs(departure["displaced"]) == pytest.approx(
                measured["mean"], rel=DEPARTURE_TOLERANCE, abs=0.0
            )
            assert agrees != departure["turns_both_ways"], (
                f"{specimen.name!r} reports {departure['displaced']:g} mm against a "
                f"measured {measured['mean']:g} mm and says it bends both ways is "
                f"{departure['turns_both_ways']} - so the flag and the figure "
                "disagree about whether this shape's displacement nets"
            )


class TestTheCorpusItself:
    def test_every_specimen_ran(self, artifacts):
        """A specimen silently missing from the run is a corpus that shrank."""
        missing = [s.name for s in corpus.specimens() if s.name not in artifacts]
        assert not missing, f"these specimens produced no artifact: {missing}"

    def test_the_corpus_scores_both_sides_of_the_flatness_question(self, artifacts):
        """A gate that only ever sees one answer is not scoring the question.

        Three things have to be in the corpus for the per-specimen comparison to
        mean anything: a face the reading declines to sample, a face it samples
        that bends, and a face it declines that the kernel says bends - which is
        the free-form surface ``isPlanar`` admits, and the only one where the
        skip could take a demand with it.
        """
        skipped = bent = free = 0
        for name in artifacts:
            flat = _record(artifacts, name).get("flatness")
            if flat is None:
                continue
            skipped += flat["skipped"]
            bent += flat["asked"] if flat["worst_asked"] > 0.0 else 0
            free += flat["skipped"] if flat["worst_skipped"] > 0.0 else 0
        assert skipped > 0, "no face anywhere in the corpus is skipped as planar"
        assert bent > 0, "no face anywhere in the corpus is asked and answers a bend"
        assert free > 0, (
            "every face the corpus skips answers a curvature of exactly zero, so "
            "nothing here exercises a free-form surface the kernel calls planar - "
            "which is the case where the skip could cost a demand"
        )

    #: The sheets carrying one clearance, drawn with different planes around it.
    AROUND_ONE_CLEARANCE = ("sheet_clearance", "sheet_clearance_wide", "sheet_clearance_round")

    def test_the_corpus_puts_stations_to_the_kernel_for_the_refusal_test_to_score(self, artifacts):
        """The guard on the refusal test above, which passes on an empty record.

        A probe that stopped filling the tally, or an argument dropped on the
        way down to a site, would leave every specimen with nothing declined and
        the whole corpus green. What is held here is that the corpus asks the
        kernel at each site that records one, since a record filled by one of
        them says nothing about the rest.
        """
        asked: dict[str, int] = {}
        for name in artifacts:
            record = _record(artifacts, name)
            for read in (record, record.get("round_trip", {})):
                for source, counts in read.get("refused", {}).items():
                    asked[source] = asked.get(source, 0) + counts[0]
        assert sum(asked.values()) > 0, "no specimen put a station to the kernel"
        # Each marker carries the space in front of it, because a source names
        # the body first and a body labelled 'wedge' holds 'edge' inside its
        # own name.
        for site in (
            " curving on face",
            " through face",
            " rim curving",
            " outline",
            " edge",
            "the gap between",
        ):
            assert any(offered for source, offered in asked.items() if site in source), (
                f"no specimen reaches the site that records {site!r}"
            )

    def test_a_clearance_is_followed_the_same_whatever_plane_is_around_it(self, artifacts):
        """One hole in several ground planes, and the answer may not know which.

        The tolerance above is a bound and this is an identity: a criterion
        stated against the sheet passes any bound loose enough while still
        handing the engine a different hole in each plane. That they agree at all
        is what says the curve at hand is what is being followed.

        Compared as shares rather than as lengths, since a round ground plane's
        own rim departs by its own radius and a square one's sides do not depart
        at all.
        """
        shares = {
            name: _record(artifacts, name)["pieces"][0]["outline_share"]
            for name in self.AROUND_ONE_CLEARANCE
        }
        assert min(shares.values()) > 0.0, (
            "the measurement went quiet, so the three agree about nothing"
        )
        first = shares[self.AROUND_ONE_CLEARANCE[0]]
        assert all(share == pytest.approx(first, rel=1e-9, abs=0.0) for share in shares.values()), (
            "one clearance is followed to different shares of its own radius "
            f"depending on the plane it was cut in: {shares}"
        )

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

    def test_the_corpus_reaches_both_materials_and_both_drawings(self, artifacts):
        """The two branches that would otherwise rest on the hand-written stub.

        The mesher asks a conductor and a dielectric different questions, and it
        pairs bodies without caring which drawn object each came from. A corpus
        of one metal object per specimen exercises neither against a kernel: the
        element count runs nowhere, and the pairing runs only between lumps of
        one compound.

        The two layers are counted here as well, by how many axes carry the
        wall. A count is allocated as ``t / (n |m|)`` per axis, so a layer whose
        normal lies on an axis spends it on that one alone and a layer whose
        normal sweeps a plane spends it on two - which is the composition the
        rule exists for, and the case where reading a thickness off a bounding
        box would answer something else entirely. A corpus holding only the
        first would leave that unmeasured while reading as covered.

        Within ``sqrt(DIMENSIONS)`` of the shape's own finest, that being the
        most the allocation can spread one demand over the axes it points
        along. Against the shape's own rather than against the wall, so this
        counts where the demand went and leaves what it is worth to the gate
        that asserts it.
        """
        assert [s.name for s in corpus.specimens() if s.beside is not None], (
            "no specimen draws a second object, so a gap is only ever measured between lumps of one"
        )
        layers = [s for s in corpus.specimens() if s.dielectric]
        assert layers, "no specimen is a dielectric, so nothing is ever counted across"

        carrying = {}
        for specimen in layers:
            demands = _record(artifacts, specimen.name)["demands"]
            finest = min(size for axis in demands for _, _, size in axis)
            carrying[specimen.name] = sum(
                1
                for axis in demands
                if axis and min(size for _, _, size in axis) <= finest * math.sqrt(DIMENSIONS)
            )
        assert 1 in carrying.values(), f"no layer has its wall on an axis: {carrying}"
        assert any(axes > 1 for axes in carrying.values()), (
            f"every layer has its wall on an axis, so nothing composes one: {carrying}"
        )

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

    def test_one_outline_is_cut_the_same_whether_or_not_it_was_extruded(self, artifacts):
        """A conductor is drawn by extruding an outline as often as by drawing the
        outline, and the two are the same conductor.

        Both are held exactly by a rectilinear grid and neither is described by a
        bounding box: a trace with a bend in it has arms far narrower than the
        span of the box around the whole of it, and everything that sizes a cell
        from a conductor's width or measures what the grid left of one reads that
        span. So what the drawing habit may not decide is whether the pieces
        arrive at all.

        Compared in the plane and not in the thickness, that being the one thing
        the two drawings genuinely differ in.
        """
        flat, standing = (_record(artifacts, name) for name in ("sheet_meander", "extrude_meander"))
        for name, record in (("sheet_meander", flat), ("extrude_meander", standing)):
            assert record["status"] == "meshed", record.get("reason")
            assert not any(piece["faces"] for piece in record["pieces"]), (
                f"{name!r} came back triangulated, so nothing can read a width off it"
            )

        def footprints(record):
            return sorted(
                (piece["lower"][0], piece["lower"][1], piece["upper"][0], piece["upper"][1])
                for piece in record["pieces"]
            )

        # To a tolerance and not exactly: one side went through the kernel's
        # extrude and is read back off a cap face, and a coordinate reached two
        # ways need not agree in its last bits.
        assert footprints(standing) == pytest.approx(footprints(flat), abs=FLATNESS), (
            "the same outline was cut into "
            f"{len(standing['pieces'])} pieces extruded and "
            f"{len(flat['pieces'])} drawn flat, and they do not cover the same "
            "ground"
        )

        # And the thickness is the one thing they may differ in, so it is the one
        # thing worth saying they do: a rectangle stood through no thickness at
        # all is a sheet, and the solid was not drawn as one.
        for piece in standing["pieces"]:
            assert piece["upper"][2] - piece["lower"][2] == pytest.approx(
                corpus.MEANDER_THICKNESS, rel=CUT_TOLERANCE, abs=0.0
            ), "an extruded piece does not stand through the thickness it was drawn"

        # The cut is exact rather than a fit, and pieces that tile add up. Asked
        # of both, and read off what was emitted rather than off what the cut
        # believed: a proof that came out on boxes nobody got is not a proof, and
        # a pair that agreed by both collapsing to one box would pass everything
        # above.
        for name, record, drawn in (
            ("sheet_meander", flat, flat["measured"]["area"]),
            (
                "extrude_meander",
                standing,
                standing["measured"]["volume"] / corpus.MEANDER_THICKNESS,
            ),
        ):
            laid = sum(
                (piece["upper"][0] - piece["lower"][0]) * (piece["upper"][1] - piece["lower"][1])
                for piece in record["pieces"]
            )
            assert laid == pytest.approx(drawn, rel=CUT_TOLERANCE, abs=0.0), (
                f"{name!r} was cut into pieces covering {laid:g} where the shape encloses {drawn:g}"
            )

    def test_a_solid_bounded_by_square_planes_is_cut_rather_than_triangulated(self, artifacts):
        """A rectilinear grid holds such a solid with nothing left over, and what
        a triangulation of one costs is that its box is wider than the metal in
        it - which everything that sizes a cell from a conductor's width, and
        everything that measures what the grid left of one, is reading.

        Selected by the shape's own faces rather than from a list of names, so a
        specimen drawn into the corpus is covered without this being touched.
        Two are named all the same: an empty selection is how this test would
        pass by measuring nothing, and those two are the ones no sweep makes -
        a step and a hollow shell, each about half its own box.
        """
        square = [
            name
            for name, record in artifacts.items()
            if record["status"] == "meshed"
            and record.get("measured", {}).get("square")
            and record["measured"]["solids"]
        ]
        for name in ("boolean_to_a_false_prism", "thickness"):
            assert name in square, (
                f"{name!r} is bounded by planes square to the grid and is not among "
                f"{square}, so what follows is not being asked of the shapes it is for"
            )
        for name in square:
            record = _record(artifacts, name)
            assert not any(piece["faces"] for piece in record["pieces"]), (
                f"{name!r} is bounded by planes square to the grid and came back "
                "triangulated, so its box is what anything reads a width off"
            )
            held = sum(
                math.prod(piece["upper"][dim] - piece["lower"][dim] for dim in range(3))
                for piece in record["pieces"]
            )
            assert held == pytest.approx(
                record["measured"]["volume"], rel=CUT_TOLERANCE, abs=0.0
            ), (
                f"{name!r} was cut into pieces holding {held:g} where the shape "
                f"encloses {record['measured']['volume']:g}"
            )

    def test_a_via_between_two_pads_is_cut_into_the_blocks_it_was_drawn_as(self, artifacts):
        """The volume identity above cannot see this one: a pad cut into strips
        at the via's walls holds the drawing's volume exactly, every strip of it
        being metal that was drawn.
        """
        record = _record(artifacts, "boolean_to_a_stack")
        boxes = sorted((tuple(piece["lower"]), tuple(piece["upper"])) for piece in record["pieces"])
        assert boxes == [
            ((0.0, 0.0, 0.0), (4.0, 4.0, 1.0)),
            ((0.0, 0.0, 2.0), (4.0, 4.0, 3.0)),
            ((1.0, 1.0, 1.0), (3.0, 3.0, 2.0)),
        ]

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


class TestAJoinAsksTheSameHoweverThePartWasTurned:
    """The plane rule, scored on faces the kernel built rather than on stubs.

    A join's demand is stated over the whole plane of directions square to its
    edge: no direction across the edge runs wider than the edge size through a
    cell there. The stub-kernel tests hold that at every dihedral and every
    orientation, but they build the pair of normals themselves, so they cannot
    say that a real kernel's faces come back with the same thing.

    The blade is where it matters. The direction between two faces that have
    closed on each other is where the field escapes, and it is the direction a
    demand per face left to whatever the rest of the grid gave it; the line the
    demand is stated across is the cross product of the two normals, which is
    worse conditioned the nearer the faces come.

    So the line is scored against the kernel's own answer for it, and the size
    against the edge size the drawing declared. Neither is read off the other.
    """

    #: The two drawings of one blade. Only how the part was put down differs.
    BLADES = ("blade", "blade_turned")

    #: How far a direction may sit from where it is claimed to be, as a distance
    #: between unit vectors. Far under any real disagreement about where an edge
    #: runs, which is a fraction of a radian.
    SAME_LINE = 1e-9

    def lines(self, record):
        """Every direction a demand on this shape was stated across, as units."""
        return [scaled(join["tangent"]) for join in record["joins"]]

    def widest(self, record):
        """The widest cell each demand across an edge leaves, along the plane it
        covers. An axis the edge runs along comes back unconstrained."""
        return [
            widest_across(
                [math.inf if size is None else size for size in join["cells"]], join["tangent"]
            )
            for join in record["joins"]
        ]

    def test_the_corpus_carries_a_join_closed_to_nearly_flat(self, artifacts):
        """The specimen's purpose, read off the kernel's own normals rather than
        declared. A corner is where a rule naming directions and a rule naming
        the plane nearly agree, so a blade that quietly opened into a corner
        would go on reading as coverage while measuring nothing new.
        """
        for name in self.BLADES:
            sharpest = _record(artifacts, name)["measured"]["sharpest_join"]
            assert sharpest is not None, f"{name!r} carries no join at all"
            assert sharpest > 180.0 - 2.0 * corpus.BLADE_DEGREES, (
                f"{name!r}'s faces disagree by {sharpest:.3f} degrees, which is a "
                "corner rather than a feather edge - so the end of the dihedral "
                "this specimen exists for is not in the corpus"
            )

    def test_the_shapes_that_say_they_run_out_to_a_tip_are_the_ones_that_do(self, artifacts):
        """A specimen declares whether it thins to nothing, and that declaration
        is scored against the kernel: the shapes that say they have a tip are the
        shapes whose faces close furthest, far enough apart that nothing is near
        the line. What the declaration decides is which of two round-trip
        comparisons a shape gets, so a flag that had drifted would ask the wrong
        one of a shape that could answer the other.
        """
        closing = {}
        for specimen in corpus.specimens():
            if specimen.name not in artifacts or artifacts[specimen.name]["status"] != "meshed":
                continue
            sharpest = _record(artifacts, specimen.name)["measured"]["sharpest_join"]
            if sharpest is not None:
                closing[specimen.name] = (specimen.tip, sharpest)
        tips = [angle for declared, angle in closing.values() if declared]
        rest = [angle for declared, angle in closing.values() if not declared]
        assert tips, "no specimen declares a tip, so the flag guards nothing"
        assert min(tips) > max(rest), (
            f"a shape that declares no tip closes further than one that does: {closing}"
        )
        assert min(tips) - max(rest) > SHARP_DEGREES, (
            f"the nearest thing to a tip that is not one closes to {max(rest):.2f} "
            f"degrees against {min(tips):.2f}, which is closer than the band "
            "separating a fillet from an edge - so which shapes are tips is "
            "being decided by the declaration rather than by the drawings"
        )

    def tip(self, record):
        """The drawn join whose faces have closed furthest on this shape."""
        return max(record["measured"]["joins_drawn"], key=lambda one: one["disagreement"])

    @staticmethod
    def on_the_edge(place, join):
        """Whether ``place`` lies on the drawn ``join``, away from either end.

        A demand is placed at a point the kernel gave for a parameter strictly
        inside its own edge, so this attributes it to one join and never to the
        two an end is shared with. Both blades are straight-edged, which is what
        lets an edge be a segment here.
        """
        first, second = join["ends"]
        run = [b - a for a, b in zip(first, second)]
        length = math.dist(first, second)
        along = sum((p - a) * r for p, a, r in zip(place, first, run)) / (length * length)
        if not 1e-6 < along < 1.0 - 1e-6:
            return False
        return math.dist(place, [a + along * r for a, r in zip(first, run)]) < 1e-6

    def test_the_feather_edge_is_itself_asked_about(self, artifacts):
        """The join the specimens were drawn for, named and found.

        Everything else in this class is satisfied by the ordinary corners: the
        blade runs edges parallel, so the tip's demand asks along a line the
        corners also ask along and for the same cells. A rule that stopped
        emitting at the sharp end would take that demand away and leave every
        other assertion here standing on what the corners contribute.

        So the tip is picked out by the kernel's own reading of how far its faces
        have closed, and what is asserted is that a demand was made on that edge,
        at a place lying along it.
        """
        for name in self.BLADES:
            record = _record(artifacts, name)
            tip = self.tip(record)
            asked = [join for join in record["joins"] if self.on_the_edge(join["place"], tip)]
            assert asked, (
                f"{name!r} closes to {tip['disagreement']:.3f} degrees along the edge "
                f"between {tip['ends'][0]} and {tip['ends'][1]}, and no demand was "
                "made anywhere on it - the sharpest join on the shape asks for nothing"
            )
            for join in asked:
                stated, line = scaled(join["tangent"]), scaled(tip["line"])
                apart = min(math.dist(stated, line), math.dist(stated, [-c for c in line]))
                assert apart < self.SAME_LINE, (
                    f"a demand at {join['place']} on {name!r}'s feather edge is "
                    f"stated across {stated}, and the edge runs along {line}"
                )

    def test_a_demand_per_face_would_have_left_that_edge_coarser(self, artifacts):
        """Why the rule is stated over the plane, measured on the drawing.

        From the tip's own two normals: what a demand per face would have left
        along the way out of the tip, against what the plane demand leaves there.
        The pair is built here from the kernel's normals and the same criterion a
        gap is held to.

        It is worst where the faces have closed onto a coordinate plane rather
        than where the part is turned, each per-face demand then holding one axis
        and letting the others go. So the square blade is the bad case and the
        turned one is milder, and the plane demand answers the edge size for both.
        """
        for name in self.BLADES:
            record = _record(artifacts, name)
            tip = self.tip(record)
            pair = [separation(corpus.EDGE_SIZE, normal) for normal in tip["normals"]]
            left = widest_across(
                [min(sizes[dim] for sizes in pair) for dim in range(DIMENSIONS)], tip["line"]
            )
            assert left > corpus.EDGE_SIZE * (1.0 + 1e-9), (
                f"a demand per face would have left {name!r}'s feather edge at "
                f"{left:.6f} across, which is inside the edge size - so this "
                "specimen does not separate the rule from the one it replaced"
            )

    def test_the_line_a_demand_is_stated_across_is_the_line_the_edge_runs_along(self, artifacts):
        """The cross product of the two normals against the curve's own tangent.

        As the faces close, the product they are crossed into shrinks with the
        sine of the angle between them, so on a feather edge the line is computed
        from a very small quantity. Everything else about the demand is
        arithmetic on top of that line and would agree with itself whatever line
        it was handed.

        Both drawings are straight-edged throughout, so the kernel's tangent at
        one place on an edge is that edge's direction everywhere on it.
        """
        for name in self.BLADES:
            record = _record(artifacts, name)
            drawn = [scaled(one["line"]) for one in record["measured"]["joins_drawn"]]
            assert drawn, f"{name!r} carries no join the kernel would name a line for"
            for stated in self.lines(record):
                apart = min(
                    min(math.dist(stated, line), math.dist(stated, [-c for c in line]))
                    for line in drawn
                )
                assert apart < self.SAME_LINE, (
                    f"a demand on {name!r} is stated across {stated}, which is no "
                    f"edge of the drawing - the nearest runs {apart:.3g} away"
                )

    def test_one_of_them_lies_on_an_axis_and_the_other_on_none(self, artifacts):
        """What makes the pair a comparison. Both drawings are the same blade,
        and if both had come to rest the same way round the answers below would
        agree for a reason that says nothing about the rule.

        On an axis means two of the three components gone, not one: an edge with
        one component gone lies in a coordinate plane, which the square blade's
        hypotenuse does while running along no axis at all.
        """
        square, turned = (self.lines(_record(artifacts, name)) for name in self.BLADES)
        assert any(sum(1 for c in line if abs(c) < 1e-9) == 2 for line in square), (
            f"no edge of the square blade runs along an axis: {square}"
        )
        assert all(min(abs(c) for c in line) > 1e-3 for line in turned), (
            f"an edge of the turned blade still lies in a coordinate plane: {turned}"
        )

    def test_the_turned_blade_carries_the_square_one_s_edges_turned(self, artifacts):
        """One shape at two orientations, and the drawing says which turning.

        Without this the pair is two blades that happen to look alike, and every
        comparison below could be satisfied by a kernel that had lost an edge on
        one of them. The rotation is the test's own, from the figures the corpus
        declares the specimen with.
        """
        square, turned = (self.lines(_record(artifacts, name)) for name in self.BLADES)
        assert len(square) == len(turned), (
            f"the blades carry different numbers of edge demands, {len(square)} "
            f"against {len(turned)}, so they are not one shape drawn twice"
        )
        for line in square:
            want = scaled(rotated(line, corpus.BLADE_TURN_ABOUT, corpus.BLADE_TURN_DEGREES))
            apart = min(
                min(math.dist(want, other), math.dist(want, [-c for c in other]))
                for other in turned
            )
            assert apart < self.SAME_LINE, (
                f"the square blade's edge along {line} has no counterpart in the "
                f"turned one; the nearest runs {apart:.3g} away"
            )

    def test_every_demand_across_an_edge_spends_the_edge_size(self, artifacts):
        """The criterion itself, on kernel-built faces and against the size the
        drawing declared. Equality and not a bound: a demand that asked for less
        would refine directions nothing is singular along, and one that asked for
        more would leave the way out of the tip to whatever the rest of the grid
        gave it.

        The line it is spent across is the previous test's business. What is here
        is that the spending is right, and that a demand is stated across a plane
        at all - a rule naming one direction has no line to sweep around and
        would arrive as a shape asking nothing across any edge.
        """
        for name in self.BLADES:
            widest = self.widest(_record(artifacts, name))
            assert widest, f"{name!r} asks nothing across any edge"
            for spent in widest:
                assert spent == pytest.approx(corpus.EDGE_SIZE, rel=1e-3, abs=0.0), (
                    f"a demand on {name!r} leaves {spent:.6f} across its edge, "
                    f"where the metal edge size is {corpus.EDGE_SIZE}"
                )

    def test_and_the_axes_are_left_differently(self, artifacts):
        """Which is what says the turning reached the measurement at all.

        The two blades must not agree about what each axis is left: turning the
        part turns the plane the demand covers, so one statement lands on the
        axes differently. A pair that agreed here would be a pair the rotation
        never reached, and everything above would be one drawing measured twice.
        """
        finest = [
            [
                min(
                    join["cells"][dim]
                    for join in _record(artifacts, name)["joins"]
                    if join["cells"][dim] is not None
                )
                for dim in range(DIMENSIONS)
            ]
            for name in self.BLADES
        ]
        assert finest[0] != pytest.approx(finest[1], rel=1e-3, abs=0.0), (
            "the two drawings leave every axis the same cell, so the turning "
            f"reached nothing and what the pair agrees about is not a measurement: {finest}"
        )
