# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Which faces the grid holds, over shapes the real CAD kernel made.

Where a conductor's boundary lands is decided by two answers that have to be the
same one: which surfaces are flat and square to an axis, so the mesher pins a
line to each, and which are grown half a cell before the engine samples them.
Pin without holding and the face stands half a cell out; hold without pinning and
the wall lands wherever the grading left a line. Both come from
``staircase._surfaces``, and this is where that is checked against drawings
nobody here tessellated.

It runs on the kernel's own triangulations because the two cases that matter are
its: a tangent join, where a fillet runs into the face it is blended into at no
angle at all, and how a curve's facets fall either side of a tangency. A
hand-written triangulation can only model both.
"""

from __future__ import annotations

import numpy as np
import pytest

from Microwave.Solvers.openems import staircase
from tests import corpus
from tests.conftest import corpus_record as _record

pytestmark = pytest.mark.slow

#: Specimens whose flat faces all meet their neighbours at a corner, so the width
#: test has nothing to act on and must not reach them. Named rather than
#: discovered, since what they are for is to fail if the rule ever starts
#: deciding shapes it does not decide today.
BOUNDED_BY_CORNERS = (
    "box_rotated",
    "wedge",
    "boolean_fuse",
    "chamfer",
    "cone",
    "loft",
    "step_round_trip",
    "tilted_skin",
)


def _pieces(record):
    return [piece for piece in record["pieces"] if piece.get("faces")]


def _planes(record):
    return tuple(
        plane for piece in _pieces(record) for plane in staircase.flat_planes(*_surface(piece))
    )


def _surface(piece):
    return piece["vertices"], piece["faces"]


def _drawn(piece, axis):
    """The two extremes of a piece on one axis, as the drawing left them."""
    points = np.asarray(piece["vertices"], dtype=float)
    return float(points[:, axis].min()), float(points[:, axis].max())


class TestAFilletDoesNotHideTheFaceItIsBlendedInto:
    """A filleted box is ordinary CAD and every one of its faces is square to an
    axis. The fillet meets each of them at no angle at all, so what recognises
    them cannot be the angle."""

    def test_all_six_faces_are_offered_where_they_were_drawn(self, artifacts):
        record = _record(artifacts, "fillet")
        piece = _pieces(record)[0]
        wanted = sorted((axis, at) for axis in range(3) for at in _drawn(piece, axis))
        offered = staircase.flat_planes(*_surface(piece))
        assert [axis for axis, _ in offered] == [axis for axis, _ in wanted]
        assert [at for _, at in offered] == pytest.approx(
            [at for _, at in wanted], rel=0.0, abs=1e-9
        )

    def test_and_each_of_them_moves_by_its_clearance_and_no_more(self, artifacts):
        """The other half of the same rule. A plane asked for and then grown
        anyway is the face half a cell out, which is worse than either."""
        record = _record(artifacts, "fillet")
        piece = _pieces(record)[0]
        points = np.asarray(piece["vertices"], dtype=float)
        cell = 0.25
        grid = [
            np.arange(points[:, dim].min() - 2 * cell, points[:, dim].max() + 2 * cell, cell)
            for dim in range(3)
        ]
        step = np.asarray(staircase.grown(*_surface(piece), grid)) - points
        for axis, at in staircase.flat_planes(*_surface(piece)):
            on = np.isclose(points[:, axis], at, atol=1e-9)
            assert on.sum() > 1
            outward = 1.0 if at > points[:, axis].mean() else -1.0
            assert step[on, axis] * outward == pytest.approx(
                staircase.PINNED_CLEARANCE * cell, rel=1e-9, abs=0.0
            )

    def test_while_the_fillet_itself_still_grows_by_half_a_cell(self, artifacts):
        """The companion, so the two above cannot pass by holding everything."""
        record = _record(artifacts, "fillet")
        piece = _pieces(record)[0]
        points = np.asarray(piece["vertices"], dtype=float)
        cell = 0.25
        grid = [
            np.arange(points[:, dim].min() - 2 * cell, points[:, dim].max() + 2 * cell, cell)
            for dim in range(3)
        ]
        step = np.asarray(staircase.grown(*_surface(piece), grid)) - points
        held = np.zeros(len(points), dtype=bool)
        for axis, at in staircase.flat_planes(*_surface(piece)):
            held |= np.isclose(points[:, axis], at, atol=1e-9)
        assert np.linalg.norm(step[~held], axis=1) == pytest.approx(
            staircase.GROWN_BY * cell, rel=1e-9, abs=0.0
        )


class TestACurveIsNotGivenAFaceItDoesNotHave:
    """A tessellated curve lands rows in one plane wherever the phase puts them
    either side of a tangency - along a cylinder's length, right round a torus'
    equator. Each such plane is a chord standing a fraction of a facet inside the
    surface it claims to be, and an anchor there is refused outright when another
    lands within the cell floor of it."""

    @pytest.mark.parametrize(
        "specimen,axis,offered",
        [
            ("cylinder", 2, 2),
            ("torus", 2, 0),
            ("pipe", 2, 2),
            ("substrate_rolled", 2, 2),
            ("sphere", 2, 0),
        ],
    )
    def test_only_the_faces_it_has_are_offered(self, artifacts, specimen, axis, offered):
        record = _record(artifacts, specimen)
        planes = _planes(record)
        assert len(planes) == offered
        assert all(held == axis for held, _ in planes)

    def test_the_width_a_run_is_judged_by_is_the_width_it_has(self, artifacts):
        """A torus' tangent band is an annulus that reaches right across the
        shape, so any width read off a span answers about the reach and not about
        the band. Held to the annulus the drawing has."""
        record = _record(artifacts, "torus")
        piece = _pieces(record)[0]
        points = np.asarray(piece["vertices"], dtype=float)
        corners = np.asarray(piece["faces"], dtype=int)
        normals = staircase._face_normals(points[corners])
        square = staircase._square_axes(normals / np.linalg.norm(normals, axis=1, keepdims=True))
        band = [
            face
            for face in np.where(square == 2)[0]
            if points[corners[face]][:, 2].mean() > points[:, 2].mean()
        ]
        assert band
        radius = np.linalg.norm(points[corners[band]].reshape(-1, 3)[:, :2], axis=1)
        # Not to the last digit: the band's two rims are polygons rather than
        # circles, so its area and its perimeter each fall a little short of the
        # annulus they approximate.
        assert staircase._width(points, corners, band) == pytest.approx(
            float(radius.max() - radius.min()), rel=1e-3, abs=0.0
        )


class TestTheWidthTestDecidesOnlyWhatItIsFor:
    """Its one question is whether a flat strip lying in the middle of a curve is
    a face of its own. A face bounded by corners is never asked, and most of the
    corpus is such a face - so what protects those is that the rule cannot reach
    them, and nothing else would notice if that changed."""

    @pytest.mark.parametrize("specimen", BOUNDED_BY_CORNERS)
    def test_a_face_bounded_by_corners_does_not_depend_on_the_width_at_all(
        self, artifacts, specimen, monkeypatch
    ):
        record = _record(artifacts, specimen)
        settled = _planes(record)
        assert settled
        for demanded in (0.0, 1e9):
            monkeypatch.setattr(staircase, "FACE_IS_WIDER_BY", demanded)
            assert _planes(record) == settled, (
                f"{specimen}'s faces moved when the width test was set to "
                f"{demanded}, so it is deciding them"
            )

    def test_and_no_run_it_does_decide_lies_anywhere_near_the_constant(self, artifacts):
        """The evidence for the number, rather than the number itself.

        Every run the test can act on is either a chord of the curve it lies in
        or a face several times wider than one, and the constant sits in the
        empty ground between the two populations.
        """
        folded, kept = [], []
        for name in corpus.specimens():
            record = artifacts.get(name.name)
            if record is None or record["status"] != "meshed":
                continue
            for piece in _pieces(record):
                for ratio in _judged(piece):
                    (kept if ratio > staircase.FACE_IS_WIDER_BY else folded).append(ratio)
        assert folded and kept
        assert max(folded) < staircase.FACE_IS_WIDER_BY / 1.5
        assert min(kept) > staircase.FACE_IS_WIDER_BY * 4.0


class TestTheWindingDecidesNothingButTheDirection:
    """``flat_planes`` and ``grown`` read one triangulation two ways.

    ``grown`` steps along the outward normal, so it winds the set first;
    ``flat_planes`` reduces the direction to an axis and reads the triangulation
    as it arrived. Both docstrings say they hold the same faces, and that rests
    on every test between the grouping and the axis being even in the sign. A
    comment cannot fail when one of them stops being even. This can.
    """

    def test_the_grouping_is_the_same_wound_either_way(self, artifacts):
        judged = 0
        for piece in _every_piece(artifacts):
            points, corners, normals, _ = piece
            drawn = staircase._surfaces(points, corners, normals)
            wound = staircase._surfaces(points, corners, -normals)
            judged += len(drawn)

            assert {frozenset(members) for members in drawn.values()} == {
                frozenset(members) for members in wound.values()
            }
        assert judged

    def test_each_surface_keeps_its_axis_and_reverses_its_direction(self, artifacts):
        held = 0
        for points, corners, normals, areas in _every_piece(artifacts):
            for members in staircase._surfaces(points, corners, normals).values():
                drawn = staircase._pinned_normal(members, normals, areas)
                wound = staircase._pinned_normal(members, -normals, areas)

                assert (drawn is None) == (wound is None)
                if drawn is not None:
                    held += 1
                    assert np.array_equal(drawn, -wound)
        assert held


def _every_piece(artifacts):
    """Each meshed piece in the corpus, as the arrays the rules are read on."""
    for name in corpus.specimens():
        record = artifacts.get(name.name)
        if record is None or record["status"] != "meshed":
            continue
        for piece in _pieces(record):
            points = np.asarray(piece["vertices"], dtype=float)
            corners = np.asarray(piece["faces"], dtype=int)
            normals = staircase._face_normals(points[corners])
            yield points, corners, normals, np.linalg.norm(normals, axis=1)


def _judged(piece):
    """How wide each run the width test reaches is, against what it runs into.

    Read out of the rule itself rather than recomputed here: a copy of the
    decision would go on reporting the ground the threshold sits in after the
    rule underneath it moved.
    """
    points = np.asarray(piece["vertices"], dtype=float)
    corners = np.asarray(piece["faces"], dtype=int)
    reached: list = []
    staircase._surfaces(points, corners, staircase._face_normals(points[corners]), judged=reached)
    return [ratio for _, ratio in reached]
