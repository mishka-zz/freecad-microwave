# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A size asked for at a place, laid by Gmsh and read back off the elements.

What is refused before Gmsh is asked is ``tests/test_gmsh_mesher.py``'s. Here
the drawings are boxes and rectangles Gmsh draws itself and writes out, since
the kernel a shape was drawn in decides nothing about how an element is sized.

Needs Gmsh and skips itself without one.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import replace

import numpy
import pytest

from tests.conftest import needed

gmsh = needed(
    "gmsh", "gmsh is not on this interpreter, so the mesher is unreachable", module_level=True
)

from Microwave.Gmsh import labels as label_map  # noqa: E402
from Microwave.Gmsh.mesh import TURNED, _marking, mesh  # noqa: E402
from Microwave.Gmsh.places import (  # noqa: E402
    OTHER_SOURCES,
    _classes,
    _figures,
    _length,
    _sampling,
)
from Microwave.Gmsh.vocabulary import (  # noqa: E402
    AtRim,
    Demand,
    Mark,
    Near,
    Piece,
    Profile,
    Refused,
    Unmeshed,
    Within,
)
from Microwave.Solvers import gmsh_mesh_report  # noqa: E402

#: Straight-sided volumes: nothing here is about the shape of an element.
VOLUME = Profile(top=3, element_order=1)

#: The air a place stands in, a long box, and the bulk size over it. The box is
#: long so that one end of it stands further from the sheet than the growth
#: below reaches.
LENGTH, SIDE = 60.0, 20.0
BULK = 5.0

#: The sheet a rim is asked of, across the near end of the air, in the plane
#: halfway up.
SHEET = ((5.0, 5.0), (15.0, 15.0))
HEIGHT = SIDE / 2.0

#: A size a tenth of the bulk, and the growth back to the bulk from it.
FINE = BULK / 10.0
GROWTH = 1.5

#: Where the elements are counted as far from the sheet: past the end of the
#: sheet by more than the size takes to grow back to the bulk.
FAR = SHEET[1][0] + (BULK - FINE) / (GROWTH - 1.0)


def _drawn(directory: pathlib.Path, name: str, build) -> str:
    """Draw one shape in a Gmsh model of its own and write it where a caller would."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        path = str(directory / f"{name}.brep")
        gmsh.write(path)
        return path
    finally:
        gmsh.finalize()


def air_only(directory: pathlib.Path) -> str:
    return _drawn(directory, "air", lambda occ: occ.addBox(0, 0, 0, LENGTH, SIDE, SIDE))


def air_and_sheet(directory: pathlib.Path) -> list[Piece]:
    air = _drawn(directory, "air", lambda occ: occ.addBox(0, 0, 0, LENGTH, SIDE, SIDE))
    (x0, y0), (x1, y1) = SHEET
    sheet = _drawn(
        directory, "sheet", lambda occ: occ.addRectangle(x0, y0, HEIGHT, x1 - x0, y1 - y0)
    )
    return [Piece("air", 3, air), Piece("sheet", 2, sheet)]


#: What is left where the air ends.
WALLS = "walls"


#: Where ``tests/drawings_probe.py`` wrote the drawings, under a real FreeCAD.
DRAWN = pathlib.Path(__file__).resolve().parent / "_drawings"


def meshed(directory, places=(), growth=GROWTH, name="model", pieces=None):
    return mesh(
        pieces if pieces is not None else air_and_sheet(directory),
        Demand(coarsest=BULK, finest=0.0, growth=growth, places=tuple(places)),
        VOLUME,
        str(directory),
        name,
        remainder=WALLS,
    )


def tetrahedra(path: str) -> numpy.ndarray:
    """The corners of every tetrahedron the file holds, as an array of four points each."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(path)
        tags, flat, _ = gmsh.model.mesh.getNodes()
        index = numpy.zeros(int(tags.max()) + 1, dtype=numpy.int64)
        index[tags.astype(numpy.int64)] = numpy.arange(len(tags))
        at = numpy.asarray(flat).reshape(-1, 3)
        _, nodes = gmsh.model.mesh.getElementsByType(
            gmsh.model.mesh.getElementType("Tetrahedron", 1)
        )
        return at[index[numpy.asarray(nodes, dtype=numpy.int64).reshape(-1, 4)]]
    finally:
        gmsh.finalize()


def polygon(occ, x: float, y: float, z: float, radius: float, sides: int) -> None:
    """A regular polygon across the plane at ``z``, drawn as one face of many short
    sides."""
    corners = [
        occ.addPoint(
            x + radius * math.cos(2 * math.pi * at / sides),
            y + radius * math.sin(2 * math.pi * at / sides),
            z,
        )
        for at in range(sides)
    ]
    lines = [occ.addLine(corners[at], corners[(at + 1) % sides]) for at in range(sides)]
    occ.addPlaneSurface([occ.addCurveLoop(lines)])


#: Where the poles of an uneven spline stand along the side it is drawn as: most
#: of them over the first hundredth of it, so its parameter covers hundreds of
#: times more length per unit at the far end than at the near one.
CLUSTERED = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 12.0, 24.0)


def sheet_with_a_spline_side(occ, poles) -> None:
    """A rectangle whose long side is a B-spline through ``poles`` along a straight
    line, and whose other three sides are lines.

    The side is straight whatever the poles are. What the poles decide is how
    fast the parameter of the spline runs along it.
    """
    (x0, y0), (_, y1) = SHEET
    along = [occ.addPoint(x0 + at, y0, HEIGHT) for at in poles]
    side = occ.addBSpline(along, degree=3)
    far = occ.addPoint(x0 + poles[-1], y1, HEIGHT)
    near = occ.addPoint(x0, y1, HEIGHT)
    loop = occ.addCurveLoop(
        [
            side,
            occ.addLine(along[-1], far),
            occ.addLine(far, near),
            occ.addLine(near, along[0]),
        ]
    )
    occ.addPlaneSurface([loop])
    occ.remove([(0, tag) for tag in along[1:-1]])


def mean_edges(corners: numpy.ndarray) -> numpy.ndarray:
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return numpy.mean(
        [numpy.linalg.norm(corners[:, i] - corners[:, j], axis=1) for i, j in pairs], axis=0
    )


class TestASizeAtARim:
    def test_the_elements_along_the_rim_follow_the_size_asked(self, tmp_path):
        """The figure set against the size asked is the size along the rim, and
        it moves with the ask: a rim asked for at nearly the bulk is laid at
        nearly the bulk, and one asked for at a tenth of it at a tenth."""
        fine = meshed(tmp_path, [AtRim("rim", "sheet", FINE)], name="fine").reached["rim"]
        coarse = meshed(tmp_path, [AtRim("rim", "sheet", 0.9 * BULK)], name="coarse").reached["rim"]
        assert fine.asked == FINE
        assert fine.reached is not None and coarse.reached is not None
        assert fine.reached < 2.0 * FINE
        assert fine.reached < coarse.reached / 4.0

    def test_the_size_holds_along_the_whole_of_a_long_rim(self, tmp_path):
        """A distance is measured from points laid along the rim, and a rim many
        times longer than its size needs as many of them, or the size between
        two is the size a step further out."""
        air = _drawn(tmp_path, "air", lambda occ: occ.addBox(0, 0, 0, LENGTH, SIDE, SIDE))
        strip = _drawn(tmp_path, "strip", lambda occ: occ.addRectangle(5.0, 9.0, HEIGHT, 50.0, 2.0))
        got = meshed(
            tmp_path,
            [AtRim("rim", "strip", FINE)],
            growth=3.0,
            pieces=[Piece("air", 3, air), Piece("strip", 2, strip)],
        )
        reached = got.reached["rim"].reached
        assert reached is not None and reached < 1.5 * FINE

    def test_a_rim_no_longer_than_its_size_is_laid_as_one_a_little_longer_is(self, tmp_path):
        """A sheet whose every side is as long as the size asked of its rim is
        refined round it as the same rim asked a hundredth finer is, rather than
        left at the bulk."""
        side = 1.0
        air = _drawn(tmp_path, "air", lambda occ: occ.addBox(0, 0, 0, SIDE, SIDE, SIDE))
        speck = _drawn(
            tmp_path,
            "speck",
            lambda occ: occ.addRectangle(HEIGHT, HEIGHT, HEIGHT, side, side),
        )
        pieces = [Piece("air", 3, air), Piece("speck", 2, speck)]
        standing = [
            meshed(tmp_path, [AtRim("rim", "speck", size)], name=f"s{size:g}", pieces=pieces)
            .reached["rim"]
            .standing
            for size in (side, 0.99 * side)
        ]
        assert None not in standing
        at_side, finer = standing
        assert at_side < 1.25 * finer

    def test_a_rim_is_laid_the_same_however_its_parameter_runs_along_it(self, tmp_path):
        """The same straight side, drawn as a spline whose parameter covers far
        more length per unit at one end than at the other, and drawn as one whose
        parameter runs evenly. Gmsh measures a distance from points laid at even
        steps of the parameter, so the uneven one needs more of them to hold the
        same spacing along its length."""
        air = _drawn(tmp_path, "air", lambda occ: occ.addBox(0, 0, 0, LENGTH, SIDE, SIDE))
        reached = []
        for name, poles in (("even", (0.0, 8.0, 16.0, 24.0)), ("uneven", CLUSTERED)):
            sheet = _drawn(
                tmp_path, name, lambda occ, poles=poles: sheet_with_a_spline_side(occ, poles)
            )
            got = meshed(
                tmp_path,
                [AtRim("rim", "sheet", FINE)],
                growth=3.0,
                name=f"spline_{name}",
                pieces=[Piece("air", 3, air), Piece("sheet", 2, sheet)],
            )
            reached.append(got.reached["rim"].reached)
        even, uneven = reached
        assert even is not None and uneven is not None
        assert uneven < 1.25 * even

    def test_the_elements_standing_on_the_rim_are_counted(self, tmp_path):
        """A finer rim has more elements standing on it."""
        fine = meshed(tmp_path, [AtRim("rim", "sheet", FINE)], name="fine").reached["rim"]
        coarse = meshed(tmp_path, [AtRim("rim", "sheet", 2 * FINE)], name="coarse").reached["rim"]
        assert fine.standing is not None
        assert fine.elements > coarse.elements > 0

    def test_the_size_grows_away_from_the_rim_at_the_growth_asked(self, tmp_path):
        """A step past the end of the sheet, a steep growth has already reached
        elements several times coarser than a gentle one has."""
        pieces = air_and_sheet(tmp_path)
        past = []
        for growth in (1.2, 3.0):
            got = meshed(
                tmp_path,
                [AtRim("rim", "sheet", FINE)],
                growth=growth,
                name=f"g{growth:g}",
                pieces=pieces,
            )
            corners = tetrahedra(got.path)
            along = corners.mean(axis=1)[:, 0] - SHEET[1][0]
            past.append(numpy.median(mean_edges(corners[(along > 1.0) & (along < 3.0)])))
        gentle, steep = past
        assert steep > 2.0 * gentle

    def test_nothing_far_from_a_place_is_refined(self, tmp_path):
        """Past the distance the size takes to grow back, the mesh is the one the
        bulk gives."""
        plain = tetrahedra(meshed(tmp_path, name="plain").path)
        refined = tetrahedra(meshed(tmp_path, [AtRim("rim", "sheet", FINE)], name="rim").path)
        far = [
            mean_edges(corners[corners.mean(axis=1)[:, 0] > FAR]) for corners in (plain, refined)
        ]
        assert all(len(edges) for edges in far), "nothing stands far from the sheet"
        assert numpy.median(far[1]) == pytest.approx(numpy.median(far[0]), rel=0.1)

    def test_the_size_near_the_sheet_is_finer_than_at_its_rim_alone(self, tmp_path):
        """A size near a face sizes the whole face, where one at its rim leaves the
        middle of it to grow."""
        near = meshed(tmp_path, [Near("near", "sheet", FINE)], name="near")
        rim = meshed(tmp_path, [AtRim("rim", "sheet", FINE)], name="rim")
        assert near.labels["sheet"].edges.longest < rim.labels["sheet"].edges.longest

    def test_a_face_cut_in_two_keeps_the_rim_it_was_drawn_with(self):
        """Two rectangles meeting along a side are one face drawn twice, and the
        side they share is inside it."""
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("halves")
            occ = gmsh.model.occ
            one = occ.addRectangle(0, 0, 0, 3, 2)
            other = occ.addRectangle(3, 0, 0, 4, 2)
            _, parts = occ.fragment([(2, one)], [(2, other)])
            occ.synchronize()
            faces = [tag for part in parts for _, tag in part]
            rim = label_map.rim(2, faces)
            length = sum(occ.getMass(1, curve) for curve in rim)
            curves = {curve for _, curve in gmsh.model.getEntities(1)}
            shared = {curve for curve in curves if occ.getCenterOfMass(1, curve)[0] == 3.0}
            alone = [label_map.rim(2, [face]) for face in faces]
        finally:
            gmsh.finalize()
        assert length == pytest.approx(2 * (7 + 2), rel=1e-9)
        assert len(shared) == 1
        assert set(rim) == curves - shared
        assert all(shared <= set(one) for one in alone)

    def test_a_label_closed_on_itself_has_no_rim_and_lays_nothing(self, tmp_path):
        """What is left where a box ends is a closed surface, round which no curve
        runs. The mesh is the one of no place, and the answer says nothing was
        laid."""
        pieces = air_and_sheet(tmp_path)
        plain = meshed(tmp_path, name="plain", pieces=pieces)
        closed = meshed(tmp_path, [AtRim("shell", WALLS, FINE)], name="closed", pieces=pieces)
        assert pathlib.Path(closed.path).read_bytes() == pathlib.Path(plain.path).read_bytes()
        shell = closed.reached["shell"]
        assert (shell.laid, shell.elements, shell.reached, shell.standing) == (False, 0, None, None)
        opened = meshed(tmp_path, [AtRim("rim", "sheet", FINE)], name="open", pieces=pieces)
        assert opened.reached["rim"].laid

    def test_a_fold_is_inside_a_label_and_not_on_its_rim(self):
        """Two faces meeting along a curve at an angle share it, as two halves of
        one flat face do."""
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("fold")
            occ = gmsh.model.occ
            floor = occ.addRectangle(0, 0, 0, 3, 2)
            wall = occ.addRectangle(0, 0, 0, 3, 2)
            occ.rotate([(2, wall)], 0, 0, 0, 1, 0, 0, 1.5707963267948966)
            _, parts = occ.fragment([(2, floor)], [(2, wall)])
            occ.synchronize()
            faces = [tag for part in parts for _, tag in part]
            rim = label_map.rim(2, faces)
            length = sum(occ.getMass(1, curve) for curve in rim)
        finally:
            gmsh.finalize()
        assert length == pytest.approx(2 * (2 + 2) + 2 * 3, rel=1e-9)


class TestASizeThroughoutALabel:
    def test_a_size_throughout_a_label_is_laid_in_it(self, tmp_path):
        """The figure set against it is the mean edge, and an element's longest
        edge is longer than its mean one."""
        got = meshed(tmp_path, [Within("air", "air", 2 * FINE)])
        reached = got.reached["air"]
        assert reached.reached is not None and reached.standing is not None
        assert FINE < reached.reached < 2 * (2 * FINE)
        assert reached.reached < reached.standing

    def test_a_size_throughout_a_body_reaches_its_faces(self, tmp_path):
        """The faces of a body standing in the air are meshed before its inside, and
        a size laid on the inside alone leaves the elements on them as coarse as
        the bulk."""
        air = _drawn(tmp_path, "air", lambda occ: occ.addBox(0, 0, 0, LENGTH, SIDE, SIDE))
        block = _drawn(tmp_path, "block", lambda occ: occ.addBox(30.0, 5.0, 5.0, 10.0, 10.0, 10.0))
        size = 2 * FINE
        got = meshed(
            tmp_path,
            [Within("body", "block", size)],
            pieces=[Piece("air", 3, air), Piece("block", 3, block, priority=1)],
        )
        assert got.labels["block"].edges.longest < 3.0 * size < BULK

    def test_a_body_grows_back_to_the_bulk_at_the_growth_asked(self, tmp_path):
        """A gentler growth lays a longer transition round the body, and so more
        elements outside it."""
        block = _drawn(tmp_path, "block", lambda occ: occ.addBox(30.0, 5.0, 5.0, 10.0, 10.0, 10.0))
        near = [Near("block", "block", 2 * FINE)]
        marks = [Mark("block", 3, block)]
        gentle = tetrahedra(marked(tmp_path, marks, near, growth=1.2, name="gentle").path)
        steep = tetrahedra(marked(tmp_path, marks, near, growth=3.0, name="steep").path)
        assert len(gentle) > len(steep)


def ring(occ, outer: float = 9.0, inner: float = 7.0) -> None:
    """A flat ring standing across the air, halfway up and round a point on its
    axis, drawn as one face with a hole."""
    x, y, z = RING_CENTRE
    loops = [occ.addCurveLoop([occ.addCircle(x, y, z, radius)]) for radius in (outer, inner)]
    occ.addPlaneSurface(loops)


#: Where the ring stands: its centre, far enough from every wall that the
#: ring's hole is air on every side.
RING_CENTRE = (20.0, SIDE / 2.0, SIDE / 2.0)


def marked(directory, marks, places, pieces=None, name="model", growth=GROWTH):
    return mesh(
        pieces if pieces is not None else [Piece("air", 3, air_only(directory))],
        Demand(coarsest=BULK, finest=0.0, growth=growth, places=tuple(places)),
        VOLUME,
        str(directory),
        name,
        remainder=WALLS,
        marks=marks,
    )


class TestASizeAtAMark:
    """A mark says where a place stands and claims nothing, so the size is laid
    at the shape it is and nowhere its bounding box adds."""

    def test_a_ring_is_sized_along_itself_and_its_hole_is_left_to_the_bulk(self, tmp_path):
        """The middle of the hole stands two sizes' growth from the ring and
        further; a box round the ring would lay the ring's size there."""
        drawn = _drawn(tmp_path, "ring", ring)
        got = marked(tmp_path, [Mark("ring", 2, drawn)], [Near("ring", "ring", FINE)])
        corners = tetrahedra(got.path)
        middle = numpy.linalg.norm(corners.mean(axis=1) - RING_CENTRE, axis=1) < FINE * 3
        assert middle.any()
        assert mean_edges(corners[middle]).min() > 3 * FINE
        along = got.reached["ring"]
        assert along.laid and along.along is not None
        # Every point of the ring stands within one size of a point the field
        # measures from, so the field there is at most the growth times the size.
        assert FINE <= along.along <= GROWTH * FINE
        assert along.along < along.reached

    def test_a_mark_drawn_as_a_sheet_is_changes_no_label(self, tmp_path):
        """It stands on the sheet's own entities, so the labels come back as
        they would without it, and a size asked at it is the mesh a size asked at
        the sheet is."""
        pieces = air_and_sheet(tmp_path)
        sheet = next(piece.file for piece in pieces if piece.label == "sheet")
        at_label = meshed(tmp_path, [Near("near", "sheet", FINE)], name="label", pieces=pieces)
        at_mark = marked(
            tmp_path,
            [Mark("again", 2, sheet)],
            [Near("near", "again", FINE)],
            pieces=pieces,
            name="mark",
        )
        # Importing the shape twice moves nodes by rounding, and nothing else.
        assert {name: replace(label, edges=None) for name, label in at_mark.labels.items()} == {
            name: replace(label, edges=None) for name, label in at_label.labels.items()
        }
        for name, label in at_mark.labels.items():
            assert label.edges.longest == pytest.approx(at_label.labels[name].edges.longest)
        assert at_mark.reached["near"].elements == at_label.reached["near"].elements
        assert at_mark.reached["near"].along == pytest.approx(at_label.reached["near"].along)
        assert len(tetrahedra(at_mark.path)) == len(tetrahedra(at_label.path))

    def test_a_body_is_sized_throughout_and_grows_away_from_its_faces(self, tmp_path):
        """A distance is measured from points, curves and surfaces, so near a body
        of the filled dimension the size is laid in it and grows from its rim."""
        block = _drawn(tmp_path, "block", lambda occ: occ.addBox(30.0, 5.0, 5.0, 10.0, 10.0, 10.0))
        near = marked(tmp_path, [Mark("block", 3, block)], [Near("near", "block", 2 * FINE)])
        within = marked(
            tmp_path, [Mark("block", 3, block)], [Within("in", "block", 2 * FINE)], name="within"
        )
        inside, stated = near.reached["near"], within.reached["in"]
        assert inside.elements == stated.elements > 0
        assert inside.reached == pytest.approx(stated.reached, rel=0.1)
        assert len(tetrahedra(near.path)) > len(tetrahedra(within.path))
        assert near.labels["air"].size == pytest.approx(LENGTH * SIDE * SIDE)

    def test_a_mark_reaching_past_the_model_marks_what_is_inside_it(self, tmp_path):
        """What it became where no label was drawn goes before anything is meshed."""
        across = _drawn(
            tmp_path, "across", lambda occ: occ.addRectangle(LENGTH - 10.0, 5.0, HEIGHT, 20.0, 10.0)
        )
        bare = marked(tmp_path, [], [], name="bare")
        got = marked(tmp_path, [Mark("across", 2, across)], [Near("near", "across", FINE)])
        assert got.labels.keys() == bare.labels.keys()
        assert got.labels["air"].size == pytest.approx(bare.labels["air"].size)
        assert got.labels[WALLS].size == pytest.approx(bare.labels[WALLS].size)
        corners = tetrahedra(got.path)
        assert corners[..., 0].max() == pytest.approx(LENGTH)
        assert got.reached["near"].laid and got.reached["near"].along < GROWTH * FINE

    def test_a_body_reaching_past_the_model_adds_nothing_to_it(self, tmp_path):
        """What only the mark became would be a region no label covers."""
        across = _drawn(
            tmp_path, "across", lambda occ: occ.addBox(LENGTH - 5.0, 5.0, 5.0, 10.0, 10.0, 10.0)
        )
        got = marked(tmp_path, [Mark("across", 3, across)], [Near("near", "across", FINE)])
        assert got.labels["air"].size == pytest.approx(LENGTH * SIDE * SIDE)
        assert got.reached["near"].elements > 0

    def test_what_a_mark_became_outside_the_model_is_taken_out(self, tmp_path):
        """A face standing past the model bounds nothing filled and stands in
        nothing, so no element would be meshed to it."""
        air = air_only(tmp_path)
        across = _drawn(
            tmp_path, "across", lambda occ: occ.addRectangle(LENGTH - 10.0, 5.0, HEIGHT, 20.0, 10.0)
        )
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("across")
            drawn, _ = label_map.load([Piece("air", 3, air), Mark("across", 2, across)])
            resolved, _ = label_map.fragment(drawn)
            assert len(resolved["across"]) == 2
            kept, marking = _marking(resolved, {"across"}, 3)
            assert kept.keys() == {"air"}
            (inside,) = [
                tag for dim, tag in marking["across"] if (dim, tag) in gmsh.model.getEntities(2)
            ]
            assert gmsh.model.getBoundingBox(2, inside)[3] == pytest.approx(LENGTH)
        finally:
            gmsh.finalize()

    def test_a_mark_wholly_outside_the_model_lays_nothing_and_says_so(self, tmp_path):
        beyond = _drawn(
            tmp_path, "beyond", lambda occ: occ.addRectangle(LENGTH + 10.0, 5.0, HEIGHT, 5.0, 5.0)
        )
        got = marked(tmp_path, [Mark("beyond", 2, beyond)], [Near("near", "beyond", FINE)])
        assert not got.reached["near"].laid
        assert got.reached["near"].elements == 0


class TestAMarkInsideABody:
    """A point or a curve standing inside a volume is not a node or a chain of
    edges the volume is meshed round: the elements there follow the size asked
    at it, as they follow it round a face."""

    def sized(self, tmp_path, dimension, build, size):
        drawn = _drawn(tmp_path, f"inside{dimension}", build)
        return marked(
            tmp_path,
            [Mark("inside", dimension, drawn)],
            [Near("near", "inside", size)],
            name=f"inside{dimension}_{size:g}",
        ).reached["near"]

    @staticmethod
    def point(occ):
        occ.addPoint(*RING_CENTRE)

    @staticmethod
    def line(occ):
        x, y, z = RING_CENTRE
        occ.addLine(occ.addPoint(x - 5.0, y, z), occ.addPoint(x + 5.0, y, z))

    def test_the_elements_holding_a_point_follow_the_size_asked_there(self, tmp_path):
        """Halving the size halves the elements round the point: the size is
        the field's and not the bulk's. A node the volume were meshed round
        would leave them at the bulk whatever was asked."""
        fine, coarse = (self.sized(tmp_path, 0, self.point, size) for size in (FINE, 2 * FINE))
        assert fine.laid and fine.elements > 0
        assert fine.reached < BULK / 2
        assert 1.5 < coarse.reached / fine.reached < 2.5

    def test_the_elements_along_a_curve_follow_the_size_asked_there(self, tmp_path):
        fine, coarse = (self.sized(tmp_path, 1, self.line, size) for size in (FINE, 2 * FINE))
        assert fine.laid and fine.elements > 0 and fine.along is None
        assert fine.standing < BULK / 2
        assert 1.5 < coarse.standing / fine.standing < 2.5

    def test_a_curve_a_label_holds_stays_one_the_volume_is_meshed_round(self, tmp_path):
        """A mark drawn on a label's own curve stands on the label's entity, and
        what the label holds is the label's to have meshed."""
        wire = _drawn(tmp_path, "wire", self.line)
        got = marked(
            tmp_path,
            [Mark("again", 1, wire)],
            [Near("near", "again", FINE)],
            pieces=[Piece("air", 3, air_only(tmp_path)), Piece("wire", 1, wire)],
            name="wired",
        )
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(got.path)
            (group,) = [tag for dim, tag in gmsh.model.getPhysicalGroups(1)]
            on_wire = set(gmsh.model.mesh.getNodesForPhysicalGroup(1, group)[0])
            _, _, corners = gmsh.model.mesh.getElements(3)
            assert on_wire <= set(numpy.concatenate(corners))
        finally:
            gmsh.finalize()

    def test_a_point_outside_the_model_lays_nothing(self, tmp_path):
        got = self.sized(tmp_path, 0, lambda occ: occ.addPoint(-5.0, 1.0, 1.0), FINE)
        assert not got.laid and got.elements == 0


class TestAFaceAMarkCuts:
    """A mark cuts the faces it crosses, and the label holding such a face holds
    every piece of it: what the label says about the face is what it said uncut."""

    FLOOR = (10.0, 5.0, 20.0, 10.0)

    def floored(self, directory, marks, name, halves=False):
        x, y, dx, dy = self.FLOOR
        floor = _drawn(directory, "floor", lambda occ: occ.addRectangle(x, y, 0.0, dx, dy))
        if halves:
            pieces = [
                Piece(
                    "air",
                    3,
                    _drawn(directory, "near", lambda occ: occ.addBox(0, 0, 0, 20, SIDE, SIDE)),
                ),
                Piece(
                    "slab",
                    3,
                    _drawn(
                        directory, "far", lambda occ: occ.addBox(20, 0, 0, LENGTH - 20, SIDE, SIDE)
                    ),
                ),
            ]
        else:
            pieces = [Piece("air", 3, air_only(directory))]
        return marked(
            directory,
            marks,
            [Near(mark.name, mark.name, FINE) for mark in marks],
            pieces=[*pieces, Piece("floor", 2, floor)],
            name=name,
        )

    def test_a_face_a_body_crosses_is_held_whole_and_stands_on_one_label(self, tmp_path):
        """A body across the face cuts it into pieces, and the pieces together
        are the face: its box, its area and the region on it are unchanged."""
        block = _drawn(tmp_path, "block", lambda occ: occ.addBox(15.0, 0.0, -1.0, 5.0, SIDE, 3.0))
        bare = self.floored(tmp_path, [], "bare").labels["floor"]
        cut = self.floored(tmp_path, [Mark("block", 3, block)], "cut").labels["floor"]
        assert len(cut.entities) > len(bare.entities) == 1
        assert cut.beside == bare.beside == ("air",)
        assert cut.size == pytest.approx(bare.size)
        assert cut.lower == pytest.approx(bare.lower) and cut.upper == pytest.approx(bare.upper)

    def test_a_face_two_regions_stand_on_names_both(self, tmp_path):
        bare = self.floored(tmp_path, [], "halves", halves=True).labels["floor"]
        assert bare.beside == ("air", "slab")

    def test_a_label_of_the_filled_dimension_names_nothing_beside_it(self, tmp_path):
        """A tag names an entity within its dimension only, so a volume and a
        face may carry one number, and here they do."""
        got = self.floored(tmp_path, [], "halves", halves=True).labels
        faces = {tag for label in got.values() if label.dimension == 2 for tag in label.entities}
        assert set(got["air"].entities) & faces and set(got["slab"].entities) & faces
        assert got["air"].beside == got["slab"].beside == ()

    def test_a_point_or_a_curve_embedded_in_a_face_stands_on_the_model(self):
        """What stands inside the model is found through faces as well as
        volumes: a point or a curve the fragmenting left on a face is embedded
        in the face and in no volume."""
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("loose")
            occ = gmsh.model.occ
            box = occ.addBox(0, 0, 0, 10, 10, 10)
            on_face = occ.addPoint(5, 2, 10)
            along_face = occ.addLine(occ.addPoint(3, 3, 10), occ.addPoint(6, 6, 10))
            _, became = occ.fragment([(3, box)], [(0, on_face), (1, along_face)])
            occ.synchronize()
            beyond = occ.addPoint(5, 2, 20)
            occ.synchronize()
            marks = {entity for pieces in became[1:] for entity in pieces} | {(0, beyond)}
            assert label_map.loose(marks, 3) == {(0, beyond)}
        finally:
            gmsh.finalize()

    def test_a_body_outside_the_model_against_its_skin_cuts_nothing(self, tmp_path):
        """What a mark holds of the model is taken before anything is cut, so a
        body touching the air from outside leaves the air's faces as they were."""
        against = _drawn(
            tmp_path, "against", lambda occ: occ.addBox(LENGTH, 5.0, 5.0, 5.0, 10.0, 10.0)
        )
        bare = marked(tmp_path, [], [], name="bare")
        got = marked(tmp_path, [Mark("against", 3, against)], [Near("near", "against", FINE)])
        assert not got.reached["near"].laid
        assert got.labels[WALLS].entities == bare.labels[WALLS].entities
        assert len(tetrahedra(got.path)) == len(tetrahedra(bare.path))


class TestWhatTheReportSaysIsAtEachPlace:
    def test_a_body_in_the_rings_hole_and_the_ring_are_told_apart(self, tmp_path):
        """A size throughout a disc filling the hole and a size near the ring are
        two places, and each line says how many elements are at it and what
        share of the model they are, counted against the elements the file
        holds."""
        drawn = _drawn(tmp_path, "ring", ring)
        x, y, z = RING_CENTRE
        disc = _drawn(tmp_path, "disc", lambda occ: occ.addCylinder(x, y, z - 1.0, 0, 0, 2.0, 6.0))
        # A block standing across the ring, so the ring is cut into several faces
        # and the model is filled by two labels.
        block = _drawn(
            tmp_path, "block", lambda occ: occ.addBox(x + 5.0, y - 3.0, 0, 8.0, 6.0, SIDE)
        )
        pieces = [Piece("air", 3, air_only(tmp_path)), Piece("block", 3, block, priority=1)]
        demand = [Within("hole", "disc", 2 * FINE), Near("ring", "ring", FINE)]
        got = marked(
            tmp_path, [Mark("ring", 2, drawn), Mark("disc", 3, disc)], demand, pieces=pieces
        )
        filling = got.labels["air"].elements + got.labels["block"].elements
        assert got.elements == len(tetrahedra(got.path)) == filling
        lines = gmsh_mesh_report.describe(
            got, Demand(coarsest=BULK, finest=0.0, growth=GROWTH, places=tuple(demand))
        )
        hole, around = got.reached["hole"], got.reached["ring"]
        assert hole.elements != around.elements
        for opening, reached, where in (
            ("Mesh: throughout 'hole'", hole, "in it"),
            ("Mesh: at 'ring'", around, "touching it"),
        ):
            (line,) = [line for line in lines if line.startswith(opening)]
            share = 100.0 * reached.elements / got.elements
            assert f"the {reached.elements} elements {where} are {share:.3g}% of the model" in line
        (line,) = [line for line in lines if line.startswith("Mesh: at 'ring'")]
        assert f"over its {math.pi * (9.0**2 - 7.0**2):.4g} square mm" in line


class TestAPlaceCutIntoManyFaces:
    """Gmsh lays a ``Distance`` field's sampling on every surface of it, whatever
    that surface's size, so the classes keep a small face from being sampled at a
    large one's count."""

    def drawn(self, tmp_path):
        """A plane across the air, and cubes standing across it in two rows, each
        a third larger than the one before, so the faces the plane is cut into
        spread over sizes that no one class holds."""
        plane = _drawn(tmp_path, "plane", lambda occ: occ.addRectangle(0, 0, HEIGHT, LENGTH, SIDE))

        def grains(occ):
            for row in range(2):
                x = 1.0
                for column in range(10):
                    side = 0.5 * 1.3**column
                    occ.addBox(x, 2.0 + 9.0 * row, HEIGHT - side / 2, side, side, side)
                    x += side + 1.0

        return [Mark("plane", 2, plane), Piece("grains", 3, _drawn(tmp_path, "grains", grains))]

    def test_every_face_is_sampled_within_the_class_of_its_own_need(self, tmp_path):
        """The faces the plane becomes once the cubes cut it up, in a model of the
        test's own."""
        drawn = self.drawn(tmp_path)
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("cut")
            loaded, _ = label_map.load(drawn)
            resolved, _ = label_map.fragment(loaded)
            faces = [tag for _, tag in resolved["plane"]]
            classes = _classes(2, faces, FINE)
            own = {tag: _sampling(_length(2, tag), FINE) for tag in faces}
        finally:
            gmsh.finalize()
        assert len(faces) > 20
        assert sorted(tag for _, members in classes for tag in members) == sorted(faces)
        # Each face is sampled at most twice as finely along each direction as
        # it needs, so it carries at most four times the points it needs: the
        # bound the module states, whatever CLASS is set to.
        for sampled, members in classes:
            assert all(own[tag] <= sampled <= 2 * own[tag] for tag in members)
        laid = sum(sampled**2 * len(members) for sampled, members in classes)
        assert laid <= 4 * sum(needed**2 for needed in own.values())
        assert len(classes) < len(faces)


class TestWhatALabelsElementsAre:
    def test_a_refined_face_reads_finer_than_one_that_is_not(self, tmp_path):
        """Two sheets at the two ends of the air, one sized and one not, in one
        mesh. The air round the sized one is coarser than it, and the one far
        from any place is as coarse as the bulk."""
        air = _drawn(tmp_path, "air", lambda occ: occ.addBox(0, 0, 0, LENGTH, SIDE, SIDE))
        (x0, y0), (x1, y1) = SHEET
        sized = _drawn(
            tmp_path, "sized", lambda occ: occ.addRectangle(x0, y0, HEIGHT, x1 - x0, y1 - y0)
        )
        other = _drawn(
            tmp_path,
            "other",
            lambda occ: occ.addRectangle(LENGTH - x1, y0, HEIGHT, x1 - x0, y1 - y0),
        )
        got = meshed(
            tmp_path,
            [Near("near", "sized", FINE)],
            pieces=[Piece("air", 3, air), Piece("sized", 2, sized), Piece("other", 2, other)],
        )
        near, far, round_it = (got.labels[name].edges for name in ("sized", "other", "air"))
        assert near.longest < 2.0 * FINE
        assert far.longest > BULK / 2.0
        assert near.longest < round_it.longest

    def test_a_label_of_curves_reads_the_edges_along_it(self, tmp_path):
        """A wire standing in the air is meshed as lines, and a line has an edge."""
        pieces = air_and_sheet(tmp_path)

        def line(occ):
            occ.addLine(occ.addPoint(20.0, 5.0, 5.0), occ.addPoint(40.0, 5.0, 5.0))

        wire = _drawn(tmp_path, "wire", line)
        got = meshed(tmp_path, pieces=[*pieces, Piece("wire", 1, wire)])
        edges = got.labels["wire"].edges
        assert edges is not None
        assert 0.0 < edges.shortest <= edges.longest <= 20.0

    def test_the_mesh_reads_the_extremes_over_the_labels_it_fills(self, tmp_path):
        """A block sized finer than the air round it: the block reads finer, and the
        whole mesh reads from the shortest of either to the longest of either."""
        air = _drawn(tmp_path, "air", lambda occ: occ.addBox(0, 0, 0, LENGTH, SIDE, SIDE))
        block = _drawn(tmp_path, "block", lambda occ: occ.addBox(30.0, 5.0, 5.0, 10.0, 10.0, 10.0))
        got = meshed(
            tmp_path,
            [Within("body", "block", 2 * FINE)],
            pieces=[Piece("air", 3, air), Piece("block", 3, block, priority=1)],
        )
        inside, outside = got.labels["block"].edges, got.labels["air"].edges
        assert inside.longest < outside.longest
        assert got.edges.shortest == min(inside.shortest, outside.shortest)
        assert got.edges.longest == max(inside.longest, outside.longest)


class TestASizeAtALabelOfPoints:
    def test_a_place_of_points_reads_the_edges_leaving_them(self, tmp_path):
        """No element lies along a point, so the figure set against the size asked
        is the longest edge running from it. A finer size leaves a shorter one."""
        air = _drawn(tmp_path, "air", lambda occ: occ.addBox(0, 0, 0, LENGTH, SIDE, SIDE))
        dot = _drawn(
            tmp_path, "dot", lambda occ: occ.addPoint(LENGTH / 2.0, SIDE / 2.0, SIDE / 2.0)
        )
        pieces = [Piece("air", 3, air), Piece("dot", 0, dot)]
        leaving = []
        for size in (FINE, 4 * FINE):
            got = meshed(
                tmp_path,
                [Near("dot", "dot", size)],
                growth=1.2,
                name=f"dot{size:g}",
                pieces=pieces,
            )
            reached = got.reached["dot"]
            assert reached.reached is not None and reached.elements > 0
            leaving.append(reached.reached)
        fine, coarse = leaving
        assert fine < coarse


class TestASizeAtASurface:
    def test_the_size_holds_across_a_face_with_many_short_sides(self, tmp_path):
        """A distance to a surface is measured from points laid across it, and a
        face whose sides are each shorter than the size needs them laid by the
        face's own extent, or the size between two is the size a step further
        out."""
        air = _drawn(tmp_path, "air", lambda occ: occ.addBox(0, 0, 0, LENGTH, SIDE, SIDE))
        disk = _drawn(tmp_path, "disk", lambda occ: polygon(occ, 20.0, SIDE / 2.0, HEIGHT, 8.0, 64))
        size = 2 * FINE
        got = meshed(
            tmp_path,
            [Near("near", "disk", size)],
            pieces=[Piece("air", 3, air), Piece("disk", 2, disk)],
        )
        reached = got.reached["near"].reached
        assert reached is not None and reached < 1.5 * size

    def test_a_rim_of_volumes_is_a_surface_and_holds_its_size(self, tmp_path):
        """The rim of a block is its faces, sampled as a surface is."""
        air = _drawn(tmp_path, "air", lambda occ: occ.addBox(0, 0, 0, LENGTH, SIDE, SIDE))
        block = _drawn(tmp_path, "block", lambda occ: occ.addBox(20.0, 2.0, 2.0, 30.0, 16.0, 16.0))
        size = 4 * FINE
        got = meshed(
            tmp_path,
            [AtRim("faces", "block", size)],
            pieces=[Piece("air", 3, air), Piece("block", 3, block, priority=1)],
        )
        reached = got.reached["faces"].reached
        assert reached is not None and reached < 1.5 * size


class TestWhatAPlaceLeavesAlone:
    def test_growth_is_read_nowhere_without_a_place(self, tmp_path):
        """The mesh of a demand stating no place is the mesh of its two bounds."""
        pieces = air_and_sheet(tmp_path)
        bare = meshed(tmp_path, growth=0.0, name="bare", pieces=pieces)
        grown = meshed(tmp_path, growth=GROWTH, name="grown", pieces=pieces)
        assert pathlib.Path(bare.path).read_bytes() == pathlib.Path(grown.path).read_bytes()
        assert bare.reached == grown.reached == {}

    def sources_seen(self, monkeypatch, pieces, directory, remainder=""):
        """The other size sources as Gmsh holds them when it is asked for the
        elements, for a demand sized round each turn that lays no place. No
        element is made."""
        seen = {}

        def looking(dim):
            seen.update({option: gmsh.option.getNumber(option) for option in OTHER_SOURCES})
            raise RuntimeError("looked, and made nothing")

        monkeypatch.setattr(gmsh.model.mesh, "generate", looking)
        try:
            mesh(
                pieces,
                Demand(coarsest=BULK, finest=0.0, per_turn=6),
                VOLUME,
                str(directory),
                "m",
                remainder=remainder,
            )
        except Unmeshed:
            pass
        return seen

    def test_sizing_round_a_turn_turns_the_other_sources_off_where_a_surface_bends(
        self, tmp_path, monkeypatch
    ):
        """Carried from the boundary into a volume, the size a sharp curve asks
        for would fill the volume at the size of its sharpest point."""
        drawn = json.loads((DRAWN / "cone_on_its_apex.manifest.json").read_text(encoding="utf-8"))
        cone = [
            Piece(one["label"], one["dim"], str(DRAWN / one["file"]), one["priority"])
            for one in drawn["pieces"]
        ]
        assert self.sources_seen(monkeypatch, cone, tmp_path) == dict.fromkeys(OTHER_SOURCES, 0.0)

    def test_a_drawing_with_no_surface_that_bends_keeps_them(self, tmp_path, monkeypatch):
        """Its mesh is the one it was before sizing round a turn was asked."""
        seen = self.sources_seen(monkeypatch, air_and_sheet(tmp_path), tmp_path, WALLS)
        assert all(value != 0.0 for value in seen.values()), seen

    def test_a_borrowed_session_gets_its_size_sources_back(self, tmp_path):
        """Every other source of a size is turned off where a place is stated, and
        each is an option of the session rather than of the model."""
        assert set(OTHER_SOURCES) <= set(TURNED)
        pieces = air_and_sheet(tmp_path)
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            before = {option: gmsh.option.getNumber(option) for option in OTHER_SOURCES}
            meshed(tmp_path, [AtRim("rim", "sheet", FINE)], pieces=pieces)
            after = {option: gmsh.option.getNumber(option) for option in OTHER_SOURCES}
        finally:
            gmsh.finalize()
        assert after == before
        assert any(before.values()), "every source was already off, so nothing was put back"


#: An iris across the air from its ceiling, as the room between the x of its
#: two faces and the depth it hangs down.
IRIS = (25.0, 35.0, 6.0)


def notched(directory: pathlib.Path) -> list[Piece]:
    """The air with an iris cut out of it: the wall is the iris's faces."""
    start, stop, depth = IRIS

    def build(occ):
        box = occ.addBox(0, 0, 0, LENGTH, SIDE, SIDE)
        iris = occ.addBox(start, 0, SIDE - depth, stop - start, SIDE, depth)
        occ.cut([(3, box)], [(3, iris)])

    return [Piece("air", 3, _drawn(directory, "notched", build))]


def iris_of_metal(directory: pathlib.Path) -> list[Piece]:
    """The whole air, and the iris drawn in it as a body that leaves the model."""
    start, stop, depth = IRIS
    iris = _drawn(
        directory,
        "iris",
        lambda occ: occ.addBox(start, 0, SIDE - depth, stop - start, SIDE, depth),
    )
    return [Piece("air", 3, air_only(directory)), Piece("iris", 3, iris, priority=1, leaves=True)]


def housed(directory: pathlib.Path) -> list[Piece]:
    """The air inside a housing of metal a wall thick all round, which leaves."""

    def build(occ):
        outer = occ.addBox(-2, -2, -2, LENGTH + 4, SIDE + 4, SIDE + 4)
        occ.cut([(3, outer)], [(3, occ.addBox(0, 0, 0, LENGTH, SIDE, SIDE))])

    housing = _drawn(directory, "housing", build)
    return [Piece("air", 3, air_only(directory)), Piece("housing", 3, housing, leaves=True)]


def turned(directory, pieces, labels, name):
    """The mesh with a rim asked where the room turns round each label."""
    return mesh(
        pieces,
        Demand(
            coarsest=BULK,
            finest=0.0,
            growth=GROWTH,
            places=tuple(AtRim(label, label, FINE, reentrant=True) for label in labels),
            walls=tuple(labels),
        ),
        VOLUME,
        str(directory),
        name,
        remainder=WALLS,
    )


class TestASizeWhereTheRoomTurns:
    """A rim asked ``reentrant`` is laid at the edges the room turns round past
    half a turn, whichever way the metal round them was drawn."""

    def test_an_iris_drawn_as_metal_and_as_a_notch_in_the_air_refine_its_lower_edges(
        self, tmp_path
    ):
        """Its two lower edges are steps the room turns three quarters round, and
        where it meets the guide's walls the room stands in a corner."""
        start, stop, depth = IRIS
        notch = turned(tmp_path, notched(tmp_path), [WALLS], "notch")
        metal = turned(tmp_path, iris_of_metal(tmp_path), [WALLS, "iris"], "metal")
        drawn = notch.reached[WALLS]
        body, rest = metal.reached["iris"], metal.reached[WALLS]
        assert drawn.laid and body.laid and not rest.laid
        assert drawn.extent == pytest.approx(2 * SIDE, rel=1e-9)
        assert body.extent == pytest.approx(drawn.extent, rel=1e-9)
        assert drawn.left and body.left and rest.left

    def test_a_housing_round_the_air_lays_nothing_at_its_corners(self, tmp_path):
        """The mesh is the one no rim is asked of."""
        pieces = housed(tmp_path)
        housing = turned(tmp_path, pieces, ["housing"], "housing")
        plain = meshed(tmp_path, name="plain", pieces=pieces)
        assert (housing.reached["housing"].laid, housing.reached["housing"].left) == (False, 12)
        assert housing.elements == plain.elements

    def test_a_ridge_is_refined_past_a_degree_beyond_half_a_turn(self, tmp_path):
        """A ridge of three quarters of a degree each side is a turn of one and a
        half past half, and one of a quarter each side is half a degree past it,
        which the kernel's error on a tangent joint could make of a flat one."""
        one = turned(tmp_path, kinked(tmp_path, 0.75, 0.25), [WALLS], "kinked")
        assert one.reached[WALLS].extent == pytest.approx(LENGTH, rel=1e-9)
        none = turned(tmp_path, kinked(tmp_path, 0.25, 0.25), [WALLS], "flat")
        assert not none.reached[WALLS].laid

    def test_a_sheet_across_the_air_is_refined_at_its_free_edges_alone(self, tmp_path):
        """Its edges on the walls stand in two corners each, and the sheet is drawn
        in two halves, whose shared edge is flat."""

        def build(occ):
            occ.addRectangle(20.0, 0.0, HEIGHT, 5.0, SIDE)
            occ.addRectangle(25.0, 0.0, HEIGHT, 5.0, SIDE)

        pieces = [
            Piece("air", 3, air_only(tmp_path)),
            Piece("sheet", 2, _drawn(tmp_path, "across", build)),
        ]
        got = turned(tmp_path, pieces, ["sheet", WALLS], "across")
        assert got.reached["sheet"].extent == pytest.approx(2 * SIDE, rel=1e-9)
        assert got.reached["sheet"].left == 5


def kinked(directory: pathlib.Path, floor: float, ceiling: float) -> list[Piece]:
    """The air as a prism along x whose floor rises to a ridge along the middle
    at ``floor`` degrees each side, and whose ceiling dips to one at ``ceiling``:
    the room turns round each ridge by half a turn and twice the angle."""

    def build(occ):
        rise = SIDE / 2.0 * math.tan(math.radians(floor))
        dip = SIDE / 2.0 * math.tan(math.radians(ceiling))
        corners = [
            (0.0, 0.0),
            (SIDE / 2.0, rise),
            (SIDE, 0.0),
            (SIDE, SIDE),
            (SIDE / 2.0, SIDE - dip),
            (0.0, SIDE),
        ]
        points = [occ.addPoint(0.0, y, z) for y, z in corners]
        lines = [occ.addLine(one, points[(k + 1) % len(points)]) for k, one in enumerate(points)]
        face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        occ.extrude([(2, face)], LENGTH, 0.0, 0.0)

    return [Piece("air", 3, _drawn(directory, "kinked", build))]


class TestAPlaceOnALabelThatHoldsNothing:
    def test_a_remainder_the_drawn_labels_leave_nothing_for_is_refused(self, tmp_path):
        """Every face where the air ends is drawn under a label of its own, so no
        group is made for the remainder, and a place on it would lay a field over
        no entity."""
        air = _drawn(tmp_path, "air", lambda occ: occ.addBox(0, 0, 0, LENGTH, SIDE, SIDE))

        def skin(occ):
            occ.remove([(3, occ.addBox(0, 0, 0, LENGTH, SIDE, SIDE))])

        pieces = [Piece("air", 3, air), Piece("skin", 2, _drawn(tmp_path, "skin", skin))]
        assert WALLS not in meshed(tmp_path, pieces=pieces, name="bare").labels
        with pytest.raises(Refused) as refused:
            meshed(tmp_path, [Near("near", WALLS, FINE)], pieces=pieces, name="placed")
        assert "'near' names the label 'walls', which holds nothing" in str(refused.value)


class TestHowAPlaceIsSampledAndMeasured:
    """Arithmetic the answer rests on, asked of the helpers directly."""

    def test_no_point_of_a_curve_stands_further_than_the_size_from_a_sampled_point(self):
        """Gmsh lays a ``Distance`` field's points along a curve at the fractions
        ``i / (n - 1)`` of it for ``i`` from 1 to ``n - 2``, and none at its ends
        (Gmsh 4.15.2, ``src/mesh/Field.cpp:2602``). Lengths a whole number of sizes
        long are where one point too few shows."""
        size = 0.5
        for multiple in (0.5, 1.0, 2.0, 3.0, 8.0, 8.5):
            length = multiple * size
            gmsh.initialize()
            try:
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.model.add("line")
                occ = gmsh.model.occ
                line = occ.addLine(occ.addPoint(0, 0, 0), occ.addPoint(length, 0, 0))
                occ.synchronize()
                ((count, _),) = _classes(1, [line], size)
            finally:
                gmsh.finalize()
            laid = [length * i / (count - 1) for i in range(1, count - 1)]
            assert laid, multiple
            gaps = numpy.diff([0.0, *laid, length])
            assert max(gaps[0], gaps[-1], *(gaps[1:-1] / 2.0)) <= size, multiple

    def test_no_point_of_a_curve_whose_parameter_runs_unevenly_stands_further_either(self):
        """Gmsh lays the points at even steps of the parameter, and the parameter
        of a spline covers more length per unit along one stretch than along
        another. The gaps are measured along the curve, where the caller reads
        them."""
        size = 0.5
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("spline")
            occ = gmsh.model.occ
            poles = [occ.addPoint(at, 0, 0) for at in CLUSTERED]
            curve = occ.addBSpline(poles, degree=3)
            occ.synchronize()
            ((count, _),) = _classes(1, [curve], size)
            (low,), (high,) = gmsh.model.getParametrizationBounds(1, curve)
            dense = numpy.linspace(low, high, 20001)
            along = numpy.asarray(gmsh.model.getValue(1, curve, list(dense))).reshape(-1, 3)
            laid = [low + (high - low) * i / (count - 1) for i in range(1, count - 1)]
        finally:
            gmsh.finalize()
        walked = numpy.concatenate(
            [[0.0], numpy.cumsum(numpy.linalg.norm(numpy.diff(along, axis=0), axis=1))]
        )
        at = numpy.interp(laid, dense, walked)
        assert len(at)
        gaps = numpy.diff([0.0, *at, walked[-1]])
        assert max(gaps[0], gaps[-1], *(gaps[1:-1] / 2.0)) <= size

    def test_the_curves_of_one_place_are_one_field_sampled_for_the_longest(self):
        """A curve is given the sampling itself rather than its square, so a
        place of curves stays the one field it has always been."""
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("lines")
            occ = gmsh.model.occ
            long = occ.addLine(occ.addPoint(0, 0, 0), occ.addPoint(30.0, 0, 0))
            short = occ.addLine(occ.addPoint(0, 1, 0), occ.addPoint(3.0, 1, 0))
            occ.synchronize()
            classes = _classes(1, [long, short], FINE)
        finally:
            gmsh.finalize()
        assert classes == [(_sampling(30.0, FINE), [long, short])]

    def test_one_element_far_coarser_than_the_rest_does_not_move_the_figure(self):
        """The figure beside the elements at a place is their median."""
        longest = numpy.array([1.0, 1.0, 1.0, 9.0])
        figures = _figures(1.0, None, longest, 2)
        assert (figures.standing, figures.elements) == (1.0, len(longest))
        assert _figures(1.0, None, numpy.zeros(0), 2).standing is None
