# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The label map over drawings FreeCAD made, and what the mesher says about each.

``tests/test_gmsh_mesher.py`` asks the same questions of maps written by hand,
which is where a refusal's arithmetic is pinned. Here the map comes from the
kernel, over shapes a user could draw, so what is being asked is whether the
mechanism holds on the drawing rather than on the map.

Needs Gmsh and skips itself without one. The files it reads are written by
``tests/drawings_probe.py`` under a real FreeCAD and are checked in, so nothing
here needs the CAD kernel either.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
from dataclasses import replace

import numpy
import pytest

from tests.conftest import needed
from tests.drawings import COAX_RADIUS, MESHED, REFUSED, UNMESHED, Drawing, drawings, ends

gmsh = needed(
    "gmsh", "gmsh is not on this interpreter, so the mesher is unreachable", module_level=True
)

from Microwave.Gmsh import coverage  # noqa: E402
from Microwave.Gmsh import labels as label_map  # noqa: E402
from Microwave.Gmsh.mesh import ALGORITHM, QUALITY, TURNED, mesh  # noqa: E402
from Microwave.Gmsh.vocabulary import (  # noqa: E402
    BOTH,
    FORMATS,
    FRONTIER,
    INTERIOR,
    AtRim,
    Demand,
    Mark,
    Near,
    Piece,
    Profile,
    Refused,
    Uncut,
    Unmeshed,
    Within,
)

#: Where ``tests/drawings_probe.py`` wrote them.
DRAWN = pathlib.Path(__file__).resolve().parent / "_drawings"

#: What a drawing that states no size of its own is meshed to. Coarse, because
#: most of these are not about an element size: what is asked is which entity
#: ended up under which label. Read off the corpus rather than written again, so
#: the drawings that do state a size are stating a difference from this one.
DEMAND = Demand(coarsest=Drawing.coarsest, finest=Drawing.finest)

#: A volume and its boundary, straight-sided. The order and the curving are
#: turned by the cases that are about them.
VOLUME = Profile(top=3, element_order=1, curved=False)

#: Sizes that are not sizes, written here because a literal ``float("nan")`` in
#: a parameter list reads as a value rather than as the absence of one.
NOT_A_NUMBER = float("nan")
WITHOUT_END = float("inf")

#: The highest element order asked of a format that states no limit of its own.
#: Higher orders exist; this is as far as the round trip through every format
#: here has been measured.
HIGHEST = 4

#: How many nodes a tetrahedron of each element order carries, which is what a
#: file read back is asked to show. Arithmetic on the order rather than a
#: measurement: an order-n tetrahedron carries the number of monomials of degree
#: at most n in three variables.
NODES_PER_TET = {1: 4, 2: 10, 3: 20, 4: 35}

#: An element order past anything Gmsh builds. Which orders it does build is
#: Gmsh's own business and is not written down here; what this holds is that a
#: refusal from inside Gmsh reaches the caller as one.
BEYOND_GMSH = 20

#: How far a sum over a mesh's elements may stand above the drawing's own
#: measure and still be rounding. Each element is a determinant of differences
#: of coordinates and the sum runs over every one of them, so the error grows
#: with how many there are rather than staying at one epsilon.
SUMMED = 1e-9


def profile_for(drawing):
    """The profile a drawing is meshed with, which is the drawing's own."""
    return Profile(
        top=drawing.top,
        element_order=drawing.element_order,
        curved=drawing.curved,
    )


def read_back(path):
    """Open a written mesh in a session of its own, and say what it holds.

    Its own session because the mesher closes the one it opened. What comes
    back is the physical groups under the names they were written with, every
    element quality, and the extreme element edges - so what the mesher
    reported is compared against the file rather than against the object that
    wrote it.

    Every corner is measured against every other, which is the element's edges
    on a simplex and its edges plus its diagonals on anything else. Gmsh gives
    this package simplices, and a shape that changed that would show up here as
    a disagreement rather than pass unnoticed.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        groups = {
            gmsh.model.getPhysicalName(dim, tag): (dim, tag)
            for dim, tag in gmsh.model.getPhysicalGroups()
        }
        node_tags, flat, _ = gmsh.model.mesh.getNodes()
        at = dict(zip([int(tag) for tag in node_tags], numpy.reshape(flat, (-1, 3)), strict=True))
        quality = {}
        counted = {}
        spanned = {}
        for dim in (1, 2, 3):
            kinds, element_tags, listing = gmsh.model.mesh.getElements(dim)
            every = [tag for group in element_tags for tag in group]
            counted[dim] = len(every)
            if every and dim > 1:
                quality[dim] = float(min(gmsh.model.mesh.getElementQualities(every, QUALITY)))
            lengths = []
            for kind, listed in zip(kinds, listing, strict=True):
                # From the element's own corner nodes rather than from Gmsh's
                # edge helper, which is what the mesher uses: an arithmetic of
                # its own is the only way this can disagree with it.
                _, _, _, nodes, _, corners = gmsh.model.mesh.getElementProperties(int(kind))
                block = numpy.reshape(listed, (-1, nodes))[:, :corners]
                for element in block:
                    places = [at[int(node)] for node in element]
                    lengths.extend(
                        float(numpy.linalg.norm(places[one] - places[other]))
                        for one in range(corners)
                        for other in range(one + 1, corners)
                    )
            if lengths:
                spanned[dim] = (float(min(lengths)), float(max(lengths)))
        return groups, quality, counted, spanned
    finally:
        gmsh.finalize()


def manifest(name):
    """What the probe wrote for one drawing."""
    return json.loads((DRAWN / f"{name}.manifest.json").read_text())


def pieces(name):
    """One drawing, as the caller would hand it over."""
    return [
        Piece(
            p["label"],
            p["dim"],
            str(DRAWN / p["file"]),
            p["priority"],
            divides=p.get("divides", False),
            leaves=p.get("leaves", False),
        )
        for p in manifest(name)["pieces"]
    ]


def demand_for(drawing):
    """The element size this drawing is meshed to."""
    return Demand(coarsest=drawing.coarsest, finest=drawing.finest, per_turn=drawing.per_turn)


#: The four triangular faces of a tetrahedron, as offsets into its corners. A
#: higher order element carries more nodes and the first four are still these,
#: since the corners come before the nodes added between them.
FACES_OF_A_TET = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def frontier_area(path):
    """The area of the faces one tetrahedron uses, and how many more than two do.

    Every triangular face of a tetrahedron is either shared with the
    tetrahedron on the other side of it or on the frontier of the meshed
    region. So the faces used once are that frontier, and their area is what
    the drawing's own frontier measures - the skin of the fused shapes, a
    cavity's wall among them, which is not the same as the outside of anything.
    Two regions meshed without being joined put the surface between them in the
    file twice, once from each side, and the frontier runs over by it.

    A face more than two tetrahedra use is the same fault at its coarsest.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        at = {int(tag): numpy.array(coordinates[3 * i : 3 * i + 3]) for i, tag in enumerate(tags)}
        used = {}
        kinds, every, nodes = gmsh.model.mesh.getElements(3)
        for kind, elements, listed in zip(kinds, every, nodes):
            per = gmsh.model.mesh.getElementProperties(kind)[3]
            rows = numpy.array(listed, dtype=int).reshape(len(elements), per)
            for row in rows:
                for face in FACES_OF_A_TET:
                    corners = tuple(sorted(int(row[i]) for i in face))
                    used[corners] = used.get(corners, 0) + 1
        area = sum(
            0.5 * float(numpy.linalg.norm(numpy.cross(at[b] - at[a], at[c] - at[a])))
            for (a, b, c), count in used.items()
            if count == 1
        )
        return area, len([1 for count in used.values() if count > 2])
    finally:
        gmsh.finalize()


def filled_measure(path, top):
    """What the written mesh occupies at the filled dimension.

    Summed over the elements themselves rather than read off the model, because
    what is being asked is whether the file holds the drawing - and a Gmsh that
    filled nothing still writes the surfaces, the groups and the header.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        if not len(tags):
            return 0.0
        at = {int(tag): coordinates[3 * i : 3 * i + 3] for i, tag in enumerate(tags)}
        total = 0.0
        kinds, every, nodes = gmsh.model.mesh.getElements(top)
        for kind, elements, listed in zip(kinds, every, nodes):
            per = gmsh.model.mesh.getElementProperties(kind)[3]
            rows = numpy.array(listed, dtype=int).reshape(len(elements), per)
            for row in rows:
                corners = [at[int(node)] for node in row[: top + 1]]
                total += _simplex(corners, top)
        return total
    finally:
        gmsh.finalize()


def _simplex(corners, top):
    """The volume of a tetrahedron or the area of a triangle, from its corners.

    The corners alone, so a second-order element is measured as the straight one
    through the same four - which is what a straight-sided profile asks for and
    is short of a curved one by what the curving bought.
    """
    if top == 3:
        a, b, c, d = corners
        return float(abs(numpy.dot(numpy.cross(b - a, c - a), d - a))) / 6.0
    a, b, c = corners
    return float(numpy.linalg.norm(numpy.cross(b - a, c - a))) / 2.0


@pytest.fixture(scope="module")
def drawn():
    """Every drawing the probe wrote, by name."""
    assert DRAWN.is_dir(), f"{DRAWN} is missing; run tests/drawings_probe.py under FreeCAD"
    return {drawing.name: drawing for drawing in drawings()}


def test_the_probe_wrote_every_drawing(drawn):
    """The floor under every case below, which would otherwise fail one at a
    time on a missing file and say nothing about why."""
    missing = sorted(name for name in drawn if not (DRAWN / f"{name}.manifest.json").is_file())
    assert not missing, f"these were never drawn: {missing}"


@pytest.mark.parametrize("name", [d.name for d in drawings()])
def test_what_was_drawn_is_what_the_drawing_declares(name, drawn):
    """The checked-in files are compared with the builder they came from.

    The probe makes this comparison too, and only while it runs. The case it
    cannot cover is the one that matters: a builder edited and the probe not run
    again leaves stale files that every case below reads happily, under labels
    the module does not state.
    """
    drawing = drawn[name]
    listed = manifest(name)["pieces"]
    written = tuple((piece["label"], piece["dim"]) for piece in listed)
    assert written == drawing.declares, (
        f"{name} was drawn as {written} and declares {drawing.declares}; "
        f"run tests/drawings_probe.py under FreeCAD"
    )
    stated = {piece["label"]: piece["priority"] for piece in listed}
    asked = dict(drawing.priority)
    assert stated == {label: asked.pop(label, 0) for label in stated} and not asked, (
        f"{name} was drawn at {stated} and states {drawing.priority}; either the probe "
        f"has not been run under FreeCAD since, or an order names a label the drawing "
        f"does not"
    )
    cutting = tuple(sorted(piece["label"] for piece in listed if piece.get("divides")))
    assert cutting == tuple(sorted(drawing.divides)), (
        f"{name} was drawn with {cutting} dividing what is filled and states "
        f"{drawing.divides}; either the probe has not been run under FreeCAD since, or "
        f"it names a label the drawing does not"
    )
    going = tuple(sorted(piece["label"] for piece in listed if piece.get("leaves")))
    assert going == tuple(sorted(drawing.leaves)), (
        f"{name} was drawn with {going} leaving the model and states {drawing.leaves}; "
        f"either the probe has not been run under FreeCAD since, or it names a label the "
        f"drawing does not"
    )


@pytest.mark.parametrize("name", [d.name for d in drawings() if d.expect == MESHED])
def test_a_sound_drawing_meshes(name, drawn, tmp_path):
    """Every label reaches the mesh, and each is a group of its own dimension."""
    drawing = drawn[name]
    got = mesh(pieces(name), demand_for(drawing), profile_for(drawing), str(tmp_path), name)
    subject = drawing.subject
    assert pathlib.Path(got.path).is_file(), subject
    given = {piece.label for piece in pieces(name)}
    assert set(got.labels) == given, subject
    for label, where in got.labels.items():
        assert where.entities, f"{subject}: {label} reached nothing"


@pytest.mark.parametrize("name", [d.name for d in drawings() if d.expect == MESHED])
def test_the_written_file_carries_what_the_mesher_reported(name, drawn, tmp_path):
    """The labels are read back out of the file rather than off the object.

    A group's name in the file is the whole point of a label reaching the mesh,
    and the object that wrote it agrees with itself whatever it wrote. So the
    file is opened again and asked.

    The worst quality is compared the same way, against every element the file
    holds - which is what says the reported number is the worst rather than
    merely one of them. How closely it has to agree is the drawing's, because
    the file is where a coordinate stops being exact.
    """
    drawing = drawn[name]
    got = mesh(pieces(name), demand_for(drawing), profile_for(drawing), str(tmp_path), name)
    groups, quality, counted, spanned = read_back(got.path)
    assert set(groups) == set(got.labels), drawing.subject
    for label, where in got.labels.items():
        assert groups[label] == (where.dimension, where.tag), drawing.subject
    assert quality == pytest.approx(got.worst_quality, rel=drawing.round_trip), drawing.subject
    assert counted[drawing.top] > 0, f"{drawing.subject}: nothing filled the top dimension"
    # The extreme edges the same way, and over the dimension the profile filled:
    # a length asked for is answered by a length, and the file is where that
    # answer either survived or did not.
    shortest, longest = spanned[drawing.top]
    assert got.edges.shortest == pytest.approx(shortest, rel=drawing.round_trip, abs=0.0), (
        drawing.subject
    )
    assert got.edges.longest == pytest.approx(longest, rel=drawing.round_trip, abs=0.0), (
        drawing.subject
    )


@pytest.mark.parametrize("name", [d.name for d in drawings() if d.expect == MESHED and d.top == 3])
def test_the_elements_meet_across_an_interface(name, drawn, tmp_path):
    """The regions share the faces between them rather than each carrying its own.

    This is what the fragmenting is for, and nothing else here would notice it
    missing: two regions built apart can carry every label, fill every entity
    and add up to the drawing's own measure while the surface between them is
    in the file twice and no element on one side meets an element on the other.
    A solver reads that as two bodies that do not touch.

    Measured against the drawing's own frontier rather than against a figure of
    its own, since the frontier of a joined mesh is exactly that. A curved
    body's frontier is chords of it and stands inside, by what the drawing
    states it loses.
    """
    drawing = drawn[name]
    got = mesh(pieces(name), demand_for(drawing), profile_for(drawing), str(tmp_path), name)
    area, over = frontier_area(got.path)
    assert over == 0, f"{drawing.subject}: {over} faces carry more than two elements"
    short = 1.0 - area / manifest(name)["outside"]
    assert short >= -SUMMED, f"{drawing.subject}: the frontier runs over the drawing's own"
    assert short <= drawing.loses, f"{drawing.subject}: the frontier is short by {short}"


@pytest.mark.parametrize("name", [d.name for d in drawings() if d.expect == MESHED and d.top == 3])
def test_the_groups_of_the_filled_dimension_divide_the_mesh(name, drawn, tmp_path):
    """One element, one label, and between them the whole of the mesh.

    This is what a backend rests on: it gives an element one attribute, so an
    element in two groups arrives either doubled or under a name neither caller
    wrote. Asked of the written file rather than of the label map, because the
    map is about the drawing and this is about what a reader of the file gets.

    Only the disjoint half is asked here. Gmsh writes the elements of the
    entities in a group and leaves the rest, so every element in the file is in
    a group by construction and comparing the two sets asserts nothing. That
    the groups cover the drawing is what
    ``test_the_mesh_fills_the_drawing_it_was_made_from`` measures, against the
    drawing rather than against the file.
    """
    drawing = drawn[name]
    got = mesh(pieces(name), demand_for(drawing), profile_for(drawing), str(tmp_path), name)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(got.path)
        every = set()
        for dim, group in gmsh.model.getPhysicalGroups(drawing.top):
            mine = set()
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, group):
                _, tags, _ = gmsh.model.mesh.getElements(dim, entity)
                mine.update(int(tag) for listed in tags for tag in listed)
            shared = every & mine
            assert not shared, (
                f"{drawing.subject}: {len(shared)} elements are in two groups of "
                f"dimension {drawing.top}"
            )
            every |= mine
        assert every, f"{drawing.subject}: no group of dimension {drawing.top} holds anything"
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("name", [d.name for d in drawings() if d.priority and d.expect == MESHED])
def test_the_answer_says_which_pieces_two_labels_were_drawn_over(name, drawn, tmp_path):
    """Nothing else in the answer records that a label was drawn over a region
    it did not keep.

    A caller reads this to check that what it drew over what is what it meant.
    An ordinary drawing puts a part inside a domain and every piece of the part
    comes back here, so the record is not a warning - it is the only place the
    overlap is visible at all once the groups are made.
    """
    drawing = drawn[name]
    got = mesh(pieces(name), demand_for(drawing), profile_for(drawing), str(tmp_path), name)
    assert got.settled, f"{drawing.subject}: the drawing has an overlap and the answer says none"
    stated = dict(drawing.priority)
    for one in got.settled:
        assert one.took not in one.gave_up
        assert stated.get(one.took, 0) > max(stated.get(label, 0) for label in one.gave_up), (
            f"{one.took} took a piece from {one.gave_up} without standing above them"
        )
        if one.took in drawing.leaves:
            # The piece left the model, and the label holds the faces it left.
            assert got.labels[one.took].dimension == one.dimension - 1
        else:
            assert one.tag in got.labels[one.took].entities
        for label in one.gave_up:
            assert one.tag not in got.labels[label].entities
        assert one.place, "a piece is named by where it is, since a tag reaches nothing drawn"


@pytest.mark.parametrize(
    "name", [d.name for d in drawings() if not d.priority and d.expect == MESHED]
)
def test_a_drawing_with_no_overlap_in_it_settles_nothing(name, drawn, tmp_path):
    """The record is empty rather than absent, which is what a caller that
    subtracted before handing over sees."""
    drawing = drawn[name]
    got = mesh(pieces(name), demand_for(drawing), profile_for(drawing), str(tmp_path), name)
    assert got.settled == ()


@pytest.mark.parametrize("name", [d.name for d in drawings() if d.divides])
def test_the_answer_says_what_a_dividing_label_leaves_standing_apart(name, drawn, tmp_path):
    """What is filled, asked again with the dividing faces joining nothing.

    The faces still join the region, so nothing else in the answer shows this:
    a skin sealing a cavity and a skin the region wraps round are one drawing
    to every other question the mesher asks.
    """
    drawing = drawn[name]
    got = mesh(pieces(name), demand_for(drawing), profile_for(drawing), str(tmp_path), name)
    assert len(got.parted) == drawing.parts, drawing.subject
    for part in got.parted:
        assert part.place, "a part is named by where it is, since a tag reaches nothing drawn"
        assert set(part.labels) <= set(got.labels), "a part names a label the mesh does not carry"
        assert set(drawing.divides) & set(part.labels), (
            f"{drawing.subject}: a part the dividing labels cut stands on none of them"
        )
    held: set[str] = set()
    for part in got.parted:
        held |= set(part.labels)
    assert held == set(got.labels), (
        f"{drawing.subject}: the parts between them stand on every label of the mesh"
    )


@pytest.mark.parametrize(
    "name", [d.name for d in drawings() if not d.divides and d.expect == MESHED]
)
def test_a_drawing_that_names_nothing_dividing_is_not_parted(name, drawn, tmp_path):
    """Empty rather than one part, so a caller that asked nothing is told nothing
    rather than being told its model holds together."""
    drawing = drawn[name]
    got = mesh(pieces(name), demand_for(drawing), profile_for(drawing), str(tmp_path), name)
    assert got.parted == ()


def test_a_sealed_cavity_and_the_region_round_it_are_told_apart(drawn, tmp_path):
    """The one thing the parts are for: which labels stand on the part that is
    closed off, so a caller can see it holds nothing it drives with."""
    drawing = drawn["skin_sealing_a_cavity"]
    got = mesh(
        pieces(drawing.name),
        demand_for(drawing),
        profile_for(drawing),
        str(tmp_path),
        drawing.name,
    )
    inside = [part for part in got.parted if "walls" not in part.labels]
    outside = [part for part in got.parted if "walls" in part.labels]
    assert len(inside) == len(outside) == 1, (
        "the part inside the skin is the one the walls do not bound"
    )
    assert set(inside[0].labels) == {"air", "skin"}, (
        "the sealed part is filled by the air label and bounded by the skin alone"
    )
    # The place is the only thing that reaches a user, so it is held against the
    # drawing rather than merely asserted non-empty: the sealed part is the box
    # _skin_sealing_a_cavity draws inside the other, and a place computed off the
    # wrong entities lands somewhere else.
    assert _corners(inside[0].place) != _corners(outside[0].place)
    low, high = _corners(inside[0].place)
    around_low, around_high = _corners(outside[0].place)
    assert all(a <= b for a, b in zip(around_low, low, strict=True))
    assert all(a >= b for a, b in zip(around_high, high, strict=True))
    assert [round(value) for value in low] == [10, 7, 3]
    assert [round(value) for value in high] == [18, 13, 7]


def _meshed(name, tmp_path, profile=None, demand=None):
    """One corpus drawing meshed as it states, or with the profile or the
    demand given."""
    drawing = drawn_by_name(name)
    return mesh(
        pieces(name),
        demand or demand_for(drawing),
        profile or profile_for(drawing),
        str(tmp_path),
        name,
    )


def drawn_by_name(name):
    """The corpus drawing of that name."""
    return next(drawing for drawing in drawings() if drawing.name == name)


def test_a_body_that_leaves_holds_the_faces_it_leaves_behind(tmp_path):
    """The whole of the post's skin, where the model now ends against the air."""
    got = _meshed("post_in_the_box", tmp_path)
    post = got.labels["post"]
    assert post.dimension == 2
    assert post.sits == FRONTIER
    assert post.beside == ("air",)
    # _post_in_the_box draws an 8 x 6 x 4 box.
    assert post.size == pytest.approx(2 * (8 * 6 + 8 * 4 + 6 * 4), rel=1e-9)
    assert got.trimmed == ()


def test_what_a_body_leaves_on_a_wall_leaves_with_it(tmp_path):
    """The plate lies on the floor and the roof, so its faces there bound
    nothing once it has gone: they leave, and what the walls were drawn over
    there is reported rather than refused."""
    got = _meshed("plate_standing_in_the_box", tmp_path)
    # _plate_standing_in_the_box draws a 10 x 1 x 10 plate the full height of the box.
    assert got.labels["plate"].size == pytest.approx(2 * (10 * 10 + 1 * 10), rel=1e-9)
    assert got.labels["walls"].size == pytest.approx(
        2 * (30 * 20 + 30 * 10 + 20 * 10) - 2 * (10 * 1), rel=1e-9
    )
    (cut,) = got.trimmed
    assert (cut.label, cut.by) == ("walls", ("plate",))
    low, high = _corners(cut.place)
    assert [round(value, 3) for value in low] == [10, 9.5, 0]
    assert [round(value, 3) for value in high] == [20, 10.5, 10]


def test_what_a_body_leaves_on_the_frontier_is_not_among_its_faces(tmp_path):
    """With nothing drawn on the walls, the plate's faces on the floor and the
    roof bound nothing once it has gone, and nothing claims them to hide it."""
    name = "plate_standing_in_the_box"
    drawing = drawn_by_name(name)
    got = mesh(
        [piece for piece in pieces(name) if piece.label != "walls"],
        demand_for(drawing),
        profile_for(drawing),
        str(tmp_path),
        name,
        remainder="walls",
    )
    assert got.labels["plate"].size == pytest.approx(2 * (10 * 10 + 1 * 10), rel=1e-9)
    assert got.trimmed == ()


def test_a_sheet_drawn_on_a_face_of_a_body_keeps_it(tmp_path):
    """The face is the sheet's, so no contest is settled over it a second time
    and the body holds the other five."""
    got = _meshed("sheet_on_a_post_face", tmp_path)
    assert got.labels["sheet"].size == pytest.approx(8 * 6, rel=1e-9)
    assert got.labels["sheet"].sits == FRONTIER
    assert got.labels["post"].size == pytest.approx(8 * 6 + 2 * (8 * 4 + 6 * 4), rel=1e-9)
    assert not set(got.labels["sheet"].entities) & set(got.labels["post"].entities)


def test_a_piece_a_region_above_a_body_keeps_is_settled_once(tmp_path):
    """The block stands above the post, so the piece the three were drawn over
    stays the block's, and the air gives it up once rather than once to each."""
    got = _meshed("post_under_a_higher_block", tmp_path)
    tags = [(one.dimension, one.tag) for one in got.settled]
    assert len(tags) == len(set(tags)), got.settled
    shared = [one for one in got.settled if set(one.gave_up) == {"air", "post"}]
    assert [one.took for one in shared] == ["block"]


def test_a_sheet_through_a_body_keeps_what_stands_outside_it(tmp_path):
    got = _meshed("post_through_a_sheet", tmp_path)
    # _post_through_a_sheet draws a 16 x 8 sheet through the post's 6 x 4 section.
    assert got.labels["sheet"].size == pytest.approx(16 * 8 - 6 * 4, rel=1e-9)
    (cut,) = got.trimmed
    assert (cut.label, cut.by) == ("sheet", ("post",))
    assert got.labels["post"].size == pytest.approx(2 * (8 * 6 + 8 * 4 + 6 * 4), rel=1e-9)


def test_a_body_filling_the_cross_section_joins_what_it_stood_between(tmp_path):
    """Asked whether the drawing holds together, the air either side of the
    plate is one part: the drawing joined it through the plate, as it would
    through a sheet in its place. Asked what the plate's condition leaves, it
    is two."""
    got = _meshed("plate_across_the_box", tmp_path, profile=replace(VOLUME, connected=True))
    assert len(got.parted) == 2
    assert all("plate" in part.labels for part in got.parted)


def test_the_rim_of_a_body_that_leaves_is_every_edge_of_it(tmp_path):
    """Its skin closes, so the rim a sheet's would be is empty; the edges of
    the body are where a size at its rim is wanted."""
    drawing = drawn_by_name("post_in_the_box")
    asked = replace(
        demand_for(drawing),
        growth=1.5,
        places=(AtRim(name="edges", label="post", size=drawing.finest),),
    )
    got = _meshed("post_in_the_box", tmp_path, demand=asked)
    assert got.reached["edges"].laid


def test_a_mark_on_a_body_that_leaves_stands_on_its_skin(tmp_path):
    """A size asked near the post is asked of the faces it left, since the model
    no longer holds the post."""
    drawing = drawn_by_name("post_in_the_box")
    asked = replace(
        demand_for(drawing),
        growth=1.5,
        places=(Near(name="round the post", label="post mark", size=drawing.finest),),
    )
    got = mesh(
        pieces("post_in_the_box"),
        asked,
        profile_for(drawing),
        str(tmp_path),
        "post_in_the_box",
        marks=(Mark("post mark", 3, str(DRAWN / "post_in_the_box.1.brep")),),
    )
    reached = got.reached["round the post"]
    assert reached.laid
    assert reached.dimension == 2


def test_a_body_through_a_wall_keeps_only_what_stood_in_the_model(tmp_path):
    """What stood outside the box was the post's alone and goes; the wall under
    the post's cross-section bounded nothing but the post, and goes with it."""
    got = _meshed("post_through_the_wall", tmp_path)
    # _post_through_the_wall draws the post 4 mm into the box, 6 x 4 across.
    assert got.labels["post"].size == pytest.approx(6 * 4 + 2 * (4 * 6 + 4 * 4), rel=1e-9)
    (cut,) = got.trimmed
    assert (cut.label, cut.by) == ("walls", ("post",))


def test_two_bodies_that_touch_leave_together(tmp_path):
    """The face between them bounds nothing once both have gone, and the sheet
    across them loses a part to each."""
    got = _meshed("two_posts_across_a_sheet", tmp_path)
    # _two_posts_across_a_sheet draws two 4 x 6 x 4 posts side by side.
    for name in ("left", "right"):
        assert got.labels[name].size == pytest.approx(2 * (4 * 6 + 4 * 4 + 6 * 4) - 6 * 4)
    (cut,) = got.trimmed
    assert (cut.label, cut.by) == ("sheet", ("left", "right"))
    assert got.labels["sheet"].size == pytest.approx(12 * 8 - 8 * 4, rel=1e-9)


def _box(path, corner, size):
    """A box written as a BREP file by Gmsh's own kernel, for a mark."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.addBox(*corner, *size)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return str(path)


def _marked(tmp_path, place, corner, size):
    """The post in the box, meshed with a place at a mark drawn as a box."""
    drawing = drawn_by_name("post_in_the_box")
    asked = replace(demand_for(drawing), growth=1.5, places=(place,))
    return mesh(
        pieces("post_in_the_box"),
        asked,
        profile_for(drawing),
        str(tmp_path),
        "post_in_the_box",
        marks=(Mark(place.label, 3, _box(tmp_path / "mark.brep", corner, size)),),
    )


def test_a_mark_wholly_inside_a_body_stands_on_the_skin_it_left(tmp_path):
    """The mark reaches no face of the post, and still a size near it is asked
    of the post's skin rather than of nothing."""
    got = _marked(tmp_path, Near("near", "inside", 2.0), (12.0, 9.0, 4.0), (2.0, 2.0, 2.0))
    assert got.reached["near"].laid
    assert got.reached["near"].dimension == 2


def test_a_mark_partly_inside_a_body_keeps_what_stays(tmp_path):
    got = _marked(tmp_path, Near("near", "across", 2.0), (14.0, 5.0, 4.0), (8.0, 4.0, 2.0))
    assert got.reached["near"].laid
    assert got.reached["near"].dimension == 3


def test_a_size_throughout_a_mark_the_body_took_is_refused(tmp_path):
    """The mark stands on the post's skin, which holds no element of a volume."""
    with pytest.raises(Refused) as refused:
        _marked(tmp_path, Within("within", "inside", 2.0), (12.0, 9.0, 4.0), (2.0, 2.0, 2.0))
    assert "'within' asks for a size throughout 'inside'" in str(refused.value)


def test_a_seam_is_no_edge_of_a_body(tmp_path):
    """A cylinder's side meets itself along a seam, and its edges are the two
    circles at its ends."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 1.0)
        gmsh.model.occ.synchronize()
        faces = [tag for _, tag in gmsh.model.getEntities(2)]
        found = label_map.bounding(2, faces)
        circles = [tag for tag in found if gmsh.model.getType(1, tag) == "Circle"]
        assert found == sorted(circles) and len(circles) == 2, found
    finally:
        gmsh.finalize()


def _corners(place):
    """The two corners a place names, as the mesher writes one."""
    low, high = place.split(" to ")
    return (
        [float(value) for value in low.strip("()").split(", ")],
        [float(value) for value in high.strip("()").split(", ")],
    )


def test_a_piece_below_the_coverage_questions_is_still_named_by_where_it_is(tmp_path):
    """A tag out of a fragmented model reaches nothing a user drew.

    The coverage questions are asked in the filled dimension and the one below
    it, so those are the dimensions whose coordinates were once gathered - and a
    label may stand lower than either. This drawing puts two of them on one
    curve under a volume profile, which is where a refusal used to name a bare
    number.
    """
    drawing = next(d for d in drawings() if d.name == "edge_drawn_twice")
    with pytest.raises(Refused) as raised:
        mesh(
            pieces("edge_drawn_twice"),
            demand_for(drawing),
            profile_for(drawing),
            str(tmp_path),
            "placed",
        )
    contest = [line for line in raised.value.complaints if "at dimension 1" in line]
    assert len(contest) == 1, raised.value.complaints
    assert " to (" in contest[0], f"the piece is named by a bare tag: {contest[0]}"


@pytest.mark.parametrize("name", [d.name for d in drawings() if d.expect == REFUSED])
def test_a_drawing_that_describes_no_mesh_is_refused_by_name(name, drawn, tmp_path):
    """And the refusal says which of them it is, in the labels the drawing used."""
    drawing = drawn[name]
    with pytest.raises(Refused) as raised:
        mesh(pieces(name), demand_for(drawing), profile_for(drawing), str(tmp_path), name)
    said = " ".join(raised.value.complaints)
    assert drawing.complaint in said, f"{drawing.subject}: got {said}"

    written = {label for label, _ in drawing.declares}
    for line in raised.value.complaints:
        assert any(label in line for label in written) or " at (" in line, (
            f"{drawing.subject}: this reaches nothing the caller drew: {line}"
        )


@pytest.mark.parametrize("name", [d.name for d in drawings() if d.expect == MESHED])
def test_the_mesh_fills_the_drawing_it_was_made_from(name, drawn, tmp_path):
    """The elements are summed and compared with what the kernel measured the
    drawing at, which is the one question the label map cannot answer.

    Every other case here asks whether the right entity carries the right name.
    A mesh can carry every name and be a mesh of something else: Gmsh warns
    about a region it could not fill, reports no error, and writes the surfaces
    and the groups anyway. What separates that file from a sound one is how
    much of the drawing is in it.

    A straight-sided element fills a straight-sided region exactly, so most of
    these hold to rounding. A curved body is approximated by chords and stands
    inside what it was drawn as, and the drawing states what that costs it.
    """
    drawing = drawn[name]
    got = mesh(pieces(name), demand_for(drawing), profile_for(drawing), str(tmp_path), name)
    short = 1.0 - filled_measure(got.path, drawing.top) / manifest(name)["measure"]
    assert short >= -SUMMED, f"{drawing.subject}: the mesh is larger than the drawing"
    assert short <= drawing.loses, f"{drawing.subject}: the mesh is short of the drawing by {short}"


@pytest.mark.parametrize("name", [d.name for d in drawings() if d.expect == UNMESHED])
def test_a_drawing_gmsh_cannot_fill_is_not_handed_back_as_a_mesh(name, drawn, tmp_path):
    """Gmsh finishing is not Gmsh filling what it was given.

    A body that passes through itself is fragmented with a warning, meshed to a
    surface and no tetrahedra with another, and reported as no errors at all.
    Left alone, that is a written file, a full label map and a region a solver
    was to fill holding nothing - and no exit code anywhere says so.

    What Gmsh said comes back with it, because Gmsh says it to a terminal and a
    caller running the mesher beside its document has none.
    """
    drawing = drawn[name]
    with pytest.raises(Unmeshed) as raised:
        mesh(pieces(name), demand_for(drawing), profile_for(drawing), str(tmp_path), name)
    said = " ".join(raised.value.complaints)
    assert drawing.complaint in said, f"{drawing.subject}: got {said}"
    assert raised.value.said, f"{drawing.subject}: what Gmsh said about it was dropped"
    assert not list(tmp_path.iterdir()), f"{drawing.subject}: a file was left behind"


#: The line that says whether a solid in a BREP file is free - made part of no
#: other shape - and the same line saying it is not. The flags are written in
#: the order free, modified, checked, orientable, closed, infinite, convex.
FREE_SOLID = "So\n\n1100000"
PART_OF_ANOTHER = "So\n\n0100000"


def test_a_drawing_the_kernel_cannot_cut_names_the_labels_and_meshes_nothing(tmp_path):
    """The kernel fails to cut a curve inside a solid inside another where the
    file marks the inner solid as part of another shape, and a caller told to
    refine would change nothing."""
    with pytest.raises(Uncut) as raised:
        mesh(pieces("curve_inside_a_post"), DEMAND, VOLUME, str(tmp_path), "curve_inside_a_post")
    (complaint,) = raised.value.complaints
    assert "'air', 'curve', 'post', 'walls'" in complaint
    assert raised.value.said


def test_the_same_drawing_whose_file_marks_the_solid_free_is_cut_and_meshed(tmp_path):
    drawn = pieces("curve_inside_a_post")
    post = next(piece for piece in drawn if piece.label == "post")
    text = pathlib.Path(post.file).read_text(encoding="utf-8")
    assert text.count(PART_OF_ANOTHER) == 1
    freed = tmp_path / "freed.brep"
    freed.write_text(text.replace(PART_OF_ANOTHER, FREE_SOLID), encoding="utf-8")
    drawn = [replace(piece, file=str(freed)) if piece is post else piece for piece in drawn]
    got = mesh(drawn, DEMAND, VOLUME, str(tmp_path), "curve_inside_a_post")
    assert pathlib.Path(got.path).is_file()


def test_a_mark_the_kernel_cannot_clip_names_the_mark_and_meshes_nothing(tmp_path):
    drawn = [piece for piece in pieces("curve_inside_a_post") if piece.label != "curve"]
    curve = next(piece for piece in pieces("curve_inside_a_post") if piece.label == "curve")
    with pytest.raises(Uncut) as raised:
        mesh(
            drawn,
            DEMAND,
            VOLUME,
            str(tmp_path),
            "curve_inside_a_post",
            marks=(Mark("curve mark", 1, curve.file),),
        )
    (complaint,) = raised.value.complaints
    assert "'curve mark'" in complaint


def test_elements_turned_over_are_named_by_the_label_holding_them(drawn, tmp_path):
    drawing = drawn["curving_turns_elements_over"]
    with pytest.raises(Unmeshed) as raised:
        mesh(
            pieces(drawing.name),
            demand_for(drawing),
            profile_for(drawing),
            str(tmp_path),
            drawing.name,
        )
    said = " ".join(raised.value.complaints)
    assert "turned inside out, which lie partly outside the drawing, in 'walls' at (" in said


def test_a_curving_pass_that_fails_in_a_session_it_did_not_open_is_its_own(drawn, tmp_path):
    """A caller whose session carries on past an error would otherwise have an
    error from making the elements skip the pass and be read as its failure."""
    drawing = drawn["curving_pass_fails"]
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.AbortOnError", 0)
        with pytest.raises(Unmeshed) as raised:
            mesh(
                pieces(drawing.name),
                demand_for(drawing),
                profile_for(drawing),
                str(tmp_path),
                drawing.name,
            )
        assert "critical value" in raised.value.complaints[0]
        assert gmsh.option.getNumber("General.AbortOnError") == 0
    finally:
        gmsh.finalize()


def test_an_error_making_the_elements_stops_them_in_a_session_it_did_not_open(tmp_path):
    """Told not to throw, Gmsh carries on past an error to the next stage, so
    the elements are made throwing whatever the caller's session says."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.AbortOnError", 0)
        with pytest.raises(Unmeshed) as raised:
            mesh(
                pieces("plain_box"),
                DEMAND,
                Profile(top=3, element_order=BEYOND_GMSH, curved=True),
                str(tmp_path),
                "ordered",
            )
        assert str(BEYOND_GMSH) in raised.value.complaints[0]
        assert gmsh.option.getNumber("General.AbortOnError") == 0
    finally:
        gmsh.finalize()


def group_area(path, name):
    """The area of the surface elements a written mesh holds under one group.

    Summed off the file rather than read off the model, and with an arithmetic
    of its own, so it can disagree with what :func:`frontier_area` computes from
    the tetrahedra on the other side of those triangles.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        at = {int(tag): numpy.array(coordinates[3 * i : 3 * i + 3]) for i, tag in enumerate(tags)}
        wanted = next(
            tag
            for dim, tag in gmsh.model.getPhysicalGroups(2)
            if gmsh.model.getPhysicalName(dim, tag) == name
        )
        total = 0.0
        for entity in gmsh.model.getEntitiesForPhysicalGroup(2, wanted):
            kinds, every, nodes = gmsh.model.mesh.getElements(2, entity)
            for kind, elements, listed in zip(kinds, every, nodes, strict=True):
                per = gmsh.model.mesh.getElementProperties(kind)[3]
                rows = numpy.array(listed, dtype=int).reshape(len(elements), per)
                for row in rows:
                    a, b, c = (at[int(node)] for node in row[:3])
                    total += 0.5 * float(numpy.linalg.norm(numpy.cross(b - a, c - a)))
        return total
    finally:
        gmsh.finalize()


def test_a_face_no_label_claims_is_meshed_and_not_written(tmp_path):
    """The drawing behind the rule that the reported quality is taken over the
    physical groups rather than over the model.

    A face between two solids is claimed by neither volume label and needs no
    label of its own, so it is meshed and left out of the file. A figure taken
    over the model is then about elements a reader of the file cannot find.

    What is held here is the difference in the sets. The worst element happens
    to be the same one either way on this drawing, so the figure alone does not
    show it.
    """
    drawing = next(d for d in drawings() if d.name == "hidden_interface")
    got = mesh(
        pieces(drawing.name),
        demand_for(drawing),
        profile_for(drawing),
        str(tmp_path),
        drawing.name,
    )
    _, quality, counted, _ = read_back(got.path)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("hidden")
        label_map.fragment(label_map.load(pieces(drawing.name))[0])
        gmsh.option.setNumber("Mesh.MeshSizeMax", drawing.coarsest)
        gmsh.option.setNumber("Mesh.MeshSizeMin", drawing.finest)
        gmsh.model.mesh.generate(3)
        _, tags, _ = gmsh.model.mesh.getElements(2)
        meshed = sum(len(group) for group in tags)
    finally:
        gmsh.finalize()
    assert meshed > counted[2], "the interface reached the file after all"
    assert quality == pytest.approx(got.worst_quality, rel=drawing.round_trip)


def test_a_remainder_is_the_frontier_of_the_meshed_model_and_nothing_else(tmp_path):
    """The drawing the wall label was getting wrong: two solids on a shared
    face.

    Handed over without the label the drawing puts on its outside, and with a
    name for what is left where the model ends, the mesher fills that in off the
    fragmented topology. The face between the two solids is inside the model and
    so is not in it - which is what a caller enumerating a solid's faces cannot
    tell, both solids listing that face as their own.

    The area is compared against the frontier computed from the tetrahedra on
    the other side of those triangles, so the two are arrived at from different
    elements of the same file.
    """
    drawing = next(d for d in drawings() if d.name == "hidden_interface")
    drawn = [piece for piece in pieces(drawing.name) if piece.label != "walls"]
    got = mesh(
        drawn,
        demand_for(drawing),
        profile_for(drawing),
        str(tmp_path),
        drawing.name,
        remainder="walls",
    )
    assert got.labels["walls"].sits == FRONTIER
    outside, shared = frontier_area(got.path)
    assert shared == 0
    assert group_area(got.path, "walls") == pytest.approx(outside, rel=drawing.round_trip)


def test_a_label_carries_the_box_round_its_nodes_and_what_it_holds(tmp_path):
    """What a caller places on a face, as a backend places the line it reads a
    port's voltage along, is read off the label as meshed rather than off the
    drawing, which the fragmenting may have cut. The box is held to the nodes of
    the written file and the measure to the area of its elements, each arrived at
    by an arithmetic of its own."""
    drawing = next(d for d in drawings() if d.name == "hidden_interface")
    drawn = [piece for piece in pieces(drawing.name) if piece.label != "walls"]
    got = mesh(
        drawn,
        demand_for(drawing),
        profile_for(drawing),
        str(tmp_path),
        drawing.name,
        remainder="walls",
    )
    boxes = {}
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(got.path)
        for name, label in got.labels.items():
            _, coordinates = gmsh.model.mesh.getNodesForPhysicalGroup(label.dimension, label.tag)
            nodes = numpy.asarray(coordinates).reshape(-1, 3)
            boxes[name] = (tuple(nodes.min(axis=0)), tuple(nodes.max(axis=0)))
    finally:
        gmsh.finalize()

    # Every label's box, and among them one that stands clear of the origin.
    assert any(any(corner) for corner, _ in boxes.values())
    for name, label in got.labels.items():
        assert (label.lower, label.upper) == boxes[name], name
    walls = got.labels["walls"]
    assert walls.size == pytest.approx(group_area(got.path, "walls"), rel=drawing.round_trip)


@pytest.mark.parametrize("top", [2, 3])
def test_what_stands_inside_the_filled_dimension_is_listed_twice_against_it(top):
    """A face inside a box, or a line inside a rectangle, divides nothing, and the
    kernel embeds it rather than listing it in any boundary. It is on both sides
    of the entity holding it, so it is listed against that entity twice - and
    against nothing else, so it joins that entity to nothing but itself."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        if top == 3:
            holder = occ.addBox(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
            inner = occ.addRectangle(2.0, 2.0, 5.0, 6.0, 6.0)
        else:
            holder = occ.addRectangle(0.0, 0.0, 0.0, 10.0, 10.0)
            inner = occ.addLine(occ.addPoint(2.0, 5.0, 0.0), occ.addPoint(8.0, 5.0, 0.0))
        _, became = occ.fragment([(top, holder)], [(top - 1, inner)])
        occ.synchronize()
        ((_, filled),) = [entity for entity in became[0] if entity[0] == top]
        ((_, standing),) = [entity for entity in became[1] if entity[0] == top - 1]
        assert (top - 1, standing) in gmsh.model.mesh.getEmbedded(top, filled)
        assert label_map.sides(top)[standing] == [filled, filled]
    finally:
        gmsh.finalize()


def test_a_point_or_a_curve_standing_inside_a_volume_is_no_face_of_it():
    """Only what stands inside at the dimension below the filled one is on both
    sides of it. A point and a curve are embedded in a volume too, and their tags
    are counted apart from the faces' - so one read as a face would add a side to
    whichever face carries the same number, and a face where the model ends would
    read as inside it."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        box = occ.addBox(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
        point = occ.addPoint(5.0, 5.0, 5.0)
        line = occ.addLine(occ.addPoint(2.0, 2.0, 2.0), occ.addPoint(8.0, 8.0, 8.0))
        occ.fragment([(3, box)], [(0, point), (1, line)])
        occ.synchronize()
        ((_, volume),) = gmsh.model.getEntities(3)
        embedded = gmsh.model.mesh.getEmbedded(3, volume)
        assert {dim for dim, _ in embedded} == {0, 1}
        faces = sorted(tag for _, tag in gmsh.model.getEntities(2))
        assert label_map.sides(3) == {face: [volume] for face in faces}
    finally:
        gmsh.finalize()


def sharing(path, name):
    """How many tetrahedra hold each triangle a written mesh has under one group.

    Two for a triangle the volume stands on both sides of, one for a triangle
    where the meshed region ends, and none for one no element lies against.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        used = {}
        kinds, every, nodes = gmsh.model.mesh.getElements(3)
        for kind, elements, listed in zip(kinds, every, nodes, strict=True):
            per = gmsh.model.mesh.getElementProperties(kind)[3]
            for row in numpy.array(listed, dtype=int).reshape(len(elements), per):
                for face in FACES_OF_A_TET:
                    corners = tuple(sorted(int(row[i]) for i in face))
                    used[corners] = used.get(corners, 0) + 1
        wanted = next(
            tag
            for dim, tag in gmsh.model.getPhysicalGroups(2)
            if gmsh.model.getPhysicalName(dim, tag) == name
        )
        counts = []
        for entity in gmsh.model.getEntitiesForPhysicalGroup(2, wanted):
            kinds, every, nodes = gmsh.model.mesh.getElements(2, entity)
            for kind, elements, listed in zip(kinds, every, nodes, strict=True):
                per = gmsh.model.mesh.getElementProperties(kind)[3]
                for row in numpy.array(listed, dtype=int).reshape(len(elements), per):
                    counts.append(used.get(tuple(sorted(int(node) for node in row[:3])), 0))
        return counts
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("order", [1, 2])
def test_a_face_standing_inside_a_volume_is_meshed_into_it(order, tmp_path):
    """The kernel embeds a face that does not divide the volume it stands in,
    rather than listing it in the volume's boundary, and the volume is meshed to
    it from both sides. So a label on it is inside the model, and every triangle
    on it is shared by the tetrahedra on either side.

    At the second order as well, curved, because the nodes moved there are
    moved on every surface the mesh holds and an embedded one is among them.
    """
    drawing = next(d for d in drawings() if d.name == "embedded_port")
    got = mesh(
        pieces(drawing.name),
        demand_for(drawing),
        replace(profile_for(drawing), element_order=order, curved=order > 1),
        str(tmp_path),
        drawing.name,
    )
    assert got.labels["port"].sits == INTERIOR
    counts = sharing(got.path, "port")
    assert counts and set(counts) == {2}, sorted(set(counts))


def test_a_remainder_the_labels_leave_nothing_for_names_no_group(tmp_path):
    """Every face where the model ends is claimed, so nothing is left over and
    the answer carries no label of the remainder's name. A face standing
    inside the model is never left over, so it does not change that."""
    drawing = next(d for d in drawings() if d.name == "embedded_port")
    got = mesh(
        pieces(drawing.name),
        demand_for(drawing),
        profile_for(drawing),
        str(tmp_path),
        drawing.name,
        remainder="rest",
    )
    assert "rest" not in got.labels
    assert sorted(got.labels) == ["air", "port", "walls"]


def test_the_same_drawing_without_a_remainder_is_refused_for_the_bare_frontier(tmp_path):
    """The remainder is the caller answering that refusal rather than a way
    round it: with no name for those faces, nothing in the file says what the
    mesh ends against there.
    """
    drawing = next(d for d in drawings() if d.name == "hidden_interface")
    drawn = [piece for piece in pieces(drawing.name) if piece.label != "walls"]
    with pytest.raises(Refused, match="the model ends at these"):
        mesh(drawn, demand_for(drawing), profile_for(drawing), str(tmp_path), drawing.name)


def test_a_remainder_carries_over_to_a_method_that_fills_a_surface(tmp_path):
    """The same mechanism one dimension down, on a drawing the mesher refuses
    today for exactly the thing a remainder answers.

    ``open_shell`` is a shell that does not close, and the rim is where the
    surface ends. A caller filling a surface names its rim the way a caller
    filling a volume names its skin, and the refusal it gets otherwise is the
    same sentence.
    """
    drawing = next(d for d in drawings() if d.name == "open_shell")
    got = mesh(
        pieces(drawing.name),
        demand_for(drawing),
        profile_for(drawing),
        str(tmp_path),
        drawing.name,
        remainder="rim",
    )
    assert got.labels["rim"].dimension == drawing.top - 1
    assert got.labels["rim"].sits == FRONTIER
    assert got.labels["rim"].entities


def test_a_remainder_named_over_a_label_the_caller_drew_is_refused(tmp_path):
    """One group cannot hold both what was drawn and what is left over."""
    drawing = next(d for d in drawings() if d.name == "hidden_interface")
    with pytest.raises(Refused, match="Give one of them another name"):
        mesh(
            pieces(drawing.name),
            demand_for(drawing),
            profile_for(drawing),
            str(tmp_path),
            drawing.name,
            remainder="walls",
        )


def test_a_region_left_empty_is_named_by_where_it_is(tmp_path):
    """A caller cannot find a fragment tag in anything it drew, so the region
    that was left empty is given in the coordinates the shapes arrived in.

    The drawing here is one sound body beside one that passes through itself, so
    the complaint has to pick out which of the two.
    """
    drawing = next(d for d in drawings() if d.name == "partly_unmeshed")
    with pytest.raises(Unmeshed) as raised:
        mesh(
            pieces(drawing.name),
            demand_for(drawing),
            profile_for(drawing),
            str(tmp_path),
            drawing.name,
        )
    said = " ".join(raised.value.complaints)
    assert " at (" in said, f"this reaches nowhere in the drawing: {said}"
    assert "60" in said, f"the crossed body was drawn from x of 60, and this says {said}"


def test_a_sliver_is_reported_in_the_quality_and_refused_nowhere(tmp_path):
    """A body that cannot hold a well-shaped element at any size meshes, fills
    itself, and says what it is in one number.

    Refusing it is not the mesher's to do: a thin conductor is a thing people
    draw, and what a sliver costs the answer depends on the method reading it.
    Saying nothing is not available either, because every other diagnostic of
    the run sits among its siblings'. So the worst element quality comes back,
    and on these drawings it stands orders below what a well-proportioned one
    reaches.
    """
    worst = {}
    for drawing in drawings():
        # A body a few nanometres across meshed to millimetre elements is
        # neither a sliver nor well proportioned: its quality is set by the size
        # asked for, which is the drawing's own business and nothing it is.
        if drawing.expect != MESHED or drawing.top != 3 or "speck" in drawing.tags:
            continue
        got = mesh(
            pieces(drawing.name),
            demand_for(drawing),
            profile_for(drawing),
            str(tmp_path),
            drawing.name,
        )
        worst[drawing.name] = got.worst_quality[3]
    slivers = {name for name, quality in worst.items() if "sliver" in drawn_tags(name)}
    assert slivers, "the corpus carries no sliver to compare against"
    assert max(worst[name] for name in slivers) < min(
        quality for name, quality in worst.items() if name not in slivers
    ), f"a sliver does not separate itself from a sound drawing: {worst}"


def drawn_tags(name):
    """The tags of one drawing, by name."""
    return next(drawing.tags for drawing in drawings() if drawing.name == name)


@pytest.mark.parametrize(
    ("demand", "profile", "complaint"),
    [
        (Demand(coarsest=0.0, finest=0.0), VOLUME, "an element has a size"),
        (Demand(coarsest=-5.0, finest=-5.0), VOLUME, "an element has a size"),
        (Demand(coarsest=-5.0, finest=-5.0), VOLUME, "a length is not less than nothing"),
        (Demand(coarsest=1.0, finest=10.0), VOLUME, "the floor stands above the ceiling"),
        (Demand(coarsest=NOT_A_NUMBER, finest=0.0), VOLUME, "which is not a length"),
        (Demand(coarsest=WITHOUT_END, finest=0.0), VOLUME, "which is not a length"),
        (Demand(coarsest=8.0, finest=NOT_A_NUMBER), VOLUME, "which is not a length"),
        (Demand(coarsest=8.0, finest=2.0), Profile(top=3, element_order=0), "the lowest an"),
        (Demand(coarsest=8.0, finest=2.0), Profile(top=3, element_order=-1), "the lowest an"),
    ],
)
def test_a_request_that_describes_no_element_is_refused_before_gmsh_is_asked(
    demand, profile, complaint, tmp_path
):
    """Gmsh stops on a ceiling of nothing or less and takes the rest without a
    word.

    Where it stops, the message names a quantity internal to it, which reaches a
    user as the name of a variable in somebody else's program. Where it takes
    the request it returns a mesh answering something nobody asked for, and
    :func:`Microwave.Gmsh.coverage.asked` says which of those each case is.
    """
    with pytest.raises(Refused) as raised:
        mesh(pieces("plain_box"), demand, profile, str(tmp_path), "asked")
    assert complaint in " ".join(raised.value.complaints)


@pytest.mark.parametrize(
    ("profile", "complaint"),
    [
        (Profile(top=3, written="stl"), "what is written here is"),
        (Profile(top=3, element_order=3, written="unv"), "writes the rest straight sided"),
        (Profile(top=3, element_order=4, written="unv"), "writes the rest straight sided"),
    ],
)
def test_a_format_that_cannot_hold_the_request_is_refused(profile, complaint, tmp_path):
    """A writer that holds less than Gmsh builds writes what it holds and says
    nothing, so the file answers a request nobody made.

    Measured: UNV keeps element order two and writes anything above as
    straight-sided elements. Every figure the mesher would then report is about
    elements the file does not carry.
    """
    with pytest.raises(Refused) as raised:
        mesh(pieces("plain_box"), DEMAND, profile, str(tmp_path), "formatted")
    assert complaint in " ".join(raised.value.complaints)


@pytest.mark.parametrize("written", sorted(FORMATS))
def test_a_name_of_the_stated_length_survives_the_writer(written, tmp_path):
    """The constant the refusals are built on, measured against the writers.

    Two labels alike as far as a name reaches arrive as one, so the mesher
    refuses a longer one. Written down and never checked, that number is a claim
    about three file formats standing on nothing: shorten it and every refusal
    still passes, lengthen it and labels are mangled with nothing said.

    So a name of exactly the stated length is written and read back, and one
    byte longer is written outside the mesher and read back to show what the
    refusal is for - twice, the second time with a name whose characters are
    not its bytes, since that is where a guard counting the wrong one lets two
    labels through to arrive under one name.
    """
    keeps = FORMATS[written].name_length
    kept = "a" * (keeps - 1) + "Z"
    drawn = pieces("plain_box")
    got = mesh(
        [Piece(kept if p.label == "air" else p.label, p.dimension, p.file) for p in drawn],
        DEMAND,
        Profile(top=3, written=written),
        str(tmp_path),
        "named",
    )
    groups, _, _, _ = read_back(got.path)
    assert kept in groups, f"{written} did not keep a name of {keeps}: {sorted(groups)}"

    over = "a" * keeps + "Z"
    assert _written_outside_the_mesher(over, written, tmp_path) != over, (
        f"{written} kept a name longer than {keeps}, so the refusal guards nothing"
    )

    outside_ascii = "\u00e9" * (keeps // 2) + "Z"
    assert len(outside_ascii) <= keeps, "the specimen has to pass a guard that counts characters"
    assert _written_outside_the_mesher(outside_ascii, written, tmp_path) != outside_ascii, (
        f"{written} keeps a name of {len(outside_ascii)} characters and "
        f"{len(outside_ascii.encode())} bytes, so the length is not measured in bytes"
    )


def _written_outside_the_mesher(label, written, tmp_path):
    """Write one group under one name, straight through Gmsh, and read it back.

    Driven here rather than through the mesher, because the mesher refuses this
    name - and a check made by patching the thing under test checks the patch.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("long")
        gmsh.model.occ.addBox(0, 0, 0, 10, 10, 10)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [1], 1, name=label)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 9.0)
        gmsh.model.mesh.generate(3)
        format = FORMATS[written]
        if format.revision is not None:
            gmsh.option.setNumber("Mesh.MshFileVersion", format.revision)
        path = str(tmp_path / f"long.{format.suffix}")
        gmsh.write(path)
        gmsh.model.remove()
        gmsh.model.add("read")
        gmsh.open(path)
        names = [gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups()]
        return names[0] if names else ""
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("written", sorted(FORMATS))
def test_the_file_holds_the_element_order_the_profile_asked_for(written, tmp_path):
    """The control on the case above, which is what makes the refusal worth
    making.

    Asked for the highest order each format keeps, the file has to come back
    carrying it. A writer that quietly dropped to straight-sided elements would
    pass every other case here: the groups are right, the labels are right, and
    the shape of an element is the one thing nothing else looks at.
    """
    holds = FORMATS[written].holds_order or HIGHEST
    profile = Profile(top=3, element_order=holds, curved=True, written=written)
    got = mesh(pieces("plain_box"), DEMAND, profile, str(tmp_path), written)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(got.path)
        kinds, _, _ = gmsh.model.mesh.getElements(3)
        nodes = {gmsh.model.mesh.getElementProperties(kind)[3] for kind in kinds}
    finally:
        gmsh.finalize()
    assert nodes == {NODES_PER_TET[holds]}, (
        f"{written} was asked for order {holds} and the file holds {nodes}"
    )


#: The format a numbered copy is asked in here. The one whose writer rewrites a
#: name, which is the reason a copy is numbered at all.
COPY = "unv"

#: How a UNV group record says what each of its members is: a node, or an
#: element. Read off files Gmsh wrote with and without its groups of nodes.
UNV_NODE, UNV_ELEMENT = 7, 8


def unv_members(path):
    """What kinds of member the groups in a UNV file hold.

    The groups are the file's dataset 2477. Each group is a line ending in its
    member count, a line naming it, and its members four numbers apiece, of
    which the first says what kind of thing the member is.
    """
    lines = pathlib.Path(path).read_text().splitlines()
    kinds = set()
    at = next(index for index, line in enumerate(lines) if line.strip() == "2477") + 1
    while lines[at].strip() != "-1":
        count = int(lines[at].split()[-1])
        at += 2
        members = []
        while len(members) < 4 * count:
            members.extend(int(word) for word in lines[at].split())
            at += 1
        kinds.update(members[::4])
    return kinds


def test_the_numbered_copy_is_the_mesh_under_its_tags(tmp_path):
    """The second file holds the elements the first holds, curved as the first
    is, with each group under its tag - and the first keeps the labels.

    Taken over a curved interface at the order the copy's format keeps, so a
    copy written before the high-order pass, or from another model, differs in
    the elements' shapes rather than only in their count.
    """
    profile = Profile(top=3, element_order=FORMATS[COPY].holds_order, curved=True)
    got = mesh(pieces("coax"), DEMAND, profile, str(tmp_path), "coax", numbered_as=COPY)
    assert got.numbered.endswith(f".{FORMATS[COPY].suffix}")

    named, quality, counted, spanned = read_back(got.path)
    numbered, copy_quality, copy_counted, copy_spanned = read_back(got.numbered)
    assert named == {label: (on.dimension, on.tag) for label, on in got.labels.items()}
    assert numbered == {str(on.tag): (on.dimension, on.tag) for on in got.labels.values()}
    # One group to a label. A reader handed a group of nodes beside each group of
    # elements finds two groups under a name that was written once.
    assert unv_members(got.numbered) == {UNV_ELEMENT}
    assert copy_counted == counted
    assert copy_quality == pytest.approx(quality, rel=1e-9)
    assert copy_spanned.keys() == spanned.keys()
    for dim, extremes in spanned.items():
        assert copy_spanned[dim] == pytest.approx(extremes, rel=1e-9)


def test_a_label_spelled_as_another_groups_number_leaves_every_number_in_place(tmp_path):
    """A label is whatever the caller typed, and one of them may be a figure. The
    volume here is labelled with the tag the faces' group is given, so the two
    groups carry that name at once on the way to being numbered."""
    drawn = [
        Piece("2" if piece.label == "air" else piece.label, piece.dimension, piece.file)
        for piece in pieces("plain_box")
    ]
    got = mesh(drawn, DEMAND, VOLUME, str(tmp_path), "figures", numbered_as=COPY)
    assert sorted(on.tag for on in got.labels.values()) == [1, 2]
    numbered, _, _, _ = read_back(got.numbered)
    assert numbered == {str(on.tag): (on.dimension, on.tag) for on in got.labels.values()}


def test_a_copy_is_written_in_the_revision_its_format_names(tmp_path):
    """The revision is an option of the session, and the first file may have
    been written in a format that names none."""
    profile = Profile(top=3, written=COPY)
    got = mesh(pieces("plain_box"), DEMAND, profile, str(tmp_path), "first", numbered_as="msh22")
    assert pathlib.Path(got.numbered).read_text().splitlines()[1].startswith("2.2")


def test_no_copy_is_written_unless_one_is_asked_for(tmp_path):
    got = mesh(pieces("plain_box"), DEMAND, VOLUME, str(tmp_path), "alone")
    assert got.numbered == ""
    assert sorted(path.name for path in tmp_path.iterdir()) == [pathlib.Path(got.path).name]


@pytest.mark.parametrize(
    ("profile", "numbered_as", "complaint"),
    [
        (VOLUME, "stl", "what is written here is"),
        (Profile(top=3, element_order=3, curved=True), COPY, "which keeps order 2"),
        (Profile(top=3, written="msh22"), "msh41", "the copy would replace the mesh"),
        (Profile(top=3, written="msh41"), "msh22", "the copy would replace the mesh"),
    ],
)
def test_a_copy_that_cannot_hold_the_mesh_is_refused(profile, numbered_as, complaint, tmp_path):
    """A copy is the same mesh, so it is held to what the first file is held to;
    and it is written beside the first under the same name, so a format sharing
    the first one's suffix would leave the caller holding the copy where it
    asked for the mesh."""
    with pytest.raises(Refused) as raised:
        mesh(pieces("plain_box"), DEMAND, profile, str(tmp_path), "copied", numbered_as=numbered_as)
    assert complaint in " ".join(raised.value.complaints)
    assert list(tmp_path.iterdir()) == []


def test_a_directory_that_is_not_there_is_said_before_the_mesh_is_built(tmp_path):
    """Gmsh writes at the end, so a path that cannot be written to is found after
    every expensive thing has been done and thrown away."""
    with pytest.raises(Refused) as raised:
        mesh(pieces("bare_walls"), DEMAND, VOLUME, str(tmp_path / "nowhere"), "somewhere")
    assert raised.value.complaints == (
        f"the directory to write into is not there: {tmp_path / 'nowhere'}",
    ), "the drawing was read before the directory was looked at"


@pytest.mark.parametrize(
    ("named", "complaint"),
    [
        ("a/path", "is not a file name"),
        ("", "is not a file name"),
    ],
)
def test_a_name_that_is_a_path_is_refused(named, complaint, tmp_path):
    """The name is what the file is called, and it is joined to the directory
    that was checked. A separator in it writes somewhere else."""
    with pytest.raises(Refused) as raised:
        mesh(pieces("plain_box"), DEMAND, VOLUME, str(tmp_path), named)
    assert complaint in " ".join(raised.value.complaints)


def test_a_file_the_caller_named_and_did_not_draw_is_said_in_its_own_terms(tmp_path):
    """Handed to Gmsh, a path that is not there stops the import with a message
    naming the path and nothing else - not the label the caller put on it, and
    not which of several pieces it was."""
    drawn = pieces("plain_box")
    missing = [
        Piece(piece.label, piece.dimension, str(tmp_path / "never_drawn.brep"))
        if piece.label == "walls"
        else piece
        for piece in drawn
    ]
    with pytest.raises(Refused) as raised:
        mesh(missing, DEMAND, VOLUME, str(tmp_path), "missing")
    said = " ".join(raised.value.complaints)
    assert "the file drawn for the label walls is not there" in said


def test_a_dimension_no_mesh_is_made_of_is_said_before_the_drawing_is_read(tmp_path):
    """The dimension to fill is part of the request and needs no drawing, so it
    is answered with the rest of the request.

    Asked after the import instead, it is answered by the coverage questions -
    which are written in that dimension and the one below it, and have nothing
    to say outside them.
    """
    unreadable = tmp_path / "not_a_shape.brep"
    unreadable.write_text("this is not a BREP\n")
    handed = [Piece("air", 3, str(unreadable))]
    with pytest.raises(Refused) as raised:
        mesh(handed, DEMAND, Profile(top=0), str(tmp_path), "flat")
    assert raised.value.complaints == (
        "the profile asks to fill dimension 0, and a mesh is made of 1, 2, 3",
    ), "the drawing was read before the dimension was looked at"


def test_the_request_is_judged_before_the_drawing_is_read(tmp_path):
    """A drawing whose labelling is also wrong hears about the size and nothing
    else.

    Nothing has been imported when the size is judged, so nothing is known
    about the labels - and the coverage questions asked of a map that was never
    built answer that every label bounds nothing, which names the caller's own
    labels and is about none of them.
    """
    with pytest.raises(Refused) as raised:
        mesh(pieces("bare_walls"), Demand(coarsest=0.0, finest=0.0), VOLUME, str(tmp_path), "both")
    assert len(raised.value.complaints) == 1, raised.value.complaints
    assert "an element has a size" in raised.value.complaints[0]


def test_an_order_gmsh_will_not_build_comes_back_as_gmsh_said_it(tmp_path):
    """The largest element order is Gmsh's to say, and it says it by name.

    Nothing here knows where that limit is, and a number written down to stand
    in for it would be a limit of ours claiming to be Gmsh's. So the order goes
    through, Gmsh stops, and what it stopped on comes back with the refusal -
    where a bare exception out of the binding names an element type and a
    number and reaches nothing the caller set.
    """
    profile = Profile(top=3, element_order=BEYOND_GMSH)
    with pytest.raises(Unmeshed) as raised:
        mesh(pieces("plain_box"), DEMAND, profile, str(tmp_path), "ordered")
    assert str(BEYOND_GMSH) in " ".join(raised.value.complaints)
    assert any(line.startswith("Error") for line in raised.value.said)


def test_a_floor_above_the_ceiling_would_otherwise_mesh_to_the_ceiling(tmp_path):
    """The control on the case above, and it is what makes it worth refusing.

    Gmsh is driven here rather than through the mesher, because the mesher
    refuses the request. What it shows is that the floor is not merely ignored:
    the mesh that comes back is the one the ceiling asks for, which is finer
    than the one the caller thought it had asked for.
    """
    fine = _elements_from_gmsh("plain_box", Demand(coarsest=1.0, finest=0.0))
    contradicted = _elements_from_gmsh("plain_box", Demand(coarsest=1.0, finest=10.0))
    coarse = _elements_from_gmsh("plain_box", Demand(coarsest=10.0, finest=0.0))
    assert contradicted == fine, "the floor changed the mesh after all"
    assert contradicted > coarse, "the ceiling is not what was met"


def _elements_from_gmsh(name, demand):
    """How many elements Gmsh makes of one drawing at one size, driven directly."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        label_map.fragment(label_map.load(pieces(name))[0])
        gmsh.option.setNumber("Mesh.MeshSizeMax", demand.coarsest)
        gmsh.option.setNumber("Mesh.MeshSizeMin", demand.finest)
        gmsh.model.mesh.generate(3)
        _, tags, _ = gmsh.model.mesh.getElements(3)
        return sum(len(group) for group in tags)
    finally:
        gmsh.finalize()


def test_a_curve_the_surface_stands_on_both_sides_of_is_inside_it(tmp_path):
    """A sphere's surface closes on a seam, and meets itself there.

    Asked for the boundary of that surface with the entities combined, Gmsh
    cancels the seam against itself and returns it nowhere - so a caller
    labelling it is refused for a curve that bounds nothing. Asked without
    combining, the seam is returned twice, which is what it is: the surface
    stands on both sides of it, and a condition on it is a condition inside the
    meshed region.
    """
    drawing = next(d for d in drawings() if d.name == "sphere_seam")
    got = mesh(
        pieces(drawing.name),
        demand_for(drawing),
        profile_for(drawing),
        str(tmp_path),
        drawing.name,
    )
    assert got.labels["seam"].sits == INTERIOR
    assert got.labels["skin"].sits is None, "a label of the filled dimension bounds nothing"


def test_a_small_body_keeps_its_faces_however_far_off_the_rest_is(tmp_path):
    """A body of a tenth of a millimetre has faces of a hundredth of a square
    millimetre, and the model it is drawn in is a million millimetres across.

    Nothing in the boundary count may be scaled by the model: the drawing is
    then refused by naming faces that bound the only volumes the caller drew.
    What the count is scaled by is each entity's own size, which rim_far_off
    holds a dimension down and this holds here.
    """
    drawing = next(d for d in drawings() if d.name == "small_and_far_apart")
    got = mesh(
        pieces(drawing.name),
        demand_for(drawing),
        VOLUME,
        str(tmp_path),
        drawing.name,
    )
    assert len(got.labels["walls"].entities) == 12, "six faces on each of the two bodies"
    assert got.labels["walls"].sits == FRONTIER


def test_a_curve_of_no_length_is_not_somewhere_the_model_ends(tmp_path):
    """A sphere's own surface is bounded by its two poles, each a curve with no
    length, and each is listed once - which reads exactly like a curve where
    the model ends and no label claims.

    Nothing can be meshed on a point, so a caller asked to label one is asked
    for a thing it cannot draw. What separates the two is the measure, and what
    the kernel returns for a degenerate curve is the rounding on an exact zero.
    """
    got = mesh(pieces("sphere_skin"), DEMAND, Profile(top=2), str(tmp_path), "sphere_skin")
    assert set(got.labels) == {"skin"}


def test_the_wall_of_a_void_is_where_the_model_ends_and_not_its_outside(tmp_path):
    """A body with a void in it ends at the void's wall as much as at its skin,
    and the count says so: each bounds one volume.

    So the word is the topology's. A label holding only cavity walls and one
    holding only the outer skin are the same answer here, and both are where a
    condition on the edge of the meshed region belongs.
    """
    got = mesh(pieces("cavity"), DEMAND, VOLUME, str(tmp_path), "cavity")
    assert got.labels["walls"].sits == FRONTIER
    assert len(got.labels["walls"].entities) == 12, "six faces outside and six within"


def test_a_face_between_two_solids_is_reported_and_not_refused(tmp_path):
    """The wall label that swallows an interface, which the mesher can see and
    cannot decide.

    A perfect conductor on a surface inside the model divides it, and a surface
    inside the model is also exactly what a port is. So the mesher says where
    the label's pieces lie and the layer that knows what the label means acts
    on it.
    """
    got = mesh(pieces("shared_interface"), DEMAND, VOLUME, str(tmp_path), "shared_interface")
    assert got.labels["walls"].sits == BOTH
    assert got.labels["air"].sits is None, "a label of the filled dimension bounds nothing"
    assert got.labels["substrate"].sits is None


def _parts_of(name, top):
    """What a drawing fills, split where nothing below joins it, off the kernel's
    own topology and with no element built."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        drawn, _ = label_map.load(pieces(name))
        label_map.fragment(drawn)
        return coverage.parts(label_map.sides(top), label_map.filled(top))
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("name", [d.name for d in drawings() if "kernel_fails" not in d.tags])
def test_what_comes_apart_is_what_the_drawing_says_comes_apart(name, drawn):
    """Two bodies a hair apart, touching along an edge, touching at a point, or
    standing a long way off are each two parts; two solids on a shared face, a
    part inside a box and a pin inside a sleeve are one.

    The kernel joins two bodies only closer than its own tolerance, so the
    drawing a millionth of a millimetre apart is the one this is for. The
    others are here because each could be read either way: an edge two volumes
    share is where both end rather than a way through.

    Every drawing rather than those that mesh, because the flag is a statement
    about the drawing and the split is asked before an element is built. All
    but those the kernel fails to cut, which are never split.
    """
    drawing = drawn[name]
    found = _parts_of(name, drawing.top)
    assert (len(found) > 1) == drawing.apart, f"{drawing.subject}: {found}"


def _drew(resolved, top):
    """The volumes the labels drew, which is what the mesher asks about."""
    return {tag for pieces in resolved.values() for dim, tag in pieces if dim == top}


def _overlapping(name, top):
    """Each body the fragmenting left standing inside another, with no element built."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        drawn, _ = label_map.load(pieces(name))
        resolved, _ = label_map.fragment(drawn)
        return label_map.overlapping(top, _drew(resolved, top))
    finally:
        gmsh.finalize()


@pytest.mark.parametrize(
    "name", [d.name for d in (*drawings(), *ends()) if "kernel_fails" not in d.tags]
)
def test_a_body_left_uncut_is_found_and_nothing_else_is(name):
    """Over every drawing the kernel cuts, those that end the model at a face
    included.

    A body the kernel leaves standing inside another, under a label of its own
    or in the holder's compound, free in the domain or in a corner of it, is
    found. Its neighbours are not, and each is here for what would have found
    it. A body just large enough to be cut out, a ball of the small body's size
    and a cylinder smaller still have only faces that bound two volumes. The same body cut
    out with a corner at the middle of the domain is where the centre of the
    domain less it falls, and the domain holds that centre with it. On the rest
    a face that bounds one volume has another volume round part or all of its
    rim: the body against the top wall,
    the needle two blocks overlapping along an edge are cut into, a post through
    a slab, a block in a notch, the body at the bottom of a pocket, an L of
    trace, and a lid over a narrow groove. A ring round a post has its centre
    in the post, and a slab in a slot has the block a hair beyond each broad
    face.
    """
    drawing = next(d for d in (*drawings(), *ends()) if d.name == name)
    found = _overlapping(name, drawing.top)
    assert bool(found) == ("uncut" in drawing.tags), f"{drawing.subject}: {found}"


@pytest.mark.parametrize(
    ("name", "at"),
    [("speck_in_a_box", "(13, 8, 4)"), ("speck_at_the_middle_of_a_ball", "(-5.1e-06, ")],
)
def test_a_body_left_uncut_is_the_one_complaint_and_names_the_labels(name, at, tmp_path):
    """Nothing else is asked of the drawing, since everything else reads what
    bounds what. Asked for one region, it would otherwise also be told the body
    comes apart from the rest, and to make the two share a face. The holder is
    not said to stand inside the body where its centre is inside it, as the
    ball's is."""
    with pytest.raises(Refused) as refused:
        mesh(pieces(name), DEMAND, replace(VOLUME, connected=True), str(tmp_path), "speck")
    (said,) = refused.value.complaints
    assert said.startswith("a body drawn for speck stands inside the region drawn for air, at ")
    assert at in said, said


@pytest.mark.parametrize("name", [d.name for d in drawings() if "uncut" in d.tags])
def test_a_body_left_uncut_is_refused_with_a_remainder_named(name, tmp_path):
    """A remainder takes every face that bounds one volume and no label claims,
    which here is each face of the body, so without the refusal the body would
    come back as where the model ends. It is the route a backend writing a
    condition on the frontier takes."""
    with pytest.raises(Refused) as refused:
        mesh(pieces(name), DEMAND, VOLUME, str(tmp_path), "rest", remainder="rest")
    assert any("did not cut it out" in line for line in refused.value.complaints)


@pytest.mark.parametrize(
    ("name", "complaint"),
    [
        ("speck_in_a_box", "did not cut it out"),
        ("speck_in_a_part", "a piece turned inside out there"),
        ("part_inside_out", "is wound inside out"),
    ],
)
def test_a_file_carrying_what_its_label_does_not_declare_is_named_beside_it(
    name, complaint, tmp_path
):
    """The refusals for what the fragmenting would get wrong stand alone except
    for this, which is about the files rather than the fragmented model."""
    handed = [
        *pieces(name),
        *[p for p in pieces("solid_dropped_inside") if p.label == "wire"],
    ]
    with pytest.raises(Refused) as refused:
        mesh(handed, DEMAND, VOLUME, str(tmp_path), "both")
    said = refused.value.complaints
    assert len(said) == 2, said
    assert complaint in said[0] and "does not declare" in said[1], said


def _inside_out(name, fragmented):
    """The volumes wound inside out among those drawn, or among those the
    fragmenting returned, with no element built."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        drawn, _ = label_map.load(pieces(name))
        if not fragmented:
            return label_map.inside_out(entity for _, entity in drawn)
        resolved, _ = label_map.fragment(drawn)
        return label_map.inside_out({entity for got in resolved.values() for entity in got})
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("name", [d.name for d in (*drawings(), *ends())])
def test_a_solid_drawn_inside_out_is_found_and_nothing_else_is(name):
    """Over every drawing as it was drawn. A solid crossing itself, whose two
    lobes cancel to a measure of rounding on zero, is not found."""
    drawing = next(d for d in (*drawings(), *ends()) if d.name == name)
    found = _inside_out(name, fragmented=False)
    assert bool(found) == ("inside_out" in drawing.tags), f"{drawing.subject}: {found}"


@pytest.mark.parametrize("factor", [1e-3, 1e4])
@pytest.mark.parametrize("name", ["part_inside_out", "partly_unmeshed"])
def test_whether_a_solid_is_inside_out_does_not_follow_the_scale_it_is_drawn_at(name, factor):
    """The same drawing enlarged or shrunk about the origin. The solid drawn
    inside out is found at every scale, and the solid crossing itself, whose
    lobes cancel, at none: what its measure rounds on grows with its area and its
    size, and a floor that grew with less would count it once it is large."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        drawn, _ = label_map.load(pieces(name))
        gmsh.model.occ.dilate([entity for _, entity in drawn], 0, 0, 0, factor, factor, factor)
        gmsh.model.occ.synchronize()
        found = label_map.inside_out(entity for _, entity in drawn)
    finally:
        gmsh.finalize()
    assert bool(found) == (name == "part_inside_out"), found


@pytest.mark.parametrize(
    "name",
    [
        d.name
        for d in (*drawings(), *ends())
        if "inside_out" not in d.tags and "kernel_fails" not in d.tags
    ],
)
def test_a_piece_turned_inside_out_is_found_and_nothing_else_is(name):
    """Over every drawing the fragmenting is asked about, which is all but those
    drawn inside out. Each small body the kernel turns a piece round for is
    found: inside a part, inside a cylinder, inside a needle laid across its
    box, and two of them touching under one label and under two. So is nothing
    else, the small bodies it leaves uncut in the other way included."""
    drawing = next(d for d in (*drawings(), *ends()) if d.name == name)
    found = _inside_out(name, fragmented=True)
    assert bool(found) == ("turned" in drawing.tags), f"{drawing.subject}: {found}"


def _on_the_palace_route(name, tmp_path):
    """What the mesher says of a drawing on the route the Palace adapter takes:
    one region asked for, and a remainder named."""
    with pytest.raises(Refused) as refused:
        mesh(
            pieces(name),
            DEMAND,
            replace(VOLUME, connected=True),
            str(tmp_path),
            "palace",
            remainder="rest",
        )
    return refused.value.complaints


@pytest.mark.parametrize(
    "name", [d.name for d in drawings() if {"turned", "inside_out"} & set(d.tags)]
)
def test_what_the_fragmenting_would_get_wrong_is_the_one_complaint_on_the_palace_route(
    name, tmp_path
):
    """Nothing is asked after it, since everything after it reads what bounds
    what. Asked for one region, a drawing the kernel turned a piece round for
    would otherwise be told its parts do not meet, and to make them share a
    face."""
    drawing = next(d for d in drawings() if d.name == name)
    (complaint,) = _on_the_palace_route(name, tmp_path)
    assert drawing.complaint in complaint, complaint


@pytest.mark.parametrize(
    ("name", "said"),
    [
        (
            "speck_in_a_part",
            "the kernel failed to cut the region drawn for air at (10, 7, 3) to (18, 13, 7), "
            "round what was drawn for part, speck: ",
        ),
        (
            "specks_touching_under_two_labels",
            "the kernel failed to cut the region drawn for air at (13, 8, 4) to (13.00002, "
            "8.00001, 4.00001), round what was drawn for one, two: ",
        ),
        ("part_inside_out", "the solid drawn for part at (10, 7, 3) to (18, 13, 7) is wound"),
        ("domain_inside_out", "the solid drawn for air at (-1e-07, -1e-07, -1e-07) to (30, "),
    ],
)
def test_what_the_fragmenting_would_get_wrong_is_named_where_it_is(name, said, tmp_path):
    """A piece turned inside out goes to the label of the region the kernel
    failed to cut and stands where the hole in it should be, so the place is
    the part's, or the small bodies' own where they stand free; what stood
    there is named by the labels round the piece. A solid drawn inside out is
    named by its label and placed where it was drawn."""
    (complaint,) = _on_the_palace_route(name, tmp_path)
    assert complaint.startswith(said), complaint


def test_a_piece_turned_inside_out_is_refused_before_a_body_left_uncut_is_asked_for(tmp_path):
    """Whether a body was left uncut is read off what bounds what, and a piece
    turned inside out makes that false. So a drawing carrying both is told of
    the turned piece alone: here the small body in a part, and another standing
    free in a corner of the domain."""
    stray = [
        replace(piece, label="stray", priority=3)
        for piece in pieces("speck_in_a_corner")
        if piece.label == "speck"
    ]
    with pytest.raises(Refused) as refused:
        mesh([*pieces("speck_in_a_part"), *stray], DEMAND, VOLUME, str(tmp_path), "both")
    (complaint,) = refused.value.complaints
    assert "a piece turned inside out there" in complaint, complaint


def test_a_stray_body_of_the_holders_own_label_is_named_by_it():
    """The body and what holds it were drawn under one label, which is the only
    name there is for either."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("stray")
        drawn, _ = label_map.load(pieces("speck_in_one_compound"))
        resolved, _ = label_map.fragment(drawn)
        (said,) = coverage.uncut(label_map.overlapping(3, _drew(resolved, 3)), resolved, 3)
    finally:
        gmsh.finalize()
    assert said.startswith("a body drawn for air stands inside the region drawn for air"), said


@pytest.mark.parametrize("name", ["crossing_port"])
def test_the_kernel_is_asked_only_about_points_inside_the_volumes_box(name, monkeypatch):
    """The kernel counts points far outside a small cylinder as inside it, and a
    volume holds nothing outside its own box, so a point outside the box is
    answered by the box. Here the box of a volume meets a face of another whose
    centre is outside it."""
    asked = []
    through = gmsh.model.isInside

    def counted(dim, tag, coord, parametric=False):
        if dim == 3:
            asked.append((gmsh.model.getBoundingBox(3, tag), list(coord)))
        return through(dim, tag, coord, parametric=parametric)

    monkeypatch.setattr(gmsh.model, "isInside", counted)
    _overlapping(name, 3)
    for box, point in asked:
        assert all(box[axis] <= point[axis] <= box[axis + 3] for axis in range(3)), (box, point)


def test_a_face_no_other_volumes_box_meets_asks_the_kernel_nothing(monkeypatch):
    """Bodies standing apart in a row. A volume whose box does not meet a face's
    box does not hold the face, so the box answers for the kernel on a drawing
    of many bodies."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("row")
        row = [(3, gmsh.model.occ.addBox(3.0 * n, 0, 0, 1, 1, 1)) for n in range(8)]
        gmsh.model.occ.fragment(row, [])
        gmsh.model.occ.synchronize()
        asked = []
        through = gmsh.model.isInside

        def counted(dim, tag, coord, parametric=False):
            asked.append((dim, tag))
            return through(dim, tag, coord, parametric=parametric)

        monkeypatch.setattr(gmsh.model, "isInside", counted)
        assert label_map.overlapping(3, [tag for _, tag in row]) == []
    finally:
        gmsh.finalize()
    assert asked == []


def test_one_region_asked_for_refuses_a_drawing_that_comes_apart_before_meshing(tmp_path):
    """Refused before an element is built, and each part is named by what was
    drawn over it and by where it is - both bodies carry one label here, so the
    place is what tells them apart."""
    with pytest.raises(Refused) as refused:
        mesh(
            pieces("apart_below_tolerance"),
            DEMAND,
            replace(VOLUME, connected=True),
            str(tmp_path),
            "apart_below_tolerance",
        )
    said = str(refused.value)
    assert "has to be one region, and no face joins these parts" in said
    boxes = [
        [float(value) for value in (near + ", " + far).split(", ")]
        for near, far in re.findall(r"the part drawn as air, at \(([^)]*)\) to \(([^)]*)\)", said)
    ]
    # The kernel pads a box by its own tolerance, and the drawing's corners are
    # whole millimetres, so a corner is compared to far less than the gap and
    # far more than the padding.
    drawn = [[0, 0, 0, 10, 10, 10], [10, 0, 0, 20, 10, 10]]
    assert boxes == [pytest.approx(box, abs=1e-6) for box in drawn], said
    assert not list(tmp_path.iterdir()), "a file was written for a refused drawing"


def test_a_part_of_many_entities_is_named_by_one_box_around_them_all(drawn, tmp_path):
    """The small open shell is five faces and one part. It is named by the box
    the five span together, which is the body that was drawn, rather than by
    any one face of it."""
    drawing = drawn["rim_far_off"]
    with pytest.raises(Refused) as refused:
        mesh(
            pieces(drawing.name),
            demand_for(drawing),
            replace(profile_for(drawing), connected=True),
            str(tmp_path),
            drawing.name,
        )
    said = str(refused.value)
    skin = re.search(r"the part drawn as skin, at \(([^)]*)\) to \(([^)]*)\)", said)
    assert skin is not None, said
    box = [float(value) for value in f"{skin[1]}, {skin[2]}".split(", ")]
    assert box == pytest.approx([0, 0, 0, 1e-4, 1e-4, 1e-4], abs=1e-6), said


def _refusal_over(name, drawn, tmp_path):
    """What the mesher says of one corpus drawing asked to fill one region."""
    drawing = drawn[name]
    with pytest.raises(Refused) as refused:
        mesh(
            pieces(name),
            demand_for(drawing),
            replace(profile_for(drawing), connected=True),
            str(tmp_path),
            name,
        )
    return next(line for line in refused.value.complaints if "has to be one region" in line)


def _closest(said):
    """How close each part is said to come to another, in the order named."""
    return [float(value) for value in re.findall(r"within (\S+) mm of another part", said)]


def _points(said):
    """The first coordinate of each point a part is said to come closest at."""
    return [float(value) for value in re.findall(r"of another part at \(([^,]+),", said)]


def test_a_gap_finer_than_a_box_can_show_is_stated_as_a_distance(drawn, tmp_path):
    """The two bodies are a millionth of a millimetre apart, and at the digits a
    place is written to the two boxes print as meeting. The distance is what
    says the gap is there, and how wide."""
    said = _refusal_over("apart_below_tolerance", drawn, tmp_path)
    assert _closest(said) == [pytest.approx(1e-6, rel=1e-3)] * 2, said


def test_bodies_touching_along_an_edge_are_said_to_come_within_nothing(drawn, tmp_path):
    """Where they touch is where a user would look, and the distance says the
    two parts meet there rather than stop short."""
    said = _refusal_over("touching_at_an_edge", drawn, tmp_path)
    assert _closest(said) == [pytest.approx(0.0, abs=1e-9)] * 2, said


def test_the_closest_pair_is_found_where_the_nearest_boxes_are_not_it(tmp_path):
    """A small box stands in a torus' hole, so their boxes overlap while the
    ring is far off. A second body stands a short way from the box with boxes
    apart. The walk has to go past the pair whose boxes touch to find the
    distance that is the answer."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("nearest")
        box = gmsh.model.occ.addBox(-0.5, -0.5, -0.5, 1.0, 1.0, 1.0)
        ring = gmsh.model.occ.addTorus(0.0, 0.0, 0.0, 10.0, 1.0)
        beside = gmsh.model.occ.addBox(2.5, -0.5, -0.5, 1.0, 1.0, 1.0)
        gmsh.model.occ.synchronize()
        said = label_map.nearest(3, [[box], [ring, beside]])
    finally:
        gmsh.finalize()
    assert _closest(" ".join(said)) == [pytest.approx(2.0, rel=1e-9)] * 2, said
    assert _points(said[0]) == [pytest.approx(0.5, abs=1e-9)], "the point is on the box"
    assert _points(said[1]) == [pytest.approx(2.5, abs=1e-9)], "the point is on the body beside"


@pytest.mark.parametrize("side", [1.0, -1.0], ids=["near above", "near below"])
def test_a_pair_whose_boxes_stand_further_off_than_the_answer_is_not_measured(side, monkeypatch):
    """The kernel measures a pair at a time, and a part of many entities beside
    another is many pairs. A body one side of the box is near, and one on the
    other side far: once the near one is measured, the far one's box already
    stands further off than that, whichever side it is on."""
    measured = []
    measure = gmsh.model.occ.getDistance

    def counted(*pair):
        measured.append(pair)
        return measure(*pair)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("pruned")
        box = gmsh.model.occ.addBox(-0.5, 0.0, 0.0, 1.0, 1.0, 1.0)
        near = gmsh.model.occ.addBox(side * 2.0 - 0.5, 0.0, 0.0, 1.0, 1.0, 1.0)
        far = gmsh.model.occ.addBox(-side * 51.0 - 0.5, 0.0, 0.0, 1.0, 1.0, 1.0)
        gmsh.model.occ.synchronize()
        monkeypatch.setattr(gmsh.model.occ, "getDistance", counted)
        said = label_map.nearest(3, [[box], [far, near]])
    finally:
        gmsh.finalize()
    assert _closest(said[0]) == [pytest.approx(1.0, rel=1e-9)], said
    assert [pair[3] for pair in measured] == [near]


def test_a_part_every_other_part_is_far_from_is_still_measured():
    """Two parts close together and a third far off. The pair between the near
    two cannot tell the far one anything, and the pair that can stands further
    off than the near two already are from each other - so it is measured for
    the far part's sake, which has nothing yet."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("three")
        one = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        near = gmsh.model.occ.addBox(2.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        far = gmsh.model.occ.addBox(-51.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        gmsh.model.occ.synchronize()
        said = label_map.nearest(3, [[one], [near], [far]])
    finally:
        gmsh.finalize()
    assert _closest(" ".join(said)) == [pytest.approx(d, rel=1e-9) for d in (1.0, 1.0, 50.0)]


def _swept_from_a_spline():
    """A solid swept from a spline whose widest point lies between the points it
    passes through, with a body a little way off that bulge and another a
    little further below the solid, whose box stands nearer.

    The kernel's box of such a solid falls short of the bulge, so the body
    beside it has the wider box gap and the smaller distance.
    """
    occ = gmsh.model.occ
    through = [(1.0, 0.0), (1.4, 0.3), (1.45, 0.7), (1.0, 1.0)]
    points = [occ.addPoint(x, 0.0, z) for x, z in through]
    spline = occ.addSpline(points)
    occ.synchronize()
    low, high = gmsh.model.getParametrizationBounds(1, spline)
    along = [low[0] + (high[0] - low[0]) * step / 4000 for step in range(4001)]
    bulge = max((gmsh.model.getValue(1, spline, [at]) for at in along), key=lambda at: at[0])

    # Turned a sixth of a turn back and swept a third of one, so the bulge is
    # met halfway through the sweep, where it was drawn.
    face = occ.addPlaneSurface([occ.addCurveLoop([spline, occ.addLine(points[-1], points[0])])])
    occ.rotate([(2, face)], 0, 0, 0, 0, 0, 1, -math.pi / 3)
    swept = occ.revolve([(2, face)], 0, 0, 0, 0, 0, 1, 2 * math.pi / 3)
    beside = occ.addBox(bulge[0] + 4e-5, -0.05, bulge[2] - 0.05, 0.1, 0.1, 0.1)
    below = occ.addBox(0.95, -0.05, -1.0, 0.1, 0.1, 1.0 - 6e-5)
    occ.synchronize()
    return next(tag for dim, tag in swept if dim == 3), beside, below


def test_a_box_that_falls_short_of_its_entity_does_not_hide_the_closest_pair():
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("swept")
        swept, beside, below = _swept_from_a_spline()
        box = gmsh.model.getBoundingBox(3, swept)
        near = gmsh.model.getBoundingBox(3, beside)
        assert near[0] - box[3] > gmsh.model.occ.getDistance(3, swept, 3, beside)[0], (
            "the specimen no longer has a box falling short of its bulge"
        )
        said = label_map.nearest(3, [[swept], [beside, below]])
    finally:
        gmsh.finalize()
    assert _closest(said[0]) == [pytest.approx(4e-5, rel=1e-3)], said


@pytest.mark.parametrize("fails", ["raises", "answers less than nothing"])
def test_a_distance_the_kernel_could_not_measure_is_not_said(fails, monkeypatch):
    """Gmsh reports a failed distance as a negative one, and raises on an entity
    it cannot measure. Neither is a distance, so the part is given no sentence
    rather than one about a gap of minus one millimetre."""

    def failing(*pair):
        if fails == "raises":
            raise Exception("could not compute distance")
        return (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("unmeasured")
        one = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        other = gmsh.model.occ.addBox(2.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        gmsh.model.occ.synchronize()
        monkeypatch.setattr(gmsh.model.occ, "getDistance", failing)
        said = label_map.nearest(3, [[one], [other]])
    finally:
        gmsh.finalize()
    assert said == ["", ""]


def test_one_region_asked_for_leaves_a_drawing_that_holds_together_as_it_was(tmp_path):
    name = "shared_interface"
    (tmp_path / "loose").mkdir()
    loose = mesh(pieces(name), DEMAND, VOLUME, str(tmp_path / "loose"), name)
    held = mesh(pieces(name), DEMAND, replace(VOLUME, connected=True), str(tmp_path), name)
    assert held.labels == loose.labels


def test_a_drawn_face_where_a_solid_already_has_one_is_unified(tmp_path):
    """A wall label costs the drawing of the face and nothing else.

    The box is drawn, and its own six faces are drawn again beside it. If the
    two were kept apart the mesh would carry a duplicate surface and the volume
    would be bounded by faces no label claims.
    """
    got = mesh(pieces("plain_box"), DEMAND, VOLUME, str(tmp_path), "plain_box")
    assert len(got.labels["walls"].entities) == 6
    assert got.labels["walls"].sits == FRONTIER
    assert len(got.labels["air"].entities) == 1


def test_a_label_follows_its_shapes_through_being_cut_up(tmp_path):
    """Six drawn faces reach the mesh as more than six, because a face standing
    inside the solid cuts four of them in half. Nothing matches them back: the
    fragmenting says which pieces each input became."""
    got = mesh(pieces("split_cavity"), DEMAND, VOLUME, str(tmp_path), "split_cavity")
    assert len(got.labels["walls"].entities) == 10, "four of the six drawn faces are halved"
    assert len(got.labels["air"].entities) == 2, "the port face cuts the volume in two"
    assert got.labels["port"].sits == INTERIOR


def test_one_drawn_shape_is_the_identity_and_not_an_empty_map():
    """Fewer than two entities and Gmsh fragments nothing, returning no map at
    all - so a resolver reading the returned map loses every label on the
    simplest drawing there is.

    Asked below the front door, because the same drawing is refused for its
    unclaimed exterior and the refusal would hide this.
    """
    gmsh.initialize()
    try:
        gmsh.model.add("lone_box")
        drawn, dropped = label_map.load(pieces("lone_box"))
        resolved = label_map.fragment(drawn)[0]
    finally:
        gmsh.finalize()
    assert dropped == []
    assert list(resolved) == ["air"]
    assert len(resolved["air"]) == 1


@pytest.mark.parametrize("written", sorted(FORMATS))
def test_a_tag_is_counted_once_for_the_whole_mesh(written, tmp_path):
    """Gmsh would take a tag unique only within its dimension, and one of the
    formats written here has a single namespace for group numbers.

    Number a surface group and a volume group both one, write UNV, and the file
    comes back with one name on both - so a caller looking its own label up in
    the file finds another label's elements under it.
    """
    profile = Profile(top=3, written=written)
    got = mesh(pieces("split_cavity"), DEMAND, profile, str(tmp_path), written)
    tags = [where.tag for where in got.labels.values()]
    assert len(set(tags)) == len(tags), f"two groups share a number: {got.labels}"
    groups, _, _, _ = read_back(got.path)
    assert set(groups) == set(got.labels), f"{written} lost a label: {sorted(groups)}"


def test_a_label_the_written_format_cannot_carry_is_refused(tmp_path):
    """A label is a string the caller chose, and two of them the formats mangle
    into one. Refused rather than written, because what comes back then is
    another label's elements under this one's name."""
    with pytest.raises(Refused) as raised:
        drawn = pieces("plain_box")
        mesh(
            [
                Piece('a"b' if piece.label == "air" else piece.label, piece.dimension, piece.file)
                for piece in drawn
            ],
            DEMAND,
            VOLUME,
            str(tmp_path),
            "quoted",
        )
    assert "cuts the name at" in " ".join(raised.value.complaints)


def test_a_drawing_that_never_fills_a_volume_is_the_same_code(tmp_path):
    """The refusals are stated in the dimension being filled and the one below
    it, so a method that meshes the surface of a closed body and never fills
    anything asks the same questions.

    On a closed surface every curve bounds two faces, so the question about an
    outside asks nothing - and the drawing passes with one label and no wall
    label at all, which under a volume profile it could not.
    """
    got = mesh(pieces("closed_shell"), DEMAND, Profile(top=2), str(tmp_path), "closed_shell")
    assert set(got.labels) == {"skin"}
    assert got.labels["skin"].dimension == 2
    assert set(got.worst_quality) == {2}, "nothing of a dimension nobody asked for"
    assert set(got.algorithm) == {2}, "an algorithm for a dimension nobody asked for"
    _, _, counted, _ = read_back(got.path)
    assert counted[3] == 0, "the volume was filled where the profile said not to fill it"


def test_the_profile_carries_a_value_the_first_backend_never_turns(tmp_path):
    """Straight-sided and curved are the same drawing, the same element size and
    the same order, and they are not the same mesh.

    The value has no unit a user could check and no reason for a user to hold an
    opinion about: a straight-sided mesh into a volume method is legal where the
    same mesh into a method working on the surface of a closed body is a
    refusal.
    """
    flat = Profile(top=3, element_order=2, curved=False)
    curved = Profile(top=3, element_order=2, curved=True)
    straight = mesh(pieces("coax"), DEMAND, flat, str(tmp_path), "flat")
    bent = mesh(pieces("coax"), DEMAND, curved, str(tmp_path), "curved")
    assert _on_the_drawn_radius(bent.path) > _on_the_drawn_radius(straight.path), (
        "curving put no node on the surface it was drawn on"
    )


def _worst_without_the_optimising_pass(name, demand, order):
    """Mesh the same model with the nodes placed on the surface and left there.

    Driven here rather than by turning the package's own knob off, because a
    comparison made by patching the thing under test compares the patch. The map
    is the package's, since what this is about is the meshing options.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        label_map.fragment(label_map.load(pieces(name))[0])
        gmsh.option.setNumber("Mesh.MeshSizeMax", demand.coarsest)
        gmsh.option.setNumber("Mesh.MeshSizeMin", demand.finest)
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
        gmsh.model.mesh.generate(3)
        _, tags, _ = gmsh.model.mesh.getElements(3)
        every = [tag for group in tags for tag in group]
        return float(min(gmsh.model.mesh.getElementQualities(every, QUALITY)))
    finally:
        gmsh.finalize()


def test_a_curved_element_is_measured_between_its_corners(tmp_path):
    """A second-order element carries a node in the middle of every edge.

    Measured to those, every edge reads as half of itself and the reported
    range is about something the drawing does not have. So the corners are what
    is measured, and the file is asked with arithmetic of its own.
    """
    curved = Profile(top=3, element_order=2, curved=True)
    got = mesh(pieces("coax"), DEMAND, curved, str(tmp_path), "curved_edges")
    _, _, _, spanned = read_back(got.path)
    assert got.edges.shortest == pytest.approx(spanned[3][0], rel=1e-9, abs=0.0)
    assert got.edges.longest == pytest.approx(spanned[3][1], rel=1e-9, abs=0.0)


def test_curving_asks_for_the_pass_that_follows_it(tmp_path):
    """Placing a node on a drawn surface is one option and repairing what that
    does to the element is another, and asking for the first alone leaves the
    worst element worse than it need be.

    So the package asks for both together, and a caller has no way to take one.
    """
    curved = Profile(top=3, element_order=2, curved=True)
    got = mesh(pieces("coax"), DEMAND, curved, str(tmp_path), "curved_pass")
    alone = _worst_without_the_optimising_pass("coax", DEMAND, 2)
    assert got.worst_quality[3] > alone, "the pass that follows the placement was not asked for"


def test_curving_does_nothing_where_there_is_no_node_to_curve(tmp_path):
    """The control on the case above, and the stronger half of it.

    At element order one there is no node between the corners, so the value has
    nothing to move and the two meshes have to be the same. A knob that changed
    the answer here would be changing something other than what it says.
    """
    flat = Profile(top=3, element_order=1, curved=False)
    curved = Profile(top=3, element_order=1, curved=True)
    straight = mesh(pieces("coax"), DEMAND, flat, str(tmp_path), "flat_first")
    bent = mesh(pieces("coax"), DEMAND, curved, str(tmp_path), "curved_first")
    assert straight.worst_quality == bent.worst_quality
    on_the_surface = _on_the_drawn_radius(straight.path)
    assert on_the_surface > 0, "no node lies on the drawn surface at all, so nothing is compared"
    assert _on_the_drawn_radius(bent.path) == on_the_surface


#: The radius of the coaxial specimen's outer wall, read off the drawing that
#: made it rather than written again. A node lying on that circle is a node on
#: the surface rather than on a chord between two corners of an element.
DRAWN_RADIUS = COAX_RADIUS


def _on_the_drawn_radius(path):
    """How many of the mesh's nodes lie on the drawn cylinder."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        _, coordinates, _ = gmsh.model.mesh.getNodes()
        places = coordinates.reshape(-1, 3)
        return sum(1 for x, y, _ in places if abs(math.hypot(x, y) - DRAWN_RADIUS) < 1e-9)
    finally:
        gmsh.finalize()


def test_each_half_of_the_demand_reaches_the_mesh_on_its_own(tmp_path):
    """One is varied at a time, so neither can be answered by the other.

    Both are Gmsh options over the whole model rather than a policy, and this is
    the range where the request decides: ask for an element coarser than the
    drawing's own smallest feature and the drawing decides instead, which is a
    property of the drawing.
    """
    coarse = mesh(pieces("plain_box"), Demand(10.0, 0.75), VOLUME, str(tmp_path), "coarse")
    fine = mesh(pieces("plain_box"), Demand(3.0, 0.75), VOLUME, str(tmp_path), "fine")
    _, _, few, _ = read_back(coarse.path)
    _, _, many, _ = read_back(fine.path)
    assert many[3] > few[3], "the coarsest element asked for changed nothing"

    floored = mesh(pieces("plain_box"), Demand(10.0, 4.0), VOLUME, str(tmp_path), "floored")
    _, _, fewer, _ = read_back(floored.path)
    assert few[3] > fewer[3], "the finest element asked for changed nothing"


def test_the_size_reported_answers_the_size_asked_for(tmp_path):
    """The demand is a length, so what says whether it was met is a length.

    Each half is varied on its own again, and each is scored on the end of the
    reported range it governs: the ceiling on the longest edge, the floor on
    the shortest. A count of elements cannot answer this - the same count
    arrives from a mesh graded differently - and the count is what a reader has
    without this.
    """
    coarse = mesh(pieces("plain_box"), Demand(10.0, 0.0), VOLUME, str(tmp_path), "coarse")
    fine = mesh(pieces("plain_box"), Demand(3.0, 0.0), VOLUME, str(tmp_path), "fine")
    assert fine.edges.longest < coarse.edges.longest, "the ceiling moved no reported edge"

    floored = mesh(pieces("plain_box"), Demand(10.0, 4.0), VOLUME, str(tmp_path), "floored")
    assert floored.edges.shortest > coarse.edges.shortest, "the floor moved no reported edge"
    assert floored.edges.shortest <= floored.edges.longest


@pytest.mark.parametrize("written", sorted(FORMATS))
def test_the_file_is_written_in_the_format_asked_for(written, tmp_path):
    """A reader wanting one revision of Gmsh's own format will not take the
    other, so which was written is not a detail the mesher may choose."""
    profile = Profile(top=3, written=written)
    got = mesh(pieces("plain_box"), DEMAND, profile, str(tmp_path), written)
    assert got.path.endswith(f".{FORMATS[written].suffix}")
    digits = "".join(letter for letter in written if letter.isdigit())
    if digits:
        stated = ".".join(digits)
        first = pathlib.Path(got.path).read_text().splitlines()[1]
        assert first.startswith(stated), f"{written} was written as {first}"


def test_it_leaves_a_session_it_did_not_open(tmp_path):
    """Gmsh's own initialize is not counted, so a bare finalize in this package
    would tear down whatever the caller had open - and a caller that meshes two
    drawings in one session would lose the first.

    A session is more than the models in it. Removing a model makes another of
    them current, and a Gmsh option belongs to the session rather than to a
    model, so a caller that meshes between two of its own calls would find both
    moved under it and nothing said.
    """
    gmsh.initialize()
    try:
        gmsh.model.add("the_caller_s_own")
        gmsh.model.add("and_another")
        gmsh.model.setCurrent("the_caller_s_own")
        held = {option: 1.0 + number for number, option in enumerate(TURNED)}
        for option, value in held.items():
            gmsh.option.setNumber(option, value)

        mesh(pieces("plain_box"), DEMAND, VOLUME, str(tmp_path), "first")
        assert gmsh.isInitialized()
        assert sorted(gmsh.model.list()) == ["", "and_another", "the_caller_s_own"], (
            f"the mesher's own model was left in the session: {gmsh.model.list()}"
        )
        assert gmsh.model.getCurrent() == "the_caller_s_own", "the current model moved"
        for option, value in held.items():
            assert gmsh.option.getNumber(option) == value, f"{option} moved"

        mesh(pieces("plain_box"), DEMAND, VOLUME, str(tmp_path), "second")
        assert gmsh.isInitialized()
        assert gmsh.model.getCurrent() == "the_caller_s_own"
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def test_a_borrowed_sessions_scaling_does_not_reach_the_file(tmp_path):
    """Gmsh's writer multiplies every coordinate by an option of the session.

    A caller that set it for its own work would otherwise get a mesh of a
    device the size it asked for and a solve of one that size times the factor,
    with nothing said. The written file is in the units the shapes were drawn
    in, so the option is set on the way out and put back afterwards.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("Mesh.ScalingFactor", 2.0)
        got = mesh(pieces("plain_box"), DEMAND, VOLUME, str(tmp_path), "scaled")
        assert gmsh.option.getNumber("Mesh.ScalingFactor") == 2.0, "the caller's value moved"
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
    _, _, _, spanned = read_back(got.path)
    assert got.edges.longest == pytest.approx(spanned[3][1], rel=1e-9, abs=0.0)


def test_a_borrowed_sessions_writer_options_do_not_reach_the_files(tmp_path):
    """What a writer keeps besides the grouped elements is an option of the
    session as well: every element or only those in a group, and a group of
    nodes beside each group of elements or not.

    Either reaching the file makes it answer another question. Elements in no
    group are elements no label says anything about, and a reader handed a
    group of nodes as well finds two groups where one label was written.
    """
    alone = mesh(pieces("split_cavity"), DEMAND, VOLUME, str(tmp_path), "alone", numbered_as=COPY)
    theirs = {
        "Mesh.SaveAll": 1,
        "Mesh.SaveGroupsOfNodes": 1,
        "Mesh.SaveGroupsOfElements": 0,
    }
    gmsh.initialize()
    try:
        for option, value in theirs.items():
            gmsh.option.setNumber(option, value)
        got = mesh(
            pieces("split_cavity"), DEMAND, VOLUME, str(tmp_path), "borrowed", numbered_as=COPY
        )
        for option, value in theirs.items():
            assert gmsh.option.getNumber(option) == value, f"the caller's {option} moved"
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
    for written, lent in ((alone.path, got.path), (alone.numbered, got.numbered)):
        groups, _, counted, _ = read_back(written)
        assert read_back(lent)[0] == groups
        assert read_back(lent)[2] == counted


def test_in_a_borrowed_session_it_repeats_nothing_of_the_callers_log(tmp_path):
    """A logger belongs to the session, so a caller's own holds everything since
    the caller started it.

    Read without asking whose it is, that puts another run's lines on this run's
    refusal - and the caller is the one who can read them, because the caller
    has the terminal. A warning rather than an error, because Gmsh raises on the
    next call it is given while a last error stands.
    """
    gmsh.initialize()
    try:
        gmsh.logger.start()
        gmsh.logger.write("a warning from something else entirely", "warning")
        drawing = next(d for d in drawings() if d.name == "self_intersecting")
        with pytest.raises(Unmeshed) as raised:
            mesh(
                pieces(drawing.name),
                demand_for(drawing),
                profile_for(drawing),
                str(tmp_path),
                drawing.name,
            )
        assert raised.value.said == (), f"this is the caller's log: {raised.value.said}"
    finally:
        gmsh.logger.stop()
        gmsh.finalize()


def test_handing_over_nothing_is_refused_rather_than_answered(tmp_path):
    """An empty drawing meshes to an empty file, which is a silent no-op with a
    path on it."""
    with pytest.raises(Refused) as raised:
        mesh([], DEMAND, VOLUME, str(tmp_path), "nothing")
    assert "nothing to mesh" in " ".join(raised.value.complaints)


def test_what_built_it_comes_back_with_it(tmp_path):
    """The Gmsh and the algorithm, because both are choices made below the
    caller: a drawing, an element size and a profile do not fix which algorithm
    ran, and the elements differ with it."""
    got = mesh(pieces("plain_box"), DEMAND, VOLUME, str(tmp_path), "plain_box")
    gmsh.initialize()
    try:
        running = gmsh.option.getString("General.Version")
        asked = {dim: int(gmsh.option.getNumber(name)) for dim, name in ALGORITHM.items()}
    finally:
        gmsh.finalize()
    assert got.version == running, "the version reported is not the Gmsh that is here"
    assert got.algorithm == asked, "the algorithm reported is not the one the option holds"


#: WR-42 in mm, run along x: the broad wall along y and the narrow one along z.
GUIDE = (20.0, 10.7, 4.3)


def _quad(occ, corners):
    """A flat face through four corners, in the order they are given."""
    points = [occ.addPoint(*corner) for corner in corners]
    lines = [occ.addLine(points[i], points[(i + 1) % 4]) for i in range(4)]
    return occ.addPlaneSurface([occ.addCurveLoop(lines)])


def _port_face(occ, x):
    """The guide's cross-section at ``x``."""
    length, broad, narrow = GUIDE
    return _quad(occ, [(x, 0.0, 0.0), (x, broad, 0.0), (x, broad, narrow), (x, 0.0, narrow)])


def _flat(occ, x0, x1, y0, y1, z):
    """A horizontal sheet."""
    return _quad(occ, [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)])


def _written(tmp_path, name, dimension, build):
    """One shape drawn by the kernel and written to a file of its own, as a
    piece of the drawing."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        path = tmp_path / f"{name}.brep"
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return Piece(name, dimension, str(path))


def _fragmented(*builds):
    """Each shape drawn and fragmented with the rest, as the mesher fragments a
    drawing; the entities each shape became, by dimension and tag."""
    occ = gmsh.model.occ
    drawn = [build(occ) for build in builds]
    _, became = occ.fragment(drawn[:1], drawn[1:])
    occ.synchronize()
    return became


def test_what_stands_on_each_labelled_face_comes_back_with_the_mesh(tmp_path):
    """A sheet across the guide running into the port's face divides that face in
    two, and the curve between the halves is the sheet's own edge - which is
    the one fact a caller has to read off the mesh to know the port is two
    guides. Every face a label holds is listed, the remainder's included."""
    length, broad, narrow = GUIDE
    got = mesh(
        [
            _written(tmp_path, "air", 3, lambda occ: occ.addBox(0.0, 0.0, 0.0, *GUIDE)),
            _written(tmp_path, "port", 2, lambda occ: _port_face(occ, 0.0)),
            _written(
                tmp_path, "sheet", 2, lambda occ: _flat(occ, 0.0, length, 0.0, broad, narrow / 2)
            ),
        ],
        DEMAND,
        VOLUME,
        str(tmp_path),
        "divided",
        remainder="wall",
    )
    on_face = got.rims[2]
    assert set(on_face) == {
        face for label in got.labels.values() if label.dimension == 2 for face in label.entities
    }
    below, above = got.labels["port"].entities
    (between,) = set(on_face[below]) & set(on_face[above])
    (sheet,) = got.labels["sheet"].entities
    assert between in on_face[sheet]
    assert all(len(got.rims[1][curve]) == 2 for _, curve in on_face[below])


def test_each_curve_of_a_labelled_face_says_where_it_runs(tmp_path):
    """The port's face is square to x, so each of its curves spans y or z and
    nothing along x - and its sides along y span the broad wall's width, which
    the tag alone cannot say. A shape carrying a coarse tolerance is drawn by
    the CAD kernel, and ``tests/palace_lumped_port_probe.py`` meshes one."""
    length, broad, narrow = GUIDE

    def guide(occ):
        return occ.addBox(0.0, 0.0, 0.0, *GUIDE)

    got = mesh(
        [
            _written(tmp_path, "air", 3, guide),
            _written(tmp_path, "port", 2, lambda occ: _port_face(occ, 0.0)),
        ],
        DEMAND,
        VOLUME,
        str(tmp_path),
        "bounds",
        remainder="wall",
    )
    (face,) = got.labels["port"].entities
    boxes = [got.bounds[1][curve] for _, curve in got.rims[2][face]]
    spans = [[high[i] - low[i] for i in range(3)] for low, high in boxes]
    assert set(got.bounds[1]) == set(got.rims[1])
    assert all(span[0] == 0.0 for span in spans)
    assert sorted(round(max(span), 9) for span in spans) == [narrow, narrow, broad, broad]


def test_a_strip_clear_of_the_walls_stands_on_the_ports_face_touching_nothing_there():
    """Its edge is embedded in the face rather than dividing it, and is listed
    on it all the same, with end points no curve round the face shares."""
    length, broad, narrow = GUIDE
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        became = _fragmented(
            lambda occ: (3, occ.addBox(0.0, 0.0, 0.0, *GUIDE)),
            lambda occ: (2, _port_face(occ, 0.0)),
            lambda occ: (2, _flat(occ, 0.0, length / 2, 3.0, broad - 3.0, narrow / 2)),
        )
        ((_, port),) = became[1]
        ((_, strip),) = became[2]
        found = label_map.rims([port, strip], 3)
        (edge,) = set(found[2][port]) & set(found[2][strip])
        around = [curve for _, curve in found[2][port] if (1, curve) != edge]
        assert not set(found[1][edge[1]]) & {end for c in around for end in found[1][c]}
    finally:
        gmsh.finalize()


def test_a_closed_curve_is_bounded_by_its_one_point():
    """Asked combined, the kernel drops a point listed twice with opposite
    orientation, and a circle is bounded by its one point that way - so the
    edge of a round port would end nowhere and join nothing."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        became = _fragmented(
            lambda occ: (3, occ.addCylinder(0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 5.0)),
            lambda occ: (2, occ.addDisk(0.0, 0.0, 0.0, 5.0, 5.0, zAxis=[1.0, 0.0, 0.0])),
        )
        ((_, port),) = became[1]
        found = label_map.rims([port], 3)
        ((_, circle),) = found[2][port]
        assert len(found[1][circle]) == 1
    finally:
        gmsh.finalize()


def test_a_point_left_on_a_face_is_listed_on_it_as_a_point():
    """A sheet touching the port's face at one corner leaves a point embedded
    in it, and that point is where the sheet's own edges end. A point's number
    is counted apart from a curve's, so it is listed with its dimension."""
    length, broad, narrow = GUIDE
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        became = _fragmented(
            lambda occ: (3, occ.addBox(0.0, 0.0, 0.0, *GUIDE)),
            lambda occ: (2, _port_face(occ, 0.0)),
            lambda occ: (
                2,
                _quad(
                    occ,
                    [
                        (0.0, broad / 2, narrow / 2),
                        (5.0, broad / 2 - 2.0, narrow / 2),
                        (10.0, broad / 2, narrow / 2),
                        (5.0, broad / 2 + 2.0, narrow / 2),
                    ],
                ),
            ),
        )
        ((_, port),) = became[1]
        ((_, sheet),) = became[2]
        left = [(dim, tag) for dim, tag in gmsh.model.mesh.getEmbedded(2, port) if dim == 0]
        assert left, "the corner left nothing standing on the face"
        bounding = gmsh.model.getBoundary([(2, port)], combined=False, oriented=False)
        found = label_map.rims([port, sheet], 3)
        assert set(found[2][port]) == {(1, abs(tag)) for _, tag in bounding} | set(left)
        corners = {end for _, edge in found[2][sheet] for end in found[1][edge]}
        assert set(left) <= corners
    finally:
        gmsh.finalize()
