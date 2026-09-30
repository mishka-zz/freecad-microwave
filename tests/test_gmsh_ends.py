# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A face handed a way into the model, and what the mesher leaves out behind it.

The decision is asked of maps written by hand, where each refusal's reasoning
is pinned with nothing else in the way, and again of drawings FreeCAD made,
where the kernel says which side of a face each volume stands on and the mesh
written afterwards is measured - so what is asked there is whether the part
left out is the run-on the drawing carries and nothing more.

The drawings are ``tests.drawings.ends``, written by ``tests/drawings_probe.py``
under a real FreeCAD and checked in. Needs Gmsh for the second half and skips it
without one.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
from dataclasses import replace

import pytest

from Microwave.Gmsh import ends
from Microwave.Gmsh.ends import AHEAD, BEHIND, UNTOLD
from Microwave.Gmsh.vocabulary import FRONTIER, Piece
from tests.conftest import needed
from tests.drawings import (
    COAX_INSET,
    COAX_LINE,
    GUIDE,
    GUIDE_INSET,
    MESHED,
    REFUSED,
)
from tests.drawings import (
    ends as drawn_ends,
)

#: Where the probe wrote them.
DRAWN = pathlib.Path(__file__).resolve().parent / "_drawings"


def _where(dimension, tags):
    return f"{dimension}:{','.join(str(tag) for tag in tags)}"


def piece(label, inward=None, dimension=2):
    return Piece(label=label, dimension=dimension, file=f"/{label}.brep", inward=inward)


class TestWhatACallerMayStateAsAWayIn:
    def test_each_labels_way_in_is_read_once(self):
        facing, said = ends.stated(
            [piece("air", dimension=3), piece("p", (1, 0, 0)), piece("p", (1, 0, 0))], 3
        )
        assert facing == {"p": (1.0, 0.0, 0.0)}
        assert said == []

    @pytest.mark.parametrize(
        ("inward", "complaint"),
        [
            ((1.0, 0.0), "three finite numbers"),
            ((1.0, math.nan, 0.0), "three finite numbers"),
            ((1.0, math.inf, 0.0), "three finite numbers"),
            ("x", "three finite numbers"),
            ((0.0, 0.0, 0.0), "no length"),
        ],
    )
    def test_a_direction_that_is_not_one_is_refused(self, inward, complaint):
        _, said = ends.stated([piece("p", inward)], 3)
        assert len(said) == 1 and complaint in said[0] and "p" in said[0]

    def test_a_way_in_is_stated_where_the_model_ends(self):
        """A volume label handed one names a place the model does not end at."""
        _, said = ends.stated([piece("air", (1, 0, 0), dimension=3)], 3)
        assert "dimension 3" in said[0] and "ends at dimension 2" in said[0]

    def test_a_way_in_is_read_only_into_a_volume(self):
        _, said = ends.stated([piece("rim", (1, 0, 0), dimension=1)], 2)
        assert "fills dimension 2" in said[0]

    def test_a_way_in_on_some_of_a_labels_pieces_only_is_refused(self):
        facing, said = ends.stated([piece("p", (1, 0, 0)), piece("p")], 3)
        assert facing == {}
        assert said == [
            "the label p is handed a way into the model on some of its pieces and not on "
            "others, and a label has one"
        ]

    def test_a_truth_value_is_not_a_direction(self):
        _, said = ends.stated([piece("p", (True, 0, 0))], 3)
        assert "three finite numbers" in said[0]

    def test_two_ways_in_for_one_label_are_refused(self):
        facing, said = ends.stated([piece("p", (1, 0, 0)), piece("p", (-1, 0, 0))], 3)
        assert said == [
            "the label p is handed more than one way into the model, and a label has one"
        ]


#: A guide cut by two planes into three volumes along it: 1 behind face 10,
#: 2 between, 3 behind face 20. Faces 30 to 33 are its walls, each on one volume.
GUIDE_SIDES = {10: [1, 2], 20: [2, 3], 30: [1], 31: [2], 32: [3], 33: [2]}
IN_ORDER = {10: {1: BEHIND, 2: AHEAD}, 20: {2: AHEAD, 3: BEHIND}}
PORTS = {10: "p1", 20: "p2"}
AIR = {1: frozenset({"air"}), 2: frozenset({"air"}), 3: frozenset({"air"})}
ONE_BODY = [("air", [1, 2, 3])]


def slab(start, stop):
    """The corners of a stretch of guide of unit cross-section along x."""
    return [(x, y, z) for x in (start, stop) for y in (0.0, 1.0) for z in (0.0, 1.0)]


#: What the kernel would measure of that guide: each volume a stretch of unit
#: cross-section, each face of unit area.
STRETCHES = {1: (0.0, 5.0), 2: (5.0, 45.0), 3: (45.0, 50.0), 4: (50.0, 55.0)}
VOLUMES = {tag: stop - start for tag, (start, stop) in STRETCHES.items()}


def unit_faces(faces, volumes, direction, depth, volume=VOLUMES):
    """Faces of unit area swept to a depth, against volumes that fill as much
    of the sweep as they hold - which is what a run-on of that cross-section is."""
    sweep = len(faces) * depth
    return sweep, min(sweep, sum(volume[tag] for tag in volumes))


MEASURED = ends.Measured(
    volume=VOLUMES,
    corners={tag: slab(*stretch) for tag, stretch in STRETCHES.items()},
    inward={"p1": (1.0, 0.0, 0.0), "p2": (-1.0, 0.0, 0.0)},
    swept=unit_faces,
)


def decide(
    facing=IN_ORDER,
    held=AIR,
    drawn=ONE_BODY,
    sides=GUIDE_SIDES,
    filled=(1, 2, 3),
    measured=MEASURED,
    ports=PORTS,
):
    return ends.behind(sides, facing, ports, filled, held, drawn, measured, _where, 3)


class TestWhatIsLeftOut:
    def test_the_run_on_behind_each_face_is_left_out_and_nothing_else(self):
        left, said = decide()
        assert said == []
        assert left == [("p1", [1]), ("p2", [3])]

    def test_faces_that_face_in_leave_out_nothing(self):
        facing = {10: {2: AHEAD}, 20: {2: AHEAD}}
        left, said = decide(facing=facing, sides={10: [2], 20: [2], 31: [2]}, filled=(2,))
        assert (left, said) == ([], [])

    def test_a_cross_section_of_several_regions_is_carried_on_region_by_region(self):
        """A guide loaded with a slab along its whole length: each piece of the
        face has the same region on both sides of it."""
        sides = {10: [1, 2], 11: [5, 6], 20: [2, 3], 21: [6, 7], 40: [1, 5], 41: [2, 6], 42: [3, 7]}
        facing = {
            10: {1: BEHIND, 2: AHEAD},
            11: {5: BEHIND, 6: AHEAD},
            20: {2: AHEAD, 3: BEHIND},
            21: {6: AHEAD, 7: BEHIND},
        }
        loaded = frozenset({"air", "slab"})
        held = {**AIR, 5: loaded, 6: loaded, 7: loaded}
        halves = {5: (0.0, 5.0), 6: (5.0, 45.0), 7: (45.0, 50.0)}
        volume = {**VOLUMES, **{t: b - a for t, (a, b) in halves.items()}}
        measured = ends.Measured(
            volume=volume,
            corners={**MEASURED.corners, **{t: slab(*h) for t, h in halves.items()}},
            inward=MEASURED.inward,
            swept=lambda *asked: unit_faces(*asked, volume=volume),
        )
        left, said = decide(
            facing=facing,
            held=held,
            sides=sides,
            filled=(1, 2, 3, 5, 6, 7),
            drawn=[("air", [1, 2, 3, 5, 6, 7]), ("slab", [5, 6, 7])],
            measured=measured,
            ports={10: "p1", 11: "p1", 20: "p2", 21: "p2"},
        )
        assert said == []
        assert left == [("p1", [1, 5]), ("p2", [3, 7])]

    def test_a_loaded_run_on_is_swept_once_for_each_region_it_holds(self):
        """So a run-on whose one region keeps its place while the other moves is
        not measured as one lump that happens to fill the sweep."""
        sides = {10: [1, 2], 11: [5, 6], 20: [2, 3], 21: [6, 7], 40: [1, 5], 41: [2, 6], 42: [3, 7]}
        facing = {
            10: {1: BEHIND, 2: AHEAD},
            11: {5: BEHIND, 6: AHEAD},
            20: {2: AHEAD, 3: BEHIND},
            21: {6: AHEAD, 7: BEHIND},
        }
        loaded = frozenset({"air", "slab"})
        held = {**AIR, 5: loaded, 6: loaded, 7: loaded}
        halves = {5: (0.0, 5.0), 6: (5.0, 45.0), 7: (45.0, 50.0)}
        volume = {**VOLUMES, **{t: b - a for t, (a, b) in halves.items()}}
        asked = []

        def recorded(faces, volumes, direction, depth):
            asked.append((faces, volumes))
            return unit_faces(faces, volumes, direction, depth, volume=volume)

        decide(
            facing=facing,
            held=held,
            sides=sides,
            filled=(1, 2, 3, 5, 6, 7),
            drawn=[("air", [1, 2, 3, 5, 6, 7]), ("slab", [5, 6, 7])],
            measured=ends.Measured(
                volume=volume,
                corners={**MEASURED.corners, **{t: slab(*h) for t, h in halves.items()}},
                inward=MEASURED.inward,
                swept=recorded,
            ),
            ports={10: "p1", 11: "p1", 20: "p2", 21: "p2"},
        )
        assert asked == [([10], [1]), ([11], [5]), ([20], [3]), ([21], [7])]

    def test_a_direction_of_any_length_reads_the_same_depth(self):
        longer = replace(MEASURED, inward={"p1": (2.0, 0.0, 0.0), "p2": (-0.5, 0.0, 0.0)})
        assert decide(measured=longer) == decide()

    def test_a_part_joined_through_walls_is_left_out_whole(self):
        """Parts are joined through every face nobody handed a direction, so a
        run-on the kernel cut into several volumes goes as one."""
        sides = {**GUIDE_SIDES, 40: [3, 4]}
        held = {**AIR, 4: frozenset({"air"})}
        left, said = decide(
            sides=sides, held=held, filled=(1, 2, 3, 4), drawn=[("air", [1, 2, 3, 4])]
        )
        assert said == []
        assert left == [("p1", [1]), ("p2", [3, 4])]


class TestWhatIsRefusedRatherThanLeftOut:
    def test_a_face_whose_one_region_stands_behind_it_faces_out(self):
        facing = {10: {2: BEHIND}, 20: {2: AHEAD}}
        _, said = decide(facing=facing, sides={10: [2], 20: [2]}, filled=(2,))
        assert said == [
            "the label p1 faces out of the model at 2:10: the one region it bounds there "
            "stands the other way from the direction it was handed"
        ]

    @pytest.mark.parametrize(
        "stands",
        [{1: UNTOLD, 2: AHEAD}, {1: AHEAD, 2: AHEAD}, {1: BEHIND, 2: BEHIND}],
        ids=["untold", "both ahead", "both behind"],
    )
    def test_a_face_whose_sides_are_not_told_apart_is_refused(self, stands):
        _, said = decide(facing={**IN_ORDER, 10: stands})
        assert said == ["which side of the label p1 the model stands on cannot be told at 2:10"]

    def test_a_part_in_front_of_one_face_and_behind_another_is_not_left_out(self):
        facing = {10: {1: BEHIND, 2: AHEAD}, 20: {2: BEHIND, 3: AHEAD}}
        left, said = decide(facing=facing)
        assert left == []
        assert len(said) == 1
        assert "stands in front of p1 and behind p2" in said[0] and "3:2" in said[0]

    def test_a_part_behind_two_faces_is_not_left_out(self):
        facing = {10: {1: AHEAD, 2: BEHIND}, 20: {2: BEHIND, 3: AHEAD}}
        left, said = decide(facing=facing)
        assert left == []
        assert said == [
            "the part of the model at 3:2 stands behind p1, p2 at once, so leaving it out "
            "leaves out what stands between them. One of them faces the wrong way"
        ]

    def test_a_part_that_is_a_whole_drawn_shape_is_not_left_out(self):
        """A face pointed at where two bodies of one region meet."""
        drawn = [("air", [1]), ("air", [2, 3])]
        left, said = decide(drawn=drawn)
        assert left == []
        assert len(said) == 1
        assert "stands behind p1" in said[0] and "the whole of a shape drawn for air" in said[0]

    def test_a_part_holding_another_region_is_not_left_out(self):
        """The region deeper in than the face, where nothing across the face
        says what it should be."""
        sides = {**GUIDE_SIDES, 40: [3, 4]}
        held = {**AIR, 4: frozenset({"air", "slab"})}
        _, said = decide(sides=sides, held=held, filled=(1, 2, 3, 4), drawn=[("air", [1, 2, 3, 4])])
        assert len(said) == 1 and "holds slab, which does not stand in front of it" in said[0]

    def test_a_region_meeting_the_face_from_behind_only_is_not_left_out(self):
        held = {**AIR, 1: frozenset({"air", "slab"})}
        _, said = decide(held=held)
        assert len(said) == 1
        assert "drawn as air, slab where what stands in front is drawn as air" in said[0]

    def test_a_part_holding_a_piece_nothing_was_drawn_over_is_not_left_out(self):
        sides = {**GUIDE_SIDES, 40: [3, 4]}
        _, said = decide(sides=sides, filled=(1, 2, 3, 4))
        assert len(said) == 1 and "holds a piece drawn as nothing" in said[0]

    def test_a_region_behind_a_piece_of_the_face_that_another_stands_in_front_of(self):
        """Each piece of the face is asked about what stands on either side of
        it, so a part meeting two regions across one face is refused where the
        two sides of a piece disagree."""
        sides = {10: [1, 2], 11: [1, 4], 20: [2, 3], 40: [2, 4]}
        facing = {10: {1: BEHIND, 2: AHEAD}, 11: {1: BEHIND, 4: AHEAD}, 20: {2: AHEAD, 3: BEHIND}}
        held = {**AIR, 4: frozenset({"substrate"})}
        _, said = decide(
            facing=facing, held=held, sides=sides, filled=(1, 2, 3, 4), ports={**PORTS, 11: "p1"}
        )
        assert said == [
            "the part of the model at 3:1 stands behind p1, and what stands behind its face "
            "is drawn as air where what stands in front is drawn as substrate, so what would "
            "be left out is not what stands in front of it carried on"
        ]

    def test_a_part_that_is_not_its_face_carried_straight_on_is_not_left_out(self):
        """What a device behind a face turned the wrong way looks like: its
        volume is not the face's area times its depth."""
        narrowed = {**VOLUMES, 3: 4.0}
        left, said = decide(
            measured=replace(
                MEASURED,
                volume=narrowed,
                swept=lambda *asked: unit_faces(*asked, volume=narrowed),
            )
        )
        assert left == []
        assert said == [
            "the part of the model at 3:3 stands behind p2, and it is not the face carried "
            "straight on: of the 4 cubic millimetres drawn as air behind it, 0 stand outside "
            "the face swept to the depth of 5 mm, and 1 of that sweep is not drawn as air. "
            "What is left out behind a face is the face carried on unchanged, and a part "
            "that is not is part of what is meshed - or the face is turned the wrong way"
        ]

    def test_what_stands_outside_the_sweep_is_as_much_a_refusal_as_what_is_missing(self):
        """A run-on bulging past its face: all of the sweep is drawn, and more."""

        def bulging(faces, volumes, direction, depth):
            sweep = len(faces) * depth
            return sweep, sweep

        left, said = decide(measured=replace(MEASURED, swept=bulging, volume={**VOLUMES, 3: 6.0}))
        assert left == [] and "1 stand outside the face swept" in said[0]

    def test_each_region_is_swept_from_the_pieces_of_the_face_it_stands_behind(self):
        asked = []

        def recorded(faces, volumes, direction, depth):
            asked.append((faces, volumes, tuple(direction), depth))
            return unit_faces(faces, volumes, direction, depth)

        decide(measured=replace(MEASURED, swept=recorded))
        assert asked == [([10], [1], (1.0, 0.0, 0.0), 5.0), ([20], [3], (-1.0, 0.0, 0.0), 5.0)]

    def test_nothing_is_left_out_while_anything_is_refused(self):
        """One part may go and the other may not, and neither goes."""
        drawn = [("air", [1, 2]), ("air", [3])]
        left, said = decide(drawn=drawn)
        assert left == [] and len(said) == 1


class TestWhatALabelWouldLose:
    def test_a_label_below_the_filled_dimension_on_a_run_on_is_named(self):
        said = ends.lost(
            {"air": [(3, 1), (3, 2)], "walls": [(2, 30), (2, 31)]},
            {(3, 1), (2, 30)},
            3,
            ["p1"],
            _where,
        )
        assert said == [
            "the label walls was drawn on what stands behind p1, at 2:30, and would be left "
            "out with it"
        ]

    def test_a_region_losing_its_run_on_is_not_a_loss(self):
        assert ends.lost({"air": [(3, 1), (3, 2)]}, {(3, 1)}, 3, ["p1"], _where) == []


# ---------------------------------------------------------------- over drawings

gmsh = needed(
    "gmsh", "gmsh is not on this interpreter, so the mesher is unreachable", module_level=True
)

from Microwave.Gmsh.mesh import mesh  # noqa: E402
from Microwave.Gmsh.vocabulary import Refused  # noqa: E402
from tests.test_gmsh_labels import (  # noqa: E402
    demand_for,
    filled_measure,
    group_area,
    profile_for,
)

#: What the drawings here call what the model ends against. Named rather than
#: drawn, since a label drawn over a run-on is refused.
WALL = "wall"


def ending(name):
    return next(drawing for drawing in drawn_ends() if drawing.name == name)


def pieces(name):
    """One drawing as the caller hands it over, directions and all."""
    listed = json.loads((DRAWN / f"{name}.manifest.json").read_text())["pieces"]
    return [
        Piece(
            p["label"],
            p["dim"],
            str(DRAWN / p["file"]),
            p["priority"],
            inward=None if "inward" not in p else tuple(p["inward"]),
        )
        for p in listed
    ]


def meshed(name, tmp_path, drawn=None):
    drawing = ending(name)
    return mesh(
        drawn if drawn is not None else pieces(name),
        demand_for(drawing),
        profile_for(drawing),
        str(tmp_path),
        name,
        remainder=WALL,
    )


@pytest.mark.parametrize("drawing", drawn_ends(), ids=lambda drawing: drawing.name)
def test_the_probe_wrote_each_drawing_as_declared(drawing):
    listed = json.loads((DRAWN / f"{drawing.name}.manifest.json").read_text())["pieces"]
    assert tuple((p["label"], p["dim"]) for p in listed) == drawing.declares
    stated = {p["label"]: tuple(p["inward"]) for p in listed if "inward" in p}
    assert stated == dict(drawing.inward), "the checked-in files predate the drawing"


@pytest.mark.parametrize(
    "drawing",
    [drawing for drawing in drawn_ends() if drawing.expect == REFUSED],
    ids=lambda drawing: drawing.name,
)
def test_a_drawing_that_cannot_be_ended_there_is_refused_before_meshing(drawing, tmp_path):
    with pytest.raises(Refused) as refused:
        meshed(drawing.name, tmp_path)
    assert drawing.complaint in str(refused.value), f"{drawing.subject}: {refused.value}"
    assert not list(tmp_path.iterdir()), "a file was written for a drawing that was refused"


def test_the_guide_is_meshed_between_its_planes(tmp_path):
    """What is filled is the guide between the planes, what the model ends
    against is its walls between them, and each plane is where it ends."""
    length, broad, narrow = GUIDE
    between = length - 2 * GUIDE_INSET
    got = meshed("guide_run_on", tmp_path)
    assert len(got.labels["air"].entities) == 1, "the label kept what was left out"
    assert filled_measure(got.path, 3) == pytest.approx(between * broad * narrow, rel=1e-9)
    assert group_area(got.path, WALL) == pytest.approx(2 * between * (broad + narrow), rel=1e-9)
    assert got.labels["port1"].sits == FRONTIER
    assert got.labels["port2"].sits == FRONTIER


def corners(place):
    """A place as the mesher writes one, back as its six numbers."""
    return [float(value) for value in re.findall(r"-?[0-9.]+(?:e-?[0-9]+)?", place)]


def test_the_answer_says_what_was_left_out_and_where(tmp_path):
    length, broad, narrow = GUIDE
    got = meshed("guide_run_on", tmp_path)
    assert [(gone.behind, gone.held) for gone in got.left_out] == [
        ("port1", ("air",)),
        ("port2", ("air",)),
    ]
    first, second = (corners(gone.place) for gone in got.left_out)
    assert first == pytest.approx([0, 0, 0, GUIDE_INSET, broad, narrow], abs=1e-6)
    assert second == pytest.approx([length - GUIDE_INSET, 0, 0, length, broad, narrow], abs=1e-6)


def test_faces_that_face_into_the_model_leave_it_whole(tmp_path):
    length, broad, narrow = GUIDE
    got = meshed("guide_ends", tmp_path)
    assert got.left_out == ()
    assert filled_measure(got.path, 3) == pytest.approx(length * broad * narrow, rel=1e-9)


def test_a_loaded_guide_is_carried_on_region_by_region(tmp_path):
    length, broad, narrow = GUIDE
    between = length - 2 * GUIDE_INSET
    got = meshed("loaded_guide_run_on", tmp_path)
    assert [(gone.behind, gone.held) for gone in got.left_out] == [
        ("port1", ("air", "slab")),
        ("port2", ("air", "slab")),
    ]
    assert filled_measure(got.path, 3) == pytest.approx(between * broad * narrow, rel=1e-9)


def test_a_straight_guide_whose_one_port_faces_out_is_meshed_short_and_says_so(tmp_path):
    """The bound of the decision: the device behind a port turned the wrong way
    is refused wherever it is not the port plane carried on, and a straight
    guide is exactly that. What was left out is in the answer."""
    length, broad, narrow = GUIDE
    got = meshed("straight_guide_behind_a_reversed_port", tmp_path)
    (gone,) = got.left_out
    assert corners(gone.place) == pytest.approx(
        [GUIDE_INSET, 0, 0, length, broad, narrow], abs=1e-6
    )
    assert filled_measure(got.path, 3) == pytest.approx(GUIDE_INSET * broad * narrow, rel=1e-9)


def test_a_coaxial_line_is_meshed_between_its_rings(tmp_path):
    """A ring's middle is in the hole, so the point asked about has to be
    looked for on the face."""
    drawing = ending("coax_run_on")
    outer, inner, length = COAX_LINE
    got = meshed("coax_run_on", tmp_path)
    assert [gone.behind for gone in got.left_out] == ["port1", "port2"]
    between = math.pi * (outer**2 - inner**2) * (length - 2 * COAX_INSET)
    measured = filled_measure(got.path, 3)
    assert measured <= between * (1 + 1e-9)
    assert measured == pytest.approx(between, rel=drawing.loses)


def test_a_face_standing_along_its_way_in_is_refused_as_untold(tmp_path):
    """The one direction the test cannot read a side off: along the face."""
    drawn = [
        replace(one, inward=(0.0, 1.0, 0.0)) if one.label == "port1" else one
        for one in pieces("guide_run_on")
    ]
    with pytest.raises(Refused, match="which side of the label port1 the model stands on"):
        meshed("guide_run_on", tmp_path, drawn)


@pytest.mark.parametrize("holds", [0, 1], ids=["neither point", "both points"])
def test_a_volume_the_kernel_does_not_place_on_one_side_is_untold(holds, tmp_path, monkeypatch):
    """What the kernel answers where it does not separate the two sides, which
    no drawing here reaches: a face on the end of the guide, whose one volume
    would otherwise be read as standing behind it or in front of it."""
    asked = gmsh.model.isInside

    def undecided(dim, tag, coord, parametric=False):
        return holds if dim == 3 else asked(dim, tag, coord, parametric)

    monkeypatch.setattr(gmsh.model, "isInside", undecided)
    with pytest.raises(Refused, match="which side of the label port1 the model stands on"):
        meshed("guide_ends", tmp_path)


def test_a_way_in_that_is_not_one_is_refused_before_the_drawing_is_read(tmp_path):
    drawn = [
        replace(one, inward=(0.0, 0.0, 0.0), file=str(tmp_path / "nowhere.brep"))
        if one.label == "port1"
        else one
        for one in pieces("guide_run_on")
    ]
    with pytest.raises(Refused) as refused:
        meshed("guide_run_on", tmp_path, drawn)
    assert any("no length" in line for line in refused.value.complaints)


def test_two_labels_handed_a_way_in_over_one_face_are_refused(tmp_path):
    """Nothing says which of the two directions the face ends the model by."""
    drawn = pieces("guide_run_on")
    first = next(one for one in drawn if one.label == "port1")
    drawn.append(replace(first, label="port1b", inward=(-1.0, 0.0, 0.0)))
    with pytest.raises(Refused, match="the labels port1, port1b are each handed a way"):
        meshed("guide_run_on", tmp_path, drawn)


def test_without_a_way_in_the_planes_are_inside_the_model(tmp_path):
    """The same drawing handed over as it was before: nothing is left out."""
    drawn = [replace(one, inward=None) for one in pieces("guide_run_on")]
    got = meshed("guide_run_on", tmp_path, drawn)
    assert got.left_out == ()
    assert got.labels["port1"].sits != FRONTIER


def test_every_meshed_drawing_here_is_meshed(tmp_path):
    """The meshed half of the population, each asked to come back as a mesh."""
    for drawing in drawn_ends():
        if drawing.expect != MESHED:
            continue
        (tmp_path / drawing.name).mkdir()
        got = meshed(drawing.name, tmp_path / drawing.name)
        assert pathlib.Path(got.path).is_file(), drawing.subject
