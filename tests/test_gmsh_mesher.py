# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the mesher refuses, and what a caller needs no mesher to say.

The questions asked of a label map are arithmetic over a map: which entity each
label got, which entities of the filled dimension each entity below it bounds,
and what a file held that its label did not declare. So they are asked
here of maps written by hand, one specimen per refusal, and none of it needs
Gmsh - which is also the claim the first test makes.

``tests/test_gmsh_labels.py`` is the other half: the same questions over maps
the kernel itself produced from shapes FreeCAD drew.
"""

from __future__ import annotations

import ast
import pathlib
import re
import subprocess
import sys

import Microwave
from Microwave.Gmsh import coverage
from Microwave.Gmsh.vocabulary import (
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
    Unmeshed,
    Within,
)
from tests.repo import ROOT

PACKAGE = pathlib.Path(Microwave.__file__).resolve().parent


def test_stating_a_request_needs_no_mesher():
    """A caller builds a request and reads an answer in the process holding the
    document, where there is no Gmsh and must not need to be.

    In a child interpreter, because this suite has already imported the half
    that does import Gmsh, and ``sys.modules`` would answer for the run rather
    than for the module.
    """
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import Microwave.Gmsh.vocabulary\n"
            "import Microwave.Gmsh.coverage\n"
            "import Microwave.Gmsh.ends\n"
            "print(','.join(n for n in sys.modules if n.split('.')[0] == 'gmsh'))\n",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "", f"stating a request pulled in {done.stdout.strip()}"


class TestWhereALabelSits:
    """The report a wall condition is decided on, and the mesher cannot decide it.

    A surface inside the model is exactly what a port is, and it is also a
    perfect conductor cutting a cavity in half. So this is reported and never
    refused, and the layer that knows which of the two it is answers it.
    """

    def test_bounding_one_is_where_the_model_ends(self):
        """Which is not the same as the outside of it: the wall of a cavity
        bounds one volume and is enclosed by the body."""
        assert coverage.sits([4, 5], {4: 1, 5: 1}) == FRONTIER

    def test_bounding_two_is_inside(self):
        assert coverage.sits([4], {4: 2}) == INTERIOR

    def test_a_label_can_hold_some_of_each(self):
        assert coverage.sits([4, 5], {4: 1, 5: 2}) == BOTH

    def test_reading_it_does_not_consume_what_it_was_given(self):
        """Both halves are read off the same entities, and an iterator read
        twice answers the second question about nothing. That failure is
        invisible on a label lying wholly outside, which is every label in the
        simplest drawing there is."""
        assert coverage.sits(iter([4]), {4: 2}) == INTERIOR

    def test_bounding_nothing_is_no_answer(self):
        assert coverage.sits([4], {}) is None


#: A drawing that meshes: one volume, its boundary claimed by a wall label, and
#: a face inside it under a label of its own. The tags are arbitrary and only
#: their agreement between the arguments means anything.
SOUND = {
    "air": [(3, 1), (3, 2)],
    "walls": [(2, 1), (2, 2), (2, 3)],
    "port": [(2, 4)],
}
BOUNDS = {1: 1, 2: 1, 3: 1, 4: 2}
FILLED = [1, 2]


class TestWhatTheMapIsAskedInTheProfilesDimensions:
    """Each refusal names a different thing a caller did, and each names the
    labels the caller wrote.

    Stated in the dimension being filled and the one below it, so the same code
    asks it of a volume method, of a method working on the surface of a closed
    body, and of one working on a cross-section.
    """

    def test_a_sound_drawing_draws_no_complaint(self):
        assert coverage.complaints(SOUND, BOUNDS, FILLED, 3) == []

    def test_handing_over_nothing_is_not_a_sound_drawing(self):
        said = coverage.complaints({}, {}, [], 3)
        assert said == ["nothing was handed over, so there is nothing to mesh"]

    def test_a_drawing_with_nothing_of_the_dimension_being_filled(self):
        """A label may stand below the dimension the profile fills - a curve
        under a volume profile is legal - so a drawing of nothing but those is
        a labelling nobody refuses and a mesh with no element of what was asked
        for. Nothing handed over at all is the sentence above this one, and
        saying both would be one event reported twice.
        """
        said = coverage.complaints({"edge": [(1, 1)]}, {}, [], 3)
        assert len(said) == 1
        assert "reaches dimension 3" in said[0]

    def test_a_region_no_label_covers(self):
        said = coverage.complaints(SOUND, BOUNDS, [*FILLED, 7], 3)
        assert len(said) == 1
        assert "dimension 3" in said[0] and said[0].endswith(": 7")

    def test_where_the_model_ends_left_unclaimed(self):
        said = coverage.complaints(SOUND, {**BOUNDS, 9: 1}, FILLED, 3)
        assert len(said) == 1
        assert "the model ends" in said[0] and said[0].endswith(": 9")

    def test_a_piece_nothing_claims_is_named_by_where_it_is(self):
        """Neither coverage question has a label to name, and a tag out of a
        fragmented model reaches nothing a user drew."""
        where = {(2, 9): "(0, 0, 0) to (1, 1, 1)", (3, 7): "(2, 2, 2) to (3, 3, 3)"}
        ends = coverage.complaints(SOUND, {**BOUNDS, 9: 1}, FILLED, 3, place=where)
        assert ends[0].endswith("9 at (0, 0, 0) to (1, 1, 1)")
        left = coverage.complaints(SOUND, BOUNDS, [*FILLED, 7], 3, place=where)
        assert left[0].endswith("7 at (2, 2, 2) to (3, 3, 3)")

    def test_a_label_the_written_format_cannot_carry(self):
        """Two labels alike as far as a group name reaches arrive as one, and a
        label cut off at a quote is not the label the caller asked about.

        The length a writer keeps is in bytes, so a name outside ASCII reaches
        it at half the characters. Counted the other way this label passes the
        guard and is written truncated, and two of them arrive under one name
        with neither caller's own in the file.
        """
        keeps = FORMATS[Profile.written].name_length
        outside_ascii = "\u00e9" * (keeps // 2) + "Z"
        assert len(outside_ascii) < keeps < len(outside_ascii.encode())
        for wrong, why in (
            ("m" * (keeps + 1), "longer"),
            (outside_ascii, "longer"),
            ('a"b', "cuts the name at"),
            ("", "empty"),
        ):
            said = coverage.complaints({**SOUND, wrong: [(2, 4)]}, BOUNDS, FILLED, 3)
            assert any(why in line for line in said), (wrong, said)

    def test_what_a_name_may_carry_belongs_to_the_format(self):
        """A space reaches one writer intact and another as an underscore, so a
        rule wide enough for every format refuses labels two of them hold.

        Measured: UNV cuts a name at a tab and writes a space as an underscore,
        so two labels differing only there arrive under one name. Gmsh's own
        formats carry both.
        """
        spaced = {**SOUND, "wall one": [(2, 3)]}
        spaced["walls"] = [(2, 1), (2, 2)]
        assert coverage.complaints(spaced, BOUNDS, FILLED, 3, "msh22") == []
        assert coverage.complaints(spaced, BOUNDS, FILLED, 3, "msh41") == []
        said = coverage.complaints(spaced, BOUNDS, FILLED, 3, "unv")
        assert any("wall one" in line and "unv" in line for line in said), said

    def test_a_label_above_the_dimension_being_filled(self):
        """Nothing in the mesh would carry it, and the file would come back
        without it while the answer said it was there."""
        said = coverage.complaints({"skin": [(2, 1)], "air": [(3, 1)]}, {1: 1}, [1], 2)
        assert any("above the 2" in line for line in said), said

    def test_a_dimension_no_mesh_is_made_of(self):
        said = coverage.complaints(SOUND, BOUNDS, FILLED, 0)
        assert said == ["the profile asks to fill dimension 0, and a mesh is made of 1, 2, 3"]

    def test_a_labelled_piece_the_drawing_puts_nowhere(self):
        whisker = {**SOUND, "port": [(2, 4), (2, 8)]}
        said = coverage.complaints(whisker, BOUNDS, FILLED, 3)
        assert len(said) == 1
        assert said[0].startswith("the label port ") and said[0].endswith(": 8")
        assert "outside the model" not in said[0], (
            "a piece inside the model that divides nothing bounds nothing too"
        )

    def test_a_label_that_reached_nothing(self):
        """A label with no entity under it has no dimension to be a group of.

        Every other question here skips it: it stands on no dimension, so it is
        neither above the one being filled nor below it, and nothing of the
        drawing is under it. Left unsaid, it reaches the point where a group
        is made and takes the first of no entities.
        """
        said = coverage.complaints({**SOUND, "empty": []}, BOUNDS, FILLED, 3)
        assert any("the label empty reached nothing" in line for line in said), said

    def test_a_label_spread_over_two_dimensions(self):
        both = {**SOUND, "port": [(2, 4), (3, 1)]}
        said = coverage.complaints(both, BOUNDS, FILLED, 3)
        assert any("port" in line and "two labels" in line for line in said)

    def test_a_shape_handed_over_and_dropped(self):
        said = coverage.complaints(SOUND, BOUNDS, FILLED, 3, dropped=[("port", "port.brep", (2,))])
        assert len(said) == 1
        assert "port" in said[0] and "port.brep" in said[0]

    def test_a_file_holding_nothing_of_its_own_dimension(self):
        said = coverage.complaints(SOUND, BOUNDS, FILLED, 3, dropped=[("port", "port.brep", ())])
        assert len(said) == 1
        assert "port" in said[0] and "nothing" in said[0]

    def test_a_tag_of_the_filled_dimension_is_not_a_tag_of_the_one_below(self):
        """Both coverage questions read a tag out of a map that holds two
        dimensions, and a tag is unique only within its own.

        Without the dimension in the comparison a region left over is answered
        by a face that happens to carry its number, and a wall left unclaimed is
        answered by a volume that does.
        """
        left_over = coverage.complaints({"walls": [(2, 3)]}, {3: 2}, [3], 3)
        assert any(
            line.endswith("dimension 3, and these are left over: 3") for line in left_over
        ), left_over

        unclaimed = coverage.complaints({"air": [(3, 1)]}, {1: 1}, [1], 3)
        assert any("the model ends" in line for line in unclaimed), unclaimed

    def test_a_closed_surface_has_nowhere_the_model_ends(self):
        """The question about a frontier asks nothing where there is none.

        On the surface of a closed body every curve bounds two faces, so the
        same code that refuses an unlabelled wall of a volume refuses nothing
        here - which is what makes it nobody's in particular.
        """
        shell = {"skin": [(2, 1), (2, 2)]}
        assert coverage.complaints(shell, {1: 2, 2: 2, 3: 2}, [1, 2], 2) == []


#: The words the mesher's public vocabulary must not contain, which are what a
#: caller's own layer decides. A mesher that knows what a port is has begun to
#: know whose port it is.
MEANINGS = ("port", "material", "boundary", "excitation")

#: The package, and where the adapter names below are read from.
MESHER = PACKAGE / "Gmsh"
ADAPTERS = PACKAGE / "Solvers"


class TestTheLabelForWhatIsLeftWhereTheModelEnds:
    """A caller naming a remainder is asking the mesher which faces those are.

    The alternative is a caller enumerating a solid's faces off the drawing,
    which cannot tell the outside of a model from the face between two bodies
    that touch - and a perfect conductor on the second cuts the region the field
    is in into pieces that are then solved apart.
    """

    def test_it_is_what_no_label_claims_where_the_model_ends(self):
        assert coverage.remaining(SOUND, {**BOUNDS, 9: 1}, 3) == [9]

    def test_a_face_inside_the_model_is_never_in_it(self):
        """Not filtered down to the frontier: built out of it. That is what a
        condition which must not divide the region rests on."""
        assert coverage.remaining(SOUND, {**BOUNDS, 9: 2}, 3) == []

    def test_naming_one_is_what_turns_a_bare_frontier_from_a_refusal(self):
        assert coverage.complaints(SOUND, {**BOUNDS, 9: 1}, FILLED, 3, remainder="rest") == []

    def test_a_remainder_the_labels_leave_nothing_for_is_not_a_refusal(self):
        """Every entity where the model ends is claimed by somebody, so there is
        nothing left over. That is a sound drawing - every wall of a guide drawn
        as metal of its own - and what follows is no group, not a complaint."""
        assert coverage.complaints(SOUND, BOUNDS, FILLED, 3, remainder="rest") == []

    def test_a_drawing_that_ends_nowhere_is_told_so(self):
        """A closed surface has no curve bounding one face, so a caller filling
        it and naming a remainder has asked for a label with nowhere to stand.

        Held apart from the case above it: there the model ends and the labels
        took all of it, and here it ends nowhere for a label to take.
        """
        said = coverage.complaints({"skin": [(2, 1)]}, {}, [1], 2, remainder="rim")
        assert len(said) == 1
        assert "the drawing ends nowhere for it to stand on" in said[0]

    def test_a_remainder_the_caller_also_drew_is_refused(self):
        """One group would hold both what was drawn and what is left over."""
        said = coverage.complaints(SOUND, {**BOUNDS, 9: 1}, FILLED, 3, remainder="port")
        assert len(said) == 1
        assert "Give one of them another name" in said[0]

    def test_a_remainder_the_written_format_cannot_carry_is_refused(self):
        said = coverage.complaints(SOUND, {**BOUNDS, 9: 1}, FILLED, 3, remainder='a"b')
        assert len(said) == 1
        assert "cuts the name at" in said[0]

    def test_a_drawing_that_reached_nothing_is_not_told_about_the_remainder_too(self):
        """The one event is that nothing was handed over, and a remainder
        catching nothing there is that same event said a second way."""
        said = coverage.complaints({}, {}, [], 3, remainder="rest")
        assert said == ["nothing was handed over, so there is nothing to mesh"]


#: Two volumes and the faces around them, as a drawing of two bodies that miss
#: each other leaves them: every face bounds one volume and none bounds both.
#: The tags are arbitrary and only their agreement between the arguments means
#: anything.
TWO_BODIES = {
    "near": [(3, 1)],
    "far": [(3, 2)],
    "walls": [(2, 1), (2, 2)],
}
AROUND = ["(0, 0, 0) to (1, 1, 1)", "(2, 0, 0) to (3, 1, 1)"]


class TestWhatIsFilledHoldsTogether:
    """Asked where the profile says what is filled has to be one region.

    Two volumes are one region where a face bounds both. The kernel joins two
    bodies only closer than its own tolerance, so a gap nobody meant leaves
    every face bounding one volume, and a caller that puts a condition on where
    the model ends has closed each body off - which nothing in the mesh says.
    """

    def test_a_face_bounding_both_joins_them(self):
        assert coverage.parts({4: [1, 2]}, [1, 2]) == [[1, 2]]

    def test_a_chain_joins_its_ends(self):
        """Neither end bounds a face the other does, and they are still one
        region through the one between them."""
        assert coverage.parts({5: [1, 2], 6: [2, 3]}, [3, 1, 2]) == [[1, 2, 3]]

    def test_what_no_face_joins_is_two_parts(self):
        assert coverage.parts({4: [1], 5: [2]}, [1, 2]) == [[1], [2]]

    def test_an_entity_joined_to_itself_is_joined_to_nothing_else(self):
        """What a seam is: a sphere's surface is listed against the curve it
        closes on twice."""
        assert coverage.parts({7: [1, 1]}, [1, 2]) == [[1], [2]]

    def test_an_entity_no_face_lists_is_a_part_of_its_own(self):
        assert coverage.parts({}, [1]) == [[1]]

    def test_the_parts_come_in_the_order_of_their_first_tags(self):
        assert coverage.parts({9: [4, 3]}, [4, 3, 1]) == [[1], [3, 4]]

    def test_a_region_that_holds_together_draws_no_complaint(self):
        assert coverage.apart(SOUND, [[1, 2]], 3) == []

    def test_each_part_is_named_by_what_was_drawn_over_it_and_where_it_is(self):
        """A tag reaches nothing a user drew, so the part carries the labels and
        one place around it, and the labels on the dimension below are not what
        it is made of."""
        said = coverage.apart(TWO_BODIES, [[1], [2]], 3, AROUND)
        assert len(said) == 1
        assert "the part drawn as near, at (0, 0, 0) to (1, 1, 1)" in said[0]
        assert "the part drawn as far, at (2, 0, 0) to (3, 1, 1)" in said[0]
        assert "walls" not in said[0]

    def test_one_label_whose_shapes_came_apart_is_named_on_each_part(self):
        """Both bodies bound to one material is the ordinary drawing, and the
        place is then all that tells the two parts apart."""
        drawn = {"air": [(3, 1), (3, 2)], "walls": [(2, 1), (2, 2)]}
        said = coverage.apart(drawn, [[1], [2]], 3, AROUND)
        assert "drawn as air, at (0, 0, 0)" in said[0]
        assert "drawn as air, at (2, 0, 0)" in said[0]

    def test_a_part_holding_several_labels_names_them_all(self):
        drawn = {**TWO_BODIES, "also": [(3, 3)]}
        said = coverage.apart(drawn, [[1, 3], [2]], 3)
        assert "the part drawn as also, near;" in said[0]

    def test_what_joins_two_parts_is_named_in_the_dimension_filled(self):
        """A method filling a surface is joined across a curve, and told so."""
        drawn = {"near": [(2, 1)], "far": [(2, 2)]}
        said = coverage.apart(drawn, [[1], [2]], 2)
        assert "no curve joins these parts" in said[0]
        assert "have to share a curve" in said[0]


def _defined(path):
    """Every name a module declares: itself, its classes, its functions and
    every argument they take, and anything assigned at module or class scope.

    Every argument, which means the ones after a bare star and the ones before a
    slash as well as the ordinary ones - a rule that reads only the ordinary
    ones is a rule a keyword argument walks past.

    Names and not the whole text, because the text is where the boundary is
    argued and Gmsh's own API is called - ``getBoundary`` is the kernel's word
    for the faces of a solid, and a rule reading it as ours would ban asking
    the question the refusals are made of.
    """
    found = {path.stem}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ClassDef):
            found.add(node.name)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            found.add(node.name)
            taken = node.args
            every = [*taken.posonlyargs, *taken.args, *taken.kwonlyargs]
            every.extend(one for one in (taken.vararg, taken.kwarg) if one is not None)
            found.update(argument.arg for argument in every)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            found.add(node.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.add(node.target.id)
    return found


def _words(name):
    """One identifier, as the words it was written from.

    Split rather than searched for, because a word is a substring of names that
    have nothing to do with it: ``imported`` carries ``port``, and a rule
    reading that as a caller's vocabulary bans the ordinary word for what an
    import returned.
    """
    return {word.lower() for word in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+", name)}


class TestWhoOwnsAPieceTwoLabelsWereDrawnOver:
    """One piece carries one label, so where two shapes overlap somebody says
    which of them owns the region they share.

    Not the map: a part meant to stand inside a domain and a body mistyped into
    another are the same shape, and no reading of the drawing separates them.
    So the caller states an order, and where it states none - which is what
    every label gets - the drawing comes back refused.
    """

    #: A wall label and a port label drawn over one face, which is the everyday
    #: shape of the contest and the one the corpus carries twice.
    OVER = {**SOUND, "port": [(2, 3)]}

    def test_no_order_leaves_the_piece_unsettled(self):
        taken, settled, stuck = coverage.settle(self.OVER, {})
        assert settled == []
        assert [entity for entity, _ in stuck] == [(2, 3)]
        said = coverage.unsettled(self.OVER, taken, stuck)
        assert len(said) == 1
        assert "port" in said[0] and "walls" in said[0]

    def test_the_order_gives_the_piece_to_the_higher(self):
        taken, settled, stuck = coverage.settle(self.OVER, {"port": 1})
        assert stuck == []
        assert taken["port"] == [(2, 3)]
        assert (2, 3) not in taken["walls"]
        assert [(one.took, one.gave_up) for one in settled] == [("port", ("walls",))]

    def test_the_refusal_says_which_region_stands_inside_which(self):
        """Which the caller cannot get cheaply anywhere else, and which says what
        to cut out of what."""
        taken, _, stuck = coverage.settle(self.OVER, {})
        assert coverage.unsettled(self.OVER, taken, stuck)[0].endswith(
            "the region drawn for port stands inside the rest"
        )

    def test_regions_that_cross_are_said_to_cross(self):
        """An order settles this as readily as it settles a nesting, so the
        sentence says only how the regions stand."""
        crossing = {"one": [(3, 1), (3, 2)], "two": [(3, 2), (3, 3)]}
        taken, _, stuck = coverage.settle(crossing, {"one": 1, "two": 1})
        said = coverage.unsettled(crossing, taken, stuck)
        assert "the regions cross" in said[0], said
        assert coverage.settle(crossing, {"one": 2, "two": 1})[2] == [], (
            "an order settles a crossing, so the sentence must not say it cannot"
        )

    def test_one_region_under_two_labels_is_said_to_be_one_region(self):
        twice = {"one": [(3, 1)], "two": [(3, 1)]}
        taken, _, stuck = coverage.settle(twice, {})
        said = coverage.unsettled(twice, taken, stuck)
        assert "one region is drawn under every one of them" in said[0], said

    def test_several_labels_tied_at_the_innermost_region(self):
        """One region under two labels, both inside a third. Neither the
        sentence for a nesting nor the one for regions that cross is true of
        it, and reading it as either tells the caller something false. No
        order over this drawing produces a mesh: it leaves one of the two
        labels nothing however it is stated.
        """
        tied = {"a": [(3, 1)], "b": [(3, 1)], "c": [(3, 1), (3, 2)]}
        taken, _, stuck = coverage.settle(tied, {})
        said = coverage.unsettled(tied, taken, stuck)
        assert said[0].endswith(
            "one region is drawn under a, b alike and stands inside the rest"
        ), said
        for order in ({"a": 2, "b": 2, "c": 1}, {"a": 1, "b": 1, "c": 2}, {"a": 2, "b": 1}):
            taken, _, stuck = coverage.settle(tied, order)
            assert coverage.unsettled(tied, taken, stuck), (
                f"{order} was taken and the drawing meshed, so the sentence is wrong"
            )

    def test_a_label_the_order_leaves_holding_nothing(self):
        """Which is a group with no element in it, and a caller looking its own
        label up in the file finding nothing under it."""
        filled = {"air": [(3, 1), (3, 2)], "lower": [(3, 1)], "upper": [(3, 2)]}
        taken, _, stuck = coverage.settle(filled, {"lower": 1, "upper": 1})
        assert stuck == []
        said = coverage.unsettled(filled, taken, stuck)
        assert len(said) == 1
        assert said[0].startswith("the label air kept nothing"), said

    def test_a_priority_that_is_not_a_whole_number(self):
        """Nothing orders a value against itself, so the highest of a set
        holding one matches no member and the contest goes to nobody."""
        order, said = coverage.ordered([Piece("air", 3, "air.brep", float("nan"))])
        assert order == {}
        assert said == [
            "the label air is handed over at priority nan, and one label stands above "
            "another by a whole number rather than by a float"
        ], said

    def test_a_label_handed_over_at_two_orders(self):
        handed = [
            Piece("air", 3, "air.brep", 0),
            Piece("air", 3, "more_air.brep", 1),
            Piece("port", 2, "port.brep", 1),
        ]
        order, said = coverage.ordered(handed)
        assert order == {"air": 1, "port": 1}
        assert len(said) == 1
        assert "the label air is handed over at priority 0, 1" in said[0], said


class TestWhatALabelDividesWhatIsFilledOn:
    """A caller says a face of its label joins nothing, and the mesher asks what
    is filled a second time with those faces out of the joining."""

    def test_a_label_stating_it_at_the_filled_dimension(self):
        """A region is what is filled rather than something the field meets, so
        saying it divides describes no drawing."""
        cutting, said = coverage.dividing([Piece("air", 3, "air.brep", divides=True)], 3)
        assert cutting == {"air"}
        assert said == [
            "the label air is handed over as dividing what is filled and declared at "
            "dimension 3, and what divides dimension 3 is a face of dimension 2"
        ], said

    def test_a_label_stating_it_below_the_faces(self):
        cutting, said = coverage.dividing([Piece("wire", 1, "wire.brep", divides=True)], 3)
        assert cutting == {"wire"}
        assert len(said) == 1
        assert "declared at dimension 1" in said[0], said

    def test_a_label_handed_over_both_ways(self):
        """One face of a label dividing while another does not is two labels."""
        handed = [
            Piece("skin", 2, "front.brep", divides=True),
            Piece("skin", 2, "back.brep"),
        ]
        cutting, said = coverage.dividing(handed, 3)
        assert cutting == set()
        assert said == [
            "the label skin is handed over both as dividing what is filled and as not, "
            "and a label is one or the other"
        ], said

    def test_a_drawing_naming_none(self):
        cutting, said = coverage.dividing([Piece("air", 3, "air.brep")], 3)
        assert (cutting, said) == (set(), [])

    def test_anything_but_true_or_false(self):
        """A value that is neither reads as one of them under a coercion nobody
        wrote down, as a priority that is not a whole number would."""
        cutting, said = coverage.dividing([Piece("skin", 2, "skin.brep", divides="yes")], 3)
        assert cutting == set()
        assert said == [
            "the label skin is handed over as dividing what is filled by 'yes', of type "
            "str, and a label either is or is not"
        ], said

    def test_the_faces_named_stop_joining_what_they_bound(self):
        """Two volumes a face joins are one part, and none once it is a cut."""
        sides = {7: [1, 2], 8: [1], 9: [2]}
        assert coverage.parts(sides, [1, 2]) == [[1, 2]]
        assert coverage.parts(sides, [1, 2], cut={7}) == [[1], [2]]

    def test_a_cut_face_that_joined_nothing_leaves_the_parts_alone(self):
        """A face where the model ends bounds one volume, so cutting on it
        changes nothing - which is why a wall built out of the frontier need
        never say it divides."""
        sides = {7: [1, 2], 8: [1], 9: [2]}
        assert coverage.parts(sides, [1, 2], cut={8, 9}) == [[1, 2]]


class TestWhatALabelLeavingTheModelIsHandedOverAs:
    """A caller says what a label fills is not part of the model, and the label
    keeps the faces it leaves behind."""

    def test_only_what_is_filled_can_leave(self):
        going, said = coverage.leaving([Piece("skin", 2, "skin.brep", leaves=True)], 3)
        assert going == {"skin"}
        assert said == [
            "the label skin is handed over as leaving the model and declared at dimension 2, "
            "and only what fills dimension 3 can leave it"
        ], said

    def test_a_label_handed_over_both_ways(self):
        handed = [
            Piece("post", 3, "one.brep", leaves=True),
            Piece("post", 3, "other.brep"),
        ]
        going, said = coverage.leaving(handed, 3)
        assert going == set()
        assert said == [
            "the label post is handed over both as leaving the model and as not, and a "
            "label is one or the other"
        ], said

    def test_anything_but_true_or_false(self):
        going, said = coverage.leaving([Piece("post", 3, "post.brep", leaves=1)], 3)
        assert going == set()
        assert said == [
            "the label post is handed over as leaving the model by 1, of type int, and a "
            "label either is or is not"
        ], said

    def test_a_drawing_naming_none(self):
        assert coverage.leaving([Piece("air", 3, "air.brep")], 3) == (set(), [])

    def test_a_label_that_leaves_may_say_it_divides(self):
        """It is declared filled and ends as faces of the dimension below, which
        is what a label that divides is."""
        handed = [Piece("post", 3, "post.brep", divides=True, leaves=True)]
        assert coverage.dividing(handed, 3) == ({"post"}, [])

    def test_what_a_body_that_left_joined_is_one_part(self):
        """Two volumes nothing joins are two parts, and one where the drawing
        joined them through something that left the model."""
        sides = {8: [1], 9: [2]}
        assert coverage.parts(sides, [1, 2]) == [[1], [2]]
        assert coverage.parts(sides, [1, 2], joined=[[1, 2]]) == [[1, 2]]


class TestTheMesherKeepsNoBackendsVocabulary:
    """The hazard this package is written against, stated as an assertion.

    A shared vocabulary bends when it is written in the terms of whoever
    consumes it first and then stretched for the second. Nothing written now
    sees that coming in general; what can be held is that the words of the layer
    above are not in this one.
    """

    def test_the_reading_finds_the_package(self):
        """The floor under the rules below, each of which holds on an empty set."""
        found = {path.stem: _defined(path) for path in MESHER.glob("*.py")}
        assert {"vocabulary", "labels", "coverage", "mesh"} <= set(found)
        assert "Piece" in found["vocabulary"], "the reading found no class"
        assert "complaints" in found["coverage"], "the reading found no function"

    def test_the_words_a_name_is_read_as_are_the_words_it_was_written_from(self):
        """The floor under the rule below, which a reading that split nothing
        would satisfy by finding every word in every name."""
        assert _words("imported") == {"imported"}
        assert _words("port_faces") == {"port", "faces"}
        assert _words("WavePortOffset") == {"wave", "port", "offset"}

    def test_no_name_it_declares_says_what_a_label_means(self):
        wrong = {}
        for path in MESHER.rglob("*.py"):
            named = sorted(name for name in _defined(path) if _words(name) & set(MEANINGS))
            if named:
                wrong[path.name] = named
        assert not wrong, f"the mesher has acquired a caller's vocabulary: {wrong}"

    def test_it_spells_no_document_kind_and_no_backend(self):
        """In the whole text and not only in the names, which is the shape a
        layering guard cannot have: a backend named as a string is invisible to
        anything reading imports.

        The kinds come from the document layer's own answer rather than from a
        list here, so a kind added tomorrow is covered as it stands.
        """
        from Microwave.Objects.kinds import kinds

        forbidden = set(kinds()) | {
            path.name
            for path in ADAPTERS.iterdir()
            if path.is_dir() and not path.name.startswith("_")
        }
        assert forbidden, "nothing was read, so this holds vacuously"
        wrong = {}
        for path in MESHER.rglob("*.py"):
            text = path.read_text()
            spelt = sorted(
                name for name in forbidden if re.search(rf"\b{name}\b", text, re.IGNORECASE)
            )
            if spelt:
                wrong[path.name] = spelt
        assert not wrong, f"the mesher spells what is above it: {wrong}"


def test_a_refusal_carries_every_complaint_and_says_them_all():
    why = Refused(["the first thing", "the second thing"])
    assert why.complaints == ("the first thing", "the second thing")
    assert "the first thing" in str(why) and "the second thing" in str(why)


def test_what_gmsh_said_travels_with_the_absence_of_a_mesh():
    """Gmsh writes what it did to a terminal. A caller running the mesher beside
    its document has none, so the lines it wants read are carried on the
    exception instead of being the only account and going nowhere."""
    why = Unmeshed(["there is no mesh"], ["Warning : no tetrahedra in region 1"])
    assert why.complaints == ("there is no mesh",)
    assert why.said == ("Warning : no tetrahedra in region 1",)
    assert Unmeshed(["alone"]).said == ()


class TestWhichElementsAreNotShapes:
    """The quality measure is signed, and what the sign means is a rule with no
    Gmsh in it.

    A sliver is a body people draw and comes back reported. An element at or
    below zero occupies space the drawing does not, and the run is stopped on
    it.
    """

    def test_a_sliver_is_a_shape(self):
        assert coverage.inverted({2: 5.8e-05, 3: 7.1e-05}) == []

    def test_an_element_turned_inside_out_is_not(self):
        assert coverage.inverted({2: -0.98, 3: 0.3}) == [2]

    def test_a_flat_element_is_not_either(self):
        """Zero is no shape rather than a small one, and the sign alone does not
        say so."""
        assert coverage.inverted({2: 0.0, 3: 0.3}) == [2]

    def test_every_dimension_that_lost_its_shape_is_named(self):
        assert coverage.inverted({2: -0.1, 3: -0.2}) == [2, 3]

    def test_nothing_measured_is_nothing_wrong(self):
        assert coverage.inverted({}) == []


class TestWhatIsAskedForBeforeAnythingIsDrawn:
    """The size and the order are three numbers, and every question about them
    is answerable with no drawing at all.

    Which is why they are answered first. Gmsh takes most of these and meshes
    something else, and the one it stops on it stops on with a message naming a
    quantity internal to it.
    """

    SOUND = Profile(top=3, element_order=1)

    def test_a_size_a_drawing_can_be_meshed_to_draws_no_complaint(self):
        assert coverage.asked(Demand(coarsest=8.0, finest=2.0), self.SOUND) == []
        assert coverage.asked(Demand(coarsest=8.0, finest=0.0), self.SOUND) == []

    def test_a_ceiling_of_nothing_or_less(self):
        assert "an element has a size" in " ".join(
            coverage.asked(Demand(coarsest=0.0, finest=0.0), self.SOUND)
        )
        assert "an element has a size" in " ".join(
            coverage.asked(Demand(coarsest=-1.0, finest=0.0), self.SOUND)
        )

    def test_a_floor_below_nothing(self):
        assert "a length is not less than nothing" in " ".join(
            coverage.asked(Demand(coarsest=8.0, finest=-1.0), self.SOUND)
        )

    def test_a_floor_above_the_ceiling(self):
        """Taken by Gmsh and meshed to the ceiling, so the request that was met
        is not the one that was made."""
        assert "the floor stands above the ceiling" in " ".join(
            coverage.asked(Demand(coarsest=1.0, finest=10.0), self.SOUND)
        )

    def test_a_size_that_is_not_a_length(self):
        """Taken by Gmsh as no size at all, so the drawing decides and the
        request reaches nothing."""
        for size in (float("nan"), float("inf")):
            assert "which is not a length" in " ".join(
                coverage.asked(Demand(coarsest=size, finest=0.0), self.SOUND)
            )
            assert "which is not a length" in " ".join(
                coverage.asked(Demand(coarsest=8.0, finest=size), self.SOUND)
            )

    def test_an_order_below_one(self):
        """Meshed by Gmsh as one."""
        for order in (0, -1):
            assert "the lowest an element has is one" in " ".join(
                coverage.asked(Demand(8.0, 2.0), Profile(top=3, element_order=order))
            )

    def test_a_count_round_a_turn_that_is_no_whole_number(self):
        """Kept by Gmsh as a whole number, which is not the count asked for."""
        for count in (0.5, 12.5, -1, float("nan"), float("inf")):
            assert "a count round a turn is a whole number, or none at all" in " ".join(
                coverage.asked(Demand(8.0, 2.0, per_turn=count), self.SOUND)
            )

    def test_no_count_round_a_turn_and_a_count_of_one_are_both_counts(self):
        for count in (0, 1, 6):
            assert coverage.asked(Demand(8.0, 2.0, per_turn=count), self.SOUND) == []

    def test_the_largest_order_is_left_to_gmsh(self):
        """Where that limit is belongs to Gmsh, which refuses an order it cannot
        build and names it. A number written down here to stand in for it would
        be a limit of ours claiming to be Gmsh's.
        """
        assert coverage.asked(Demand(8.0, 2.0), Profile(top=3, element_order=99)) == []


class TestWhatAPlaceIsRefusedForBeforeAnythingIsDrawn:
    """A size at a place is a few numbers and a label, and every question about
    them is answered before a drawing is read.

    Each refusal names the place by the name its caller gave it, which is what
    the caller reads its answer under.
    """

    SOUND = Profile(top=3, element_order=1)

    def asked(self, *places, coarsest=8.0, finest=0.0, growth=1.5):
        return " ".join(
            coverage.asked(
                Demand(coarsest=coarsest, finest=finest, growth=growth, places=places),
                self.SOUND,
            )
        )

    def placed(self, *places, top=3, remainder="", marks=(), walls=(), mirrors=()):
        pieces = [Piece("air", 3, "air.brep"), Piece("sheet", 2, "sheet.brep")]
        pieces.append(Piece("wire", 1, "wire.brep"))
        pieces.append(Piece("dot", 0, "dot.brep"))
        pieces.append(Piece("post", 3, "post.brep", leaves=True))
        return " ".join(
            coverage.placed(
                Demand(
                    coarsest=8.0,
                    finest=0.0,
                    growth=1.5,
                    places=places,
                    walls=walls,
                    mirrors=mirrors,
                ),
                pieces,
                top,
                remainder,
                marks,
            )
        )

    def test_a_place_of_every_kind_that_describes_a_size_draws_no_complaint(self):
        places = (
            Near("near", "sheet", 1.0),
            AtRim("rim", "sheet", 1.0),
            Within("within", "air", 1.0),
        )
        assert self.asked(*places) == ""
        assert self.placed(*places) == ""

    def test_a_size_at_or_above_the_coarsest_asks_for_nothing(self):
        """A size field refines and never coarsens, so Gmsh would mesh exactly
        what the ceiling alone gives."""
        for size in (8.0, 9.0):
            assert "'edge' asks for" in self.asked(AtRim("edge", "sheet", size))
            assert "refines and never coarsens" in self.asked(AtRim("edge", "sheet", size))

    def test_a_size_under_the_floor(self):
        said = self.asked(Near("edge", "sheet", 0.5), finest=1.0)
        assert "'edge'" in said and "the floor stands above the place" in said

    def test_a_size_that_is_not_one(self):
        for size in (0.0, -1.0, float("nan")):
            assert "'edge' asks for" in self.asked(Near("edge", "sheet", size))

    def test_a_growth_that_lays_no_transition(self):
        for growth in (1.0, 0.5, float("nan")):
            assert "grows by more than one" in self.asked(Near("edge", "sheet", 1.0), growth=growth)
            assert "grows by more than one" in self.asked(
                Within("body", "air", 1.0), Near("near", "air", 1.0), growth=growth
            )

    def test_the_growth_is_asked_of_no_demand_whose_places_lay_no_transition(self):
        """A size throughout a label stops at its boundary and reads no growth."""
        assert self.asked(Within("body", "air", 1.0), growth=0.0) == ""

    def test_the_growth_is_asked_of_no_demand_without_a_place(self):
        assert coverage.asked(Demand(coarsest=8.0, finest=0.0, growth=0.0), self.SOUND) == []

    def test_two_places_under_one_name(self):
        said = self.asked(Near("edge", "sheet", 1.0), AtRim("edge", "sheet", 1.0))
        assert "more than one place is called 'edge'" in said

    def test_a_label_no_piece_is_drawn_under(self):
        said = self.placed(Near("edge", "nothing", 1.0))
        assert "'edge' names 'nothing', and no piece is drawn under that label" in said

    def test_a_mark_is_a_name_a_place_may_name(self):
        ring = Mark("ring", 2, "ring.brep")
        assert self.placed(Near("edge", "ring", 1.0), marks=[ring]) == ""
        assert "'edge' names 'ring'" in self.placed(Near("edge", "ring", 1.0))
        assert "'edge' asks for a size throughout 'ring'" in self.placed(
            Within("edge", "ring", 1.0), marks=[ring]
        )

    def test_the_remainder_is_a_label_a_place_may_name(self):
        """It is drawn as nothing, and it is known to stand where the model ends."""
        assert self.placed(AtRim("edge", "walls", 1.0), remainder="walls") == ""
        assert "'walls'" in self.placed(AtRim("edge", "walls", 1.0))

    def test_a_rim_of_what_has_only_points_round_it(self):
        """Round a curve in a volume stand points, and round a point nothing."""
        assert "'edge' asks for a size at the rim of 'wire'" in self.placed(
            AtRim("edge", "wire", 1.0)
        )
        assert self.placed(AtRim("edge", "wire", 1.0), top=2) == ""
        assert "'edge'" in self.placed(AtRim("edge", "dot", 1.0), top=2)

    def test_the_room_turns_round_the_rim_of_faces_or_of_a_body_that_leaves(self):
        """A body that leaves the model leaves its skin, which is faces."""
        for label in ("sheet", "post"):
            assert self.placed(AtRim("edge", label, 1.0, reentrant=True), walls=[label]) == ""
        assert self.placed(AtRim("edge", "walls", 1.0, reentrant=True), remainder="walls") == ""

    def test_the_room_turns_round_no_rim_of_a_body_or_in_a_flat_mesh(self):
        """The room opens round a curve, which a rim of a body in the model is not,
        and which a mesh of faces has none of."""
        said = self.placed(AtRim("edge", "air", 1.0, reentrant=True))
        assert "'edge' asks for a size where the room turns round the rim of 'air'" in said
        assert "stands at dimension 3" in said
        said = self.placed(AtRim("edge", "sheet", 1.0, reentrant=True), top=2)
        assert "in a mesh filling dimension 2" in said

    def test_a_wall_or_a_mirror_is_a_label_of_faces(self):
        assert self.placed(walls=["sheet", "post"], mirrors=["sheet"]) == ""
        said = self.placed(walls=["air", "nothing"], mirrors=["wire"])
        assert "the wall 'air' is of dimension 3" in said
        assert "the wall 'nothing' is no label drawn" in said
        assert "the mirror 'wire' is of dimension 1" in said
        ring = Mark("ring", 2, "ring.brep")
        assert "the wall 'ring' is no label drawn" in self.placed(walls=["ring"], marks=[ring])

    def test_a_size_throughout_a_label_below_the_dimension_filled(self):
        assert "'body' asks for a size throughout 'sheet'" in self.placed(
            Within("body", "sheet", 1.0)
        )

    def test_a_size_throughout_a_label_that_leaves_the_model(self):
        """It holds no element once it has gone, and its skin is sized by a size
        near it or at its rim."""
        assert "'body' asks for a size throughout 'post', which leaves the model" in (
            self.placed(Within("body", "post", 1.0))
        )
        assert self.placed(Near("near", "post", 1.0), AtRim("rim", "post", 1.0)) == ""

    def test_a_size_near_a_volume_is_asked_of_a_volume_mesh(self):
        """It is laid throughout the volume, and above the dimension filled
        nothing in the mesh would carry it."""
        assert self.placed(Near("near", "air", 1.0)) == ""
        assert "'near' asks for a size near 'air'" in self.placed(Near("near", "air", 1.0), top=2)


class TestWhatAMarkIsRefusedForBeforeAnythingIsDrawn:
    PIECES = (Piece("air", 3, "air.brep"), Piece("sheet", 2, "sheet.brep"))

    def said(self, *marks, remainder="walls"):
        return " ".join(coverage.marked(marks, self.PIECES, 3, remainder))

    def test_a_mark_of_each_dimension_the_mesh_holds_draws_no_complaint(self):
        marks = [Mark(f"at {dim}", dim, "shape.brep") for dim in (0, 1, 2, 3)]
        assert self.said(*marks) == ""

    def test_a_mark_under_a_labels_name(self):
        """A place names a label or a mark, and one name cannot be both."""
        assert "the mark 'sheet' carries the name of a label" in self.said(
            Mark("sheet", 2, "sheet.brep")
        )
        assert "the mark 'walls'" in self.said(Mark("walls", 2, "walls.brep"))

    def test_one_name_at_two_dimensions(self):
        said = self.said(Mark("ring", 2, "face.brep"), Mark("ring", 1, "edge.brep"))
        assert "the mark 'ring' is drawn at dimensions [1, 2]" in said
        assert self.said(Mark("ring", 2, "one.brep"), Mark("ring", 2, "two.brep")) == ""

    def test_a_mark_above_the_dimension_filled(self):
        assert "the mark 'ring' is drawn at dimension 4" in self.said(Mark("ring", 4, "r.brep"))
        assert "the mark 'ring' is drawn at dimension -1" in self.said(Mark("ring", -1, "r.brep"))


class TestTheProfile:
    def test_the_top_is_the_dimension_filled(self):
        assert Profile(top=3).top == 3
        assert Profile(top=2).top == 2

    def test_a_dimension_a_mesh_is_made_of(self):
        """The profile takes it without a word; what refuses one it cannot fill
        is the run, where the message can say what a mesh is made of."""
        assert Profile(top=0).top == 0

    def test_a_surface_method_and_a_volume_method_state_the_same_field(self):
        """The value the profile carries that the first backend never sets.

        A straight-sided mesh into a volume method is legal; the same mesh into
        a method working on the surface of a closed body is a refusal. So the
        profile is not a place to put one backend's one knob.
        """
        assert Profile(top=3, element_order=2, curved=False).curved is False
        assert Profile(top=2, element_order=2, curved=True).curved is True
