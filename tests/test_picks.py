# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a pick is, beyond the box it reduces to.

The drawing and the openEMS adapter both ask this about the same port, and a
port they answer differently about is one whose picture is not what the solver
gets. So the question is asked in one module, and these are its rules.
"""

import types

import pytest

from Microwave import picks


class Element:
    """A sub-shape, answering only what :mod:`Microwave.picks` asks of it."""

    def __init__(self, area):
        self.Faces = [self] if area else []


class Shape:
    def __init__(self, **elements):
        self.Faces = []
        self._elements = elements

    def getElement(self, name):
        return self._elements[name]


class Obj:
    def __init__(self, shape):
        self.Shape = shape


def picked(*names, **elements):
    return (Obj(Shape(**elements)), list(names))


class TestWhatAPickEncloses:
    def test_an_edge_encloses_nothing(self):
        assert picks.is_outline(picked("Edge2", Edge2=Element(area=False)))

    def test_a_face_encloses_its_own_area(self):
        assert not picks.is_outline(picked("Face1", Face1=Element(area=True)))

    def test_an_edge_beside_a_face_is_not_an_outline(self):
        """Every name, not the first. Reading one would let the drawing and the
        envelope reach different answers about a port neither refuses."""
        pick = picked("Edge2", "Face1", Edge2=Element(area=False), Face1=Element(area=True))
        assert not picks.is_outline(pick)

    def test_and_the_order_they_were_picked_in_does_not_change_it(self):
        pick = picked("Face1", "Edge2", Edge2=Element(area=False), Face1=Element(area=True))
        assert not picks.is_outline(pick)

    def test_two_edges_are_still_an_outline(self):
        pick = picked("Edge2", "Edge4", Edge2=Element(area=False), Edge4=Element(area=False))
        assert picks.is_outline(pick)


class TestAPickWithNothingUnderIt:
    def test_a_whole_shape_is_read_as_itself(self):
        """``(object, [''])`` is how a link with no sub-element is spelled, and
        a sheet picked whole encloses the area it covers."""
        whole = Shape()
        whole.Faces = [whole]
        assert not picks.is_outline((Obj(whole), [""]))

    def test_an_object_with_no_shape_is_not_an_outline(self):
        """It is not anything: the caller has no box to build from either, and
        says so first. Answering "outline" would be a guess."""

        class Bare:
            Shape = None

        assert not picks.is_outline((Bare(), [""]))

    @pytest.mark.parametrize("link", [(None, []), None])
    def test_nor_is_a_link_that_names_nothing(self, link):
        """An unconfigured port reaches this before anything refuses it, so it
        answers rather than raising - and the answer is the one that leaves the
        box alone."""
        assert not picks.is_outline(link)


# ---------------------------------------------------------------------------
# Which side of a pick the body is on
# ---------------------------------------------------------------------------


def drawn(*lumps, faces, volume=None):
    """A body made of boxes, with named faces on it, as a pick's owner.

    The kernel model is the translation's own, so there is one model of a shape
    in the suite rather than two. What it cannot hold is a curve or a taper,
    which is why the same rule is scored against a real FreeCAD in
    ``tests/launch_probe.py``.
    """
    from .test_openems_document_translation import Compound, Shape

    body = Compound(*(Shape(*lump) for lump in lumps))
    for name, at in faces.items():
        body._faces[name] = Shape(*at)
    if volume is not None:
        body.Solids[0].Volume = volume
    return body


#: A trace folded back on itself: a long arm, a spine, and a short arm over it.
#: The short arm ends at x = 4 and the whole fold's box reaches x = 10, so the
#: box centre sits past the end of the arm the port is on.
FOLD = (((0, 0, 0), (10, 1, 0.5)), ((0, 0, 0), (1, 3, 0.5)), ((0, 2, 0), (4, 3, 0.5)))
SHORT_ARM_END = ((4, 2, 0), (4, 3, 0.5))


class TestWhichSideTheBodyIsOn:
    def test_a_face_at_one_end_launches_along_the_body(self):
        bar = drawn(((0, 0, 0), (10, 1, 0.5)), faces={"near": ((0, 0, 0), (0, 1, 0.5))})
        assert picks.inward((Obj(bar), ["near"]), 0) == 1

    def test_and_the_far_end_launches_the_other_way(self):
        bar = drawn(((0, 0, 0), (10, 1, 0.5)), faces={"far": ((10, 0, 0), (10, 1, 0.5))})
        assert picks.inward((Obj(bar), ["far"]), 0) == -1

    def test_a_fold_answers_from_its_own_arm_and_not_from_its_box(self):
        """The fault the whole rule exists for.

        Reading the box, this answers +X, because the long arm carries the
        centre past the short arm's end - and the metal behind that face runs
        the other way. A reversed launch solves clean with the phase inverted,
        so this is the one that has to be right rather than merely refused.
        """
        fold = drawn(*FOLD, faces={"end": SHORT_ARM_END})
        assert fold.BoundBox.XMin + fold.BoundBox.XMax > 2 * 4.0
        assert picks.inward((Obj(fold), ["end"]), 0) == -1

    def test_an_arm_ending_at_the_box_centre_is_a_drawing_and_not_a_refusal(self):
        """The second shape the box rule turned away.

        A fold whose fed arm ends exactly at the centre of the whole fold's box
        left the old rule with a zero to take the sign of, and it refused a
        legal drawing for it. There is no zero to take here: the arm is on one
        side of that face and air is on the other.
        """
        fold = drawn(
            ((0, 0, 0), (10, 1, 0.5)),
            ((0, 0, 0), (1, 3, 0.5)),
            ((0, 2, 0), (5, 3, 0.5)),
            faces={"end": ((5, 2, 0), (5, 3, 0.5))},
        )
        assert fold.BoundBox.XMin + fold.BoundBox.XMax == 2 * 5.0
        assert picks.inward((Obj(fold), ["end"]), 0) == -1

    def test_a_face_with_body_on_both_sides_says_nothing(self):
        bar = drawn(((0, 0, 0), (10, 1, 0.5)), faces={"cut": ((5, 0, 0), (5, 1, 0.5))})
        assert picks.inward((Obj(bar), ["cut"]), 0) is None

    def test_nor_does_one_standing_clear_of_it(self):
        bar = drawn(((0, 0, 0), (10, 1, 0.5)), faces={"off": ((20, 0, 0), (20, 1, 0.5))})
        assert picks.inward((Obj(bar), ["off"]), 0) is None

    def test_two_names_that_disagree_answer_neither(self):
        """A pick may name more than one element, and two ends of one bar point
        into it from opposite directions. There is no single launch there."""
        bar = drawn(
            ((0, 0, 0), (10, 1, 0.5)),
            faces={"near": ((0, 0, 0), (0, 1, 0.5)), "far": ((10, 0, 0), (10, 1, 0.5))},
        )
        assert picks.inward((Obj(bar), ["near", "far"]), 0) is None

    def test_a_solid_wound_inside_out_says_nothing(self):
        """The kernel reads such a solid as the complement of the space it
        appears to occupy, so every meeting comes back the wrong way round. It
        is refused outright where the model is translated; here it is simply
        not answered.

        Asked of the solids rather than of the shape, because a compound of a
        flat conductor and a solid one beside it reports a negative volume of
        its own with nothing in it wound the wrong way - which is a conductor
        somebody draws, and ``sheet_and_post`` in ``launches.py`` is it."""
        bar = drawn(
            ((0, 0, 0), (10, 1, 0.5)), faces={"near": ((0, 0, 0), (0, 1, 0.5))}, volume=-5.0
        )
        assert picks.inward((Obj(bar), ["near"]), 0) is None

    def test_a_kernel_that_will_not_meet_them_says_nothing_either(self):
        """It refuses one step of the two, which is the case that separates a
        refusal from an answer: read as air, the other step alone would look
        like a clean launch into the body and the shape would never have been
        asked."""
        bar = drawn(((0, 0, 0), (10, 1, 0.5)), faces={"near": ((0, 0, 0), (0, 1, 0.5))})
        met = bar.common

        def refuse_the_step_out(other):
            if other.BoundBox.XMin < 0.0:
                raise RuntimeError("BOPAlgo_Builder not done")
            return met(other)

        bar.common = refuse_the_step_out
        assert picks.inward((Obj(bar), ["near"]), 0) is None

    def test_an_object_with_no_shape_answers_nothing(self):
        class Bare:
            Shape = None

        assert picks.inward((Bare(), ["Face1"]), 0) is None

    def test_a_shape_with_nothing_in_it_says_nothing(self):
        """A pick on an object whose recompute failed. The kernel raises on its
        own account rather than answering, which is not a different case from
        the ones above and must not read as one."""
        bar = drawn(((0, 0, 0), (10, 1, 0.5)), faces={"near": ((0, 0, 0), (0, 1, 0.5))})

        def null(other):
            raise ValueError("Null input shape")

        bar.common = null
        assert picks.inward((Obj(bar), ["near"]), 0) is None


class Root:
    """The top container of a tree, answering each path with the shape it holds."""

    def __init__(self, label, **held):
        self.Label = label
        self._held = held
        self.asked = []

    def getSubObject(self, path):
        self.asked.append(path)
        return self._held[path]


class Placed(Obj):
    """An object standing in containers, as ``Parents`` lists them."""

    def __init__(self, shape, *parents, label="Septum"):
        super().__init__(shape)
        self.Label = label
        self.Parents = list(parents)


class Node:
    """A document object along a selection's path."""

    def __init__(self, label, **attributes):
        self.Label = label
        self.__dict__.update(attributes)


class Pose:
    """A placement, recording what it was composed of."""

    def __init__(self, *parts):
        self.parts = parts

    def inverse(self):
        return Pose("less", *self.parts)

    def multiply(self, other):
        return Pose(*self.parts, *other.parts)

    def toMatrix(self):
        return self.parts


class Mover(Shape):
    """A shape that hands back ``moved`` for any transformation."""

    def __init__(self, moved):
        super().__init__()
        self._moved = moved
        self.matrix = None

    def transformGeometry(self, matrix):
        self.matrix = matrix
        return self._moved


class TestWhereAShapeIsShown:
    """An object's ``Shape`` is in the coordinates of the container it stands in,
    so the shape is asked of the top container along the path down to it."""

    def test_an_object_in_no_container_is_its_own_shape(self):
        own = Shape()
        assert picks.placed(Placed(own)) is own

    def test_an_object_in_a_container_is_what_the_top_one_holds_along_the_path(self):
        shown = Shape()
        root = Root("Assembly", **{"Inner.Septum.": shown})
        assert picks.placed(Placed(Shape(), (root, "Inner.Septum."))) is shown
        assert root.asked == ["Inner.Septum."]

    def test_one_path_listed_twice_is_one_place(self):
        """A body lists a feature once for each way it claims it."""
        shown = Shape()
        root = Root("Body", **{"Pad.": shown})
        assert picks.placed(Placed(Shape(), (root, "Pad."), (root, "Pad."))) is shown

    def test_two_places_are_refused_naming_both(self):
        one, other = Root("Assembly", **{"Septum.": Shape()}), Root("Group", **{"Septum.": Shape()})
        with pytest.raises(
            picks.Unplaced, match="'Septum' stands in more than one container"
        ) as said:
            picks.placed(Placed(Shape(), (one, "Septum."), (other, "Septum.")))
        assert "Assembly.Septum." in str(said.value) and "Group.Septum." in str(said.value)
        assert str(said.value).endswith(", so its place is ambiguous. Keep it in one")

    def test_two_roots_sharing_a_label_are_two_places(self):
        one, other = (
            Root("Assembly", **{"Septum.": Shape()}),
            Root("Assembly", **{"Septum.": Shape()}),
        )
        with pytest.raises(picks.Unplaced):
            picks.placed(Placed(Shape(), (one, "Septum."), (other, "Septum.")))

    def test_a_root_in_another_document_is_not_a_place(self):
        """Another open document linking to the container lists its own root,
        and that is its drawing rather than this one's."""
        home, away = object(), object()
        shown = Shape()
        here = Root("Assembly", **{"Septum.": shown})
        there = Root("Assembly", **{"Assembly.Septum.": Shape()})
        here.Document, there.Document = home, away
        obj = Placed(Shape(), (there, "Assembly.Septum."), (here, "Septum."))
        obj.Document = home
        assert picks.placed(obj) is shown

    def test_an_element_is_taken_off_the_shape_where_it_is_shown(self):
        face = Element(area=True)
        root = Root("Assembly", **{"Septum.": Shape(Face1=face)})
        assert picks.element(Placed(Shape(), (root, "Septum.")), "Face1") is face

    def test_an_element_named_through_a_container_is_refused_naming_it(self):
        with pytest.raises(picks.Unplaced, match="picked at Inner.Septum.Face1"):
            picks.element(Placed(Shape()), "Inner.Septum.Face1")

    def test_a_pick_is_read_where_it_is_shown(self):
        face = Element(area=False)
        root = Root("Assembly", **{"Septum.": Shape(Edge2=face)})
        pick = (Placed(Shape(Edge2=Element(area=True)), (root, "Septum.")), ["Edge2"])
        assert picks.shapes(pick) == [face]
        assert picks.is_outline(pick)

    def test_an_element_of_a_link_array_stands_where_the_array_puts_it(self):
        """The array lists it among its own elements, so the path from the array
        to it is its place rather than a copy."""
        shown = Shape()
        element = Placed(Shape(), label="Array_i0")
        array = Node("Array", LinkedObject=object(), ElementList=[element])
        root = Root("Assembly", **{"Array.Array_i0.": shown})
        root.getSubObjectList = lambda path: [root, array, element]
        element.Parents = [(root, "Array.Array_i0.")]
        assert picks.placed(element) is shown

    def test_a_container_is_its_own_shape_moved_by_the_containers_above_it(self):
        """A container answers no shape along a path that ends at it, measured
        under 1.1.1, so its own is moved by the placement the path accumulates
        less its own."""
        moved = Shape()
        own = Mover(moved)
        inner = Placed(own, label="Inner")
        inner.Placement = Pose("own")
        root = Root("Outer", **{"Inner.": None})
        root.getSubObject = lambda path, retType=0: Pose("path") if retType == 3 else None
        inner.Parents = [(root, "Inner.")]
        assert picks.placed(inner) is moved
        assert own.matrix == ("path", "less", "own")

    def test_the_original_a_link_shows_is_not_placed_by_the_link(self):
        own = Shape()
        septum = Placed(own)
        link = Node("Link", LinkedObject=septum)
        root = Root("Assembly", **{"Link.Septum.": Shape()})
        root.getSubObjectList = lambda path: [root, link, septum]
        septum.Parents = [(root, "Link.Septum.")]
        assert picks.placed(septum) is own


class Top(Node):
    """The object a pick was made through, answering each path with the chain
    of objects down it."""

    def __init__(self, label, chains, **attributes):
        super().__init__(label, **attributes)
        self._chains = chains

    def getSubObjectList(self, path):
        return [self, *self._chains[path]]


def entry(obj, *names):
    return picks.Selected(obj, tuple(names))


class TestWhatAPickNames:
    """``Gui.Selection.getCompleteSelection(0)`` keeps each pick on the object
    it was made through, with the path down to what it picked, one entry to a
    pick in the order picked."""

    def test_a_pick_through_a_container_lands_on_the_object_it_reaches(self):
        box = Node("Box")
        part = Top("Assembly", {"Box.Face2": [box]})
        assert picks.resolved([entry(part, "Box.Face2")]) == ([entry(box, "Face2")], [])

    def test_a_pick_inside_the_copy_a_link_shows_is_named_and_left_out(self):
        """It would name the original, where it was drawn, not the copy picked."""
        box = Node("Box")
        link = Top("Link", {"Box.Face2": [box]}, LinkedObject=Node("Assembly"))
        assert picks.resolved([entry(link, "Box.Face2")]) == ([], ["'Box' in 'Link'"])

    def test_a_pick_on_a_link_itself_stays_on_the_link(self):
        link = Top("Link", {"Face2": []}, LinkedObject=Node("Box"))
        assert picks.resolved([entry(link, "Face2")]) == ([entry(link, "Face2")], [])

    def test_a_pick_on_an_element_of_a_link_array_lands_on_the_element(self):
        element = Node("Array_i0")
        array = Top("Array", {"0.Face2": [element]}, LinkedObject=Node("Box"))
        array.ElementList = [element]
        assert picks.resolved([entry(array, "0.Face2")]) == ([entry(element, "Face2")], [])

    def test_the_element_is_the_name_at_the_end_of_the_path(self):
        """A body answers a face with its history before the name, measured
        under 1.1.1."""
        body = Top("Body", {";Face2;:H832,F;:H833:8,F.Face2": []})
        picked = picks.resolved([entry(body, ";Face2;:H832,F;:H833:8,F.Face2")])
        assert picked == ([entry(body, "Face2")], [])

    def test_a_whole_object_is_picked_with_no_names(self):
        box = Node("Box")
        assert picks.resolved([entry(box)]) == ([entry(box)], [])

    def test_picks_through_one_container_keep_their_place_among_the_others(self):
        """A port reads its source and its reference off the order."""
        a, b, c = Node("A"), Node("B"), Node("C")
        part = Top("Assembly", {"A.Face6": [a], "B.Face5": [b]})
        picked = picks.resolved([entry(part, "A.Face6"), entry(c, "Face6"), entry(part, "B.Face5")])
        assert picked == ([entry(a, "Face6"), entry(c, "Face6"), entry(b, "Face5")], [])

    def test_one_original_picked_twice_inside_a_copy_is_named_once(self):
        box = Node("Box")
        array = Top("Array", {"Box.Face1": [box], "Box.Face2": [box]}, LinkedObject=box)
        _, copies = picks.resolved([entry(array, "Box.Face1"), entry(array, "Box.Face2")])
        assert picks.copy_note(copies).count("'Box' in 'Array'") == 1
        assert picks.copy_note(copies).endswith(". Pick the original, or the link itself")

    def test_a_pick_in_another_document_is_not_kept(self):
        """The selection spans every open document, and a link cannot reach
        into another."""
        here, there = Node("Box"), Node("Other")
        made = [
            types.SimpleNamespace(Object=here, SubElementNames=(), DocumentName="Board"),
            types.SimpleNamespace(Object=there, SubElementNames=(), DocumentName="Elsewhere"),
        ]
        assert picks.resolved(made, "Board") == ([entry(here)], [])

    def test_a_path_that_cannot_be_listed_is_kept_whole(self):
        """Read as the element at its end it would name another face; kept
        whole, :func:`picks.element` refuses it by name."""
        part = Node("Assembly")
        assert picks.resolved([entry(part, "Box.Face2")]) == ([entry(part, "Box.Face2")], [])

    def test_picks_interleaved_across_objects_keep_the_order_they_were_made_in(self):
        """A lumped port takes the first two: a source on A, then a reference
        on C, though A was picked again after."""
        a, c = Node("A"), Node("C")
        part = Top("Assembly", {"A.Face6": [a], "A.Face5": [a]})
        made = [entry(part, "A.Face6"), entry(c, "Face5"), entry(part, "A.Face5")]
        expected = [entry(a, "Face6"), entry(c, "Face5"), entry(a, "Face5")]
        assert picks.resolved(made) == (expected, [])
