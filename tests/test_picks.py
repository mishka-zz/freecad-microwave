# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a pick is, beyond the box it reduces to.

The drawing and the openEMS adapter both ask this about the same port, and a
port they answer differently about is one whose picture is not what the solver
gets. So the question is asked in one module, and these are its rules.
"""

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
    from .test_document_translation import Compound, Shape

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
