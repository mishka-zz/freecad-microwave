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
