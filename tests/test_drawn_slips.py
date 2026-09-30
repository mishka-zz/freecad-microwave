# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A slip a body of the medium's own material takes, off stand-ins for the
kernel.

Which body takes a slip, and what the kernel makes of that body grown, is driven
under a real FreeCAD by ``tests/openems_palace_rooms_probe.py``. Here the rule
is held where it decides without the kernel: a room no body of the medium
stands beside, or one reaching where the model is driven, is taken by nothing;
and a fuse the kernel answers wrongly is refused rather than solved.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Microwave import drawn

#: A slip's solid as the kernel answers it once the body beside it has taken
#: every face: nothing is left to face anything across the gap.
TAKEN = SimpleNamespace(Faces=[], Area=2.004)


def room(least=(0.0, 0.0, 0.0), most=(1.0, 1.0, 0.001), bounded_by=("Guide",), solid=None):
    """A slip, whose solid nothing may ask about unless one is given."""
    return drawn.Room(
        solid=solid,
        volume=0.001,
        surface=2.004,
        least=least,
        most=most,
        sides=(),
        bounded_by=bounded_by,
    )


class TestWhichBodyTakesASlip:
    def test_none_where_no_body_of_the_medium_stands_beside_it(self):
        assert drawn.joined(room(), {"Guide": []}, set(), {"Guide": 1.0}) is None

    def test_none_where_it_reaches_a_port_or_a_lumped_element(self):
        """The gap there is where the model is driven, so the rule stops before
        it asks the kernel anything."""
        clear = [((0.5, 0.5, 0.0), (0.6, 0.6, 0.0))]
        assert drawn.joined(room(), {"Guide": []}, {"Guide"}, {"Guide": 1.0}, clear) is None

    def test_a_box_stated_from_either_corner_is_reached_alike(self):
        """A port states its corners from the end it drives to the end it returns
        on, so the greater corner may come first."""
        clear = [((0.6, 0.6, 0.5), (0.5, 0.5, -0.5))]
        assert drawn.joined(room(), {"Guide": []}, {"Guide"}, {"Guide": 1.0}, clear) is None

    def test_a_box_it_does_not_reach_is_no_reason(self):
        clear = [((2.0, 2.0, 2.0), (3.0, 3.0, 3.0))]
        taken = room(solid=TAKEN)
        assert drawn.joined(taken, {"Guide": []}, {"Guide"}, {"Guide": 1.0}, clear) == "Guide"


class Kept:
    """A face the room keeps once the body beside it has taken it, standing
    ``at`` along one line, a unit square in area."""

    def __init__(self, at):
        self.at, self.Area = at, 1.0
        self.Faces = [self]

    def distToShape(self, other):
        return (abs(self.at - other.at),)


def kept(*faces):
    """A slip's solid whose faces the body taking it leaves as ``faces``."""
    return SimpleNamespace(Faces=list(faces), Area=2.004)


class TestThePairTest:
    """Two faces the room keeps that stand across the gap from each other leave
    it open whichever body grows. The least extent round the room is a
    millimetre, so faces nearer than a micron face each other."""

    def test_faces_across_the_gap_leave_it_open(self):
        facing = room(solid=kept(Kept(0.0), Kept(0.0005)))
        assert drawn.joined(facing, {"Guide": []}, {"Guide"}, {"Guide": 1.0}) is None

    def test_faces_that_meet_do_not(self):
        meeting = room(solid=kept(Kept(0.0), Kept(0.0)))
        assert drawn.joined(meeting, {"Guide": []}, {"Guide"}, {"Guide": 1.0}) == "Guide"

    def test_faces_further_apart_than_a_slip_do_not(self):
        apart = room(solid=kept(Kept(0.0), Kept(0.01)))
        assert drawn.joined(apart, {"Guide": []}, {"Guide"}, {"Guide": 1.0}) == "Guide"


class Fused:
    """What the kernel hands back for a body fused with rooms."""

    def __init__(self, volume, solids=1, valid=True):
        self.Volume, self.Area = volume, 10.0
        self.Solids = [object()] * solids
        self.valid = valid

    def removeSplitter(self):
        return self

    def isValid(self):
        return self.valid


def body(fused, solids=1):
    return SimpleNamespace(Volume=4.0, Solids=[object()] * solids, fuse=lambda rooms: fused)


class TestTheBodyGrown:
    def test_is_the_kernels_answer_where_it_holds_the_body_and_the_room(self):
        fused = Fused(4.001)
        assert drawn.grown(body(fused), [room()]) is fused

    def test_two_solids_the_room_bridges_come_back_as_one(self):
        fused = Fused(4.001)
        assert drawn.grown(body(fused, solids=2), [room()]) is fused

    @pytest.mark.parametrize(
        ("fused", "said"),
        [
            (Fused(4.001, valid=False), "not a valid shape"),
            (Fused(4.1), "where the two hold"),
            (Fused(4.001, solids=2), "comes apart"),
        ],
    )
    def test_is_refused_where_the_kernel_answers_something_else(self, fused, said):
        with pytest.raises(ValueError, match=said):
            drawn.grown(body(fused), [room()])
