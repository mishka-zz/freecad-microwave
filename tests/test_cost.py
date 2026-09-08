# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What measuring a drawing costs, held to how it grows rather than to a figure.

Every other test here asks whether the answer is right. A phase can be right
and unaffordable, and a suite of right answers cannot see the difference.

**The counts are read as a ratio and never as a threshold.** One drawing's cost
is a number nobody can fail - a bound written under it is a runtime written
down, which this project keeps nowhere because nothing re-derives it. The same
shape measured at two cells says how the cost grows with what is asked of it,
and that is a property of the code: a phase linear in its input stays linear
when the input doubles, on any machine and in any year.

The specimens and the two cells are ``tests/cost_probe.py``, which needs the
CAD kernel. Nothing here does. It measures each drawing and then prunes what it
measured, an axis at a time, which is the pair of phases a mesh pays for before
a line is laid.

**The gap walk is read somewhere else, and it needs no kernel.** Its samples are
spaced at the gap it is measuring rather than at the cell, so refining the cell
does not move them, and it needs a second body where these specimens are drawn
one at a time. What it spends is the arithmetic between a sample and a
containment answer, which a stand-in gives as a kernel does - so
``tests/test_lfs.py`` reads its growth, at two gaps, against the bound the
walk's own constants state.
"""

from __future__ import annotations

import os

import pytest

from tests.conftest import GROWTH_WORTH_READING, LINEAR_ENOUGH, probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "cost_probe.py")


@pytest.fixture(scope="module")
def spent(tmp_path_factory):
    """What each specimen spent at both cells, from a probe under a real FreeCAD."""
    return probe_manifest(PROBE, tmp_path_factory.mktemp("cost"), "COST_OUT", key="read")


def measured(spent, name):
    record = spent.get(name)
    assert record is not None, f"{name!r} is not in what the probe wrote"
    if "unavailable" in record:
        pytest.skip(f"{name} could not be drawn here: {record['unavailable']}")
    return record["coarse"], record["fine"]


def grew(coarse, fine, key):
    return fine[key] / coarse[key] if coarse[key] else 0.0


#: The specimens the growth is read on. What each is here for is that
#: refining the cell raises substantially more demands off it, which is what
#: the first test below asserts rather than assumes.
GROWING = ("fillet", "chamfer", "boolean_cut")


class TestTheCostGrowsNoFasterThanTheDrawing:
    """The mesher's own claim about each phase, held over a real drawing."""

    @pytest.mark.parametrize("name", GROWING)
    def test_refining_the_cell_raises_more_demands_to_grow_on(self, spent, name):
        """The guard against every ratio below being read on a shape that did
        not move. A specimen whose demands are set by its curvature alone
        raises the same set at either cell, and then a scan of any shape at all
        passes for having nothing to grow over."""
        coarse, fine = measured(spent, name)
        print(
            f"SPENT {name} raised={coarse['raised']}->{fine['raised']} "
            f"kept={coarse['kept']}->{fine['kept']} "
            f"compared={coarse['compared']}->{fine['compared']} "
            f"cast={coarse['cast']}->{fine['cast']} "
            f"tested={coarse['tested']}->{fine['tested']} "
            f"reachable={coarse['reachable']}->{fine['reachable']}"
        )
        assert grew(coarse, fine, "raised") > GROWTH_WORTH_READING

    @pytest.mark.parametrize("name", GROWING)
    def test_the_casts_follow_the_demands_they_are_made_for(self, spent, name):
        """A cast is made to answer a demand, so the two rise together. Without
        this the test below says nothing: it holds the triangles against the
        casts, and a phase whose casts went quadratic would drag the triangles
        up with them and satisfy it the whole way.
        """
        coarse, fine = measured(spent, name)
        assert grew(coarse, fine, "cast") <= grew(coarse, fine, "raised") ** LINEAR_ENOUGH

    @pytest.mark.parametrize("name", GROWING)
    def test_the_chord_costs_what_the_casts_cost(self, spent, name):
        """The index answers one cast independently of how many are made, so
        the triangles tested follow the casts rather than the sampling."""
        coarse, fine = measured(spent, name)
        assert grew(coarse, fine, "tested") <= grew(coarse, fine, "cast") ** LINEAR_ENOUGH

    @pytest.mark.parametrize("name", GROWING)
    def test_and_the_triangles_counted_are_some_and_not_all(self, spent, name):
        """Two guards on the test above, and each closes a different way of
        satisfying it for nothing. A count that stayed at zero makes its ratio
        zero and passes; and casts and triangles grow together with no index at
        all, so the ratio holds for a filter that filters nothing.
        """
        _, fine = measured(spent, name)
        assert 0 < fine["tested"] < fine["reachable"]

    @pytest.mark.xfail(
        strict=True,
        reason="the pruning scan is quadratic in the demands a drawing raises; "
        "asking the whole kept set at once made that cheap without making it linear",
    )
    @pytest.mark.parametrize("name", GROWING)
    def test_the_pruning_scan_is_linear_in_the_demands_it_is_given(self, spent, name):
        """What the scan undertakes, and does not do.

        Its own account of itself is that only demands of one size escape being
        compared against everything finer. What it compares against is every
        kept demand finer, not every kept size, so a face sampled all over at
        one curvature is one comparison per sample per later demand and the
        scan is quadratic in the drawing.

        Held as a failure rather than as the square, because the square is what
        the code does and asserting it would be the code written twice.
        """
        coarse, fine = measured(spent, name)
        assert grew(coarse, fine, "compared") <= grew(coarse, fine, "raised") ** LINEAR_ENOUGH
