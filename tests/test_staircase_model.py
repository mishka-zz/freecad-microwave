# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the staircase does to a curved conductor, and what growing it buys.

Every claim here is about a coaxial line, whose impedance is exact and depends
on its two radii through ``ln(b / a)`` alone - so a surface the engine builds
the wrong size answers with the wrong impedance and the closed form says by how
much. Both radii are wrong in the same direction, which is what makes the line
sensitive where a single surface would not be.

The instrument is :mod:`tests.staircase_model`, which reproduces openEMS' edge
rule and solves the cross-section as electrostatics. It is a model of one rule
and not of the engine, so what it is trusted for is *behaviour against the cell*
- an order, a direction, a comparison between two treatments of one grid - and
never for a figure the acceptance suite's real solves should be giving.
"""

from __future__ import annotations

import numpy as np
import pytest

from Microwave.Solvers.openems.staircase import GROWN_BY
from tests import convergence
from tests.analytic import reference
from tests.staircase_model import capacitance, impedance, recession, uniform

#: The line, in mm. Its ratio is what sets the answer; the sizes are chosen so
#: the coarsest grid below still has a conductor several cells across.
INNER = 1.0
OUTER = 3.5

#: How far the grid reaches. Past the shield the field is zero, so this only has
#: to hold the shield itself.
SPAN = 4.6

#: A second line, whose return path is pushed far enough out that the inner
#: surface carries nearly all of the error. It is what reads one convex surface
#: on its own, where the line above averages a convex surface with a concave one.
WIDE_OUTER = 10.0
WIDE_SPAN = 10.6

#: Cells to solve at, in mm. Spread far enough apart that a rate between them is
#: about the cell rather than about the scatter, and reaching a cell fine enough
#: that a wrong coefficient and a right one are no longer within reach of the
#: staircase's own jumping about.
CELLS = (0.22, 0.16, 0.12, 0.09, 0.07, 0.055)

#: Where the axis sits relative to the grid. Centred on a line is what a mesher
#: pinning lines to a drawing gives; the other two break that, one by moving
#: every line and one by spacing the two axes differently.
GOLDEN = (1 + 5**0.5) / 2


def ideal() -> float:
    return reference.coaxial_impedance(INNER, OUTER, 1.0)


def error(cell: float, grown: float, offset: float = 0.0, stretch: float = 1.0) -> float:
    """Fractional error of the staircased line, grown by ``grown`` cells."""
    x = uniform(cell, SPAN)
    y = uniform(cell * stretch, SPAN, offset)
    measured = impedance(x, y, INNER + grown * cell, OUTER - grown * cell * stretch, eps_r=1.0)
    return measured / ideal() - 1.0


class TestTheModelMeasuresTheRuleItClaimsTo:
    @pytest.mark.parametrize("axis", [0, 1])
    def test_a_capacitor_the_grid_holds_exactly_comes_out_as_arithmetic(self, axis):
        """The solve itself, against a case with no staircase in it.

        Two strips normal to an axis, their faces on grid lines, hold a uniform
        field between them and nothing else - so the capacitance is the width
        over the gap exactly, and any error is the assembly rather than the
        geometry. It is the one arrangement here whose answer owes nothing to
        the sampling rule, which is what makes it the check the rest stands on:
        the conductance weights, the energy sum, and which nodes were found to
        be a conductor at all.

        Along each axis in turn, because the transverse extent an edge carries
        is read off the *other* axis - so a weight fetched from the wrong one
        answers correctly in one orientation and not in the other. Graded across
        the gap for the same reason at one remove: a weight read from a single
        spacing rather than from each line's own neighbours also answers
        correctly until the spacings differ.

        One gap and not two. The driven conductor is the one holding the origin
        and ground is the one holding the grid's far corner, so the strip at the
        other end is tied to neither - and a conductor at no fixed potential,
        with nothing beyond it to draw charge from, sits at the potential of
        what surrounds it and stores nothing.
        """
        driven, grounded = 0.5, 1.5
        graded = np.unique(
            np.concatenate(
                [
                    np.linspace(-2.0, -driven, 13),
                    np.linspace(-driven, driven, 9),
                    np.linspace(driven, 2.0, 13),
                ]
            )
        )
        lines = [np.linspace(-1.0, 1.0, 21), np.linspace(-1.5, 1.5, 16)]
        width = lines[1 - axis]
        lines[axis] = graded

        def strips(*point):
            return (np.abs(point[axis]) <= driven) | (np.abs(point[axis]) >= grounded)

        assert capacitance(*lines, strips) == pytest.approx(
            (width[-1] - width[0]) / (grounded - driven), rel=1e-12, abs=0.0
        )

    def test_a_conductor_comes_back_smaller_than_it_was_drawn(self):
        """An edge conducts only where its sample is inside the metal.

        So the metal is inscribed: the inner conductor loses and the bore opens,
        both of which raise ``ln(b / a)``. A line reading *low* would mean the
        sign of the whole mechanism is the other way round.
        """
        for cell in CELLS:
            assert error(cell, 0.0) > 0.0

    def test_what_the_grid_gives_up_is_first_order_in_the_cell(self):
        """The rule gives up a share of the cell, so the error follows the cell.

        This is the claim the correction exists to answer, and it is the one
        thing here that must be an order rather than a size: a first-order error
        is one refinement cannot afford to remove.
        """
        errors = [abs(error(cell, 0.0)) for cell in CELLS]
        assert convergence.falls_with_every_refinement(CELLS, errors)
        assert convergence.order_of(CELLS, errors) == pytest.approx(1.0, abs=0.35)

    def test_a_grid_fine_against_the_line_answers_the_closed_form(self):
        """Nothing but the staircase is between the model and the exact answer.

        Ungrown and finely meshed it has to approach ``ln(b / a)``, which is what
        says the potential solve and the capacitance behind it are right rather
        than merely self-consistent.
        """
        assert abs(error(0.04, 0.0)) < 0.02


class TestGrowingTheConductorBeforeItIsSampled:
    def test_the_growth_helps_at_every_cell(self):
        """The property the correction is for, and the one a regression breaks.

        Stated as a comparison between two treatments of the same grid, so it
        holds whatever the model's own absolute accuracy is - and it reads
        :data:`GROWN_BY` rather than restating it, so changing the shipped
        constant is answered here rather than only by an hour of solving.
        """
        for cell in CELLS:
            assert abs(error(cell, GROWN_BY)) < abs(error(cell, 0.0))

    def test_half_a_cell_carries_the_line_past_the_drawing(self):
        """A cylinder gives up less than half a cell, so the half over-shoots.

        The half is the mean of what a surface meeting the grid at every phase
        gives up, and a sphere many cells across does meet it that way. A
        cylinder does not, and the sign of that is what is asserted: over-grown
        at every cell, never short of the drawing.
        """
        assert all(error(cell, GROWN_BY) < 0.0 for cell in CELLS)

    def test_what_the_half_leaves_behind_does_not_refine_away(self):
        """It is a bias and not a residue, so a finer mesh does not reach it.

        This is what separates a correction of the wrong size from one that is
        merely imperfect: what a rounding leaves is proportional to the cell and
        falls with it, and what a wrong coefficient leaves is not.
        """
        errors = [abs(error(cell, GROWN_BY)) for cell in CELLS]
        assert not convergence.falls_with_every_refinement(CELLS, errors)

    def test_one_convex_surface_gives_up_less_than_it_is_handed(self):
        """The bias, measured on a single surface rather than on a pair.

        The line above averages a convex surface with a concave one. Pushing the
        shield out leaves the inner surface carrying nearly all of ``ln(b / a)``,
        and what it gives up is short of the half at every cell - which is the
        whole of the open question, stated without a figure of its own.

        What is *not* asserted is that growing by this much lands the answer.
        The recession and the growth that restores it are different quantities:
        a capacitance is a step function of the size a conductor is handed, so
        the growth is quantised by the same grid the recession was measured
        against, and at some cells the shipped half happens to land nearer than
        the measured recession does.
        """
        for cell in CELLS:
            x = uniform(cell, WIDE_SPAN)
            assert 0.0 < recession(x, x, INNER, WIDE_OUTER) / cell < GROWN_BY

    def test_where_the_axis_sits_does_not_set_the_size(self):
        """Moving the grid off the axis changes the path, not the destination.

        Worth pinning because it is the plausible cure that does not work: a
        circle centred on a line meets the grid in four-fold symmetry, and
        breaking that does smooth how the error approaches its bias without
        moving the bias. Grid placement is not what the half is wrong about.
        """
        fine = CELLS[-1]
        centred = abs(error(fine, GROWN_BY))
        broken = abs(error(fine, GROWN_BY, offset=1 / GOLDEN, stretch=GOLDEN / 1.5))
        assert abs(broken - centred) < 0.01
