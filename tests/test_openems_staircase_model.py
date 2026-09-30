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
from tests.staircase_model import (
    capacitance,
    impedance,
    recession,
    stripline,
    stripline_grid,
    uniform,
)

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

#: Offsets per axis the lattice is averaged over, so that what is read off a
#: sequence is the cell rather than where the cell fell. Square, since a line has
#: two transverse axes and they do not move together.
PHASES = 4

#: How far from one the uncorrected line's order may sit. Tighter than the grown
#: one below, this error being the several times larger of the two - which is
#: :data:`SMALLER_BY`'s subject - and so the one the lattice is a smaller share
#: of.
UNGROWN_ORDER = 0.35

#: How far from one the grown line's order may sit. Wide, because what the
#: correction leaves is small enough that the lattice is a large share of it even
#: averaged. What it has to exclude is the two things that were claimed about
#: this quantity and are not true of it: a fixed bias that refinement never
#: reaches, and the second order a boundary would have if the rounding were gone
#: rather than made smaller.
GROWN_ORDER = 0.5

#: How much smaller than the rounding it replaces what the correction leaves has
#: to be. It is the coefficient that the half changes, and this is the claim in
#: place of the one about the order.
SMALLER_BY = 2.0


def ideal() -> float:
    return reference.coaxial_impedance(INNER, OUTER, 1.0)


def error(
    cell: float, grown: float, offset: float = 0.0, stretch: float = 1.0, across: float = 0.0
) -> float:
    """Fractional error of the staircased line, grown by ``grown`` cells.

    ``offset`` and ``across`` slide the lattice along one axis and the other, in
    mm, which is the free variable a single grid per cell size never records.
    """
    x = uniform(cell, SPAN, across)
    y = uniform(cell * stretch, SPAN, offset)
    measured = impedance(x, y, INNER + grown * cell, OUTER - grown * cell * stretch, eps_r=1.0)
    return measured / ideal() - 1.0


def averaged(cell: float, grown: float) -> float:
    """The error with where the lattice fell averaged out, on both axes.

    A boundary's place inside its cell moves the answer, and a line through the
    axis is an extreme of that spread rather than a sample of it - so a sweep
    read at one offset per cell can draw a flat line through a falling one. It
    is the whole reason a rate is read off this and not off one grid apiece.
    """
    return float(
        np.mean(
            [
                error(cell, grown, offset=down / PHASES * cell, across=step / PHASES * cell)
                for step in range(PHASES)
                for down in range(PHASES)
            ]
        )
    )


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

        Off the lattice average, like the corrected line below it. This error is
        large enough that one offset per cell would answer the same, but a rate
        read at one offset is not a rate whatever it happens to say.
        """
        errors = [abs(averaged(cell, 0.0)) for cell in CELLS]
        assert convergence.falls_with_every_refinement(CELLS, errors)
        assert convergence.order_of(CELLS, errors) == pytest.approx(1.0, abs=UNGROWN_ORDER)

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

    def test_what_the_half_leaves_behind_is_first_order_too(self):
        """The growth changes what the term is worth and not the order of it.

        A curved surface goes on meeting the grid at every phase however fine the
        grid is, so what the correction leaves is still a share of the cell -
        smaller by a large factor, and falling at the same rate. Both halves are
        asserted, because the size on its own is what a correction of any wrong
        coefficient also delivers.

        Read off the lattice average. What is left is small enough that where the
        grid fell is a large part of it, so a single offset per cell is not a
        sequence: it jumps about, and reading that as a line that has stopped
        falling is the mistake averaging exists to prevent.
        """
        grown = [abs(averaged(cell, GROWN_BY)) for cell in CELLS]
        ungrown = [abs(averaged(cell, 0.0)) for cell in CELLS]
        assert convergence.order_of(CELLS, grown) == pytest.approx(1.0, abs=GROWN_ORDER)
        for cell, left, rounding in zip(CELLS, grown, ungrown, strict=True):
            assert left * SMALLER_BY < rounding, (
                f"at {cell} mm the correction leaves {100 * left:.3f} % against "
                f"{100 * rounding:.3f} % for the rounding it replaces"
            )

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


class TestAStriplineCrossSection:
    """The grid the closed form is scored against, held to being the geometry
    it claims. What it answers is checked in ``test_reference.py``, against the
    conformal mapping; what is here is the shape the solve is handed."""

    def test_a_wall_inside_the_strip_is_refused(self):
        """It would otherwise run off the end of an empty list, since the
        outward grading never takes a step. A box narrower than the conductor
        in it is a caller's mistake and is named as one."""
        for wall in (0.4, 0.5):
            with pytest.raises(ValueError, match="inside a strip"):
                stripline_grid(1.0, 1.0, 20, wall)

    def test_the_grid_reaches_past_the_wall_it_was_asked_for(self):
        """The outward grading stops at the first line beyond the wall, and that
        line is where the shield is put - so what the solve encloses is the box
        it reports rather than the one it was asked for."""
        lines, _, reached = stripline_grid(1.0, 1.0, 20, 6.0)
        assert reached >= 6.0
        assert lines.max() == pytest.approx(reached, rel=1e-12, abs=0.0)

    def test_the_strip_and_both_planes_land_on_lines(self):
        """The cross-section is only the drawn one if the sampling rule finds
        it: an edge between lines is a conductor of another width."""
        lines, across, _ = stripline_grid(1.0, 1.0, 20, 6.0)
        for wanted in (-0.5, 0.0, 0.5):
            assert np.min(np.abs(lines - wanted)) < 1e-12
        for wanted in (-0.5, 0.0, 0.5):
            assert np.min(np.abs(across - wanted)) < 1e-12

    def test_the_shield_is_one_conductor(self):
        """Both planes and both walls, or the solve grounds whichever piece
        holds the grid's far corner and leaves the rest floating."""
        metal = stripline(1.0, 1.0, 6.0)
        for corner in ((6.0, 0.5), (-6.0, -0.5), (0.0, 0.5), (6.0, 0.0)):
            assert metal(np.array([corner[0]]), np.array([corner[1]]))[0]

    def test_the_strip_has_no_thickness(self):
        """It is one line of nodes. Given thickness it would be a bar, and the
        closed form it is scored against is the zero-thickness one."""
        metal = stripline(1.0, 1.0, 6.0)
        assert metal(np.array([0.0]), np.array([0.0]))[0]
        assert not metal(np.array([0.0]), np.array([0.01]))[0]
