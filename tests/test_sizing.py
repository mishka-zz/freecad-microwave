# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Unit tests for the sizing criterion and its allocation.

The criterion is the one place where the whole mesher's guarantee is stated as
arithmetic, so it is tested as arithmetic: the constraint is exactly tight, the
degenerate directions give the answers a reader can work out by hand, and the
allocation that was *not* chosen is shown doing the thing it was rejected for.
"""

import math

import numpy as np
import pytest

import Microwave.Solvers.openems.mesh as mesh
from Microwave.Solvers.openems.mesh import MaterialClass, Region, generate_mesh_lines
from Microwave.Solvers.openems.sizing import (
    Feature,
    connection,
    demands,
    elements,
    separation,
)
from tests.mesh_fixtures import DOMAIN, cell_at, params, stackup, substrate

AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
DIAGONAL = (1.0, 1.0, 1.0)


def spend(sizes, normal):
    """``sum_i |m_i| h_i``, the quantity the separation criterion bounds."""
    length = math.sqrt(sum(value**2 for value in normal))
    return math.fsum(
        abs(value) / length * size for value, size in zip(normal, sizes) if math.isfinite(size)
    )


def spanned(sizes, normal, thickness):
    """How many cell pitches lie across a slab, measured along its own normal.

    Grid planes of axis ``i`` go by at ``|m_i|/h_i`` per unit length walked
    along ``m``, so the pitch along ``m`` is the reciprocal of their sum and the
    slab is this many of them thick. Computed off the sizes rather than off the
    thickness, so it is the criterion's claim read back rather than restated.
    """
    length = math.sqrt(sum(value**2 for value in normal))
    return thickness * math.fsum(
        abs(value) / length / size for value, size in zip(normal, sizes) if math.isfinite(size)
    )


def cells_met(sizes, normal, thickness, offsets):
    """How many cells one ray across the slab lies in, from a given grid offset.

    What :func:`spanned` measures is a spacing, and this is the tally that
    spacing produces on a particular line through the layer - the two agree on
    an axis-aligned normal and part company off it, which is the distinction the
    criterion is careful about.
    """
    length = math.sqrt(sum(value**2 for value in normal))
    crossed = 0
    for value, size, start in zip(normal, sizes, offsets):
        if not math.isfinite(size):
            continue
        stop = start + thickness * abs(value) / length
        crossed += math.floor(stop / size) - math.floor(start / size)
    return crossed + 1


class TestSeparation:
    def test_the_constraint_is_exactly_tight(self):
        """No budget wasted, and none overspent - the point of the allocation."""
        for normal in [(1.0, 2.0, 3.0), (0.3, -0.9, 0.1), (1.0, 1.0, 0.0), DIAGONAL]:
            assert spend(separation(4.0, normal), normal) == pytest.approx(4.0, rel=1e-12)

    def test_an_axis_aligned_face_constrains_only_its_own_axis(self):
        """Manhattan geometry costs nothing on the axes it does not touch.

        This is the property that lets an ordinary board keep its anisotropic
        grid: a face normal to x says what x may be and leaves y and z free.
        """
        for dim, normal in enumerate(AXES):
            sizes = separation(0.5, normal)
            assert sizes[dim] == pytest.approx(0.5, rel=1e-12)
            assert all(math.isinf(sizes[other]) for other in range(3) if other != dim)

    def test_the_body_diagonal_gives_an_equal_share(self):
        """A face pointing equally at all three axes gets a cubic cell."""
        sizes = separation(1.0, DIAGONAL)
        assert sizes == pytest.approx((1 / math.sqrt(3),) * 3, rel=1e-12)

    def test_a_normal_need_not_arrive_normalised(self):
        assert separation(2.0, (0.0, 3.0, 4.0)) == pytest.approx(
            separation(2.0, (0.0, 0.6, 0.8)), rel=1e-12
        )

    def test_it_scales_with_the_thickness(self):
        one = separation(1.0, (1.0, 2.0, 0.5))
        ten = separation(10.0, (1.0, 2.0, 0.5))
        assert ten == pytest.approx(tuple(10.0 * value for value in one), rel=1e-12)

    def test_which_way_a_face_points_does_not_change_how_thick_it_is(self):
        """The criterion is even in the normal, so a sign must not reach it.

        Worth its own test because the failure is silent and in the unsafe
        direction: a component read with its sign intact falls below the floor
        for negatives, its axis comes back unconstrained, and a gap gets no
        demand at all on the axis it is measured along.
        """
        normal = (0.3, -0.9, 0.1)
        assert all(math.isfinite(value) for value in separation(1.0, normal))
        assert separation(1.0, normal) == separation(1.0, tuple(-v for v in normal))


class TestTheNormalFloor:
    """Where a component stops counting as a direction the feature faces.

    It has to be a threshold rather than an exact zero: a normal off a CAD
    kernel carries rounding, and the allocation divides by the square root of
    the component, so a component of 1e-30 would otherwise demand cells 1e15
    times finer than the feature it came from.
    """

    def test_a_component_below_it_leaves_its_axis_unconstrained(self):
        assert math.isinf(separation(1.0, (1.0, 1e-9, 0.0))[1])

    def test_a_component_above_it_does_not(self):
        assert math.isfinite(separation(1.0, (1.0, 1e-3, 0.0))[1])

    def test_and_the_axes_that_remain_are_unaffected_by_the_rounding(self):
        """Dropping a negligible component must not disturb the real ones."""
        assert separation(1.0, (1.0, 1e-9, 0.0))[0] == pytest.approx(
            separation(1.0, (1.0, 0.0, 0.0))[0], rel=1e-6
        )


class TestContinuity:
    """A face a thousandth of a degree off axis must not change the answer.

    Imperfect input is a requirement, so this is the property that chose the
    allocation. The comparison is against the allocation that minimises total
    cell count instead - an equal share among the axes the normal touches -
    which is cheaper and discontinuous, and the discontinuity is what disqualifies
    it. Asserted as a comparison rather than against a fixed tolerance: the
    allocation has a square-root cusp at an axis, so any absolute figure here
    would be pinning the cusp's shape rather than the property.
    """

    TILT = math.radians(1e-3)

    def cheapest(self, thickness, normal):
        """``h_i = t/(k|m_i|)`` over the ``k`` axes the normal touches."""
        length = math.sqrt(sum(value**2 for value in normal))
        active = [abs(value) / length for value in normal if abs(value) / length > 1e-6]
        return thickness / (len(active) * max(active))

    def test_a_hair_off_axis_barely_moves_this_allocation(self):
        square = separation(1.0, (1.0, 0.0, 0.0))[0]
        tilted = separation(1.0, (math.cos(self.TILT), math.sin(self.TILT), 0.0))[0]
        assert tilted == pytest.approx(square, rel=0.01)

    def test_and_halves_the_one_that_minimises_cell_count(self):
        square = self.cheapest(1.0, (1.0, 0.0, 0.0))
        tilted = self.cheapest(1.0, (math.cos(self.TILT), math.sin(self.TILT), 0.0))
        assert tilted == pytest.approx(square / 2.0, rel=1e-3)

    def test_the_tilted_axis_grows_without_bound_as_the_tilt_closes(self):
        """The second axis leaves ``inf`` continuously rather than jumping.

        It is the axis that was unconstrained at zero tilt, so the way it
        arrives is the whole question: it comes in enormous and grows as the
        tilt shrinks, which is what makes the limit the axis-aligned answer.
        """
        previous = 0.0
        for tilt in (1e-2, 1e-3, 1e-4, 1e-5):
            free = separation(1.0, (math.cos(tilt), math.sin(tilt), 0.0))[1]
            assert free > previous
            previous = free
        assert previous > 100.0


class TestConnection:
    def test_the_constraint_is_exactly_tight(self):
        sizes = connection(3.0)
        assert math.sqrt(sum(value**2 for value in sizes)) == pytest.approx(3.0, rel=1e-12)

    def test_a_cell_fits_inside_the_feature(self):
        """The cell whose body diagonal is the thickness."""
        assert connection(1.0) == pytest.approx((1 / math.sqrt(3),) * 3, rel=1e-12)

    def test_it_is_stronger_than_separation_at_every_normal(self):
        """Satisfying connection satisfies separation whichever way the face points.

        The reason the two can be stated against one thickness. Checked over a
        sweep rather than argued, because it is the property that makes
        ``Feature.thickness`` mean one thing.
        """
        sizes = connection(1.0)
        for normal in [
            AXES[0],
            AXES[2],
            DIAGONAL,
            (1.0, 1.0, 0.0),
            (0.2, 0.9, -0.4),
            (1.0, 1e-9, 1e-9),
        ]:
            assert spend(sizes, normal) <= 1.0 + 1e-12

    def test_and_is_never_looser(self):
        """Equality only at a body diagonal, where the two criteria coincide."""
        assert spend(connection(1.0), DIAGONAL) == pytest.approx(1.0, rel=1e-12)
        assert spend(connection(1.0), AXES[0]) < 1.0


class TestElements:
    """Counting cells across a layer, rather than fitting one inside it.

    The other two criteria bound a displacement, so each is satisfied by a
    single cell and each is stated as an inequality with slack. A count is an
    equality - but an equality about a *pitch*, which is what :func:`spanned`
    reads back, and not about the cells any one ray happens to lie in, which is
    what :func:`cells_met` reads back. Keeping those apart is most of this
    class.
    """

    SKEW = ((1.0, 2.0, 3.0), (0.3, -0.9, 0.1), (1.0, 1.0, 0.0), DIAGONAL, *AXES)

    @pytest.mark.parametrize("count", [2, 4, 9])
    def test_the_pitch_along_the_normal_is_the_thickness_over_the_count(self, count):
        """Exactly, at every normal. The criterion is that spacing itself."""
        for normal in self.SKEW:
            sizes = elements(1.6, normal, count)
            assert spanned(sizes, normal, 1.6) == pytest.approx(count, rel=1e-12)

    @pytest.mark.parametrize("count", [2, 4, 9])
    def test_and_it_does_not_move_with_the_grid(self, count):
        """Which is what makes it a property of the allocation rather than of
        where the mesher happened to start laying lines. Stated because the
        tally below is *not* that, and the two are easy to confuse."""
        for normal in self.SKEW:
            sizes = elements(1.6, normal, count)
            for share in (0.0, 0.25, 0.5, 0.75):
                offsets = [share * size if math.isfinite(size) else 0.0 for size in sizes]
                assert spanned(sizes, normal, 1.6) == pytest.approx(count, rel=1e-12)
                assert cells_met(sizes, normal, 1.6, offsets) >= 1

    @pytest.mark.parametrize("count", [2, 4, 9])
    def test_an_axis_aligned_layer_lands_the_count_on_any_grid(self, count):
        """There the pitch and the tally agree, whatever the offset - a segment
        of ``n`` cells' length lies in ``n`` cells or one more. It is the case a
        flat board is in, which is why a box can pin its two faces and be done.
        """
        sizes = elements(1.6, (0.0, 0.0, 1.0), count)
        for share in (0.0, 0.1, 0.5, 0.9):
            met = cells_met(sizes, (0.0, 0.0, 1.0), 1.6, (0.0, 0.0, share * sizes[2]))
            assert count <= met <= count + 1

    def test_but_a_tilted_layer_can_fall_short_on_one_line_through_it(self):
        """The distinction the criterion is stated carefully about, pinned so it
        cannot quietly become a guarantee.

        Off axis the pitch is realised on three coarser axis pitches, so where
        the grid sits decides how many planes a *particular* ray meets. What
        holds is the spacing; a ray is not promised the count. Anything that
        claimed otherwise would be claiming this case away.
        """
        count = 2
        sizes = elements(1.0, DIAGONAL, count)
        assert spanned(sizes, DIAGONAL, 1.0) == pytest.approx(count, rel=1e-12)
        met = {
            cells_met(sizes, DIAGONAL, 1.0, [share * size for size in sizes])
            for share in (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9)
        }
        assert min(met) < count < max(met)

    @pytest.mark.parametrize("count", [2, 4, 9])
    def test_an_axis_aligned_layer_asks_what_a_box_asks(self, count):
        """A board drawn flat reaches the mesher as a region, which asks for its
        extent over the count on the axis it is thin along. Bend the same board
        and it arrives as a triangulation with nothing but this - so the two
        have to be the same number, or one board answers differently for having
        been drawn curved."""
        for dim, normal in enumerate(AXES):
            sizes = elements(1.6, normal, count)
            assert sizes[dim] == pytest.approx(1.6 / count, rel=1e-12)
            assert all(math.isinf(sizes[other]) for other in range(3) if other != dim)

    def test_a_count_of_one_asks_for_the_layers_own_extent(self):
        """Which is the rule's own off switch, and reads as one: a cell as long
        as the slab's shadow on an axis constrains nothing that axis was not
        already doing."""
        assert elements(1.6, (0.0, 0.0, 1.0), 1)[2] == pytest.approx(1.6, rel=1e-12)
        assert elements(1.6, (1.0, 0.0, 1.0), 1)[0] == pytest.approx(1.6 * math.sqrt(2), rel=1e-12)

    def test_it_releases_an_axis_continuously(self):
        """The fault the cell-count allocation was rejected for, and the reason
        this expression shares its shape without sharing it.

        Tilting a face off an axis sends that axis to ``inf`` smoothly, because
        the divisor is a count somebody declared rather than a tally of how many
        axes the normal happens to touch.
        """
        thick = [elements(1.0, (1.0, tilt, 0.0), 4)[1] for tilt in (1e-3, 1e-4, 1e-5)]
        assert thick[0] < thick[1] < thick[2]
        for one, other in zip(thick, thick[1:]):
            assert other == pytest.approx(10.0 * one, rel=1e-6)

    def test_it_is_never_stricter_than_the_same_count_of_separations(self):
        """They answer different questions and this says which is which.

        Separation bounds how far a rounded point moves along the normal, which
        is the whole cell diagonal projected there; a count only has to place
        that many cell pitches across the layer. So stating a count as ``n``
        separations would refine a tilted layer for nothing, by a factor that
        grows with the tilt - and it is a tilted layer this exists for.
        """
        for normal in self.SKEW:
            counted = elements(1.2, normal, 4)
            over = separation(1.2 / 4, normal)
            assert all(math.isinf(one) or one >= other - 1e-12 for one, other in zip(counted, over))
        # Equal only where the normal is on an axis, so the two are genuinely
        # different demands rather than one written twice.
        assert elements(1.2, AXES[0], 4)[0] == pytest.approx(separation(1.2 / 4, AXES[0])[0])
        assert elements(1.2, DIAGONAL, 4)[0] > separation(1.2 / 4, DIAGONAL)[0]

    def test_a_normal_need_not_arrive_normalised(self):
        assert elements(1.0, (0.0, 0.0, 7.0), 4) == elements(1.0, (0.0, 0.0, 1.0), 4)

    def test_which_way_the_layer_faces_does_not_change_the_count(self):
        assert elements(1.0, (1.0, -2.0, 3.0), 4) == elements(1.0, (-1.0, 2.0, -3.0), 4)

    def test_a_count_below_one_is_refused(self):
        for bad in (0, -3):
            with pytest.raises(ValueError, match="count"):
                elements(1.0, (0.0, 0.0, 1.0), bad)


class TestFeature:
    def test_a_feature_with_a_normal_is_a_separation(self):
        gap = Feature(0.2, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 1.0), "gap")
        assert gap.cells() == separation(0.2, (1.0, 0.0, 0.0))

    def test_a_feature_without_one_is_a_connection(self):
        wire = Feature(0.2, None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "wire")
        assert wire.cells() == connection(0.2)

    def test_a_thickness_that_is_not_a_length_is_refused(self):
        for bad in (0.0, -1.0, math.inf, math.nan):
            with pytest.raises(ValueError, match="thickness"):
                Feature(bad, None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "bad")

    def test_an_inverted_corner_is_refused(self):
        with pytest.raises(ValueError, match="below lower corner"):
            Feature(1.0, None, (0.0, 0.0, 0.0), (0.0, -1.0, 0.0), "inverted")

    def test_a_feature_carrying_a_count_is_an_element_count(self):
        layer = Feature(1.6, (0.0, 0.0, 1.0), (0.0,) * 3, (1.0, 1.0, 1.6), "layer", across=4)
        assert layer.cells() == elements(1.6, (0.0, 0.0, 1.0), 4)

    def test_and_the_same_length_without_one_is_still_a_separation(self):
        """The count is what selects the criterion, so the two must part on it
        alone - a layer and a gap of one thickness ask for different cells."""
        gap = Feature(1.6, (0.0, 0.0, 1.0), (0.0,) * 3, (1.0, 1.0, 1.6), "gap")
        assert gap.cells() == separation(1.6, (0.0, 0.0, 1.0))

    def test_a_count_with_nothing_to_count_along_is_refused(self):
        with pytest.raises(ValueError, match="needs a normal"):
            Feature(1.6, None, (0.0,) * 3, (1.0,) * 3, "no direction", across=4)

    def test_a_negative_count_is_refused(self):
        with pytest.raises(ValueError, match="across"):
            Feature(1.6, (0.0, 0.0, 1.0), (0.0,) * 3, (1.0,) * 3, "backwards", across=-1)

    def test_a_relaxed_feature_asks_for_the_relaxed_size(self):
        gap = Feature(0.2, (1.0, 0.0, 0.0), (0.0,) * 3, (0.0, 1.0, 1.0), "gap", relaxed_to=1.0)
        assert gap.cells()[0] == 1.0

    def test_a_relaxation_finer_than_the_measurement_changes_nothing(self):
        drawn = dict(thickness=0.2, normal=(1.0, 0.0, 0.0), lower=(0.0,) * 3, upper=(0.0, 1.0, 1.0))
        assert Feature(**drawn, source="gap", relaxed_to=1e-6).cells() == (
            Feature(**drawn, source="gap").cells()
        )

    def test_an_axis_the_feature_does_not_constrain_stays_unconstrained(self):
        """``inf`` is how an axis says it was never asked, and a floor must not
        turn that into a finite demand covering the whole model."""
        gap = Feature(0.2, (1.0, 0.0, 0.0), (0.0,) * 3, (0.0, 1.0, 1.0), "gap", relaxed_to=1.0)
        assert gap.cells()[1] == math.inf and gap.cells()[2] == math.inf

    def test_a_normal_with_no_direction_is_refused_here_rather_than_at_meshing(self):
        """A chord walked between two points that round together leaves one, and
        the allocation would divide by its length an axis at a time - naming
        neither the body nor the face it was measured on."""
        with pytest.raises(ValueError, match="must have a direction"):
            Feature(1.6, (0.0, 0.0, 0.0), (0.0,) * 3, (1.0,) * 3, "collapsed")


class TestDemands:
    def test_an_unconstrained_axis_contributes_nothing(self):
        """Rather than a demand of ``inf`` every caller would have to filter."""
        wall = Feature(0.5, (0.0, 0.0, 1.0), (0.0, 0.0, 2.0), (1.0, 1.0, 2.0), "wall")
        per_axis = demands([wall])
        assert [len(axis) for axis in per_axis] == [0, 0, 1]
        assert per_axis[2][0].size == pytest.approx(0.5, rel=1e-12)

    def test_a_span_is_projected_onto_each_axis_it_constrains(self):
        block = Feature(1.0, None, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), "block")
        per_axis = demands([block])
        for dim, axis in enumerate(per_axis):
            assert (axis[0].lower, axis[0].upper) == (1.0 + dim, 4.0 + dim)

    def test_the_source_survives_onto_every_axis(self):
        """What a cost is attributed to, so the bill can name a face."""
        block = Feature(1.0, None, (0.0,) * 3, (1.0,) * 3, "Reflector face 7")
        assert {d.source for axis in demands([block]) for d in axis} == {"Reflector face 7"}


class TestFeaturesReachTheGrid:
    """A feature is a constraint on the finished grid, and only a constraint.

    The mesher's existing input is an axis-aligned :class:`Region`, which pins
    lines and applies the thirds rule. A feature does neither - a boundary not
    parallel to a grid plane has no coordinate to pin - so what is asserted
    here is that it lowers cells where it sits and changes nothing else.
    """

    def grid(self, features=()):
        return generate_mesh_lines(
            [substrate()],
            DOMAIN,
            params(),
            features=features,
        )

    def test_a_feature_coarser_than_the_cap_leaves_every_line_where_it_was(self):
        """Bit-identical, not merely close.

        The gate on this whole change: the acceptance figures are measured
        against grids built by this function, so anything that moves a line by
        one ulp has to be visible here rather than in a solve an hour later.

        A constraint above the cap can never win a ``min`` against it, so the
        finished grid must be the one built without it. Stated that way rather
        than as ``features=()``, which would compare a call against itself.
        """
        idle = Feature(1000.0, None, (0.0,) * 3, (1.0,) * 3, "far coarser than the cap")
        for dim in range(3):
            assert np.array_equal(self.grid()[dim], self.grid([idle])[dim])

    def test_a_feature_refines_the_axes_its_normal_touches(self):
        wall = Feature(0.2, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "z-facing gap")
        coarse = self.grid()
        fine = self.grid([wall])
        assert cell_at(fine[2], 0.0) < cell_at(coarse[2], 0.0)

    def test_and_leaves_the_axes_it_does_not(self):
        wall = Feature(0.2, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "z-facing gap")
        assert np.array_equal(self.grid()[0], self.grid([wall])[0])

    def test_a_diagonal_normal_spends_itself_across_all_three_axes(self):
        """The case the whole module exists for, end to end.

        One boundary, not parallel to any grid plane, refining all three axes by
        different amounts - and by the amounts the criterion allocated, not
        merely by some amount. The realized cell may be finer than asked, since
        a gap holds a whole number of cells, but never coarser.
        """
        normal = (1.0, 2.0, 4.0)
        skew = Feature(0.3, normal, (0.0,) * 3, (0.0,) * 3, "skew face")
        asked = separation(0.3, normal)
        coarse = self.grid()
        fine = self.grid([skew])
        sizes = [cell_at(fine[dim], 0.0) for dim in range(3)]

        for dim in range(3):
            assert sizes[dim] < cell_at(coarse[dim], 0.0)
            # A point constraint attains its minimum at a point, and the cell
            # containing that point spans a neighbourhood where the field has
            # already risen - so the realized cell sits above the asked size by
            # something less than one grading step, which is the most the field
            # can climb across one cell.
            assert asked[dim] / params().max_ratio[dim] <= sizes[dim]
            assert sizes[dim] <= asked[dim] * params().max_ratio[dim]

        # Different amounts, in the order the allocation put them: the axis the
        # normal leans on hardest gets the finest cells.
        assert sizes[2] < sizes[1] < sizes[0]

    def test_a_counted_layer_lands_the_cells_it_asked_for(self):
        """The count reaching the grid as cells rather than as an intention.

        Counted off the finished grid rather than off the demand, and scored
        against the box the same layer would have been: a region gets
        ``extent/min_lines`` across its own thickness, so whichever way the
        layer was drawn a ray through it has to cross that many cells, and the
        two routes have to land within a line of each other. Neither is held to
        the count exactly - a region here is one demand among several and a
        feature pins nothing at all - so what is asserted is the floor and the
        agreement. Drawn clear of the substrate, whose own count would otherwise
        be answering.
        """
        low, high, count = 3.0, 3.4, 4
        span = ((-8.0, -8.0, low), (8.0, 8.0, high))
        layer = Feature(high - low, (0.0, 0.0, 1.0), *span, "measured layer", across=count)
        boxed = Region(*span, MaterialClass.DIELECTRIC, "drawn layer")

        def across(lines):
            return int(np.sum((lines > low) & (lines < high))) + 1

        measured = self.grid([layer])
        drawn = generate_mesh_lines([substrate(), boxed], DOMAIN, params(min_lines=count))

        assert across(measured[2]) >= count
        assert across(drawn[2]) >= count
        assert abs(across(measured[2]) - across(drawn[2])) <= 1
        assert cell_at(measured[2], 0.5 * (low + high)) <= (high - low) / count
        # Without it the layer is not there at all - nothing else in the model
        # is asking anywhere near it, which is what makes the count the subject.
        assert across(self.grid()[2]) < count
        # The axes the layer's normal does not touch are left where they were,
        # so what it bought is the count and not a finer grid all round.
        assert np.array_equal(self.grid()[0], measured[0])

    def test_an_omnidirectional_feature_refines_all_three(self):
        wire = Feature(0.2, None, (0.0,) * 3, (0.0,) * 3, "wire")
        coarse = self.grid()
        fine = self.grid([wire])
        for dim in range(3):
            assert cell_at(fine[dim], 0.0) < cell_at(coarse[dim], 0.0)

    def test_a_feature_pins_no_line_of_its_own(self):
        """It is a constraint, never an anchor.

        A feature has no coordinate to pin, and sweeping geometry for candidates
        would put the line count back under the control of the model's face
        count - which is the thing a tessellated solid has thousands of. So
        adding one may move where lines fall and must never add a pinned one.
        """
        wire = Feature(0.05, None, (0.31,) * 3, (0.31,) * 3, "wire")
        before = self.grid()
        after = self.grid([wire])
        for dim in range(3):
            assert len(after.fixed[dim]) == len(before.fixed[dim])
            assert [line.position for line in after.fixed[dim]] == [
                line.position for line in before.fixed[dim]
            ]

    def test_a_feature_outside_the_domain_refines_nothing(self):
        """The trap the separable grid sets, and the reason it is not a refusal.

        A feature that misses the domain in one axis would still refine slabs on
        the other two, at coordinates where the geometry it was measured from is
        not. It is dropped, unlike a refinement region, which is a request
        somebody typed and is refused by name instead.
        """
        elsewhere = Feature(0.05, None, (500.0, 0.0, 500.0), (500.0, 0.0, 500.0), "elsewhere")
        for dim in range(3):
            assert np.array_equal(self.grid()[dim], self.grid([elsewhere])[dim])

    def test_a_demand_below_the_floor_reaches_the_field_unclamped(self):
        """The floor is applied once, by the field, and not again on the way in.

        A feature's size is measured off a drawing rather than typed, so one
        below the floor is imperfect input rather than a mistake to refuse -
        which is what separates it from a ``SizingRegion``, an explicit request
        that *is* refused by name. Asserted on the constraint rather than on the
        finished grid: what the floor then does to placement is the field's
        business, and testing it here would be testing that instead.
        """
        tiny = Feature(1e-9, None, (0.0,) * 3, (0.0,) * 3, "sliver")
        built = mesh._constraints([substrate()], 0, params(), -10.0, 10.0, (), [tiny])
        assert [c.size for c in built if c.source == "sliver"] == [
            pytest.approx(1e-9 / math.sqrt(3))
        ]


class TestWhatSetTheField:
    """``_SizingField.winner`` names the constraint that took the ``min``.

    The report's whole job is to say which drawn feature bought the finest plane
    on an axis, and a cell size on its own cannot say: several constraints could
    have produced it. So the argmin is asked for alongside the min, and every
    constraint the mesher builds has to be able to answer.
    """

    def field(self, regions, dim=2):
        p = params()
        pinned = mesh._snap(
            *mesh._fixed_positions(regions, dim, -5.0, 5.0, p, ()),
            float(p.min_cell),
            dim,
        )
        sources = mesh._Sources(pinned, dim)
        return mesh._settle(
            sources,
            mesh._constraints(regions, dim, p, -5.0, 5.0),
            p.ceiling,
            math.log(1.3) * 0.995,
            float(p.min_cell),
        )

    def test_where_no_constraint_reaches_it_names_nothing(self):
        """The cap set the value, and the cap is policy rather than geometry."""
        p = params()
        bare = mesh._SizingField([], p.ceiling, math.log(1.3) * 0.995, float(p.min_cell))
        assert bare.winner(-4.5) == ""

    def test_inside_a_region_it_names_that_region(self):
        assert "Substrate" in self.field([substrate()]).winner(0.8)

    def test_every_constraint_the_mesher_builds_can_be_named(self):
        """Including the ones the mesher publishes to itself.

        Seam agreement republishes each gap's realized edge cell as a point
        constraint, and those usually win near their own pinned line - being the
        finest thing there. Anonymous, they would leave the finest plane on an
        axis attributed to nothing at all.
        """
        field = self.field(stackup())
        assert all(source for source in field._sources)

    def test_a_feature_is_named_where_it_wins(self):
        wire = Feature(0.02, None, (0.0, 0.0, 0.8), (0.0, 0.0, 0.8), "Reflector face 7")
        p = params()
        field = mesh._SizingField(
            mesh._constraints([substrate()], 2, p, -5.0, 5.0, (), [wire]),
            p.ceiling,
            math.log(1.3) * 0.995,
            float(p.min_cell),
        )
        assert field.winner(0.8) == "Reflector face 7"
