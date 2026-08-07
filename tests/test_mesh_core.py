# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Unit tests for the Yee-grid core.

The mesher is the most valuable and least verifiable part of an FDTD adapter:
its output is a pile of numbers that looks equally plausible whether or not it
is right, and a defect shows up as a quietly wrong answer months later. So this
file asserts the properties the grid is *supposed* to have, directly, without a
solver.

Nothing here imports openEMS, CSXCAD or FreeCAD - that is the point of the
core being a pure function.
"""

import math

import numpy as np
import pytest

import Microwave.Solvers.openems.mesh as mesh
from Microwave.Solvers.openems.mesh import (
    MaterialClass,
    MeshError,
    MeshLines,
    MeshParams,
    Region,
    SizingRegion,
    generate_mesh_lines,
)
from tests.mesh_fixtures import (
    DOMAIN,
    assert_graded_within,
    cell_at,
    ground_sheet,
    has_line,
    params,
    pin_at,
    stackup,
    substrate,
)


class TestSmoothness:
    """Adjacent cells may not differ by more than the configured ratio.

    One-sided on purpose: a mesher that ignored the request and always graded
    at 1.05 would satisfy every case here. The *two*-sided pin - that the
    grading is at the ratio, not merely under it - is
    ``TestAgainstClosedForm::test_grading_follows_a_geometric_progression``,
    which brackets it at rtol 0.02. Two ratios rather than five: across the
    whole mutation sweep only 1.1 ever discriminated, and 1.2, 1.5 and 2.0 all
    failed together with 234 other tests or not at all.
    """

    @pytest.mark.parametrize("ratio", [1.1, 2.0])
    def test_ratio_is_respected_everywhere(self, ratio):
        lines = generate_mesh_lines(stackup(), DOMAIN, params(max_ratio=(ratio, ratio, ratio)))
        assert_graded_within(lines, ratio)

    def test_tighter_ratio_costs_more_cells(self):
        """Smoothness is a real trade-off; assert it behaves like one."""
        loose = generate_mesh_lines([substrate()], DOMAIN, params(max_ratio=(2.0,) * 3))
        tight = generate_mesh_lines([substrate()], DOMAIN, params(max_ratio=(1.1,) * 3))
        assert tight.cell_count > loose.cell_count


class TestThirdsRule:
    """Lines go one third inside a conductor edge and two thirds outside."""

    def test_lines_straddle_a_metal_edge(self):
        metal_res = 0.3
        block = Region(
            lower=(-2.0, -2.0, -1.0),
            upper=(2.0, 2.0, 1.0),
            material=MaterialClass.METAL,
            label="Block",
        )
        lines = generate_mesh_lines([block], DOMAIN, params(metal_res=metal_res, min_lines=1))

        for edge, inside, outside in (
            (-2.0, -2.0 + metal_res / 3, -2.0 - 2 * metal_res / 3),
            (2.0, 2.0 - metal_res / 3, 2.0 + 2 * metal_res / 3),
        ):
            assert has_line(lines.x, inside), f"no line inside edge {edge}"
            assert has_line(lines.x, outside), f"no line outside edge {edge}"
            assert not has_line(lines.x, edge), f"line sits on edge {edge}"

    def test_offsets_are_one_third_and_two_thirds(self):
        """Pin the ratio, not just the presence of two lines."""
        metal_res = 0.3
        block = Region(
            lower=(-2.0, -2.0, -1.0),
            upper=(2.0, 2.0, 1.0),
            material=MaterialClass.METAL,
        )
        lines = generate_mesh_lines([block], DOMAIN, params(metal_res=metal_res, min_lines=1))
        below = lines.x[lines.x < -2.0].max()
        above = lines.x[lines.x > -2.0].min()
        assert (-2.0 - below) / (above - (-2.0)) == pytest.approx(2.0, rel=1e-6)

    def test_thin_conductor_falls_back_to_plain_edges(self):
        """Thirds offsets would cross on a conductor thinner than a cell."""
        metal_res = 1.0
        thin = Region(
            lower=(-2.0, -2.0, 0.0),
            upper=(2.0, 2.0, 0.1),
            material=MaterialClass.METAL,
        )
        lines = generate_mesh_lines(
            [thin],
            DOMAIN,
            params(metal_res=metal_res, dielectric_res=metal_res, min_lines=1),
        )
        assert has_line(lines.z, 0.0)
        assert has_line(lines.z, 0.1)
        # ...and the thirds offsets must NOT be there, or the fallback did not
        # happen and the test would pass on a mesher that applied both.
        assert not has_line(lines.z, 0.0 - 2 * metal_res / 3)
        assert not has_line(lines.z, 0.1 + 2 * metal_res / 3)

    def test_dielectric_edges_get_a_line_on_them(self):
        """The thirds rule is for conductors; a dielectric interface is not one."""
        lines = generate_mesh_lines([substrate()], DOMAIN, params())
        assert has_line(lines.z, 0.0)
        assert has_line(lines.z, 1.6)


class TestConductorJoins:
    """Two conductors sharing a face are one piece of metal, not two edges."""

    def test_a_via_landing_on_a_ground_plane_keeps_the_plane_on_a_line(self):
        """The plane is a PEC sheet; it must stay pinned whatever the via does."""
        parts = [
            Region((-5, -5, 0), (5, 5, 0), MaterialClass.METAL, "GND"),
            Region((-0.5, -0.5, 0), (0.5, 0.5, 1), MaterialClass.METAL, "Via"),
        ]
        lines = generate_mesh_lines(parts, ((-8, -8, -2), (8, 8, 4)), params())
        assert has_line(lines.z, 0.0, tol=1e-12)

    def test_no_thirds_offsets_at_a_joined_face(self):
        """There is no edge singularity inside continuous metal, so resolving
        one wastes cells and fights the sheet's need for a line on the face."""
        res = 0.2
        parts = [
            Region((-5, -5, 0), (5, 5, 0), MaterialClass.METAL, "GND"),
            Region((-0.5, -0.5, 0), (0.5, 0.5, 1), MaterialClass.METAL, "Via"),
        ]
        lines = generate_mesh_lines(parts, ((-8, -8, -2), (8, 8, 4)), params(metal_res=res))
        assert not has_line(lines.z, -2 * res / 3)
        assert not has_line(lines.z, res / 3)

    def test_joining_does_not_cost_a_finer_timestep_than_separating(self):
        """Moving a via 1 um onto the plane must not make the mesh worse.

        It used to: the flush case put a line on the conductor edge and the
        repair refined the whole neighbourhood, costing 2.4x the cells and a 6x
        smaller timestep for a 1 um move, with no warning.
        """

        def build(dz):
            return generate_mesh_lines(
                [
                    Region((-5, -5, 0), (5, 5, 0), MaterialClass.METAL, "GND"),
                    Region((-0.5, -0.5, dz), (0.5, 0.5, 1 + dz), MaterialClass.METAL, "Via"),
                ],
                ((-8, -8, -2), (8, 8, 4)),
                params(),
            )

        flush, apart = build(0.0), build(0.05)

        # Both grids must be real before comparing them. `>=` and `<=` are both
        # satisfied by equality, so if the via were dropped entirely the two
        # would collapse to the same trivial mesh and the comparison would pass
        # while proving nothing.
        res = 0.2  # params() default metal_res
        for grid, dz in ((flush, 0.0), (apart, 0.05)):
            face = 1 + dz
            assert not has_line(grid.z, face), "a line sits on a conductor edge"
            assert has_line(grid.z, face - res / 3), "the via's face was not meshed"
            assert has_line(grid.z, face + 2 * res / 3)

        assert flush.smallest_cell() >= apart.smallest_cell()
        assert flush.cell_count <= apart.cell_count


class TestSheets:
    """Zero-thickness objects must land exactly on a line."""

    @pytest.mark.parametrize("z", [0.0, 0.37, 1.6, -2.25])
    def test_sheet_survives_at_awkward_positions(self, z):
        """A sheet off the natural grid is exactly the case that gets lost.

        The ``z = 0.0`` case is the plain one - a ground plane on the natural
        grid - and as a test of its own it would carry a different name and the
        same fixture.
        """
        sheet = Region(
            lower=(-8.0, -8.0, z),
            upper=(8.0, 8.0, z),
            material=MaterialClass.METAL,
            label="Sheet",
        )
        lines = generate_mesh_lines([sheet], DOMAIN, params())
        assert has_line(lines.z, z)

    @pytest.mark.parametrize("gap", [0.05, 0.011, 0.0101])
    def test_two_close_sheets_both_survive(self, gap):
        """Merging must not swallow one of a closely spaced pair.

        Parametrized down to just above the floor: at a comfortable 5x clear of
        it this never exercised the merging logic it is named for.
        """
        sheets = [
            Region((-8, -8, 0.0), (8, 8, 0.0), MaterialClass.METAL, "lower"),
            Region((-8, -8, gap), (8, 8, gap), MaterialClass.METAL, "upper"),
        ]
        lines = generate_mesh_lines(sheets, DOMAIN, params(min_cell=0.01))
        assert has_line(lines.z, 0.0, tol=1e-12)
        assert has_line(lines.z, gap, tol=1e-12)


class TestMinLines:
    """Thin regions must not be spanned by a single cell."""

    @pytest.mark.parametrize("min_lines", [2, 4, 8])
    def test_substrate_gets_the_requested_cell_count(self, min_lines):
        """Exactly the count, not at least it.

        A floor admits a mesher that silently over-refines every thin region in
        every model - ``extent / (min_lines + 1)`` gives 3, 6 and 9 cells here
        for a requested 2, 4 and 8, and cell counts rise ~1.5x per axis with the
        timestep falling to match. The mesher is exact on this fixture, so the
        assertion can be.
        """
        lines = generate_mesh_lines(
            [substrate()], DOMAIN, params(min_lines=min_lines, dielectric_res=5.0)
        )
        through = lines.z[(lines.z >= 0.0 - 1e-9) & (lines.z <= 1.6 + 1e-9)]
        assert len(through) - 1 == min_lines

    def test_asking_for_one_element_asks_for_nothing(self):
        """A count of one is the rule's own off switch, and must behave as one.

        A layer pinned at both faces already has a cell across it, so demanding
        one is demanding what is there. Issuing the constraint anyway makes the
        sizing field claim the layer's whole thickness, which rounds up to two
        cells and grades the neighbourhood down to match.
        """
        layer = Region((-5, -5, 0.0), (5, 5, 0.1), MaterialClass.DIELECTRIC, "Layer")
        domain = ((-8, -8, -2), (8, 8, 4))

        def across(count):
            lines = generate_mesh_lines(
                [layer], domain, params(min_lines=count, dielectric_res=1.0)
            ).z
            return len(lines[(lines >= -1e-12) & (lines <= 0.1 + 1e-12)]) - 1

        assert across(1) == 1
        assert across(2) == 2, "the fixture cannot tell the two counts apart"

    def test_a_clipped_region_is_sized_by_what_was_drawn(self):
        """A long board through a THROUGH face must not act like a thin one.

        The same substrate meshed twice into the same domain: once as the
        region a clip would produce (``drawn`` carries the full 100 mm), once
        as a region genuinely that small. The first must not be refined -
        nothing thin is there - and the second must be, because something is.

        This is the amplifier behind sizing ``min_lines`` from the extent the
        user drew:
        ``min_lines`` from the clipped extent let a small domain demand fine
        cells, which thinned the absorber, which shrank the domain again. On
        the microstrip example at 2.4-2.5 GHz it cost a 14.5x pitch error,
        76,006 cells and 22% of the timestep.
        """
        domain = ((-2.0, -5.0, -1.0), (2.0, 5.0, 3.0))
        settings = dict(metal_res=1.0, dielectric_res=4.0, min_lines=9)
        box = ((-2.0, -5.0, 0.0), (2.0, 5.0, 1.6))

        clipped = Region(*box, MaterialClass.DIELECTRIC, "Board", drawn=(100.0, 10.0, 1.6))
        genuine = Region(*box, MaterialClass.DIELECTRIC, "Chip")

        def across_x(region):
            lines = generate_mesh_lines([region], domain, params(**settings)).x
            return len(lines[(lines >= -2.0 - 1e-9) & (lines <= 2.0 + 1e-9)]) - 1

        assert across_x(genuine) == settings["min_lines"]
        assert across_x(clipped) == 1

    def test_clipping_does_not_change_an_axis_it_did_not_clip(self):
        """z is the same 1.6 mm either way, and must still be resolved."""
        domain = ((-2.0, -5.0, -1.0), (2.0, 5.0, 3.0))
        settings = dict(metal_res=1.0, dielectric_res=4.0, min_lines=9)
        box = ((-2.0, -5.0, 0.0), (2.0, 5.0, 1.6))

        def across_z(drawn):
            region = Region(*box, MaterialClass.DIELECTRIC, "Board", drawn=drawn)
            lines = generate_mesh_lines([region], domain, params(**settings)).z
            return len(lines[(lines >= -1e-9) & (lines <= 1.6 + 1e-9)]) - 1

        assert across_z((100.0, 10.0, 1.6)) == across_z(None) >= settings["min_lines"]

    def test_a_region_that_was_never_clipped_falls_back_to_its_extent(self):
        region = Region((0, 0, 0), (1, 2, 3), MaterialClass.DIELECTRIC, "Solid")
        assert [region.thickness(d) for d in range(3)] == [1.0, 2.0, 3.0]

    def test_a_conductor_is_not_spanned_like_a_substrate(self):
        """One box, meshed twice, differing only in what it is made of.

        Nine cells across a foil resolve nothing: the field is in the
        dielectric, and the skin depth is orders below the copper. They do set
        the smallest cell in the model, which through the Courant limit is paid
        for by every other cell in it.
        """
        settings = dict(metal_res=0.2, dielectric_res=0.5, min_lines=9)
        domain = ((-8, -8, -2), (8, 8, 4))
        foil = ((-5.0, -5.0, 0.0), (5.0, 5.0, 0.035))

        def across_z(material):
            region = Region(*foil, material, "Foil")
            lines = generate_mesh_lines([region], domain, params(**settings)).z
            return len(lines[(lines >= -1e-12) & (lines <= 0.035 + 1e-12)]) - 1

        assert across_z(MaterialClass.METAL) == 1
        assert across_z(MaterialClass.DIELECTRIC) == settings["min_lines"]

    def test_the_conductor_still_pins_both_its_faces(self):
        """Exempt from the count, not from being discretised where it is.

        openEMS builds the box between the lines it finds, so a face falling
        between two of them moves the metal. One cell across a foil is the
        answer; none is a different structure.
        """
        region = Region((-5, -5, 0.0), (5, 5, 0.035), MaterialClass.METAL, "Foil")
        lines = generate_mesh_lines([region], ((-8, -8, -2), (8, 8, 4)), params(min_lines=9))
        assert has_line(lines.z, 0.0)
        assert has_line(lines.z, 0.035)

    def test_a_demand_on_one_region_does_not_move_a_conductor_edge(self):
        """The count refines the region that asked for it, not its neighbours.

        The via's edge is a millimetre above the board, and the thirds rule
        straddling that edge is the one placement in the model chosen
        deliberately. Both counts bite inside the board - asserted, because
        two settings that are each a no-op agree about everything and prove
        nothing.
        """
        settings = dict(metal_res=0.2, dielectric_res=0.5)
        regions = [
            Region((-5, -5, -1), (5, 5, 0), MaterialClass.DIELECTRIC, "Board"),
            Region((-5, -5, 0), (5, 5, 0), MaterialClass.METAL, "GND"),
            Region((-0.5, -0.5, 0), (0.5, 0.5, 1), MaterialClass.METAL, "Via"),
        ]
        domain = ((-8, -8, -2), (8, 8, 4))

        # The board is 1 mm, so the two counts ask 0.25 and 0.125 across it,
        # both finer than the 0.5 cap.
        loose = generate_mesh_lines(regions, domain, params(min_lines=4, **settings))
        tight = generate_mesh_lines(regions, domain, params(min_lines=8, **settings))
        assert not np.array_equal(loose.z, tight.z), "neither count bit"

        # The via's top face is a conductor edge, so the thirds rule straddles
        # it rather than landing on it, and the board's count must not reach it.
        res = settings["metal_res"]
        for grid in (loose, tight):
            assert not has_line(grid.z, 1.0)
            assert has_line(grid.z, 1.0 - res / 3), "the via's face was not meshed"
            assert has_line(grid.z, 1.0 + 2 * res / 3)

    def test_does_not_force_lines_on_thick_regions(self):
        """min_lines is a floor for thin features, not a global multiplier.

        The region must be thick enough that `extent / min_lines` exceeds the
        resolution cap in *both* configurations, or the test compares two
        no-ops and proves nothing - which is exactly what it did before.
        """
        thick = Region((-8, -8, -4), (8, 8, 4), MaterialClass.DIELECTRIC, "Air")
        settings = dict(dielectric_res=0.5, metal_res=0.2)
        assert settings["dielectric_res"] < 8.0 / 4, "test would be vacuous"
        coarse = generate_mesh_lines([thick], DOMAIN, params(min_lines=2, **settings))
        fine = generate_mesh_lines([thick], DOMAIN, params(min_lines=4, **settings))
        assert coarse.shape == fine.shape


class TestAConductorAtTheAbsorber:
    """A conductor reaching the domain wall continues into the absorber.

    There is no edge there to resolve - a feed line running into the PML is
    the ordinary case - so no thirds rule and no metal_res constraint. Both
    `_fixed_positions` and `_constraints` must agree about that, and until this
    existed the `_constraints` half could be deleted outright with all 570 fast
    tests still passing. The slow gate could not see it either: the
    over-refined mesh passed Hammerstad at +0.074%, well inside 1%.
    """

    def _sheet(self, x0, x1):
        return Region((x0, -8.0, 0.0), (x1, 8.0, 0.0), MaterialClass.METAL, "Sheet")

    def test_an_edge_inside_the_domain_is_refined(self):
        """The control. Without it the test below passes on a mesher that
        refines nothing anywhere."""
        lines = generate_mesh_lines([self._sheet(-8.0, 8.0)], DOMAIN, params())
        assert cell_at(lines.x, 8.0 - 1e-6) <= 0.2 * (1 + 1e-9)

    def test_an_edge_at_the_wall_is_not(self):
        flush = generate_mesh_lines([self._sheet(-10.0, 10.0)], DOMAIN, params())
        assert cell_at(flush.x, 10.0 - 1e-6) > 0.9, (
            "the conductor was refined as if it had an edge at the domain wall"
        )

    def test_the_same_conductor_is_still_refined_off_the_wall(self):
        """It reaches the wall in x and not in y, so y must be unaffected."""
        flush = generate_mesh_lines([self._sheet(-10.0, 10.0)], DOMAIN, params())
        assert cell_at(flush.y, 8.0 - 1e-6) <= 0.2 * (1 + 1e-9)


class TestPerMaterialWavelength:
    """A wave slows inside a dielectric, so only the dielectric needs fine cells.

    One lambda taken from the slowest material in the model is the safe answer
    and an expensive one: it meshes the air around a patch on alumina 3.1x
    finer than anything there requires, which is roughly 30x the cells.
    """

    #: Vacuum bulk size; the substrate below asks for half of it, as eps_r = 4
    #: would.
    CAP = 1.0
    IN_DIELECTRIC = 0.5

    def _substrate(self):
        return Region(
            lower=(-8.0, -8.0, 0.0),
            upper=(8.0, 8.0, 1.6),
            material=MaterialClass.DIELECTRIC,
            label="Substrate",
            size=self.IN_DIELECTRIC,
        )

    def _params(self, **overrides):
        settings = dict(
            metal_res=0.2,
            dielectric_res=self.IN_DIELECTRIC,
            min_lines=1,
            cap=self.CAP,
        )
        settings.update(overrides)
        return params(**settings)

    def test_the_dielectric_is_finer_than_the_air_around_it(self):
        lines = generate_mesh_lines([self._substrate()], DOMAIN, self._params())
        inside = cell_at(lines.z, 0.8)
        outside = cell_at(lines.z, -4.0)
        assert inside <= self.IN_DIELECTRIC * (1 + 1e-9)
        assert outside > self.IN_DIELECTRIC * 1.5

    def test_air_is_capped_at_the_vacuum_size(self):
        lines = generate_mesh_lines([self._substrate()], DOMAIN, self._params())
        assert cell_at(lines.z, -4.0) <= self.CAP * (1 + 1e-9)

    def test_no_cap_means_the_grid_it_always_produced(self):
        """A caller who gives only ``dielectric_res`` gets exactly the grid they
        got before regions carried sizes: the ceiling is the bulk size, so a
        region asking for the bulk size asks for nothing."""
        settings = params(metal_res=0.2, dielectric_res=self.IN_DIELECTRIC, min_lines=1)
        assert settings.ceiling == settings.dielectric_res

        sized = generate_mesh_lines([self._substrate()], DOMAIN, settings)
        plain = generate_mesh_lines(
            [
                Region(
                    lower=self._substrate().lower,
                    upper=self._substrate().upper,
                    material=MaterialClass.DIELECTRIC,
                    label="Substrate",
                )
            ],
            DOMAIN,
            settings,
        )
        for dim in range(3):
            assert list(sized[dim]) == list(plain[dim])

    def test_two_dielectrics_get_two_different_sizes(self):
        """The whole point, and the only arrangement that can show it.

        With one dielectric its own size *is* the global one - the global one
        is taken from the slowest material, and with one material that is it.
        So a single-substrate model cannot tell "each region asks for its own
        size" from "every region asks for the global size", and a mutation that
        deleted the former survived the suite until this existed.
        """
        slow = Region(
            (-8.0, -8.0, 0.0),
            (8.0, 8.0, 1.0),
            MaterialClass.DIELECTRIC,
            "Alumina",
            size=0.25,
        )
        fast = Region(
            (-8.0, -8.0, 2.0),
            (8.0, 8.0, 4.5),
            MaterialClass.DIELECTRIC,
            "Foam",
            size=0.75,
        )
        settings = params(metal_res=0.2, dielectric_res=0.25, min_lines=1, cap=self.CAP)
        lines = generate_mesh_lines([slow, fast], DOMAIN, settings)

        assert cell_at(lines.z, 0.5) <= 0.25 * (1 + 1e-9)
        in_foam = cell_at(lines.z, 3.2)
        assert in_foam <= 0.75 * (1 + 1e-9)
        assert in_foam > 0.25 * 1.5, "the coarser dielectric was meshed at the finer one's size"

    def test_a_region_without_a_size_falls_back_to_the_global_one(self):
        plain = Region(
            lower=(-8.0, -8.0, 0.0),
            upper=(8.0, 8.0, 1.6),
            material=MaterialClass.DIELECTRIC,
            label="Substrate",
        )
        lines = generate_mesh_lines([plain], DOMAIN, self._params())
        assert cell_at(lines.z, 0.8) <= self.IN_DIELECTRIC * (1 + 1e-9)

    def test_a_conductor_is_not_coarsened_by_the_cap(self):
        """Its edges resolve a field singularity, not a wavelength, so they are
        sized by ``metal_res`` and the cap has nothing to say about them."""
        stack = [
            self._substrate(),
            Region((-3.0, -1.5, 1.6), (3.0, 1.5, 1.6), MaterialClass.METAL, "Trace"),
        ]
        lines = generate_mesh_lines(stack, DOMAIN, self._params())
        assert cell_at(lines.y, 1.5 - 0.05) <= 0.2 * (1 + 1e-9)

    def test_a_cap_finer_than_the_bulk_is_refused(self):
        """A ceiling below the size it is a ceiling for is a caller mistake."""
        with pytest.raises(MeshError, match="finer than dielectric_res"):
            params(metal_res=0.2, dielectric_res=1.0, cap=0.5)

    def test_a_region_size_must_be_positive(self):
        with pytest.raises(MeshError, match="size must be > 0"):
            Region((0, 0, 0), (1, 1, 1), MaterialClass.DIELECTRIC, "Bad", size=0.0)


class TestLocalRefinement:
    """A ``SizingRegion`` refines, and does nothing else.

    "Nothing else" is the load-bearing half. A refinement box is a statement
    about resolution, not geometry, so the one thing that must never happen is
    for the grid to snap to the box the user drew - someone refining a coupled
    gap would get lines at the box faces instead of where the gap is, and the
    mesh would silently be a picture of the annotation rather than of the model.
    """

    #: Room around the substrate so refinement has somewhere to grade out into.
    BOX = SizingRegion(lower=(-2.0, -2.0, 0.0), upper=(2.0, 2.0, 1.6), size=0.1, label="Gap")

    def test_cells_inside_the_box_are_no_larger_than_asked(self):
        lines = generate_mesh_lines([substrate()], DOMAIN, params(), sizing=[self.BOX])
        for dim, (low, high) in enumerate(zip(self.BOX.lower, self.BOX.upper)):
            inside = lines[dim][(lines[dim] >= low) & (lines[dim] <= high)]
            assert len(inside) >= 2, f"no cells inside the box on axis {dim}"
            assert np.max(np.diff(inside)) <= self.BOX.size * (1 + 1e-9)

    def test_without_the_box_those_cells_are_coarser(self):
        """Or the test above proves only that the global size was already fine."""
        lines = generate_mesh_lines([substrate()], DOMAIN, params())
        low, high = self.BOX.lower[0], self.BOX.upper[0]
        inside = lines.x[(lines.x >= low) & (lines.x <= high)]
        assert np.max(np.diff(inside)) > self.BOX.size

    def test_it_refines_only_near_itself(self):
        """A local demand that reached the whole domain would not be local.

        Not asserted as "the far cells are bit-identical": they are not, and
        should not be. Seam settling publishes each gap's realized edge size
        back into the field, so a change anywhere nudges its neighbours by
        design. What must hold is that the grid still reaches full size away
        from the box, and that refining a 4 mm box costs a fraction of
        refining the 20 mm domain.
        """
        settings = params()
        plain = generate_mesh_lines([substrate()], DOMAIN, settings)
        local = generate_mesh_lines([substrate()], DOMAIN, settings, sizing=[self.BOX])
        everywhere = generate_mesh_lines(
            [substrate()],
            DOMAIN,
            params(metal_res=self.BOX.size, dielectric_res=self.BOX.size),
        )

        # Still coarse where nothing asked for detail.
        assert np.max(np.diff(local.x)) > 0.9 * settings.dielectric_res
        # But it did do something, and far less than the global version.
        assert local.cell_count > plain.cell_count
        assert local.cell_count < everywhere.cell_count / 10

    def test_it_pins_no_line_at_its_own_faces(self):
        """The distinction between a sizing region and a region.

        Refined cells land wherever the field puts them. If the box faces were
        pinned they would appear in the anchor list, which is what a mesh report
        shows the user as "why the grid is like this".
        """
        plain = generate_mesh_lines([substrate()], DOMAIN, params())
        refined = generate_mesh_lines([substrate()], DOMAIN, params(), sizing=[self.BOX])
        for dim in range(3):
            assert [pin.position for pin in refined.fixed[dim]] == [
                pin.position for pin in plain.fixed[dim]
            ]
            assert [pin.source for pin in refined.fixed[dim]] == [
                pin.source for pin in plain.fixed[dim]
            ]

    def test_it_still_grades_smoothly(self):
        """The steepest gradient in the model, so this is where grading fails."""
        ratio = 1.3
        lines = generate_mesh_lines(
            [substrate()],
            DOMAIN,
            params(max_ratio=(ratio,) * 3),
            sizing=[self.BOX],
        )
        assert_graded_within(lines, ratio)

    def test_a_region_coarser_than_the_global_size_is_refused_by_name(self):
        coarse = SizingRegion((-2, -2, 0), (2, 2, 1.6), size=2.0, label="TooCoarse")
        with pytest.raises(MeshError) as excinfo:
            generate_mesh_lines([substrate()], DOMAIN, params(), sizing=[coarse])
        assert "TooCoarse" in str(excinfo.value)
        assert "refine only" in str(excinfo.value)

    def test_a_region_below_the_cell_floor_is_refused_by_name(self):
        """Silently clamping would hand back a grid nobody asked for."""
        settings = params()
        tiny = SizingRegion(
            (-2, -2, 0),
            (2, 2, 1.6),
            size=float(settings.min_cell) / 2,
            label="TooFine",
        )
        with pytest.raises(MeshError) as excinfo:
            generate_mesh_lines([substrate()], DOMAIN, settings, sizing=[tiny])
        assert "TooFine" in str(excinfo.value)
        assert "floor" in str(excinfo.value)

    def test_a_box_that_misses_the_domain_in_one_axis_is_refused(self):
        """The trap a separable grid sets.

        Missing in x alone is enough: the y and z spans still overlap, so the
        box would quietly refine two slabs through the middle of the model
        while sitting nowhere near it.
        """
        adrift = SizingRegion((50.0, -2.0, 0.0), (54.0, 2.0, 1.6), 0.1, label="Adrift")
        with pytest.raises(MeshError) as excinfo:
            generate_mesh_lines([substrate()], DOMAIN, params(), sizing=[adrift])
        assert "Adrift" in str(excinfo.value)
        assert "in x" in str(excinfo.value)

    def test_a_box_overhanging_the_wall_is_allowed(self):
        """Refining up to a THROUGH boundary is ordinary; the spans overlap."""
        over = SizingRegion((-12.0, -2.0, 0.0), (-6.0, 2.0, 1.6), 0.1, label="Edge")
        lines = generate_mesh_lines([substrate()], DOMAIN, params(), sizing=[over])
        assert cell_at(lines.x, -8.0) <= 0.1 * (1 + 1e-9)

    def test_its_own_min_lines_spans_a_box_the_size_alone_would_not(self):
        """A box thinner than ``size`` still gets the count it asks for.

        Measured on the cell straddling the box centre rather than by counting
        lines between its faces: the faces are deliberately not pinned, so the
        cells at each end hang over and a line count is off by up to two.
        """
        thin = dict(lower=(-2.0, -2.0, 0.4), upper=(2.0, 2.0, 0.8))
        settings = params(min_lines=1)
        loose = generate_mesh_lines(
            [substrate()], DOMAIN, settings, sizing=[SizingRegion(size=0.5, **thin)]
        )
        tight = generate_mesh_lines(
            [substrate()],
            DOMAIN,
            settings,
            sizing=[SizingRegion(size=0.5, min_lines=8, **thin)],
        )
        assert cell_at(loose.z, 0.6) > 0.4, "test would be vacuous"
        # Two-sided: an upper bound alone accepts a region cut into nine cells
        # where eight were asked for, and every over-refinement above that.
        assert 0.4 / 9 <= cell_at(tight.z, 0.6) <= 0.4 / 8 * (1 + 1e-9)

    def test_zero_min_lines_inherits_the_global_count(self):
        thin = dict(lower=(-2.0, -2.0, 0.4), upper=(2.0, 2.0, 0.8), size=0.5)
        inherited = generate_mesh_lines(
            [substrate()], DOMAIN, params(min_lines=8), sizing=[SizingRegion(**thin)]
        )
        explicit = generate_mesh_lines(
            [substrate()],
            DOMAIN,
            params(min_lines=8),
            sizing=[SizingRegion(min_lines=8, **thin)],
        )
        assert list(inherited.z) == list(explicit.z)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (dict(size=0.0), "must be > 0"),
            (dict(size=-1.0), "must be > 0"),
            (dict(size=0.1, min_lines=-1), "min_lines must be >= 0"),
            (dict(size=0.1, upper=(-3.0, 2.0, 1.6)), "below lower corner"),
        ],
    )
    def test_a_malformed_region_is_refused_on_construction(self, kwargs, message):
        settings = dict(lower=(-2.0, -2.0, 0.0), upper=(2.0, 2.0, 1.6), label="Bad")
        settings.update(kwargs)
        with pytest.raises(MeshError, match=message):
            SizingRegion(**settings)


class TestAbsorber:
    """PML layers must be uniform, or they reflect."""

    @pytest.mark.parametrize("pml_cells", [4, 8, 12])
    def test_absorber_cells_are_uniform(self, pml_cells):
        lines = generate_mesh_lines([substrate()], DOMAIN, params(pml_cells=pml_cells))
        for dim in range(3):
            spacings = np.diff(lines[dim])
            assert np.allclose(spacings[:pml_cells], spacings[0])
            assert np.allclose(spacings[-pml_cells:], spacings[-1])

    def test_absorber_lies_outside_the_requested_domain(self):
        """The caller sizes the physical box; the absorber is added beyond it."""
        lines = generate_mesh_lines([substrate()], DOMAIN, params(pml_cells=8))
        assert lines.x[0] < DOMAIN[0][0]
        assert lines.x[-1] > DOMAIN[1][0]

    def test_domain_boundary_is_still_a_grid_line(self):
        lines = generate_mesh_lines([substrate()], DOMAIN, params())
        assert has_line(lines.x, DOMAIN[0][0])
        assert has_line(lines.x, DOMAIN[1][0])

    def test_absorber_can_be_disabled(self):
        lines = generate_mesh_lines([substrate()], DOMAIN, params(pml_cells=0))
        assert lines.x[0] == pytest.approx(DOMAIN[0][0])
        assert lines.x[-1] == pytest.approx(DOMAIN[1][0])


class TestResolution:
    """Cell sizes follow the configured resolutions."""

    def test_no_cell_exceeds_the_dielectric_cap(self):
        settings = params(dielectric_res=1.0, pml_cells=0)
        lines = generate_mesh_lines([substrate()], DOMAIN, settings)
        for dim in range(3):
            assert np.max(np.diff(lines[dim])) <= settings.dielectric_res * (1 + 1e-9)

    def test_metal_edges_are_finer_than_open_space(self):
        block = Region((-2, -2, -1), (2, 2, 1), MaterialClass.METAL, "Block")
        lines = generate_mesh_lines(
            [block], DOMAIN, params(metal_res=0.1, dielectric_res=2.0, min_lines=1)
        )
        near_edge = np.min(np.diff(lines.x[np.abs(lines.x + 2.0) < 0.5]))
        assert near_edge == pytest.approx(0.1, rel=0.25), (
            f"cells at a metal edge should be about metal_res, got {near_edge}"
        )

    def test_finer_resolution_gives_more_cells(self):
        coarse = generate_mesh_lines([substrate()], DOMAIN, params(dielectric_res=2.0))
        fine = generate_mesh_lines([substrate()], DOMAIN, params(dielectric_res=0.5))
        assert fine.cell_count > coarse.cell_count


class TestValidationRaises:
    """Requirement 4 is "raise, not warn" - so test the raising, not the grid.

    Every other class here re-derives a property from the output, which stays
    green whether validation exists or not. Deleting the whole `_validate` layer
    passes the whole file. These call it directly with grids it must reject.
    """

    def _reject(self, lines, **overrides):
        settings = params(pml_cells=0, **overrides)
        mesh._validate(lines, [lines[0], lines[-1]], 0, settings)

    def test_smoothness_violation_raises(self):
        lines = [0.0, 1.0, 3.0]  # a 2:1 jump against a 1.3 limit
        with pytest.raises(MeshError, match="violates smoothness"):
            self._reject(lines)

    def test_cell_below_the_floor_raises(self):
        settings = params(pml_cells=0)
        lines = [0.0, float(settings.min_cell) / 10.0, 1.0]
        with pytest.raises(MeshError, match="below the floor"):
            mesh._validate(lines, [lines[0], lines[-1]], 0, settings)

    def test_non_increasing_lines_raise(self):
        with pytest.raises(MeshError, match="non-increasing"):
            self._reject([0.0, 1.0, 1.0, 2.0])

    def test_non_finite_lines_raise(self):
        with pytest.raises(MeshError, match="non-finite"):
            self._reject([0.0, float("nan"), 1.0])

    def test_lost_pinned_line_raises(self):
        """The check that a sheet survived must itself be checked."""
        settings = params(pml_cells=0)
        with pytest.raises(MeshError, match="lost a required grid line"):
            mesh._validate([0.0, 1.0, 2.0], [0.0, 0.5, 2.0], 0, settings)

    def test_non_uniform_absorber_raises(self):
        """Smooth everywhere, but the outer cells are not equal - a PML that
        reflects. The grid has to stay inside the ratio limit or it trips the
        smoothness check first and never reaches the absorber one."""
        lines = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.2]
        with pytest.raises(MeshError, match="absorber is not uniformly spaced"):
            mesh._validate(lines, [lines[0], lines[-1]], 0, params(pml_cells=2))

    def test_a_good_grid_is_accepted(self):
        """Guard against a validator that rejects everything."""
        lines = [float(i) for i in range(12)]
        mesh._validate(lines, [0.0, 11.0], 0, params(pml_cells=2))


class TestCellFloor:
    """Requirement 7: nothing may set the timestep by accident."""

    def test_no_cell_falls_below_the_floor(self):
        settings = params(min_cell=0.05)
        lines = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, settings)
        assert lines.smallest_cell() >= 0.05 * (1 - 1e-9)

    def test_anchors_closer_than_the_floor_are_refused_by_name(self):
        """Two conducting sheets that cannot both be resolved must say so."""
        sheets = [
            Region((-8, -8, 0.0), (8, 8, 0.0), MaterialClass.METAL, "GroundA"),
            Region((-8, -8, 1e-5), (8, 8, 1e-5), MaterialClass.METAL, "GroundB"),
        ]
        with pytest.raises(MeshError, match="GroundB"):
            generate_mesh_lines(sheets, DOMAIN, params(min_cell=0.01))

    def test_a_conducting_sheet_is_never_moved(self):
        """Relocating an anchor is undetectable downstream, so it must not happen.

        Validation compares against the positions the mesher kept, not the ones
        the caller asked for - so a sheet quietly averaged somewhere else would
        pass every check while being in the wrong place.
        """
        z = 0.371
        sheet = Region((-8, -8, z), (8, 8, z), MaterialClass.METAL, "Sheet")
        lines = generate_mesh_lines([sheet], DOMAIN, params(min_cell=0.05))
        assert has_line(lines.z, z, tol=1e-12)

    def test_the_domain_is_not_shrunk(self):
        """The caller's box is an anchor too, even with a sheet close to it."""
        sheet = Region((-8, -8, 4.9), (8, 8, 4.9), MaterialClass.METAL, "Sheet")
        lines = generate_mesh_lines([sheet], DOMAIN, params(min_cell=0.05, pml_cells=0))
        assert lines.z[-1] == pytest.approx(DOMAIN[1][2], abs=1e-12)
        assert has_line(lines.z, 4.9, tol=1e-12)

    def test_a_sheet_crowding_the_domain_wall_is_refused(self):
        """Rather than quietly moving the wall or the sheet to make room."""
        sheet = Region((-8, -8, 4.999), (8, 8, 4.999), MaterialClass.METAL, "Sheet")
        with pytest.raises(MeshError, match="domain upper bound"):
            generate_mesh_lines([sheet], DOMAIN, params(min_cell=0.05))

    def test_a_thin_copper_layer_still_meshes(self):
        """35 um copper on a 0.2 mm grid is the most ordinary feature there is.

        Both faces are anchors, so it is the floor that decides whether they
        may stand that close. A floor derived from the resolution rather than
        kept far below it would refuse every foil, mask and thin-film layer
        there is.
        """
        trace = Region((-1, -0.15, 1.6), (1, 0.15, 1.635), MaterialClass.METAL, "Trace")
        lines = generate_mesh_lines(
            [trace], ((-5, -5, 0), (5, 5, 5)), params(metal_res=0.2, dielectric_res=1.0)
        )
        assert has_line(lines.z, 1.6)
        assert has_line(lines.z, 1.635)


class TestSymmetry:
    """Requirement 8: a symmetric structure must give a symmetric grid.

    An asymmetric grid under a symmetric model excites modes that are not in
    the model, and the asymmetry that does it is far below what anyone notices
    by eye.
    """

    def test_symmetric_geometry_gives_an_exactly_symmetric_grid(self):
        block = Region((-2, -2, -1), (2, 2, 1), MaterialClass.METAL, "Block")
        lines = generate_mesh_lines([block], DOMAIN, params())
        for dim in range(3):
            axis = lines[dim]
            centre = (axis[0] + axis[-1]) / 2.0
            # Exactly, not nearly. The unfolded grid is already symmetric to
            # about 3e-14, so a loose tolerance here cannot tell whether the
            # symmetry step ran at all.
            assert np.array_equal(axis, 2 * centre - axis[::-1])

    def test_asymmetric_geometry_is_not_folded(self):
        """The bug this guards: folding a grid whose geometry is one-sided.

        The two ends of a domain always mirror each other, so testing pinned
        positions alone declares a one-sided structure symmetric and folds it,
        destroying the grading. Only a closed-form check catches that, because
        the folded grid still looks perfectly plausible.
        """
        ratio, fine, cap = 1.3, 0.05, 2.0
        sheet = Region((0.0, -5, -5), (0.0, 5, 5), MaterialClass.METAL, "Sheet")
        settings = MeshParams(
            metal_res=fine,
            dielectric_res=cap,
            max_ratio=(ratio,) * 3,
            min_lines=1,
            pml_cells=0,
        )
        lines = generate_mesh_lines([sheet], ((0.0, -5, -5), (40.0, 5, 5)), settings)
        spacings = np.diff(lines.x)
        assert spacings[0] < fine * 2, "grading at the feature was lost"
        assert spacings[-1] > spacings[0] * 10, "grid was folded; it should grow away"

    def test_symmetry_survives_an_off_centre_domain(self, monkeypatch):
        """Symmetry about a non-zero centre is where floating point gives up.

        Off centre ``(u + v) / 2`` rounds, so exactness is not available and the
        claim can only be comparative: the fold must beat the placement it
        corrects. Both grids are built here, so the bar is re-measured rather
        than remembered - a threshold in millimetres is a threshold on
        whichever grid was current when it was chosen, and it fails the next
        time anything legitimately moves a line.
        """
        block = Region((9.0, -2, -1), (11.0, 2, 1), MaterialClass.METAL, "Block")

        def residual():
            lines = generate_mesh_lines([block], ((0.0, -10, -5), (20.0, 10, 5)), params()).x
            centre = (lines[0] + lines[-1]) / 2.0
            mirrored = 2 * centre - lines[::-1]
            return np.max(np.abs(lines - mirrored)), np.spacing(np.max(np.abs(lines)))

        folded, ulp = residual()
        monkeypatch.setattr(mesh, "_symmetrize", lambda interior, positions: interior)
        unfolded, _ = residual()

        assert folded < unfolded, "the fold left the grid no more symmetric than it found it"
        # And what is left is the arithmetic rather than a grid that drifted: a
        # fold about a coordinate the mesh cannot represent exactly cannot do
        # better than the last few bits of that coordinate. The budget has to
        # be tighter than what the unfolded control already reaches, or the
        # line above subsumes it and this one asserts nothing.
        assert 8 * ulp < unfolded, "the budget is looser than an unfolded grid; it cannot bind"
        assert folded < 8 * ulp


class TestAgainstClosedForm:
    """Cases where the exact grid is derivable, so the code is checked against
    mathematics rather than against anyone's implementation.

    These are the strongest tests in the file. Everything else asserts that a
    property holds; these assert the specific numbers the algorithm must produce.
    """

    def test_unconstrained_axis_is_exactly_uniform(self):
        """With only a cap, the answer is L/ceil(L/cap), repeated.

        No geometry, no grading, nothing to trade off - if this is not exact,
        the arclength integration itself is wrong.
        """
        domain = ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
        settings = params(dielectric_res=1.0, metal_res=1.0, pml_cells=0, min_lines=1)
        lines = generate_mesh_lines([], domain, settings)

        expected_cells = math.ceil(10.0 / 1.0)
        for dim in range(3):
            spacings = np.diff(lines[dim])
            assert len(spacings) == expected_cells
            assert np.allclose(spacings, 10.0 / expected_cells, rtol=1e-9)

    def test_grading_follows_a_geometric_progression(self):
        """Away from a fine feature, cell k must be h0 * ratio**k, and there
        must be ln(cap/fine)/ln(ratio) of them.

        This is the closed-form solution of the sizing field: cells grow by a
        constant factor until they hit the cap. It pins the grading law itself,
        including that the field's Lipschitz constant is ln(ratio) and not
        ratio - 1 - the latter produces exp(ratio-1) growth, which is 3.8%
        too fast at ratio 1.3 and would fail this test.

        The ratio and the count are one property measured two ways, on one
        grid: fix the ratio to 2% and the count follows to well inside the +-2
        below. They were two tests with a byte-identical ten-line fixture and no
        mutation separated them.
        """
        ratio, fine, cap = 1.3, 0.05, 2.0
        sheet = Region(
            lower=(0.0, -5.0, -5.0),
            upper=(0.0, 5.0, 5.0),
            material=MaterialClass.METAL,
            label="Sheet",
        )
        settings = MeshParams(
            metal_res=fine,
            dielectric_res=cap,
            max_ratio=(ratio,) * 3,
            min_lines=1,
            pml_cells=0,
        )
        lines = generate_mesh_lines([sheet], ((0.0, -5.0, -5.0), (40.0, 5.0, 5.0)), settings)

        spacings = np.diff(lines.x)
        growing = spacings[spacings < cap * 0.9]
        assert len(growing) > 8, "not enough graded cells to test the progression"

        observed = growing[1:] / growing[:-1]
        assert np.allclose(observed, ratio, rtol=0.02), (
            f"cell growth {observed} is not the geometric progression {ratio}"
        )

        predicted = math.log(cap / fine) / math.log(ratio)
        assert abs(len(growing) - predicted) <= 2, (
            f"{len(growing)} graded cells against an analytic {predicted:.1f}"
        )

    def test_growth_reaches_the_cap_and_stops(self):
        """The progression is truncated by the cap, not continued past it."""
        ratio, fine, cap = 1.4, 0.05, 1.0
        sheet = Region((0.0, -5, -5), (0.0, 5, 5), MaterialClass.METAL, "Sheet")
        settings = MeshParams(
            metal_res=fine,
            dielectric_res=cap,
            max_ratio=(ratio,) * 3,
            min_lines=1,
            pml_cells=0,
        )
        lines = generate_mesh_lines([sheet], ((0.0, -5, -5), (40.0, 5, 5)), settings)
        assert np.max(np.diff(lines.x)) <= cap * (1 + 1e-9)


class TestGridWellFormedness:
    def test_lines_are_strictly_increasing(self):
        lines = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, params())
        for dim in range(3):
            assert np.all(np.diff(lines[dim]) > 0)

    def test_all_lines_are_finite(self):
        lines = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, params())
        for dim in range(3):
            assert np.all(np.isfinite(lines[dim]))

    def test_empty_region_list_still_meshes_the_domain(self):
        lines = generate_mesh_lines([], DOMAIN, params())
        assert lines.cell_count > 0

    def test_cell_count_matches_shape(self):
        """Lines, not intervals - what openEMS calls cells and what it divides
        the iteration time by. ``prod(n - 1)`` is the defensible geometric
        answer and the wrong one for this property; it read 3.4% low on the
        acceptance grid and was printed three lines above openEMS' own figure
        for the same grid. See ``MeshLines.cell_count``."""
        lines = generate_mesh_lines([substrate()], DOMAIN, params())
        nx, ny, nz = lines.shape
        assert lines.cell_count == nx * ny * nz
        assert lines.cell_count != (nx - 1) * (ny - 1) * (nz - 1)

    def test_result_is_deterministic(self):
        """Same input, same grid - otherwise nothing downstream is reproducible."""
        first = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, params())
        second = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, params())
        for dim in range(3):
            assert np.array_equal(first[dim], second[dim])


class TestRefusesLoudly:
    """An unmeshable request is an error naming the object, never a silent fudge."""

    def test_region_outside_domain_names_the_object(self):
        stray = Region((-50, -1, -1), (50, 1, 1), MaterialClass.METAL, "Antenna")
        with pytest.raises(MeshError, match="Antenna"):
            generate_mesh_lines([stray], DOMAIN, params())

    def test_region_outside_domain_names_the_axis(self):
        stray = Region((-50, -1, -1), (50, 1, 1), MaterialClass.METAL, "Antenna")
        with pytest.raises(MeshError, match=r"\bx\b"):
            generate_mesh_lines([stray], DOMAIN, params())

    def test_inverted_region_is_rejected(self):
        with pytest.raises(MeshError, match="upper corner is below"):
            Region((0, 0, 0), (-1, 1, 1), MaterialClass.METAL, "Backwards")

    def test_degenerate_domain_is_rejected(self):
        with pytest.raises(MeshError, match="no extent"):
            generate_mesh_lines([], ((0, 0, 0), (0, 10, 10)), params())

    def test_metal_res_coarser_than_dielectric_res_is_rejected(self):
        with pytest.raises(MeshError, match="metal edges need the finer grid"):
            MeshParams(metal_res=2.0, dielectric_res=1.0)

    @pytest.mark.parametrize("ratio", [0.9, 1.0])
    def test_ratio_of_one_or_less_is_rejected(self, ratio):
        with pytest.raises(MeshError, match="max_ratio must be > 1"):
            MeshParams(metal_res=0.1, dielectric_res=1.0, max_ratio=(ratio,) * 3)

    def test_negative_resolution_is_rejected(self):
        with pytest.raises(MeshError, match="resolutions must be > 0"):
            MeshParams(metal_res=-0.1, dielectric_res=1.0)


class TestTheWholeGridHasASize:
    """The per-axis limit bounds one axis; nothing bounded the product.

    Every route measured into this state came from one mistyped property on a
    model that meshes fine otherwise, and each was silent - the mesh came back
    promptly and the solve did not come back at all. The absorber is appended
    after the per-axis limit is applied, so it is not even the whole story on
    one axis.
    """

    def test_an_absurd_absorber_is_refused(self):
        """``PMLCells`` is applied per axis after everything else, so it
        multiplies the finished grid by itself three times."""
        with pytest.raises(MeshError, match="it is the product that ran away"):
            generate_mesh_lines(stackup(), DOMAIN, params(pml_cells=200_001))

    def test_a_domain_a_typo_made_enormous_is_refused(self):
        """A port offset with an extra digit drags the domain out with it. Each
        axis stays well inside the per-axis limit and the product does not."""
        huge = ((-400.0, -400.0, -400.0), (400.0, 400.0, 400.0))
        with pytest.raises(MeshError, match="it is the product that ran away"):
            generate_mesh_lines(stackup(), huge, params())

    def test_the_message_names_the_shape_and_the_memory(self):
        with pytest.raises(MeshError) as raised:
            generate_mesh_lines(stackup(), DOMAIN, params(pml_cells=200_001))
        message = str(raised.value)
        assert "lines)" in message and "GiB" in message
        assert "pml_cells" in message

    def test_the_warning_band_arrives_before_the_refusal(self):
        """Derived rather than declared, so the two cannot pass each other."""
        assert 0 < mesh.LARGE_GRID_BYTES < mesh.MAX_GRID_BYTES


class TestProvenance:
    """Every pinned line knows what pinned it.

    The mesher has always named the responsible object in its *error* messages;
    these assert the same names survive into a successful result, because that
    is what a mesh report and the preview's anchor view are built from. The
    property under test is not decoration: recovering it afterwards from
    positions alone is impossible, since a substrate's top face and a line two
    thirds of a cell outside a trace edge are the same float.
    """

    def test_every_pinned_line_is_actually_in_the_grid(self):
        lines = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, params())
        for dim in range(3):
            assert lines.fixed[dim], f"axis {dim} reported no pinned lines at all"
            for pin in lines.fixed[dim]:
                assert has_line(lines[dim], pin.position), (
                    f"{pin.source} at {pin.position} claims to be pinned but is not a grid line"
                )

    def test_pins_are_ordered_by_position(self):
        lines = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, params())
        for dim in range(3):
            positions = [pin.position for pin in lines.fixed[dim]]
            assert positions == sorted(positions)

    def test_a_conducting_sheet_names_itself_and_is_an_anchor(self):
        lines = generate_mesh_lines([ground_sheet()], DOMAIN, params())
        pin = pin_at(lines.fixed[2], 0.0)
        assert "GroundPlane" in pin.source
        assert pin.required

    def test_a_dielectric_face_names_itself_and_is_only_a_preference(self):
        """The required flag is the difference between 'must' and 'would like'."""
        lines = generate_mesh_lines([substrate()], DOMAIN, params())
        pin = pin_at(lines.fixed[2], 1.6)
        assert "Substrate" in pin.source
        assert not pin.required

    def test_the_domain_walls_are_anchors(self):
        lines = generate_mesh_lines([], DOMAIN, params())
        for dim in range(3):
            for position in (DOMAIN[0][dim], DOMAIN[1][dim]):
                pin = pin_at(lines.fixed[dim], position)
                assert pin.required
                assert "domain" in pin.source

    def test_thirds_lines_say_which_side_of_the_edge_they_sit(self):
        """The two lines around a conductor edge are not interchangeable."""
        metal_res = 0.3
        block = Region(
            lower=(-2.0, -2.0, -1.0),
            upper=(2.0, 2.0, 1.0),
            material=MaterialClass.METAL,
            label="Block",
        )
        lines = generate_mesh_lines([block], DOMAIN, params(metal_res=metal_res, min_lines=1))
        inside = pin_at(lines.fixed[0], -2.0 + metal_res / 3)
        outside = pin_at(lines.fixed[0], -2.0 - 2 * metal_res / 3)

        assert "Block" in inside.source and "inside" in inside.source
        assert "Block" in outside.source and "outside" in outside.source
        assert inside.required and outside.required

    def test_a_requested_line_says_it_was_requested(self):
        """A port's plane is pinned by nothing the user drew, so it needs a name."""
        lines = generate_mesh_lines([], DOMAIN, params(), forced=([], [], [1.234]))
        pin = pin_at(lines.fixed[2], 1.234)
        assert pin.required
        assert "requested" in pin.source

    def test_a_preference_dropped_for_crowding_is_not_claimed(self):
        """A substrate sitting on a ground plane pins one line, not two.

        The face and the sheet are at the same coordinate, so the preference is
        dropped. Reporting both would describe a grid line that the mesher
        placed for the *conductor* as belonging to the dielectric - and the
        anchor view would then show a sheet that could be moved.
        """
        lines = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, params())
        pin = pin_at(lines.fixed[2], 0.0)  # asserts exactly one
        assert "GroundPlane" in pin.source
        assert pin.required

    def test_a_preference_dropped_inside_a_thirds_span_is_not_claimed(self):
        """A dielectric face inside a conductor's thirds span gives way."""
        metal_res = 0.6
        trace = Region(
            lower=(-2.0, -2.0, 0.0),
            upper=(2.0, 2.0, 2.0),
            material=MaterialClass.METAL,
            label="Trace",
        )
        # Its face lands between the trace's inside and outside thirds lines.
        filler = Region(
            lower=(-2.0 + metal_res / 6, -8.0, -1.0),
            upper=(8.0, 8.0, -0.5),
            material=MaterialClass.DIELECTRIC,
            label="Filler",
        )
        lines = generate_mesh_lines(
            [trace, filler], DOMAIN, params(metal_res=metal_res, min_lines=1)
        )
        # Only its lower face is in the span; the far one at x = 8 is legitimate.
        sources = [pin.source for pin in lines.fixed[0]]
        assert "'Filler' lower face" not in sources, sources
        assert "'Filler' upper face" in sources, "the test lost its own geometry"
        assert not has_line(lines.x, -2.0 + metal_res / 6)

    def test_provenance_does_not_disturb_the_grid(self):
        """Carrying names must not move a line. Guards the whole refactor."""
        lines = generate_mesh_lines([substrate(), ground_sheet()], DOMAIN, params())
        bare = mesh._mesh_axis(
            [substrate(), ground_sheet()], 2, DOMAIN[0][2], DOMAIN[1][2], params()
        )
        assert np.array_equal(lines.z, bare[0])


class TestMeshLines:
    def test_indexing_matches_named_axes(self):
        lines = MeshLines(x=np.array([0.0, 1.0]), y=np.array([0.0, 2.0]), z=np.array([0.0, 3.0]))
        assert np.array_equal(lines[0], lines.x)
        assert np.array_equal(lines[1], lines.y)
        assert np.array_equal(lines[2], lines.z)

    def test_smallest_cell_spans_all_axes(self):
        lines = MeshLines(
            x=np.array([0.0, 1.0]),
            y=np.array([0.0, 0.25]),
            z=np.array([0.0, 3.0]),
        )
        assert lines.smallest_cell() == pytest.approx(0.25)


class TestAnchorsAreBitExact:
    """Pinned positions must come back as the *same float*, not a near one.

    Approximate comparison cannot see this class of bug, which is why the rest
    of this file missed it. A zero-thickness sheet occupies a zero-width
    interval, so a solver discretises it only where a grid line equals its
    position exactly. One ulp of drift and the conductor is silently absent -
    openEMS warns "Unused primitive" on stderr and simulates the structure
    without it.
    """

    def _stackup(self):
        return [
            Region((-100, -15, 0), (100, 15, 1.6), MaterialClass.DIELECTRIC, "sub"),
            Region((-100, -15, 0), (100, 15, 0), MaterialClass.METAL, "ground"),
            Region((-100, -1.5, 1.6), (100, 1.5, 1.6), MaterialClass.METAL, "trace"),
        ]

    def test_sheets_land_on_exact_lines(self):
        """This domain is centred on the substrate's midplane, so the fold
        fires - and the fold is the dangerous case, because it rewrites every
        line on the axis. The centre assertion is what says so; without it a
        later change to the domain could move the case off centre and leave the
        class testing only the easy path.
        """
        params = MeshParams(metal_res=0.5, dielectric_res=1.0, min_lines=8, pml_cells=8)
        lines = generate_mesh_lines(self._stackup(), ((-100, -27, -8), (100, 27, 9.6)), params)
        centre = (lines.z[0] + lines.z[-1]) / 2.0
        assert abs(centre - 0.8) < 1e-9, "this case is meant to be symmetric"
        for anchor in (0.0, 1.6):
            assert anchor in set(lines.z.tolist()), (
                f"no grid line exactly at z={anchor}; nearest is "
                f"{lines.z[np.argmin(np.abs(lines.z - anchor))]!r}"
            )

    def test_an_off_centre_sheet_is_exact_too(self):
        """No fold here, but the guarantee is the same."""
        regions = [
            Region((-10, -10, 0), (10, 10, 3.17), MaterialClass.DIELECTRIC, "sub"),
            Region((-10, -10, 0.41), (10, 10, 0.41), MaterialClass.METAL, "sheet"),
        ]
        params = MeshParams(metal_res=0.2, dielectric_res=0.5, pml_cells=4)
        lines = generate_mesh_lines(regions, ((-12, -12, -3), (12, 12, 7)), params)
        assert 0.41 in set(lines.z.tolist())


class TestButtedConductorsAreOnePieceOfMetal:
    """A seam is not an edge, and the grid must not pretend otherwise.

    Two conductors meeting in plane are one piece of metal: the field
    penetrates neither, so the shared face carries no singularity and the
    thirds rule - a treatment for an *isolated* edge - has nothing to resolve
    there, and neither side may claim it.

    This matters more than tidiness now that the translation cuts a drawn
    outline into rectangles by itself. Every seam it makes is one the user did
    not draw, so a seam that cost grid would tax a shape for the way it
    happened to be cut up.
    """

    COPPER = {"material": MaterialClass.METAL, "material_name": "Copper"}

    def strip(self, low, high, label, **overrides):
        settings = dict(lower=(low, -1.5, 0.0), upper=(high, 1.5, 0.0), label=label)
        settings.update(self.COPPER)
        settings.update(overrides)
        return Region(**settings)

    def axes(self, regions):
        lines = generate_mesh_lines(regions, DOMAIN, params(metal_res=0.6))
        return (lines.x, lines.y, lines.z)

    def test_a_strip_in_pieces_meshes_as_the_strip_it_was_cut_from(self):
        """The gate. No reference and no error bar - one drawing, two ways of
        expressing it, and the grids have to agree line for line."""
        whole = self.axes([self.strip(-6.0, 6.0, "whole")])
        halves = self.axes([self.strip(-6.0, 0.0, "left"), self.strip(0.0, 6.0, "right")])
        for dim, (one, cut) in enumerate(zip(whole, halves)):
            assert len(one) == len(cut), f"axis {dim}"
            assert np.allclose(one, cut, rtol=0.0, atol=0.0), f"axis {dim}"

    def test_the_agreement_does_not_depend_on_how_many_pieces(self):
        """Four, cut at coordinates that are nothing to do with the grid. A cut
        is arbitrary, so its result must be too."""
        whole = self.axes([self.strip(-6.0, 6.0, "whole")])
        quarters = self.axes(
            [
                self.strip(-6.0, -3.0, "a"),
                self.strip(-3.0, 0.0, "b"),
                self.strip(0.0, 3.5, "c"),
                self.strip(3.5, 6.0, "d"),
            ]
        )
        for one, cut in zip(whole, quarters):
            assert len(one) == len(cut)
            assert np.allclose(one, cut, rtol=0.0, atol=0.0)

    def test_a_stem_ends_at_the_bar_but_the_bar_runs_past_the_stem(self):
        """Covered, not merely touching - the distinction the whole predicate
        turns on. The stem's end face is buried in the bar and is no edge; the
        bar's face at that same plane is exposed for most of its length and
        keeps its pair of lines."""
        bar = self.strip(-6.0, 6.0, "bar")
        stem = Region(lower=(-1.5, 1.5, 0.0), upper=(1.5, 8.0, 0.0), label="stem", **self.COPPER)
        lines = generate_mesh_lines([bar, stem], DOMAIN, params(metal_res=0.6))
        # metal_res 0.6: a third inside the metal, two thirds outside.
        assert has_line(lines.y, 1.5 - 0.2)
        assert has_line(lines.y, 1.5 + 0.4)
        # And nothing pinned at the seam itself, which is what a thirds pair
        # exists to avoid.
        assert not has_line(lines.y, 1.5)

    def test_two_different_metals_still_pin_the_boundary_they_share(self):
        """No singularity, so no thirds - but a copper-to-PEC seam is a real
        property boundary, and openEMS assigns a cell that straddles it by
        priority. Off a line, the boundary moves by up to a cell and nothing
        downstream can tell."""
        lines = generate_mesh_lines(
            [
                self.strip(-6.0, 0.0, "copper"),
                self.strip(0.0, 6.0, "plate", material_name="PEC"),
            ],
            DOMAIN,
            params(metal_res=0.6),
        )
        assert has_line(lines.x, 0.0)
        assert not has_line(lines.x, -0.2)
        assert not has_line(lines.x, 0.4)

    def test_a_piece_too_thin_for_the_thirds_rule_still_knows_its_seam(self):
        """A cut can leave a piece narrower than `metal_res`, which takes the
        branch that pins both faces plainly instead of resolving them as edges.
        That branch has to ask about the seam too, or the cheapest shape to draw
        becomes the most expensive to mesh.

        The seam and the thinness are on the *same* axis here, which is the only
        arrangement that reaches the branch: a piece thick enough for the thirds
        rule is handled well above it.
        """
        thin = dict(self.COPPER)
        wide = params(metal_res=1.0)
        whole = generate_mesh_lines(
            [Region(lower=(-6.0, -0.4, 0.0), upper=(6.0, 0.4, 0.0), label="whole", **thin)],
            DOMAIN,
            wide,
        )
        halves = generate_mesh_lines(
            [
                Region(lower=(-6.0, -0.4, 0.0), upper=(6.0, 0.0, 0.0), label="near", **thin),
                Region(lower=(-6.0, 0.0, 0.0), upper=(6.0, 0.4, 0.0), label="far", **thin),
            ],
            DOMAIN,
            wide,
        )
        assert not has_line(halves.y, 0.0)
        assert len(whole.y) == len(halves.y)
        assert np.allclose(whole.y, halves.y, rtol=0.0, atol=0.0)
