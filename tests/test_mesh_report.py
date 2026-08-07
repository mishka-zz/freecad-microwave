# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Unit tests for the mesh report.

The report's whole value is that it is *true about the grid it was handed*, so
most of these build a grid whose every answer is known by hand and check the
report against arithmetic rather than against the mesher. Two of them go the
other way and run the real mesher, because a report that is self-consistent on
synthetic input and wrong on real input would pass everything else here.

Nothing imports openEMS, CSXCAD or FreeCAD.
"""

import math

import numpy as np
import pytest

import Microwave.Solvers.openems.mesh as mesh
from Microwave.Solvers.openems.mesh import (
    FixedLine,
    MaterialClass,
    MeshLines,
    Region,
    generate_mesh_lines,
)
from Microwave.Solvers.openems.model import SPEED_OF_LIGHT
from Microwave.Solvers.openems.report import Extent, mesh_report
from tests.mesh_fixtures import DOMAIN, params, stackup


def uniform(pitch=1.0, count=11, *, pinned=True) -> MeshLines:
    """A cube of uniform cells, with the two ends pinned on each axis."""
    axis = np.arange(count, dtype=float) * pitch
    ends = (
        (
            FixedLine(float(axis[0]), "lower wall", True),
            FixedLine(float(axis[-1]), "upper wall", True),
        )
        if pinned
        else ()
    )
    return MeshLines(x=axis.copy(), y=axis.copy(), z=axis.copy(), fixed=(ends,) * 3)


def cfl(dx: float, dy: float, dz: float) -> float:
    """The closed form the report is asserted against, in mm in and seconds out."""
    inverse = sum(1.0 / (d * 1e-3) ** 2 for d in (dx, dy, dz))
    return 1.0 / (SPEED_OF_LIGHT * math.sqrt(inverse))


class TestCounts:
    def test_cells_and_lines_match_the_grid(self):
        """Cells are lines multiplied, openEMS' convention: it updates and
        allocates per line, the outermost included, and divides by that product
        to report MCells/s."""
        report = mesh_report(uniform(count=11), [], params())
        assert report.lines == (11, 11, 11)
        assert report.cells == 11 * 11 * 11

    def test_cell_steps_needs_a_step_count(self):
        assert mesh_report(uniform(), [], params()).cell_steps is None

    def test_cell_steps_is_cells_times_steps(self):
        report = mesh_report(uniform(count=11), [], params(), max_timesteps=500)
        assert report.cell_steps == 1331 * 500


class TestTimestep:
    """Asserted against the closed form, never against the code's own output.

    Every comparison here passes ``abs=0.0`` deliberately. A timestep is of
    order 1e-13 s and ``pytest.approx`` defaults to an *absolute* tolerance of
    1e-12, which silently swallows any ``rel=`` given beside it - without
    ``abs=0`` these tests accept a timestep off by a factor of two, and a
    mutation run proved they did.
    """

    def test_uniform_grid_matches_the_courant_closed_form(self):
        report = mesh_report(uniform(pitch=1.0), [], params())
        assert report.timestep == pytest.approx(cfl(1.0, 1.0, 1.0), rel=1e-12, abs=0.0)

    def test_anisotropic_grid_matches_the_closed_form(self):
        lines = MeshLines(
            x=np.arange(11) * 1.0,
            y=np.arange(11) * 2.0,
            z=np.arange(11) * 4.0,
        )
        report = mesh_report(lines, [], params())
        assert report.timestep == pytest.approx(cfl(1.0, 2.0, 4.0), rel=1e-12, abs=0.0)

    def test_the_smallest_cell_on_an_axis_is_what_counts(self):
        """One sliver divides the timestep for the whole domain."""
        coarse = mesh_report(uniform(pitch=1.0), [], params()).timestep
        slivered = MeshLines(
            x=np.array([0.0, 0.01, *np.arange(1, 11, dtype=float)]),
            y=np.arange(11) * 1.0,
            z=np.arange(11) * 1.0,
        )
        assert mesh_report(slivered, [], params()).timestep == pytest.approx(
            cfl(0.01, 1.0, 1.0), rel=1e-12, abs=0.0
        )
        assert mesh_report(slivered, [], params()).timestep < coarse / 50

    def test_the_factor_scales_it_linearly(self):
        full = mesh_report(uniform(), [], params(), timestep_factor=1.0).timestep
        half = mesh_report(uniform(), [], params(), timestep_factor=0.5).timestep
        assert half == pytest.approx(full / 2.0, rel=1e-12, abs=0.0)


class TestExtents:
    """The one number that catches a model drawn in the wrong unit."""

    def test_the_domain_excludes_the_absorber(self):
        report = mesh_report(uniform(pitch=1.0, count=21), [], params(pml_cells=8))
        # 21 lines, 8 absorber cells a side: the domain is lines 8..12.
        assert report.domain.lower == (8.0, 8.0, 8.0)
        assert report.domain.upper == (12.0, 12.0, 12.0)
        assert report.domain.size == (4.0, 4.0, 4.0)

    def test_the_outer_extent_includes_it(self):
        report = mesh_report(uniform(pitch=1.0, count=21), [], params(pml_cells=8))
        assert report.outer.lower == (0.0, 0.0, 0.0)
        assert report.outer.size == (20.0, 20.0, 20.0)
        assert report.has_absorber

    def test_without_an_absorber_the_two_agree(self):
        report = mesh_report(uniform(count=11), [], params(pml_cells=0))
        assert report.domain == report.outer
        assert not report.has_absorber

    def test_a_per_axis_absorber_is_honoured(self):
        """A waveguide absorbs on two faces and is walled by metal on four."""
        report = mesh_report(uniform(pitch=1.0, count=21), [], params(pml_cells=(8, 0, 8)))
        assert report.domain.size == (4.0, 20.0, 4.0)

    def test_a_grid_too_short_for_its_absorber_falls_back(self):
        """Hand-built grids reach this; generate_mesh_lines refuses to make one."""
        report = mesh_report(uniform(count=5), [], params(pml_cells=8))
        assert report.domain == report.outer

    def test_the_summary_states_the_domain_size_and_corner(self):
        text = mesh_report(uniform(pitch=1.0, count=21), [], params(pml_cells=8)).summary()
        assert "domain 4 x 4 x 4 mm at (8, 8, 8)" in text
        assert "20 x 20 x 20 mm with the absorber" in text


class TestCoverage:
    """How much of the model survived the absorber.

    The pair of numbers that names the fault: a domain alone is a measurement, a domain
    beside the structure it was cut from is a verdict. Every case here is
    arithmetic on a hand-built grid - 21 lines at unit pitch, 8 absorber cells
    a side, so the domain is 8..12 whatever the structure does.
    """

    GRID = dict(pitch=1.0, count=21)

    def report(self, lower, upper, **kw):
        return mesh_report(
            uniform(**self.GRID), [], params(pml_cells=8), structure=Extent(lower, upper), **kw
        )

    def test_the_absorber_is_measured_off_the_grid(self):
        """Not ``pml_cells * cap`` - those two are exactly what disagree."""
        axis = self.report((8.0,) * 3, (12.0,) * 3).coverage[0]
        assert (axis.below, axis.above) == (8.0, 8.0)

    def test_a_domain_smaller_than_the_model_reports_its_share(self):
        """A THROUGH face: the structure runs out through the absorber."""
        assert self.report((0.0,) * 3, (20.0,) * 3).coverage[0].share == 0.2

    def test_an_absorber_that_ate_the_model_says_so(self):
        text = self.report((-40.0,) * 3, (60.0,) * 3).summary()
        assert "interior covers 4% of the structure" in text

    def test_a_healthy_model_says_so_in_the_same_words(self):
        text = self.report((7.0,) * 3, (13.0,) * 3).summary()
        assert "interior covers 67% of the structure" in text

    def test_a_padded_face_reports_clearance_instead_of_a_share(self):
        """Outward padding puts the domain *outside* the model, and a share
        over 100% describes nothing. 180% was the first thing this printed."""
        text = self.report((9.0,) * 3, (11.0,) * 3).summary()
        assert "1 + 1 mm of air outside the structure" in text
        assert "% of the structure" not in text

    def test_a_model_exactly_filling_the_domain_reads_as_whole(self):
        """The WR-42 fixture: the guide *is* the domain, to the millimetre."""
        text = self.report((8.0,) * 3, (12.0,) * 3).summary()
        assert "interior covers 100% of the structure" in text

    def test_an_axis_with_no_absorber_is_not_described(self):
        report = mesh_report(
            uniform(**self.GRID),
            [],
            params(pml_cells=(8, 0, 8)),
            structure=Extent((0.0,) * 3, (20.0,) * 3),
        )
        assert "  y: absorber" not in report.summary()
        assert "  x: absorber" in report.summary()

    def test_a_flat_axis_does_not_divide_by_zero(self):
        """A sheet has zero extent on one axis and still has to be reported."""
        axis = self.report((8.0, 8.0, 10.0), (12.0, 12.0, 10.0)).coverage[2]
        assert axis.share == 1.0

    def test_without_a_structure_nothing_is_claimed(self):
        report = mesh_report(uniform(**self.GRID), [], params(pml_cells=8))
        assert report.coverage is None
        assert "of the structure" not in report.summary()


class TestExtremes:
    def test_smallest_finds_the_cell_and_its_axis(self):
        lines = MeshLines(
            x=np.arange(11) * 1.0,
            y=np.array([0.0, 0.25, *np.arange(1, 11, dtype=float)]),
            z=np.arange(11) * 1.0,
        )
        report = mesh_report(lines, [], params())
        assert report.smallest.size == pytest.approx(0.25)
        assert report.smallest.axis == "y"
        assert (report.smallest.lower, report.smallest.upper) == (0.0, 0.25)

    def test_largest_finds_the_cell_and_its_axis(self):
        lines = MeshLines(
            x=np.arange(11) * 1.0,
            y=np.arange(11) * 1.0,
            z=np.array([0.0, 7.0, *np.arange(8, 18, dtype=float)]),
        )
        report = mesh_report(lines, [], params())
        assert report.largest.size == pytest.approx(7.0)
        assert report.largest.axis == "z"

    def test_the_smallest_cell_is_reported_between_its_pinned_lines(self):
        """The number prompts 'why'; the names are the answer.

        Four pins, not two: with a single pin on each side the nearest and the
        outermost are the same object, so picking either passed. This is the one
        line of the report that turns a number into an action, and the failure
        it hid names the domain bounds for every cell in the model.
        """
        axis = np.array([0.0, 1.0, 1.1, 2.0, 3.0])
        lines = MeshLines(
            x=axis,
            y=np.arange(5, dtype=float),
            z=np.arange(5, dtype=float),
            fixed=(
                (
                    FixedLine(0.0, "domain lower bound", True),
                    FixedLine(1.0, "'Trace' edge at 1, outside", True),
                    FixedLine(1.1, "'Trace' edge at 1, inside", True),
                    FixedLine(3.0, "domain upper bound", True),
                ),
                (),
                (),
            ),
        )
        report = mesh_report(lines, [], params(pml_cells=0))
        assert report.smallest.between == (
            "'Trace' edge at 1, outside",
            "'Trace' edge at 1, inside",
        )

    def test_a_cell_beyond_every_pin_is_named_as_absorber(self):
        """The absorber is added outside the last pinned line, so it has none."""
        lines = MeshLines(
            x=np.array([-1.0, 0.0, 1.0, 2.0]),
            y=np.arange(4, dtype=float),
            z=np.arange(4, dtype=float),
            fixed=((FixedLine(0.0, "domain lower bound", True),), (), ()),
        )
        report = mesh_report(lines, [], params())
        assert "absorber" in report.smallest.between[0]

    def test_worst_ratio_is_the_largest_adjacent_step(self):
        lines = MeshLines(
            x=np.array([0.0, 1.0, 3.0, 4.0]),  # 1, 2, 1 -> ratio 2 both ways
            y=np.arange(4, dtype=float),
            z=np.arange(4, dtype=float),
        )
        assert mesh_report(lines, [], params()).worst_ratio == pytest.approx(2.0)

    def test_a_shrinking_step_counts_as_much_as_a_growing_one(self):
        """A grid that only ever coarsens is graded just as hard as one that refines.

        The previous test cannot tell: its spacings are 1, 2, 1, so the forward
        ratio alone already peaks at 2 and a one-directional comparison passes
        it. Here the spacings only shrink, so forward ratios are all below 1 and
        anything that fails to look both ways reports a perfectly smooth grid.
        """
        lines = MeshLines(
            x=np.array([0.0, 2.0, 3.0]),  # 2 then 1 - shrinking only
            y=np.arange(4, dtype=float),
            z=np.arange(4, dtype=float),
        )
        assert mesh_report(lines, [], params()).worst_ratio == pytest.approx(2.0)

    def test_a_uniform_grid_has_a_ratio_of_one(self):
        assert mesh_report(uniform(), [], params()).worst_ratio == pytest.approx(1.0)

    def test_an_axis_of_one_cell_does_not_crash(self):
        """There is no adjacent pair to compare, and no ratio to report.

        ``generate_mesh_lines`` never produces such an axis, but ``mesh_report``
        takes any ``MeshLines`` - and the guard against it was previously
        unreachable from any test, so deleting it would have been invisible.
        """
        lines = MeshLines(
            x=np.array([0.0, 1.0]),
            y=np.array([0.0, 1.0]),
            z=np.array([0.0, 1.0]),
        )
        report = mesh_report(lines, [], params())
        assert report.worst_ratio == pytest.approx(1.0)
        assert report.cells == 8  # two lines on each axis, openEMS' count

    def test_a_tie_across_axes_resolves_the_same_way_every_time(self):
        """A uniform grid ties on all three axes. Which one is named is
        arbitrary, but it must not vary between runs or a report flickers."""
        first = mesh_report(uniform(), [], params())
        second = mesh_report(uniform(), [], params())
        assert first.smallest.axis == second.smallest.axis == "x"
        assert first.largest.axis == second.largest.axis == "x"


class TestFeatureResolution:
    """Cells across an object - the number that says whether it is resolved."""

    @pytest.mark.parametrize(
        "low, high, expected",
        [
            (2.0, 5.0, 3),  # flush with lines
            (2.5, 4.5, 3),  # straddling, same three cells
            (2.2, 2.8, 1),  # entirely inside one cell
            (0.0, 10.0, 10),  # the whole axis
        ],
    )
    def test_cells_across_counts_the_cells_the_object_touches(self, low, high, expected):
        region = Region((low, 0.0, 0.0), (high, 10.0, 10.0), MaterialClass.DIELECTRIC, "R")
        report = mesh_report(uniform(count=11), [region], params())
        assert report.features[0].across[0] == expected

    def test_a_sheet_reports_zero_on_its_flat_axis(self):
        sheet = Region((0.0, 0.0, 3.0), (10.0, 10.0, 3.0), MaterialClass.METAL, "GND")
        feature = mesh_report(uniform(count=11), [sheet], params()).features[0]
        assert feature.across[2] == 0
        assert feature.is_sheet

    def test_a_sheet_is_still_judged_on_the_axes_it_does_have(self):
        """A ground plane is flat in z and perfectly ordinary in x and y.

        The zero must not drag ``thinnest`` down, or every sheet in every model
        reads as unresolved and the warning becomes noise.
        """
        sheet = Region((0.0, 0.0, 3.0), (10.0, 10.0, 3.0), MaterialClass.METAL, "GND")
        feature = mesh_report(uniform(count=11), [sheet], params()).features[0]
        assert feature.thinnest == 10

    def test_the_material_is_carried_through(self):
        regions = [
            Region((0.0, 0.0, 0.0), (5.0, 5.0, 5.0), MaterialClass.METAL, "M"),
            Region((0.0, 0.0, 0.0), (5.0, 5.0, 5.0), MaterialClass.DIELECTRIC, "D"),
        ]
        report = mesh_report(uniform(count=11), regions, params())
        assert [f.material for f in report.features] == ["metal", "dielectric"]

    def test_an_ordinary_grid_is_not_called_oversized(self):
        assert mesh_report(uniform(count=11), [], params()).oversized is None

    def test_a_grid_nobody_meant_says_so_in_the_summary(self):
        """*Update Mesh* never reaches pre-flight - it meshes, draws, and
        prints this. So the one route on which every runaway grid was actually
        met is the one that has to carry the sentence."""
        side = round((mesh.LARGE_GRID_BYTES / mesh.BYTES_PER_CELL) ** (1 / 3)) + 2
        report = mesh_report(uniform(count=side), [], params())

        assert report.oversized is not None
        assert "MaxGrowthRatio" in report.oversized
        assert report.oversized in report.summary()

    def test_a_one_cell_object_is_flagged_unresolved(self):
        thin = Region((2.2, 0.0, 0.0), (2.8, 10.0, 10.0), MaterialClass.DIELECTRIC, "Thin")
        report = mesh_report(uniform(count=11), [thin], params())
        assert [f.label for f in report.unresolved] == ["Thin"]

    def test_two_cells_across_is_resolved_enough_to_pass(self):
        """The bar is *fewer than two*, and where a bar sits is the whole of it.

        Two samples through a layer is where a report stops having anything to
        say; a section that also fires at two fires on ordinary models, and one
        that fires at neither says nothing when a substrate gets a single cell.
        """
        two = Region((2.2, 0.0, 0.0), (3.8, 10.0, 10.0), MaterialClass.DIELECTRIC, "Two")
        report = mesh_report(uniform(count=11), [two], params())
        assert report.features[0].thinnest == 2
        assert report.unresolved == ()

    def test_a_well_resolved_object_is_not_flagged(self):
        fat = Region((0.0, 0.0, 0.0), (5.0, 5.0, 5.0), MaterialClass.DIELECTRIC, "Fat")
        assert mesh_report(uniform(count=11), [fat], params()).unresolved == ()

    def test_a_sheet_is_not_flagged_for_being_flat(self):
        sheet = Region((0.0, 0.0, 3.0), (10.0, 10.0, 3.0), MaterialClass.METAL, "GND")
        assert mesh_report(uniform(count=11), [sheet], params()).unresolved == ()

    def test_a_one_cell_solid_conductor_is_not_flagged(self):
        """The same box, twice, differing only in what it is made of.

        One cell through a foil is what ``min_lines`` deliberately lays, so a
        report that flags it is the mesher warning about its own policy - and
        a section that fires on every model with a foil in it is a section the
        reader learns to skip.
        """
        box = ((2.2, 0.0, 0.0), (2.8, 10.0, 10.0))

        def flagged(material):
            region = Region(*box, material, "Foil")
            return [f.label for f in mesh_report(uniform(count=11), [region], params()).unresolved]

        assert flagged(MaterialClass.METAL) == []
        assert flagged(MaterialClass.DIELECTRIC) == ["Foil"]

    def test_a_one_cell_wide_conductor_sheet_is_flagged(self):
        """A sheet has no thickness in the count, so its `1` is a width.

        The exemption above is about a foil's thickness, and ``thinnest`` skips
        the axes an object is flat on - so on a sheet it can only be a width
        or a length, and a trace one cell wide is exactly what a conductor
        exempt from the element count can quietly become.
        """
        narrow = Region((2.2, 0.0, 3.0), (2.8, 10.0, 3.0), MaterialClass.METAL, "Trace")
        report = mesh_report(uniform(count=11), [narrow], params())
        assert report.features[0].thinnest == 1
        assert [f.label for f in report.unresolved] == ["Trace"]

    def test_a_degenerate_point_is_described_rather_than_flagged(self):
        """A region with no extent on any axis has no thickness to under-resolve.

        Reachable from a degenerate CAD solid, so the behaviour is pinned rather
        than left to whatever ``min()`` of an empty list happens to do.
        """
        point = Region((3.0, 3.0, 3.0), (3.0, 3.0, 3.0), MaterialClass.METAL, "Speck")
        report = mesh_report(uniform(count=11), [point], params())
        feature = report.features[0]
        assert feature.across == (0, 0, 0)
        assert feature.thinnest == 0
        assert feature.is_sheet
        assert report.unresolved == ()
        assert "Speck: a point" in report.summary()


class TestAgainstTheRealMesher:
    """Self-consistency on synthetic grids proves nothing about real ones."""

    def test_the_report_describes_the_grid_it_was_given(self):
        regions = stackup()
        lines = generate_mesh_lines(regions, DOMAIN, params())
        report = mesh_report(lines, regions, params())

        assert report.cells == lines.cell_count
        assert report.smallest.size == pytest.approx(lines.smallest_cell())
        assert report.lines == lines.shape

    def test_min_lines_shows_up_as_cells_across_the_substrate(self):
        """A property the mesher promises, read back through the report."""
        regions = stackup()
        lines = generate_mesh_lines(regions, DOMAIN, params(min_lines=9))
        report = mesh_report(lines, regions, params(min_lines=9))
        substrate = next(f for f in report.features if f.label == "Substrate")
        assert substrate.across[2] >= 9

    def test_the_worst_ratio_stays_inside_the_grading_budget(self):
        regions = stackup()
        lines = generate_mesh_lines(regions, DOMAIN, params(max_ratio=(1.3,) * 3))
        assert mesh_report(lines, regions, params()).worst_ratio <= 1.3 * (1 + 1e-6)

    def test_every_pinned_line_survives_into_the_report(self):
        regions = stackup()
        lines = generate_mesh_lines(regions, DOMAIN, params())
        report = mesh_report(lines, regions, params())
        assert report.pinned == lines.fixed
        assert any("GroundPlane" in pin.source for pin in report.pinned[2])


class TestSummary:
    """The text is the deliverable; it has to carry the actionable parts."""

    def test_it_names_what_caused_the_smallest_cell(self):
        regions = [
            Region((-8, -8, 0.0), (8, 8, 1.6), MaterialClass.DIELECTRIC, "Substrate"),
            Region((-2, -2, 1.6), (2, 2, 1.6), MaterialClass.METAL, "Trace"),
        ]
        lines = generate_mesh_lines(regions, DOMAIN, params())
        text = mesh_report(lines, regions, params()).summary()
        assert "smallest cell" in text
        assert text.count("\n") >= 4
        for pin in lines.fixed[0] + lines.fixed[1] + lines.fixed[2]:
            if pin.source in text:
                break
        else:
            pytest.fail(f"no pinned line was named in:\n{text}")

    def test_it_reports_cell_steps_when_given_a_step_count(self):
        """The property was tested; the line that renders it was not."""
        text = mesh_report(uniform(count=11), [], params(), max_timesteps=500).summary()
        assert "665,500 cell-steps at 500 steps" in text

    def test_it_stays_quiet_about_cell_steps_when_not(self):
        assert "cell-steps" not in mesh_report(uniform(), [], params()).summary()

    def test_it_reports_the_meshing_time_when_given_one(self):
        text = mesh_report(uniform(), [], params(), elapsed=0.0074).summary()
        assert "7 ms" in text

    def test_it_stays_quiet_about_time_when_not(self):
        assert "ms" not in mesh_report(uniform(), [], params()).summary()

    def test_an_unresolved_object_is_called_out(self):
        thin = Region((2.2, 0.0, 0.0), (2.8, 10.0, 10.0), MaterialClass.DIELECTRIC, "Thin")
        text = mesh_report(uniform(count=11), [thin], params()).summary()
        assert "Thin" in text and "across at its thinnest" in text

    def test_a_sheet_is_described_as_one_rather_than_as_zero_cells(self):
        sheet = Region((0.0, 0.0, 3.0), (10.0, 10.0, 3.0), MaterialClass.METAL, "GND")
        text = mesh_report(uniform(count=11), [sheet], params()).summary()
        assert "zero thickness in z" in text
        # z is the flat axis, so it must not appear as a cell count at all.
        assert "across z" not in text
        assert "across x" in text
