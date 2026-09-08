# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What has to be true of the pillbox before a solve on it could mean anything.

None of this runs a solver. What these hold is what ``test_acceptance_pillbox``
assumes rather than measures: that the cavity carries the mode it is drawn for,
that the mode is alone where it would be read, and that the arithmetic both
sides share says what it claims.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from Microwave.Solvers.openems.materials import VACUUM_PERMITTIVITY
from tests import pillbox
from tests.analytic import reference

#: What a node comes out as when it is reached through the cosine of a right
#: angle rather than through a Bessel function's own zero. Half the modes here
#: have their node at mid-height, which is that cosine exactly.
ROUNDING = 1e-12


def _spectrum(height: float) -> dict[tuple, float]:
    """Every mode of this cavity in the band, as ``mode -> Hz``.

    Keyed by the indices rather than by the name, so a check can ask the mode
    what it does as well as what it is called.

    Enumerated rather than listed, so a height that changed which modes are
    where cannot leave a stale list behind. The indices reach past the band at
    both heights, and ``test_the_enumeration_reaches_past_the_band`` is what
    says so.
    """
    found = {}
    for kind in ("TM", "TE"):
        for order in range(4):
            for root in (1, 2, 3):
                for axial in range(4):
                    if kind == "TE" and axial == 0:
                        continue
                    at = reference.circular_cavity_frequency(
                        pillbox.RADIUS * 1e-3,
                        height * 1e-3,
                        order=order,
                        root=root,
                        axial=axial,
                        kind=kind,
                        eps_r=pillbox.EPS_R,
                    )
                    if pillbox.BAND[0] <= at <= pillbox.BAND[1]:
                        found[kind, order, root, axial] = at
    return found


class TestTheSpectrumTheProbeCanActuallyRead:
    """The spectrum case is scored against several exact roots at once, and what
    has to hold first is that each of them is a line the arrangement produces and
    that no window reaches a neighbour."""

    def test_every_mode_scored_is_in_the_band(self):
        """A mode outside the sweep has no bins to be fitted in, and a window
        clipped by the edge of the record fits a flank rather than a line."""
        spectrum = _spectrum(pillbox.TALL)
        for mode in pillbox.SPECTRUM_MODES:
            name = pillbox.mode_name(mode)
            assert mode in spectrum, f"{name} is not in the band the cavity is swept over"
            at = spectrum[mode]
            for edge in pillbox.BAND:
                assert abs(at - edge) / at > pillbox.SPECTRUM_WINDOW, (
                    f"{name} sits at {at / 1e9:.3f} GHz and its window reaches "
                    f"{100 * pillbox.SPECTRUM_WINDOW:g} % of that, past the sweep's "
                    f"{edge / 1e9:.1f} GHz"
                )

    def test_each_window_holds_one_line_the_probe_can_drive(self):
        """Two windows that overlap are two fits free to find the same line, and
        a window reaching a neighbour reports a real resonance of the same cavity
        on the wrong reference.

        Among the modes the probe can *drive*, which is the whole of what can be
        in the record. It has to be put that way here rather than over the
        spectrum entire, because `TE011` sits exactly on top of `TM111` and no
        window will ever separate those two - what separates them is that an
        element along the axis drives one of them and not the other.
        """
        spectrum = _spectrum(pillbox.TALL)
        driven = {mode: at for mode, at in spectrum.items() if pillbox.axial_field(mode) != 0.0}
        for mode in pillbox.SPECTRUM_MODES:
            want = driven[mode]
            others = [at for other, at in driven.items() if other != mode]
            nearest = min(others, key=lambda at: abs(at - want))
            assert abs(nearest - want) / want > 2 * pillbox.SPECTRUM_WINDOW, (
                f"{pillbox.mode_name(mode)}'s window reaches "
                f"{100 * pillbox.SPECTRUM_WINDOW:g} % and the next line the probe drives "
                f"is {100 * abs(nearest - want) / want:.2f} % away"
            )

    def test_and_every_line_the_probe_can_drive_is_one_of_them(self):
        """Otherwise the gate scores some of what is in the record and leaves the
        rest unaccounted for, which is the same fit wandering onto a neighbour
        wearing a different name."""
        driven = [mode for mode in _spectrum(pillbox.TALL) if pillbox.axial_field(mode) != 0.0]
        assert sorted(driven) == sorted(pillbox.SPECTRUM_MODES)

    def test_the_probe_stands_where_every_one_of_them_is_alive(self):
        """An element on the axis at mid-height reads the dominant mode and
        nothing else in the band, sitting on a node of everything with azimuthal
        variation and of everything with an odd number of half-waves along the
        axis. So where this one stands is the whole of what makes a spectrum
        readable, and a placement that had drifted onto a node would show as a
        line that would not fit rather than as anything nameable."""
        for mode in pillbox.SPECTRUM_MODES:
            share = abs(pillbox.axial_field(mode))
            assert share > 0.25, (
                f"{pillbox.mode_name(mode)} stands at {share:.3f} of its own largest "
                "where the probe is, which is a node rather than a crest"
            )

    def test_and_the_middle_is_the_placement_that_cannot_read_them(self):
        """The property the arrangement rests on, from the other side: it is the
        centre that is blind, so the move is necessary rather than tidy."""
        blind = [
            pillbox.mode_name(mode)
            for mode in pillbox.SPECTRUM_MODES
            if abs(pillbox.axial_field(mode, at=(0.0, 0.0))) < ROUNDING
        ]
        assert blind == ["TM110", "TM011", "TM111"]

    def test_a_TE_mode_is_not_among_them(self):
        """It has no axial electric field at all, so an element along the axis
        drives nothing of it - which is also what says a line found where `TE011`
        and `TM111` sit together is the second of them."""
        for mode in pillbox.SPECTRUM_MODES:
            assert mode[0] == "TM"
        assert pillbox.axial_field(("TE", 0, 1, 1)) == 0.0

    def test_two_of_the_cavity_s_modes_are_exactly_degenerate(self):
        """`TE0n1` and `TM1n1` coincide in any circular cylinder, the derivative
        of `J0` being `-J1`, so no fit at any resolution separates them and only
        the probe says which was found."""
        assert pillbox.mode_frequency(("TE", 0, 1, 1)) == pytest.approx(
            pillbox.mode_frequency(("TM", 1, 1, 1)), rel=1e-12, abs=0.0
        )


class TestTheCavityCarriesTheModeItIsDrawnFor:
    """``TE111`` falls as a cavity is stretched and the flat ``TM`` mode does
    not, so past about two radii of height the two cross and the lowest line is
    a different mode. Held against the spectrum rather than against that ratio,
    so it stays true of whatever the dimensions become."""

    def test_the_flat_mode_is_the_lowest_at_every_height(self):
        for height in (pillbox.TALL, pillbox.SHORT):
            spectrum = _spectrum(height)
            lowest = min(spectrum, key=spectrum.get)
            assert lowest == pillbox.DOMINANT, (
                f"at {height} mm the lowest mode in the band is "
                f"{pillbox.mode_name(lowest)} at {spectrum[lowest] / 1e9:.3f} GHz, not the "
                "one the fixture reads"
            )

    def test_nothing_else_is_within_reach_of_the_window(self):
        """A fit that wandered onto a neighbour would report a real resonance of
        the same cavity and look entirely healthy."""
        for height in (pillbox.TALL, pillbox.SHORT):
            spectrum = _spectrum(height)
            want = spectrum[pillbox.DOMINANT]
            others = [at for mode, at in spectrum.items() if mode != pillbox.DOMINANT]
            nearest = min(others, key=lambda at: abs(at - want))
            assert abs(nearest - want) / want > 2 * pillbox.WINDOW, (
                f"at {height} mm the window reaches {100 * pillbox.WINDOW:g} % and the "
                f"next mode is {100 * abs(nearest - want) / want:.1f} % away"
            )

    def test_the_enumeration_reaches_past_the_band(self):
        """Otherwise the claims above are about how far the loop happened to
        count rather than about the cavity.

        The frequency climbs with every index, so what binds is the first step
        *beyond* each loop with the other indices at their smallest - the
        cheapest mode the enumeration does not reach. Every one of those has to
        be out of the band already, and the corner where all three are largest
        says nothing, being out of it by a wide margin whatever the loops are.
        """
        beyond = {
            "TM": ((4, 1, 0), (0, 4, 0), (0, 1, 4)),
            "TE": ((4, 1, 1), (1, 4, 1), (1, 1, 4)),
        }
        for kind, indices in beyond.items():
            for order, root, axial in indices:
                first = reference.circular_cavity_frequency(
                    pillbox.RADIUS * 1e-3,
                    pillbox.TALL * 1e-3,
                    order=order,
                    root=root,
                    axial=axial,
                    kind=kind,
                    eps_r=pillbox.EPS_R,
                )
                assert first > pillbox.BAND[1], (
                    f"{kind}{order}{root}{axial} is at {first / 1e9:.2f} GHz, inside a band "
                    f"stopping at {pillbox.BAND[1] / 1e9:.2f}, and the enumeration never reaches it"
                )

    def test_the_height_is_not_in_the_frequency_that_is_read(self):
        """The property the whole arrangement rests on, stated where the fixture
        can lose it: the two heights are one resonance."""
        assert (
            _spectrum(pillbox.TALL)[pillbox.DOMINANT] == _spectrum(pillbox.SHORT)[pillbox.DOMINANT]
        )

    def test_the_two_heights_are_far_enough_apart_to_be_a_change(self):
        assert pillbox.SHORT < 0.75 * pillbox.TALL


class TestTheArithmeticBothSidesShare:
    def test_a_measured_frequency_reads_back_as_the_radius(self):
        assert pillbox.effective_radius(pillbox.frequency()) == pytest.approx(
            pillbox.RADIUS, rel=1e-12, abs=0.0
        )

    def test_and_a_smaller_cavity_reads_back_larger(self):
        """The sign is the whole use of it: sampling leaves the metal inscribed,
        which opens the bore out, and that has to read as a wall further out
        rather than nearer."""
        assert pillbox.effective_radius(pillbox.frequency() * 0.99) > pillbox.RADIUS

    def test_a_displacement_is_worth_its_share_of_the_radius(self):
        """The mode goes as one over the radius and depends on nothing else, so
        a wall moved by some fraction of the radius moves the frequency by the
        same fraction. Checked against the closed form rather than against the
        expression, which is the same line twice."""
        for distance in (0.01, 0.1, 0.5):
            moved = reference.circular_cavity_frequency(
                (pillbox.RADIUS + distance) * 1e-3, pillbox.TALL * 1e-3, eps_r=pillbox.EPS_R
            )
            assert abs(moved - pillbox.frequency()) / pillbox.frequency() == pytest.approx(
                pillbox.displacement_worth(distance), rel=0.05, abs=0.0
            )

    def test_the_wall_cell_is_the_one_the_wall_falls_in(self):
        lines = [0.0, 5.0, 10.0, 14.8, 15.4, 20.0]
        assert pillbox.wall_cell(lines) == pytest.approx(0.6, rel=1e-12, abs=0.0)

    def test_a_line_on_the_wall_belongs_to_the_cell_above_it(self):
        """The case that separates the two neighbours, and the one the mesher
        actually produces: it pins a line to a conductor's surface, so the wall
        lands on a line rather than between two."""
        lines = [0.0, 10.0, 14.0, pillbox.RADIUS, 15.7, 20.0]
        assert pillbox.wall_cell(lines) == pytest.approx(0.7, rel=1e-12, abs=0.0)

    def test_and_it_reads_a_uniform_grid_as_its_own_spacing(self):
        for spacing in (0.9, 0.4, 0.25):
            lines = np.arange(-20.0, 20.0 + spacing / 2, spacing)
            assert pillbox.wall_cell(lines) == pytest.approx(spacing, rel=1e-9, abs=0.0)

    def test_the_wall_phase_is_how_far_across_that_cell_it_stands(self):
        lines = [0.0, 5.0, 10.0, 14.8, 15.4, 20.0]
        assert pillbox.wall_phase(lines) == pytest.approx(1.0 / 3.0, rel=1e-12, abs=0.0)

    def test_a_line_on_the_wall_reads_as_the_bottom_of_the_cell_above_it(self):
        """The same seam :func:`wall_cell` is read at, answered the same way, so
        the two describe one cell rather than two neighbouring ones."""
        lines = [0.0, 10.0, 14.0, pillbox.RADIUS, 15.7, 20.0]
        assert pillbox.wall_phase(lines) == pytest.approx(0.0, rel=1e-12, abs=0.0)

    def test_a_phase_is_a_share_of_its_own_cell_and_not_a_length(self):
        """So a sequence holding it holds the same *shape* of sampling at every
        cell, which is the whole point of holding it."""
        for spacing in (0.9, 0.4, 0.25):
            below = pillbox.RADIUS - 0.75 * spacing
            lines = below + spacing * np.arange(-20, 21)
            assert pillbox.wall_phase(lines) == pytest.approx(0.75, rel=1e-9, abs=0.0)

    def test_the_cell_is_a_length_in_the_units_the_drawing_is_in(self):
        """The mesh resolution every case is planned at, so an answer in metres
        would leave each of them a thousand times finer than intended and only
        the solve would notice. Anchored on the wavelength it divides, which is
        the one length here that the cavity does not supply."""
        wavelength = reference.SPEED_OF_LIGHT / pillbox.BAND[1] * 1e3
        assert 10.0 < wavelength < 100.0
        for divisor in pillbox.DIVISORS:
            assert pillbox.cell_size(divisor) == pytest.approx(
                wavelength / divisor, rel=1e-12, abs=0.0
            )

    def test_the_loss_tangent_is_the_conductivity_it_was_declared_as(self):
        """A conductivity and a loss tangent are the same statement at one
        frequency, and the fit is started from the Q this implies. Restated as
        the current the field drives against the current that stores it, which
        is the definition rather than the expression."""
        at = pillbox.frequency()
        conduction = pillbox.KAPPA
        displacement = 2 * np.pi * at * VACUUM_PERMITTIVITY * pillbox.EPS_R
        assert pillbox.loss_tangent() == pytest.approx(
            conduction / displacement, rel=1e-12, abs=0.0
        )
        assert pillbox.loss_tangent(2.0 * at) == pytest.approx(
            pillbox.loss_tangent(at) / 2.0, rel=1e-12, abs=0.0
        )


class TestWhatAPolygonOfThisCylinderIsWorth:
    """A triangulation reaches the solver as a prism on an inscribed polygon, so
    the cavity it stands for is slightly the smaller one. :func:`enclosed_radius`
    is what prices that, and here it is given polygons whose answer is known.
    """

    @staticmethod
    def _prism(sides: int, radius: float, height: float):
        angle = 2 * np.pi * np.arange(sides) / sides
        rim = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
        vertices = [(*point, -height / 2) for point in rim]
        vertices += [(*point, height / 2) for point in rim]
        vertices += [(0.0, 0.0, -height / 2), (0.0, 0.0, height / 2)]
        low, high = 2 * sides, 2 * sides + 1
        faces = []
        for i in range(sides):
            j = (i + 1) % sides
            faces += [(i, j, i + sides), (j, j + sides, i + sides)]  # side
            faces += [(low, j, i), (high, i + sides, j + sides)]  # ends
        return vertices, faces

    def test_a_polygon_reads_back_as_the_circle_of_its_own_area(self):
        """Exactly, so what the function measures is the area rather than the
        radius it was drawn at."""
        for sides in (8, 32, 126):
            vertices, faces = self._prism(sides, 15.0, 15.0)
            equal_area = 15.0 * np.sqrt(sides / (2 * np.pi) * np.sin(2 * np.pi / sides))
            assert pillbox.enclosed_radius(vertices, faces, 15.0) == pytest.approx(
                equal_area, rel=1e-9, abs=0.0
            )

    def test_a_finer_polygon_loses_less(self):
        deficits = [
            15.0 - pillbox.enclosed_radius(*self._prism(sides, 15.0, 15.0), 15.0)
            for sides in (8, 16, 32, 64, 126)
        ]
        assert deficits == sorted(deficits, reverse=True)

    def test_the_winding_does_not_enter(self):
        """A kernel is free to hand back either winding, and a volume read off
        the wrong one is negative."""
        vertices, faces = self._prism(32, 15.0, 15.0)
        reversed_faces = [tuple(reversed(face)) for face in faces]
        assert pillbox.enclosed_radius(vertices, reversed_faces, 15.0) == pytest.approx(
            pillbox.enclosed_radius(vertices, faces, 15.0), rel=1e-12, abs=0.0
        )


class TestTheCaseTableIsWhatItClaims:
    def test_the_sequence_is_the_tall_cavity_at_every_cell(self):
        assert set(pillbox.SEQUENCE) <= set(pillbox.CASES)
        cells = [pillbox.cell_size(pillbox.CASES[name].divisor) for name in pillbox.SEQUENCE]
        assert cells == sorted(cells, reverse=True), "the sequence is not coarsest cell first"
        assert len(set(cells)) == len(cells)

    def test_every_other_case_changes_exactly_one_thing(self):
        """What each case says is about the thing it changed, so two at once
        would make it about neither.

        Against the point of the sequence sharing its cell, rather than against
        the coarsest: the sequence is the same cavity at several cells, so any
        of them is a fair starting point, and the spectrum is read at a cell of
        its own because the coarsest cannot separate the lines it scores.

        A case that is one half of a *pair* is measured against its own partner
        instead, that being what it is subtracted from. Which is the same rule
        stated over the right baseline rather than an exception to it.
        """
        sequence = {pillbox.CASES[name].divisor: pillbox.CASES[name] for name in pillbox.SEQUENCE}
        against = {name: pillbox.CASES[partner] for name, partner in (pillbox.ON_THE_FACE,)}
        for name, case in pillbox.CASES.items():
            if name in pillbox.SEQUENCE:
                continue
            plain = against.get(name, sequence[case.divisor])
            differ = [
                field.name
                for field in dataclasses.fields(pillbox.Case)
                if getattr(case, field.name) != getattr(plain, field.name)
            ]
            assert len(differ) == 1, (
                f"{name} differs from what it is read against in {differ}, so what it "
                "answers is about all of them or about none"
            )

    def test_the_probe_still_spans_a_cell_at_the_coarsest(self):
        """A lumped element whose two ends snap to one line has no length,
        openEMS skips it, and the run comes back NaN - which pre-flight refuses.
        That needs a cell wider than the gap; asked for here is a cell narrower
        than it, which is the conservative side of the same condition and leaves
        the element more than a single edge to be an integral over.
        """
        for name, case in pillbox.CASES.items():
            assert pillbox.cell_size(case.divisor) < case.probe, (
                f"{name}: the cell is {pillbox.cell_size(case.divisor):.4f} mm and the "
                f"probe spans {case.probe} mm, which is not room for an element"
            )

    def test_the_lattice_offsets_are_inside_one_cell(self):
        """Sliding by a whole cell is the same grid again, so an offset beyond
        one is an alignment already covered wearing another name."""
        for phase in pillbox.LATTICE_PHASES:
            assert all(0.0 < value <= 0.5 for value in phase if value)
            assert any(phase)

    def test_no_two_cases_share_a_name(self):
        """A case is its own directory, so a collision is one solve reported
        twice. Compared over whatever a case is made of rather than over a list
        written here, which a new field would silently fall out of."""
        assert len(pillbox.CASES) == len(
            {dataclasses.astuple(case) for case in pillbox.CASES.values()}
        )
