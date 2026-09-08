# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Acceptance gate: a WR-42 rectangular waveguide, against exact theory.

The microstrip gate is checked against Hammerstad, an empirical fit quoted to
about 1%. That error bar is wide enough to hide a real mistake inside: a mesh
four times too coarse at the conductor edge sat comfortably within it for a full
day of work before anyone noticed. This case exists because it has no error bar.

Everything asserted here falls straight out of separating Maxwell's equations in
a hollow rectangular pipe of perfect conductor. There is no fitting and no
quoted accuracy, so a discretisation mistake has nowhere to hide::

    kc    = sqrt((m*pi/a)^2 + (n*pi/b)^2)      transverse wavenumber
    f_c   = c * kc / 2*pi                      cutoff; c/2a for TE10
    beta  = sqrt(k^2 - kc^2)                   phase constant

It also exercises the parts of the adapter the microstrip never touches: an
**enclosed** domain - PEC on the four guide walls, absorber only on the two
ends, so the mesh must not grow sideways or the walls move and the cutoff shifts
- and a port type that lays down no conductor of its own.

What is *not* asserted, and why
-------------------------------

Not the impedance. ``RectWGPort`` computes its reference impedance analytically
from the mode and the guide dimensions rather than measuring it
(``ports.py``: ``self.ZL = k * Z0 / self.beta``, then ``self.Z_ref = self.ZL``).
Comparing that against the closed form would be comparing the closed form
against itself. What the simulation genuinely measures is the phase of the wave
that crossed the guide, and how much of the power came back.

The phase test
--------------

``S21`` between the two measurement planes is ``exp(-j*beta*L)``. Rather than
assert the phase directly - which would need the mode normalisation and every
constant offset to be exactly right - the unwrapped phase is fitted against the
*analytic* beta across the band. The slope recovers the plane separation, which
is known from the geometry. That is insensitive to any constant offset and stays
a test of the physics: if the simulated guide had a different cutoff, beta would
be wrong, and the recovered length would miss.

Geometry and mesh are openEMS' own ``Rect_Waveguide.py`` tutorial: WR-42 over
20-26 GHz, a band that is single-mode by construction (TE10 cuts off at
14.0 GHz, TE20 at 28.0 GHz).

Every assertion runs against **both drawings** of that guide - broad wall on x,
then on y. See the ``orientation`` fixture: turning the drawing must not turn the
physics.
"""

from __future__ import annotations

import numpy as np
import pytest

from Microwave.Results.sparameters import SParameters
from Microwave.Solvers.openems import plan, preflight, read, residual, run, write
from Microwave.Solvers.openems.model import (
    THROUGH,
    Frequency,
    Material,
    Port,
    Problem,
    Solid,
    Termination,
)
from Microwave.Solvers.openems.regions import MeshParams
from tests.analytic import reference

pytestmark = pytest.mark.slow

# WR-42, in mm. Standard band 18-26.5 GHz.
GUIDE_BROAD = 10.7  # a
GUIDE_NARROW = 4.3  # b
GUIDE_LENGTH = 50.0

MODE = "TE10"

FREQ_MIN = 20e9
FREQ_MAX = 26e9

# lambda/30 in vacuum at the top of the band. Free space rather than guide
# wavelength: the guide wavelength is the longer of the two, so this is the
# conservative choice.
RESOLUTION = reference.SPEED_OF_LIGHT / FREQ_MAX * 1e3 / 30

# Port planes, following the tutorial: excitation at 10 cells in from each end,
# measurement plane 5 cells further along. Both ports face inward.
PORT_INSET = 10 * RESOLUTION
PORT_DEPTH = 5 * RESOLUTION

#: Distance between the two measurement planes, in metres. The probe plane sits
#: at the port box's ``stop`` (``ports.py``: ``m_start[exc_ny] = m_stop[exc_ny]``),
#: so this is what the phase fit must recover.
PLANE_SEPARATION = (GUIDE_LENGTH - 2 * (PORT_INSET + PORT_DEPTH)) * 1e-3

TIMESTEPS = 12000

#: Numerical dispersion at 30 cells per wavelength is of order (pi/N)^2/6, about
#: 0.2%. This is set from that, not from what the run happened to produce.
LENGTH_TOLERANCE = 0.01

#: A hollow PEC guide in vacuum is lossless, so all the power that does not come
#: back must go through; the only slack is absorber leakage at the two ends, and
#: the gate prints what that comes to. Set one decade above it: a regression
#: guard with headroom, not a claim about how exact the physics is.
#:
#: One decade clear, not two. A bound two decades above the leakage guards
#: nothing - the absorber could stop working almost entirely and this would
#: still pass.
POWER_TOLERANCE = 5e-4

#: One decade clear of the match the gate reports, for the same reason.
MATCH_TOLERANCE = 0.02


def section(broad_axis: int) -> tuple[float, float]:
    """The guide's transverse extents, ``(x, y)``, with the broad wall placed."""
    extents = [0.0, 0.0]
    extents[broad_axis] = GUIDE_BROAD
    extents[1 - broad_axis] = GUIDE_NARROW
    return extents[0], extents[1]


def _problem(broad_axis: int) -> Problem:
    width, height = section(broad_axis)

    # openEMS needs no box for vacuum, but the mesher needs to know how far the
    # problem extends, and the walls are boundary conditions rather than
    # geometry. An epsilon = 1 fill states the extent and changes no physics -
    # and is where a dielectric-filled guide would put its material.
    materials = (Material(name="Air", kind="dielectric", epsilon=1.0),)
    solids = (
        Solid(
            material="Air",
            lower=(0.0, 0.0, 0.0),
            upper=(width, height, GUIDE_LENGTH),
            priority=0,
            label="Guide",
        ),
    )
    ports = (
        Port(
            number=1,
            kind="rect_waveguide",
            mode=MODE,
            start=(0.0, 0.0, PORT_INSET),
            stop=(width, height, PORT_INSET + PORT_DEPTH),
            propagation_axis=2,
            excite=True,
            label="Port 1",
        ),
        Port(
            number=2,
            kind="rect_waveguide",
            mode=MODE,
            start=(0.0, 0.0, GUIDE_LENGTH - PORT_INSET),
            stop=(width, height, GUIDE_LENGTH - PORT_INSET - PORT_DEPTH),
            propagation_axis=2,
            label="Port 2",
        ),
    )

    params = MeshParams(
        metal_res=RESOLUTION,
        dielectric_res=RESOLUTION,
        max_ratio=(1.4, 1.4, 1.4),
        min_lines=10,
        # Absorber on the ends only. The side walls are PEC boundaries, and
        # growing the mesh sideways would move them.
        pml_cells=(0, 0, 8),
    )
    grid = plan.plan_grid(
        solids,
        ports,
        materials,
        params,
        # The domain *is* the guide across the section; along it, the guide runs
        # out through the absorber so the line never sees an end.
        padding=((0, 0), (0, 0), (THROUGH, THROUGH)),
    )

    return Problem(
        title=f"WR-42 rectangular waveguide, broad wall on {'xy'[broad_axis]}",
        frequency=Frequency(start=FREQ_MIN, stop=FREQ_MAX, points=201),
        grid=grid,
        materials=materials,
        solids=solids,
        ports=ports,
        boundary=("PEC", "PEC", "PEC", "PEC", "PML_8", "PML_8"),
        termination=Termination(max_timesteps=TIMESTEPS, end_criteria=0.0),
    )


@pytest.fixture(scope="module", params=[0, 1], ids=["broad-on-x", "broad-on-y"])
def orientation(request) -> int:
    """Which transverse axis the broad wall lies along.

    Every assertion below runs against both drawings of the same guide, because
    openEMS binds ``a`` to the first transverse axis *positionally* while the
    document names the mode after the broad wall. Handing it the pair sorted
    instead - broad wall first, whatever it is bound to - launches a field that
    is not a mode of the guide: the second drawing then reflects everything,
    transmits nothing, and fits a plane separation with the wrong sign.

    The cutoff and the reference impedance are analytic in ``a`` and ``b`` alone,
    so both looked correct throughout. Only a solve catches this, which is why it
    is asserted here and not in the fast suite.
    """
    return request.param


@pytest.fixture(scope="module")
def problem(orientation) -> Problem:
    return _problem(orientation)


@pytest.fixture(scope="module")
def solved(problem, orientation, interpreter, tmp_path_factory):
    directory = tmp_path_factory.mktemp(f"waveguide-broad-{'xy'[orientation]}")
    envelope = write.write(problem, directory)
    preflight.refuse_if_blocked(preflight.check(problem))
    run.run(envelope, interpreter=interpreter)
    return read.read(directory)


def test_the_band_is_single_mode():
    """Fast, exact, and no solver: the band must carry TE10 and nothing else.

    If it did not, S21 would mix modes travelling at different phase constants
    and the fit below would be meaningless rather than merely wrong.
    """
    a, b = GUIDE_BROAD * 1e-3, GUIDE_NARROW * 1e-3
    te10 = reference.cutoff_frequency(a, b, 1, 0)
    te20 = reference.cutoff_frequency(a, b, 2, 0)
    te01 = reference.cutoff_frequency(a, b, 0, 1)

    assert te10 == pytest.approx(reference.SPEED_OF_LIGHT / (2 * a))
    assert te10 < FREQ_MIN, f"TE10 cuts off at {te10 / 1e9:.2f} GHz, inside the band"
    assert min(te20, te01) > FREQ_MAX, "a second mode propagates in this band"


def test_the_absorber_is_only_on_the_ends(problem, orientation):
    """The enclosed case, guarded without a solver.

    Growing the mesh sideways would put the PEC wall somewhere other than the
    guide wall, and every frequency in this file depends on where that wall is.
    """
    width, height = section(orientation)
    assert problem.grid.params["pml_cells"] == [0, 0, 8]
    assert problem.grid.x[0] == pytest.approx(0.0)
    assert problem.grid.x[-1] == pytest.approx(width)
    assert problem.grid.y[0] == pytest.approx(0.0)
    assert problem.grid.y[-1] == pytest.approx(height)


def test_phase_constant_matches_theory(solved, orientation):
    """The gate: the wave crossing the guide has the phase constant theory says.

    Fitted rather than compared pointwise, so a constant offset in the mode
    normalisation cannot pass or fail it - only the physics can.
    """
    # Theory takes the walls, not the axes they were drawn on.
    frequency = solved.frequency
    beta = np.real(
        reference.phase_constant(frequency, GUIDE_BROAD * 1e-3, GUIDE_NARROW * 1e-3, 1, 0)
    )
    phase = np.unwrap(np.angle(solved.s(2)))

    slope, _ = np.polyfit(beta, phase, 1)
    recovered = -slope
    error = (recovered - PLANE_SEPARATION) / PLANE_SEPARATION

    # Printed for the reason given in test_acceptance_microstrip's gate: the
    # tolerance is the right bar for the assertion and far looser than the
    # agreement, so a pass cannot tell you the number has not moved.
    print(
        f"\nGATE WR-42 plane separation = {recovered * 1e3:.4f} mm, geometry "
        f"{PLANE_SEPARATION * 1e3:.4f} mm, {error * 100:+.4f}% "
        f"[broad wall on {'xy'[orientation]}]"
    )
    assert abs(error) < LENGTH_TOLERANCE, (
        f"the phase of S21 implies the measurement planes are "
        f"{recovered * 1e3:.3f} mm apart, but the geometry puts them "
        f"{PLANE_SEPARATION * 1e3:.3f} mm apart ({error * 100:+.2f}%). Either "
        f"the guide's phase constant is wrong or its cutoff is."
    )


def test_the_guide_is_matched(solved):
    """A uniform guide running out through the absorber must barely reflect."""
    worst = float(np.max(np.abs(solved.s(1))))
    assert worst < MATCH_TOLERANCE, (
        f"|S11| reaches {worst:.4f} ({20 * np.log10(worst):.1f} dB); the guide "
        "is reflecting, so it is not behaving as an infinite line"
    )


def test_power_is_conserved(solved, orientation):
    """PEC walls and vacuum: whatever does not come back must go through.

    Exact, and independent of the phase test - it would catch a mesh that
    absorbed energy into a staircased wall while leaving the phase intact.
    """
    total = np.abs(solved.s(1)) ** 2 + np.abs(solved.s(2)) ** 2
    worst = float(np.max(np.abs(total - 1.0)))
    print(
        f"\nGATE WR-42 worst power departure = {worst:.3e}, "
        f"bound {POWER_TOLERANCE:.0e} [broad wall on {'xy'[orientation]}]"
    )
    assert worst < POWER_TOLERANCE, (
        f"|S11|^2 + |S21|^2 departs from unity by {worst:.4f} in a lossless "
        "guide; energy is going somewhere the model does not describe"
    )


def test_the_response_had_finished_when_the_run_stopped(solved, orientation):
    """Every figure above is a transform of these records, so a truncated one
    would move all of them together and none of the bounds would say why.

    A ``RectWGPort`` reads its record through a different class from the
    microstrip and the lumped gates, which is the other reason this is asserted
    once per port kind rather than once.
    """
    worst = max(solved.tail_share.values())
    print(
        f"\nGATE WR-42 worst tail share = {worst:.2e}, "
        f"bound {residual.WANTED:.0e} [broad wall on {'xy'[orientation]}]"
    )
    assert residual.unfinished(solved.tail_share) is None


class TestWhatTheGuideIsReportedAgainst:
    """One solve, two references, and only one of them is a picture of the
    guide.

    A guide has no reference impedance to type in. At 23 GHz WR-42 is 475.0 ohm
    by wave impedance, 381.8 by power-voltage and 471.0 by power-current, and
    only the ratios a reference cancels out of are convention-free - which is
    why a bench references a guide to the line standard's *own* impedance.

    Reported that way this guide is matched, because it is. Reported against a
    constant it is not, and the arithmetic saying so is correct: a uniform guide
    running out through an absorber presents exactly its own impedance, so the
    reflection a 50-ohm system sees is the mismatch closed form. That is the
    second assertion, and it is what makes the first one mean something - the
    two together say the plot changed for a reason that can be written down.
    """

    def matrix(self, solved, reference):
        return SParameters.from_runs([solved], reference=reference)

    def test_against_its_own_impedance_nothing_is_renormalised_at_all(self, solved):
        """Exactly the raw wave decomposition, bit for bit.

        Not "within MATCH_TOLERANCE", which ``test_the_guide_is_matched``
        already asserts of the same numbers and which a renormalisation to the
        guide's own impedance would pass without doing nothing. The property is
        that the pipeline left them alone, and that has no tolerance.
        """
        against_itself = self.matrix(solved, None)

        np.testing.assert_array_equal(against_itself.parameter(1, 1), solved.s(1))
        assert against_itself.self_referenced == (1, 2)

    def test_against_a_constant_it_reads_as_the_mismatch_it_is(self, solved, orientation):
        """|S11| = |(Z - R)/(Z + R)|, per frequency, with Z the guide's own.

        The same closed form the lumped-termination gate uses, and the bound is
        the residual reflection of the guide itself: the identity holds exactly
        for a line that reflects nothing, so how far the guide misses it is how
        far the guide misses being infinite.
        """
        against_fifty = self.matrix(solved, 50.0)
        z = against_fifty.impedance(1)
        expected = np.abs((z - 50.0) / (z + 50.0))
        measured = np.abs(against_fifty.parameter(1, 1))

        # From the raw run, not from a second assembly: the bound has to come
        # from outside the code being gated, or breaking the reference would
        # widen the bound by exactly as much as it moved the answer.
        slack = float(np.max(np.abs(solved.s(1))))
        worst = float(np.max(np.abs(measured - expected)))
        print(
            f"\nGATE WR-42 |S11| at 50 ohm = {measured.max():.4f} "
            f"({20 * np.log10(measured.max()):.1f} dB), against the mismatch "
            f"closed form to {worst:.2e}, slack {slack:.2e} "
            f"[broad wall on {'xy'[orientation]}]"
        )
        assert worst < slack, (
            f"|S11| referenced to 50 ohm misses |(Z - R)/(Z + R)| by {worst:.3e}, "
            f"more than the {slack:.3e} the guide's own residual reflection "
            "allows; the renormalisation is not doing what the mismatch predicts"
        )


def test_the_run_is_reproducible(solved, problem):
    assert solved.reproducible
    assert solved.matches(problem.digest())
    assert solved.provenance["cells"] == problem.grid.cell_count
