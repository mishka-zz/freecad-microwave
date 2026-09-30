# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the de-embedding answers, on runs built from a network and a line whose
values are known.

Every run here is arithmetic rather than a solve, so what comes back out is
held to what went in, to the rounding of the arithmetic. The limits
``tests/deembed.py`` states - a transformer, the order of the network, a network
per port - are built as well, and each one is held to doing what the module says
it does.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests import deembed

#: The reference the runs are stated against, in ohms.
REFERENCE = 50.0

#: A line near the reference and not on it, so a fit returning the reference
#: has not found the line.
IMPEDANCE = 49.8

#: Two electrical lengths, in radians, neither near a whole number of half
#: wavelengths.
LENGTHS = (0.7, 1.4)

#: A port's network of an inductance and a capacitance at a frequency, in ohms
#: and siemens, and one with loss in both.
ANGULAR = 2 * math.pi * 8e9
REACTIVE = (1j * ANGULAR * 26e-12, 1j * ANGULAR * 1e-15)
LOSSY = (0.02 + 1j * ANGULAR * 26e-12, 1e-6 + 1j * ANGULAR * 1e-15)

#: How closely arithmetic that went in has to come back out.
ROUNDING = 1e-9


def runs(series, shunt, topology, impedance=IMPEDANCE, lengths=LENGTHS, inside=None):
    """Each length's chain matrix, through the network at both ends.

    :param inside: a two-port standing between the network and the line at both
        ends, turned round at the far one.
    """
    x = deembed.network(series, shunt, topology)
    if inside is not None:
        x = x @ inside
    return [x @ deembed.line(impedance, theta) @ deembed.reverse(x) for theta in lengths]


@pytest.mark.parametrize("topology", deembed.TOPOLOGIES)
@pytest.mark.parametrize("network", [REACTIVE, LOSSY], ids=["reactive", "lossy"])
def test_the_line_and_the_network_come_back_out(topology, network):
    found = deembed.fit(runs(*network, topology), LENGTHS, REFERENCE, topology)
    assert found.impedance == pytest.approx(IMPEDANCE, rel=ROUNDING, abs=0.0)
    assert found.series == pytest.approx(network[0], rel=ROUNDING, abs=0.0)
    assert found.shunt == pytest.approx(network[1], rel=ROUNDING, abs=0.0)
    assert found.residual < ROUNDING


def test_the_network_is_read_as_an_inductance_and_a_capacitance():
    found = deembed.fit(
        runs(*REACTIVE, deembed.SERIES_FIRST), LENGTHS, REFERENCE, deembed.SERIES_FIRST
    )
    assert found.inductance(ANGULAR) == pytest.approx(26e-12, rel=ROUNDING, abs=0.0)
    assert found.capacitance(ANGULAR) == pytest.approx(1e-15, rel=ROUNDING, abs=0.0)


def test_a_line_that_is_not_the_drawing_leaves_a_residual():
    """The electrical length is given rather than fitted, so a line longer than
    drawn is not taken up as a length."""
    longer = [theta * 1.01 for theta in LENGTHS]
    measured = runs(*REACTIVE, deembed.SERIES_FIRST, lengths=longer)
    found = deembed.fit(measured, LENGTHS, REFERENCE, deembed.SERIES_FIRST)
    exact = deembed.fit(measured, longer, REFERENCE, deembed.SERIES_FIRST)
    assert exact.residual < ROUNDING
    assert found.residual > 1e3 * ROUNDING


@pytest.mark.parametrize("ratio", [0.9, 1.1])
def test_a_transformer_in_the_network_is_taken_into_the_impedance(ratio):
    """The one thing two lengths cannot separate, and the reason the impedance
    is conditional on the network holding no transformer."""
    measured = runs(*REACTIVE, deembed.SERIES_FIRST, inside=deembed.transformer(ratio))
    found = deembed.fit(measured, LENGTHS, REFERENCE, deembed.SERIES_FIRST)
    assert found.residual < ROUNDING
    assert found.impedance == pytest.approx(ratio**2 * IMPEDANCE, rel=ROUNDING, abs=0.0)


def test_the_two_orders_differ_by_that_transformer():
    """Fitted in the other order, a reactive network's runs are reproduced as
    well, and the impedance moves by ``(1 + Z Y)**2``."""
    series, shunt = REACTIVE
    measured = runs(series, shunt, deembed.SERIES_FIRST)
    found = deembed.fit(measured, LENGTHS, REFERENCE, deembed.SHUNT_FIRST)
    assert found.residual < ROUNDING
    assert found.impedance == pytest.approx(
        IMPEDANCE * abs(1 + series * shunt) ** 2, rel=ROUNDING, abs=0.0
    )
    assert found.series == pytest.approx(series * (1 + series * shunt), rel=ROUNDING, abs=0.0)
    assert found.shunt == pytest.approx(shunt / (1 + series * shunt), rel=ROUNDING, abs=0.0)


def test_and_a_lossy_network_is_not_reproduced_in_the_other_order():
    """Where ``Z Y`` is not real the ratio is not a real transformer, and the
    fit in the other order leaves a residual."""
    measured = runs(*LOSSY, deembed.SERIES_FIRST)
    found = deembed.fit(measured, LENGTHS, REFERENCE, deembed.SHUNT_FIRST)
    assert found.residual > 1e3 * ROUNDING


def moved(inductance, capacitance, topology=deembed.SERIES_FIRST):
    """Each length's chain matrix with a network of each port's own."""
    near = deembed.network(1j * ANGULAR * inductance[0], 1j * ANGULAR * capacitance[0], topology)
    far = deembed.network(1j * ANGULAR * inductance[1], 1j * ANGULAR * capacitance[1], topology)
    return [near @ deembed.line(IMPEDANCE, theta) @ deembed.reverse(far) for theta in LENGTHS]


def departure(inductance, capacitance):
    return max(
        float(np.max(np.abs(one - other)))
        for one, other in zip(moved(inductance, capacitance), moved((0.0, 0.0), (0.0, 0.0)))
    )


def test_a_network_moved_from_one_port_to_the_other_reaches_no_run_at_first_order():
    """Why one network serves both ports: an inductance moved from one port to
    the other with ``L / Zc**2`` of capacitance moved the same way moves the runs
    as the square of what was moved, where the same inductance added at both
    ports moves them in proportion to it."""
    exchanged = [
        departure((step, -step), (step / IMPEDANCE**2, -step / IMPEDANCE**2))
        for step in (1e-12, 2e-12)
    ]
    added = [departure((step, step), (0.0, 0.0)) for step in (1e-12, 2e-12)]
    assert exchanged[1] / exchanged[0] == pytest.approx(4.0, rel=1e-3)
    assert added[1] / added[0] == pytest.approx(2.0, rel=1e-3)
    assert exchanged[0] < 1e-2 * added[0]


def test_but_a_difference_between_the_ports_reaches_the_two_reflections():
    """Capacitance moved the other way is a difference the runs do see, and it
    is where ``S11`` and ``S22`` part."""
    step = 1e-12
    for measured in moved((step, -step), (-step / IMPEDANCE**2, step / IMPEDANCE**2)):
        s = deembed.scattering(measured, REFERENCE)
        assert abs(s[0, 0] - s[1, 1]) > 1e-4


@pytest.mark.parametrize(
    "network", [REACTIVE, (5.0 + 100.0j, 2e-3 + 5e-2j)], ids=["small", "large"]
)
def test_the_phase_across_two_lengths_is_the_lines_whatever_the_network(network):
    shorter, longer = runs(*network, deembed.SERIES_FIRST)
    assert deembed.phase_across(shorter, longer) == pytest.approx(
        LENGTHS[1] - LENGTHS[0], rel=ROUNDING, abs=0.0
    )


def test_a_phase_past_half_a_turn_comes_back_folded():
    shorter, longer = runs(*REACTIVE, deembed.SERIES_FIRST, lengths=(0.5, 0.5 + math.pi + 0.3))
    assert deembed.phase_across(shorter, longer) == pytest.approx(math.pi - 0.3, rel=ROUNDING)


def test_a_chain_matrix_and_its_scattering_matrix_are_one_two_port():
    """A matched line reflects nothing and transmits its own phase, and the two
    conversions undo each other."""
    s = deembed.scattering(deembed.line(REFERENCE, 0.7), REFERENCE)
    assert abs(s[0, 0]) < ROUNDING and abs(s[1, 1]) < ROUNDING
    assert s[1, 0] == pytest.approx(np.exp(-0.7j), rel=ROUNDING, abs=0.0)
    (measured,) = runs(*LOSSY, deembed.SHUNT_FIRST, lengths=(1.1,))
    back = deembed.abcd(deembed.scattering(measured, REFERENCE), REFERENCE)
    np.testing.assert_allclose(back, measured, rtol=ROUNDING, atol=0.0)


def test_one_run_is_refused():
    (measured,) = runs(*REACTIVE, deembed.SERIES_FIRST, lengths=(0.7,))
    with pytest.raises(ValueError, match="two runs at least"):
        deembed.fit([measured], [0.7], REFERENCE, deembed.SERIES_FIRST)


def test_an_order_it_does_not_know_is_refused():
    with pytest.raises(ValueError, match="no topology"):
        deembed.network(1.0, 1.0, "sideways")


def test_the_standard_errors_are_what_the_figures_scatter_by():
    """Runs with noise of one size added, fitted over and over: the standard
    error each fit states is the spread the fitted figures actually have."""
    generator = np.random.default_rng(7)
    series, shunt = REACTIVE
    clean = runs(series, shunt, deembed.SERIES_FIRST)
    fits = []
    for _ in range(400):
        noisy = []
        for measured in clean:
            s = deembed.scattering(measured, REFERENCE)
            noise = 1e-4 * (
                generator.standard_normal((2, 2)) + 1j * generator.standard_normal((2, 2))
            )
            noisy.append(deembed.abcd(s + (noise + noise.T) / 2, REFERENCE))
        fits.append(deembed.fit(noisy, LENGTHS, REFERENCE, deembed.SERIES_FIRST))
    for spread, stated in (
        (np.std([one.impedance for one in fits]), np.mean([one.impedance_error for one in fits])),
        (np.std([one.series.imag for one in fits]), np.mean([one.reactance_error for one in fits])),
        (
            np.std([one.shunt.imag for one in fits]),
            np.mean([one.susceptance_error for one in fits]),
        ),
    ):
        assert stated == pytest.approx(spread, rel=0.5)


def test_exact_runs_leave_no_standard_error():
    found = deembed.fit(
        runs(*REACTIVE, deembed.SERIES_FIRST), LENGTHS, REFERENCE, deembed.SERIES_FIRST
    )
    assert found.impedance_error < ROUNDING * IMPEDANCE


def test_a_line_in_a_dielectric_is_fitted_at_the_wavenumber_of_its_fill():
    """At the vacuum wavenumber the line's length is the wrong electrical length,
    and the fit leaves a residual and a wrong impedance."""
    frequency, permittivity = 3e9, 4.4
    k = deembed.wavenumber(frequency, permittivity, 1.0)
    assert k == pytest.approx(
        2 * math.pi * frequency / 299_792_458.0 * math.sqrt(permittivity), rel=ROUNDING, abs=0.0
    )
    lengths = [k * 8e-3, k * 16e-3]
    measured = runs(*REACTIVE, deembed.SERIES_FIRST, lengths=lengths)
    found = deembed.fit(measured, lengths, REFERENCE, deembed.SERIES_FIRST)
    assert found.impedance == pytest.approx(IMPEDANCE, rel=ROUNDING, abs=0.0)
    vacuum = deembed.wavenumber(frequency, 1.0, 1.0)
    wrong = deembed.fit(measured, [vacuum * 8e-3, vacuum * 16e-3], REFERENCE, deembed.SERIES_FIRST)
    assert wrong.residual > 1e3 * ROUNDING


def test_unequal_references_convert_both_ways():
    """A reciprocal two-port stated against a different reference at each port
    comes back to the same chain matrix, and its matrix stays symmetric."""
    (measured,) = runs(*LOSSY, deembed.SHUNT_FIRST, lengths=(1.1,))
    s = deembed.scattering(measured, (25.0, 100.0))
    assert s[0, 1] == pytest.approx(s[1, 0], rel=ROUNDING, abs=0.0)
    np.testing.assert_allclose(deembed.abcd(s, (25.0, 100.0)), measured, rtol=ROUNDING, atol=0.0)
    assert abs(s[0, 0] - deembed.scattering(measured, 50.0)[0, 0]) > 0.1


def test_a_reference_that_is_not_a_positive_number_is_refused():
    with pytest.raises(ValueError, match="positive number"):
        deembed.abcd(np.eye(2) * 0.1, (50.0, 0.0))
