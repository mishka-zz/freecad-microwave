# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The neutral S-parameter layer, gated against closed forms.

The load-bearing test here is :class:`TestTheWaveConventionIsCorrected`. Every
other check is bookkeeping; that one is physics, and getting it wrong produces
numbers that look entirely plausible.
"""

import re

import numpy as np
import pytest

from Microwave.Results.sparameters import (
    MIRROR,
    ResultError,
    SParameters,
    describe_reference,
)


class Run:
    """Stands in for ``Solvers.openems.read.Results``.

    A stub rather than a real solve because the physics being tested is what
    happens *after* the solver: the same arithmetic must hold whatever produced
    the wave amplitudes, and a real FDTD run would only add noise to a
    comparison against an exact closed form.
    """

    def __init__(self, frequency, excited, z_ref, volts, digest="d", reproducible=True):
        self.frequency = np.asarray(frequency, dtype=float)
        self.excited_port = excited
        self._z = {number: np.asarray(z, dtype=complex) for number, z in z_ref.items()}
        self._volts = volts
        self.provenance = {"envelope_digest": digest, "title": "stub", "cells": 1000}
        self.reproducible = reproducible
        #: What the excited port put in. A real run that put in nothing still
        #: returns a full-length array of zeros, which is the case worth being
        #: able to build here.
        self.incident = np.ones_like(self.frequency, dtype=complex)

    @property
    def ports(self):
        return dict.fromkeys(self._z)

    def port(self, number):
        return type("P", (), {"z0": self._z[number], "incident": self.incident})()

    def s(self, receiving, driving):
        return np.asarray(self._volts[(receiving, driving)], dtype=complex)


def series_resistor(z01, z02, r, frequency=(1e9, 2e9)):
    """Two runs describing a series resistor between mismatched ports.

    Built by taking the exact power-wave answer and converting *back* into the
    raw voltage ratios openEMS would report, so the test drives the code with
    what the driver actually writes.
    """
    frequency = np.asarray(frequency, dtype=float)
    ones = np.ones_like(frequency)
    den = z01 + z02 + r
    s11, s21, s22 = (z02 + r - z01) / den, 2 * np.sqrt(z01 * z02) / den, (z01 + r - z02) / den

    volts = {
        (1, 1): s11 * ones,
        (2, 1): s21 * np.sqrt(z02 / z01) * ones,
        (1, 2): s21 * np.sqrt(z01 / z02) * ones,
        (2, 2): s22 * ones,
    }
    z = {1: z01 * ones, 2: z02 * ones}
    return [
        Run(frequency, 1, z, {k: v for k, v in volts.items() if k[1] == 1}),
        Run(frequency, 2, z, {k: v for k, v in volts.items() if k[1] == 2}),
    ]


def from_z_matrix(z_matrix, zr, frequency=(1e9, 2e9)):
    """Runs describing an arbitrary two-port, built from its Z-matrix.

    Definition-free, and the only way to get a network that is *not* mirror
    symmetric: a series resistor is its own mirror image whatever impedances it
    sits between, so it can never be the negative case.
    """
    frequency = np.asarray(frequency, dtype=float)
    ones = np.ones_like(frequency)
    z_matrix = np.asarray(z_matrix, dtype=complex)
    zr = np.asarray(zr, dtype=complex)

    raw = (z_matrix - np.diag(zr)) @ np.linalg.inv(z_matrix + np.diag(zr))
    volts = {(i + 1, j + 1): raw[i, j] * ones for i in range(2) for j in range(2)}
    z = {1: zr[0] * ones, 2: zr[1] * ones}
    return [
        Run(frequency, 1, z, {k: v for k, v in volts.items() if k[1] == 1}),
        Run(frequency, 2, z, {k: v for k, v in volts.items() if k[1] == 2}),
    ]


def three_port_run(frequency=(1e9, 2e9)):
    """One run of a three-port, for the cases mirror symmetry cannot serve."""
    frequency = np.asarray(frequency, dtype=float)
    ones = np.ones_like(frequency)
    z = {n: 50.0 * ones for n in (1, 2, 3)}
    volts = {(n, 1): 0.1 * n * ones for n in (1, 2, 3)}
    return [Run(frequency, 1, z, volts)]


class TestTheWaveConventionIsCorrected:
    def test_a_series_resistor_between_mismatched_ports(self):
        """The gate. openEMS reports volts; S-parameters are normalised.

        A series resistor R between reference impedances Z01 and Z02 has, at a
        common 50 ohm reference, S11 = R/(2*50+R) and S21 = 100/(2*50+R) - no
        approximation, no fitting constants. Tolerance is 1e-6 relative because
        scikit-rf renormalises through Z-parameters, which its own source warns
        costs accuracy; the measured error is about 3e-8.
        """
        z01, z02, r = 30.0, 75.0, 20.0
        result = SParameters.from_runs(series_resistor(z01, z02, r), reference=50.0)

        expected_s11 = r / (2 * 50.0 + r)
        expected_s21 = 2 * 50.0 / (2 * 50.0 + r)

        assert result.parameter(1, 1).real == pytest.approx(expected_s11, rel=1e-6, abs=0.0)
        assert result.parameter(2, 1).real == pytest.approx(expected_s21, rel=1e-6, abs=0.0)
        assert result.parameter(1, 2).real == pytest.approx(expected_s21, rel=1e-6, abs=0.0)

    def test_skipping_the_correction_is_detectably_wrong(self, monkeypatch):
        """The same case with the normalisation disabled, through real code.

        Computing both answers from local floats and comparing them would
        assert arithmetic and exercise nothing - no mutation of
        ``sparameters.py`` could make it fail. Stubbing ``_normalisation`` to 1
        and driving ``from_runs`` instead fails if the correction ever stops
        being applied at all.
        """
        from Microwave.Results import sparameters

        z01, z02, r = 30.0, 75.0, 20.0
        runs = series_resistor(z01, z02, r)
        correct = SParameters.from_runs(runs, reference=50.0).parameter(2, 1)

        monkeypatch.setattr(sparameters, "_normalisation", lambda z: np.ones_like(z.real))
        uncorrected = SParameters.from_runs(runs, reference=50.0).parameter(2, 1)

        error = np.max(np.abs(uncorrected - correct)) / np.max(np.abs(correct))
        assert error > 0.5, f"disabling the correction only moved S21 by {error:.1%}"

    def test_a_complex_reference_impedance(self):
        """The case the whole 'pseudo' choice exists for, and the only one that
        distinguishes it from 'power' and 'traveling'.

        A microstrip's Z_ref is complex and frequency-dependent. Every other
        test here uses real impedances, where all three wave definitions agree
        and scikit-rf's own conversion is a documented no-op - so without this
        test, swapping ``s_def="pseudo"`` for either alternative passes the
        suite while putting 24% error into a microstrip result.

        Built from a Z-matrix, which is definition-free: the raw ratio openEMS
        would report is M = (Z - Zr)(Z + Zr)^-1, and the answer at a real 50 ohm
        is (Z - 50)(Z + 50)^-1, with no wave convention anywhere in either.
        """
        frequency = np.array([1e9, 2e9])
        zr = np.array([45.0 - 8.0j, 90.0 + 15.0j])
        z_matrix = np.array([[60.0 + 10.0j, 25.0 - 5.0j], [25.0 - 5.0j, 80.0 - 20.0j]])

        identity = np.eye(2)
        raw = (z_matrix - np.diag(zr)) @ np.linalg.inv(z_matrix + np.diag(zr))
        expected = (z_matrix - 50.0 * identity) @ np.linalg.inv(z_matrix + 50.0 * identity)

        ones = np.ones_like(frequency)
        z = {1: zr[0] * ones, 2: zr[1] * ones}
        volts = {(i + 1, j + 1): raw[i, j] * ones for i in range(2) for j in range(2)}
        runs = [
            Run(frequency, 1, z, {k: v for k, v in volts.items() if k[1] == 1}),
            Run(frequency, 2, z, {k: v for k, v in volts.items() if k[1] == 2}),
        ]

        result = SParameters.from_runs(runs, reference=50.0)
        assert np.allclose(result.s[0], expected, rtol=0, atol=1e-9), (
            f"complex Z_ref mishandled:\n{result.s[0]}\nexpected\n{expected}"
        )

    def test_matched_ports_need_no_correction(self):
        """With equal reference impedances the factor is exactly 1.

        Worth asserting because it is why the existing single-port acceptance
        gates never caught this: on a symmetric microstrip the bug is invisible.
        """
        result = SParameters.from_runs(series_resistor(50.0, 50.0, 20.0), reference=50.0)
        expected = 2 * 50.0 / (2 * 50.0 + 20.0)
        assert result.parameter(2, 1).real == pytest.approx(expected, rel=1e-9, abs=0.0)

    def test_a_matched_line_is_passive_and_reciprocal(self):
        """Physical sanity the library gives us for free."""
        result = SParameters.from_runs(series_resistor(50.0, 50.0, 20.0))
        assert result.network().is_passive()
        assert result.network().is_reciprocal()


class TestAPortReferencedToItself:
    """``None`` in place of a number: report this port against its own impedance.

    What a bench does. TRL references a guide to the line standard's own
    characteristic impedance, which is dispersive and typed in nowhere - and
    for a guide there is no number to type, its impedance differing by a quarter
    between the wave, power-voltage and power-current conventions.

    The closed form is the fixture's own: ``series_resistor`` is built from the
    exact power-wave S-matrix *at the two port impedances* and converted back
    into the volts openEMS reports. So referencing both ports to themselves must
    return those numbers, and nothing else does.
    """

    Z01, Z02, R = 30.0, 75.0, 20.0

    def exact(self):
        den = self.Z01 + self.Z02 + self.R
        return (
            (self.Z02 + self.R - self.Z01) / den,
            2 * np.sqrt(self.Z01 * self.Z02) / den,
            (self.Z01 + self.R - self.Z02) / den,
        )

    def test_both_ports_at_their_own_recovers_the_matrix_the_ports_define(self):
        result = SParameters.from_runs(series_resistor(self.Z01, self.Z02, self.R), reference=None)
        s11, s21, s22 = self.exact()

        assert result.parameter(1, 1).real == pytest.approx(s11, rel=1e-9, abs=0.0)
        assert result.parameter(2, 1).real == pytest.approx(s21, rel=1e-9, abs=0.0)
        assert result.parameter(2, 2).real == pytest.approx(s22, rel=1e-9, abs=0.0)
        assert result.parameter(1, 2).real == pytest.approx(s21, rel=1e-9, abs=0.0)

    def test_the_reference_it_reports_is_what_each_port_measured(self):
        runs = series_resistor(self.Z01, self.Z02, self.R)
        result = SParameters.from_runs(runs, reference=None)

        np.testing.assert_array_equal(result.reference, result.measured_impedance)

    def test_it_is_asked_per_port(self):
        """Port 1 to 50 ohm and port 2 to its own 75.

        The same closed form as a one-path measurement, and for the same reason:
        S11 means "reflection at port 1 with every other port terminated in its
        own reference impedance", so a port left where it was measured is part
        of what S11 is rather than a number left unconverted.
        """
        result = SParameters.from_runs(
            series_resistor(self.Z01, self.Z02, self.R), reference=[50.0, None]
        )

        den = 50.0 + self.Z02 + self.R
        assert result.parameter(1, 1).real == pytest.approx(
            (self.Z02 + self.R - 50.0) / den, rel=1e-6, abs=0.0
        )
        assert result.reference[0, 0] == 50.0
        assert result.reference[0, 1] == result.measured_impedance[0, 1]

    def test_a_number_still_moves_the_matrix_off_what_the_ports_measured(self):
        """The negative case, without which every assertion above passes on a
        ``from_runs`` that ignored ``reference`` entirely."""
        own = SParameters.from_runs(series_resistor(self.Z01, self.Z02, self.R), reference=None)
        fixed = SParameters.from_runs(series_resistor(self.Z01, self.Z02, self.R), reference=50.0)

        assert not np.allclose(own.parameter(1, 1), fixed.parameter(1, 1))

    def test_a_mirror_derives_its_column_in_the_one_basis_the_declaration_gives(self):
        """Two ports of a declared mirror are one port, so they have one
        impedance - and the copy S22 = S11 is only exact if the answer is
        reported in the basis the copy was taken in. Referencing each port to
        its own *measurement* instead would move the two diagonal terms by
        different amounts and undo it.
        """
        runs = series_resistor(50.0, 50.4, 20.0)
        derived = SParameters.from_runs(runs[:1], reference=None, symmetry=MIRROR)

        np.testing.assert_array_equal(derived.reference[:, 0], derived.reference[:, 1])
        np.testing.assert_allclose(derived.parameter(2, 2), derived.parameter(1, 1), rtol=0, atol=0)


class TestAOnePathMeasurement:
    """Driving one port of a two-port: S11 and S21, exactly, and nothing faked.

    The overwhelmingly common study, and the same thing a one-path VNA gives
    you - a LiteVNA has one source and two receivers, so S12 and S22 are not
    measured rather than badly measured.

    Gated against closed forms, and the closed form is the interesting part.
    **S11 means "reflection at port 1 with every other port terminated in its
    own reference impedance"**, so leaving port 2 where it was measured is not
    a number left unconverted - it is part of what S11 *is*. Moving port 2 to
    50 ohm asks a different physical question and needs S22 to answer.
    """

    Z01, Z02, R = 30.0, 75.0, 20.0

    def one_path(self, z02=None, reference=50.0):
        runs = series_resistor(self.Z01, self.Z02 if z02 is None else z02, self.R)
        return SParameters.from_runs(runs[:1], reference=reference)

    def test_the_driven_column_is_the_closed_form(self):
        """Port 1 renormalised to 50 ohm, port 2 still loaded with its own 75.

        The same series-resistor formula, with z01 = 50 because that is where
        the driven port was moved to and z02 = 75 because that is where port 2
        physically sits. Asserting the both-ports-at-50 answer here was my first
        guess and it is wrong by 86%: it describes a different termination.
        """
        result = self.one_path()

        den = 50.0 + self.Z02 + self.R
        expected_s11 = (self.Z02 + self.R - 50.0) / den
        expected_s21 = 2 * np.sqrt(50.0 * self.Z02) / den

        assert result.parameter(1, 1).real == pytest.approx(expected_s11, rel=1e-6, abs=0.0)
        assert result.parameter(2, 1).real == pytest.approx(expected_s21, rel=1e-6, abs=0.0)

    def test_it_agrees_with_the_two_run_assembly_term_for_term(self):
        """Half the solve time must not mean a different answer.

        With the undriven port already at the requested reference there is
        nothing to convert there, so the two paths must agree - which isolates
        the claim being made: the driven column comes out identical whether or
        not the other column was ever measured. That is the claim
        ``_wanted_reference`` rests on, and this is where it is held.

        1e-6, not machine precision, and the slack is scikit-rf's: it
        renormalises through Z-parameters, which its own source warns costs
        accuracy, and the two paths ask it to do different amounts of work. The
        measured disagreement is 9.5e-8, and the one-path answer is the more
        accurate of the two - it lands on 1/6 exactly.
        """
        matched = 50.0
        both = SParameters.from_runs(series_resistor(self.Z01, matched, self.R), reference=50.0)
        one = self.one_path(z02=matched)

        for receiving in (1, 2):
            np.testing.assert_allclose(
                one.parameter(receiving, 1),
                both.parameter(receiving, 1),
                rtol=1e-6,
                err_msg=f"S{receiving}1",
            )

    def test_the_termination_at_the_undriven_port_is_what_makes_them_differ(self):
        """Not a defect, and worth pinning so nobody 'fixes' it.

        The two-run matrix moves port 2 to 50 ohm as well, which changes what
        S11 is a reflection *into*. Both numbers are right; they answer
        different questions, and only the two-run one can answer the second.
        """
        both = SParameters.from_runs(series_resistor(self.Z01, self.Z02, self.R), reference=50.0)
        one = self.one_path()

        assert abs(one.parameter(1, 1)[0] - both.parameter(1, 1)[0]) > 0.1

    def test_what_was_not_measured_is_nan_and_not_zero(self):
        """A zero is a number somebody will plot, and -inf dB is a plausible
        looking notch."""
        result = self.one_path()

        assert np.all(np.isnan(result.parameter(1, 2)))
        assert np.all(np.isnan(result.parameter(2, 2)))
        assert np.all(np.isfinite(result.parameter(1, 1)))

    def test_it_says_which_columns_it_has(self):
        result = self.one_path()

        assert result.driven == (1,)
        assert result.unmeasured == (2,)
        assert not result.complete
        assert result.port_numbers == (1, 2)

    def test_a_full_sweep_is_complete(self):
        result = SParameters.from_runs(series_resistor(self.Z01, self.Z02, self.R))

        assert result.driven == (1, 2)
        assert result.unmeasured == ()
        assert result.complete

    def test_the_undriven_port_keeps_its_own_reference(self):
        """Moving it needs S22, which nobody measured. Leaving it where it is
        makes that renormalisation the identity, and the number is recorded so
        the result cannot pretend to be at 50 ohm throughout."""
        result = self.one_path()

        np.testing.assert_allclose(result.reference[:, 0].real, 50.0)
        np.testing.assert_allclose(result.reference[:, 1].real, self.Z02)

    def test_the_impedance_of_every_port_is_still_known(self):
        """Undriven ports are still *receivers*, so their Z_ref is measured in
        the one run there is. Only their column is missing."""
        result = self.one_path()

        np.testing.assert_allclose(result.impedance(2).real, self.Z02)

    def test_a_network_is_refused_by_name(self):
        with pytest.raises(ResultError, match=r"port\(s\) \[2\] were never driven"):
            self.one_path().network()

    def test_a_touchstone_file_is_refused_by_name(self, tmp_path):
        """There is no honest .s2p for a one-path measurement: the file has a
        column for every term and no way to say which were invented."""
        with pytest.raises(ResultError, match="Touchstone"):
            self.one_path().write_touchstone(tmp_path / "partial")

        assert list(tmp_path.iterdir()) == []


class TestMirrorSymmetry:
    """One solve, whole matrix - when the user says the structure is a mirror.

    The saving is real: half the FDTD time for the commonest two-port study. The
    gate is that the derived matrix must equal the *measured* one, so a
    symmetric structure solved twice and solved once give the same answer.

    A symmetric series resistor is the exact case: R between two equal
    impedances is its own mirror image, and every term has a closed form.
    """

    Z0, R = 40.0, 20.0

    def test_one_run_reproduces_the_two_run_matrix(self):
        """The whole claim, term for term."""
        runs = series_resistor(self.Z0, self.Z0, self.R)
        both = SParameters.from_runs(runs, reference=50.0)
        one = SParameters.from_runs(runs[:1], reference=50.0, symmetry=MIRROR)

        np.testing.assert_allclose(one.s, both.s, rtol=1e-6, atol=1e-12)

    def test_the_derived_terms_are_the_closed_form(self):
        """Not merely self-consistent - right.

        A series resistor between two 50 ohm references has S11 = S22 =
        R/(2*50+R) and S21 = S12 = 100/(2*50+R), whichever end you drive.
        """
        one = SParameters.from_runs(
            series_resistor(self.Z0, self.Z0, self.R)[:1],
            reference=50.0,
            symmetry=MIRROR,
        )
        expected_s11 = self.R / (2 * 50.0 + self.R)
        expected_s21 = 2 * 50.0 / (2 * 50.0 + self.R)

        assert one.parameter(2, 2).real == pytest.approx(expected_s11, rel=1e-6, abs=0.0)
        assert one.parameter(1, 2).real == pytest.approx(expected_s21, rel=1e-6, abs=0.0)

    def test_the_completed_matrix_is_referenced_throughout(self):
        """A derived column is a known column, so it gets renormalised with the
        rest - which is the point of completing before renormalising rather
        than after."""
        one = SParameters.from_runs(
            series_resistor(self.Z0, self.Z0, self.R)[:1],
            reference=50.0,
            symmetry=MIRROR,
        )
        np.testing.assert_allclose(one.reference.real, 50.0)
        assert one.complete
        assert one.unmeasured == ()

    def test_derived_columns_are_not_called_measured(self):
        one = SParameters.from_runs(
            series_resistor(self.Z0, self.Z0, self.R)[:1],
            reference=50.0,
            symmetry=MIRROR,
        )
        assert one.driven == (1,)
        assert one.derived == (2,)
        assert one.provenance["symmetry"] == MIRROR
        assert one.provenance["derived_columns"] == [2]

    def test_a_touchstone_file_becomes_possible(self, tmp_path):
        """The refusal was about missing numbers, not about principle."""
        one = SParameters.from_runs(
            series_resistor(self.Z0, self.Z0, self.R)[:1],
            reference=50.0,
            symmetry=MIRROR,
        )
        written = one.write_touchstone(tmp_path / "derived")
        assert written.is_file() and written.suffix == ".s2p"

    def test_the_file_says_which_terms_were_not_measured(self, tmp_path):
        """Otherwise it is the very thing the incomplete-matrix refusal exists
        to prevent: a .s2p whose invented terms look exactly like measured
        ones. Completing the matrix gets it past the refusal, so the file has
        to carry the assumption itself.
        """
        one = SParameters.from_runs(
            series_resistor(self.Z0, self.Z0, self.R)[:1],
            reference=50.0,
            symmetry=MIRROR,
        )
        text = one.write_touchstone(tmp_path / "derived").read_text()

        header = [line for line in text.splitlines() if line.startswith("!")]
        assert any("NOT ALL TERMS WERE MEASURED" in line for line in header)
        assert any("[2]" in line and "mirror" in line for line in header)
        # Still a Touchstone file: the option line must survive, and comments
        # are what "!" means in the format.
        assert "# Hz S RI R 50" in text

    def test_a_fully_measured_file_carries_no_such_header(self, tmp_path):
        """The annotation must mean something, so it has to be absent when
        there is nothing to annotate."""
        both = SParameters.from_runs(series_resistor(self.Z0, self.Z0, self.R), reference=50.0)
        text = both.write_touchstone(tmp_path / "measured").read_text()

        assert "NOT ALL TERMS WERE MEASURED" not in text

    def test_the_file_still_reads_back(self, tmp_path):
        """Comments are legal Touchstone, but a header is only worth writing if
        it does not cost the file its readers."""
        from Microwave.Results import _skrf

        one = SParameters.from_runs(
            series_resistor(self.Z0, self.Z0, self.R)[:1],
            reference=50.0,
            symmetry=MIRROR,
        )
        written = one.write_touchstone(tmp_path / "derived")

        back = _skrf.module().Network(str(written))
        np.testing.assert_allclose(back.s, one.s, rtol=1e-6, atol=1e-9)

    def test_declaring_it_on_a_full_sweep_changes_nothing(self):
        """Both ports driven: there is nothing to derive, and the measurement
        wins over the declaration."""
        runs = series_resistor(self.Z0, self.Z0, self.R)
        plain = SParameters.from_runs(runs, reference=50.0)
        declared = SParameters.from_runs(runs, reference=50.0, symmetry=MIRROR)

        np.testing.assert_array_equal(declared.s, plain.s)
        assert declared.derived == ()

    def test_it_derives_nothing_where_it_has_no_meaning(self):
        """A three-port driven once leaves a 2x2 block unknown, and "mirror"
        does not say which port maps onto which.

        It derives nothing and the partial matrix survives. Raising here was
        the obvious choice and is wrong twice: the runs form a perfectly
        good partial matrix, and the failure arrived minutes after the solve
        having already told the user on Check that nothing would be derived.
        """
        result = SParameters.from_runs(three_port_run(), symmetry=MIRROR, reference=50.0)

        assert result.derived == ()
        assert result.driven == (1,)
        assert result.unmeasured == (2, 3)
        assert np.all(np.isfinite(result.parameter(2, 1)))

    def test_an_unknown_symmetry_is_named(self):
        runs = series_resistor(self.Z0, self.Z0, self.R)
        with pytest.raises(ResultError, match="unknown symmetry"):
            SParameters.from_runs(runs[:1], symmetry="helical")


class TestThePreconditionTheMirrorRestsOn:
    """S22 = S11 needs both ports at one reference, so one is *made*.

    The precondition is real and cannot be handled by assumption: the
    copy was made in the measured basis and renormalisation afterwards moved
    the two diagonal terms by different amounts, undoing it. A lumped port's
    Z_ref is the number the user typed, so two of them agree exactly and the
    acceptance gates - which use lumped ports - read a mismatch of 0.0 and
    could never see it. A **microstrip** port measures
    ``sqrt(Et*dEt / (Ht*dHt))`` from its own fields, so the two ends of a
    genuinely symmetric line come back different, and the derived column
    inherited that difference.

    Two ports of a true mirror have the same Z0. A gap between them is two noisy
    estimates of one quantity, never a real asymmetry - so the gap is evidence
    about the *declaration*, and the arithmetic must not carry it into S.

    Asserted against the closed form, not against the old output: a series
    resistor R between two 50 ohm references has S11 = S22 = R/(R+100) and
    S21 = S12 = 100/(R+100), whatever impedances the ports report.
    """

    #: Series R and its exact answer at a 50 ohm reference.
    R = 20.0
    EXACT_S11 = R / (R + 100.0)
    EXACT_S21 = 100.0 / (R + 100.0)

    def one_path(self, z_a, z_b):
        runs = series_resistor(z_a, z_b, self.R)
        return SParameters.from_runs(runs[:1], reference=50.0, symmetry=MIRROR)

    @pytest.mark.parametrize("z_b", [40.0, 40.04, 44.0, 60.0])
    def test_the_derived_matrix_is_the_closed_form_whatever_the_ports_report(self, z_b):
        """The one assertion that separates every candidate fix.

        Deriving in the measured basis gave 0.2460 at z_b = 44 and 0.4831 at
        z_b = 60 against an exact 0.1667. Overwriting the undriven port's
        measured impedance with the driven one fixes S22 by putting the error
        into S11 instead. Renormalising both ports before deriving needs the
        column nobody measured. Only making the common basis out of the
        undriven port's own impedance is exact.
        """
        s = self.one_path(40.0, z_b).s[0]
        assert s[0, 0] == pytest.approx(self.EXACT_S11, abs=1e-6)
        assert s[1, 1] == pytest.approx(self.EXACT_S11, abs=1e-6)
        assert s[1, 0] == pytest.approx(self.EXACT_S21, abs=1e-6)
        assert s[0, 1] == pytest.approx(self.EXACT_S21, abs=1e-6)

    def test_the_measured_column_is_not_moved_to_prop_up_the_derived_one(self):
        """A derivation may never corrupt a measurement. Both matrices are at
        50 ohm, so the driven column must be the same in each."""
        both = SParameters.from_runs(series_resistor(40.0, 60.0, self.R), reference=50.0)
        derived = self.one_path(40.0, 60.0)
        assert derived.s[:, 0, 0] == pytest.approx(both.s[:, 0, 0], abs=1e-6)
        assert derived.s[:, 1, 0] == pytest.approx(both.s[:, 1, 0], abs=1e-6)

    def test_the_copy_survives_into_a_reflection_null(self):
        """The user's own complaint, as an assertion.

        A max-|S| metric cannot see this - which is why SYMMETRY_TOLERANCE
        never did. The error was a small *absolute* perturbation, invisible in
        the passband and several dB deep in a null: 0.1% of impedance
        disagreement showed as 3.52 dB at |S11| = -64 dB. R here is chosen to
        put |S11| near 1e-3, where the dB scale magnifies it.
        """
        runs = series_resistor(50.0, 50.05, 0.2)
        derived = SParameters.from_runs(runs[:1], reference=50.0, symmetry=MIRROR)
        s11, s22 = derived.s[0, 0, 0], derived.s[0, 1, 1]
        assert abs(s11) < 5e-3, "the fixture should sit in a null"
        gap = abs(20 * np.log10(abs(s11)) - 20 * np.log10(abs(s22)))
        assert gap < 0.01, f"{gap:.2f} dB between a column and its own source"

    def test_the_fixture_is_not_vacuous(self):
        """Both halves: the ports must genuinely disagree, and the disagreement
        must be the kind renormalisation exposes."""
        assert self.one_path(40.0, 60.0).provenance["symmetry_impedance_mismatch"] == pytest.approx(
            0.5, rel=1e-9, abs=0.0
        )

    def test_matched_ports_report_no_mismatch(self):
        result = self.one_path(40.0, 40.0)
        assert result.provenance["symmetry_impedance_mismatch"] == pytest.approx(0.0, abs=1e-12)

    def test_the_mismatch_is_recorded_not_assumed_away(self):
        result = self.one_path(40.0, 44.0)
        assert result.provenance["symmetry_impedance_mismatch"] == pytest.approx(
            0.1, rel=1e-9, abs=0.0
        )

    def test_the_mismatch_no_longer_predicts_an_error_because_there_is_none(self):
        """This assertion is the inverse of the one it replaces, deliberately.

        The old test demanded that a mismatch of x produce about x of error -
        10% in, 9.5e-2 out - and it passed, which is how a defect stayed live
        under a green suite for as long as it did. The number keeps its value
        and changes its meaning: it now says whether the *declaration* is
        credible, not how wrong the answer is.
        """
        both = SParameters.from_runs(series_resistor(40.0, 44.0, self.R), reference=50.0)
        derived = self.one_path(40.0, 44.0)
        mismatch = derived.provenance["symmetry_impedance_mismatch"]

        scale = float(np.max(np.abs(both.s)))
        error = float(np.max(np.abs(derived.s - both.s))) / scale
        assert mismatch == pytest.approx(0.1, rel=1e-9, abs=0.0)
        assert error < mismatch / 1000.0, (
            f"a {mismatch:.1%} impedance mismatch produced {error:.2%} of error"
        )

    def test_it_is_reported_rather_than_refused(self):
        """The declaration is the engineer's. What must not happen is that the
        number is unavailable, not that the run is stopped."""
        result = self.one_path(40.0, 60.0)
        assert result.complete
        assert result.derived == (2,)


class TestCheckingASymmetryClaim:
    """Two solves turn the declaration into something falsifiable.

    It measures the *network*: is S22 = S11 once both ports sit at a common
    reference? That is one of the two things the declaration needs. The other
    - that the two ports are themselves mirror images, so their reference
    impedances agree - costs no solve at all and belongs in pre-flight.
    """

    def test_a_symmetric_structure_agrees_with_itself(self):
        result = SParameters.from_runs(series_resistor(40.0, 40.0, 20.0), reference=50.0)
        assert result.mirror_disagreement() < 1e-6

    def test_a_series_resistor_is_symmetric_however_it_is_fed(self):
        """Worth pinning, because it is why this cannot be tested with one.

        A series element is its own mirror image whatever impedances it sits
        between - once both ports are renormalised to 50 ohm the asymmetry of
        the *feed* is gone. A test that used unequal port impedances to
        manufacture an asymmetric network would be asserting nothing.
        """
        result = SParameters.from_runs(series_resistor(25.0, 100.0, 20.0), reference=50.0)
        assert result.mirror_disagreement() < 1e-6

    def test_an_asymmetric_network_is_caught(self):
        """Z11 != Z22 is asymmetry no renormalisation can remove."""
        result = SParameters.from_runs(
            from_z_matrix([[60.0, 25.0], [25.0, 20.0]], [50.0, 50.0]),
            reference=50.0,
        )
        assert result.mirror_disagreement() > 0.1

    def test_a_non_reciprocal_network_is_caught_too(self):
        """The other half of the metric, which nothing else reaches.

        Every case above has Z12 = Z21, so |S12 - S21| is identically zero and
        deleting that term from ``mirror_disagreement`` changed nothing. A
        circulator-like Z-matrix is what makes it bite - and while no material
        this workbench can express is non-reciprocal today, the term is what
        would notice if one ever were.
        """
        result = SParameters.from_runs(
            from_z_matrix([[60.0, 25.0], [-25.0, 60.0]], [50.0, 50.0]),
            reference=50.0,
        )
        assert result.mirror_disagreement() > 0.1

    def test_it_refuses_to_check_a_matrix_it_derived(self):
        """Otherwise the answer is zero by construction, and a declaration
        would appear to confirm itself."""
        derived = SParameters.from_runs(
            series_resistor(40.0, 40.0, 20.0)[:1], reference=50.0, symmetry=MIRROR
        )
        with pytest.raises(ResultError, match="zero by construction"):
            derived.mirror_disagreement()

    def test_it_refuses_an_incomplete_matrix(self):
        one = SParameters.from_runs(series_resistor(40.0, 40.0, 20.0)[:1], reference=50.0)
        with pytest.raises(ResultError, match="never driven"):
            one.mirror_disagreement()


class TestWhatItRefuses:
    def test_two_runs_driving_the_same_port(self):
        runs = series_resistor(50.0, 50.0, 10.0)
        runs[1].excited_port = 1
        with pytest.raises(ResultError, match="same port"):
            SParameters.from_runs(runs)

    def test_runs_meshed_differently(self):
        """Structure is compared by cell count, not by envelope digest.

        Digests were the first attempt and were wrong: sweep() drives a
        different port per run, so each envelope differs by construction and
        the check rejected the one input it exists to accept.
        """
        runs = series_resistor(50.0, 50.0, 10.0)
        runs[1].provenance = dict(runs[1].provenance, cells=999)
        with pytest.raises(ResultError, match="meshed differently"):
            SParameters.from_runs(runs)

    def test_runs_differing_only_in_excitation_are_accepted(self):
        """The case a naive digest check rejects. This is normal input."""
        runs = series_resistor(50.0, 50.0, 10.0)
        for index, run in enumerate(runs):
            run.provenance = dict(run.provenance, envelope_digest=f"digest-{index}")
        result = SParameters.from_runs(runs)
        assert result.provenance["envelope_digest"] == {1: "digest-0", 2: "digest-1"}

    def test_runs_sampled_at_different_frequencies(self):
        runs = series_resistor(50.0, 50.0, 10.0)
        runs[1].frequency = np.array([1e9, 3e9])
        with pytest.raises(ResultError, match="different frequencies"):
            SParameters.from_runs(runs)

    def test_nothing_at_all(self):
        with pytest.raises(ResultError, match="no runs"):
            SParameters.from_runs([])


class TestWhatItCarries:
    def test_port_numbers_are_the_documents_not_zero_based(self):
        """A document numbering its ports 3 and 7 still yields a 2x2 matrix."""
        runs = series_resistor(50.0, 50.0, 10.0)
        for run in runs:
            run._z = {3: run._z[1], 7: run._z[2]}
            run._volts = {
                (3 if a == 1 else 7, 3 if b == 1 else 7): v for (a, b), v in run._volts.items()
            }
            run.excited_port = 3 if run.excited_port == 1 else 7

        result = SParameters.from_runs(runs)
        assert result.port_numbers == (3, 7)
        assert result.s.shape == (2, 2, 2)
        with pytest.raises(ResultError, match=r"no port 1 "):
            result.parameter(1, 1)

    def test_measured_impedance_is_kept_separate_from_the_reference(self):
        """QA F2 wants Z0 shown; it is not the 50 ohm the matrix is normalised to."""
        result = SParameters.from_runs(series_resistor(30.0, 75.0, 20.0), reference=50.0)
        assert result.impedance(1).real == pytest.approx(30.0, rel=1e-12, abs=0.0)
        assert result.impedance(2).real == pytest.approx(75.0, rel=1e-12, abs=0.0)
        assert np.all(result.reference == 50.0)

    def test_a_non_default_reference_is_honoured(self):
        """75 ohm, not 50. Every other test takes the default, so hard-coding
        50.0 in ``from_runs`` would otherwise pass the suite."""
        z01, z02, r = 30.0, 75.0, 20.0
        result = SParameters.from_runs(series_resistor(z01, z02, r), reference=75.0)

        expected = 2 * 75.0 / (2 * 75.0 + r)
        assert result.parameter(2, 1).real == pytest.approx(expected, rel=1e-6, abs=0.0)
        assert np.all(result.reference == 75.0)

    def test_the_reference_array_is_writeable(self):
        """``broadcast_to`` returns a read-only zero-strided view; it is copied.

        Nothing in the workbench writes to it today, but it is public state on
        a result object and handing out a read-only view is a trap.
        """
        result = SParameters.from_runs(series_resistor(50.0, 50.0, 10.0))
        result.reference[0, 0] = 51.0  # must not raise

    def test_impedance_is_averaged_across_the_runs(self):
        """A microstrip port *measures* Z_ref from its own run's fields, so the
        runs disagree slightly and no single run's value is right for every
        column. The reported impedance is the mean, and the spread is measured
        against it."""
        runs = series_resistor(50.0, 50.0, 10.0)
        runs[0]._z = {1: runs[0]._z[1] * 1.02, 2: runs[0]._z[2]}

        result = SParameters.from_runs(runs)
        assert result.impedance(1).real == pytest.approx(50.5, rel=1e-9, abs=0.0)
        # Spread is measured against the mean, not against either run: the two
        # runs are 51.0 and 50.0, so it is 0.5/50.5 and not 0.01.
        assert result.provenance["impedance_spread"] == pytest.approx(0.5 / 50.5, rel=1e-9, abs=0.0)

    def test_runs_that_disagree_wildly_about_impedance_are_refused(self):
        runs = series_resistor(50.0, 50.0, 10.0)
        runs[0]._z = {1: runs[0]._z[1] * 2.0, 2: runs[0]._z[2]}
        with pytest.raises(ResultError, match="disagree about port impedance"):
            SParameters.from_runs(runs)

    def test_a_non_finite_impedance_is_named(self):
        """A waveguide port below cutoff. Without this the failure is a
        LinAlgError from inside scikit-rf that names nothing."""
        runs = series_resistor(50.0, 50.0, 10.0)
        for run in runs:
            run._z = {1: run._z[1] * np.nan, 2: run._z[2]}
        with pytest.raises(ResultError, match="port 1 reports a non-finite"):
            SParameters.from_runs(runs)

    def test_it_names_the_run_that_produced_it_and_both_causes(self):
        """An ``MSLPort`` reads ``nan`` at every point in
        a run a *lumped* port excites, and is finite in the run it drives
        itself. Both causes are properties of the excitation, so naming only
        the measured port sends the reader to the wrong object - and this one
        is invisible from that port entirely. Advice about
        waveguide cutoff alone, in documents holding no waveguide."""
        runs = series_resistor(50.0, 50.0, 10.0)
        by_excitation = {int(run.excited_port): run for run in runs}
        bad = by_excitation[2]
        bad._z = {1: bad._z[1] * np.nan, 2: bad._z[2]}

        with pytest.raises(ResultError) as raised:
            SParameters.from_runs(runs)

        said = str(raised.value)
        assert "run driven by port 2" in said
        assert "cutoff" in said and "lumped port excites" in said

    def test_a_run_that_excited_nothing_is_refused_by_its_cause(self):
        """A lumped port whose box lies outside the grid. openEMS clips such a
        box away without comment, the run takes its full wall time, and it comes
        back with a finite port impedance beside a matrix of nan - every
        S-parameter is 0/0. The incident wave is the one place the cause is
        visible rather than the symptom."""
        runs = series_resistor(50.0, 50.0, 10.0)
        by_excitation = {int(run.excited_port): run for run in runs}
        dead = by_excitation[2]
        dead.incident = np.zeros_like(dead.frequency, dtype=complex)

        with pytest.raises(ResultError) as raised:
            SParameters.from_runs(runs)

        said = str(raised.value)
        assert "driven by port 2" in said and "no incident wave" in said
        assert "inside the grid" in said, "it should say what to look at"

    def test_a_live_run_is_not_mistaken_for_a_dead_one(self):
        """The guard is "identically zero", not "small". A well-matched port
        measuring near-nothing is an answer, not a failure."""
        runs = series_resistor(50.0, 50.0, 10.0)
        for run in runs:
            run.incident = np.full_like(run.frequency, 1e-30, dtype=complex)
        assert SParameters.from_runs(runs).ports == 2

    def test_a_nan_in_a_measured_column_never_reaches_scikit_rf(self):
        """Unchecked it arrives as ``LinAlgError: Array must not
        contain infs or NaNs`` out of ``s2z``'s inverse - minutes after the
        solve, from inside a library, naming nothing about the model, with the
        traceback in the panel's log. The zeros written into the *undriven*
        columns were already there for this reason; the measured ones were not
        asked the same question."""
        runs = series_resistor(50.0, 50.0, 10.0)
        by_excitation = {int(run.excited_port): run for run in runs}
        bad = by_excitation[1]
        bad._volts[(2, 1)] = bad._volts[(2, 1)] * np.nan

        with pytest.raises(ResultError) as raised:
            SParameters.from_runs(runs)

        said = str(raised.value)
        assert "S21 is not a number at 2 of 2 frequency points" in said
        assert "driven by port 1" in said

    def test_per_solve_provenance_is_kept_per_run(self):
        """A two-run matrix must not report one run's wall time as its own."""
        runs = series_resistor(50.0, 50.0, 10.0)
        for index, run in enumerate(runs):
            run.provenance = dict(run.provenance, wall_seconds=10.0 + index)

        result = SParameters.from_runs(runs)
        assert result.provenance["wall_seconds"] == {1: 10.0, 2: 11.0}

    def test_each_run_keeps_what_its_own_tails_were_worth(self):
        """The one field here that can say a matrix is wrong. Merged as though
        it described the structure, a two-run sweep would report whichever run
        came first and the truncated one would vanish.

        The inner keys are strings because these come off a results file, which
        JSON gives no integer keys; ``read.Results.tail_share`` is what puts
        them back, and this merge does not go through it."""
        runs = series_resistor(50.0, 50.0, 10.0)
        for index, run in enumerate(runs):
            run.provenance = dict(run.provenance, tail_share={"1": 1e-4, "2": 0.2 * index})

        result = SParameters.from_runs(runs)
        assert result.provenance["tail_share"] == {
            1: {"1": 1e-4, "2": 0.0},
            2: {"1": 1e-4, "2": 0.2},
        }

    def test_provenance_records_which_library_produced_it(self):
        result = SParameters.from_runs(series_resistor(50.0, 50.0, 10.0))
        assert result.provenance["result_library"].startswith("scikit-rf ")
        assert result.provenance["excitations"] == [1, 2]

    def test_one_irreproducible_run_taints_the_matrix(self):
        runs = series_resistor(50.0, 50.0, 10.0)
        runs[1].reproducible = False
        assert SParameters.from_runs(runs).provenance["reproducible"] is False


class TestDescribingTheReference:
    """One number could only be right by luck, so the display is a sentence."""

    def test_one_value_everywhere_reads_as_one_value(self):
        assert describe_reference(np.full((5, 2), 50.0), (1, 2)) == "50 ohm"

    def test_ports_that_differ_are_named(self):
        reference = np.tile([25.0, 100.0], (5, 1))
        assert describe_reference(reference, (1, 2)) == "port 1: 25 ohm, port 2: 100 ohm"

    def test_a_reference_that_moves_with_frequency_says_so(self):
        reference = np.column_stack([np.linspace(48.0, 52.0, 5)] * 2)
        assert describe_reference(reference, (1, 2)) == "an impedance that varies with frequency"

    def test_a_complex_reference_is_written_as_one(self):
        reference = np.full((3, 2), 49.7 - 0.8j)
        assert describe_reference(reference, (1, 2)) == "49.7 - 0.8j ohm"

    def test_a_port_at_its_own_impedance_is_named_beside_the_ones_that_have_a_number(self):
        """Folding a requested 50 ohm together with a microstrip's own
        49.7 - 0.8j into one verdict reads as "varies with frequency" and says
        nothing. Ports that disagree are listed one by one, whether they disagree
        about a number or about having one."""
        reference = np.column_stack([np.full(3, 50.0 + 0j), np.linspace(48.0, 52.0, 3) - 0.8j])
        assert (
            describe_reference(reference, (1, 2), (2,))
            == "port 1: 50 ohm, port 2: its own impedance"
        )

    def test_every_port_at_its_own_impedance_needs_no_number_at_all(self):
        """The waveguide case: there is no number, and quoting the first bin of
        a dispersive impedance would invent one."""
        reference = np.column_stack([np.linspace(470.0, 480.0, 5)] * 2)
        assert describe_reference(reference, (1, 2), (1, 2)) == "each port's own impedance"

    def test_a_lone_port_at_its_own_impedance_is_not_pluralised(self):
        """``each port's`` for a one-port would be a sentence about a set of one."""
        guide = np.linspace(470.0, 480.0, 3).reshape(3, 1)
        assert describe_reference(guide, (1,), (1,)) == "its own impedance"

    def test_a_port_at_its_own_impedance_still_reads_as_the_number_when_it_has_one(self):
        """A lumped port's own impedance is the resistance that was typed, so
        there is a number and the reason for it is not what a reader came for.
        ``its own impedance`` is for where there is nothing else to say, which
        is a guide. On the workbench's commonest study - two lumped ports at
        50 ohm, one of them undriven - it captions two curves of one column as
        though they sat on different bases.
        """
        assert describe_reference(np.full((4, 2), 50.0), (1, 2), (2,)) == "50 ohm"
        assert describe_reference(np.full((4, 1), 475.0), (1,), (1,)) == "475 ohm"

    def test_a_port_at_its_own_impedance_that_holds_still_is_named_when_it_differs(self):
        """It is a number, so it is reported as one - and it being a different
        number from the other port's is the thing worth seeing."""
        reference = np.tile([50.0, 75.0], (4, 1))
        assert describe_reference(reference, (1, 2), (2,)) == "port 1: 50 ohm, port 2: 75 ohm"

    def test_a_fixed_port_keeps_its_number_beside_one_that_disperses(self):
        """Both ports carry a number, and only one of them holds still.

        Collapsing the pair into one verdict about frequency was the old rule,
        and it threw away the number port 1 does have. Per port, the answer is
        two facts instead of one.
        """
        reference = np.column_stack([np.full(4, 50.0), np.linspace(470.0, 480.0, 4)])
        assert describe_reference(reference, (1, 2)) == (
            "port 1: 50 ohm, port 2: an impedance that varies with frequency"
        )

    def test_every_shape_reads_as_english_after_the_words_referenced_to(self):
        """The one sentence the chart, the ``Reference`` property and both
        Touchstone refusals share. A bracketed list or a bare verb phrase reads
        as a debug print in every one of them.
        """
        band = np.linspace(48.0, 52.0, 4)
        shapes = [
            (np.full((4, 2), 50.0), (1, 2), ()),
            (np.column_stack([band] * 2), (1, 2), ()),
            (np.column_stack([band] * 2), (1, 2), (1, 2)),
            (np.full((4, 1), 475.0), (1,), (1,)),
            (np.tile([25.0, 100.0], (4, 1)), (1, 2), ()),
            (np.column_stack([np.full(4, 50.0), band]), (1, 2), (2,)),
            (np.column_stack([np.full(4, 50.0), band]), (1, 2), ()),
        ]
        for reference, numbers, at_own in shapes:
            sentence = f"referenced to {describe_reference(reference, numbers, at_own)}"
            assert "[" not in sentence and "(s)" not in sentence, sentence
            # A verb where the object belongs: every label has to be a noun
            # phrase, because half of them are read after "port 3:" as well.
            assert re.match(r"referenced to (an?|its|each|port|-?\d)", sentence), sentence

    def test_a_number_that_happens_to_equal_what_the_port_measured_is_still_a_number(self):
        """The ordinary two-port: two lumped ports at 50 ohm, asked for 50.

        openEMS sets a lumped port's Z_ref to its resistance exactly, so the
        reference and the measurement agree to the bit - and inferring intent
        from that agreement reported the commonest study in the workbench as
        referenced to nothing in particular.
        """
        runs = series_resistor(50.0, 50.0, 20.0)
        result = SParameters.from_runs(runs, reference=50.0)

        np.testing.assert_array_equal(result.reference, result.measured_impedance)
        assert result.self_referenced == ()
        assert result.reference_description() == "50 ohm"

    def test_a_derived_mirror_asked_for_nothing_is_recorded_and_reads_as_its_number(self):
        """Both ports sit at the pair's one common impedance, which is neither
        port's own measurement - so only ``from_runs`` can record what it did,
        and it does. What the reader is shown is still the number, because there
        is one and it is what every term on the chart is against: the same
        impedance that ``one_reference`` would let Touchstone put in its option
        line, where a phrase about whose impedance it is has nowhere to go.
        """
        runs = series_resistor(50.0, 50.4, 20.0)[:1]
        result = SParameters.from_runs(runs, reference=None, symmetry=MIRROR)

        assert result.self_referenced == (1, 2)
        assert result.one_reference
        assert result.reference_description() == "50.4 ohm"


class TestTheStubStillMatchesTheRealThing:
    """Everything above drives ``Run``, not ``read.Results``.

    That is deliberate - exact stubs are what let the physics be checked
    against a closed form instead of against FDTD noise - but it means a
    rename in ``read.py`` would leave every test in this file green while
    ``from_runs`` no longer worked on real output. The acceptance test would
    catch it, four minutes and an openEMS install later. This is the cheap
    version of that check.
    """

    def test_the_surface_from_runs_uses_exists_on_read_results(self):
        from Microwave.Solvers.openems import read

        for name in ("frequency", "excited_port", "ports", "provenance", "reproducible"):
            assert hasattr(read.Results, name) or name in read.Results.__dataclass_fields__, (
                f"read.Results has no {name!r}; the stub in this file has drifted"
            )
        for name in ("port", "s"):
            assert callable(getattr(read.Results, name, None)), (
                f"read.Results.{name}() is gone; the stub in this file has drifted"
            )
        assert "z0" in read.PortResult.__dataclass_fields__


class TestTouchstone:
    def test_it_round_trips_through_a_file(self, tmp_path):
        from Microwave.Results import _skrf

        result = SParameters.from_runs(series_resistor(30.0, 75.0, 20.0), reference=50.0)
        written = result.write_touchstone(tmp_path / "device")

        assert written.name == "device.s2p"
        again = _skrf.module().Network(str(written))
        assert np.allclose(again.s, result.s, rtol=0, atol=1e-9)
        assert np.allclose(again.z0.real, 50.0)

    def test_a_reference_the_format_cannot_hold_is_refused_here(self, tmp_path):
        """One real number for every port at every frequency, or no file.

        Three ways to miss that, and all three used to arrive as scikit-rf's own
        sentence out of a vendored library, after the save dialog, naming
        nothing about the model: ports at different impedances, which is the
        ordinary 30/75 study; a reference that varies with frequency; and a
        complex one.
        """
        runs = series_resistor(30.0, 75.0, 20.0)
        for reference in (None, [30.0, 75.0], [50.0, None]):
            result = SParameters.from_runs(runs, reference=reference)
            with pytest.raises(ResultError, match="one real reference impedance"):
                result.write_touchstone(tmp_path / "device")
            assert not (tmp_path / "device.s2p").exists()

    def test_a_mirror_whose_ports_end_up_at_different_references_is_refused(self, tmp_path):
        """The header would claim S22 is a copy of S11, and it would be off.

        A mirror declares the two ports identical, so asking for one against a
        number and the other against itself asks for the copy to be taken in one
        basis and reported in another - the final renormalisation then moves
        the two diagonal terms by different amounts. ``Gui/symmetry`` warns
        before the solve; the file cannot be written either way, because two
        references are two references.
        """
        result = SParameters.from_runs(
            series_resistor(50.0, 50.4, 20.0)[:1], reference=[50.0, None], symmetry=MIRROR
        )
        assert result.complete

        with pytest.raises(ResultError, match="one real reference impedance"):
            result.write_touchstone(tmp_path / "device")

    def test_the_refusal_says_what_the_reference_actually_is(self, tmp_path):
        result = SParameters.from_runs(series_resistor(30.0, 75.0, 20.0), reference=[25.0, 100.0])
        with pytest.raises(ResultError, match=re.escape("port 1: 25 ohm, port 2: 100 ohm")):
            result.write_touchstone(tmp_path / "device")

    def test_a_given_suffix_is_not_doubled(self, tmp_path):
        """scikit-rf appends .sNp itself; passing one would give device.s2p.s2p."""
        result = SParameters.from_runs(series_resistor(50.0, 50.0, 10.0))
        written = result.write_touchstone(tmp_path / "device.s2p")
        assert written.name == "device.s2p"
        assert written.is_file()

    def test_the_file_says_what_made_it(self, tmp_path):
        """The file is handed on with nothing beside it, so it names the tool."""
        result = SParameters.from_runs(series_resistor(50.0, 50.0, 10.0))
        text = result.write_touchstone(tmp_path / "device").read_text()
        assert "Produced by   FreeCAD Microwave" in text

    def test_it_says_what_the_numbers_are_of(self, tmp_path):
        """Network size, span and reference - what makes the file readable
        without the document it came from."""
        result = SParameters.from_runs(series_resistor(50.0, 50.0, 10.0))
        text = result.write_touchstone(tmp_path / "device").read_text()

        assert "! Network       2-port" in text
        assert "! Frequency     1 GHz to 2 GHz, 2 points" in text
        assert "! Reference     50 ohm" in text

    def test_it_is_dated(self, tmp_path):
        """Sortable, unambiguous and zoned. A local timestamp in a file handed
        across a timezone is worse than none."""
        result = SParameters.from_runs(series_resistor(50.0, 50.0, 10.0))
        text = result.write_touchstone(tmp_path / "device").read_text()

        dated = [line for line in text.splitlines() if line.startswith("! Date")]
        assert len(dated) == 1, dated
        assert re.fullmatch(r"! Date {10}\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC", dated[0])

    def test_it_carries_both_versions(self, tmp_path):
        """They answer different questions: the simulator's is what produced the
        numbers, ours is what arranged them."""
        from Microwave import __version__

        text = self._fully_described().write_touchstone(tmp_path / "device").read_text()

        assert f"! Produced by   FreeCAD Microwave {__version__}" in text
        assert "! Simulator     openEMS 0.0.36" in text

    @pytest.mark.parametrize(
        "title",
        [
            # "! gamma" and "! port impedance" are HFSS extensions, and the
            # reader consumes the lines *after* them as floats - so a study
            # named this way eats the option line and the file stops parsing.
            "Gamma sweep",
            "Port impedance study",
            # An embedded newline splits one comment into a second line with no
            # "!" on it, which is a data line as far as the reader is concerned.
            "Two\nport",
            # An ordinary FreeCAD label, and the format predates any encoding
            # assumption.
            "50 Ω taper",
            "45° bend",
            # Long enough that a header stops being readable.
            "A" * 200,
            "# not an option line",
            "",
        ],
    )
    def test_a_study_name_cannot_break_the_file(self, tmp_path, title):
        """The name is whatever the user typed in the tree, and it lands in a
        comment. A Touchstone comment is not inert."""
        from Microwave.Results import _skrf

        runs = series_resistor(50.0, 50.0, 10.0)
        for run in runs:
            run.provenance = dict(run.provenance, title=title)
        result = SParameters.from_runs(runs, reference=50.0)

        written = result.write_touchstone(tmp_path / "device")

        text = written.read_text()
        text.encode("ascii")
        header = text.split("# Hz")[0].splitlines()
        assert all(line.startswith("!") for line in header), header
        # A header line is read at a glance beside seven others, and a FreeCAD
        # label has no length limit at all.
        assert max(len(line) for line in header) <= 80, header
        back = _skrf.module().Network(str(written))
        np.testing.assert_allclose(back.s, result.s, rtol=1e-6, atol=1e-9)

    def test_the_study_name_is_labelled_not_bare(self, tmp_path):
        """What keeps it off the front of the line, which is where the reader
        looks for its keywords."""
        runs = series_resistor(50.0, 50.0, 10.0)
        for run in runs:
            run.provenance = dict(run.provenance, title="Gamma sweep")
        result = SParameters.from_runs(runs, reference=50.0)

        text = result.write_touchstone(tmp_path / "device").read_text()

        assert "! Study         Gamma sweep" in text
        assert "! Gamma sweep" not in text

    def _fully_described(self, reference=75.0):
        """A result carrying everything the header can say.

        The default fixture's provenance names no solver, so the simulator line
        is absent from it and every assertion about that line would pass on a
        file that never had one. Same for the reference: at 50 ohm a hardcoded
        "50 ohm" is indistinguishable from one read off the data.
        """
        runs = series_resistor(reference, reference, 10.0)
        for run in runs:
            run.provenance = dict(
                run.provenance,
                title="Lowpass stub filter",
                solver="openEMS",
                solver_version="0.0.36",
            )
        return SParameters.from_runs(runs, reference=reference)

    def test_it_names_the_simulator_that_produced_the_numbers(self, tmp_path):
        """Not the workbench's version but the engine's: reproducing a result
        needs it the way reproducing a bench measurement needs the firmware
        revision."""
        text = self._fully_described().write_touchstone(tmp_path / "device").read_text()

        assert "! Simulator     openEMS 0.0.36" in text
        assert "! Study         Lowpass stub filter" in text

    def test_the_reference_is_read_off_the_data(self, tmp_path):
        text = self._fully_described(reference=75.0).write_touchstone(tmp_path / "d").read_text()

        assert "! Reference     75 ohm" in text

    def test_it_is_plain_ascii(self, tmp_path):
        """Touchstone predates any encoding assumption and downstream parsers
        are old. "ohm", never a symbol.

        On the fullest header there is, so every line is judged - one built
        from a bare result leaves out the lines most likely to carry a stray
        symbol, because they are the ones quoting text from elsewhere.
        """
        text = self._fully_described().write_touchstone(tmp_path / "device").read_text()

        text.encode("ascii")

    def test_every_header_line_is_a_touchstone_comment(self, tmp_path):
        """A header line that is not commented is data as far as a parser is
        concerned, and a decorative rule would be read as a frequency point."""
        result = SParameters.from_runs(series_resistor(50.0, 50.0, 10.0))
        text = result.write_touchstone(tmp_path / "device").read_text()

        head = text.split("# Hz")[0].splitlines()
        assert head, "the option line moved; this no longer reads the header"
        assert all(line.startswith("!") for line in head), head

    def test_the_vendored_library_does_not_sign_it_too(self, tmp_path):
        """scikit-rf writes ``Created with skrf ...`` unless told not to."""
        result = SParameters.from_runs(series_resistor(50.0, 50.0, 10.0))
        text = result.write_touchstone(tmp_path / "device").read_text()
        assert "Created with skrf" not in text
        assert text.count("FreeCAD Microwave") == 1


BAND = (1e9, 2e9, 3e9, 4e9)


def one_bad_point(index=1, factor=2.0, frequency=BAND):
    """Two runs of a matched series resistor that disagree about Z0 at one point.

    The disagreement is *local*, which is the whole subject: a microstrip port's
    measured impedance goes indeterminate near a standing-wave null and is fine
    a bin either side, so a fixture that spoils the whole band could not tell a
    per-point rule from the band-wide one it replaced.
    """
    runs = series_resistor(50.0, 50.0, 10.0, frequency=frequency)
    scale = np.ones(len(frequency), dtype=complex)
    scale[index] = factor
    runs[0]._z = {1: runs[0]._z[1] * scale, 2: runs[0]._z[2]}
    return runs


class TestOneBadPointDoesNotKillTheSweep:
    """A resonant DUT makes the impedance extraction fail at a few frequencies.

    A band-wide maximum instead lets a handful of bad points
    refused the other 195 - minutes of FDTD thrown away over a resonance. It
    is per point now, and the points that fail come back ``nan``.
    """

    def test_the_bad_point_is_named(self):
        result = SParameters.from_runs(one_bad_point(index=1))
        assert result.discarded == (1,)

    def test_the_whole_matrix_is_blanked_at_that_point(self):
        """Every term, not one port's row and column. Renormalising mixes them
        all at a given frequency, so one bad Z0 spoils that point entirely -
        and the Z0 that went wrong here belongs to port 1 alone, so ``s[1,1,1]``
        is the corner that distinguishes the two."""
        result = SParameters.from_runs(one_bad_point(index=1))
        assert np.all(np.isnan(result.s[1]))

    def test_its_neighbours_are_untouched(self):
        """Renormalisation does not mix across frequencies. That is what makes
        the rest of the band sound rather than merely unexamined."""
        result = SParameters.from_runs(one_bad_point(index=1))
        assert np.all(np.isfinite(result.s[[0, 2, 3]]))

    def test_the_surviving_points_equal_a_sweep_that_never_had_it(self):
        """The strongest form of "the rest of the band is untouched": the same
        numbers as solving only the good frequencies.

        To 1e-6 and not to the last bit, and the gap is scikit-rf's rather than
        ours: ``renormalize`` short-circuits when the whole
        ``z0`` array already equals the target and otherwise goes round through
        Z-parameters for *every* frequency, so one differing point moves all of
        them by ~6e-8 - the same residue :func:`_normalisation` records.

        It still discriminates, because the failure this guards against is not
        subtle: a band-wide mean impedance, which is what this replaced, moves
        these points by a third.
        """
        spoiled = SParameters.from_runs(one_bad_point(index=1)).usable()
        clean = SParameters.from_runs(series_resistor(50.0, 50.0, 10.0, frequency=(1e9, 3e9, 4e9)))
        assert spoiled.s == pytest.approx(clean.s, rel=1e-6, abs=0.0)
        assert spoiled.frequency == pytest.approx(clean.frequency, rel=1e-12, abs=0.0)

    def test_a_disagreement_under_the_tolerance_is_kept(self):
        """|f-1|/(f+1) is the spread against the mean, so 1.1978 gives 9%."""
        result = SParameters.from_runs(one_bad_point(index=1, factor=1.1978))
        assert result.provenance["impedance_spread"] < 0.10
        assert result.discarded == ()
        assert np.all(np.isfinite(result.s))

    def test_a_disagreement_over_the_tolerance_is_not(self):
        """And 1.2472 gives 11%. The pair brackets the constant: a tolerance
        moved either way fails one of them. It says nothing about ``>`` against
        ``>=``, and cannot - no factor lands the spread exactly on 0.10 in
        binary, so the boundary itself is unobservable from here."""
        result = SParameters.from_runs(one_bad_point(index=1, factor=1.2472))
        assert result.provenance["impedance_spread"] > 0.10
        assert result.discarded == (1,)

    def test_every_point_bad_is_still_a_refusal(self):
        """Nothing survives, so there is no partial result to hand back - and
        a matrix of pure nan is the one thing worse than a refusal."""
        runs = series_resistor(50.0, 50.0, 10.0, frequency=BAND)
        runs[0]._z = {1: runs[0]._z[1] * 2.0, 2: runs[0]._z[2]}
        with pytest.raises(ResultError, match="at every frequency"):
            SParameters.from_runs(runs)

    def test_one_run_never_discards(self):
        """Nothing to disagree with: the mean of one run is that run, so every
        difference is zero. This needs no special case in the source and has
        none - it is here because it is the *common* case (one solve, S11 and
        S21) and so the one where a silent failure would cost most.
        """
        runs = series_resistor(50.0, 50.0, 10.0, frequency=BAND)[:1]
        result = SParameters.from_runs(runs)
        assert result.discarded == ()
        assert result.provenance["impedance_spread"] == 0.0
        assert np.all(np.isfinite(result.parameter(1, 1)))

    def test_runs_that_are_wrong_the_same_way_are_not_caught(self):
        """Named, because the docstrings around this could be read as a promise.

        It compares runs against each other, so two extractions that fail
        identically - both ports' probes inside the same junction - agree
        perfectly and pass - on a real solve they agree exactly while Z0
        comes back wildly reactive for a real line.
        """
        runs = series_resistor(50.0, 50.0, 10.0, frequency=BAND)
        for run in runs:
            run._z = {1: run._z[1] * 4.0, 2: run._z[2]}

        result = SParameters.from_runs(runs)
        assert result.discarded == ()
        assert result.provenance["impedance_spread"] == 0.0

    def test_the_indices_come_back_sorted_and_unique(self):
        """``blank_span`` reads the ends of this tuple, and ``usable``
        indexes with it. A caller building one by hand - ``load`` does, from a
        document somebody could have edited - must not be able to make either
        of those wrong."""
        full = SParameters.from_runs(one_bad_point(index=1))
        result = SParameters(
            frequency=full.frequency,
            s=full.s,
            port_numbers=full.port_numbers,
            reference=full.reference,
            measured_impedance=full.measured_impedance,
            discarded=(3, 1, 3),
        )
        assert result.discarded == (1, 3)

    def test_the_spread_reported_is_the_worst_one(self):
        """Not the worst *surviving* one. A result with holes should read as
        alarming, because it is."""
        result = SParameters.from_runs(one_bad_point(index=1, factor=2.0))
        assert result.provenance["impedance_spread"] == pytest.approx(1.0 / 3.0, rel=1e-9, abs=0.0)


class TestASolveThatReturnedNoFieldAtAll:
    """Every term ``nan``, with nothing having been discarded.

    The refusals asked the *bookkeeping* - ``driven``, ``derived``,
    ``discarded`` - and never the array, so this passed all of them and wrote
    a Touchstone file of ``nan`` tokens. It is not hypothetical: a waveguide
    port whose plane missed the grid excites nothing, ``|uf_inc|`` is zero and
    every term comes back ``nan``, and a single run has nobody to disagree with,
    so ``discarded`` stays empty.
    """

    def blank(self, ports=2, driven=None):
        n = len(np.atleast_1d(np.arange(ports)))
        frequency = np.array([1e9, 2e9, 3e9])
        return SParameters(
            frequency=frequency,
            s=np.full((3, n, n), np.nan, dtype=complex),
            port_numbers=tuple(range(1, n + 1)),
            driven=driven,
            reference=np.full((3, n), 50.0, dtype=complex),
            measured_impedance=np.full((3, n), 50.0, dtype=complex),
        )

    def test_the_bookkeeping_alone_calls_it_complete(self):
        """The premise. If this ever fails the test below proves nothing."""
        result = self.blank()
        assert result.complete and not result.discarded

    def test_a_network_is_refused(self):
        with pytest.raises(ResultError, match="hold no numbers"):
            self.blank().network()

    def test_a_touchstone_file_is_refused(self, tmp_path):
        with pytest.raises(ResultError, match="cannot write a Touchstone file"):
            self.blank().write_touchstone(tmp_path / "empty")

    def test_the_mirror_is_not_reported_as_confirmed(self):
        """``if gap > 1e-3`` is False for NaN, so this answered "the mirror
        declaration holds" about a result with no numbers in it - the one
        check that makes the claim falsifiable, answering yes to everything."""
        with pytest.raises(ResultError, match="no field|nothing to compare"):
            self.blank().mirror_disagreement()

    def test_the_refusal_says_the_solve_returned_nothing(self):
        """Not "the runs disagreed": nothing disagreed, and sending the user to
        look at measurement planes for a port that excited nothing wastes the
        one message that could have said what to check."""
        with pytest.raises(ResultError, match="grid lines"):
            self.blank().network()

    def test_usable_does_not_hand_back_the_same_dead_end(self):
        """``_kept`` was fixed to ask the array; ``usable`` was left guarding on
        ``discarded``, so it returned ``self`` for a solve with nothing
        discarded - and the refusal that sent the user here said "call
        usable() for the points that do"."""
        blank = self.blank()
        assert blank.usable() is not blank
        assert blank.usable().frequency.size == 0

    def test_the_refusal_does_not_offer_a_way_out_that_is_not_one(self):
        with pytest.raises(ResultError) as raised:
            self.blank().network()
        assert "usable()" not in str(raised.value)
        assert "nothing to keep" in str(raised.value)

    def test_a_legitimately_partial_matrix_is_untouched(self):
        """An undriven column is *supposed* to be nan. Asking the array must
        not turn the ordinary one-path measurement into a refusal."""
        result = self.blank(driven=(1,))
        object.__setattr__(
            result,
            "s",
            np.where(np.arange(2)[None, None, :] == 0, 0.1 + 0.2j, np.nan) * np.ones((3, 2, 2)),
        )
        assert result._kept.all()


class TestWhatABandWithAHoleRefuses:
    """``nan`` in a Network propagates through every operation it has and
    arrives at the far end as a plot of nothing. Both doors are shut."""

    def result(self):
        return SParameters.from_runs(one_bad_point(index=1))

    def test_a_network_is_refused(self):
        with pytest.raises(ResultError, match="hold no numbers"):
            self.result().network()

    def test_a_touchstone_file_is_refused_by_name(self, tmp_path):
        """By name, because ``network()`` below would refuse it anyway - and
        would say "cannot build a Network" to somebody who asked for a file.
        The same reason ``_require_complete`` is called in both places."""
        with pytest.raises(ResultError, match="cannot write a Touchstone file"):
            self.result().write_touchstone(tmp_path / "holed")

    def test_the_refusal_names_the_way_out(self):
        with pytest.raises(ResultError, match=r"usable\(\)"):
            self.result().network()

    def test_the_refusal_names_where_the_hole_is(self):
        with pytest.raises(ResultError, match=r"at 2 GHz"):
            self.result().network()

    def test_two_holes_are_reported_as_two_and_not_as_a_band(self):
        """Holes at the ends of a sweep are not nine gigahertz of dead band, and
        "between X and Y" said they were."""
        result = SParameters.from_runs(one_bad_point(index=0, factor=2.0))
        spread = SParameters(
            frequency=result.frequency,
            s=result.s,
            port_numbers=result.port_numbers,
            reference=result.reference,
            measured_impedance=result.measured_impedance,
            discarded=(0, 3),
        )
        assert spread.blank_span() == "lowest 1 GHz, highest 4 GHz"

    def test_one_hole_is_not_reported_as_a_span(self):
        assert self.result().blank_span() == "at 2 GHz"

    def test_usable_opens_both(self, tmp_path):
        usable = self.result().usable()
        assert usable.network() is not None
        assert usable.write_touchstone(tmp_path / "trimmed").is_file()

    def test_usable_trims_every_array_together(self):
        """One array left at full length is an object whose columns no longer
        line up with its frequency axis: ``impedance()`` then answers for the
        wrong bin, and ``store`` writes a list the document refuses to reload.
        """
        usable = self.result().usable()
        points = usable.frequency.size
        assert points == 3
        for name in ("s", "reference", "measured_impedance"):
            assert len(getattr(usable, name)) == points, name

    def test_usable_is_the_same_object_when_there_is_no_hole(self):
        """Nothing to trim, nothing to copy - and a caller that always calls
        it should not pay for a copy of a 201-point matrix."""
        result = SParameters.from_runs(series_resistor(50.0, 50.0, 10.0))
        assert result.usable() is result

    def test_the_touchstone_header_says_the_band_is_incomplete(self, tmp_path):
        """A shorter frequency list is indistinguishable from a sweep that was
        never asked for those points. The file has to say which it is."""
        runs = series_resistor(50.0, 50.0, 10.0, frequency=BAND)
        scale = np.array([2.0, 1.0, 1.0, 2.0], dtype=complex)
        runs[0]._z = {1: runs[0]._z[1] * scale, 2: runs[0]._z[2]}
        holed = SParameters.from_runs(runs)
        assert holed.discarded == (0, 3)

        text = holed.usable().write_touchstone(tmp_path / "trimmed").read_text()
        assert "2 FREQUENCY POINT(S) ARE MISSING" in text
        # Both ends, so a header that reported only one of them - or swapped
        # them - is visible. One hole could not tell.
        assert "lowest 1 GHz, highest 4 GHz" in text
        assert text.splitlines()[0].startswith("!")

    def test_a_complete_band_says_nothing(self, tmp_path):
        result = SParameters.from_runs(series_resistor(50.0, 50.0, 10.0))
        text = result.write_touchstone(tmp_path / "whole").read_text()
        assert "MISSING" not in text

    def test_a_symmetry_claim_is_still_checked_over_what_is_left(self):
        """Refusing here would be a second reason the panel says nothing about
        symmetry, and the declaration is about the structure, not the band."""
        result = SParameters.from_runs(one_bad_point(index=1))
        assert result.mirror_disagreement() < 1e-6


class TestWhereTheFileActuallyLands:
    """``touchstone_path``: the name a write would use.

    Public because a save dialog checks for an existing file under the name the
    user typed. If the suffix rule moves that to a different file, the dialog
    asked about the wrong one - and the answer was destructive.
    """

    def subject(self, ports=2):
        numbers = tuple(range(1, ports + 1))
        return SParameters(
            frequency=np.array([1e9, 2e9]),
            s=np.zeros((2, ports, ports), dtype=complex),
            port_numbers=numbers,
            reference=np.full((2, ports), 50.0),
            measured_impedance=np.full((2, ports), 50.0 + 0j),
        )

    @pytest.mark.parametrize(
        "typed, expected",
        [
            ("device", "device.s2p"),
            ("device.s2p", "device.s2p"),
            ("device.S2P", "device.S2P"),
            # The bug this exists for. ``with_suffix`` replaces whatever follows
            # the last dot, and a frequency or a version in a filename is
            # indistinguishable from a suffix - so these three collapsed onto
            # one name, and any file already using it was overwritten with no
            # prompt, the dialog having asked about a name that did not exist.
            ("lowpass_2.4GHz", "lowpass_2.4GHz.s2p"),
            ("lowpass_2.9GHz", "lowpass_2.9GHz.s2p"),
            ("rev1.2", "rev1.2.s2p"),
            ("run.sim", "run.sim.s2p"),
            ("model.step", "model.step.s2p"),
            # A Touchstone suffix for the wrong port count is appended to rather
            # than corrected: a name that says something the user did not ask
            # for is worse than an ugly one.
            ("device.s3p", "device.s3p.s2p"),
        ],
    )
    def test_the_suffix_is_appended_and_never_substituted(self, typed, expected):
        assert self.subject().touchstone_path(typed).name == expected

    def test_the_port_count_comes_from_the_matrix(self):
        assert self.subject(ports=1).touchstone_path("x").name == "x.s1p"
        assert self.subject(ports=3).touchstone_path("x").name == "x.s3p"

    def test_a_one_port_keeps_its_own_suffix_and_not_a_two_port_one(self):
        """``.s1p`` is a real Touchstone suffix, so a rule that matched any of
        them would leave a one-port file named after a two-port."""
        assert self.subject(ports=1).touchstone_path("x.s1p").name == "x.s1p"
        assert self.subject(ports=2).touchstone_path("x.s1p").name == "x.s1p.s2p"

    def test_the_directory_survives(self, tmp_path):
        landed = self.subject().touchstone_path(tmp_path / "sub" / "a.4b")
        assert landed.parent == tmp_path / "sub"
        assert landed.name == "a.4b.s2p"

    def test_a_write_lands_where_it_said_it_would(self, tmp_path):
        """The two must not drift apart: the caller checks one and gets the
        other, which is exactly the overwrite this was found through."""
        subject = self.subject()
        assert subject.write_touchstone(tmp_path / "v1.2") == subject.touchstone_path(
            tmp_path / "v1.2"
        )

    def test_a_dotted_name_does_not_overwrite_its_neighbour(self, tmp_path):
        """The failure end to end: ``lowpass_2.4GHz`` is
        written as ``lowpass_2.s2p`` and destroy an unrelated file."""
        bystander = tmp_path / "lowpass_2.s2p"
        bystander.write_text("not ours")

        self.subject().write_touchstone(tmp_path / "lowpass_2.4GHz")

        assert bystander.read_text() == "not ours"
        assert (tmp_path / "lowpass_2.4GHz.s2p").is_file()
