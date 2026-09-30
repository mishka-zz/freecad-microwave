# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Two S-matrices of one drawing, compared term by term, or refused by name."""

import numpy as np
import pytest

from Microwave.Results.compared import DRAWING, SIGN_BY_RULE, Incomparable, compare
from Microwave.Results.sparameters import SParameters

BAND = np.linspace(20e9, 26e9, 7)


def line(transmission=0.9, reflection=0.1, frequency=BAND, **fields):
    """A matched-looking two-port whose terms turn with frequency."""
    phase = np.exp(-1j * np.linspace(0.0, 3.0, frequency.size))
    s = np.empty((frequency.size, 2, 2), dtype=complex)
    s[:, 0, 0] = s[:, 1, 1] = reflection * phase
    s[:, 1, 0] = s[:, 0, 1] = transmission * phase**2
    reference = fields.pop("reference", np.full((frequency.size, 2), 50.0 + 0j))
    return SParameters(
        frequency=np.asarray(frequency, dtype=float),
        s=s,
        port_numbers=fields.pop("port_numbers", (1, 2)),
        reference=reference,
        measured_impedance=reference.copy(),
        **fields,
    )


def turned(result, port):
    """``result`` with one port's mode taken the other way round."""
    signs = np.array([-1.0 if number == port else 1.0 for number in result.port_numbers])
    s = result.s * signs[None, :, None] * signs[None, None, :]
    return SParameters(**{**result.__dict__, "s": s})


def worst(comparison, name):
    return next(term for term in comparison.terms if term.name == name)


class TestWhatIsStated:
    def test_one_matrix_against_itself_differs_by_nothing(self):
        comparison = compare(line(), "openEMS", line(), "Palace")
        assert [term.worst for term in comparison.terms] == [0.0] * 4
        assert all("differs by at most 0 at" in comparison.line(term) for term in comparison.terms)

    def test_the_worst_difference_is_found_where_it_is_with_both_levels(self):
        first = line()
        s = first.s.copy()
        s[4, 1, 0] += 0.01
        second = SParameters(**{**first.__dict__, "s": s})
        term = worst(compare(first, "openEMS", second, "Palace"), "S21")
        assert term.worst == pytest.approx(0.01, rel=1e-9)
        assert term.at == BAND[4]
        assert term.first == pytest.approx(abs(first.s[4, 1, 0]), rel=1e-12)
        assert term.second == pytest.approx(abs(s[4, 1, 0]), rel=1e-12)

    def test_the_difference_is_of_the_terms_and_not_of_their_decibels(self):
        """Two reflections deep in a null are 20 dB apart and differ by next to
        nothing, which is how far apart they are."""
        comparison = compare(line(reflection=1e-4), "openEMS", line(reflection=1e-5), "Palace")
        assert worst(comparison, "S11").worst == pytest.approx(9e-5, rel=1e-9)

    def test_each_line_names_the_term_the_frequency_and_both_solvers(self):
        comparison = compare(line(), "openEMS", line(transmission=0.8), "Palace")
        said = comparison.line(worst(comparison, "S21"))
        assert said.startswith("S21 differs by at most 0.1 (-20.0 dB) at ")
        assert "is 0.9 in what openEMS solved and 0.8 in what Palace solved" in said


class TestPorts:
    def test_one_set_of_ports_listed_in_another_order_is_one_set(self):
        first = line(port_numbers=(2, 5))
        second = SParameters(
            **{
                **first.__dict__,
                "s": first.s[:, ::-1, ::-1],
                "port_numbers": (5, 2),
            }
        )
        comparison = compare(first, "openEMS", second, "Palace")
        assert comparison.port_numbers == (2, 5)
        assert [term.worst for term in comparison.terms] == [0.0] * 4


class TestWhatIsRefused:
    def test_two_sets_of_ports(self):
        with pytest.raises(Incomparable, match=r"ports 1 and 2 .* ports 1 and 3"):
            compare(line(), "openEMS", line(port_numbers=(1, 3)), "Palace")

    def test_two_drawings(self):
        with pytest.raises(Incomparable, match="two different drawings"):
            compare(
                line(provenance={DRAWING: "a"}),
                "openEMS",
                line(provenance={DRAWING: "b"}),
                "Palace",
            )

    def test_a_result_that_does_not_say_what_it_was_solved_from_is_compared_and_said(self):
        comparison = compare(line(provenance={DRAWING: "a"}), "openEMS", line(), "Palace")
        assert comparison.notes() == [
            "what Palace solved does not say what drawing it was solved from, so the two "
            "are not known to be of one drawing"
        ]

    def test_two_results_of_one_drawing_say_nothing_about_it(self):
        one = {DRAWING: "a"}
        assert (
            compare(line(provenance=one), "openEMS", line(provenance=one), "Palace").notes() == []
        )

    def test_no_frequency_in_common(self):
        with pytest.raises(Incomparable, match="share no frequency"):
            compare(line(), "openEMS", line(frequency=BAND + 1e6), "Palace")

    def test_no_term_both_measured(self):
        first, second = line(), line()
        s1, s2 = first.s.copy(), second.s.copy()
        s1[:, :, 1] = np.nan
        s2[:, :, 0] = np.nan
        with pytest.raises(Incomparable, match="no term was measured by both"):
            compare(
                SParameters(**{**first.__dict__, "s": s1, "driven": (1,)}),
                "openEMS",
                SParameters(**{**second.__dict__, "s": s2, "driven": (2,)}),
                "Palace",
            )


class TestReferencing:
    def test_each_port_at_its_own_impedance_is_one_referencing_whatever_its_number(self):
        """A wave impedance and a power-voltage impedance of one mode are two
        numbers for one reference."""
        guide = np.full((BAND.size, 2), 528.6 + 0j)
        stated = np.full((BAND.size, 2), 424.3 + 0j)
        compare(
            line(reference=guide, self_referenced=(1, 2)),
            "openEMS",
            line(reference=stated, self_referenced=(1, 2)),
            "Palace",
        )

    def test_a_resistance_and_the_same_number_asked_for_are_one_referencing(self):
        compare(line(), "openEMS", line(self_referenced=(1, 2)), "Palace")

    def test_two_numbers_are_one_where_a_message_could_not_tell_them_apart(self):
        closer = np.full((BAND.size, 2), 50.0 * (1 + 3e-9) + 0j)
        compare(line(), "openEMS", line(reference=closer), "Palace")

    def test_a_number_against_a_mode_is_refused_naming_the_port(self):
        unstated = np.full((BAND.size, 2), np.nan + 0j)
        with pytest.raises(
            Incomparable, match=r"port 1 is referenced to 50 ohm .* to its own impedance"
        ):
            compare(
                line(),
                "openEMS",
                line(reference=unstated, self_referenced=(1, 2)),
                "Palace",
            )


class TestFrequencies:
    def test_only_the_frequencies_both_solved_are_compared_and_the_rest_are_counted(self):
        denser = np.linspace(20e9, 26e9, 13)
        comparison = compare(line(), "openEMS", line(frequency=denser), "Palace")
        assert np.array_equal(comparison.frequency, BAND)
        assert comparison.unshared == denser.size - BAND.size
        assert (
            f"{denser.size - BAND.size} frequencies solved by one of the two alone are not "
            "compared" in comparison.notes()
        )

    def test_one_frequency_is_one(self):
        comparison = compare(line(), "openEMS", line(frequency=np.append(BAND, 27e9)), "Palace")
        assert "one frequency solved by one of the two alone is not compared" in comparison.notes()

    def test_a_band_read_back_from_a_table_is_the_band_it_was_written_from(self):
        """A frequency read back from a solver's table carries the figures the
        table holds, and parts from the one computed by about one in a
        billion."""
        rounded = BAND * (1.0 + 1.2e-9)
        assert compare(line(), "openEMS", line(frequency=rounded), "Palace").unshared == 0

    def test_a_sweep_finer_than_the_tolerance_is_paired_point_for_point(self):
        """Each frequency is paired with its own and not with its neighbours, so
        a narrow sweep against itself differs by nothing."""
        narrow = 10e9 + np.arange(11) * 2e3
        comparison = compare(line(frequency=narrow), "openEMS", line(frequency=narrow), "Palace")
        assert np.array_equal(comparison.frequency, narrow)
        assert [term.worst for term in comparison.terms] == [0.0] * 4
        assert comparison.unshared == 0

    def test_a_frequency_held_twice_is_one_frequency(self):
        twice = np.append(BAND, BAND[2])
        comparison = compare(line(), "openEMS", line(frequency=twice), "Palace")
        assert np.array_equal(comparison.frequency, BAND)
        assert comparison.unshared == 0

    def test_every_frequency_shared_says_nothing_about_them(self):
        assert compare(line(), "openEMS", line(), "Palace").unshared == 0


class TestTheSignOfAMode:
    def test_a_port_signed_by_a_rule_is_turned_back_and_said_to_be(self):
        second = turned(line(provenance={SIGN_BY_RULE: [2]}), 2)
        comparison = compare(line(), "openEMS", second, "Palace")
        assert comparison.turned == ((2, "Palace"),)
        assert worst(comparison, "S21").worst == pytest.approx(0.0, abs=1e-15)
        assert comparison.notes()[0] == (
            "port 2 of what Palace solved was turned half a cycle, because Palace chose "
            "the sign of the mode there by a rule of its own"
        )

    def test_a_port_is_turned_in_the_matrix_whose_solver_chose_its_sign(self):
        """Filed second, openEMS is the second matrix, and the port Palace
        signed is still turned in what Palace solved."""
        first = turned(line(provenance={SIGN_BY_RULE: [2]}), 2)
        comparison = compare(first, "Palace", line(), "openEMS")
        assert comparison.turned == ((2, "Palace"),)
        assert np.array_equal(comparison.second, line().s)
        assert worst(comparison, "S21").worst == pytest.approx(0.0, abs=1e-15)
        assert comparison.notes()[0].startswith("port 2 of what Palace solved was turned")

    def test_a_port_whose_sign_the_drawing_fixes_is_left_as_it_came(self):
        comparison = compare(line(), "openEMS", turned(line(), 2), "Palace")
        assert comparison.turned == ()
        assert worst(comparison, "S21").worst == pytest.approx(1.8, rel=1e-9)

    def test_a_port_signed_by_a_rule_is_not_turned_where_that_brings_nothing_closer(self):
        comparison = compare(line(), "openEMS", line(provenance={SIGN_BY_RULE: [1, 2]}), "Palace")
        assert comparison.turned == ()
        assert not any("turned" in said for said in comparison.lines())

    def test_of_two_ways_of_turning_that_give_one_matrix_the_smaller_is_named(self):
        second = turned(line(provenance={SIGN_BY_RULE: np.array([1, 2])}), 2)
        assert compare(line(), "openEMS", second, "Palace").turned == ((1, "Palace"),)

    def test_the_reflections_stay_as_they_are(self):
        second = turned(line(provenance={SIGN_BY_RULE: [2]}), 2)
        comparison = compare(line(), "openEMS", second, "Palace")
        assert np.array_equal(comparison.second[:, 1, 1], second.s[:, 1, 1])
