# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A Touchstone file read into a matrix: ``Results/touchstone.py``.

Every file the workbench writes reads back as the matrix it was written from,
which is the round trip a user makes by exporting a run and importing it into
another document. A foreign file is held to closed forms of the format: a value
in each of its formats and units, a Z parameter normalised to the option line,
a power wave at a complex reference. What it does not read it refuses by name.
"""

import re

import numpy as np
import pytest

from Microwave.Results import touchstone
from Microwave.Results.sparameters import SParameters
from Microwave.Results.touchstone import OWN, STATED, Unreadable, declared, read


def written(tmp_path, result, name="table", per_point=False):
    """``result`` written by the workbench and read back, with the file's bytes."""
    path = result.write_touchstone(tmp_path / name, per_point=per_point)
    data = path.read_bytes()
    return read(path.name, data), data


def matrix(ports=2, points=7, reference=50.0, **fields):
    """Every term distinct, so a transposed read cannot pass."""
    frequency = np.linspace(1e9, 3e9, points)
    index = np.indices((points, ports, ports))
    s = 0.05 * (index[0] + 1) + 0.1 * index[1] + 0.01 * index[2] + 0.02j * (index[2] - index[1])
    z = np.broadcast_to(np.asarray(reference, dtype=float), (points, ports)).astype(complex)
    return SParameters(
        frequency=frequency,
        s=s.astype(complex),
        port_numbers=tuple(range(1, ports + 1)),
        reference=z,
        measured_impedance=z.copy(),
        **fields,
    )


def same(found, expected):
    np.testing.assert_array_equal(found.frequency, expected.frequency)
    np.testing.assert_array_equal(found.s, expected.s)
    np.testing.assert_array_equal(found.reference, expected.reference)


class TestWhatTheWorkbenchWritesReadsBack:
    @pytest.mark.parametrize("ports", [1, 2, 3, 4])
    def test_every_port_count(self, tmp_path, ports):
        result = matrix(ports=ports)
        got, _ = written(tmp_path, result)
        same(got.result, result)
        assert got.result.port_numbers == result.port_numbers

    def test_a_reference_other_than_fifty_ohm(self, tmp_path):
        result = matrix(reference=75.0)
        got, _ = written(tmp_path, result)
        same(got.result, result)
        assert got.suggested == STATED

    def test_a_reference_stated_at_each_point_is_suggested_as_the_ports_own(self, tmp_path):
        """A guide's reference referenced to its own mode moves across the band,
        and the file states it at each point."""
        guide = np.outer(np.linspace(400.0, 600.0, 7), [1.0, 1.1])
        result = matrix(reference=guide, self_referenced=(1, 2))
        got, _ = written(tmp_path, result, per_point=True)
        same(got.result, result)
        assert got.suggested == OWN
        assert declared(got.result, OWN).reference_description() == (result.reference_description())

    def test_one_reference_per_port_is_read_as_stated(self, tmp_path):
        result = matrix(reference=[50.0, 75.0])
        got, _ = written(tmp_path, result, per_point=True)
        same(got.result, result)
        assert got.suggested == STATED

    def test_a_column_completed_from_a_symmetry_reads_back_as_written(self, tmp_path):
        result = matrix(driven=(1,), derived=(2,), provenance={"symmetry": "mirror"})
        got, _ = written(tmp_path, result)
        same(got.result, result)

    def test_the_header_comes_back_as_comments(self, tmp_path):
        got, _ = written(tmp_path, matrix(provenance={"title": "Filter"}))
        assert any("Filter" in line for line in got.comments)
        assert any("FreeCAD Microwave" in line for line in got.comments)


def file(tmp_path, name, text):
    path = tmp_path / name
    path.write_bytes(text.encode("latin-1"))
    return read(name, path.read_bytes())


class TestAForeignFile:
    def test_magnitude_in_decibels_and_angle_in_degrees(self, tmp_path):
        got = file(tmp_path, "a.s1p", "# MHz S DB R 50\n100 -6.0205999 90\n")
        assert got.result.frequency == pytest.approx([100e6], rel=1e-15)
        assert got.result.s[0, 0, 0] == pytest.approx(0.5j, abs=1e-9)

    def test_magnitude_and_angle(self, tmp_path):
        got = file(tmp_path, "a.s1p", "# kHz S MA R 50\n1 0.5 180\n")
        assert got.result.frequency == pytest.approx([1e3], rel=1e-15)
        assert got.result.s[0, 0, 0] == pytest.approx(-0.5, abs=1e-12)

    def test_a_two_port_line_is_s11_s21_s12_s22(self, tmp_path):
        got = file(tmp_path, "a.s2p", "# GHz S RI R 50\n1 0.11 0 0.21 0 0.12 0 0.22 0\n")
        assert got.result.s[0].real.tolist() == [[0.11, 0.12], [0.21, 0.22]]

    def test_a_z_parameter_is_normalised_to_the_option_line(self, tmp_path):
        """Touchstone 1 normalises Z to R: 2 at R 75 is 150 ohm, which
        reflects (150 - 75) / (150 + 75)."""
        got = file(tmp_path, "z.s1p", "# GHz Z RI R 75\n1 2 0\n")
        assert got.result.s[0, 0, 0] == pytest.approx(1 / 3, abs=1e-12)
        assert "Z parameters turned into S parameters at 75 ohm" in got.said

    def test_a_reference_per_port_from_a_version_two_file(self, tmp_path):
        got = file(
            tmp_path,
            "v2.s2p",
            "[Version] 2.0\n# GHz S RI R 50\n[Number of Ports] 2\n"
            "[Two-Port Data Order] 21_12\n[Number of Frequencies] 1\n[Reference] 50 75\n"
            "[Network Data]\n1 0.1 0 0.9 0 0.9 0 0.1 0\n[End]\n",
        )
        assert got.result.reference.tolist() == [[50, 75]]
        assert got.result.reference_description() == "port 1: 50 ohm, port 2: 75 ohm"

    def test_the_noise_block_is_left_out_and_said_to_be(self, tmp_path):
        got = file(
            tmp_path,
            "amp.s2p",
            "# GHz S MA R 50\n1 0.1 0 0.2 0 0.3 0 0.4 0\n2 0.1 0 0.2 0 0.3 0 0.4 0\n"
            "1 0.8 0.5 45 0.3\n2 0.9 0.45 60 0.35\n",
        )
        assert got.result.frequency.tolist() == [1e9, 2e9]
        assert "noise parameters left out" in got.said

    def test_hfss_port_impedances_are_read_as_power_waves(self, tmp_path):
        """HFSS states its matrix in its own wave definition unless the file
        names another, and a matrix here is of power waves. At a complex
        reference Z0 a load Z reflects (Z - Z0) / (Z + Z0) as a travelling wave
        and (Z - conj(Z0)) / (Z + Z0) as a power wave."""
        z0, load = 50 + 20j, 80 - 10j
        travelling = (load - z0) / (load + z0)
        got = file(
            tmp_path,
            "hfss.s1p",
            f"# GHz S RI R 50\n1 {travelling.real!r} {travelling.imag!r}\n"
            f"! Port Impedance {z0.real} {z0.imag}\n",
        )
        assert got.result.s[0, 0, 0] == pytest.approx(
            (load - z0.conjugate()) / (load + z0), abs=1e-12
        )
        assert got.result.reference[0, 0] == z0

    def test_a_comment_in_utf_8_is_read_as_written(self, tmp_path):
        path = tmp_path / "a.s1p"
        path.write_bytes("! 50 Ω load\n# GHz S RI R 50\n1 0 0\n".encode())
        assert read(path.name, path.read_bytes()).comments == (" 50 Ω load",)

    def test_the_comments_ahead_of_the_option_line_are_kept(self, tmp_path):
        got = file(tmp_path, "a.s1p", "! VNA ZNB20\n! Cal: TRL\n# GHz S RI R 50\n1 0 0\n")
        assert [line.strip() for line in got.comments] == ["VNA ZNB20", "Cal: TRL"]


class TestWhatItRefuses:
    @pytest.mark.parametrize("kind", ["Y", "H", "G"])
    def test_parameters_whose_normalisation_the_format_leaves_open(self, tmp_path, kind):
        with pytest.raises(
            Unreadable,
            match=re.escape(
                f"{kind} parameters in a Touchstone 1 file. Only S parameters, and Z parameters "
                "in a Touchstone 1 file, can be imported. Export S parameters from the "
                "instrument or the simulator"
            ),
        ):
            file(tmp_path, "a.s2p", f"# GHz {kind} RI R 50\n1 0.1 0 0.2 0 0.3 0 0.4 0\n")

    def test_z_parameters_in_a_version_two_file(self, tmp_path):
        with pytest.raises(Unreadable, match="Z parameters in a Touchstone 2.0"):
            file(
                tmp_path,
                "a.s1p",
                "[Version] 2.0\n# GHz Z RI R 50\n[Number of Ports] 1\n"
                "[Number of Frequencies] 1\n[Network Data]\n1 60 0\n[End]\n",
            )

    def test_mixed_mode_data(self, tmp_path):
        zeros = " ".join(["0"] * 32)
        with pytest.raises(
            Unreadable,
            match=re.escape(
                "holds mixed-mode data. Only single-ended S parameters can be imported. Export "
                "single-ended S parameters from the instrument or the simulator"
            ),
        ):
            file(
                tmp_path,
                "a.s4p",
                "[Version] 2.0\n# GHz S RI R 50\n[Number of Ports] 4\n"
                "[Number of Frequencies] 1\n[Mixed-Mode Order] D2,1 D1,2 C2,1 C1,2\n"
                f"[Network Data]\n1 {zeros}\n[End]\n",
            )

    def test_a_word_where_a_number_belongs_names_its_line(self, tmp_path):
        with pytest.raises(Unreadable, match="line 3: holds 'x' where a number belongs"):
            file(tmp_path, "a.s1p", "! one\n# GHz S RI R 50\n1 0.1 x\n")

    def test_a_frequency_listed_twice(self, tmp_path):
        with pytest.raises(Unreadable, match="line 3: lists 1 after 1"):
            file(tmp_path, "a.s1p", "# GHz S RI R 50\n1 0.1 0\n1 0.2 0\n")

    def test_frequencies_that_do_not_rise(self, tmp_path):
        with pytest.raises(Unreadable, match="line 3: lists 1 after 2: frequencies rise"):
            file(tmp_path, "a.s1p", "# GHz S RI R 50\n2 0.1 0\n1 0.1 0\n")

    def test_a_file_of_no_points(self, tmp_path):
        with pytest.raises(Unreadable, match="no frequency point"):
            file(tmp_path, "a.s1p", "# GHz S RI R 50\n")

    def test_a_name_that_says_no_port_count(self, tmp_path):
        with pytest.raises(
            Unreadable,
            match=re.escape(
                "a.ts: the file name does not end in .sNp, where N is the number of ports. "
                "Rename the file to end in .sNp"
            ),
        ):
            file(tmp_path, "a.ts", "# GHz S RI R 50\n1 0.1 0\n")

    def test_data_that_does_not_fit_the_ports_its_name_says(self, tmp_path):
        with pytest.raises(Unreadable, match="3-port"):
            file(tmp_path, "a.s3p", "# GHz S RI R 50\n1 0.1 0 0.2 0 0.3 0 0.4 0\n")

    def test_words_in_an_information_block_are_not_data(self, tmp_path):
        got = file(
            tmp_path,
            "a.s1p",
            "[Version] 2.0\n# GHz S RI R 50\n[Number of Ports] 1\n[Begin Information]\n"
            "Part Number widget\n[End Information]\n[Number of Frequencies] 1\n"
            "[Network Data]\n1 0.1 0\n[End]\n",
        )
        assert got.result.s[0, 0, 0] == pytest.approx(0.1, abs=0.0)


class TestACommentIsNeverData:
    """Every case an adversarial tester read wrong through a parser that took
    comments for data or dropped data for noise."""

    def test_a_comment_naming_gamma_leaves_the_option_line_alone(self, tmp_path):
        got = file(tmp_path, "a.s1p", "! Gamma of a 50 ohm load\n# MHz S RI R 50\n100 0.1 0.2\n")
        assert got.result.frequency.tolist() == [1e8]
        assert got.result.s[0, 0, 0] == 0.1 + 0.2j

    def test_a_comment_naming_gamma_between_points_drops_no_point(self, tmp_path):
        got = file(
            tmp_path, "a.s1p", "# GHz S RI R 50\n1 0.1 0.2\n! Gamma tuned\n3 0.3 0.4\n4 0.5 0.6\n"
        )
        assert got.result.frequency.tolist() == [1e9, 3e9, 4e9]

    def test_a_comment_naming_port_impedance_ahead_of_the_option_line(self, tmp_path):
        got = file(
            tmp_path, "a.s1p", "! Port impedance 50 ohm everywhere\n# MHz S RI R 50\n100 0.1 0.2\n"
        )
        assert got.result.s[0, 0, 0] == 0.1 + 0.2j
        assert got.result.reference.tolist() == [[50]]

    def test_a_comment_on_the_option_line_states_no_reference(self, tmp_path):
        got = file(tmp_path, "a.s1p", "# GHz S MA ! 3 dB attenuator\n1 0.5 0\n")
        assert got.result.reference.tolist() == [[50]]

    @pytest.mark.parametrize("matrix_format", ["Lower", "Upper"])
    def test_a_triangle_in_either_two_port_order(self, tmp_path, matrix_format):
        got = file(
            tmp_path,
            "a.s2p",
            "[Version] 2.0\n# GHz S RI R 50\n[Number of Ports] 2\n"
            "[Two-Port Data Order] 21_12\n[Number of Frequencies] 1\n"
            f"[Matrix Format] {matrix_format}\n[Network Data]\n1 0.11 0 0.21 0 0.22 0\n[End]\n",
        )
        assert got.result.s[0].real.tolist() == [[0.11, 0.21], [0.21, 0.22]]

    def test_a_falling_frequency_in_a_two_port_is_refused_and_not_read_as_noise(self, tmp_path):
        """Noise lines hold five numbers, and these hold nine."""
        row = " 1 0 2 0 3 0 4 0"
        with pytest.raises(Unreadable, match="line 4: lists 2 after 3: frequencies rise"):
            file(tmp_path, "a.s2p", f"# GHz S RI R 50\n1{row}\n3{row}\n2{row}\n4{row}\n")

    def test_noise_may_start_at_the_last_network_frequency(self, tmp_path):
        got = file(
            tmp_path,
            "a.s2p",
            "# GHz S MA R 50\n2 .95 -26 3.57 157 .04 76 .66 -14\n"
            "22 .60 -144 1.30 40 .14 40 .56 -85\n22 .7 .64 69 .38\n",
        )
        assert got.result.frequency.tolist() == [2e9, 22e9]
        assert "noise parameters left out" in got.said

    def test_the_specifications_four_port_example(self, tmp_path):
        rows = (
            "5 0.60 161.24 0.40 -42.20 0.42 -66.58 0.53 -79.34\n"
            "0.40 -42.20 0.60 161.20 0.53 -79.34 0.42 -66.58\n"
            "0.42 -66.58 0.53 -79.34 0.60 161.24 0.40 -42.20\n"
            "0.53 -79.34 0.42 -66.58 0.40 -42.20 0.60 161.24\n"
        )
        got = file(tmp_path, "a.s4p", f"# GHZ S MA R 50\n{rows}")
        assert abs(got.result.s[0, 1, 1]) == pytest.approx(0.60, abs=1e-12)
        assert np.angle(got.result.s[0, 0, 2], deg=True) == pytest.approx(-66.58, abs=1e-9)

    @pytest.mark.parametrize(
        ("text", "why"),
        [
            ("# GHz S RI R 50\n1 0.5 0\nnan 0.4 0\n", "'nan' where a number belongs"),
            ("# GHz S RI R 50\n1 0.5 0\ninf 0.4 0\n", "'inf' where a number belongs"),
            ("# GHz S RI R 50\n1 nan 0\n", "'nan' where a number belongs"),
            ("# GHz S RI R 50\n1_0 0.5 0\n", "'1_0' where a number belongs"),
            ("# GHz S RI R -50\n1 0.5 0\n", "'-50' for R"),
            ("# GHz S RI R 50j\n1 0.5 0\n", "'50j' for R"),
            ("# GHz S RI R75\n1 0.5 0\n", "the option line holds 'R75'"),
            ("# THz S RI R 50\n1 0.5 0\n", "the option line holds 'THz'"),
            ("1 0.5 0.1\n# MHz S RI R 75\n", "before the option line"),
            ("# GHz S RI R 50\n-1 0.5 0\n", "negative frequency"),
            ("# GHz S RI R 50\n1 0.5\n", "partway through a frequency point"),
            ("# GHz S RI R 50\n1 0.5 0 0.4\n", "does not start a frequency point"),
            ("1 0.5 0\n", "before the option line"),
            ("! nothing\n", "no option line"),
        ],
    )
    def test_a_line_the_format_does_not_allow(self, tmp_path, text, why):
        with pytest.raises(Unreadable, match=re.escape(why)):
            file(tmp_path, "a.s1p", text)

    @pytest.mark.parametrize(
        ("body", "why"),
        [
            ("[Number of Frequencies] 5\n[Network Data]\n1 0.1 0\n[End]\n", "holds 1"),
            ("[Number of Frequencies] 1\n[Network Data]\n1 0.1 0\n[End]\n2 0.1 0\n", "after [End]"),
            ("[Number of Frequencies] 1\n[Network Data]\n1 0.1 0\n", "does not close with [End]"),
            ("[Unknown Thing] 3\n", "[Unknown Thing]"),
        ],
    )
    def test_a_version_two_file_that_says_otherwise_than_it_holds(self, tmp_path, body, why):
        with pytest.raises(Unreadable, match=re.escape(why)):
            file(tmp_path, "a.s1p", f"[Version] 2.0\n# GHz S RI R 50\n[Number of Ports] 1\n{body}")

    def test_a_version_two_two_port_states_its_data_order(self, tmp_path):
        with pytest.raises(Unreadable, match=r"no \[Two-Port Data Order\]"):
            file(
                tmp_path,
                "a.s2p",
                "[Version] 2.0\n# GHz S RI R 50\n[Number of Ports] 2\n[Number of Frequencies] 1\n"
                "[Network Data]\n1 0.1 0 0.2 0 0.3 0 0.4 0\n[End]\n",
            )

    def test_another_version_is_refused(self, tmp_path):
        with pytest.raises(Unreadable, match="Touchstone 2.1"):
            file(tmp_path, "a.s1p", "[Version] 2.1\n# GHz S RI R 50\n")

    def test_z_beside_hfss_port_impedances_is_refused(self, tmp_path):
        with pytest.raises(Unreadable, match="beside HFSS port impedances"):
            file(
                tmp_path,
                "a.s2p",
                "# GHz Z RI R 50\n1 2 0 1 0 1 0 2 0\n! Port Impedance 50 0 100 0\n",
            )

    def test_a_port_impedance_missing_from_a_point_is_refused(self, tmp_path):
        with pytest.raises(Unreadable, match="after 1 of its 2 points"):
            file(
                tmp_path, "a.s1p", "# GHz S RI R 50\n1 0.2 0.1\n! Port Impedance 40 0\n2 0.3 0.1\n"
            )

    def test_a_comment_naming_port_impedance_with_no_numbers_is_a_comment(self, tmp_path):
        got = file(tmp_path, "a.s1p", "# GHz S RI R 50\n1 0.2 0.1\n! Port impedance is 50 ohm\n")
        assert got.result.reference.tolist() == [[50]]

    def test_the_option_line_in_any_order_and_any_case(self, tmp_path):
        got = file(tmp_path, "a.s1p", "#\tr 75 ri MHZ s\n100 0.1 0.2\n")
        assert got.result.reference.tolist() == [[75]]
        assert got.result.frequency.tolist() == [1e8]

    def test_a_second_option_line_is_ignored(self, tmp_path):
        got = file(tmp_path, "a.s1p", "# GHz S RI R 50\n# MHz S MA R 75\n1 0.1 0.2\n")
        assert got.result.frequency.tolist() == [1e9]

    def test_a_byte_order_mark_and_carriage_returns(self, tmp_path):
        path = tmp_path / "a.s1p"
        path.write_bytes(b"\xef\xbb\xbf# GHz S RI R 50\r1 0.1 0.2\r2 0.3 0.4\r")
        assert read(path.name, path.read_bytes()).result.frequency.tolist() == [1e9, 2e9]


class TestWhatEachLineMayHold:
    """The rest of what the format allows and does not, line by line."""

    @pytest.mark.parametrize(
        "comment",
        [
            "# GHz S RI R 50\n! Port Impedance 40 0\n1 0.1 0\n",
            "# GHz S RI R 50\n1 0.1 0\n! Port Impedance\n",
        ],
    )
    def test_a_port_impedance_comment_out_of_place_or_empty_is_a_comment(self, tmp_path, comment):
        assert file(tmp_path, "a.s1p", comment).result.reference.tolist() == [[50]]

    def test_a_port_impedance_comment_in_a_version_two_file_is_a_comment(self, tmp_path):
        got = file(
            tmp_path,
            "a.s1p",
            "[Version] 2.0\n# GHz S RI R 50\n[Number of Ports] 1\n[Network Data]\n"
            "1 0.1 0\n! Port Impedance 40 0\n[End]\n",
        )
        assert got.result.reference.tolist() == [[50]]

    def test_a_port_impedance_no_port_can_have(self, tmp_path):
        with pytest.raises(Unreadable, match="line 3: states a port impedance whose real part"):
            file(tmp_path, "a.s1p", "# GHz S RI R 50\n1 0.1 0\n! Port Impedance -40 0\n")

    def test_two_port_impedances_after_one_point_and_none_after_the_next(self, tmp_path):
        with pytest.raises(Unreadable, match="needs one after each point"):
            file(
                tmp_path,
                "a.s1p",
                "# GHz S RI R\n1 0.1 0\n! Port Impedance 40 0\n! Port Impedance 40 0\n2 0.1 0\n",
            )

    def test_r_with_no_number_and_no_reference_at_each_point(self, tmp_path):
        with pytest.raises(Unreadable, match="R with no value"):
            file(tmp_path, "a.s1p", "# GHz S RI R\n1 0.1 0\n")

    @pytest.mark.parametrize(
        ("body", "why"),
        [
            ("[Number of Ports] 2\n", "states 2 ports, and the file name says 1"),
            ("[Number of Ports] 1\n[Two-Port Data Order] 11_22\n", "two-port data order '11_22'"),
            ("[Number of Ports] 1\n[Matrix Format] Diagonal\n", "matrix format 'Diagonal'"),
            ("[Number of Ports] 1\n[Reference] 50 75\n", "a reference for more than 1 ports"),
            ("[Number of Ports] 1\n1 0.1 0\n", "outside [Network Data]"),
            ("[Number of Ports] 1\n[Version] 2.0\n", "[Version] after the option line"),
        ],
    )
    def test_a_version_two_keyword_out_of_place(self, tmp_path, body, why):
        with pytest.raises(Unreadable, match=re.escape(why)):
            file(tmp_path, "a.s1p", f"[Version] 2.0\n# GHz S RI R 50\n{body}")

    @pytest.mark.parametrize(
        ("order", "s21"),
        [("12_21", 0.3), ("21_12", 0.2)],
    )
    def test_a_version_two_two_port_in_its_stated_order(self, tmp_path, order, s21):
        got = file(
            tmp_path,
            "a.s2p",
            f"[Version] 2.0\n# GHz S RI R 50\n[Number of Ports] 2\n[Two-Port Data Order] {order}\n"
            "[Network Data]\n1 0.1 0 0.2 0 0.3 0 0.4 0\n[End]\n",
        )
        assert got.result.s[0, 1, 0] == s21

    @pytest.mark.parametrize(
        ("matrix_format", "row"),
        [
            ("Lower", "1 11 0\n21 0 22 0\n31 0 32 0 33 0\n"),
            ("Upper", "1 11 0 21 0 31 0\n22 0 32 0\n33 0\n"),
        ],
    )
    def test_a_three_port_triangle_fills_its_mirror(self, tmp_path, matrix_format, row):
        got = file(
            tmp_path,
            "a.s3p",
            "[Version] 2.0\n# GHz S RI R 50\n[Number of Ports] 3\n"
            f"[Matrix Format] {matrix_format}\n[Network Data]\n{row}[End]\n",
        )
        assert got.result.s[0].real.tolist() == [[11, 21, 31], [21, 22, 32], [31, 32, 33]]


class TestWhatASecondAttackFound:
    """Every case the second adversarial tester read wrong or refused for the
    wrong reason."""

    @pytest.mark.parametrize(
        "name,text",
        [
            ("z.s1p", "# GHz Z RI R 50\n1 -1 0\n"),
            ("z.s2p", "# GHz Z RI R 50\n1 0 0 1 0 1 0 0 0\n"),
        ],
    )
    def test_a_z_with_no_s(self, tmp_path, name, text):
        with pytest.raises(Unreadable, match="z \\+ 1 is singular"):
            file(tmp_path, name, text)

    def test_one_port_lines_in_a_file_named_for_two_ports(self, tmp_path):
        lines = "".join(f"{k} 0.{k} -{k}0\n" for k in range(1, 7))
        with pytest.raises(
            Unreadable, match="line 2: starts a point that runs over more than one line"
        ):
            file(tmp_path, "a.s2p", f"# GHz S MA R 50\n{lines}")

    def test_a_row_of_a_four_port_that_does_not_open_its_line(self, tmp_path):
        row = " 0.1 0 0.2 0 0.3 0 0.4 0"
        text = f"# GHz S RI R 50\n1{row}{row}\n{row}\n{row}\n"
        with pytest.raises(Unreadable, match="line 2: does not start row 2"):
            file(tmp_path, "a.s4p", text)

    @pytest.mark.parametrize("mark", [b"\x0c", b"\x0b", b"\x1c", "\u2028".encode(), b"\x85"])
    def test_a_comment_holding_what_other_text_breaks_lines_at(self, tmp_path, mark):
        path = tmp_path / "a.s1p"
        path.write_bytes(b"# GHz S RI R 50\n1 0.5 0 ! note" + mark + b" 2 0.25 0\n")
        assert read(path.name, path.read_bytes()).result.frequency.tolist() == [1e9]

    @pytest.mark.parametrize(
        "text",
        [
            "# GHz S RI R 50\n1 1e999 0\n",
            "# GHz S RI R 1e999\n1 0.5 0\n",
            "# GHz S DB R 50\n1 1e5 0\n",
            "# GHz S RI R 50\n1e300 0.5 0\n",
        ],
    )
    def test_a_value_too_large_to_hold(self, tmp_path, text):
        with pytest.raises(Unreadable, match="too large to hold|positive number"):
            file(tmp_path, "a.s1p", text)

    def test_a_port_impedance_with_no_space_after_its_name(self, tmp_path):
        got = file(tmp_path, "a.s1p", "# GHz S RI R 50\n1 0.1 0\n! Port Impedance75 0\n")
        assert got.result.reference.tolist() == [[75]]

    def test_a_port_impedance_followed_by_a_comment(self, tmp_path):
        got = file(tmp_path, "a.s1p", "# GHz S RI R 50\n1 0.1 0\n! Port Impedance 75 0 ! extra\n")
        assert got.result.reference.tolist() == [[75]]

    def test_a_port_impedance_of_the_wrong_count(self, tmp_path):
        with pytest.raises(
            Unreadable, match="states 2 numbers for the port impedances of a 2-port"
        ):
            file(tmp_path, "a.s2p", "# GHz S RI R 50\n1 0 0 1 0 1 0 0 0\n! Port Impedance 75 0\n")

    def test_a_port_impedance_too_large_to_hold(self, tmp_path):
        with pytest.raises(Unreadable, match="a port impedance too large to hold"):
            file(tmp_path, "a.s1p", "# GHz S RI R 50\n1 0.1 0\n! Port Impedance 1e999 0\n")

    def test_a_one_port_holds_no_noise(self, tmp_path):
        with pytest.raises(Unreadable, match="line 3: lists 0.5 after 1: frequencies rise"):
            file(tmp_path, "a.s1p", "# GHz S RI R 50\n1 0.1 0\n0.5 1 2 3 4\n")

    def test_a_noise_line_of_other_than_five_numbers(self, tmp_path):
        with pytest.raises(Unreadable, match="line 3: is a line of noise parameters"):
            file(tmp_path, "a.s2p", "# GHz S MA R 50\n2 1 0 1 0 1 0 1 0\n1 1 0.5 45\n")

    def test_utf_16_text(self, tmp_path):
        path = tmp_path / "a.s1p"
        path.write_bytes("# GHz S RI R 50\n1 0.1 0\n".encode("utf-16"))
        with pytest.raises(
            Unreadable,
            match=re.escape("holds a NUL byte, as UTF-16 text does. Save the file as ASCII text"),
        ):
            read(path.name, path.read_bytes())

    def test_a_byte_order_mark_ahead_of_latin_1(self, tmp_path):
        path = tmp_path / "a.s1p"
        path.write_bytes(b"\xef\xbb\xbf! \xb5m\n# GHz S RI R 50\n1 0.1 0\n")
        assert read(path.name, path.read_bytes()).result.frequency.tolist() == [1e9]

    def test_an_information_block_left_open(self, tmp_path):
        with pytest.raises(Unreadable, match=re.escape("holds no [End Information]")):
            file(tmp_path, "a.s1p", "[Version] 2.0\n# GHz S RI R 50\n[Begin Information]\nx\n")

    def test_data_after_a_reference_with_no_network_data_keyword(self, tmp_path):
        with pytest.raises(Unreadable, match=re.escape("outside [Network Data]")):
            file(
                tmp_path,
                "a.s1p",
                "[Version] 2.0\n# GHz S RI R 50\n[Number of Ports] 1\n[Reference] 50\n1 0.1 0\n",
            )


class TestWhatAThirdAttackFound:
    """The cases a third adversarial tester refuted the reader with."""

    def test_a_port_count_too_long_to_be_one(self):
        with pytest.raises(Unreadable, match="the file name does not end in .sNp"):
            read("a.s" + "9" * 5000 + "p", b"# GHz S RI R 50\n1 0.1 0\n")

    def test_a_count_too_long_to_be_one(self, tmp_path):
        with pytest.raises(Unreadable, match="where a count belongs"):
            file(
                tmp_path, "a.s1p", "[Version] 2.0\n# GHz S RI R 50\n[Number of Ports] " + "1" * 5000
            )

    @pytest.mark.parametrize("keyword", ["Network Data", "End", "End Information"])
    def test_a_value_after_a_keyword_that_takes_none(self, tmp_path, keyword):
        with pytest.raises(Unreadable, match=re.escape(f"after [{keyword}], which takes no value")):
            file(
                tmp_path,
                "a.s1p",
                "[Version] 2.0\n# GHz S RI R 50\n[Number of Ports] 1\n[Begin Information]\n"
                f"[End Information]\n[Network Data]\n[{keyword}] 1 0.9 0\n2 0.4 0\n[End]\n",
            )

    @pytest.mark.parametrize(
        ("line", "s11"),
        [("1 1e155 0", 1.0), ("1 1e-300 0", -1.0)],
    )
    def test_a_one_port_z_of_any_size(self, tmp_path, line, s11):
        got = file(tmp_path, "z.s1p", f"# GHz Z RI R 50\n{line}\n")
        assert got.result.s[0, 0, 0] == pytest.approx(s11, abs=1e-12)

    def test_the_point_with_no_s_is_the_one_named(self, tmp_path):
        with pytest.raises(Unreadable, match="at 2 GHz that cannot be converted"):
            file(tmp_path, "z.s1p", "# GHz Z RI R 50\n1 2 0\n2 -1 0\n")

    def test_two_opens_of_any_size(self, tmp_path):
        got = file(tmp_path, "z.s2p", "# GHz Z RI R 50\n1 1e160 0 0 0 0 0 1e160 0\n")
        np.testing.assert_allclose(got.result.s[0], np.eye(2), atol=1e-12)

    @pytest.mark.parametrize("value", ["1e15", "1e16"])
    def test_a_z_whose_s_cannot_be_told_to_six_figures(self, tmp_path, value):
        """A shunt open, all four terms equal: z + 1 is singular to within its
        rounding."""
        line = " ".join([f"{value} 0"] * 4)
        with pytest.raises(Unreadable, match="to S at the option line's R to six figures"):
            file(tmp_path, "z.s2p", f"# GHz Z RI R 50\n1 {line}\n")

    def test_a_shunt_open_of_moderate_size_is_nearly_a_thru(self, tmp_path):
        line = " ".join(["1e4 0"] * 4)
        got = file(tmp_path, "z.s2p", f"# GHz Z RI R 50\n1 {line}\n")
        z = 1e4
        # The solve loses the figures the condition number of z + 1 has.
        assert got.result.s[0, 1, 0] == pytest.approx(2 * z / (2 * z + 1), rel=1e-9)

    @pytest.mark.parametrize(
        ("option", "word"),
        [("# GHz MHz S RI", "'MHz'"), ("# S Z RI", "'Z'"), ("# RI R 50 R 75", "'R'")],
    )
    def test_an_option_named_twice(self, tmp_path, option, word):
        with pytest.raises(Unreadable, match=f"names {word} and another value of the same kind"):
            file(tmp_path, "a.s1p", f"{option}\n1 0.1 0\n")

    def test_a_negative_magnitude(self, tmp_path):
        with pytest.raises(Unreadable, match="negative magnitude"):
            file(tmp_path, "a.s1p", "# GHz S MA R 50\n1 -0.5 0\n")

    @pytest.mark.parametrize("space", ["\xa0", "\u3000", "\x0b"])
    def test_a_space_the_format_does_not_name(self, tmp_path, space):
        path = tmp_path / "a.s1p"
        path.write_bytes(f"# GHz S RI R 50\n{space}1{space}0.1 0\n".encode())
        with pytest.raises(Unreadable, match="where a number belongs"):
            read(path.name, path.read_bytes())

    def test_a_three_port_frequency_alone_on_its_line(self, tmp_path):
        row = "0.1 0 0.2 0 0.3 0"
        with pytest.raises(Unreadable, match="line 2: holds a frequency alone"):
            file(tmp_path, "a.s3p", f"# GHz S RI R 50\n1\n{row}\n{row}\n{row}\n")


class TestTheDeclaration:
    def test_as_the_file_states_changes_nothing(self):
        result = matrix()
        assert declared(result, STATED) is result

    def test_its_own_impedance_moves_no_number_of_the_matrix(self):
        result = matrix()
        np.testing.assert_array_equal(declared(result, OWN).s, result.s)

    def test_its_own_impedance_drops_a_number_the_file_states_once(self):
        """The option line's 50 over a TRL calibration is not the line's
        impedance, so the port is left with none."""
        own = declared(matrix(), OWN)
        assert own.self_referenced == (1, 2)
        assert not np.isfinite(own.reference).any()
        assert own.reference_description() == "each port's own impedance"

    def test_its_own_impedance_keeps_numbers_stated_at_each_point(self, tmp_path):
        """Each port keeps them, a lumped port's 50 that does not move beside a
        guide's that does."""
        stated = np.stack([np.full(7, 50.0), np.linspace(400.0, 600.0, 7)], axis=1)
        got, _ = written(tmp_path, matrix(reference=stated), per_point=True)
        np.testing.assert_array_equal(declared(got.result, OWN).reference, stated)

    def test_anything_else_is_refused(self):
        with pytest.raises(ValueError, match="no such declaration"):
            declared(matrix(), "Fixed impedance")


class TestTheFileItself:
    def test_the_digest_names_its_kind(self):
        assert touchstone.digest(b"abc").startswith("sha256:ba7816bf")

    @pytest.mark.parametrize("data", ["! Ω\n".encode(), "! \xb5\n".encode("latin-1")])
    def test_the_text_gives_back_the_bytes(self, data):
        text, encoding = touchstone.text_of(data)
        assert text.encode(encoding) == data
