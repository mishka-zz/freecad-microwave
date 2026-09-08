# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What has to be true of the gate report, without running a single gate.

The report's whole value is that its standing part - which gate, what it is
scored against, whose bar it is - stays true while the numbers come out of the
run. So the standing part is what is checked here: that it covers the gates on
disk in both directions, and that the reading of a run's output picks up
everything a gate said and nothing else.
"""

from __future__ import annotations

import ast
import subprocess

import pytest

from tests import gate_report

#: Two rows to render, so that a section can be told from its neighbour.
GATE, OTHER = gate_report.GATES[0], gate_report.GATES[1]


def test_every_acceptance_file_has_a_row():
    """A gate added without a row would print its figures under no claim at all,
    which is the one thing this report exists to prevent."""
    on_disk = {path.name for path in gate_report.TESTS.glob("test_acceptance_*.py")}
    described = {gate.path.name for gate in gate_report.GATES}
    assert on_disk == described


def test_no_row_outlives_its_file():
    for gate in gate_report.GATES:
        assert gate.path.exists(), f"{gate.name} is described but not there"


def test_no_gate_is_described_twice():
    names = [gate.name for gate in gate_report.GATES]
    assert len(set(names)) == len(names)


def test_every_row_says_which_activity_it_is():
    """The distinction the raw output cannot make, and the reason for the file.
    Held against the three the module defines rather than against "not empty",
    so a fourth spelling of one of them fails here."""
    allowed = {gate_report.VERIFICATION, gate_report.VALIDATION, gate_report.IDENTITY}
    for gate in gate_report.GATES:
        assert gate.activity in allowed

    exact = [gate for gate in gate_report.GATES if gate.activity == gate_report.VERIFICATION]
    assert exact, "a suite with no exact reference in it would need a different report"


def test_every_row_says_what_decides_it():
    allowed = {
        gate_report.STUDIED,
        gate_report.EXTRAPOLATED,
        gate_report.DECLARED,
        gate_report.COMPOSED,
        gate_report.GRADED,
    }
    for gate in gate_report.GATES:
        assert gate.criterion in allowed

    studied = [gate for gate in gate_report.GATES if gate.criterion == gate_report.STUDIED]
    assert studied, (
        "no gate computes the interval it is held to, so every bar in this suite "
        "is one somebody chose - which is the thing the report exists to expose"
    )


def test_every_row_says_what_it_is_scored_against():
    for gate in gate_report.GATES:
        assert gate.reference.strip()
        assert gate.isolates.strip()


def test_a_gate_scored_against_a_fit_is_never_called_verification():
    """The two are decided together and by hand, so they can disagree. A row
    reading "exact" beside an activity of validation, or the reverse, is the
    error the whole page would then repeat."""
    for gate in gate_report.GATES:
        exact = gate.reference.startswith("exact")
        called_exact = gate.activity == gate_report.VERIFICATION
        assert exact == called_exact, (
            f"{gate.name} is scored against {gate.reference!r} and calls itself {gate.activity}"
        )


def test_only_a_validation_gate_composes_a_validation_uncertainty():
    """The three uncertainties in quadrature are what you compose when the
    reference has an error of its own. Composing one against an exact reference
    would be inventing a term for it."""
    for gate in gate_report.GATES:
        if gate.criterion == gate_report.COMPOSED:
            assert gate.activity == gate_report.VALIDATION, (
                f"{gate.name} composes a validation uncertainty and is {gate.activity}"
            )


def test_a_gate_does_what_its_criterion_says_it_does():
    """The row and the file are written apart, so they can drift - and this is
    the drift that matters, because the criterion is the part a reader uses to
    decide how much a figure is worth.

    Checked against the *call* rather than against prose, since each criterion
    has exactly one thing it means the gate does: an interval is asked whether it
    covers the reference, an extrapolated limit comes off a study, a composed
    uncertainty is assembled and compared against, a graded curve is scored on
    the published scale. A row that claims one of those while the file makes no
    such call is a claim about method that nothing performs.
    """
    performs = {
        gate_report.STUDIED: ".covers(",
        gate_report.EXTRAPOLATED: "convergence.uncertainty_of",
        gate_report.COMPOSED: "validation.Comparison",
        gate_report.GRADED: "fsv.compare",
    }
    for gate in gate_report.GATES:
        call = performs.get(gate.criterion)
        if call is None:
            continue
        assert call in gate.path.read_text(), (
            f"{gate.name} is reported as deciding by {gate.criterion!r}, "
            f"and nothing in {gate.path.name} calls {call}"
        )


def test_a_declared_bound_is_not_quietly_doing_more():
    """The other direction, and the weaker half: a gate that has grown a
    refinement study should stop being described as holding a single mesh."""
    for gate in gate_report.GATES:
        if gate.criterion != gate_report.DECLARED:
            continue
        assert "convergence.uncertainty_of" not in gate.path.read_text(), (
            f"{gate.name} runs a refinement study and is still reported as a "
            "bound argued on a single mesh"
        )


class TestReadingARunBack:
    OUTPUT = "\n".join(
        [
            "collected 3 items",
            "",
            "GATE stripline gap-10: cell 0.2000 mm, Z 49.5367 ohm",
            "a line that merely mentions GATE in passing",
            "  GATE stripline order: the error falls as the cell to the power 0.94",
            "...",
            "3 passed in 12.34s",
        ]
    )

    def test_it_takes_the_figures_and_leaves_the_prose(self):
        assert gate_report.readings_from(self.OUTPUT) == (
            "stripline gap-10: cell 0.2000 mm, Z 49.5367 ohm",
            "stripline order: the error falls as the cell to the power 0.94",
        )

    def test_it_finds_the_outcome(self):
        assert gate_report.outcome_from(self.OUTPUT) == "3 passed in 12.34s"

    def test_a_run_that_said_nothing_has_no_outcome_rather_than_a_wrong_one(self):
        assert gate_report.outcome_from("") == "no outcome reported"

    @pytest.mark.parametrize(
        "summary", ("1 failed, 2 passed in 3.00s", "9 skipped in 0.40s", "2 errors in 1.00s")
    )
    def test_it_finds_an_unhappy_outcome_as_readily_as_a_happy_one(self, summary):
        """The report is worth least when everything passed, so the reading has
        to survive the cases it is actually for."""
        assert gate_report.outcome_from(f"...\n{summary}") == summary


def run(gate, *lines, summary, code=0):
    """One gate's run, shaped the way ``pytest -q -s`` writes one.

    Through the output rather than around it, because that is the only thing a
    real run hands over and a reading assembled by hand can describe a run that
    could not happen - figures on a file that skipped, a failure with a passing
    summary.

    Under ``-q`` pytest writes one progress character per test with no newline,
    so every print that does not open with one arrives with that character in
    front of it. A helper writing clean lines tests the reader against output no
    run produces.
    """
    printed = "".join(("" if first else ".") + f"GATE {line}\n" for first, line in enumerate(lines))
    return gate_report.Reading(gate=gate, output=f"{printed}.\n{summary}\n", code=code, seconds=1.0)


class TestWhatTheReportSays:
    @pytest.fixture(scope="class")
    def rendered(self):
        readings = [
            run(gate_report.GATES[0], "waveguide something = 1.0", summary="17 passed in 14.00s"),
            run(gate_report.GATES[1], summary="9 skipped in 0.40s"),
        ]
        return gate_report.render(readings)

    def test_every_figure_reaches_the_page(self, rendered):
        assert "waveguide something = 1.0" in rendered

    def test_each_gate_carries_its_reference_its_activity_and_its_criterion(self, rendered):
        for gate in gate_report.GATES[:2]:
            assert gate.reference in rendered
            assert gate.activity in rendered
            assert gate.criterion in rendered

    def test_a_skipped_gate_says_so_rather_than_reading_as_silent(self, rendered):
        """An empty section and a section saying nothing ran look the same to a
        reader and mean opposite things. Held against what the module says for
        that verdict rather than against a sentence quoted here."""
        assert gate_report.SAID[gate_report.MEASURED_NOTHING] in rendered

    def test_the_page_states_where_its_numbers_came_from(self, rendered):
        assert "tests.gate_report" in rendered


class TestAClauseBecomesASentence:
    """The table holds clauses, because they read as clauses under a bold
    lead-in; the page needs them to start a line as well."""

    def test_it_capitalises_and_stops(self):
        assert gate_report.sentence("two round walls") == "Two round walls."

    def test_it_does_not_stop_twice(self):
        assert gate_report.sentence("a guide on two axes.") == "A guide on two axes."

    def test_it_leaves_the_rest_of_the_words_alone(self):
        """Several of these carry a word that must not be lowered or raised -
        a mode name, a formula's author, an axis."""
        assert gate_report.sentence("what Hammerstad is worth on x") == (
            "What Hammerstad is worth on x."
        )


class TestEveryGateHereDoesPrintAFigure:
    """The invariant the "printed nothing" verdict rests on.

    That verdict calls a passing, figureless gate a gate that stopped measuring,
    which would be wrong of a gate that legitimately asserts in silence - so
    whether any such gate exists is asked of the files rather than assumed.
    """

    @staticmethod
    def literals(path):
        """Every string literal in a file, f-strings included."""
        found = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.append(node.value)
            elif isinstance(node, ast.JoinedStr):
                found += [
                    part.value
                    for part in node.values
                    if isinstance(part, ast.Constant) and isinstance(part.value, str)
                ]
        return found

    def test_no_gate_asserts_in_silence(self):
        for gate in gate_report.GATES:
            printed = [text for text in self.literals(gate.path) if gate_report.MARKER in text]
            assert printed, (
                f"{gate.name} prints no figure, so a passing run of it would be "
                f"reported as a gate that stopped measuring"
            )


class TestWhatARunIsWorth:
    """Each state a run can be in, built from what pytest would write.

    Assembled through the output because that is the only thing a real run hands
    over: a verdict read off a reading somebody constructed proves nothing about
    a reading the tool makes for itself.
    """

    def test_a_gate_that_passed_and_printed_is_the_only_evidence(self):
        assert run(GATE, "z = 49.5", summary="3 passed in 12.34s").verdict == gate_report.SOLVED

    def test_a_failure_is_a_failure_even_where_most_of_it_passed(self):
        """A gate prints before it asserts, so a failing run carries figures
        and differs from a passing one in nothing a reader scans for but
        pytest's own last line."""
        reading = run(GATE, "z = 49.5", summary="1 failed, 2 passed in 3.00s")
        assert reading.verdict == gate_report.FAILED
        assert reading.lines == ("z = 49.5",)

    def test_an_error_is_not_read_as_a_pass(self):
        assert run(GATE, summary="2 errors in 1.00s").verdict == gate_report.FAILED
        assert run(GATE, summary="1 error in 1.00s").verdict == gate_report.FAILED

    def test_a_machine_that_cannot_solve_is_told_apart_from_a_broken_gate(self):
        assert run(GATE, summary="9 skipped in 0.40s").verdict == gate_report.MEASURED_NOTHING

    def test_the_passes_that_need_no_engine_do_not_make_it_a_partial_run(self):
        """Every gate holds tests that need no engine, so a machine that cannot
        solve reports passes beside its skips. A verdict keyed on the skips alone
        calls that a partial run and lays a missing engine at the workbench's
        door."""
        assert run(GATE, summary="3 passed, 14 skipped in 0.67s").verdict == (
            gate_report.MEASURED_NOTHING
        )

    def test_a_gate_that_passed_and_said_nothing_is_not_a_pass(self):
        assert run(GATE, summary="3 passed in 12.34s").verdict == gate_report.SILENT

    def test_a_file_that_measured_some_of_it_is_not_read_as_a_whole_one(self):
        """It prints figures and passes what it ran, so it reads as an ordinary
        result, and which subset ran is nowhere on the page."""
        reading = run(GATE, "z = 49.5", summary="1 passed, 8 skipped in 4.00s")
        assert reading.verdict == gate_report.PART_RAN

    def test_a_run_that_never_reported_is_not_read_as_empty(self):
        nothing = gate_report.Reading(gate=GATE, output="Traceback ...", code=1, seconds=1.0)
        assert nothing.verdict == gate_report.NOTHING_RAN

    def test_a_count_of_expected_failures_is_not_a_count_of_failures(self):
        """`xfailed` carries the word and is the opposite outcome."""
        assert run(GATE, "z = 49.5", summary="1 xfailed, 2 passed in 3.00s").verdict == (
            gate_report.SOLVED
        )


class TestTheStatusSaysWhetherToBelieveThePage:
    """The half a reader cannot supply. CI runs the fast tier and no gate, so
    this exit status is the only thing that fails when the workbench stops
    agreeing with the physics."""

    def test_only_every_gate_solving_is_success(self):
        assert gate_report.status_of([run(GATE, "z", summary="3 passed in 1.00s")]) == 0

    def test_a_single_failure_among_many_passes_still_fails(self):
        assert (
            gate_report.status_of(
                [
                    run(GATE, "z", summary="3 passed in 1.00s"),
                    run(OTHER, "y", summary="1 failed, 2 passed in 1.00s"),
                ]
            )
            == 1
        )

    def test_a_run_that_measured_nothing_fails_differently(self):
        """Not success, because nothing was measured; not a failure of the
        workbench either, because nothing was asked of it."""
        assert gate_report.status_of([run(GATE, summary="9 skipped in 0.40s")]) == 3

    def test_one_gate_solving_and_one_measuring_nothing_is_neither_case(self):
        assert (
            gate_report.status_of(
                [
                    run(GATE, "z", summary="3 passed in 1.00s"),
                    run(OTHER, summary="9 skipped in 0.40s"),
                ]
            )
            == 1
        )

    def test_a_gate_that_printed_nothing_fails_the_run(self):
        assert gate_report.status_of([run(GATE, summary="3 passed in 1.00s")]) == 1


class TestAFailedGateIsNotQuotedAsAMeasurement:
    @pytest.fixture(scope="class")
    def rendered(self):
        return gate_report.render(
            [
                run(GATE, "waveguide z = 1.0", summary="1 failed, 2 passed in 3.00s"),
                run(OTHER, "coax z = 50.0", summary="4 passed in 9.00s"),
            ]
        )

    def test_the_page_says_which_gate_failed(self, rendered):
        assert f"{GATE.name} - {gate_report.FAILED}" in rendered

    def test_the_figures_are_labelled_before_they_are_quoted(self, rendered):
        """They are still worth printing - the number that failed is what a
        reader came for - but not under the claim they no longer support."""
        assert gate_report.SAID[gate_report.FAILED] in rendered
        assert rendered.index(gate_report.SAID[gate_report.FAILED]) < rendered.index(
            "waveguide z = 1.0"
        )

    def test_the_gate_that_did_pass_is_not_labelled_with_it(self, rendered):
        after = rendered[rendered.index(f"## {OTHER.name}") :]
        assert gate_report.SAID[gate_report.FAILED] not in after

    def test_the_verdict_is_in_the_table_as_well_as_in_the_prose(self, rendered):
        assert f"| {GATE.name} | {gate_report.FAILED} |" in rendered


class TestWhatDidNotRunReadsDifferentlyFromWhatSaidNothing:
    """Two states a blank section covers equally, meaning opposite things: one
    is a machine that could not solve, the other a gate that stopped measuring.
    Neither is legible as an absence."""

    @staticmethod
    def section(rendered, gate):
        after = rendered[rendered.index(f"## {gate.name}") :]
        return after

    def test_each_says_which_it_is(self):
        rendered = gate_report.render(
            [
                run(GATE, summary="9 skipped in 0.40s"),
                run(OTHER, summary="3 passed in 12.34s"),
            ]
        )
        assert gate_report.SAID[gate_report.MEASURED_NOTHING] in self.section(rendered, GATE)
        assert gate_report.SAID[gate_report.SILENT] in self.section(rendered, OTHER)

    def test_the_headline_answers_before_the_table(self):
        rendered = gate_report.render([run(GATE, summary="9 skipped in 0.40s")])
        assert rendered.index(gate_report.MEASURED_NOTHING) < rendered.index("| gate |")

    def test_a_page_of_solved_gates_says_so(self):
        assert "Every gate solved and passed." in gate_report.render(
            [run(GATE, "z", summary="3 passed in 1.00s")]
        )


class TestWhatPytestWritesRatherThanWhatItMeant:
    """The reader is given the output of a real run, which is not tidy. Each
    shape here is one ``pytest -q -s`` produces."""

    def test_a_figure_behind_a_progress_character_is_still_a_figure(self):
        """Under ``-q`` pytest writes one character per test and no newline, so
        a print that does not open with one lands on that line."""
        assert gate_report.readings_from(".GATE cavity: 9.0001 GHz\n") == ("cavity: 9.0001 GHz",)

    def test_a_run_of_them_is_stripped_and_not_just_one(self):
        assert gate_report.readings_from("..sFGATE coax: 50.0 ohm\n") == ("coax: 50.0 ohm",)

    def test_a_gate_whose_figures_are_all_behind_one_is_not_called_silent(self):
        """The verdict that would otherwise be reached says the gate stopped
        measuring, which would be a false alarm printed by the release form."""
        reading = gate_report.Reading(
            gate=GATE, output=".GATE z = 1.0\n.\n2 passed in 1.00s\n", code=0, seconds=1.0
        )
        assert reading.verdict == gate_report.SOLVED

    def test_prose_that_mentions_the_marker_is_still_not_a_figure(self):
        assert gate_report.readings_from("a line that merely mentions GATE in passing\n") == ()

    def test_a_word_starting_with_a_progress_character_is_not_stripped_into_one(self):
        assert gate_report.readings_from("Exception raised before GATE x\n") == ()

    def test_a_figure_is_quoted_as_it_was_printed(self):
        """Only the front is stripped. A figure ends in whatever it ends in, and
        several of these end in a full stop that is part of the sentence."""
        assert gate_report.readings_from("GATE the error falls as the cell to 0.94.\n") == (
            "the error falls as the cell to 0.94.",
        )


class TestPytestsExitCodeIsConsulted:
    """A summary line counts assertions, and this suite fails a session for
    something no assertion knows about - `pytest_sessionfinish` sets the exit
    status where a tier solved more than its budget, with every test passing."""

    def test_a_passing_summary_with_a_non_zero_exit_is_not_evidence(self):
        assert run(GATE, "z = 1.0", summary="3 passed in 1.00s", code=1).verdict == (
            gate_report.REFUSED
        )

    def test_it_fails_the_run(self):
        assert gate_report.status_of([run(GATE, "z", summary="3 passed in 1.00s", code=1)]) == 1

    def test_a_gate_that_actually_failed_is_still_reported_as_failing(self):
        """The more specific verdict wins: both are non-zero exits, and only one
        of them says the workbench disagrees with the physics."""
        assert run(GATE, "z", summary="1 failed, 2 passed in 1.00s", code=1).verdict == (
            gate_report.FAILED
        )

    def test_the_page_says_which_it_was(self):
        rendered = gate_report.render([run(GATE, "z = 1.0", summary="3 passed in 1.00s", code=1)])
        assert gate_report.SAID[gate_report.REFUSED] in rendered


class TestOnlyPytestAnswersForPytest:
    def test_the_summary_is_read_from_standard_output_alone(self, monkeypatch):
        """Standard error arrives after pytest's summary and the summary is
        found by walking backwards, so a line of chatter that counts anything
        would answer for the run. Stubbed rather than solved: what is checked is
        which stream is read, not what a gate does."""
        seen = subprocess.CompletedProcess(
            args=(),
            returncode=0,
            stdout="GATE z = 1.0\n.\n1 passed in 1.00s\n",
            stderr="engine: 3 passed segments of the guide were skipped\n",
        )
        monkeypatch.setattr(gate_report.subprocess, "run", lambda *a, **k: seen)
        reading = gate_report.solve(GATE)
        assert reading.outcome == "1 passed in 1.00s"
        assert reading.verdict == gate_report.SOLVED

    def test_the_exit_code_reaches_the_reading(self, monkeypatch):
        seen = subprocess.CompletedProcess(
            args=(), returncode=1, stdout="GATE z = 1.0\n.\n1 passed in 1.00s\n", stderr=""
        )
        monkeypatch.setattr(gate_report.subprocess, "run", lambda *a, **k: seen)
        assert gate_report.solve(GATE).verdict == gate_report.REFUSED


class TestNamingAGateThatIsNotHere:
    def test_it_refuses_rather_than_running_the_ones_it_knows(self, capsys):
        """Asking for one gate that exists and one that does not would
        otherwise run the first and report a whole run of everything asked
        for."""
        assert gate_report.main(["--only", GATE.name, "--only", "nosuch"]) == 2
        assert "nosuch" in capsys.readouterr().err

    def test_it_names_every_one_it_did_not_know(self, capsys):
        assert gate_report.main(["--only", "nosuch", "--only", "neither"]) == 2
        said = capsys.readouterr().err
        assert "nosuch" in said and "neither" in said


class TestTheCommandCarriesTheVerdictOut:
    """The status is computed in one place and returned from another, and only
    the command is what anybody runs. Driven with the solving stubbed, because
    what is under test is the wiring rather than any gate."""

    @staticmethod
    def answering(monkeypatch, **outputs):
        def instead(gate, interpreter=None):
            return gate_report.Reading(gate=gate, output=outputs[gate.name], code=0, seconds=1.0)

        monkeypatch.setattr(gate_report, "solve", instead)

    def test_a_gate_that_failed_makes_the_command_fail(self, monkeypatch, tmp_path, capsys):
        self.answering(monkeypatch, **{GATE.name: "GATE z = 1.0\n.\n1 failed in 1.00s\n"})
        page = tmp_path / "report.md"
        assert gate_report.main(["--only", GATE.name, "--out", str(page)]) == 1
        assert gate_report.SAID[gate_report.FAILED] in page.read_text()
        assert GATE.name in capsys.readouterr().err

    def test_a_gate_that_solved_makes_it_succeed(self, monkeypatch, tmp_path):
        self.answering(monkeypatch, **{GATE.name: "GATE z = 1.0\n.\n1 passed in 1.00s\n"})
        page = tmp_path / "report.md"
        assert gate_report.main(["--only", GATE.name, "--out", str(page)]) == 0
        assert "Every gate solved and passed." in page.read_text()

    def test_a_machine_that_measured_nothing_says_which_fault_it_was(self, monkeypatch, tmp_path):
        self.answering(monkeypatch, **{GATE.name: ".\n3 passed, 14 skipped in 0.67s\n"})
        page = tmp_path / "report.md"
        assert gate_report.main(["--only", GATE.name, "--out", str(page)]) == 3
        assert gate_report.SAID[gate_report.MEASURED_NOTHING] in page.read_text()


class TestAPageWithNoGateOnIt:
    """Vacuous truth is the failure mode: "nothing is unsolved" holds of nothing
    at all, and would put a headline of success over an empty page."""

    def test_it_does_not_report_success(self):
        assert "Every gate solved and passed." not in gate_report.render([])

    def test_nor_does_the_status(self):
        assert gate_report.status_of([]) != 0

    def test_a_reading_that_can_be_walked_once_still_fills_the_page(self):
        """The page walks its readings for the headline, for the table and for
        the sections, so one that empties on the first walk would carry a
        headline over nothing."""
        one = run(GATE, "z = 1.0", summary="1 passed in 1.00s")
        rendered = gate_report.render(reading for reading in [one])
        assert "z = 1.0" in rendered
        assert f"| {GATE.name} |" in rendered
