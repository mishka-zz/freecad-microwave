# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How a run's log is said, whichever backend says it: the tone each line is
shown in, where a blank line stands, and what a failure says."""

from types import SimpleNamespace

from Microwave.Gui.run_steps import FAILED, WARNED, WARNING, Steps, pointed

#: A refinement's lines, as a run's log held them.
REFINES = [
    "Refines Run 4 (openEMS): Mesh Settings (openEMS) ElementsPerWavelength 30 -> 42.4",
    "Run 4 (openEMS) took 16 s of processor time",
]


class Shown(Steps):
    """Steps that keep what they showed, line and tone."""

    def __init__(self):
        self.logged = []
        self.shown = []
        self.status = []

    def show_status(self, text, color):
        self.status.append((text, color))

    def show_line(self, line, tone=None):
        self.shown.append((line, tone))

    def clear_shown(self):
        self.shown.clear()


def test_a_line_that_warns_is_shown_as_a_warning_and_the_rest_plainly():
    steps = Shown()
    steps.log(f"{WARNING}Symmetry: the two ports measured reference impedances 1.00% apart")
    steps.log("Port 1: 476.23 + 0.00j ohm at 23 GHz")
    assert steps.shown == [
        (f"{WARNING}Symmetry: the two ports measured reference impedances 1.00% apart", WARNED),
        ("Port 1: 476.23 + 0.00j ohm at 23 GHz", None),
    ]


def test_a_blank_line_is_said_once_and_never_first():
    steps = Shown()
    for line in ("", "Palace runs on 4 processes", "", "", "Palace version: v0.18.1", ""):
        steps.log(line)
    assert steps.logged == ["Palace runs on 4 processes", "", "Palace version: v0.18.1", ""]
    assert [line for line, _ in steps.shown] == steps.logged


def test_a_failure_is_red_and_each_line_of_it_is_shown_as_one():
    """Even a line that would warn, since it says why the step failed."""
    steps = Shown()
    steps.fail("Run failed - see the log", "openEMS was not found", None, f"{WARNING}x")
    assert steps.status == [("Run failed - see the log", "red")]
    assert steps.shown == [("openEMS was not found", FAILED), (f"{WARNING}x", FAILED)]


def test_a_refinements_lines_are_set_apart_from_what_follows():
    steps = Shown()
    steps.refinement = SimpleNamespace(lines=lambda: list(REFINES))
    steps.restart_log()
    steps.log("2 of 2 ports are marked as sources")
    assert steps.logged[1:] == [*REFINES, "", "2 of 2 ports are marked as sources"]


def test_a_run_counts_as_its_warnings_what_it_showed_as_warnings():
    steps = Shown()
    steps.log(f"{WARNING}Symmetry: the two ports measured reference impedances 1.00% apart")
    steps.log("Port 1: 476.23 + 0.00j ohm at 23 GHz")
    kept = steps.kept(SimpleNamespace(ended=lambda log, warned: warned))
    assert kept == [line for line, tone in steps.shown if tone == WARNED]


def test_the_property_that_names_a_program_is_said_only_where_it_is_blank():
    blank = SimpleNamespace(SolverPath="", Label="Solver (Palace)")
    named = SimpleNamespace(SolverPath="/opt/palace/bin/palace", Label="Solver (Palace)")
    assert pointed(blank, "SolverPath", "the palace binary") == (
        "Set SolverPath on 'Solver (Palace)' to the palace binary."
    )
    assert pointed(named, "SolverPath", "the palace binary") is None
