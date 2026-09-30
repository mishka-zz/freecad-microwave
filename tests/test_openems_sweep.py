# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A run is a sweep: one solve per port, assembled into one matrix.

The task panel is the only route from the GUI to the solver, and until now it
drove one port and plotted one curve. What is tested here is the part that
changed - how a sweep is laid out on disk, how its runs are checked against
the envelopes they were supposed to come from, and what happens to the matrix
afterwards.

Qt is stubbed rather than skipped. PySide is not installed in the test
environment, so an ``importorskip`` would mean this never runs at all - which
is how the panel came to have no coverage the first time.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from Microwave.Objects.analysis import createEMAnalysis
from Microwave.Results.sparameters import ResultError, SParameters


@pytest.fixture(scope="module")
def panel_module():
    """``Microwave.Gui.openems_task_panel``, imported against ``conftest``'s
    stubbed Qt.

    The stubs come from ``conftest``, which installs them at import - before any
    test module loads - so a ``setdefault`` dance here could
    never fire. They are also never removed: doing so would not un-import the
    module, whose classes are bound to the stub objects either way, and would only
    make a later import build a *second*, differently-stubbed copy.
    """
    from Microwave.Gui import openems_task_panel

    return openems_task_panel


class Port:
    def __init__(self, number):
        self.number = number


class Problem:
    """Enough of an envelope for the panel: which port it drives, and its digest."""

    def __init__(self, number):
        self.excited_port = Port(number)
        self._digest = f"{number:064x}"

    def digest(self):
        return self._digest


class Run:
    """Enough of an adapter ``Results`` for the provenance report."""

    def __init__(
        self,
        port,
        digest=None,
        reproducible=True,
        tail_share=None,
        smallest_response=1.0,
        provenance=None,
    ):
        self.excited_port = port
        self.reproducible = reproducible
        self.tail_share = {port: 1e-4} if tail_share is None else tail_share
        self.smallest_response = smallest_response
        self.provenance = provenance or {}
        self._digest = digest if digest is not None else f"{port:064x}"

    def matches(self, digest):
        return self._digest == digest


class Signal:
    """Enough of a Qt signal to watch it wired up and taken down again.

    Refuses a disconnect with nothing on the other end, where PySide only warns
    and returns ``False``. Stricter than the real thing on purpose: the fake
    must not be the more forgiving of the two, or a redundant teardown passes
    here and prints four warnings per close in FreeCAD's report view.
    """

    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def disconnect(self):
        if not self.slots:
            raise RuntimeError("disconnecting a signal that has no slots")
        self.slots.clear()


class Button:
    """Enough of a ``QPushButton`` to see what it is wired to.

    Stands in for the real one because ``QtWidgets`` is a single mock: every
    ``QPushButton(...)`` call answers with the same object, so four buttons
    would be one and nothing could tell them apart.
    """

    def __init__(self, label=""):
        self.label = label
        self.enabled = True
        self.clicked = Signal()

    def setToolTip(self, text):
        pass

    def setEnabled(self, enabled):
        self.enabled = enabled


class Label:
    """Enough of a ``QLabel`` to read back what the panel wrote on it.

    Stands in for the real one for the reason :class:`Button` does: every
    ``QLabel(...)`` answers with the one mock, so two labels would be one.
    """

    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


def panel(panel_module, **attributes):
    """A panel with no widgets.

    ``__new__`` rather than ``__init__``: constructing the real thing builds a
    Qt form and reads the document's mesh state, and neither has anything to do
    with what these tests are about. The methods under test are the real ones.
    """
    instance = panel_module.SimulationTaskPanel.__new__(panel_module.SimulationTaskPanel)
    instance.logged = []
    instance.status = []
    instance.problems = []
    instance.geometry_lines = []
    # A study that declares no symmetry, which is the default and what most of
    # these tests are about. Override by passing ``analysis=``.
    instance.analysis = type("Study", (), {"Symmetry": "None"})()
    #: What ``setup_ui`` builds, wired the way it wires it: a button and the
    #: slot it drives.
    instance.buttons = tuple((Button(), lambda: None) for _ in range(2))
    for button, slot in instance.buttons:
        button.clicked.connect(slot)
    #: The line naming which solver Run starts. Written on every translation.
    instance.label_solver = Label()
    instance._stage = ""
    instance.status_color = None
    instance.log = instance.logged.append

    def set_status(text, colour=None):
        instance.status_color = colour
        instance.status.append((text, colour))

    instance.set_status = set_status
    for name, value in attributes.items():
        setattr(instance, name, value)
    return instance


def matrix(port_numbers=(1, 2), points=3):
    ports = len(port_numbers)
    return SParameters(
        frequency=np.linspace(1e9, 3e9, points),
        s=np.zeros((points, ports, ports), dtype=complex),
        port_numbers=tuple(port_numbers),
        reference=np.full((points, ports), 50.0),
        measured_impedance=np.full((points, ports), 50.0 + 0j),
        # What an assembly of openEMS runs carries forward from them, and what
        # a stored matrix is filed under.
        provenance={"solver": "openEMS"},
    )


class TestWhereTheRunsGo:
    """One directory per driven port, always - including for one port.

    A single run is a sweep of length one. Two layouts would be two things to
    reason about at the moment something has already gone wrong.
    """

    def test_each_run_gets_a_directory_named_after_its_port(
        self, panel_module, tmp_path, monkeypatch
    ):
        written = []
        monkeypatch.setattr(
            panel_module.write,
            "write",
            lambda problem, directory, report=(): (
                written.append(directory) or f"{directory}/e.json"
            ),
        )
        subject = panel(panel_module)

        envelopes = panel_module.SimulationTaskPanel.write_envelopes(
            subject, [Problem(1), Problem(2)], str(tmp_path)
        )

        assert [port for port, _ in envelopes] == [1, 2]
        assert [str(tmp_path / "port1"), str(tmp_path / "port2")] == written
        assert (tmp_path / "port1").is_dir() and (tmp_path / "port2").is_dir()

    def test_a_single_port_run_is_not_a_special_case(self, panel_module, tmp_path, monkeypatch):
        monkeypatch.setattr(
            panel_module.write,
            "write",
            lambda problem, directory, report=(): f"{directory}/e.json",
        )
        subject = panel(panel_module)

        panel_module.SimulationTaskPanel.write_envelopes(subject, [Problem(1)], str(tmp_path))

        assert (tmp_path / "port1").is_dir()

    def test_the_log_says_where_each_envelope_went(self, panel_module, tmp_path, monkeypatch):
        monkeypatch.setattr(
            panel_module.write,
            "write",
            lambda problem, directory, report=(): f"{directory}/e.json",
        )
        subject = panel(panel_module)

        panel_module.SimulationTaskPanel.write_envelopes(subject, [Problem(3)], str(tmp_path))

        assert any("Port 3" in line and "port3" in line for line in subject.logged)


class TestTheSweepRunsThemInOrder:
    """``run.sweep``, which is what a Run button now does.

    Tested here and not through ``SimulationWorker``: a ``QThread`` subclass
    cannot be instantiated without Qt, so anything that lives inside one is
    unreachable from a test. The worker is a signal adapter over this and
    nothing else, which is the point.
    """

    @pytest.fixture
    def solver(self, monkeypatch):
        from Microwave.Solvers.openems import run as solver_run

        return solver_run, monkeypatch

    def test_it_solves_every_envelope_and_keeps_the_results(self, solver):
        solver_run, monkeypatch = solver
        solved = []

        def fake_run(envelope, **kwargs):
            solved.append(envelope)
            return f"{envelope}/results.json"

        monkeypatch.setattr(solver_run, "run", fake_run)

        found = solver_run.sweep([(1, "one"), (2, "two")])

        assert solved == ["one", "two"]
        assert found == ["one/results.json", "two/results.json"]

    def test_it_says_which_run_is_in_flight(self, solver):
        """The progress line has to name the run, not just the cell count.

        Three identical "Solving 406,000 cells..." in a row says nothing about
        how far through a sweep is.
        """
        solver_run, monkeypatch = solver
        monkeypatch.setattr(solver_run, "run", lambda envelope, **kwargs: f"{envelope}/r.json")
        stages = []

        solver_run.sweep([(1, "one"), (4, "two")], on_stage=lambda *args: stages.append(args))

        assert stages == [(1, 2, 1), (2, 2, 4)]

    def test_one_failure_stops_the_sweep(self, solver):
        """A matrix missing a column is not a matrix.

        Carrying on would spend minutes producing something that cannot be
        assembled, burying the message that says why.
        """
        solver_run, monkeypatch = solver
        attempted = []

        def fake_run(envelope, **kwargs):
            attempted.append(envelope)
            raise RuntimeError("the solver died")

        monkeypatch.setattr(solver_run, "run", fake_run)

        with pytest.raises(RuntimeError, match="the solver died"):
            solver_run.sweep([(1, "one"), (2, "two"), (3, "three")])

        assert attempted == ["one"]

    def test_every_run_is_given_the_same_way_to_stop(self, solver):
        """One handle for the whole sweep, or Stop would only reach the run that
        happened to be in flight when it was built."""
        solver_run, monkeypatch = solver
        handed = []
        monkeypatch.setattr(
            solver_run,
            "run",
            lambda envelope, cancel=None, **kwargs: handed.append(cancel),
        )
        cancel = solver_run.Cancellation()

        solver_run.sweep([(1, "one"), (2, "two")], cancel=cancel)

        assert handed == [cancel, cancel]

    def test_build_only_produces_no_results_and_that_is_not_a_failure(self, solver):
        solver_run, monkeypatch = solver
        monkeypatch.setattr(solver_run, "run", lambda envelope, **kwargs: None)

        assert solver_run.sweep([(1, "one")], build_only=True) == []


class TestSayingWhatTheSweepWillReach:
    """The cost, and the gap, reported on Check rather than after the solve."""

    def problems(self, driven, all_ports):
        found = []
        for number in driven:
            problem = Problem(number)
            problem.ports = [Port(n) for n in all_ports]
            found.append(problem)
        return found

    def test_it_says_how_many_solves_a_run_is(self, panel_module):
        subject = panel(panel_module, problems=self.problems([1, 2], [1, 2]))

        panel_module.SimulationTaskPanel.report_coverage(subject)

        assert any("2 solves" in line for line in subject.logged)

    def test_one_port_is_not_announced_as_a_sweep(self, panel_module):
        subject = panel(panel_module, problems=self.problems([1], [1]))

        panel_module.SimulationTaskPanel.report_coverage(subject)

        assert subject.logged == []

    def test_a_port_nobody_drives_is_named_before_the_solve(self, panel_module):
        """Not a fault - the usual way to run a two-port, at half the cost.

        Said on Check so the trade is visible before the minutes are spent, and
        naming the terms that *will* be measured is the part that makes it a
        decision rather than a warning."""
        subject = panel(panel_module, problems=self.problems([1], [1, 2]))

        missing = panel_module.SimulationTaskPanel.report_coverage(subject)

        assert missing == [2]
        line = next(line for line in subject.logged if "not marked as sources" in line)
        assert "[2]" in line
        assert "S11" in line and "S21" in line
        assert "S22" not in line

    def test_full_coverage_reports_nothing_missing(self, panel_module):
        subject = panel(panel_module, problems=self.problems([1, 2], [1, 2]))

        assert panel_module.SimulationTaskPanel.report_coverage(subject) == []


class TestADeclaredSymmetry:
    """The study says the device is a mirror; the panel acts on it and says so."""

    def study(self, symmetry):
        return type("Study", (), {"Symmetry": symmetry})()

    def problems(self, driven, all_ports):
        found = []
        for number in driven:
            problem = Problem(number)
            problem.ports = [Port(n) for n in all_ports]
            found.append(problem)
        return found

    def test_an_undriven_column_is_reported_as_derived_not_missing(self, panel_module, monkeypatch):
        """Same document, opposite meaning: with the declaration the missing
        column is a saving, without it it is a gap."""
        monkeypatch.setattr(panel_module.symmetry_check, "mirror_warnings", lambda problem: [])
        subject = panel(
            panel_module,
            analysis=self.study("Mirror"),
            problems=self.problems([1], [1, 2]),
        )

        panel_module.SimulationTaskPanel.report_coverage(subject)

        line = next(line for line in subject.logged if "[2]" in line)
        assert "derived rather than solved" in line
        assert "half the run time" in line

    def test_the_cheap_asymmetry_checks_are_reported(self, panel_module, monkeypatch):
        monkeypatch.setattr(
            panel_module.symmetry_check,
            "mirror_warnings",
            lambda problem: ["the ports are different kinds"],
        )
        subject = panel(
            panel_module,
            analysis=self.study("Mirror"),
            problems=self.problems([1], [1, 2]),
        )

        panel_module.SimulationTaskPanel.report_coverage(subject)

        assert any("the ports are different kinds" in line for line in subject.logged)

    def test_no_declaration_means_no_symmetry_chatter(self, panel_module, monkeypatch):
        """Nothing is checked, and nothing is said, on a study that never
        claimed anything."""
        monkeypatch.setattr(
            panel_module.symmetry_check,
            "mirror_warnings",
            lambda problem: ["should never be reached"],
        )
        subject = panel(
            panel_module,
            analysis=self.study("None"),
            problems=self.problems([1], [1, 2]),
        )

        panel_module.SimulationTaskPanel.report_coverage(subject)

        assert not any("Symmetry:" in line for line in subject.logged)

    def test_a_two_run_sweep_tests_the_claim(self, panel_module):
        """Free verification: both ends were solved, so the declaration can be
        checked rather than believed."""
        subject = panel(panel_module, analysis=self.study("Mirror"))
        held = matrix()

        panel_module.SimulationTaskPanel.report_symmetry(subject, held, "mirror")

        assert any("matches S11" in line for line in subject.logged)

    def test_a_device_that_is_not_a_mirror_is_reported(self, panel_module):
        subject = panel(panel_module, analysis=self.study("Mirror"))
        held = matrix()
        held.s[:, 1, 1] = 0.9  # S22 far from S11

        panel_module.SimulationTaskPanel.report_symmetry(subject, held, "mirror")

        assert any("not the mirror" in line for line in subject.logged)

    def derived_matrix(self, mismatch=0.0):
        held = matrix()
        object.__setattr__(held, "driven", (1,))
        object.__setattr__(held, "derived", (2,))
        held.provenance["symmetry"] = "mirror"
        held.provenance["symmetry_impedance_mismatch"] = mismatch
        return held

    def test_collect_results_says_which_columns_were_derived(self, panel_module, monkeypatch, doc):
        """The whole point of tracking it. Storing a derived column and saying
        nothing would make it indistinguishable from a measured one at the only
        moment anybody is looking."""
        analysis = createEMAnalysis(doc)
        analysis.Symmetry = "Mirror"
        monkeypatch.setattr(panel_module.results_glue, "load_runs", lambda locations: [Run(1)])
        monkeypatch.setattr(
            panel_module.results_glue, "assemble", lambda runs, **kw: self.derived_matrix()
        )
        monkeypatch.setattr(panel_module, "show_matrix", lambda result: None)
        subject = panel(panel_module, analysis=analysis, problems=[Problem(1)])

        panel_module.SimulationTaskPanel.collect_results(subject, ["a/results.json"])

        assert any("[2] were derived" in line for line in subject.logged)

    def test_a_large_impedance_mismatch_is_reported_with_the_matrix(
        self, panel_module, monkeypatch, doc
    ):
        """The derivation takes the two ports to be one, so a disagreement
        between the impedances they measured is how far the declaration is from
        what was measured. Silence there is the failure mode with no symptom."""
        analysis = createEMAnalysis(doc)
        analysis.Symmetry = "Mirror"
        monkeypatch.setattr(panel_module.results_glue, "load_runs", lambda locations: [Run(1)])
        monkeypatch.setattr(
            panel_module.results_glue,
            "assemble",
            lambda runs, **kw: self.derived_matrix(mismatch=0.08),
        )
        monkeypatch.setattr(panel_module, "show_matrix", lambda result: None)
        subject = panel(panel_module, analysis=analysis, problems=[Problem(1)])

        panel_module.SimulationTaskPanel.collect_results(subject, ["a/results.json"])

        assert any("8.00% apart" in line for line in subject.logged)

    @pytest.mark.parametrize(("smallest", "warned"), [((0.1, 1.0), True), ((1.0, 1.0), False)])
    def test_what_an_undriven_port_sent_back_is_warned_of_past_the_tail_bar(
        self, panel_module, monkeypatch, doc, smallest, warned
    ):
        """The same bar a response still ringing is held to, scaled by the
        smallest response any run of the study reads."""
        held = matrix()
        object.__setattr__(held, "driven", (1,))
        held.provenance["sent_back"] = {"1": {"bound": 0.003, "frequency": 6e9, "port": 2}}
        analysis = createEMAnalysis(doc)
        monkeypatch.setattr(
            panel_module.results_glue,
            "load_runs",
            lambda locations: [Run(n, smallest_response=s) for n, s in zip((1, 2), smallest)],
        )
        monkeypatch.setattr(panel_module.results_glue, "assemble", lambda runs, **kw: held)
        monkeypatch.setattr(panel_module, "show_matrix", lambda result: None)
        subject = panel(panel_module, analysis=analysis, problems=[Problem(1), Problem(2)])

        panel_module.SimulationTaskPanel.collect_results(subject, ["a/results.json"])

        said = [line for line in subject.logged if "up to about 0.003 (-50.5 dB)" in line]
        assert len(said) == int(warned)
        assert all(line.startswith("WARNING: A term of column 1 can be off S") for line in said)

    @pytest.mark.parametrize(("smallest", "warned"), [((0.1, 1.0), True), ((1.0, 1.0), False)])
    def test_a_matrix_departing_from_reciprocity_is_warned_of_past_the_tail_bar(
        self, panel_module, monkeypatch, doc, smallest, warned
    ):
        """The departure is held to the bar that what an undriven port sent back
        is held to, scaled by the smallest response any run of the study reads."""
        held = matrix()
        held.s[1, 1, 0] = 0.003
        analysis = createEMAnalysis(doc)
        monkeypatch.setattr(
            panel_module.results_glue,
            "load_runs",
            lambda locations: [Run(n, smallest_response=s) for n, s in zip((1, 2), smallest)],
        )
        monkeypatch.setattr(panel_module.results_glue, "assemble", lambda runs, **kw: held)
        monkeypatch.setattr(panel_module, "show_matrix", lambda result: None)
        subject = panel(panel_module, analysis=analysis, problems=[Problem(1), Problem(2)])

        panel_module.SimulationTaskPanel.collect_results(subject, ["a/results.json"])

        said = [line for line in subject.logged if "the device's two are equal" in line]
        assert len(said) == int(warned)
        assert all(
            line.startswith("WARNING: S21 and S12 differ by up to 0.003 (-50.5 dB) at 2 GHz")
            for line in said
        )

    @pytest.mark.parametrize(("factor", "warned"), [(3.0, True), (1.5, False)])
    def test_a_reference_that_magnifies_the_solves_error_is_warned_of(
        self, panel_module, monkeypatch, doc, factor, warned
    ):
        """Past a factor of two the renormalisation can add more error than the
        solve left."""
        held = matrix()
        held.provenance["magnified"] = {
            "factor": factor,
            "frequency": 2e10,
            "ports": {"1": {"reference": [50.0, 0.0], "own": [475.0, 0.0], "vswr": 9.5}},
        }
        analysis = createEMAnalysis(doc)
        monkeypatch.setattr(panel_module.results_glue, "load_runs", lambda locations: [Run(1)])
        monkeypatch.setattr(panel_module.results_glue, "assemble", lambda runs, **kw: held)
        monkeypatch.setattr(panel_module, "show_matrix", lambda result: None)
        subject = panel(panel_module, analysis=analysis, problems=[Problem(1)])

        panel_module.SimulationTaskPanel.collect_results(subject, ["a/results.json"])

        said = [line for line in subject.logged if "magnify an error in the solve" in line]
        assert len(said) == int(warned)
        assert all(
            line.startswith(
                "WARNING: Port 1 is reported against 50 ohm and its own impedance is 475 ohm"
            )
            and line.endswith(
                "Reference it to its own impedance, where the error is not magnified."
            )
            for line in said
        )

    def test_a_small_mismatch_is_not_worth_saying(self, panel_module, monkeypatch, doc):
        analysis = createEMAnalysis(doc)
        analysis.Symmetry = "Mirror"
        monkeypatch.setattr(panel_module.results_glue, "load_runs", lambda locations: [Run(1)])
        monkeypatch.setattr(
            panel_module.results_glue,
            "assemble",
            lambda runs, **kw: self.derived_matrix(mismatch=1e-9),
        )
        monkeypatch.setattr(panel_module, "show_matrix", lambda result: None)
        subject = panel(panel_module, analysis=analysis, problems=[Problem(1)])

        panel_module.SimulationTaskPanel.collect_results(subject, ["a/results.json"])

        assert not any("apart" in line for line in subject.logged)

    def test_collect_results_says_which_material_the_other_backend_modelled_otherwise(
        self, panel_module, monkeypatch, doc
    ):
        from Microwave.Gui import results as glue

        analysis = createEMAnalysis(doc)
        palace = matrix()
        palace.provenance.update(
            solver="Palace",
            modelled=[{"material": "FR4", "held": "loss tangent", "loss_tangent": 0.02}],
        )
        glue.record(analysis, palace)
        folded = matrix()
        folded.provenance["modelled"] = [
            {
                "material": "FR4",
                "held": "conductivity",
                "conductivity": 0.02,
                "folded": 0.02,
                "at": 2e9,
            }
        ]
        monkeypatch.setattr(panel_module.results_glue, "load_runs", lambda locations: [Run(1)])
        monkeypatch.setattr(panel_module.results_glue, "assemble", lambda runs, **kw: folded)
        monkeypatch.setattr(panel_module, "show_matrix", lambda result: None)
        subject = panel(panel_module, analysis=analysis, problems=[Problem(1)])

        panel_module.SimulationTaskPanel.collect_results(subject, ["a/results.json"])

        assert any(
            line.startswith("'FR4' is not one model in the two: openEMS solved")
            for line in subject.logged
        )

    def comparing(self, panel_module, monkeypatch, doc, drawn=""):
        """The openEMS panel filing a matrix beside a Palace one, and what it logged."""
        from Microwave.Gui import results as glue

        analysis = createEMAnalysis(doc)
        palace = matrix()
        palace.provenance.update(solver="Palace")
        glue.record(analysis, palace)
        answer = matrix()
        answer.s[:, 1, 0] *= 0.9
        monkeypatch.setattr(panel_module.results_glue, "load_runs", lambda locations: [Run(1)])
        monkeypatch.setattr(panel_module.results_glue, "assemble", lambda runs, **kw: answer)
        monkeypatch.setattr(panel_module, "show_matrix", lambda result: None)
        subject = panel(panel_module, analysis=analysis, problems=[Problem(1)])
        subject.drawn = drawn
        panel_module.SimulationTaskPanel.collect_results(subject, ["a/results.json"])
        return analysis, subject.logged

    def test_collect_results_says_how_far_the_other_backends_matrix_stands(
        self, panel_module, monkeypatch, doc
    ):
        _, logged = self.comparing(panel_module, monkeypatch, doc)
        assert any(line.startswith("S21 differs by at most ") for line in logged)

    def test_the_matrix_is_filed_as_solved_from_the_drawing_read_when_the_run_started(
        self, panel_module, monkeypatch, doc
    ):
        from Microwave.Gui import results as glue
        from Microwave.Objects.results import load
        from Microwave.Results.compared import DRAWING

        analysis, logged = self.comparing(panel_module, monkeypatch, doc, drawn="started")
        assert load(glue.find_results(analysis, "openEMS")).provenance[DRAWING] == "started"
        assert any(
            line.startswith("Not compared with what Palace solved:")
            and "two different drawings" in line
            for line in logged
        )

    def test_a_two_run_sweep_reports_the_check_through_collect_results(
        self, panel_module, monkeypatch, doc
    ):
        """Through the real call site, not by calling report_symmetry directly.

        Deleting the call from ``collect_results`` survived every other test:
        each one either drove the method itself or handed it a matrix it
        declines to check.
        """
        analysis = createEMAnalysis(doc)
        analysis.Symmetry = "Mirror"
        monkeypatch.setattr(
            panel_module.results_glue, "load_runs", lambda locations: [Run(1), Run(2)]
        )
        monkeypatch.setattr(panel_module.results_glue, "assemble", lambda runs, **kw: matrix())
        monkeypatch.setattr(panel_module, "show_matrix", lambda result: None)
        subject = panel(panel_module, analysis=analysis, problems=[Problem(1), Problem(2)])

        panel_module.SimulationTaskPanel.collect_results(subject, ["a", "b"])

        assert any("matches S11" in line for line in subject.logged)

    def holed_matrix(self, points=5, discarded=(2,)):
        """A matrix with frequency points that hold no numbers.

        The band has a hole in it where the solves disagreed about port
        impedance - a *point* failure, not a missing column, and the panel is
        the only place a user ever learns it happened.
        """
        full = matrix(points=points)
        s = np.array(full.s)
        s[list(discarded)] = np.nan + 1j * np.nan
        return SParameters(
            frequency=full.frequency,
            s=s,
            port_numbers=full.port_numbers,
            reference=full.reference,
            measured_impedance=full.measured_impedance,
            discarded=discarded,
        )

    def with_matrix(self, panel_module, monkeypatch, doc, result):
        analysis = createEMAnalysis(doc)
        monkeypatch.setattr(
            panel_module.results_glue, "load_runs", lambda locations: [Run(1), Run(2)]
        )
        monkeypatch.setattr(panel_module.results_glue, "assemble", lambda runs, **kw: result)
        monkeypatch.setattr(panel_module, "show_matrix", lambda result: None)
        subject = panel(panel_module, analysis=analysis, problems=[Problem(1), Problem(2)])
        panel_module.SimulationTaskPanel.collect_results(subject, ["a", "b"])
        return subject

    def test_blank_frequency_points_are_reported(self, panel_module, monkeypatch, doc):
        """The plot breaks its line there by itself, which reads as "nothing was
        asked for" rather than "this went wrong". Nothing else says which."""
        subject = self.with_matrix(panel_module, monkeypatch, doc, self.holed_matrix())
        assert any("1 of 5 frequency points are blank" in line for line in subject.logged)

    def test_they_change_the_status_from_green(self, panel_module, monkeypatch, doc):
        """``on_finished`` sets green just before this runs, and a result with
        holes in it is not a clean finish."""
        subject = self.with_matrix(panel_module, monkeypatch, doc, self.holed_matrix())
        assert subject.status[-1][1] == "orange"

    def test_points_where_a_port_carries_no_power_are_reported_as_that(
        self, panel_module, monkeypatch, doc
    ):
        """A guide below its mode's cutoff. Nothing was discarded, and the
        advice about microstrip ports and measurement planes is not this."""
        below = self.holed_matrix(points=5, discarded=())
        below.s[:2] = np.nan + 1j * np.nan
        below.measured_impedance[:2] = -3000j
        subject = self.with_matrix(panel_module, monkeypatch, doc, below)
        assert subject.status[-1][1] == "orange"
        assert any(
            "2 of 5 frequency points are blank" in line and "carry no power there" in line
            for line in subject.logged
        )
        assert not any("microstrip" in line or "disagreed" in line for line in subject.logged)

    def test_a_whole_band_says_nothing_about_blanks(self, panel_module, monkeypatch, doc):
        subject = self.with_matrix(panel_module, monkeypatch, doc, matrix())
        assert not any("blank" in line for line in subject.logged)

    def test_the_impedance_quoted_is_not_one_of_the_blank_points(
        self, panel_module, monkeypatch, doc
    ):
        """``impedance_lines`` takes band centre, and a half-wave resonance puts
        its null there by construction - so the point most likely to be junk
        was the one printed as fact, three lines under the warning that says it
        is blank. ``measured_impedance`` is deliberately *not* blanked: it is the
        record of what the extraction did.
        """
        holed = self.holed_matrix(points=5, discarded=(2,))
        holed.measured_impedance[2] = 125.0 + 0j
        subject = self.with_matrix(panel_module, monkeypatch, doc, holed)
        assert not any("125.00" in line for line in subject.logged)
        assert any("50.00" in line for line in subject.logged)

    def test_a_three_port_is_not_promised_a_derivation(self, panel_module, monkeypatch):
        """Mirror symmetry completes a two-port driven at one end and nothing
        else. Promising otherwise on Check is a promise broken after the solve,
        and the pre-flight warning right above it says the opposite."""
        monkeypatch.setattr(panel_module.symmetry_check, "mirror_warnings", lambda problem: [])
        subject = panel(
            panel_module,
            analysis=self.study("Mirror"),
            problems=self.problems([1], [1, 2, 3]),
        )

        panel_module.SimulationTaskPanel.report_coverage(subject)

        assert not any("derived rather than solved" in line for line in subject.logged)
        assert any("not marked as sources" in line for line in subject.logged)

    def test_a_derived_matrix_is_not_checked_against_itself(self, panel_module):
        """The answer would be zero by construction, and a declaration would
        appear to confirm itself."""
        subject = panel(panel_module, analysis=self.study("Mirror"))
        held = SParameters(
            frequency=np.linspace(1e9, 3e9, 3),
            s=np.zeros((3, 2, 2), dtype=complex),
            port_numbers=(1, 2),
            reference=np.full((3, 2), 50.0),
            measured_impedance=np.full((3, 2), 50.0 + 0j),
            driven=(1,),
            derived=(2,),
        )

        panel_module.SimulationTaskPanel.report_symmetry(subject, held, "mirror")

        assert subject.logged == []


class TestThePanelNamesTheSolverItStarts:
    """One panel, one backend, and a study that may hold a solver of each.

    Until a study could hold two solvers there was nothing to name. Now a
    drawing marked up for the other backend can sit in front of this panel, and
    what stops it being answered by this one without a word is that the panel
    says which solver Run starts.
    """

    def _palace(self, doc, analysis):
        from Microwave.Objects.solver import createEMSolverPalace

        solver = createEMSolverPalace(doc)
        analysis.addObject(solver)
        return solver

    def _openems_only(self, analysis):
        return [
            obj
            for obj in analysis.Group
            if type(getattr(obj, "Proxy", None)).__name__ != "EMSolverOpenEMS"
        ]

    def test_the_one_a_new_study_ships_with_is_named(self, panel_module, doc):
        analysis = createEMAnalysis(doc)
        assert panel_module.solver_line(analysis) == "openEMS"

    def test_a_study_holding_both_says_which_one_run_starts(self, panel_module, doc):
        analysis = createEMAnalysis(doc)
        self._palace(doc, analysis)
        line = panel_module.solver_line(analysis)
        assert line == "openEMS - Run leaves Palace alone"

    def test_a_study_holding_only_the_other_backends_says_run_leaves_it(self, panel_module, doc):
        """Amber rather than red. Nothing has gone wrong: it is a good study for
        a backend this panel does not start."""
        analysis = createEMAnalysis(doc)
        self._palace(doc, analysis)
        analysis.Group = self._openems_only(analysis)
        line = panel_module.solver_line(analysis)
        assert "orange" in line
        assert "Palace - nothing here starts openEMS" in line

    def test_a_study_holding_no_solver_says_so(self, panel_module, doc):
        analysis = createEMAnalysis(doc)
        analysis.Group = self._openems_only(analysis)
        assert panel_module.solver_line(analysis) == "(none)"

    def test_a_renamed_solver_still_names_the_backend_that_runs(self, panel_module, doc):
        """A label is whatever the user typed, so a study whose solvers are
        called Coarse and Fine would otherwise name no backend at all."""
        analysis = createEMAnalysis(doc)
        for obj in analysis.Group:
            if type(getattr(obj, "Proxy", None)).__name__ == "EMSolverOpenEMS":
                obj.Label = "Coarse"
        assert panel_module.solver_line(analysis) == "Coarse (openEMS)"

    def test_a_label_the_user_typed_is_escaped(self, panel_module, doc):
        """A Label is whatever was typed, and Qt reads one holding ``<`` as
        markup and swallows the rest of the line."""
        analysis = createEMAnalysis(doc)
        self._palace(doc, analysis).Label = "<b>theirs"
        # Both sides of the line, because the panel writes its own solver's
        # label as readily as the one it leaves alone.
        for obj in analysis.Group:
            if type(getattr(obj, "Proxy", None)).__name__ == "EMSolverOpenEMS":
                obj.Label = "<i>mine"
        line = panel_module.solver_line(analysis)
        assert "&lt;i&gt;mine" in line
        assert "&lt;b&gt;theirs" in line
        assert "<i>mine" not in line
        assert "<b>theirs" not in line

    def test_a_study_holding_this_panels_solver_is_not_refused_here(self, panel_module, doc):
        analysis = createEMAnalysis(doc)
        self._palace(doc, analysis)
        assert panel_module.not_this_panel_s_solver(analysis) == ""

    def test_a_study_holding_none_is_left_to_the_adapters_own_refusal(self, panel_module, doc):
        """The adapter says a study holds no solver of its kind and asks for
        one, which is the right thing to say when the study holds nothing."""
        analysis = createEMAnalysis(doc)
        analysis.Group = self._openems_only(analysis)
        assert panel_module.not_this_panel_s_solver(analysis) == ""

    def test_a_study_holding_only_the_other_backends_is_refused_by_name(self, panel_module, doc):
        """Said instead of the adapter's, which asks for a solver to be added in
        front of a user looking at one in the tree."""
        analysis = createEMAnalysis(doc)
        self._palace(doc, analysis)
        analysis.Group = self._openems_only(analysis)
        said = panel_module.not_this_panel_s_solver(analysis)
        assert "'Palace'" in said
        assert "EM Analysis" in said

    def test_the_refusal_names_the_backend_this_panel_starts(self, panel_module, doc):
        """Taken from that adapter's own declaration rather than written out,
        so the name the panel says is the name the adapter answers to."""
        from Microwave.Solvers.openems.capabilities import capabilities

        analysis = createEMAnalysis(doc)
        self._palace(doc, analysis)
        analysis.Group = self._openems_only(analysis)
        assert capabilities().solver in panel_module.not_this_panel_s_solver(analysis)

    def test_the_directory_line_does_not_tell_a_saved_document_to_save(self, panel_module, doc):
        """Where a run is written is a property of this panel's solver, and a
        study holding none has no such property. Saving is the remedy for an
        unset directory, not for an absent one."""
        analysis = createEMAnalysis(doc)
        self._palace(doc, analysis)
        analysis.Group = self._openems_only(analysis)
        under_test = panel(panel_module, analysis=analysis, label_simdir=Label())
        panel_module.SimulationTaskPanel.update_simdir_label(under_test)
        assert under_test.label_simdir.text == "<b>Directory:</b> (none)"
        assert "save" not in under_test.label_simdir.text

    def test_a_study_that_has_the_solver_still_says_to_save(self, panel_module, doc):
        under_test = panel(panel_module, analysis=createEMAnalysis(doc), label_simdir=Label())
        panel_module.SimulationTaskPanel.update_simdir_label(under_test)
        assert "save the document" in under_test.label_simdir.text

    def test_translating_stops_there_rather_than_asking_the_adapter(
        self, panel_module, doc, monkeypatch
    ):
        """The refusal is what the user gets, and no translation is attempted -
        the adapter's own would speak about a solver that is not the subject."""
        analysis = createEMAnalysis(doc)
        self._palace(doc, analysis)
        analysis.Group = self._openems_only(analysis)

        def refuse(_analysis):
            raise AssertionError("the adapter was asked to translate")

        monkeypatch.setattr(panel_module.document, "sweep_and_report", refuse)
        under_test = panel(
            panel_module,
            analysis=analysis,
            update_mesh_label=lambda: None,
        )
        assert panel_module.SimulationTaskPanel.translate(under_test) is None
        assert under_test.status[-1][1] == "red"
        assert any("Palace" in line for line in under_test.logged)
        assert "Palace" in under_test.label_solver.text

    def test_update_mesh_stops_there_too(self, panel_module, doc, monkeypatch):
        """The preview is this backend's grid, laid from what its translation
        reads, so the button meets the same study and the same false remedy."""
        analysis = createEMAnalysis(doc)
        self._palace(doc, analysis)
        analysis.Group = self._openems_only(analysis)

        def refuse(_analysis):
            raise AssertionError("the mesher was asked for a grid")

        monkeypatch.setattr(panel_module.openems_mesh_preview, "refresh", refuse, raising=False)
        under_test = panel(
            panel_module,
            analysis=analysis,
            log_view=MagicMock(),
            update_mesh_label=MagicMock(),
        )
        panel_module.SimulationTaskPanel.on_update_mesh(under_test)
        assert under_test.status[-1][1] == "red"
        assert any("Palace" in line for line in under_test.logged)

    def test_update_mesh_on_a_study_it_works_with_still_meshes(
        self, panel_module, doc, monkeypatch
    ):
        analysis = createEMAnalysis(doc)
        self._palace(doc, analysis)
        built = MagicMock(summary=MagicMock(return_value="summary"), unresolved=(), oversized=None)
        monkeypatch.setattr(
            panel_module.openems_mesh_preview,
            "refresh",
            lambda _analysis: (None, built),
            raising=False,
        )
        under_test = panel(
            panel_module,
            analysis=analysis,
            log_view=MagicMock(),
            update_mesh_label=MagicMock(),
        )
        panel_module.SimulationTaskPanel.on_update_mesh(under_test)
        assert under_test.status[-1] == ("Mesh drawn", "green")

    def test_a_study_this_panel_can_run_still_reaches_the_adapter(
        self, panel_module, doc, monkeypatch
    ):
        analysis = createEMAnalysis(doc)
        self._palace(doc, analysis)
        reached = []

        def translate(_analysis):
            reached.append(_analysis)
            raise panel_module.TranslationError("stopped here on purpose")

        monkeypatch.setattr(panel_module.document, "sweep_and_report", translate)
        under_test = panel(
            panel_module,
            analysis=analysis,
            update_mesh_label=lambda: None,
        )
        assert panel_module.SimulationTaskPanel.translate(under_test) is None
        assert reached == [analysis]
        assert "Run leaves Palace alone" in under_test.label_solver.text


class TestTheLogSaysWhatWroteIt:
    """A run's log is the one a user pastes into a bug report, so it is stamped.

    Only that one. Meshing and checking clear the same widget and say nothing:
    a banner on every press is a banner nobody reads.
    """

    def restarted(self, panel_module):
        subject = panel(panel_module, log_view=MagicMock())
        panel_module.SimulationTaskPanel.restart_log(subject)
        return subject

    def test_the_version_is_the_first_line(self, panel_module):
        from Microwave import __version__

        subject = self.restarted(panel_module)

        assert subject.logged[0] == f"FreeCAD Microwave {__version__}"

    def test_it_clears_what_was_there(self, panel_module):
        subject = self.restarted(panel_module)

        subject.log_view.clear.assert_called_once()

    def test_check_does_not_stamp(self, panel_module, monkeypatch):
        subject = TestTheVerdictCheckPaints().translate_returning(
            panel_module, monkeypatch, [1, 2], [1, 2]
        )
        subject.log_view = MagicMock()

        panel_module.SimulationTaskPanel.on_check(subject)

        assert not any("FreeCAD Microwave" in line for line in subject.logged)

    def test_the_footer_says_it_without_a_press(self, panel_module):
        """What the panel shows when nothing has been run. The widget cannot be
        built without a display; what it says can."""
        from Microwave import __version__
        from Microwave.Gui.run_panel import version_line

        assert __version__ in version_line()


class TestTheVerdictCheckPaints:
    """Check's headline must not contradict its own log.

    Testing the *rendered* label for the word "warning" paints every verdict
    phrased any other way green - including "no S-matrix will be stored", which
    is the one most worth reading.
    """

    def translate_returning(
        self, panel_module, monkeypatch, driven, all_ports, findings=(), said=()
    ):
        """Drive the real ``translate`` with the document and pre-flight stubbed."""
        problems = []
        for number in driven:
            problem = Problem(number)
            problem.ports = [Port(n) for n in all_ports]
            problems.append(problem)

        # One call, because one translation produces both - a panel asking
        # twice would read a document that may have moved between the two.
        monkeypatch.setattr(
            panel_module.document, "sweep_and_report", lambda analysis: (problems, list(said))
        )
        monkeypatch.setattr(panel_module.preflight, "check", lambda problem: list(findings))
        monkeypatch.setattr(panel_module.preflight, "refusals", lambda f: [])

        subject = panel(panel_module, analysis=object())
        subject.update_mesh_label = lambda: None
        subject.log_view = MagicMock()
        return subject

    def test_what_the_shapes_lost_is_said_in_the_log(self, panel_module, monkeypatch):
        """Check is where the minutes are decided, so a difference between what
        was drawn and what the solver will be given has to be readable there."""
        subject = self.translate_returning(
            panel_module,
            monkeypatch,
            [1],
            [1],
            said=["Shell is solved 0.003 mm inside the drawing."],
        )

        panel_module.SimulationTaskPanel.on_check(subject)

        assert any("0.003 mm inside" in line for line in subject.logged)

    def test_and_it_goes_into_every_run_directory_the_sweep_writes(
        self, panel_module, tmp_path, monkeypatch
    ):
        """A directory meant to be attachable to a bug report on its own has to
        carry it, and it is one fact about the drawing rather than one per run.
        """
        written = []
        monkeypatch.setattr(
            panel_module.write,
            "write",
            lambda problem, directory, report=(): (
                written.append(list(report)) or f"{directory}/e.json"
            ),
        )
        subject = panel(panel_module, geometry_lines=["Shell is solved 0.003 mm inside."])

        panel_module.SimulationTaskPanel.write_envelopes(
            subject, [Problem(1), Problem(2)], str(tmp_path)
        )

        assert written == [["Shell is solved 0.003 mm inside."]] * 2

    def test_a_clean_model_is_green(self, panel_module, monkeypatch):
        subject = self.translate_returning(panel_module, monkeypatch, [1, 2], [1, 2])

        panel_module.SimulationTaskPanel.on_check(subject)

        assert subject.status[-1] == ("Ready to run", "green")

    def test_a_deliberately_partial_sweep_is_also_green(self, panel_module, monkeypatch):
        """Leaving a port undriven is a choice, and the usual one for a
        two-port. Colouring it as a problem would train people to ignore the
        colour - the log line is where the trade is explained."""
        subject = self.translate_returning(panel_module, monkeypatch, [1], [1, 2])

        panel_module.SimulationTaskPanel.on_check(subject)

        assert subject.status[-1] == ("Ready to run", "green")
        assert any("not marked as sources" in line for line in subject.logged)

    def test_a_preflight_warning_is_not_painted_over(self, panel_module, monkeypatch):
        """Searching the rendered label for the amber verdict
        for the word "warning"."""
        # The real dataclass, not a stub with one attribute: the panel now
        # counts subjects as well as reading severity, and a fake that carries
        # only what today's code touches goes stale the next time it touches
        # one thing more.
        warning = panel_module.preflight.Finding(
            panel_module.preflight.WARN, "Trace", "it is thinner than a skin depth"
        )
        subject = self.translate_returning(
            panel_module, monkeypatch, [1, 2], [1, 2], findings=[warning]
        )

        panel_module.SimulationTaskPanel.on_check(subject)

        assert subject.status[-1][1] == "orange"

    def test_the_headline_counts_objects_and_not_lines(self, panel_module, monkeypatch):
        """One grouped finding can stand for any number of objects, and the
        headline is the only number the user sees before opening the log.
        Counting lines would report a wall three solids run through as one
        problem."""
        warning = panel_module.preflight.Finding(
            panel_module.preflight.WARN,
            ("Substrate", "Ground", "Trace"),
            "it extends past the wall",
        )
        subject = self.translate_returning(
            panel_module, monkeypatch, [1, 2], [1, 2], findings=[warning]
        )

        panel_module.SimulationTaskPanel.on_check(subject)

        assert subject.status[-1] == ("3 warning(s) - see the log", "orange")

    def test_the_refusal_headline_counts_objects_too(self, panel_module, monkeypatch):
        """What the severity filter selects is not under test here - the
        helper stubs ``refusals`` away and this puts everything back through
        it. What is under test is that the red headline counts objects, and
        that is the only claim to read off it.
        """
        refusal = panel_module.preflight.Finding(
            panel_module.preflight.REFUSE, ("Board A", "Board B"), "it is outside the grid"
        )
        subject = self.translate_returning(
            panel_module, monkeypatch, [1, 2], [1, 2], findings=[refusal]
        )
        monkeypatch.setattr(panel_module.preflight, "refusals", lambda f: list(f))

        panel_module.SimulationTaskPanel.on_check(subject)

        assert subject.status[-1] == ("2 refusal(s) - see the log", "red")


class TestTheProgressLine:
    """Three identical "Solving 406,000 cells..." says nothing about progress."""

    def test_it_names_the_run_in_flight(self, panel_module):
        """The detail is what ``driver.py`` actually writes - a bare integer
        and the line shape. A count alone here is a number the
        driver has never emitted: the panel stripped the prefix and passed the
        rest through, so the test's own formatting stood in for the panel's."""
        subject = panel(panel_module)

        panel_module.SimulationTaskPanel.on_stage(subject, 2, 3, 5)
        panel_module.SimulationTaskPanel.on_marker(subject, "GRID", "cells=406000 lines=100x70x58")

        assert subject.status[-1][0] == ("Run 2/3 (port 5): Solving 406,000 cells (100x70x58)...")

    def test_the_shape_travels_with_the_count(self, panel_module):
        """openEMS prints ``Dimensions: 141x117x57 = 940329 Cells`` into the
        same log a few lines later. Counting intervals rather than lines puts a
        different number beside openEMS' own with nothing saying they describe
        one grid; the shape is what lets a reader see they do, without
        arithmetic."""
        subject = panel(panel_module)
        panel_module.SimulationTaskPanel.on_marker(subject, "GRID", "cells=940329 lines=141x117x57")
        assert subject.status[-1][0] == "Solving 940,329 cells (141x117x57)..."

    def test_a_grid_marker_it_cannot_read_is_shown_not_swallowed(self, panel_module):
        """The same rule the unknown-marker case follows."""
        subject = panel(panel_module)
        panel_module.SimulationTaskPanel.on_marker(subject, "GRID", "who knows")
        assert subject.status[-1][0] == "Solving who knows..."

    def test_a_single_run_is_not_dressed_up_as_a_sweep(self, panel_module):
        subject = panel(panel_module)

        panel_module.SimulationTaskPanel.on_stage(subject, 1, 1, 1)
        panel_module.SimulationTaskPanel.on_marker(subject, "SOLVER_STARTED", "")

        assert subject._stage == ""
        assert subject.status[-1] == ("Solving...", None)

    def test_an_error_keeps_the_run_it_came_from(self, panel_module):
        """Which of four solves failed is the first thing anyone asks."""
        subject = panel(panel_module)

        panel_module.SimulationTaskPanel.on_stage(subject, 3, 4, 3)
        panel_module.SimulationTaskPanel.on_marker(subject, "ERROR", "kind=engine")

        text, colour = subject.status[-1]
        assert colour == "red"
        assert text.startswith("Run 3/4 (port 3): ")

    def test_an_unknown_marker_is_shown_rather_than_swallowed(self, panel_module):
        """A new marker in the driver must degrade to visible, not to silence."""
        subject = panel(panel_module)

        panel_module.SimulationTaskPanel.on_marker(subject, "NEW_THING", "detail")

        assert subject.status[-1][0] == "NEW_THING"


class TestCheckingTheRunsCameBack:
    def test_a_run_from_a_different_envelope_is_called_out(self, panel_module):
        subject = panel(panel_module, problems=[Problem(1), Problem(2)])

        panel_module.SimulationTaskPanel.report_provenance(
            subject, [Run(1), Run(2, digest="stale")]
        )

        assert any("port 2" in line and "different input" in line for line in subject.logged)
        assert not any("port 1" in line for line in subject.logged)

    def test_a_missing_run_is_named(self, panel_module):
        subject = panel(panel_module, problems=[Problem(1), Problem(2)])

        panel_module.SimulationTaskPanel.report_provenance(subject, [Run(1)])

        assert any("no results came back for port 2" in line for line in subject.logged)

    def test_runs_are_matched_by_the_port_they_drove_not_by_position(self, panel_module):
        """A sweep whose runs came back out of order is exactly what this
        catches; a positional check would compare the wrong pair and report
        two failures where there are none."""
        subject = panel(panel_module, problems=[Problem(1), Problem(2)])

        panel_module.SimulationTaskPanel.report_provenance(subject, [Run(2), Run(1)])

        assert subject.logged == []

    def test_energy_termination_is_reported_once_for_the_sweep(self, panel_module):
        subject = panel(panel_module, problems=[Problem(1), Problem(2)])

        panel_module.SimulationTaskPanel.report_provenance(
            subject, [Run(1), Run(2, reproducible=False)]
        )

        assert sum("energy termination" in line for line in subject.logged) == 1

    def test_a_run_that_stopped_before_its_response_did_is_called_out(self, panel_module):
        """Per run, and naming which one. The driver says it too, on a marker
        that reaches the status line and is gone by the next one - the log is
        where a user reads it after the solve."""
        subject = panel(panel_module, problems=[Problem(1), Problem(2)])

        panel_module.SimulationTaskPanel.report_provenance(
            subject, [Run(1, tail_share={1: 1e-4, 2: 0.06}), Run(2)]
        )

        unfinished = [line for line in subject.logged if "stopped before the response" in line]
        assert len(unfinished) == 1
        assert "run 1" in unfinished[0] and "6% at port 2" in unfinished[0]

    def test_a_sweep_that_finished_says_nothing_about_it(self, panel_module):
        """Otherwise the line is in every log and stops being read."""
        subject = panel(panel_module, problems=[Problem(1), Problem(2)])

        panel_module.SimulationTaskPanel.report_provenance(subject, [Run(1), Run(2)])

        assert not any("stopped before the response" in line for line in subject.logged)

    def test_each_run_is_judged_against_the_response_its_study_reads(self, panel_module):
        """Off the result rather than off the panel's own copy of the study.
        A result opened here is judged the way the driver judged it, and a run
        far under the bar at full scale is nowhere near one at a stopband."""
        subject = panel(panel_module, problems=[Problem(1), Problem(2)])

        panel_module.SimulationTaskPanel.report_provenance(
            subject,
            [Run(1, tail_share={1: 1e-3}), Run(2, tail_share={2: 1e-3}, smallest_response=0.01)],
        )

        unfinished = [line for line in subject.logged if "stopped before the response" in line]
        assert len(unfinished) == 1
        assert "run 2" in unfinished[0] and "-40 dB" in unfinished[0]

    def test_what_the_waveguide_ports_read_is_called_out_from_the_result(self, panel_module):
        """A port left as openEMS read it, and field besides its mode read at a
        port, are each said off the run's own record, naming the run and the
        port, as the driver says them."""
        from Microwave.Solvers.openems import nearfield

        subject = panel(panel_module, problems=[Problem(1), Problem(2), Problem(3)])
        refused = {"port_reading": {"2": {"corrected": False, "reason": "R", "refused": True}}}
        port = nearfield.Weighed(
            depth=1.0, differ=[0.0, 2 * nearfield.BAR**2], decay=[1e9, 1e9], shallow=None
        )
        record = nearfield.record({2: port}, {}, {}, [20e9, 21e9])
        near = {nearfield.KEY: record}

        panel_module.SimulationTaskPanel.report_provenance(
            subject,
            [
                Run(1, provenance=refused),
                Run(2, provenance=near),
                Run(3, provenance=near, tail_share={3: 0.5}),
            ],
        )

        warned = [line for line in subject.logged if line.startswith("WARNING")]
        assert len(warned) == 3
        assert "run 1" in warned[0] and "port 2: its waveguide reading is left" in warned[0]
        assert "run 2" in warned[1] and "the largest part of it at port 2" in warned[1]
        # A record cut short says so, and the planes it moved apart are not blamed.
        assert "run 3" in warned[2] and "stopped before the response" in warned[2]

    def test_how_much_power_the_waveguide_ports_account_for_is_called_out(self, panel_module):
        """Said off the run's own record, naming the run, as the driver says it,
        and only where the record says it is sound."""
        from Microwave.Solvers.openems import balance

        subject = panel(panel_module, problems=[Problem(1), Problem(2)])
        closed = balance.record([0.0, 2 * balance.BAR], [0.0, 0.0], [20e9, 21e9], False, None)
        lossy = balance.record([0.0, 2 * balance.BAR], [0.0, 0.0], [20e9, 21e9], True, None)

        panel_module.SimulationTaskPanel.report_provenance(
            subject,
            [Run(1, provenance={balance.KEY: closed}), Run(2, provenance={balance.KEY: lossy})],
        )

        warned = [line for line in subject.logged if line.startswith("WARNING")]
        assert len(warned) == 1
        assert warned[0].startswith("WARNING: run 1: the waveguide ports account for 0.2% less")


class TestWhatHappensToTheMatrix:
    @pytest.fixture
    def collected(self, panel_module, monkeypatch, doc):
        """Run ``collect_results`` over a canned sweep, into a real group."""
        analysis = createEMAnalysis(doc)
        monkeypatch.setattr(
            panel_module.results_glue, "load_runs", lambda directories: [Run(1), Run(2)]
        )
        assembled_with = {}

        def fake_assemble(runs, **kw):
            assembled_with.update(kw)
            return matrix()

        monkeypatch.setattr(panel_module.results_glue, "assemble", fake_assemble)
        plotted = []
        monkeypatch.setattr(panel_module, "show_matrix", lambda result: plotted.append(result))
        subject = panel(panel_module, analysis=analysis, problems=[Problem(1), Problem(2)])
        return subject, analysis, plotted, panel_module, assembled_with

    def test_the_study_s_symmetry_declaration_reaches_the_assembly(self, collected):
        """Otherwise a Mirror study would solve one port and store a matrix with
        a hole in it - the feature silently absent, with nothing to see."""
        subject, _, _, panel_module, assembled_with = collected
        subject.analysis.Symmetry = "Mirror"

        panel_module.SimulationTaskPanel.collect_results(subject, ["a/results.json"])

        assert assembled_with.get("symmetry") == "mirror"

    def test_each_port_s_reference_impedance_reaches_the_assembly(self, collected):
        """Otherwise every matrix is renormalised to 50 and labelled 50.

        ``assemble`` defaults to 50, so a caller passing nothing sends a
        75-ohm study back referenced to 50 - in the panel and in every
        Touchstone file. A test elsewhere passing
        ``reference=75.0`` directly and proved the *parameter* worked, which is
        exactly why the missing argument went unseen. Assert the wiring, not
        the parameter.
        """
        from Microwave.Objects.ports import createEMPortLumped

        subject, analysis, _, panel_module, assembled_with = collected
        for number, impedance in ((1, 75.0), (2, 100.0)):
            port = createEMPortLumped(f"P{number}", doc=analysis.Document)
            port.Number = number
            port.ReferenceImpedance = impedance
            analysis.Group = analysis.Group + [port]

        panel_module.SimulationTaskPanel.collect_results(subject, ["a/results.json"])

        assert assembled_with.get("reference") == [75.0, 100.0]

    def test_a_port_referenced_to_itself_reaches_the_assembly_as_nothing(self, collected):
        """``None``, and specifically not the number beside it.

        The editor hides ``ReferenceImpedance`` in this mode, so whatever it
        holds is stale - reading it would renormalise a dispersive guide to a
        number the user cannot see and can no longer change.
        """
        from Microwave.Objects.ports import PORT_IMPEDANCE, createEMPortLumped

        subject, analysis, _, panel_module, assembled_with = collected
        for number in (1, 2):
            port = createEMPortLumped(f"P{number}", doc=analysis.Document)
            port.Number = number
            port.ReferenceImpedance = 12.5
            if number == 2:
                port.ReferencedTo = PORT_IMPEDANCE
            analysis.Group = analysis.Group + [port]

        panel_module.SimulationTaskPanel.collect_results(subject, ["a/results.json"])

        assert assembled_with.get("reference") == [12.5, None]

    def test_a_study_claiming_nothing_passes_nothing(self, collected):
        subject, _, _, panel_module, assembled_with = collected

        panel_module.SimulationTaskPanel.collect_results(subject, ["a/results.json"])

        assert assembled_with.get("symmetry") is None

    def test_the_matrix_lands_in_the_document(self, collected):
        subject, analysis, _, panel_module, _kw = collected

        panel_module.SimulationTaskPanel.collect_results(
            subject, ["a/results.json", "b/results.json"]
        )

        from Microwave.Gui.results import stored

        assert stored(analysis, "openEMS").port_numbers == (1, 2)

    def test_it_is_stored_before_it_is_plotted(self, collected, monkeypatch):
        """A plot is a window someone closes; the matrix is the answer.

        Losing minutes of FDTD because a matplotlib backend misbehaved would be
        an absurd trade, so the order is not incidental.
        """
        subject, analysis, _, panel_module, _kw = collected
        monkeypatch.setattr(
            panel_module,
            "show_matrix",
            lambda result: (_ for _ in ()).throw(RuntimeError("no backend")),
        )

        panel_module.SimulationTaskPanel.collect_results(subject, ["a/results.json"])

        from Microwave.Gui.results import stored

        assert stored(analysis, "openEMS") is not None

    def test_the_impedance_each_port_measured_is_reported(self, collected):
        subject, _, _, panel_module, _kw = collected

        panel_module.SimulationTaskPanel.collect_results(subject, ["a/results.json"])

        assert sum(line.startswith("Port ") for line in subject.logged) == 2

    def test_it_plots_the_whole_matrix_and_not_a_run(self, collected):
        subject, _, plotted, panel_module, _kw = collected

        panel_module.SimulationTaskPanel.collect_results(subject, ["a/results.json"])

        assert len(plotted) == 1
        assert isinstance(plotted[0], SParameters)

    def test_a_refusal_to_assemble_is_shown_verbatim(self, panel_module, monkeypatch, doc):
        """The refusal names what disagreed and what to do about it. Replacing
        it with "could not assemble" would throw away the whole check."""
        analysis = createEMAnalysis(doc)
        monkeypatch.setattr(panel_module.results_glue, "load_runs", lambda directories: [Run(1)])

        def refuse(runs, **kw):
            raise ResultError("the runs were meshed differently (12 cells against 9)")

        monkeypatch.setattr(panel_module.results_glue, "assemble", refuse)
        subject = panel(panel_module, analysis=analysis, problems=[Problem(1)])

        panel_module.SimulationTaskPanel.collect_results(subject, ["a/results.json"])

        assert any("meshed differently" in line for line in subject.logged)
        from Microwave.Gui.results import stored

        assert stored(analysis, "openEMS") is None

    def test_no_results_at_all_is_not_a_success(self, panel_module):
        subject = panel(panel_module)

        panel_module.SimulationTaskPanel.collect_results(subject, [])

        assert subject.status[-1] == ("The solver wrote no results", "red")


class Worker:
    """Enough of ``SimulationWorker`` for the panel: a way to be stopped, the
    signals it is listened to through, and whether it is still running."""

    def __init__(self, slots=(), running=True):
        from Microwave.Solvers.cancellation import Cancellation

        self.cancellation = Cancellation()
        self.running = running
        self.results_paths = []
        #: How ``wait`` was called, and whether a stop had been asked for by
        #: then - the two things that decide whether the thread can be
        #: destroyed safely.
        self.waited = None
        self.asked_to_stop_first = None
        for name in slots:
            setattr(self, name, Signal())

    def isRunning(self):
        return self.running

    def start(self):
        self.running = True

    def wait(self, *deadline):
        self.waited = deadline
        self.asked_to_stop_first = self.cancellation.requested
        self.running = False
        return True


class TestTheButtonRowIsOneTable:
    """``setup_ui``, which nothing else in the suite reaches.

    The row, the wiring and the shutdown all read ``self.buttons``, so what is
    left to get wrong is the table itself: a button laid out but not listed is
    connected by nobody and disconnected by nobody - and a button listed but
    not connected is a dead control the user presses.
    """

    @pytest.fixture
    def built(self, panel_module, monkeypatch):
        """A real panel, built the real way, against fake buttons."""
        monkeypatch.setattr(panel_module.QtWidgets, "QPushButton", Button)
        analysis = MagicMock()
        analysis.Document.FileName = ""
        return panel_module.SimulationTaskPanel(analysis)

    def test_it_holds_every_button_the_panel_shows(self, built):
        assert [button.label for button, _ in built.buttons] == [
            "Update Mesh",
            "Check",
            "Run",
            "Stop",
        ]

    def test_each_one_is_wired_to_what_the_table_says_it_drives(self, built):
        assert [button.clicked.slots for button, _ in built.buttons] == [
            [slot] for _, slot in built.buttons
        ]


class TestStoppingARun:
    """Closing the panel mid-solve destroyed the running ``QThread``: the 3000 ms
    wait returned ``False``, the thread was destroyed anyway, and FreeCAD went
    down with ``Abort trap: 6`` leaving openEMS on every core.

    The wait was never the fault. A wait that can expire is not a shutdown, it
    only moves the crash; what makes waiting safe is having asked the run to
    stop first.

    Ending the thread is half of it. The other half is the wiring: a connection
    that outlives one of its two ends is torn down through whichever end went
    first, and FreeCAD destroys the form's widgets and releases the panel from
    inside one C++ destructor. So nothing may still be connected by then -
    whether the run ended on its own, was stopped, or was replaced by another.
    """

    def running(self, panel_module, **attributes):
        subject = panel(
            panel_module,
            btn_stop=MagicMock(),
            set_ui_busy=MagicMock(),
            collect_results=MagicMock(),
            **attributes,
        )
        # Wired up the way ``on_run`` wires it, from the same table, so what a
        # release has to take down is whatever a run put up.
        subject.worker = worker = Worker(subject.worker_slots())
        for name, slot in subject.worker_slots().items():
            getattr(worker, name).connect(slot)
        return subject, worker

    def test_closing_the_panel_asks_the_run_to_stop_before_waiting_on_it(self, panel_module):
        subject, worker = self.running(panel_module)

        panel_module.SimulationTaskPanel.shutdown(subject)

        assert worker.asked_to_stop_first is True

    def test_the_wait_carries_no_deadline(self, panel_module):
        """A deadline here is a number chosen to hide a blocking call rather
        than to bound one, and when it expires the process aborts."""
        subject, worker = self.running(panel_module)

        panel_module.SimulationTaskPanel.shutdown(subject)

        assert worker.waited == ()

    def test_closing_a_panel_with_nothing_running_waits_on_nothing(self, panel_module):
        subject, worker = self.running(panel_module)
        worker.running = False

        panel_module.SimulationTaskPanel.shutdown(subject)

        assert worker.waited is None

    def test_a_run_that_has_already_ended_is_still_let_go_of(self, panel_module):
        """Stop, then close. There is nothing to wait for, which is not the same
        as nothing to do."""
        subject, worker = self.running(panel_module)
        worker.running = False

        panel_module.SimulationTaskPanel.shutdown(subject)

        assert subject.worker is None

    def test_the_panel_listens_to_everything_the_worker_reports(self, panel_module):
        """Named rather than derived, because ``SimulationWorker`` declares its
        signals in a ``QThread`` subclass and Qt here is a stub - the class
        object is unreachable, so the table would otherwise be both the wiring
        and its own oracle, and an entry could be dropped without a sound.

        These five are the contract with ``SimulationWorker``. Adding a sixth
        signal there means adding it here, which is the point.
        """
        subject = panel(panel_module)

        assert set(panel_module.SimulationTaskPanel.worker_slots(subject)) == {
            "marker_received",
            "log_received",
            "error_occurred",
            "finished_signal",
            "stage_started",
        }

    def test_every_signal_the_panel_listens_to_is_disconnected(self, panel_module):
        subject, worker = self.running(panel_module)

        panel_module.SimulationTaskPanel.shutdown(subject)

        assert all(getattr(worker, name).slots == [] for name in subject.worker_slots())

    def test_the_buttons_are_unwired_as_well(self, panel_module):
        """The widgets are the other end, and they go in the same destructor."""
        subject, _ = self.running(panel_module)
        buttons = [button for button, _ in subject.buttons]

        panel_module.SimulationTaskPanel.shutdown(subject)

        assert all(button.clicked.slots == [] for button in buttons)

    def test_closing_twice_costs_nothing(self, panel_module):
        subject, _ = self.running(panel_module)

        panel_module.SimulationTaskPanel.shutdown(subject)
        panel_module.SimulationTaskPanel.shutdown(subject)

        assert subject.worker is None

    def second_run(self, panel_module, monkeypatch, **attributes):
        """Run, Stop, Run - carried through to the worker that replaces the
        first one, which is the only place the release can be skipped without
        anything closing.

        The replacement is a fake rather than the real ``SimulationWorker``,
        which cannot be built here: it is a ``QThread`` subclass, and against
        stubbed Qt the class object is a mock that answers exactly one call.
        """
        from types import SimpleNamespace

        subject, first = self.running(
            panel_module,
            restart_log=lambda: None,
            resolve_directory=lambda: "/nowhere",
            translate=lambda: [Problem(1)],
            write_envelopes=lambda problems, base: [(1, "port1/openems.json")],
            **attributes,
        )
        first.running = False
        monkeypatch.setattr(
            panel_module.Objects,
            "solver_of",
            lambda analysis: SimpleNamespace(SolverPython="python3"),
        )
        replacement = Worker(subject.worker_slots(), running=False)
        monkeypatch.setattr(panel_module, "SimulationWorker", lambda *arguments: replacement)

        panel_module.SimulationTaskPanel.on_run(subject)
        return subject, first, replacement

    def test_a_second_run_lets_go_of_the_first(self, panel_module, monkeypatch):
        """A replaced worker that keeps its connections is the same dangling
        pair as a closed panel, reached without closing anything."""
        subject, first, _ = self.second_run(panel_module, monkeypatch)

        assert all(getattr(first, name).slots == [] for name in subject.worker_slots())

    def test_the_replacement_is_wired_from_the_same_table(self, panel_module, monkeypatch):
        """Or the release would be taking down connections the run never made,
        and leaving the ones it did."""
        subject, _, replacement = self.second_run(panel_module, monkeypatch)

        assert subject.worker is replacement
        for name, slot in subject.worker_slots().items():
            assert getattr(replacement, name).slots == [slot]

    def gui(self, panel_module, monkeypatch, active_document):
        """A fresh ``FreeCADGui``. The stub in ``conftest`` is one object for the
        whole session, so a call counted here would be a call counted for every
        test that ran before it."""
        from Microwave.Gui import run_panel

        stub = MagicMock()
        stub.ActiveDocument = active_document
        monkeypatch.setattr(run_panel, "FreeCADGui", stub)
        return stub

    def test_close_reaches_the_view_provider_rather_than_the_widget(
        self, panel_module, monkeypatch
    ):
        """``resetEdit`` runs ``unsetEdit``, which shuts the panel down and
        releases the document's edit lock. Closing the widget leaves the lock
        on, and tree interaction frozen with it."""
        subject, worker = self.running(panel_module)
        gui = self.gui(panel_module, monkeypatch, MagicMock())

        panel_module.SimulationTaskPanel.reject(subject)

        gui.ActiveDocument.resetEdit.assert_called_once_with()
        assert worker.waited is None, "the ViewProvider's unsetEdit does the shutdown"

    def test_with_no_gui_document_it_shuts_down_and_closes_itself(self, panel_module, monkeypatch):
        """There is no edit to reset and no lock to release, so nothing else
        will close the dialog - and a shutdown on its own leaves it on screen
        with every button unwired."""
        subject, _ = self.running(panel_module)
        gui = self.gui(panel_module, monkeypatch, None)

        panel_module.SimulationTaskPanel.reject(subject)

        assert subject.worker is None
        gui.Control.closeDialog.assert_called_once_with()

    def test_stop_asks_and_returns_rather_than_blocking_the_gui(self, panel_module):
        """The solver can take seconds to die. Waiting for it on the GUI thread
        freezes FreeCAD for exactly as long."""
        subject, worker = self.running(panel_module)

        panel_module.SimulationTaskPanel.on_stop(subject)

        assert worker.cancellation.requested is True
        assert worker.waited is None

    def test_stop_says_so_while_the_solver_is_still_dying(self, panel_module):
        subject, _ = self.running(panel_module)

        panel_module.SimulationTaskPanel.on_stop(subject)

        assert subject.status[-1] == ("Stopping...", "orange")

    def test_a_stopped_run_is_not_reported_as_completed(self, panel_module):
        """``on_finished`` is reached either way. Without this the panel showed
        the last progress line and a re-enabled Run button, and said nothing
        about the results that are not there."""
        subject, worker = self.running(panel_module)
        worker.cancellation.cancel()

        panel_module.SimulationTaskPanel.on_finished(subject, False)

        assert subject.status[-1] == ("Stopped - nothing from this run is kept", "orange")
        subject.collect_results.assert_not_called()

    def test_a_stopped_run_that_managed_to_finish_still_keeps_nothing(self, panel_module):
        """The last run of a sweep can complete between the click and the
        signal. Its column is one of several and the rest were never solved."""
        subject, worker = self.running(panel_module)
        worker.cancellation.cancel()
        worker.results_paths = ["port1/results.json"]

        panel_module.SimulationTaskPanel.on_finished(subject, True)

        subject.collect_results.assert_not_called()

    def test_a_run_that_failed_is_not_relabelled_as_stopped(self, panel_module):
        subject, _ = self.running(panel_module)

        panel_module.SimulationTaskPanel.on_finished(subject, False)

        assert not any("Stopped" in text for text, _ in subject.status)


class TestWhatUpdateMeshSays:
    """The status line after a mesh, which is the only verdict most users read.

    Every measured way into a runaway grid was met on this button, and this
    route never reaches pre-flight - it meshes, draws and prints the report.
    So a grid nobody meant has to be visible here or it is not visible at all.
    """

    def running(self, panel_module, **report):
        subject = panel(
            panel_module,
            log_view=MagicMock(),
            update_mesh_label=MagicMock(),
        )
        built = MagicMock(summary=MagicMock(return_value="summary"), unresolved=(), **report)
        return subject, built

    def _drawn(self, panel_module, monkeypatch, built):
        monkeypatch.setattr(
            panel_module.openems_mesh_preview,
            "refresh",
            lambda analysis: (None, built),
            raising=False,
        )
        subject, _ = self.running(panel_module)
        panel_module.SimulationTaskPanel.on_update_mesh(subject)
        return subject

    def test_an_ordinary_mesh_is_green(self, panel_module, monkeypatch):
        _, built = self.running(panel_module, oversized=None)
        subject = self._drawn(panel_module, monkeypatch, built)
        assert subject.status[-1] == ("Mesh drawn", "green")

    def test_a_grid_nobody_meant_is_not_green(self, panel_module, monkeypatch):
        _, built = self.running(panel_module, oversized="this grid is 3.2 GiB")
        subject = self._drawn(panel_module, monkeypatch, built)
        text, colour = subject.status[-1]
        assert colour == "orange"
        assert "large" in text
