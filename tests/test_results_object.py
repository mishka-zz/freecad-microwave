# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""``EMSParameters``: the S-matrix as something the document owns.

Two questions, and they are different. Does a matrix survive the round trip
through FreeCAD properties unchanged - which is the whole reason results are
stored in the document rather than in a file beside it? And does the glue in
``Gui/results.py`` put exactly one of these in the right analysis?

Nothing here imports FreeCAD: ``conftest`` stubs it, so a document object is
exercised as a document object rather than as a data class that happens to
resemble one.
"""

import json
import re
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from Microwave.Gui import results as glue
from Microwave.Objects.results import (
    BULK,
    SUMMARY,
    createEMSParameters,
    load,
    store,
)
from Microwave.Results.compared import SIGN_BY_RULE
from Microwave.Results.sparameters import IMPEDANCE_STATED, MIRROR, ResultError, SParameters
from Microwave.Solvers.palace import balance
from Microwave.Solvers.palace import read as palace_read
from Microwave.Solvers.palace.read import POWER_VOLTAGE


def matrix(port_numbers=(1, 2), points=4, reference=50.0):
    """A matrix with every term distinct, so a transposed store cannot pass."""
    ports = len(port_numbers)
    frequency = np.linspace(1e9, 4e9, points)
    # s[f, i, j] = (f+1) + 0.1*i + 0.01*j, complex with a different imaginary
    # rule again - no two entries share a value.
    index = np.indices((points, ports, ports))
    s = (
        (index[0] + 1)
        + 0.1 * index[1]
        + 0.01 * index[2]
        + 1j * (0.5 - 0.2 * index[0] + 0.03 * index[1] - 0.007 * index[2])
    )
    return SParameters(
        frequency=frequency,
        s=s.astype(complex),
        port_numbers=tuple(port_numbers),
        # Broadcast, so a caller can pass a scalar, one value per port, or a
        # full (frequency, port) array - the round trip has to survive all
        # three, and a fixture that only ever produces 50 ohm everywhere cannot
        # tell whether load() read the array or invented it.
        reference=np.broadcast_to(np.asarray(reference, dtype=float), (points, ports)).copy(),
        # Deliberately complex and per port: a microstrip's Z0 is, and storing
        # only its magnitude would be a loss no later reader could detect.
        measured_impedance=np.stack(
            [
                np.full(points, 48.0 + 1.5j),
                np.full(points, 51.0 - 2.5j),
            ][:ports],
            axis=1,
        ),
        provenance={"solver": "openEMS", "cells": 406000, "excitations": [1, 2]},
    )


def from_backend(solver, result):
    """``result`` as though ``solver`` had produced it."""
    return replace(result, provenance={**result.provenance, "solver": solver})


class TestTheRoundTrip:
    """What goes into the document must be what comes out of it.

    This is the claim the whole storage design rests on. If a matrix does not
    survive save and reopen exactly, the alternative - a Touchstone file
    beside the document - was the better choice after all.
    """

    def test_every_number_comes_back(self, doc):
        """Ports out of order, and a reference that varies on both axes.

        Both are deliberate. Ascending port numbers would let ``load`` sort them
        and pass; a uniform 50 ohm reference would let it return a constant and
        pass. Neither shortcut is hypothetical - both were tried as mutations
        and must not survive this test.
        """
        obj = createEMSParameters(doc)
        original = matrix(
            port_numbers=(5, 2),
            reference=np.column_stack([np.linspace(48.0, 52.0, 4), np.linspace(70.0, 80.0, 4)]),
        )
        store(obj, original)

        back = load(obj)

        np.testing.assert_array_equal(back.frequency, original.frequency)
        np.testing.assert_array_equal(back.s, original.s)
        np.testing.assert_array_equal(back.reference, original.reference)
        np.testing.assert_array_equal(back.measured_impedance, original.measured_impedance)
        assert back.port_numbers == original.port_numbers

    def test_which_ports_sit_at_their_own_impedance_comes_back(self, doc):
        """A *driven* one, because an undriven port is folded back into that
        record on the way out whatever was stored - so it is the one shape where
        dropping the field is visible. What reads it is ``Results/tdr.py``,
        which refuses a step response at any port in it.
        """
        original = matrix(reference=np.column_stack([np.linspace(470.0, 480.0, 4)] * 2))
        obj = createEMSParameters(doc)
        store(
            obj,
            SParameters(
                frequency=original.frequency,
                s=original.s,
                port_numbers=original.port_numbers,
                reference=original.reference,
                measured_impedance=original.reference,
                self_referenced=original.port_numbers,
            ),
        )

        assert load(obj).self_referenced == original.port_numbers

    def test_it_is_exact_and_not_merely_close(self, doc):
        """Bit for bit, which is the point of storing doubles rather than text.

        ``abs=0.0`` deliberately: pytest.approx carries a default absolute
        tolerance of 1e-12 that would accept a store rounding to picohertz, and
        a 4 GHz sample rounded to 1e-12 relative is still 4 mHz out.
        """
        obj = createEMSParameters(doc)
        original = matrix()
        store(obj, original)
        back = load(obj)

        assert back.frequency.tolist() == original.frequency.tolist()
        assert back.s.ravel().tolist() == original.s.ravel().tolist()

    def test_the_port_numbers_are_the_document_s_own(self, doc):
        """A two-port carved out of a five-port keeps its numbers.

        ``SParameters`` indexes rows by position and names them by document
        port number, and losing the mapping would silently relabel every curve.
        """
        obj = createEMSParameters(doc)
        store(obj, matrix(port_numbers=(2, 5)))
        assert load(obj).port_numbers == (2, 5)

    def test_a_one_port_result_is_not_a_special_case(self, doc):
        obj = createEMSParameters(doc)
        store(obj, matrix(port_numbers=(3,), points=7))
        back = load(obj)
        assert back.s.shape == (7, 1, 1)
        assert back.port_numbers == (3,)

    def test_the_stored_lists_are_plain_floats(self, doc):
        """Every one of them, because a numpy array is silently destroyed.

        Measured on FreeCAD 1.1.1: ``App::PropertyFloatList`` handed a numpy
        array stores *N copies of its last element* - ``np.arange(5.0)`` reads
        back as ``[4.0, 4.0, 4.0, 4.0, 4.0]``, as plain floats, right length,
        right final sample. A frequency axis reopens as a flat line at the top
        of the band and looks like physics.

        The stub in ``conftest`` stores whatever it is handed, so this test
        cannot reproduce the corruption - it pins the *conversion* instead,
        which is the thing that must not be simplified away.

        Driven from ``BULK`` rather than a list written out here, so a array
        added later is covered the day it is added. ``type() is``, not
        ``isinstance``: ``numpy.float64`` passes an isinstance check against
        ``float`` and is exactly what must not get through.
        """
        obj = createEMSParameters(doc)
        store(obj, matrix())
        for name in BULK:
            values = getattr(obj, name)
            assert type(values) is list, f"{name} is {type(values)}, not a list"
            for value in values:
                assert type(value) in (int, float), f"{name} holds {type(value)}"

    def test_the_stored_counts_are_plain_ints(self, doc):
        """The same rule as above, one step along, where FreeCAD is *stricter*.

        Measured on FreeCAD 1.1.1: a numpy array into ``App::PropertyFloatList``
        is quietly corrupted, but a numpy int into ``App::PropertyInteger``
        raises ``TypeError: type must be int, not numpy.int64`` - out of
        ``record()`` in the GUI, after the solve has finished.

        Unlike the list conversions, this one cannot be killed by deleting the
        ``int()`` beside it, and that is worth saying rather than mistaking for
        a gap. ``ndarray.size`` and ``len()`` both return a Python ``int``
        already, so neither source can produce a numpy scalar today. What this
        guards is the *next* source: ``np.sum(mask)`` and ``np.count_nonzero``
        are the natural way to write "how many points did we keep", both return
        ``numpy.int64``, and the stub accepts one silently - so the change
        would look fine everywhere except in a real FreeCAD.

        Counts only. ``FrequencyStart`` and ``FrequencyStop`` are quantity
        properties and come back as a ``Quantity`` from a real FreeCAD and a
        ``MockQuantity`` here, so what a test can pin about them is their value,
        which ``test_the_summary_describes_the_matrix`` does.
        """
        obj = createEMSParameters(doc)
        store(obj, matrix())
        for name in ("Ports", "Points"):
            assert type(getattr(obj, name)) is int, f"{name} is not a plain int"

    def test_the_document_layer_does_not_import_scikit_rf(self):
        """Checked in a clean interpreter, because this suite has already
        imported it - an in-process assertion would pass or fail on test
        ordering rather than on the claim.

        The claim: opening a document costs numpy and the standard library.
        scikit-rf 1.13.0 eagerly pulls in pandas and scipy, and matplotlib where
        it is installed - about a second - and ``Objects.results`` names
        ``SParameters`` at module scope,
        so it is one careless import away from paying that at workbench start.
        """
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        code = (
            # A bare module is enough: nothing in Objects/ touches FreeCAD at
            # import time, it only needs the name to bind.
            "import sys, types;"
            "sys.modules['FreeCAD'] = types.ModuleType('FreeCAD');"
            "import Microwave.Objects;"
            "print(','.join(sorted(n for n in sys.modules "
            "if n.split('.')[0] in ('skrf', 'pandas', 'matplotlib'))))"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=root
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "", (
            f"importing the document layer pulled in {done.stdout.strip()}"
        )

    def test_it_does_not_need_the_result_library(self, doc, monkeypatch):
        """A document full of results opens without paying for scikit-rf.

        1.13.0 eagerly imports pandas and scipy, and matplotlib where it is
        installed - about a second.
        Storing and reading a matrix is numpy arithmetic and nothing else, and
        this is what keeps it that way: the resolver is made to raise, and the
        round trip must not notice.
        """
        from Microwave.Results import _skrf

        def refuse():
            raise AssertionError("the round trip reached for scikit-rf")

        monkeypatch.setattr(_skrf, "module", refuse)

        obj = createEMSParameters(doc)
        store(obj, matrix())
        assert load(obj).s.shape == (4, 2, 2)


class TestAPartialMatrixInTheDocument:
    """A one-path result has to survive save and reopen as *itself*.

    The failure to avoid is a partial matrix reopening as a complete one: the
    nan columns would come back as numbers and the ports nobody drove would
    look measured.
    """

    def partial(self, points=4):
        full = matrix(points=points)
        s = np.array(full.s)
        s[:, :, 1] = np.nan + 1j * np.nan
        reference = np.array(full.reference, dtype=complex)
        reference[:, 1] = full.measured_impedance[:, 1]
        return SParameters(
            frequency=full.frequency,
            s=s,
            port_numbers=full.port_numbers,
            reference=reference,
            measured_impedance=full.measured_impedance,
            driven=(1,),
        )

    def test_which_columns_exist_comes_back(self, doc):
        obj = createEMSParameters(doc)
        store(obj, self.partial())

        back = load(obj)
        assert back.driven == (1,)
        assert back.unmeasured == (2,)
        assert not back.complete

    def test_the_holes_come_back_as_holes(self, doc):
        obj = createEMSParameters(doc)
        store(obj, self.partial())

        back = load(obj)
        assert np.all(np.isnan(back.parameter(1, 2)))
        assert np.all(np.isfinite(back.parameter(2, 1)))

    def test_a_port_nobody_drove_at_fifty_ohm_reads_as_fifty_ohm(self, doc):
        """The commonest study in the workbench, and it is what the chart says.

        Two lumped ports, one driven. Nothing renormalises the undriven column,
        because that would need the terms it does not have - but its own
        impedance *is* the 50 ohm that was typed, so nothing about the matrix
        distinguishes it from a port asked for 50. A basis naming it apart
        puts the driven column's two curves on apparently different bases.
        """
        original = self.partial()
        lumped = np.full(original.reference.shape, 50.0 + 0j)
        obj = createEMSParameters(doc)
        store(
            obj,
            SParameters(
                frequency=original.frequency,
                s=original.s,
                port_numbers=original.port_numbers,
                reference=lumped,
                measured_impedance=lumped,
                driven=(1,),
            ),
        )

        assert obj.Reference == "50 ohm"

    def test_a_port_nobody_drove_at_an_impedance_that_moves_is_named(self, doc):
        """And here there is no number to give it, which is the case the phrase
        exists for. It is also the only shape in this file where the record of
        which ports sit at their own impedance changes anything readable, so it
        is where both halves of that record are held: the sentence ``store``
        composes, and the field it writes for ``load`` to bring back.
        """
        original = self.partial()
        own = np.linspace(470.0, 480.0, original.frequency.size)
        reference = np.array(original.reference, dtype=complex)
        reference[:, 1] = own
        measured = np.array(original.measured_impedance, dtype=complex)
        measured[:, 1] = own
        obj = createEMSParameters(doc)
        store(
            obj,
            SParameters(
                frequency=original.frequency,
                s=original.s,
                port_numbers=original.port_numbers,
                reference=reference,
                measured_impedance=measured,
                driven=(1,),
            ),
        )

        assert obj.Reference == "port 1: 50 ohm, port 2: its own impedance"
        assert load(obj).self_referenced == (2,)

    def test_a_complex_reference_survives(self, doc):
        """The undriven port keeps its own measured impedance, which for a
        microstrip is complex. Storing the reference as a float list would
        discard the imaginary part with no warning."""
        obj = createEMSParameters(doc)
        original = self.partial()
        store(obj, original)

        back = load(obj)
        assert back.reference[0, 1].imag != 0.0
        np.testing.assert_array_equal(back.reference, original.reference)

    def test_a_complete_matrix_still_says_so(self, doc):
        obj = createEMSParameters(doc)
        store(obj, matrix())
        assert load(obj).complete

    def test_a_derived_column_reopens_as_derived(self, doc):
        """Complete, but not measured. Losing that distinction would let a
        column filled from the user's symmetry claim reopen as evidence for
        it - and ``mirror_disagreement`` would then confirm the claim from
        its own output."""
        partial = self.partial()
        completed = SParameters(
            frequency=partial.frequency,
            s=np.nan_to_num(partial.s),
            port_numbers=partial.port_numbers,
            reference=np.full_like(partial.reference, 50.0),
            measured_impedance=partial.measured_impedance,
            driven=(1,),
            derived=(2,),
        )
        obj = createEMSParameters(doc)
        store(obj, completed)

        back = load(obj)
        assert back.driven == (1,)
        assert back.derived == (2,)
        assert back.complete
        assert back.unmeasured == ()

    def test_a_derived_column_is_not_recorded_as_self_referenced(self, doc):
        """A derived column *is* renormalised, along with the ones that were
        solved, so it does not belong in the record of what sits at its own
        impedance. The label is not what holds this: a port at a constant 50
        reads as 50 ohm either way. ``Results/tdr.py`` is - it refuses a step
        response at any port in that record, so folding a derived column in
        turns a port that can be read into one that is refused, and the refusal
        names a reason that is not the one.
        """
        partial = self.partial()
        completed = SParameters(
            frequency=partial.frequency,
            s=np.nan_to_num(partial.s),
            port_numbers=partial.port_numbers,
            reference=np.full_like(partial.reference, 50.0),
            measured_impedance=partial.measured_impedance,
            driven=(1,),
            derived=(2,),
        )
        obj = createEMSParameters(doc)
        store(obj, completed)

        assert load(obj).self_referenced == ()
        assert obj.Reference == "50 ohm"


class TestWhatTheDocumentShows:
    def test_the_bulk_arrays_are_hidden_and_the_summary_is_read_only(self, doc):
        """A 2004-element float list in the property editor is noise, and
        editing one by hand could only make the object lie about a solve."""
        obj = createEMSParameters(doc)
        for name in BULK:
            assert obj._editor_modes[name] == 2, name
        for name in SUMMARY:
            assert obj._editor_modes[name] == 1, name

    def test_the_summary_describes_the_matrix(self, doc):
        obj = createEMSParameters(doc)
        store(obj, matrix(points=9))

        assert obj.Ports == 2
        assert obj.Points == 9
        assert float(obj.FrequencyStart) == pytest.approx(1e9, rel=1e-12, abs=0.0)
        assert float(obj.FrequencyStop) == pytest.approx(4e9, rel=1e-12, abs=0.0)
        assert obj.Reference == "50 ohm"

    def test_provenance_survives_as_readable_json(self, doc):
        obj = createEMSParameters(doc)
        store(obj, matrix())
        assert json.loads(obj.Provenance)["cells"] == 406000
        assert load(obj).provenance["solver"] == "openEMS"

    def test_provenance_that_json_cannot_express_does_not_cost_the_object(self, doc):
        """An adapter is free to record whatever it measured.

        A single unserialisable entry must not take the whole matrix with it -
        losing minutes of FDTD because a provenance value was a numpy scalar
        would be an absurd trade.
        """
        obj = createEMSParameters(doc)
        result = matrix()
        result.provenance["awkward"] = {1, 2, 3}
        store(obj, result)
        assert "awkward" in json.loads(obj.Provenance)
        assert load(obj).s.shape == (4, 2, 2)

    def test_an_unserialisable_provenance_key_does_not_either(self, doc):
        """``default=`` covers values and not keys.

        With only ``default=str`` a single odd key raised ``TypeError`` out of
        ``store`` - exactly the trade the paragraph above calls absurd, from
        the other side of the colon.
        """
        obj = createEMSParameters(doc)
        result = matrix()
        result.provenance[frozenset({"odd"})] = 1
        store(obj, result)

        assert json.loads(obj.Provenance)["solver"] == "openEMS"
        assert load(obj).s.shape == (4, 2, 2)

    def test_keys_that_cannot_be_compared_do_not_either(self, doc):
        """``sort_keys`` compares them, and str against int raises."""
        obj = createEMSParameters(doc)
        result = matrix()
        result.provenance[7] = "seven"
        store(obj, result)

        assert obj.Provenance
        assert load(obj).s.shape == (4, 2, 2)

    def test_a_provenance_that_falls_back_keeps_the_backend_it_is_filed_under(self, doc):
        """A nested dict holding an int key and a str key defeats even the
        stringified top level. The fallback keeps the backend's name, or the
        next run of that backend passes the result over and files another."""
        from Microwave.Objects.results import backend

        obj = createEMSParameters(doc)
        result = matrix()
        result.provenance["nested"] = {1: "one", "two": 2}
        store(obj, result)

        assert "unserialisable" in json.loads(obj.Provenance)
        assert backend(obj) == "openEMS"


class TestItRefusesRatherThanGuesses:
    def test_an_empty_object_says_to_run_the_analysis(self, doc):
        obj = createEMSParameters(doc)
        with pytest.raises(
            ResultError, match=re.escape("holds no S-matrix yet. Run Simulation to fill it")
        ):
            load(obj)

    def test_a_truncated_matrix_is_refused_by_name(self, doc):
        """Reshaping whatever is there would hand physics code an array of the
        right shape and the wrong contents - the one failure with no symptom."""
        obj = createEMSParameters(doc)
        store(obj, matrix())
        obj.ScatteringReal = list(obj.ScatteringReal)[:-1]

        with pytest.raises(ResultError, match="ScatteringReal"):
            load(obj)

    def test_a_short_impedance_list_is_refused_too(self, doc):
        obj = createEMSParameters(doc)
        store(obj, matrix())
        obj.PortImpedanceImag = list(obj.PortImpedanceImag)[:2]

        with pytest.raises(ResultError, match="PortImpedanceImag"):
            load(obj)


class FakePort:
    def __init__(self, z0, incident, reflected):
        self.z0 = z0
        self.incident = incident
        self.reflected = reflected


class FakeRun:
    """One solve, shaped like ``Solvers.openems.read.Results``.

    Enough for ``from_runs``: the frequencies, the ports it saw, the one it
    drove, each port's reference impedance, and the volts it measured.
    """

    def __init__(self, excited, frequency, impedances, values):
        self.frequency = frequency
        self.excited_port = excited
        self.ports = tuple(impedances)
        self.reproducible = True
        self.provenance = {"cells": 1000, "envelope_digest": f"d{excited}"}
        self._impedances = impedances
        self._values = values

    def port(self, number):
        """Each port's waves as a run with every other port matched makes them:
        a unit wave into the one driven, none into the rest, and each sending
        back its ratio."""
        driven = float(number == self.excited_port)
        return FakePort(
            np.full(self.frequency.size, self._impedances[number]),
            np.full(self.frequency.size, driven, dtype=complex),
            self.s(number, self.excited_port),
        )

    def s(self, receiving, driving):
        return np.full(self.frequency.size, self._values[(receiving, driving)])


class TestAssembly:
    """``assemble`` and the reference it assembles to.

    Both were stubbed out of every other test in this file, so the 50 ohm
    default - which every stored matrix is normalised to - was pinned by
    nothing at all.
    """

    def runs(self):
        frequency = np.linspace(1e9, 2e9, 3)
        impedances = {1: 50.0 + 0j, 2: 50.0 + 0j}
        return [
            FakeRun(1, frequency, impedances, {(1, 1): 0.2 + 0j, (2, 1): 0.8 + 0j}),
            FakeRun(2, frequency, impedances, {(1, 2): 0.8 + 0j, (2, 2): 0.2 + 0j}),
        ]

    def test_it_normalises_to_fifty_ohms_unless_told_otherwise(self):
        assembled = glue.assemble(self.runs())

        assert np.allclose(assembled.reference, glue.REFERENCE)
        assert np.allclose(assembled.reference, 50.0)

    def test_the_reference_is_honoured_when_it_is_given(self):
        assembled = glue.assemble(self.runs(), reference=75.0)
        assert np.allclose(assembled.reference, 75.0)

    def test_matched_ports_pass_their_volts_through_unchanged(self):
        """Every port at the reference impedance, so the pseudo-wave correction
        and the renormalisation are both identities - what comes out is what
        the runs measured, and any stray factor shows up immediately."""
        assembled = glue.assemble(self.runs())

        assert assembled.port_numbers == (1, 2)
        np.testing.assert_allclose(assembled.parameter(2, 1), 0.8, atol=1e-9)
        np.testing.assert_allclose(assembled.parameter(1, 1), 0.2, atol=1e-9)


class TestTheGlueThatFilesIt:
    def analysis(self, doc):
        obj = doc.addObject("App::DocumentObjectGroupPython", "EMAnalysis")
        return obj

    def test_the_first_run_is_what_puts_it_in_the_tree(self, doc):
        analysis = self.analysis(doc)
        assert glue.find_results(analysis, "openEMS") is None

        stored = glue.record(analysis, matrix())

        [run] = glue.runs_of(analysis)
        assert glue.find_results(analysis, "openEMS") is stored
        assert run in analysis.Group
        assert run.Group == [stored]
        assert glue.run_of(stored) is run

    def test_the_filed_run_does_not_arrive_marked_stale(self, doc):
        """S-Parameters arrive under the analysis carrying the touched mark,
        and only Recompute clears it. The mark tells the user the object in
        front of them is out of date - on a matrix measured a second earlier.
        Purged rather than recomputed: ``execute`` has nothing to do."""
        analysis = self.analysis(doc)
        stored = glue.record(analysis, matrix())
        assert "Touched" not in stored.State
        assert "Touched" not in glue.run_of(stored).State

    def test_filing_the_run_leaves_the_study_as_it_found_it(self, doc):
        """Adding the run touches the study, and a study marked for recompute
        after a run reads as one the run has made stale."""
        analysis = self.analysis(doc)
        analysis.purgeTouched()
        glue.record(analysis, matrix())
        assert "Touched" not in analysis.State

    def test_a_study_an_edit_marked_keeps_its_mark(self, doc):
        """The mark an edit put there before the run still says the study
        changed, and recomputing is what clears it."""
        analysis = self.analysis(doc)
        assert "Touched" in analysis.State
        glue.record(analysis, matrix())
        assert "Touched" in analysis.State

    def test_filing_the_run_is_one_undo_step(self, doc):
        """Untransacted, Ctrl-Z after a run leaves the run in the tree and
        undoes something earlier instead. Named for what disappears."""
        analysis = self.analysis(doc)
        glue.record(analysis, matrix())
        assert doc.transactions == [("Store Results", "commit")]

    def test_the_run_and_its_matrix_are_created_inside_it(self, doc, monkeypatch):
        """As for the mesh: wrapping nothing is not the fix. Hoisting the
        creation above the block leaves the run in the tree after Ctrl-Z, and
        nothing but this notices."""
        analysis = self.analysis(doc)
        opened_when = []
        for name in ("createEMSParameters", "createEMRun"):
            real = getattr(glue, name)

            def spy(*args, real=real):
                opened_when.append(doc._open)
                return real(*args)

            monkeypatch.setattr(glue, name, spy)
        glue.record(analysis, matrix())
        assert opened_when == ["Store Results", "Store Results"]

    def test_a_failure_while_filing_aborts_rather_than_commits(self, doc, monkeypatch):
        analysis = self.analysis(doc)
        monkeypatch.setattr(
            glue,
            "store",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with pytest.raises(RuntimeError):
            glue.record(analysis, matrix())
        assert doc.transactions == [("Store Results", "abort")]

    def test_a_second_run_is_kept_beside_the_first(self, doc):
        """A design is iterated by comparing this run with the last one, so a
        run that replaced the last one would destroy the comparison it was made
        for."""
        analysis = self.analysis(doc)
        first = glue.record(analysis, matrix(points=4))
        second = glue.record(analysis, matrix(points=6))

        assert first is not second
        assert glue.results_of(analysis) == [first, second]
        assert load(first).frequency.size == 4
        assert glue.find_results(analysis, "openEMS") is second

    def test_the_newest_is_the_highest_number_whatever_its_clock_said(self, doc):
        """A start is one machine's clock and orders nothing across two
        machines or a change of the clock; the number is the document's."""
        analysis = self.analysis(doc)
        first = glue.record(
            analysis, matrix(), replace(glue.begin(analysis), started="2026-10-03T12:00:00+02:00")
        )
        second = glue.record(
            analysis, matrix(), replace(glue.begin(analysis), started="2026-10-03T11:00:00+02:00")
        )
        assert glue.run_of(second).Number > glue.run_of(first).Number
        assert glue.find_results(analysis, "openEMS") is second

    def test_a_run_whose_matrix_was_deleted_answers_nothing(self, doc):
        """It is no run to compare with, draw or export, and the run before it
        is the newest that answers."""
        analysis = self.analysis(doc)
        first = glue.record(analysis, matrix())
        emptied = glue.record(analysis, matrix())
        doc.removeObject(emptied.Name)
        assert glue.find_results(analysis, "openEMS") is first
        assert glue.results_of(analysis) == [first]

    def test_it_reads_back_what_it_filed(self, doc):
        analysis = self.analysis(doc)
        glue.record(analysis, matrix(port_numbers=(2, 5)))
        assert glue.stored(analysis, "openEMS").port_numbers == (2, 5)

    def test_an_analysis_with_no_result_has_none(self, doc):
        assert glue.stored(self.analysis(doc), "openEMS") is None

    def test_it_ignores_everything_else_in_the_study(self, doc):
        """A real analysis group holds a solver, a mesh policy, ports, material
        bindings and a preview. Picking the first member would find one of
        those, and every one of them would be the wrong answer."""
        analysis = self.analysis(doc)
        for name in ("EMSolverOpenEMS", "EMMeshPolicy", "EMPortLumped"):
            other = doc.addObject("App::FeaturePython", name)
            other.Proxy = type(name, (), {})()
            analysis.addObject(other)

        stored = glue.record(analysis, matrix())

        found = glue.find_results(analysis, "openEMS")
        # By kind, not by position. Asserting only ``found is stored`` passed
        # with the kind test removed entirely: ``record`` and ``find_results``
        # would both pick the same wrong member and agree with each other.
        assert type(found.Proxy).__name__ == "EMSParameters"
        assert found is stored

    def test_a_matrix_standing_in_the_study_is_no_backends_run(self, doc):
        """A matrix on its own - imported, or saved by an earlier build - is a
        table and not a run, so it is nobody's newest answer."""
        analysis = self.analysis(doc)
        alone = createEMSParameters(doc)
        analysis.addObject(alone)

        assert glue.find_results(analysis, "openEMS") is None
        assert glue.stored(analysis, "openEMS") is None
        assert glue.results_of(analysis) == [alone]

    def test_each_backend_keeps_its_own_answer(self, doc):
        """One drawing answered by two backends is the only measurement of what
        the staircase costs on the user's own shape, and an answer that
        overwrote the other would leave nothing to compare it with."""
        analysis = self.analysis(doc)
        fdtd = glue.record(analysis, matrix(port_numbers=(1, 2)))
        fem = glue.record(analysis, from_backend("Palace", matrix(port_numbers=(1,))))

        assert fdtd is not fem
        assert glue.stored(analysis, "openEMS").ports == 2
        assert glue.stored(analysis, "Palace").ports == 1

    def test_a_rerun_replaces_nothing(self, doc):
        analysis = self.analysis(doc)
        fdtd = glue.record(analysis, matrix(points=4))
        fem = glue.record(analysis, from_backend("Palace", matrix(points=5)))
        again = glue.record(analysis, matrix(points=6))

        assert again is not fdtd
        assert load(fdtd).frequency.size == 4
        assert load(fem).frequency.size == 5
        assert glue.stored(analysis, "openEMS").frequency.size == 6

    def test_a_run_is_labelled_with_its_number_and_backend_and_nothing_else(self, doc):
        """Several sit side by side in the tree, and the label is what a user
        tells them apart by. The number is the document's, so FreeCAD never
        numbers a label itself."""
        analysis = self.analysis(doc)
        first = glue.record(analysis, matrix())
        second = glue.record(analysis, from_backend("Palace", matrix()))
        assert [glue.run_of(found).Label for found in (first, second)] == [
            "Run 1 (openEMS)",
            "Run 2 (Palace)",
        ]
        assert [found.Label for found in (first, second)] == [
            "S-Parameters (run 1, openEMS)",
            "S-Parameters (run 2, Palace)",
        ]

    def test_a_damaged_result_raises_rather_than_reading_as_absent(self, doc):
        """``None`` means absent and nothing else.

        Reporting damage as "no result yet" would offer a Run button as the fix
        for a problem running does not fix, and would hide the message that
        says what is actually wrong.
        """
        analysis = self.analysis(doc)
        found = glue.record(analysis, matrix())
        found.ScatteringImag = list(found.ScatteringImag)[:3]

        with pytest.raises(ResultError, match="ScatteringImag"):
            glue.stored(analysis, "openEMS")

    def test_two_studies_do_not_share_one_matrix(self, doc):
        """Per analysis, not per document. A document-wide lookup would
        overwrite the first study's answer with the second's."""
        one, two = self.analysis(doc), self.analysis(doc)
        glue.record(one, matrix(port_numbers=(1, 2)))
        glue.record(two, matrix(port_numbers=(3,)))

        assert glue.stored(one, "openEMS").port_numbers == (1, 2)
        assert glue.stored(two, "openEMS").port_numbers == (3,)

    def test_each_run_gets_its_own_directory(self):
        assert glue.directory_for("/tmp/board_openems", 2).endswith("board_openems/port2")

    def test_a_result_with_no_samples_reports_nothing_rather_than_raising(self):
        """A zero-point band is not something a solve produces, but the readout
        runs on the success path where an IndexError would be reported as
        "plotting failed" over a result that is fine."""
        assert list(glue.impedance_lines(matrix(points=0))) == []

    def test_the_impedance_readout_names_the_frequency_it_belongs_to(self):
        """A microstrip's Z0 moves across the band, and a number quoted without
        its frequency is how a dispersion curve becomes a constant."""
        lines = list(glue.impedance_lines(matrix(points=5)))

        assert lines == [
            "Port 1: 48.00 + 1.50j ohm at 2.5 GHz",
            "Port 2: 51.00 - 2.50j ohm at 2.5 GHz",
        ]


class TestWhatPalaceAnswers:
    """A Palace table, as the result layer holds it.

    One run carries every excitation, and every port is referenced to its own
    mode, with a number only where Palace read the port's voltage. So what is
    checked is where each column lands, that nothing claims an impedance nobody
    gave, and that every reader of the reference says so rather than printing
    nan."""

    def answer(self, driven=(1, 2)):
        frequency = np.linspace(20e9, 26e9, 3)
        out = (1, 2)
        # Every entry distinct, and each column carrying its driving port in
        # the imaginary part, so a column put under the wrong port is caught.
        table = np.empty((frequency.size, len(out), len(driven)), dtype=complex)
        for column, port in enumerate(driven):
            for row in range(len(out)):
                table[:, row, column] = 0.1 * (row + 1) + 1j * port + np.arange(frequency.size)
        # What leaves through each face is what the column says, less the watt
        # that came in at the driven port, and a tenth of a percent more at
        # port 2 - so the account falls short by that and nothing else.
        flux = np.abs(table) ** 2
        for column, port in enumerate(driven):
            flux[:, out.index(port), column] -= 1.0
            flux[:, out.index(2), column] += 1e-3
        return SimpleNamespace(
            frequency=frequency,
            out=out,
            driven=driven,
            matrix=table,
            flux=flux,
            radiated=None,
            dissipates=True,
            modelled=(),
            stated={},
        )

    @pytest.mark.parametrize(("kind", "alike"), [(palace_read.RESISTANCE, True), ("", False)])
    def test_a_lumped_port_is_compared_at_its_resistance_and_a_guide_at_no_number(
        self, kind, alike
    ):
        """A lumped port's own reference is its element's resistance, so a
        table held to that number elsewhere is differenced against it. A guide's
        is its own mode, which no number stands for, so the difference names
        the port."""
        from Microwave.Results import overlay

        answer = self.answer()
        answer.impedance = np.full((answer.frequency.size, 2), 50.0)
        answer.stated = {1: kind, 2: kind} if kind else {}
        palace = glue.from_palace(answer)
        fixed = replace(palace, self_referenced=(), provenance={})
        tables = (overlay.Table(fixed, "openEMS"), overlay.Table(palace, "Palace"))
        apart = overlay.difference(*tables, 2, 1)
        assert float(apart.value.max()) == 0.0
        if alike:
            assert apart.unalike == ()
        else:
            assert apart.unalike[0] == "port 1 is at its own mode in Palace"

    def test_a_guide_read_and_one_not_read_are_one_basis(self):
        """Each is at its own mode, whatever figure its impedance has or lacks."""
        from Microwave.Results import overlay

        read = self.answer()
        read.impedance = np.full((read.frequency.size, 2), 420.0)
        tables = [
            overlay.Table(glue.from_palace(read), "Run 1 (Palace)"),
            overlay.Table(glue.from_palace(self.answer()), "Run 2 (Palace)"),
        ]
        assert overlay.difference(*tables, 2, 1).unalike == ()

    def test_a_guide_against_a_line_at_its_own_impedance_is_two_bases(self):
        """A line's own impedance states no kind, and a mode is not one."""
        from Microwave.Results import overlay

        palace = glue.from_palace(self.answer())
        line = replace(
            palace,
            reference=np.outer(np.linspace(45.0, 55.0, 3), [1.0, 1.0]) + 0j,
            provenance={},
        )
        apart = overlay.difference(
            overlay.Table(palace, "Run 1 (Palace)"), overlay.Table(line, "Run 2 (openEMS)"), 2, 1
        )
        assert apart.unalike == (
            "port 1 is at its own mode in Run 1 (Palace), its own impedance in Run 2 (openEMS)",
            "port 2 is at its own mode in Run 1 (Palace), its own impedance in Run 2 (openEMS)",
        )

    def test_each_column_stands_under_the_port_that_drove_it(self):
        answer = self.answer(driven=(2,))
        result = glue.from_palace(answer, title="WR-42")
        np.testing.assert_array_equal(result.parameter(1, 2), answer.matrix[:, 0, 0])
        np.testing.assert_array_equal(result.parameter(2, 2), answer.matrix[:, 1, 0])

    def test_a_port_nobody_drove_is_an_unmeasured_column(self):
        result = glue.from_palace(self.answer(driven=(2,)))
        assert result.unmeasured == (1,)
        assert np.isnan(result.parameter(1, 1)).all()

    def test_what_the_matrix_leaves_unaccounted_for_is_kept_with_it(self):
        """The matrix cannot show the power a port absorbed besides its mode,
        and the log that said so scrolls away."""
        answer = self.answer(driven=(1,))
        kept = glue.from_palace(answer).provenance["unaccounted"]
        (short,) = balance.shortfalls(answer)
        assert kept == {
            "1": {
                "share": short.share,
                "frequency": short.frequency,
                "port": short.port,
                "radiated": None,
                "dissipates": True,
                "lumped": False,
            }
        }
        assert short.port == 2

    def test_no_impedance_is_claimed_for_a_port_the_run_stated_none_for(self):
        """A port with no figure is a wave port, since a lumped one always
        states its resistance, so it is named as at its own mode."""
        result = glue.from_palace(self.answer())
        assert result.self_referenced == (1, 2)
        assert np.isnan(result.reference).all()
        assert np.isnan(result.measured_impedance).all()
        assert result.provenance[IMPEDANCE_STATED] == {
            "1": palace_read.UNREAD,
            "2": palace_read.UNREAD,
        }

    def test_a_port_palace_signed_by_its_own_rule_is_named_in_the_provenance(self):
        answer = self.answer()
        answer.sign_by_rule = (2,)
        assert glue.from_palace(answer).provenance[SIGN_BY_RULE] == [2]

    def test_a_run_whose_every_port_was_given_a_line_names_none(self):
        assert SIGN_BY_RULE not in glue.from_palace(self.answer()).provenance

    def test_the_impedance_the_run_stated_is_the_reference_and_is_named(self):
        """Palace states a port's power-voltage impedance where it read the
        port's voltage, and that is the number the matrix is referenced to."""
        answer = self.answer()
        stated = np.column_stack([np.full(3, np.nan), [420.0, 400.0, 380.0]])
        answer.impedance = stated
        result = glue.from_palace(answer)

        assert np.isnan(result.reference[:, 0]).all()
        np.testing.assert_array_equal(result.reference[:, 1].real, stated[:, 1])
        np.testing.assert_array_equal(result.measured_impedance, result.reference)
        assert result.provenance[IMPEDANCE_STATED] == {"1": palace_read.UNREAD, "2": POWER_VOLTAGE}

    def test_a_guide_stated_at_every_port_is_offered_as_palace_writes_it(self, doc):
        """The per-point file, with the advice Palace can take: a fixed
        reference is refused there, so the caveat does not send the user to it."""
        answer = self.answer()
        answer.impedance = np.column_stack([[420.0, 400.0, 380.0], [420.0, 400.0, 380.0]])
        obj = createEMSParameters(doc)
        store(obj, glue.from_palace(answer))

        export = glue.touchstone_export(obj)

        assert export.per_point and export.refusal == ""
        assert "Palace reports a wave port against its own mode alone" in export.caveat
        assert "ReferencedTo" not in export.caveat
        assert "do not cascade into each other" in export.caveat

    def test_lumped_ports_of_two_resistances_are_offered_in_their_own_terms(self, doc):
        """A lumped port is referenced to its resistance, so the caveat names
        what gives the file one reference, and says nothing about guides."""
        answer = self.answer()
        answer.impedance = np.column_stack([np.full(3, 50.0), np.full(3, 25.0)])
        answer.stated = {1: palace_read.RESISTANCE, 2: palace_read.RESISTANCE}
        obj = createEMSParameters(doc)
        store(obj, glue.from_palace(answer))

        export = glue.touchstone_export(obj)

        assert export.per_point and export.refusal == ""
        assert "give every lumped port the same Resistance" in export.caveat
        assert "wave port" not in export.caveat and "guide" not in export.caveat

    def test_a_declared_mirror_fills_the_column_nobody_drove(self):
        """Palace terminates a port it does not drive in its own mode, so the run
        driving port 2 is the one that drove port 1 with the two exchanged."""
        answer = self.answer(driven=(1,))
        result = glue.from_palace(answer, symmetry=MIRROR)
        assert result.derived == (2,)
        assert result.unmeasured == ()
        np.testing.assert_array_equal(result.parameter(2, 2), answer.matrix[:, 0, 0])
        np.testing.assert_array_equal(result.parameter(1, 2), answer.matrix[:, 1, 0])
        assert result.provenance["symmetry"] == MIRROR
        assert result.provenance["derived_columns"] == [2]

    def test_the_ports_stated_impedances_say_how_far_the_declaration_is_off(self):
        answer = self.answer(driven=(1,))
        answer.impedance = np.column_stack([np.full(3, 400.0), np.full(3, 404.0)])
        result = glue.from_palace(answer, symmetry=MIRROR)
        assert result.provenance["symmetry_impedance_mismatch"] == pytest.approx(0.01, rel=1e-12)

    def test_a_port_stating_no_impedance_leaves_the_mismatch_unsaid(self):
        result = glue.from_palace(self.answer(driven=(1,)), symmetry=MIRROR)
        assert "symmetry_impedance_mismatch" not in result.provenance

    def test_a_mirror_derives_nothing_where_both_ports_were_driven(self):
        """Both columns are measured, and the declaration is then something to
        compare them against rather than to fill from."""
        answer = self.answer()
        result = glue.from_palace(answer, symmetry=MIRROR)
        assert result.derived == ()
        np.testing.assert_array_equal(result.s, glue.from_palace(answer).s)

    def test_no_declaration_leaves_the_undriven_column_unmeasured(self):
        assert glue.from_palace(self.answer(driven=(1,))).unmeasured == (2,)

    def test_the_reference_reads_as_each_ports_own(self):
        """The chart's basis and the property editor both print this line,
        and a nan that reached either would name a figure nobody gave."""
        assert glue.from_palace(self.answer()).reference_description() == (
            "each port's own impedance"
        )

    def test_the_backend_is_what_it_is_filed_under(self, doc):
        analysis = doc.addObject("App::DocumentObjectGroupPython", "EMAnalysis")
        glue.record(analysis, glue.from_palace(self.answer(), title="WR-42"))
        assert glue.stored(analysis, "Palace").port_numbers == (1, 2)
        assert glue.stored(analysis, "openEMS") is None

    def test_it_survives_the_document(self, doc):
        obj = createEMSParameters(doc)
        result = glue.from_palace(self.answer(driven=(1,)))
        store(obj, result)
        back = load(obj)
        assert back.self_referenced == (1, 2)
        assert back.drivers == (1,)
        np.testing.assert_array_equal(back.s, result.s)

    def test_a_partial_answer_is_refused_for_what_nothing_changes(self, doc):
        """A port left undriven is a column another solve would fill - and the
        file would still be refused for the reference, so the advice to solve
        again, or to declare a symmetry, is advice that changes nothing."""
        obj = createEMSParameters(doc)
        store(obj, glue.from_palace(self.answer(driven=(1,))))
        export = glue.touchstone_export(obj)
        assert "states no number" in export.refusal
        assert "symmetric" not in export.refusal

    def test_a_touchstone_export_is_refused_without_advice_that_cannot_be_taken(self, doc):
        """The other refusal tells the user to give every port a fixed
        reference, which this backend's wave port refuses by name."""
        obj = createEMSParameters(doc)
        store(obj, glue.from_palace(self.answer()))
        export = glue.touchstone_export(obj)
        assert "states no number" in export.refusal
        assert "ReferenceImpedance" not in export.refusal


class TestABandWithAHoleInTheDocument:
    """Frequency points that hold no numbers must reopen as holes too.

    A different failure from the partial matrix above: those are missing
    *columns*, these are missing *points*, and the two are stored separately
    because they mean different things. Reopening without the record would give
    a matrix full of nan that nothing could explain - and, worse, one that
    ``write_touchstone`` would happily write.
    """

    def holed(self, points=4, discarded=(1,)):
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

    def test_which_points_are_blank_comes_back(self, doc):
        obj = createEMSParameters(doc)
        store(obj, self.holed(discarded=(1, 3)))
        assert load(obj).discarded == (1, 3)

    def test_the_holes_come_back_as_holes(self, doc):
        obj = createEMSParameters(doc)
        store(obj, self.holed())

        back = load(obj)
        assert np.all(np.isnan(back.s[1]))
        assert np.all(np.isfinite(back.s[[0, 2, 3]]))

    def test_a_reopened_result_still_refuses_to_export(self, doc, tmp_path):
        """The record is what makes the refusal survive the round trip. Without
        it the reopened matrix looks whole and writes nan into a Touchstone
        file, where nothing downstream can tell."""
        obj = createEMSParameters(doc)
        store(obj, self.holed())
        with pytest.raises(ResultError, match="hold no numbers"):
            load(obj).write_touchstone(tmp_path / "reopened")


class TestABandBelowCutoffInTheDocument:
    """A guide whose band starts below its mode's cutoff, reopened.

    Which points carried no power is read off the ports' impedance, which the
    document keeps, so a reopened result says it as the assembled one did and
    exports the points above cutoff.
    """

    def below(self, own=False, points=5, unpowered=2):
        full = matrix(points=points)
        s = np.array(full.s)
        s[:unpowered] = np.nan + 1j * np.nan
        measured = np.full((points, 2), 450.0 + 0j)
        measured[:unpowered] = -3000j
        return SParameters(
            frequency=full.frequency,
            s=s,
            port_numbers=full.port_numbers,
            reference=measured.copy() if own else full.reference,
            measured_impedance=measured,
            self_referenced=full.port_numbers if own else (),
        )

    def holder(self, doc, **kw):
        obj = createEMSParameters(doc)
        store(obj, self.below(**kw))
        return obj

    def test_which_points_carried_no_power_comes_back(self, doc):
        back = load(self.holder(doc))
        assert back.unpowered == (0, 1)
        assert back.discarded == ()

    def test_numbers_where_a_port_carries_no_power_are_not_kept(self, doc):
        """The assembly blanks such a point, but a document written by
        something else may hold numbers there, normalised by nothing."""
        written = self.below()
        written.s[:2] = 0.5 + 0j
        obj = createEMSParameters(doc)
        store(obj, written)
        back = load(obj)
        assert back.kept.tolist() == [False, False, True, True, True]
        assert back.usable().frequency.size == 3

    @pytest.mark.parametrize("own", [False, True])
    def test_the_rest_is_offered_and_the_caveat_says_why(self, doc, own):
        """At its ports' own impedance the reference below cutoff is imaginary,
        and asking about it before the points are dropped refused the whole
        export as one at a complex reference."""
        export = glue.touchstone_export(self.holder(doc, own=own))
        assert export.refusal == ""
        assert export.result.frequency.size == 3
        assert "2 of 5 frequency points hold no numbers" in export.caveat
        assert "Ports 1 and 2 carry no power there" in export.caveat
        assert export.caveat.endswith("Write the other 3?")
        assert export.per_point is False

    def test_the_impedance_quoted_is_above_cutoff(self):
        """Band centre is below cutoff here, where the impedance is a port's
        that carries no power."""
        lines = list(glue.impedance_lines(self.below(unpowered=3)))
        assert all("450.00" in line for line in lines), lines


class TestDecidingWhatATouchstoneExportWrites:
    """``Gui.results.touchstone_export``: the whole judgement, without Qt.

    A ``.sNp`` has a column for every term, a row for every frequency and one
    reference impedance for the lot, and no way to mark any of them as invented
    or as varying. The three ways this workbench can fall short of that are
    different problems with different fixes, and the command on the toolbar is
    wiring on top of the answer.
    """

    def holder(self, doc, result):
        obj = createEMSParameters(doc)
        store(obj, result)
        return obj

    def test_a_whole_matrix_is_written_as_it_stands(self, doc):
        export = glue.touchstone_export(self.holder(doc, matrix()))
        assert export.refusal == ""
        assert export.caveat == ""
        assert export.result is not None
        assert export.suffix == ".s2p"

    def test_a_partial_matrix_is_refused_and_the_ports_are_named(self, doc):
        """No file at all. The fix is another solve, which this cannot do -
        so it says which ports and what to do, rather than offering a dialog."""
        full = matrix()
        s = np.array(full.s)
        s[:, :, 1] = np.nan + 1j * np.nan
        partial = SParameters(
            frequency=full.frequency,
            s=s,
            port_numbers=full.port_numbers,
            reference=full.reference,
            measured_impedance=full.measured_impedance,
            driven=(1,),
        )
        export = glue.touchstone_export(self.holder(doc, partial))
        assert export.result is None
        assert "[2]" in export.refusal
        assert export.refusal.endswith(
            "Set Driven to true on those ports and run again, or, if the device is "
            "mirror-symmetric, set the study's Symmetry to Mirror and run again."
        )

    def referenced(self, reference, self_referenced=()):
        full = matrix()
        return SParameters(
            frequency=full.frequency,
            s=full.s,
            port_numbers=full.port_numbers,
            reference=np.asarray(reference, dtype=complex),
            measured_impedance=full.measured_impedance,
            self_referenced=self_referenced,
        )

    def test_a_complex_reference_is_refused_before_the_dialog(self, doc):
        """The whole point of asking here rather than only in ``write_touchstone``.

        Asked only there, this case reaches the user as a refusal *after* they
        have picked a filename. The wanted text has to be one only the
        description can supply: the refusal's own boilerplate ends "what it was
        measured against is preserved", so anything matching that reads as a
        pass without the description in it.
        """
        full = matrix()
        export = glue.touchstone_export(
            self.holder(doc, self.referenced(full.measured_impedance, (1, 2)))
        )

        assert export.result is None
        assert "port 2: 51 - 2.5j ohm" in export.refusal, export.refusal
        # A port referenced to itself hides ReferenceImpedance, so the advice
        # has to name the property that shows it again.
        assert export.refusal.endswith(
            "Set every port's ReferencedTo to Fixed impedance with the same "
            "ReferenceImpedance and run again, or keep the result in the document, where "
            "its reference is preserved."
        )
        assert "ReferenceImpedance" in export.refusal

    def test_a_real_reference_that_is_not_one_number_is_offered_at_each_point(self, doc):
        """The ordinary 30/75 study, and a guide at its own impedance, which is
        every guide as it is made. Each is written with its reference at each
        point once the user has read what a reader that ignores that form
        takes instead."""
        full = matrix()
        points = full.frequency.size
        guide = np.tile(np.linspace(600.0, 430.0, points)[:, None], (1, 2))
        for reference, self_referenced, wanted in (
            (np.tile([30.0, 75.0], (points, 1)), (), "port 1: 30 ohm"),
            (guide, (1, 2), "each port's own impedance, which spans 170 ohm"),
        ):
            export = glue.touchstone_export(
                self.holder(doc, self.referenced(reference, self_referenced))
            )

            assert export.refusal == ""
            assert export.result is not None
            assert export.per_point
            assert wanted in export.caveat, export.caveat
            assert "may take every port at 50 ohm" in export.caveat
            assert "ReferencedTo to Fixed impedance" in export.caveat
            assert export.caveat.endswith("Write it?")
            # No port says which of a guide's impedances it states, so there is
            # nothing to warn a cascade about.
            assert "cascade" not in export.caveat

    def test_one_reference_is_written_on_the_option_line_without_asking(self, doc):
        export = glue.touchstone_export(self.holder(doc, matrix()))
        assert not export.per_point

    def test_the_reference_is_judged_on_the_points_the_file_holds(self, doc):
        """A point the solves disagreed about is left out, and so is whatever
        reference it carried. The file holds one number, and asks nothing about
        it."""
        full = matrix(points=3)
        s = np.array(full.s)
        s[1] = np.nan + 1j * np.nan
        reference = np.full((3, 2), 50.0)
        reference[1] = 70.0
        holed = replace(full, s=s, reference=reference, discarded=(1,))

        export = glue.touchstone_export(self.holder(doc, holed))

        assert not export.per_point
        assert "50 ohm" not in export.caveat and "Write the other 2?" in export.caveat

    def test_a_band_with_holes_is_offered_rather_than_refused(self, doc):
        """The other shortfall, and the opposite answer: a file *can* be
        written, of the points that hold numbers, once the user has said so."""
        full = matrix(points=5)
        s = np.array(full.s)
        s[[1, 3]] = np.nan + 1j * np.nan
        holed = SParameters(
            frequency=full.frequency,
            s=s,
            port_numbers=full.port_numbers,
            reference=full.reference,
            measured_impedance=full.measured_impedance,
            discarded=(1, 3),
        )
        export = glue.touchstone_export(self.holder(doc, holed))

        assert export.refusal == ""
        assert "2 of 5" in export.caveat
        assert export.result.frequency.size == 3
        assert np.all(np.isfinite(export.result.s))

    def test_the_caveat_says_where_the_holes_are(self, doc):
        """Two holes at the ends of a band are not one dead region between
        them - the message a user acts on has to distinguish those."""
        full = matrix(points=5)
        s = np.array(full.s)
        s[[0, 4]] = np.nan + 1j * np.nan
        holed = SParameters(
            frequency=full.frequency,
            s=s,
            port_numbers=full.port_numbers,
            reference=full.reference,
            measured_impedance=full.measured_impedance,
            discarded=(0, 4),
        )
        assert (
            "lowest 1 GHz, highest 4 GHz" in glue.touchstone_export(self.holder(doc, holed)).caveat
        )

    def test_an_empty_result_object_is_refused_by_its_own_message(self, doc):
        """``load`` already refuses this and says to run the analysis. Catching
        it and rewording would be a second copy of that rule."""
        export = glue.touchstone_export(createEMSParameters(doc))
        assert export.result is None
        assert "Run Simulation to fill it" in export.refusal

    def test_the_default_name_carries_the_document_and_the_result(self, doc):
        obj = self.holder(doc, matrix())
        obj.Label = "Line rev B"
        assert glue.touchstone_export(obj).stem == "Unnamed-Line_rev_B"

    def test_the_default_name_of_a_run_names_the_run(self, doc):
        """Several runs of one study export to several files, and the name is
        what tells them apart."""
        analysis = doc.addObject("App::DocumentObjectGroupPython", "EMAnalysis")
        held = glue.record(analysis, matrix())
        assert glue.touchstone_export(held).stem == "Unnamed-S-Parameters_run_1_openEMS"

    def test_a_label_cannot_redirect_the_write(self, doc):
        """A FreeCAD label is free text. Left alone, ``"../../etc/x"`` is a
        default filename that walks out of the directory the dialog opened in.
        """
        obj = self.holder(doc, matrix())
        obj.Label = "../../etc/passwd"
        stem = glue.touchstone_export(obj).stem
        assert stem == "Unnamed-_etc_passwd"

    def test_a_label_of_nothing_but_punctuation_still_names_a_file(self, doc, monkeypatch):
        """Sanitising can consume the whole name. An empty stem hands Qt a
        dialog opened on a directory with no filename in it."""
        obj = self.holder(doc, matrix())
        # monkeypatch, not an assignment with a finally: the document stub is a
        # session-wide singleton and ``reset_document`` only clears its objects,
        # so an instance attribute set here outlives the test. Assigning
        # restored the *class* attribute in a finally and left the instance one
        # in place - cleanup that reads as protection and is a no-op.
        monkeypatch.setattr(obj.Document, "Name", "///", raising=False)
        obj.Label = "///"
        assert glue.touchstone_export(obj).stem == "sparameters"

    def test_a_result_with_no_title_can_still_be_written(self, doc, tmp_path):
        """scikit-rf demands a filename and falls back to the Network's name,
        which is ``provenance["title"]`` - and a Problem's title defaults to
        empty. Without the filename passed through, this died inside a vendored
        library on "Network must have a name", after the solve."""
        export = glue.touchstone_export(self.holder(doc, matrix()))
        assert not export.result.provenance.get("title")
        assert export.result.write_touchstone(tmp_path / export.stem).is_file()


class TestTheFileThatComesOut:
    """End to end: a document object in, a Touchstone file on disk."""

    def test_it_round_trips_through_the_export(self, doc, tmp_path):
        from Microwave.Results import _skrf

        original = matrix()
        obj = createEMSParameters(doc)
        store(obj, original)

        export = glue.touchstone_export(obj)
        written = export.result.write_touchstone(tmp_path / export.stem)

        assert written.name.endswith(".s2p")
        again = _skrf.module().Network(str(written))
        assert np.allclose(again.s, original.s, rtol=0, atol=1e-9)

    def test_a_trimmed_file_says_what_is_missing(self, doc, tmp_path):
        """The band in the file is shorter than the sweep that produced it, and
        nothing downstream can tell that from a sweep never asked for those
        points. The header is the only place it can be said."""
        full = matrix(points=5)
        s = np.array(full.s)
        s[[1, 3]] = np.nan + 1j * np.nan
        obj = createEMSParameters(doc)
        store(
            obj,
            SParameters(
                frequency=full.frequency,
                s=s,
                port_numbers=full.port_numbers,
                reference=full.reference,
                measured_impedance=full.measured_impedance,
                discarded=(1, 3),
            ),
        )

        export = glue.touchstone_export(obj)
        text = export.result.write_touchstone(tmp_path / export.stem).read_text()
        assert "2 FREQUENCY POINT(S) ARE MISSING" in text
        # linspace(1e9, 4e9, 5), so indices 1 and 3 are 1.75 and 3.25 GHz.
        assert "lowest 1.75 GHz, highest 3.25 GHz" in text


class TestAResultWithNothingLeftInIt:
    """Every frequency point discarded.

    ``from_runs`` refuses this outright, so it arrives only from a document
    written by something else - which ``load``'s own docstring treats as a
    real case. Without a refusal here the caveat offered to write zero points,
    and saying yes reached an IndexError from inside scikit-rf: the exact
    failure that asking ahead exists to prevent.
    """

    def holder(self, doc, points=3):
        full = matrix(points=points)
        obj = createEMSParameters(doc)
        store(
            obj,
            SParameters(
                frequency=full.frequency,
                s=np.full_like(np.asarray(full.s), np.nan + 1j * np.nan),
                port_numbers=full.port_numbers,
                reference=full.reference,
                measured_impedance=full.measured_impedance,
                discarded=tuple(range(points)),
            ),
        )
        return obj

    def test_it_is_refused_rather_than_offered(self, doc):
        export = glue.touchstone_export(self.holder(doc))
        assert export.result is None
        assert export.caveat == ""
        assert "holds no number at any of its 3 frequency points" in export.refusal

    def test_the_refusal_says_what_to_do(self, doc):
        assert "measurement plane" in glue.touchstone_export(self.holder(doc)).refusal

    def test_one_point_surviving_is_still_an_offer(self, doc):
        """The boundary: this refuses "nothing left", not "nearly nothing".
        A one-point Touchstone file is legal and occasionally what you want."""
        full = matrix(points=3)
        s = np.array(full.s)
        s[[0, 2]] = np.nan + 1j * np.nan
        obj = createEMSParameters(doc)
        store(
            obj,
            SParameters(
                frequency=full.frequency,
                s=s,
                port_numbers=full.port_numbers,
                reference=full.reference,
                measured_impedance=full.measured_impedance,
                discarded=(0, 2),
            ),
        )

        export = glue.touchstone_export(obj)
        assert export.refusal == ""
        assert export.result.frequency.size == 1


class TestWhatARunKeeps:
    """The run's own record, beside its matrix: what it was asked, what it said,
    what it warned of and what it cost."""

    def test_the_record_lands_on_the_run(self, doc):
        analysis = doc.addObject("App::DocumentObjectGroupPython", "EMAnalysis")
        begun = glue.begin(analysis, "/runs/board", processes=4)
        kept = replace(begun.ended("one\nWARNING: two", ["WARNING: two"]), wall=12.5, cpu=40.0)
        run = glue.run_of(glue.record(analysis, matrix(), kept))
        assert (run.Number, run.Directory, run.Processes) == (1, "/runs/board/run1", 4)
        assert (run.WallSeconds, run.ProcessorSeconds) == (12.5, 40.0)
        assert run.Log == "one\nWARNING: two"
        assert run.Diagnostics == ["WARNING: two"]
        assert run.Started == begun.started
        assert json.loads(run.Snapshot) == json.loads(json.dumps(begun.snapshot))

    def test_what_it_cost_is_measured_from_when_it_began(self, doc, monkeypatch):
        analysis = doc.addObject("App::DocumentObjectGroupPython", "EMAnalysis")
        clock = iter([100.0, 103.5])
        cpu = iter([10.0, 25.0])
        monkeypatch.setattr(glue.time, "monotonic", lambda: next(clock))
        monkeypatch.setattr(glue, "children_cpu", lambda: next(cpu))
        ended = glue.begin(analysis).ended("")
        assert (ended.wall, ended.cpu) == (3.5, 15.0)

    def test_a_number_a_directory_already_carries_is_passed_over(self, doc, tmp_path):
        """A run stopped or failed keeps nothing in the document and keeps its
        directory, which no later run writes into."""
        analysis = doc.addObject("App::DocumentObjectGroupPython", "EMAnalysis")
        (tmp_path / "run1").mkdir()
        (tmp_path / "run2").mkdir()
        assert glue.begin(analysis, tmp_path).directory == str(tmp_path / "run3")
        assert glue.begin(analysis).directory == ""

    def test_the_number_is_one_past_every_run_in_the_document(self, doc):
        """Two studies in one document number their runs as one sequence, so
        FreeCAD never numbers a run's label of its own accord."""
        one = doc.addObject("App::DocumentObjectGroupPython", "EMAnalysis")
        two = doc.addObject("App::DocumentObjectGroupPython", "EMAnalysis")
        glue.record(one, matrix())
        glue.record(two, matrix())
        assert [run.Label for run in glue.runs_of(two)] == ["Run 2 (openEMS)"]


class TestRunsOnTwoBackends:
    """What a run says it was asked otherwise names nothing its own backend
    does not read, and names the switch where the run before was another
    backend's."""

    def study(self, doc):
        from Microwave.Objects import createEMAnalysis, createEMGmshMesh, createEMSolverPalace

        analysis = createEMAnalysis(doc)
        palace = createEMSolverPalace(doc)
        analysis.addObject(palace)
        analysis.addObject(createEMGmshMesh(doc))
        return analysis, palace

    def changes(self, analysis):
        return {run.Label: (list(run.Changes), run.Label2) for run in glue.runs_of(analysis)}

    def test_another_backends_setting_is_not_said_of_a_run(self, doc):
        analysis, palace = self.study(doc)
        glue.record(analysis, matrix())
        palace.Order = palace.Order + 1
        glue.record(analysis, matrix())
        assert self.changes(analysis)["Run 2 (openEMS)"] == (
            [],
            "against Run 1 (openEMS): nothing it was asked changed",
        )

    @pytest.mark.parametrize(("amended", "said"), [("EMGmshMesh", False), ("EMYeeGrid", True)])
    def test_a_region_is_said_of_a_run_only_on_the_backend_it_amends(self, doc, amended, said):
        from Microwave.Objects import createEMMeshRegion, members
        from Microwave.Objects.kinds import kind_of

        analysis, _ = self.study(doc)
        (settings,) = [one for one in members(analysis) if kind_of(one) == amended]
        region = createEMMeshRegion(doc, settings)
        region.ElementSize = 0.5
        analysis.addObject(region)
        glue.record(analysis, matrix())
        region.ElementSize = 0.25
        glue.record(analysis, matrix())
        changes, _ = self.changes(analysis)["Run 2 (openEMS)"]
        assert any("ElementSize" in change for change in changes) is said

    def test_a_refinement_names_as_kept_only_what_its_own_backend_lays(self, doc):
        """A refinement turns one backend's mesh, so the lengths and counts it
        leaves as stated are that backend's settings' and its regions'."""
        from Microwave.Gui import refine
        from Microwave.Objects import createEMMeshRegion, members
        from Microwave.Objects.kinds import kind_of

        analysis, _ = self.study(doc)
        held = {kind_of(one): one for one in members(analysis)}
        for kind, label in (("EMYeeGrid", "Ours"), ("EMGmshMesh", "Theirs")):
            held[kind].MinElementSize = 0.01
            region = createEMMeshRegion(doc, held[kind])
            region.Label, region.ElementSize = label, 0.5
            analysis.addObject(region)
        from Microwave.Objects import createEMYeeGrid

        stray = createEMMeshRegion(doc, createEMYeeGrid(doc))
        stray.Label, stray.ElementSize = "Elsewhere", 0.5
        analysis.addObject(stray)
        said = refine._kept(analysis, ("EMSolverOpenEMS", "EMYeeGrid"))
        assert [line.split(" stays ")[0] for line in said] == [
            f"{held['EMYeeGrid'].Label} MinElementSize",
            f"{held['EMYeeGrid'].Label} MinElementsAcross",
            "Ours ElementSize",
        ]

    def test_a_record_narrowed_to_a_backend_keeps_only_what_that_backend_reaches(self):
        """The drawing a region of the other backend alone names, and that
        region's shapes, leave the record with the region."""
        from Microwave.Results import snapshots

        asked = {
            "members": [
                {"name": "Grid", "kind": "EMYeeGrid"},
                {"name": "Gmsh", "kind": "EMGmshMesh"},
                {"name": "Ours", "kind": "EMMeshRegion", "MeshSettings": {"name": "Grid"}},
                {"name": "Theirs", "kind": "EMMeshRegion", "MeshSettings": {"name": "Gmsh"}},
            ],
            "drawn": [
                {"name": "Board", glue.BY: ["Ours", "Theirs"]},
                {"name": "Helper", glue.BY: ["Theirs"]},
                {"name": "Sheet", glue.BY: ["Study"]},
            ],
            snapshots.SHAPES: {"Ours": "a", "Theirs": "b", "Grid": "c", "Gmsh": "d"},
        }
        kept = glue.narrowed(asked, ("EMSolverOpenEMS", "EMYeeGrid"))
        assert [member["name"] for member in kept["members"]] == ["Grid", "Ours"]
        assert [obj["name"] for obj in kept["drawn"]] == ["Board", "Sheet"]
        assert kept[snapshots.SHAPES] == {"Ours": "a", "Grid": "c"}

    def test_each_backends_regions_are_neither_new_nor_gone_across_a_switch(self, doc):
        from Microwave.Objects import createEMMeshRegion, members
        from Microwave.Objects.kinds import kind_of

        analysis, _ = self.study(doc)
        for one in list(members(analysis)):
            if kind_of(one) in ("EMYeeGrid", "EMGmshMesh"):
                analysis.addObject(createEMMeshRegion(doc, one))
        glue.record(analysis, matrix())
        glue.record(analysis, from_backend("Palace", matrix()))
        assert self.changes(analysis)["Run 2 (Palace)"][0] == ["solved on Palace"]

    def test_its_own_region_changed_across_another_backends_run_is_said_since_its_last(self, doc):
        from Microwave.Objects import createEMMeshRegion, members
        from Microwave.Objects.kinds import kind_of

        analysis, _ = self.study(doc)
        (grid,) = [one for one in members(analysis) if kind_of(one) == "EMYeeGrid"]
        region = createEMMeshRegion(doc, grid)
        region.Label, region.ElementSize = "Ours", 0.5
        analysis.addObject(region)
        glue.record(analysis, matrix())
        glue.record(analysis, from_backend("Palace", matrix()))
        region.ElementSize = 0.25
        glue.record(analysis, matrix())
        changes, _ = self.changes(analysis)["Run 3 (openEMS)"]
        assert any(
            change.startswith("Ours ElementSize") and change.endswith("since Run 1 (openEMS)")
            for change in changes
        ), changes

    def test_a_run_after_another_backends_says_the_switch_and_no_solver_new(self, doc):
        analysis, _ = self.study(doc)
        glue.record(analysis, matrix())
        glue.record(analysis, from_backend("Palace", matrix()))
        assert self.changes(analysis)["Run 2 (Palace)"] == (
            ["solved on Palace"],
            "against Run 1 (openEMS): solved on Palace",
        )

    def test_its_own_setting_changed_across_another_backends_run_is_said_since_its_last(self, doc):
        analysis, palace = self.study(doc)
        glue.record(analysis, from_backend("Palace", matrix()))
        palace.Order = palace.Order + 1
        glue.record(analysis, from_backend("Palace", matrix()))
        was = palace.Order
        palace.Order = was + 1
        glue.record(analysis, matrix())
        glue.record(analysis, from_backend("Palace", matrix()))
        assert self.changes(analysis)["Run 4 (Palace)"][0] == [
            "solved on Palace",
            f"Solver (Palace) Order {was} -> {was + 1} since Run 2 (Palace)",
        ]

    def test_a_run_whose_backend_has_no_run_before_says_the_switch_alone(self, doc):
        analysis, palace = self.study(doc)
        glue.record(analysis, matrix())
        palace.Order = palace.Order + 1
        glue.record(analysis, from_backend("Palace", matrix()))
        assert self.changes(analysis)["Run 2 (Palace)"][0] == ["solved on Palace"]

    def test_a_run_read_back_holds_no_solver_its_backend_does_not_read(self, doc):
        analysis, _ = self.study(doc)
        held = glue.record(analysis, matrix())
        asked = glue.asked_of(glue.run_of(held))
        assert {member["kind"] for member in asked["members"]} >= {"EMSolverOpenEMS", "EMYeeGrid"}
        assert not {member["kind"] for member in asked["members"]} & {
            "EMSolverPalace",
            "EMGmshMesh",
        }

    def test_the_overlay_figures_a_setting_only_for_the_runs_that_read_it(self, doc):
        from Microwave.Gui.plot_s_params import UNHELD, overlay_chart

        analysis, palace = self.study(doc)
        held = [glue.record(analysis, from_backend("Palace", matrix()))]
        held.append(glue.record(analysis, matrix()))
        palace.Order = palace.Order + 1
        held.append(glue.record(analysis, from_backend("Palace", matrix())))
        drawn = overlay_chart(glue.overlaid(held), chosen=(2, 1))
        assert [series.name for series in drawn.series] == [
            f"S21, Run 1 (Palace), Solver (Palace) Order {palace.Order - 1}",
            "S21, Run 2 (openEMS)",
            f"S21, Run 3 (Palace), Solver (Palace) Order {palace.Order}",
        ]
        order = drawn.table.columns.index("Solver (Palace) Order")
        assert [row[order] for row in drawn.table.rows] == [
            str(palace.Order - 1),
            UNHELD,
            str(palace.Order),
        ]

    def test_its_own_setting_changed_between_two_of_its_runs_is_said_once(self, doc):
        from Microwave.Objects import solver_of

        analysis, _ = self.study(doc)
        solver = solver_of(analysis)
        was = solver.MaxTimesteps
        glue.record(analysis, matrix())
        solver.MaxTimesteps = was * 2
        glue.record(analysis, matrix())
        assert self.changes(analysis)["Run 2 (openEMS)"][0] == [
            f"Solver (openEMS) MaxTimesteps {was} -> {was * 2}"
        ]

    def test_a_run_that_kept_no_record_is_compared_with_nothing(self, doc):
        analysis, _ = self.study(doc)
        first = glue.run_of(glue.record(analysis, from_backend("Palace", matrix())))
        first.Snapshot = ""
        glue.record(analysis, matrix())
        glue.record(analysis, from_backend("Palace", matrix()))
        found = self.changes(analysis)
        assert found["Run 2 (openEMS)"] == ([], "")
        assert found["Run 3 (Palace)"][0] == ["solved on Palace"]

    def test_a_table_no_run_kept_has_no_figure_and_no_mark(self, doc):
        from Microwave.Gui.plot_s_params import overlay_chart
        from Microwave.Objects.results import createEMSParameters, store

        analysis, palace = self.study(doc)
        held = [glue.record(analysis, from_backend("Palace", matrix()))]
        palace.Order = palace.Order + 1
        held.append(glue.record(analysis, from_backend("Palace", matrix())))
        alone = createEMSParameters(doc)
        store(alone, matrix())
        drawn = overlay_chart(glue.overlaid([*held, alone]), chosen=(2, 1))
        order = drawn.table.columns.index("Solver (Palace) Order")
        assert drawn.table.rows[-1][order] == ""

    def test_a_refinement_is_not_refused_for_another_backends_setting(self, doc):
        from Microwave.Gui import refine
        from Microwave.Gui.openems_steps import OpenEMSSteps

        analysis, palace = self.study(doc)
        run = glue.run_of(glue.record(analysis, matrix()))
        palace.Order = palace.Order + 1
        refined = refine.refinement(run, OpenEMSSteps.REFINES, OpenEMSSteps.OWNS)
        assert refined.run is run

    def test_a_backend_this_layer_does_not_know_is_said_as_the_switch_alone(self, doc):
        from Microwave.Objects.kinds import recipe_kinds, solver_kinds

        analysis, _ = self.study(doc)
        glue.record(analysis, matrix())
        glue.record(analysis, from_backend("NEC2", matrix()))
        glue.record(analysis, matrix())
        found = self.changes(analysis)
        assert found["Run 2 (NEC2)"][0] == ["solved on NEC2"]
        assert found["Run 3 (openEMS)"][0] == ["solved on openEMS"]
        [unknown] = [run for run in glue.runs_of(analysis) if run.Solver == "NEC2"]
        known = solver_kinds() | recipe_kinds()
        assert not {member["kind"] for member in glue.asked_of(unknown)["members"]} & known

    def bound(self, doc, palace, chained):
        """A variable set whose Order the Palace solver's Order is bound to, and
        which, ``chained``, is itself bound to its Base. Nothing else reads it."""
        held = doc.addObject("App::FeaturePython", "Vars")
        held.addProperty("App::PropertyInteger", "Order", "Vars")
        held.Order = palace.Order
        if chained:
            held.addProperty("App::PropertyInteger", "Base", "Vars")
            held.Base = palace.Order - 1
            held.setExpression("Order", "Base + 1")
        palace.setExpression("Order", "Vars.Order")
        object.__setattr__(palace, "OutList", [held])
        return held

    def test_what_only_another_backends_solver_reads_is_not_said_through_a_chain(self, doc):
        from Microwave.Gui import refine
        from Microwave.Gui.openems_steps import OpenEMSSteps

        analysis, palace = self.study(doc)
        held = self.bound(doc, palace, chained=True)
        run = glue.run_of(glue.record(analysis, matrix()))
        held.Base += 1
        held.Order += 1
        palace.Order = held.Order
        glue.record(analysis, matrix())
        assert self.changes(analysis)["Run 2 (openEMS)"][0] == []
        assert refine.refinement(run, OpenEMSSteps.REFINES, OpenEMSSteps.OWNS).run is run

    def test_a_value_only_another_backends_solver_reads_marks_the_run_as_holding_none(self, doc):
        from Microwave.Gui.plot_s_params import UNHELD, overlay_chart

        analysis, palace = self.study(doc)
        held = self.bound(doc, palace, chained=False)
        runs = []
        for backend in ("Palace", "openEMS", "Palace"):
            runs.append(glue.record(analysis, from_backend(backend, matrix())))
            held.Order += 1
            palace.Order = held.Order
        drawn = overlay_chart(glue.overlaid(runs), chosen=(2, 1))
        assert drawn.series[1].name == "S21, Run 2 (openEMS)"
        assert [row[drawn.table.columns.index("Order")] for row in drawn.table.rows][1] == UNHELD
