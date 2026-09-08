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
from Microwave.Results.sparameters import ResultError, SParameters


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
        distinguishes it from a port asked for 50. A footnote naming it apart
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


class TestItRefusesRatherThanGuesses:
    def test_an_empty_object_says_to_run_the_analysis(self, doc):
        obj = createEMSParameters(doc)
        with pytest.raises(ResultError, match="no S-matrix yet"):
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
    def __init__(self, z0, incident=None):
        self.z0 = z0
        #: A run that excited nothing returns zeros here, and from_runs refuses
        #: on it before the nan it causes can reach scikit-rf.
        self.incident = np.ones_like(np.asarray(z0)) if incident is None else incident


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
        return FakePort(np.full(self.frequency.size, self._impedances[number]))

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
        assert glue.find_results(analysis) is None

        stored = glue.record(analysis, matrix())

        assert glue.find_results(analysis) is stored
        assert stored in analysis.Group

    def test_the_filed_matrix_does_not_arrive_marked_stale(self, doc):
        """S-Parameters arrive under the analysis carrying the touched mark,
        and only Recompute clears it. The mark tells the user the object in
        front of them is out of date - on a matrix measured a second earlier.
        Purged rather than recomputed: ``execute`` has nothing to do."""
        analysis = self.analysis(doc)
        stored = glue.record(analysis, matrix())
        assert "Touched" not in stored.State

    def test_filing_the_matrix_is_one_undo_step(self, doc):
        """Untransacted, Ctrl-Z after a run leaves the S-parameters in the tree
        and undoes something earlier instead. Named for what disappears."""
        analysis = self.analysis(doc)
        glue.record(analysis, matrix())
        assert doc.transactions == [("Store Results", "commit")]

    def test_the_result_object_is_created_inside_it(self, doc, monkeypatch):
        """As for the mesh: wrapping nothing is not the fix. Hoisting the
        creation above the block leaves the matrix in the tree after Ctrl-Z, and
        nothing but this notices."""
        analysis = self.analysis(doc)
        opened_when = []
        real = glue.createEMSParameters

        def spy(document):
            opened_when.append(document._open)
            return real(document)

        monkeypatch.setattr(glue, "createEMSParameters", spy)
        glue.record(analysis, matrix())
        assert opened_when == ["Store Results"]

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

    def test_a_second_run_overwrites_rather_than_accumulates(self, doc):
        """A study holds the answer to the question it currently asks.

        Keeping every historical matrix would turn a document into a filing
        cabinet nobody asked for; a Touchstone export is how you keep one.
        """
        analysis = self.analysis(doc)
        first = glue.record(analysis, matrix(points=4))
        second = glue.record(analysis, matrix(points=6))

        assert first is second
        assert len([m for m in analysis.Group if m is second]) == 1
        assert load(second).frequency.size == 6

    def test_it_reads_back_what_it_filed(self, doc):
        analysis = self.analysis(doc)
        glue.record(analysis, matrix(port_numbers=(2, 5)))
        assert glue.stored(analysis).port_numbers == (2, 5)

    def test_an_analysis_with_no_result_has_none(self, doc):
        assert glue.stored(self.analysis(doc)) is None

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

        found = glue.find_results(analysis)
        # By kind, not by position. Asserting only ``found is stored`` passed
        # with the kind test removed entirely: ``record`` and ``find_results``
        # would both pick the same wrong member and agree with each other.
        assert type(found.Proxy).__name__ == "EMSParameters"
        assert found is stored

    def test_a_result_object_created_but_never_filled_reads_as_absent(self, doc):
        """``store`` can fail - a read-only document, a full disk - and the
        object is created first. Empty is *absent*, and only damage raises."""
        analysis = self.analysis(doc)
        analysis.addObject(createEMSParameters(doc))

        assert glue.stored(analysis) is None

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
            glue.stored(analysis)

    def test_two_studies_do_not_share_one_matrix(self, doc):
        """Per analysis, not per document. A document-wide lookup would
        overwrite the first study's answer with the second's."""
        one, two = self.analysis(doc), self.analysis(doc)
        glue.record(one, matrix(port_numbers=(1, 2)))
        glue.record(two, matrix(port_numbers=(3,)))

        assert glue.stored(one).port_numbers == (1, 2)
        assert glue.stored(two).port_numbers == (3,)

    def test_each_run_gets_its_own_directory(self):
        assert glue.directory_for("/tmp/board_sim", 2).endswith("board_sim/port2")

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
        assert "excitation" in export.refusal

    def test_a_reference_the_format_cannot_hold_is_refused_before_the_dialog(self, doc):
        """The whole point of asking here rather than only in ``write_touchstone``.

        Asked only there, this case reaches the user as scikit-rf's own
        sentence, out of a vendored library, *after* they have picked a
        filename - and it does so for the ordinary 30/75 study, not only for a
        port referenced to itself.
        """
        full = matrix()
        for reference, self_referenced, wanted in (
            (np.tile([30.0, 75.0], (full.frequency.size, 1)), (), "port 1: 30 ohm"),
            # Each port at its own complex impedance. The wanted text has to be
            # one only the description can supply: the refusal's own boilerplate
            # ends "what it was measured against is preserved", so anything
            # matching that reads as a pass without the description in it.
            (full.measured_impedance, (1, 2), "port 2: 51 - 2.5j ohm"),
        ):
            result = SParameters(
                frequency=full.frequency,
                s=full.s,
                port_numbers=full.port_numbers,
                reference=np.asarray(reference, dtype=complex),
                measured_impedance=full.measured_impedance,
                self_referenced=self_referenced,
            )
            export = glue.touchstone_export(self.holder(doc, result))

            assert export.result is None
            assert wanted in export.refusal, export.refusal
            assert "ReferenceImpedance" in export.refusal

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
        assert "Run the analysis" in export.refusal

    def test_the_default_name_carries_the_document_and_the_result(self, doc):
        obj = self.holder(doc, matrix())
        obj.Label = "Line rev B"
        assert glue.touchstone_export(obj).stem == "Unnamed-Line_rev_B"

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
        assert "all 3 of its frequency points" in export.refusal

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
