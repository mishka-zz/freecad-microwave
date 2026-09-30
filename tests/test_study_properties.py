# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the study states, and whether each backend answers it.

``EMAnalysis`` holds what the device is asked, in quantities the answer can be
compared against: the band, the points across it, the device's symmetry, and
the smallest response that matters. Nothing on it names a method. How one
backend covers the band - a pulse, timesteps, an adaptive sweep - is that
backend's solver object's.

Held the way ``tests/test_mesh_recipes.py`` holds the mesh kinds:

* one table, so a property that names a method is caught on the study rather
  than by whichever test its author had in mind;
* a static net, so a property no backend reads is a silent no-op caught here;
* a case per demand on each backend, because the net cannot see whether a read
  it finds reaches the run.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest

from Microwave.Objects.analysis import (
    MIRROR_SYMMETRY,
    NO_SYMMETRY,
    createEMAnalysis,
)
from Microwave.Objects.kinds import classes, recipe_kinds, solver_kinds

#: Everything the study carries, and the value each arrives at.
CARRIES = {
    "FrequencyStart": 1.0e9,
    "FrequencyStop": 10.0e9,
    "NumFrequencyPoints": 501,
    "Symmetry": NO_SYMMETRY,
    "SmallestResponse": 0.0,
}

#: What each adapter reads off the study to state its run.
READ_BY_EVERY_ADAPTER = ("FrequencyStart", "FrequencyStop", "NumFrequencyPoints")

#: What every adapter reads through one reader in ``Solvers/properties.py``,
#: by the name of the property and then of the reader each adapter calls.
READ_THROUGH_A_SHARED_READER = {"SmallestResponse": "smallest_response"}

#: What the result layer reads to assemble each backend's matrix: a mirror fills
#: the column a run did not drive, whichever backend ran it.
READ_BY_THE_RESULT_LAYER = ("Symmetry",)

#: A property that names how one backend covers the band.
RETIRED = ("Waveform",)


def _unread(properties, where):
    """Which of ``properties`` the source at ``where``, a file or a directory,
    never reads."""
    from .test_mesh_recipes import _unread as unread

    if where.is_file():
        return [name for name in properties if name not in where.read_text(encoding="utf-8")]
    return unread(properties, where)


def _called(where):
    """Every name called as a plain function anywhere under ``where``."""
    called = set()
    for path in sorted(where.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(node.func.id)
    return called


def _held(obj, name):
    found = getattr(obj, name)
    return float(getattr(found, "Value", found)) if name.startswith("Frequency") else found


class TestTheStudyDeclaresWhatIsAsked:
    def test_it_carries_what_the_table_says_and_nothing_else(self, doc):
        study = createEMAnalysis(doc)
        from Microwave.Objects.staleness import FREECADS_OWN

        carried = set(study.PropertiesList) - set(FREECADS_OWN) - {"Group", "_GroupTouched"}
        assert carried == set(CARRIES)

    def test_it_arrives_at_the_values_it_was_argued_into(self, doc):
        study = createEMAnalysis(doc)
        assert {name: _held(study, name) for name in CARRIES} == CARRIES

    def test_it_carries_no_property_of_a_solver_or_a_recipe(self, doc):
        """A property one backend's object carries names that backend's method,
        so the same name on the study would be a second copy of it."""
        from tests.conftest import declared_by

        study = set(createEMAnalysis(doc).PropertiesList)
        methods = solver_kinds() | recipe_kinds() | {"EMMeshPolicy"}
        for kind, cls in classes().items():
            if kind not in methods:
                continue
            module = cls.__module__.rsplit(".", 1)[-1]
            shared = (study & set(declared_by(module, kind))) - {"Label"}
            assert not shared, f"{kind} also carries {sorted(shared)}"

    def test_the_excitation_is_the_openems_solvers(self, doc):
        study = createEMAnalysis(doc)
        assert not any(hasattr(study, name) for name in RETIRED)

    def test_every_property_is_placed_in_one_reader(self):
        """A property added tomorrow fails here until somebody says who reads it."""
        placed = [READ_BY_EVERY_ADAPTER, READ_THROUGH_A_SHARED_READER, READ_BY_THE_RESULT_LAYER]
        assert sorted(name for reader in placed for name in reader) == sorted(CARRIES)


class TestNoStudyPropertyIsANoOp:
    """Every property the study offers is read by what answers it."""

    @pytest.mark.parametrize("backend", ["openems", "palace"])
    def test_each_adapter_reads_what_it_states_its_run_from(self, backend):
        from .test_mesh_recipes import ADAPTERS

        assert _unread(READ_BY_EVERY_ADAPTER, ADAPTERS / backend) == []

    @pytest.mark.parametrize("backend", ["openems", "palace"])
    def test_each_adapter_calls_the_shared_reader(self, backend):
        from .test_mesh_recipes import ADAPTERS

        shared = ADAPTERS / "properties.py"
        assert _unread(READ_THROUGH_A_SHARED_READER, shared) == []
        assert set(READ_THROUGH_A_SHARED_READER.values()) <= _called(ADAPTERS / backend)

    def test_the_result_layer_reads_what_it_assembles_from(self):
        from .test_mesh_recipes import ADAPTERS

        results = ADAPTERS.parent / "Gui" / "results.py"
        assert _unread(READ_BY_THE_RESULT_LAYER, results) == []

    def test_a_waveform_left_on_the_study_reaches_no_run(self):
        """``Waveform`` is still a name the openEMS adapter reads, off its own
        solver, so the net cannot say where it is read from. A study carrying
        one the adapter would refuse is translated as though it carried none."""
        from Microwave.Solvers.openems import document

        from .test_openems_document_translation import analysis, model

        stranded = analysis()
        stranded.Waveform = "Sinusoid"
        assert document.problem(model(analysis=stranded).Objects[0]) is not None


class TestEachDemandMovesTheOpenemsRun:
    def problem(self, **study):
        from Microwave.Solvers.openems import document

        from .test_openems_document_translation import analysis, model

        return document.problem(model(analysis=analysis(**study)).Objects[0])

    def test_the_band_is_the_one_the_pulse_covers(self):
        asked = self.problem(FrequencyStart=2e9, FrequencyStop=9e9, NumFrequencyPoints=33)
        assert (asked.frequency.start, asked.frequency.stop) == (2e9, 9e9)
        assert asked.frequency.points == 33

    def test_the_smallest_response_is_what_the_record_is_held_to(self):
        assert self.problem(SmallestResponse=-40.0).smallest_response == pytest.approx(0.01)


class TestEachDemandMovesThePalaceRun:
    def problem(self, **study):
        from Microwave.Solvers.palace.document import problem

        from .test_palace_adapter import guide

        return problem(guide(**study)[0])

    def test_the_band_is_the_one_swept(self):
        asked = self.problem(FrequencyStart=21e9, FrequencyStop=25e9, NumFrequencyPoints=5)
        assert (asked.sweep.start, asked.sweep.stop, asked.sweep.points) == (21e9, 25e9, 5)

    def test_the_smallest_response_is_what_the_reduced_model_is_held_to(self):
        assert self.problem(SmallestResponse=-40.0).smallest == pytest.approx(0.01)

    def test_a_response_that_is_not_a_depth_is_refused_by_name(self):
        from Microwave.Solvers.errors import TranslationError

        with pytest.raises(TranslationError, match="SmallestResponse"):
            self.problem(SmallestResponse=3.0)


class TestTheMirrorIsAppliedWhicheverBackendRan:
    """The declaration is about the device, so each backend's matrix is filled
    from it alike. ``tests/test_sparameters.py`` holds the fill on openEMS runs."""

    def test_the_study_declares_it_in_the_result_layers_words(self):
        from types import SimpleNamespace

        from Microwave.Gui import results
        from Microwave.Results.sparameters import MIRROR

        assert results.declared_symmetry(SimpleNamespace(Symmetry=MIRROR_SYMMETRY)) == MIRROR
        assert results.declared_symmetry(SimpleNamespace(Symmetry=NO_SYMMETRY)) is None

    def test_a_palace_answer_driven_from_one_port_is_filled(self):
        from types import SimpleNamespace

        from Microwave.Gui import results
        from Microwave.Results.sparameters import MIRROR

        frequency = np.linspace(20e9, 26e9, 3)
        answer = SimpleNamespace(
            frequency=frequency,
            out=(1, 2),
            driven=(1,),
            matrix=np.full((3, 2, 1), 0.3 + 0.1j),
            flux=np.zeros((3, 2, 1)),
            radiated=None,
            dissipates=False,
            modelled=(),
        )
        assert results.from_palace(answer, symmetry=MIRROR).derived == (2,)
