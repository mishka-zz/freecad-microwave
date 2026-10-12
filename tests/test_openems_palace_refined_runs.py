# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A run solved again one step finer from its menu, on the shipped WR-42 guide,
on each backend, through the panel a user presses Run on.

What the stubs cannot say is the engines': whether the step reaches a real
solve, whether the refined run is filed beside the run it refines and says what
changed against it, and whether the move its log states is the move between the
two stored matrices. ``openems_palace_refined_runs_probe.py`` does it under a
real FreeCAD with its main window up.

Held back to the release run: two solves compared with one another are a study
of the mesh, which no fixture of the graph names.
"""

import os
import re

import pytest

from Microwave.Gui import refine

from .conftest import probe_manifest

pytestmark = [pytest.mark.slow, pytest.mark.release]

PROBE = os.path.join(os.path.dirname(__file__), "openems_palace_refined_runs_probe.py")

LABELS = ["Run 1 (openEMS)", "Run 2 (Palace)", "Run 3 (openEMS)", "Run 4 (Palace)"]

#: Each refined run, and the run it refines.
REFINES = {"Run 3 (openEMS)": "Run 1 (openEMS)", "Run 4 (Palace)": "Run 2 (Palace)"}


@pytest.fixture(scope="module")
def made(tmp_path_factory, interpreter):
    out = tmp_path_factory.mktemp("refined")
    manifest = probe_manifest(
        PROBE,
        out,
        "REFINED_OUT",
        key=None,
        environment={"MICROWAVE_OPENEMS_PYTHON": str(interpreter)},
    )
    if "missing" in manifest:
        pytest.skip(f"this machine cannot solve on both backends: {manifest['missing']}")
    return manifest


def run(made, label):
    return next(found for found in made["runs"] if found["label"] == label)


def test_each_refined_run_is_kept_beside_the_run_it_refines(made):
    assert [found["label"] for found in made["runs"]] == LABELS


@pytest.mark.parametrize("label", list(REFINES))
def test_the_refined_run_says_what_changed_against_the_run_it_refines(made, label):
    found = run(made, label)
    assert found["description"].startswith(f"against {REFINES[label]}: ")
    turned = {
        "Run 3 (openEMS)": {"ElementsPerWavelength", "MaxTimesteps"},
        "Run 4 (Palace)": {"Order"},
    }[label]
    assert {line.split()[-4] for line in found["changes"]} == turned


def test_the_step_reached_the_solve_and_the_study_kept_its_own_values(made):
    before = made["held"]["before"]
    assert made["held"]["after"] == before
    asked = {
        found["label"]: {
            (member["kind"], key): member[key]
            for member in found["snapshot"]["members"]
            for key in ("ElementsPerWavelength", "MaxTimesteps", "Order")
            if key in member
        }
        for found in made["runs"]
    }
    openems, palace = asked["Run 3 (openEMS)"], asked["Run 4 (Palace)"]
    assert openems[("EMYeeGrid", "ElementsPerWavelength")] == refine.finer(before["per wavelength"])
    assert openems[("EMSolverOpenEMS", "MaxTimesteps")] == refine.longer(before["timesteps"])
    assert palace[("EMSolverPalace", "Order")] == before["order"] + 1


@pytest.mark.parametrize("label", list(REFINES))
def test_the_move_the_log_states_is_the_move_between_the_stored_matrices(made, label):
    found = run(made, label)
    [head] = [line for line in found["log"] if line.startswith(f"From {REFINES[label]} to {label}")]
    assert head.endswith(refine.NOT_CONVERGED)
    stated = float(re.search(r"at most ([0-9.e+-]+)", head).group(1))
    apart = made["apart"][label]
    assert apart["points"] > 0
    assert stated == pytest.approx(apart["largest"], rel=0.06, abs=0.0)


@pytest.mark.parametrize("label", list(REFINES))
def test_the_panel_said_what_the_step_turns_before_run_was_pressed(made, label):
    offered = made["said"][f"{REFINES[label]} offered"]
    assert offered[0].startswith(f"Refines {REFINES[label]}: ")


@pytest.mark.parametrize("label", list(REFINES))
def test_the_panel_drew_the_two_runs_with_their_difference(made, label):
    [drawn] = made["drawn"][REFINES[label]]
    assert drawn["tables"] == [REFINES[label], label]
    assert drawn["against"] == 0
    assert drawn["chosen"] is not None
