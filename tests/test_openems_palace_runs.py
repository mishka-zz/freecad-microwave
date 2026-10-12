# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Every run kept, on the shipped WR-42 guide, solved through each backend's own
panel, edited, and solved again on each.

What the stubs cannot say is FreeCAD's and the engines': whether a run filed by
a real panel holds its matrix, its log and its cost in a directory of its own,
whether the runs are drawn on one set of axes told apart by what changed,
whether undo takes a run back whole, and whether all of it survives a save and
a reopen. ``openems_palace_runs_probe.py`` does it under a real FreeCAD with its
main window up, and reopens the saved file in another.

Held back to the release run: solves compared with one another are a study of the
runs, which no fixture of the graph names.
"""

import os

import pytest

from .conftest import probe_manifest

pytestmark = [pytest.mark.slow, pytest.mark.release]

HERE = os.path.dirname(__file__)
PROBE = os.path.join(HERE, "openems_palace_runs_probe.py")

LABELS = ["Run 1 (openEMS)", "Run 2 (Palace)", "Run 3 (openEMS)", "Run 4 (Palace)"]


@pytest.fixture(scope="module")
def made(tmp_path_factory, interpreter):
    out = tmp_path_factory.mktemp("runs")
    manifest = probe_manifest(
        PROBE,
        out / "made",
        "RUNS_OUT",
        key=None,
        environment={"MICROWAVE_OPENEMS_PYTHON": str(interpreter)},
    )
    if "missing" in manifest:
        pytest.skip(f"this machine cannot solve on both backends: {manifest['missing']}")
    return out, manifest


@pytest.fixture(scope="module")
def reopened(made):
    out, manifest = made
    return probe_manifest(
        PROBE,
        out / "reopened",
        "RUNS_OUT",
        key=None,
        environment={"RUNS_REOPENED": manifest["document"]},
    )


def test_each_run_is_kept_beside_the_last_in_a_group_of_its_own(made):
    _, manifest = made
    runs = manifest["runs"]
    assert [run["label"] for run in runs] == LABELS
    assert [run["kinds"] for run in runs] == [["EMSParameters"]] * 4
    assert [run["result"] for run in runs] == [
        "S-Parameters (run 1, openEMS)",
        "S-Parameters (run 2, Palace)",
        "S-Parameters (run 3, openEMS)",
        "S-Parameters (run 4, Palace)",
    ]
    assert all(run["ports"] == [1, 2] for run in runs)


def test_each_run_wrote_into_a_directory_of_its_own(made):
    _, manifest = made
    directories = [run["directory"] for run in manifest["runs"]]
    assert len(set(directories)) == 4
    assert all(run["wrote"] for run in manifest["runs"])


def test_each_run_keeps_what_it_said_and_what_it_warned_of(made):
    _, manifest = made
    for run in manifest["runs"]:
        assert run["log"][0].startswith("FreeCAD Microwave ")
        assert set(run["diagnostics"]) <= set(run["log"])
        assert all(line.startswith(("WARNING: ", "[warn] ")) for line in run["diagnostics"])


def test_each_run_keeps_what_it_cost(made):
    _, manifest = made
    for run in manifest["runs"]:
        assert run["wall"] > 0.0
        assert run["cpu"] > 0.0
        assert run["processes"] >= 1


def test_the_table_under_the_overlay_says_the_band_moved_and_nothing_else(made):
    """Where a run wrote is the machine's and changes no answer, so the first
    Palace run setting its directory is not said: the runs before the edit
    share one row of figures, and the runs after it another."""
    _, manifest = made
    drawn = manifest["overlay"]
    assert drawn["notes"][0].startswith("Each table referenced to ")
    table = drawn["table"]
    assert table["columns"] == [
        "Table",
        "Grid",
        "WR-42 waveguide FrequencyStart (GHz)",
        "WR-42 waveguide FrequencyPoints",
    ]
    assert [row[0] for row in table["rows"]] == LABELS
    first, second, third, fourth = (row[1:] for row in table["rows"])
    assert first == second and third == fourth and first != third
    assert all(a != b for a, b in zip(first, third, strict=True))


def test_every_run_is_drawn_on_one_set_of_axes_with_its_difference(made):
    """Each on its own grid; a difference against the first on the coarser of
    the two grids inside the band both cover."""
    _, manifest = made
    drawn = manifest["overlay"]
    first, *later = manifest["runs"]
    named = [", ".join(name.split(", ")[:2]) for name in drawn["series"][: len(LABELS)]]
    assert named == [f"S21, {label}" for label in LABELS]
    for run in later:
        apart = drawn["differences"][f"|S21 difference|, {run['label']} against {first['label']}"]
        low, high = (
            max(first["frequency"][0], run["frequency"][0]),
            min(first["frequency"][-1], run["frequency"][-1]),
        )
        grids = [
            [value for value in grid["frequency"] if low <= value <= high] for grid in (first, run)
        ]
        assert apart == min(grids, key=len)
    delays = [", ".join(name.split(", ")[:2]) for name in drawn["delay"]]
    assert delays == [f"S21, {label}" for label in LABELS]


def test_the_impedance_of_the_runs_is_refused_saying_a_guide_has_none(made):
    """A guide carries nothing below its cutoff, so no sweep gives it a
    low-pass step response, and the refusal names each run rather than asking
    for a sweep from near DC."""
    _, manifest = made
    said = manifest["impedance"]["refused"]
    for label in LABELS:
        assert f"{label}: cannot take a step response at port 1: it is a waveguide port" in said
    assert "near DC" not in said


def test_the_newest_run_is_compared_with_the_other_backends_newest(made):
    """What the last run's log says of the other backend as it is filed."""
    _, manifest = made
    said = manifest["cross-check"]
    assert said and not any(line.startswith("Not compared") for line in said)
    assert any(
        line.startswith("S21 differs by at most ")
        and "in what Run 3 (openEMS) solved and " in line
        and line.endswith(" in what Run 4 (Palace) solved")
        for line in said
    )


def test_no_translation_reads_inside_a_run(made):
    _, manifest = made
    assert manifest["ports translated"] == {"openEMS": 2, "Palace": 2}
    assert "EMSParameters" not in manifest["members"]
    assert manifest["members"].count("EMRun") == 4


def test_a_run_is_one_undo_step(made):
    _, manifest = made
    undo = manifest["undo"]
    assert undo["before"] == LABELS
    assert undo["undone"] == LABELS[:3]
    assert undo["redone"] == LABELS


def test_the_runs_come_back_whole_from_the_saved_document(made, reopened):
    _, manifest = made
    assert reopened["proxies"] == ["EMRun"] * 4
    assert reopened["runs"] == manifest["runs"]
    assert reopened["overlay"] == manifest["overlay"]
