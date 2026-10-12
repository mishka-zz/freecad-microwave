# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The shipped WR-42 guide solved without the GUI on each backend, saved, and
opened again with it.

What the stubs cannot say is FreeCAD's and the engines': what ``freecadcmd``
does with a script's arguments and its exit status, what a save with no GUI
leaves out of the file, what a script that raised does a second time, and
whether the file opens in the GUI with the runs in the study and the view as it
was. The shipped script is run as a user runs it, and
``openems_palace_batch_probe.py`` does the rest.

Held back to the release run: the second run is read against the first, which
is a comparison of solves.
"""

import hashlib
import os
import subprocess
import zipfile

import pytest

from .conftest import _freecadcmd, probe_manifest

pytestmark = [pytest.mark.slow, pytest.mark.release]

HERE = os.path.dirname(__file__)
PROBE = os.path.join(HERE, "openems_palace_batch_probe.py")
SCRIPT = os.path.join(os.path.dirname(HERE), "microwave_solve.py")


def probe(out, case, **environment):
    return probe_manifest(
        PROBE, out / case, "BATCH_OUT", key=None, environment={"BATCH_CASE": case, **environment}
    )


def shipped(document, *arguments, environment=None):
    """Run the shipped script on ``document`` as a user does. Returns the exit
    status and what it printed."""
    finished = subprocess.run(
        [_freecadcmd(), SCRIPT, "--", document, *arguments],
        capture_output=True,
        text=True,
        env={**os.environ, **(environment or {})},
        cwd=os.path.dirname(document),
    )
    return finished.returncode, finished.stdout


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, interpreter):
    out = tmp_path_factory.mktemp("batch")
    made = probe(out, "prepare")
    if "missing" in made:
        pytest.skip(f"this machine cannot solve on both backends: {made['missing']}")
    return out, made, {"MICROWAVE_OPENEMS_PYTHON": str(interpreter)}


@pytest.fixture(scope="module")
def on_openems(prepared):
    _, made, environment = prepared
    return shipped(made["guide"], "--solver", "openEMS", environment=environment)


@pytest.fixture(scope="module")
def on_palace(prepared, on_openems):
    out, made, environment = prepared
    return probe(out, "raised", BATCH_GUIDE=made["guide"], **environment)


@pytest.fixture(scope="module")
def reopened(prepared, on_palace):
    out, made, _ = prepared
    return probe(out, "reopened", BATCH_REOPENED=made["guide"])


def test_the_shipped_script_solves_on_openems_and_exits_naming_the_file(prepared, on_openems):
    _, made, _ = prepared
    status, said = on_openems
    assert status == 0, said[-4000:]
    assert "Solving 'WR-42 waveguide' on 'Solver (openEMS)'" in said
    assert f"Saved in {made['guide']}." in said


def test_a_script_that_raises_after_the_solve_solves_once(on_palace):
    # freecadcmd runs it a second time, and the second asks again.
    [(first, said), (second, said_again)] = on_palace["statuses"]
    assert (first, second) == (0, 0)
    assert any(line.startswith("Solving 'WR-42 waveguide' on 'Solver (Palace)'") for line in said)
    assert "Nothing is solved again." in said_again
    assert not any(line.startswith("Solving") for line in said_again)


def test_the_runs_are_in_the_study_and_the_second_names_what_changed_since_the_first(reopened):
    runs = {
        found["label"]: found for found in reopened["objects"].values() if found["kind"] == "EMRun"
    }
    assert sorted(runs) == ["Run 1 (openEMS)", "Run 2 (Palace)"]
    assert {found["study"] for found in runs.values()} == {"WR-42 waveguide"}
    assert runs["Run 2 (Palace)"]["label2"].startswith("against Run 1 (openEMS)")
    assert {found["provider"] for found in runs.values()} == {"EMRunViewProvider"}


def test_the_file_reopens_with_every_object_shown_as_it_was_saved(prepared, reopened):
    _, made, _ = prepared
    assert "GuiDocument.xml" in reopened["entries"]
    shown = {name: found["visible"] for name, found in reopened["objects"].items()}
    assert made["visibility"]
    assert {name: shown[name] for name in made["visibility"]} == made["visibility"]


def test_a_document_that_does_not_recompute_is_refused_and_left_as_it_was(prepared):
    _, made, environment = prepared
    before = digest(made["broken"])
    status, said = shipped(made["broken"], environment=environment)
    assert status == 2, said[-4000:]
    assert "'GuideAir' does not recompute" in said
    assert digest(made["broken"]) == before


def test_a_directory_this_machine_cannot_write_is_refused_naming_rundir(prepared):
    _, made, environment = prepared
    status, said = shipped(made["elsewhere"], "--solver", "openEMS", environment=environment)
    assert status == 2, said[-4000:]
    assert "--rundir" in said


def test_a_file_changed_while_it_was_solved_is_kept_and_the_runs_saved_beside_it(prepared):
    out, made, environment = prepared
    answer = probe(out, "changed", BATCH_GUIDE=made["meddled"], **environment)
    assert answer["status"] == 4
    assert answer["tail"] == "written by another program"
    [beside] = answer["beside"]
    with zipfile.ZipFile(beside) as read:
        document = read.read("Document.xml").decode("utf-8")
    assert 'name="EMRun"' in document
