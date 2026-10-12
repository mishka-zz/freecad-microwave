# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A solver deleted through FreeCAD's own Delete takes what it brought to its
study: its backend's mesh settings, the refinement regions amending them and
the mesh made from them - in one undo step, asking nothing, and leaving the
mesh policy and the other backend's objects where they were.

The stubs cannot say it: Delete, its question and its undo step are
FreeCAD's. ``solver_deletion_probe.py`` drives them under a real FreeCAD with
its main window up.
"""

import os

import pytest

from .conftest import probe_manifest

PROBE = os.path.join(os.path.dirname(__file__), "solver_deletion_probe.py")

pytestmark = pytest.mark.slow

#: What each backend brings, by kind, and its mesh by the kind FreeCAD gives it.
BROUGHT = {
    "Palace": {"EMSolverPalace", "EMGmshMesh", "FemMeshProxy"},
    "openEMS": {"EMSolverOpenEMS", "EMYeeGrid", "EMMeshPreview"},
}


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    return probe_manifest(PROBE, tmp_path_factory.mktemp("deletion"), "DELETION_OUT", key=None)


def kinds(held):
    return {kind for kind, _ in held}


def regions(held):
    return sum(kind == "EMMeshRegion" for kind, _ in held)


@pytest.mark.parametrize(
    ("backend", "step"), [("Palace", "Palace deleted"), ("openEMS", "openEMS deleted")]
)
def test_a_solver_deleted_takes_what_it_brought_and_nothing_else(made, backend, step):
    before, after = made["before"], made[step]
    other = "openEMS" if backend == "Palace" else "Palace"
    assert not kinds(after) & BROUGHT[backend]
    assert BROUGHT[other] <= kinds(after)
    assert regions(before) - regions(after) == 1
    assert {"EMMeshPolicy", "EMPortRectWaveguide", "EMMaterialBinding"} <= kinds(after)
    assert len(before) - len(after) == len(BROUGHT[backend]) + 1


def test_the_delete_is_one_undo_step_that_puts_everything_back(made):
    assert made["undo names"] == ["Delete"]
    assert made["undone"] == made["before"]


def test_nothing_is_asked_on_the_way(made):
    assert made["asked"] == []
