# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Where FreeCAD's model tree shows what a study holds: a solver over its mesh
settings and the mesh made from them, the mesh settings over the regions
amending them, everything else under the study, and each object once - after a
mesh is made, a region added and turned, and all of it undone and redone.

The stubs cannot say it, since the tree is FreeCAD's and reads an object's
children again only when it is told to. ``tree_nesting_probe.py`` reads
the tree under a real FreeCAD with its main window up.
"""

import os

import pytest

from .conftest import probe_manifest

PROBE = os.path.join(os.path.dirname(__file__), "tree_nesting_probe.py")

STEPS = (
    "opened",
    "meshed on openEMS",
    "meshed for Palace",
    "region added",
    "region turned",
    "region unlinked",
    "unlink undone",
    "undone",
    "redone",
)

#: The steps after which a mesh of each backend stands in the study.
MESHED = {
    "openEMS": STEPS[1:],
    "Palace": (
        "meshed for Palace",
        "region added",
        "region turned",
        "region unlinked",
        "unlink undone",
        "redone",
    ),
}


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    return probe_manifest(PROBE, tmp_path_factory.mktemp("tree"), "TREE_OUT", key=None)


def under(rows, label):
    """The labels of the rows directly under the row ``label``."""
    [at] = [index for index, (_, text) in enumerate(rows) if text == label]
    depth = rows[at][0]
    found = []
    for row_depth, text in rows[at + 1 :]:
        if row_depth <= depth:
            break
        if row_depth == depth + 1:
            found.append(text)
    return found


def study(rows):
    """The labels of the rows directly under the study."""
    return under(rows, rows[0][1])


@pytest.mark.parametrize("step", STEPS)
def test_each_object_of_the_study_is_shown_once(made, step):
    texts = [text for _, text in made[step]]
    assert len(texts) == len(set(texts)), made[step]


@pytest.mark.parametrize("step", STEPS)
def test_a_solver_shows_its_mesh_settings_first_and_the_study_does_not(made, step):
    rows, roles = made[step], made["roles"]
    for backend in ("openEMS", "Palace"):
        assert under(rows, roles[f"{backend} solver"])[0] == roles[f"{backend} settings"]
        assert roles[f"{backend} settings"] not in study(rows)


@pytest.mark.parametrize(
    ("backend", "step"), [(b, s) for b, steps in MESHED.items() for s in steps]
)
def test_a_mesh_is_shown_under_its_solver_beside_its_settings(made, backend, step):
    rows, roles = made[step], made["roles"]
    assert under(rows, roles[f"{backend} solver"]) == [
        roles[f"{backend} settings"],
        roles[f"{backend} mesh"],
    ]
    assert roles[f"{backend} mesh"] not in under(rows, roles[f"{backend} settings"])
    assert roles[f"{backend} mesh"] not in study(rows)


def test_a_mesh_undone_is_gone_from_under_its_solver(made):
    roles = made["roles"]
    assert under(made["undone"], roles["Palace solver"]) == [roles["Palace settings"]]


def test_a_region_added_with_a_solver_selected_is_shown_under_its_settings(made):
    rows, roles = made["region added"], made["roles"]
    assert under(rows, roles["Palace settings"]) == [made["region"]]
    assert made["region"] not in study(rows)


def test_a_region_turned_to_other_settings_moves_under_them(made):
    rows, roles = made["region turned"], made["roles"]
    assert under(rows, roles["openEMS settings"]) == [made["region"]]
    assert under(rows, roles["Palace settings"]) == []


def test_a_region_amending_nothing_is_shown_under_the_study(made):
    rows, roles = made["region unlinked"], made["roles"]
    assert made["region"] in study(rows)
    assert made["region"] not in under(rows, roles["openEMS settings"])


def test_a_region_unlinked_and_undone_is_back_under_the_settings_it_amended(made):
    rows, roles = made["unlink undone"], made["roles"]
    assert under(rows, roles["openEMS settings"]) == [made["region"]]
    assert made["region"] not in study(rows)


def test_a_region_undone_is_gone(made):
    assert made["region"] not in [text for _, text in made["undone"]]
