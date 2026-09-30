# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The mesh a Mesh on Palace makes is in the document, as a user sees it.

What the suite's stubs cannot say is FreeCAD's: whether its reader takes the
copy the mesher wrote, whether the object is drawn and can be hidden, what undo
does to a mesh replaced, and whether all of it survives a save and a reopen. So
``palace_mesh_probe.py`` does it under a real FreeCAD with its main window up,
``palace_mesh_reopen_probe.py`` opens the saved file in another, and this reads
what the probes wrote.

Skips where the machine has no Palace or no Gmsh, as the gate does, or no
FreeCAD.
"""

import os

import pytest

from .conftest import probe_manifest

HERE = os.path.dirname(__file__)
PROBE = os.path.join(HERE, "palace_mesh_probe.py")
REOPEN = os.path.join(HERE, "palace_mesh_reopen_probe.py")


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    out = tmp_path_factory.mktemp("palace_mesh")
    manifest = probe_manifest(PROBE, out / "made", "PALACE_MESH_OUT", key=None)
    if "missing" in manifest:
        pytest.skip(f"this machine cannot mesh for Palace: {manifest['missing']}")
    return out, manifest


@pytest.fixture(scope="module")
def reopened(made):
    out, _ = made
    return probe_manifest(REOPEN, out / "reopened", "PALACE_MESH_REOPENED", key=None)


def test_every_label_comes_back_as_a_group_under_its_own_name(made):
    """Including one the copy's writer would have rewritten."""
    _, manifest = made
    assert "Port 1 (in)" in manifest["labels"]
    for which in ("first", "second"):
        groups = manifest[which]["groups"]
        assert sorted(groups) == manifest["labels"], which
        assert all(groups.values()), f"a group came back empty: {groups}"


def test_it_is_drawn_by_freecads_own_view_provider_and_can_be_hidden(made):
    _, manifest = made
    first = manifest["first"]
    assert first["type"] == "Fem::FemMeshObject"
    assert first["label"] == "Mesh (Palace)"
    assert first["provider"] == "ViewProviderFemMesh"
    assert first["visible"] is True
    assert manifest["hidden"] is False
    assert manifest["shown again"] is True


def test_a_second_mesh_replaces_the_first_and_keeps_it_hidden(made):
    """Under the first one's label: FreeCAD keeps labels unique, so one taken
    while the old object is still there comes back with a suffix."""
    _, manifest = made
    assert manifest["objects"] == 1
    assert manifest["second"]["label"] == "Mesh (Palace)"
    assert manifest["second"]["volumes"] != manifest["first"]["volumes"]
    assert manifest["second"]["visible"] is False


def test_undo_brings_the_first_mesh_back_and_redo_the_second(made):
    _, manifest = made
    assert manifest["undone"] == manifest["first"]["volumes"]
    assert manifest["redone"] == manifest["second"]["volumes"]


def test_the_study_translates_with_the_mesh_in_it(made):
    _, manifest = made
    assert manifest["translates"] is True


def test_a_reopened_document_holds_the_mesh_as_it_was_saved(made, reopened):
    """In a FreeCAD that had not imported FreeCAD's FEM module before the file
    was opened, so what brings it in is the document."""
    _, manifest = made
    assert reopened["imported before"] is False
    assert reopened["found"] is True
    assert reopened["made for"] == "Palace"
    assert reopened["seen"] == manifest["second"]


def test_what_it_cost_the_file_is_printed(made):
    """Not asserted: the figure is an observation, and is written to the
    record."""
    _, manifest = made
    print(
        f"MESH IN DOCUMENT: {manifest['elements']} elements added {manifest['bytes']} bytes "
        f"to the saved file, {manifest['bytes'] / manifest['elements']:.1f} an element"
    )
    assert manifest["bytes"] > 0


CURRENT = {"status": "Current", "label": "Mesh (Palace)", "says": None}
STALE = {
    "status": "Out of date",
    "label": "Mesh (Palace) - out of date",
    "says": "the model has changed since the mesh was made",
}


def test_a_mesh_just_made_matches_the_study(made):
    _, manifest = made
    assert manifest["marks"]["made"] == CURRENT
    assert manifest["marks"]["saved"] == CURRENT


@pytest.mark.parametrize("edit", ["excitation", "points"])
def test_an_edit_no_mesh_reads_leaves_it_matching(made, edit):
    _, manifest = made
    assert manifest["marks"][edit] == CURRENT


@pytest.mark.parametrize("edit", ["body", "band", "recipe", "region's body", "port out"])
def test_an_edit_to_what_it_was_built_from_marks_it_where_the_tree_shows_it(made, edit):
    """A body through the links and FreeCAD's recompute - one nothing binds
    through the link the mesh holds to it - the band through the study, a port
    through the study's membership."""
    _, manifest = made
    assert manifest["marks"][edit] == STALE


def test_an_edit_awaiting_a_recompute_is_reported_before_the_mark_is_written(made):
    _, manifest = made
    assert manifest["marks"]["body before a recompute"] == {**CURRENT, "says": STALE["says"]}


def test_a_second_study_numbered_by_freecad_is_marked_all_the_same(made):
    """The label the mark writes on the second study's mesh is the first
    study's already, and FreeCAD numbers it."""
    _, manifest = made
    first, then = manifest["marks"]["second study"], manifest["marks"]["second study's body"]
    assert first["status"] == "Current" and first["label"].startswith(CURRENT["label"])
    assert then["status"] == STALE["status"] and then["says"] == STALE["says"]
    assert then["label"].rstrip("0123456789") == STALE["label"]


def test_a_result_beside_the_mesh_it_was_solved_on_is_marked_once_that_is_deleted(made):
    """Deleting the mesh changes what the study holds, which is when the study
    asks again."""
    _, manifest = made
    assert manifest["marks"]["result beside its mesh"] == "S-Parameters (Palace)"
    assert manifest["marks"]["result, its mesh deleted"] == (
        "S-Parameters (Palace) - not solved on the mesh shown"
    )


def test_undoing_the_edit_undoes_the_mark_and_the_panel_still_asks_for_a_mesh(made):
    """The mark is written in the step that made the edit, so it goes with it.
    The undo leaves what it put back touched, and the recompute that clears
    that reaches the mesh: it reads out of date until the next Mesh, as the
    other backend's preview does, rather than current before it is known to
    be."""
    _, manifest = made
    assert manifest["marks"]["body undone"] == {**CURRENT, "says": STALE["says"]}
    assert manifest["marks"]["body undone, recomputed"] == STALE


def test_undo_and_redo_bring_each_mesh_back_with_its_own_mark(made):
    """The first mesh was made before the recipe moved for the second, so it
    comes back out of date, and the second comes back current. What the panel
    says is not asked: a redo leaves objects touched, and the panel then asks
    for a mesh until a recompute has run."""
    _, manifest = made
    marks = [(each["status"], each["label"]) for each in manifest["trace"]]
    current, stale = (CURRENT["status"], CURRENT["label"]), (STALE["status"], STALE["label"])
    assert marks == [current, stale, current, current]


def test_a_reopened_mesh_says_what_it_said_and_is_marked_by_the_next_edit(reopened):
    assert reopened["marks"]["reopened"] == CURRENT
    assert reopened["marks"]["body"] == STALE
