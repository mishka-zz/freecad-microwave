# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A Touchstone file through FreeCAD's own File menu, under a real FreeCAD, and
the table it became after a save and a reopen.

What the stubs cannot say is FreeCAD's: whether ``Init.py`` registers the types
FreeCAD's menu offers, whether the importer call the menu makes files one table
in one undo step, whether a property edited by the user restates the table's
reference, and whether the file held in the document comes back from a saved
one as it went in. ``file_menu_probe.py`` does it under ``freecadcmd`` with the
main window up, and reopens the saved file in another.
"""

import os

import pytest

from .conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "file_menu_probe.py")


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    out = tmp_path_factory.mktemp("file_menu")
    return out, probe_manifest(PROBE, out / "made", "FILE_MENU_OUT", key=None)


@pytest.fixture(scope="module")
def reopened(made):
    out, manifest = made
    return probe_manifest(
        PROBE,
        out / "reopened",
        "FILE_MENU_OUT",
        key=None,
        environment={"FILE_MENU_REOPENED": manifest["document"]},
    )


def test_the_file_menu_offers_touchstone_for_one_to_four_ports(made):
    _, manifest = made
    module = ["Microwave.Gui.file_menu"]
    assert manifest["import types"] == {"s1p": module, "S2P": module, "s4p": module, "s5p": []}
    assert manifest["export types"] == {"s1p": module, "s2p": module, "s5p": []}


def test_an_imported_file_is_one_table_in_the_study_selected(made):
    _, manifest = made
    assert manifest["in the study"]
    assert manifest["undo names"][0] == "Import Touchstone"
    assert "Import" not in manifest["undo names"]


def test_a_file_the_workbench_exported_reads_back_as_the_matrix_it_was(made):
    _, manifest = made
    assert manifest["exported"]
    assert manifest["imported"]["s"] == manifest["the run's"]["s"]
    assert manifest["imported"]["frequency"] == manifest["the run's"]["frequency"]
    assert manifest["imported"]["held"] == manifest["imported"]["digest"] == manifest["source"]


def test_it_is_a_table_to_the_overlay_and_the_chart(made):
    _, manifest = made
    assert manifest["overlay"] == [
        "S21, Run 1 (openEMS)",
        "S21, run",
        "|S21 difference|, run against Run 1 (openEMS)",
    ]
    assert manifest["chart title"].endswith(", run")


def test_an_edited_declaration_and_port_map_restate_the_table(made):
    _, manifest = made
    assert manifest["declared own"]["reference"] == "each port's own impedance"
    assert manifest["moved"]["ports"] == [2, 1]


def test_the_table_is_written_back_whole_and_one_table_only(made):
    _, manifest = made
    assert manifest["written back whole"]
    assert manifest["two tables refused"]


def test_an_import_is_undone_whole(made):
    _, manifest = made
    undo = manifest["undo"]
    assert undo["after"] == undo["before"] + 1
    assert undo["undone"] == undo["before"]


def test_open_makes_a_document_holding_the_table_at_its_root(made):
    _, manifest = made
    assert manifest["opened"]["at the root"] == ["run"]


def test_the_table_comes_back_from_the_saved_document_as_it_was_left(made, reopened):
    _, manifest = made
    back = reopened["imported"]
    assert back["proxy"] == "EMSParameters"
    assert back["held"] == manifest["source"]
    assert back["declared"] == "Port impedance"
    assert back["reference"] == "each port's own impedance"
    assert back["s"] == manifest["imported"]["s"]
