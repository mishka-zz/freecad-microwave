# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a run says changed since the study's run before it, under a real
FreeCAD, for each kind of edit a user makes, and after a save and a reopen.

What the stubs cannot say is FreeCAD's: what an expression, a spreadsheet, a
variable set, a sketch's named dimension and a placement look like to the
snapshot, and whether a shape reads back from a saved file as it was written.
``run_changes_probe.py`` does it under ``freecadcmd`` with the main window up,
and reopens the saved file in another.
"""

import os

import pytest

from .conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "run_changes_probe.py")


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    out = tmp_path_factory.mktemp("run_changes")
    return out, probe_manifest(PROBE, out / "made", "RUN_CHANGES_OUT", key=None)


@pytest.fixture(scope="module")
def reopened(made):
    out, manifest = made
    return probe_manifest(
        PROBE,
        out / "reopened",
        "RUN_CHANGES_OUT",
        key=None,
        environment={"RUN_CHANGES_REOPENED": manifest["document"]},
    )


def changes(made, edit):
    _, manifest = made
    return manifest["edits"][edit]["changes"]


def test_a_spreadsheet_cell_is_named_with_what_it_drives(made):
    """The cell drives the stub's width, and the pad's length through a cell
    computed from it."""
    [line] = changes(made, "spreadsheet")
    assert line.startswith("Params StubLen ")
    assert line.endswith(" mm (Pad Length, Params Half, Stub Width)")


def test_a_variable_set_value_is_named_with_the_dimensions_it_drives(made):
    [line] = changes(made, "variable set")
    assert line.startswith("Vars TraceW ") and line.endswith(" mm (Trace Width)")


def test_an_edited_expression_names_the_value_it_drives_and_both_texts(made):
    [line] = changes(made, "expression")
    assert line.startswith("Stub Width ")
    assert line.endswith("(=Params.StubLen -> =Params.StubLen * 2)")


def test_a_materials_constant_and_a_ports_reference_are_named(made):
    assert [line.split()[:2] for line in changes(made, "material")] == [["FR4", "LossTangent"]]
    assert changes(made, "port") == ["solved on Palace", "Port1 ReferenceImpedance 50 -> 75"]


def test_a_materials_constant_driven_from_a_cell_is_named_where_it_was_set(made):
    [line] = changes(made, "material from a cell")
    assert line.startswith("Params Er ") and line.endswith(" (FR4 Permittivity)")


def test_what_changes_no_answer_says_nothing(made):
    assert changes(made, "hidden") == []


def test_the_band_read_from_a_cell_is_named_where_it_was_set_in_hertz(made):
    _, manifest = made
    [line] = changes(made, "band")
    assert line.startswith("Params Fmax ")
    assert line.endswith(f" GHz ({manifest['study']} FrequencyStop)")


def test_the_bands_start_set_directly_is_said_in_hertz(made):
    _, manifest = made
    [line] = changes(made, "band start")
    assert line.startswith(f"{manifest['study']} FrequencyStart ") and line.endswith(" GHz")


def test_an_attachment_offset_is_said_as_its_move(made):
    assert changes(made, "offset") == ["Pad AttachmentOffset moved +0.1 mm along Z"]


def test_a_placement_bound_to_a_cell_is_said_as_the_move_beside_the_cell(made):
    """A placement is said as the move it made, since a part of it bound to a
    cell does not say what else moved it."""
    cell, move = changes(made, "bound placement")
    assert cell.startswith("Params ViaX ") and cell.endswith(" mm")
    assert move == "Via moved +0.25 mm along X"


def test_a_quantity_typed_with_a_unit_of_symbols_is_a_value_and_no_formula(made):
    assert changes(made, "typed area") == ["Params Area 4 -> 5 mm^2 (Pad Width)"]


def test_a_rename_is_no_change(made):
    assert changes(made, "renamed") == []


def test_a_shape_drawn_anew_and_moved_at_once_is_said_both_ways(made):
    move, shape = changes(made, "drawn anew and moved")
    assert move == "Via moved +1 mm along Y" and shape == "the shape of Via changed"


def test_a_solids_own_dimension_is_named_in_its_unit_and_nothing_it_carries(made):
    """The pad attached to the substrate's top moves with it, which is the
    attachment's doing and not an edit. The run before was solved on Palace."""
    switched, line = changes(made, "dimension")
    assert switched == "solved on openEMS"
    assert line.startswith("Substrate Height ") and line.endswith(" mm")


def test_a_placement_is_said_as_the_move(made):
    assert changes(made, "placement") == ["Stub moved +0.5 mm along X"]


def test_a_shape_drawn_anew_is_said_as_that_shape(made):
    assert changes(made, "drawn anew") == ["the shape of Via changed"]


def test_each_run_describes_itself_against_the_run_before_it(made):
    _, manifest = made
    assert manifest["first"]["description"] == ""
    palace = manifest["edits"]["port"]["label"]
    assert manifest["edits"]["dimension"]["description"].startswith(f"against {palace}: ")
    assert manifest["unchanged"]["description"].endswith(": nothing it was asked changed")


def test_a_shape_freecad_caches_on_the_study_is_nothing_asked(made):
    """FreeCAD keeps the shape of a study it was asked for in a property of its
    own that it does not save. A Palace mesh made before is not out of date
    for it."""
    _, manifest = made
    assert manifest["study"] in manifest["cached"]
    assert not manifest["mesh inputs moved"]
    assert manifest["unchanged"]["changes"] == []
    assert manifest["departs"] == []


def test_each_curve_of_the_overlay_carries_its_own_values(made):
    """The first edit moved the stub's length and the second the trace's
    width, so each value differs from the curve before it where it was edited
    and nowhere else."""
    _, manifest = made
    said = [name.split(", ", 2)[2].split(", ") for name in manifest["legend"]]
    assert all(len(entry) == 2 for entry in said)
    stub = [entry[0] for entry in said]
    trace = [entry[1] for entry in said]
    assert all(value.startswith("StubLen ") and value.endswith(" mm") for value in stub)
    assert all(value.startswith("TraceW ") and value.endswith(" mm") for value in trace)
    assert stub[0] != stub[1] == stub[2]
    assert trace[0] == trace[1] != trace[2]


def test_a_sketchs_named_dimension_is_a_value_bound_by_its_expression(made):
    _, manifest = made
    sketch = manifest["sketch"]
    assert sketch["units"] == {"Constraints.Reach": "mm", "Constraints.Tilt": "deg"}
    assert sketch["expressions"] == {"Constraints.Reach": "Vars.TraceW * 2"}


def test_a_sketch_keeps_what_the_user_sets_and_not_what_it_computes(made):
    """Whether it is fully constrained is the solver's answer, a reference
    dimension measures the sketch, and the attacher engine reads otherwise
    once the document is opened again."""
    _, manifest = made
    kept = manifest["sketch"]["kept"]
    assert "Constraints.Reach" in kept
    for computed in ("FullyConstrained", "Constraints.Measured", "AttacherEngine"):
        assert computed not in kept


def test_a_sketchs_named_angle_is_kept_in_the_degrees_it_was_set_in(made):
    _, manifest = made
    assert manifest["sketch"]["values"]["Constraints.Tilt"] == pytest.approx(30.0, abs=1e-9)


def test_runs_read_back_as_they_were_filed(made, reopened):
    _, manifest = made
    back = {run["label"]: run for run in reopened["runs"]}
    for run in manifest["edits"].values():
        assert back[run["label"]] == run


def test_nothing_changed_across_a_save_and_a_reopen(reopened):
    assert reopened["since the last run"] == []


def test_the_overlay_puts_each_runs_figures_in_a_table_on_its_details_tab(made):
    """Drawn in FreeCAD's own plot tab: a row to a run, marked with the colour
    its curve was drawn in, the values that differ as columns with their unit
    in the heading, on the details tab under the notes."""
    _, manifest = made
    shown = manifest["shown"]
    assert shown["headings"] == ["", "Table", "StubLen (mm)", "TraceW (mm)"]
    assert [row[0] for row in shown["rows"]] == [
        "Run 1 (openEMS)",
        "Run 2 (openEMS)",
        "Run 3 (openEMS)",
    ]
    stub = [row[1] for row in shown["rows"]]
    trace = [row[2] for row in shown["rows"]]
    assert stub[0] != stub[1] == stub[2] and trace[0] == trace[1] != trace[2]
    assert shown["marks"] == shown["curves"][: len(shown["marks"])]
    assert shown["tabs"][:2] == ["Chart", "Details"]
    assert shown["plot on the first"] and shown["table and notes on the second"]
    assert shown["notes over the table"]
    assert shown["notes"].startswith("Each table referenced to ")
    assert shown["legend laid out"] is False
    assert shown["every row in view"]


def test_a_difference_chosen_through_its_selector_puts_its_warnings_on_a_tab(made):
    """Three runs of three drawings: differenced against the first, each other
    run is drawn against it and said to stand on another drawing, on a tab
    that is shown only while it says something and counts what it says."""
    from Microwave.Gui import charts

    _, manifest = made
    warned = manifest["warned"]
    assert warned["opened"]["shown"] == [True, True, False]
    differenced = warned["differenced"]
    lines = differenced["warnings"].split("\n\n")
    assert [line for line in lines if line.startswith("Different bases: ")] == [
        f"Different bases: Run 1 (openEMS) and Run {n} (openEMS) were solved from different "
        "drawings"
        for n in (2, 3)
    ]
    assert differenced["tabs"][2] == charts.WARNINGS_TAB.format(len(lines))
    assert differenced["shown"] == [True, True, True]
    assert sum("difference" in name for name in differenced["curves"]) == 2
    assert differenced["heading"] == warned["opened"]["heading"]
    assert warned["taken off"]["shown"] == [True, True, False]
    assert warned["taken off"]["warnings"] == ""
    assert warned["open once hidden"] != 2


def test_a_table_of_many_values_keeps_each_run_on_one_line_in_view(made):
    _, manifest = made
    far = manifest["shown far apart"]
    assert len(far["headings"]) > 4 and far["every row in view"] and far["every heading whole"]


def test_a_legend_name_of_any_length_leaves_the_axes_where_they_were(made):
    _, manifest = made
    assert manifest["axes"]["long"] == manifest["axes"]["short"]


def test_a_basis_every_run_shares_is_said_once_in_the_notes(made):
    _, manifest = made
    assert manifest["notes"][0].startswith("Each table referenced to ")
    assert sum("referenced to" in note or " step " in note for note in manifest["notes"]) == 1
