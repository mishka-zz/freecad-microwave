# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What two runs' snapshots say changed, where the change is in the drawing:
a value set on a solid, a spreadsheet or a variable set, a value an expression
drives, a placement and a shape; and what a run's description and an overlay's
legend make of it.

The snapshots are built by hand in the form ``Gui/described.py::drawn`` writes.
``tests/test_run_changes_reopen.py`` holds the same against a real FreeCAD.
"""

import copy
import math

from Microwave.Results import snapshots


def solid(name, label=None, inputs=(), shape="a", placed=None, **values):
    said = {
        "name": name,
        "label": label or name,
        "values": {key: value for key, (value, _, _) in values.items()},
        "units": {key: unit for key, (_, unit, _) in values.items() if unit},
        "expressions": {key: text for key, (_, _, text) in values.items() if text},
        "inputs": list(inputs),
        "shape": shape,
    }
    if placed is not None:
        said["Placement"] = placed
    return said


def sheet(name="Spreadsheet", label="Params", **values):
    said = solid(name, label, **values)
    del said["shape"]
    return said


def study(*drawn, permittivity=4.3, shapes="s"):
    material = {"name": "EMMaterial", "label": "FR4", "kind": "EMMaterial"}
    material["Permittivity"] = permittivity
    return {
        "drawing": "d",
        snapshots.SHAPES: shapes,
        "study": {"name": "EMAnalysis", "label": "Study", "FrequencyStop": 1e10},
        "members": [
            {
                "name": "Binding",
                "label": "Substrate",
                "kind": "EMMaterialBinding",
                "Material": material,
            }
        ],
        "drawn": list(drawn),
    }


def driven(width=3.0, other=1.0, expression="Spreadsheet.StubW", shape="a"):
    """A spreadsheet's StubW, and a stub whose width an expression binds to it."""
    return study(
        sheet(StubW=(width, "mm", ""), Other=(other, "mm", "")),
        solid(
            "Stub",
            inputs=["Spreadsheet"],
            shape=shape,
            Width=(width if "* 2" not in expression else 2 * width, "mm", expression),
            Length=(10.0, "mm", ""),
        ),
        shapes=shape,
    )


class TestADrivenValueIsSaidWhereItWasSet:
    def test_the_cell_is_named_with_the_dimensions_it_drives(self):
        assert snapshots.changed(driven(), driven(3.2, shape="b")) == [
            "Params StubW 3 -> 3.2 mm (Stub Width)"
        ]

    def test_a_value_reached_by_label_is_followed_as_one_reached_by_name(self):
        before = driven(expression="<<Params>>.StubW")
        after = driven(3.2, expression="<<Params>>.StubW", shape="b")
        assert snapshots.changed(before, after) == ["Params StubW 3 -> 3.2 mm (Stub Width)"]

    def test_a_value_driven_through_another_follows_the_first(self):
        def chain(width):
            return study(
                sheet(StubW=(width, "mm", "")),
                solid(
                    "Stub",
                    Width=(width, "mm", "Spreadsheet.StubW"),
                    Height=(2 * width, "mm", "Width * 2"),
                ),
            )

        assert snapshots.changed(chain(3.0), chain(3.5)) == [
            "Params StubW 3 -> 3.5 mm (Stub Width, Stub Height)"
        ]

    def test_a_cell_nothing_reads_changes_no_answer_and_is_not_said(self):
        assert snapshots.changed(driven(), driven(other=2.0)) == []
        assert snapshots.legend([driven(), driven(other=2.0)]) == ["", ""]

    def test_a_name_that_only_begins_like_the_cell_is_not_a_reading_of_it(self):
        before = driven(expression="Spreadsheet.StubWide")
        after = driven(3.2, expression="Spreadsheet.StubWide", shape="b")
        after["drawn"][0]["values"]["StubWide"] = 9.0
        assert snapshots.changed(before, after) == ["Stub Width 3 -> 3.2 mm"]

    def test_an_edited_expression_names_the_value_it_drives_and_both_texts(self):
        after = driven(expression="Spreadsheet.StubW * 2", shape="b")
        assert snapshots.changed(driven(), after) == [
            "Stub Width 3 -> 6 mm (=Spreadsheet.StubW -> =Spreadsheet.StubW * 2)"
        ]

    def test_a_value_newly_bound_says_its_expression(self):
        before = driven(shape="a")
        before["drawn"][1]["expressions"] = {}
        before["drawn"][1]["values"]["Width"] = 2.0
        assert snapshots.changed(before, driven(shape="b")) == [
            "Stub Width 2 -> 3 mm (now =Spreadsheet.StubW)"
        ]

    def test_a_value_moved_by_what_no_snapshot_holds_is_said_without_its_text(self):
        """The expression reads a value the snapshot does not keep, and stayed
        as it was, so the value is said and the expression is not."""

        def bound(width):
            return study(solid("Stub", Width=(width, "mm", "Elsewhere.W")))

        assert snapshots.changed(bound(3.0), bound(3.2)) == ["Stub Width 3 -> 3.2 mm"]

    def test_an_expression_rewritten_to_the_same_value_says_nothing(self):
        after = driven()
        after["drawn"][1]["expressions"]["Width"] = "Spreadsheet.StubW + 0"
        assert snapshots.changed(driven(), after) == []


class TestAPlacementIsSaidAsTheMove:
    def placed(self, position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)):
        return study(solid("Stub", placed={"position": list(position), "rotation": list(rotation)}))

    def test_a_move_along_one_axis_names_the_axis(self):
        assert snapshots.changed(self.placed(), self.placed((0.0, -1.5, 0.0))) == [
            "Stub moved -1.5 mm along Y"
        ]

    def test_a_move_along_several_axes_names_the_step(self):
        assert snapshots.changed(self.placed(), self.placed((0.5, 0.0, 2.0))) == [
            "Stub moved by (+0.5, 0, +2) mm"
        ]

    def test_a_turn_about_an_axis_names_the_angle_and_the_axis(self):
        half = math.sqrt(0.5)
        quarter = self.placed(rotation=(0.0, 0.0, half, half))
        assert snapshots.changed(self.placed(), quarter) == ["Stub turned 90 deg about Z"]

    def test_a_turn_is_said_the_short_way_round(self):
        """From 170 degrees about Z to -170, which is twenty degrees on."""

        def about_z(degrees):
            half = math.radians(degrees) / 2
            return (0.0, 0.0, math.sin(half), math.cos(half))

        before, after = self.placed(rotation=about_z(170)), self.placed(rotation=about_z(-170))
        [line] = snapshots.changed(before, after)
        assert line.startswith("Stub turned 20") and line.endswith(" deg about Z")

    def test_a_turn_is_measured_from_where_it_stood(self):
        half = math.sqrt(0.5)
        quarter = self.placed(rotation=(0.0, 0.0, half, half))
        back = self.placed()
        assert snapshots.changed(quarter, back) == ["Stub turned -90 deg about Z"]


class TestAShapeChangedByNothingNamedIsSaidAsThatShape:
    def test_a_shape_drawn_anew_is_named(self):
        before = study(solid("Via", shape="a"), shapes="s")
        after = study(solid("Via", shape="b"), shapes="t")
        assert snapshots.changed(before, after) == ["the shape of Via changed"]

    def test_a_shape_made_from_one_drawn_anew_is_not(self):
        before = study(solid("Sketch", shape="a"), solid("Pad", inputs=["Sketch"], shape="p"))
        after = study(solid("Sketch", shape="b"), solid("Pad", inputs=["Sketch"], shape="q"))
        assert snapshots.changed(before, after) == ["the shape of Sketch changed"]

    def test_a_shape_its_own_value_moved_is_said_by_the_value(self):
        before = study(solid("Box", shape="a", Width=(3.0, "mm", "")))
        after = study(solid("Box", shape="b", Width=(4.0, "mm", "")))
        assert snapshots.changed(before, after) == ["Box Width 3 -> 4 mm"]

    def test_a_snapshot_holding_its_drawing_says_nothing_of_the_shapes_as_a_whole(self):
        before = study(solid("Box", shape="a", Width=(3.0, "mm", "")), shapes="s")
        after = study(solid("Box", shape="b", Width=(4.0, "mm", "")), shapes="t")
        assert "the shapes the study names changed" not in snapshots.changed(before, after)


class TestAFigureIsSaidInItsUnit:
    def test_a_frequency_is_said_in_hertz_with_one_prefix(self):
        before, after = study(), study()
        before["study"]["units"] = after["study"]["units"] = {"FrequencyStop": "1/s"}
        after["study"]["FrequencyStop"] = 1.2e10
        assert snapshots.changed(before, after) == ["Study FrequencyStop 10 -> 12 GHz"]

    def test_the_units_and_expressions_a_member_keeps_are_no_values_of_it(self):
        before, after = study(), copy.deepcopy(study())
        before["study"]["units"] = {"FrequencyStop": "Hz"}
        after["study"]["units"] = {"FrequencyStop": "1/s"}
        before["study"]["expressions"] = {"FrequencyStop": "Params.Top"}
        after["study"]["expressions"] = {"FrequencyStop": "Params.Top * 1"}
        assert snapshots.changed(before, after) == []


class TestARunsDescription:
    def test_it_names_the_run_it_is_against(self):
        said = snapshots.summary("Run 3 (openEMS)", ["Params StubW 3 -> 3.2 mm"])
        assert said == "against Run 3 (openEMS): Params StubW 3 -> 3.2 mm"

    def test_nothing_changed_is_said(self):
        assert snapshots.summary("Run 3 (Palace)", []).endswith(": nothing it was asked changed")

    def test_what_does_not_fit_is_counted(self):
        lines = [f"Params Cell{index} 1 -> 2 mm" for index in range(9)]
        said = snapshots.summary("Run 3 (openEMS)", lines)
        assert len(said) <= snapshots.SUMMARY
        shown = said.split(": ", 1)[1].split("; ")
        assert shown[-1] == f"{len(lines) - len(shown) + 1} more"
        assert shown[:-1] == lines[: len(shown) - 1]

    def test_the_first_change_is_said_whole_however_long(self):
        line = "Params " + "x" * snapshots.SUMMARY + " 1 -> 2 mm"
        assert snapshots.summary("Run 1 (openEMS)", [line, "Vars Gap 1 -> 2 mm"]).endswith(
            line + "; 1 more"
        )


class TestTheLegend:
    def test_each_entry_carries_its_own_value_of_what_differs(self):
        said = snapshots.legend([driven(3.0), driven(3.2), driven(3.4)])
        assert said == ["StubW 3 mm", "StubW 3.2 mm", "StubW 3.4 mm"]

    def test_figures_are_as_few_as_tell_the_values_apart(self):
        said = snapshots.legend([driven(3.0), driven(3.0001)])
        assert said[0] != said[1]
        assert said == ["StubW 3 mm", "StubW 3.0001 mm"]

    def test_a_driven_value_follows_its_source_and_is_not_named(self):
        said = snapshots.legend([driven(3.0), driven(3.2)])
        assert all("Width" not in entry for entry in said)

    def test_past_two_values_the_legend_names_none(self):
        def many(level):
            cells = sheet(**{f"Cell{index}": (level + index, "mm", "") for index in range(5)})
            reader = solid("Box", inputs=["Spreadsheet"])
            reader["expressions"] = {"Height": " + ".join(f"Spreadsheet.Cell{i}" for i in range(5))}
            return study(cells, reader)

        assert snapshots.legend([many(0.0), many(1.0)]) == ["", ""]
        assert len(snapshots.columns([many(0.0), many(1.0)])) > snapshots.LEGEND

    def test_a_key_two_objects_share_is_named_with_its_owner(self):
        def two(width):
            return study(
                solid("Stub", Width=(width, "mm", "")), solid("Trace", Width=(width, "mm", ""))
            )

        assert snapshots.legend([two(1.0), two(2.0)]) == [
            "Stub Width 1 mm, Trace Width 1 mm",
            "Stub Width 2 mm, Trace Width 2 mm",
        ]

    def test_a_table_with_no_snapshot_has_an_empty_entry(self):
        assert snapshots.legend([driven(3.0), {}, driven(3.2)]) == [
            "StubW 3 mm",
            "",
            "StubW 3.2 mm",
        ]

    def test_no_snapshot_at_all_names_nothing(self):
        assert snapshots.legend([{}, {}]) == ["", ""]

    def test_one_snapshot_alone_names_nothing(self):
        assert snapshots.legend([driven(3.0), {}]) == ["", ""]

    def test_a_placement_has_no_figure_and_is_left_out(self):
        def placed(x):
            return study(
                solid("Stub", placed={"position": [x, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]})
            )

        assert snapshots.legend([placed(0.0), placed(1.0)]) == ["", ""]


class TestWhatAFirstReadingFound:
    """Each a pair of snapshots a reader of the first build found said wrongly."""

    def test_a_shape_drawn_anew_is_said_beside_an_edit_to_a_cell_it_does_not_read(self):
        def placed(other, shape):
            return study(
                sheet(Z=(1.0, "mm", ""), Other=(other, "mm", "")),
                solid(
                    "Via", inputs=["Spreadsheet"], shape=shape, Height=(1.0, "mm", "Spreadsheet.Z")
                ),
            )

        assert snapshots.changed(placed(1.0, "a"), placed(2.0, "b")) == ["the shape of Via changed"]

    def test_a_shape_made_from_something_new_is_said_beside_another_edit(self):
        def padded(other, sketch, shape):
            return study(
                sheet(Other=(other, "mm", "")),
                solid(sketch, shape="s"),
                solid("Pad", inputs=[sketch], shape=shape),
            )

        lines = snapshots.changed(padded(1.0, "Sketch", "a"), padded(2.0, "Sketch001", "b"))
        assert lines == ["the shape of Pad changed"]

    def test_an_expression_edited_with_the_cell_it_reads_is_said_on_its_own_line(self):
        after = driven(3.2, expression="Spreadsheet.StubW * 2", shape="b")
        assert snapshots.changed(driven(), after) == [
            "Params StubW 3 -> 3.2 mm",
            "Stub Width 3 -> 6.4 mm (=Spreadsheet.StubW -> =Spreadsheet.StubW * 2)",
        ]

    def test_curves_that_differ_carry_entries_that_differ(self):
        doubled = "Spreadsheet.StubW * 2"
        said = snapshots.legend(
            [driven(3.0), driven(3.0, expression=doubled), driven(3.2, expression=doubled)]
        )
        assert len(set(said)) == len(said)

    def test_a_cell_computed_from_another_follows_it(self):
        def computed(width):
            return study(
                sheet(StubW=(width, "mm", ""), Double=(2 * width, "mm", "StubW * 2")),
                solid(
                    "Stub", inputs=["Spreadsheet"], Width=(2 * width, "mm", "Spreadsheet.Double")
                ),
            )

        assert snapshots.changed(computed(3.0), computed(3.2)) == [
            "Params StubW 3 -> 3.2 mm (Params Double, Stub Width)"
        ]
        assert snapshots.legend([computed(3.0), computed(3.2)]) == ["StubW 3 mm", "StubW 3.2 mm"]

    def test_a_bound_placement_whose_expression_changed_says_both_texts(self):
        def placed(x, text):
            said = solid(
                "Stub", placed={"position": [x, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]}
            )
            said["expressions"] = {"Placement": text}
            return study(said)

        lines = snapshots.changed(placed(1.0, "Elsewhere.X"), placed(2.0, "Elsewhere.X * 2"))
        assert lines == ["Stub moved +1 mm along X (=Elsewhere.X -> =Elsewhere.X * 2)"]

    def test_a_placement_held_in_another_property_is_said_as_a_move_of_it(self):
        def offset(z):
            return study(
                solid(
                    "Pad",
                    AttachmentOffset=(
                        {"position": [0.0, 0.0, z], "rotation": [0.0, 0.0, 0.0, 1.0]},
                        "",
                        "",
                    ),
                )
            )

        assert snapshots.changed(offset(0.0), offset(0.1)) == [
            "Pad AttachmentOffset moved +0.1 mm along Z"
        ]
        assert snapshots.legend([offset(0.0), offset(0.1)]) == ["", ""]


class TestWhatTheSecondMutationRunAsked:
    def test_a_cell_whose_formula_reads_nothing_kept_is_where_a_change_was_set(self):
        def computed(value):
            return study(
                sheet(A=(value, "mm", "Elsewhere.X")),
                solid("Stub", inputs=["Spreadsheet"], Width=(value, "mm", "Spreadsheet.A")),
            )

        assert snapshots.changed(computed(1.0), computed(2.0)) == [
            "Params A 1 -> 2 mm (Stub Width)"
        ]

    def test_a_shape_moved_by_a_value_its_expressions_read_is_not_said_as_drawn_anew(self):
        def bound(z, shape):
            via = solid("Via", inputs=["Spreadsheet"], shape=shape)
            via["expressions"] = {"Kept": "Spreadsheet.Z"}
            return study(sheet(Z=(z, "mm", "")), via)

        assert snapshots.changed(bound(1.0, "a"), bound(2.0, "b")) == ["Params Z 1 -> 2 mm"]

    def test_the_shapes_moved_where_no_shape_of_the_drawing_did_is_said(self):
        before = study(solid("Box", shape="a"), shapes="s")
        after = study(solid("Box", shape="a"), shapes="t")
        assert snapshots.changed(before, after) == ["the shapes the study names changed"]


class TestWhatAnAdversaryFound:
    """Each a pair of snapshots an adversarial tester found said wrongly."""

    def test_a_rename_is_no_change(self):
        def named(label):
            said = study()
            said["members"][0]["References"] = [{"name": "Trace", "label": label, "elements": [""]}]
            said["members"][0]["Material"]["label"] = label + " FR4"
            return said

        assert snapshots.changed(named("Trace"), named("Trace line")) == []

    def test_a_link_to_a_whole_object_is_said_without_an_empty_element(self):
        def linked(name):
            said = study()
            said["members"][0]["References"] = [{"name": name, "label": name, "elements": [""]}]
            return said

        assert snapshots.changed(linked("Trace"), linked("Stub")) == [
            "Substrate References [Trace] -> [Stub]"
        ]

    def test_a_value_read_through_its_objects_shape_follows_it(self):
        def stacked(height):
            return study(
                solid("Substrate", Height=(height, "mm", ""), shape=f"s{height}"),
                solid(
                    "Stub",
                    inputs=["Substrate"],
                    placed={"position": [0.0, 0.0, height], "rotation": [0.0, 0.0, 0.0, 1.0]},
                ),
            )

        stacked_before, stacked_after = stacked(1.6), stacked(1.8)
        stacked_before["drawn"][1]["expressions"] = {"Placement": "Substrate.Shape.BoundBox.ZMax"}
        stacked_after["drawn"][1]["expressions"] = {"Placement": "Substrate.Shape.BoundBox.ZMax"}
        assert snapshots.changed(stacked_before, stacked_after) == [
            "Substrate Height 1.6 -> 1.8 mm",
            "Stub moved +0.2 mm along Z",
        ]

    def test_a_cell_read_by_its_address_follows_its_alias(self):
        def computed(width, text):
            cells = sheet(Wid=(width, "mm", ""), Hgt=(2 * width, "mm", text))
            cells["addresses"] = {"Wid": "B1", "Hgt": "B2"}
            reader = solid(
                "Stub", inputs=["Spreadsheet"], Width=(2 * width, "mm", "Spreadsheet.Hgt")
            )
            return study(cells, reader)

        for text in ("B1 * 2", "$B$1 * 2"):
            assert snapshots.changed(computed(4.0, text), computed(4.5, text)) == [
                "Params Wid 4 -> 4.5 mm (Params Hgt, Stub Width)"
            ]

    def test_a_placement_bound_in_part_and_moved_besides_is_said_as_the_move(self):
        def placed(x, y):
            stub = solid("Stub", placed={"position": [x, y, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]})
            stub["expressions"] = {"Placement": "Spreadsheet.StubX"}
            return study(sheet(StubX=(x, "mm", "")), stub)

        assert snapshots.changed(placed(-1.5, 0.0), placed(-1.0, 0.7)) == [
            "Params StubX -1.5 -> -1 mm",
            "Stub moved by (+0.5, +0.7, 0) mm",
        ]

    def test_a_shape_drawn_anew_and_moved_is_said_both_ways(self):
        def via(x, shape):
            return study(
                solid(
                    "Via",
                    shape=shape,
                    placed={"position": [x, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
                )
            )

        assert snapshots.changed(via(0.0, "a"), via(1.0, "b")) == [
            "Via moved +1 mm along X",
            "the shape of Via changed",
        ]

    def test_a_shape_moved_by_what_it_links_to_is_not_drawn_anew(self):
        def attached(z, shape):
            frame = solid(
                "LCS", placed={"position": [0.0, 0.0, z], "rotation": [0.0, 0.0, 0.0, 1.0]}
            )
            del frame["shape"]
            return study(frame, solid("OnLCS", inputs=["LCS"], shape=shape))

        assert snapshots.changed(attached(0.0, "a"), attached(3.0, "b")) == [
            "LCS moved +3 mm along Z"
        ]

    def test_a_value_nothing_reads_in_a_variable_set_is_not_said(self):
        def held(unused):
            variables = sheet("VarSet", "Design vars", Unused=(unused, "mm", ""))
            return study(variables)

        assert snapshots.changed(held(1.0), held(2.0)) == []

    def test_a_half_turn_is_said_about_the_positive_axis(self):
        def turned(q):
            return study(solid("Stub", placed={"position": [0.0, 0.0, 0.0], "rotation": list(q)}))

        assert snapshots.changed(turned((0.0, 0.0, 0.0, 1.0)), turned((0.0, -1.0, 0.0, 0.0))) == [
            "Stub turned 180 deg about Y"
        ]

    def test_a_changed_link_is_said_once_and_not_again_as_the_shapes(self):
        def port(edge, shapes):
            said = study(shapes=shapes)
            said["members"].append(
                {
                    "name": "Port1",
                    "label": "Port1",
                    "kind": "EMPortMicrostrip",
                    "TraceEnd": {"name": "Trace", "label": "Trace", "elements": [edge]},
                }
            )
            return said

        assert snapshots.changed(port("Edge1", "s"), port("Edge3", "t")) == [
            "Port1 TraceEnd Trace Edge1 -> Trace Edge3"
        ]

    def test_a_dimension_read_through_another_objects_shape_follows_its_value(self):
        def sized(height):
            return study(
                solid("Substrate", Height=(height, "mm", ""), shape=f"s{height}"),
                solid(
                    "Stub",
                    inputs=["Substrate"],
                    Width=(height, "mm", "Substrate.Shape.BoundBox.ZMax"),
                ),
            )

        assert snapshots.changed(sized(1.6), sized(1.8)) == [
            "Substrate Height 1.6 -> 1.8 mm (Stub Width)"
        ]


class TestWhatNoFigureSays:
    def test_a_move_and_a_rewritten_expression_are_kept_and_a_figure_is_not(self):
        before = driven(3.0)
        after = driven(3.2, expression="Spreadsheet.StubW * 2", shape="b")
        after["drawn"][1]["Placement"] = {
            "position": [1.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        }
        before["drawn"][1]["Placement"] = {
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        }
        shown = snapshots.columns([before, after])
        assert snapshots.unfigured(before, after, shown) == [
            "Stub Width =Spreadsheet.StubW -> =Spreadsheet.StubW * 2",
            "Stub moved +1 mm along X",
        ]

    def test_a_value_a_column_holds_is_said_by_none(self):
        before, after = driven(3.0), driven(3.2, shape="b")
        assert snapshots.unfigured(before, after, snapshots.columns([before, after])) == []

    def test_a_value_two_runs_of_three_hold_is_a_column_and_the_member_gone_is_said(self):
        port = {"name": "Port2", "label": "Port 2", "kind": "EMPortLumped", "Impedance": 50.0}
        runs = [study() for _ in range(3)]
        for said, extra in zip(runs, ([port], [{**port, "Impedance": 75.0}], []), strict=True):
            said["members"] += extra
        shown = snapshots.columns(runs)
        assert [(column.name, column.figures) for column in shown] == [
            ("Port 2 Impedance", ("50", "75", ""))
        ]
        assert snapshots.unfigured(runs[0], runs[1], shown) == []
        assert snapshots.unfigured(runs[0], runs[2], shown) == ["Port 2 is gone"]

    def test_a_value_no_column_holds_is_said_in_full(self):
        port = {"name": "Port2", "label": "Port 2", "kind": "EMPortLumped", "Impedance": 50.0}
        runs = [study(), study()]
        runs[0]["members"].append(port)
        runs[1]["members"].append({**port, "Impedance": 75.0})
        assert snapshots.unfigured(runs[0], runs[1]) == ["Port 2 Impedance 50 -> 75"]


class TestWhatAColumnIsCalled:
    def test_a_cell_with_an_alias_is_called_by_it_and_one_without_by_its_sheet(self):
        def sheets(width, aliased):
            cells = sheet(**{("StubW" if aliased else "B2"): (width, "mm", "")})
            cells["addresses"] = {"StubW": "B2"} if aliased else {}
            reader = solid(
                "Stub",
                inputs=["Spreadsheet"],
                Width=(width, "mm", "Spreadsheet." + ("StubW" if aliased else "B2")),
            )
            return study(cells, reader)

        assert [c.name for c in snapshots.columns([sheets(3.0, True), sheets(3.5, True)])] == [
            "StubW"
        ]
        assert [c.name for c in snapshots.columns([sheets(3.0, False), sheets(3.5, False)])] == [
            "Params B2"
        ]

    def test_one_name_in_two_variable_sets_is_called_with_each_sets_label(self):
        def held(width):
            return study(
                sheet("VarSet", "Stub vars", W=(width, "mm", "")),
                sheet("VarSet001", "Trace vars", W=(width, "mm", "")),
                solid(
                    "Stub",
                    inputs=["VarSet", "VarSet001"],
                    Width=(width, "mm", "VarSet.W + VarSet001.W"),
                ),
            )

        assert [c.name for c in snapshots.columns([held(1.0), held(2.0)])] == [
            "Stub vars W",
            "Trace vars W",
        ]


def solved_on(backend, order=None, steps=None, permittivity=4.3):
    """A study as a run on ``backend`` is read: its own solver beside what
    both backends read, and the kinds its backend alone reads."""
    asked = study(permittivity=permittivity)
    own = {
        "openEMS": ("EMSolverOpenEMS", "EMYeeGrid"),
        "Palace": ("EMSolverPalace", "EMGmshMesh"),
    }[backend]
    solver = {"name": own[0], "label": f"Solver ({backend})", "kind": own[0]}
    if order is not None:
        solver["Order"] = order
    if steps is not None:
        solver["MaxTimesteps"] = steps
    asked["members"].append(solver)
    return {**asked, snapshots.BACKEND: backend, snapshots.OWN: list(own)}


class TestRunsOnTwoBackends:
    def test_a_run_on_another_backend_says_the_switch_and_no_solver_new_or_gone(self):
        assert snapshots.changed(solved_on("openEMS", steps=30000), solved_on("Palace", 3)) == [
            "solved on Palace"
        ]

    def test_a_value_both_backends_read_is_said_beside_the_switch(self):
        assert snapshots.changed(
            solved_on("openEMS", steps=30000), solved_on("Palace", 3, permittivity=4.6)
        ) == ["solved on Palace", "FR4 Permittivity 4.3 -> 4.6"]

    def test_the_drawing_changed_is_still_said_beside_the_switch(self):
        after = {**solved_on("Palace", 3), "drawing": "e"}
        assert snapshots.changed(solved_on("openEMS", steps=30000), after) == [
            "solved on Palace",
            "the drawing changed",
        ]

    def test_two_runs_on_one_backend_say_their_own_solvers_change(self):
        assert snapshots.changed(solved_on("Palace", 3), solved_on("Palace", 4)) == [
            "Solver (Palace) Order 3 -> 4"
        ]

    def test_a_record_that_names_no_backend_is_compared_whole(self):
        """A run read where its backend is not known, and the study a refinement
        reads, name none, and a solver gone from one is then said gone."""
        before = solved_on("openEMS", steps=30000)
        after = solved_on("Palace", 3)
        del after[snapshots.OWN], after[snapshots.BACKEND]
        assert snapshots.changed(before, after) == [
            "Solver (openEMS) is gone",
            "Solver (Palace) is new",
        ]

    def test_what_a_backend_alone_reads_is_its_own_solver(self):
        assert [m["name"] for m in snapshots.owned(solved_on("Palace", 3))["members"]] == [
            "EMSolverPalace"
        ]
        assert snapshots.owned(study()) == {}

    def test_a_value_only_some_runs_hold_is_figured_for_those_that_hold_it(self):
        asked = [solved_on("openEMS", steps=30000), solved_on("Palace", 2), solved_on("Palace", 3)]
        [column] = snapshots.columns(asked)
        assert (column.name, column.figures) == ("Solver (Palace) Order", ("", "2", "3"))
        assert snapshots.legend(asked) == ["", "Solver (Palace) Order 2", "Solver (Palace) Order 3"]

    def test_a_value_one_run_alone_holds_differs_from_nothing(self):
        asked = [solved_on("openEMS", steps=30000), solved_on("Palace", 3)]
        assert snapshots.columns(asked) == []

    def test_a_row_named_for_its_run_does_not_say_the_switch_again(self):
        before = solved_on("openEMS", steps=30000)
        after = solved_on("Palace", 3, permittivity=4.6)
        assert snapshots.unfigured(before, after) == ["FR4 Permittivity 4.3 -> 4.6"]

    def test_a_value_its_source_moved_follows_it_where_the_first_run_holds_neither(self):
        asked = [study(), driven(3.0), driven(3.2, shape="b")]
        assert [column.name for column in snapshots.columns(asked)] == ["StubW"]

    def test_a_column_says_which_runs_hold_it_whatever_the_figure(self):
        asked = [solved_on("openEMS", steps=30000), solved_on("Palace", 2), solved_on("Palace", 3)]
        asked[1]["members"][-1]["Order"] = ""
        [column] = snapshots.columns(asked)
        assert (column.figures, column.held) == (("", "", "3"), (False, True, True))

    def test_a_cell_read_only_by_what_its_unread_sheet_binds_is_not_said(self):
        def held(width):
            unread = sheet(StubW=(width, "mm", ""))
            unread["expressions"]["Note"] = "StubW"
            return study(unread)

        assert snapshots.changed(held(3.0), held(3.2)) == []
