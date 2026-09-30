# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How a result says its backend modelled each lossy material.

What each adapter records is held against what it gave its solver in the
adapter's own tests. What is held here is everything above: the words, the
comparison of two backends' records, and each place a user reads them - the
stored result, the line under its chart, a Touchstone header, and the log of a
run filed beside another backend's answer.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from Microwave.Gui import results as glue
from Microwave.Gui.plot_s_params import chart_text
from Microwave.Objects.analysis import createEMAnalysis
from Microwave.Objects.results import createEMSParameters, store
from Microwave.Results import modelled
from Microwave.Results.sparameters import ResultError, SParameters
from tests.test_results_object import matrix
from tests.test_sparameters import series_resistor

#: A loss tangent, as one backend holds it and the other folds it.
HELD = {"material": "FR4", "held": "loss tangent", "loss_tangent": 0.02}
FOLDED = {
    "material": "FR4",
    "held": "conductivity",
    "conductivity": 0.0245,
    "folded": 0.0245,
    "at": 5e9,
}
#: A conductivity alone, which both backends hold across the band.
CONDUCTING = {"material": "FR4", "held": "conductivity", "conductivity": 0.5}
#: A sheet inside the region, as each backend takes one.
CURRENT = {"material": "Brass", "sheet": "net current", "conductivity": 1.57e7, "thickness": 0.5}
FACES = {
    "material": "Brass",
    "sheet": "surface impedance",
    "conductivity": 1.57e7,
    "thickness": 0.0,
    "faces": ["inside"],
}

#: The outside of a study open to free space on Palace, as its run records it.
OPEN = {
    "boundary": "absorbing",
    "order": 1,
    "clearance": 39.97,
    "faces": ["YMin", "YMax", "ZMax"],
    "magnetic": ["XMin"],
}
#: The outside of the same drawing on openEMS, as its run records it: an absorbing
#: layer of cells, with the structure running out through one pair of faces.
LAYER = {
    "boundary": "absorbing layer",
    "cells": [8, 8, 8, 8, 8, 8],
    "clearance": 39.97,
    "faces": ["YMin", "YMax", "ZMax"],
    "through": ["XMin", "XMax"],
}


def with_records(result, solver, *records):
    return replace(
        result, provenance={**result.provenance, "solver": solver, "modelled": list(records)}
    )


class TestTheWords:
    def test_a_loss_tangent_held_across_the_band(self):
        assert modelled.said([HELD]) == ["'FR4': a loss tangent of 0.02, held across the band"]

    def test_a_loss_tangent_folded_says_where_and_what_that_does(self):
        (line,) = modelled.said([FOLDED])
        assert line.startswith("'FR4': a loss tangent folded at 5 GHz into a conductivity")
        assert "falls as one over frequency" in line
        assert "added to its own" not in line

    def test_a_conductivity_a_loss_tangent_was_added_to_is_stated_apart(self):
        (line,) = modelled.said([{**FOLDED, "conductivity": 0.0245 + 0.5}])
        assert "into a conductivity of 0.0245 S/m, added to its own 0.5 S/m" in line

    def test_a_conductivity_alone(self):
        assert modelled.said([CONDUCTING]) == [
            "'FR4': a conductivity of 0.5 S/m, held across the band"
        ]

    def test_a_sheet_carrying_its_current(self):
        (line,) = modelled.said([CURRENT])
        assert "0.5 mm thick" in line and line.endswith("carrying the net current through it")

    @pytest.mark.parametrize(
        "faces, said",
        [
            (["inside"], "on each of its two faces, passing nothing between them"),
            (["boundary"], "on the one face where the model ends"),
            (["boundary", "inside"], "where it stands inside the model"),
        ],
    )
    def test_a_sheet_carrying_its_surface_impedance(self, faces, said):
        (line,) = modelled.said([{**FACES, "faces": faces}])
        assert said in line
        assert "thick" not in line

    def test_one_clause_for_a_sheet_standing_both_inside_and_where_the_model_ends(self):
        assert modelled.brief([{**FACES, "faces": ["boundary", "inside"]}]) == (
            "'Brass' sheet by its faces inside and where the model ends"
        )
        assert modelled.brief([{**FACES, "faces": ["boundary"]}]) == (
            "'Brass' sheet where the model ends"
        )

    def test_one_clause_for_a_loss_tangent_folded_into_a_conductivity_of_its_own(self):
        assert modelled.brief([{**FOLDED, "conductivity": 0.5245}]) == (
            "'FR4' loss tangent folded at 5 GHz, added to its conductivity"
        )

    def test_one_clause_for_a_chart(self):
        assert modelled.brief([FOLDED, FACES]) == (
            "'FR4' loss tangent folded at 5 GHz; 'Brass' sheet by its two faces"
        )
        assert modelled.brief([]) == ""


class TestTheOutside:
    def test_an_open_outside_is_stated_whole(self):
        (said,) = modelled.said([OPEN])
        assert said == (
            "The outside: an absorbing condition of order 1 on YMin, YMax and ZMax, "
            "39.97 mm from the structure; the air beside a port's face on XMin is a "
            "magnetic wall"
        )

    def test_it_is_no_part_of_what_the_chart_says_of_loss(self):
        assert modelled.brief([HELD, OPEN]) == modelled.brief([HELD])

    def test_an_absorbing_layer_is_stated_with_its_depth_and_what_runs_through_it(self):
        (said,) = modelled.said([LAYER])
        assert said == (
            "The outside: an absorbing layer 8/8/8/8/8/8 cells deep, 39.97 mm clear of the "
            "structure on YMin, YMax and ZMax; the structure runs out through it on XMin "
            "and XMax"
        )

    def test_a_backend_that_records_no_outside_is_not_called_another_model(self):
        """A run that recorded no outside says nothing about its own, so one side
        holding an outside and the other none is not a difference, and the
        materials are still compared."""
        assert modelled.apart([HELD, OPEN], "Palace", [HELD], "openEMS") == []
        assert modelled.apart([HELD, OPEN], "Palace", [FOLDED], "openEMS") == modelled.apart(
            [HELD], "Palace", [FOLDED], "openEMS"
        )


class TestTheTwoOutsidesCompared:
    """A condition on a surface and a layer of cells absorb differently, and one's
    order is not the other's depth. What both state is which faces they held open
    and how far the boundary stood, and those are what is compared."""

    def test_the_same_faces_at_the_same_distance_are_not_a_difference(self):
        assert modelled.apart([OPEN], "Palace", [LAYER], "openEMS") == []

    def test_different_faces_held_open_are_named_on_both_sides(self):
        (line,) = modelled.apart([OPEN], "Palace", [{**LAYER, "faces": ["ZMax"]}], "openEMS")
        assert line == (
            "the two runs held different faces open: Palace held YMax, YMin "
            "and ZMax, and openEMS held ZMax"
        )

    def test_a_run_that_held_none_open_is_said_to_have_held_none(self):
        (line,) = modelled.apart([OPEN], "Palace", [{**LAYER, "faces": []}], "openEMS")
        assert "openEMS held none" in line

    def test_boundaries_at_different_distances_are_named_with_both(self):
        (line,) = modelled.apart([OPEN], "Palace", [{**LAYER, "clearance": 12.0}], "openEMS")
        assert line == (
            "the two runs stood their boundaries different distances from the structure: "
            "39.97 mm in what Palace solved, and 12 mm in what openEMS solved"
        )

    def test_a_distance_rounded_on_its_way_through_a_file_is_the_same_distance(self):
        # The last digits a provenance file's rounding moves, stated here rather
        # than read off the tolerance, which would move with it.
        near = {**LAYER, "clearance": 39.97 * (1.0 + 1e-12)}
        assert modelled.apart([OPEN], "Palace", [near], "openEMS") == []

    def test_runs_standing_in_different_media_are_named_with_both(self):
        (line,) = modelled.apart([OPEN, {"medium": "PTFE"}], "Palace", [LAYER], "openEMS")
        assert line == (
            "the two runs stood in different media: 'PTFE' in what Palace solved, and vacuum "
            "in what openEMS solved"
        )

    def test_one_medium_on_both_is_no_difference(self):
        mine, theirs = [OPEN, {"medium": "PTFE"}], [LAYER, {"medium": "PTFE"}]
        assert modelled.apart(mine, "Palace", theirs, "openEMS") == []

    def test_closed_runs_in_different_media_are_told_apart(self):
        """A closed run records no outside, and its medium is compared all the
        same."""
        (line,) = modelled.apart([{"medium": "PTFE"}], "Palace", [], "openEMS")
        assert "'PTFE' in what Palace solved, and vacuum in what openEMS solved" in line

    def test_a_medium_is_no_loss_under_a_chart(self):
        assert modelled.brief([{"medium": "PTFE"}]) == ""

    def test_the_order_and_the_depth_are_not_compared(self):
        """Neither means what the other means, so neither is a difference."""
        assert modelled.apart([OPEN], "Palace", [{**LAYER, "cells": [4] * 6}], "openEMS") == []
        assert modelled.apart([{**OPEN, "order": 2}], "a", [OPEN], "b") == []

    def test_the_materials_are_compared_beside_the_outsides(self):
        lines = modelled.apart(
            [HELD, OPEN], "Palace", [FOLDED, {**LAYER, "clearance": 12.0}], "openEMS"
        )
        assert len(lines) == 2
        assert any("stood their boundaries different distances" in line for line in lines)
        assert any("'FR4' is not one model" in line for line in lines)

    def test_a_touchstone_header_states_it_under_its_own_caption(self, tmp_path):
        runs = series_resistor(50.0, 50.0, 10.0)
        for run in runs:
            run.provenance = dict(
                run.provenance, solver="Palace", modelled=[HELD, OPEN, {"medium": "PTFE"}]
            )
        text = SParameters.from_runs(runs).write_touchstone(tmp_path / "device").read_text()
        head = text.split("# Hz")[0].splitlines()
        captions = [
            line[2:16].strip()
            for line in head
            if line.startswith(("! Loss", "! Boundary", "! Medium"))
        ]
        assert captions == ["Loss", "Boundary", "Medium"]


class TestTwoBackendsCompared:
    def test_a_loss_tangent_held_and_one_folded_are_two_models(self):
        (line,) = modelled.apart([HELD], "Palace", [FOLDED], "openEMS")
        assert line.startswith("'FR4' is not one model in the two: Palace solved a loss tangent")
        assert "and openEMS a loss tangent folded at 5 GHz" in line

    def test_a_sheet_of_two_faces_and_one_of_its_current_are_two_models(self):
        (line,) = modelled.apart([FACES], "Palace", [CURRENT], "openEMS")
        assert "'Brass' is not one model" in line

    def test_a_sheet_where_the_model_ends_is_not_one_standing_inside_it(self):
        assert modelled.apart([FACES], "a", [{**FACES, "faces": ["boundary"]}], "b")

    def test_one_model_computed_two_ways_is_one_model(self):
        """A folded conductivity is a loss tangent times a frequency, so two
        runs of one model need not agree on a value."""
        assert modelled.apart([CONDUCTING], "a", [{**CONDUCTING, "conductivity": 0.51}], "b") == []
        assert modelled.apart([FOLDED], "a", [{**FOLDED, "at": 6e9}], "b") == []
        assert modelled.apart([FOLDED], "a", [CONDUCTING], "b") == []

    def test_a_material_one_run_did_not_record_is_named_with_what_the_other_solved(self):
        """Renamed, or made lossless, between the two runs: the two answer
        different drawings, and the line says which material and what solved it."""
        assert modelled.apart([HELD], "Palace", [CURRENT], "openEMS") == [
            "'FR4' is lossy in what Palace solved and not in what openEMS solved: "
            "Palace solved a loss tangent of 0.02, held across the band",
            "'Brass' is lossy in what openEMS solved and not in what Palace solved: "
            f"openEMS solved {modelled.said([CURRENT])[0].split(': ', 1)[1]}",
        ]

    def test_two_materials_sharing_a_label_are_compared_whole(self):
        """FreeCAD can be set to let two objects share a label, and a comparison
        keyed by the label alone would read one of them."""
        both = [HELD, CONDUCTING]
        assert modelled.apart(both, "a", both, "b") == []
        (line,) = modelled.apart(both, "a", [CONDUCTING, FOLDED], "b")
        assert modelled.apart(both, "a", [HELD, {**CURRENT, "material": "FR4"}], "b")
        assert "a loss tangent of 0.02, held across the band; a conductivity of 0.5" in line


class TestWhereAUserReadsIt:
    def test_the_stored_result_states_it(self, doc):
        holder = store(createEMSParameters(doc), with_records(matrix(), "Palace", HELD, FACES))
        assert list(holder.Modelled) == modelled.said([HELD, FACES])

    def test_the_stored_result_states_it_where_the_rest_of_the_record_cannot_be_written(self, doc):
        """The record is kept however the rest of the provenance fares, since it
        is read off the result in hand and not off the JSON written beside it."""
        result = with_records(matrix(), "Palace", HELD)
        # Keys of two types in one nested mapping defeat ``sort_keys``, which
        # is what sends the JSON down its fallback.
        result = replace(result, provenance={**result.provenance, "x": {1: "a", "b": 2}})
        holder = store(createEMSParameters(doc), result)
        assert "unserialisable" in json.loads(holder.Provenance)
        assert list(holder.Modelled) == modelled.said([HELD])

    def test_a_result_saved_before_the_property_existed_is_refused_and_kept(self, doc):
        """FreeCAD restores the properties an object was saved with, so a result
        from an earlier build has no ``Modelled``. Refused by name before
        anything is written, so the matrix it held is the one it still holds."""
        holder = store(createEMSParameters(doc), matrix())
        holder._props.pop("Modelled")
        kept = list(holder.ScatteringReal)
        with pytest.raises(ResultError, match="'S-Parameters' has no Modelled: it was saved"):
            store(holder, with_records(matrix(points=2), "Palace", HELD))
        assert list(holder.ScatteringReal) == kept

    def test_a_result_of_a_lossless_model_states_nothing(self, doc):
        assert list(store(createEMSParameters(doc), matrix()).Modelled) == []

    def test_the_line_under_the_chart_states_it_after_the_reference(self):
        footnote = chart_text(with_records(matrix(), "openEMS", FOLDED)).footnote
        assert footnote.startswith("Referenced to ")
        assert footnote.endswith(". Loss: 'FR4' loss tangent folded at 5 GHz")

    def test_a_line_too_long_for_the_chart_points_at_the_property_instead(self):
        from Microwave.Gui.plot_s_params import LONGEST_BASIS

        many = [FOLDED] * 8
        footnote = chart_text(with_records(matrix(), "openEMS", *many)).footnote
        assert len(footnote) <= LONGEST_BASIS
        assert footnote.endswith("Loss: as the result's Modelled property states for each material")

    def test_a_chart_of_a_lossless_model_says_nothing_of_loss(self):
        assert "Loss" not in chart_text(matrix()).footnote

    def test_a_touchstone_header_states_it_whole_on_comment_lines(self, tmp_path):
        runs = series_resistor(50.0, 50.0, 10.0)
        for run in runs:
            run.provenance = dict(run.provenance, solver="openEMS", modelled=[FOLDED, CURRENT])
        text = SParameters.from_runs(runs).write_touchstone(tmp_path / "device").read_text()
        head = text.split("# Hz")[0].splitlines()
        loss = [line for line in head if line.startswith("! Loss")]
        assert len(loss) == 2
        block = head[head.index(loss[0]) : -1]
        assert all(line.startswith("! ") and len(line) <= 78 for line in block)
        assert " ".join(line[16:] for line in block) == " ".join(modelled.said([FOLDED, CURRENT]))

    def test_a_touchstone_header_of_a_lossless_model_says_nothing_of_loss(self, tmp_path):
        text = SParameters.from_runs(series_resistor(50.0, 50.0, 10.0)).write_touchstone(
            tmp_path / "device"
        )
        assert "! Loss" not in text.read_text()

    def test_palace_s_record_reaches_the_result(self):
        frequency = np.linspace(20e9, 26e9, 3)
        answer = SimpleNamespace(
            frequency=frequency,
            out=(1, 2),
            driven=(1, 2),
            matrix=np.zeros((3, 2, 2), dtype=complex),
            flux=np.zeros((3, 2, 2)),
            radiated=None,
            dissipates=True,
            modelled=(HELD, FACES),
        )
        assert glue.from_palace(answer).provenance["modelled"] == [HELD, FACES]


class TestARunFiledBesideAnotherBackends:
    def test_it_names_each_material_the_two_modelled_otherwise(self, doc):
        analysis = createEMAnalysis(doc)
        glue.record(analysis, with_records(matrix(), "openEMS", FOLDED, CURRENT))
        said = glue.beside(analysis, with_records(matrix(), "Palace", HELD, FACES))
        lines = [line for line in said if " is not one model" in line]
        assert [line.split(" is not")[0] for line in lines] == ["'FR4'", "'Brass'"]
        assert all(": Palace solved " in line for line in lines)

    def test_it_says_nothing_where_the_two_modelled_alike(self, doc):
        analysis = createEMAnalysis(doc)
        glue.record(analysis, with_records(matrix(), "openEMS", CONDUCTING))
        said = glue.beside(analysis, with_records(matrix(), "Palace", CONDUCTING))
        assert not [line for line in said if "model" in line]

    def test_it_does_not_compare_a_backend_with_its_own_last_answer(self, doc):
        analysis = createEMAnalysis(doc)
        glue.record(analysis, with_records(matrix(), "Palace", FOLDED))
        assert glue.beside(analysis, with_records(matrix(), "Palace", HELD)) == []


class TestRunsOfOneMatrix:
    def test_runs_given_their_loss_otherwise_are_refused(self):
        """The provenance keeps the first run's record for the whole matrix."""
        runs = series_resistor(50.0, 50.0, 10.0)
        runs[0].provenance = dict(runs[0].provenance, modelled=[FOLDED])
        runs[1].provenance = dict(runs[1].provenance, modelled=[CONDUCTING])
        with pytest.raises(ResultError, match="describe different models"):
            SParameters.from_runs(runs)

    def test_runs_given_it_alike_keep_the_record(self):
        runs = series_resistor(50.0, 50.0, 10.0)
        for run in runs:
            run.provenance = dict(run.provenance, modelled=[FOLDED])
        assert SParameters.from_runs(runs).provenance["modelled"] == [FOLDED]
