# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A run solved again one step finer: what each backend turns, what the
translation is handed, what the study holds afterwards, what the new run says
changed and how far it moved, and what is refused.

The step arithmetic is checked here directly. What the stubs cannot say is
FreeCAD's: whether a value set for one translation and put back leaves the
study, its recompute marks and its undo stack as they were, and whether the
panel a run's menu opens holds the refinement. ``openems_palace_refine_probe.py`` does that
under ``freecadcmd`` on ``examples/stepped_line.FCStd``, filing made-up
matrices rather than solving.
"""

import math
import os

import pytest

from Microwave.Gui import refine

from .conftest import probe_manifest

PROBE = os.path.join(os.path.dirname(__file__), "openems_palace_refine_probe.py")


class TestTheStep:
    def test_two_finer_steps_double_the_cells_per_wavelength(self):
        assert refine.finer(refine.finer(20.0)) == 40.0
        assert refine.finer(refine.finer(28.3)) == pytest.approx(56.6, rel=1e-3)

    @pytest.mark.parametrize("asked", [20.0, 15.0, 7.5, 33.0])
    def test_a_finer_step_is_the_square_root_of_two_to_three_figures(self, asked):
        assert refine.finer(asked) == pytest.approx(asked * math.sqrt(2.0), rel=5e-3)

    @pytest.mark.parametrize("asked", [1, 20000, 28285, 30000])
    def test_a_longer_run_is_never_shorter_than_the_square_root_of_two(self, asked):
        longer = refine.longer(asked)
        assert isinstance(longer, int)
        assert asked * math.sqrt(2.0) <= longer < asked * math.sqrt(2.0) + 1

    def test_a_higher_order_is_one_more(self):
        assert refine.higher(3) == 4
        assert isinstance(refine.higher(3.0), int)


class TestTheMeshLabel:
    """The openEMS panel's mesh line is read from the study, which holds the
    refined values only while the run's envelopes are written."""

    @pytest.mark.parametrize("refined", [True, False])
    def test_it_is_read_again_once_a_refined_run_has_begun_and_only_then(self, refined):
        from types import SimpleNamespace

        from Microwave.Gui.openems_steps import OpenEMSSteps

        held = refine.Refinement(None, "", "", (), (), ()) if refined else None
        read = []
        steps = SimpleNamespace(
            resolve_directory=lambda: "/nowhere",
            take_refinement=lambda: held,
            _begun=lambda base: [],
            update_mesh_label=lambda: read.append(True),
        )

        OpenEMSSteps.begin_run(steps)

        assert read == ([True] if refined else [])


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    out = tmp_path_factory.mktemp("refine")
    manifest = probe_manifest(PROBE, out, "REFINE_OUT", key=None)
    assert "error" not in manifest, manifest.get("error")
    return manifest


@pytest.mark.slow
class TestUnderFreeCAD:
    def test_the_finer_grid_reaches_the_translation(self, made):
        plain, first, again = (
            made["openEMS plain"],
            made["openEMS"]["first"],
            made["openEMS"]["again"],
        )
        assert plain["cells"] < first["cells"] < again["cells"]
        assert first["timesteps"] == refine.longer(plain["timesteps"])
        assert again["timesteps"] == refine.longer(first["timesteps"])

    def test_the_higher_order_reaches_the_translation(self, made):
        assert made["Palace"]["job"]
        assert made["Palace"]["order"] == made["before"]["order"] + 1

    def test_the_study_holds_what_it_held_and_nothing_is_marked(self, made):
        assert made["after"] == made["before"]
        for case in (made["openEMS"]["first"], made["openEMS"]["again"], made["Palace"]):
            assert case["held"] == made["before"]

    def test_a_refinement_leaves_nothing_to_undo_but_the_runs_it_files(self, made):
        undo = made["undo"]
        filed = undo["after"] - undo["before"]
        assert filed == 3
        assert set(undo["names"][:filed]) == {"Store Results"}

    def test_the_new_run_says_what_changed_against_the_run_it_refines(self, made):
        first = made["openEMS"]["first"]["run"]
        assert sorted(first["changes"]) == [
            "Mesh Settings (openEMS) ElementsPerWavelength 20 -> 28.3",
            f"Solver (openEMS) MaxTimesteps 20000 -> {refine.longer(20000)}",
        ]
        assert first["description"].startswith("against Run 1 (openEMS): ")
        again = made["openEMS"]["again"]["run"]
        assert "Mesh Settings (openEMS) ElementsPerWavelength 28.3 -> 40" in again["changes"]
        assert again["description"].startswith(f"against {first['label']}: ")
        palace = made["Palace"]["run"]
        assert palace["changes"] == ["Solver (Palace) Order 3 -> 4"]
        assert palace["description"].startswith("against Run 2 (Palace): ")

    def test_the_move_is_the_largest_difference_and_never_a_convergence(self, made):
        """The made-up transmission from port 1 falls along the band to 0.95 of
        0.9, so it moves most at the top of the band, by 0.045."""
        [head, *terms] = made["openEMS"]["first"]["moved"]
        assert head.startswith("From Run 1 (openEMS) to Run 3 (openEMS), ")
        assert "at most 0.045 " in head and "in S21 at 10 GHz." in head
        assert head.endswith(refine.NOT_CONVERGED)
        assert [term.split()[0] for term in terms] == ["S11", "S12", "S21", "S22"]
        assert not any("converged" in term for term in terms)

    def test_the_move_is_said_by_the_steps_own_filing_and_kept_in_the_run_s_log(self, made):
        for case in (made["openEMS"]["first"], made["openEMS"]["again"], made["Palace"]):
            assert case["moved"][0].startswith("From Run ")
            assert case["moved"][0].endswith(refine.NOT_CONVERGED)

    def test_a_filed_refined_run_leaves_no_refinement_waiting(self, made):
        for case in (made["openEMS"]["first"], made["openEMS"]["again"], made["Palace"]):
            assert not case["waiting"]

    def test_the_study_s_values_are_put_back_when_the_block_raises(self, made):
        assert made["waits"]["raised"] == made["before"]

    def test_an_edit_made_while_a_refinement_waits_is_kept_and_not_refined_from(self, made):
        edited = made["waits"]["edited"]
        assert edited["timesteps asked"] == refine.longer(20000)
        assert edited["held"]["timesteps"] == 50000
        assert edited["held"]["touched"] == []

    def test_a_study_changed_while_a_refinement_waits_is_refused_at_run(self, made):
        changed = made["waits"]["changed"]
        assert not changed["began"]
        assert changed["status"] == "red"
        assert any("Substrate Height 1.6 -> 2.1 mm" in line for line in changed["log"])
        assert changed["waiting"]

    def test_palace_check_translates_what_run_would_solve(self, made):
        checked = made["waits"]["Palace checked"]
        order = made["before"]["order"]
        assert any(
            f"field of order {order + 1} within each element" in line for line in checked["log"]
        )
        assert checked["held"] == made["before"]

    def test_nothing_reads_the_raised_order_through_a_cell(self, made):
        """The Gmsh recipe reads the order through a spreadsheet cell, so a
        recompute while the order is raised would mesh another element."""
        cell = made["waits"]["Palace read through a cell"]
        assert cell["job"]
        assert cell["per wavelength while written"] == [cell["asked"]]
        assert cell["per wavelength after"] == cell["asked"]
        assert not cell["touched"]

    def test_an_edit_not_yet_recomputed_is_refused_on_palace(self, made):
        edited = made["waits"]["Palace edited, not recomputed"]
        assert not edited["job"]
        assert edited["status"] == "red"
        assert any("Substrate Height 1.6 -> 2 mm" in line for line in edited["log"])

    def test_a_run_whose_source_lost_its_matrix_is_said_as_not_compared(self, made):
        [line] = made["waits"]["gone"]
        assert line.startswith("Not compared with Run ")
        assert line.endswith(": its S-parameters are no longer in the study")

    def test_what_a_refinement_turns_is_said_before_it_runs(self, made):
        [turned] = [line for line in made["openEMS lines"] if line.startswith("Refines ")]
        assert turned == (
            "Refines Run 1 (openEMS): Mesh Settings (openEMS) ElementsPerWavelength 20 -> 28.3; "
            f"Solver (openEMS) MaxTimesteps 20000 -> {refine.longer(20000)}"
        )
        assert made["Palace"]["log"][1] == "Refines Run 2 (Palace): Solver (Palace) Order 3 -> 4"

    def test_a_length_the_problem_states_is_named_as_kept_where_the_mesh_is_turned(self, made):
        lines = made["region lines"]
        kept = "so the cells it sizes are not refined"
        assert f"Mesh Refinement ElementSize stays 0.2 mm, {kept}" in lines["openEMS"]
        assert any(
            line.startswith("Mesh Settings (openEMS) MinElementsAcross stays ")
            for line in lines["openEMS"]
        )
        assert not any(" stays " in line for line in lines["Palace"])
        assert not any("Unused Refinement" in line for line in lines["openEMS"])
        assert not any("Palace Refinement" in line for line in lines["openEMS"])
        assert not any(" stays 0," in line for line in lines["openEMS"])

    def test_the_other_backends_region_edited_leaves_a_run_refinable(self, made):
        assert made["past the other backend's region"] == ""

    def test_the_panel_a_run_opens_holds_the_refinement_and_says_so(self, made):
        panel = made["panel"]
        assert panel["open"]
        assert panel["refines"] == "Run 1 (openEMS)"
        assert "again one step finer" in panel["status"]
        assert panel["log"][0].startswith("Refines Run 1 (openEMS): ")

    def test_check_keeps_a_waiting_refinement_and_says_it_again(self, made):
        checked = made["panel"]["checked"]
        assert checked["waiting"]
        assert checked["cells"] == made["openEMS"]["first"]["cells"]
        assert checked["log"][0].startswith("Refines Run 1 (openEMS): ")

    def test_a_refined_run_is_drawn_against_the_run_it_refines_where_it_moved_most(self, made):
        panel = made["panel"]
        assert panel["drew"] and not panel["plain drew"]
        assert panel["drawn"] == [
            {"tables": ["Run 1 (openEMS)", "Run 3 (openEMS)"], "chosen": [2, 1], "against": 0}
        ]

    def test_a_study_asking_otherwise_is_refused_naming_what_changed(self, made):
        refused = made["refusals"]
        assert "Substrate Height 1.6 -> 2.1 mm" in refused["drawing"]
        assert "Mesh Refinement is new" in refused["a region added"]
        assert "bound to an expression" in refused["bound to an expression"]

    @pytest.mark.parametrize(
        "case",
        [
            "drawing put back",
            "the other backend's order",
            "the order it turns",
            "the value it turns",
        ],
    )
    def test_what_the_step_turns_and_another_backend_reads_are_not_refused(self, made, case):
        assert made["refusals"][case] is None
