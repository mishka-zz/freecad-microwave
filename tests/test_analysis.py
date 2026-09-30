# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""``EMAnalysis``: the study container, and how objects find the one they join.

The behaviour under test is *ownership*. A document scan - one simulation per
document, taking every port and binding it finds - fails the tree, which will
not nest a freshly created object, and fails two studies over one geometry,
which then have nothing to scope themselves to.
"""

import pytest

from Microwave.Objects.analysis import (
    NoAnalysis,
    NoSolver,
    analyses,
    analysis_of,
    createEMAnalysis,
    find_analysis,
    solver_of,
    solver_to_run,
    solvers_in,
)
from Microwave.Objects.mesh import createEMMeshRegion


def kind(obj):
    return type(getattr(obj, "Proxy", None)).__name__


class TestWhatCreatingOneGives:
    def test_it_comes_with_a_solver_its_grid_and_a_mesh_policy(self, doc):
        """None of the three is useful alone, and assembling them by hand on
        every new document is three clicks to reach the state everyone wants.

        Palace's recipe is not here: a study carries the settings of the
        backends it is run on, and ``Add Palace Solver`` adds that one.
        """
        analysis = createEMAnalysis(doc)
        assert sorted(kind(o) for o in analysis.Group) == [
            "EMMeshPolicy",
            "EMSolverOpenEMS",
            "EMYeeGrid",
        ]

    def test_it_carries_the_band(self, doc):
        analysis = createEMAnalysis(doc)
        assert float(analysis.FrequencyStart) == pytest.approx(1e9)
        assert float(analysis.FrequencyStop) == pytest.approx(10e9)
        assert analysis.NumFrequencyPoints == 501

    def test_a_frequency_behaves_as_a_number(self, doc):
        """A band's two ends go in a set, and order against each other.

        Ordinary things to do with two numbers, asserted because the stub
        standing in for a frequency property is a ``float`` subclass with its
        own ``__eq__`` - one line away from being unhashable, and then this
        fails for a reason that has nothing to do with the workbench.
        """
        analysis = createEMAnalysis(doc)
        band = {analysis.FrequencyStart, analysis.FrequencyStop}
        assert len(band) == 2
        assert min(band) < max(band)

    def test_it_is_a_real_group(self, doc):
        """Not an ``App::FeaturePython`` with a claimChildren.

        This is the whole fix: FreeCAD maintains ``Group`` itself, so an object
        added after the tree was drawn lands inside it. A presentation hook is
        asked when the tree feels like asking, which is what left a refinement
        region at document root.
        """
        assert "DocumentObjectGroup" in createEMAnalysis(doc).TypeId

    def test_two_of_them_can_coexist(self, doc):
        """The point of a container: without one, a second simulation has to be
        refused outright, nothing being able to say which objects are whose."""
        first, second = createEMAnalysis(doc), createEMAnalysis(doc)
        assert first is not second
        assert len(analyses(doc)) == 2
        assert not set(map(id, first.Group)) & set(map(id, second.Group))


class TestFindingTheOneAnObjectBelongsTo:
    def test_a_member_finds_its_analysis(self, doc):
        analysis = createEMAnalysis(doc)
        assert analysis_of(solver_of(analysis)) is analysis

    def test_an_analysis_is_its_own(self, doc):
        analysis = createEMAnalysis(doc)
        assert analysis_of(analysis) is analysis

    def test_a_nested_member_still_belongs(self, doc):
        """Sorting twenty ports into a subgroup is housekeeping, not a way to
        drop them out of the study."""
        analysis = createEMAnalysis(doc)
        folder = doc.addObject("App::DocumentObjectGroup", "Ports")
        region = createEMMeshRegion(doc)
        folder.addObject(region)
        analysis.addObject(folder)

        assert analysis_of(region) is analysis

    def test_a_loose_object_belongs_to_nothing(self, doc):
        createEMAnalysis(doc)
        assert analysis_of(createEMMeshRegion(doc)) is None

    def test_membership_does_not_leak_between_studies(self, doc):
        first, second = createEMAnalysis(doc), createEMAnalysis(doc)
        region = createEMMeshRegion(doc)
        second.addObject(region)

        assert analysis_of(region) is second
        assert analysis_of(region) is not first


class TestTheSymmetryDeclaration:
    """A statement about the device, so it sits on the study beside the band.

    Its whole value is that it lets one solve stand for two, which means a
    wrong default would silently halve the information in every result. The
    default has to be the honest one.
    """

    def test_it_defaults_to_claiming_nothing(self, doc):
        analysis = createEMAnalysis(doc)
        assert analysis.Symmetry == "None"

    def test_mirror_is_the_other_choice(self, doc):
        analysis = createEMAnalysis(doc)
        assert analysis.getEnumerationsOfProperty("Symmetry") == ["None", "Mirror"]

    def test_the_glue_reads_it_in_the_result_layer_s_vocabulary(self, doc):
        """The document says "Mirror"; the neutral layer says "mirror". Two
        vocabularies on purpose - one is a user-facing enumeration, the other
        is a physics flag - and this is the single point they meet."""
        from Microwave.Gui.results import declared_symmetry
        from Microwave.Results.sparameters import MIRROR

        analysis = createEMAnalysis(doc)
        assert declared_symmetry(analysis) is None

        analysis.Symmetry = "Mirror"
        assert declared_symmetry(analysis) == MIRROR


class TestTheSmallestResponseDeclaration:
    """How far down the response is read, which is a statement about the study.

    It sits beside the band for the same reason the symmetry does: nothing in a
    solved sweep distinguishes a term that is the point of the exercise from a
    matched line's own reflection, so it is declared and never inferred. What it
    reaches is the bar each backend's shortcut is judged against, and nothing
    else.
    """

    def test_it_defaults_to_full_scale(self, doc):
        """Full scale is what a study that declares nothing is held to, and a
        study nobody has thought about declares nothing."""
        analysis = createEMAnalysis(doc)
        assert analysis.SmallestResponse == 0.0

    def test_the_default_reaches_the_envelope_as_what_it_calls_full_scale(self, doc):
        """A study declaring nothing spells it zero dB on the property and a
        magnitude of one in the envelope, and only the translation between them
        holds the two spellings together."""
        from dataclasses import fields

        from Microwave.Solvers.openems import document
        from Microwave.Solvers.openems.model import Problem

        undeclared = {field.name: field.default for field in fields(Problem)}
        assert document.smallest_response(createEMAnalysis(doc)) == pytest.approx(
            undeclared["smallest_response"]
        )


class TestWalkingAGroup:
    """``members`` is the one walk, shared by everything that asks.

    It replaced three copies - ``analysis_of``'s, the mesh preview's and the
    result glue's - of which only the adapter's had a cycle guard.
    """

    def test_it_follows_nested_groups(self, doc):
        from Microwave.Objects.analysis import members

        analysis = createEMAnalysis(doc)
        folder = doc.addObject("App::DocumentObjectGroup", "Ports")
        region = createEMMeshRegion(doc)
        folder.addObject(region)
        analysis.addObject(folder)

        found = members(analysis)
        assert folder in found and region in found

    def test_a_cycle_terminates_instead_of_hanging_freecad(self, doc):
        """A document is a file a user can edit, and this runs inside GUI
        callbacks - an infinite loop here hangs FreeCAD with nothing in the
        log to say why."""
        from Microwave.Objects.analysis import members

        outer = doc.addObject("App::DocumentObjectGroup", "Outer")
        inner = doc.addObject("App::DocumentObjectGroup", "Inner")
        outer.addObject(inner)
        inner.addObject(outer)

        found = members(outer)

        assert inner in found
        assert len(found) == 2

    def test_an_object_appearing_twice_is_listed_once(self, doc):
        from Microwave.Objects.analysis import members

        outer = doc.addObject("App::DocumentObjectGroup", "Outer")
        folder = doc.addObject("App::DocumentObjectGroup", "Folder")
        region = createEMMeshRegion(doc)
        folder.addObject(region)
        outer.addObject(folder)
        outer.addObject(region)

        assert members(outer).count(region) == 1


class TestChoosingWhereANewObjectGoes:
    def test_the_only_analysis_is_the_obvious_answer(self, doc):
        analysis = createEMAnalysis(doc)
        assert find_analysis(doc) is analysis

    def test_no_analysis_says_to_make_one(self, doc):
        with pytest.raises(NoAnalysis, match="no EM analysis"):
            find_analysis(doc)

    def test_two_analyses_refuse_rather_than_guess(self, doc):
        """Picking one would put the user's new port in a study they were not
        looking at, and nothing on screen would say so."""
        first, second = createEMAnalysis(doc), createEMAnalysis(doc)
        first.Label, second.Label = "Passband", "Stopband"
        with pytest.raises(NoAnalysis, match="Passband, Stopband"):
            find_analysis(doc)

    def test_the_selection_decides_when_there_are_several(self, doc):
        first, second = createEMAnalysis(doc), createEMAnalysis(doc)
        assert find_analysis(doc, [second]) is second
        assert find_analysis(doc, [first]) is first

    def test_selecting_anything_inside_one_is_enough(self, doc):
        """A port of the study you were just editing is the study you meant."""
        _first, second = createEMAnalysis(doc), createEMAnalysis(doc)
        assert find_analysis(doc, [solver_of(second)]) is second

    def test_a_selection_outside_any_analysis_is_ignored(self, doc):
        """Selecting a solid is not a statement about which study to add to;
        the single-analysis rule should still answer."""
        analysis = createEMAnalysis(doc)
        loose = doc.addObject("Part::Box", "Substrate")
        assert find_analysis(doc, [loose]) is analysis


class TestFindingTheSolver:
    def test_it_is_the_one_in_the_group(self, doc):
        analysis = createEMAnalysis(doc)
        assert kind(solver_of(analysis)) == "EMSolverOpenEMS"

    def test_a_study_without_one_says_none_rather_than_raising(self, doc):
        """The panel has to open on a half-built study - reading *why* it
        cannot run is what it is for. The adapter is what refuses, by name."""
        analysis = createEMAnalysis(doc)
        analysis.Group = [o for o in analysis.Group if kind(o) != "EMSolverOpenEMS"]
        assert solver_of(analysis) is None

    def test_a_solver_sorted_into_a_subgroup_is_still_found(self, doc):
        """Sorting objects into a subgroup is ordinary housekeeping, and the
        adapter's own contents() follows one. The two have to find the same
        object: a panel that finds none opens with a blank directory, and the
        run it starts is written somewhere the user did not ask for."""
        analysis = createEMAnalysis(doc)
        found = solver_of(analysis)
        subgroup = doc.addObject("App::DocumentObjectGroup", "Solvers")
        analysis.Group = [o for o in analysis.Group if o is not found]
        analysis.addObject(subgroup)
        subgroup.addObject(found)
        assert solver_of(analysis) is found

    def test_another_backends_solver_standing_first_is_not_taken(self, doc):
        """A study holds one solver per backend, and this lookup wants one of
        them. What the panel reads off the object is the interpreter that owns
        the openEMS bindings and the directory a run is written into, neither of
        which the other backend has a word for - so taking whichever solver came
        first hands the panel properties that are not there.

        Standing first is the case, because that is the order a group is read
        in and the one a user makes by dragging.
        """
        from Microwave.Objects.solver import createEMSolverPalace

        analysis = createEMAnalysis(doc)
        analysis.Group = [createEMSolverPalace(doc), *analysis.Group]
        assert kind(solver_of(analysis)) == "EMSolverOpenEMS"


class TestEverySolverAStudyHolds:
    """What answers *which backend is this study for*.

    Whatever has to say which solver a run starts asks this. What counts as one
    is derived from the solver classes rather than typed out, so ``solvers_in``
    itself needs no edit when a third backend lands - and the test below, which
    holds it to the derived set, fails until this class puts one in the study.
    That failure is the point: it is where somebody is told that a new backend
    has to be answered for here.
    """

    def _palace(self, doc, analysis):
        from Microwave.Objects.solver import createEMSolverPalace

        solver = createEMSolverPalace(doc)
        analysis.addObject(solver)
        return solver

    def test_a_new_study_holds_the_one_it_was_created_with(self, doc):
        analysis = createEMAnalysis(doc)
        assert [kind(obj) for obj in solvers_in(analysis)] == ["EMSolverOpenEMS"]

    def test_a_second_backends_solver_is_one_of_them(self, doc):
        analysis = createEMAnalysis(doc)
        self._palace(doc, analysis)
        assert {kind(obj) for obj in solvers_in(analysis)} == {
            "EMSolverOpenEMS",
            "EMSolverPalace",
        }

    def test_it_is_every_kind_the_workbench_calls_a_solver(self, doc):
        """Derived rather than typed out, which is what stops a third backend
        being forgotten here the way a hand-written list forgets one."""
        from Microwave.Objects.kinds import solver_kinds

        analysis = createEMAnalysis(doc)
        self._palace(doc, analysis)
        assert {kind(obj) for obj in solvers_in(analysis)} == solver_kinds()

    def test_nothing_else_in_the_study_is_one(self, doc):
        analysis = createEMAnalysis(doc)
        analysis.addObject(createEMMeshRegion(doc))
        assert all(kind(obj) in ("EMSolverOpenEMS",) for obj in solvers_in(analysis))
        assert len(solvers_in(analysis)) == 1

    def test_a_solver_sorted_into_a_subgroup_is_still_one_of_them(self, doc):
        """The same housekeeping ``solver_of`` follows, and the two have to
        agree about what a study holds."""
        analysis = createEMAnalysis(doc)
        found = solver_of(analysis)
        subgroup = doc.addObject("App::DocumentObjectGroup", "Solvers")
        analysis.Group = [o for o in analysis.Group if o is not found]
        analysis.addObject(subgroup)
        subgroup.addObject(found)
        assert solvers_in(analysis) == [found]

    def test_a_study_holding_none_is_empty_rather_than_refused(self, doc):
        """A panel opens on a half-built study, and reading why it cannot run is
        what it is for."""
        analysis = createEMAnalysis(doc)
        analysis.Group = [o for o in analysis.Group if kind(o) != "EMSolverOpenEMS"]
        assert solvers_in(analysis) == []

    def test_they_come_back_in_the_order_the_group_holds_them(self, doc):
        analysis = createEMAnalysis(doc)
        palace = self._palace(doc, analysis)
        analysis.Group = [palace, *[o for o in analysis.Group if o is not palace]]
        assert solvers_in(analysis)[0] is palace


class TestChoosingTheSolverToRun:
    """A run is minutes, and a study answered by the backend nobody asked for
    says nothing about having been. So the choice is the selection, then the
    only solver, and otherwise a refusal naming what is there."""

    def _both(self, doc):
        from Microwave.Objects.solver import createEMSolverPalace

        analysis = createEMAnalysis(doc)
        fem = createEMSolverPalace(doc)
        analysis.addObject(fem)
        return analysis, solver_of(analysis), fem

    def test_the_only_one_needs_no_selection(self, doc):
        analysis = createEMAnalysis(doc)
        assert solver_to_run(analysis) is solver_of(analysis)

    def test_the_one_selected_is_the_one_run(self, doc):
        analysis, fdtd, fem = self._both(doc)
        assert solver_to_run(analysis, [fem]) is fem
        assert solver_to_run(analysis, [fdtd]) is fdtd

    def test_both_selected_is_refused_rather_than_the_first_taken(self, doc):
        analysis, fdtd, fem = self._both(doc)
        with pytest.raises(NoSolver, match="Select the one to run"):
            solver_to_run(analysis, [fem, fdtd])

    def test_both_and_neither_selected_is_refused_naming_both(self, doc):
        analysis, fdtd, fem = self._both(doc)
        with pytest.raises(NoSolver) as refused:
            solver_to_run(analysis, [analysis])
        assert repr(fdtd.Label) in str(refused.value)
        assert repr(fem.Label) in str(refused.value)

    def test_another_studys_solver_selected_names_nothing_here(self, doc):
        """It falls through to this study's own rule rather than running a
        solver of a study the user was not looking at."""
        analysis = createEMAnalysis(doc)
        _, _, fem = self._both(doc)
        assert solver_to_run(analysis, [fem]) is solver_of(analysis)

    def test_a_study_with_none_says_how_to_add_one(self, doc):
        analysis = createEMAnalysis(doc)
        analysis.Group = [obj for obj in analysis.Group if obj is not solver_of(analysis)]
        with pytest.raises(NoSolver, match="Add openEMS Solver or Add Palace Solver"):
            solver_to_run(analysis)


class TestWhatARefinementMayBeAimedAt:
    """The other half of the DAG fix, and the more insidious half.

    *Add Mesh Refinement* pre-fills ``References`` from the selection, which is
    almost always right - and was catastrophically wrong for one selection:
    the study itself. Group membership is a dependency edge, so a region inside
    an analysis that references that analysis is a two-node cycle. FreeCAD
    prints *"The graph must be a DAG"* once and then simply stops being able to
    order the recompute: touching the region marks the analysis, touching the
    analysis marks the region, forever.
    """

    def selection(self, *objects):
        """Stand-ins for ``Gui.Selection.getSelectionEx()`` entries."""
        return [type("Sel", (), {"Object": obj, "SubElementNames": ()})() for obj in objects]

    def test_a_solid_comes_through_whole(self, doc):
        from Microwave.Objects.mesh import references_from

        board = doc.addObject("Part::Box", "Substrate")
        assert references_from(self.selection(board)) == [(board, [""])]

    def test_a_face_selection_keeps_its_sub_elements(self, doc):
        from Microwave.Objects.mesh import references_from

        board = doc.addObject("Part::Box", "Substrate")
        chosen = type("Sel", (), {"Object": board, "SubElementNames": ("Face2",)})()
        assert references_from([chosen]) == [(board, ["Face2"])]

    def test_the_study_itself_is_not_a_reference(self, doc):
        """The selection that closed the cycle."""
        from Microwave.Objects.mesh import references_from

        assert references_from(self.selection(createEMAnalysis(doc))) == []

    def test_our_own_objects_are_not_references(self, doc):
        """A refinement resolves a feature. A mesh policy is not one - and the
        preview would be taken by a Shape test alone, being a Part::FeaturePython.
        """
        from Microwave.Objects.mesh import references_from
        from Microwave.Objects.preview import createEMMeshPreview

        analysis = createEMAnalysis(doc)
        chosen = [*analysis.Group, createEMMeshPreview(doc)]
        assert references_from(self.selection(*chosen)) == []

    def test_geometry_survives_a_mixed_selection(self, doc):
        """Dropping the whole selection because one entry was ours would be a
        silent no-op - the region would come back referencing nothing."""
        from Microwave.Objects.mesh import references_from

        analysis = createEMAnalysis(doc)
        board = doc.addObject("Part::Box", "Substrate")
        assert references_from(self.selection(analysis, board)) == [(board, [""])]


class TestWhatTheWorkbenchRecognisesAsItsOwn:
    """Every place that asks "is this object ours?" must give the same answer.

    Testing the proxy class name for a prefix couples recognition to naming:
    rename the classes and the workbench silently stops recognising its own
    objects, which costs them a view provider and lets a refinement region
    reference something nothing will draw. Deriving the set from the classes is
    the one form that cannot fall out of step with them.
    """

    def test_it_names_every_document_object(self):
        from Microwave.Objects.kinds import kinds

        expected = {
            "EMAnalysis",
            "EMSolverOpenEMS",
            "EMSolverPalace",
            "EMMeshPolicy",
            "EMMeshRegion",
            "EMMeshPreview",
            "EMYeeGrid",
            "EMGmshMesh",
            "EMMaterial",
            "EMMaterialBinding",
            "EMPortCoaxial",
            "EMPortLumped",
            "EMPortMicrostrip",
            "EMPortRectWaveguide",
            "EMSParameters",
        }
        assert kinds() == expected

    def test_it_does_not_name_the_mixin_every_object_inherits(self):
        """``ViewProviderRestored`` is imported into every object module, so a
        naive scan of module contents collects it once per module."""
        from Microwave.Objects.kinds import kinds

        assert "ViewProviderRestored" not in kinds()

    def test_it_does_not_name_the_base_the_solver_kinds_share(self):
        """``EMSolverBase`` is what makes a solver askable-for by its class
        rather than by a prefix on its name. Nothing carries it as a ``Proxy``,
        and a kind with no view provider passes the sweep's filter and then
        finds nothing to attach."""
        from Microwave.Objects.kinds import kinds

        assert "EMSolverBase" not in kinds()

    def test_it_does_not_name_the_base_the_port_kinds_share(self):
        """``EMPortBase`` is shared behaviour, not a kind: no document object
        ever carries it as a ``Proxy``.

        The set is what the view-provider sweep filters on, so listing a class
        with no provider lets an object of it pass the filter and then find
        nothing, doing nothing and saying nothing. Dormant, because no factory
        can produce one, which is exactly how such an entry survives a surface
        audit."""
        from Microwave.Objects.kinds import kinds

        assert "EMPortBase" not in kinds()

    def test_the_solver_kinds_are_the_kinds_whose_class_is_a_solver(self):
        """Derived, so a backend added tomorrow is a solver here as it stands.

        Compared against a typed-out set as well, because the derivation is
        what the preview's exclusions and the dispatch of a run both read: a
        class that stopped inheriting the base would answer this quietly and
        take both with it.
        """
        from Microwave.Objects.kinds import kinds, solver_kinds

        assert solver_kinds() == {"EMSolverOpenEMS", "EMSolverPalace"}
        assert solver_kinds() <= kinds()

    def test_every_solver_but_the_one_the_drawing_is_laid_from_is_left_out(self):
        """``NOT_MESHED_FROM`` mirrors ``Gui/openems_mesh_preview.py::_link``
        by hand.

        The half a second backend moves is derivable, so it is derived: a
        solver the preview is not laid from and nobody excluded leaves the
        study's membership comparison unequal for good.
        """
        from Microwave.Objects.analysis import NOT_MESHED_FROM
        from Microwave.Objects.kinds import solver_kinds
        from Microwave.Solvers.openems.document import SOLVER

        assert SOLVER not in NOT_MESHED_FROM, (
            "the study stops counting the solver its own drawing is laid from, "
            "so moving that solver out of the study says nothing"
        )
        unaccounted = sorted(solver_kinds() - {SOLVER} - set(NOT_MESHED_FROM))
        assert not unaccounted, unaccounted

    def test_every_recipe_but_the_one_the_drawing_is_laid_from_is_left_out(self):
        """The same pairing for a mesh recipe, derived the same way.

        A recipe added tomorrow for a third pipeline is caught here. Left out of
        the list it counts as membership the preview never links, which leaves
        the comparison unequal for good and the badge stale with no press of
        Update Mesh able to clear it.
        """
        from Microwave.Objects.analysis import NOT_MESHED_FROM
        from Microwave.Objects.kinds import recipe_kinds
        from Microwave.Solvers.openems.document import RECIPE

        assert recipe_kinds(), "the derivation found no recipe, so this checks nothing"
        assert RECIPE not in NOT_MESHED_FROM, (
            "the study stops counting the recipe its own drawing is laid from, "
            "so moving that recipe out of the study says nothing"
        )
        unaccounted = sorted(recipe_kinds() - {RECIPE} - set(NOT_MESHED_FROM))
        assert not unaccounted, unaccounted

    def test_a_mesh_freecad_draws_leaves_out_every_solver_and_every_other_recipe(self):
        """``NOT_FEM_MESHED_FROM`` is held to the same derivation. The Palace
        solver is left out too: the order of the elements the mesher lays is
        fixed, so nothing on it reaches the mesh."""
        from Microwave.Objects.analysis import NOT_FEM_MESHED_FROM
        from Microwave.Objects.kinds import recipe_kinds, solver_kinds
        from Microwave.Solvers.palace.document import RECIPE

        assert RECIPE not in NOT_FEM_MESHED_FROM
        unaccounted = sorted(
            (solver_kinds() | recipe_kinds()) - {RECIPE} - set(NOT_FEM_MESHED_FROM)
        )
        assert not unaccounted, unaccounted

    def test_a_foreign_object_is_not_ours(self, doc):
        from Microwave.Objects.kinds import is_ours

        assert not is_ours(doc.addObject("Part::Box", "Substrate"))

    def test_every_one_of_ours_is_ours(self, doc):
        from Microwave.Objects.kinds import is_ours
        from Microwave.Objects.preview import createEMMeshPreview

        analysis = createEMAnalysis(doc)
        for obj in (analysis, *analysis.Group, createEMMeshPreview(doc)):
            assert is_ours(obj), obj.Name
