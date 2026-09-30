# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What each class says moves no cell, and what it does about it.

``Microwave/Objects/staleness.py`` says why the declaration is a complement and
which way it fails. This pins the list on each class and the two things the
list has to reach: the status that keeps a linked object's quiet edit off the
dependency graph, and the study's own write, which is what the study has
instead of a link.

Whether the status actually suppresses FreeCAD's touch is FreeCAD's decision
and is counted under a real one by ``tests/test_preview_recompute.py``. The
stub records what it was told. What is here is the half a stub can answer:
which names were declared, and that each of them is a property the object
carries.
"""

import pytest

from Microwave.Objects import analysis, kinds, materials, mesh, ports, preview, results, solver
from Microwave.Objects.analysis import MEMBERSHIP
from Microwave.Objects.staleness import FREECADS_OWN

#: Every class here that declares anything, with the call that makes one.
DECLARING = {
    "EMAnalysis": analysis.createEMAnalysis,
    "EMMeshPreview": preview.createEMMeshPreview,
    "EMPortCoaxial": ports.createEMPortCoaxial,
    "EMPortLumped": ports.createEMPortLumped,
    "EMPortMicrostrip": ports.createEMPortMicrostrip,
    "EMPortRectWaveguide": ports.createEMPortRectWaveguide,
    "EMSolverOpenEMS": solver.createEMSolverOpenEMS,
    "EMSolverPalace": solver.createEMSolverPalace,
}

#: Every class that declares nothing, so that a class starting to declare has
#: to be named in one list or the other rather than slipping past both.
SILENT = (
    materials.EMMaterial,
    materials.EMMaterialBinding,
    mesh.EMMeshPolicy,
    mesh.EMMeshRegion,
    mesh.EMYeeGrid,
    mesh.EMGmshMesh,
    results.EMSParameters,
)


def _made(name, doc):
    return DECLARING[name](doc=doc)


class TestWhatEachClassDeclares:
    def test_a_class_declaring_nothing_says_so_by_carrying_the_empty_default(self):
        """A mesh policy leaves ``CurveTolerance`` undeclared although it moves
        no cell - measured over a hundredfold range on three curved specimens.
        It sets the triangulation the key is taken over, and a badge quieter
        than the key is the direction this must not fail in. Every other
        property a policy, a region, a material or a binding carries reaches
        the mesher.

        Each recipe declares nothing too, and for two different reasons. On the
        Yee grid every property sizes or places a cell. On the Gmsh mesh none
        of them does, and the way that is said is
        ``Objects/analysis.py::NOT_MESHED_FROM``: the openEMS preview neither
        links that object nor counts it as membership, so no edit to it reaches
        the badge to be quieted.
        """
        for klass in SILENT:
            assert klass.MOVES_NO_CELL == (), klass.__name__

    @pytest.mark.parametrize("name", sorted(DECLARING))
    def test_every_declared_name_is_a_property_the_object_carries(self, name, doc):
        """A misspelt name declares nothing and reads as a declaration.
        FreeCAD refuses the status on a property that is not there, and the
        loop skips it rather than throwing, which is what a file written before
        the class needs and what would hide this."""
        obj = _made(name, doc)
        missing = [
            declared for declared in type(obj.Proxy).MOVES_NO_CELL if not hasattr(obj, declared)
        ]
        assert not missing, missing

    @pytest.mark.parametrize("name", sorted(DECLARING))
    def test_a_freshly_made_object_carries_the_status(self, name, doc):
        """The mixin puts it back on a restored object. Nothing puts it on a
        new one but the class's own ``__init__``, and an object made without it
        would fire a recompute for every quiet edit until it had been saved and
        read back."""
        obj = _made(name, doc)
        marked = {
            declared for declared, status in obj._property_status.items() if "Output" in status
        }
        assert marked == set(type(obj.Proxy).MOVES_NO_CELL)

    @pytest.mark.parametrize("name", sorted(DECLARING))
    def test_nothing_of_freecads_own_is_declared(self, name, doc):
        """``Placement`` is the one that matters: FreeCAD accepts the call and
        goes on touching the object, so a class listing one would be stating
        something the platform does not honour."""
        obj = _made(name, doc)
        assert not set(type(obj.Proxy).MOVES_NO_CELL) & set(FREECADS_OWN)


class TestTheStudyMarksThePreviewItself:
    """The study holds the preview in its group, group membership is a
    dependency edge, and a link back would close a cycle. So the band reaches
    the badge by a write rather than by a recompute."""

    def _study_with_preview(self, doc):
        """A study whose drawing was made from what the study now holds.

        ``MeshedFrom`` is filled the way
        ``Gui/openems_mesh_preview.py::_link`` fills it, because membership
        is answered by comparing the two.
        """
        study = analysis.createEMAnalysis(doc=doc)
        drawing = preview.createEMMeshPreview()
        study.addObject(drawing)
        drawing.MeshedFrom = [
            member for member in study.Group if kinds.kind_of(member) != "EMMeshPreview"
        ]
        drawing.Status = preview.CURRENT
        return study, drawing

    def test_moving_the_band_ages_the_drawing(self, doc):
        study, drawing = self._study_with_preview(doc)
        study.Proxy.onChanged(study, "FrequencyStop")
        assert drawing.Status == preview.OUT_OF_DATE

    @pytest.mark.parametrize(
        "name",
        # Written out rather than read off the class under test. Taking the
        # list from MOVES_NO_CELL would make a name dropped from it stop being
        # tested for, which is the fault this asks about.
        ["NumFrequencyPoints", "SmallestResponse", "Symmetry"],
    )
    def test_what_the_mesher_never_reads_leaves_it_alone(self, name, doc):
        study, drawing = self._study_with_preview(doc)
        study.Proxy.onChanged(study, name)
        assert drawing.Status == preview.CURRENT

    @pytest.mark.parametrize("name", [own for own in FREECADS_OWN if own not in MEMBERSHIP])
    def test_freecads_own_properties_leave_it_alone(self, name, doc):
        """The filter is what is under test, so each name is offered whether or
        not a group carries that property: ``onChanged`` is handed a name."""
        study, drawing = self._study_with_preview(doc)
        study.Proxy.onChanged(study, name)
        assert drawing.Status == preview.CURRENT

    def test_a_property_nobody_declared_ages_the_drawing(self, doc):
        """The complement is the point. A property added to the study tomorrow
        marks the drawing stale until somebody says it moves nothing."""
        study, drawing = self._study_with_preview(doc)
        study.Proxy.onChanged(study, "SomethingAddedLater")
        assert drawing.Status == preview.OUT_OF_DATE

    def test_a_study_with_no_preview_is_not_an_error(self, doc):
        """This runs from inside ``onChanged``, where a traceback reaches the
        user over an ordinary property edit."""
        study = analysis.createEMAnalysis(doc=doc)
        study.Proxy.onChanged(study, "FrequencyStop")

    def test_a_restoring_document_is_left_alone(self, doc):
        """FreeCAD fires ``onChanged`` for every property of every object it
        reads back, and a file's own preview describes the document that file
        holds."""
        study, drawing = self._study_with_preview(doc)
        doc.Restoring = True
        try:
            study.Proxy.onChanged(study, "FrequencyStop")
        finally:
            doc.Restoring = False
        assert drawing.Status == preview.CURRENT

    def test_an_undo_or_a_redo_is_left_alone(self, doc):
        """Each puts back the badge with the edit it was written for."""
        study, drawing = self._study_with_preview(doc)
        doc.Transacting = True
        try:
            study.Proxy.onChanged(study, "FrequencyStop")
        finally:
            del doc.Transacting
        assert drawing.Status == preview.CURRENT

    def test_the_preview_of_another_study_is_left_alone(self, doc):
        """Two studies over one board have two grids, and the lookup is by
        membership."""
        study, drawing = self._study_with_preview(doc)
        _, others_drawing = self._study_with_preview(doc)
        study.Proxy.onChanged(study, "FrequencyStop")
        assert others_drawing.Status == preview.CURRENT
        assert drawing.Status == preview.OUT_OF_DATE


class TestMembership:
    """What the study holds is a grid input, and no property announces it.

    ``Gui/openems_mesh_preview.py::_link`` rebuilds the preview's link list at
    each Update Mesh and nowhere else, so an object added afterwards is in no
    list: it reaches the preview through nothing, and neither does any later
    edit to it. Comparing what the study holds against what the drawing was
    made from is what answers that, and it translates nothing.
    """

    def _drawn(self, doc):
        study = analysis.createEMAnalysis(doc=doc)
        drawing = preview.createEMMeshPreview()
        study.addObject(drawing)
        drawing.MeshedFrom = [
            member for member in study.Group if kinds.kind_of(member) != "EMMeshPreview"
        ]
        drawing.Status = preview.CURRENT
        return study, drawing

    def test_a_study_holding_what_it_was_drawn_from_is_left_alone(self, doc):
        study, drawing = self._drawn(doc)
        study.Proxy.onChanged(study, "Group")
        assert drawing.Status == preview.CURRENT

    def test_a_refinement_region_added_after_the_drawing_ages_it(self, doc):
        """The case the drawing cannot see any other way. A region created
        after Update Mesh is in no link list, so its ``ElementSize`` is
        invisible for as long as the region is."""
        study, drawing = self._drawn(doc)
        study.addObject(mesh.createEMMeshRegion(doc=doc))
        study.Proxy.onChanged(study, "Group")
        assert drawing.Status == preview.OUT_OF_DATE

    def test_a_port_taken_out_of_the_study_ages_the_drawing(self, doc):
        study, drawing = self._drawn(doc)
        port = ports.createEMPortLumped(doc=doc)
        study.addObject(port)
        drawing.MeshedFrom = [*drawing.MeshedFrom, port]
        drawing.Status = preview.CURRENT

        study.Group.remove(port)
        study.Proxy.onChanged(study, "Group")
        assert drawing.Status == preview.OUT_OF_DATE

    def test_a_region_added_to_a_subgroup_of_the_study_ages_the_drawing(self, doc):
        """Sorting objects into a subgroup is ordinary housekeeping, and both
        ``members`` and ``contents`` follow one. FreeCAD fires ``Group`` on the
        subgroup and ``_GroupTouched`` on the study, so the study has to listen
        for both or a whole tidy tree goes unwatched."""
        study, drawing = self._drawn(doc)
        subgroup = doc.addObject("App::DocumentObjectGroup", "Refinements")
        study.addObject(subgroup)
        drawing.MeshedFrom = [*drawing.MeshedFrom]
        drawing.Status = preview.CURRENT

        subgroup.addObject(mesh.createEMMeshRegion(doc=doc))
        study.Proxy.onChanged(study, "_GroupTouched")
        assert drawing.Status == preview.OUT_OF_DATE

    def test_a_material_kept_in_the_study_leaves_the_drawing_alone(self, doc):
        """A grid is not laid from where a material sits: it is reached through
        the binding that names it. A material dragged into the study is in the
        group and never in the link list, so counting it would leave the two
        sets unequal for good and the badge stale for every later change."""
        study, drawing = self._drawn(doc)
        study.addObject(materials.createEMMaterial(doc=doc))
        study.Proxy.onChanged(study, "Group")
        assert drawing.Status == preview.CURRENT

    def test_a_result_arriving_leaves_the_drawing_alone(self, doc):
        """A solve puts its result in the study. A mesh reported stale because
        results arrived is a worse fire than the one this puts out."""
        study, drawing = self._drawn(doc)
        study.addObject(results.createEMSParameters(doc=doc))
        study.Proxy.onChanged(study, "Group")
        assert drawing.Status == preview.CURRENT

    def test_a_second_backends_solver_leaves_the_drawing_alone(self, doc):
        """The drawing is one backend's grid, laid from what that backend's
        translation reads - and it reads its own solver and no other. Counting
        the second one would leave the two sets unequal for good: the badge
        would come up amber and stay there through every remesh, with nothing
        the user could do about it."""
        study, drawing = self._drawn(doc)
        study.addObject(solver.createEMSolverPalace(doc=doc))
        study.Proxy.onChanged(study, "Group")
        assert drawing.Status == preview.CURRENT

    def test_a_study_with_no_drawing_is_not_an_error(self, doc):
        study = analysis.createEMAnalysis(doc=doc)
        study.Proxy.onChanged(study, "Group")


class TestTheStudyMarksAMeshFreeCADDraws:
    """The same writes reach a mesh the study shows as FreeCAD's own object,
    which cannot link the study either - see ``Objects/fem_mesh.py``."""

    def _meshed(self, doc):
        from Microwave.Gui import fem_mesh as shown

        study = analysis.createEMAnalysis(doc=doc)
        study.addObject(solver.createEMSolverPalace(doc=doc))
        study.addObject(mesh.createEMGmshMesh(doc=doc))
        made = shown.put(
            study,
            "Palace",
            "a mesh",
            key=shown.inputs(study),
            linked=shown.built_from(study),
        )
        assert made.Status == preview.CURRENT
        return study, made

    @pytest.mark.parametrize("name", ["FrequencyStart", "FrequencyStop", "SomethingAddedLater"])
    def test_what_the_study_does_not_declare_ages_it(self, name, doc):
        study, made = self._meshed(doc)
        study.Proxy.onChanged(study, name)
        assert made.Status == preview.OUT_OF_DATE
        assert made.Label == "Mesh (Palace) - out of date"

    @pytest.mark.parametrize(
        "name",
        ["NumFrequencyPoints", "SmallestResponse", "Symmetry", "Label"],
    )
    def test_what_it_declares_leaves_it_alone(self, name, doc):
        study, made = self._meshed(doc)
        study.Proxy.onChanged(study, name)
        assert (made.Status, made.Label) == (preview.CURRENT, "Mesh (Palace)")

    def test_a_study_holding_what_it_was_meshed_from_is_left_alone(self, doc):
        study, made = self._meshed(doc)
        study.Proxy.onChanged(study, "Group")
        assert made.Status == preview.CURRENT

    def test_a_region_added_after_the_mesh_ages_it(self, doc):
        study, made = self._meshed(doc)
        study.addObject(mesh.createEMMeshRegion(doc=doc))
        study.Proxy.onChanged(study, "Group")
        assert made.Status == preview.OUT_OF_DATE

    @pytest.mark.parametrize(
        "added",
        [
            lambda doc: materials.createEMMaterial(doc=doc),
            lambda doc: results.createEMSParameters(doc=doc),
            lambda doc: solver.createEMSolverOpenEMS(doc=doc),
            lambda doc: preview.createEMMeshPreview(doc),
        ],
    )
    def test_what_no_mesh_of_it_is_built_from_leaves_it_alone(self, added, doc):
        study, made = self._meshed(doc)
        study.addObject(added(doc))
        study.Proxy.onChanged(study, "Group")
        assert made.Status == preview.CURRENT

    def test_an_undo_or_a_redo_is_left_alone(self, doc):
        """Each puts back the marks with the edits they were written for, and
        replays the group before the links it compares against."""
        study, made = self._meshed(doc)
        study.addObject(mesh.createEMMeshRegion(doc=doc))
        doc.Transacting = True
        try:
            study.Proxy.onChanged(study, "Group")
            study.Proxy.onChanged(study, "FrequencyStop")
        finally:
            del doc.Transacting
        assert made.Status == preview.CURRENT

    def test_a_label_freecad_numbered_to_keep_it_unique_is_marked(self, doc):
        study, made = self._meshed(doc)
        made.Label = "Mesh (Palace)001"
        study.Proxy.onChanged(study, "FrequencyStop")
        assert made.Label == "Mesh (Palace) - out of date"

    def test_a_label_somebody_wrote_is_left_alone_and_the_mark_still_written(self, doc):
        study, made = self._meshed(doc)
        made.Label = "Iris mesh"
        study.Proxy.onChanged(study, "FrequencyStop")
        assert (made.Status, made.Label) == (preview.OUT_OF_DATE, "Iris mesh")

    def test_a_mesh_that_records_no_status_is_left_alone(self, doc):
        """A mesh object carrying the backend it was made for and nothing else."""
        from Microwave.Objects.fem_mesh import MADE_FOR, TYPE

        study = analysis.createEMAnalysis(doc=doc)
        old = doc.addObject(TYPE, "Mesh")
        old.addProperty("App::PropertyString", MADE_FOR)
        old.MeshedFor = "Palace"
        old.Label = "Mesh (Palace)"
        study.addObject(old)
        study.Proxy.onChanged(study, "FrequencyStop")
        assert old.Label == "Mesh (Palace)"
        # Asked directly as well, since the study swallows a failure here.
        from Microwave.Objects.fem_mesh import age

        age(old)
        assert old.Label == "Mesh (Palace)"


class TestARecomputeReachingAMeshFreeCADDraws:
    """FreeCAD recomputes such a mesh only through what it links, and runs no
    ``execute`` of ours for it, so the document observer writes the mark."""

    def test_it_is_marked_stale(self, doc):
        from Microwave.Gui import fem_mesh as shown
        from Microwave.ViewProviders import _DocumentWatcher

        study = analysis.createEMAnalysis(doc=doc)
        made = shown.put(study, "Palace", "a mesh", key=shown.inputs(study))
        _DocumentWatcher().slotRecomputedObject(made)
        assert (made.Status, made.Label) == (preview.OUT_OF_DATE, "Mesh (Palace) - out of date")

    def test_a_mesh_nobody_made_for_a_backend_is_left_alone(self, doc):
        """It may carry a property of the same name for a reason of its own."""
        from Microwave.Objects.fem_mesh import TYPE
        from Microwave.ViewProviders import _DocumentWatcher

        theirs = doc.addObject(TYPE, "TheirMesh")
        theirs.addProperty("App::PropertyEnumeration", "Status")
        theirs.Status = [preview.CURRENT, preview.OUT_OF_DATE]
        theirs.Label = "Mesh (Palace)"
        _DocumentWatcher().slotRecomputedObject(theirs)
        assert (theirs.Status, theirs.Label) == (preview.CURRENT, "Mesh (Palace)")


class TestTheStudyPairsAResultWithTheMeshItShows:
    """What the study holds decides which mesh it shows, so a change to that is
    when the study asks again."""

    def _solved(self, doc):
        import json

        from Microwave.Gui import fem_mesh as shown

        study = analysis.createEMAnalysis(doc=doc)
        made = shown.put(study, "Palace", "a mesh", key=shown.inputs(study), identity="solved")
        result = results.createEMSParameters(doc=doc)
        result.Provenance = json.dumps({"solver": "Palace", results.SOLVED_ON: "solved"})
        result.Label = results.label("Palace")
        study.addObject(result)
        return study, made, result

    def test_the_mesh_deleted_marks_the_result(self, doc):
        study, made, result = self._solved(doc)
        doc.removeObject(made.Name)
        study.Proxy.onChanged(study, "Group")
        assert result.Label == results.label("Palace", apart=True)

    def test_another_mesh_shown_marks_it_and_its_own_back_unmarks_it(self, doc):
        from Microwave.Gui import fem_mesh as shown

        study, made, result = self._solved(doc)
        spare = shown.put(study, "Elsewhere", "a mesh", identity="spare")
        spare.MeshedFor = "Palace"
        study.Group = [member for member in study.Group if member is not made]
        study.Proxy.onChanged(study, "Group")
        assert result.Label == results.label("Palace", apart=True)
        study.Group = [member for member in study.Group if member is not spare]
        study.addObject(made)
        study.Proxy.onChanged(study, "Group")
        assert result.Label == results.label("Palace")

    def test_an_undo_is_left_alone(self, doc):
        study, made, result = self._solved(doc)
        doc.removeObject(made.Name)
        doc.Transacting = True
        try:
            study.Proxy.onChanged(study, "Group")
        finally:
            del doc.Transacting
        assert result.Label == results.label("Palace")
