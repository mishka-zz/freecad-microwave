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
}

#: Every class that declares nothing, so that a class starting to declare has
#: to be named in one list or the other rather than slipping past both.
SILENT = (
    materials.EMMaterial,
    materials.EMMaterialBinding,
    mesh.EMMeshPolicy,
    mesh.EMMeshRegion,
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

        ``MeshedFrom`` is filled the way ``Gui/mesh_preview.py::_link`` fills
        it, because membership is answered by comparing the two.
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
        ["NumFrequencyPoints", "SmallestResponse", "Symmetry", "Waveform"],
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

    ``Gui/mesh_preview.py::_link`` rebuilds the preview's link list at each
    Update Mesh and nowhere else, so an object added afterwards is in no list:
    it reaches the preview through nothing, and neither does any later edit to
    it. Comparing what the study holds against what the drawing was made from
    is what answers that, and it translates nothing.
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

    def test_a_study_with_no_drawing_is_not_an_error(self, doc):
        study = analysis.createEMAnalysis(doc=doc)
        study.Proxy.onChanged(study, "Group")
