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
    GAUSSIAN,
    NoAnalysis,
    analyses,
    analysis_of,
    createEMAnalysis,
    find_analysis,
    solver_of,
)
from Microwave.Objects.mesh import createEMMeshRegion


def kind(obj):
    return type(getattr(obj, "Proxy", None)).__name__


class TestWhatCreatingOneGives:
    def test_it_comes_with_a_solver_and_a_mesh_policy(self, doc):
        """None of the three is useful alone, and assembling them by hand on
        every new document is three clicks to reach the state everyone wants."""
        analysis = createEMAnalysis(doc)
        assert sorted(kind(o) for o in analysis.Group) == [
            "EMMeshPolicy",
            "EMSolverOpenEMS",
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
    reaches is the bar a truncated run is judged against, and nothing else.
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
        assert document._smallest_response(createEMAnalysis(doc)) == pytest.approx(
            undeclared["smallest_response"]
        )


class TestTheExcitationDeclaration:
    """An enumeration of one, and the one is what the engine is driven with.

    The property is kept for what it names rather than for what it offers: a
    study saying how it is excited is describing the problem, and the day an
    adapter learns a second waveform there is somewhere for it to go. What is
    forbidden is the other order - offering the choice first, where it changes
    no envelope and no answer.
    """

    def test_it_offers_only_what_the_adapter_can_produce(self, doc):
        analysis = createEMAnalysis(doc)
        assert analysis.getEnumerationsOfProperty("Waveform") == [GAUSSIAN]
        assert analysis.Waveform == GAUSSIAN

    def test_the_adapter_accepts_exactly_the_value_the_property_offers(self):
        """The two copies of the string, in one place.

        They are copies because the adapter may not import the document layer
        - that reaches ``FreeCAD`` - so nothing but this holds them
        together, and a document offering a waveform the translation refuses is
        a workbench that cannot solve its own default.
        """
        from Microwave.Solvers.openems import policy

        assert policy.GAUSSIAN == GAUSSIAN


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
            "EMMeshPolicy",
            "EMMeshRegion",
            "EMMeshPreview",
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

    def test_a_foreign_object_is_not_ours(self, doc):
        from Microwave.Objects.kinds import is_ours

        assert not is_ours(doc.addObject("Part::Box", "Substrate"))

    def test_every_one_of_ours_is_ours(self, doc):
        from Microwave.Objects.kinds import is_ours
        from Microwave.Objects.preview import createEMMeshPreview

        analysis = createEMAnalysis(doc)
        for obj in (analysis, *analysis.Group, createEMMeshPreview(doc)):
            assert is_ours(obj), obj.Name
