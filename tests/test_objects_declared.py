# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""An object restored from a document an earlier build saved is named for every
way it departs from its class, and a study holding one is refused in one
sentence.

On the stubs, which record what each property was declared as.
``tests/test_openems_palace_declared.py`` holds the same under a real FreeCAD,
on a document saved and opened again.
"""

import pytest

from Microwave.Objects.declared import declarations, departures, linked
from Microwave.Objects.kinds import classes
from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding
from Microwave.Objects.mesh import createEMMeshPolicy
from Microwave.Solvers.errors import TranslationError
from Microwave.Solvers.properties import out_of_step


def made(doc, kind):
    obj = doc.addObject("App::FeaturePython", kind)
    classes()[kind](obj)
    return obj


class TestWhatDeparts:
    @pytest.mark.parametrize("kind", sorted(classes()))
    def test_an_object_this_build_makes_departs_in_nothing(self, doc, kind):
        assert departures(made(doc, kind)).phrases == ()

    def test_a_property_added_since_is_named_as_missing(self, doc):
        policy = createEMMeshPolicy(doc)
        policy.removeProperty("Medium")
        policy.removeProperty("Clearance")
        departed = departures(policy)
        assert departed.phrases == ("has no Clearance, Medium",)
        assert departed.stops

    def test_a_property_retired_since_is_named_with_its_value(self, doc):
        policy = createEMMeshPolicy(doc)
        policy.addProperty("App::PropertyFloat", "ElementsPerWavelength", "Mesh")
        policy.ElementsPerWavelength = 30.0
        departed = departures(policy)
        assert departed.phrases == (
            "carries ElementsPerWavelength 30.0, which this build does not declare",
        )
        assert not departed.stops, "a property the user added would stop every run"

    def test_a_link_retired_since_is_named_by_what_it_links(self, doc):
        binding = createEMMaterialBinding("Binding", doc)
        binding.addProperty("App::PropertyLinkList", "Holds", "Material")
        binding.Holds = [createEMMaterial("Copper", doc)]
        said = "carries Holds ['Copper'], which this build does not declare"
        assert departures(binding).phrases == (said,)

    def test_an_enumeration_saved_with_other_choices_is_named(self, doc):
        """FreeCAD writes an enumeration's choices into the document, so a
        choice added since is not one the saved object offers."""
        policy = createEMMeshPolicy(doc)
        policy.PaddingXMin = ["Air", "Through"]
        departed = departures(policy)
        assert departed.phrases == (
            "offers Air, Through for PaddingXMin where this build offers Air, Through, Ends",
        )
        assert departed.stops

    def test_a_property_saved_as_another_type_is_named(self, doc):
        policy = createEMMeshPolicy(doc)
        declared = policy.getTypeIdOfProperty("Clearance")
        policy.removeProperty("Clearance")
        policy.addProperty("App::PropertyInteger", "Clearance", "Domain")
        departed = departures(policy)
        assert departed.phrases == (
            f"holds Clearance as App::PropertyInteger where this build declares {declared}",
        )
        assert departed.stops

    def test_every_departure_of_one_object_is_named_at_once(self, doc):
        policy = createEMMeshPolicy(doc)
        policy.removeProperty("Medium")
        policy.addProperty("App::PropertyFloat", "MaxGrowthRatio", "Mesh")
        policy.PaddingZMax = ["Air", "Through"]
        said = departures(policy).phrases
        assert [phrase.split()[0] for phrase in said] == ["has", "carries", "offers"]

    def test_an_enumeration_saved_with_its_choices_reordered_is_named_so(self, doc):
        """FreeCAD keeps an enumeration's value as an index into its choices."""
        policy = createEMMeshPolicy(doc)
        policy.PaddingXMin = ["Ends", "Air", "Through"]
        departed = departures(policy)
        assert departed.phrases == (
            "orders the choices for PaddingXMin Ends, Air, Through where this build orders "
            "them Air, Through, Ends",
        )
        assert departed.stops

    def test_a_value_that_cannot_be_read_is_said_to_be_so(self, doc):
        """A property no class declares may hold anything, such as an object since
        deleted, and reading it must not raise in place of the refusal."""

        class Deleted:
            @property
            def Label(self):
                raise ReferenceError("Cannot access attribute 'Label' of deleted object")

        policy = createEMMeshPolicy(doc)
        policy.addProperty("App::PropertyPythonObject", "Box", "Base")
        policy.Box = Deleted()
        assert (
            departures(policy).carried
            == "carries Box (unreadable), which this build does not declare"
        )

    def test_a_long_value_is_cut(self, doc):
        policy = createEMMeshPolicy(doc)
        policy.addProperty("App::PropertyFloatList", "Samples", "Base")
        policy.Samples = [float(sample) for sample in range(1000)]
        assert len(departures(policy).carried) < 200

    def test_what_a_binding_links_is_its_material_and_the_bodies_it_binds(self, doc):
        """Objects alone, whatever form the link property holds them in."""
        material = createEMMaterial("Copper", doc)
        body = doc.addObject("Part::Feature", "Trace")
        binding = createEMMaterialBinding("Binding", doc)
        binding.Material = material
        binding.References = [(body, ["Face1"])]
        assert sorted(obj.Name for obj in linked(binding)) == sorted([material.Name, body.Name])

    def test_what_a_policy_links_is_its_medium_and_nothing_it_states(self, doc):
        medium = createEMMaterial("Water", doc)
        policy = createEMMeshPolicy(doc)
        policy.Medium = medium
        assert linked(policy) == (medium,)

    def test_an_object_of_no_kind_here_departs_in_nothing(self, doc):
        """Another workbench's object carries what its own class declares."""
        foreign = doc.addObject("App::FeaturePython", "Foreign")
        foreign.addProperty("App::PropertyLength", "Width", "Base")
        assert departures(foreign).phrases == ()

    def test_the_proxy_answers_for_its_object(self, doc):
        policy = createEMMeshPolicy(doc)
        policy.removeProperty("Medium")
        assert policy.Proxy.departures(policy) == departures(policy)

    def test_the_declarations_leave_the_active_document_as_it_was(self, doc):
        held = [obj.Name for obj in doc.Objects]
        declarations.cache_clear()
        assert set(declarations()) == set(classes())
        assert [obj.Name for obj in doc.Objects] == held


class TestOneRefusal:
    def test_every_object_that_departs_is_named_in_one_refusal(self, doc):
        policy = createEMMeshPolicy(doc)
        policy.removeProperty("Medium")
        material = createEMMaterial("Copper", doc)
        material.removeProperty("DispersionFrequency")
        with pytest.raises(TranslationError) as refused:
            out_of_step([policy, material])
        said = str(refused.value)
        assert "'Mesh Policy' has no Medium" in said
        assert "'Copper' has no DispersionFrequency" in said

    def test_a_study_whose_objects_are_this_builds_is_left_alone(self, doc):
        out_of_step([createEMMeshPolicy(doc), createEMMaterial("Copper", doc)])

    def test_a_material_is_reached_through_what_links_it(self, doc):
        """A material stands at the document's root, outside every study."""
        material = createEMMaterial("Copper", doc)
        material.removeProperty("DispersionFrequency")
        binding = createEMMaterialBinding("Binding", doc)
        binding.Material = material
        with pytest.raises(TranslationError, match="'Copper' has no DispersionFrequency"):
            out_of_step([binding])

    def test_a_link_the_user_added_is_not_followed(self, doc):
        """No run reads through it, so what it links to stops nothing."""
        material = createEMMaterial("Copper", doc)
        material.removeProperty("DispersionFrequency")
        binding = createEMMaterialBinding("Binding", doc)
        binding.addProperty("App::PropertyLink", "Datasheet", "Base")
        binding.Datasheet = material
        out_of_step([binding])

    def test_the_policy_s_medium_is_followed(self, doc):
        medium = createEMMaterial("Water", doc)
        medium.removeProperty("DispersionFrequency")
        policy = createEMMeshPolicy(doc)
        policy.Medium = medium
        with pytest.raises(TranslationError, match="'Water' has no DispersionFrequency"):
            out_of_step([policy])

    def test_a_property_no_class_declares_stops_nothing_by_itself(self, doc):
        """It may be one the user added in the property editor."""
        policy = createEMMeshPolicy(doc)
        policy.addProperty("App::PropertyLength", "Width", "Base")
        out_of_step([policy])

    def test_it_is_named_with_its_value_where_the_run_is_stopped(self, doc):
        """The object made anew does not hold what the user typed into it."""
        policy = createEMMeshPolicy(doc)
        policy.addProperty("App::PropertyFloat", "EdgeRefinement", "Mesh")
        policy.EdgeRefinement = 3.0
        material = createEMMaterial("Copper", doc)
        material.removeProperty("DispersionFrequency")
        with pytest.raises(TranslationError) as refused:
            out_of_step([policy, material])
        said = str(refused.value)
        assert "'Mesh Policy' carries EdgeRefinement 3.0" in said
        assert "Delete Property" in said, "FreeCAD's own words for the editor's command"
