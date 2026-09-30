# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A mesh region is read once, and both backends refuse it in the same words."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Microwave.Solvers import mesh_regions
from Microwave.Solvers.errors import TranslationError
from Microwave.Solvers.openems import document as openems_document
from Microwave.Solvers.openems import geometry as openems_geometry
from Microwave.Solvers.palace import document as palace_document
from tests import test_openems_document_translation as openems_study
from tests import test_palace_adapter as palace_study


class Shape:
    """A shape whose elements a pick can name, each a shape of its own."""

    def __init__(self, name, elements=None):
        self.name = name
        self.elements = dict(elements or {})

    def getElement(self, name):
        return self.elements[name]


def drawn(name, shape):
    return SimpleNamespace(
        Proxy=type("Part::Feature", (), {})(), Label=name, Name=name, Shape=shape
    )


def region(references, **overrides):
    properties = dict(
        Mode="Refine", References=references, ElementSize=0.5, MinElementsAcross=0, Enabled=True
    )
    properties.update(overrides)
    return SimpleNamespace(
        Proxy=type("EMMeshRegion", (), {})(), Label="Ring", Name="EMMeshRegion", **properties
    )


def palace_regions(*stated):
    """What the Palace adapter reads the regions as, in a study binding nothing."""
    return palace_document._refinements(SimpleNamespace(refinements=stated), [], {}, [], {}, {})


class TestWhatARegionNames:
    def test_a_vertex_is_named_by_its_element_and_read_as_that_element(self):
        corner = Shape("corner")
        rod = drawn("Rod", Shape("rod", {"Vertex3": corner}))
        (named,) = mesh_regions.read(region([(rod, ["Vertex3"])])).named
        assert (named.name, named.subject, named.shape) == (
            "Rod:Vertex3",
            ("Rod", "Vertex3"),
            corner,
        )

    def test_a_compound_named_whole_is_read_as_its_whole_shape(self):
        """A boolean hands back a compound, and naming it whole names what it holds."""
        compound = Shape("compound")
        fused = drawn("Fusion", compound)
        (named,) = mesh_regions.read(region([fused])).named
        assert (named.name, named.subject, named.shape) == ("Fusion", ("Fusion", ""), compound)

    def test_each_element_of_one_reference_is_named_apart(self):
        rod = drawn("Rod", Shape("rod", {"Face1": Shape("top"), "Edge2": Shape("edge")}))
        named = mesh_regions.read(region([(rod, ["Face1", "Edge2"])])).named
        assert [one.name for one in named] == ["Rod:Face1", "Rod:Edge2"]

    def test_a_coarsening_reads_no_shape(self):
        """It names what a binding sizes, so an element an edit left behind is not
        resolved."""
        rod = drawn("Rod", Shape("rod"))
        (named,) = mesh_regions.read(region([(rod, ["Face9"])], Mode="Coarsen")).named
        assert (named.subject, named.shape) == (("Rod", "Face9"), None)

    def test_the_subject_is_the_objects_name_rather_than_its_label(self):
        rod = drawn("Rod", Shape("rod"))
        rod.Name = "Box001"
        (named,) = mesh_regions.read(region([rod])).named
        assert (named.name, named.subject) == ("Rod", ("Box001", ""))


FAULTS = [
    (dict(ElementSize=0.0), "has no element size set"),
    (dict(ElementSize=float("nan")), "has no element size set"),
    (dict(References=[]), "refines nothing"),
    (dict(References=[], Mode="Coarsen"), "coarsens nothing"),
    (dict(Mode="Sideways"), "Mode is 'Sideways'"),
    (dict(MinElementsAcross=-1), "MinElementsAcross is -1"),
    (dict(Mode="Coarsen", MinElementsAcross=3), "set to Coarsen and asks for 3"),
    (dict(References=[drawn("Nothing", None)]), "'Nothing', which has no shape"),
]


class TestBothBackendsRefuseInOneVoice:
    @pytest.mark.parametrize(("changed", "said"), FAULTS)
    def test_the_reader_refuses_by_name(self, changed, said):
        with pytest.raises(TranslationError, match="'Ring'") as refused:
            mesh_regions.read(region([drawn("Rod", Shape("rod"))], **changed))
        assert said in str(refused.value)

    @pytest.mark.parametrize(("changed", "said"), FAULTS)
    def test_each_backend_refuses_in_the_readers_words(self, changed, said):
        """Each backend's own translation of a whole study, so a backend that
        stopped reading regions through the reader would answer otherwise."""
        with pytest.raises(TranslationError) as read:
            mesh_regions.read(region([drawn("Rod", Shape("rod"))], **changed))
        doc = openems_study.model()
        openems_study._add(
            doc,
            openems_study.refinement("Ring", openems_study.part(doc, "Trace"), **changed),
        )
        with pytest.raises(TranslationError) as openems:
            openems_document.problem(doc.Objects[0])
        found, body = palace_study.guide()
        found.Group = [*found.Group, palace_study.refinement([(body, [""])], **changed)]
        with pytest.raises(TranslationError) as palace:
            palace_document.problem(found)
        assert said in str(read.value)
        assert str(palace.value) == str(openems.value) == str(read.value)

    def test_each_backend_names_what_a_region_names_alike(self):
        box = SimpleNamespace(XMin=0.0, YMin=0.0, ZMin=0.0, XMax=1.0, YMax=1.0, ZMax=1.0)
        top = Shape("top")
        top.BoundBox, top.Faces = box, [top]
        whole = Shape("rod", {"Face1": top})
        whole.BoundBox, whole.Solids = box, [whole]
        rod = drawn("Rod", whole)
        stated = region([(rod, ["Face1"]), rod])
        (palace,) = palace_regions(stated)
        openems = openems_geometry._sizing_regions([stated])
        assert [mark.name for mark in palace.marks] == [one.label for one in openems]
        assert [one.label for one in openems] == ["Ring on Rod:Face1", "Ring on Rod"]

    def test_a_region_left_out_is_read_by_neither(self):
        stated = region([], Enabled=False, ElementSize=0.0, Mode="Sideways")
        assert palace_regions(stated) == []
        assert openems_document.relaxations([stated]) == {}
        assert openems_geometry._sizing_regions([stated]) == ()
