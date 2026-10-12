# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The document's own materials.

An ``EMMaterial`` holds values, and the solve reads them. A catalog is a
starting point rather than a live link. Values are copied in once, and with them
the entry's measured table where it has one. Where they came from is recorded
beside them, in the Provenance group.

An engineer who measures their own laminate and types 4.15 over the catalog's
4.3 keeps that number. ``SourceDigest`` records the values as the catalog stated
them, and translation compares it with the values the object holds now. While
the two agree, each study solves at the table row nearest its own band. Once
they differ, the material has been edited, and every study solves at the values
shown. The catalog itself is never read at translation, so a ``.FCStd`` solves
unchanged on a machine with no catalogs installed at all.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import FreeCAD

from ._vp_hook import ViewProviderRestored

if TYPE_CHECKING:
    from ..Materials.model import Catalog, MaterialEntry


class EMMaterial(ViewProviderRestored):
    def __init__(self, obj):
        obj.addProperty(
            "App::PropertyEnumeration",
            "MaterialType",
            "Material",
            "Dielectric, PEC for a perfect conductor, or Conducting Sheet for a metal"
            " of finite Conductivity and Thickness",
        )
        obj.MaterialType = [
            "Dielectric",
            "PEC",
            "Conducting Sheet",
            "Frequency Dependent Dielectric",
        ]
        obj.addProperty("App::PropertyFloat", "Permittivity", "Material", "Relative permittivity")
        obj.Permittivity = 1.0
        obj.addProperty("App::PropertyFloat", "Permeability", "Material", "Relative permeability")
        obj.Permeability = 1.0
        obj.addProperty(
            "App::PropertyFloat", "Conductivity", "Material", "Conductivity, in S/m (0 = none)"
        )
        obj.Conductivity = 0.0
        obj.addProperty(
            "App::PropertyFloat", "LossTangent", "Material", "Dielectric loss tangent (0 = none)"
        )
        obj.LossTangent = 0.0
        obj.addProperty(
            "App::PropertyLength",
            "Thickness",
            "Material",
            "Metal thickness of a Conducting Sheet. It sets the loss and is not drawn",
        )
        obj.Thickness = 0.035
        obj.addProperty(
            "App::PropertyColor", "Color", "Material", "Colour of the shapes bound to this material"
        )
        obj.Color = (0.8, 0.8, 0.8)

        # The frequency Permittivity and LossTangent are quoted at. Editable,
        # because an engineer typing a measured permittivity of their own has to
        # state where it was measured. Zero means unstated, which is what a
        # hand-made material starts as.
        obj.addProperty(
            "App::PropertyFrequency",
            "MeasuredAt",
            "Material",
            "Frequency Permittivity and LossTangent are quoted at (0 = not stated)",
        )
        obj.MeasuredAt = 0.0

        # The catalog's own table, row by row, where the entry has one. Read-only,
        # because a second place to edit the values would skip the rule that an
        # edited material keeps what was typed. Solvers/materials.py reads it.
        for name, description in (
            ("DispersionFrequency", "Frequency of each measured row in the catalog"),
            ("DispersionPermittivity", "Permittivity the catalog states at each row"),
            ("DispersionLossTangent", "Loss tangent the catalog states at each row"),
        ):
            obj.addProperty(
                "App::PropertyFloatList",
                name,
                "Material",
                f"{description}. Each study solves at the row nearest its band centre"
                " until a value of this material is edited, and at the values shown after",
            )
            obj.setEditorMode(name, 1)

        # Provenance. Read-only in the editor, because these record what
        # happened rather than set anything. Editing Source would not fetch
        # anything, and a property that looks like it does something and does
        # not is a silent no-op.
        for name, description in (
            ("Source", "Catalog and material this came from, as catalog:material"),
            ("SourceCatalog", "Name and version of that catalog when it was picked"),
            ("SourceDigest", "Checksum of the values as the catalog stated them"),
            ("Description", "The catalog's description of this material"),
        ):
            obj.addProperty("App::PropertyString", name, "Provenance", description)
            obj.setEditorMode(name, 1)

        obj.Proxy = self

    def execute(self, obj):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


class EMMaterialBinding(ViewProviderRestored):
    def __init__(self, obj):
        obj.addProperty(
            "App::PropertyLink", "Material", "Binding", "The material References are made of"
        )
        obj.addProperty(
            "App::PropertyLinkSubList",
            "References",
            "Binding",
            "The bodies or faces made of Material",
        )
        obj.Proxy = self

    def execute(self, obj):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


def createEMMaterial(name: str = "EMMaterial", doc: Any = None) -> Any:
    doc = doc or FreeCAD.ActiveDocument
    # The internal Name has to be an identifier, and a catalog's name is not one
    # ("FR-4 TG155", "Copper, 1 oz"). FreeCAD would silently mangle or reject
    # such a name, so the readable form goes on the Label.
    obj = doc.addObject("App::FeaturePython", _identifier(name))
    obj.Label = name
    EMMaterial(obj)
    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMMaterial")
    return obj


def _identifier(name: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in name)
    return cleaned.strip("_") or "EMMaterial"


#: What a binding made without a name of its own is called in the tree.
BINDING_LABEL = "Material Binding"


def createEMMaterialBinding(name: str = "EMMaterialBinding", doc: Any = None) -> Any:
    doc = doc or FreeCAD.ActiveDocument
    obj = doc.addObject("App::FeaturePython", name)
    EMMaterialBinding(obj)
    if name == "EMMaterialBinding":
        obj.Label = BINDING_LABEL
    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMMaterialBinding")
    return obj


def fill_binding(
    binding: Any, selection: Iterable[Any], study: Any, copies: Sequence[str] = ()
) -> list[str]:
    """Set a binding's two links from what the user had picked. Returns the rest.

    The two halves of a pick are distinguishable: one of the objects is an
    ``EMMaterial`` and the others are geometry. Order says nothing here and is
    not read. A port is different - which of two copper faces is the trace
    cannot be read off the shapes - so ``port_setup`` documents an order and
    honours it.

    The binding is made either way, as a port is. Translation refuses one that
    names no material and one that binds to nothing, both by name, so leaving a
    link empty costs a message and never a wrong answer. Half a binding with the
    other half to fill in is more use than no binding and an error.

    ``study`` is the study the binding joins. A pick that depends on it is left
    out of ``References`` and named, since a link to it would close a cycle.
    ``copies`` names the picks :func:`Microwave.picks.resolved` left out, made
    inside the copy a link shows.
    """
    from ..picks import copy_note
    from .kinds import kind_of
    from .mesh import cycle_note, reaching, references_from

    picked = [getattr(chosen, "Object", chosen) for chosen in selection]
    materials = [obj for obj in picked if kind_of(obj) == "EMMaterial"]
    references = references_from(selection, study)

    notes = []
    if len(materials) == 1:
        binding.Material = materials[0]
    elif not materials:
        notes.append("no material was selected; set Material in the property editor")
    else:
        # Order will not break this tie. A binding carries one material, and
        # taking the first would be an assignment the user never made and would
        # not see without opening the property editor.
        named = ", ".join(repr(obj.Label) for obj in materials)
        notes.append(
            f"{len(materials)} materials were selected ({named}), and a binding takes "
            "one. Set Material in the property editor"
        )

    if copies:
        notes.append(copy_note(copies))
    if cyclic := reaching(selection, study):
        notes.append(cycle_note(cyclic))
    if references:
        binding.References = references
    elif not cyclic and not copies:
        notes.append("no geometry was selected; set References in the property editor")
    return notes


def apply_entry(obj: Any, entry: MaterialEntry, catalog: Catalog) -> Any:
    """Copy one catalog entry's values onto an EMMaterial, with its table and
    its provenance."""
    from ..Materials.model import DOCUMENT_TYPES, MaterialRef

    obj.MaterialType = DOCUMENT_TYPES[entry.kind]
    obj.Permittivity = entry.epsilon_r
    obj.Permeability = entry.mu_r
    obj.Conductivity = entry.conductivity
    obj.LossTangent = entry.loss_tangent
    obj.Thickness = entry.thickness
    obj.MeasuredAt = entry.measured_at
    obj.Color = entry.rgb()
    obj.DispersionFrequency = [row.frequency for row in entry.rows]
    obj.DispersionPermittivity = [row.epsilon_r for row in entry.rows]
    obj.DispersionLossTangent = [row.loss_tangent for row in entry.rows]

    obj.Source = str(MaterialRef(catalog.id, entry.id))
    obj.SourceCatalog = f"{catalog.name} {catalog.version}".strip()
    obj.SourceDigest = entry.digest()
    obj.Description = entry.description
    return obj


def sourced_from(doc: Any, ref: object) -> list[Any]:
    """Every material in the document carrying ``ref``, in document order.

    More than one is legitimate. A stackup carries an FR-4 layer on each side of
    a Rogers core with their own numbers, and an uncatalogued laminate is
    entered by taking the nearest catalog row and editing it. The command that
    adds them names the ones already there, and does not refuse.

    A label cannot say whether a material came from a given reference, because
    :func:`label_for` numbers by what is free rather than by what a material
    came from. This function is the only thing that can.
    """
    wanted = str(ref)
    return [obj for obj in doc.Objects if getattr(obj, "Source", "") == wanted]


def label_for(doc: Any, entry: MaterialEntry, catalog: Catalog) -> str:
    """``Generic FR4 (1)`` - catalog, material, and which one of them.

    Every part is written unconditionally. Adding a part only when the document
    forces it would mean deciding what a collision is with, in order to choose
    between a catalog and a number, and there is no right answer: a board solid
    named after its laminate, or two catalogs sharing a display name, turn one
    suffix into the other's meaning. Writing both always means neither has to
    stand in for the other.

    Numbering from ``(1)`` rather than leaving the first bare follows the same
    argument. In a tree where the first is ``Generic FR4`` and the second
    ``Generic FR4 (2)``, the reader has to notice an absence. Numbering both puts
    the answer in the same place on every row.

    The number cannot come from FreeCAD. With duplicate labels disallowed, which
    is the default, FreeCAD resolves a collision by appending its own counter,
    and on a name that already ends in a digit that reads as a different part:
    ``Generic FR4001``.
    """
    labelled = {obj.Label for obj in doc.Objects}
    stem = f"{catalog.name} {entry.name}"
    number = 1
    while f"{stem} ({number})" in labelled:
        number += 1
    return f"{stem} ({number})"


def create_from_entry(doc: Any, entry: MaterialEntry, catalog: Catalog) -> Any:
    """A new EMMaterial at document root, carrying one catalog entry."""
    obj = createEMMaterial(label_for(doc, entry, catalog), doc=doc)
    return apply_entry(obj, entry, catalog)
