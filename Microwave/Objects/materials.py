# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The document's own materials.

An ``EMMaterial`` holds *values*, and the solve reads those values and nothing
else. Where they came from is recorded beside them, in the Provenance group, and
is never consulted at translation time - which is what makes a ``.FCStd``
solve unchanged on a machine with no catalogs installed at all.

That is the whole relationship between this module and ``Microwave.Materials``:
a catalog is a **starting point**, not a live link. Values are copied in once.
An engineer who measures their own laminate and types 4.15 over the catalog's
4.3 keeps that number for ever; the provenance then says the material has been
edited since it was imported, which is information rather than a fault.
"""

import FreeCAD

from ._vp_hook import ViewProviderRestored


class EMMaterial(ViewProviderRestored):
    def __init__(self, obj):
        obj.addProperty("App::PropertyEnumeration", "MaterialType", "Material")
        obj.MaterialType = ["Dielectric", "PEC", "ConductingSheet", "FrequencyDependentDielectric"]
        obj.addProperty("App::PropertyFloat", "Permittivity", "Material")
        obj.Permittivity = 1.0
        obj.addProperty("App::PropertyFloat", "Permeability", "Material")
        obj.Permeability = 1.0
        obj.addProperty("App::PropertyFloat", "Conductivity", "Material")
        obj.Conductivity = 0.0
        obj.addProperty("App::PropertyFloat", "LossTangent", "Material")
        obj.LossTangent = 0.0
        obj.addProperty("App::PropertyLength", "Thickness", "Material")
        obj.Thickness = 0.035
        obj.addProperty("App::PropertyColor", "Color", "Material")
        obj.Color = (0.8, 0.8, 0.8)

        # The frequency Permittivity and LossTangent are quoted at. Editable,
        # because typing your own measured permittivity means saying where you
        # measured it. Zero means unstated, which is what a hand-made material
        # starts as.
        obj.addProperty(
            "App::PropertyFrequency",
            "MeasuredAt",
            "Material",
            "Frequency the permittivity and loss tangent are quoted at (0 = unstated)",
        )
        obj.MeasuredAt = 0.0

        # Provenance. Read-only in the editor because it is a record of what
        # happened, not a setting: editing Source would not fetch anything, and
        # a property that looks like it does something and does not is a
        # silent no-op.
        for name, description in (
            ("Source", "Catalog and material this came from, as catalog:material"),
            ("SourceCatalog", "Name and version of that catalog when it was picked"),
            ("SourceDigest", "Fingerprint of the values as the catalog stated them"),
            ("Description", "What the catalog says this material is"),
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
        obj.addProperty("App::PropertyLink", "Material", "Binding")
        obj.addProperty("App::PropertyLinkSubList", "References", "Binding")
        obj.Proxy = self

    def execute(self, obj):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


def createEMMaterial(name="EMMaterial", doc=None):
    doc = doc or FreeCAD.ActiveDocument
    # The internal Name has to be an identifier; a catalog's name does not
    # ("FR-4 TG155", "Copper, 1 oz"). FreeCAD would silently mangle or reject
    # it, so the readable form goes on the Label where it belongs.
    obj = doc.addObject("App::FeaturePython", _identifier(name))
    obj.Label = name
    EMMaterial(obj)
    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMMaterial")
    return obj


def _identifier(name):
    cleaned = "".join(character if character.isalnum() else "_" for character in name)
    return cleaned.strip("_") or "EMMaterial"


def createEMMaterialBinding(name="EMMaterialBinding"):
    obj = FreeCAD.ActiveDocument.addObject("App::FeaturePython", name)
    EMMaterialBinding(obj)
    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMMaterialBinding")
    return obj


def fill_binding(binding, selection):
    """Set a binding's two links from what the user had picked. Returns the rest.

    A pick tells its own halves apart: one of the objects is an ``EMMaterial``
    and the others are geometry. Order says nothing and is not read - unlike a
    port, where which of two copper faces is the trace is genuinely unknowable
    from the shapes, so ``port_setup`` documents an order and honours it.

    The binding is made either way, as a port is. Translation refuses one that
    names no material and one that binds to nothing, both by name, so leaving a
    link empty costs a message and never a wrong answer - and half a binding
    with the other half to fill in is a better place to stand than no binding
    and an error.
    """
    from .kinds import kind_of
    from .mesh import references_from

    picked = [getattr(chosen, "Object", chosen) for chosen in selection]
    materials = [obj for obj in picked if kind_of(obj) == "EMMaterial"]
    references = references_from(selection)

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
            f"{len(materials)} materials were selected ({named}); a binding carries "
            "one, so set Material in the property editor"
        )

    if references:
        binding.References = references
    else:
        notes.append("no geometry was selected; set References in the property editor")
    return notes


def apply_entry(obj, entry, catalog):
    """Copy one catalog entry's values onto an EMMaterial, with its provenance."""
    from ..Materials.model import MaterialRef

    obj.MaterialType = _MATERIAL_TYPES[entry.kind]
    obj.Permittivity = entry.epsilon_r
    obj.Permeability = entry.mu_r
    obj.Conductivity = entry.conductivity
    obj.LossTangent = entry.loss_tangent
    obj.Thickness = entry.thickness
    obj.MeasuredAt = entry.measured_at
    obj.Color = entry.rgb()

    obj.Source = str(MaterialRef(catalog.id, entry.id))
    obj.SourceCatalog = f"{catalog.name} {catalog.version}".strip()
    obj.SourceDigest = entry.digest()
    obj.Description = entry.description
    return obj


#: The catalog's vocabulary, which is the adapter's, mapped onto the document
#: object's enumeration. Two spellings of one set exist because the enumeration
#: is what a FreeCAD user sees in a dropdown and the other is what a file says.
_MATERIAL_TYPES = {
    "dielectric": "Dielectric",
    "pec": "PEC",
    "conducting_sheet": "ConductingSheet",
}


def sourced_from(doc, ref):
    """Every material in the document carrying ``ref``, in document order.

    Plural, because more than one is legitimate: a stackup wants an FR-4 layer
    on each side of a Rogers core with their own numbers, and the way an
    uncatalogued laminate gets entered is to take the nearest catalog row and
    edit it. The command that adds them names the ones already there; it does
    not refuse.

    This is the only thing that can answer "is this one of those?", because a
    label cannot: :func:`label_for` numbers by what is free, not by what a
    material came from.
    """
    wanted = str(ref)
    return [obj for obj in doc.Objects if getattr(obj, "Source", "") == wanted]


def label_for(doc, entry, catalog):
    """``Generic FR4 (1)`` - catalog, material, and which one of them.

    All three parts unconditionally, which is the whole of the rule. Every
    version of this that earned a part only when the document forced it had to
    decide *what a collision was with* in order to pick between a catalog and a
    number, and every such decision was wrong somewhere: a board solid named
    after its laminate, or two catalogs sharing a display name, turned one
    suffix into the other's meaning. Saying both always means neither has to
    stand in for the other, and there is nothing left to get wrong.

    Numbering from ``(1)`` rather than leaving the first bare is the same
    argument. A tree where the first is ``Generic FR4`` and the second
    ``Generic FR4 (2)`` asks the reader to notice an absence; numbering both
    puts the answer in the same place on every row.

    The number cannot come from FreeCAD. With duplicate labels disallowed -
    the default - it resolves a collision by appending its own counter, and
    on a name that already ends in a digit that reads as a different part:
    ``Generic FR4001``.
    """
    labelled = {obj.Label for obj in doc.Objects}
    stem = f"{catalog.name} {entry.name}"
    number = 1
    while f"{stem} ({number})" in labelled:
        number += 1
    return f"{stem} ({number})"


def create_from_entry(doc, entry, catalog):
    """A new EMMaterial at document root, carrying one catalog entry."""
    obj = createEMMaterial(label_for(doc, entry, catalog), doc=doc)
    return apply_entry(obj, entry, catalog)
