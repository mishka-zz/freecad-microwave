# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What each kind of edit costs a recompute, and where it leaves the badge.

Run by ``test_preview_recompute`` under ``freecadcmd``. The suite's stubs cannot
answer this: whether a property assignment touches its object is FreeCAD's
decision, and a stub is free to model it either way, so a stub asserting the
answer would be asserting itself. What a recompute then reaches is FreeCAD's
too.

Every case is one change to one open document, followed by one recompute. What
is recorded is how many times the preview's ``execute`` ran, how many times the
staleness key was derived, what the object's state was after the change, and
where the badge ended up.

The sweep at the end asks the other half: over every value property of every
object a study holds, does the badge fire wherever the key moves. That is the
direction the declaration must not be wrong in - a false "out of date" costs a
button press, and a false "current" is a drawing that lies. It leaves out what
points rather than measures: a link is not repointed here, because that is
membership, which the named cases cover.

``manifest.json`` names what was counted and is written last, so its existence
rather than the exit status is what says this finished - see
``tests/conftest.py``'s ``probe_manifest``.
"""

import glob
import json
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402

from Microwave.Gui import mesh_preview  # noqa: E402
from Microwave.Objects import mesh, results  # noqa: E402
from Microwave.Objects import preview as preview_objects  # noqa: E402
from Microwave.Objects.preview import DISPLAY_PROPERTIES  # noqa: E402
from Microwave.Solvers.openems import document  # noqa: E402

#: Filled by the counted wrapper below, and reset before each case.
ASKS = [0]
EXECUTES = [0]


def _instrument():
    real_digest = document.grid_inputs_digest

    def counted(analysis):
        ASKS[0] += 1
        return real_digest(analysis)

    document.grid_inputs_digest = counted

    real_execute = preview_objects.EMMeshPreview.execute

    def watched(self, obj):
        EXECUTES[0] += 1
        return real_execute(self, obj)

    preview_objects.EMMeshPreview.execute = watched


def _case(doc, preview, change):
    """One change, one recompute, and what they cost."""
    ASKS[0] = EXECUTES[0] = 0
    change()
    state = list(preview.State)
    doc.recompute()
    return {
        "asks": ASKS[0],
        "executes": EXECUTES[0],
        "state": state,
        "badge": str(preview.Status),
    }


def _analysis(doc):
    for obj in doc.Objects:
        if type(getattr(obj, "Proxy", None)).__name__ == "EMAnalysis":
            return obj
    return None


def _a_bound_solid(analysis):
    """The first solid a binding points at, or ``None``."""
    for binding in document.contents(analysis).bindings:
        for reference in getattr(binding, "References", ()) or ():
            return reference[0] if isinstance(reference, tuple) else reference
    return None


def _edit_then_nudge(preview, found):
    """Move the mesh policy, then move a slice before anything recomputes."""
    found.settings.ElementsPerWavelength = float(found.settings.ElementsPerWavelength) + 5.0
    _nudge(preview, "SliceY")


def _bump(obj, name):
    """Move one property of one object to some other setting."""
    value = getattr(obj, name)
    kind = obj.getTypeIdOfProperty(name)
    if kind == "App::PropertyBool":
        setattr(obj, name, not value)
    elif kind == "App::PropertyInteger":
        setattr(obj, name, int(value) + 1000)
    elif kind == "App::PropertyEnumeration":
        choices = [c for c in obj.getEnumerationsOfProperty(name) if c != str(value)]
        setattr(obj, name, choices[0])
    elif kind in ("App::PropertyString", "App::PropertyPath"):
        setattr(obj, name, str(value) + "x")
    elif hasattr(value, "Value"):
        setattr(obj, name, float(value.Value) * 1.1 + 0.1)
    else:
        setattr(obj, name, float(value) * 1.1 + 0.1)


def _nudge(preview, name):
    """Move one display property to some other setting."""
    if name == "Display":
        modes = preview.getEnumerationsOfProperty("Display")
        preview.Display = next(mode for mode in modes if mode != str(preview.Display))
    elif name.startswith("ShowSlice"):
        setattr(preview, name, not getattr(preview, name))
    else:
        setattr(preview, name, float(getattr(preview, name)) + 0.5)


def _counted(path):
    doc = FreeCAD.openDocument(path)
    analysis = _analysis(doc)
    if analysis is None:
        FreeCAD.closeDocument(doc.Name)
        return None

    preview, _ = mesh_preview.refresh(analysis)
    doc.recompute()
    found = document.contents(analysis)
    solid = _a_bound_solid(analysis)

    cases = {}
    # The display properties first, while the badge still reads Current: a case
    # that asks nothing can only be told from one that asks and finds nothing
    # by the count, and the badge is what says the drawing still matches.
    for name in sorted(DISPLAY_PROPERTIES):
        cases[name] = _case(doc, preview, partial(_nudge, preview, name))

    # Then everything else a class declares moves no cell, on the same footing:
    # the declaration is what has to keep these off the graph.
    for name in sorted(type(found.solver.Proxy).MOVES_NO_CELL):
        if hasattr(found.solver, name):
            cases[name] = _case(doc, preview, partial(_bump, found.solver, name))
    if found.ports:
        cases["Excitation"] = _case(doc, preview, partial(_bump, found.ports[0], "Excitation"))
    cases["NumFrequencyPoints"] = _case(
        doc, preview, partial(_bump, analysis, "NumFrequencyPoints")
    )

    cases["nothing at all"] = _case(doc, preview, lambda: None)
    # Membership moves the grid and no property announces it, so the study
    # compares what it holds against what the drawing was made from. A result
    # object is what that comparison has to leave alone: a solve puts one in
    # the study, and a mesh reported stale because results arrived is a worse
    # fire than the one the comparison puts out.
    cases["a result arriving"] = _case(
        doc, preview, partial(analysis.addObject, results.createEMSParameters(doc))
    )
    # A material is reached through the binding that names it, wherever the
    # tree keeps it, so tidying one into the study moves no cell. Counting it
    # would leave the two sets unequal for good, and every later change stale.
    material = next((getattr(binding, "Material", None) for binding in found.bindings), None)
    if material is not None:
        cases["a material moved into the study"] = _case(
            doc, preview, partial(analysis.addObject, material)
        )

    # Everything below moves a cell, so the badge stays where they leave it and
    # the cases above have to come first.
    #
    # The first is an edit made while the drawing is being looked at from a
    # different angle. The redraw purges the touched flag, so this is where a
    # purge that swallowed somebody else's touch would show.
    cases["an edit behind a slice nudge"] = _case(
        doc,
        preview,
        partial(_edit_then_nudge, preview, found),
    )
    # The band reaches no link. The study holds the preview in its group, so a
    # link back would close a cycle, and the study marks the drawing itself.
    cases["the frequency band"] = _case(doc, preview, partial(_bump, analysis, "FrequencyStop"))
    if solid is not None:
        cases["a bound solid touched"] = _case(doc, preview, solid.touch)
    cases["the mesh policy"] = _case(
        doc,
        preview,
        lambda: setattr(
            found.settings,
            "ElementsPerWavelength",
            float(found.settings.ElementsPerWavelength) + 5.0,
        ),
    )
    if solid is not None and hasattr(solid, "Length"):
        cases["a bound solid"] = _case(
            doc, preview, lambda: setattr(solid, "Length", float(solid.Length) + 0.1)
        )
    # Last, because it leaves the study holding something the drawing was not
    # made from. A region created after Update Mesh is in no link list, so
    # nothing about it reaches the preview - not the addition, and not one
    # later edit to its element size, across a save and until the next Update
    # Mesh.
    cases["a region arriving"] = _case(
        doc, preview, partial(analysis.addObject, mesh.createEMMeshRegion(doc))
    )
    # And into a subgroup of the study, which is where the study hears
    # _GroupTouched rather than Group. Sorting objects into one is ordinary
    # housekeeping, and both members() and contents() follow it.
    subgroup = doc.addObject("App::DocumentObjectGroup", "Refinements")
    analysis.addObject(subgroup)
    mesh_preview._mark_current(preview)
    cases["a region arriving in a subgroup"] = _case(
        doc, preview, partial(subgroup.addObject, mesh.createEMMeshRegion(doc))
    )

    FreeCAD.closeDocument(doc.Name)
    return cases


#: Property types the sweep has no perturbation for. A link is not moved
#: because moving one rewrites what the study owns, which is a different
#: subject - see the membership note in ``Gui/mesh_preview.py::_link``.
UNSWEPT = ("Link", "XLink", "Expression")


def _tries(obj, name):
    """Every value worth setting this property to, in order.

    An enumeration is offered all of its other choices rather than the first:
    one boundary word leaves the absorber where it was while another moves it,
    and trying one choice reports the property as no input at all.
    """
    value = getattr(obj, name)
    kind = obj.getTypeIdOfProperty(name)
    if any(word in kind for word in UNSWEPT):
        return []
    if kind == "App::PropertyEnumeration":
        return [choice for choice in obj.getEnumerationsOfProperty(name) if choice != str(value)]
    if kind == "App::PropertyBool":
        return [not value]
    if kind == "App::PropertyInteger":
        return [int(value) + 1]
    if kind in ("App::PropertyString", "App::PropertyPath"):
        return [str(value) + "x"]
    if kind == "App::PropertyPlacement":
        moved = value.copy()
        moved.Base.x += 0.01
        return [moved]
    if hasattr(value, "Value"):
        return [float(value.Value) * 1.1 + 0.1]
    if isinstance(value, float):
        return [value * 1.1 + 0.1]
    return []


def _settled(doc, preview):
    doc.recompute()
    preview.Status = preview_objects.CURRENT
    preview.purgeTouched()


def _declared(path):
    """Whether the badge fires wherever the key moves, property by property.

    The badge and the key are two answers to one question, and the badge is now
    the one the tree shows. It is allowed to be louder than the key. It is not
    allowed to be quieter, and this is what says so.
    """
    doc = FreeCAD.openDocument(path)
    analysis = _analysis(doc)
    if analysis is None:
        FreeCAD.closeDocument(doc.Name)
        return None

    preview, _ = mesh_preview.refresh(analysis)
    found = document.contents(analysis)
    owned = [
        ("study", analysis),
        ("solver", found.solver),
        ("policy", found.settings),
        *[("binding", obj) for obj in found.bindings],
        *[("port", obj) for obj in found.ports],
        *[("region", obj) for obj in found.refinements],
    ]

    swept = {}
    for kind, obj in owned:
        for name in list(obj.PropertiesList):
            try:
                tries = _tries(obj, name)
            except Exception:
                tries = []
            if not tries:
                continue
            before = getattr(obj, name)
            moved = False
            badges = set()
            for new in tries:
                _settled(doc, preview)
                key = document.grid_inputs_digest(analysis)
                try:
                    setattr(obj, name, new)
                except Exception:
                    continue
                doc.recompute()
                badges.add(str(preview.Status))
                try:
                    moved = moved or document.grid_inputs_digest(analysis) != key
                except Exception:
                    moved = True
                try:
                    setattr(obj, name, before)
                except Exception:
                    pass
            entry = swept.setdefault(f"{kind}.{name}", {"grid": False, "badges": []})
            entry["grid"] = entry["grid"] or moved
            entry["badges"] = sorted(set(entry["badges"]) | badges)
    FreeCAD.closeDocument(doc.Name)
    return swept


def main():
    out = os.environ["PREVIEW_RECOMPUTE_OUT"]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _instrument()

    counted = {}
    declared = {}
    for path in sorted(glob.glob(os.path.join(here, "examples", "*.FCStd"))):
        cases = _counted(path)
        if cases is not None:
            counted[os.path.basename(path)] = cases
        swept = _declared(path)
        if swept is not None:
            declared[os.path.basename(path)] = swept

    with open(os.path.join(out, "manifest.json"), "w") as manifest:
        json.dump({"counted": counted, "declared": declared}, manifest)


if __name__ in ("__main__", "preview_recompute_probe"):
    main()
