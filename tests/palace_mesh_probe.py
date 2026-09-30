# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Put the mesh of a WR-42 guide in its document, as a Mesh on Palace does.

Run by ``test_palace_mesh_shown`` under ``freecadcmd``::

    PALACE_MESH_OUT=tests/_palace_mesh freecadcmd tests/palace_mesh_probe.py

A real FreeCAD because every question here is FreeCAD's: whether its reader
takes the copy the mesher wrote, whether the object is drawn and can be hidden,
what undo does to it, and what it costs a saved file. The main window is shown,
because a view provider exists only with one.

The guide is the Palace gate's own, drawn as two bodies, so the labels are the
bindings, the ports and the wall - and the first port is renamed to something
the copy's writer would rewrite.

It meshes at the first of ``SIZES`` and again at the second, so the second mesh
is a different mesh replacing the first, and it hides the first before the
second arrives. The document is saved with the second mesh hidden, and
``palace_mesh_reopen_probe.py`` opens it in a FreeCAD of its own.

Palace is never started. The stage that prepares a run finds one all the same,
so a machine without it skips, as the gate does.

``manifest.json`` is written last - see ``tests/conftest.py``'s
``probe_manifest``.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402
import FreeCADGui  # noqa: E402

from Microwave.Gui import fem_mesh  # noqa: E402
from Microwave.Objects.mesh import createEMMeshRegion  # noqa: E402
from Microwave.Objects.results import SOLVED_ON, createEMSParameters  # noqa: E402
from Microwave.Objects.results import label as result_label  # noqa: E402
from Microwave.Solvers.palace import document, pipeline  # noqa: E402
from Microwave.Solvers.palace.capabilities import capabilities  # noqa: E402
from tests.palace_waveguide_probe import build, missing  # noqa: E402

#: What the first port is called: a space and brackets, which the UNV writer
#: rewrites and a document keeps.
RENAMED = "Port 1 (in)"

#: The element sizes, as elements per wavelength. Far enough apart that the
#: meshes differ.
SIZES = (4, 7)

#: The document, beside the manifest.
SAVED = "guide.FCStd"

#: The document :func:`marks` saves, beside it.
MARKS = "marks.FCStd"


def groups(shown):
    """Every group in a mesh, as its name and how many members it has."""
    return {shown.getGroupName(group): len(shown.getGroupElements(group)) for group in shown.Groups}


def meshed(analysis, out, size):
    """Mesh the study at ``size`` and put the mesh in it, as the panel does."""
    document.contents(analysis).recipe.ElementsPerWavelength = size
    analysis.Document.recompute()
    key, linked = fem_mesh.inputs(analysis), fem_mesh.built_from(analysis)
    prepared = pipeline.prepare(analysis, os.path.join(out, f"run{size}"))
    made = pipeline.meshed(prepared, numbered_as=fem_mesh.FORMAT)
    shown, _ = fem_mesh.record(
        analysis,
        capabilities().solver,
        fem_mesh.read(made),
        key=key,
        identity=fem_mesh.identity(made.path),
        linked=linked,
    )
    return made, shown


def mark(analysis):
    """What the study's mesh says of itself, and what the panel would say."""
    found = fem_mesh.find(analysis, capabilities().solver)
    return {
        "status": str(found.Status),
        "label": found.Label,
        "says": fem_mesh.staleness(analysis, capabilities().solver),
    }


def edited(doc, change):
    """Make ``change`` as the property editor does: in a step of its own, then
    a recompute."""
    doc.openTransaction("Edit")
    change()
    doc.recompute()
    doc.commitTransaction()


def marks(out):
    """Every kind of edit, on a study meshed afresh before each that marks it,
    and what the mesh says after it. The document is saved with a mesh that
    matches, for the reopening probe to edit."""
    doc = FreeCAD.newDocument("palace_marks")
    analysis, _ = build(doc)
    # A body nothing binds, which a region alone names.
    refined = doc.addObject("Part::Box", "Refined")
    refined.Length = refined.Width = refined.Height = 1.0
    refined.Placement.Base = FreeCAD.Vector(5.0, 2.0, 1.0)
    region = createEMMeshRegion(doc)
    region.References = [(refined, [""])]
    region.ElementSize = 1.0
    analysis.addObject(region)
    doc.recompute()
    doc.UndoMode = 1
    body = doc.getObject("GuideAir1")
    port = next(obj for obj in doc.Objects if obj.Label == "Port1")
    size = SIZES[0]
    answer = {}

    meshed(analysis, out, size)
    answer["made"] = mark(analysis)
    edited(doc, lambda: setattr(port, "Excitation", not port.Excitation))
    answer["excitation"] = mark(analysis)
    edited(doc, lambda: setattr(analysis, "NumFrequencyPoints", analysis.NumFrequencyPoints + 1))
    answer["points"] = mark(analysis)
    doc.openTransaction("Edit")
    body.Height = float(body.Height) * 1.01
    answer["body before a recompute"] = mark(analysis)
    doc.recompute()
    doc.commitTransaction()
    answer["body"] = mark(analysis)
    doc.undo()
    answer["body undone"] = mark(analysis)
    doc.recompute()
    answer["body undone, recomputed"] = mark(analysis)

    for kind, change in (
        ("band", lambda: setattr(analysis, "FrequencyStart", float(analysis.FrequencyStart) * 0.9)),
        ("recipe", lambda: setattr(document.contents(analysis).recipe, "MaxGrowthRatio", 1.5)),
        ("region's body", lambda: setattr(refined, "Height", 1.5)),
    ):
        meshed(analysis, out, size)
        edited(doc, change)
        answer[kind] = mark(analysis)

    meshed(analysis, out, size)
    answer["saved"] = mark(analysis)
    path = os.path.join(out, MARKS)
    doc.saveAs(path)
    answer["document"] = path
    # After the save, since a study of one port is not the one reopened.
    edited(doc, lambda: analysis.removeObject(port))
    answer["port out"] = mark(analysis)

    # A second study in the document, whose mesh FreeCAD labels with a number.
    other, _ = build(doc)
    meshed(other, out, size)
    answer["second study"] = mark(other)
    second_body = next(
        member
        for member in doc.Objects
        if member.TypeId == "Part::Box"
        and member.Name.startswith("GuideAir")
        and member is not body
    )
    edited(doc, lambda: setattr(second_body, "Height", float(second_body.Height) * 1.01))
    answer["second study's body"] = mark(other)

    # A result solved on the second study's mesh, and that mesh then deleted.
    result = createEMSParameters(doc)
    shown = fem_mesh.find(other, capabilities().solver)
    solved_on = {"solver": capabilities().solver, SOLVED_ON: shown.Identity}
    result.Provenance = json.dumps(solved_on)
    result.Label = result_label(capabilities().solver)
    other.addObject(result)
    doc.recompute()
    answer["result beside its mesh"] = result.Label
    doc.removeObject(shown.Name)
    answer["result, its mesh deleted"] = result.Label
    return answer


def seen(obj):
    """What a user sees of the object."""
    view = obj.ViewObject
    return {
        "type": obj.TypeId,
        "label": obj.Label,
        "provider": type(view).__name__ if view is not None else None,
        "modes": list(view.listDisplayModes()) if view is not None else [],
        "visible": bool(view.Visibility) if view is not None else None,
        "volumes": obj.FemMesh.VolumeCount,
        "groups": groups(obj.FemMesh),
    }


def main(out):
    FreeCADGui.showMainWindow()
    os.makedirs(out, exist_ok=True)
    answer = {}
    absent = missing()
    if absent:
        answer["missing"] = absent
    else:
        doc = FreeCAD.newDocument("palace_mesh")
        analysis, _ = build(doc, cut=True)
        port = next(obj for obj in doc.Objects if obj.Label == "Port1")
        port.Label = RENAMED
        path = os.path.join(out, SAVED)
        doc.saveAs(path)
        bare = os.path.getsize(path)
        doc.UndoMode = 1

        first, shown = meshed(analysis, out, SIZES[0])
        answer["labels"] = sorted(first.labels)
        answer["first"] = seen(shown)
        view = shown.ViewObject
        view.Visibility = False
        answer["hidden"] = bool(view.Visibility)
        view.Visibility = True
        answer["shown again"] = bool(view.Visibility)
        view.Visibility = False

        second, shown = meshed(analysis, out, SIZES[1])
        answer["second"] = seen(shown)
        answer["objects"] = sum(obj.TypeId == fem_mesh.TYPE for obj in doc.Objects)
        # The study still translates, with FreeCAD's object in it.
        answer["translates"] = bool(document.problem(analysis).regions)

        answer["trace"] = [mark(analysis)]
        doc.undo()
        answer["undone"] = fem_mesh.find(analysis, capabilities().solver).FemMesh.VolumeCount
        answer["trace"].append(mark(analysis))
        doc.redo()
        answer["redone"] = fem_mesh.find(analysis, capabilities().solver).FemMesh.VolumeCount
        answer["trace"].append(mark(analysis))

        doc.save()
        answer["trace"].append(mark(analysis))
        answer["bytes"] = os.path.getsize(path) - bare
        final = fem_mesh.find(analysis, capabilities().solver).FemMesh
        answer["elements"] = final.VolumeCount + final.FaceCount
        answer["document"] = path
        answer["marks"] = marks(out)

    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle, indent=2)


# Guarded, since the reopening probe imports what a user sees of the object
# from here, and freecadcmd runs a script under its own name rather than as
# __main__.
if "PALACE_MESH_OUT" in os.environ:
    main(os.environ["PALACE_MESH_OUT"])
