# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A study made to look as an earlier build saved it, saved, opened again and
translated by both backends, and every kind of object as this build makes it,
saved and opened again.

Run by ``test_openems_palace_declared`` under ``freecadcmd``::

    DECLARED_OUT=tests/_declared freecadcmd tests/openems_palace_declared_probe.py

``examples/waveguide_wr42.py``'s guide is drawn twice, each answered by a study
holding both solvers. One is left as this build makes it. The other has a
property taken off, one put on and an enumeration given the choices it had
before a choice was added, on the policy, a port and a material, as a document
an earlier build wrote comes back. What each backend's translation says of each
after a save and a reopen is handed over. So is a study of this build whose
policy carries a property the user added, what each kind of object departs in
once saved and opened again, and which document is open and active, and how
many steps its undo holds, before and after the declarations are read.

``manifest.json`` is written last, so its existence rather than the exit status
is what says this finished.
"""

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "examples"))

import FreeCAD  # noqa: E402

from Microwave import Objects  # noqa: E402
from Microwave.Objects.analysis import members  # noqa: E402
from Microwave.Objects.declared import declarations, departures  # noqa: E402
from Microwave.Objects.kinds import classes, kind_of  # noqa: E402
from Microwave.Objects.mesh import createEMGmshMesh  # noqa: E402
from Microwave.Objects.solver import createEMSolverPalace  # noqa: E402
from Microwave.Solvers.openems import document as openems  # noqa: E402
from Microwave.Solvers.palace import document as palace  # noqa: E402


def _example():
    """``examples/waveguide_wr42.py``, loaded under a name of its own.

    Imported by its own name it builds and saves the example document, because
    its guard fires for the name ``freecadcmd`` runs it under.
    """
    spec = importlib.util.spec_from_file_location(
        "wr42_declared", os.path.join(HERE, "examples", "waveguide_wr42.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WR42 = _example()


def drawn(name):
    doc = FreeCAD.newDocument(name)
    air, planes = WR42.geometry(doc)
    analysis = WR42.study(doc)
    WR42.markup(analysis, air, planes)
    analysis.addObject(createEMSolverPalace(doc))
    analysis.addObject(createEMGmshMesh(doc))
    doc.recompute()
    return analysis


def aged(analysis):
    """What an earlier build's file holds, on the policy, a port and a material."""
    found = members(analysis)
    policy = next(obj for obj in found if kind_of(obj) == "EMMeshPolicy")
    port = next(obj for obj in found if kind_of(obj).startswith("EMPort"))
    binding = next(obj for obj in found if kind_of(obj) == "EMMaterialBinding")
    policy.removeProperty("Medium")
    policy.addProperty("App::PropertyFloat", "ElementsPerWavelength", "Mesh")
    policy.ElementsPerWavelength = 30.0
    policy.PaddingXMin = ["Air", "Through"]
    port.removeProperty("ReferenceDepth")
    port.addProperty("App::PropertyLength", "Length", "Waveguide")
    port.Length = 2.5
    binding.Material.removeProperty("DispersionFrequency")
    return {
        "policy": policy.Label,
        "port": port.Label,
        "material": binding.Material.Label,
    }


def added_by_hand(analysis):
    """Properties the user added in the property editor: a note on the policy,
    and a link on a binding to a material an earlier build saved, which no
    binding binds."""
    found = members(analysis)
    policy = next(obj for obj in found if kind_of(obj) == "EMMeshPolicy")
    policy.addProperty("App::PropertyString", "Note", "Base")
    policy.Note = "measured on the bench"
    binding = next(obj for obj in found if kind_of(obj) == "EMMaterialBinding")
    old = Objects.createEMMaterial("Datasheet", doc=analysis.Document)
    old.removeProperty("DispersionFrequency")
    binding.addProperty("App::PropertyLink", "Datasheet", "Base")
    binding.Datasheet = old


def every_kind(out):
    """What each kind, as its create function makes it, departs in once saved
    and opened again."""
    doc = FreeCAD.newDocument("kinds")
    names = {kind: getattr(Objects, f"create{kind}")(doc=doc).Name for kind in classes()}
    path = os.path.join(out, "kinds.FCStd")
    doc.saveAs(path)
    FreeCAD.closeDocument(doc.Name)
    doc = FreeCAD.openDocument(path)
    found = {kind: list(departures(doc.getObject(name)).phrases) for kind, name in names.items()}
    FreeCAD.closeDocument(doc.Name)
    return found


def state():
    active = FreeCAD.ActiveDocument
    return {
        "open": sorted(FreeCAD.listDocuments()),
        "active": active.Name if active else None,
        "undo": active.UndoCount if active else None,
    }


def reopened(analysis, out):
    path = os.path.join(out, f"{analysis.Document.Name}.FCStd")
    name = analysis.Name
    analysis.Document.saveAs(path)
    FreeCAD.closeDocument(analysis.Document.Name)
    return FreeCAD.openDocument(path).getObject(name)


def said(translate, analysis):
    try:
        translate(analysis)
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    answer = {}
    current, old, own = drawn("current"), drawn("aged"), drawn("own")
    current.Document.UndoMode = 1
    current.Document.openTransaction("An edit")
    current.Label = "Edited"
    current.Document.commitTransaction()
    FreeCAD.setActiveDocument(current.Document.Name)
    answer["made"] = sorted(analysis.Document.Name for analysis in (current, old, own))
    answer["before"] = state()
    declarations.cache_clear()
    declarations()
    answer["after"] = state()
    answer["kinds"] = every_kind(OUT)
    answer["labels"] = aged(old)
    added_by_hand(own)
    for name, analysis in (("current", current), ("aged", old), ("own", own)):
        analysis = reopened(analysis, OUT)
        answer[name] = {
            "openEMS": said(openems.problem, analysis),
            "Palace": said(palace.problem, analysis),
        }
    with open(os.path.join(OUT, "manifest.json"), "w") as handle:
        json.dump(answer, handle, indent=2)


OUT = os.environ.get("DECLARED_OUT", ".")

# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "openems_palace_declared_probe"):
    main()
