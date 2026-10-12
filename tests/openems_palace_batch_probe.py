# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Solve the shipped WR-42 guide without the GUI, from a script of one's own,
and write down what came of it.

Run by ``tests/test_openems_palace_batch.py`` under ``freecadcmd``, one case to
a process, the case named by ``BATCH_CASE``::

    BATCH_OUT=tests/_batch BATCH_CASE=prepare freecadcmd tests/openems_palace_batch_probe.py

``prepare``
    copies the guide twice, and writes it twice more: once with the guide's box drawn
    with no length, which does not recompute, and once with the openEMS
    solver's directory under a file, which cannot be made.
``raised``
    solves the copy on Palace and then raises, as a script of a user's might,
    so that ``freecadcmd`` runs it a second time.
``changed``
    solves the copy on openEMS and writes to its file while the run is filed,
    as another program might.
``reopened``
    opens ``BATCH_REOPENED`` with FreeCAD's main window up and reads it back.

``manifest.json`` names what was written and is written last - see
``tests/conftest.py``'s ``probe_manifest``. In ``raised`` it is written on each
pass, before the probe raises.
"""

import json
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402

from Microwave import batch  # noqa: E402

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "waveguide_wr42.FCStd"
)

#: What ``prepare`` writes, beside the manifest.
GUIDE = "guide.FCStd"
MEDDLED = "meddled.FCStd"
BROKEN = "broken.FCStd"
ELSEWHERE = "elsewhere.FCStd"


def visibility(path):
    """Each object's visibility as the GUI entries of the file at ``path`` hold it."""
    with zipfile.ZipFile(path) as read:
        if batch.GUI_DOCUMENT not in read.namelist():
            return {}
        text = read.read(batch.GUI_DOCUMENT).decode("utf-8")
    import xml.etree.ElementTree as tree

    found = {}
    for provider in tree.fromstring(text).iter("ViewProvider"):
        for prop in provider.iter("Property"):
            if prop.get("name") == "Visibility":
                found[provider.get("name")] = prop.find("Bool").get("value") == "true"
    return found


def prepare(out):
    from Microwave.Solvers.openems import run as openems_run
    from tests.palace_waveguide_probe import missing

    absent = missing()
    if not absent:
        try:
            openems_run.find_interpreter()
        except openems_run.EngineNotFound as error:
            absent = str(error)
    if absent:
        return {"missing": absent}
    guide = shutil.copy(EXAMPLE, os.path.join(out, GUIDE))
    meddled = shutil.copy(EXAMPLE, os.path.join(out, MEDDLED))
    doc = FreeCAD.openDocument(EXAMPLE)
    doc.GuideAir.Length = 0
    doc.saveAs(os.path.join(out, BROKEN))
    FreeCAD.closeDocument(doc.Name)
    doc = FreeCAD.openDocument(EXAMPLE)
    blocking = os.path.join(out, "a file")
    with open(blocking, "w") as stream:
        stream.write("in the way")
    doc.EMSolverOpenEMS.RunDirectory = os.path.join(blocking, "runs")
    doc.saveAs(os.path.join(out, ELSEWHERE))
    FreeCAD.closeDocument(doc.Name)
    return {
        "guide": guide,
        "meddled": meddled,
        "broken": os.path.join(out, BROKEN),
        "elsewhere": os.path.join(out, ELSEWHERE),
        "visibility": visibility(EXAMPLE),
    }


def raised(out, guide):
    """Solve, write down how it ended, then raise."""
    record = os.path.join(out, "statuses.json")
    statuses = []
    if os.path.exists(record):
        with open(record) as stream:
            statuses = json.load(stream)
    said = []
    printed = batch.say
    batch.say = lambda line: said.append(line) or printed(line)
    statuses.append([int(batch.solve(guide, solver="Palace")), said])
    batch.say = printed
    with open(record, "w") as stream:
        json.dump(statuses, stream)
    write(out, {"statuses": statuses})
    raise RuntimeError("a script of the user's fails after the solve")


def changed(out, guide):
    original = batch.OpenEMSRun.collect_results

    def collect_and_meddle(self, paths):
        filed = original(self, paths)
        with open(guide, "ab") as stream:
            stream.write(b"written by another program")
        return filed

    batch.OpenEMSRun.collect_results = collect_and_meddle
    status = batch.solve(guide, solver="openEMS")
    folder = os.path.dirname(guide)
    beside = sorted(name for name in os.listdir(folder) if ".run" in name)
    with open(guide, "rb") as stream:
        tail = stream.read()[-len(b"written by another program") :]
    return {
        "status": int(status),
        "beside": [os.path.join(folder, name) for name in beside],
        "tail": tail.decode(),
    }


def reopened(path):
    import FreeCADGui

    FreeCADGui.showMainWindow()
    from Microwave.Objects.analysis import analysis_of
    from Microwave.Objects.kinds import kind_of

    doc = FreeCAD.openDocument(path)
    objects = {}
    for obj in doc.Objects:
        study = analysis_of(obj)
        view = obj.ViewObject
        objects[obj.Name] = {
            "kind": kind_of(obj) or obj.TypeId,
            "label": obj.Label,
            "label2": obj.Label2,
            "study": study.Label if study is not None else "",
            "visible": bool(view.Visibility) if view is not None else None,
            "provider": type(getattr(view, "Proxy", None)).__name__,
        }
    with zipfile.ZipFile(path) as read:
        names = read.namelist()
    return {"objects": objects, "entries": names}


def write(out, answer):
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle, indent=2, default=str)


def main(out):
    os.makedirs(out, exist_ok=True)
    case = os.environ["BATCH_CASE"]
    if case == "raised":
        raised(out, os.environ["BATCH_GUIDE"])
    elif case == "changed":
        write(out, changed(out, os.environ["BATCH_GUIDE"]))
    elif case == "reopened":
        write(out, reopened(os.environ["BATCH_REOPENED"]))
    else:
        write(out, prepare(out))


main(os.environ["BATCH_OUT"])
