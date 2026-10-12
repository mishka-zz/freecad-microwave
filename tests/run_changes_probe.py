# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""File a run after each kind of edit a user makes to a study, and write down
what each run says changed.

Run by ``tests/test_run_changes_reopen.py`` under ``freecadcmd``::

    RUN_CHANGES_OUT=tests/_run_changes/made freecadcmd tests/run_changes_probe.py
    RUN_CHANGES_OUT=tests/_run_changes/reopened \\
        RUN_CHANGES_REOPENED=tests/_run_changes/made/notch.FCStd \\
        freecadcmd tests/run_changes_probe.py

The first pass opens a copy of ``examples/stub_notch.FCStd`` with FreeCAD's main
window up. It drives the stub's width and the substrate's permittivity from a
spreadsheet and the trace's width from a variable set through expressions. It
adds a via drawn as a plain shape and a pad attached to the substrate's top,
with the pad's length computed in the spreadsheet from the stub's width by the
cell's address and its width from an area typed with a unit of symbols, and
the via's position and the band's stop read from the spreadsheet too. It files
a run without a solve, then makes one edit of each kind and files a run after
each: the spreadsheet's cell, the variable set's value, the expression itself,
a material's constant set directly and from a cell, a port's reference, a
solid's own dimension, a solid's placement, the via's shape drawn anew, a solid
hidden, the band's stop, the pad's attachment offset, the via's position, and
the band's start, a solid renamed, the via drawn anew and moved at once, and
the area. A
variable set holds a value to every figure a double holds, which the pad's
height reads. It then saves the document. The second, in a FreeCAD of its
own, opens what the first saved and reads the runs back.

``manifest.json`` names what was written and is written last - see
``tests/conftest.py``'s ``probe_manifest``.
"""

import json
import os
import shutil
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCADGui  # noqa: E402

FreeCADGui.showMainWindow()

import FreeCAD  # noqa: E402
import numpy as np  # noqa: E402
import Part  # noqa: E402

from Microwave.Gui import results as glue  # noqa: E402
from Microwave.Gui.described import drawn  # noqa: E402
from Microwave.Gui.fem_mesh import inputs as mesh_inputs  # noqa: E402
from Microwave.Gui.plot_s_params import overlay_chart  # noqa: E402
from Microwave.Objects.analysis import analyses  # noqa: E402
from Microwave.Objects.declared import departures  # noqa: E402
from Microwave.Objects.runs import snapshot as snapshot_of  # noqa: E402
from Microwave.Results import overlay, snapshots  # noqa: E402
from Microwave.Results.sparameters import SParameters  # noqa: E402

#: The property FreeCAD caches an object's shape in once asked for it
#: (``src/Mod/Part/App/PropertyTopoShape.cpp``).
CACHE = "_Part_ShapeCache"

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "stub_notch.FCStd"
)

SAVED = "notch.FCStd"


def solved(solver):
    """A matrix as a run would file it."""
    frequency = np.linspace(1e9, 9e9, 9)
    s = np.full((9, 2, 2), 0.1 + 0.05j)
    z = np.full((9, 2), 50.0 + 0j)
    return SParameters(
        frequency=frequency,
        s=s,
        port_numbers=(1, 2),
        reference=z,
        measured_impedance=z.copy(),
        provenance={"solver": solver},
    )


def filed(study, solver="openEMS"):
    """File a run of ``study`` as it stands, and return it."""
    return glue.run_of(glue.record(study, solved(solver)))


def painted(window):
    """A picture of ``window`` once Qt has laid it out and painted it."""
    from PySide import QtWidgets

    application = QtWidgets.QApplication.instance()
    for _ in range(50):
        application.processEvents()
    return window.grab()


def axes_with_a_name(name):
    """Where the axes stand in a plot tab drawing two curves, one of them
    called ``name``, as a fraction of the figure."""
    from Microwave.Gui import charts

    x = np.linspace(1.0, 2.0, 11)
    window = charts.render(
        charts.Chart(
            series=(charts.Series(x, -x, name), charts.Series(x, -2 * x, "S21")),
            x_label="Frequency (GHz)",
            y_label="Magnitude (dB)",
            heading="S11",
        ),
        name[:20],
    )
    window.resize(1000, 800)
    window.fig.canvas.draw()
    return list(window.axes.get_position().bounds)


def said(run):
    return {"label": run.Label, "description": run.Label2, "changes": list(run.Changes)}


def shown(tables, out, picture="overlay.png"):
    """The overlay drawn in FreeCAD's own plot tab: what its table holds, the
    colour each row is marked with and the colour of each curve, which of its
    tabs holds what, and a picture of each tab."""
    from Microwave.Gui.plot_s_params import show_overlay

    window = show_overlay(tables)
    held = window.table
    window.resize(1000, 800)
    painted(window).save(os.path.join(out, picture))
    window.tabs.setCurrentIndex(1)
    painted(window).save(os.path.join(out, "details-" + picture))
    window.tabs.setCurrentIndex(0)
    plot, details = window.tabs.widget(0), window.tabs.widget(1)
    legend = window.axes.get_legend()
    return {
        "tabs": [window.tabs.tabText(i) for i in range(window.tabs.count())],
        "plot on the first": plot.isAncestorOf(window.canvas)
        and plot.isAncestorOf(window.mpl_toolbar),
        "table and notes on the second": details.isAncestorOf(held)
        and details.isAncestorOf(window.notes),
        "notes over the table": window.details.indexOf(window.notes) < window.details.indexOf(held),
        "notes": window.notes.text(),
        "legend laid out": legend is not None and legend.get_in_layout(),
        "headings": [held.horizontalHeaderItem(i).text() for i in range(held.columnCount())],
        "rows": [
            [held.item(r, c).text() for c in range(1, held.columnCount())]
            for r in range(held.rowCount())
        ],
        "marks": [held.item(r, 0).background().color().name() for r in range(held.rowCount())],
        "curves": [line.line.get_color() for line in window.series],
        "every row in view": held.verticalScrollBar().maximum() == 0,
        "every heading whole": all(
            held.columnWidth(column)
            > held.horizontalHeader()
            .fontMetrics()
            .horizontalAdvance(held.horizontalHeaderItem(column).text())
            for column in range(1, held.columnCount())
        ),
    }


def warned(tables):
    """The overlay drawn in FreeCAD's own plot tab, its difference chosen
    through its own selector and then taken off again with the warnings tab
    open: each tab's title, whether the warnings tab is shown, what it says,
    and which tab is open once it is hidden."""
    from PySide import QtWidgets

    from Microwave.Gui.plot_s_params import NO_DIFFERENCE, show_overlay

    window = show_overlay(tables, chosen=(2, 1))
    tabs = window.tabs
    [against] = [
        box
        for box in window.findChildren(QtWidgets.QComboBox)
        if box.count() and box.itemText(0) == NO_DIFFERENCE
    ]

    def read():
        return {
            "tabs": [tabs.tabText(i) for i in range(tabs.count())],
            "shown": [tabs.isTabVisible(i) for i in range(tabs.count())],
            "warnings": window.warnings.text(),
            "heading": window.fig._suptitle.get_text(),
            "curves": [line.name for line in window.series],
        }

    found = {"opened": read()}
    against.setCurrentIndex(1)
    found["differenced"] = read()
    tabs.setCurrentIndex(window.warnings_tab)
    against.setCurrentIndex(0)
    found["taken off"] = read()
    found["open once hidden"] = tabs.currentIndex()
    return found


def first(out):
    document = FreeCAD.openDocument(shutil.copy(EXAMPLE, os.path.join(out, SAVED)))
    [study] = analyses(document)
    get = document.getObject
    stub, trace, board = get("Stub"), get("Trace"), get("Substrate")

    sheet = document.addObject("Spreadsheet::Sheet", "Params")
    sheet.set("A1", "StubLen")
    sheet.set("B1", f"={stub.Width.Value} mm")
    sheet.setAlias("B1", "StubLen")
    variables = document.addObject("App::VarSet", "Vars")
    variables.addProperty("App::PropertyLength", "TraceW", "Base")
    variables.TraceW = trace.Width
    variables.addProperty("App::PropertyFloat", "Ratio", "Base")
    # Every figure a double holds, which a file written and read back does not.
    variables.Ratio = 0.1 + 0.2
    sheet.set("A2", "Er")
    sheet.set("B2", str(get("FR4").Permittivity))
    sheet.setAlias("B2", "Er")
    sheet.set("A3", "Half")
    sheet.set("B3", "=B1 / 2")
    sheet.setAlias("B3", "Half")
    sheet.set("A4", "Fmax")
    sheet.set("B4", f"={study.FrequencyStop.Value / 1e9} GHz")
    sheet.setAlias("B4", "Fmax")
    sheet.set("A6", "Area")
    sheet.set("B6", "=4 (mm ^ 2)")
    sheet.setAlias("B6", "Area")
    sheet.set("A5", "ViaX")
    sheet.set("B5", "=0 mm")
    sheet.setAlias("B5", "ViaX")
    study.setExpression("FrequencyStop", "Params.Fmax")
    stub.setExpression("Width", "Params.StubLen")
    trace.setExpression("Width", "Vars.TraceW")
    get("FR4").setExpression("Permittivity", "Params.Er")
    via = document.addObject("Part::Feature", "Via")
    via.Shape = Part.makeCylinder(0.3, 1.5)
    pad = document.addObject("Part::Box", "Pad")
    pad.Length = pad.Width = pad.Height = 0.5
    pad.AttachmentSupport = [(board, "Face6")]
    pad.MapMode = "FlatFace"
    pad.setExpression("Length", "Params.Half")
    pad.setExpression("Height", "Vars.Ratio * 1 mm")
    pad.setExpression("Width", "Params.Area / 8 mm")
    via.setExpression(".Placement.Base.x", "Params.ViaX")
    binding = get("TraceBinding")
    binding.References = [*binding.References, (via, ("",)), (pad, ("",))]
    document.recompute()

    runs = [filed(study)]
    edits = {
        "spreadsheet": lambda: sheet.set("B1", f"={stub.Width.Value + 0.5} mm"),
        "variable set": lambda: setattr(variables, "TraceW", trace.Width.Value + 0.25),
        "expression": lambda: stub.setExpression("Width", "Params.StubLen * 2"),
        "material": lambda: setattr(get("FR4"), "LossTangent", get("FR4").LossTangent * 2),
        "material from a cell": lambda: sheet.set("B2", str(get("FR4").Permittivity + 0.1)),
        "port": lambda: setattr(get("Port1"), "ReferenceImpedance", 75.0),
        "dimension": lambda: setattr(board, "Height", board.Height.Value + 0.2),
        "placement": lambda: setattr(
            stub,
            "Placement",
            stub.Placement.multiply(
                FreeCAD.Placement(FreeCAD.Vector(0.5, 0, 0), FreeCAD.Rotation())
            ),
        ),
        "drawn anew": lambda: setattr(via, "Shape", Part.makeCylinder(0.4, 1.5)),
        "hidden": lambda: setattr(board, "Visibility", False),
        "band": lambda: sheet.set("B4", f"={study.FrequencyStop.Value / 1e9 + 1} GHz"),
        "offset": lambda: setattr(
            pad,
            "AttachmentOffset",
            FreeCAD.Placement(FreeCAD.Vector(0, 0, 0.1), FreeCAD.Rotation()),
        ),
        "bound placement": lambda: sheet.set("B5", "=0.25 mm"),
        "band start": lambda: setattr(study, "FrequencyStart", study.FrequencyStart.Value + 0.5e9),
        "renamed": lambda: setattr(trace, "Label", "Trace line"),
        "typed area": lambda: sheet.set("B6", "=5 (mm ^ 2)"),
        "drawn anew and moved": lambda: (
            setattr(via, "Shape", Part.makeCylinder(0.3, 1.5)),
            setattr(
                via, "Placement", FreeCAD.Placement(FreeCAD.Vector(0, 1, 0), FreeCAD.Rotation())
            ),
        ),
    }
    answer = {"edits": {}}
    for name, edit in edits.items():
        edit()
        document.recompute()
        runs.append(filed(study, "Palace" if name == "port" else "openEMS"))
        answer["edits"][name] = said(runs[-1])
    answer["first"] = said(runs[0])
    answer["study"] = study.Label
    # FreeCAD caches the shape of a study it was asked for in a property of its
    # own, and two runs filed one after the other find it holding different
    # values. The mesh's inputs are read before and after.
    inputs_before = mesh_inputs(study)
    for obj in document.Objects:
        Part.getShape(obj)
    answer["mesh inputs moved"] = mesh_inputs(study) != inputs_before
    answer["cached"] = [obj.Label for obj in document.Objects if CACHE in obj.PropertiesList]
    answer["departs"] = [obj.Label for obj in [study, *study.Group] if departures(obj).phrases]
    filed(study)
    answer["unchanged"] = said(filed(study))

    drawn_over = overlay_chart(glue.overlaid(runs[:3]), overlay.DB, (2, 1))
    answer["legend"] = [series.name for series in drawn_over.series]
    answer["notes"] = list(drawn_over.notes)
    answer["shown"] = shown(glue.overlaid(runs[:3]), out)
    answer["warned"] = warned(glue.overlaid(runs[:3]))
    answer["shown far apart"] = shown(glue.overlaid([runs[0], runs[-1]]), out, "far.png")
    answer["axes"] = {
        "short": axes_with_a_name("S11, Run 1 (openEMS)"),
        "long": axes_with_a_name("S11, Run 1 (openEMS), " + "Substrate Height 1.6 mm, " * 12),
    }

    sketch = document.addObject("Sketcher::SketchObject", "Outline")
    import Sketcher

    sketch.addGeometry(Part.LineSegment(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(4, 0, 0)))
    sketch.renameConstraint(
        sketch.addConstraint(Sketcher.Constraint("DistanceX", 0, 1, 0, 2, 4.0)), "Reach"
    )
    sketch.setExpression("Constraints.Reach", "Vars.TraceW * 2")
    tilt = sketch.addConstraint(Sketcher.Constraint("Angle", 0, 1.0))
    sketch.renameConstraint(tilt, "Tilt")
    sketch.setDatum(tilt, FreeCAD.Units.Quantity("30 deg"))
    sketch.addGeometry(Part.LineSegment(FreeCAD.Vector(0, 1, 0), FreeCAD.Vector(3, 1, 0)))
    measured = sketch.addConstraint(Sketcher.Constraint("DistanceX", 1, 1, 1, 2, 3.0))
    sketch.setDriving(measured, False)
    sketch.renameConstraint(measured, "Measured")
    document.recompute()
    kept = drawn(sketch)
    answer["sketch"] = {
        "values": {k: v for k, v in kept["values"].items() if k.startswith("Constraints.")},
        "units": {k: v for k, v in kept["units"].items() if k.startswith("Constraints.")},
        "expressions": kept["expressions"],
        "kept": sorted(kept["values"]),
    }
    document.removeObject("Outline")
    document.recompute()

    answer["snapshot now"] = glue.snapshot(study)
    document.save()
    answer["document"] = document.FileName
    return answer


def reopened(path):
    document = FreeCAD.openDocument(path)
    [study] = analyses(document)
    runs = sorted(glue.runs_of(study), key=lambda run: int(run.Number))
    return {
        "runs": [said(run) for run in runs],
        "since the last run": snapshots.changed(snapshot_of(runs[-1]), glue.snapshot(study)),
    }


def main():
    out = os.environ["RUN_CHANGES_OUT"]
    os.makedirs(out, exist_ok=True)
    again = os.environ.get("RUN_CHANGES_REOPENED")
    answer = reopened(again) if again else first(out)
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(answer, handle, indent=1, default=str)


try:
    main()
except Exception:
    traceback.print_exc()
