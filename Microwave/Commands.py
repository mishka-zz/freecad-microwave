# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The toolbar and menu, with one class per button.

Each command is a ``GetResources`` (icon, label, tooltip), an ``Activated`` and
an inherited ``IsActive``. The helpers at the top of this file hold what is
worth knowing about all of them, rather than the classes:

* every creation command joins the active analysis through ``_add``, so a new
  object lands in one place across the whole workbench;
* every one of them runs inside a transaction, because FreeCAD does not wrap a
  Python command in one and Ctrl-Z would otherwise do nothing;
* every one of them is disabled with no document open, because FreeCAD leaves a
  Python command enabled unless it says otherwise;
* whatever any of them refuses, or leaves unfinished, goes through
  ``Gui.notify``, which states the choice of channel.

Commands hold no logic of their own. ``Objects.port_setup`` reads a selection,
the adapter translates, and anything with a dialog is in ``Gui/``.
"""

import os

import FreeCAD

from Microwave import Objects
from Microwave.Gui import notify
from Microwave.Objects import port_setup
from Microwave.undo import transaction
from Microwave.ViewProviders import icon as get_icon_path


def _selected():
    """What the user has picked, as document objects. Empty outside the GUI."""
    try:
        import FreeCADGui

        return list(FreeCADGui.Selection.getSelection())
    except Exception:  # pragma: no cover - console mode
        return []


def _target_analysis(doc, title):
    """The analysis a new object joins, or ``None`` after saying why not.

    Every creation command goes through here, so a new object joins the analysis
    in one place across the whole workbench. ``title`` is the command's own
    name, so the refusal says which button raised it.
    """
    try:
        return Objects.find_analysis(doc, _selected())
    except Objects.NoAnalysis as error:
        notify.refused(title, str(error))
        return None


def _add(doc, label, make):
    """Create an object inside the active analysis, undoably.

    FreeCAD does not wrap a Python command in a transaction. Without one, Ctrl-Z
    does nothing and the object stays. Measured on 1.1.1, UndoCount stays 0. The
    user then adds a second object, and refusals name the first, which reads as
    naming the wrong object.

    ``Microwave/undo.py`` holds the rule and the quirks behind it. It is the one
    place that writes the open, abort and commit sequence.
    """
    analysis = _target_analysis(doc, label)
    if analysis is None:
        return None
    with transaction(doc, label):
        obj = make()
        analysis.addObject(obj)
    doc.recompute()
    return obj


class _Command:
    """Base for every command here. It supplies ``IsActive``.

    FreeCAD leaves a Python command enabled unless it says otherwise. The
    default is True and there is no warning. Undeclared, the whole toolbar is
    live with no document open, and pressing anything raises on
    ``FreeCAD.ActiveDocument`` being ``None``, several frames from anything the
    user recognises.

    The bar is that a document is open, and not that an analysis exists. Where
    an analysis is the thing missing, ``_target_analysis`` names it and states
    what to do about it, which a button that has gone grey cannot do. Having no
    document needs no explanation.
    """

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class EMAnalysisCommand(_Command):
    def GetResources(self):
        return {
            "Pixmap": get_icon_path("Analysis.svg"),
            "MenuText": "Create EM Analysis",
            "ToolTip": "Create a study: a frequency band, a solver and a mesh policy",
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        with transaction(doc, "Create EM Analysis"):
            Objects.createEMAnalysis(doc)
        doc.recompute()


class EMMaterialCommand(_Command):
    def GetResources(self):
        return {
            "Pixmap": get_icon_path("Material.svg"),
            "MenuText": "Create Material",
            "ToolTip": "Create an EM Material",
        }

    def Activated(self):
        # The material is not created in the analysis. Materials are a shared
        # library, since two studies over one board use the same FR4, so they
        # live at document root and bindings reference them.
        doc = FreeCAD.ActiveDocument
        with transaction(doc, "Create Material"):
            Objects.createEMMaterial()
        doc.recompute()


class EMMaterialFromCatalogCommand(_Command):
    """Pick materials out of the installed catalogs.

    The catalogs live outside the document, so installing more of them costs the
    tree nothing. Only what the user picks becomes an object. The command copies
    the values in and never links to them, so the saved file still solves on a
    machine that has none of those catalogs; see ``Objects.materials``.
    """

    def GetResources(self):
        return {
            "Pixmap": get_icon_path("MaterialCatalog.svg"),
            "MenuText": "Add Material from Catalog...",
            "ToolTip": "Pick a material from the installed catalogs. Values are "
            "copied into the document, so the file still solves on a "
            "machine without them",
        }

    def Activated(self):
        from Microwave.Gui.material_picker import choose_materials

        doc = FreeCAD.ActiveDocument
        chosen = choose_materials(doc)
        if not chosen:
            return
        notes = []
        with transaction(doc, "Add Material from Catalog"):
            for catalog, entry in chosen:
                ref = catalog.ref(entry)
                # This makes a second copy and reports it. Both readings are
                # real: the user picked the same laminate twice by mistake, or
                # the user wants two of it - a stackup with FR-4 on both sides
                # of a core, or the nearest catalog row taken to be edited into
                # a laminate no catalog has. Only the user can tell those apart,
                # so this reports what it did instead of choosing for them.
                #
                # The note names the existing copies rather than counting them.
                # A count here and the number in the label answer nearly the
                # same question from different sets, and they disagree once a
                # copy is deleted. The user can act on a label.
                already = [obj.Label for obj in Objects.sourced_from(doc, ref)]
                obj = Objects.create_from_entry(doc, entry, catalog)
                if already:
                    notes.append(
                        f"{ref} is already in this document as "
                        f"{', '.join(repr(label) for label in already)}; "
                        f"{obj.Label!r} is another copy of it. Delete it if "
                        "that was not what you meant"
                    )
        doc.recompute()
        notify.noted("Add Material from Catalog", notes)


class EMMaterialBindingCommand(_Command):
    def GetResources(self):
        return {
            "Pixmap": get_icon_path("MaterialBinding.svg"),
            "MenuText": "Bind Material to Shape",
            "ToolTip": "Give geometry a material. Select a material and the "
            "solids or faces it is made of, in any order",
        }

    def Activated(self):
        import FreeCADGui

        selection = FreeCADGui.Selection.getSelectionEx()
        notes = []

        def make():
            binding = Objects.createEMMaterialBinding()
            notes.extend(
                f"{binding.Label}: {note}" for note in Objects.fill_binding(binding, selection)
            )
            return binding

        _add(FreeCAD.ActiveDocument, "Bind Material to Shape", make)
        notify.noted("Bind Material to Shape", notes)


class _PortCommand(_Command):
    """One port kind, made from what the user picked.

    The picks and the axes carry the same information. The face the wave enters
    through fixes the direction of travel, and the position of the ground fixes
    the direction of the field. The command reads both off the geometry rather
    than making the user restate them; see ``Objects.port_setup``. Whatever it
    cannot read, it reports, and it creates the port either way.

    Geometry cannot supply the order of the selection. Two copper faces do not
    distinguish the trace from the ground. Each subclass names its order in the
    tooltip.
    """

    #: (icon, menu text, tooltip, factory, filler)
    kind: tuple

    def GetResources(self):
        icon, text, tip, _, _ = self.kind
        return {"Pixmap": get_icon_path(icon), "MenuText": text, "ToolTip": tip}

    def Activated(self):
        import FreeCADGui

        _, text, _, factory, fill = self.kind
        picks = port_setup.picks_from(FreeCADGui.Selection.getSelectionEx())
        notes = []

        def make():
            port = factory(doc=FreeCAD.ActiveDocument)
            notes.extend(f"{port.Label}: {note}" for note in fill(port, picks))
            return port

        _add(FreeCAD.ActiveDocument, text, make)
        # A port from an incomplete pick keeps its default axes and looks
        # finished in the tree, so this reports what could not be inferred.
        # Pre-flight reports it again before a run, for a user who dismissed
        # this.
        notify.noted(text, notes)


class EMPortMicrostripCommand(_PortCommand):
    kind = (
        "PortMicrostrip.svg",
        "Add Microstrip Port",
        "Add a microstrip port. Select the end face of the trace, then the "
        "ground plane it is referenced to",
        Objects.createEMPortMicrostrip,
        port_setup.fill_microstrip,
    )


class EMPortLumpedCommand(_PortCommand):
    kind = (
        "PortLumped.svg",
        "Add Lumped Port",
        "Add a lumped port. Select the source face, then the reference face across the gap",
        Objects.createEMPortLumped,
        port_setup.fill_lumped,
    )


class EMPortWaveguideCommand(_PortCommand):
    kind = (
        "PortWaveguide.svg",
        "Add Waveguide Port",
        "Add a rectangular waveguide port. Select the guide's cross-section",
        Objects.createEMPortRectWaveguide,
        port_setup.fill_waveguide,
    )


class EMPortCoaxialCommand(_PortCommand):
    kind = (
        "PortCoaxial.svg",
        "Add Coaxial Port",
        "Add a coaxial port. Select the ring between the inner conductor and "
        "the shield, at the end of the line",
        Objects.createEMPortCoaxial,
        port_setup.fill_coaxial,
    )


class EMMeshRegionCommand(_Command):
    def GetResources(self):
        return {
            "Pixmap": get_icon_path("MeshRegion.svg"),
            "MenuText": "Add Mesh Refinement",
            "ToolTip": "Set the element size around some geometry: finer than "
            "the policy, or coarser where the detail does not matter",
        }

    def Activated(self):
        import FreeCADGui

        # A region referencing nothing refuses the next mesh by name. That is
        # right, but it is a poor way to meet a new feature, and whatever is
        # selected is almost certainly what the region is for. ElementSize
        # stays at zero, because there is no honest default without knowing the
        # model, and the refusal says which property to set.
        selection = FreeCADGui.Selection.getSelectionEx()
        notes = []

        def make():
            obj = Objects.createEMMeshRegion(FreeCAD.ActiveDocument)
            references = Objects.references_from(selection)
            if references:
                obj.References = references
            else:
                # Otherwise the refusal arrives at the next mesh, naming an
                # object added several actions ago.
                notes.append(
                    f"{obj.Label}: nothing was selected for References, so this "
                    "region is aimed at nothing; pick the geometry in the property editor"
                )
            return obj

        _add(FreeCAD.ActiveDocument, "Add Mesh Refinement", make)
        notify.noted("Add Mesh Refinement", notes)


class UpdateMeshCommand(_Command):
    """Mesh the study and draw the grid, without opening the panel.

    This runs the same action as the panel's button. It is on the toolbar
    because a user inspects the meshing repeatedly while modelling, and reaching
    it through a modal task dialog every time would make them stop doing it.
    """

    def GetResources(self):
        return {
            "Pixmap": get_icon_path("UpdateMesh.svg"),
            "MenuText": "Update Mesh",
            "ToolTip": "Mesh the study and draw the grid",
        }

    def Activated(self):
        from Microwave.Gui import mesh_preview
        from Microwave.Solvers.openems import document

        doc = FreeCAD.ActiveDocument
        analysis = _target_analysis(doc, "Update Mesh")
        if analysis is None:
            return
        try:
            _, report = mesh_preview.refresh(analysis)
        except (document.TranslationError, document.MeshError) as error:
            notify.refused("Update Mesh", str(error))
            return
        FreeCAD.Console.PrintMessage(report.summary() + "\n")
        doc.recompute()


class RunCommand(_Command):
    def GetResources(self):
        return {
            "Pixmap": get_icon_path("Run.svg"),
            "MenuText": "Run Simulation",
            "ToolTip": "Open the simulation control panel",
        }

    def Activated(self):
        import FreeCADGui

        doc = FreeCAD.ActiveDocument
        analysis = _target_analysis(doc, "Run Simulation")
        if analysis is None:
            return
        FreeCADGui.ActiveDocument.setEdit(analysis.Name, 0)


class ExportTouchstoneCommand(_Command):
    """Write the study's S-matrix as a Touchstone file.

    This is the one way a result leaves the document. The document stores a
    matrix, and the next run overwrites it, so exporting is how a matrix is
    kept. It also hands the numbers to everything else in the field, because
    ``.sNp`` is the one format every simulator and VNA reads.

    ``Gui.results.touchstone_export`` makes all the judgements, and it imports
    no Qt. What is left here is a message box, a file dialog and a write.
    """

    def GetResources(self):
        return {
            "Pixmap": get_icon_path("ExportTouchstone.svg"),
            "MenuText": "Export Touchstone",
            "ToolTip": "Write this study's S-parameters as a Touchstone (.sNp) file",
        }

    def Activated(self):
        parent = notify.main_window()
        holder = _result_or_complain("Export Touchstone")
        if holder is not None:
            export_touchstone(holder, parent)


def _result_or_complain(title: str):
    """The stored matrix this command was aimed at, or ``None`` having said why.

    Something can go wrong before any result is read: the study has nothing to
    act on, or it holds more than one matrix with no way to tell which was
    meant. Neither of those is a refusal from the chart or the export, and
    reporting either is better than reporting "cannot read a result" about the
    wrong object.
    """
    from Microwave.Gui import results as results_glue

    try:
        return results_glue.result_in_hand(FreeCAD.ActiveDocument, _selected())
    except (Objects.NoAnalysis, results_glue.NoResult) as error:
        notify.refused(title, str(error))
        return None


class PlotSParametersCommand(_Command):
    """Draw the study's S-matrix.

    This opens the same chart as the result's own double-click, from the toolbar
    instead. A user who has just pressed Run is already looking at the toolbar,
    and a result they have not yet found in the tree can still be reached there.
    """

    def GetResources(self):
        return {
            "Pixmap": get_icon_path("SParameters.svg"),
            "MenuText": "Plot S-parameters",
            "ToolTip": "Draw every measured term of this study's S-matrix, in dB",
        }

    def Activated(self):
        from Microwave.Gui.plot_s_params import show_matrix
        from Microwave.Objects.results import load

        holder = _result_or_complain("Plot S-parameters")
        if holder is None:
            return
        try:
            show_matrix(load(holder))
        except Exception as error:
            notify.refused("Plot S-parameters", f"cannot plot {_label(holder)}: {error}")


class PlotImpedanceCommand(_Command):
    """Draw impedance along the line, from one port's reflection.

    The port is the lowest one that can produce a trace, so the ordinary
    two-port study needs no choice made before the chart appears. Where no port
    can, this uses the lowest anyway and the refusal says what to change. A
    button that does nothing when pressed tells the user nothing. Every port a
    study offers is on the result's own right-click menu.
    """

    def GetResources(self):
        return {
            "Pixmap": get_icon_path("Impedance.svg"),
            "MenuText": "Plot impedance along the line",
            "ToolTip": "Draw impedance against distance or time, from a port's reflection",
        }

    def Activated(self):
        from Microwave.Gui.plot_tdr import show_trace
        from Microwave.Gui.views import impedance_view, traceable_ports
        from Microwave.Objects.results import load

        holder = _result_or_complain("Plot impedance")
        if holder is None:
            return
        try:
            result = load(holder)
            ports = traceable_ports(result) or list(result.port_numbers[:1])
            show_trace(impedance_view(holder, ports[0]))
        except Exception as error:
            notify.refused("Plot impedance", f"cannot plot {_label(holder)}: {error}")


def _label(obj) -> str:
    return str(getattr(obj, "Label", None) or getattr(obj, "Name", "the result"))


def export_touchstone(holder, parent=None):
    """Write one ``EMSParameters`` object to a Touchstone file, asking as needed.

    This takes the result object rather than the analysis, so the tree's
    right-click entry can hand over the thing that was clicked. Re-deriving it
    from the selection exports the wrong matrix: an analysis holding two results
    always gives the first, a result dragged out of its group falls back to
    whatever the document's one analysis holds, and Qt preserves a
    multi-selection on right-click, so the study picked can be the one selected
    earlier.

    ``Gui.results.touchstone_export`` makes all the judgements, and it imports
    no Qt. What is left here is a message box, a file dialog and a write.
    """
    from PySide import QtWidgets

    from Microwave.Gui import results as results_glue

    export = results_glue.touchstone_export(holder)
    if export.refusal:
        notify.refused("Export Touchstone", export.refusal)
        return
    if export.caveat:
        answer = QtWidgets.QMessageBox.question(
            parent,
            "Export Touchstone",
            export.caveat,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Yes,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

    document = getattr(holder, "Document", None)
    start = os.path.join(_start_directory(document), export.stem + export.suffix)
    chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent,
        "Export Touchstone",
        start,
        f"Touchstone (*{export.suffix});;All files (*)",
    )
    if not chosen:
        return

    # The dialog's own overwrite prompt asked about the name the user typed. If
    # the suffix moves that to a different file, the prompt asked about the
    # wrong one, and this check is the only thing between an unrelated file and
    # a silent overwrite.
    target = export.result.touchstone_path(chosen)
    if str(target) != str(chosen) and target.exists():
        answer = QtWidgets.QMessageBox.question(
            parent,
            "Export Touchstone",
            f"This will be written as {target.name}, which already exists.\n\nOverwrite it?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

    try:
        written = export.result.write_touchstone(chosen)
    except Exception as error:
        notify.refused("Export Touchstone", f"could not write the file: {error}")
        return
    FreeCAD.Console.PrintMessage(f"Microwave: wrote {written}\n")


def _start_directory(doc) -> str:
    """Where the save dialog opens: beside the document, or the user's home.

    A document that has never been saved has an empty ``FileName``, and handing
    Qt an empty directory opens it wherever the process happens to be. For
    FreeCAD on macOS that is ``/``.
    """
    path = str(getattr(doc, "FileName", "") or "")
    return os.path.dirname(path) if path else os.path.expanduser("~")


#: Every command: the id FreeCAD knows it by, the class, and the toolbar it sits
#: on. Registration, the toolbars and the menu all read this one table.
#:
#: Typing the id twice - once to register, once in a toolbar tuple - admits a
#: command that is registered and reachable from nowhere, or on a toolbar and
#: registered nowhere. One table makes both states impossible to write down,
#: rather than leaving them for a test to catch.
COMMANDS = (
    ("Microwave_Analysis", EMAnalysisCommand, "Microwave"),
    ("Microwave_Run", RunCommand, "Microwave"),
    ("Microwave_PlotSParameters", PlotSParametersCommand, "Microwave Results"),
    ("Microwave_PlotImpedance", PlotImpedanceCommand, "Microwave Results"),
    ("Microwave_ExportTouchstone", ExportTouchstoneCommand, "Microwave Results"),
    ("Microwave_MaterialFromCatalog", EMMaterialFromCatalogCommand, "Microwave Materials"),
    ("Microwave_Material", EMMaterialCommand, "Microwave Materials"),
    ("Microwave_MaterialBinding", EMMaterialBindingCommand, "Microwave Materials"),
    ("Microwave_PortMicrostrip", EMPortMicrostripCommand, "Microwave Ports"),
    ("Microwave_PortLumped", EMPortLumpedCommand, "Microwave Ports"),
    ("Microwave_PortWaveguide", EMPortWaveguideCommand, "Microwave Ports"),
    ("Microwave_MeshRegion", EMMeshRegionCommand, "Microwave Mesh"),
    ("Microwave_UpdateMesh", UpdateMeshCommand, "Microwave Mesh"),
)

#: Commands that exist and are deliberately not registered, in the same shape as
#: :data:`COMMANDS` so one can be moved between the two.
#:
#: The coaxial port translates, solves and is scored against a closed form, so
#: none of it is deleted or allowed to rot. It is not ready to be put in front
#: of somebody, and a user finds any command that is registered. Nothing here
#: reaches the toolbars or the menu, which both read ``COMMANDS`` alone. The
#: document object, its view provider and the adapter's support for it all stay,
#: so a document that holds one still opens and still solves.
WITHHELD = (("Microwave_PortCoaxial", EMPortCoaxialCommand, "Microwave Ports"),)


def _toolbars():
    """One toolbar per group, in table order, every command its own button.

    There are few enough commands that they all fit, and a visible button is
    easier to find than one behind a dropdown. A dropdown shows whichever
    command was used last, so the others stop existing until a user goes
    looking. FreeCAD makes each of these a separate dockable toolbar the user
    can move or hide, and Assembly, CAM and BIM all split themselves this way.

    The study toolbar holds both ends of the workflow: create and Run.
    Everything a finished run produces is on Microwave Results, and the next
    output goes there too. Touchstone export sits there with the charts,
    because it takes a result out of the document rather than setting one up.
    """
    groups: dict[str, list[str]] = {}
    for name, _, group in COMMANDS:
        groups.setdefault(group, []).append(name)
    return tuple((group, tuple(names)) for group, names in groups.items())


TOOLBARS = _toolbars()


def menu():
    """The same commands, in the same groups, as one flat menu.

    The menu uses separators rather than submenus, so the whole workbench is
    readable in one glance. The toolbars are flat for the same reason. FreeCAD
    spells a separator as the literal string in the command list, and CAM's menu
    is built this way.
    """
    entries: list[str] = []
    for _, commands in TOOLBARS:
        if entries:
            entries.append("Separator")
        entries.extend(commands)
    return entries


def register_commands():
    import FreeCADGui

    for name, command, _ in COMMANDS:
        FreeCADGui.addCommand(name, command())
