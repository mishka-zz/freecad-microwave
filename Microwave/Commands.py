# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The toolbar and menu, with one class per button.

Each command is a ``GetResources`` (icon, label, tooltip), an ``Activated`` and
an inherited ``IsActive``. The helpers at the top of this file hold what is
worth knowing about all of them, rather than the classes:

* every command that creates one object finds the analysis it joins through
  ``_target_analysis``, so a new object lands in one place across the whole
  workbench. ``_add`` hands that analysis to what makes the object and files the
  object there. ``_SolverCommand``, which creates several, does the same thing
  in a loop, and Add Mesh Refinement, which asks the study which settings it
  amends before making anything, files its region itself;
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
from Microwave.Objects.kinds import kind_of
from Microwave.picks import copy_note, resolved
from Microwave.undo import transaction
from Microwave.ViewProviders import icon as get_icon_path


def _selected():
    """What the user has picked, as document objects. Empty outside the GUI."""
    try:
        import FreeCADGui

        return list(FreeCADGui.Selection.getSelection())
    except Exception:  # pragma: no cover - console mode
        return []


def _selection():
    """What the user picked, each pick on the object that owns it, and the
    picks made inside the copy a link shows, which ``picks.resolved`` leaves
    out."""
    import FreeCADGui

    document = FreeCAD.ActiveDocument
    return resolved(
        FreeCADGui.Selection.getCompleteSelection(0), document.Name if document else None
    )


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
    """Create an object inside the active analysis, undoably. ``make`` is
    handed that analysis and returns the object.

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
        obj = make(analysis)
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
            "ToolTip": "Create a study with a frequency band, an openEMS solver, its "
            "mesh settings and a mesh policy",
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
            "ToolTip": "Create a material at the document root. Bind it to shapes with "
            "Bind Material to Shape",
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
            "ToolTip": "Add materials from the installed catalogs. The values are "
            "copied into the document, so the file solves without the catalogs",
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
                        f"{', '.join(repr(label) for label in already)}, so "
                        f"{obj.Label!r} is another copy. Delete {obj.Label!r} "
                        "if you did not want a copy"
                    )
        doc.recompute()
        notify.noted("Add Material from Catalog", notes)


class EMMaterialBindingCommand(_Command):
    def GetResources(self):
        return {
            "Pixmap": get_icon_path("MaterialBinding.svg"),
            "MenuText": "Bind Material to Shape",
            "ToolTip": "Bind a material to shapes. Select the material and the "
            "solids or faces made of it, in any order",
        }

    def Activated(self):
        selection, copies = _selection()
        notes = []

        def make(analysis):
            # The study the binding joins decides which picks would close a
            # cycle.
            binding = Objects.createEMMaterialBinding()
            notes.extend(
                f"{binding.Label}: {note}"
                for note in Objects.fill_binding(binding, selection, analysis, copies)
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
        _, text, _, factory, fill = self.kind
        selection, copies = _selection()
        picks = port_setup.picks_from(selection)
        notes = []

        def make(_analysis):
            port = factory(doc=FreeCAD.ActiveDocument)
            if copies:
                # A pick left out moves every pick after it up a place, and a
                # port reads which face is which off that order, so it fills
                # nothing.
                notes.append(f"{port.Label}: {copy_note(copies)}")
            else:
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
        "Add a rectangular waveguide port. Select the guide's cross-section face, "
        "and the guide's body too if that face is a separate object",
        Objects.createEMPortRectWaveguide,
        port_setup.fill_waveguide,
    )


class EMPortCoaxialCommand(_PortCommand):
    kind = (
        "PortCoaxial.svg",
        "Add Coaxial Port",
        "Add a coaxial port. Select the ring between the inner conductor and "
        "the shield at the end of the line, and the line's body too if the ring "
        "is a separate object",
        Objects.createEMPortCoaxial,
        port_setup.fill_coaxial,
    )


class EMMeshRegionCommand(_Command):
    def GetResources(self):
        return {
            "Pixmap": get_icon_path("MeshRegion.svg"),
            "MenuText": "Add Mesh Refinement",
            "ToolTip": "Set a finer or coarser element size around geometry for one "
            "solver. Select the geometry, and the solver if the study has two",
        }

    def Activated(self):
        # A region referencing nothing refuses the next mesh by name. That is
        # right, but it is a poor way to meet a new feature, and whatever is
        # selected is almost certainly what the region is for. ElementSize
        # stays at zero, because there is no honest default without knowing the
        # model, and the refusal says which property to set.
        title = "Add Mesh Refinement"
        doc = FreeCAD.ActiveDocument
        selection, copies = _selection()
        analysis = _target_analysis(doc, title)
        if analysis is None:
            return
        try:
            settings = Objects.settings_to_amend(analysis, [chosen.Object for chosen in selection])
        except Objects.NoSettings as error:
            notify.refused(title, str(error))
            return
        notes = []

        def make():
            obj = Objects.createEMMeshRegion(doc, settings)
            references = Objects.references_from(selection, analysis)
            if copies:
                notes.append(f"{obj.Label}: {copy_note(copies)}")
            if cyclic := Objects.reaching(selection, analysis):
                notes.append(f"{obj.Label}: {Objects.cycle_note(cyclic)}")
            if references:
                obj.References = references
            elif not cyclic and not copies:
                # Otherwise the refusal arrives at the next mesh, naming an
                # object added several actions ago.
                notes.append(
                    f"{obj.Label}: References is empty because nothing was selected. "
                    "Set References in the property editor"
                )
            return obj

        with transaction(doc, title):
            analysis.addObject(make())
        doc.recompute()
        notify.noted(title, notes)


class MeshCommand(_Command):
    """Mesh the study on one backend, as that backend's panel does.

    It is on the toolbar because a user inspects the meshing repeatedly while
    modelling. The backend is the one the selection names, then the study's
    only one, as Run picks it. openEMS's grid is laid and drawn without a
    panel. Gmsh meshes for Palace on the panel's worker, so Palace's panel
    opens and meshes, and says what it made.
    """

    TITLE = "Mesh"

    def GetResources(self):
        return {
            "Pixmap": get_icon_path("Mesh.svg"),
            "MenuText": self.TITLE,
            "ToolTip": "Mesh the study for its solver and show the mesh. If the study has "
            "more than one solver, select the solver, its mesh settings or its mesh first",
        }

    def Activated(self):
        from Microwave.Gui.panels import open_run

        doc = FreeCAD.ActiveDocument
        analysis = _target_analysis(doc, self.TITLE)
        if analysis is None:
            return
        try:
            solver = Objects.solver_to_run(analysis, _solvers_named(analysis, _selected()))
        except Objects.NoSolver as error:
            notify.refused(self.TITLE, str(error))
            return
        if kind_of(solver) != "EMSolverOpenEMS":
            # A panel already open is not meshed through: meshing there stops
            # whatever it is running, which the refusal says it does not.
            if not open_run(self.TITLE, solver):
                return
            panel = getattr(getattr(solver.ViewObject, "Proxy", None), "panel", None)
            if panel is not None:
                panel.on_mesh()
            return
        from Microwave.Gui import openems_mesh_preview
        from Microwave.Solvers.openems import document

        try:
            _, report = openems_mesh_preview.refresh(analysis)
        except (document.TranslationError, document.MeshError) as error:
            notify.refused(self.TITLE, str(error))
            return
        FreeCAD.Console.PrintMessage(report.summary() + "\n")
        doc.recompute()


def _solvers_named(analysis, selection):
    """The solvers of ``analysis`` that ``selection`` names: each solver picked,
    and the solver whose mesh settings or mesh is picked, as the tree shows them
    under it."""
    named = []
    for solver in Objects.solvers_in(analysis):
        shown = [solver]
        waiting = [solver]
        while waiting:
            claims = getattr(
                getattr(getattr(waiting.pop(), "ViewObject", None), "Proxy", None),
                "claimChildren",
                None,
            )
            for child in claims() if claims is not None else ():
                shown.append(child)
                waiting.append(child)
        if any(chosen is obj for chosen in selection for obj in shown):
            named.append(solver)
    return named


class RunCommand(_Command):
    """Open the panel for the solver the study is run on.

    The solver the selection names, by itself, its mesh settings or its mesh,
    then the study's only one. A study holding
    one of each backend with neither selected is refused by name rather than
    guessed at: a run is minutes, and an answer from the backend nobody meant
    says nothing about it.
    """

    def GetResources(self):
        return {
            "Pixmap": get_icon_path("Run.svg"),
            "MenuText": "Run Simulation",
            "ToolTip": "Open the run panel for the study's solver. If the study has more "
            "than one solver, select the one to run first",
        }

    def Activated(self):
        from Microwave.Gui.panels import open_run

        doc = FreeCAD.ActiveDocument
        analysis = _target_analysis(doc, "Run Simulation")
        if analysis is None:
            return
        try:
            solver = Objects.solver_to_run(analysis, _solvers_named(analysis, _selected()))
        except Objects.NoSolver as error:
            notify.refused("Run Simulation", str(error))
            return
        open_run("Run Simulation", solver)


class _SolverCommand(_Command):
    """What makes a study runnable on one backend, added to the study.

    A backend needs its solver, its own mesh recipe, and the mesh policy every
    backend reads. The command adds whichever of those the study lacks, so a
    study that has the solver and lacks the recipe is repaired by pressing the
    button rather than by deleting the solver, which would cost the user its
    absorber, its timesteps and its paths.

    A study holds one solver of each backend and each backend's translation
    refuses a second, so the command refuses a study already holding all of
    them - at the moment it is pressed rather than at a Run some time later,
    naming the solver already there.
    """

    #: ``(icon, backend, what that backend needs as kind and factory)``. The
    #: solver leads: it is what the button is named for, what the refusal names,
    #: and the one object whose arrival alone needs no explaining.
    backend: tuple

    def GetResources(self):
        icon, name, _ = self.backend
        return {
            "Pixmap": get_icon_path(icon),
            "MenuText": f"Add {name} Solver",
            "ToolTip": f"Add the {name} solver and its mesh settings to the study",
        }

    def Activated(self):
        _, name, wanted = self.backend
        title = f"Add {name} Solver"
        doc = FreeCAD.ActiveDocument
        analysis = _target_analysis(doc, title)
        if analysis is None:
            return

        (solver_kind, _), *_ = wanted
        held = {kind_of(member) for member in Objects.members(analysis)}
        missing = [factory for kind, factory in wanted if kind not in held]
        missing_kinds = [kind for kind, _ in wanted if kind not in held]
        if not missing:
            solver = next(
                obj for obj in Objects.solvers_in(analysis) if kind_of(obj) == solver_kind
            )
            notify.refused(
                title,
                f"{analysis.Label!r} already has the {name} solver {solver.Label!r}. "
                f"A study takes one {name} solver",
            )
            return

        with transaction(doc, title):
            added = [factory(doc) for factory in missing]
            for obj in added:
                analysis.addObject(obj)
        doc.recompute()

        # A backend and its recipe always arrive together, because no command
        # adds a recipe alone, so a study gaining a second backend is the
        # ordinary case and shows itself in the tree. What is worth a word is a
        # study that already had the solver and was missing its companions: the
        # user pressed a button that says it adds a solver and got something
        # else.
        if solver_kind not in missing_kinds:
            names = ", ".join(repr(obj.Label) for obj in added)
            notify.noted(
                title,
                [
                    f"{analysis.Label!r} already had its {name} solver, so only these "
                    f"were added: {names}"
                ],
            )


class EMSolverOpenEMSCommand(_SolverCommand):
    backend = (
        "SolverOpenEMS.svg",
        "openEMS",
        (
            ("EMSolverOpenEMS", Objects.createEMSolverOpenEMS),
            ("EMYeeGrid", Objects.createEMYeeGrid),
            ("EMMeshPolicy", Objects.createEMMeshPolicy),
        ),
    )


class EMSolverPalaceCommand(_SolverCommand):
    backend = (
        "SolverPalace.svg",
        "Palace",
        (
            ("EMSolverPalace", Objects.createEMSolverPalace),
            ("EMGmshMesh", Objects.createEMGmshMesh),
            ("EMMeshPolicy", Objects.createEMMeshPolicy),
        ),
    )


class AddSolverCommand:
    """The solvers a study can be given, behind one toolbar button.

    The one dropdown on the toolbars, because it is the one group that grows
    with each backend. FreeCAD's FEM gives its solvers one button the same way
    (``src/Mod/Fem/femcommands/commands.py``). The menu lists each.
    """

    def GetResources(self):
        return {
            "Pixmap": get_icon_path("SolverOpenEMS.svg"),
            "MenuText": "Add Solver",
            "ToolTip": "Add a solver and its mesh settings to the study",
        }

    def GetCommands(self):
        return ADD_SOLVER

    def GetDefaultCommand(self):
        return 0

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class RefineRunCommand(_Command):
    """Solve a kept run again one step finer, from the panel of its backend.

    The run picked in the tree, or the run whose matrix is picked; a matrix no
    run holds is refused. With nothing of either picked, the study's newest run
    where every run it keeps was solved on one backend.
    Anything else is refused naming the backends, since a refinement of a run
    nobody meant measures nothing anyone asked.
    """

    TITLE = "Solve Again One Step Finer"

    def GetResources(self):
        return {
            "Pixmap": get_icon_path("RefineRun.svg"),
            "MenuText": self.TITLE,
            "ToolTip": "Solve a run again one step finer on the same solver, and report "
            "how far the S-parameters moved. If the study has runs from more than one "
            "solver, select the run first",
        }

    def Activated(self):
        from Microwave.Gui import results as glue
        from Microwave.Gui.panels import open_refined

        picked = _runs_picked()
        if len(picked) > 1:
            names = ", ".join(repr(run.Label) for run in picked)
            notify.refused(self.TITLE, f"{names} are all selected. Select one run to refine")
            return
        alone = [table for table in _tables_picked() if glue.run_of(table) is None]
        if alone:
            names = ", ".join(repr(table.Label) for table in alone)
            notify.refused(
                self.TITLE,
                f"{names} did not come from a run of a study, so there is no solve to "
                "repeat. Select a run",
            )
            return
        if not picked:
            analysis = _target_analysis(FreeCAD.ActiveDocument, self.TITLE)
            if analysis is None:
                return
            runs = list(glue.runs_of(analysis))
            backends = sorted({str(run.Solver) for run in runs})
            if not runs:
                notify.refused(
                    self.TITLE,
                    f"{analysis.Label!r} has no run to refine. Run the study first",
                )
                return
            if len(backends) > 1:
                notify.refused(
                    self.TITLE,
                    f"{analysis.Label!r} has runs from {', '.join(backends)}. "
                    "Select the run to refine in the tree",
                )
                return
            picked = runs[-1:]
        open_refined(self.TITLE, picked[0])


def _runs_picked():
    """The runs picked in the tree, each once: a run, or the run holding a
    picked matrix, in the order picked."""
    from Microwave.Gui import results as glue
    from Microwave.Objects.runs import is_run

    picked = []
    for chosen in _selected():
        run = chosen if is_run(chosen) else glue.run_of(chosen)
        if run is not None and not any(run is held for held in picked):
            picked.append(run)
    return picked


class ShowRunLogCommand(_Command):
    """Show what a kept run said while it ran, in a window of its own.

    Each run picked in the tree, or the run whose matrix is picked. Greyed
    while nothing picked is a run or a run's matrix.
    """

    TITLE = "Show Run Log"

    def GetResources(self):
        return {
            "Pixmap": get_icon_path("RunLog.svg"),
            "MenuText": self.TITLE,
            "ToolTip": "Show the log of each selected run. "
            "Select a run or its S-parameters in the tree first",
        }

    def IsActive(self):
        return super().IsActive() and bool(_runs_picked())

    def Activated(self):
        from Microwave.Gui.run_panel import show_log

        for run in _runs_picked():
            show_log(run)


class ExportTouchstoneCommand(_Command):
    """Write the S-matrix picked in the tree as a Touchstone file. Greyed unless
    one table is picked: a run, or S-parameters.

    This is the one way a result leaves the document, and it hands the numbers
    to everything else in the field, because ``.sNp`` is the one format every
    simulator and VNA reads.

    ``Gui.results.touchstone_export`` makes all the judgements, and it imports
    no Qt. What is left here is a message box, a file dialog and a write.
    """

    TITLE = "Export Touchstone..."

    def GetResources(self):
        return {
            "Pixmap": get_icon_path("ExportTouchstone.svg"),
            "MenuText": self.TITLE,
            "ToolTip": "Write the selected run's S-parameters as a Touchstone (.sNp) file. "
            "Select one run or its S-parameters in the tree first",
        }

    def IsActive(self):
        return super().IsActive() and len(_tables_picked()) == 1

    def Activated(self):
        picked = _tables_picked()
        if len(picked) == 1:
            export_touchstone(picked[0], notify.main_window())


def _tables_picked():
    """The S-parameter tables picked in the tree, each once: a matrix, or the
    matrix of a run, in the order picked."""
    from Microwave.Gui import results as glue

    return glue.tables_in(_selected())


class PlotSParametersCommand(_Command):
    """Draw the runs and S-parameters picked in the tree on one set of axes.

    One table draws every term it measured, and several draw one term each with
    a table to difference against, as ``Gui/plot_s_params.py`` builds them.
    Greyed while nothing picked is a table: a run, or S-parameters.
    """

    TITLE = "Plot S-Parameters"

    def GetResources(self):
        return {
            "Pixmap": get_icon_path("SParameters.svg"),
            "MenuText": self.TITLE,
            "ToolTip": "Plot the selected runs and S-parameters on one chart. "
            "Select them in the tree first, with Ctrl held for several",
        }

    def IsActive(self):
        return super().IsActive() and bool(_tables_picked())

    def Activated(self):
        plot_s_parameters(_selected())


def plot_s_parameters(selection):
    """Draw the tables ``selection`` names on one set of axes, or say why not."""
    from Microwave.Gui import results as results_glue
    from Microwave.Gui.plot_s_params import show_overlay

    try:
        show_overlay(results_glue.overlaid(selection))
    except results_glue.NoResult as error:
        notify.refused(PlotSParametersCommand.TITLE, str(error))
    except Exception as error:
        notify.refused(PlotSParametersCommand.TITLE, f"cannot plot: {error}")


class PlotImpedanceCommand(_Command):
    """Draw impedance along the line, from a port's reflection.

    Each table selected in the tree on one set of axes at one port and one rise
    time. The port is the lowest one every table can produce a trace at, so the
    ordinary two-port study needs no choice made before the chart appears, and
    the chart offers the others. Where no port can, the refusal says what to
    change. Greyed while nothing picked is a table.
    """

    TITLE = "Plot Impedance along the Line"

    def GetResources(self):
        return {
            "Pixmap": get_icon_path("Impedance.svg"),
            "MenuText": self.TITLE,
            "ToolTip": (
                "Plot impedance against distance or time, from a port's reflection. "
                "Select runs or S-parameters in the tree first"
            ),
        }

    def IsActive(self):
        return super().IsActive() and bool(_tables_picked())

    def Activated(self):
        plot_impedance(_selected())


def plot_impedance(selection):
    """Draw the step response of the tables ``selection`` names, or say why not."""
    from Microwave.Gui import results as results_glue
    from Microwave.Gui.plot_tdr import show_traces

    try:
        show_traces(results_glue.overlaid(selection))
    except results_glue.NoResult as error:
        notify.refused(PlotImpedanceCommand.TITLE, str(error))
    except Exception as error:
        notify.refused(PlotImpedanceCommand.TITLE, f"cannot plot: {error}")


def export_touchstone(holder, parent=None, chosen=None):
    """Write one ``EMSParameters`` object to a Touchstone file, asking as needed.

    ``chosen`` is the file FreeCAD's own export dialog was given, where File >
    Export asked for one, and the command asks for none of its own.

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

    if chosen is None:
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
    target = export.target(chosen)
    if str(target) != str(chosen) and target.exists():
        answer = QtWidgets.QMessageBox.question(
            parent,
            "Export Touchstone",
            f"The file will be saved as {target.name}, which already exists.\n\nOverwrite it?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

    try:
        written = export.write(chosen)
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


#: Where a command is reached: a toolbar button and a menu entry, a menu entry
#: alone, or a toolbar button alone.
BOTH, MENU, TOOLBAR = "both", "menu", "toolbar"

#: Every command: the id FreeCAD knows it by, the class, the group it sits in,
#: and where it is reached. Registration, the toolbars and the menu all read
#: this one table.
#:
#: Typing the id twice - once to register, once in a toolbar tuple - admits a
#: command that is registered and reachable from nowhere, or on a toolbar and
#: registered nowhere. One table makes both states impossible to write down,
#: rather than leaving them for a test to catch.
COMMANDS = (
    ("Microwave_Analysis", EMAnalysisCommand, "Microwave", BOTH),
    ("Microwave_AddSolver", AddSolverCommand, "Microwave", TOOLBAR),
    ("Microwave_SolverOpenEMS", EMSolverOpenEMSCommand, "Microwave", MENU),
    ("Microwave_SolverPalace", EMSolverPalaceCommand, "Microwave", MENU),
    ("Microwave_MaterialFromCatalog", EMMaterialFromCatalogCommand, "Microwave Materials", BOTH),
    ("Microwave_Material", EMMaterialCommand, "Microwave Materials", BOTH),
    ("Microwave_MaterialBinding", EMMaterialBindingCommand, "Microwave Materials", BOTH),
    ("Microwave_PortMicrostrip", EMPortMicrostripCommand, "Microwave Ports", BOTH),
    ("Microwave_PortLumped", EMPortLumpedCommand, "Microwave Ports", BOTH),
    ("Microwave_PortWaveguide", EMPortWaveguideCommand, "Microwave Ports", BOTH),
    ("Microwave_MeshRegion", EMMeshRegionCommand, "Microwave Mesh", BOTH),
    ("Microwave_Mesh", MeshCommand, "Microwave Mesh", BOTH),
    ("Microwave_Run", RunCommand, "Microwave Solve", BOTH),
    ("Microwave_RefineRun", RefineRunCommand, "Microwave Solve", BOTH),
    ("Microwave_PlotSParameters", PlotSParametersCommand, "Microwave Results", BOTH),
    ("Microwave_PlotImpedance", PlotImpedanceCommand, "Microwave Results", BOTH),
    ("Microwave_ExportTouchstone", ExportTouchstoneCommand, "Microwave Results", BOTH),
    ("Microwave_ShowRunLog", ShowRunLogCommand, "Microwave Results", BOTH),
)

#: The commands behind the Add Solver button, in table order: one per backend.
ADD_SOLVER = tuple(name for name, command, *_ in COMMANDS if issubclass(command, _SolverCommand))

#: Commands that exist and are deliberately not registered, in the same shape as
#: :data:`COMMANDS` so one can be moved between the two.
#:
#: The coaxial port translates, solves and is scored against a closed form, so
#: none of it is deleted or allowed to rot. It is not ready to be put in front
#: of somebody, and a user finds any command that is registered. Nothing here
#: reaches the toolbars or the menu, which both read ``COMMANDS`` alone. The
#: document object, its view provider and the adapter's support for it all stay,
#: so a document that holds one still opens and still solves.
WITHHELD = (("Microwave_PortCoaxial", EMPortCoaxialCommand, "Microwave Ports", BOTH),)


def _grouped(reached):
    """The commands reached where ``reached`` says, by group in table order."""
    groups: dict[str, list[str]] = {}
    for name, _, group, where in COMMANDS:
        if where in reached:
            groups.setdefault(group, []).append(name)
    return tuple((group, tuple(names)) for group, names in groups.items())


def _toolbars():
    """One toolbar per group, in table order, every command its own button
    except the solvers.

    A visible button is easier to find than one behind a dropdown, which shows
    whichever command was used last. The solvers are the exception: they are
    the one group that grows with each backend, and each is pressed once per
    study. FreeCAD makes each toolbar a separate dockable one the user can move
    or hide, and Assembly, CAM, BIM and FEM all split themselves this way.

    The groups follow what is done to a study in order: made and given its
    solvers, its materials and ports, meshed, solved, and read.
    Touchstone export sits with the charts, because it takes a result out of
    the document rather than setting one up.
    """
    return _grouped((BOTH, TOOLBAR))


TOOLBARS = _toolbars()


def menu():
    """The same groups as one flat menu, each solver an entry of its own.

    The menu uses separators rather than submenus, so the whole workbench is
    readable in one glance. FreeCAD spells a separator as the literal string in
    the command list, and CAM's menu is built this way.
    """
    entries: list[str] = []
    for _, commands in _grouped((BOTH, MENU)):
        if entries:
            entries.append("Separator")
        entries.extend(commands)
    return entries


def register_commands():
    import FreeCADGui

    for name, command, *_ in COMMANDS:
        FreeCADGui.addCommand(name, command())
