# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The study container: what a run is about, and what belongs to it.

A solver object standing in for the study container fails. Ownership becomes a
document scan: ``Solvers/openems/document.contents()`` would walk
``document.Objects`` and take every binding, port and region it finds, so a
second simulation inherits all of them and has to be refused. And the tree shows
the wrong thing: ``claimChildren`` is a presentation hook and adding an object
does not call it, so a freshly created refinement region sits at document root.
A ``Group`` is real ownership - FreeCAD tracks membership, drag-and-drop works,
and deleting the analysis takes its contents with it.

This object is therefore an ``App::DocumentObjectGroupPython``, as FEM's
analysis container is, and the group holds what belongs to the study.

What it holds and what it does not
----------------------------------

In the group: the solver, the mesh policy, each backend's mesh recipe, the
material bindings, the ports, the refinement regions, the preview.

Not in the group: the geometry, and the material library. Ports and bindings
reference real CAD solids, and a user's ``Part::Box`` is theirs. Materials are a
shared library at document root, so several analyses over one board use the same
FR4.

There is no ``Solver`` link beside the group. A link that duplicates group
membership can point outside it, and membership already says what the study
owns. A study may hold one solver per backend, so that a drawing marked up once
is answered by each of them without either erasing the other. Two of one kind
are refused by that backend's own translation.
"""

import FreeCAD

from ._vp_hook import ViewProviderRestored
from .fem_mesh import BUILT_FROM, age, made_for
from .kinds import is_ours, kind_of, recipe_kinds, solver_kinds
from .preview import OUT_OF_DATE
from .runs import RUN
from .staleness import FREECADS_OWN

#: Values of ``EMAnalysis.Symmetry``. The first is the default: a result then
#: contains only what was measured.
NO_SYMMETRY = "None"
#: The device is its own mirror image about the plane between its two ports,
#: and the two ports are identical. See ``Results/sparameters.MIRROR``.
MIRROR_SYMMETRY = "Mirror"


class EMAnalysis(ViewProviderRestored):
    """What the device is asked, plus ownership through ``Group``.

    These properties are on the study rather than on a backend. The band a
    device is characterised over is a statement about the problem. How openEMS
    covers it - one pulse, and how many timesteps it takes - is not, so
    ``EMSolverOpenEMS`` keeps the waveform, the absorber, the thread count and
    the interpreter, and holds none of these.
    """

    #: What the mesher never reads. The band is not here: every element size
    #: is derived from it, so moving either end moves every cell.
    #:
    #: The study is the one owned object the preview cannot link - it holds the
    #: preview in its group, group membership is a dependency edge, and a link
    #: back would close a cycle. So :meth:`onChanged` reads this list and marks
    #: the drawing stale itself.
    #:
    #: The list is also handed to the marking every declaring class applies, so
    #: these carry the status as well. Nothing depends on it here, and a class
    #: declaring one thing in one place is worth more than the calls saved by
    #: an exception.
    MOVES_NO_CELL = (
        "FrequencyPoints",
        "ResponseFloor",
        "Symmetry",
    )

    def __init__(self, obj):
        obj.addProperty(
            "App::PropertyFrequency",
            "FrequencyStart",
            "Analysis",
            "Lowest frequency of the band the study is solved over",
        )
        obj.FrequencyStart = "1.0 GHz"
        obj.addProperty(
            "App::PropertyFrequency",
            "FrequencyStop",
            "Analysis",
            "Highest frequency of the band the study is solved over",
        )
        obj.FrequencyStop = "10.0 GHz"
        obj.addProperty(
            "App::PropertyInteger",
            "FrequencyPoints",
            "Analysis",
            "Number of frequency points in the results",
        )
        obj.FrequencyPoints = 501

        # This is a statement about the device, so it sits here beside the band
        # rather than on a solver: what the structure is cannot depend on which
        # backend looks at it. No geometry check can make the statement for the
        # user - a board symmetric to within a via's placement is symmetric for
        # this purpose - so it is declared and never inferred.
        #
        # A symmetric two-port driven from one port needs one solve rather than
        # two. S22 = S11 by the mirror, and S12 = S21 by reciprocity, which every
        # material this workbench can express obeys. That halves the solve for
        # the commonest two-port study, on either backend.
        #
        # An enumeration rather than a checkbox, because other symmetries exist:
        # a symmetric 4-port coupler relates more terms than a mirror does, and a
        # bool would have to be replaced rather than extended.
        obj.addProperty(
            "App::PropertyEnumeration",
            "Symmetry",
            "Analysis",
            "Mirror: a two-port whose identical ports are mirror images of each"
            " other. The S-parameters of a port not driven are then taken from"
            " the driven one",
        )
        obj.Symmetry = [NO_SYMMETRY, MIRROR_SYMMETRY]

        # How far down the response is read. That decides how large an error a
        # run's shortcut may leave in it: a stopband of -40 dB is |S21| = 0.01,
        # so an error that is negligible beside a response of one is the whole of
        # that term. Zero asks for full scale, and holds the run to the same
        # share of unity every matched structure is held to.
        #
        # The figure is declared rather than measured off the sweep, which is why
        # it sits on the study. A solved response has small terms in it either
        # way, and nothing in the numbers says whether a -40 dB term is the point
        # of the exercise or a rounding error beneath a matched line. Only
        # whoever asked for the run knows. Reading it off the answer would also
        # be circular: an error fills a null in, so the run that most needs the
        # bar is the one that reports the shallowest notch to set it from.
        #
        # It sets the bar each backend holds its own shortcut to - a record
        # stopped short, a reduced model - and changes no setting of the run.
        # See ``Solvers/properties.WANTED``.
        obj.addProperty(
            "App::PropertyFloat",
            "ResponseFloor",
            "Analysis",
            "Lowest response level the results are read at, in dB: 0 for full"
            " scale, negative for a stopband or a notch",
        )
        obj.ResponseFloor = 0.0

        self.declare_what_moves_no_cell(obj)

        obj.Proxy = self

    def onChanged(self, obj, prop):
        """Mark the study's mesh preview stale when this edit moved a cell.

        The preview is told directly because it cannot be told by the graph:
        see :attr:`MOVES_NO_CELL`. Every other owned class reaches it through a
        link, and states the same thing by carrying a property status.

        The declaration is a complement, so a property added to this class
        tomorrow marks the drawing stale until somebody says it moves nothing.
        ``Objects/staleness.py`` says why that is the direction to fail in.

        Membership is answered by comparing rather than by declaring -
        :attr:`MEMBERSHIP` and :func:`_the_meshed_membership_moved`.

        A mesh FreeCAD draws is told the same things the same way, since it
        cannot link the study either: see ``Objects/fem_mesh.py``. What moves no
        cell here moves no element either, since no mesher reads it. The band
        sizes both and is not among them.

        A change of the group asks the tree to read again what each member
        holds in it, on an edit, an undo and a redo alike: see :func:`renest`.
        On restore no member has a view to ask yet.

        Restore is left alone otherwise. FreeCAD fires this for every property of every
        object it reads back, and a file's own preview describes the document
        that file holds. So are an undo and a redo, which put back each mark
        with the edit it was written for. Replayed, the group changes before the
        links of a mesh it holds come back, and a comparison made in between
        would mark a mesh that matches.
        """
        if prop == "Group":
            renest(obj)
        document = getattr(obj, "Document", None)
        if document is None or getattr(document, "Restoring", False):
            return
        if getattr(document, "Transacting", False):
            return
        if prop in MEMBERSHIP:
            if _the_meshed_membership_moved(obj):
                mark_preview_stale(obj)
            mark_meshes_stale(obj, _meshes_whose_membership_moved(obj))
            return
        if prop in type(self).MOVES_NO_CELL or prop in FREECADS_OWN:
            return
        mark_preview_stale(obj)
        mark_meshes_stale(obj, fem_meshes(obj))

    def execute(self, obj):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


def createEMAnalysis(doc=None):
    """A study runnable on openEMS: that solver, its grid settings, and the policy.

    They are created together because none of them is useful alone: an analysis
    with no solver cannot run, one with no mesh policy cannot say what the
    device asks of a mesh, and one with no Yee grid gives openEMS' mesher no
    cell size. Assembling that by hand costs the user a command apiece on every
    new document, to reach the state they always want. FEM's Analysis container
    command does the same.

    Palace's recipe is not made here. ``Commands._SolverCommand`` makes each
    backend's, so a study carries the settings of the backends it is run on
    rather than of every backend installed.
    """
    doc = doc or FreeCAD.ActiveDocument

    obj = doc.addObject("App::DocumentObjectGroupPython", "EMAnalysis")
    EMAnalysis(obj)
    obj.Label = "EM Analysis"

    from ._vp_hook import inject_view_provider
    from .mesh import createEMMeshPolicy, createEMYeeGrid
    from .solver import createEMSolverOpenEMS

    inject_view_provider(obj, "EMAnalysis")

    obj.addObject(createEMSolverOpenEMS(doc))
    obj.addObject(createEMYeeGrid(doc))
    obj.addObject(createEMMeshPolicy(doc))
    return obj


# ---------------------------------------------------------------------------
# Finding the analysis an object belongs to, or should join
# ---------------------------------------------------------------------------


class NoAnalysis(LookupError):
    """No analysis could be chosen. The message says what to do about it."""


def _label(obj):
    return str(getattr(obj, "Label", None) or getattr(obj, "Name", "?"))


def analyses(doc):
    """Every analysis in ``doc``, in document order."""
    return [obj for obj in doc.Objects if kind_of(obj) == "EMAnalysis"]


def analysis_of(obj):
    """The analysis holding ``obj``, or ``None``.

    Nested groups are followed. A user may sort a study's ports into a subgroup,
    and those ports stay part of the study. So are runs, since a matrix a run
    holds is still the study's answer.
    """
    if obj is None:
        return None
    if kind_of(obj) == "EMAnalysis":
        return obj
    document = getattr(obj, "Document", None)
    if document is None:
        return None
    for analysis in analyses(document):
        if _holds(analysis, obj):
            return analysis
    return None


def members(group, seen=None):
    """Everything in a group, following nested groups once each, and stopping
    at a run.

    The one definition for the GUI layer. A run is a member and what it holds is
    not: a run records a solve that happened, and nothing it holds is part of
    what the study asks now. See ``Objects/runs.py``.

    ``seen`` guards against a cycle. A document is a file a user can edit, and a
    cycle here is an infinite loop inside a GUI callback, which hangs FreeCAD
    with nothing in the log to say why. ``seen`` holds identities rather than
    names, because two documents may hold objects called ``EMPortLumped`` and
    neither should hide the other.
    """
    return _walk(group, set() if seen is None else seen, into_runs=False)


def _walk(group, seen, into_runs):
    found = []
    for member in getattr(group, "Group", None) or ():
        if id(member) in seen:
            continue
        seen.add(id(member))
        found.append(member)
        if getattr(member, "Group", None) is not None and (into_runs or kind_of(member) != RUN):
            found.extend(_walk(member, seen, into_runs))
    return found


def _holds(group, obj):
    """Whether ``obj`` stands anywhere in ``group``, inside a run as well, so a
    result a run holds still names its study."""
    return any(member is obj for member in _walk(group, set(), into_runs=True))


def find_preview(analysis):
    """The analysis's mesh preview, or ``None``. One per analysis.

    The lookup is per analysis rather than per document. Two studies over one
    board have two grids, and a document-wide lookup would hand the second
    study the first one's picture.
    """
    for member in members(analysis):
        if kind_of(member) == "EMMeshPreview":
            return member
    return None


#: The properties that say the study's membership may have moved. ``Group`` is
#: the study's own. ``_GroupTouched`` is what FreeCAD fires when a group
#: *inside* the study changes, and sorting twenty ports into a subgroup is
#: ordinary housekeeping that ``members`` and ``contents`` both follow, so the
#: trigger has to follow it too.
MEMBERSHIP = ("Group", "_GroupTouched")

#: What a study may hold that ``Gui/openems_mesh_preview.py::_link`` never
#: records. Everything else in the group reaches the mesher, so a member added
#: or taken away moves the grid.
#:
#: This list mirrors that one, and a kind added to this workbench tomorrow is
#: counted here until somebody adds it there. That leaves the comparison
#: permanently unequal and the badge permanently stale, which is loud rather
#: than quiet - and ``tests/test_preview_recompute.py`` catches it on the
#: shipped examples, where a result object arriving has to leave the drawing
#: alone.
#:
#: A material is here because a grid is not laid from where it sits. It is
#: reached through the binding that names it, wherever the tree keeps it, so
#: dragging one into the study moves no cell.
#:
#: Another backend's solver is here because this preview is one backend's grid.
#: It is laid from what that adapter's translation reads, and that translation
#: reads its own solver and no other, so a study that gains a second one gains
#: nothing the grid depends on. Another backend's mesh recipe is here for the
#: same reason: it holds the primitives of a discretisation this grid is not.
#: ``tests/test_analysis.py`` derives which those are from the solver classes
#: and from the recipe classes, so a third backend is caught here rather than by
#: a badge the user cannot clear.
NOT_MESHED_FROM = (
    "EMGmshMesh",
    "EMMaterial",
    "EMMeshPreview",
    "EMRun",
    "EMSParameters",
    "EMSolverPalace",
)


def _what_the_grid_is_laid_from(objects):
    """The names among ``objects`` that a study owns and a grid is laid from."""
    return _built_from(objects, NOT_MESHED_FROM)


def left_out(obj, excluded):
    """Whether ``obj`` is of a kind in ``excluded``, or a refinement region
    amending a recipe of such a kind - which is another pipeline's to read."""
    if kind_of(obj) == "EMMeshRegion":
        return kind_of(getattr(obj, "MeshSettings", None)) in excluded
    return kind_of(obj) in excluded


def _built_from(objects, excluded):
    """The names among ``objects`` that a study owns and are not ``excluded``.

    Anything this workbench does not own is left out, so the geometry a
    binding points at does not count as membership: it is in a mesh's link list
    and never in the study's group. So is a mesh FreeCAD draws.
    """
    return {obj.Name for obj in objects if is_ours(obj) and not left_out(obj, excluded)}


def _the_meshed_membership_moved(analysis):
    """Whether the study holds a different set of what the grid is laid from.

    ``Gui/openems_mesh_preview.py::_link`` records that set on the preview,
    and it is rebuilt there and nowhere else. So a port dragged out of the
    study, or a refinement region created after the last Mesh, moves
    the grid and reaches the preview through no link at all - the object is in
    no list, and every later edit to it is invisible for the same reason,
    across a save and until the next Mesh. Comparing the two sets walks
    the group and translates nothing.

    Nested groups are followed, because ``members`` and ``contents`` follow
    them. What is left out is on :data:`NOT_MESHED_FROM`.

    Nothing re-checks membership when a file opens. A study saved holding a
    member its preview was never linked from reopens saying the drawing
    matches, and is told only when something changes. The panel derives the key
    whenever it opens, which is what reports such a study.
    """
    preview = find_preview(analysis)
    if preview is None:
        return False
    return _what_the_grid_is_laid_from(members(analysis)) != _what_the_grid_is_laid_from(
        getattr(preview, "MeshedFrom", None) or ()
    )


#: What a study may hold that a mesh FreeCAD draws is not built from. The
#: counterpart of :data:`NOT_MESHED_FROM` for the tetrahedral mesh, and
#: ``Gui/fem_mesh.py::built_from`` links everything else a study holds. The
#: openEMS solver, its grid and its preview are another backend's; the Palace
#: solver's settings are read once a mesh is made, since the order of the
#: elements the mesher lays is fixed rather than chosen.
NOT_FEM_MESHED_FROM = (
    "EMMaterial",
    "EMMeshPreview",
    "EMRun",
    "EMSParameters",
    "EMSolverOpenEMS",
    "EMSolverPalace",
    "EMYeeGrid",
)


def fem_meshes(analysis):
    """Every mesh FreeCAD draws that the study holds, as ``Gui/fem_mesh.py``
    puts one there."""
    return [member for member in members(analysis) if made_for(member)]


def _meshes_whose_membership_moved(analysis):
    """The study's meshes FreeCAD draws that were built from a different set of
    members than the study holds now. :func:`_the_meshed_membership_moved` for
    the preview says why a comparison is the question to ask."""
    held = _built_from(members(analysis), NOT_FEM_MESHED_FROM)
    return [
        mesh
        for mesh in fem_meshes(analysis)
        if _built_from(getattr(mesh, BUILT_FROM, None) or (), NOT_FEM_MESHED_FROM) != held
    ]


def mark_meshes_stale(analysis, meshes):
    """Say that each of ``meshes`` no longer describes the document.

    Failures are swallowed, for the reason :func:`mark_preview_stale` gives.
    """
    for mesh in meshes:
        try:
            age(mesh)
        except Exception as error:  # pragma: no cover - the panel still has the key
            label = getattr(analysis, "Label", "")
            FreeCAD.Console.PrintWarning(
                f"Microwave: could not mark the mesh in {label!r} out of date: {error}\n"
            )


def mark_preview_stale(analysis):
    """Say that this study's drawing no longer describes the document.

    A study with no preview yet, or one whose preview came back from a file
    without its ``Proxy``, has nothing to tell and this does nothing.

    Failures are swallowed. This runs from inside ``onChanged``, where an
    exception reaches the user as a traceback over an ordinary property edit,
    and the panel re-derives the staleness key whenever it opens.
    """
    try:
        preview = find_preview(analysis)
        if preview is not None:
            preview.Status = OUT_OF_DATE
    except Exception as error:  # pragma: no cover - the panel still has the key
        FreeCAD.Console.PrintWarning(
            f"Microwave: could not mark the mesh preview out of date: {error}\n"
        )


def find_analysis(doc, selection=()):
    """The analysis a newly created object should join.

    Chosen from the selection first: a port of one study, picked while another
    object is added, names the study meant. Chosen from the document next, when
    the document holds exactly one analysis. Anything else raises
    :class:`NoAnalysis` naming the problem, rather than picking one and putting
    the user's new port in a study they were not looking at.

    There is no persisted "active analysis". FEM has one, and a user then has to
    work out why an object went where it did. A rule derived from what is on
    screen cannot go stale.
    """
    for chosen in selection:
        found = analysis_of(chosen)
        if found is not None:
            return found

    found = analyses(doc)
    if len(found) == 1:
        return found[0]
    if not found:
        raise NoAnalysis(
            "this document has no EM analysis to add to. Create one with Create EM Analysis"
        )
    names = ", ".join(_label(obj) for obj in found)
    raise NoAnalysis(
        f"this document holds {len(found)} analyses ({names}). Select the one to "
        "add to, or an object inside it, and try again"
    )


def solvers_in(analysis):
    """Every solver object in an analysis, in group order.

    Which kinds count is derived from the classes rather than typed out, so a
    backend added tomorrow answers here as it stands - ``Objects/kinds.py``
    states that rule and ``solver_kinds`` is what reads it.

    For whatever has to say which backend a study is answered by. Nothing is
    refused here: a study holding none and a study holding one of each are both
    ordinary, and reading why a study cannot run is what a panel is for. Nested
    groups are followed, for the reason :func:`solver_of` gives.
    """
    wanted = solver_kinds()
    return [member for member in members(analysis) if kind_of(member) in wanted]


def companions(solver):
    """What ``solver`` brings to its study and no other member reads: its
    backend's mesh settings, each refinement region amending them, and each
    mesh made from them - the regions and meshes first, the settings last.

    The study's mesh policy is every backend's, and a run is a result, so
    neither is among them. A region amending the settings is one wherever it is
    filed in the study, since a region left amending nothing is refused by every
    backend.
    """
    study = analysis_of(solver)
    wanted = getattr(getattr(solver, "Proxy", None), "RECIPE", None)
    if study is None or wanted is None:
        return []
    held = members(study)
    recipes = [member for member in held if kind_of(member) == wanted]
    amending = [
        member
        for member in held
        if any(getattr(member, "MeshSettings", None) is recipe for recipe in recipes)
        or any(
            linked is recipe
            for linked in getattr(member, "MeshedFrom", ()) or ()
            for recipe in recipes
        )
    ]
    return amending + recipes


def solver_of(analysis):
    """The openEMS solver object in an analysis, or ``None``.

    A study may hold a second backend's solver beside it, and this is not it.
    What a caller reads off the object is FDTD vocabulary - the run length, the
    timestep factor, the interpreter that owns the bindings, the directory a run
    is written into - and the other backend has a word for none of it.

    For the GUI, which needs somewhere to read those before it knows whether the
    model translates. Nested groups are followed, because the adapter's own
    ``contents()`` follows them and the two have to find the same object. This
    one stays quiet where there is none, so a panel can open on a half-built
    study.
    """
    for member in members(analysis):
        if kind_of(member) == "EMSolverOpenEMS":
            return member
    return None


class NoSolver(LookupError):
    """No solver could be chosen to run. The message says what to do about it."""


class NoSettings(LookupError):
    """No mesh settings could be chosen for a refinement region to amend. The
    message says what to do about it."""


def settings_to_amend(analysis, selection=()):
    """The mesh settings a new refinement region in ``analysis`` amends.

    The selection first, as :func:`solver_to_run` reads it: a solver of this
    study, its mesh settings, a mesh made from them or a region amending them
    names that backend's settings, and a selection naming two backends names
    none. Then the settings of the study's only solver. Anything else raises
    :class:`NoSettings` naming what is there, since a region on the backend
    nobody meant refines nothing the user looks at.
    """
    held = members(analysis)
    recipes = [member for member in held if kind_of(member) in recipe_kinds()]

    def recipe_of_kind(kind):
        return next((recipe for recipe in recipes if kind_of(recipe) == kind), None)

    def named_by(chosen):
        if any(chosen is recipe for recipe in recipes):
            return chosen
        if not any(chosen is member for member in held):
            return None
        if kind_of(chosen) == "EMMeshRegion":
            linked = getattr(chosen, "MeshSettings", None)
            return linked if any(linked is recipe for recipe in recipes) else None
        if kind_of(chosen) in solver_kinds():
            return recipe_of_kind(getattr(chosen.Proxy, "RECIPE", None))
        for linked in getattr(chosen, "MeshedFrom", None) or ():
            if any(linked is recipe for recipe in recipes):
                return linked
        return None

    picked = []
    for chosen in selection:
        found = named_by(chosen)
        if found is not None and not any(found is one for one in picked):
            picked.append(found)
    label = _label(analysis)
    if len(picked) == 1:
        return picked[0]
    if picked:
        names = ", ".join(repr(_label(recipe)) for recipe in picked)
        raise NoSettings(
            f"The selection names {names} of {label!r}, and a refinement region "
            "refines the mesh of one solver. Select the solver, mesh settings or "
            "mesh of one solver only"
        )
    solvers = solvers_in(analysis)
    if not solvers:
        raise NoSettings(
            f"{label!r} holds no solver, so a refinement region has no mesh to "
            "refine. Add one with Add openEMS Solver or Add Palace Solver"
        )
    if len(solvers) > 1:
        names = ", ".join(repr(_label(solver)) for solver in solvers)
        raise NoSettings(
            f"{label!r} holds {names}, and a refinement region refines the mesh of "
            "one of them. Select that solver together with the geometry, and add "
            "the region again"
        )
    (solver,) = solvers
    found = recipe_of_kind(getattr(solver.Proxy, "RECIPE", None))
    if found is None:
        raise NoSettings(
            f"{label!r} holds {_label(solver)!r} but not its mesh settings. "
            "Add the same solver again to restore them"
        )
    return found


def renest(analysis):
    """Ask the tree to read again the children of ``analysis`` and of each
    member of it that holds children in it.

    The tree reads an object's children again only when a property of that
    object changes (``src/Gui/Tree.cpp``), so a mesh joining the study would
    stand at the document's root rather than under the solver it was made for,
    and a region turned to other settings would stay under the first.
    ``ShowInTree`` is a property of the view and counts as a change, and
    setting it to what it holds changes no object of the document. There is no
    view without the interface, and nothing to ask.
    """
    for member in [analysis, *(getattr(analysis, "Group", ()) or ())]:
        view = getattr(member, "ViewObject", None)
        if getattr(getattr(view, "Proxy", None), "claimChildren", None) is not None:
            view.ShowInTree = view.ShowInTree


def solver_to_run(analysis, selection=()):
    """The solver a Run opens the panel for.

    The selection first: one solver of this study picked in the tree names the
    backend meant, and more than one names none. Then the study's only solver.
    Anything else raises
    :class:`NoSolver` naming what is there, rather than picking one: a run is
    minutes, and a study answered by the backend nobody asked for says nothing
    about having been.

    A solver of another study in the selection names nothing here, and the rule
    falls through to this study's own.
    """
    found = solvers_in(analysis)
    picked = [solver for solver in found if any(chosen is solver for chosen in selection)]
    if len(picked) == 1:
        return picked[0]
    label = _label(analysis)
    if picked:
        names = ", ".join(repr(_label(solver)) for solver in picked)
        raise NoSolver(f"{names} of {label!r} are all selected. Select only the one to run")
    if len(found) == 1:
        return found[0]
    if not found:
        raise NoSolver(
            f"{label!r} holds no solver. Add one with Add openEMS Solver or Add Palace Solver"
        )
    names = ", ".join(repr(_label(solver)) for solver in found)
    raise NoSolver(
        f"{label!r} holds the solvers {names}. Select the one to run in the tree, "
        "or double-click it"
    )
