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

In the group: the solver, the mesh policy, the material bindings, the ports,
the refinement regions, the preview.

Not in the group: the geometry, and the material library. Ports and bindings
reference real CAD solids, and a user's ``Part::Box`` is theirs. Materials are a
shared library at document root, so several analyses over one board use the same
FR4.

There is no ``Solver`` link beside the group. A link that duplicates group
membership can point outside it, and nothing needs disambiguating while one
analysis holds one solver. A study holding two solvers is refused by name
instead. A run starts from the study, and nothing in the study says which of the
two it starts with.
"""

import FreeCAD

from ._vp_hook import ViewProviderRestored
from .kinds import is_ours, kind_of
from .preview import OUT_OF_DATE
from .staleness import FREECADS_OWN

#: The whole of ``EMAnalysis.Waveform``. It names the pulse's envelope, which is
#: the part a study chooses. The carrier on it belongs to the adapter, and there
#: is one adapter, so this is the one excitation the workbench can produce.
#: ``Solvers/openems/policy._waveform`` refuses anything else by name, and
#: repeats the string because it may not import this module.
GAUSSIAN = "Gaussian"

#: Values of ``EMAnalysis.Symmetry``. The first is the default: a result then
#: contains only what was measured.
NO_SYMMETRY = "None"
#: The device is its own mirror image about the plane between its two ports,
#: and the two ports are identical. See ``Results/sparameters.MIRROR``.
MIRROR_SYMMETRY = "Mirror"


class EMAnalysis(ViewProviderRestored):
    """Frequency band and excitation intent, plus ownership through ``Group``.

    These properties are on the study rather than on a backend. The band a
    device is characterised over is a statement about the problem. How many
    timesteps openEMS takes to answer it is not, so ``EMSolverOpenEMS`` keeps
    the boundaries, the PML depth, the thread count and the interpreter, and
    holds none of these.
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
        "NumFrequencyPoints",
        "SmallestResponse",
        "Symmetry",
        "Waveform",
    )

    def __init__(self, obj):
        obj.addProperty(
            "App::PropertyFrequency",
            "FrequencyStart",
            "Analysis",
            "Start of the band being characterised",
        )
        obj.FrequencyStart = "1.0 GHz"
        obj.addProperty(
            "App::PropertyFrequency",
            "FrequencyStop",
            "Analysis",
            "End of the band being characterised",
        )
        obj.FrequencyStop = "10.0 GHz"
        obj.addProperty(
            "App::PropertyInteger",
            "NumFrequencyPoints",
            "Analysis",
            "How many points the results are reported at",
        )
        obj.NumFrequencyPoints = 501

        # An enumeration with one value. The band above says what is being
        # measured, and this says what it is measured with. A broadband FDTD run
        # answers the whole band from a single Gaussian pulse. A study naming its
        # excitation describes the problem rather than choosing a backend, so the
        # property belongs here whether or not there is anything to choose yet.
        #
        # The list holds one value because a setting that reaches nothing is a
        # silent no-op. A sinusoid or a step is a capability an adapter has to
        # grow, and until one does, offering it changes no envelope and no
        # answer. Each joins this list beside the wiring that honours it.
        obj.addProperty(
            "App::PropertyEnumeration",
            "Waveform",
            "Analysis",
            "Excitation the band is measured with",
        )
        obj.Waveform = [GAUSSIAN]

        # This is a statement about the device, so it sits here beside the band
        # rather than on a solver: what the structure is cannot depend on which
        # backend looks at it. No geometry check can make the statement for the
        # user - a board symmetric to within a via's placement is symmetric for
        # this purpose - so it is declared and never inferred.
        #
        # A time-domain solver drives one port per run, so a symmetric two-port
        # needs one solve rather than two. S22 = S11 by the mirror, and
        # S12 = S21 by reciprocity, which every material this workbench can
        # express obeys. That halves the FDTD time for the commonest two-port
        # study.
        #
        # An enumeration rather than a checkbox, because other symmetries exist:
        # a symmetric 4-port coupler relates more terms than a mirror does, and a
        # bool would have to be replaced rather than extended.
        obj.addProperty(
            "App::PropertyEnumeration",
            "Symmetry",
            "Analysis",
            "Symmetry of the device, used to derive the S-matrix terms a single"
            " solve cannot measure",
        )
        obj.Symmetry = [NO_SYMMETRY, MIRROR_SYMMETRY]

        # How far down the response is read. That decides how much leakage a
        # truncated run may leave in it: a stopband of -40 dB is |S21| = 0.01,
        # so leakage that is negligible beside a response of one is the whole of
        # that term. Zero asks for full scale, and holds the run to the same
        # share of unity every matched structure is held to.
        #
        # The figure is declared rather than measured off the sweep, which is why
        # it sits on the study. A solved response has small terms in it either
        # way, and nothing in the numbers says whether a -40 dB term is the point
        # of the exercise or a rounding error beneath a matched line. Only
        # whoever asked for the run knows. Reading it off the answer would also
        # be circular: leakage fills a null in, so the run that most needs the
        # bar is the one that reports the shallowest notch to set it from.
        #
        # It reaches the residual check and nothing else. No run is solved
        # differently for it. See ``Solvers/openems/residual.WANTED``.
        obj.addProperty(
            "App::PropertyFloat",
            "SmallestResponse",
            "Analysis",
            "Smallest response this study reads, in dB (0 = full scale)",
        )
        obj.SmallestResponse = 0.0

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

        Restore is left alone. FreeCAD fires this for every property of every
        object it reads back, and a file's own preview describes the document
        that file holds.
        """
        document = getattr(obj, "Document", None)
        if document is None or getattr(document, "Restoring", False):
            return
        if prop in MEMBERSHIP:
            if _the_meshed_membership_moved(obj):
                mark_preview_stale(obj)
            return
        if prop in type(self).MOVES_NO_CELL or prop in FREECADS_OWN:
            return
        mark_preview_stale(obj)

    def execute(self, obj):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


def createEMAnalysis(doc=None):
    """A study with an openEMS solver and a mesh policy already in it.

    They are created together because none of them is useful alone: an analysis
    with no solver cannot run, and an analysis with no mesh policy cannot mesh.
    Assembling that by hand costs the user a command apiece on every new
    document, to reach the state they always want. FEM's Analysis container
    command does the same.
    """
    doc = doc or FreeCAD.ActiveDocument

    obj = doc.addObject("App::DocumentObjectGroupPython", "EMAnalysis")
    EMAnalysis(obj)
    obj.Label = "EM Analysis"

    from ._vp_hook import inject_view_provider
    from .mesh import createEMMeshPolicy
    from .solver import createEMSolverOpenEMS

    inject_view_provider(obj, "EMAnalysis")

    obj.addObject(createEMSolverOpenEMS(doc))
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
    and those ports stay part of the study.
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
    """Everything in a group, following nested groups once each.

    The one definition for the GUI layer. ``seen`` guards against a cycle. A
    document is a file a user can edit, and a cycle here is an infinite loop
    inside a GUI callback, which hangs FreeCAD with nothing in the log to say
    why. ``seen`` holds identities rather than names, because two documents may
    hold objects called ``EMPortLumped`` and neither should hide the other.
    """
    seen = set() if seen is None else seen
    found = []
    for member in getattr(group, "Group", None) or ():
        if id(member) in seen:
            continue
        seen.add(id(member))
        found.append(member)
        if getattr(member, "Group", None) is not None:
            found.extend(members(member, seen))
    return found


def _holds(group, obj):
    return any(member is obj for member in members(group))


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

#: What a study may hold that ``Gui/mesh_preview.py::_link`` never records.
#: Everything else in the group reaches the mesher, so a member added or taken
#: away moves the grid.
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
NOT_MESHED_FROM = ("EMMaterial", "EMMeshPreview", "EMSParameters")


def _what_the_grid_is_laid_from(objects):
    """The names among ``objects`` that a study owns and a grid is laid from.

    Anything this workbench does not own is left out, so the geometry a
    binding points at does not count as membership: it is in the preview's
    link list and never in the study's group.
    """
    return {obj.Name for obj in objects if is_ours(obj) and kind_of(obj) not in NOT_MESHED_FROM}


def _the_meshed_membership_moved(analysis):
    """Whether the study holds a different set of what the grid is laid from.

    ``Gui/mesh_preview.py::_link`` records that set on the preview, and it is
    rebuilt there and nowhere else. So a port dragged out of the study, or a
    refinement region created after the last Update Mesh, moves the grid and
    reaches the preview through no link at all - the object is in no list, and
    every later edit to it is invisible for the same reason, across a save and
    until the next Update Mesh. Comparing the two sets walks the group and
    translates nothing.

    Nested groups are followed, because ``members`` and ``contents`` follow
    them. What is left out is on :data:`NOT_MESHED_FROM`.

    A file written before this comparison existed can hold a member the drawing
    was never linked from, and it reopens saying the drawing matches: nothing
    re-checks membership on open, and the study is told only when something
    changes. The panel derives the key whenever it opens, which is what reports
    that file.
    """
    preview = find_preview(analysis)
    if preview is None:
        return False
    return _what_the_grid_is_laid_from(members(analysis)) != _what_the_grid_is_laid_from(
        getattr(preview, "MeshedFrom", None) or ()
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
        FreeCAD.Console.PrintWarning(f"Microwave: could not age the mesh preview: {error}\n")


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
            "this document has no EM analysis, so there is nothing for a new "
            "object to belong to. Create one first"
        )
    names = ", ".join(_label(obj) for obj in found)
    raise NoAnalysis(
        f"this document holds {len(found)} analyses ({names}). Select the one "
        "you mean - or anything inside it - before adding to it"
    )


def solver_of(analysis):
    """The openEMS solver object in an analysis, or ``None``.

    For the GUI, which needs somewhere to read ``SimDir`` and ``SolverPython``
    before it knows whether the model translates. The adapter's own
    ``contents()`` finds the same object and refuses by name when it is missing.
    This one stays quiet, so a panel can open on a half-built study.
    """
    for member in getattr(analysis, "Group", None) or ():
        if kind_of(member) == "EMSolverOpenEMS":
            return member
    return None
