# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The study container: what a run is *about*, and what belongs to it.

A solver object standing in for the study container fails:

* **Ownership becomes a document scan.** ``Solvers/openems/document.contents()``
  would walk ``document.Objects`` and take every binding, port and region it
  finds, so a second simulation inherits all of them and has to be refused.
* **The tree lies.** ``claimChildren`` is a *presentation* hook and adding an
  object does not call it, so a freshly created refinement region sits at
  document root. A ``Group`` is real ownership - FreeCAD tracks membership,
  drag-and-drop works, and deleting the analysis takes its contents with it.

So this is an ``App::DocumentObjectGroupPython``, exactly as FEM's analysis
container is, and the group *is* the answer to "what belongs to this study?".

What it holds and what it does not
----------------------------------

In the group: the solver, the mesh policy, the material bindings, the ports,
the refinement regions, the preview.

**Not** in the group: the geometry, and the material library. Ports and
bindings *reference* real CAD solids, and a user's ``Part::Box`` is theirs.
Materials are a shared library at document root; several analyses over one
board use the same FR4.

There is **no** ``Solver`` link beside the group. A link that duplicates group
membership is a link that can point outside it, and nothing needs
disambiguating while one analysis holds one solver. A study holding an openEMS
and a NEC2 solver at once makes the choice by selecting the one to run, as FEM
does.
"""

import FreeCAD

from ._vp_hook import ViewProviderRestored
from .kinds import kind_of

#: The whole of ``EMAnalysis.Waveform``. openEMS is driven by ``SetGaussExcite``
#: and the adapter offers no other call, so this is the one excitation any of
#: this can produce; ``Solvers/openems/policy._waveform`` refuses anything
#: else by name, and repeats the string because it may not import this module.
GAUSSIAN = "Gaussian"

#: Values of ``EMAnalysis.Symmetry``. The first is the default, and it is the
#: honest one: a result then contains only what was measured.
NO_SYMMETRY = "None"
#: The device is its own mirror image about the plane between its two ports,
#: and the two ports are identical. See ``Results/sparameters.MIRROR``.
MIRROR_SYMMETRY = "Mirror"


class EMAnalysis(ViewProviderRestored):
    """Frequency band and excitation intent, plus ownership through ``Group``.

    These properties are on the *study*, not on a backend. What band the device
    is being characterised over is a statement about the problem; how many
    timesteps openEMS takes to answer is not. That split is why ``EMSolverOpenEMS``
    kept the boundaries, the PML depth, the thread count and the interpreter and
    gave these up: it is a solver object and nothing more.
    """

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

        # An enumeration of one. The band above says what is being measured;
        # this says what the measuring is done *with*, and a broadband FDTD run
        # answers the whole band from a single Gaussian pulse - so a study
        # naming its excitation is describing the problem, not choosing a
        # backend, and the property belongs here whether or not there is
        # anything to choose yet.
        #
        # One value, because a setting that reaches nothing is a silent no-op:
        # a sinusoid or a step is a capability an adapter has to grow, and until
        # one does, offering it is a choice that changes no envelope and no
        # answer. Each joins this list beside the wiring that honours it.
        obj.addProperty(
            "App::PropertyEnumeration",
            "Waveform",
            "Analysis",
            "Excitation the band is measured with",
        )
        obj.Waveform = [GAUSSIAN]

        # A statement about the *device*, which is why it is here beside the
        # band and not on a solver: what the structure is cannot depend on which
        # backend looks at it. It is also a statement no geometry check can
        # make on a user's behalf - a board symmetric to within a via's placement is
        # symmetric for this purpose - so it is declared, never inferred.
        #
        # What it buys: a time-domain solver drives one port per run, so a
        # symmetric two-port needs one solve rather than two. S22 = S11 by the
        # mirror and S12 = S21 by reciprocity, which every material this
        # workbench can express obeys. Half the FDTD time for the commonest
        # two-port study.
        #
        # Enumeration rather than a checkbox because other symmetries exist -
        # a symmetric 4-port coupler relates more terms than a mirror does -
        # and a bool would have to be replaced rather than extended.
        obj.addProperty(
            "App::PropertyEnumeration",
            "Symmetry",
            "Analysis",
            "Symmetry of the device, used to derive the S-matrix terms a single"
            " solve cannot measure",
        )
        obj.Symmetry = [NO_SYMMETRY, MIRROR_SYMMETRY]

        # How far down the response is read, which decides how much leakage a
        # truncated run may leave in it: a stopband of -40 dB *is* |S21| = 0.01,
        # so leakage that is nothing beside a response of one is the whole of
        # that. Zero asks for full scale, and holds the run to the same share of
        # unity every matched structure is held to.
        #
        # Declared rather than measured off the sweep, and that is what puts it
        # on the study. A solved response has small terms in it either way, and
        # nothing in the numbers says whether a -40 dB term is the point of the
        # exercise or a rounding error beneath a matched line - only whoever
        # asked for the run knows. Reading it off the answer would also be
        # circular: leakage fills a null in, so the run that most needs the bar
        # is the one that reports the shallowest notch to set it from.
        #
        # It reaches the residual check and nothing else - no run is solved
        # differently for it. See ``Solvers/openems/residual.WANTED``.
        obj.addProperty(
            "App::PropertyFloat",
            "SmallestResponse",
            "Analysis",
            "Smallest response this study reads, in dB (0 = full scale)",
        )
        obj.SmallestResponse = 0.0

        obj.Proxy = self

    def execute(self, obj):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


def createEMAnalysis(doc=None):
    """A study with an openEMS solver and a mesh policy already in it.

    All three at once because none of them is useful alone: an analysis with no
    solver cannot run and an analysis with no mesh policy cannot mesh, and
    making the user assemble that by hand on every new document is three clicks
    to reach the state they always want. FEM's *Analysis container* command does
    the same.
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
    """No analysis could be chosen, and the message says what to do about it."""


def _label(obj):
    return str(getattr(obj, "Label", None) or getattr(obj, "Name", "?"))


def analyses(doc):
    """Every analysis in ``doc``, in document order."""
    return [obj for obj in doc.Objects if kind_of(obj) == "EMAnalysis"]


def analysis_of(obj):
    """The analysis holding ``obj``, or ``None``.

    Nested groups are followed, because a user is entitled to sort a study's
    ports into a subgroup and it should not stop being part of the study.
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

    The one definition for the GUI layer. ``seen`` guards against a cycle: a
    document is a file a user can edit, and a cycle here is an infinite loop
    inside a GUI callback, which hangs FreeCAD with nothing in the log to say
    why. It is kept by identity rather than by name, since two documents may
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


def find_analysis(doc, selection=()):
    """The analysis a newly created object should join.

    Chosen from the selection first - a port of one study picked while another was
    added names the study meant - then from the
    document when it holds exactly one. Anything else raises
    :class:`NoAnalysis` naming the problem, rather than picking one and putting
    the user's new port in a study they were not looking at.

    Deliberately no persisted "active analysis". FEM has one and it is a
    standing source of "why did that go there?"; a rule derived from what is on
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
    ``contents()`` finds the same object and refuses by name when it is missing;
    this one stays quiet so a panel can open on a half-built study.
    """
    for member in getattr(analysis, "Group", None) or ():
        if kind_of(member) == "EMSolverOpenEMS":
            return member
    return None
