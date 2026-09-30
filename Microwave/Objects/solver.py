# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""One class per backend, each holding what is that backend's own.

The band a device is characterised over is a statement about the problem, so it
sits on the study and every solver here answers the same one. What is left is
each backend's own vocabulary, and none of it carries over: a PML depth is
counted in cells of a grid the other backend never lays, and a polynomial order
describes a field inside an element the other one has no word for.

A study may hold one solver of each kind, so that a drawing marked up once is
answered by each of them. Two of one kind are refused by the command that adds
one and by that backend's own translation.
"""

import FreeCAD

from ._vp_hook import ViewProviderRestored

#: How an openEMS face absorbs, the first being the default. The adapter states
#: the list again, because this module imports FreeCAD and the adapter does not;
#: a test holds the two together.
ABSORBERS = ("PML", "Mur")

#: What ``EMSolverOpenEMS.Waveform`` offers: the envelope of the one pulse that
#: covers the band. The carrier on it is the adapter's. A shape the adapter
#: cannot produce joins beside the wiring that honours it, because a choice that
#: reaches nothing is a silent no-op. Stated again by the adapter for the reason
#: ``ABSORBERS`` is.
WAVEFORMS = ("Gaussian",)

#: How a Palace sweep reaches the points across the band, the first being the
#: default. Stated again by the adapter for the reason ``ABSORBERS`` is.
SWEEPS = ("Adaptive", "Discrete")


class EMSolverBase(ViewProviderRestored):
    """What every solver is, carrying no property at all.

    A property here would reach every backend, and there is nothing left for one
    to say: what several backends need alike is a statement about the problem
    and belongs to the study. What this class carries instead is the answer to
    whether a document object is a solver.
    ``Objects/kinds.py::solver_kinds`` reads it. Asking by the class rather than
    by a prefix on a name is the rule that module gives.

    The methods below are what FreeCAD asks of any proxy that stores nothing of
    its own.
    """

    def execute(self, obj):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


class EMSolverOpenEMS(EMSolverBase):
    """The openEMS solver: this backend's own settings, and nothing neutral.

    FEM calls the same thing ``solverbase.Proxy``. The frequency band sits on
    :class:`~.analysis.EMAnalysis` instead. The band a device is characterised
    over is a statement about the problem, and every solver answering it needs
    the same one.

    What is left is FDTD vocabulary throughout: the pulse that covers the band,
    the absorber, timesteps, a stability factor, the interpreter that owns the
    bindings. A property belongs
    here when it is written in that vocabulary. Which face of the domain is a
    wall and which absorbs is not: it follows from what the mesh policy's
    ``Padding`` says lies beyond each face, which every backend reads. What is
    left here is how this one absorbs - a perfectly matched layer so many cells
    deep, or Mur's first-order condition - because "PML with 8 cells" is not a
    neutral concept.
    """

    #: What the mesher never reads. The absorber and its depth are not here:
    #: a perfectly matched layer is cells, and Mur's condition takes none.
    #: Measured over the shipped examples and a WR-90 part - see
    #: ``Objects/staleness.py`` for what the declaration is for.
    MOVES_NO_CELL = (
        "EnergyDecay",
        "MaxTimesteps",
        "SimDir",
        "SolverPython",
        "Threads",
        "TimestepFactor",
        "Waveform",
    )

    def __init__(self, obj):
        # How every face that absorbs does it. Which faces those are is the mesh
        # policy's Padding, and openems.policy._boundary is the rule.
        obj.addProperty(
            "App::PropertyEnumeration",
            "Absorber",
            "Absorber",
            "How each face the mesh policy leaves open absorbs: PML, a perfectly"
            " matched layer PMLCells deep, or Mur, a first-order condition on the"
            " face itself",
        )
        obj.Absorber = list(ABSORBERS)

        obj.addProperty("App::PropertyInteger", "PMLCells", "Absorber", "PML thickness in cells")
        obj.PMLCells = 8

        # Solver settings
        #
        # A time-domain run covers the whole band with one pulse, and this names
        # its envelope. A frequency-domain solve is time-harmonic and has no
        # pulse, so the property is this backend's rather than the study's. The
        # mesher does not read it, so it is not the Yee grid's either.
        obj.addProperty(
            "App::PropertyEnumeration",
            "Waveform",
            "Solver",
            "Envelope of the pulse that covers the band in one run",
        )
        obj.Waveform = list(WAVEFORMS)
        obj.addProperty(
            "App::PropertyInteger", "MaxTimesteps", "Solver", "Maximum number of timesteps"
        )
        # With EnergyDecay at 0 below, this is the run length rather than a
        # ceiling, and every step is taken. The same number is
        # openems.model.DEFAULT_TIMESTEPS, which carries what it rests on, and a
        # test keeps the two equal.
        obj.MaxTimesteps = 30000
        # Zero disables energy termination. That is the only reproducible
        # setting, and openems.policy._termination says why, and why a positive
        # level is refused.
        obj.addProperty(
            "App::PropertyFloat",
            "EnergyDecay",
            "Solver",
            "Let openEMS stop early once the energy has fallen below this level under "
            "its peak, in dB and negative: -40 is 40 dB down (0 = never). openEMS tests "
            "the level every four seconds of wall clock, so a shorter run takes every step",
        )
        obj.EnergyDecay = 0.0
        # Reaches openEMS' SetTimeStepFactor, which applies it only below 1, so
        # the default leaves the engine on its own step. Below 1 the factor
        # trades simulated time for stability: the same MaxTimesteps covers
        # proportionally less of it, and pre-flight reports that. Outside (0, 1]
        # the translation refuses. openEMS steps at full size there anyway,
        # warning on stderr below zero and saying nothing at all above one. See
        # model.check_timestep_factor.
        obj.addProperty(
            "App::PropertyFloat",
            "TimestepFactor",
            "Solver",
            "Scale openEMS' own timestep for stability; 1 = leave it alone (0, 1]",
        )
        obj.TimestepFactor = 1.0
        obj.addProperty("App::PropertyInteger", "Threads", "Solver", "Number of threads (0=auto)")
        obj.Threads = 0

        # Infrastructure.
        #
        # Every property here must reach something. A setting the user can
        # change that reaches nothing is a silent no-op. Wire it through to the
        # engine, or do not offer it.
        obj.addProperty("App::PropertyPath", "SimDir", "Infrastructure", "Simulation directory")
        # The interpreter that owns the openEMS bindings. It is not FreeCAD's,
        # which is why the solver runs as a subprocess. Empty means search:
        # $MICROWAVE_OPENEMS_PYTHON, this interpreter, then python3 on PATH,
        # each verified by importing the bindings.
        obj.addProperty(
            "App::PropertyString",
            "SolverPython",
            "Infrastructure",
            "Python with the openEMS bindings (blank = search)",
        )
        obj.SolverPython = ""

        self.declare_what_moves_no_cell(obj)

        # No mesh settings link, to either object that carries them. What the
        # device asks of any mesh is ``EMMeshPolicy`` and what this pipeline
        # does about it is ``EMYeeGrid``, and both belong to the study and are
        # found in the study's Group by kind. A link here would be a second
        # statement about membership, and one that can point into another
        # analysis.

        obj.Proxy = self


class EMSolverPalace(EMSolverBase):
    """The Palace solver: this backend's own settings, and nothing neutral.

    Nothing this object carries has a counterpart on the other solver. Palace
    solves the frequency domain on tetrahedra, so there is no timestep to scale
    and no absorber made of cells to give a depth to. There are no bindings to
    prove in an unknown Python either: Palace is one program, run as one, and
    what has to be found is a binary rather than an interpreter. What there is
    instead is the order of the polynomial the field is written in inside each
    element, and how a sweep reaches the points across the band. And there is
    where a run is written, how many processes it runs
    on, and which launcher and which mesher answer it.

    The band and the points across it are the study's own properties. What the
    device asks of any mesh is the mesh policy's and the element size asked for
    is ``EMGmshMesh``'s, and both objects are found in the study's Group.
    """

    #: Every property this class carries. What decides where an element goes is
    #: the drawing, the mesh policy and ``EMGmshMesh``, and what is here is how
    #: the field is written inside an element already placed - so a property
    #: added to this
    #: class is judged against that before it lands. The mesh preview is the
    #: other backend's grid and is not laid from this object at all, which is
    #: ``Objects/analysis.py::NOT_MESHED_FROM``.
    MOVES_NO_CELL = (
        "MPILauncher",
        "MesherPython",
        "Order",
        "Processes",
        "SimDir",
        "SolverPath",
        "Sweep",
        "SweepSolves",
        "SweepTolerance",
    )

    def __init__(self, obj):
        # A higher order carries more of the field inside each element, so it
        # is what makes the answer stop depending on the shape of the elements
        # the mesher happened to produce. Three is the default for that reason.
        # It is neither a demand the document states nor anything a user has a
        # unit to check, which is why it is here and not on a mesh object.
        obj.addProperty(
            "App::PropertyInteger",
            "Order",
            "Solver",
            "Polynomial order of the field within an element",
        )
        obj.Order = 3

        # A band of many points costs a full solve at each of them where the
        # sweep is discrete. An adaptive sweep solves a few in full and answers
        # every point from a reduced model built of them, and it is the default
        # because a study asks for hundreds of points. Where the points are no
        # more than SweepSolves, the adapter solves each in full whatever this
        # says, since that costs no more than the most an adaptive sweep may
        # take and is exact at every point.
        obj.addProperty(
            "App::PropertyEnumeration",
            "Sweep",
            "Sweep",
            "Adaptive: solve a few frequencies in full and answer every point from a "
            "reduced model built of them. Discrete: solve every point in full",
        )
        obj.Sweep = list(SWEEPS)
        obj.addProperty(
            "App::PropertyFloat",
            "SweepTolerance",
            "Sweep",
            "The field error, relative to the field, an adaptive sweep's reduced model "
            "is built to. It bounds no scattering parameter directly",
        )
        obj.SweepTolerance = 1e-3
        obj.addProperty(
            "App::PropertyInteger",
            "SweepSolves",
            "Sweep",
            "The most full solves an adaptive sweep takes for each driven port. A sweep "
            "that takes them all without converging is refused",
        )
        obj.SweepSolves = 20

        # Every property here must reach something, for the reason
        # ``EMSolverOpenEMS`` gives above. Those below are read by the panel
        # that runs this backend, ``Gui/palace_panel.py``, and by nothing that
        # translates the document.
        obj.addProperty(
            "App::PropertyInteger",
            "Processes",
            "Solver",
            "How many MPI processes Palace runs on",
        )
        # One, because Open MPI refuses more ranks than it counts slots on the
        # machine, every machine has one, and a default that fails on some
        # machine is a default nobody can rely on. A user raises it.
        obj.Processes = 1

        obj.addProperty("App::PropertyPath", "SimDir", "Infrastructure", "Simulation directory")
        # Palace is one program started through its own launcher, so what is
        # found is a file rather than an interpreter to prove. Empty means
        # search: $MICROWAVE_PALACE, then palace on PATH. A path given here is
        # used as given.
        obj.addProperty(
            "App::PropertyString",
            "SolverPath",
            "Infrastructure",
            "The palace launcher (blank = search)",
        )
        obj.SolverPath = ""
        # The palace launcher starts its processes through an MPI launcher it
        # looks for on PATH, and on macOS a FreeCAD started from Finder or the Dock is
        # given the system's PATH alone. Empty means mpirun on PATH.
        obj.addProperty(
            "App::PropertyString",
            "MPILauncher",
            "Infrastructure",
            "The mpirun Palace starts its processes with (blank = mpirun on PATH)",
        )
        obj.MPILauncher = ""
        # The mesher runs in a Python of its own, because Gmsh is a C++ library
        # that takes its process down with it. Empty means search:
        # $MICROWAVE_GMSH_PYTHON, the Python beside FreeCAD's, then python3 on
        # PATH, each proved by importing gmsh.
        obj.addProperty(
            "App::PropertyString",
            "MesherPython",
            "Infrastructure",
            "Python that can import gmsh (blank = search)",
        )
        obj.MesherPython = ""

        self.declare_what_moves_no_cell(obj)

        obj.Proxy = self


def createEMSolverOpenEMS(doc=None):
    """The openEMS solver, unowned. The analysis files it away."""
    doc = doc or FreeCAD.ActiveDocument

    obj = doc.addObject("App::FeaturePython", "EMSolverOpenEMS")
    EMSolverOpenEMS(obj)
    obj.Label = "openEMS"

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMSolverOpenEMS")
    return obj


def createEMSolverPalace(doc=None):
    """The Palace solver, unowned. The analysis files it away.

    Not created with the study. ``createEMAnalysis`` puts in the backend whose
    route from a drawing to a result is complete, so that a new study runs; a
    second solver is added by whoever wants that backend.
    """
    doc = doc or FreeCAD.ActiveDocument

    obj = doc.addObject("App::FeaturePython", "EMSolverPalace")
    EMSolverPalace(obj)
    obj.Label = "Palace"

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMSolverPalace")
    return obj
