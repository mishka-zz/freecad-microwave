# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

import FreeCAD

from ._vp_hook import ViewProviderRestored


class EMSolverOpenEMS(ViewProviderRestored):
    """The openEMS solver: this backend's own settings, and nothing neutral.

    FEM calls the same thing ``solverbase.Proxy``. The frequency band and the
    waveform sit on :class:`~.analysis.EMAnalysis` instead. The band a device is
    characterised over is a statement about the problem, and every solver
    answering it needs the same one.

    What is left is FDTD vocabulary throughout: PML depth, timesteps, a
    stability factor, the interpreter that owns the bindings. A property belongs
    here when it is written in that vocabulary. Boundary conditions are the same
    case - "PML with 8 cells" is not a neutral concept, so it lives on the
    solver and each adapter exposes its own.
    """

    #: What the mesher never reads. The boundaries and the PML depth are not
    #: here: an absorber is cells, and changing a wall to PEC takes them away.
    #: Measured over the shipped examples and a WR-90 part - see
    #: ``Objects/staleness.py`` for what the declaration is for.
    MOVES_NO_CELL = (
        "EnergyDecay",
        "MaxTimesteps",
        "SimDir",
        "SolverPython",
        "Threads",
        "TimestepFactor",
    )

    def __init__(self, obj):
        # Boundary conditions
        boundaries = ["PML", "PEC", "PMC", "Mur", "Periodic"]
        for axis in ["X", "Y", "Z"]:
            for side in ["Min", "Max"]:
                pname = f"Boundary{axis}{side}"
                obj.addProperty(
                    "App::PropertyEnumeration",
                    pname,
                    "Boundaries",
                    f"Boundary condition on {axis}{side}",
                )
                setattr(obj, pname, boundaries)

        obj.addProperty("App::PropertyInteger", "PMLCells", "Boundaries", "PML thickness in cells")
        obj.PMLCells = 8

        # Solver settings
        obj.addProperty(
            "App::PropertyInteger", "MaxTimesteps", "Solver", "Maximum number of timesteps"
        )
        # With EnergyDecay at 0 below, this is the run length rather than a
        # ceiling, and every step is taken. The same number is
        # openems.model.DEFAULT_TIMESTEPS, which carries what it rests on, and a
        # test keeps the two equal.
        obj.MaxTimesteps = 30000
        # Zero disables energy termination. That is the only reproducible
        # setting, and openems.policy._termination says why.
        obj.addProperty(
            "App::PropertyFloat",
            "EnergyDecay",
            "Solver",
            "Stop early once the energy falls this far below its peak, in dB (0 = never)",
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

        # No MeshSettings link. Mesh policy is neutral - ``EMMeshPolicy``, read
        # by every backend - so it belongs to the study and is found in the
        # study's Group. A link here would be a second statement about
        # membership, and one that can point into another analysis.

        obj.Proxy = self

    def execute(self, obj):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


def createEMSolverOpenEMS(doc=None):
    """The openEMS solver, unowned. The analysis files it away."""
    doc = doc or FreeCAD.ActiveDocument

    obj = doc.addObject("App::FeaturePython", "EMSolverOpenEMS")
    EMSolverOpenEMS(obj)
    obj.Label = "openEMS"

    from ._vp_hook import inject_view_provider

    inject_view_provider(obj, "EMSolverOpenEMS")
    return obj
