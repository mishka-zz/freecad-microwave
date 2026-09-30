# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

import pytest

from Microwave.Objects.mesh import createEMMeshPolicy, createEMYeeGrid
from Microwave.Objects.solver import createEMSolverOpenEMS


def solver():
    return createEMSolverOpenEMS()


def mesh_policy():
    return createEMMeshPolicy()


def yee_grid():
    return createEMYeeGrid()


def test_the_solver_carries_no_neutral_property():
    """The band belongs to the study, not to a backend.

    Not a naming preference: every solver answering the same question needs the
    same band, and a copy on the openEMS object is a copy that can disagree with
    the NEC2 one. Asserted as an absence, because the way this regresses is
    somebody adding a convenience property back.
    """
    sim = solver()
    for name in ("FrequencyStart", "FrequencyStop", "NumFrequencyPoints"):
        assert not hasattr(sim, name), f"{name} belongs on EMAnalysis"


class TestTheExcitationDeclaration:
    """An enumeration of one, and the one is what the engine is driven with.

    The property names how a time-domain run covers the band, which a
    frequency-domain solve has no word for, so it is this backend's. What is
    forbidden is offering a choice before the adapter can honour it: a shape
    offered first changes no envelope and no answer.
    """

    def test_it_offers_only_what_the_adapter_can_produce(self):
        from Microwave.Solvers.openems import policy

        sim = solver()
        assert sim.getEnumerationsOfProperty("Waveform") == [policy.GAUSSIAN]
        assert sim.Waveform == policy.GAUSSIAN

    def test_the_study_states_no_excitation(self, doc):
        """The study is what the device is asked, and every backend answers it."""
        from Microwave.Objects.analysis import createEMAnalysis

        assert not hasattr(createEMAnalysis(doc), "Waveform")


def test_simulation_defaults():
    sim = solver()
    # The absorber: how every face the mesh policy leaves open absorbs.
    assert sim.Absorber == "PML"
    assert sim.PMLCells == 8

    # Solver
    assert sim.MaxTimesteps == 30000
    assert sim.TimestepFactor == 1.0
    assert sim.Threads == 0

    # Infrastructure
    assert sim.SimDir == ""  # blank means "beside the document"
    assert sim.SolverPython == ""  # blank means "search for the bindings"


def test_energy_termination_is_off_by_default():
    """Zero dB reads as "never stop early", and that is deliberate.

    openEMS re-evaluates its energy criterion inside a branch gated on four
    seconds of wall clock, so an energy-terminated run stops at a step count
    that depends on machine load, and repeat runs of one byte-identical
    configuration scatter across most of the gate's tolerance.
    """
    assert solver().EnergyDecay == 0.0


class TestThePalaceSolver:
    """The second backend's own object, and what reads it.

    Nothing but a string connects a property to the adapter that reads it, so
    the reading is driven rather than described. Asserting the names here alone
    would pass just as happily against a translation that reads none of them.
    """

    def solver(self):
        from Microwave.Objects.solver import createEMSolverPalace

        return createEMSolverPalace()

    def test_it_carries_no_neutral_property(self):
        """The band belongs to the study. Two solvers in one study make that
        load-bearing rather than tidy: a copy here is a copy that can disagree
        with the band the other backend answers, over one drawing."""
        sim = self.solver()
        for name in ("FrequencyStart", "FrequencyStop", "NumFrequencyPoints"):
            assert not hasattr(sim, name), f"{name} belongs on EMAnalysis"

    def test_it_carries_nothing_of_the_other_backend_s(self):
        """A timestep, an absorber counted in cells and an interpreter to find
        are FDTD's and are meaningless here. Asserted as an absence, because the
        way this regresses is somebody copying the class that was already
        there."""
        sim = self.solver()
        for name in (
            "Absorber",
            "EnergyDecay",
            "MaxTimesteps",
            "PMLCells",
            "SolverPython",
            "Threads",
            "TimestepFactor",
            "Waveform",
        ):
            assert not hasattr(sim, name), f"{name} is the other backend's word"

    def test_it_carries_nothing_nobody_reads(self):
        """A setting that reaches nothing is a silent no-op. This pins the set;
        that each is read is asserted where it is read - the order and the sweep by the
        adapter, below, and the rest by the panel that runs this backend, in
        ``tests/test_palace_panel.py``. Stated as the whole set, so a property
        added here fails until somebody names it beside the code that reads it.
        FreeCAD's own bookkeeping is taken out, which is what
        ``Objects/staleness.py`` collects it for."""
        from Microwave.Objects.staleness import FREECADS_OWN

        carried = set(self.solver().PropertiesList) - set(FREECADS_OWN)
        assert carried == {
            "MPILauncher",
            "MesherPython",
            "Order",
            "Processes",
            "SimDir",
            "SolverPath",
            "Sweep",
            "SweepSolves",
            "SweepTolerance",
        }

    def test_it_runs_on_one_process_unless_told_otherwise(self):
        """Open MPI refuses more ranks than it counts slots on the machine, and
        every machine has one. A default that fails somewhere is not one
        anybody can rely on."""
        assert self.solver().Processes == 1

    def test_the_launchers_and_the_mesher_are_searched_for_unless_named(self):
        sim = self.solver()
        assert sim.SolverPath == ""
        assert sim.MPILauncher == ""
        assert sim.MesherPython == ""

    def test_nothing_it_carries_moves_a_cell(self):
        """A drift guard, and what it guards is a judgement. What decides where
        an element goes is the drawing and the mesh policy, and what is here is
        how the field is written inside an element already placed - so the
        declaration covers the whole set. Adding a property fails this until
        somebody has decided which side of that it falls on."""
        from Microwave.Objects.staleness import FREECADS_OWN

        sim = self.solver()
        carried = set(sim.PropertiesList) - set(FREECADS_OWN)
        assert set(type(sim.Proxy).MOVES_NO_CELL) == carried

    def test_the_adapter_reads_the_order_off_the_object(self):
        """The value, not the default: comparing three against three passes on
        a translation that hard-codes it and never reads the property."""
        from Microwave.Solvers.palace.document import _order

        sim = self.solver()
        assert _order(sim) == sim.Order
        sim.Order = 4
        assert _order(sim) == 4

    def test_the_order_it_ships_is_the_one_that_absorbs_the_mesh(self):
        """The default, so a change to it fails somewhere cheap. Which order
        stops the mesh deciding the answer is measured rather than derived, and
        the measurement is not in this repository."""
        assert self.solver().Order == 3

    def test_the_adapter_refuses_an_order_that_is_not_one(self):
        """The property is an integer the user may type into, so the floor is
        the translation's rather than the editor's."""
        from Microwave.Solvers.errors import TranslationError
        from Microwave.Solvers.palace.document import _order

        sim = self.solver()
        sim.Order = 0
        with pytest.raises(TranslationError, match="Order is 0"):
            _order(sim)


def test_the_mesh_policy_defaults_to_air_padding():
    """The policy says which faces are open to free space and how far the air
    reaches past them, and 0 is the length derived from the band."""
    mesh = mesh_policy()
    for axis in ["X", "Y", "Z"]:
        for side in ["Min", "Max"]:
            assert getattr(mesh, f"Padding{axis}{side}") == "Air"
    assert float(mesh.Clearance) == 0.0


def test_the_default_mesh_follows_the_rule_the_gate_measured():
    """The values, so a change to one of them fails somewhere cheap.

    ``ElementsPerWavelength`` only means a length once a frequency and a
    permittivity exist; the refinement means the same thing whatever the model
    is, which is why it is the one a gate can be solved at several values of.
    ``tests/test_acceptance_openems_microstrip.py`` is that gate, and it
    prints what coarsening this refinement costs a line whose impedance is
    extracted.

    That gate writes the refinement out as a constant of its own, so the two are
    compared here rather than left to agree. Without it the gate goes on
    measuring a refinement nobody ships the moment this default moves, and every
    figure it prints is about a policy that is no longer the policy.
    """
    from tests.test_acceptance_openems_microstrip import REFINEMENT

    mesh, grid = mesh_policy(), yee_grid()

    assert grid.ElementsPerWavelength == pytest.approx(20.0)
    assert grid.EdgeRefinement == pytest.approx(6.0)
    assert grid.EdgeRefinement == pytest.approx(REFINEMENT), (
        "the microstrip gate solves its ladder around a refinement this grid no "
        "longer ships, so what it prints is not what a user gets"
    )
    assert mesh.MinElementsAcross >= 9, "a thin substrate carries the whole field"
    assert grid.MaxGrowthRatio > 1.0, "a ratio of 1 forbids grading"


def test_the_floor_on_element_size_defaults_to_derived():
    """Zero is the sentinel for "work it out", not a floor of zero."""
    assert float(mesh_policy().MinElementSize) == 0.0


def test_the_mesh_policy_speaks_no_solver_s_dialect():
    """The neutral object is read by FDTD, MoM and FEM adapters alike.

    "Cell" is FDTD's word, "segment" is MoM's, "tetrahedron" is FEM's. Naming
    the neutral layer after the first backend is what the adapter architecture
    exists to avoid, and it is the kind of thing that creeps back one property
    at a time.

    No carve-out: a count of cells is the Yee grid's primitive and the grid is
    named for that method, so the rule applies to the policy whole.
    """
    mesh = mesh_policy()
    owned = [p for p in mesh.PropertiesList if p not in ("Label", "Label2")]
    for name in owned:
        lowered = name.lower()
        assert "cell" not in lowered, name
        assert "segment" not in lowered and "tet" not in lowered, name


class TestTheRefinementObject:
    """``EMMeshRegion``'s properties and the adapter that reads them.

    Nothing but a string connects the two halves, so this drives the real
    object through the real translation. Asserting the property *names* here
    instead would pass just as happily with a translation that reads none of
    them.
    """

    def _region(self, lower, upper, **overrides):
        from Microwave.Objects.mesh import createEMMeshRegion

        class BoundBox:
            def __init__(self, lower, upper):
                (self.XMin, self.YMin, self.ZMin) = lower
                (self.XMax, self.YMax, self.ZMax) = upper

        class Shape:
            def __init__(self, lower, upper):
                self.BoundBox = BoundBox(lower, upper)

        class Target:
            Label = "Pad"
            Name = "Pad"

        Target.Shape = Shape(lower, upper)

        obj = createEMMeshRegion()
        obj.References = [(Target(), [])]
        for name, value in overrides.items():
            setattr(obj, name, value)
        return obj

    def test_a_created_region_translates_to_a_sizing_region(self):
        from Microwave.Solvers.openems import document

        obj = self._region(
            (-1.0, -2.0, 0.0), (1.0, 2.0, 1.6), ElementSize=0.05, MinElementsAcross=6
        )
        (sizing,) = document._sizing_regions([obj])

        assert sizing.lower == (-1.0, -2.0, 0.0)
        assert sizing.upper == (1.0, 2.0, 1.6)
        assert sizing.size == pytest.approx(0.05)
        # A value, not the default: comparing 0 against 0 passes on a
        # translation that hard-codes it and never reads the property.
        assert sizing.min_lines == 6
        assert "Pad" in sizing.label, "an error must be able to name what it hit"

    def test_it_is_enabled_when_created(self):
        """A region created and then ignored until Enabled was ticked would
        look like a mesher that does not work."""
        assert self._region((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)).Enabled is True

    def test_it_speaks_no_solver_s_dialect(self):
        """Neutral, like every object in this layer. FDTD cells, MoM segments
        and FEM tetrahedra are all "elements" here."""
        obj = self._region((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0))
        for name in obj.PropertiesList:
            lowered = name.lower()
            assert "cell" not in lowered, name
            assert "segment" not in lowered and "tet" not in lowered, name


FACES = ("XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax")


def test_the_solver_states_no_condition_on_any_face():
    """Which face is a wall and which absorbs follows from the mesh policy's
    Padding, which every backend reads. Asserted as an absence, because the way
    this regresses is a per-face property coming back beside the policy's and
    disagreeing with it."""
    sim = solver()
    for face in FACES:
        assert not hasattr(sim, f"Boundary{face}"), face


def test_every_absorber_the_dropdown_offers_is_built():
    """Read off the enumeration rather than a typed-out list, so an absorber
    cannot be offered without this deciding what the adapter builds for it -
    and the adapter's own copy of the list is held to the object's."""
    from Microwave.Objects.solver import ABSORBERS
    from Microwave.Solvers.openems import policy

    sim = solver()
    offered = sim.getEnumerationsOfProperty("Absorber")
    assert tuple(offered) == ABSORBERS == policy.ABSORBERS

    built = {}
    for kind in offered:
        sim.Absorber = kind
        built[kind] = policy._absorbing(sim)
    # PML is PMLCells deep, and Mur is openEMS' word for its own condition.
    assert built == {"PML": f"PML_{sim.PMLCells}", "Mur": "MUR"}


def test_an_absorber_this_adapter_does_not_build_is_refused_by_name():
    """A file can offer a longer list than the class does, since FreeCAD
    restores an enumeration's choices from the document."""
    from Microwave.Solvers.openems import policy
    from Microwave.Solvers.openems.properties import TranslationError

    class Saved:
        Label = "openEMS"
        Absorber = "Periodic"
        PMLCells = 8

    with pytest.raises(TranslationError, match="Absorber is 'Periodic'"):
        policy._absorbing(Saved())
