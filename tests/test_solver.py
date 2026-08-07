# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

import pytest

from Microwave.Objects.mesh import createEMMeshPolicy
from Microwave.Objects.solver import createEMSolverOpenEMS


def solver():
    return createEMSolverOpenEMS()


def mesh_policy():
    return createEMMeshPolicy()


def test_the_solver_carries_no_neutral_property():
    """The band belongs to the study, not to a backend.

    Not a naming preference: every solver answering the same question needs the
    same band, and a copy on the openEMS object is a copy that can disagree with
    the NEC2 one. Asserted as an absence, because the way this regresses is
    somebody adding a convenience property back.
    """
    sim = solver()
    for name in ("FrequencyStart", "FrequencyStop", "NumFrequencyPoints", "Waveform"):
        assert not hasattr(sim, name), f"{name} belongs on EMAnalysis"


def test_simulation_defaults():
    sim = solver()
    # Boundaries
    faces = ["XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax"]
    for face in faces:
        assert getattr(sim, f"Boundary{face}") == "PML"
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


def test_the_mesh_policy_defaults_to_air_padding():
    mesh = mesh_policy()
    for axis in ["X", "Y", "Z"]:
        for side in ["Min", "Max"]:
            assert getattr(mesh, f"Padding{axis}{side}") == "Air"
            assert getattr(mesh, f"AirCells{axis}{side}") == 8


def test_the_default_mesh_follows_the_rule_the_gate_measured():
    """``EdgeRefinement`` is what the defaults are *for*.

    ``ElementsPerWavelength`` only means a length once a frequency and a
    permittivity exist. The refinement does not depend on either, and it is the
    quantity that was measured: at the same cell count, refining conductor edges
    by 2 instead of 6 moved the microstrip gate's extracted impedance by 1.4%,
    more than its whole tolerance. See tests/test_acceptance_microstrip.py.
    """
    mesh = mesh_policy()

    assert mesh.ElementsPerWavelength == pytest.approx(20.0)
    assert mesh.EdgeRefinement == pytest.approx(6.0)
    assert mesh.MinElementsAcross >= 9, "a thin substrate carries the whole field"
    assert mesh.MaxGrowthRatio > 1.0, "a ratio of 1 forbids grading"


def test_the_floor_on_element_size_defaults_to_derived():
    """Zero is the sentinel for "work it out", not a floor of zero."""
    assert float(mesh_policy().MinElementSize) == 0.0


def test_the_mesh_policy_speaks_no_solver_s_dialect():
    """The neutral object is read by FDTD, MoM and FEM adapters alike.

    "Cell" is FDTD's word, "segment" is MoM's, "tetrahedron" is FEM's. Naming
    the neutral layer after the first backend is what the adapter architecture
    exists to avoid, and it is the kind of thing that creeps back one property
    at a time.
    """
    mesh = mesh_policy()
    owned = [p for p in mesh.PropertiesList if p not in ("Label", "Label2")]
    for name in owned:
        lowered = name.lower()
        assert "cell" not in lowered or lowered.startswith("aircells"), name
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


@pytest.mark.parametrize("face", FACES)
def test_setting_one_boundary_leaves_the_other_five_alone(face):
    """Six faces, six independent properties.

    One test per face, not a 6x6 loop putting thirty-six assertions under one
    name, most of them exercising conftest's ``__setattr__`` rather than the
    workbench.
    """
    sim = solver()
    for other in FACES:
        setattr(sim, f"Boundary{other}", "PML")
    setattr(sim, f"Boundary{face}", "PEC")

    assert getattr(sim, f"Boundary{face}") == "PEC"
    assert [getattr(sim, f"Boundary{o}") for o in FACES if o != face] == ["PML"] * 5


def test_a_boundary_refuses_a_condition_that_is_not_one():
    with pytest.raises(ValueError, match="is not in allowed choices"):
        solver().BoundaryXMin = "Garbage"


def test_every_boundary_the_dropdown_offers_is_translated_or_refused_by_name():
    """The document offers ``Periodic``; openEMS' adapter has no word for it.

    That is allowed - P3 asks for a loud refusal, not for the two lists to be
    the same - but it has to *be* a refusal that names the property, not a
    silent substitution into something that solves. Read off the enumeration
    rather than a typed-out list, so a sixth condition cannot be added without
    this deciding which of the two it is.

    Replaces a test whose own comment said it verified the conftest stub.
    """
    from Microwave.Solvers.openems.document import TranslationError, _boundary

    sim = solver()
    offered = sim.getEnumerationsOfProperty("BoundaryXMin")
    assert set(offered) == {"PML", "PEC", "PMC", "Mur", "Periodic"}, offered

    refused, translated = {}, set()
    for condition in offered:
        for face in FACES:
            setattr(sim, f"Boundary{face}", condition)
        try:
            words = _boundary(sim)
        except TranslationError as refusal:
            refused[condition] = str(refusal)
        else:
            assert len(words) == len(FACES), condition
            translated.add(condition)

    # Which way round matters, and asserting only "one or the other" is what let
    # a mutant that translated Periodic to PEC pass: a silent PEC would wall a
    # structure that was meant to repeat, and it would solve.
    assert translated == {"PML", "PEC", "PMC", "Mur"}
    assert set(refused) == {"Periodic"}
    for condition, message in refused.items():
        assert condition in message, message
        assert "BoundaryXMin" in message, message
