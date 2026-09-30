# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Acceptance gate: a strip dipole in a medium answers as the dipole in vacuum.

Two sheets of perfect conductor end to end with a gap between them and one
lumped port across the gap, open on every face, solved twice on one grid: in
vacuum, and in a medium of relative permittivity four over half the band with
half the port's resistance. Maxwell's equations scale exactly that way, so the
two are one dipole the same wavelengths long, and the two matrices are one.

What this gate exercises
------------------------

The medium as the driver lays it: one box of its material under every solid,
reaching past the grid, so the room round the dipole and the perfectly matched
layer beyond it are the medium. A medium laid short of the absorber would leave
a vacuum layer the wave reflects off, and one the layer did not follow would
reflect at the layer's face.

What it is held to
------------------

What moving the absorber moves the answer. A third run stands the absorber
nearer the vacuum dipole, and the medium's matrix is held closer to the vacuum
one than that.

The two are not one system of updates to the last place. What does not scale
parts them, among it the layer's grading, which openEMS computes from the
impedance of vacuum (``FDTD/extensions/operator_ext_upml.cpp``,
``Operator_Ext_UPML::BuildExtension``).

The medium's pulse lasts longer by the root of its permittivity, and pre-flight
estimates a run's length from the grid's step in vacuum, so the medium's run is
given that many times the steps. openEMS takes its step from the cells'
materials and steps the medium longer, so that run records longer than it
needs, which moves nothing the gate reads.

It solves a comparison, so it runs before a release. Before a commit it checks
that the two drawings are one grid, which costs no solve.
"""

from __future__ import annotations

import numpy as np
import pytest

from Microwave.Results.sparameters import SParameters
from Microwave.Solvers.openems import plan, preflight, read, run, write
from Microwave.Solvers.openems.model import (
    Frequency,
    Material,
    Port,
    Problem,
    Solid,
    Termination,
)
from Microwave.Solvers.openems.regions import MeshParams
from Microwave.units import SPEED_OF_LIGHT

pytestmark = pytest.mark.slow

#: The dipole, in mm: tip to tip, the width of each strip, and the gap the port
#: lies across.
LENGTH = 48.0
WIDTH = 2.0
GAP = 1.0

#: The band in vacuum, in Hz, and the port's resistance there, in ohms. The band
#: holds the half-wave resonance of a dipole this long, and is wide enough that
#: the pulse exciting it is short against the run.
FREQ_MIN = 1.5e9
FREQ_MAX = 4.5e9
RESISTANCE = 73.0

#: The medium's relative permittivity. Its root is two, so the band and the
#: resistance it divides stay exact.
MEDIUM = 4.0

#: How far the absorber stands from the dipole, in mm, and the nearer distance
#: the scale is taken from.
CLEARANCE = 60.0
NEARER = 30.0

#: Cells across a wavelength at the top of the band, and how much finer a
#: conductor's edge is.
PER_WAVELENGTH = 20.0
EDGE = 6.0

#: Steps of a run in vacuum. See the module's docstring for the medium's.
TIMESTEPS = 80000


def _problem(scale: float, clearance: float) -> Problem:
    """The dipole in a medium of ``scale`` squared, over the band divided by
    ``scale``, with the absorber ``clearance`` from it."""
    top = FREQ_MAX / scale
    cap = SPEED_OF_LIGHT / top * 1e3 / PER_WAVELENGTH
    bulk = cap / scale
    materials = [Material(name="Arms", kind="pec")]
    medium = ""
    if scale != 1.0:
        materials.append(Material(name="Medium", kind="dielectric", epsilon=scale**2))
        medium = "Medium"
    half = LENGTH / 2
    solids = (
        Solid(
            material="Arms",
            lower=(-half, -WIDTH / 2, 0.0),
            upper=(-GAP / 2, WIDTH / 2, 0.0),
            priority=5,
            label="ArmLeft",
        ),
        Solid(
            material="Arms",
            lower=(GAP / 2, -WIDTH / 2, 0.0),
            upper=(half, WIDTH / 2, 0.0),
            priority=5,
            label="ArmRight",
        ),
    )
    resistance = RESISTANCE / scale
    ports = (
        Port(
            number=1,
            kind="lumped",
            start=(-GAP / 2, -WIDTH / 2, 0.0),
            stop=(GAP / 2, WIDTH / 2, 0.0),
            propagation_axis=1,
            excitation_axis=0,
            excite=True,
            feed_resistance=resistance,
            reference_impedance=resistance,
            label="Feed",
        ),
    )
    params = MeshParams(
        metal_res=bulk / EDGE,
        dielectric_res=bulk,
        max_ratio=(1.4, 1.4, 1.4),
        pml_cells=8,
        cap=cap,
        medium=scale**2,
    )
    grid = plan.plan_grid(solids, ports, tuple(materials), params, ((clearance,) * 2,) * 3)
    return Problem(
        title="strip dipole",
        frequency=Frequency(start=FREQ_MIN / scale, stop=top, points=41),
        grid=grid,
        materials=tuple(materials),
        solids=solids,
        ports=ports,
        boundary=("PML_8",) * 6,
        termination=Termination(max_timesteps=round(TIMESTEPS * scale), end_criteria=0.0),
        medium=medium,
    )


@pytest.fixture(scope="module")
def paired(interpreter, tmp_path_factory):
    """S11 of the vacuum dipole, the dipole in the medium, and the vacuum dipole
    with its absorber nearer, by name."""
    answers = {}
    for name, scale, clearance in (
        ("vacuum", 1.0, CLEARANCE),
        ("medium", MEDIUM**0.5, CLEARANCE),
        ("nearer", 1.0, NEARER),
    ):
        problem = _problem(scale, clearance)
        preflight.refuse_if_blocked(preflight.check(problem))
        envelope = write.write(problem, tmp_path_factory.mktemp(f"dipole_{name}"))
        run.run(envelope, interpreter=interpreter)
        solved = read.read(envelope.parent)
        matrix = SParameters.from_runs([solved], reference=RESISTANCE / scale)
        answers[name] = (problem, matrix.parameter(1, 1))
    return answers


def test_the_medium_is_meshed_as_the_vacuum_it_scales_to():
    """The same wavelengths in the medium are the same cells, so the grid laid
    from the medium's band and the vacuum's is one grid."""
    vacuum = _problem(1.0, CLEARANCE)
    medium = _problem(MEDIUM**0.5, CLEARANCE)
    assert medium.medium == "Medium"
    for dim in range(3):
        assert np.allclose(medium.grid[dim], vacuum.grid[dim], rtol=0.0, atol=1e-9)


def test_a_dipole_in_a_medium_answers_as_the_dipole_in_vacuum(paired):
    """See the module's docstring."""
    _, vacuum = paired["vacuum"]
    _, medium = paired["medium"]
    _, nearer = paired["nearer"]
    parted = float(np.abs(medium - vacuum).max())
    moved = float(np.abs(nearer - vacuum).max())
    print(
        f"\nGATE strip dipole on openEMS in a medium of relative permittivity {MEDIUM:g}: "
        f"S11 parts from the vacuum dipole's by {parted:.2g}, where moving the absorber from "
        f"{CLEARANCE:g} to {NEARER:g} mm moves it {moved:.2g}"
    )
    assert parted < moved, (
        f"the dipole in the medium parts from the vacuum one by {parted:.3g}, as much as "
        f"moving the absorber moves it, {moved:.3g}: the medium is not what the run solved"
    )
