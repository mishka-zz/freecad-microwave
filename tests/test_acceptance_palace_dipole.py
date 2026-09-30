# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Acceptance gate: a strip dipole in free space, answered by Palace.

Two sheets of perfect conductor end to end with a gap between them, one lumped
port across the gap, and every face of the domain set to ``Air``. Nothing is
bound to a dielectric, so the air the adapter reserves round the sheets is the
whole region, and the power the port puts in leaves by two ways only: back
through the port, and out through the open surface.

What this gate exercises
------------------------

A study open to free space on the second backend: the reserved air, its
absorbing condition, the power Palace measures through it, and the reading of
that power back into the run. A drawing of metal alone, which a closed study
refuses for having no region, stands in the air as its region.

What is held
------------

Conservation, at every sample. The sheets are perfect conductors and the air is
vacuum, so nothing dissipates. What the matrix reflects and what leaves through
the open surface add to what went in, and ``1 - |S11|^2 - P_open`` is the share
neither accounts for. It is held to ``balance.BAR``, the share the run itself
warns past. An open surface that is not written, or written without its flux,
leaves the whole radiated share unaccounted for. A flux read with the wrong sign
leaves twice it.

What is printed and not held
----------------------------

The resonance, where the input reactance crosses zero. The closed forms for a
strip dipole - induced EMF on a wire of a quarter of the strip's width, fed
across a gap of no width - carry no stated accuracy at this width and gap, so no
tolerance for them can be derived, and the resonance is printed for a reader.

The estimate the run states of how far the open surface moved the matrix. It is
formed from the larger of the dipole mode's reflection and the slanted wave's.
The dipole mode's falls as the surface stands farther off, and no distance
lowers the slanted wave's, so from this clearance outward the slanted wave's is
the larger and the estimate stays near it times the radiated share. Two
clearances compared against it would pass by far more than any move the surface
makes, and would hold nothing. It is printed, and the gate asserts only that the
run states it.

What the release run adds
-------------------------

The dipole in a medium of relative permittivity four, over half the band with
half the resistance. Maxwell's equations scale exactly, so that dipole is the
vacuum one: the same wavelengths long in its medium, meshed to the same
elements, answering the same matrix. It holds that the medium fills the reserved
air, that the mesh is laid at its wavelength, and that the absorbing condition
takes it: the two meshes are one mesh, and the two matrices part by less than
moving the open surface nearer moves the vacuum one. A medium the condition did
not take would reflect at the surface, and a mesh laid at the vacuum's wavelength
would be another mesh.

The other comparisons a release run could add, the dipole at a second clearance
or on a finer mesh, hold nothing against the estimate, as above, and against a
closed form there is no tolerance to hold them to. So the gate itself solves one
mesh, before a commit, and the second clearance is solved only as the scale the
medium is held to.

Why a subprocess
----------------

For the reason ``tests/test_acceptance_palace_waveguide.py`` gives: the
drawing needs the CAD kernel, and ``tests/palace_dipole_probe.py`` runs under
``freecadcmd`` to draw it and drive the run. A machine without Palace or Gmsh
makes the probe say which, and the gate skips.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from Microwave.Solvers.palace import balance
from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_dipole_probe.py")

#: How far the open surface stands from the dipole, in mm: half a wavelength at
#: the bottom of the band.
CLEARANCE = 60.0

#: The mesh, in elements per wavelength at the top of the band.
ONE_MESH = 6.0

#: The run, by the name the probe writes it under.
CASE = f"dipole_{CLEARANCE:g}mm_{ONE_MESH:g}_per_wavelength"

#: The medium's relative permittivity in the release run. Its root is two, so
#: the band and the resistance it divides stay exact.
MEDIUM = 4.0

#: The nearer clearance the release run moves the open surface to, in mm.
NEARER = 30.0


def _solved(tmp_path_factory, clearance, permittivity=1.0):
    """The dipole on :data:`ONE_MESH` at ``clearance`` in a medium of
    ``permittivity``, as the probe left it."""
    case = f"dipole_{clearance:g}mm_{ONE_MESH:g}_per_wavelength"
    if permittivity != 1.0:
        case += f"_in_{permittivity:g}"
    manifest = probe_manifest(
        PROBE,
        tmp_path_factory.mktemp("palace_dipole"),
        "DIPOLE_PALACE_OUT",
        key=None,
        environment={
            "DIPOLE_PALACE_CLEARANCES": f"{clearance:g}",
            "DIPOLE_PALACE_MESHES": f"{ONE_MESH:g}",
            "DIPOLE_PALACE_MEDIUM": f"{permittivity:g}",
        },
    )
    if manifest.get("missing"):
        pytest.skip(f"this machine cannot drive Palace: {manifest['missing']}")
    assert manifest["cases"] == [case], (
        f"the probe wrote {manifest['cases']} rather than {[case]}, so it stopped "
        "before every run finished. A gate that is not run is not a gate that passed"
    )
    run = manifest[case]
    assert (run["out"], run["driven"]) == ([1], [1])
    run["s11"] = (np.asarray(run["real"]) + 1j * np.asarray(run["imaginary"]))[:, 0, 0]
    run["frequency"] = np.asarray(run["frequency"], dtype=float)
    run["resistance"] = manifest["resistance"]
    return run


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """The dipole on :data:`ONE_MESH` at :data:`CLEARANCE`."""
    return _solved(tmp_path_factory, CLEARANCE)


@pytest.fixture(scope="module")
def in_a_medium(tmp_path_factory):
    """The same dipole in :data:`MEDIUM`, scaled to it."""
    return _solved(tmp_path_factory, CLEARANCE, MEDIUM)


@pytest.fixture(scope="module")
def nearer(tmp_path_factory):
    """The vacuum dipole with its open surface at :data:`NEARER`."""
    return _solved(tmp_path_factory, NEARER)


def described(run: dict) -> str:
    return (
        f"{run['elements_per_wavelength']:g} elements per wavelength asked "
        f"{run['element']:.4g} mm, edge refinement {run['edge_refinement']:g}, order "
        f"{run['order']}, {run['unknowns']} unknowns, open surface "
        f"{run['clearance']:g} mm from the dipole, {run['seconds']:.0f} s"
    )


def test_power_is_conserved(solved):
    """What the port reflects and what leaves through the open surface add to what
    went in, at every sample."""
    assert solved["radiated"] is not None, "the run measured no power through an open surface"
    radiated = np.asarray(solved["radiated"], dtype=float)[:, 0]
    left = 1.0 - np.abs(solved["s11"]) ** 2 - radiated
    worst = int(np.argmax(np.abs(left)))
    print(
        f"\nGATE strip dipole on Palace ({described(solved)}): 1 - |S11|^2 - P_open = "
        + ", ".join(
            f"{value:+.2e} at {f / 1e9:g} GHz" for value, f in zip(left, solved["frequency"])
        )
        + f"; P_open {radiated.min():.4f} to {radiated.max():.4f}"
    )
    assert abs(left[worst]) <= balance.BAR, (
        f"at {solved['frequency'][worst] / 1e9:g} GHz the reflection and the open surface "
        f"leave {left[worst]:+.3g} of the power unaccounted for, past the {balance.BAR:g} "
        "the run warns at, in a model where nothing dissipates"
    )


def test_the_resonance_is_printed(solved):
    """Where the input reactance crosses zero, interpolated linearly between the
    samples either side. Not held: see the module's docstring."""
    s11 = solved["s11"]
    impedance = solved["resistance"] * (1 + s11) / (1 - s11)
    frequency = solved["frequency"]
    reactance = impedance.imag
    crossings = [
        index for index in range(frequency.size - 1) if reactance[index] <= 0 < reactance[index + 1]
    ]
    if not crossings:
        print(
            f"\nGATE strip dipole on Palace ({described(solved)}): the input reactance does "
            f"not cross zero between {frequency[0] / 1e9:g} and {frequency[-1] / 1e9:g} GHz"
        )
        return
    index = crossings[0]
    share = -reactance[index] / (reactance[index + 1] - reactance[index])
    resonance = frequency[index] + share * (frequency[index + 1] - frequency[index])
    resistance = impedance.real[index] + share * (impedance.real[index + 1] - impedance.real[index])
    print(
        f"\nGATE strip dipole on Palace ({described(solved)}): resonance "
        f"{resonance / 1e9:.4f} GHz, input resistance there {resistance:.2f} ohm"
    )


def test_the_run_states_its_estimate_of_the_open_surface(solved):
    """The estimate is printed and not held against anything: see the module's
    docstring. That the run states it is held."""
    assert solved["opened"], "the run did not state where the open surface stands"
    assert solved["estimate"] is not None, (
        "the run stated no estimate of how far the open surface moved the matrix"
    )
    print(
        f"\nGATE strip dipole on Palace ({described(solved)}): the run estimates the open "
        f"surface moved S11 by {solved['estimate']:.2g}"
    )


@pytest.mark.release
def test_a_medium_the_run_is_scaled_to_answers_as_vacuum(solved, in_a_medium, nearer):
    """The medium fills the reserved air, the mesh is laid at its wavelength, and
    the absorbing condition takes it. See the module's docstring."""
    assert in_a_medium["frequency"] == pytest.approx(solved["frequency"] / MEDIUM**0.5, rel=1e-12)
    assert (in_a_medium["element"], in_a_medium["unknowns"]) == (
        solved["element"],
        solved["unknowns"],
    ), "the medium was meshed to another size than the vacuum it scales to"
    parted = float(np.abs(in_a_medium["s11"] - solved["s11"]).max())
    moved = float(np.abs(nearer["s11"] - solved["s11"]).max())
    print(
        f"\nGATE strip dipole on Palace in a medium of relative permittivity {MEDIUM:g} "
        f"({described(in_a_medium)}): S11 parts from the vacuum dipole's by {parted:.2g}, "
        f"where moving the open surface from {CLEARANCE:g} to {NEARER:g} mm moves it {moved:.2g}"
    )
    assert parted < moved, (
        f"the dipole in the medium parts from the vacuum one by {parted:.3g}, as much as "
        f"moving the open surface moves it, {moved:.3g}: the medium is not what the run solved"
    )
