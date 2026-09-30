# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A sheet of metal running into a wave port's face, on Palace.

Palace solves a port's mode on the port's face alone, with the metal meeting it
as the walls of that cross-section, and takes the mode it ranks first. A sheet
standing clear of the walls, or a perfect conductor touching the face at a
point, makes it a guide of two conductors, and the port drives the wave between
them rather than TE10: refused once the mesh exists, naming the port and the
sheet. A sheet dividing the face into guides that each carry a wave leaves the
port taking one of them while what the others carry goes missing from the
matrix: refused by the port's mode run before the band is solved, naming the
sheet as what divides the face.

A sheet joined to the wall that divides nothing - a fin off the floor, end to
end - makes a ridged guide, whose own mode is the one the port takes. That is
solved, and held to conserving power: the metal is perfect and each port absorbs
its own mode, so what the two ports see between them is what went in.

A sheet of brass touching the face at a point divides nothing and adds no
conductor there, since its condition is one along a curve. It is solved, and
the port takes up part of the field it leaves at the face: the run warns of
what the matrix leaves unaccounted for, naming that port.

What these drawings do not reach is a face that is one guide with one conductor
round it and still carries a second propagating mode, which the topology of the
face cannot tell.

Why a subprocess
----------------

For the reason ``tests/test_acceptance_palace_waveguide.py`` gives: the
drawing needs the CAD kernel, and ``tests/palace_metal_at_a_port_probe.py``
runs under ``freecadcmd`` to draw it and drive the run. A machine without
Palace or Gmsh makes the probe say which, and the test skips.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from Microwave.Solvers.palace.balance import WARNING
from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_metal_at_a_port_probe.py")

#: How far the power that came back out of the ridged guide may fall short of
#: what went in. The metal is perfect and each port absorbs its own mode, so
#: what falls short is the discretisation's.
POWER_TOLERANCE = 1e-3

#: What the probe draws, as it names each drawing: the sheets that divide a
#: port's face, those standing on it apart from the wall, the one joined to the
#: wall that divides nothing, and the brass one touching the face at a point.
DIVIDING = ("through", "brass_through", "to_the_middle")
APART = ("strip", "corner")
RIDGED = ("fin",)
ABSORBED = ("brass_corner",)


@pytest.fixture(scope="module")
def run_manifest(tmp_path_factory):
    """Every drawing, driven under a real FreeCAD, as the probe left them."""
    out = tmp_path_factory.mktemp("palace_port_metal")
    manifest = probe_manifest(PROBE, out, "METAL_AT_A_PORT_OUT", key=None)
    if manifest.get("missing"):
        pytest.skip(f"this machine cannot drive Palace: {manifest['missing']}")
    drawn = [*DIVIDING, *APART, *RIDGED, *ABSORBED]
    assert sorted(manifest["cases"]) == sorted(drawn), (
        f"the probe wrote {manifest['cases']} rather than {drawn}, so it stopped "
        "before every run finished"
    )
    return manifest


def refused(run_manifest, drawing):
    """What the run said about a drawing, having checked it said it before the band ran."""
    case = run_manifest[drawing]
    assert case["said"], f"{drawing} was solved, and its port's face is not one guide"
    assert case["meshed"] and not case["solved"], (
        f"{drawing} meshed {case['meshed']} and solved {case['solved']}: the refusal "
        "comes once the mesh exists and before the band is solved"
    )
    said = case["said"]
    assert "'Port1'" in said and repr(run_manifest["sheet"]) in said, said
    return said


@pytest.mark.parametrize("drawing", DIVIDING)
def test_a_sheet_dividing_a_ports_face_is_refused_naming_the_port_and_the_sheet(
    run_manifest, drawing
):
    assert "divides, whose parts carry 2 modes" in refused(run_manifest, drawing)


@pytest.mark.parametrize("drawing", APART)
def test_a_sheet_clear_of_the_walls_at_a_ports_face_is_refused_as_a_second_conductor(
    run_manifest, drawing
):
    assert "2 pieces" in refused(run_manifest, drawing)


@pytest.mark.parametrize("drawing", RIDGED)
def test_a_fin_off_the_wall_through_the_ports_is_solved_and_loses_nothing(run_manifest, drawing):
    case = run_manifest[drawing]
    assert not case["said"], case["said"]
    s = np.asarray(case["real"]) + 1j * np.asarray(case["imaginary"])
    driven = case["driven"].index(1)
    came_back = sum(np.abs(s[:, case["out"].index(port), driven]) ** 2 for port in (1, 2))
    worst = float(np.max(np.abs(1.0 - came_back)))
    print(f"\nWR-42 with a sheet at a port on Palace ({drawing}): power lost {worst:.2e}")
    assert worst < POWER_TOLERANCE, (
        f"{drawing} returned {came_back} of the power that went in: the ports are not "
        "absorbing the mode the guide carries"
    )


@pytest.mark.parametrize("drawing", ABSORBED)
def test_brass_touching_a_ports_face_at_a_point_is_solved_and_warned_of(run_manifest, drawing):
    case = run_manifest[drawing]
    assert not case["said"], case["said"]
    (line,) = case["stated"]
    print(f"\nWR-42 with a sheet at a port on Palace ({drawing}): {line}")
    assert line.startswith(WARNING) and "Driven from 'Port1'" in line, line
    assert "through 'Port1'" in line, line
