# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How many modes a port's face carries, on Palace.

A wave port absorbs the one mode it was asked for. A second mode propagating at
its face carries power the matrix never counts, and nothing in the matrix says
so: two fins of unequal length, a strip joined to a side wall, a notched septum,
a square guide with a plate in it, WR-42 above its TE20 cutoff with a strip off
the centre line. Every port face there is one guide with one conductor round
it, so nothing in the drawing's topology separates them from a guide that
answers right. The modes of each port's face do: a boundary mode run Palace
makes on the face at the top of the band, before the band is solved, and a face
carrying more than one mode is refused naming the port and the wave numbers.

A sheet 2.5 mm from a side wall divides each port face into two guides, and the
narrower is below cutoff across the band, so each face carries one mode. That
is solved, and held to conserving power: the metal is perfect and each port
absorbs its own mode, so what the two ports see between them is what went in.

Why a subprocess
----------------

For the reason ``tests/test_acceptance_palace_waveguide.py`` gives: the
drawing needs the CAD kernel, and ``tests/palace_port_modes_probe.py`` runs
under ``freecadcmd`` to draw it and drive the run. A machine without Palace
or Gmsh makes the probe say which, and the test skips.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_port_modes_probe.py")

#: How far the power that came back out of the guide carrying one mode may fall
#: short of what went in. The metal is perfect and each port absorbs its own
#: mode, so what falls short is the discretisation's.
POWER_TOLERANCE = 1e-3

#: What the probe draws, as it names each drawing: the guides whose port face
#: carries more than one mode at the top of the band, and the one that carries
#: one.
SEVERAL = ("unequal_fins", "strip_joined", "notched_septum", "square_plate", "above_te20")
ONE = ("off_centre",)


@pytest.fixture(scope="module")
def run_manifest(tmp_path_factory):
    """Every drawing, driven under a real FreeCAD, as the probe left them."""
    out = tmp_path_factory.mktemp("palace_port_modes")
    manifest = probe_manifest(PROBE, out, "PORT_MODES_OUT", key=None)
    if manifest.get("missing"):
        pytest.skip(f"this machine cannot drive Palace: {manifest['missing']}")
    drawn = [*SEVERAL, *ONE]
    assert sorted(manifest["cases"]) == sorted(drawn), (
        f"the probe wrote {manifest['cases']} rather than {drawn}, so it stopped "
        "before every run finished"
    )
    return manifest


@pytest.mark.parametrize("drawing", SEVERAL)
def test_a_port_face_carrying_a_second_mode_is_refused_before_the_band(run_manifest, drawing):
    case = run_manifest[drawing]
    said = case["said"]
    assert said, f"{drawing} was solved, and its port's face carries more than one mode"
    assert not case["solved"], f"{drawing} was refused after the band was solved"
    assert "'Port1'" in said and " modes at " in said and " per metre" in said, said


@pytest.mark.parametrize("drawing", ONE)
def test_a_port_face_carrying_one_mode_is_solved_and_loses_nothing(run_manifest, drawing):
    case = run_manifest[drawing]
    assert not case["said"], case["said"]
    assert len(case["carried"]) == 4, (
        f"the run said {case['carried']} of the ports' modes, where each port is asked at "
        "each end of the band"
    )
    s = np.asarray(case["real"]) + 1j * np.asarray(case["imaginary"])
    driven = case["driven"].index(1)
    came_back = sum(np.abs(s[:, case["out"].index(port), driven]) ** 2 for port in (1, 2))
    worst = float(np.max(np.abs(1.0 - came_back)))
    print(f"\nWR-42 divided off centre on Palace: power lost {worst:.2e}")
    assert worst < POWER_TOLERANCE, (
        f"{drawing} returned {came_back} of the power that went in: the ports are not "
        "absorbing the mode the guide carries"
    )
