# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Two wave ports whose faces meet, on Palace.

Palace solves each port's mode with every other port's faces as a perfect
conductor. Two ports side by side on the start of WR-42, meeting along a curve
no metal holds, are each solved with a wall there that the field between them
does not have: split across the height, each half keeps its wave and part of
the power is lost at the faces; split across the width, each half is below
cutoff. Both are refused once the mesh exists, naming the two ports. A sheet of
perfect conductor along the shared curve is the wall each port is solved with,
and that drawing is solved: each half sends half its power on to the far port,
and the power is conserved.

Why a subprocess
----------------

For the reason ``tests/test_acceptance_palace_waveguide.py`` gives: the
drawing needs the CAD kernel, and ``tests/palace_ports_meeting_probe.py``
runs under ``freecadcmd`` to draw it and drive the run. A machine without
Palace or Gmsh makes the probe say which, and the test skips.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from Microwave.Solvers.palace.balance import BAR
from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_ports_meeting_probe.py")

MEETING = ("across_the_height", "across_the_width")
HELD = ("held_by_a_sheet",)


@pytest.fixture(scope="module")
def run_manifest(tmp_path_factory):
    """Every drawing, driven under a real FreeCAD, as the probe left them."""
    out = tmp_path_factory.mktemp("palace_ports_meeting")
    manifest = probe_manifest(PROBE, out, "PORTS_MEETING_OUT", key=None)
    if manifest.get("missing"):
        pytest.skip(f"this machine cannot drive Palace: {manifest['missing']}")
    drawn = [*MEETING, *HELD]
    assert sorted(manifest["cases"]) == sorted(drawn), (
        f"the probe wrote {manifest['cases']} rather than {drawn}, so it stopped "
        "before every run finished"
    )
    return manifest


@pytest.mark.parametrize("drawing", MEETING)
def test_two_ports_meeting_along_a_curve_no_metal_holds_are_refused(run_manifest, drawing):
    said = run_manifest[drawing]["said"]
    assert said.startswith(
        "'Port1' and 'Port2' stand on faces that meet along a curve no metal holds"
    ), said


def test_two_ports_either_side_of_a_sheet_are_solved_and_conserve_power(run_manifest):
    case = run_manifest["held_by_a_sheet"]
    assert case["said"] == "", case["said"]
    s = np.asarray(case["real"]) + 1j * np.asarray(case["imaginary"])
    departure = np.max(np.abs(1.0 - np.sum(np.abs(s) ** 2, axis=1)))
    print(f"\nWR-42 split at its start by a sheet on Palace: worst power departure {departure:.2e}")
    assert departure <= BAR
    far = case["out"].index(3)
    assert np.allclose(np.abs(s[:, far, :]) ** 2, 0.5, atol=BAR)
