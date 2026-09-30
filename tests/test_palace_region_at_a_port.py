# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A refinement region drawn across a wave port's face leaves the port's
voltage line as it was.

The mesher cuts a region into the model where it meets it, so a region crossing
a port's face cuts the face into pieces. The line each port's voltage is read
along fixes the sign of the port's mode, and a port written without one comes
back from Palace with its transmission turned half a cycle. So each drawing
here is held to the line the same guide carries with no region at all, and to
having cut the face, which is what makes the comparison say anything.

Why a subprocess
----------------

The regions need the CAD kernel and the mesh needs Gmsh, so
``tests/palace_region_at_a_port_probe.py`` runs under ``freecadcmd``. Nothing
is solved: the line is in the configuration. Skips where the machine has no
Palace or no Gmsh.
"""

from __future__ import annotations

import os

import pytest

from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_region_at_a_port_probe.py")

CUT = ("face_part", "body", "edge")


@pytest.fixture(scope="module")
def run_manifest(tmp_path_factory):
    out = tmp_path_factory.mktemp("palace_region_at_a_port")
    manifest = probe_manifest(PROBE, out, "REGION_AT_A_PORT_OUT", key=None)
    if "missing" in manifest:
        pytest.skip(f"this machine cannot mesh for Palace: {manifest['missing']}")
    drawn = ["none", *CUT]
    assert manifest["cases"] == drawn, (
        f"the probe wrote {manifest['cases']} rather than {drawn}, so it stopped "
        "before every drawing was meshed"
    )
    return manifest


def test_the_guide_with_no_region_carries_a_line_at_each_port(run_manifest):
    lines = run_manifest["none"]["lines"]
    assert sorted(lines) == ["1", "2"]
    assert all(line is not None for line in lines.values())
    assert run_manifest["none"]["pieces"]["Port1"] == 1


@pytest.mark.parametrize("drawing", CUT)
def test_a_region_cutting_a_ports_face_leaves_each_ports_line_as_it_was(run_manifest, drawing):
    case = run_manifest[drawing]
    assert case["pieces"]["Port1"] > 1, "the region did not cut the port's face"
    assert case["lines"] == run_manifest["none"]["lines"]
