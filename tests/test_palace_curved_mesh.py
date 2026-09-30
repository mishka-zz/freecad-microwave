# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A body curved tighter than the element the wavelength asks for meshes on
Palace, and a line inside metal a refinement names is refused by name.

A second-order element follows a curved surface only across a limited arc, and
Gmsh turns one laid across a tighter curve inside out. So each curved surface
is sized by its radius as well as by the wavelength, and a torus, a cone at its
apex, a narrow groove, a thin rod and a thin via mesh. The torus sized by the
wavelength alone does not, and the run says how to make its elements finer.
The kernel fails to cut the line, and the run names the refinement and says
nothing about the size of the elements.

Why a subprocess
----------------

The drawings need the CAD kernel and the mesh needs Gmsh, so
``tests/palace_curved_mesh_probe.py`` runs under ``freecadcmd``. Nothing is
solved. Where the machine has no Gmsh, the meshes skip.
"""

from __future__ import annotations

import math
import os

import pytest

from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_curved_mesh_probe.py")

#: The elements the probe asks round a full turn, and the thinnest radius it
#: draws, in mm.
PER_TURN = 6.0
THINNEST = 0.15

#: Each drawing meshed.
MESHED = (
    "torus_open",
    "cone_over_a_hump",
    "groove_under_the_top",
    "rod_of_pec_open_0.15",
    "rod_of_pec_open_1.5",
    "rod_of_pec_closed_0.15",
    "rod_of_pec_closed_1.5",
    "via_closed_0.15",
    "via_closed_1.5",
    "rod_of_the_dielectric_open",
)

#: Each drawing Gmsh cannot fill.
UNMESHED = ("torus_open_sized_by_wavelength_alone", "line_in_a_post_closed")


@pytest.fixture(scope="module")
def run_manifest(tmp_path_factory):
    out = tmp_path_factory.mktemp("palace_curved_mesh")
    manifest = probe_manifest(PROBE, out, "CURVED_MESH_OUT", key=None)
    drawn = [*MESHED, *UNMESHED]
    assert sorted(manifest["cases"]) == sorted(drawn), (
        f"the probe wrote {manifest['cases']} rather than {drawn}"
    )
    for case in drawn:
        assert "failed" not in manifest[case], (case, manifest[case]["failed"])
        assert "refused" not in manifest[case], (case, manifest[case]["refused"])
    if "missing" in manifest:
        pytest.skip(f"this machine cannot mesh for Palace: {manifest['missing']}")
    return manifest


@pytest.mark.parametrize("case", MESHED)
def test_a_body_curved_tighter_than_the_element_meshes(run_manifest, case):
    assert run_manifest[case].get("mesh") == "meshed", (case, run_manifest[case])
    assert run_manifest[case]["worst"] > 0.0, (case, run_manifest[case])


@pytest.mark.parametrize("case", ["rod_of_pec_open_0.15", "via_closed_0.15"])
def test_a_thin_round_body_is_meshed_at_its_own_radius(run_manifest, case):
    """The rim of a sheet is refined far coarser than this radius, so the
    shortest element is the rod's."""
    assert run_manifest[case]["shortest"] <= 2.0 * math.pi * THINNEST / PER_TURN


def test_a_torus_sized_by_the_wavelength_alone_is_told_how_to_make_its_elements_finer(
    run_manifest,
):
    said = run_manifest["torus_open_sized_by_wavelength_alone"].get("mesh", "")
    assert said.startswith("refused: "), said
    assert "set ElementsPerTurn on the Gmsh mesh above zero" in said


def test_a_line_inside_metal_a_refinement_names_is_refused_by_name_and_not_by_size(
    run_manifest,
):
    """The kernel fails to cut a line inside a body inside another once the body
    has been used in a cut, and the elements' size changes nothing there."""
    said = run_manifest["line_in_a_post_closed"].get("mesh", "")
    assert said.startswith("refused: Gmsh could not cut the mark 'Mesh Refinement on Line'"), said
    assert "ElementsPerTurn" not in said
