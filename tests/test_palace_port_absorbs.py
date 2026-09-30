# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a wave port absorbs besides its own mode, on Palace.

A wave port takes its own mode at its face and takes up part of whatever else
reaches it. A post a millimetre from a port leaves its evanescent field on the
face, and the port takes some of it: power that stands in no term of the
matrix. So does a post 20 mm from each port where the band's top nears TE20's
cutoff and TE20 dies slowly, and so does an H-plane T there. After the band is
solved the run states what the matrix leaves unaccounted for from each driven
port, as a warning past the bar, and returns the matrix either way.

What the model turns into heat is not counted: brass walls and an FR4 filling
lose more power than the bar, and neither is warned of. A plain guide, a post on
the centre line, which excites no TE20, and the off-centre post over the lower
part of the band are not warned of either. Nor is a guide whose lower half is
PTFE from port to port: each port's face crosses two materials and carries a
mode with an electric field along the guide, which the port is matched to.

Why a subprocess
----------------

For the reason ``tests/test_acceptance_palace_waveguide.py`` gives: the
drawing needs the CAD kernel, and ``tests/palace_port_absorbs_probe.py`` runs
under ``freecadcmd`` to draw it and drive the run. A machine without Palace
or Gmsh makes the probe say which, and the test skips.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from Microwave.Solvers.palace.balance import BAR, WARNING
from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_port_absorbs_probe.py")

#: What the probe draws, as it names each drawing: those whose ports absorb field
#: besides their mode, those among them where a post stands at port 1, and those
#: whose ports do not.
ABSORBING = ("close_brass_post", "close_post_filled", "near_te20", "tee")
AT_PORT_1 = ("close_brass_post", "close_post_filled")
CLEAN = ("plain", "brass_walls", "filled", "centred_near_te20", "low", "slab")

#: The drawings whose models turn power into heat, and those among the clean
#: ones that lose more of it than the bar.
DISSIPATING = ("close_brass_post", "close_post_filled", "brass_walls", "filled", "slab")
HEATED = ("brass_walls", "filled")


@pytest.fixture(scope="module")
def run_manifest(tmp_path_factory):
    """Every drawing, driven under a real FreeCAD, as the probe left them."""
    out = tmp_path_factory.mktemp("palace_port_absorbs")
    manifest = probe_manifest(PROBE, out, "PORT_ABSORBS_OUT", key=None)
    if manifest.get("missing"):
        pytest.skip(f"this machine cannot drive Palace: {manifest['missing']}")
    drawn = [*ABSORBING, *CLEAN]
    assert sorted(manifest["cases"]) == sorted(drawn), (
        f"the probe wrote {manifest['cases']} rather than {drawn}, so it stopped "
        "before every run finished"
    )
    return manifest


def kept(case):
    """The share of the driven port's power the matrix accounts for, per sample."""
    s = np.asarray(case["real"]) + 1j * np.asarray(case["imaginary"])
    return np.sum(np.abs(s[:, :, 0]) ** 2, axis=1)


def the_one(case):
    """The run's one driven port's shortfall and the line stating it."""
    assert case["driven"] == [1]
    (short,) = case["shortfalls"]
    (line,) = case["stated"]
    return short, line


@pytest.mark.parametrize("drawing", ABSORBING)
def test_a_port_absorbing_field_besides_its_mode_is_warned_of(run_manifest, drawing):
    short, line = the_one(run_manifest[drawing])
    print(
        f"\nWR-42 on Palace ({drawing}): {short['share']:.2e} unaccounted for at "
        f"{short['frequency'] / 1e9:.6g} GHz, most through port {short['port']}"
    )
    assert short["share"] > BAR
    assert line.startswith(WARNING) and "Driven from 'Port1'" in line, line
    assert f"{short['share'] * 100:.3g}% of the power unaccounted for" in line, line


@pytest.mark.parametrize("drawing", AT_PORT_1)
def test_a_post_at_a_port_is_named_as_taken_by_that_port(run_manifest, drawing):
    short, line = the_one(run_manifest[drawing])
    assert short["port"] == 1
    assert "through 'Port1'" in line, line


@pytest.mark.parametrize("drawing", CLEAN)
def test_a_port_absorbing_nothing_else_is_stated_and_not_warned_of(run_manifest, drawing):
    short, line = the_one(run_manifest[drawing])
    print(
        f"\nWR-42 on Palace ({drawing}): {short['share']:.2e} unaccounted for at "
        f"{short['frequency'] / 1e9:.6g} GHz"
    )
    assert abs(short["share"]) <= BAR
    assert not line.startswith(WARNING), line


@pytest.mark.parametrize("drawing", HEATED)
def test_what_the_model_dissipates_is_not_counted_as_the_ports(run_manifest, drawing):
    """The model loses more than the bar, and the run does not warn of it."""
    case = run_manifest[drawing]
    assert case["dissipates"]
    assert np.min(kept(case)) < 1.0 - BAR
    short, _ = the_one(case)
    assert abs(short["share"]) <= BAR


@pytest.mark.parametrize("drawing", [*ABSORBING, *CLEAN])
def test_a_model_that_dissipates_nothing_is_held_to_its_matrix(run_manifest, drawing):
    case = run_manifest[drawing]
    assert case["dissipates"] == (drawing in DISSIPATING)
    if case["dissipates"]:
        return
    short, _ = the_one(case)
    sample = case["frequency"].index(short["frequency"])
    assert short["share"] == pytest.approx(1.0 - kept(case)[sample], abs=1e-12)
