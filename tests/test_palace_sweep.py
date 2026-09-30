# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""An adaptive Palace sweep against the same study solved at every point.

A resonator between two irises in WR-42 with brass walls is swept adaptively
over many points, and again over few enough that each is solved in full. Where
the two bands share a point inside the band the answers agree, and the adaptive
run says where it solved in full. The ends of the band are left out: an adaptive
sweep solves both in full. The tolerance is on the field and bounds no
S-parameter, so the bar the answers are held to is that tolerance taken as a
bound on each term, which is a choice and not a consequence of it. The same
study allowed the fewest full solves and a tolerance no model reaches in them
is refused by the name of each driven port.

The test runs before a commit although it compares two solves. It is the one
run of the adaptive route on a real Palace, and costs less than a gate.

Why a subprocess
----------------

For the reason ``tests/test_acceptance_palace_waveguide.py`` gives: the
drawing needs the CAD kernel, and ``tests/palace_sweep_probe.py`` runs under
``freecadcmd`` to draw it and drive the run. A machine without Palace or Gmsh
makes the probe say which, and the test skips.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from Microwave.Solvers.palace import run
from Microwave.Solvers.palace.config import FEWEST_SOLVES
from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_sweep_probe.py")

#: What the solver object is made with, which the adaptive run is swept to.
TOLERANCE = 1e-3


@pytest.fixture(scope="module")
def swept(tmp_path_factory):
    """Each run, driven under a real FreeCAD, as the probe left it."""
    out = tmp_path_factory.mktemp("palace_sweep")
    manifest = probe_manifest(PROBE, out, "PALACE_SWEEP_OUT", key=None)
    if manifest.get("missing"):
        pytest.skip(f"this machine cannot drive Palace: {manifest['missing']}")
    assert manifest["cases"] == ["adaptive", "discrete", "capped"], (
        f"the probe wrote {manifest['cases']}, so it stopped before every run finished"
    )
    return manifest


def matrix(case):
    return np.asarray(case["real"]) + 1j * np.asarray(case["imaginary"])


def test_the_adaptive_run_agrees_with_every_point_solved_in_full(swept):
    adaptive, discrete = swept["adaptive"], swept["discrete"]
    band = np.asarray(adaptive["frequency"])
    inside = discrete["frequency"][1:-1]
    shared = [int(np.argmin(np.abs(band - f))) for f in inside]
    assert np.abs(band[shared] - inside).max() < 1.0
    difference = np.abs(matrix(adaptive)[shared] - matrix(discrete)[1:-1])
    print(f"GATE palace sweep: largest |dS| {difference.max():.3g} against {TOLERANCE:g}")
    assert difference.max() < TOLERANCE


def test_the_adaptive_run_says_where_each_driven_port_was_solved_in_full(swept):
    stated = [line for line in swept["adaptive"]["said"] if "full solves, at " in line]
    assert [line.split(":")[0] for line in stated] == ["'Port1'", "'Port2'"]
    assert all(f"within {TOLERANCE:g}" in line for line in stated)


def test_the_adaptive_run_checks_its_model_against_full_solves_and_it_holds(swept):
    """The run solves the band again in full where the model is likeliest to be
    wrong, and on this resonator the model holds there to inside the bar a study
    read at full scale sets."""
    (checked,) = [line for line in swept["adaptive"]["said"] if "off the full solve by" in line]
    print(f"GATE palace sweep check: {checked}")
    assert "SweepTolerance" not in checked


def test_a_run_at_every_point_is_not_checked_again(swept):
    assert not [line for line in swept["discrete"]["said"] if "solved again in full" in line]


def test_the_run_at_every_point_says_so(swept):
    assert "Each of the 6 points is solved in full" in swept["discrete"]["said"]


def test_a_sweep_that_takes_every_solve_it_is_allowed_is_refused(swept):
    capped = swept["capped"]
    assert capped["refused"].startswith("'Port1': ")
    assert f"all {FEWEST_SOLVES} full solves" in capped["refused"]
    (first, *_) = run.sampled(capped["log"])
    assert not first.converged
