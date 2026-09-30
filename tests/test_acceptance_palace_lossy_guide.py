# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Acceptance gate: a WR-42 guide losing power to brass, answered by Palace.

A guide whose walls conduct finitely loses power as the wave travels, at a rate
the lossless mode's field on each wall decides. Perturbing that field with the
surface resistance of a metal thick against its skin depth gives, per face::

    Rs      = sqrt(omega mu0 / 2 sigma)               surface resistance
    P       = omega mu0 beta a^3 b / 4 pi^2           power the mode carries
    broad   = Rs a / 4 (1 + (beta a / pi)^2)          lost on one broad face
    narrow  = Rs b / 2                                lost on one narrow face
    alpha   = (faces lost on) / 2 P                   nepers per metre

The metal is the bundled catalog's brass, a conducting sheet, drawn these ways:

* the four side faces of the guide, so the metal is where the model ends - two
  broad faces and two narrow ones;
* a plane across the guide at half its height, with the perfect wall round it.
  TE10's field does not vary across the narrow wall, so each face of the plane
  meets what a broad wall does - two broad faces.
* a plane on one broad wall, short of both ends, with the perfect wall round
  it - one broad face. It meets no port's face, so each port's mode is solved
  with it as the perfect wall it stands in.

Each is drawn at two lengths, and the gate reads the power lost over the added
length: ``1 - |S11|^2 - |S21|^2`` at each, and ``alpha`` from the ratio of what
came through. What the port faces and the plane's two ends do is the same at
both lengths and cancels, and no fitted constant is left.

What this gate exercises
------------------------

A conducting sheet reaching the second backend as the surface impedance of its
metal, on the outside of the model and inside it. A sheet that never reached the
run as that leaves a lossless guide, and the fall is nothing. A plane carrying
the condition on one face rather than both loses half of it; walls with the
narrow pair left perfect lose less by that pair's share, which the closed form
puts past a sixth everywhere in this band. The bar below is between those and
the closed form.

What it does not see is the thickness. Brass is hundreds of skin depths thick
across this band, so the adapter gives Palace no thickness, and with none the
side of the model a sheet stands on changes nothing Palace computes either. Both
are held by the unit tests on what is written, and not by a solve here.

A film whose sheet resistance is of the order of the wave impedance lets much
of the wave through, and Palace would stop it at both faces. The probe draws one
across the guide, and the run refuses it before anything is written.

It also runs the adapter's rule for a wave port meeting a finite conductor - the
walls meet both ends. Palace can hang there on more than one process, and the
adapter gives each lossless material a loss tangent far below a double's
precision, which makes every rank assemble alike. The probe asks for more
than one process, and the run is held to have used them and said why.

Why a subprocess
----------------

For the reason ``tests/test_acceptance_palace_waveguide.py`` gives: the
drawing needs the CAD kernel, and ``tests/palace_lossy_guide_probe.py`` runs
under ``freecadcmd`` to draw it and drive the run. A machine without Palace
or Gmsh makes the probe say which, and the gate skips.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.analytic import reference
from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_lossy_guide_probe.py")

#: How far the attenuation read off the two lengths may sit from the closed
#: form, as a fraction of it.
#:
#: A plane carrying the condition on one face misses by half, and walls with the
#: narrow pair left perfect by more than a sixth, so one percent stands between
#: those and the metal reaching the run as drawn. The closed form perturbs the
#: lossless mode's field and leaves out what the loss does to the field itself,
#: which is of the order of the surface resistance over the wave impedance and
#: far under the bar. What the run reached is on the gate's own line.
ATTENUATION_TOLERANCE = 0.01

#: The two lengths each drawing is made at, in mm, as the probe draws them.
LENGTHS = (20.0, 60.0)

#: The faces each drawing loses power on: broad, narrow.
FACES = {"walls": (2, 2), "sheet": (2, 0), "strip": (1, 0)}

#: Every drawing the gate solves, by the name the probe writes it under.
SOLVED = [f"{drawing}_{length:g}" for drawing in FACES for length in LENGTHS]


@pytest.fixture(scope="module")
def run_manifest(tmp_path_factory):
    """Every drawing, driven under a real FreeCAD, as the probe left them."""
    out = tmp_path_factory.mktemp("palace_lossy_guide")
    manifest = probe_manifest(PROBE, out, "LOSSY_GUIDE_PALACE_OUT", key=None)
    if manifest.get("missing"):
        pytest.skip(f"this machine cannot drive Palace: {manifest['missing']}")
    assert manifest["cases"] == SOLVED, (
        f"the probe wrote {manifest['cases']} rather than {SOLVED}, so it stopped "
        "before every run finished. A gate that is not run is not a gate that passed"
    )
    return manifest


def entry(read: dict, out: int, driven: int) -> np.ndarray:
    """``S[out][driven]`` by port number rather than by position."""
    s = np.asarray(read["real"], dtype=float) + 1j * np.asarray(read["imaginary"], dtype=float)
    return s[:, read["out"].index(out), read["driven"].index(driven)]


def through(read: dict) -> np.ndarray:
    """The power that came back out of the guide, of what went into it."""
    return np.abs(entry(read, 1, 1)) ** 2 + np.abs(entry(read, 2, 1)) ** 2


@pytest.mark.parametrize("drawing", list(FACES))
def test_the_brass_takes_off_what_its_surface_resistance_does(run_manifest, drawing):
    """The gate: the added length of lossy guide loses what theory says it does."""
    short, long = (run_manifest[f"{drawing}_{length:g}"] for length in LENGTHS)
    frequency = np.asarray(short["frequency"], dtype=float)
    assert np.array_equal(frequency, np.asarray(long["frequency"], dtype=float))

    broad, narrow = FACES[drawing]
    expected = reference.wall_attenuation(
        frequency,
        run_manifest["broad"] * 1e-3,
        run_manifest["narrow"] * 1e-3,
        short["conductivity"],
        broad=broad,
        narrow=narrow,
    )
    added = (long["length"] - short["length"]) * 1e-3
    alpha = -np.log(through(long) / through(short)) / (2 * added)
    error = alpha / expected - 1.0
    worst = float(np.max(np.abs(error)))

    print(
        f"\nGATE WR-42 of brass on Palace ({drawing}): attenuation over "
        f"{added * 1e3:g} mm = {np.min(alpha):.5f} to {np.max(alpha):.5f} Np/m, "
        f"worst departure from the surface resistance {worst * 100:.4f}% "
        f"[order {short['order']}]"
    )
    assert worst < ATTENUATION_TOLERANCE, (
        f"the added length of guide loses {alpha} Np/m against {expected} from the "
        f"brass's surface resistance on {broad} broad and {narrow} narrow faces, "
        f"{worst * 100:.2f}% at worst, so the metal the run solved is not the one drawn"
    )


def test_a_resistive_film_across_the_guide_is_refused_naming_its_sheet_resistance(run_manifest):
    """The solver takes each face of a sheet inside the region as a wall, so a
    film that lets most of the wave through would come back as one that stops
    it. Refused before anything is drawn for the mesher."""
    film = run_manifest["film"]
    assert "'BrassBinding' binds 'Film'" in film["said"], film["said"]
    assert "1000 ohms a square" in film["said"], film["said"]
    assert not film["drawn"], "the film's refusal came after its shapes were written"


@pytest.mark.parametrize("case", SOLVED)
def test_a_run_carrying_brass_keeps_its_processes_and_says_why(run_manifest, case):
    asked = run_manifest[case]["asked"]
    said = run_manifest[case]["processes"]
    assert asked > 1, "the probe asks for one process, so no rank could disagree"
    assert [line for line in said if f"Running with {asked} MPI processes" in line], (
        f"the run asked for {asked} processes and said {said} about them"
    )
    assert [
        line for line in said if line.startswith("Each lossless material is given a loss tangent")
    ], f"the run gave the materials no loss tangent and said {said}"
