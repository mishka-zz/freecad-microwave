# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Acceptance gate: metal drawn inside a WR-42 guide, answered by Palace.

A septum of perfect conductor stands in the plane at half the broad wall, the
full height of the guide, for a length along it. Alongside it the guide is two
guides half as wide, and across the whole band both are below cutoff, so the
wave crossing the septum decays rather than travels. Separating Maxwell's
equations in each half gives the rate::

    w     = a / 2                              width of each half
    kc    = pi / w                             transverse wavenumber, TE10
    alpha = sqrt(kc^2 - k0^2)                  attenuation, nepers per metre

What the junctions at either end of the septum scatter is not in that. So the
septum is drawn at two lengths and the gate reads how much further |S21| falls
over the longer one: what each junction does to the wave crossing once is the
same at both lengths and cancels, and what is left is ``alpha`` times the added
length, with no fitted constant. What does not cancel is the wave that crosses
the septum more than once, which falls with the shorter septum's length as the
square of the wave crossing once, and the next mode the junctions excite, which
falls faster still.

What this gate exercises
------------------------

Metal drawn inside the region the field is in, two ways. The septum is drawn as
a sheet, which is how every board in this workbench draws its metal, bound to a
PEC material, and it is not a face of the region: the guide runs round it on
both sides. It has to reach the run as a condition on a face inside the model.

It is also drawn as a body of thickness ``t``, which has to leave the region
with its faces carrying the condition. Each half-guide beside it is then
``(a - t) / 2`` wide rather than ``a / 2``, and the body is held to the closed
form of that width.

A septum that never reached the run leaves one guide, whose wave travels, and
|S21| then falls by nothing. A septum solved at a thickness it was not drawn
with moves the half-width, and with it ``alpha``, by far more than the bar
below. So does a body left in the region as a volume of vacuum, since the wave
then crosses it.

Why a subprocess
----------------

For the reason ``tests/test_acceptance_palace_waveguide.py`` gives: the
drawing needs the CAD kernel, and ``tests/palace_septum_probe.py`` runs under
``freecadcmd`` to draw it and drive the run. A machine without Palace or Gmsh
makes the probe say which, and the gate skips.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.analytic import reference
from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_septum_probe.py")

#: How far the fall of |S21| between the two lengths may sit from the closed
#: form, as a fraction of it.
#:
#: The closed form leaves out the wave that crosses the septum more than once and
#: every mode past the first, and neither is written out here: what the run
#: reached with both in it is on the gate's own line. The bar is what separates
#: the septum reaching the run from it not. A septum missing from the run misses
#: by the whole fall, and the half-width of a septum half a millimetre thick
#: against that of a sheet moves ``alpha`` by tens of percent near the top of the
#: band, so one percent stands between them whichever way it is drawn.
FALL_TOLERANCE = 0.01

#: The septum is lossless and the wave port absorbs its own mode, so the power
#: that went in comes back out of one port or the other.
POWER_TOLERANCE = 1e-3

#: The two lengths the septum is drawn at, in mm, as the probe draws them.
LENGTHS = (8.0, 16.0)

#: How the septum is drawn, as the probe names it.
SHAPES = ("sheet", "body")

#: Every drawing the gate solves, by the name the probe writes it under.
SOLVED = [f"{shape}_{length:g}" for shape in SHAPES for length in LENGTHS]


@pytest.fixture(scope="module")
def run_manifest(tmp_path_factory):
    """Every drawing, driven under a real FreeCAD, as the probe left them."""
    out = tmp_path_factory.mktemp("palace_septum")
    manifest = probe_manifest(PROBE, out, "SEPTUM_PALACE_OUT", key=None)
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


@pytest.mark.parametrize("shape", SHAPES)
def test_the_septum_attenuates_at_the_rate_of_the_half_width_guide(run_manifest, shape):
    """The gate: the extra length of septum takes off what theory says it does."""
    short, long = (run_manifest[f"{shape}_{length:g}"] for length in LENGTHS)
    frequency = np.asarray(short["frequency"], dtype=float)
    assert np.array_equal(frequency, np.asarray(long["frequency"], dtype=float))

    thickness = run_manifest["thickness"] if shape == "body" else 0.0
    half = (run_manifest["broad"] - thickness) / 2.0
    alpha = np.imag(reference.phase_constant(frequency, half * 1e-3, run_manifest["narrow"] * 1e-3))
    assert np.all(alpha > 0), "the band reaches the cutoff of the half-width guide"
    added = (long["length"] - short["length"]) * 1e-3
    expected = -alpha * added

    fall = np.log(np.abs(entry(long, 2, 1))) - np.log(np.abs(entry(short, 2, 1)))
    error = fall / expected - 1.0
    worst = float(np.max(np.abs(error)))

    print(
        f"\nGATE WR-42 septum on Palace (a {shape}): fall of |S21| over "
        f"{added * 1e3:g} mm = {np.min(fall) * 20 / np.log(10):.3f} to "
        f"{np.max(fall) * 20 / np.log(10):.3f} dB, worst departure from the "
        f"half-width guide {worst * 100:.4f}% [order {short['order']}]"
    )
    assert worst < FALL_TOLERANCE, (
        f"the longer septum takes off {fall} nepers against {expected} from the "
        f"half-width guide's TE10 attenuation, {worst * 100:.2f}% at worst, so "
        "the septum the run solved is not the one drawn"
    )


@pytest.mark.parametrize("case", SOLVED)
def test_power_is_conserved(run_manifest, case):
    read = run_manifest[case]
    column = np.abs(entry(read, 1, 1)) ** 2 + np.abs(entry(read, 2, 1)) ** 2
    worst = float(np.max(np.abs(column - 1.0)))
    print(f"\nGATE WR-42 septum on Palace ({case}): worst power departure = {worst:.3e}")
    assert worst < POWER_TOLERANCE, (
        f"a column of the matrix carries {worst:.3g} more or less power than went "
        "into it, and a septum of perfect conductor in a hollow guide is lossless"
    )
