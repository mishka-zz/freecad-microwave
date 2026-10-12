# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Acceptance gate: the filter somebody else built, against what their VNA said,
answered by Palace.

The stepped-impedance low-pass of Chen, Chen and Wang, transcribed in
:data:`tests.published.CHEN_2025`, solved on Palace from
``examples/stepped_lowpass_measured.FCStd`` exactly as that document is shipped,
and held to the two figures the paper states for the board it fabricated and
measured. ``tests/test_acceptance_openems_lowpass.py`` holds openEMS to the same
two figures, and says why a gate against an instrument is worth having beside
the closed forms.

What is scored
--------------

The same two claims, read the same way and held to the same tolerance, all from
``tests/lowpass_reading.py``: the -3 dB corner measured from the passband,
within :data:`~tests.lowpass_reading.CORNER_TOLERANCE`, and a transmission below
the published floor above the published frequency. The absolute -3 dB crossing
is printed beside them and asserted on by nothing.

The stopband is an inequality, as on openEMS. Their board carries the loss of
two connectors that this one does not, so transmitting more than they measured
is a disagreement, and transmitting less is not.

The board Palace is given is the board the paper specifies
----------------------------------------------------------

The adapter hands the mesher the faces as drawn, and every element conforms to
them, so the narrow sections are modelled at the width they are drawn at. The
openEMS gate draws them wider, by the share its grid inscribes, to hand openEMS
the width the paper specifies. Palace needs no such drawing, and this gate
solves the document the paper dimensions.

The loss tangent reaches Palace as a loss tangent, and Palace sets the imaginary
part of the permittivity to the real part times it at every frequency
(``palace/models/materialoperator.cpp``, "Material permittivity"). So the
passband droop here is not the band-centre conductivity's that the openEMS gate
describes.

What limits the comparison
--------------------------

The example carries the backend's defaults: the order the solver is made with,
the elements per wavelength and the edge refinement the mesh is made with, the
count across a narrow sheet the policy lays, and the clearance the policy derives
from the band. A test that solves nothing holds that it does. The gate then
solves the same document three times more, each with one setting moved, and
holds each move of the corner below
:data:`~tests.lowpass_reading.REFERENCE_HALF_WIDTH`, half the last digit the
published corner is printed to:

- **the narrow sections twice as finely across**, by a Mesh Refinement at each
  edge the count is laid at, at half the size laid there. A strip's impedance is
  set by the field across its width, and the narrow sections are the ones a size
  set by the wavelength spans in about one element. openEMS sizes its cell
  across a conductor from its width, and Palace lays the count across a narrow
  sheet, so this is where the two backends need the same thing by different
  rules;
- **one order lower**, on a mesh made from the same settings. The move is the
  change between two orders, and is not taken as the error left at the shipped
  order: nothing here measures how much each order removes;
- **the open surface** :data:`FARTHER` **times as far off.** The move is what
  that changes. A first-order absorbing condition reflects at either distance, so
  the move is the corner's sensitivity to where the surface stands rather than
  the cost of having one.

Each move is held alone, and moves inside the bar do not bound their sum. A move
past the bar says that the setting moved, and not the reference, limits the
comparison, and the assertion names the setting.

What the openEMS gate does and this one does not
------------------------------------------------

A refinement study with an order of convergence, and the sensitivity to the
laminate. Both are claims about the board and the reference, and the openEMS gate
makes them. The moves above answer the question this gate adds: whether
Palace's defaults are fine enough for the board.

Why a subprocess
----------------

For the reason ``tests/test_acceptance_palace_waveguide.py`` gives: the drawing
needs the CAD kernel, and ``tests/palace_lowpass_probe.py`` runs under
``freecadcmd`` to open the document and drive the run. A machine without Palace
or Gmsh makes the probe say which, and the gate skips.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.conftest import probe_manifest
from tests.lowpass_reading import (
    CORNER_TOLERANCE,
    PAPER,
    REFERENCE_HALF_WIDTH,
    corner,
    first_crossing,
    passband_db,
)

pytestmark = pytest.mark.slow

# Every solve here is of the costliest board a Palace gate solves, so each test
# that solves is held back to the release run, and the test that solves nothing
# is not. A lumped port, an open board and a stepped trace are each exercised
# before a commit by a gate costing a fraction of this one.
release = pytest.mark.release

PROBE = os.path.join(os.path.dirname(__file__), "palace_lowpass_probe.py")

#: How many times the shipped clearance the farther case stands its open
#: surface off.
FARTHER = 2.0


def _probed(tmp_path_factory, cases):
    return probe_manifest(
        PROBE,
        tmp_path_factory.mktemp("palace_lowpass"),
        "LOWPASS_PALACE_OUT",
        key=None,
        environment={
            "LOWPASS_PALACE_CASES": ",".join(cases),
            "LOWPASS_PALACE_FARTHER": f"{FARTHER!r}",
        },
    )


def _solved(tmp_path_factory, case):
    """The example solved as ``case`` names it, as the probe left it."""
    manifest = _probed(tmp_path_factory, [case])
    if manifest.get("missing"):
        pytest.skip(f"this machine cannot drive Palace: {manifest['missing']}")
    assert manifest["cases"] == [case], (
        f"the probe wrote {manifest['cases']} rather than {[case]}, so it stopped "
        "before the run finished. A gate that is not run is not a gate that passed"
    )
    run = manifest[case]
    assert run["driven"] == [1], f"the run drove {run['driven']} rather than port 1"
    matrix = np.asarray(run["real"]) + 1j * np.asarray(run["imaginary"])
    run["frequency"] = np.asarray(run["frequency"], dtype=float)
    run["decibels"] = 20.0 * np.log10(np.abs(matrix[:, run["out"].index(2), 0]))
    run["corner"] = corner(run["frequency"], run["decibels"])
    return run


@pytest.fixture(scope="module")
def shipped(tmp_path_factory):
    return _solved(tmp_path_factory, "shipped")


@pytest.fixture(scope="module")
def coarser(tmp_path_factory):
    return _solved(tmp_path_factory, "coarser")


@pytest.fixture(scope="module")
def farther(tmp_path_factory):
    return _solved(tmp_path_factory, "farther")


@pytest.fixture(scope="module")
def finer(tmp_path_factory):
    return _solved(tmp_path_factory, "finer")


def described(run: dict) -> str:
    return (
        f"order {run['order']}, narrow sections at {run['narrowest']:.4g} mm, "
        f"open surface {run['clearance']:.4g} mm off"
    )


def test_each_case_moves_one_setting_from_the_defaults(tmp_path_factory):
    """The shipped document carries what a solver, a mesh and a policy are made
    with, and each other case moves the one setting it is named for. Solves
    nothing, so a guard below cannot pass for a case that changed nothing."""
    manifest = _probed(tmp_path_factory, [])
    made, found = manifest["made"], manifest["set"]
    shipped = found["shipped"]
    assert {key: shipped[key] for key in made} == made, (
        f"the example sets {shipped}, where a new solver, mesh and policy are made "
        f"with {made}, so the gate does not solve the defaults it says it does"
    )
    assert found["coarser"] == {**shipped, "order": shipped["order"] - 1}
    farther = found["farther"]
    assert farther["clearance"] == pytest.approx(FARTHER * shipped["clearance"], rel=1e-9)
    assert without(farther, "clearance", "stated_clearance") == without(
        shipped, "clearance", "stated_clearance"
    )
    assert shipped["narrowest"] is not None, "the example lays no count across a narrow sheet"
    finer = found["finer"]
    assert finer["narrowest"] == pytest.approx(shipped["narrowest"] / 2.0, rel=1e-9)
    assert without(finer, "narrowest") == without(shipped, "narrowest")


def without(found: dict, *keys: str) -> dict:
    return {key: value for key, value in found.items() if key not in keys}


@release
def test_the_fixture_is_not_vacuous(shipped):
    """There has to be a passband for the corner to be the edge of.

    Asserted as the contrast between the two bands, for the reason the openEMS
    gate gives: a floor under the passband is a tolerance on insertion loss, and
    a board uniformly dead or uniformly transparent has no contrast at all.
    """
    frequency, decibels = shipped["frequency"], shipped["decibels"]
    passband = passband_db(frequency, decibels)
    stopband = float(decibels[frequency >= PAPER.stopband_from].max())
    assert passband - stopband > -PAPER.stopband / 2.0, (
        f"passband {passband:.2f} dB and stopband {stopband:.2f} dB are not two bands"
    )


@release
def test_the_corner_is_where_the_instrument_found_it(shipped):
    """The gate. Their -3 dB point, against Palace's, on their board."""
    found = shipped["corner"]
    passband = passband_db(shipped["frequency"], shipped["decibels"])
    error = found / PAPER.corner - 1.0
    print(
        f"\nGATE published low-pass on Palace ({described(shipped)}): corner = "
        f"{found / 1e9:.4f} GHz (3 dB below a passband of {passband:+.2f} dB), "
        f"measured {PAPER.corner / 1e9:.3g} GHz, {error:+.2%}, bound {CORNER_TOLERANCE:.0%}"
    )
    assert found == pytest.approx(PAPER.corner, rel=CORNER_TOLERANCE)


@release
def test_the_stopband_is_as_deep_as_the_paper_reports(shipped):
    """The other gate, and an inequality rather than an agreement."""
    frequency, decibels = shipped["frequency"], shipped["decibels"]
    above = frequency >= PAPER.stopband_from
    assert above.any(), "no samples in the stopband"
    worst = float(decibels[above].max())
    at = float(frequency[above][int(np.argmax(decibels[above]))])
    print(
        f"\nGATE published low-pass on Palace ({described(shipped)}): stopband = "
        f"{worst:.2f} dB at {at / 1e9:.3f} GHz, published below {PAPER.stopband:.0f} dB "
        f"above {PAPER.stopband_from / 1e9:.0f} GHz"
    )
    assert worst <= PAPER.stopband


@release
def test_where_the_absolute_three_decibel_crossing_lands(shipped):
    """Reported, and asserted on by nothing: the gap between the two crossings is
    the passband droop, a fact about the modelled dissipation."""
    frequency, decibels = shipped["frequency"], shipped["decibels"]
    absolute = first_crossing(frequency, decibels, -3.0)
    print(
        f"\nGATE published low-pass on Palace ({described(shipped)}): absolute -3 dB = "
        f"{absolute / 1e9:.4f} GHz, {absolute / PAPER.corner - 1:+.2%} against the "
        f"measurement and {absolute / shipped['corner'] - 1:+.2%} against the corner "
        "(not asserted)"
    )


@release
@pytest.mark.parametrize(
    "moved, key, setting",
    [
        ("finer", "narrowest", "the count across the narrow sections"),
        ("coarser", "order", "the order"),
        ("farther", "clearance", "the clearance"),
    ],
)
def test_moving_one_setting_moves_the_corner_inside_the_reference(
    shipped, moved, key, setting, request
):
    """The corner moves less than the published figure's half-width when one
    setting moves. See the module docstring."""
    other = request.getfixturevalue(moved)
    assert other[key] != shipped[key], f"the {moved} case solved {setting} the example ships"
    move = other["corner"] - shipped["corner"]
    print(
        f"\nGATE published low-pass on Palace: {setting} moved, {described(shipped)} to "
        f"{described(other)}, moves the corner {move / 1e6:+.3g} MHz "
        f"({move / PAPER.corner:+.2%}), against the reference's half-width of "
        f"{REFERENCE_HALF_WIDTH / 1e6:.3g} MHz"
    )
    assert abs(move) < REFERENCE_HALF_WIDTH, (
        f"moving {setting} from {described(shipped)} to {described(other)} moves the "
        f"corner {move / 1e6:+.4g} MHz, past the {REFERENCE_HALF_WIDTH / 1e6:.3g} MHz "
        f"half-width of the published figure, so {setting} the example ships with, and "
        "not the reference, limits this comparison"
    )
