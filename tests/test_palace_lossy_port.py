# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A wave port in a lossy filling, on Palace.

Palace matches a wave port to the real part of its mode's propagation constant
and drops the imaginary part (``palace/models/waveportoperator.cpp:1734-1748``).
A mode that loses power as it travels meets a face matched to another line, and
the face reflects part of it. WR-42 filled from port to port with one lossy
dielectric reflects nothing, so what S11 returns there is the port's own: the
share the adapter computes from the constant and states before the band. Past a
thousandth of the power the run is refused before the band. A filling lossy
enough to hide the guide's own mode from the mode run is refused saying so.

The port solves its mode again at every frequency, choosing among the pairs its
eigensolver converged on (``palace/models/modeeigensolver.cpp:635-650``), and
past a large loss that can be another mode than the one the mode run found. So
the guide that is solved is held to its ports taking, at every frequency, a
mode that propagates, and at the ends of the band the one the mode run found.

Why a subprocess
----------------

For the reason ``tests/test_acceptance_palace_waveguide.py`` gives: the
drawing needs the CAD kernel, and ``tests/palace_lossy_port_probe.py`` runs
under ``freecadcmd`` to draw it and drive the run. A machine without Palace
or Gmsh makes the probe say which, and the test skips.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from Microwave.Solvers.palace.balance import BAR
from Microwave.Solvers.palace.modes import propagates, reflected
from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_lossy_port_probe.py")

DRAWN = ("board", "absorbing", "conducting")

#: How far the reflection Palace returns may stand from the share the constant
#: states, as a share of it. The port's mode and the guide's field are one
#: discretisation, so the two agree far closer than this.
AGREEMENT = 0.02

#: How far the constant a port took at an end of the band may stand from the one
#: its mode run found there, as a share of it. Both are one solve of one face,
#: and Palace prints four significant figures.
SAME = 1e-3


@pytest.fixture(scope="module")
def run_manifest(tmp_path_factory):
    """Every drawing, driven under a real FreeCAD, as the probe left them."""
    out = tmp_path_factory.mktemp("palace_lossy_port")
    manifest = probe_manifest(PROBE, out, "LOSSY_PORT_OUT", key=None)
    if manifest.get("missing"):
        pytest.skip(f"this machine cannot drive Palace: {manifest['missing']}")
    assert sorted(manifest["cases"]) == sorted(DRAWN), (
        f"the probe wrote {manifest['cases']} rather than {DRAWN}, so it stopped "
        "before every run finished"
    )
    return manifest


def carried(case, port, edge):
    """The one propagating constant a port's mode run found at an end of the band."""
    (only,) = [
        complex(real, imaginary)
        for real, imaginary in case["found"][f"{port}/{edge}"]
        if propagates(complex(real, imaginary))
    ]
    return only


def test_a_face_reflecting_past_the_bar_is_refused_before_the_band(run_manifest):
    case = run_manifest["absorbing"]
    assert not case["solved"], "the band was solved"
    assert "as though the model did" in case["said"], case["said"]
    assert "'Port1'" in case["said"] and "the bottom of the band" in case["said"]


def test_a_filling_that_hides_the_guides_mode_is_refused_saying_so(run_manifest):
    case = run_manifest["conducting"]
    assert not case["solved"], "the band was solved"
    assert "A lossy filling can also hide the guide's own mode" in case["said"], case["said"]


def test_a_face_reflecting_under_the_bar_is_solved_and_states_it(run_manifest):
    case = run_manifest["board"]
    assert case["solved"] and not case["said"], case["said"]
    assert case["carried"] and all("its face reflects" in line for line in case["carried"])
    bottom = carried(case, 1, "bottom")
    print(f"\nWR-42 filled, loss tangent 0.02, on Palace: port 1 reflects {reflected(bottom):.3g}")
    assert 4 * reflected(bottom) <= BAR


def two_faces(constant, length):
    """|S11|^2 of a guide of this constant between two port lines of its real part.

    A TE mode's impedance is omega mu over its constant, so each line is written
    as the reciprocal of its constant, and the guide is a line of ``length``
    metres terminated by the second port's.
    """
    port, guide = 1 / constant.real, 1 / constant
    turn = np.tan(constant * length)
    seen = guide * (port + 1j * guide * turn) / (guide + 1j * port * turn)
    return abs((seen - port) / (seen + port)) ** 2


def test_what_the_faces_reflect_is_what_the_constant_says(run_manifest):
    """A uniform guide reflects nothing, so S11 is the two port faces' own: each
    reflects its share, and the second's returns along the guide."""
    case = run_manifest["board"]
    s11 = np.abs(np.asarray(case["real"])[:, 0, 0] + 1j * np.asarray(case["imaginary"])[:, 0, 0])
    first = [complex(real, imaginary) for port, real, imaginary in case["taken"] if port == 1]
    assert len(first) == len(s11)
    for measured, constant in zip(s11**2, first, strict=True):
        assert measured == pytest.approx(two_faces(constant, case["length"] / 1e3), rel=AGREEMENT)
        assert measured <= 4 * reflected(constant) * (1 + AGREEMENT)


def test_each_port_takes_the_mode_its_face_carries(run_manifest):
    case = run_manifest["board"]
    taken = case["taken"]
    assert {port for port, _, _ in taken} == {1, 2}
    assert all(propagates(complex(real, imaginary)) for _, real, imaginary in taken)
    samples = len(taken) // 2
    for port in (1, 2):
        mine = [complex(real, imaginary) for at, real, imaginary in taken if at == port]
        assert len(mine) == samples
        for edge, constant in (("bottom", mine[0]), ("top", mine[-1])):
            found = carried(case, port, edge)
            assert abs(constant - found) <= SAME * abs(found), (port, edge)
