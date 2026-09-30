# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Acceptance gate: a WR-42 guide answered by Palace, from a real document.

The same guide, the same band and the same closed form as
``tests/test_acceptance_openems_waveguide.py``, answered by the other backend
and reached the way a user reaches it - a drawing marked up with document
objects, handed to the adapter, meshed, solved, and read back as a matrix.
Everything asserted falls out of separating Maxwell's equations in a hollow
rectangular pipe of perfect conductor, so there is no fitted constant and no
quoted accuracy::

    kc   = pi / a                              transverse wavenumber, TE10
    beta = sqrt(k0^2 - kc^2)                   phase constant
    S21  = exp(-j * beta * L)

The phase is fitted rather than compared point by point. The slope of the
unwrapped phase against the analytic beta is the distance the wave crossed, and
a constant offset is absorbed by the fit - which matters here, because a wave
port on this backend is reported against its own mode and there is no reference
impedance in the run to normalise from.

What this gate exercises that the other one cannot
--------------------------------------------------

A conforming tetrahedral mesh rather than a staircased grid. A port standing on
the guide's own end face, which is where a wave leaves when there is no
absorber to run out through. And the whole route from a document to a matrix in
one call, which is the join every stage of this adapter answers into.

**The same guide is drawn twice**, once as one body and once as two that meet
halfway along, and every figure is read off both. The drawing is not the device:
a region drawn in two pieces is one region, and the face where they meet is
inside the model rather than on its boundary. What that catches is a wall
condition written over that face - which sends back a guide half as long, with a
clean exit, a full set of tables and no warning about any of it.

The two are numbered differently as well. The mesher numbers its groups in the
order of their names, and the one body's region sorts after the ports where the
two bodies' sort before them - so the region stands above the ports in one mesh
and below them in the other, and the two answers agreeing says the numbering
does not reach the answer either.

**It is also drawn as two bodies a micron apart**. The study fills the gap with
vacuum, as each body is, so one body grows over it and the guide is solved as
one, and the run says which body grew. That drawing is held to the one body's
answer as the others are.

**And the shipped example is solved**, drawn by its own script for the other
backend: its ports stand on planes drawn across the air, the guide runs on past
each of them, and the mesher leaves out what stands behind each plane. The
ports are referred a few millimetres past each plane, so Palace moves each
answer along the guide by the port's ``Offset``; the separation it recovers,
and the phase it leaves over, are those of the reference planes, read off the
document. A sign or a unit wrong in the ``Offset`` moves the phase by twice the
wave's phase over the depth, which the phase left over separates outright.
Scored against its own planes only.

Why a subprocess
----------------

This gate starts from a drawing, and the interpreter the suite runs under has no
CAD kernel. ``tests/palace_waveguide_probe.py`` runs under ``freecadcmd``, draws
the guide and drives the run; this side holds what came back against the closed
form. The mesher and Palace are processes of their own either way, and the
adapter is what starts them. A machine with neither makes the probe say which,
and the gate skips.

Both are found the way the adapter finds them anywhere: Palace on the path or
at ``$MICROWAVE_PALACE``, and a Python that can import Gmsh at
``$MICROWAVE_GMSH_PYTHON``. Nothing else is guessed at, so an installation the
path does not reach makes this gate skip until one of those names it.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.analytic import reference
from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_waveguide_probe.py")

#: How far the recovered separation may sit from the length drawn.
#:
#: The bar separates a run that carried the mode from one that did not, and the
#: two are nowhere near each other. A run that carries it recovers the length to
#: parts in a hundred thousand. A run that found a different mode, or met a mesh
#: too coarse to carry one, is out by tens of percent. One percent stands
#: between them, and it is not a claim about how exact this method is - what the
#: run reached is on the gate's own line, and it is far inside this.
LENGTH_TOLERANCE = 0.01

#: What a straight line fitted to noise looks like. A run that carried the mode
#: has an unwrapped phase that *is* the line, to a small fraction of a degree;
#: a run that did not still yields a slope, and a plausible length with it. So
#: the residual is read beside the length rather than instead of it, and this
#: bar is loose because the two cases are orders of magnitude apart.
RESIDUAL_TOLERANCE_DEGREES = 1.0

#: A uniform guide with a wave port at each end reflects nothing but its own
#: discretisation. What this catches is a port on the wrong face, an ordinal
#: that reached a mode the far end does not launch, or a wall condition that
#: never landed - each of which sends back a large fraction of what went in.
MATCH_TOLERANCE = 1e-2

#: A hollow perfect guide in vacuum is lossless and a wave port absorbs its own
#: mode, so each column of the matrix carries one power in and one out.
POWER_TOLERANCE = 1e-3

#: How far two drawings of this guide may disagree about a term of the
#: matrix. The bar separates the drawing from the device: a wall over the face
#: where the two bodies meet shorts the guide in the middle, and the through
#: term then falls from one to nothing, which is a disagreement of order one.
#: What two conforming meshes of one guide reach is far inside this, and what
#: they reached is on the gate's own line.
DRAWING_TOLERANCE = 1e-3

#: The guide is reciprocal and its two ends are identical, so the two
#: off-diagonal terms are one number measured twice. They come from different
#: solves, which is what makes the comparison worth making.
RECIPROCITY_TOLERANCE = 1e-3

#: How much phase a straight guide's transmission may carry beyond the wave's
#: own. A mode turned the other way at one port is half a turn, so the bar
#: separates the two outright; what the guide reaches is on the gate's line.
SIGN_TOLERANCE_DEGREES = 1.0

#: How far each port's stated impedance may stand from TE10's power-voltage
#: impedance. Palace reads the voltage off the mode it solved on the port's
#: face, so this holds the face's mode and the line across it together; a line
#: off the middle of the broad side reads a smaller voltage, and one that missed
#: the face reads none.
IMPEDANCE_TOLERANCE = 1e-3


#: The drawings of this one guide, by the name the probe writes each under,
#: against how each says the same device. The second is two bodies that touch,
#: whose shared face is inside the model: a wall written across it is a perfect
#: conductor cutting the guide, and what comes back is two guides half as long,
#: measured on a run that exits zero and warns about nothing. The third is the
#: guide's walls with nothing bound inside, where the medium the study fills
#: undrawn room with is the guide's air. So every figure below is read off each
#: drawing, and each is held to agreeing with the first.
DRAWN = {
    "wr42": "one body",
    "wr42_cut": "two bodies that touch",
    "wr42_apart": "two bodies a micron apart",
    "wr42_walls": "its walls, with nothing bound inside",
}

#: The drawing of two bodies a micron apart, which one body grows over.
APART = "wr42_apart"

#: The shipped example, which is scored against theory like the drawings in
#: ``DRAWN`` and not against them, its ports standing at planes of its own.
EXAMPLE = "wr42_example"

#: Every drawing the gate solves, against how each says the device.
SOLVED = {
    **DRAWN,
    EXAMPLE: "the shipped example, run on past its port planes and referred past them",
}


@pytest.fixture(scope="module")
def run_manifest(tmp_path_factory):
    """Every drawing, driven under a real FreeCAD, as the probe left them."""
    out = tmp_path_factory.mktemp("palace_guide")
    manifest = probe_manifest(PROBE, out, "WAVEGUIDE_PALACE_OUT", key=None)
    if manifest.get("missing"):
        pytest.skip(f"this machine cannot drive Palace: {manifest['missing']}")
    assert manifest["cases"] == list(SOLVED), (
        f"the probe wrote {manifest['cases']} rather than {list(SOLVED)}, so it "
        "stopped before every run finished. A gate that is not run is not a "
        "gate that passed"
    )
    return manifest


@pytest.fixture(scope="module", params=list(SOLVED))
def solved(request, run_manifest):
    """One of the drawings, as a matrix."""
    read = run_manifest[request.param]
    read["s"] = np.asarray(read["real"], dtype=float) + 1j * np.asarray(
        read["imaginary"], dtype=float
    )
    read["frequency"] = np.asarray(read["frequency"], dtype=float)
    read["drawn"] = SOLVED[request.param]
    return read


def entry(solved, out: int, driven: int) -> np.ndarray:
    """``S[out][driven]`` by port number rather than by position."""
    return solved["s"][:, solved["out"].index(out), solved["driven"].index(driven)]


def test_phase_constant_matches_theory(solved):
    """The gate: the wave that crossed the guide has the phase theory gives it.

    The recovered separation is the distance between the faces the ports stand
    on: the guide's own length where they stand on its ends, and the distance
    between the planes where the guide runs on past them.
    """
    beta = np.real(
        reference.phase_constant(
            solved["frequency"], solved["broad"] * 1e-3, solved["narrow"] * 1e-3, 1, 0
        )
    )
    phase = np.unwrap(np.angle(entry(solved, 2, 1)))
    slope, offset = np.polyfit(beta, phase, 1)
    recovered = -slope
    drawn = solved["length"] * 1e-3
    error = (recovered - drawn) / drawn
    residual = float(np.max(np.abs(np.polyval([slope, offset], beta) - phase)))

    print(
        f"\nGATE WR-42 on Palace ({solved['drawn']}): "
        f"plane separation = {recovered * 1e3:.4f} mm, "
        f"geometry {drawn * 1e3:.4f} mm, {error * 100:+.4f}%, fit residual "
        f"{np.degrees(residual):.3e} deg "
        f"[order {solved['order']}, {solved['elements_per_wavelength']} elements "
        "per wavelength]"
    )
    assert np.degrees(residual) < RESIDUAL_TOLERANCE_DEGREES, (
        f"the unwrapped phase departs from a straight line in beta by "
        f"{np.degrees(residual):.3g} degrees, so the slope below is arithmetic "
        "over something that is not one mode crossing the guide"
    )
    assert abs(error) < LENGTH_TOLERANCE, (
        f"the phase of S21 implies the port faces are {recovered * 1e3:.3f} mm "
        f"apart, and the guide is drawn {drawn * 1e3:.3f} mm long "
        f"({error * 100:+.2f}%). Either its phase constant is wrong or its "
        "cutoff is"
    )


def test_each_ports_mode_is_turned_one_way(solved):
    """The line each port's voltage is read along turns its mode, and it runs
    upwards along the same axis at both ends, so a straight guide transmits
    the wave's own phase, ``-beta L``, with nothing left over. A mode turned the
    other way at one port leaves half a turn, and the length the slope gives is
    blind to it."""
    beta = np.real(
        reference.phase_constant(
            solved["frequency"], solved["broad"] * 1e-3, solved["narrow"] * 1e-3, 1, 0
        )
    )
    left = np.angle(entry(solved, 2, 1) * np.exp(1j * beta * solved["length"] * 1e-3))
    worst = float(np.degrees(np.max(np.abs(left))))
    print(f"\nGATE WR-42 on Palace ({solved['drawn']}): worst phase left over = {worst:.3e} deg")
    assert worst < SIGN_TOLERANCE_DEGREES, (
        f"S21 carries {worst:.3g} degrees beyond the wave's own phase, so the ports' "
        "modes are not turned one way"
    )


def test_the_guide_is_matched(solved):
    """Read at both ends, because a fault confined to one port reaches the other
    end only through the through term."""
    worst = max(float(np.max(np.abs(entry(solved, port, port)))) for port in solved["driven"])
    print(f"\nGATE WR-42 on Palace ({solved['drawn']}): worst reflection = {worst:.3e}")
    assert worst < MATCH_TOLERANCE, (
        f"|S11| reaches {worst:.4g}; a uniform guide with a wave port on each "
        "end sends back only its own discretisation"
    )


def test_power_is_conserved(solved):
    worst = 0.0
    for driving in solved["driven"]:
        column = sum(np.abs(entry(solved, measured, driving)) ** 2 for measured in solved["out"])
        worst = max(worst, float(np.max(np.abs(column - 1.0))))
    print(f"\nGATE WR-42 on Palace ({solved['drawn']}): worst power departure = {worst:.3e}")
    assert worst < POWER_TOLERANCE, (
        f"a column of the matrix carries {worst:.3g} more or less power than "
        "went into it, and a hollow guide of perfect conductor in vacuum is "
        "lossless"
    )


def test_each_port_states_its_power_voltage_impedance(solved):
    """The number a file of this guide carries for each port, which is what a
    renormalisation or a cascade with another guide's file reads."""
    theory = reference.power_voltage_impedance(
        solved["frequency"], solved["broad"] * 1e-3, solved["narrow"] * 1e-3
    )
    assert solved["impedance"] is not None, "the run read no port's voltage"
    stated = np.asarray(solved["impedance"], dtype=float)
    worst = float(np.max(np.abs(stated / theory[:, None] - 1.0)))
    print(
        f"\nGATE WR-42 on Palace ({solved['drawn']}): worst power-voltage impedance "
        f"departure = {worst:.3e}"
    )
    assert stated.shape == (solved["frequency"].size, len(solved["out"]))
    assert worst < IMPEDANCE_TOLERANCE, (
        f"a port states an impedance {worst:.3g} away from TE10's power-voltage "
        "impedance, so the line its voltage is read along is not across the middle "
        "of the guide, or the mode is not TE10"
    )


def test_the_two_solves_agree_about_the_same_number(solved):
    """The guide is reciprocal, and each off-diagonal term is a different solve.

    Nothing in the run couples them. Palace loops over the excitations and
    solves each one, measuring every port after each solve
    (``palace/drivers/drivensolver.cpp:154-223``), so a column comes from the
    solve that drove its port and no off-diagonal term is filled from the one
    opposite it. This is two measurements of one quantity rather than a symmetry
    the arithmetic imposed.
    """
    gap = float(np.max(np.abs(entry(solved, 2, 1) - entry(solved, 1, 2))))
    print(f"\nGATE WR-42 on Palace ({solved['drawn']}): worst reciprocity gap = {gap:.3e}")
    assert gap < RECIPROCITY_TOLERANCE, (
        f"S21 and S12 differ by {gap:.3g}, and a hollow guide is reciprocal"
    )


def test_the_drawing_is_not_the_device(run_manifest):
    """One guide drawn several ways answers the same, and no reference is involved.

    Each drawing is scored against theory above, and a fault that moved them
    alike would pass there. This compares them with each other, which is what
    catches the drawing reaching the answer: a region drawn as two bodies is one
    region, and the face where they meet carries no condition; and room nobody
    bound inside walls of metal is the air a region of vacuum would be.
    """
    matrices = {}
    for case in DRAWN:
        read = run_manifest[case]
        # Compared entry by entry, so the two have to be indexed alike. Both
        # documents number their ports the same way and nothing here reorders
        # them, which is a claim rather than an arrangement.
        assert (read["out"], read["driven"]) == (
            run_manifest[next(iter(DRAWN))]["out"],
            run_manifest[next(iter(DRAWN))]["driven"],
        )
        matrices[case] = np.asarray(read["real"], dtype=float) + 1j * np.asarray(
            read["imaginary"], dtype=float
        )
    whole, *others = DRAWN
    for other in others:
        gap = float(np.max(np.abs(matrices[whole] - matrices[other])))
        print(
            f"\nGATE WR-42 on Palace: worst gap between {DRAWN[whole]} and "
            f"{DRAWN[other]} = {gap:.3e}"
        )
        assert gap < DRAWING_TOLERANCE, (
            f"the guide drawn as {DRAWN[whole]} and as {DRAWN[other]} answer {gap:.3g} "
            "apart, and they are one device"
        )


def test_the_room_between_the_walls_and_the_box_is_left_out(run_manifest):
    """The guide drawn as its walls stands in the box round it, and the room its
    chamfered edges and the blind hole in its top leave there is sealed off by
    the metal: said before the solve, one line to a room."""
    walls = run_manifest["wr42_walls"]
    said = walls["sealed"]
    corners = [line for line in said if "'HousingBinding' and the domain's sides" in line]
    hole = [line for line in said if "'HousingBinding' and the domain's side ZMax" in line]
    assert len(corners) == 4, said
    assert len(hole) == 1, said
    # The rooms left out are no reserved air, so no side of the box is said to
    # carry a wall the drawing does not cover: the housing, the ports' faces and
    # the rooms left out cover each.
    assert walls["walled"] == [], walls["walled"]


def test_the_corners_of_the_housing_carry_no_refinement(run_manifest):
    """The room turns a quarter turn round each inner edge of the housing, where
    the field is not singular, so the edges are meshed as the corners of the
    guide drawn as its air are. Said before the solve, under the housing's
    binding."""
    (line,) = run_manifest["wr42_walls"]["unturned"]
    assert "'HousingBinding'" in line


def test_what_stands_behind_each_plane_of_the_shipped_guide_is_left_out(run_manifest):
    """Said before the solve, one line per part, naming the port it stood
    behind and what it was drawn as."""
    said = run_manifest[EXAMPLE]["left out"]
    assert len(said) == 2, said
    assert "'Port1'" in said[0] and "'Port2'" in said[1]
    assert all("'GuideAirBinding'" in line for line in said)


def test_a_guide_drawn_as_two_bodies_a_hair_apart_says_one_grew_over_the_gap(run_manifest):
    """The run says where the gap was, how thin, and which body is solved over
    it, since the model solved is not the drawing."""
    (said,) = run_manifest[APART]["joined"]
    assert "0.001 mm thick" in said
    assert "GuideAir1Binding" in said and "GuideAir2Binding" in said
    assert run_manifest["wr42"]["joined"] == run_manifest["wr42_cut"]["joined"] == []
