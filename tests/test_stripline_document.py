# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The shipped stripline, held against the line the acceptance gate solves.

``examples/stripline_50ohm.FCStd`` gives a user a route from the GUI to a number
that owes nothing to a solver: a stripline is TEM, so conformal mapping gives its
impedance exactly, where every microstrip expression is a fit. It translates to
the structure ``tests/stripline.py`` builds by hand and
``test_acceptance_stripline`` scores, so what that gate measured about this line
is a measurement about this file. The comparison is therefore an equality rather
than a second solve, over what reaches the engine rather than what only a reader
sees - the title, the labels, and what the two sides call their materials.

The one difference is the strip. The gate draws none - ``MSLPort`` lays it -
while a document has to draw one, because that is what a port's ``TraceEnd`` is
picked off. Both are the same material and the drawn sheet lies inside the port's
own box, so CSXCAD gives every cell there to the port's primitive and the drawn
one is reported unused.

Marked slow because it starts FreeCAD, not because it solves. Nothing here does.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from Microwave.Solvers.openems import preflight
from Microwave.Solvers.openems.model import Problem
from tests import stripline
from tests.conftest import draw_cases

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "stripline_document_probe.py")

#: The name the probe writes under, which is the document's own stem.
NAME = "stripline_50ohm"

#: How far apart two grid lines may be and still be the same line, in mm. One
#: nanometre: both sides are built by the same mesher from independently
#: declared dimensions, so this allows for the arithmetic reaching a line by two
#: routes and nothing else.
SAME_LINE = 1e-9


@pytest.fixture(scope="module")
def translated(tmp_path_factory):
    """The shipped document, opened under a real FreeCAD and translated."""
    drawn = draw_cases(
        PROBE, tmp_path_factory.mktemp("stripline_document"), "STRIPLINE_DOCUMENT_OUT"
    )
    assert NAME in drawn, (
        f"the probe opened {NAME}.FCStd and found no EMAnalysis in it. A "
        "document restored with the workbench reached by sys.path instead of "
        "installed in Mod/ comes back with Proxy objects set to None, partially "
        "and without saying so - install it and run this again"
    )
    return Problem.from_dict(json.loads((drawn[NAME] / "openems.json").read_text()))


@pytest.fixture(scope="module")
def hand_built():
    """The same line as the acceptance gate builds it, at its operating point."""
    return stripline.problem(**stripline.cases()[stripline.NOMINAL])


def test_the_two_routes_plan_one_grid(translated, hand_built):
    """Line by line rather than by cell count: two different grids agree on a
    count whenever they happen to be the same size."""
    for axis in ("x", "y", "z"):
        ours, theirs = getattr(translated.grid, axis), getattr(hand_built.grid, axis)
        assert len(ours) == len(theirs), (
            f"the document's {axis} grid has {len(ours)} lines against the "
            f"gate's {len(theirs)}, so the shipped file is no longer the "
            "structure the gate measured"
        )
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=SAME_LINE)


def test_both_are_asked_the_same_question(translated, hand_built):
    """Everything about the run that is not the grid and not the geometry.

    Each of these moves the answer on its own: a shorter record truncates the
    response the DFT is taken of, a wall that absorbs where the gate's conducts
    is a different line, a timestep factor below one covers less simulated time
    for the same step count, and the length unit scales the whole drawing.
    """
    for field in (
        "frequency",
        "boundary",
        "termination",
        "length_unit",
        "timestep_factor",
        "smallest_response",
    ):
        assert getattr(translated, field) == getattr(hand_built, field), (
            f"the document's {field} is {getattr(translated, field)!r} against "
            f"the gate's {getattr(hand_built, field)!r}"
        )


def test_the_fill_is_the_same_region_of_the_same_stuff(translated, hand_built):
    """Compared by what it is rather than by what it is called: the two sides
    name their materials for different readers, and neither name reaches the
    engine's physics."""
    ours = _dielectrics(translated)
    theirs = _dielectrics(hand_built)
    assert len(ours) == len(theirs) == 1, (
        f"the document holds {len(ours)} dielectric solids and the gate "
        f"{len(theirs)}; this line is one box of fill"
    )
    assert ours[0].lower == theirs[0].lower
    assert ours[0].upper == theirs[0].upper
    assert _as_built(translated, ours[0].material) == _as_built(hand_built, theirs[0].material)


def test_the_port_lays_the_same_metal(translated, hand_built):
    """What the strip is made of, which no other comparison here reaches.

    The grid does not carry it: a lossy sheet in the same place asks the mesher
    for the same lines, so two envelopes can plan one grid and hand the engine a
    perfect conductor and a resistive one. The port fields are compared by name,
    and the metal is a name.
    """
    assert _as_built(translated, translated.ports[0].metal) == _as_built(
        hand_built, hand_built.ports[0].metal
    )


def test_the_port_is_the_same_port(translated, hand_built):
    """Everything about it except what it is called and what its metal is
    called. Corner ordering is in here: reversing ``start`` and ``stop`` inverts
    the excitation and still solves cleanly."""
    (ours,), (theirs,) = translated.ports, hand_built.ports
    for field in (
        "number",
        "kind",
        "start",
        "stop",
        "propagation_axis",
        "excitation_axis",
        "excite",
        "feed_shift",
        "measurement_shift",
        "feed_resistance",
        "reference_impedance",
    ):
        assert getattr(ours, field) == getattr(theirs, field), (
            f"the document's port has {field} = {getattr(ours, field)!r} against "
            f"the gate's {getattr(theirs, field)!r}"
        )


def test_the_only_extra_conductor_is_one_the_port_covers(translated, hand_built):
    """The one difference between the two envelopes, and why it is not one.

    A document draws its strip because a port is picked off a face and takes its
    metal from what that face is bound to. ``MSLPort`` then lays its own strip
    over the whole port box in the same material, CSXCAD assigns each cell to one
    primitive, and the drawn sheet is left with none - which openEMS reports as
    an unused primitive.

    Containment is what makes that benign, so containment is what is asserted: a
    sheet reaching past the port box would be metal the port does not lay.
    """
    port = translated.ports[0]
    extra = [solid for solid in translated.solids if solid.material == port.metal]
    assert len(extra) == 1, (
        f"{len(extra)} solids are made of the port's own metal; the strip is one "
        "sheet and the port lays the rest"
    )
    sheet = extra[0]
    # The gate's own name for its metal, not the document's: the two envelopes
    # name their materials separately, so this side's name against that side's
    # solids is always false and guards nothing.
    theirs = hand_built.ports[0].metal
    assert not any(solid.material == theirs for solid in hand_built.solids), (
        "the gate has grown a drawn conductor of its own, so the difference this "
        "test exists to account for is no longer the difference"
    )

    down = port.excitation_axis
    assert sheet.lower[down] == sheet.upper[down] == port.start[down], (
        f"the strip sits from {sheet.lower[down]:g} to {sheet.upper[down]:g} on "
        f"the excitation axis and the port lays its metal at {port.start[down]:g}"
    )
    for axis in range(3):
        low, high = sorted((port.start[axis], port.stop[axis]))
        assert low <= sheet.lower[axis] and sheet.upper[axis] <= high, (
            f"the drawn strip reaches outside the port's box on "
            f"{'xyz'[axis]}, so it is metal the port does not lay"
        )


def test_the_document_would_be_allowed_to_run(translated):
    """Pre-flight over the shipped file: a finding here is one every reader of
    this example sees before anything solves."""
    findings = preflight.check(translated)
    assert not findings, f"{NAME}.FCStd draws pre-flight findings: {findings}"


def _dielectrics(problem):
    kinds = {material.name: material.kind for material in problem.materials}
    return [solid for solid in problem.solids if kinds[solid.material] == "dielectric"]


#: What each kind of material carries into the engine, beyond being that kind.
#: Taken from the branches of
#: ``Microwave/Solvers/openems/driver.py::_add_material``, which is the whole of
#: what CSXCAD is told - a perfect conductor is a name and nothing else, so a
#: thickness the document object carries reaches no argument on this route.
_BUILT_FROM = {
    "pec": (),
    "conducting_sheet": ("conductivity", "thickness"),
    "lossy_dielectric": ("epsilon", "mu", "kappa"),
    "dielectric": ("epsilon", "mu"),
}


def _as_built(problem, name):
    """One material as the arguments the engine would be given for it.

    Not the dataclass and not its name: each envelope carries whatever fields its
    own route filled in, so comparing the objects finds differences that reach
    nothing, while the names differ always.
    """
    material = next(found for found in problem.materials if found.name == name)
    fields = _BUILT_FROM[material.kind]
    return (material.kind, *(getattr(material, field) for field in fields))
