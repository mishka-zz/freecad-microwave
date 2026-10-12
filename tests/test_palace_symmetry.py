# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checking a declared mirror against what a Palace run is given.

The same policy as the other backend's: nothing is refused, and what has to be
right is which mismatches are said. The ports, regions and metals are the
adapter's own types and the problem holding them is a stand-in for what the
check reads. Each shape is a box of one piece, whose products of inertia about
its own centre are zero.

Why a subprocess
----------------

A drawn shape is read for its mass, its centre of mass and its products of
inertia, which are the CAD kernel's, and a trace drawn as one face is not a box.
``tests/palace_symmetry_probe.py`` draws boards under ``freecadcmd`` and asks
the check about each. Nothing is translated, meshed or solved.
"""

import math
import os
from types import SimpleNamespace

import pytest

from Microwave.Gui.symmetry import palace_mirror_warnings
from Microwave.Solvers.palace.problem import (
    Conductor,
    Element,
    Feed,
    LossyConductor,
    LumpedFeed,
    Region,
)
from tests.conftest import probe_manifest

PROBE = os.path.join(os.path.dirname(__file__), "palace_symmetry_probe.py")


def box(lower, upper):
    """A shape as the check reads one: its bounding box, and itself as one piece,
    a body where it has three extents and a sheet where it has two."""
    corners = dict(zip(("XMin", "YMin", "ZMin"), lower, strict=True))
    corners.update(zip(("XMax", "YMax", "ZMax"), upper, strict=True))
    extents = [high - low for low, high in zip(lower, upper, strict=True)]
    piece = SimpleNamespace(
        Faces=[],
        Mass=math.prod(extent for extent in extents if extent > 0),
        CenterOfMass=tuple((low + high) / 2 for low, high in zip(lower, upper, strict=True)),
        MatrixOfInertia=SimpleNamespace(
            **{f"A{row}{column}": 0.0 for row in range(1, 4) for column in range(1, 4)}
        ),
    )
    body = all(extent > 0 for extent in extents)
    return SimpleNamespace(
        BoundBox=SimpleNamespace(**corners),
        Solids=[piece] if body else [],
        Faces=[] if body else [piece],
    )


def described(ports, regions, conductors=()):
    """What the check reads of a Palace problem."""
    return SimpleNamespace(
        ports=tuple(sorted(ports, key=lambda port: port.number)),
        regions=tuple(regions),
        conductors=tuple(conductors),
        lossy=(),
    )


def lumped(number, x, resistance=50.0):
    """A lumped port across the board's height at ``x``, under a trace two
    millimetres wide."""
    element = Element(label=f"Port{number}", shapes=(box((x, -1, 0), (x, 1, 1.6)),), direction="+Z")
    return LumpedFeed(
        label=f"Port{number}", number=number, elements=(element,), resistance=resistance
    )


def board(ports, extra=()):
    """A uniform trace over a ground plane, forty millimetres long on X."""
    return described(
        ports,
        regions=(Region(label="FR4", shapes=(box((-20, -5, 0), (20, 5, 1.6)),), filling=None),),
        conductors=(
            Conductor(label="Trace", shapes=(box((-20, -1, 1.6), (20, 1, 1.6)),)),
            Conductor(label="Ground", shapes=(box((-20, -5, 0), (20, 5, 0)),)),
            *extra,
        ),
    )


def wave(number, x, offset=0.0):
    """A wave port across a WR-42 guide drawn along X, at ``x``."""
    face = box((x, 0, 0), (x, 10.7, 4.3))
    return Feed(
        label=f"Port{number}", shapes=(face,), number=number, behind=(x, 5, 2), offset=offset
    )


def guide(ports):
    return described(
        ports,
        regions=(Region(label="Air", shapes=(box((0, 0, 0), (50, 10.7, 4.3)),), filling=None),),
    )


class TestAMirroredStudyIsLeftAlone:
    def test_a_uniform_board_with_matched_ports_says_nothing(self):
        assert palace_mirror_warnings(board((lumped(1, -20), lumped(2, 20)))) == []

    def test_a_guide_with_its_planes_mirrored_says_nothing(self):
        assert palace_mirror_warnings(guide((wave(1, 5), wave(2, 45)))) == []

    def test_a_guide_drawn_along_y_is_mirrored_along_y(self):
        """The mirror is the plane between the two ports, whichever axis they
        lie on."""
        ports = []
        for number, y in ((1, 5), (2, 45)):
            face = box((0, y, 0), (10.7, y, 4.3))
            ports.append(
                Feed(label=f"Port{number}", shapes=(face,), number=number, behind=(5, y, 2))
            )
        air = Region(label="Air", shapes=(box((0, 0, 0), (10.7, 50, 4.3)),), filling=None)
        assert palace_mirror_warnings(described(ports, (air,))) == []


class TestThePortsMustBeMirrors:
    def test_two_lumped_ports_of_different_resistance_are_named(self):
        found = palace_mirror_warnings(board((lumped(1, -20), lumped(2, 20, resistance=25.0))))
        assert any(
            w.startswith("Port 1 has a reference impedance of 50 ohm and port 2 of 25 ohm")
            for w in found
        ), found

    def test_two_wave_ports_referred_at_different_depths_are_named(self):
        found = palace_mirror_warnings(guide((wave(1, 5), wave(2, 45, offset=1.0))))
        assert any("different reference depths (0 against 1)" in w for w in found), found

    def test_a_port_off_the_mirror_is_named_with_how_far(self):
        found = palace_mirror_warnings(guide((wave(1, 5), wave(2, 43))))
        assert any(w.startswith("Port 2 stands 2 mm from where port 1's") for w in found), found

    def test_a_port_off_by_rounding_is_not_named(self):
        """A coordinate that differs in its last digits is the same place."""
        assert palace_mirror_warnings(guide((wave(1, 5), wave(2, 45 + 1e-9)))) == []

    def test_a_port_moved_across_the_mirror_is_named(self):
        """At the mirrored place along the guide and a millimetre aside."""
        element = Element(label="Port2", shapes=(box((20, 0, 0), (20, 2, 1.6)),), direction="+Z")
        moved = LumpedFeed(label="Port2", number=2, elements=(element,), resistance=50.0)
        found = palace_mirror_warnings(board((lumped(1, -20), moved)))
        assert any(w.startswith("Port 2 stands 1 mm from where port 1's") for w in found), found

    def test_a_port_of_two_elements_is_compared_by_the_box_round_both(self):
        """Port 1 spans the trace in two elements, port 2 in one element over half
        of it."""

        def element(low, high):
            return Element(
                label="Port1", shapes=(box((-20, low, 0), (-20, high, 1.6)),), direction="+Z"
            )

        split = LumpedFeed(
            label="Port1", number=1, elements=(element(-1, 0), element(0, 1)), resistance=50.0
        )
        half = Element(label="Port2", shapes=(box((20, 0, 0), (20, 1, 1.6)),), direction="+Z")
        found = palace_mirror_warnings(
            board((split, LumpedFeed(label="Port2", number=2, elements=(half,), resistance=50.0)))
        )
        assert any(w.startswith("Port 2 stands 1 mm from where port 1's") for w in found), found


class TestTheDrawingMustBeAMirror:
    def test_a_stub_on_one_side_is_seen(self):
        stub = Conductor(label="Trace", shapes=(box((-12, 1, 1.6), (-8, 3, 1.6)),))
        found = palace_mirror_warnings(board((lumped(1, -20), lumped(2, 20)), extra=(stub,)))
        assert any("not a mirror image about the middle" in w for w in found), found

    def test_mirrored_stubs_bound_apart_are_partners(self):
        """Two perfect conductors are one material whichever binding holds each."""
        stubs = (
            Conductor(label="StubL", shapes=(box((-12, 1, 1.6), (-8, 3, 1.6)),)),
            Conductor(label="StubR", shapes=(box((8, 1, 1.6), (12, 3, 1.6)),)),
        )
        assert palace_mirror_warnings(board((lumped(1, -20), lumped(2, 20)), extra=stubs)) == []

    def test_mirrored_lossy_sheets_are_matched_by_their_material(self):
        def sheet(name, material, low, high):
            shape = box((low, 1, 1.6), (high, 3, 1.6))
            return LossyConductor(
                label=name,
                shapes=(shape,),
                conductivity=5.8e7,
                permeability=1.0,
                thickness=0.0,
                material=material,
            )

        ports = (lumped(1, -20), lumped(2, 20))
        alike = described(ports, board(ports).regions, board(ports).conductors)
        alike.lossy = (sheet("CuL", "Copper", -12, -8), sheet("CuR", "Copper", 8, 12))
        assert palace_mirror_warnings(alike) == []
        alike.lossy = (sheet("CuL", "Copper", -12, -8), sheet("AlR", "Aluminium", 8, 12))
        found = palace_mirror_warnings(alike)
        assert any("not a mirror image about the middle" in w for w in found), found

    def test_mirrored_regions_are_matched_by_their_material(self):
        """Two bindings of one dielectric, one each side, are a mirror; two
        dielectrics are not."""

        def half(name, material, low, high):
            return Region(
                label=name,
                shapes=(box((low, -5, 0), (high, 5, 1.6)),),
                filling=None,
                material=material,
            )

        ports = (lumped(1, -20), lumped(2, 20))
        alike = (half("Left", "FR4", -20, 0), half("Right", "FR4", 0, 20))
        assert palace_mirror_warnings(described(ports, alike)) == []
        unlike = (half("Left", "FR4", -20, 0), half("Right", "Rogers", 0, 20))
        found = palace_mirror_warnings(described(ports, unlike))
        assert any("not a mirror image about the middle" in w for w in found), found

    def test_stubs_mirrored_along_the_line_and_not_across_it_are_seen(self):
        """Each stub has its partner's distance from the mirror, on the other
        side of the trace."""
        stubs = (
            Conductor(label="StubL", shapes=(box((-12, 1, 1.6), (-8, 3, 1.6)),)),
            Conductor(label="StubR", shapes=(box((8, -3, 1.6), (12, -1, 1.6)),)),
        )
        found = palace_mirror_warnings(board((lumped(1, -20), lumped(2, 20)), extra=stubs))
        assert any("not a mirror image about the middle" in w for w in found), found

    def test_a_body_is_not_the_partner_of_a_sheet_of_its_metal(self):
        """A body and a sheet of one area at mirrored places: a volume is not
        an area."""
        stubs = (
            Conductor(label="Body", shapes=(box((-12, 1, 1.1), (-8, 3, 2.1)),)),
            Conductor(label="Sheet", shapes=(box((8, 1, 1.6), (12, 3, 1.6)),)),
        )
        found = palace_mirror_warnings(board((lumped(1, -20), lumped(2, 20)), extra=stubs))
        assert any("not a mirror image about the middle" in w for w in found), found

    def test_a_drawing_far_from_the_origin_is_judged_where_it_stands(self):
        """A board a hundred metres along Y, with a stub too small to see on one
        side: the moments are taken about the drawing, not about the origin."""
        far = 1e5
        ports = []
        for number, x in ((1, -20), (2, 20)):
            face = box((x, far - 1, 0), (x, far + 1, 1.6))
            element = Element(label=f"Port{number}", shapes=(face,), direction="+Z")
            ports.append(
                LumpedFeed(
                    label=f"Port{number}", number=number, elements=(element,), resistance=50.0
                )
            )
        study = described(
            ports,
            regions=(
                Region(
                    label="FR4", shapes=(box((-20, far - 5, 0), (20, far + 5, 1.6)),), filling=None
                ),
            ),
            conductors=(
                Conductor(label="Trace", shapes=(box((-20, far - 1, 1.6), (20, far + 1, 1.6)),)),
                Conductor(label="Ground", shapes=(box((-20, far - 5, 0), (20, far + 5, 0)),)),
                Conductor(
                    label="Speck", shapes=(box((-10, far + 2, 1.6), (-9.9, far + 2.01, 1.6)),)
                ),
            ),
        )
        assert palace_mirror_warnings(study) == []

    def test_mirrored_stubs_of_one_metal_are_accepted(self):
        stubs = (
            Conductor(
                label="Stubs",
                shapes=(box((-12, 1, 1.6), (-8, 3, 1.6)), box((8, 1, 1.6), (12, 3, 1.6))),
            ),
        )
        assert palace_mirror_warnings(board((lumped(1, -20), lumped(2, 20)), extra=stubs)) == []


class TestWhereAMirrorDoesNotApply:
    def test_a_three_port_is_told_nothing_will_be_derived(self):
        (said,) = palace_mirror_warnings(board((lumped(1, -20), lumped(2, 20), lumped(3, 0))))
        assert "3 port(s)" in said
        assert "nothing will be derived" in said


#: Boards whose drawing is a mirror, however its trace was cut into shapes.
MIRRORED = (
    "steps_one_face",
    "steps_three_faces",
    "ell_cut_differently",
    "holes_on_both_sides",
    "after_step",
    "sheet_with_a_body_mirrored_by_a_loose_one",
    "board_turned_inside_out",
    "vias_on_both_sides_of_a_wide_board",
    "ell_bodies_one_turned_inside_out",
)

#: Boards whose drawing is not, though a box round each shape is.
NOT_MIRRORED = (
    "steps_unmirrored_one_face",
    "stub_drawn_into_the_trace",
    "hole_on_one_side",
    "via_on_one_side",
    "via_on_one_side_of_a_wide_board",
    "sheet_with_a_body_on_one_side",
)


@pytest.fixture(scope="module")
def drawn(tmp_path_factory):
    """What the check said of each board the probe drew."""
    out = tmp_path_factory.mktemp("palace_symmetry")
    said = probe_manifest(PROBE, out, "PALACE_SYMMETRY_OUT")
    assert sorted(said) == sorted((*MIRRORED, *NOT_MIRRORED)), sorted(said)
    return said


@pytest.mark.slow
@pytest.mark.parametrize("board_drawn", MIRRORED)
def test_a_mirrored_board_says_nothing_however_it_was_cut(drawn, board_drawn):
    assert drawn[board_drawn] == []


@pytest.mark.slow
@pytest.mark.parametrize("board_drawn", NOT_MIRRORED)
def test_a_board_that_is_not_a_mirror_is_said_to_be_none(drawn, board_drawn):
    (said,) = drawn[board_drawn]
    assert "not a mirror image about the middle" in said
