# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A study open to free space on Palace: the air reserved round the structure, the
absorbing condition on its open sides, and what the run says about them.

The drawing is the stripline ``test_palace_lumped`` draws, with its lumped port at
each end, and what the adapter asks the CAD kernel is answered by the same box
arithmetic. Two answers are added here: the reserved box, and the area of a face
nothing covers. The skin stand-in is taught one thing a real fuse does, that a body
standing inside another is no part of the skin, which is every body inside the
reserved box.
"""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from Microwave import drawn, units
from Microwave.Gmsh.vocabulary import FRONTIER, INTERIOR, Within
from Microwave.Solvers.errors import TranslationError
from Microwave.Solvers.palace import balance, pipeline, policy
from Microwave.Solvers.palace.attributes import check, configured
from Microwave.Solvers.palace.config import ABSORBING_ORDER, Driven, Material, Sweep
from Microwave.Solvers.palace.document import OPEN_SURFACE, problem
from Microwave.Solvers.palace.problem import (
    BACKGROUND_PRIORITY,
    METAL_PRIORITY,
    OPEN,
    PLANE_PRIORITY,
    SPACE,
    Conductor,
)
from Microwave.Solvers.palace.read import FLUX_TABLE, TABLE, Scattering, scattering
from tests.test_palace_adapter import mesh_settings, with_members
from tests.test_palace_lumped import (
    BOTTOM,
    HALF,
    SEPARATION,
    SHIELD,
    TOP,
    Box,
    block,
    face,
    kernel,  # noqa: F401 - the kernel's stand-in, used by every test here
    lumped,
    lumped_run,
    material,
    meshed,
    metal_binding,
    obj,
    refused,
    stripline,
)

#: The stripline's structure, as the box round everything bound: the fill, which
#: the strip lies inside.
STRUCTURE = ((-HALF, -SHIELD / 2, 0.0), (HALF, SHIELD / 2, SEPARATION))

#: A second body standing on the stripline's top, as wide and as long.
CAP = ((-HALF, -SHIELD / 2, SEPARATION), (HALF, SHIELD / 2, SEPARATION + 1.0))

#: Every face of the domain, spelt as the policy spells them.
FACES = [f"{axis}{side}" for axis in "XYZ" for side in ("Min", "Max")]

#: A clearance, in mm, that no figure of the drawing shares.
CLEARANCE = 3.7


def _contained(inner, outer):
    return all(
        outer.lower[i] <= inner.lower[i] and inner.upper[i] <= outer.upper[i] for i in range(3)
    )


def _overlap(one, cover):
    """The area of a flat box ``one`` that ``cover`` stands on or holds."""
    (flat,) = [i for i in range(3) if one.lower[i] == one.upper[i]]
    at = one.lower[flat]
    if not cover.lower[flat] <= at <= cover.upper[flat]:
        return 0.0
    area = 1.0
    for i in range(3):
        if i != flat:
            area *= max(0.0, min(one.upper[i], cover.upper[i]) - max(one.lower[i], cover.lower[i]))
    return area


class Parts:
    """What is left of a flat box once others are cut from it, as the rectangles
    that make it up."""

    def __init__(self, rectangles):
        self.rectangles = rectangles
        self.Area = sum(_area(one) for one in rectangles)


def _area(one):
    return math.prod(high - low for low, high in zip(one.lower, one.upper) if high > low)


def _uncovered(one, covers):
    """``one`` less every flat box in ``covers`` lying in its plane, cut into the
    cells the edges of all of them make."""
    (flat,) = [i for i in range(3) if one.lower[i] == one.upper[i]]
    across = [i for i in range(3) if i != flat]
    lying = [c for c in covers if c.lower[flat] == c.upper[flat] == one.lower[flat]]
    cuts = [
        sorted(
            {one.lower[i], one.upper[i]}
            | {
                min(max(v, one.lower[i]), one.upper[i])
                for c in lying
                for v in (c.lower[i], c.upper[i])
            }
        )
        for i in across
    ]
    cells = []
    for a, b in zip(cuts[0], cuts[0][1:]):
        for c, d in zip(cuts[1], cuts[1][1:]):
            low, high = list(one.lower), list(one.upper)
            low[across[0]], high[across[0]], low[across[1]], high[across[1]] = a, b, c, d
            middle = [(lo + hi) / 2 for lo, hi in zip(low, high)]
            if not any(
                all(cover.lower[i] <= middle[i] <= cover.upper[i] for i in across)
                for cover in lying
            ):
                cells.append(face("air", low, high))
    return Parts(cells)


@pytest.fixture(autouse=True)
def reserved(kernel, monkeypatch):  # noqa: F811 - the fixture is imported to be depended on
    """The reserved box, the uncovered area and shape, and a skin that leaves out a
    body standing inside another. The covers of one face never overlap here."""
    skin = drawn.skin
    apart = drawn.apart
    made = []

    def box(lower, upper):
        body = block("reserved", tuple(lower), tuple(upper))
        made.append(body)
        return body

    def fused(bodies):
        kept = [
            body
            for body in bodies
            if not any(other is not body and _contained(body, other) for other in bodies)
        ]
        return skin(kept)

    def bare(faces, covers):
        return sum(
            math.prod(high - low for low, high in zip(one.lower, one.upper) if high > low)
            - sum(_overlap(one, cover) for cover in covers)
            for one in faces
        )

    def spread(shape, faces):
        return apart(
            shape,
            [one for f in faces for one in (f.rectangles if isinstance(f, Parts) else (f,))],
        )

    monkeypatch.setattr(drawn, "box", box)
    monkeypatch.setattr(drawn, "skin", fused)
    monkeypatch.setattr(drawn, "bare", bare)
    monkeypatch.setattr(drawn, "uncovered", _uncovered)
    monkeypatch.setattr(drawn, "apart", spread)
    return made


def _shapes(shapes):
    """Each shape as where it stands, the stand-ins comparing by identity."""
    return tuple((shape.name, shape.lower, shape.upper) for shape in shapes)


def _stated(described):
    """What a Problem says, with each shape replaced by where it stands, so two
    translations of one study compare by what they state rather than by which
    objects the kernel handed back."""
    return (
        tuple((r.label, _shapes(r.shapes), r.filling, r.material) for r in described.regions),
        tuple((label, dim, _shapes(shapes)) for label, dim, shapes in described.labelled),
        described.reserved
        and (described.reserved.faces, described.reserved.lower, described.reserved.upper),
        described.unwalled,
        tuple(str(place) for place in described.demand.places),
    )


def settings_of(line):
    return next(member for member in line.study.Group if hasattr(member, "PaddingXMin"))


def opened(faces=("ZMax",), clearance=CLEARANCE, through=(), permittivity=None, line=None):
    """The stripline with ``faces`` open to free space and ``through`` run out
    through, every other face ending on the drawing."""
    line = line or stripline()
    settings = settings_of(line)
    settings.Clearance = clearance
    for name in faces:
        setattr(settings, f"Padding{name}", "Air")
    for name in through:
        setattr(settings, f"Padding{name}", "Through")
    if permittivity is not None:
        fill = next(m for m in line.study.Group if getattr(m, "Label", "") == "FillBinding")
        fill.Material.Permittivity = permittivity
    return line


def closed(line=None):
    line = line or stripline()
    settings_of(line).Clearance = CLEARANCE
    return line


def room(least, most, sides=(), bounded_by=()):
    """A room the kernel answers, standing for the box from ``least`` to
    ``most``, which holds a point inside that box or on it."""

    def holds(point, tolerance, faces):
        at = (point.x, point.y, point.z)
        return all(
            low - tolerance <= one <= high + tolerance
            for one, low, high in zip(at, least, most, strict=True)
        )

    def apart(shape):
        gaps = [
            max(low - shape.upper[i], shape.lower[i] - high, 0.0)
            for i, (low, high) in enumerate(zip(least, most, strict=True))
        ]
        return (math.hypot(*gaps),)

    x, y, z = (high - low for low, high in zip(least, most, strict=True))
    return drawn.Room(
        solid=SimpleNamespace(
            isInside=holds, distToShape=apart, Faces=[], lower=tuple(least), upper=tuple(most)
        ),
        volume=x * y * z,
        surface=2.0 * (x * y + y * z + z * x),
        least=least,
        most=most,
        sides=tuple(sides),
        bounded_by=tuple(bounded_by),
    )


#: A room the drawing closes off inside itself, reaching no side of the box.
CAVITY = room((-1.0, -1.0, 0.5), (1.0, 1.0, 1.5), bounded_by=("FillBinding#0",))

#: A room above the strip at its middle, reaching the top of the box, that only
#: the strip bounds. No lumped element stands in it: those are at the ends.
ABOVE = room((-1.0, -1.0, SEPARATION), (1.0, 1.0, SEPARATION + 1.0), ("ZMax",), ("StripBinding#0",))


class TestTheRoomTheDrawingLeaves:
    """Every study asks which rooms its drawing leaves in the box round it. Each
    is vacuum, but the room metal seals off against a side of the box, which the
    box leaves out."""

    def test_it_asks_with_regions_and_solid_metal_held_and_sheets_and_ports_dividing(
        self, rooms_asked
    ):
        """The face a lumped element lies in is not among them: it is a magnetic
        wall where the model ends, and says nothing about what fills the room
        beyond it."""
        _, asked = rooms_asked
        described = problem(closed().study)
        ((_, _, held, sheets),) = asked
        assert all(len(group) == 1 for group in [*held.values(), *sheets.values()])
        assert {key.rsplit("#", 1)[0] for key in held} == {
            region.label for region in described.regions
        }
        assert {key.rsplit("#", 1)[0] for key in sheets} == {
            *(one.label for one in described.conductors if not one.solid),
            *(one.label for one in described.lossy),
            *(one.label for one in described.feeds),
        }
        assert described.planes
        for plane in described.planes:
            for shape in plane.shapes:
                assert not any(one is shape for group in sheets.values() for one in group)

    def test_an_open_study_asks_in_the_box_it_reserves(self, rooms_asked):
        _, asked = rooms_asked
        described = problem(opened().study)
        ((lower, upper, _, _),) = asked
        assert (tuple(lower), tuple(upper)) == (described.reserved.lower, described.reserved.upper)

    def test_room_in_a_closed_study_is_vacuum_and_opens_no_side(self, rooms_asked):
        record, _ = rooms_asked
        record.answer = [CAVITY]
        described = problem(closed().study)
        assert described.reserved.faces == ()
        assert described.reserved.shapes == ()
        assert described.opened is None
        (space,) = [region for region in described.regions if region.label == SPACE]
        assert space.filling.slowing == 1.0
        assert OPEN not in [label for label, _, _ in described.labelled]

    def test_the_drawn_regions_are_laid_at_their_own_size(self, rooms_asked):
        record, _ = rooms_asked
        record.answer = [CAVITY]
        described = problem(opened(faces=(), permittivity=4.0).study)
        slower = {region.label for region in described.regions if region.filling.slowing > 1.0}
        assert slower
        within = {place.label for place in described.demand.places if isinstance(place, Within)}
        assert within == slower

    def test_room_no_thicker_than_the_kernel_holds_reserves_nothing(self, rooms_asked):
        record, _ = rooms_asked
        record.answer = [CAVITY._replace(volume=drawn.ROOM * CAVITY.surface / 2.0)]
        assert problem(closed().study).reserved is None

    def test_room_a_dielectric_body_stands_against_thinner_than_a_slip_is_refused(
        self, rooms_asked
    ):
        """Bodies drawn to meet that miss each other leave room a thousandth of
        their size thick, and the mesh would lay elements that thin across it.
        The fill is not the medium's material, so no body takes the room."""
        record, _ = rooms_asked
        least = min(SEPARATION, SHIELD, 2 * HALF)
        slip = drawn.SLIP * least * CAVITY.surface / 2.0
        record.answer = [CAVITY._replace(volume=0.9 * slip)]
        with pytest.raises(TranslationError) as refused:
            problem(opened(faces=(), permittivity=2.2).study)
        said = str(refused.value)
        assert "from (-1, -1, 0.5) to (1, 1, 1.5) mm" in said
        assert "between 'FillBinding'" in said
        assert "Move the bodies to meet, or draw a body filling the gap" in said
        record.answer = [CAVITY._replace(volume=1.1 * slip)]
        assert problem(closed().study).reserved is not None

    def test_a_slip_a_body_of_the_medium_takes_grows_that_body_and_is_stated(
        self, rooms_asked, monkeypatch
    ):
        """The fill is vacuum, as the medium is, so the room a slip leaves beside
        it is the fill grown. The kernel's answer to which body takes it and
        what that body becomes is driven under a real FreeCAD by
        ``tests/openems_palace_rooms_probe.py``; here the translation carries
        both through."""
        record, _ = rooms_asked
        least = min(SEPARATION, SHIELD, 2 * HALF)
        record.answer = [CAVITY._replace(volume=0.9 * drawn.SLIP * least * CAVITY.surface / 2.0)]
        grown = object()
        asked = []

        def joined(room, held, kin, extents, clear):
            asked.append(sorted(kin))
            return "FillBinding#0"

        monkeypatch.setattr(drawn, "joined", joined)
        monkeypatch.setattr(drawn, "grown", lambda shape, taken: grown)
        described = problem(closed().study)
        assert asked == [["FillBinding#0"]]
        (fill,) = [region for region in described.regions if region.label == "FillBinding"]
        assert fill.shapes[0] is grown
        (said,) = policy.joining(described.joined)
        assert "from (-1, -1, 0.5) to (1, 1, 1.5) mm" in said
        assert "is solved as part of 'FillBinding'" in said
        assert described.reserved is None

    def test_a_slip_grows_its_body_when_a_lossy_sheet_rebinds_the_regions(
        self, rooms_asked, monkeypatch
    ):
        """A closed study that reserves air binds its regions again to judge a
        sheet of finite conductivity against the medium, and the body a slip
        grew stays grown."""
        record, _ = rooms_asked
        least = min(SEPARATION, SHIELD, 2 * HALF)
        record.answer = [
            CAVITY._replace(volume=0.9 * drawn.SLIP * least * CAVITY.surface / 2.0),
            CAVITY,
        ]
        grown = []

        def grow(shape, taken):
            grown.append(block("grown", shape.lower, shape.upper))
            return grown[-1]

        monkeypatch.setattr(drawn, "joined", lambda *_: "FillBinding#0")
        monkeypatch.setattr(drawn, "grown", grow)
        line = closed()
        film = material("ConductingSheet", "Film")
        film.Conductivity, film.Thickness = 5.8e7, 0.035
        sheet = obj(
            "Part::Plane", "FilmSheet", Shape=face("film", (-HALF, -1.0, 3.0), (HALF, 1.0, 3.0))
        )
        binding = obj("EMMaterialBinding", "FilmBinding", Material=film, References=[(sheet, [])])
        line.study.Group = [*line.study.Group, binding]
        described = problem(line.study)
        assert described.reserved is not None and described.lossy
        (fill,) = [region for region in described.regions if region.label == "FillBinding"]
        assert fill.shapes[0] is grown[-1]

    def test_a_slip_reaching_where_the_model_is_driven_is_refused(self, rooms_asked):
        """The fill is vacuum, as the medium is, but the room reaches the lumped
        ports' elements, where a gap is where the drive stands."""
        record, _ = rooms_asked
        everywhere = room((-1e3, -1e3, -1e3), (1e3, 1e3, 1e3), bounded_by=("FillBinding#0",))
        record.answer = [everywhere._replace(volume=1e-6 * everywhere.surface / 2.0)]
        with pytest.raises(TranslationError, match="drawn to meet that miss each other"):
            problem(closed().study)

    def test_a_slip_the_kernel_cannot_measure_is_refused_saying_so(self, rooms_asked, monkeypatch):
        record, _ = rooms_asked
        least = min(SEPARATION, SHIELD, 2 * HALF)
        record.answer = [CAVITY._replace(volume=0.9 * drawn.SLIP * least * CAVITY.surface / 2.0)]

        def joined(*_):
            raise RuntimeError("the shape is broken")

        monkeypatch.setattr(drawn, "joined", joined)
        with pytest.raises(TranslationError, match="drawn to meet that miss each other") as told:
            problem(closed().study)
        assert "The CAD kernel could not measure it: the shape is broken" in str(told.value)

    def test_a_fuse_the_kernel_answers_wrongly_is_refused_saying_so(self, rooms_asked, monkeypatch):
        record, _ = rooms_asked
        least = min(SEPARATION, SHIELD, 2 * HALF)
        record.answer = [CAVITY._replace(volume=0.9 * drawn.SLIP * least * CAVITY.surface / 2.0)]
        monkeypatch.setattr(drawn, "joined", lambda *_: "FillBinding#0")

        def grown(shape, taken):
            raise ValueError("the kernel's fuse of the body and the room is not a valid shape")

        monkeypatch.setattr(drawn, "grown", grown)
        with pytest.raises(TranslationError, match="drawn to meet that miss each other") as told:
            problem(closed().study)
        assert "not a valid shape" in str(told.value)

    @pytest.fixture
    def cut(self, monkeypatch):
        """Each list of solids the reserved box is cut by."""
        made = drawn.box
        taken = []

        def box(lower, upper):
            body = made(lower, upper)
            body.cut = lambda others: taken.append(others) or body
            return body

        monkeypatch.setattr(drawn, "box", box)
        return taken

    def test_room_metal_seals_off_against_a_side_is_left_out_and_stated(self, rooms_asked, cut):
        record, _ = rooms_asked
        record.answer = [CAVITY, ABOVE]
        described = problem(closed().study)
        assert cut == [[ABOVE.solid]]
        (sealed,) = described.reserved.sealed
        assert (sealed.least, sealed.most, sealed.bounded_by) == (
            ABOVE.least,
            ABOVE.most,
            ("StripBinding",),
        )
        (line,) = [
            one
            for one in policy.reserving(described.reserved, described.sweep)
            if "left out of the model" in one
        ]
        assert "from (-1, -1, 2) to (1, 1, 3) mm, 4 mm^3" in line
        assert "'StripBinding' and the domain's side ZMax close it off" in line

    def test_room_inside_a_shell_of_metal_is_left_out_as_well(self, rooms_asked, cut):
        """Nothing drives a room metal closes off inside the drawing either, so a
        shell of metal is the metal body it was drawn as."""
        record, _ = rooms_asked
        shell = room((-1.0, -1.0, 0.5), (1.0, 1.0, 1.5), bounded_by=("StripBinding#0",))
        record.answer = [CAVITY, shell]
        (sealed,) = problem(closed().study).reserved.sealed
        assert (sealed.least, sealed.sides) == (shell.least, ())
        assert cut == [[shell.solid]]

    @staticmethod
    def clear_of(side):
        """The closed stripline with each port driven to the fill's face across
        from ``side`` alone, so no element reaches the side a bound stands off."""
        line = stripline()
        across = [TOP] if side == "ZMin" else [BOTTOM]
        ports = [
            lumped(number, line.strip, line.fill, edge, ReferenceEntity=(line.fill, across))
            for number, edge in ((1, "Edge1"), (2, "Edge2"))
        ]
        line.study.Group = [m for m in line.study.Group if not hasattr(m, "Resistance")] + ports
        return closed(line)

    @staticmethod
    def standing_off(monkeypatch, side, by=0.05):
        """Every bound shape's bound standing ``by`` off its body on ``side``, as
        the kernel's tighter box says, and the points on it falling a hundredth of
        that inside the bound on every side, as a mesh's nodes fall inside a
        curved extreme and short of nothing else here."""
        dim, lowest = "XYZ".index(side[0]), side.endswith("Min")

        def moved(shape):
            low, high = list(shape.lower), list(shape.upper)
            if lowest:
                low[dim] += by
            else:
                high[dim] -= by
            return tuple(low), tuple(high)

        def reached(shape):
            low = [one + by / 100.0 for one in shape.lower]
            high = [one - by / 100.0 for one in shape.upper]
            return SimpleNamespace(
                XMin=low[0], YMin=low[1], ZMin=low[2], XMax=high[0], YMax=high[1], ZMax=high[2]
            )

        monkeypatch.setattr(drawn, "tightest", moved)
        monkeypatch.setattr(drawn, "reached", reached)

    @pytest.fixture
    def parted(self, monkeypatch):
        """Each room parted, with the box it was parted at; the piece beyond is
        :data:`BEYOND` and the piece inside is :data:`CAVITY`."""
        asked = []

        def part(room, lower, upper, within, held, sheets):
            asked.append((room, lower, upper, within))
            return [self.BEYOND], [CAVITY]

        monkeypatch.setattr(drawn, "parted", part)
        return asked

    #: The room between a body and a side of the box its bound stands off.
    BEYOND = room(
        (-HALF, -SHIELD / 2, -0.05), (HALF, SHIELD / 2, 0.0), ("ZMin",), ("FillBinding#0",)
    )

    #: A room reaching the bottom of the box.
    BELOW = room(
        (-HALF, -SHIELD / 2, -0.05), (HALF, SHIELD / 2, 0.5), ("ZMin",), ("FillBinding#0",)
    )

    def test_a_room_reaching_a_side_the_bound_stands_off_is_parted_there(
        self, rooms_asked, cut, parted, monkeypatch
    ):
        """The box is built on a bound that stands off a curved face. A room
        reaching that side, on a side the domain ends on, is parted where the
        drawing reaches: the piece beyond is the bound's and is left out, and the
        piece inside is the drawing's."""
        record, _ = rooms_asked
        record.answer = [CAVITY, self.BELOW]
        self.standing_off(monkeypatch, "ZMin")
        described = problem(self.clear_of("ZMin").study)
        ((which, lower, upper, within),) = parted
        assert which is self.BELOW
        assert within[1] == upper
        assert within[0][:2] == lower[:2]
        assert within[0][2] == pytest.approx(lower[2] + 0.0005, rel=1e-12, abs=0.0)
        (left,) = described.reserved.sealed
        assert (left.stands_off, left.bounded_by) == (True, ("FillBinding",))
        assert cut == [[self.BEYOND.solid]]
        (line,) = [
            one
            for one in policy.reserving(described.reserved, described.sweep)
            if "left out of the model" in one
        ]
        assert "'FillBinding' and the box, which is built on a bound standing off" in line

    def test_room_beyond_the_drawing_bounded_by_nothing_is_stated_as_the_drawing_s(
        self, rooms_asked, cut, parted, monkeypatch
    ):
        record, _ = rooms_asked
        record.answer = [CAVITY, self.BELOW]
        self.standing_off(monkeypatch, "ZMin")
        monkeypatch.setattr(self, "BEYOND", self.BEYOND._replace(bounded_by=()))
        described = problem(self.clear_of("ZMin").study)
        (line,) = [
            one
            for one in policy.reserving(described.reserved, described.sweep)
            if "left out of the model" in one
        ]
        assert "between the drawing and the box" in line

    def test_nothing_is_parted_where_no_bound_stands_off(self, rooms_asked, parted):
        record, _ = rooms_asked
        record.answer = [CAVITY, self.BELOW]
        problem(closed().study)
        assert parted == []

    def test_a_room_reaching_no_side_the_bound_stands_off_is_not_parted(
        self, rooms_asked, parted, monkeypatch
    ):
        record, _ = rooms_asked
        record.answer = [CAVITY, self.BELOW]
        self.standing_off(monkeypatch, "ZMax")
        problem(self.clear_of("ZMax").study)
        assert parted == []

    def test_a_side_open_to_free_space_is_not_parted(self, rooms_asked, parted, monkeypatch):
        """Air is reserved beyond an open side, so no room there is the bound's."""
        record, _ = rooms_asked
        record.answer = [CAVITY, ABOVE]
        self.standing_off(monkeypatch, "ZMax")
        problem(opened().study)
        assert parted == []

    @pytest.mark.parametrize(
        ("padding", "ends"), [("Ends", True), ("Air", False), ("Through", False)]
    )
    def test_only_a_side_the_domain_ends_on_is_moved(self, padding, ends):
        from Microwave.Solvers.palace import document

        settings = SimpleNamespace(**{f"Padding{face}": "Ends" for face in policy.FACES})
        settings.PaddingZMax = padding
        assert ("ZMax" in document._ends(settings)) is ends

    def test_room_sealed_off_alone_reserves_nothing(self, rooms_asked):
        """Left out, it is where the model ends, which is what a drawing that
        fills its box already is."""
        record, _ = rooms_asked
        record.answer = [ABOVE]
        assert problem(closed().study).reserved is None

    @pytest.mark.parametrize(
        "other",
        [
            ABOVE._replace(bounded_by=("StripBinding#0", "FillBinding#0")),
            room(
                (-HALF - 1.0, -SHIELD / 2, 0.0),
                (HALF + 1.0, SHIELD / 2, SEPARATION),
                ("ZMax",),
                ("StripBinding#0",),
            ),
        ],
        ids=["bounded_by_a_region", "holding_a_lumped_element"],
    )
    def test_room_is_not_sealed(self, rooms_asked, other):
        record, _ = rooms_asked
        record.answer = [CAVITY, other]
        assert problem(closed().study).reserved.sealed == ()

    def test_room_reaching_an_open_side_is_not_sealed(self, rooms_asked):
        record, _ = rooms_asked
        record.answer = [ABOVE]
        assert problem(opened(faces=("ZMax",)).study).reserved.sealed == ()

    def test_a_shape_the_kernel_cannot_cut_is_refused_by_the_study(self, monkeypatch):
        def broken(*args):
            raise RuntimeError("BRep_API: command not done")

        monkeypatch.setattr(drawn, "rooms", broken)
        with pytest.raises(TranslationError, match="could not be cut from the box"):
            problem(closed().study)


class TestWhichStudiesAreOpen:
    def test_a_study_with_no_air_face_reserves_nothing(self):
        """Closed, the region is the bound bodies and every face nobody drew on is
        the wall: nothing is reserved and nothing is labelled open."""
        described = problem(closed().study)
        assert described.reserved is None
        assert described.unwalled == ()
        assert SPACE not in [region.label for region in described.regions]
        assert OPEN not in [label for label, _, _ in described.labelled]
        assert not any(isinstance(place, Within) for place in described.demand.places)

    @pytest.mark.parametrize("face_name", FACES)
    def test_one_air_face_opens_the_study(self, face_name):
        described = problem(opened(faces=(face_name,)).study)
        assert described.reserved.faces == (face_name,)
        (space,) = [region for region in described.regions if region.label == SPACE]
        assert space.filling.slowing == 1.0
        assert (OPEN, 2) in [(label, dim) for label, dim, _ in described.labelled]

    def test_the_other_pipelines_recipe_in_the_study_changes_nothing(self):
        """What this backend solves is decided by the study's own statements. A Yee
        grid belongs to the other pipeline, and a study holding one - because the
        user compares the two backends on one drawing - is translated as if it did
        not. The first version of the split that made these recipes read this
        backend's air off that object, and answered a closed box where the study
        said free space."""
        line = opened()
        described = problem(line.study)
        grid = obj(
            "EMYeeGrid",
            "YeeGrid",
            ElementsPerWavelength=20.0,
            EdgeRefinement=6.0,
            MaxGrowthRatio=1.3,
        )
        line.study.Group = [*line.study.Group, grid]
        assert _stated(problem(line.study)) == _stated(described)


class TestTheReservedBox:
    @pytest.mark.parametrize("face_name", FACES)
    def test_it_grows_by_the_clearance_on_an_air_face_and_nowhere_else(self, face_name):
        described = problem(opened(faces=(face_name,)).study)
        lower, upper = [list(corner) for corner in STRUCTURE]
        dim = "XYZ".index(face_name[0])
        if face_name.endswith("Min"):
            lower[dim] -= CLEARANCE
        else:
            upper[dim] += CLEARANCE
        assert described.reserved.lower == pytest.approx(tuple(lower), rel=0.0, abs=1e-12)
        assert described.reserved.upper == pytest.approx(tuple(upper), rel=0.0, abs=1e-12)

    def test_it_is_flush_on_a_face_the_structure_runs_out_through(self):
        """Each port's elements lie in an end face, so a lumped port's plane spans
        each end of the reserved box and ``Through`` is honoured there."""
        described = problem(opened(faces=("ZMax",), through=("XMin", "XMax")).study)
        assert described.reserved.lower[0] == -HALF
        assert described.reserved.upper[0] == HALF

    def test_a_through_face_no_port_spans_is_refused(self):
        with pytest.raises(TranslationError, match="PaddingYMin"):
            problem(opened(faces=("ZMax",), through=("YMin",)).study)

    def test_through_beside_a_lumped_port_stays_refused_in_a_closed_study(self):
        """Only the reserved box's side is honoured by a lumped plane. A closed
        study is translated as it was."""
        line = closed()
        settings_of(line).PaddingXMin = "Through"
        with pytest.raises(TranslationError, match="PaddingXMin"):
            problem(line.study)

    def test_the_clearance_is_derived_where_it_is_nought(self):
        described = problem(opened(clearance=0.0).study)
        top = 8e9
        assert described.reserved.clearance == pytest.approx(
            0.4 * units.SPEED_OF_LIGHT / top * units.MM_PER_M, rel=1e-15, abs=0.0
        )

    def test_it_stands_round_metal_bound_outside_the_bodies(self):
        """The box is round every shape a binding hands over, sheets included."""
        line = opened()
        lid = face("lid", (-HALF, -SHIELD / 2, SEPARATION + 1.0), (HALF, SHIELD / 2, 3.0))
        line.study.Group = [*line.study.Group, metal_binding("LidMetal", lid)]
        described = problem(line.study)
        assert described.reserved.upper[2] == pytest.approx(3.0 + CLEARANCE, rel=0.0, abs=1e-12)

    def test_it_is_handed_over_below_every_drawn_region(self, tmp_path):
        """A body standing inside the box takes its own space only where the box
        gives it up, and at one priority the mesher refuses each such piece."""
        described = problem(opened().study)
        pieces, _ = pipeline.write.draw(described, tmp_path)
        priority = {piece.label: piece.priority for piece in pieces}
        assert priority[SPACE] == BACKGROUND_PRIORITY
        assert priority["FillBinding"] > BACKGROUND_PRIORITY
        assert {priority[plane.label] for plane in described.planes} == {PLANE_PRIORITY}
        elements = {e.label for feed in described.lumped for e in feed.elements}
        assert min(priority[name] for name in elements) > PLANE_PRIORITY


def sides(beside):
    """The sides a run of :class:`Beside` records names, in the order it holds
    them."""
    return tuple(one.side for one in beside)


class TestTheOpenSides:
    @pytest.mark.parametrize("face_name", FACES)
    def test_each_is_a_rectangle_from_the_boxs_own_corners(self, face_name, kernel):  # noqa: F811
        before = len(kernel)
        described = problem(opened(faces=(face_name,)).study)
        dim = "XYZ".index(face_name[0])
        side = described.reserved.lower if face_name.endswith("Min") else described.reserved.upper
        (asked,) = [
            one
            for one in kernel[before:]
            if one[2:] == (described.reserved.lower, described.reserved.upper)
            and one[:2] == (side[dim], dim)
        ]
        (rectangle,) = described.reserved.shapes
        assert rectangle.lower[dim] == rectangle.upper[dim] == side[dim]

    def test_one_ends_where_the_drawing_does_on_a_side_the_study_ends_on(self, monkeypatch):
        """The bound stands off a curved extreme at the bottom, which the study
        ends on. The room between the drawing and the bound is left out, and the
        open side beside it stops where the drawing reaches."""
        TestTheRoomTheDrawingLeaves.standing_off(monkeypatch, "ZMin", by=1.0)
        line = opened(faces=("XMax",), line=TestTheRoomTheDrawingLeaves.clear_of("ZMin"))
        described = problem(line.study)
        (rectangle,) = described.reserved.shapes
        bottom = STRUCTURE[0][2] + 0.01
        assert described.reserved.lower[2] == STRUCTURE[0][2]
        assert rectangle.lower[2] == pytest.approx(bottom, rel=0.0, abs=1e-12)
        assert rectangle.upper[2] == described.reserved.upper[2]

    def test_every_air_face_is_under_one_label(self):
        described = problem(opened(faces=("YMin", "YMax", "ZMax")).study)
        assert described.reserved.faces == ("YMin", "YMax", "ZMax")
        assert len(described.reserved.shapes) == 3
        (row,) = [row for row in described.labelled if row[0] == OPEN]
        assert row[2] == described.reserved.shapes

    def test_an_element_inside_the_air_faces_the_open_surface(self):
        """Every face open, the elements stand inside the model with the reserved
        air in front of them and behind them, as far as the clearance."""
        described = problem(opened(faces=FACES).study)
        (element, _) = described.lumped[0].elements
        assert described.planes == ()
        assert element.facing[0][1] == OPEN_SURFACE
        assert element.facing[0][2] == pytest.approx(CLEARANCE, rel=0.0, abs=1e-12)

    def test_an_element_side_along_a_wall_side_of_the_box_is_still_refused(self):
        """A side of the box the policy ends on is the wall, open study or not."""
        with pytest.raises(TranslationError, match="runs along the edge of the region"):
            problem(opened(faces=("ZMax",), line=stripline(width=SHIELD)).study)

    def test_an_element_side_along_an_open_side_is_not_the_wall(self):
        """An open side is not the wall. It stands the clearance clear of every
        bound shape, so it meets a side of an element only where the clearance
        is below the kernel's tolerance, and there nothing shorts the element."""
        from Microwave.portbox import FLATNESS

        line = opened(faces=("YMin", "YMax"), clearance=FLATNESS / 10, line=stripline(width=SHIELD))
        assert problem(line.study).lumped


def _end_face(x):
    """The fill's face at one end, as a lumped plane lying in it keeps it."""
    return ((x, -SHIELD / 2, 0.0), (x, SHIELD / 2, SEPARATION))


def _end_side(x):
    """The whole end side of the reserved box open at the top, as a lumped plane
    on a Through side keeps it."""
    return ((x, -SHIELD / 2, 0.0), (x, SHIELD / 2, SEPARATION + CLEARANCE))


class TestAPortsPlaneOnASideOfTheBox:
    """Each lumped port's elements lie in an end face of the fill. Open at the top,
    each end side of the reserved air holds that face and air above it."""

    def test_on_an_ends_side_the_plane_is_the_faces_drawn_there(self):
        described = problem(opened(faces=("ZMax",)).study)
        corners = sorted(
            (shape.lower, shape.upper) for plane in described.planes for shape in plane.shapes
        )
        assert corners == sorted([_end_face(-HALF), _end_face(HALF)])
        assert described.reserved.magnetic == ()

    def test_on_a_through_side_the_plane_is_the_whole_side(self):
        """The air above the fill is part of the plane, so the field the line
        holds in the air meets a magnetic wall there, as the rest of it does."""
        described = problem(opened(faces=("ZMax",), through=("XMin", "XMax")).study)
        corners = sorted(
            (shape.lower, shape.upper) for plane in described.planes for shape in plane.shapes
        )
        assert corners == sorted([_end_side(-HALF), _end_side(HALF)])
        assert sides(described.reserved.magnetic) == ("XMin", "XMax")

    def test_the_open_surface_is_the_air_sides_alone(self):
        """Nothing on a Through side carries the absorbing condition."""
        for through in ((), ("XMin", "XMax")):
            described = problem(opened(faces=("ZMax",), through=through).study)
            assert described.reserved.faces == ("ZMax",)
            (top,) = described.reserved.shapes
            assert top.lower[2] == top.upper[2] == SEPARATION + CLEARANCE

    def test_a_through_side_the_drawn_faces_cover_whole_is_not_named(self):
        """Open at one end alone, the other end side is the fill's end face and
        no air stands beside it."""
        described = problem(opened(faces=("XMin",), through=("XMax",)).study)
        assert described.reserved.magnetic == ()

    def test_an_element_side_meeting_the_air_on_an_ends_side_is_refused(self):
        """The strip as wide as the fill: each element's sides run along the end
        face's edges, with air beside them in the side. On an Ends side that air
        is the wall, which shorts the element."""
        line = opened(faces=("YMin", "YMax"), line=stripline(width=SHIELD))
        with pytest.raises(TranslationError, match="runs along the edge of the region"):
            problem(line.study)

    def test_and_taken_on_a_through_side_where_that_air_is_the_plane(self):
        line = opened(
            faces=("YMin", "YMax"), through=("XMin", "XMax"), line=stripline(width=SHIELD)
        )
        assert sides(problem(line.study).reserved.magnetic) == ("XMin", "XMax")

    @staticmethod
    def _undrawn(through=()):
        """The fill drawn short of the strip's near end, and each port referenced
        to a body bound to nothing: the near element lies in the XMin side of the
        box, which the strip alone reaches."""
        parts = [((-5.0, -SHIELD / 2, 0.0), (HALF, SHIELD / 2, SEPARATION))]
        line = opened(faces=("ZMax",), through=through, line=stripline(parts=parts))
        unbound = obj("Part::Box", "Unbound", Shape=block("unbound", *STRUCTURE))
        line.study.Group = [
            *(m for m in line.study.Group if not m.Label.startswith("Port")),
            lumped(1, line.strip, unbound, "Edge1"),
            lumped(2, line.strip, unbound, "Edge2"),
        ]
        return line

    def test_an_element_in_an_ends_side_where_no_body_is_drawn_is_refused(self):
        with pytest.raises(TranslationError, match="where no body bound to a dielectric"):
            problem(self._undrawn().study)

    def test_and_taken_in_a_through_side_which_is_the_plane_whole(self):
        described = problem(self._undrawn(through=("XMin", "XMax")).study)
        assert "XMin" in sides(described.reserved.magnetic)

    def test_the_run_states_the_magnetic_wall_before_it_starts(self):
        """A line per side, each with the area of that side the air holds and the
        area of the whole side."""
        described = problem(opened(faces=("ZMax",), through=("XMin", "XMax")).study)
        said = [
            line
            for line in policy.reserving(described.reserved, described.sweep)
            if "magnetic wall" in line
        ]
        assert [
            line.startswith(f"The air beside the lumped ports' faces in {side} ")
            for line, side in zip(said, ("XMin", "XMax"), strict=True)
        ] == [True, True]
        whole = SHIELD * (SEPARATION + CLEARANCE)
        assert all(f"{whole:.4g} mm^2 side" in line for line in said)
        assert all(f"{SHIELD * CLEARANCE:.4g} mm^2" in line for line in said)
        assert all(
            "reflected" in line and "no figure stated here counts it" in line for line in said
        )

    def test_and_says_nothing_of_it_on_an_ends_side(self):
        described = problem(opened(faces=("ZMax",)).study)
        lines = policy.reserving(described.reserved, described.sweep)
        assert not any("magnetic wall" in line for line in lines)

    @pytest.mark.parametrize("through", [(), ("XMin", "XMax")])
    def test_the_wall_is_written_pec_the_plane_pmc_and_the_air_absorbing(self, through):
        """Open at the top: the other sides of the box are the wall, each plane
        is a magnetic wall, and the open surface absorbs, each under its own
        attributes."""
        described = problem(opened(faces=("ZMax",), through=through).study)
        mesh = meshed(described)
        written = json.loads(configured(described, mesh, "results").to_json())["Boundaries"]
        tags = {name: one.tag for name, one in mesh.labels.items()}
        metal = [tags[conductor.label] for conductor in described.conductors]
        assert written["PEC"]["Attributes"] == sorted([tags[described.wall], *metal])
        assert written["PMC"]["Attributes"] == sorted(tags[p.label] for p in described.planes)
        assert written["Absorbing"]["Attributes"] == [tags[OPEN]]


def metal_alone(faces=FACES):
    """The stripline's strip and ports with nothing bound to the fill: metal in
    free space, open on ``faces``."""
    line = opened(faces=faces)
    line.study.Group = [m for m in line.study.Group if getattr(m, "Label", "") != "FillBinding"]
    return line


class TestMetalAlone:
    def test_is_taken_in_an_open_study_with_the_reserved_air_as_its_region(self):
        described = problem(metal_alone().study)
        assert [region.label for region in described.regions] == [SPACE]
        assert described.reserved.thinnest is None
        assert described.unwalled == ()
        line = policy.reserving(described.reserved, described.sweep)[0]
        assert "smallest extent" not in line

    def test_is_refused_in_a_closed_one(self):
        line = metal_alone()
        for name in FACES:
            setattr(settings_of(line), f"Padding{name}", "Ends")
        with pytest.raises(TranslationError, match="binds no dielectric to anything"):
            problem(line.study)

    def test_a_sheet_of_finite_conductivity_is_judged_against_the_vacuum(self):
        """A film a micrometre thick at a thousand siemens a metre lets most of a
        wave in vacuum through, and the vacuum is the one medium here."""
        line = metal_alone()
        film = material("ConductingSheet", "Film")
        film.Conductivity, film.Thickness = 1e3, 1e-3
        sheet = obj(
            "Part::Plane", "FilmSheet", Shape=face("film", (-HALF, -1.0, 3.0), (HALF, 1.0, 3.0))
        )
        binding = obj("EMMaterialBinding", "FilmBinding", Material=film, References=[(sheet, [])])
        line.study.Group = [*line.study.Group, binding]
        free = units.VACUUM_PERMEABILITY * units.SPEED_OF_LIGHT
        with pytest.raises(TranslationError, match=f"in a medium of the model of {free:.4g} ohms"):
            problem(line.study)


class TestNamesTheAdapterGives:
    @pytest.mark.parametrize("taken", [SPACE, OPEN])
    def test_an_object_called_one_is_refused_in_an_open_study(self, taken):
        line = opened()
        fill = next(m for m in line.study.Group if getattr(m, "Label", "") == "FillBinding")
        fill.Label = taken
        with pytest.raises(TranslationError, match=f"an object is called '{taken}'"):
            problem(line.study)

    def test_and_taken_in_a_closed_one(self):
        line = closed()
        fill = next(m for m in line.study.Group if getattr(m, "Label", "") == "FillBinding")
        fill.Label = SPACE
        assert problem(line.study).reserved is None


class TestAWavePort:
    def test_is_refused_in_an_open_study_by_name(self):
        study = with_members(settings=mesh_settings(PaddingZMax="Air"))
        with pytest.raises(TranslationError) as raised:
            problem(study)
        said = str(raised.value)
        assert "is a waveguide port" in said
        assert "PaddingZMax" in said
        assert "lumped ports only" in said

    def test_is_taken_in_a_closed_one(self):
        assert problem(with_members()).reserved is None


class TestTheSizePerRegion:
    def test_the_air_is_meshed_at_the_vacuum_size_and_a_slower_region_at_its_own(self):
        closed_size = problem(closed(line=opened(faces=(), permittivity=4.0)).study)
        described = problem(opened(permittivity=4.0).study)
        vacuum = policy.element_size(10.0, 8e9, 1.0)
        assert described.demand.coarsest == vacuum
        (within,) = [place for place in described.demand.places if isinstance(place, Within)]
        assert (within.label, within.size) == ("FillBinding", policy.element_size(10.0, 8e9, 4.0))
        assert closed_size.demand.coarsest == within.size

    def test_a_region_no_slower_than_vacuum_is_laid_at_the_size_everywhere(self):
        described = problem(opened().study)
        assert not any(isinstance(place, Within) for place in described.demand.places)

    def test_a_rim_is_refined_from_the_slowest_region_as_in_a_closed_study(self):
        shut = problem(closed(line=opened(faces=(), permittivity=4.0)).study)
        described = problem(opened(permittivity=4.0).study)

        def rims(demand):
            return [place.size for place in demand.places if not isinstance(place, Within)]

        assert rims(described.demand) == rims(shut.demand)
        (creases,) = [record for record in described.unlaid if hasattr(record, "rims")]
        (shut_creases,) = [record for record in shut.unlaid if hasattr(record, "rims")]
        assert creases.size == shut_creases.size


class TestAFaceOfARegionStandingInTheAir:
    """The statement is made per face. What a closed study would have walled and
    the reserved air now stands against is named, whatever else the region carries
    elsewhere."""

    def test_each_face_is_stated_with_its_own_area(self):
        """The fill is vacuum, and open at the top it stands in the reserved air
        across its whole top face and nowhere else."""
        (said,) = problem(opened().study).unwalled
        assert said.region == "FillBinding"
        (top,) = said.faces
        assert top.where == f"the face across Z at {SEPARATION:g} mm"
        assert top.side == "ZMax"
        assert top.area == pytest.approx(2 * HALF * SHIELD, rel=1e-12, abs=0.0)

    def test_is_stated_whatever_fills_it(self):
        """A stripline filled with a dielectric loses its shield planes to the air
        as a vacuum-filled one does."""
        (said,) = problem(opened(permittivity=2.2).study).unwalled
        assert said.region == "FillBinding"
        assert said.area == pytest.approx(2 * HALF * SHIELD, rel=1e-12, abs=0.0)

    def test_is_stated_on_every_face_it_stands_in_the_air_across(self):
        """Open on every face, the fill stands in the air over its whole skin,
        named face by face. The strip lies inside it and meets its end faces along
        a line alone."""
        (said,) = problem(opened(faces=FACES).study).unwalled
        whole = 2 * (2 * HALF * SHIELD + 2 * HALF * SEPARATION + SHIELD * SEPARATION)
        assert said.area == pytest.approx(whole, rel=1e-12, abs=0.0)
        assert len(said.faces) == 6
        assert {face.side for face in said.faces} == set(FACES)

    def test_a_face_metal_covers_whole_is_not_stated_and_the_rest_of_them_are(self):
        """A lid over part of the top: the top is named over what the lid leaves
        bare, which is what a closed study would have walled."""
        line = opened(faces=("YMin", "ZMax"))
        lid = face("lid", (-HALF, -1.0, SEPARATION), (HALF, 1.0, SEPARATION))
        line.study.Group = [*line.study.Group, metal_binding("LidMetal", lid)]
        (said,) = problem(line.study).unwalled
        stated = {face.side: face.area for face in said.faces}
        assert stated["ZMax"] == pytest.approx(2 * HALF * (SHIELD - 2.0), rel=1e-12, abs=0.0)
        assert stated["YMin"] == pytest.approx(2 * HALF * SEPARATION, rel=1e-12, abs=0.0)

    def test_metal_on_one_face_does_not_silence_another(self):
        """A ground over the whole of the bottom face, which lies in a wall side of
        the box: the bottom is not stated and the top still is."""
        line = opened()
        ground = face("ground", (-HALF, -SHIELD / 2, 0.0), (HALF, SHIELD / 2, 0.0))
        line.study.Group = [*line.study.Group, metal_binding("GroundMetal", ground)]
        (said,) = problem(line.study).unwalled
        assert [face.side for face in said.faces] == ["ZMax"]

    def test_the_side_named_is_the_one_the_face_lies_in_the_plane_of(self):
        """Open at XMin and at the top: the fill's top face reaches the structure's
        XMin extreme as well, and setting PaddingXMin to Ends would not wall it. The
        side named is the one the face is square to."""
        (said,) = problem(opened(faces=("XMin", "ZMax")).study).unwalled
        top = {face.where: face.side for face in said.faces}
        assert top[f"the face across Z at {SEPARATION:g} mm"] == "ZMax"

    def test_a_face_lying_in_a_side_the_policy_ends_on_is_not_stated(self):
        """Open at the top alone: the bottom face lies in the ZMin side of the box,
        where the model still ends, so nothing of it stands in the air."""
        (said,) = problem(opened().study).unwalled
        assert not any(face.where.endswith("at 0 mm") for face in said.faces)

    def test_a_face_another_region_stands_on_meets_no_air(self):
        """A second region stands on the fill's top: the fill's top face is covered
        and the second region's own top is stated."""
        cap = obj("Part::Box", "Cap", Shape=block("cap", *CAP))
        binding = obj(
            "EMMaterialBinding",
            "CapBinding",
            Material=material("Dielectric", "Cap"),
            References=[(cap, [])],
        )
        line = opened(line=stripline(extra=(binding,)))
        (said,) = problem(line.study).unwalled
        assert said.region == "CapBinding"
        assert said.area == pytest.approx(2 * HALF * SHIELD, rel=1e-12, abs=0.0)

    def test_it_is_said_before_the_run_and_not_refused(self):
        described = problem(opened(permittivity=2.2).study)
        lines = policy.reserving(described.reserved, described.sweep, described.unwalled)
        (said,) = [line for line in lines if line.startswith("'FillBinding'")]
        assert f"the face across Z at {SEPARATION:g} mm" in said
        assert "toward PaddingZMax" in said
        assert "none is a wall" in said
        assert "draw metal on it, or set the Padding it stands toward to Ends" in said


class TestTheWallTheAdapterStandsInTheAir:
    """A side the policy ends the domain on is as wide as the box, so the drawing
    covers it only where a body reaches the box's corners. The rest is reserved air
    carrying a perfect wall nobody drew, and the run names it."""

    def test_a_side_wider_than_the_drawing_is_stated_with_its_area(self):
        """Open at the top: each of the four flanking sides is as tall as the box
        and the fill reaches only the drawing's own height."""
        described = problem(opened(faces=("ZMax",)).study)
        stated = {one.side: one for one in described.reserved.walled}
        assert sorted(stated) == ["XMax", "XMin", "YMax", "YMin"]
        assert stated["XMin"].whole == pytest.approx(
            SHIELD * (SEPARATION + CLEARANCE), rel=1e-12, abs=0.0
        )
        assert stated["XMin"].area == pytest.approx(SHIELD * CLEARANCE, rel=1e-12, abs=0.0)

    def test_a_side_the_drawing_covers_whole_is_not_stated(self):
        """Open at the top alone, the bottom side of the box is the fill's own
        bottom face: the drawing holds all of it."""
        described = problem(opened(faces=("ZMax",)).study)
        assert "ZMin" not in [one.side for one in described.reserved.walled]

    def test_a_through_side_is_stated_as_the_plane_and_not_as_the_wall(self):
        described = problem(opened(faces=("ZMax",), through=("XMin", "XMax")).study)
        assert sorted(one.side for one in described.reserved.walled) == ["YMax", "YMin"]
        assert sides(described.reserved.magnetic) == ("XMin", "XMax")

    def test_the_run_states_it_with_both_ways_to_change_it(self):
        described = problem(opened(faces=("ZMax",)).study)
        said = [
            line
            for line in policy.reserving(described.reserved, described.sweep)
            if line.startswith("XMin of the reserved air")
        ]
        (line,) = said
        assert f"{SHIELD * CLEARANCE:.4g} mm^2" in line
        assert "the drawing does not cover" in line
        assert "Draw the conductor out to the edge of the air" in line
        assert "set PaddingXMin to Air" in line

    def test_a_closed_study_states_none_of_it(self):
        assert problem(closed().study).reserved is None


class TestTheBoundTheBoxIsBuiltOn:
    """The box is measured to ``Shape.BoundBox``, which is exact for a face square
    to an axis and stands off a curved surface. The run states how far, and nothing
    is built on the tighter bound."""

    def test_nothing_is_stated_where_the_bound_is_the_body_s_own(self):
        described = problem(opened().study)
        assert described.reserved.overstated is None
        assert not any(
            "stands up to" in line for line in policy.reserving(described.reserved, described.sweep)
        )

    def test_a_tighter_bound_equal_to_the_kernel_s_states_nothing(self, monkeypatch):
        """The kernel's own answer for a box is the box, and nothing stands off."""
        monkeypatch.setattr(drawn, "tightest", lambda shape: (shape.lower, shape.upper))
        assert problem(opened().study).reserved.overstated is None

    def test_the_tighter_bound_is_the_kernel_s_optimal_box(self):
        """Asked of ``optimalBoundingBox``, and nothing where the kernel has none."""
        from types import SimpleNamespace

        def box(low, high):
            return SimpleNamespace(XMin=low, YMin=low, ZMin=low, XMax=high, YMax=high, ZMax=high)

        shape = SimpleNamespace(BoundBox=box(-6.5, 6.5), optimalBoundingBox=lambda: box(-6.0, 6.0))
        shape.copy = lambda: shape
        assert drawn.tightest(shape) == ((-6.0,) * 3, (6.0,) * 3)
        plain = SimpleNamespace(BoundBox=box(0.0, 1.0))
        plain.copy = lambda: plain
        assert drawn.tightest(plain) is None

    def test_a_bound_standing_off_the_body_is_stated_with_the_distance(self, monkeypatch):
        """The kernel's answer for a curved body is wider than the body. Here the
        tighter bound is made to come back a tenth of a millimetre inside it."""

        def tighter(shape):
            return (
                tuple(low + 0.1 for low in shape.lower),
                tuple(high - 0.1 for high in shape.upper),
            )

        monkeypatch.setattr(drawn, "tightest", tighter)
        described = problem(opened().study)
        name, stands = described.reserved.overstated
        assert name in ("FillBinding", "StripMetal")
        assert stands == pytest.approx(0.1, rel=1e-9, abs=0.0)
        (line,) = [
            one
            for one in policy.reserving(described.reserved, described.sweep)
            if "stands up to" in one
        ]
        assert f"{stands:.4g} mm past the body of {name!r}" in line
        assert "an open side of the box is that much further" in line

    def test_the_box_is_built_on_the_kernel_s_bound_and_not_the_tighter_one(self, monkeypatch):
        """The tighter bound is not an enclosing one, so nothing is measured from
        it: the box stands where it stood."""

        def tighter(shape):
            return (
                tuple(low + 0.1 for low in shape.lower),
                tuple(high - 0.1 for high in shape.upper),
            )

        plain = problem(opened().study).reserved
        monkeypatch.setattr(drawn, "tightest", tighter)
        said = problem(opened().study).reserved
        assert (said.lower, said.upper) == (plain.lower, plain.upper)


class TestAClearanceThatReachesNothing:
    def test_a_closed_study_states_that_it_lays_nothing(self):
        described = problem(closed().study)
        (record,) = [one for one in described.unlaid if hasattr(one, "clearance")]
        assert record.clearance == CLEARANCE
        (said,) = [
            line for line in policy.said(described.unlaid, meshed(described)) if "Clearance" in line
        ]
        assert said.startswith(f"'MeshSettings': Clearance {CLEARANCE:.4g} mm lays nothing")
        assert "no face of this study is Air" in said
        assert "Set a face to Air to reserve that space" in said

    def test_an_open_study_states_nothing_of_it(self):
        described = problem(opened().study)
        assert not any(hasattr(one, "clearance") for one in described.unlaid)

    def test_a_clearance_of_nought_states_nothing(self):
        """Nought means derived, so nothing was stated and nothing goes unlaid."""
        line = closed()
        settings_of(line).Clearance = 0.0
        assert not any(hasattr(one, "clearance") for one in problem(line.study).unlaid)


class TestAStructureFlatAcrossAnAxis:
    """A dipole is two sheets in one plane. The air reserved round it is a solid
    box, and a box spanning nothing along an axis holds no field."""

    def test_is_refused_naming_the_axis_and_what_to_set(self):
        with pytest.raises(TranslationError) as raised:
            problem(metal_alone(faces=("XMin",)).study)
        said = str(raised.value)
        assert "lies in one plane across Z" in said
        assert "PaddingZMin is Ends" in said and "PaddingZMax is Ends" in said
        assert "Set PaddingZMin or PaddingZMax to Air" in said

    def test_is_taken_where_air_is_reserved_across_it(self):
        """The strip is flat in z, and air on both faces across z gives the box a
        thickness of two clearances."""
        described = problem(metal_alone().study)
        assert described.reserved.upper[2] - described.reserved.lower[2] == pytest.approx(
            2 * CLEARANCE, rel=1e-12, abs=0.0
        )

    @pytest.mark.parametrize("air", [("ZMin",), ("ZMax",), ("ZMin", "ZMax")])
    def test_air_on_either_face_across_the_axis_is_enough(self, air):
        """One face is what gives the box a thickness, and either one does."""
        from Microwave.Solvers.palace.document import (
            _check_the_structure_spans_every_axis,
            contents,
        )

        found = contents(closed().study)
        flat = (0.0, 0.0, 1.0)
        _check_the_structure_spans_every_axis(found, (0.0, 0.0, 1.0), (1.0, 1.0, 1.0), air)
        with pytest.raises(TranslationError, match="one plane across Z"):
            _check_the_structure_spans_every_axis(found, flat, (1.0, 1.0, 1.0), ())

    def test_the_kernel_is_never_asked_for_the_box(self, reserved):
        with pytest.raises(TranslationError, match="lies in one plane across Z"):
            problem(metal_alone(faces=("XMin",)).study)
        assert reserved == []


class TestThePolicyIsRefusedBeforeTheBoxIsReserved:
    def test_a_through_face_no_plane_can_lie_in_costs_no_box(self, reserved):
        with pytest.raises(TranslationError, match="PaddingYMin"):
            problem(opened(faces=("ZMax",), through=("YMin",)).study)
        assert reserved == []

    def test_a_curve_tolerance_costs_no_box(self, reserved):
        line = opened()
        settings_of(line).CurveTolerance = 0.1
        with pytest.raises(TranslationError, match="CurveTolerance"):
            problem(line.study)
        assert reserved == []

    def test_a_through_side_a_plane_does_lie_in_is_still_taken(self):
        """The check before the box is asked of every side an element's plane may
        lie in, so it refuses none the fused model would have honoured."""
        assert problem(opened(faces=("ZMax",), through=("XMin", "XMax")).study).reserved is not None


def laid_at(line, lower, upper, drive):
    """``line`` with a third lumped port whose element is the rectangle from
    ``lower`` to ``upper``, driven along axis ``drive`` between the ends of two
    sheets bound to nothing."""
    ends = []
    for index, at in enumerate((lower[drive], upper[drive])):
        low, high = list(lower), list(upper)
        low[drive] = high[drive] = at
        sheet = Box(f"unbound {index}", low, high)
        sheet.elements = {"Edge1": Box(f"unbound {index} edge", low, high)}
        ends.append(obj("Part::Feature", f"Unbound{index}", Shape=sheet))
    port = lumped(
        3,
        None,
        None,
        None,
        ExcitationAxis="XYZ"[drive],
        SourceEntity=(ends[0], ["Edge1"]),
        ReferenceEntity=(ends[1], ["Edge1"]),
    )
    line.study.Group = [*line.study.Group, port]
    return line


def beyond(face_name, by, reach=1.0):
    """A rectangle past ``face_name`` of the box an open study with every face
    open reserves round :data:`STRUCTURE`, from ``by`` to ``by + reach`` beyond
    that side, driven along the next axis and flat across the one after."""
    dim = "XYZ".index(face_name[0])
    drive, flat = (dim + 1) % 3, (dim + 2) % 3
    lower = [STRUCTURE[0][i] - CLEARANCE for i in range(3)]
    upper = [STRUCTURE[1][i] + CLEARANCE for i in range(3)]
    low, high = [0.0] * 3, [0.0] * 3
    if face_name.endswith("Max"):
        low[dim], high[dim] = upper[dim] + by, upper[dim] + by + reach
    else:
        low[dim], high[dim] = lower[dim] - by - reach, lower[dim] - by
    middle = (STRUCTURE[0][drive] + STRUCTURE[1][drive]) / 2
    low[drive], high[drive] = middle - 0.5, middle + 0.5
    low[flat] = high[flat] = (STRUCTURE[0][flat] + STRUCTURE[1][flat]) / 2
    return low, high, drive


class TestAnElementOutsideTheModel:
    """A lumped element is laid where the conductors picked for it face each
    other. Picked on shapes bound to nothing, it can stand where the model does
    not reach, and the mesher meshes nothing there."""

    @pytest.mark.parametrize("face_name", FACES)
    def test_one_wholly_past_an_open_side_is_refused_naming_the_side(self, face_name):
        low, high, drive = beyond(face_name, 1.0)
        said = refused(
            laid_at(opened(faces=FACES), low, high, drive),
            "'Port3' lays 'Port3 element 1' past the side",
            "Bind the conductor at each end of the element to a material",
        )
        edge = [*STRUCTURE[0], *STRUCTURE[1]][
            "XYZ".index(face_name[0]) + 3 * face_name.endswith("Max")
        ]
        side = edge + (CLEARANCE if face_name.endswith("Max") else -CLEARANCE)
        assert f"{face_name} at {face_name[0]}={side:.6g}" in said

    def test_one_past_two_sides_names_both(self):
        low, high, drive = beyond("XMax", 1.0)
        low[drive] = STRUCTURE[1][drive] + CLEARANCE + 1.0
        high[drive] = low[drive] + 1.0
        refused(
            laid_at(opened(faces=FACES), low, high, drive),
            "past the side XMax at X=",
            " and the side YMax at Y=",
        )

    def test_one_past_where_a_curved_body_ends_is_refused_there(self, monkeypatch):
        """On a side the study ends on, the model ends where the drawing reaches,
        which stands inside a bound that stands off a curved extreme."""
        TestTheRoomTheDrawingLeaves.standing_off(monkeypatch, "XMax", by=1.0)
        reach = HALF - 0.01
        low, high = (reach + 0.002, 0.0, 0.5), (HALF - 0.002, 0.0, 1.5)
        refused(laid_at(closed(), low, high, 2), f"past the side XMax at X={reach:.6g}")

    def test_one_reaching_across_an_open_side_is_refused(self):
        low, high, drive = beyond("XMax", -0.5)
        refused(laid_at(opened(faces=FACES), low, high, drive), "past the side XMax")

    def test_one_past_a_side_the_study_ends_on_is_refused(self):
        low, high, drive = beyond("XMax", 1.0 - CLEARANCE)
        refused(
            laid_at(opened(faces=("ZMax",)), low, high, drive),
            f"past the side XMax at X={HALF:.6g}",
        )

    def test_one_past_a_side_of_a_closed_study_is_refused(self):
        low, high, drive = beyond("XMax", 1.0 - CLEARANCE)
        refused(laid_at(closed(), low, high, drive), f"past the side XMax at X={HALF:.6g}")

    def test_one_lying_in_an_open_side_is_refused_naming_it(self):
        top = SEPARATION + CLEARANCE
        low, high = (-1.0, -1.0, top), (1.0, 1.0, top)
        refused(
            laid_at(opened(faces=("ZMax",)), low, high, 0),
            f"'Port3 element 1' in the side ZMax at Z={top:.6g}, which is open to free space",
        )

    def test_one_lying_in_a_side_the_study_ends_on_is_left_to_the_plane(self):
        low, high = (-1.0, -1.0, SEPARATION), (1.0, 1.0, SEPARATION)
        planes = problem(laid_at(opened(faces=("XMax",)), low, high, 0).study).planes
        assert [plane.ports for plane in planes if "Port3" in plane.ports] == [("Port3",)]

    def test_one_inside_is_judged_against_the_open_side_across_its_own_axis(self):
        """The element stands in the reserved air above the fill, flat across X
        where the open top side stands along Z."""
        top = SEPARATION + CLEARANCE
        low, high = (top, -1.0, SEPARATION + 1.0), (top, 1.0, SEPARATION + 2.0)
        problem(laid_at(opened(faces=("ZMax",)), low, high, 2).study)

    @pytest.mark.parametrize("face_name", FACES)
    def test_one_along_a_side_from_inside_is_not_refused_for_it(self, face_name):
        low, high, drive = beyond(face_name, -1.0)
        problem(laid_at(opened(faces=FACES), low, high, drive).study)


def touching_asked(monkeypatch, side=None, point=(1.0, 2.0, 3.0), kept=True):
    """Every shape :func:`Microwave.drawn.touching` is asked about, as
    ``(shape, dim, sign, at)``, with ``point`` the answer on ``side`` and
    nothing on every other. The point lies on a room the model keeps where
    ``kept`` says so, and the solids it is asked against are recorded under
    ``on``."""
    asked = []
    wanted = side and ("XYZ".index(side[0]), 1.0 if side.endswith("Max") else -1.0)

    def touching(shape, dim, sign, at):
        asked.append((shape, dim, sign, at))
        return point if (dim, sign) == wanted else None

    def on(at, solids):
        touching.on.append(list(solids))
        return kept

    touching.on = []
    monkeypatch.setattr(drawn, "touching", touching)
    monkeypatch.setattr(drawn, "on", on)
    return asked


class TestAWallTouchingACurve:
    """On a side the study ends on, the wall stands where the drawing reaches. A
    face curving away from it there meets it at a point or along a line."""

    def test_a_body_the_wall_touches_where_it_curves_is_refused_naming_the_side(self, monkeypatch):
        touching_asked(monkeypatch, "XMax")
        said = refused(
            closed(),
            f"' meets the side XMax at X={HALF:.6g} only where it curves away from it, "
            "at (1, 2, 3) mm",
            "Set XMax to Air, or end the drawing on a flat face at that side",
        )
        assert said.startswith("'FillBinding'") or said.startswith("'StripBinding'"), said

    def test_the_point_and_the_side_are_written_without_traces_of_arithmetic(self, monkeypatch):
        from Microwave.portbox import FLATNESS

        touching_asked(monkeypatch, "ZMin", point=(FLATNESS / 10, -FLATNESS / 10, 3.0))
        refused(closed(), "meets the side ZMin at Z=0 only", "at (0, 0, 3) mm")

    def test_a_touch_on_room_the_model_leaves_out_is_taken(self, monkeypatch):
        """Metal seals off the room between itself and the side, and the model
        leaves that room out, so nothing is meshed at the touch."""
        touching_asked(monkeypatch, "XMax", kept=False)
        problem(closed().study)

    def test_the_touch_is_asked_against_the_rooms_the_model_keeps(self, monkeypatch, rooms_asked):
        record, _ = rooms_asked
        record.answer = [CAVITY]
        touching_asked(monkeypatch, "XMax")
        refused(closed(), "' meets the side XMax at ")
        (solids,) = drawn.touching.on
        assert solids == [CAVITY.solid]

    def test_only_the_sides_open_to_free_space_are_not_asked(self, monkeypatch):
        """An open side stands the clearance off the drawing, so nothing touches
        it. The clearance is below a nanometre here, so a side asked
        would find the drawing at it."""
        from Microwave.portbox import FLATNESS

        asked = touching_asked(monkeypatch)
        problem(opened(faces=("YMin", "ZMax"), clearance=FLATNESS / 10).study)
        flush = {("XYZ".index(one[0]), 1.0 if one.endswith("Max") else -1.0) for one in FACES}
        flush -= {(1, -1.0), (2, 1.0)}
        assert {(dim, sign) for _, dim, sign, _ in asked} == flush

    def test_a_through_side_is_asked_as_a_side_the_study_ends_on_is(self, monkeypatch):
        """The box stands on the drawing on a ``Through`` side too. A lumped
        port's plane lies in that side, so the remedy leaves the side as it is."""
        touching_asked(monkeypatch, "XMax")
        said = refused(
            opened(faces=("ZMax",), through=("XMin", "XMax")),
            f"' meets the side XMax at X={HALF:.6g} only where it curves away from it",
            "Give the body a flat where it meets the side, or move it clear of the side",
        )
        assert "Set XMax to Air" not in said

    def test_an_open_study_with_every_side_open_asks_nothing(self, monkeypatch):
        asked = touching_asked(monkeypatch)
        problem(opened(faces=FACES).study)
        assert asked == []

    def test_only_a_shape_reaching_the_side_is_asked(self, monkeypatch):
        """The strip runs the length of the fill, and stands inside it across the
        other two axes."""
        asked = touching_asked(monkeypatch)
        problem(closed().study)
        for shape, dim, sign, at in asked:
            reach = (shape.upper if sign > 0 else shape.lower)[dim]
            assert reach == pytest.approx(at, rel=0.0, abs=1e-12), (shape.name, dim, sign)
        assert {dim for shape, dim, _, _ in asked if shape.name == "strip"} == {0}

    def test_the_wall_is_asked_about_where_the_drawing_reaches(self, monkeypatch):
        """The bound stands off a curved extreme, and the wall stands where the
        drawing's own points reach, inside it."""
        TestTheRoomTheDrawingLeaves.standing_off(monkeypatch, "ZMin", by=1.0)
        asked = touching_asked(monkeypatch)
        problem(TestTheRoomTheDrawingLeaves.clear_of("ZMin").study)
        walls = [at for _, dim, sign, at in asked if (dim, sign) == (2, -1.0)]
        assert walls
        assert walls == pytest.approx([STRUCTURE[0][2] + 0.01] * len(walls), rel=0.0, abs=1e-12)


class TestWhatAThroughFaceIsToldToDo:
    def test_an_open_study_is_told_to_lay_a_lumped_port_in_that_side(self):
        with pytest.raises(TranslationError) as raised:
            problem(opened(faces=("ZMax",), through=("YMin",)).study)
        said = str(raised.value)
        assert "lay a lumped port whose elements lie in that side of the reserved air" in said
        assert said.index("Set each of those to Air") < said.index("lay a lumped port")
        assert "waveguide port" not in said

    def test_a_closed_study_is_not_told_to_add_a_port_that_stands_there_already(self):
        """A lumped port's plane is honoured on a side of the reserved air, and a
        closed study reserves none - so the advice is a waveguide port or Air."""
        line = closed()
        settings_of(line).PaddingXMin = "Through"
        with pytest.raises(TranslationError) as raised:
            problem(line.study)
        said = str(raised.value)
        assert "stand a waveguide port across that face" in said
        assert said.index("Set each of those to Air") < said.index("stand a waveguide port")
        assert "lumped port whose" not in said


class TestTheThinnestBodyTheClearanceIsStatedAgainst:
    @staticmethod
    def _with_a_puck(monkeypatch):
        """A thin dielectric puck buried inside the fill, thinner than the fill,
        and the fill cut round it. The boxes here cannot be cut, so the two are
        said to share no volume."""
        monkeypatch.setattr(drawn, "shared_volume", lambda *_: 0.0)
        puck = obj("Part::Box", "Puck", Shape=block("puck", (-1.0, -1.0, 0.2), (1.0, 1.0, 0.7)))
        binding = obj(
            "EMMaterialBinding",
            "PuckBinding",
            Material=material("Dielectric", "Puck"),
            References=[(puck, [])],
        )
        line = opened(line=stripline(extra=(binding,)))
        return line

    def test_a_body_buried_inside_another_is_not_it(self, monkeypatch):
        """The puck stands in no air, so it sets no distance the open surface
        stands in, and the fill's own cross-section is what is stated."""
        described = problem(self._with_a_puck(monkeypatch).study)
        assert [one.region for one in described.unwalled] == ["FillBinding"]
        assert described.reserved.thinnest == ("FillBinding", SEPARATION)

    def test_the_run_states_that_body_s_extent(self, monkeypatch):
        described = problem(self._with_a_puck(monkeypatch).study)
        line = policy.reserving(described.reserved, described.sweep)[0]
        assert f"{SEPARATION:.4g} mm smallest extent of 'FillBinding'" in line


class TestASideWithinTheTolerance:
    """``Clearance`` is a length a user types, and one below the kernel's tolerance
    puts an open side of the box within the tolerance of the body it stands off."""

    def test_a_face_in_an_open_side_stands_in_the_air_whatever_the_clearance(self):
        from Microwave.portbox import FLATNESS

        near = problem(opened(faces=("ZMax",), clearance=FLATNESS).study)
        far = problem(opened(faces=("ZMax",), clearance=1.0).study)
        assert [one.region for one in near.unwalled] == [one.region for one in far.unwalled]
        assert [face.side for one in near.unwalled for face in one.faces] == ["ZMax"]

    def test_the_nearer_side_of_an_axis_decides_which_one_a_face_lies_in(self):
        from Microwave.Solvers.palace.document import _side_of
        from Microwave.Solvers.palace.problem import Reserved

        box = Reserved(
            shapes=(),
            faces=("ZMin",),
            clearance=1e-7,
            lower=(0.0, 0.0, -1e-7),
            upper=(1.0, 1.0, 0.0),
            thinnest=None,
        )
        flat = face("flat", (0.0, 0.0, 0.0), (1.0, 1.0, 0.0))
        assert _side_of(flat, box) == "ZMax"


class TestTheCapOnTheLinearSolversIterations:
    def test_an_open_run_is_given_the_open_cap_and_a_closed_run_the_closed_one(self):
        from Microwave.Solvers.palace.config import LINEAR_ITERATIONS, OPEN_LINEAR_ITERATIONS

        def cap(line):
            described = problem(line.study)
            written = json.loads(configured(described, meshed(described), "results").to_json())
            return written["Solver"]["Linear"]["MaxIts"]

        assert cap(opened()) == OPEN_LINEAR_ITERATIONS
        assert cap(closed()) == LINEAR_ITERATIONS
        assert OPEN_LINEAR_ITERATIONS > LINEAR_ITERATIONS


class TestTheAbsorbingConditionWritten:
    def test_the_open_label_carries_it_at_the_first_order(self):
        described = problem(opened().study)
        mesh = meshed(described)
        run = configured(described, mesh, "results")
        tag = mesh.labels[OPEN].tag
        assert run.absorbing == (tag,)
        written = json.loads(run.to_json())["Boundaries"]
        assert written["Absorbing"] == {"Attributes": [tag], "Order": ABSORBING_ORDER}
        assert ABSORBING_ORDER == 1

    def test_the_space_is_filled_with_vacuum(self):
        described = problem(opened(permittivity=4.0).study)
        mesh = meshed(described)
        run = configured(described, mesh, "results")
        (space,) = [m for m in run.materials if m.attributes == (mesh.labels[SPACE].tag,)]
        assert space.filling.permittivity == 1.0

    def test_a_closed_run_writes_none(self):
        described = problem(closed().study)
        written = json.loads(configured(described, meshed(described), "results").to_json())
        assert "Absorbing" not in written["Boundaries"]
        assert len(written["Boundaries"]["Postprocessing"]["SurfaceFlux"]) == 2

    def test_the_power_through_it_is_asked_two_sided_past_every_port(self):
        described = problem(opened().study)
        mesh = meshed(described)
        fluxes = json.loads(configured(described, mesh, "results").to_json())["Boundaries"][
            "Postprocessing"
        ]["SurfaceFlux"]
        (radiated,) = [one for one in fluxes if one["Attributes"] == [mesh.labels[OPEN].tag]]
        assert radiated == {
            "Index": 3,
            "Attributes": [mesh.labels[OPEN].tag],
            "Type": "Power",
            "TwoSided": True,
        }
        assert "Center" not in radiated

    def test_a_run_whose_outside_is_absorbing_alone_needs_no_metal(self):
        run = lumped_run(perfect_conductor=(), magnetic_conductor=(), absorbing=(6,))
        assert json.loads(run.to_json())["Boundaries"]["Absorbing"]["Attributes"] == [6]
        with pytest.raises(ValueError, match="no attribute carries metal"):
            lumped_run(perfect_conductor=(), magnetic_conductor=())

    def test_an_absorbing_attribute_carrying_another_condition_cannot_be_written(self):
        with pytest.raises(ValueError, match="more than one condition"):
            lumped_run(absorbing=(6,))

    def test_the_open_label_inside_the_model_is_refused(self):
        described = problem(opened().study)
        mesh = meshed(described)
        labels = dict(mesh.labels)
        labels[OPEN] = replace(labels[OPEN], sits=INTERIOR)
        with pytest.raises(TranslationError, match="open sides of the reserved air"):
            check(described, replace(mesh, labels=labels))
        assert mesh.labels[OPEN].sits == FRONTIER
        check(described, mesh)


def open_tables(directory, radiated_in, flux_driven=0.9, flux_passive=-0.05):
    """A two-port matrix driven from port 1, the power through each port as
    Palace measures a lumped port's, and the power through the open surface as
    Palace measures a two-sided flux: what goes in."""
    directory.mkdir(parents=True, exist_ok=True)
    frequencies = [2.0, 5.0, 8.0]
    head = ["f (GHz)"]
    rows = [[f] for f in frequencies]
    for out in (1, 2):
        head += [f"|S[{out}][1]| (dB)", f"arg(S[{out}][1]) (deg.)"]
        for row in rows:
            row += [-20.0 if out == 1 else -1.0, 0.0]
    (directory / TABLE).write_text(
        ",".join(head) + "\n" + "\n".join(",".join(str(v) for v in row) for row in rows) + "\n"
    )
    head = ["f (GHz)", "Φ_pow[1] (W)", "Φ_pow[2] (W)", "Φ_pow[3] (W)"]
    body = "\n".join(
        f"{f},{flux_driven},{flux_passive},{given}"
        for f, given in zip(frequencies, radiated_in, strict=True)
    )
    (directory / FLUX_TABLE).write_text(",".join(head) + "\n" + body + "\n")


class TestThePowerThroughTheOpenSurfaceRead:
    def run(self):
        return lumped_run(absorbing=(11,))

    def test_it_is_read_with_the_sign_turned(self, tmp_path):
        open_tables(tmp_path, [-0.01, -0.02, -0.03])
        answer = scattering(tmp_path, self.run())
        np.testing.assert_array_equal(answer.radiated[:, 0], [0.01, 0.02, 0.03])

    def test_a_closed_run_reads_none(self, tmp_path):
        open_tables(tmp_path, [0.0, 0.0, 0.0])
        assert scattering(tmp_path, lumped_run()).radiated is None


def open_answer(radiated, reflected=0.1, transmitted=0.9, dissipates=False, flux=None):
    """One driven port's column at three frequencies, and what left through the
    open surface at each."""
    frequency = np.array([2e9, 5e9, 8e9])
    matrix = np.zeros((3, 2, 1), dtype=complex)
    matrix[:, 0, 0] = math.sqrt(reflected)
    matrix[:, 1, 0] = math.sqrt(transmitted)
    return Scattering(
        frequency=frequency,
        out=(1, 2),
        driven=(1,),
        matrix=matrix,
        flux=np.zeros((3, 2, 1)) if flux is None else flux,
        dissipates=dissipates,
        radiated=np.asarray(radiated, dtype=float).reshape(3, 1),
    )


class TestTheBalanceOfAnOpenRun:
    def test_what_left_through_the_open_surface_is_accounted_for(self):
        """Nothing is lost: the matrix holds 0.99 of the watt and the open surface
        took the rest."""
        (shortfall,) = balance.shortfalls(open_answer([0.01, 0.01, 0.01], transmitted=0.89))
        assert shortfall.share == pytest.approx(0.0, rel=0.0, abs=1e-12)
        assert shortfall.radiated == pytest.approx(0.01, rel=1e-12, abs=0.0)

    def test_where_the_model_dissipates_the_heat_is_what_the_faces_leave(self):
        """The faces' powers sum to what the model took in, less nothing: part of
        it is heat and part left through the open surface, so the heat is the
        faces' sum less the radiated share, and the account closes."""
        flux = np.zeros((3, 2, 1))
        flux[:, 0, 0] = -(1.0 - 0.1)
        flux[:, 1, 0] = 0.8
        answer = open_answer([0.05, 0.05, 0.05], transmitted=0.8, dissipates=True, flux=flux)
        (shortfall,) = balance.shortfalls(answer)
        assert shortfall.share == pytest.approx(0.0, rel=0.0, abs=1e-12)

    def test_the_line_says_what_left_through_the_open_surface(self):
        shortfall = balance.Shortfall(1, 5e9, 2e-3, 2, radiated=0.05)
        said = balance.said(shortfall, {1: "Port1", 2: "Port2"}, lumped={1, 2})
        assert "after the 5% that left through the open surface" in said

    def test_more_accounted_than_went_in_is_sent_to_the_mesh_at_the_open_surface(self):
        shortfall = balance.Shortfall(1, 5e9, -2e-2, 1, radiated=0.3)
        said = balance.said(shortfall, {1: "Port1", 2: "Port2"}, lumped={1, 2})
        assert said.startswith(balance.WARNING)
        assert "refine the mesh at the open surface" in said

    def test_where_the_model_dissipates_the_open_surface_is_not_in_the_figure(self):
        """The heat is read off the faces less what left through the open surface,
        so the radiated share cancels: the same faces and matrix give the same
        share whatever the open surface took."""
        flux = np.zeros((3, 2, 1))
        flux[:, 0, 0] = -(1.0 - 0.1)
        flux[:, 1, 0] = 0.8
        shares = [
            balance.shortfalls(
                open_answer([left] * 3, transmitted=0.8, dissipates=True, flux=flux)
            )[0]
            for left in (0.01, 0.05)
        ]
        assert shares[0].share == pytest.approx(shares[1].share, rel=0.0, abs=1e-15)
        assert all(one.dissipates for one in shares)

    def test_where_the_model_dissipates_the_mesh_at_the_open_surface_is_not_blamed(self):
        shortfall = balance.Shortfall(2, 1e9, -2e-2, 1, radiated=0.05, dissipates=True)
        said = balance.said(shortfall, NAMES, lumped={1, 2})
        assert said.startswith(balance.WARNING)
        assert "at the open surface" not in said
        assert "does not enter this figure" in said
        assert said.endswith("Refine the mesh")

    def test_where_the_model_dissipates_the_share_is_not_said_to_follow_the_open_surface(self):
        shortfall = balance.Shortfall(1, 5e9, 2e-3, 2, radiated=0.05, dissipates=True)
        said = balance.said(shortfall, NAMES, lumped={1, 2})
        assert "after the" not in said
        assert "compares the matrix with the ports' faces alone" in said

    def test_a_closed_run_says_nothing_of_it(self):
        shortfall = balance.Shortfall(1, 5e9, 2e-3, 2)
        said = balance.said(shortfall, {1: "Port1", 2: "Port2"}, lumped={1, 2})
        assert "open surface" not in said


def first_order(frequency, distance):
    """The closed form, written here from the wave number and the distance."""
    k = 2.0 * math.pi * frequency / 299_792_458.0
    a = distance * 1e-3
    return (4.0 * (k * a) ** 4 + 1.0) ** -0.5


#: Each port's label, by its number.
NAMES = {1: "Port1", 2: "Port2"}


def far_corner():
    """The stripline open on YMin and ZMax: the angle from square at which a line
    from the structure's top face meets the far corner of the ZMax side, and what
    a first-order side reflects there, written from the drawing's own numbers."""
    across = math.hypot(2 * HALF, SHIELD + CLEARANCE)
    angle = math.atan2(across, CLEARANCE)
    return angle, (1 - math.cos(angle)) / (1 + math.cos(angle))


def two_driven(first, second):
    """Both ports of the stripline driven, each sending ``first`` and ``second``
    of its watt through the open surface at three frequencies."""
    frequency = np.array([2e9, 5e9, 8e9])
    matrix = np.zeros((3, 2, 2), dtype=complex)
    matrix[:, 0, 0] = matrix[:, 1, 1] = 0.1
    return Scattering(
        frequency=frequency,
        out=(1, 2),
        driven=(1, 2),
        matrix=matrix,
        flux=np.zeros((3, 2, 2)),
        dissipates=False,
        radiated=np.array([first, second], dtype=float).T,
    )


class TestWhatTheRunSaysOfTheOpenSurface:
    def test_the_reflection_is_the_first_order_closed_form(self):
        for frequency, distance in ((1e9, 12.0), (3e9, 40.0), (10e9, 7.5)):
            assert policy.reflection(frequency, distance) == pytest.approx(
                first_order(frequency, distance), rel=1e-12, abs=0.0
            )

    def test_it_approaches_whole_close_in_and_falls_as_the_square_far_off(self):
        assert policy.reflection(1e6, 1.0) == pytest.approx(1.0, rel=1e-9, abs=0.0)
        near, far = policy.reflection(10e9, 100.0), policy.reflection(10e9, 200.0)
        assert far / near == pytest.approx(0.25, rel=1e-3, abs=0.0)

    def test_before_the_run_it_states_where_the_surface_stands_and_what_it_reflects(self):
        described = problem(opened(faces=("YMin", "ZMax"), permittivity=2.2).study)
        line = policy.reserving(described.reserved, described.sweep, described.unwalled)[0]
        assert line.startswith("The open surface is YMin and ZMax of the reserved air, 3.7 mm")
        share = CLEARANCE / (units.SPEED_OF_LIGHT / 2e9 * 1e3)
        assert f"{share:.2g} of the free-space wavelength at 2 GHz" in line
        assert f"{CLEARANCE / SEPARATION:.3g} times the 2 mm smallest extent" in line
        # 3.7 mm is a small fraction of a wavelength at 2 GHz: the dipole mode
        # comes back all but whole, and reads as a hundred.
        assert first_order(2e9, CLEARANCE) > 0.995
        assert "reflects 100% of the dipole mode" in line
        angle, slant = far_corner()
        assert 10.0 < slant * 100 < 99.5
        assert (
            f"{slant * 100:.0f}% of a plane wave meeting a side {math.degrees(angle):.3g} " in line
        )
        assert "The dipole mode's is the larger" in line

    def test_the_slanted_wave_is_stated_where_it_reflects_more(self):
        """Far enough out the dipole mode comes back weakly, and no distance
        lowers what a flat side reflects at a slant."""
        described = problem(opened(faces=("YMin", "ZMax"), clearance=60.0).study)
        line = policy.reserving(described.reserved, described.sweep)[0]
        assert first_order(2e9, 60.0) < policy.slanted(policy.steepest(described.reserved))
        assert "The slanted wave's is the larger" in line

    def test_no_figure_is_stated_as_a_bound(self):
        described = problem(opened(faces=("YMin", "ZMax")).study)
        lines = policy.reserving(described.reserved, described.sweep, described.unwalled)
        assert not any("up to" in line for line in lines)
        said = balance.moved_said(
            balance.moved(open_answer([0.02, 0.01, 0.001]), CLEARANCE)[0], NAMES
        )
        assert "up to" not in said
        assert "an estimate and not a bound" in said

    def test_the_steepest_angle_is_to_the_far_corner_of_an_open_side(self):
        """From the near face of the structure's box to the far corner of the
        ZMax side: across the whole length, and from one edge of the structure
        to the far edge of the grown YMin side."""
        described = problem(opened(faces=("YMin", "ZMax")).study)
        angle, _ = far_corner()
        assert policy.steepest(described.reserved) == pytest.approx(angle, rel=1e-12, abs=0.0)

    def test_a_flat_side_reflects_a_slanted_wave_by_the_cosine(self):
        for degrees in (0.0, 30.0, 60.0, 89.0):
            cosine = math.cos(math.radians(degrees))
            assert policy.slanted(math.radians(degrees)) == pytest.approx(
                (1 - cosine) / (1 + cosine), rel=1e-12, abs=1e-15
            )

    def test_a_share_near_a_hundred_reads_as_one(self):
        assert policy.percent(0.9997, 2) == "100%"
        assert policy.percent(0.99995) == "100%"
        assert policy.percent(0.5834, 2) == "58%"
        assert policy.percent(0.000534) == "0.0534%"
        assert "e" not in policy.percent(1e-7)

    def test_the_clearance_is_stated_against_the_thinnest_body(self):
        """The fill drawn as two bodies, the second a millimetre wide beside the
        first: that one is the thinnest."""
        parts = [STRUCTURE, ((-HALF, SHIELD / 2, 0.0), (HALF, SHIELD / 2 + 1.0, SEPARATION))]
        described = problem(opened(line=stripline(parts=parts)).study)
        assert described.reserved.thinnest == ("FillBinding", 1.0)

    def test_after_the_run_it_estimates_how_far_the_surface_moved_the_column(self):
        """The share that left times the larger reflection, at each sample, and
        stated where it is largest."""
        answer = open_answer([0.02, 0.01, 0.001])
        slant = 0.05
        (one,) = balance.moved(answer, CLEARANCE, slant)
        estimates = [
            share * max(first_order(f, CLEARANCE), slant)
            for share, f in zip([0.02, 0.01, 0.001], [2e9, 5e9, 8e9], strict=True)
        ]
        best = int(np.argmax(estimates))
        assert one.frequency == [2e9, 5e9, 8e9][best]
        assert one.estimate == pytest.approx(estimates[best], rel=1e-12, abs=0.0)
        said = balance.moved_said(one, NAMES)
        assert f"by {estimates[best]:.2g}" in said
        assert f"{[0.02, 0.01, 0.001][best] * 100:.3g}% of the power left" in said

    def test_the_slanted_reflection_is_taken_where_it_is_the_larger(self):
        """Far out the dipole mode reflects little, and the slant decides."""
        answer = open_answer([0.01, 0.01, 0.01])
        (one,) = balance.moved(answer, 300.0, 0.3)
        assert one.reflection == 0.3
        assert one.estimate == pytest.approx(0.3 * 0.01, rel=1e-12, abs=0.0)

    def test_the_largest_estimate_is_stated_rather_than_the_largest_share(self):
        """Where the share is largest at the top of the band and the surface
        reflects most at the bottom, the product decides."""
        shares = [0.01, 0.0101, 0.0102]
        (one,) = balance.moved(open_answer(shares), CLEARANCE)
        assert int(np.argmax(shares)) == 2
        assert one.frequency == 2e9

    def test_a_term_off_the_diagonal_takes_the_root_of_both_shares(self):
        """A port that radiates little beside one that radiates much: the term
        between them is the reflection times the root of the two shares, which
        is past the reflection times the quiet port's own."""
        answer = two_driven([0.001] * 3, [0.1] * 3)
        quiet, loud = balance.moved(answer, 300.0, 0.2)
        assert (quiet.excitation, quiet.across) == (1, 2)
        assert quiet.estimate == pytest.approx(0.2 * math.sqrt(0.001 * 0.1), rel=1e-12, abs=0.0)
        assert quiet.estimate > 0.2 * 0.001
        assert loud.estimate == pytest.approx(0.2 * 0.1, rel=1e-12, abs=0.0)
        said = balance.moved_said(quiet, NAMES)
        assert "the 10% 'Port2' sent through it" in said

    def test_the_sample_is_where_the_term_off_the_diagonal_is_largest(self):
        """The quiet port's own share is flat across the band and the loud one's
        peaks in the middle: the term between them, and so the column's
        estimate, is largest there."""
        quiet, _ = balance.moved(two_driven([0.001] * 3, [0.01, 0.1, 0.01]), 300.0, 0.2)
        assert quiet.frequency == 5e9
        assert quiet.widest == 0.1

    def test_a_port_not_driven_is_said_to_go_unestimated(self):
        (one,) = balance.moved(open_answer([0.01] * 3), CLEARANCE)
        (both, _) = balance.moved(two_driven([0.01] * 3, [0.01] * 3), CLEARANCE)
        assert "not estimated" in balance.moved_said(one, NAMES)
        assert "not estimated" not in balance.moved_said(both, NAMES)

    def test_a_closed_run_states_nothing(self):
        answer = replace(open_answer([0.0, 0.0, 0.0]), radiated=None)
        assert balance.moved(answer, CLEARANCE) == ()

    def test_the_log_takes_the_slanted_reflection_where_it_is_the_larger(
        self, tmp_path, monkeypatch
    ):
        """Far out, the estimate after the run is formed from what a flat side
        reflects at the steepest angle, which the geometry alone sets."""
        described = problem(opened(faces=("YMin", "ZMax"), clearance=60.0).study)
        order = []
        monkeypatch.setattr(pipeline.run, "solve", lambda *_args, **_kwargs: "")
        monkeypatch.setattr(
            pipeline.read, "scattering", lambda *_args: open_answer([0.01, 0.01, 0.01])
        )
        prepared = pipeline.Prepared(
            problem=described, directory=tmp_path, binary=tmp_path / "palace", pieces=()
        )
        pipeline.finish(prepared, meshed(described), 1, on_output=order.append)
        across = math.hypot(2 * HALF, SHIELD + 60.0)
        cosine = math.cos(math.atan2(across, 60.0))
        slant = (1 - cosine) / (1 + cosine)
        assert first_order(2e9, 60.0) < slant
        (said,) = [line for line in order if "estimates it moved" in line]
        assert f"by {slant * 0.01:.2g}" in said

    def test_both_figures_reach_the_log_of_a_run(self, tmp_path, monkeypatch):
        """The first before the solve and the second after it, beside what the
        run already says there."""
        described = problem(opened().study)
        mesh = meshed(described)
        order = []

        def solving(*_args, **_kwargs):
            order.append("SOLVE")
            return ""

        def answered(_directory, run):
            return open_answer([0.02, 0.01, 0.001])

        monkeypatch.setattr(pipeline.run, "solve", solving)
        monkeypatch.setattr(pipeline.read, "scattering", answered)
        prepared = pipeline.Prepared(
            problem=described, directory=tmp_path, binary=tmp_path / "palace", pieces=()
        )
        pipeline.finish(prepared, mesh, 1, on_output=order.append)
        solved = order.index("SOLVE")
        assert any(line.startswith("The open surface is") for line in order[:solved])
        assert any("It is an estimate and not a bound" in line for line in order[solved:])


class TestTheResultRecordsItsOutside:
    def test_an_open_run_records_the_condition_where_it_stands(self):
        described = problem(opened(faces=("ZMax",), through=("XMin", "XMax")).study)
        (record,) = pipeline.modelled(described, meshed(described))
        assert record == {
            "boundary": "absorbing",
            "order": ABSORBING_ORDER,
            "clearance": CLEARANCE,
            "faces": ["ZMax"],
            "magnetic": ["XMin", "XMax"],
        }

    def test_a_closed_run_records_none(self):
        described = problem(closed().study)
        assert pipeline.modelled(described, meshed(described)) == ()


def test_the_driven_run_of_an_open_study_is_a_value_like_any_other():
    """What the configuration says of the open surface is readable back without
    Palace and without a mesh."""
    run = Driven(
        mesh="m.msh",
        output="results",
        materials=(Material(attributes=(1,)),),
        perfect_conductor=(),
        ports=lumped_run().ports,
        sweep=Sweep(2e9, 8e9, 3),
        order=2,
        absorbing=(9,),
    )
    assert run.radiated == 3
    assert json.loads(run.to_json())["Boundaries"]["Absorbing"]["Order"] == 1


def test_a_label_the_mesh_does_not_carry_fails_rather_than_being_dropped():
    """Every label but the wall is looked up as it is."""
    described = problem(opened().study)
    mesh = meshed(described)
    labels = {name: one for name, one in mesh.labels.items() if name != OPEN}
    with pytest.raises(KeyError):
        configured(described, replace(mesh, labels=labels), "results")


def test_a_side_meeting_the_open_surface_cannot_stand_within_the_clearance():
    """Every open rectangle stands the clearance clear of the box round every
    bound shape, so no drawn face reaches it."""
    described = problem(opened(faces=FACES).study)
    for rectangle in described.reserved.shapes:
        (flat,) = [i for i in range(3) if rectangle.lower[i] == rectangle.upper[i]]
        at = rectangle.lower[flat]
        assert min(abs(at - STRUCTURE[0][flat]), abs(at - STRUCTURE[1][flat])) == pytest.approx(
            CLEARANCE, rel=0.0, abs=1e-12
        )


def test_the_reserved_box_is_a_body_of_its_own(reserved):
    """One box is made per translation, and it is the region handed over."""
    described = problem(opened().study)
    (space,) = [region for region in described.regions if region.label == SPACE]
    assert space.shapes == (reserved[-1],)
    assert isinstance(reserved[-1], Box)


class TestEveryRegionStandsAtOnePriority:
    """No region is ordered against another: each stands below metal and above
    the reserved air, two regions over one space are refused before the mesher
    is asked, and the bodies of one binding are one region."""

    def test_regions_stand_together_below_metal(self, tmp_path, monkeypatch):
        described = problem(
            TestTheThinnestBodyTheClearanceIsStatedAgainst._with_a_puck(monkeypatch).study
        )
        post = Conductor(
            label="Post", shapes=(block("post", (-0.2, -0.2, 0.2), (0.2, 0.2, 0.7)),), solid=True
        )
        described = replace(described, conductors=(*described.conductors, post))
        pieces, _ = pipeline.write.draw(described, tmp_path)
        priority = {piece.label: piece.priority for piece in pieces}
        assert priority[SPACE] == BACKGROUND_PRIORITY
        assert priority["FillBinding"] == priority["PuckBinding"] > BACKGROUND_PRIORITY
        assert priority["Post"] == METAL_PRIORITY > priority["FillBinding"]

    def test_a_region_inside_another_is_refused_naming_the_cut(self):
        puck = obj("Part::Box", "Puck", Shape=block("puck", (-1.0, -1.0, 0.2), (1.0, 1.0, 0.7)))
        binding = obj(
            "EMMaterialBinding",
            "PuckBinding",
            Material=material("Dielectric", "Puck"),
            References=[(puck, [])],
        )
        with pytest.raises(TranslationError) as refused:
            problem(opened(line=stripline(extra=(binding,))).study)
        said = str(refused.value)
        assert "'Puck' stands inside" in said
        assert "Cut 'Puck' out of" in said

    def test_two_bodies_one_binding_covers_are_one_region_wherever_they_overlap(self):
        line = opened()
        fill = next(m for m in line.study.Group if getattr(m, "Label", "") == "FillBinding")
        body, _ = fill.References[0]
        twin = obj("Part::Box", "Twin", Shape=block("twin", (-1.0, -1.0, 0.2), (1.0, 1.0, 0.7)))
        fill.References = [(body, []), (twin, [])]
        (region,) = [one for one in problem(line.study).regions if one.label == "FillBinding"]
        assert len(region.shapes) == 2


def ptfe(**overrides):
    """A dielectric of the shipped catalog, to link as the medium."""
    chosen = material("Dielectric", "PTFE")
    chosen.Permittivity = 2.1
    for key, value in overrides.items():
        setattr(chosen, key, value)
    return chosen


def in_ptfe(line=None, **overrides):
    """The stripline open on its top face, in a medium of PTFE."""
    line = opened(line=line)
    settings_of(line).Medium = ptfe(**overrides)
    return line


class TestTheMediumFillsTheReservedAir:
    """The policy's ``Medium`` is what fills the air reserved round the
    structure, out to the open surface the absorbing condition lies on."""

    def test_the_space_is_filled_with_it_and_carries_its_name(self):
        described = problem(in_ptfe().study)
        (space,) = [region for region in described.regions if region.label == SPACE]
        assert (space.filling.permittivity, space.material) == (2.1, "PTFE")
        mesh = meshed(described)
        run = configured(described, mesh, "results")
        (written,) = [m for m in run.materials if m.attributes == (mesh.labels[SPACE].tag,)]
        assert written.filling.permittivity == 2.1

    def test_the_clearance_is_derived_from_the_wavelength_in_it(self):
        vacuum = problem(opened(clearance=0.0).study).reserved.clearance
        line = opened(clearance=0.0)
        settings_of(line).Medium = ptfe()
        filled = problem(line.study).reserved.clearance
        assert filled == pytest.approx(vacuum / math.sqrt(2.1), rel=1e-12, abs=0.0)

    def test_the_reserved_air_is_laid_at_the_size_in_it(self):
        vacuum = problem(opened().study).demand.coarsest
        filled = problem(in_ptfe().study).demand.coarsest
        assert filled == pytest.approx(vacuum / math.sqrt(2.1), rel=1e-12)

    def test_the_run_states_it_and_the_surface_in_its_wavelength(self):
        described = problem(in_ptfe(LossTangent=0.01).study)
        lines = policy.reserving(described.reserved, described.sweep)
        assert lines[0].startswith("Every space no bound body fills is 'PTFE'")
        assert "'PTFE' without its loss" in lines[1]
        bottom = described.sweep.start
        wavelength = units.SPEED_OF_LIGHT / bottom / math.sqrt(2.1) * units.MM_PER_M
        share = (
            f"{CLEARANCE / wavelength:.2g} of the wavelength in 'PTFE' at {bottom / 1e9:.4g} GHz"
        )
        assert share in lines[2]

    def test_a_region_s_bare_face_is_said_to_meet_it(self):
        described = problem(in_ptfe().study)
        lines = policy.reserving(described.reserved, described.sweep, described.unwalled)
        assert described.unwalled
        assert any("between the region and the medium 'PTFE'" in line for line in lines)

    def test_the_run_records_it_apart_from_its_outside(self):
        described = problem(in_ptfe().study)
        records = pipeline.modelled(described, meshed(described))
        assert {"medium": "PTFE"} in records
        (outside,) = [one for one in records if "boundary" in one]
        assert "medium" not in outside

    def test_the_condition_reflects_the_dipole_mode_at_the_wavenumber_in_it(self):
        """The dipole mode is reflected by what ``k a`` is, and a medium slowing
        a wave by the root of four doubles the wavenumber at one frequency."""
        assert policy.reflection(1e9, 30.0, 4.0) == policy.reflection(2e9, 30.0)

    def test_a_medium_the_drawing_leaves_no_room_is_said_to_fill_nothing(self):
        line = closed()
        settings_of(line).Medium = ptfe()
        described = problem(line.study)
        assert described.reserved is None
        lines = policy.said(described.unlaid, meshed(described))
        assert any("Medium 'PTFE' fills nothing" in one for one in lines)

    @pytest.mark.parametrize("declared", ["PEC", "ConductingSheet"])
    def test_a_metal_is_refused_naming_the_policy_and_the_material(self, declared):
        line = opened()
        settings_of(line).Medium = material(declared, "Brass")
        with pytest.raises(TranslationError, match=f"Medium links 'Brass', a {declared}"):
            problem(line.study)

    def test_after_the_run_the_estimate_reads_the_wavenumber_in_it(self, tmp_path, monkeypatch):
        """The dipole mode's reflection is taken in the medium the open surface
        stands in, which reflects less than vacuum at one distance."""
        line = opened(clearance=30.0)
        settings_of(line).Medium = ptfe()
        described = problem(line.study)
        order = []
        monkeypatch.setattr(pipeline.run, "solve", lambda *_args, **_kwargs: "")
        monkeypatch.setattr(
            pipeline.read, "scattering", lambda *_args: open_answer([0.02, 0.01, 0.001])
        )
        prepared = pipeline.Prepared(
            problem=described, directory=tmp_path, binary=tmp_path / "palace", pieces=()
        )
        pipeline.finish(prepared, meshed(described), 1, on_output=order.append)
        slant = policy.slanted(policy.steepest(described.reserved))
        (one,) = balance.moved(open_answer([0.02, 0.01, 0.001]), 30.0, slant, 2.1)
        (said,) = [line for line in order if "estimates it moved" in line]
        assert f"by {one.estimate:.2g}" in said
        (vacuum,) = balance.moved(open_answer([0.02, 0.01, 0.001]), 30.0, slant)
        assert f"{one.estimate:.2g}" != f"{vacuum.estimate:.2g}"
