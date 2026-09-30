# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The check that says whether openEMS reads each dielectric anywhere.

openEMS averages a cell's material at the middle of the cell and a quarter of
the way in from each of its lines, and a dielectric none of those points falls
inside is solved as though it were not drawn. CSXCAD still marks it used where a
grid line lies on its face, so nothing else says so. The cases below place a
body against the reading points of a grid whose lines are whole millimetres, so
every point a body is meant to miss or to hold is arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from Microwave.Solvers.openems import dielectrics
from Microwave.Solvers.openems.model import Material, MeshGrid, Problem, Solid
from Microwave.Solvers.openems.preflight.finding import REFUSE, WARN
from tests.openems_triangulated import ball, bar, extent

#: A grid of whole millimetres, so a cell's reading points stand at a quarter,
#: a half and three quarters of each millimetre.
LINES = np.arange(0.0, 11.0, 1.0)

FILM = Material(name="Film", kind="dielectric", epsilon=10.0)
AIR = Material(name="Air", kind="dielectric")
FERRITE = Material(name="Ferrite", kind="dielectric", mu=4.0)
LOSSY = Material(name="Lossy", kind="lossy_dielectric", kappa=0.5)
LEAKY = Material(name="Leaky", kind="lossy_dielectric", conductivity=0.5)
SHEET = Material(name="Sheet", kind="conducting_sheet", conductivity=5.8e7, thickness=0.035)
COPPER = Material(name="Copper", kind="pec")
BOTH = Material(name="Both", kind="dielectric", epsilon=4.0, mu=4.0)


class Model:
    """Only the fields the check reads off a problem."""

    def __init__(
        self, solids, materials=(FILM, AIR, FERRITE, LOSSY, LEAKY, SHEET, COPPER, BOTH), medium=""
    ):
        self.grid = MeshGrid(x=LINES, y=LINES.copy(), z=LINES.copy())
        self.materials, self.solids, self.medium = materials, solids, medium
        self.grown_by, self.pinned_clearance = 0.0, 0.0


def layer(low: float, high: float, material: str = "Film", label: str = "Layer") -> Solid:
    """A box across most of the grid, spanning ``low`` to ``high`` on z."""
    return Solid(material=material, lower=(2.0, 2.0, low), upper=(8.0, 8.0, high), label=label)


def said(*solids: Solid) -> list:
    return dielectrics.check(Model(solids))


class TestALayerIsReadWhereAPointFallsInsideIt:
    @pytest.mark.parametrize(
        "low, high",
        [
            (5.0, 5.2),  # a line on its lower face, under a quarter of the cell
            (4.8, 5.0),  # a line on its upper face
            (5.3, 5.45),  # between the quarter and the middle, no line on it
            (4.9, 5.1),  # across a line, short of the quarter on either side
            (5.0, 5.0),  # flat, on a line
            (5.6, 5.6),  # flat, off every line and every reading point
            (5.25, 5.25),  # flat, exactly on a quarter point
            (5.0, 5.25),  # reaching the quarter point and no further
            (5.25, 5.3),  # starting on it
        ],
    )
    def test_a_layer_no_reading_point_falls_inside_is_refused(self, low, high):
        found = said(layer(low, high))
        assert [finding.severity for finding in found] == [REFUSE]
        assert found[0].subjects == ("Layer",)

    @pytest.mark.parametrize(
        "low, high",
        [
            (5.0, 5.26),  # past the quarter point
            (5.45, 5.55),  # holds the middle
            (5.0, 6.0),  # a whole cell
        ],
    )
    def test_a_layer_a_reading_point_falls_inside_is_not(self, low, high):
        """A point on the boundary is not counted: the engine moves the
        structure before it builds it, and whether the point is taken then
        follows its rounding."""
        assert said(layer(low, high)) == []

    def test_the_message_states_the_thickness_the_cell_and_what_to_do(self):
        (finding,) = said(layer(5.0, 5.2))
        assert "0.2 mm across on the z axis" in finding.message
        assert "stand 0.5 mm apart" in finding.message
        assert "MinElementSize" in finding.message
        assert "thicker than 0.5 mm" in finding.message

    def test_the_thickness_it_asks_for_is_read_wherever_the_layer_stands(self):
        """The gap between the points either side of it, rounded up."""
        (finding,) = said(layer(5.76, 6.24))
        assert "thicker than 0.5 mm" in finding.message
        for low in np.arange(4.0, 6.0, 0.01):
            assert said(layer(low, low + 0.5001)) == [], low

    @pytest.mark.parametrize("thickness", [0.0, 5e-10])
    def test_a_flat_layer_is_told_it_holds_no_volume(self, thickness):
        """Flat within the grid's own tolerance."""
        (finding,) = said(layer(5.0, 5.0 + thickness))
        assert "flat on the z axis" in finding.message

    def test_the_figure_it_asks_for_is_rounded_up(self):
        assert dielectrics._up(0.12341) == "0.1235"
        assert dielectrics._up(0.5) == "0.5"


class TestWhatTheCheckAsksAbout:
    def test_vacuum_is_not_asked_about(self):
        """The run is the same with it or without it."""
        assert said(layer(5.0, 5.1, material="Air")) == []

    def test_a_body_of_the_medium_is_not_asked_about(self):
        assert dielectrics.check(Model([layer(5.0, 5.1)], medium="Film")) == []

    def test_a_gap_of_vacuum_in_a_magnetic_medium_is(self):
        """Their permittivities agree and their permeabilities do not."""
        found = dielectrics.check(Model([layer(5.0, 5.1, material="Air")], medium="Ferrite"))
        assert [finding.severity for finding in found] == [REFUSE]

    def test_a_gap_of_vacuum_in_a_dielectric_medium_is(self):
        """It is solved as the medium where the engine does not read it."""
        found = dielectrics.check(Model([layer(5.0, 5.1, material="Air")], medium="Film"))
        assert [finding.severity for finding in found] == [REFUSE]

    def test_metal_is_not_asked_about(self):
        """A conductor is sampled at the lines themselves, which is a different
        rule and a different check."""
        assert said(layer(5.0, 5.1, material="Copper")) == []

    def test_a_magnetic_body_is_asked_about(self):
        assert said(layer(5.0, 5.1, material="Ferrite"))

    def test_a_body_missing_both_ways_is_named_once(self):
        assert len(said(layer(5.0, 5.1, material="Both"))) == 1

    @pytest.mark.parametrize("material", ["Lossy", "Leaky"])
    def test_a_lossy_dielectric_is_asked_about(self, material):
        assert said(layer(5.0, 5.1, material=material))

    def test_a_conducting_sheet_is_not_asked_about(self):
        assert said(layer(5.0, 5.0, material="Sheet")) == []

    def test_a_body_is_read_where_any_piece_of_it_is(self):
        """A drawing cut into boxes arrives as numbered pieces, and a sliver of
        it beside a piece that is read is not missing."""
        assert said(layer(5.0, 5.1, label="Board#1"), layer(5.1, 6.5, label="Board#2")) == []

    def test_a_film_on_a_board_of_another_material_is_still_refused(self):
        found = said(layer(5.0, 5.1), layer(5.1, 6.5, material="Lossy", label="Board"))
        assert [finding.subjects for finding in found] == [("Layer",)]

    def test_a_film_meeting_a_body_of_its_material_at_a_corner_is_refused(self):
        """A corner shares no cell."""
        film = Solid(material="Film", lower=(0.5, 0.5, 5.0), upper=(4.0, 4.0, 5.1), label="Film")
        block = Solid(material="Film", lower=(4.0, 4.0, 5.1), upper=(6.0, 6.0, 7.0), label="Block")
        assert [finding.subjects for finding in said(film, block)] == [("Film",)]

    def test_a_film_meeting_a_body_of_its_material_along_an_edge_is_refused(self):
        film = Solid(material="Film", lower=(2.0, 0.5, 5.0), upper=(8.0, 4.0, 5.1), label="Film")
        block = Solid(material="Film", lower=(2.0, 4.0, 5.1), upper=(8.0, 6.0, 7.0), label="Block")
        assert [finding.subjects for finding in said(film, block)] == [("Film",)]

    def test_a_film_apart_from_a_body_of_its_material_is_refused(self):
        found = said(layer(5.0, 5.1), layer(7.0, 8.5, label="Block"))
        assert [finding.subjects for finding in found] == [("Layer",)]

    def test_a_region_of_several_pieces_names_each(self):
        found = said(layer(5.0, 5.1, label="Film#1"), layer(5.1, 5.2, label="Film#2"))
        assert [finding.subjects for finding in found] == [("Film#1", "Film#2")]
        assert "0.2 mm across on the z axis" in found[0].message

    def test_each_body_missing_is_named_once(self):
        found = said(layer(5.0, 5.1, label="One"), layer(3.0, 3.1, label="Two"), layer(7.0, 7.9))
        assert sorted(name for finding in found for name in finding.subjects) == ["One", "Two"]


class TestPermeabilityIsReadTheOtherWayRound:
    """A wire along x through the middles of its cells in y and z.

    Permittivity is read half a cell along an edge and a quarter across it, so
    no edge reads the wire: an edge along x needs a quarter point in y and z,
    and the wire holds only the middles. Permeability is read a quarter along
    and half across, so the edge along x reads it.
    """

    @staticmethod
    def wire(material: str) -> Solid:
        return Solid(material=material, lower=(2.0, 5.45, 5.45), upper=(8.0, 5.55, 5.55))

    def test_a_dielectric_wire_is_refused(self):
        assert [finding.severity for finding in said(self.wire("Film"))] == [REFUSE]

    def test_a_magnetic_one_is_read(self):
        assert said(self.wire("Ferrite")) == []


def hollow_box(outer: float, wall: float, centre=(5.0, 5.0, 5.0)) -> Solid:
    """A closed box with a closed box cut out of it, as one triangulation.

    The inner surface is wound inward, so the two together bound the wall.
    """
    vertices, faces = bar((outer, outer, outer), 0.0, centre)
    inside, inward = bar((outer - 2 * wall,) * 3, 0.0, centre)
    shift = len(vertices)
    vertices = list(vertices) + list(inside)
    faces = list(faces) + [(a + shift, c + shift, b + shift) for a, b, c in inward]
    lower, upper = extent(vertices)
    return Solid(
        material="Film",
        lower=lower,
        upper=upper,
        vertices=tuple(map(tuple, vertices)),
        faces=tuple(map(tuple, faces)),
        label="Shell",
    )


class TestATriangulatedBodyIsAskedPointByPoint:
    """A shell whose walls stand between the reading points on every axis.

    Its box holds reading points on every axis, so the box alone cannot refuse
    it. The walls run from 2.05 mm to 2.1 mm and their mirror, where no reading
    point falls.
    """

    def test_a_shell_whose_walls_miss_every_point_is_refused(self):
        found = said(hollow_box(5.9, 0.05))
        assert [finding.severity for finding in found] == [REFUSE]
        assert "every one of those points near this object" in found[0].message

    def test_a_solid_one_is_read(self):
        assert said(hollow_box(5.9, 2.9)) == []

    def test_a_body_the_budget_cannot_cover_is_said_to_be_unchecked(self, monkeypatch):
        """Rather than refused on points nobody asked about, or passed."""
        monkeypatch.setattr(dielectrics, "BUDGET", 24 * 8)
        found = said(hollow_box(5.9, 0.05))
        assert [finding.severity for finding in found] == [WARN]
        assert "has not been checked" in found[0].message

    def test_the_budget_is_spent_nearest_the_middle_first(self, monkeypatch):
        """A body filling its box is found on the first points asked, so a
        budget of one point's triangles is enough for it."""
        monkeypatch.setattr(dielectrics, "BUDGET", 24)
        assert said(hollow_box(5.9, 2.9)) == []


class TestAShapeIsAskedWhereItsBoxIsNotEnough:
    def test_a_round_body_is_read(self):
        """Its box holds reading points outside it as well as inside it."""
        assert said(sphere(2.9)) == []

    def test_a_budget_of_one_point_asks_the_middle(self, monkeypatch):
        """The corners of a ball's box are outside the ball."""
        body = sphere(2.9)
        monkeypatch.setattr(dielectrics, "BUDGET", len(body.faces))
        assert said(body) == []

    def test_a_budget_short_of_one_point_asks_nothing(self, monkeypatch):
        body = sphere(2.9)
        monkeypatch.setattr(dielectrics, "BUDGET", len(body.faces) - 1)
        assert [finding.severity for finding in said(body)] == [WARN]

    def test_the_budget_is_the_bodys_rather_than_each_pieces(self, monkeypatch):
        """A shell spends it and the piece of the same body beyond is not asked."""
        shell = hollow_box(5.9, 0.05)
        monkeypatch.setattr(dielectrics, "BUDGET", len(shell.faces) * 8)
        vertices, faces = bar((1.9, 1.9, 1.9), 0.0, (5.0, 5.0, 9.0))
        lower, upper = extent(vertices)
        beyond = Solid(
            material="Film",
            lower=lower,
            upper=upper,
            vertices=tuple(map(tuple, vertices)),
            faces=tuple(map(tuple, faces)),
            label="Shell",
        )
        assert said(beyond) == [], "the piece beyond is read when it is asked"
        assert [finding.severity for finding in said(shell, beyond)] == [WARN]

    def test_a_sheet_on_a_reading_plane_is_refused_as_flat(self):
        """Its outline holds reading points, and the engine would read it there
        and nowhere a little above or below: the answer would follow the grid."""
        sheet = Solid(
            material="Film",
            lower=(2.0, 2.0, 5.25),
            upper=(8.0, 8.0, 5.25),
            vertices=((2.0, 2.0, 5.25), (8.0, 2.0, 5.25), (8.0, 8.0, 5.25), (2.0, 8.0, 5.25)),
            faces=((0, 1, 2), (0, 2, 3)),
            sheet_normal=2,
            label="Sheet",
        )
        (found,) = said(sheet)
        assert found.severity == REFUSE
        assert "flat on the z axis" in found.message


def sphere(radius: float, centre=(5.0, 5.0, 5.0), label: str = "Ball") -> Solid:
    vertices, faces = ball(radius, centre)
    lower, upper = extent(vertices)
    return Solid(
        material="Film",
        lower=lower,
        upper=upper,
        vertices=tuple(map(tuple, vertices)),
        faces=tuple(map(tuple, faces)),
        label=label,
    )


class TestAPointOnTheSurfaceIsNotCounted:
    """A prism along x whose section is the triangle (5, 5), (5.5, 5), (5, 5.5)
    in y and z. The one reading point its box holds away from the box's faces
    in y and z, (5.25, 5.25), lies on its slanted face."""

    def test_a_body_read_only_on_its_surface_is_refused(self):
        section = ((5.0, 5.0), (5.5, 5.0), (5.0, 5.5))
        vertices = tuple((x, y, z) for x in (2.0, 8.0) for y, z in section)
        quads = ((0, 3, 4, 1), (0, 2, 5, 3), (1, 4, 5, 2))
        faces = ((0, 1, 2), (3, 5, 4)) + tuple(
            tri for a, b, c, d in quads for tri in ((a, b, c), (a, c, d))
        )
        prism = Solid(
            material="Film",
            lower=(2.0, 5.0, 5.0),
            upper=(8.0, 5.5, 5.5),
            vertices=vertices,
            faces=faces,
            label="Prism",
        )
        found = said(prism)
        assert [finding.severity for finding in found] == [REFUSE]
        assert "on its surface" in found[0].message


class TestTheReadingPoints:
    """Where openEMS reads a material, from ``Operator::AverageMatQuarterCell``."""

    def test_they_stand_at_the_middle_and_the_quarters_of_every_cell(self):
        lines = np.array([0.0, 1.0, 3.0])
        assert list(dielectrics._at(lines, (0.5,))) == [0.5, 2.0]
        assert list(dielectrics._at(lines, (0.25, 0.75))) == [0.25, 0.75, 1.5, 2.5]


class TestEveryRouteAsksIt:
    """Asserted through what the driver reports, which is where every route
    passes."""

    def problem(self, low: float, high: float) -> Problem:
        from tests.test_openems_adapter import MATERIALS, build_problem

        return build_problem(
            solids=(layer(low, high),),
            grid=MeshGrid(x=LINES, y=LINES.copy(), z=LINES.copy()),
            materials=(*MATERIALS, FILM),
        )

    def test_a_layer_the_grid_does_not_read_is_refused_by_the_run(self):
        from Microwave.Solvers.openems import driver, preflight

        found = driver.everything_wrong(self.problem(5.0, 5.1))
        assert any(finding.subjects == ("Layer",) for finding in preflight.refusals(found))

    def test_a_layer_it_reads_is_not(self):
        from Microwave.Solvers.openems import driver

        found = driver.everything_wrong(self.problem(5.0, 5.5))
        assert not any(finding.subjects == ("Layer",) for finding in found)
