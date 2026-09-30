# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A lumped port on Palace: what it becomes, what is written for it, and what is refused.

The drawing is a stripline's cross-section: a fill of vacuum, and a strip of
perfect conductor across the middle of it, each port driven from the strip's
end edge to the fill's bottom and top faces. Every test here runs where no solver
and no CAD kernel is installed. The shapes are boxes along the axes, and what
the adapter asks the kernel - a rectangle, a segment, the distance between
shapes, the area two faces share and the skin of the region's bodies - is
answered by box arithmetic in its place. For rectangles and segments along the
axes that arithmetic is the kernel's answer. The skin is not: the stand-in
drops a face two bodies share only where the two faces are the same rectangle,
and merges rectangles in one plane only where they make a rectangle.
"""

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Microwave import drawn
from Microwave.Gmsh.vocabulary import BOTH, FRONTIER, INTERIOR, Edges, Label, Mesh, Part, Trimmed
from Microwave.Gui import results as results_glue
from Microwave.Results.sparameters import IMPEDANCE_STATED, LUMPED_RESISTANCE
from Microwave.Solvers.errors import TranslationError
from Microwave.Solvers.palace import balance, pipeline
from Microwave.Solvers.palace.attributes import between_the_boundary, check, configured
from Microwave.Solvers.palace.config import (
    Driven,
    LumpedElement,
    LumpedPort,
    Material,
    Sweep,
    WavePort,
)
from Microwave.Solvers.palace.document import problem
from Microwave.Solvers.palace.problem import PLANE_PRIORITY
from Microwave.Solvers.palace.read import (
    FLUX_TABLE,
    POWER_VOLTAGE,
    RESISTANCE,
    TABLE,
    Scattering,
    scattering,
)
from tests.conftest import only_declared
from tests.test_palace_adapter import palace_solver

#: The stripline examples/stripline_50ohm.py draws, across: the strip's width,
#: the ground planes' separation and the shield's width, in mm. The line is
#: shorter, which nothing here depends on.
WIDTH, SEPARATION, SHIELD = 2.9, 2.0, 8.0
HALF = 10.0

AXES = "XYZ"


class Box:
    """A shape as the adapter reads one, standing for the box it spans.

    A face is flat across one axis and a segment across two. A body is given
    its faces by :func:`block`.
    """

    def __init__(self, name, lower, upper, solid=False):
        self.name = name
        self.lower, self.upper = tuple(lower), tuple(upper)
        extents = [b - a for a, b in zip(lower, upper)]
        self.BoundBox = SimpleNamespace(
            **{f"{axis}Min": lower[i] for i, axis in enumerate(AXES)},
            **{f"{axis}Max": upper[i] for i, axis in enumerate(AXES)},
            **{f"{axis}Length": extents[i] for i, axis in enumerate(AXES)},
            DiagonalLength=math.hypot(*extents),
        )
        centre = [(a + b) / 2 for a, b in zip(lower, upper)]
        self.CenterOfMass = SimpleNamespace(x=centre[0], y=centre[1], z=centre[2])
        flat = sum(1 for extent in extents if extent == 0.0)
        self.Solids = [self] if solid else []
        self.Faces = [] if flat >= 2 else [self]
        self.elements = {}

    def getElement(self, name):
        return self.elements[name]

    def exportBrep(self, path):
        Path(path).write_text(self.name, encoding="utf-8")

    @property
    def Volume(self):
        return math.prod(b - a for a, b in zip(self.lower, self.upper)) if self.Solids else 0.0

    def isValid(self):
        return True

    def cut(self, other, fuzzy=0.0):
        """What this box keeps outside another, measured and not drawn: one slab
        over this box's largest face, and a face where the slab is no thicker
        than ``fuzzy``."""
        left = self.Volume - self.common(other).Volume
        extents = [b - a for a, b in zip(self.lower, self.upper)]
        faces = [extents[a] * extents[b] for a, b in ((0, 1), (0, 2), (1, 2))]
        area = 2.0 * sum(faces)
        if left <= 1e-12 * self.Volume or left / max(faces) <= fuzzy:
            return SimpleNamespace(Solids=[])
        return SimpleNamespace(Solids=[SimpleNamespace(Volume=left, Area=area)])

    def common(self, other, fuzzy=0.0):
        """The box two boxes share, measured as the kernel measures a solid, and
        a face where it is no thicker than ``fuzzy`` across some axis."""
        lower = [max(a, b) for a, b in zip(self.lower, other.lower)]
        upper = [min(a, b) for a, b in zip(self.upper, other.upper)]
        extents = [max(b - a, 0.0) for a, b in zip(lower, upper)]
        x, y, z = extents
        volume = 0.0 if min(extents) <= fuzzy else x * y * z
        solids = [SimpleNamespace(Volume=volume)] if volume > 0.0 else []
        return SimpleNamespace(Volume=volume, Area=2.0 * (x * y + y * z + z * x), Solids=solids)


def face(name, lower, upper):
    return Box(name, lower, upper)


def block(name, lower, upper):
    """A body, with a face at each end along x, then y, then z."""
    body = Box(name, lower, upper, solid=True)
    faces = []
    for axis in range(3):
        for end in (lower, upper):
            low, high = list(lower), list(upper)
            low[axis] = high[axis] = end[axis]
            faces.append(face(f"{name} at {AXES[axis]} {end[axis]:g}", low, high))
    body.Faces = faces
    body.elements = {f"Face{index}": one for index, one in enumerate(faces, start=1)}
    return body


#: Which of a block's faces is which, as :func:`block` numbers them.
NEAR_END, FAR_END, LEFT_SIDE, RIGHT_SIDE, BOTTOM, TOP = (f"Face{n}" for n in range(1, 7))


def strip_sheet(width=WIDTH, start=-HALF):
    sheet = face("strip", (start, -width / 2, 1.0), (HALF, width / 2, 1.0))
    sheet.elements = {
        "Face1": sheet,
        "Edge1": Box("strip near end", (start, -width / 2, 1.0), (start, width / 2, 1.0)),
        "Edge2": Box("strip far end", (HALF, -width / 2, 1.0), (HALF, width / 2, 1.0)),
    }
    return sheet


def obj(kind, name, **properties):
    return SimpleNamespace(Proxy=type(kind, (), {})(), Label=name, Name=name, **properties)


def material(kind, name):
    return obj(
        "EMMaterial",
        name,
        MaterialType=kind,
        Permittivity=1.0,
        Permeability=1.0,
        LossTangent=0.0,
        Conductivity=0.0,
        Thickness=0.035,
        MeasuredAt=0.0,
        SourceDigest="",
    )


def lumped(number, strip, fill, edge, **overrides):
    properties = dict(
        Number=number,
        Excitation=True,
        ReferencedTo="Port impedance",
        ReferenceImpedance=50.0,
        Resistance=50.0,
        ExcitationAxis="Z",
        SourceEntity=(strip, [edge]),
        ReferenceEntity=(fill, [BOTTOM, TOP]),
    )
    properties.update(overrides)
    return obj("EMPortLumped", f"Port{number}", **properties)


def metal_binding(name, *shapes):
    sheets = [obj("Part::Plane", f"{name}{i}", Shape=shape) for i, shape in enumerate(shapes)]
    return obj(
        "EMMaterialBinding",
        name,
        Material=material("PEC", "PEC"),
        References=[(sheet, []) for sheet in sheets],
    )


def stripline(width=WIDTH, parts=None, ports=None, extra=()):
    """A study of the stripline, a port at each end, and what it holds by name.

    ``parts`` draws the fill as several bodies, the first of them the one the
    ports' references are faces of.
    """
    boxes = parts or [((-HALF, -SHIELD / 2, 0.0), (HALF, SHIELD / 2, SEPARATION))]
    bodies = [
        obj("Part::Box", "Fill" if i == 0 else f"Fill{i}", Shape=block(f"fill{i}", *corners))
        for i, corners in enumerate(boxes)
    ]
    fill = bodies[0]
    strip = obj("Part::Plane", "Strip", Shape=strip_sheet(width))
    settings = obj(
        "EMMeshPolicy",
        "MeshSettings",
        **only_declared(
            "mesh",
            "EMMeshPolicy",
            dict(
                MinElementSize=0.0,
                CurveTolerance=0.0,
                MinElementsAcross=9,
                Clearance=0.0,
                Medium=None,
                **{f"Padding{axis}{side}": "Ends" for axis in AXES for side in ("Min", "Max")},
            ),
        ),
    )
    tetrahedra = obj(
        "EMGmshMesh",
        "GmshMesh",
        **only_declared(
            "mesh",
            "EMGmshMesh",
            dict(
                ElementsPerWavelength=10.0,
                EdgeRefinement=8.0,
                ElementsPerTurn=6,
                MaxGrowthRatio=1.6,
            ),
        ),
    )
    members = [
        palace_solver(),
        settings,
        tetrahedra,
        obj(
            "EMMaterialBinding",
            "FillBinding",
            Material=material("Dielectric", "Vacuum"),
            References=[(body, []) for body in bodies],
        ),
        obj(
            "EMMaterialBinding",
            "StripBinding",
            Material=material("PEC", "PEC"),
            References=[(strip, [])],
        ),
        *(
            ports
            if ports is not None
            else (lumped(1, strip, fill, "Edge1"), lumped(2, strip, fill, "Edge2"))
        ),
        *extra,
    ]
    study = obj(
        "EMAnalysis",
        "Study",
        FrequencyStart=2e9,
        FrequencyStop=8e9,
        NumFrequencyPoints=3,
        SmallestResponse=0.0,
        Group=members,
    )
    return SimpleNamespace(study=study, fill=fill, strip=strip)


def _gap(one, other):
    return math.sqrt(
        sum(
            max(0.0, one.lower[i] - other.upper[i], other.lower[i] - one.upper[i]) ** 2
            for i in range(3)
        )
    )


def _merged(faces):
    """Rectangles in one plane that abut along a whole side, as one rectangle.

    What removing the splitters does to a skin whose faces are rectangles along
    the axes; a union of rectangles that is no rectangle is left apart.
    """
    faces = list(faces)
    merged = True
    while merged:
        merged = False
        for i, one in enumerate(faces):
            for j, other in enumerate(faces):
                if j <= i:
                    continue
                flat = [k for k in range(3) if one.lower[k] == one.upper[k]]
                if len(flat) != 1 or other.lower[flat[0]] != other.upper[flat[0]]:
                    continue
                if other.lower[flat[0]] != one.lower[flat[0]]:
                    continue
                across = [k for k in range(3) if k != flat[0]]
                for along, same in (across, across[::-1]):
                    if (one.lower[same], one.upper[same]) != (other.lower[same], other.upper[same]):
                        continue
                    if (
                        one.upper[along] == other.lower[along]
                        or other.upper[along] == one.lower[along]
                    ):
                        low = [min(a, b) for a, b in zip(one.lower, other.lower)]
                        high = [max(a, b) for a, b in zip(one.upper, other.upper)]
                        faces[i] = face(one.name, low, high)
                        del faces[j]
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break
    return faces


@pytest.fixture(autouse=True)
def kernel(monkeypatch):
    """What the adapter asks the kernel, answered off boxes."""
    asked = []

    def rectangle(position, axis, lower, upper):
        asked.append((position, axis, tuple(lower), tuple(upper)))
        low, high = list(lower), list(upper)
        low[axis] = high[axis] = position
        return Box("element", low, high)

    def segment(start, stop):
        return Box(
            "side",
            [min(a, b) for a, b in zip(start, stop)],
            [max(a, b) for a, b in zip(start, stop)],
        )

    def apart(shape, faces):
        return min((_gap(shape, one) for one in faces), default=math.inf)

    def shared_area(one, other):
        flat = [i for i in range(3) if one.lower[i] == one.upper[i]]
        if len(flat) != 1 or other.lower[flat[0]] != other.upper[flat[0]]:
            return 0.0
        if other.lower[flat[0]] != one.lower[flat[0]]:
            return 0.0
        area = 1.0
        for i in range(3):
            if i != flat[0]:
                area *= max(
                    0.0, min(one.upper[i], other.upper[i]) - max(one.lower[i], other.lower[i])
                )
        return area

    def skin(bodies):
        faces = [(index, one) for index, body in enumerate(bodies) for one in body.Faces]
        kept = [
            one
            for index, one in faces
            if not any(
                other_index != index and (other.lower, other.upper) == (one.lower, one.upper)
                for other_index, other in faces
            )
        ]
        return _merged(kept)

    monkeypatch.setattr(drawn, "rectangle", rectangle)
    monkeypatch.setattr(drawn, "segment", segment)
    monkeypatch.setattr(drawn, "apart", apart)
    monkeypatch.setattr(drawn, "shared_area", shared_area)
    monkeypatch.setattr(drawn, "skin", skin)
    return asked


def with_port(line=None, **overrides):
    line = line or stripline()
    ports = [
        lumped(1, line.strip, line.fill, "Edge1", **overrides),
        lumped(2, line.strip, line.fill, "Edge2"),
    ]
    line.study.Group = [m for m in line.study.Group if not hasattr(m, "Resistance")] + ports
    return line


def refused(line, *fragments):
    with pytest.raises(TranslationError) as raised:
        problem(line.study)
    said = str(raised.value)
    for fragment in fragments:
        assert fragment in said, said
    return said


class TestWhatALumpedPortBecomes:
    def test_one_element_for_each_face_its_reference_names(self):
        (first, second) = problem(stripline().study).lumped
        assert [element.label for element in first.elements] == [
            "Port1 element 1",
            "Port1 element 2",
        ]
        assert (first.number, second.number) == (1, 2)
        assert first.resistance == 50.0 and first.excited

    @pytest.mark.parametrize("stated", ["Z", "-Z"])
    def test_each_element_is_driven_from_the_source_to_its_reference(self, stated):
        """The axis's own sign is discarded, as the other backend discards it:
        the strip is the source, so the element to the bottom face is driven
        down and the one to the top face up, whichever way the axis was named."""
        (first, _) = problem(with_port(ExcitationAxis=stated).study).lumped
        assert [element.direction for element in first.elements] == ["-Z", "+Z"]

    def test_the_element_is_the_rectangle_where_the_source_faces_its_reference(self, kernel):
        problem(stripline().study)
        position, axis, lower, upper = kernel[0]
        assert (position, axis) == (-HALF, 0)
        assert lower[1:] == (-WIDTH / 2, 0.0) and upper[1:] == (WIDTH / 2, 1.0)

    def test_the_face_it_lies_in_is_a_mirror_the_mesher_reads_an_opening_against(self):
        """Outside its elements the plane is a magnetic wall where the model ends,
        and the room past it is its reflection."""
        described = problem(stripline().study)
        assert described.demand.mirrors == (
            "rest of the face of Port1",
            "rest of the face of Port2",
        )

    def test_the_face_it_lies_in_is_a_plane_named_after_its_port(self):
        described = problem(stripline().study)
        assert [feed.planes for feed in described.lumped] == [
            ("rest of the face of Port1",),
            ("rest of the face of Port2",),
        ]
        (near, _) = described.planes
        assert near.ports == ("Port1",)
        (merged,) = near.shapes
        assert (merged.lower, merged.upper) == (
            (-HALF, -SHIELD / 2, 0.0),
            (-HALF, SHIELD / 2, SEPARATION),
        )

    def test_the_rest_of_the_face_gives_every_piece_up_to_what_else_was_drawn(self, tmp_path):
        """What fails if a lumped port's leftover could be the wall: a plane
        drawn at the default priority ties with an element over its piece, and
        a plane not drawn at all leaves the rest of the face to the wall."""
        described = problem(stripline().study)
        files = {
            name: tuple(str(tmp_path / f"{name}-{i}") for i, _ in enumerate(shapes))
            for name, _, shapes in described.labelled
        }
        priority = {piece.label: piece.priority for piece in described.pieces(files)}
        planes = {plane.label for plane in described.planes}
        assert planes and planes <= set(priority)
        assert all(priority[label] == PLANE_PRIORITY for label in planes)
        assert all(p > PLANE_PRIORITY for name, p in priority.items() if name not in planes)

    def test_a_fill_split_across_the_element_is_one_plane(self):
        """A substrate drawn as three bodies side by side, the middle one as
        wide as the trace: each side of the element lies on the face between
        two bodies, which is inside the region and not the wall. The kernel's
        skin merges the three end faces into one, as the stand-in here does."""
        parts = [
            ((-HALF, -WIDTH / 2, 0.0), (HALF, WIDTH / 2, SEPARATION)),
            ((-HALF, -SHIELD / 2, 0.0), (HALF, -WIDTH / 2, SEPARATION)),
            ((-HALF, WIDTH / 2, 0.0), (HALF, SHIELD / 2, SEPARATION)),
        ]
        described = problem(stripline(parts=parts).study)
        (near, _) = described.planes
        assert [shape.name for shape in near.shapes] == ["fill0 at X -10"]

    def test_a_fill_in_layers_across_the_element_is_one_plane(self):
        """Layers split below the strip: the element to the bottom face lies on
        two bodies' end faces, neither of which holds all of it."""
        parts = [
            ((-HALF, -SHIELD / 2, 0.0), (HALF, SHIELD / 2, 0.5)),
            ((-HALF, -SHIELD / 2, 0.5), (HALF, SHIELD / 2, SEPARATION)),
        ]
        line = stripline(parts=parts)
        line = with_port(line, ReferenceEntity=(line.fill, [BOTTOM]))
        line.study.Group = [m for m in line.study.Group if m.Label != "Port2"]
        described = problem(line.study)
        (plane,) = described.planes
        (merged,) = plane.shapes
        assert (merged.lower, merged.upper) == (
            (-HALF, -SHIELD / 2, 0.0),
            (-HALF, SHIELD / 2, SEPARATION),
        )

    def test_two_ports_in_one_face_share_its_plane(self):
        """Two traces ending on one face would otherwise draw two labels over
        its rest at one priority, which the mesher refuses."""
        line = stripline()
        near_left = face("left", (-HALF, -3.5, 1.0), (HALF, -2.5, 1.0))
        near_left.elements = {"Edge1": Box("left end", (-HALF, -3.5, 1.0), (-HALF, -2.5, 1.0))}
        near_right = face("right", (-HALF, 2.5, 1.0), (HALF, 3.5, 1.0))
        near_right.elements = {"Edge1": Box("right end", (-HALF, 2.5, 1.0), (-HALF, 3.5, 1.0))}
        left = obj("Part::Plane", "Left", Shape=near_left)
        right = obj("Part::Plane", "Right", Shape=near_right)
        ports = [
            lumped(1, left, line.fill, "Edge1", ReferenceEntity=(line.fill, [BOTTOM])),
            lumped(2, right, line.fill, "Edge1", ReferenceEntity=(line.fill, [BOTTOM])),
        ]
        study = stripline(ports=ports).study
        study.Group = [m for m in study.Group if m.Label != "StripBinding"]
        described = problem(study)
        (plane,) = described.planes
        assert plane.label == "rest of the face of Port1 and Port2"
        assert [feed.planes for feed in described.lumped] == [(plane.label,), (plane.label,)]

    def test_every_port_is_listed_whatever_its_kind(self):
        assert [port.number for port in problem(stripline().study).ports] == [1, 2]


def beside_the_side(z_low, z_high):
    """A sheet in the end face, from the element's side outward, over part of its height."""
    return face("beside", (-HALF, WIDTH / 2, z_low), (-HALF, SHIELD / 2, z_high))


class TestWhatALumpedPortRefusesBeforeAMesh:
    """Each refusal names the port. Where its advice is a claim, the drawing the
    advice describes is translated."""

    def test_a_face_the_structure_runs_out_through(self):
        """A lumped port's element lies in a face rather than spanning one, and
        the rest of that face carries a wall of its own - so no element bounds a
        face of the domain, and a face stated ``Through`` beside one is a wall
        like any other. Only a wave port's plane makes that face the port's.
        """
        line = stripline()
        settings = next(m for m in line.study.Group if hasattr(m, "PaddingXMin"))
        settings.PaddingXMin = "Through"
        refused(line, "PaddingXMin", "runs out through the absorber")

    def test_no_resistance(self):
        refused(with_port(Resistance=0.0), "'Port1': Resistance is 0 ohm", "PEC")
        assert problem(with_port(Resistance=50.0).study).lumped[0].resistance == 50.0

    @pytest.mark.parametrize("resistance", [-50.0, math.nan, math.inf])
    def test_a_resistance_that_is_not_positive(self, resistance):
        refused(with_port(Resistance=resistance), "'Port1': Resistance is")

    def test_a_fixed_reference_other_than_the_resistance(self):
        refused(
            with_port(ReferencedTo="Fixed impedance", ReferenceImpedance=75.0),
            "'Port1': ReferencedTo is 'Fixed impedance' at ReferenceImpedance 75",
            "Set ReferenceImpedance to 50, or ReferencedTo to 'Port impedance'",
        )
        problem(with_port(ReferencedTo="Fixed impedance", ReferenceImpedance=50.0).study)
        problem(with_port(ReferencedTo="Port impedance", ReferenceImpedance=75.0).study)

    def test_an_axis_that_names_none(self):
        refused(with_port(ExcitationAxis=""), "'Port1': ExcitationAxis is ''")

    def test_an_unset_source(self):
        refused(with_port(SourceEntity=None), "'Port1': SourceEntity is unset")

    def test_a_reference_naming_no_face(self):
        line = stripline()
        refused(
            with_port(line, ReferenceEntity=(line.fill, [])), "'Port1': ReferenceEntity", "'Fill'"
        )

    def test_a_source_naming_two_picks(self):
        line = stripline()
        refused(
            with_port(line, SourceEntity=(line.strip, ["Edge1", "Edge2"])),
            "'Port1': SourceEntity names 2 sub-elements of 'Strip'",
        )

    def test_a_pick_spanning_the_axis_it_drives_across(self):
        line = stripline()
        refused(
            with_port(line, ReferenceEntity=(line.fill, [NEAR_END])),
            "'Port1': ReferenceEntity names Face1 of 'Fill', which spans 2 mm along Z",
            "Select the face or edge at the gap",
        )
        problem(with_port(line, ReferenceEntity=(line.fill, [BOTTOM])).study)

    def test_picks_that_do_not_face_each_other(self):
        line = stripline()
        line.fill.Shape.elements["Face7"] = face("corner", (8.0, 3.0, 0.0), (9.0, 4.0, 0.0))
        refused(
            with_port(line, ReferenceEntity=(line.fill, ["Face7"])),
            "'Port1' driven to Face7 of 'Fill': SourceEntity and ReferenceEntity do not overlap",
            "Pick a reference that faces the source",
        )
        problem(with_port(line, ReferenceEntity=(line.fill, [BOTTOM])).study)

    def test_a_source_edge_at_an_angle_to_the_axes(self):
        line = stripline()
        slant = Box("slant", (-HALF, -WIDTH / 2, 1.0), (-HALF + 1.0, WIDTH / 2, 1.0))
        slant.Faces = []
        line.strip.Shape.elements["Edge3"] = slant
        refused(
            with_port(line, SourceEntity=(line.strip, ["Edge3"])),
            "'Port1' is driven from an edge that runs at an angle to the axes",
            "Draw the trace's end along an axis",
        )
        problem(with_port(line, SourceEntity=(line.strip, ["Edge1"])).study)

    def test_a_source_face_spanning_both_axes_across_the_gap(self):
        line = stripline()
        refused(
            with_port(line, SourceEntity=(line.strip, ["Face1"])),
            "'Port1' is driven from a face that spans the two axes across Z",
            "Select the edge",
        )
        problem(with_port(line, SourceEntity=(line.strip, ["Edge1"])).study)

    @pytest.mark.parametrize(
        "heights",
        [(0.0, 1.0), (0.30, 0.33), (0.0, 0.4)],
        ids=["the_whole_side", "between_where_points_were_sampled", "joined_to_an_end"],
    )
    def test_a_sheet_of_metal_along_a_side_of_the_element(self, heights):
        line = stripline()
        line.study.Group = [
            *line.study.Group,
            metal_binding("BesideMetal", beside_the_side(*heights)),
        ]
        refused(
            line,
            "'Port1' lays 'Port1 element 1', which 'BesideMetal' meets along a side",
            "Keep the metal off the element's sides",
        )
        clear = face("clear", (-HALF, WIDTH / 2 + 0.5, heights[0]), (-HALF, SHIELD / 2, heights[1]))
        line.study.Group = [
            *(m for m in line.study.Group if m.Label != "BesideMetal"),
            metal_binding("ClearMetal", clear),
        ]
        problem(line.study)

    def test_a_sheet_touching_only_the_end_the_element_drives_to_is_taken(self):
        """The ground under the element meets its side at a corner, as the end
        metal does, and not along it."""
        line = stripline()
        ground = face("ground", (-HALF, -SHIELD / 2, 0.0), (HALF, SHIELD / 2, 0.0))
        line.study.Group = [*line.study.Group, metal_binding("GroundMetal", ground)]
        problem(line.study)

    def test_the_edge_of_the_region_along_a_side_of_the_element(self):
        """A strip as wide as the shield puts each side of its elements on a
        side wall, which is the wall."""
        refused(
            stripline(width=SHIELD),
            "'Port1' lays 'Port1 element 1', whose side runs along the edge of the region",
            "Draw the element clear of the edge of the region",
        )
        problem(stripline(width=SHIELD - 1.0).study)

    def test_an_element_partly_on_the_boundary(self):
        """The fill above half the gap runs on past the end, so the element
        stands on the boundary below and inside the region above."""
        parts = [
            ((-HALF, -SHIELD / 2, 0.0), (HALF, SHIELD / 2, 0.5)),
            ((-HALF - 2.0, -SHIELD / 2, 0.5), (HALF, SHIELD / 2, SEPARATION)),
        ]
        line = stripline(parts=parts)
        line = with_port(line, ReferenceEntity=(line.fill, [BOTTOM]))
        line.study.Group = [m for m in line.study.Group if m.Label != "Port2"]
        refused(line, "'Port1' lays 'Port1 element 1', which lies partly on the boundary")
        parts[1] = ((-HALF, -SHIELD / 2, 0.5), (HALF, SHIELD / 2, SEPARATION))
        line = stripline(parts=parts)
        line = with_port(line, ReferenceEntity=(line.fill, [BOTTOM]))
        line.study.Group = [m for m in line.study.Group if m.Label != "Port2"]
        problem(line.study)

    def test_an_element_a_face_covers_a_band_of_is_partly_on_the_boundary(self):
        """The fill ends at the element's plane only across a band thirty
        micrometres wide inside its left side, and runs on past it everywhere
        else. The threshold a face has to cover is the area of a band one
        tolerance wide round the element's outline - what a face meeting the
        element along its edge covers by rounding - and thirty tolerances is a
        face covering part of the element, not one meeting its edge."""
        band = 3e-5
        parts = [
            ((-HALF, -SHIELD / 2, 0.0), (HALF, -WIDTH / 2 + band, SEPARATION)),
            ((-HALF - 2.0, -WIDTH / 2 + band, 0.0), (HALF, SHIELD / 2, SEPARATION)),
        ]
        line = stripline(parts=parts)
        ground = face("ground", (-HALF - 2.0, -SHIELD / 2, 0.0), (HALF, SHIELD / 2, 0.0))
        ground.elements = {"Face1": ground}
        sheet = obj("Part::Plane", "Ground", Shape=ground)
        port = lumped(1, line.strip, sheet, "Edge1", ReferenceEntity=(sheet, ["Face1"]))
        line.study.Group = [
            *(m for m in line.study.Group if not hasattr(m, "Resistance")),
            port,
            obj(
                "EMMaterialBinding",
                "GroundMetal",
                Material=material("PEC", "PEC"),
                References=[(sheet, [])],
            ),
        ]
        refused(line, "'Port1' lays 'Port1 element 1', which lies partly on the boundary")

    def test_a_coplanar_face_touching_the_element_is_no_part_of_its_plane(self):
        """A body standing past the end face above the strip puts the element
        inside the region. The fill's end face below it touches the element at
        its end and covers none of it, and stays the wall. The fill is drawn in
        two layers so that the stand-in skin, which drops only faces two bodies
        share whole, finds the region's end where the kernel does."""
        parts = [
            ((-HALF, -SHIELD / 2, 1.0), (HALF, SHIELD / 2, SEPARATION)),
            ((-HALF, -SHIELD / 2, 0.0), (HALF, SHIELD / 2, 1.0)),
            ((-HALF - 2.0, -SHIELD / 2, 1.0), (-HALF, SHIELD / 2, SEPARATION)),
        ]
        line = stripline(parts=parts)
        line = with_port(line, ReferenceEntity=(line.fill, [TOP]))
        line.study.Group = [m for m in line.study.Group if m.Label != "Port2"]
        described = problem(line.study)
        assert described.planes == ()
        assert described.lumped[0].planes == ()

    def test_a_dielectric_bound_to_what_encloses_no_volume(self):
        line = stripline()
        shell = obj("Part::Feature", "Shell", Shape=face("shell", (0.0, 0.0, 0.0), (1.0, 1.0, 0.0)))
        (fill_binding,) = [m for m in line.study.Group if m.Label == "FillBinding"]
        fill_binding.References = [*fill_binding.References, (shell, [])]
        refused(
            line, "'FillBinding'", "'Shell', which encloses no volume", "Bind it to a closed body"
        )

    def test_surfaces_the_kernel_cannot_make_a_solid_of(self, monkeypatch):
        def failing(shape):
            raise drawn.Unmeasured("BRep_API: command not done")

        monkeypatch.setattr(drawn, "filling", failing)
        refused(
            stripline(), "'FillBinding'", "could not make a solid", "BRep_API: command not done"
        )

    def test_a_fuse_the_kernel_cannot_make(self, monkeypatch):
        def failing(bodies):
            raise ValueError("Null input shape")

        monkeypatch.setattr(drawn, "skin", failing)
        refused(stripline(), "could not be fused", "Null input shape", "closed bodies")

    def test_a_lumped_port_beside_a_wave_port(self):
        line = stripline()
        wave = obj(
            "EMPortRectWaveguide",
            "Guide",
            Number=3,
            Excitation=True,
            ReferencedTo="Port impedance",
            ReferenceImpedance=50.0,
            Mode="TE10",
            ReferenceDepth=0.0,
            PropagationAxis="X",
            CrossSection=(line.fill, [NEAR_END]),
        )
        line.study.Group = [*line.study.Group, wave]
        refused(line, "'Guide' is a waveguide port and 'Port1' a lumped port")


def meshed(
    described,
    plane_sits=FRONTIER,
    element_sits=FRONTIER,
    side_metal=False,
    part_of_a_side=False,
    end_metal=True,
    via=False,
    strip_on_the_face=False,
    wall_at_both_ends=False,
    strip_at_both_ends=False,
):
    """The mesh of the stripline as it comes back, one port's element at a time.

    Each element's face is bounded by four curves running round it, each given
    the box round it: the end on the strip, across y at z 1, the end on the
    wall, across y at z 0, and its two sides, along z, which the plane holds.
    ``side_metal`` gives one side to the wall. ``part_of_a_side`` splits one side
    where a sheet ends, and gives the wall the part joined to the wall's end.
    ``end_metal`` false takes the strip off its end. ``via`` gives the strip's
    face a curve the wall holds as well, away from every element, as a sheet
    standing from the strip to the ground joins the two. ``strip_on_the_face``
    places the strip where the model ends, as a trace lying on the region's face
    stands. ``wall_at_both_ends`` gives the wall the element's end on the strip
    as well. ``strip_at_both_ends`` gives the strip the end on the wall instead,
    as two sheets under one binding across a series gap hold both ends.
    """
    labels = {}
    tag = 1
    for name, dimension, _ in described.labelled:
        sits = None if dimension == 3 else FRONTIER
        if name == "StripBinding":
            sits = FRONTIER if strip_on_the_face else INTERIOR
        if any(name == plane.label for plane in described.planes):
            sits = plane_sits
        if any(name == e.label for feed in described.lumped for e in feed.elements):
            sits = element_sits
        labels[name] = Label(dimension=dimension, tag=tag, entities=(tag,), sits=sits)
        tag += 1
    labels[described.wall] = Label(dimension=2, tag=tag, entities=(tag,), sits=FRONTIER)
    on_face = {label.tag: () for label in labels.values() if label.dimension == 2}
    on_curve = {}
    bounds = {}
    curve, point = 100, 1000
    strip = labels["StripBinding"].tag
    wall = labels[described.wall].tag
    half = WIDTH / 2

    def held_by(owner, *curves):
        if owner is not None:
            on_face[owner] = (*on_face[owner], *((1, c) for c in curves))

    for feed in described.lumped:
        rest = labels[feed.planes[0]].tag if feed.planes else None
        for element in feed.elements:
            top, bottom, left, right, split = range(curve, curve + 5)
            corner = list(range(point, point + 5))
            on_curve[top] = ((0, corner[0]), (0, corner[1]))
            on_curve[bottom] = ((0, corner[2]), (0, corner[3]))
            on_curve[right] = ((0, corner[1]), (0, corner[3]))
            bounds[top] = ((-HALF, -half, 1.0), (-HALF, half, 1.0))
            bounds[bottom] = ((-HALF, -half, 0.0), (-HALF, half, 0.0))
            bounds[right] = ((-HALF, half, 0.0), (-HALF, half, 1.0))
            if part_of_a_side:
                on_curve[left] = ((0, corner[4]), (0, corner[0]))
                on_curve[split] = ((0, corner[2]), (0, corner[4]))
                bounds[left] = ((-HALF, -half, 0.5), (-HALF, -half, 1.0))
                bounds[split] = ((-HALF, -half, 0.0), (-HALF, -half, 0.5))
                sides = (left, split, right)
            else:
                on_curve[left] = ((0, corner[0]), (0, corner[2]))
                bounds[left] = ((-HALF, -half, 0.0), (-HALF, -half, 1.0))
                sides = (left, right)
            on_face[labels[element.label].tag] = tuple((1, c) for c in (top, bottom, *sides))
            if end_metal:
                held_by(strip, top)
            if wall_at_both_ends:
                held_by(wall, top)
            held_by(strip if strip_at_both_ends else wall, bottom)
            if side_metal:
                held_by(wall, left)
                held_by(rest, right)
            elif part_of_a_side:
                held_by(wall, split)
                held_by(rest, left, right)
            else:
                held_by(rest, left, right)
            curve += 5
            point += 5
    if via:
        shared = curve
        on_curve[shared] = ((0, point), (0, point + 1))
        bounds[shared] = ((0.0, -half, 1.0), (0.0, half, 1.0))
        held_by(strip, shared)
        held_by(wall, shared)
    return Mesh(
        path="/run/model.msh",
        labels=labels,
        worst_quality={},
        edges=Edges(shortest=0.0, longest=0.0),
        settled=(),
        version="",
        algorithm={},
        rims={2: on_face, 1: on_curve},
        bounds={1: bounds},
    )


class TestAPartALumpedElementStandsOn:
    """A lumped element drives a field across its own face, so it joins two
    parts rather than dividing them, and a part holding one is reached."""

    def test_neither_an_element_nor_the_face_it_lies_in_divides_the_region(self):
        """An element drives a field across its own face. The rest of that face
        is a magnetic wall only where the model ends, and inside the model it
        carries nothing at all - so neither is a cut."""
        described = problem(stripline().study)
        elements = {element.label for feed in described.lumped for element in feed.elements}
        assert not elements & described.dividing
        assert not {plane.label for plane in described.planes} & described.dividing
        assert described.dividing == {described.conductors[0].label}

    def test_a_part_holding_one_is_taken(self):
        described = problem(stripline().study)
        driven = described.lumped[0].elements[0].label
        mesh = replace(
            meshed(described),
            parted=(Part(labels=(described.regions[0].label, driven), place="(0, 0, 0)"),),
        )
        check(described, mesh)

    def test_a_part_holding_none_is_refused(self):
        described = problem(stripline().study)
        metal = described.conductors[0].label
        mesh = replace(
            meshed(described),
            parted=(Part(labels=(described.regions[0].label, metal), place="(1, 2, 3)"),),
        )
        with pytest.raises(TranslationError) as refused:
            check(described, mesh)
        assert "no port stands on it" in str(refused.value)


class TestAnElementPartlyInsideMetal:
    """The part of an element inside a body of metal leaves the mesh with it,
    and Palace sizes an element off what it is given."""

    def test_an_element_that_lost_part_of_itself_is_refused(self):
        described = problem(stripline().study)
        element = described.lumped[0].elements[0].label
        mesh = replace(
            meshed(described),
            trimmed=(Trimmed(label=element, by=("Via",), place="(1, 2, 3) to (4, 5, 6)"),),
        )
        with pytest.raises(TranslationError) as refused:
            check(described, mesh)
        assert f"{described.lumped[0].label!r} lays an element partly inside 'Via'" in str(
            refused.value
        )
        assert "at (1, 2, 3) to (4, 5, 6)" in str(refused.value)

    def test_a_sheet_that_lost_part_of_itself_is_taken(self):
        """What a body of metal fills is metal whichever label was drawn there."""
        described = problem(stripline().study)
        metal = described.conductors[0].label
        mesh = replace(
            meshed(described),
            trimmed=(Trimmed(label=metal, by=("Via",), place="(1, 2, 3) to (4, 5, 6)"),),
        )
        check(described, mesh)


class TestWhatTheMeshDecides:
    def test_the_rest_of_the_face_where_the_model_ends_is_a_magnetic_wall(self):
        described = problem(stripline().study)
        mesh = meshed(described)
        run = configured(described, mesh, "results")
        planes = {mesh.labels[plane.label].tag for plane in described.planes}
        assert set(run.magnetic_conductor) == planes
        assert not planes & set(run.perfect_conductor)
        written = json.loads(run.to_json())["Boundaries"]
        assert sorted(written["PMC"]["Attributes"]) == sorted(planes)
        assert not planes & set(written["PEC"]["Attributes"])

    def test_the_rest_of_a_face_inside_the_model_carries_nothing(self):
        described = problem(stripline().study)
        run = configured(described, meshed(described, plane_sits=INTERIOR), "results")
        assert run.magnetic_conductor == ()
        assert "PMC" not in json.loads(run.to_json())["Boundaries"]

    def test_a_face_partly_inside_and_partly_where_it_ends_is_refused(self):
        described = problem(stripline().study)
        with pytest.raises(TranslationError) as raised:
            check(described, meshed(described, plane_sits=BOTH))
        assert "the elements of 'Port1' lie in a face that stands partly inside" in str(
            raised.value
        )

    def test_each_port_is_written_with_its_elements_and_resistance(self):
        described = problem(stripline().study)
        mesh = meshed(described)
        written = json.loads(configured(described, mesh, "results").to_json())["Boundaries"]
        assert "WavePort" not in written
        (first, second) = written["LumpedPort"]
        assert first == {
            "Index": 1,
            "R": 50.0,
            "Excitation": 1,
            "Elements": [
                {"Attributes": [mesh.labels["Port1 element 1"].tag], "Direction": "-Z"},
                {"Attributes": [mesh.labels["Port1 element 2"].tag], "Direction": "+Z"},
            ],
        }
        assert second["Excitation"] == 2
        assert "Active" not in first

    def test_the_power_through_a_port_is_asked_of_both_sides_of_its_elements(self):
        described = problem(stripline().study)
        mesh = meshed(described)
        written = json.loads(configured(described, mesh, "results").to_json())["Boundaries"]
        (first, _) = written["Postprocessing"]["SurfaceFlux"]
        assert first == {
            "Index": 1,
            "Attributes": sorted(
                [mesh.labels["Port1 element 1"].tag, mesh.labels["Port1 element 2"].tag]
            ),
            "Type": "Power",
            "TwoSided": True,
        }

    def test_an_element_between_two_conductors_is_taken(self):
        described = problem(stripline().study)
        check(described, meshed(described))

    @pytest.mark.parametrize("drawing", ["side_metal", "part_of_a_side"])
    def test_the_wall_along_a_side_is_refused_naming_the_port(self, drawing):
        """Along the whole side, or along the part of it joined to an end, where
        counting the metal round the element in pieces finds two."""
        described = problem(stripline().study)
        with pytest.raises(TranslationError) as raised:
            check(described, meshed(described, **{drawing: True}))
        said = str(raised.value)
        assert said.startswith(
            "'Port1' lays 'Port1 element 1', which the region's own boundary meets along a side"
        ), said
        assert "Draw the element clear of the edge of the region" in said

    def test_an_end_meeting_no_metal_is_refused_naming_the_port(self):
        described = problem(stripline().study)
        with pytest.raises(TranslationError) as raised:
            check(described, meshed(described, end_metal=False))
        assert str(raised.value).startswith(
            "'Port1' lays 'Port1 element 1', which meets metal at one end only"
        )

    def test_a_strip_joined_to_the_wall_away_from_the_element_is_taken_and_not_said(self):
        """A via, a shorted stub or a line running into the wall joins the strip
        to the ground somewhere along the line, and the element still drives
        across a gap between two pieces of metal."""
        described = problem(stripline().study)
        mesh = meshed(described, via=True)
        check(described, mesh)
        assert between_the_boundary(described, mesh) == []

    def test_one_binding_at_both_ends_is_taken_and_not_said(self):
        """Two sheets under one binding across a series gap: a binding is not a
        piece of metal."""
        described = problem(stripline().study)
        mesh = meshed(described, strip_at_both_ends=True)
        check(described, mesh)
        assert between_the_boundary(described, mesh) == []

    def test_a_trace_lying_on_the_region_s_face_is_taken_and_said(self):
        described = problem(stripline().study)
        mesh = meshed(described, strip_on_the_face=True)
        check(described, mesh)
        said = between_the_boundary(described, mesh)
        assert said[0] == (
            "'Port1' lays 'Port1 element 1', both of whose ends lie on the region's "
            "boundary, which this solver makes a perfect conductor where no metal is drawn, "
            "so the element is driven between two parts of that boundary. 'StripBinding' "
            "lies on the region's boundary at an end, and stands on the wall there"
        )

    def test_a_body_of_metal_at_an_end_is_not_the_boundary(self):
        """Every face a body leaves stands where the model ends, and the body is
        still a conductor of its own inside the region: an element from it to
        the wall drives between two conductors."""
        drawn = problem(stripline().study)
        described = replace(
            drawn, conductors=tuple(replace(one, solid=True) for one in drawn.conductors)
        )
        mesh = meshed(drawn, strip_on_the_face=True)
        assert between_the_boundary(described, mesh) == []

    def test_both_ends_on_the_wall_are_said_with_no_metal_named(self):
        described = problem(stripline().study)
        mesh = meshed(described, wall_at_both_ends=True)
        check(described, mesh)
        said = between_the_boundary(described, mesh)
        assert said[0] == (
            "'Port1' lays 'Port1 element 1', both of whose ends lie on the region's "
            "boundary, which this solver makes a perfect conductor where no metal is drawn, "
            "so the element is driven between two parts of that boundary"
        )

    def test_a_strip_inside_the_region_is_not_said(self):
        described = problem(stripline().study)
        assert between_the_boundary(described, meshed(described)) == []

    def test_what_is_said_is_a_warning_before_the_solve(self, tmp_path, monkeypatch):
        described = problem(stripline().study)
        mesh = replace(meshed(described, strip_on_the_face=True), path=str(tmp_path / "m.msh"))
        said = []

        def solving(config, processes, solver=None, on_output=None, cancel=None, launcher=None):
            said.append("solved")
            raise RuntimeError("stop")

        monkeypatch.setattr(pipeline.run, "solve", solving)
        prepared = pipeline.Prepared(
            problem=described, directory=tmp_path, binary=Path("palace"), pieces=()
        )
        with pytest.raises(RuntimeError):
            pipeline.finish(prepared, mesh, 1, on_output=said.append)
        warned = [line for line in said if line.startswith(balance.WARNING)]
        assert warned and said.index(warned[0]) < said.index("solved")
        assert "both of whose ends lie on the region's boundary" in warned[0]

    def test_what_stands_beside_each_port_is_said(self):
        described = problem(stripline().study)
        said = pipeline.beside(described, meshed(described))
        assert said[0].startswith(
            "'Port1': the whole of the region's face its elements lie in, 'rest of the face "
            "of Port1', is a magnetic wall wherever no metal is drawn on it"
        )

    def test_a_port_inside_the_model_says_what_stands_behind_it(self):
        described = problem(stripline().study)
        said = pipeline.beside(
            described, meshed(described, plane_sits=INTERIOR, element_sits=INTERIOR)
        )
        assert said[0].startswith("'Port1' stands inside the model")

    def test_an_element_inside_the_region_says_what_faces_it_and_how_far(self):
        """The strip ends five micrometres short of the end face, so its element
        lies on no face and the end face stands that far behind it."""
        line = stripline()
        line.strip.Shape = strip_sheet(start=-HALF + 5e-6)
        line.study.Group = [m for m in line.study.Group if m.Label != "Port2"]
        described = problem(line.study)
        (feed,) = described.lumped
        assert feed.planes == ()
        (first, _) = feed.elements
        assert first.facing[0][:2] == ("-X", "the region's boundary")
        assert first.facing[0][2] == pytest.approx(5e-6, rel=1e-6, abs=0.0)
        assert first.facing[1][:2] == ("+X", "the region's boundary")
        mesh = meshed(replace(described, planes=()), element_sits=INTERIOR)
        (said, _) = pipeline.beside(described, mesh)
        assert "on its -X side the region's boundary, 5e-06 mm away" in said, said

    def test_the_nearest_of_what_faces_an_element_is_named(self):
        """A sheet half a millimetre behind an element inside the region, and
        the end face five millimetres behind it on the same side."""
        line = stripline()
        line.strip.Shape = strip_sheet(start=-5.0)
        behind = face("behind", (-5.5, -1.0, 0.2), (-5.5, 1.0, 0.8))
        line.study.Group = [
            *(m for m in line.study.Group if m.Label != "Port2"),
            metal_binding("BehindMetal", behind),
        ]
        described = problem(line.study)
        (feed,) = described.lumped
        mesh = meshed(replace(described, planes=()), element_sits=INTERIOR)
        (said, _) = pipeline.beside(described, mesh)
        assert "on its -X side 'BehindMetal', 0.5 mm away" in said, said


def lumped_run(**changed):
    written = dict(
        mesh="line.msh",
        output="results",
        materials=(Material(attributes=(1,)),),
        perfect_conductor=(6, 9),
        magnetic_conductor=(7, 8),
        ports=(
            LumpedPort(
                index=1,
                elements=(LumpedElement((2,), "-Z"), LumpedElement((3,), "+Z")),
                resistance=50.0,
                excited=True,
            ),
            LumpedPort(index=2, elements=(LumpedElement((4,), "-Z"),), resistance=50.0),
        ),
        sweep=Sweep(2e9, 8e9, 3),
        order=2,
    )
    written.update(changed)
    return Driven(**written)


class TestTheLumpedPortWritten:
    @pytest.mark.parametrize("resistance", [0.0, -1.0, math.nan])
    def test_a_resistance_that_is_not_positive_cannot_be_written(self, resistance):
        with pytest.raises(ValueError, match="a resistance is positive"):
            LumpedPort(index=1, elements=(LumpedElement((2,), "+Z"),), resistance=resistance)

    def test_a_direction_that_is_not_an_axis_cannot_be_written(self):
        with pytest.raises(ValueError, match="one of \\+X"):
            LumpedElement((2,), "Z")

    def test_a_run_holding_both_kinds_of_port_cannot_be_written(self):
        guide = WavePort(index=3, attributes=(10,), behind=(0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="a wave port and a lumped port"):
            lumped_run(ports=(*lumped_run().ports, guide))

    def test_a_face_carrying_two_conditions_cannot_be_written(self):
        with pytest.raises(ValueError, match="more than one condition"):
            lumped_run(magnetic_conductor=(7, 9))

    def test_an_undriven_port_carries_no_excitation(self):
        (_, second) = json.loads(lumped_run().to_json())["Boundaries"]["LumpedPort"]
        assert "Excitation" not in second


def tables(directory, flux_driven, flux_passive):
    """A two-port matrix driven from port 1, and the power through each port as
    Palace measures it: what each port gives the model."""
    directory.mkdir(parents=True, exist_ok=True)
    frequencies = [2.0, 5.0, 8.0]
    head = ["f (GHz)"]
    rows = [[f] for f in frequencies]
    for out in (1, 2):
        head += [f"|S[{out}][1]| (dB)", f"arg(S[{out}][1]) (deg.)"]
        for row in rows:
            row += [-40.0 if out == 1 else -0.001, 0.0]
    (directory / TABLE).write_text(
        ",".join(head) + "\n" + "\n".join(",".join(str(v) for v in row) for row in rows) + "\n"
    )
    head = ["f (GHz)", "Φ_pow[1] (W)", "Φ_pow[2] (W)"]
    body = "\n".join(f"{f},{flux_driven},{flux_passive}" for f in frequencies)
    (directory / FLUX_TABLE).write_text(",".join(head) + "\n" + body + "\n")


class TestTheAnswerRead:
    def run(self):
        return lumped_run(
            ports=(
                LumpedPort(1, (LumpedElement((2,), "-Z"),), 50.0, excited=True),
                LumpedPort(2, (LumpedElement((4,), "-Z"),), 25.0),
            )
        )

    def test_a_lumped_port_states_its_resistance(self, tmp_path):
        tables(tmp_path, 0.99, -0.99)
        answer = scattering(tmp_path, self.run())
        assert np.all(answer.impedance[:, 0] == 50.0)
        assert np.all(answer.impedance[:, 1] == 25.0)
        assert answer.stated == {1: RESISTANCE, 2: RESISTANCE}

    def test_the_power_a_port_gives_the_model_is_read_as_what_leaves_it(self, tmp_path):
        """Palace measures what the elements give the model on both sides, and
        every figure the balance reads is what leaves: the driven port's is
        negative and the one it drives positive."""
        tables(tmp_path, 0.99, -0.98)
        answer = scattering(tmp_path, self.run())
        assert np.all(answer.flux[:, 0, 0] == -0.99)
        assert np.all(answer.flux[:, 1, 0] == 0.98)

    def test_the_result_names_the_reference_by_the_kind_of_port(self, tmp_path):
        tables(tmp_path, 0.99, -0.99)
        stored = results_glue.from_palace(scattering(tmp_path, self.run()), "line")
        assert stored.provenance[IMPEDANCE_STATED] == {
            "1": LUMPED_RESISTANCE,
            "2": LUMPED_RESISTANCE,
        }
        assert np.all(np.asarray(stored.reference)[:, 1] == 25.0)

    def test_the_file_header_says_a_lumped_port_is_no_guide(self, tmp_path):
        tables(tmp_path, 0.99, -0.99)
        stored = results_glue.from_palace(scattering(tmp_path, self.run()), "line")
        said = " ".join(stored._reference_lines())
        assert "resistance of its lumped element" in said
        assert "guide" not in said

    def test_a_wave_port_is_still_named_by_its_power_voltage_impedance(self):
        answer = Scattering(
            frequency=np.array([1e9]),
            out=(1,),
            driven=(1,),
            matrix=np.zeros((1, 1, 1), dtype=complex),
            flux=np.zeros((1, 1, 1)),
            dissipates=False,
            impedance=np.array([[40.0]]),
        )
        stored = results_glue.from_palace(answer, "guide")
        assert stored.provenance[IMPEDANCE_STATED] == {"1": POWER_VOLTAGE}

    def test_a_shortfall_through_a_lumped_port_gives_the_share_and_no_advice(self):
        shortfall = balance.Shortfall(excitation=1, frequency=8e9, share=2e-3, port=2)
        said = balance.said(shortfall, {1: "Port1", 2: "Port2"}, lumped={1, 2})
        assert said == (
            f"{balance.WARNING}Driven from 'Port1', the matrix leaves 0.2% of the power "
            "unaccounted for at 8 GHz. Most of it left through 'Port2', a lumped port"
        )


class TestTheRunOfALumpedPort:
    def finish(self, tmp_path, monkeypatch, conducting=False):
        described = problem(stripline().study)
        mesh = replace(meshed(described), path=str(tmp_path / "model.msh"))
        started = []

        def solving(config, processes, solver=None, on_output=None, cancel=None, launcher=None):
            started.append((Path(config).name, processes))
            return ""

        def answered(directory, run):
            return Scattering(
                frequency=np.array([2e9]),
                out=(1, 2),
                driven=(1, 2),
                matrix=np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=complex),
                flux=np.array([[[-1.0, 1.0], [1.0, -1.0]]]),
                dissipates=False,
            )

        monkeypatch.setattr(pipeline.run, "solve", solving)
        monkeypatch.setattr(pipeline.read, "scattering", answered)
        if conducting:
            real = pipeline.attributes.configured

            def with_a_sheet(problem_, mesh_, output):
                from Microwave.Solvers.palace.config import SurfaceConductivity

                return replace(
                    real(problem_, mesh_, output),
                    conducting=(SurfaceConductivity(attributes=(99,), conductivity=5.8e7),),
                )

            monkeypatch.setattr(pipeline.attributes, "configured", with_a_sheet)
        prepared = pipeline.Prepared(
            problem=described, directory=tmp_path, binary=Path("palace"), pieces=()
        )
        said = []
        pipeline.finish(prepared, mesh, 2, on_output=said.append)
        return started, said

    def test_no_mode_run_is_asked_for(self, tmp_path, monkeypatch):
        started, said = self.finish(tmp_path, monkeypatch)
        assert started == [("palace.json", 2)]
        assert any("is a magnetic wall" in line for line in said)

    def test_a_run_of_lumped_ports_beside_a_lossy_sheet_is_written_as_drawn(
        self, tmp_path, monkeypatch
    ):
        """The loss tangent is for the mode solve of a wave port whose face meets
        a finite conductor, and a lumped port has no mode solve."""
        started, said = self.finish(tmp_path, monkeypatch, conducting=True)
        assert started == [("palace.json", 2)]
        assert not any("loss tangent" in line for line in said)
        written = json.loads((tmp_path / "palace.json").read_text())
        assert not any("LossTan" in material for material in written["Domains"]["Materials"])
