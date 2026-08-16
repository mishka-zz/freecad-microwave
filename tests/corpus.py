# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The shapes a user can actually draw, as a corpus the geometry layer is run over.

The suite's hand-written ``Shape`` stub is a model of the CAD kernel, and a model
holds only the behaviours somebody thought to put in it - so a defect that lives
in the difference between the model and the kernel is invisible to every test
written against the model. These specimens come from the kernel.

They are organised by **constructor**, because that is the axis a user moves
along - a cone is drawn by the cone primitive, by revolving a triangle, or by
importing a STEP file, and the geometry layer is required not to care which.
Adding a constructor is a row here, not a branch anywhere else.

The dimensions are arbitrary and mean nothing on their own; what each specimen is
for is written beside it.

This module is imported by the probe that runs under FreeCAD and by the tests
that read what the probe wrote, so the kernel is imported lazily: under a plain
Python there is none, and the names still have to be enumerable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from Microwave.Solvers.openems.sizing import fits_inside

__all__ = ["CELL_CAP", "EDGE_SIZE", "Specimen", "Unavailable", "round_trip", "specimens"]

#: The coarsest cell the demands are measured against, and the size a metal edge
#: asks for, in mm. Here rather than in the probe because the reader of what the
#: probe wrote needs the cap too: a demand at or above it can never win the
#: sizing field's minimum, so it is the ceiling any comparison of two demand sets
#: is made under.
#:
#: Neither is a policy. What is compared is one shape's demands against the same
#: shape's after a round trip, so any pair that lets these dimensions produce
#: demands at all would do - and the cap has to be coarse enough that even the
#: gentlest curvature drawn here is still fine enough to bind, or the smoothest
#: specimens compare two empty sets.
CELL_CAP = 5.0
EDGE_SIZE = 0.25

#: The thickness a conductor drawn as a surface is given here, in mm. The edge
#: size stands in for the cell the metal is meshed at, which is what the adapter
#: measures the thickness against; what it has to be for these specimens is
#: small against every surface drawn with ``skin`` set, and large enough that
#: the kernel has something to offset. Put through the same inverse the adapter
#: uses, so a specimen is offered what a document would be offered.
SKIN = fits_inside(EDGE_SIZE)


class Unavailable(Exception):
    """This machine cannot draw the specimen, and that says nothing about the
    geometry layer. Raised for a missing prerequisite - a font, an importer -
    and never for a shape the kernel declined to build."""


#: Where each platform keeps a font with outline glyphs. A specimen that needs
#: one and finds none reports itself unavailable rather than undrawable, since a
#: machine's font list says nothing about the geometry layer.
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


@dataclass(frozen=True)
class Specimen:
    """One shape, and what it is in the corpus to say.

    :param name: Stable identifier. It names the artifact the probe writes and
        the test that reads it, so it is not renamed casually.
    :param build: Takes the ``Part`` module and returns a shape. What else the
        kernel offers is imported inside the builder that wants it, so importing
        this module needs no kernel at all and the names stay enumerable.
    :param subject: What this specimen is here to exercise, in a few words. It
        goes into the failure message, so a red corpus entry says what broke
        rather than only which name broke.
    :param expect: ``meshed`` where the geometry layer is required to accept the
        shape, ``refused`` where it is required to reject it with a message
        naming the object. Anything else is a defect either way, which is the
        whole point of stating it: a shape that quietly vanishes passes a test
        that only asks "did it not crash".
    :param tags: Coarse groupings a test can select on.
    :param wall: The shape's own thinnest cross-section, where it has one and it
        is what the specimen is here to test. Declared from the dimensions the
        builder draws with, so it is the drawing's figure and not a measurement
        of the code. A specimen leaving it unset is not gated on it - a shape
        with no single thickness has no such figure to state.
    :param gap: The clearance between two lumps of the one object, declared the
        same way and for the same reason. A gap that closes is two conductors
        shorted, which is a run that completes and answers about a device
        nobody drew.
    :param skin: Whether to run this one as a conductor, which is the only kind
        of material a surface carrying no thickness is given one. Off by
        default, so every specimen already here keeps being asked the question
        it was written to answer.
    """

    name: str
    build: Callable[[Any], Any]
    subject: str
    expect: str = "meshed"
    tags: tuple[str, ...] = field(default_factory=tuple)
    wall: float | None = None
    gap: float | None = None
    skin: bool = False


def _vector(part: Any, x: float, y: float, z: float) -> Any:
    """A kernel vector. ``Part`` does not carry one, so it comes from the app."""
    import FreeCAD

    return FreeCAD.Vector(x, y, z)


# ----------------------------------------------------------------- primitives


def _box(part: Any) -> Any:
    return part.makeBox(6.0, 4.0, 2.0)


def _box_rotated(part: Any) -> Any:
    """A box no rectilinear grid holds. The bounding box is honest and useless."""
    shape = part.makeBox(6.0, 4.0, 2.0)
    shape.rotate(_vector(part, 0, 0, 0), _vector(part, 0, 0, 1), 30.0)
    return shape


def _cylinder(part: Any) -> Any:
    return part.makeCylinder(3.0, 8.0)


def _cone(part: Any) -> Any:
    """A curved solid drawn as a primitive. Where a view provider exists, the
    kernel answers ``tessellate`` with the display mesh built for it, which is
    cut to a cosmetic deviation rather than to anything the solver asked for."""
    return part.makeCone(4.0, 1.0, 6.0)


def _sphere(part: Any) -> Any:
    """One face and a seam edge whose curve type the kernel will not name."""
    return part.makeSphere(4.0)


def _torus(part: Any) -> Any:
    """Genus one. The hole is the part a containment check has to get right."""
    return part.makeTorus(6.0, 2.0)


def _wedge(part: Any) -> Any:
    return part.makeWedge(0.0, 0.0, 0.0, 2.0, 2.0, 5.0, 4.0, 3.0, 3.0, 4.0)


def _ellipsoid(part: Any) -> Any:
    """Curvature that varies over a face, so one radius cannot stand for it."""
    import FreeCAD

    shape = part.makeSphere(3.0)
    matrix = FreeCAD.Matrix()
    matrix.scale(1.0, 2.0, 0.5)
    return shape.transformGeometry(matrix)


# ------------------------------------------------------------------ booleans


def _cut(part: Any) -> Any:
    """A hole through a block. Returns a compound, as every boolean does."""
    return part.makeBox(8.0, 8.0, 3.0).cut(
        part.makeCylinder(1.5, 9.0, _vector(part, 4.0, 4.0, -3.0))
    )


def _fuse(part: Any) -> Any:
    return part.makeBox(6.0, 6.0, 3.0).fuse(part.makeSphere(2.5, _vector(part, 6.0, 3.0, 1.5)))


def _common(part: Any) -> Any:
    return part.makeBox(6.0, 6.0, 6.0).common(part.makeSphere(4.0, _vector(part, 3.0, 3.0, 3.0)))


def _cut_to_a_box(part: Any) -> Any:
    """A boolean whose *result* is a box. The layer must judge the measurement
    and not the history: this has to be held exactly, like a drawn box."""
    return part.makeBox(8.0, 4.0, 2.0).cut(
        part.makeBox(4.0, 4.0, 2.0, _vector(part, 4.0, 0.0, 0.0))
    )


# ------------------------------------------------------------------- features


def _fillet(part: Any) -> Any:
    """Curvature that is a local detail on a thick body, which is the case the
    curvature bound reads as a thickness and is wrong about."""
    shape = part.makeBox(8.0, 6.0, 3.0)
    return shape.makeFillet(0.6, shape.Edges)


def _chamfer(part: Any) -> Any:
    shape = part.makeBox(8.0, 6.0, 3.0)
    return shape.makeChamfer(0.6, shape.Edges)


def _thickness(part: Any) -> Any:
    """A hollow shell of finite wall. The wall is the feature, and no witness
    pair between distinct solids can see it."""
    shape = part.makeBox(8.0, 6.0, 4.0)
    return shape.makeThickness([shape.Faces[0]], -0.5, 1e-3)


#: The wall :func:`_hollow_sphere` is drawn with, and the outer radius it is
#: drawn on. Named because the specimen declares the first to the gate that
#: reads it, and a wall stated twice is a wall the two can disagree about.
HOLLOW_WALL = 0.5
HOLLOW_RADIUS = 6.0


def _hollow_sphere(part: Any) -> Any:
    """A wall with no edge anywhere on it, which is what makes it a gate.

    Every other hollow specimen is a box, and a box has sharp joins - those ask
    for cells finer than the wall does, so they answer the question before the
    wall is ever consulted. A sphere has none: the only thing on it that can ask
    for a cell to fit inside the wall is the wall.

    Its curvature cannot stand in either. The radius the surface carries is the
    sphere's, an order above the wall, so a rule that reads thickness off
    curvature reads this shape as thick.
    """
    return part.makeSphere(HOLLOW_RADIUS).cut(part.makeSphere(HOLLOW_RADIUS - HOLLOW_WALL))


# --------------------------------------------------------------------- sweeps


def _extrude(part: Any) -> Any:
    return part.Face(part.Wire(part.makeCircle(3.0).Edges)).extrude(_vector(part, 0.0, 0.0, 5.0))


def _revolve(part: Any) -> Any:
    """A cone by revolution. Nothing downstream may tell it from the primitive."""
    outline = part.makePolygon(
        [
            _vector(part, 0.0, 0.0, 0.0),
            _vector(part, 4.0, 0.0, 0.0),
            _vector(part, 1.0, 0.0, 6.0),
            _vector(part, 0.0, 0.0, 6.0),
            _vector(part, 0.0, 0.0, 0.0),
        ]
    )
    return part.Face(outline).revolve(
        _vector(part, 0.0, 0.0, 0.0), _vector(part, 0.0, 0.0, 1.0), 360.0
    )


def _loft(part: Any) -> Any:
    return part.makeLoft(
        [
            part.Wire(part.makeCircle(3.0).Edges),
            part.Wire(part.makeCircle(1.0, _vector(part, 0.0, 0.0, 5.0)).Edges),
        ],
        True,
    )


def _pipe(part: Any) -> Any:
    """A round conductor that lies along no axis, which is what the connection
    criterion exists for."""
    spine = part.Wire(
        part.makePolygon(
            [
                _vector(part, 0.0, 0.0, 0.0),
                _vector(part, 0.0, 0.0, 5.0),
                _vector(part, 4.0, 0.0, 8.0),
            ]
        )
    )
    return spine.makePipeShell([part.Wire(part.makeCircle(0.8).Edges)], True, True)


def _wire_dipole(part: Any) -> Any:
    """A thin round conductor lying along no axis: a wire antenna.

    Its volume is a millionth of the box that bounds it, which is the regime
    where a triangulation is at its worst - the fineness a request opens at
    follows the extent of the box rather than the wire's own diameter, so the
    first answer is a prism inscribed in the circle and misses a third of the
    metal. Nothing thinner than this is drawn anywhere else here, and the check
    that the triangulation is still the shape has nothing else to bite on.
    """
    return part.makeCylinder(
        0.02, 100.0, _vector(part, 0.0, 0.0, 0.0), _vector(part, 1.0, 1.0, 1.0)
    )


def _helix(part: Any) -> Any:
    """A coil: thin, curved, and along every axis at once."""
    spine = part.makeHelix(4.0, 8.0, 2.5)
    section = part.Wire(part.makeCircle(1.0, _vector(part, 2.5, 0.0, 0.0)).Edges)
    return spine.makePipeShell([section], True, True)


# ------------------------------------------------------------------- flat area


def _rectangle(part: Any) -> Any:
    """A ground plane drawn as a face. Held exactly, and an anchor everywhere."""
    return part.Face(
        part.makePolygon(
            [
                _vector(part, 0.0, 0.0, 0.0),
                _vector(part, 8.0, 0.0, 0.0),
                _vector(part, 8.0, 5.0, 0.0),
                _vector(part, 0.0, 5.0, 0.0),
                _vector(part, 0.0, 0.0, 0.0),
            ]
        )
    )


def _disc(part: Any) -> Any:
    """A flat outline that is not rectangles, so it is held as triangles and its
    boundary is the only thing that can ask for a line near it."""
    return part.Face(part.Wire(part.makeCircle(4.0).Edges))


def _annulus(part: Any) -> Any:
    """A flat area with a hole, so the triangulation has to carry the hole and
    not the outer contour alone."""
    outer = part.Face(part.Wire(part.makeCircle(4.0).Edges))
    inner = part.Face(part.Wire(part.makeCircle(1.5).Edges))
    return outer.cut(inner)


def _sheet_on_x(part: Any) -> Any:
    """A flat area normal to x, and not a rectangle, so it is held as triangles
    rather than as a box and reaches the polygon primitive at all. Its two
    in-plane axes are y then z, which is both the cycle and ascending order."""
    return part.Face(
        part.makePolygon(
            [
                _vector(part, 2.0, 0.0, 0.0),
                _vector(part, 2.0, 6.0, 0.0),
                _vector(part, 2.0, 6.0, 4.0),
                _vector(part, 2.0, 0.0, 0.0),
            ]
        )
    )


def _sheet_on_y(part: Any) -> Any:
    """A flat area normal to y, and the one that tells a cycle from a sort.

    A polygon's two coordinate lists are read as axes ``(n+1)%3`` and
    ``(n+2)%3``. That is the same as the remaining axes in ascending order for a
    sheet normal to x or to z, and the reverse of it for one normal to y - so a
    sheet laid the obvious way arrives mirrored, and only here does it show.
    """
    shape = part.Face(
        part.makePolygon(
            [
                _vector(part, 0.0, 0.0, 0.0),
                _vector(part, 8.0, 0.0, 0.0),
                _vector(part, 8.0, 0.0, 3.0),
                _vector(part, 0.0, 0.0, 0.0),
            ]
        )
    )
    return shape


def _sheet_disc_elevated(part: Any) -> Any:
    """A curved flat area away from every coordinate plane.

    Only two coordinates and an elevation are sent, so where the elevation comes
    from is a choice - and one made from a face lying at zero is right whichever
    corner of the box it was read off.
    """
    shape = _disc(part)
    shape.translate(_vector(part, 1.0, 2.0, 3.5))
    return shape


def _rectangle_off_plane(part: Any) -> Any:
    """Flat, and not on a coordinate plane. Its elevation is the anchor."""
    shape = _rectangle(part)
    shape.translate(_vector(part, -2.0, -1.0, 3.5))
    return shape


def _text(part: Any) -> Any:
    """Outline text: many small curved edges bounding one flat area, with
    counters, which is the densest boundary a drawing produces by accident."""
    import Draft

    font = _font()
    if font is None:
        raise Unavailable("no font with outline glyphs was found on this machine")
    string = Draft.make_shapestring("FDTD", font, 6.0)
    string.Document.recompute()
    return string.Shape


def _font() -> str | None:
    import os

    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------- awkwardness


def _negative_octant(part: Any) -> Any:
    """Wholly below the origin on every axis, and curved so that it is emitted
    as a polyhedron. The engine builds the ray for its containment test by
    scaling the bounding box maximum, which for this shape points back into it -
    a box would take the other path and never reach that code."""
    return part.makeCylinder(2.0, 5.0, _vector(part, -8.0, -8.0, -9.0))


def _straddling_origin(part: Any) -> Any:
    return part.makeCylinder(2.0, 6.0, _vector(part, -1.0, -1.0, -3.0))


def _thin_sliver(part: Any) -> Any:
    """A conductor far thinner than the domain it sits in, which is the ordinary
    shape of a foil and the one whose thickness is easiest to lose."""
    return part.makeBox(8.0, 6.0, 0.035)


def _near_flat(part: Any) -> Any:
    """Thin enough to look flat and thick enough not to be. Whichever way it is
    classified, it must not be refused for a thickness nobody drew."""
    return part.makeBox(8.0, 6.0, 1e-6)


def _two_disjoint(part: Any) -> Any:
    """One object, two pieces, and a bounding box holding the gap between them."""
    return part.makeCompound(
        [
            part.makeBox(3.0, 3.0, 3.0),
            part.makeBox(3.0, 3.0, 3.0, _vector(part, 6.0, 0.0, 0.0)),
        ]
    )


def _touching_at_a_corner(part: Any) -> Any:
    """Two blocks sharing one vertex. The engine conducts along edges that share
    a node, so whether this is one conductor or two is a real question - and a
    polyhedron cannot represent the vertex at all."""
    return part.makeCompound(
        [
            part.makeBox(3.0, 3.0, 3.0),
            part.makeBox(3.0, 3.0, 3.0, _vector(part, 3.0, 3.0, 3.0)),
        ]
    )


def _narrow_gap(part: Any) -> Any:
    """Two conductors a hair apart, where the gap between them is smaller than
    either of the bodies and is the only thing separating one from two."""
    return part.makeCompound(
        [
            part.makeBox(6.0, 3.0, 1.0),
            part.makeBox(6.0, 3.0, 1.0, _vector(part, 0.0, 3.05, 0.0)),
        ]
    )


#: The gap :func:`_sphere_pair` is drawn with, and the radius of each lump.
#: Named for the reason :data:`HOLLOW_WALL` is - the specimen declares the gap
#: to the gate that reads it, and a figure stated twice is one the two can
#: disagree about. The gap is under the cell the radius asks for through
#: :data:`~Microwave.Solvers.openems.lfs.SURFACE_FIDELITY`, so it is the finest
#: thing anywhere on the shape, and the gate asserts that rather than assuming
#: it.
PAIR_GAP = 0.3
PAIR_RADIUS = 5.0


def _sphere_pair(part: Any) -> Any:
    """A gap between two lumps of one object, with nothing else to measure it.

    The lumps are spheres for the reason :func:`_hollow_sphere` is one: a sphere
    carries no sharp join, and a radius this size asks for cells well coarser
    than this gap, so the only thing on the drawing that can ask for a cell to
    fit between them is the gap itself. Two cylinders - the pair of vias this is
    really about - would have their rims asking for the metal edge size all
    round the gap, and would pass whether or not anything measured it.

    Both lumps are curved, so both reach the mesher as triangles. A pair of
    boxes would not: each pins its own faces and the thirds rule sizes what is
    between them, so nothing here would be exercised.
    """
    return part.makeCompound(
        [
            part.makeSphere(PAIR_RADIUS),
            part.makeSphere(PAIR_RADIUS, _vector(part, 2.0 * PAIR_RADIUS + PAIR_GAP, 0.0, 0.0)),
        ]
    )


#: The gap :func:`_pad_pair` is drawn with, and the radius of each pad. Declared
#: for the reason :data:`PAIR_GAP` is, and finer than the pads' own rims ask for
#: anywhere, so the gap is the finest thing on the drawing and the gate that
#: reads it can assert the demand rather than merely clear it.
PAD_GAP = 0.1
PAD_RADIUS = 4.0


def _pad_pair(part: Any) -> Any:
    """A gap between two *sheets* of one object, drawn in one operation.

    What makes this its own case beside :func:`_sphere_pair` is that a surface
    is not decomposed the way a volume is: a shell is one surface however many
    faces went into it, so the regions of a compound carrying no volume cannot
    be its faces.

    Round for the reason the spheres are round: a pair of rectangles is cut into
    boxes that pin their own grid lines, and the clearance between them is then
    sized by the thirds rule rather than by anything that measured it.
    """
    return part.makeCompound(
        [
            part.Face(part.Wire(part.makeCircle(PAD_RADIUS).Edges)),
            part.Face(
                part.Wire(
                    part.makeCircle(
                        PAD_RADIUS, _vector(part, 2.0 * PAD_RADIUS + PAD_GAP, 0.0, 0.0)
                    ).Edges
                )
            ),
        ]
    )


def _mixed_compound(part: Any) -> Any:
    """Solids and a loose face in one object. Whether that face is a conductor
    of its own or the leftover of a construction cannot be read from a drawing,
    and emitting only the solids would drop it without saying so."""
    return part.makeCompound(
        [
            part.makeBox(3.0, 3.0, 3.0),
            part.makeBox(3.0, 3.0, 3.0, _vector(part, 6.0, 0.0, 0.0)),
            part.Face(
                part.makePolygon(
                    [
                        _vector(part, 0.0, 0.0, 5.0),
                        _vector(part, 9.0, 0.0, 5.0),
                        _vector(part, 9.0, 9.0, 5.0),
                        _vector(part, 0.0, 9.0, 5.0),
                        _vector(part, 0.0, 0.0, 5.0),
                    ]
                )
            ),
        ]
    )


def _inside_out(part: Any) -> Any:
    """A box whose faces are wound the other way. The kernel reads it as the
    complement: its volume is negative, every point in the box answers outside,
    and every point beyond answers inside."""
    return part.makeBox(4.0, 3.0, 2.0).reversed()


def _open_shell(part: Any) -> Any:
    """Three faces of a box. It reports a volume, so nothing that trusts
    ``Volume`` can tell it from a solid - and the engine reads an open surface as
    a sheet that contains no point at all, printing nothing."""
    return part.Shell(part.makeBox(4.0, 4.0, 4.0).Faces[:3])


def _self_intersecting(part: Any) -> Any:
    """Two solids overlapping as a compound rather than fused, so the same space
    is claimed twice."""
    return part.makeCompound(
        [
            part.makeBox(4.0, 4.0, 4.0),
            part.makeBox(4.0, 4.0, 4.0, _vector(part, 2.0, 2.0, 2.0)),
        ]
    )


def _tilted_sheet(part: Any) -> Any:
    """Flat, and flat on no axis. There is no plane to declare it at."""
    shape = _rectangle(part)
    shape.rotate(_vector(part, 0, 0, 0), _vector(part, 1, 0, 0), 35.0)
    return shape


def _pipe_wall(part: Any) -> Any:
    """A cylinder's lateral face, open at both ends. How a coaxial shield's bore
    reads in CAD, and the drawing a conductor with no thickness arrives as."""
    return part.makeCylinder(5.0, 20.0).Faces[0]


def _tapered_wall(part: Any) -> Any:
    """A cone's lateral face: a horn, and a wall whose radius varies along it."""
    return part.makeCone(5.0, 2.0, 20.0).Faces[0]


def _reflector(part: Any) -> Any:
    """Half a cylinder wall - open on three sides, and metal on both faces, so
    which way the offset runs cannot be read off the drawing."""
    return part.makeCylinder(
        5.0, 20.0, _vector(part, 0, 0, 0), _vector(part, 0, 0, 1), 180.0
    ).Faces[0]


def _creased_wall(part: Any) -> Any:
    """Three faces of a box: a shell with two right-angle creases, where an
    offset of one face runs into the offset of its neighbour."""
    return part.Shell(part.makeBox(8.0, 6.0, 4.0).Faces[:3])


def _narrow_skin(part: Any) -> Any:
    """A ribbon far longer than it is wide, drawn as metal with no thickness.

    Narrower across than the thickness a skin would be given, so what would be
    solved is a bar. It is the case a root of the area cannot see - that reads
    this as wide as its diagonal. Tilted, because a ribbon lying on an axis is
    held as the area it is and never asks for a thickness at all."""
    shape = part.makePlane(20.0, SKIN / 2.0)
    shape.rotate(_vector(part, 0, 0, 0), _vector(part, 1, 0, 0), 35.0)
    return shape


def _tilted_skin(part: Any) -> Any:
    """The tilted sheet again, as metal. Flat, and flat on no axis - so it has
    no elevation to be laid at and is given thickness instead."""
    return _tilted_sheet(part)


def _degenerate_line(part: Any) -> Any:
    """A wire. A region needs area, and this has none - the kernel will not even
    build a box that thin, so the honest way to draw it is as the curve it is."""
    return part.Wire([part.makeLine(_vector(part, 0.0, 0.0, 0.0), _vector(part, 8.0, 0.0, 0.0))])


def _tiny_against_the_domain(part: Any) -> Any:
    """Orders of magnitude smaller than the domain, and away from its walls."""
    return part.makeBox(0.01, 0.01, 0.01, _vector(part, 5.0, 5.0, 5.0))


# ------------------------------------------------------------------ round trip


def _stepped(part: Any) -> Any:
    """A cone through STEP. The export strips every trace of what drew it, so
    this and the primitive have to be treated identically."""
    return round_trip(part, _cone(part), "step")


def _brepped(part: Any) -> Any:
    return round_trip(part, _cylinder(part), "brep")


def round_trip(part: Any, shape: Any, suffix: str) -> Any:
    """The same shape, written to a file and read back.

    Public because it is two things at once: how the round-trip specimens above
    are drawn, and the transformation every *other* specimen is put through to
    check that the geometry layer treats a shape and its round trip alike. One
    copy, so the specimen and the invariant cannot drift apart.

    ``step`` is the one that strips: it carries surfaces and the curves trimming
    them, and no proxy, no parametric history and no FreeCAD type, so anything
    that came back different was read off something other than the geometry.
    ``brep`` is the kernel's own format and keeps far more, which makes it the
    weaker transformation and the one a specimen asks for by name.
    """
    import os
    import tempfile

    handle, path = tempfile.mkstemp(suffix=f".{suffix}")
    os.close(handle)
    try:
        if suffix == "step":
            shape.exportStep(path)
            return part.read(path)
        shape.exportBrep(path)
        restored = part.Shape()
        restored.importBrep(path)
        return restored
    finally:
        os.unlink(path)


def specimens() -> tuple[Specimen, ...]:
    """The corpus. One entry per constructor, and the awkward cases by name."""
    return (
        Specimen("box", _box, "the case a rectilinear grid holds exactly", tags=("primitive",)),
        Specimen(
            "box_rotated",
            _box_rotated,
            "a box no grid holds, whose bounding box is a plausible wrong answer",
            tags=("primitive",),
        ),
        Specimen("cylinder", _cylinder, "a curved face with no witness pair", tags=("primitive",)),
        Specimen(
            "cone", _cone, "the shape whose triangulation was a display mesh", tags=("primitive",)
        ),
        Specimen("sphere", _sphere, "one face, and a seam with no curve type", tags=("primitive",)),
        Specimen(
            "torus", _torus, "genus one, so containment has to find the hole", tags=("primitive",)
        ),
        Specimen("wedge", _wedge, "planar faces on none of the axes", tags=("primitive",)),
        Specimen(
            "ellipsoid", _ellipsoid, "curvature that varies across a face", tags=("primitive",)
        ),
        Specimen(
            "boolean_cut", _cut, "a hole, and a compound rather than a solid", tags=("boolean",)
        ),
        Specimen("boolean_fuse", _fuse, "a curved body joined to a flat one", tags=("boolean",)),
        Specimen(
            "boolean_common", _common, "a curved face bounded by planar ones", tags=("boolean",)
        ),
        Specimen(
            "boolean_to_a_box",
            _cut_to_a_box,
            "a boolean whose result is a box, judged by measurement not history",
            tags=("boolean",),
        ),
        Specimen(
            "fillet", _fillet, "small curvature as a detail on a thick body", tags=("feature",)
        ),
        Specimen("chamfer", _chamfer, "a sharp join replaced by two", tags=("feature",)),
        Specimen(
            "thickness", _thickness, "a wall no inter-solid witness pair sees", tags=("feature",)
        ),
        Specimen(
            "hollow_sphere",
            _hollow_sphere,
            "a wall with no edge and no curvature that reveals it",
            tags=("feature",),
            wall=HOLLOW_WALL,
        ),
        Specimen("extrude", _extrude, "a cylinder that was never the primitive", tags=("sweep",)),
        Specimen("revolve", _revolve, "a cone that was never the primitive", tags=("sweep",)),
        Specimen("loft", _loft, "a ruled surface between two circles", tags=("sweep",)),
        Specimen("pipe", _pipe, "a round conductor lying along no axis", tags=("sweep",)),
        Specimen("helix", _helix, "a thin conductor curving through every axis", tags=("sweep",)),
        Specimen(
            "wire_dipole",
            _wire_dipole,
            "a solid whose volume is a millionth of its own box",
            tags=("primitive",),
        ),
        Specimen("sheet_rectangle", _rectangle, "a flat area held exactly", tags=("sheet",)),
        Specimen("sheet_disc", _disc, "a flat area whose outline is curved", tags=("sheet",)),
        Specimen("sheet_annulus", _annulus, "a flat area with a hole in it", tags=("sheet",)),
        Specimen("sheet_on_x", _sheet_on_x, "a flat area normal to x", tags=("sheet",)),
        Specimen(
            "sheet_on_y",
            _sheet_on_y,
            "a flat area normal to y, where a coordinate cycle differs from a sort",
            tags=("sheet",),
        ),
        Specimen(
            "sheet_disc_elevated",
            _sheet_disc_elevated,
            "a curved flat area at a plane that is not zero",
            tags=("sheet",),
        ),
        Specimen(
            "sheet_elevated",
            _rectangle_off_plane,
            "a flat area away from a coordinate plane",
            tags=("sheet",),
        ),
        Specimen(
            "sheet_text",
            _text,
            "many curved edges bounding one area, with counters",
            tags=("sheet",),
        ),
        Specimen(
            "step_round_trip",
            _stepped,
            "a cone with every trace of its constructor gone",
            tags=("import",),
        ),
        Specimen(
            "brep_round_trip",
            _brepped,
            "a cylinder through the kernel's own format",
            tags=("import",),
        ),
        Specimen(
            "negative_octant",
            _negative_octant,
            "wholly below the origin, where the parity ray inverts",
            tags=("awkward",),
        ),
        Specimen(
            "straddling_origin",
            _straddling_origin,
            "across the origin on every axis",
            tags=("awkward",),
        ),
        Specimen(
            "thin_sliver",
            _thin_sliver,
            "a conductor far thinner than the domain",
            tags=("awkward",),
        ),
        Specimen(
            "near_flat",
            _near_flat,
            "too thin to be a solid, too thick to be a sheet",
            tags=("awkward",),
        ),
        Specimen("two_disjoint", _two_disjoint, "one object in two pieces", tags=("awkward",)),
        Specimen(
            "touching_at_a_corner",
            _touching_at_a_corner,
            "two blocks sharing one vertex",
            tags=("awkward",),
        ),
        Specimen("narrow_gap", _narrow_gap, "a gap that shorts if it is missed", tags=("awkward",)),
        Specimen(
            "sphere_pair",
            _sphere_pair,
            "a gap between two lumps of one object, and nothing else to measure it",
            tags=("awkward",),
            gap=PAIR_GAP,
        ),
        Specimen(
            "pad_pair",
            _pad_pair,
            "a gap between two sheets of one object, which holds no solid to split",
            tags=("awkward",),
            gap=PAD_GAP,
        ),
        Specimen(
            "tiny_against_the_domain",
            _tiny_against_the_domain,
            "small enough to grade across the domain",
            tags=("awkward",),
        ),
        Specimen(
            "self_intersecting",
            _self_intersecting,
            "the same space claimed twice",
            tags=("awkward",),
        ),
        Specimen(
            "inside_out",
            _inside_out,
            "a solid wound the other way, which the kernel reads as everything else",
            expect="refused",
            tags=("awkward",),
        ),
        Specimen(
            "mixed_compound",
            _mixed_compound,
            "solids and a loose face in one object, part volume and part surface",
            expect="refused",
            tags=("awkward",),
        ),
        Specimen(
            "open_shell",
            _open_shell,
            "an unclosed surface that reports a volume anyway",
            expect="refused",
            tags=("awkward",),
        ),
        Specimen(
            "tilted_sheet",
            _tilted_sheet,
            "flat, on no axis, so there is no plane to declare",
            expect="refused",
            tags=("awkward", "sheet"),
        ),
        Specimen(
            "degenerate_line",
            _degenerate_line,
            "flat in two axes, so it has no area to model",
            expect="refused",
            tags=("awkward",),
        ),
        Specimen(
            "pipe_wall",
            _pipe_wall,
            "a conductor drawn as a curved surface, open at both ends",
            tags=("skin",),
            skin=True,
        ),
        Specimen(
            "tapered_wall",
            _tapered_wall,
            "a skin whose radius varies along it",
            tags=("skin",),
            skin=True,
        ),
        Specimen(
            "reflector",
            _reflector,
            "a skin open on three sides, with metal wanted on both of its faces",
            tags=("skin",),
            skin=True,
        ),
        Specimen(
            "creased_wall",
            _creased_wall,
            "a skin whose faces meet at a corner, where one offset runs into the next",
            tags=("skin",),
            skin=True,
        ),
        Specimen(
            "tilted_skin",
            _tilted_skin,
            "flat on no axis, so it has no elevation and takes a thickness instead",
            tags=("skin", "sheet"),
            skin=True,
        ),
        Specimen(
            "narrow_skin",
            _narrow_skin,
            "a skin narrower across than the thickness it would be given",
            expect="refused",
            tags=("skin", "sheet"),
            skin=True,
        ),
    )
