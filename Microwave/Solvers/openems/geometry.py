# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A drawn shape, as the regions openEMS will hold it in.

openEMS holds a region either as an axis-aligned box or as a closed triangulated
surface. This module decides which of the two a shape is, and it reads that off
the shape's own volume rather than off whatever drew it: a shape that fills its
bounding box is sent as a box, one that does not is sent as its own
triangulation, and being curved, rotated or swept is not by itself a reason to
refuse anything.

Refused here, by name:

* a shape flat in two of the three axes, which is a line or a point and has no
  area to model;
* a shape that is part volume and part surface, since which of the two a face
  belonging to no solid was meant to be cannot be read from the drawing;
* a solid wound inside out, which describes everything *except* the space it
  appears to occupy and so is not bounded;
* a *dielectric* surface that encloses no volume and lies flat on none of the
  three axes: a layer with no thickness is laid at an elevation, a curved or a
  tilted one has none, and its thickness is most of what the layer does so
  nothing here may supply one. A **conductor** drawn that way is given a
  thickness instead, the field inside metal being zero either way;
* a shape whose own triangulation comes back open, or comes back having lost
  enough of the drawing to be a different object.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass, replace
from typing import Any

from ... import picks, portbox
from .mesh import SizingRegion
from .model import DIMENSIONS
from .properties import AXIS_NAMES, TranslationError, _label, _value
from .rectilinear import RectilinearError, rectangles
from .surface import covered_area, enclosed_volume, surface_fault

#: Re-exported, not redefined: see :data:`Microwave.portbox.FLATNESS` for the
#: number and the reasoning. A second copy here is a fact that drifts.
FLATNESS = portbox.FLATNESS


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in millimetres, corners sorted."""

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    @property
    def extents(self) -> tuple[float, float, float]:
        return tuple(b - a for a, b in zip(self.lower, self.upper))  # type: ignore[return-value]

    def middle(self, dim: int) -> float:
        return 0.5 * (self.lower[dim] + self.upper[dim])

    def nearest(self, dim: int, to: float) -> float:
        """Whichever face along ``dim`` lies closer to ``to``."""
        low, high = self.lower[dim], self.upper[dim]
        return low if abs(low - to) <= abs(high - to) else high

    def is_flat(self, dim: int) -> bool:
        return self.extents[dim] <= FLATNESS

    def as_pair(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The form ``Microwave.portbox`` takes: two corner triples."""
        return (self.lower, self.upper)


Vertex = tuple[float, float, float]


Triangle = tuple[int, int, int]


@dataclass(frozen=True)
class Piece:
    """One region a shape occupies, and how the adapter will hold it.

    ``box`` is always where it is. ``faces`` is empty where a rectilinear grid
    holds the shape exactly and the box *is* the shape; where it is not, the
    triangulated boundary is carried as well and the box is that
    triangulation's own extent.

    ``shape`` is the geometry this region *is*, which for a compound decomposed
    into its solids is one of them rather than the whole. It travels with the
    piece because one shape yields as many pieces as it needs, so nothing about
    where a piece sits in a list says what it was read from - and a mesher
    handed the compound instead measures the distance from a shape to itself,
    which is zero and reads as touching. It has no default: a region is always a
    region *of* something, and the field is here to make every place that builds
    one say which.

    What form it is in is not asked here. The :class:`~.model.Solid` a piece
    becomes answers that, and asking it there is what keeps one answer.
    """

    label: str
    box: Box
    shape: Any
    vertices: tuple[Vertex, ...] = ()
    faces: tuple[Triangle, ...] = ()
    #: The axis the piece is flat on, where its triangles cover an area rather
    #: than bound a volume. ``None`` for a box and for a solid.
    sheet_normal: int | None = None
    #: The thickness this region was given because the drawing carried none, in
    #: mm, or zero where the drawing carried its own. See :func:`_thickened`.
    thickened: float = 0.0


#: How fine a triangulation is asked for, as a fraction of the solid's own
#: narrowest extent. Polygonisation error and the grid's staircase are
#: independent and add, so the triangulation wants to be the smaller of the two.
#:
#: It is asked against the shape and not against the cell size, which is the
#: quantity that actually matters, because no grid exists yet when a document is
#: translated - the grid is decided from the geometry this produces. So this is
#: a request rather than a guarantee, and FreeCAD treats it as a hint in any
#: case: what is checked afterwards is how much volume the triangulation lost.
DEFLECTION_OF_EXTENT = 0.1


#: How much of a shape's own measure - the volume a solid's triangles enclose,
#: the area a sheet's cover - the triangulation may lose before the drawing and
#: what gets solved are different objects. A requested deflection is only a hint
#: - FreeCAD returned the same mesh for two different requests - so what the
#: triangulation achieved is measured here rather than assumed.
MAX_SHAPE_LOSS = 0.01


#: How close a shape's own measure has to be to its bounding box's before the
#: box *is* the shape. Relative, so it means the same for a foil and for a
#: waveguide.
BOX_TOLERANCE = 1e-6


#: Below this fraction of the space a shape spans, the figure the kernel
#: computes across its faces is rounding rather than a region - which is what
#: separates a surface from a solid, and is a different question from filling a
#: box however alike the two comparisons look. Both sides of it are measured
#: against the corpus rather than declared here.
NOT_A_REGION = 1e-12


#: How many times the shape is asked for a triangulation before it is refused.
#: The hint is honoured so loosely that a sharpening often returns the same mesh,
#: so there have to be enough attempts to leave that band. Each after the first
#: is four times finer, so the last asks for a 256th of the first.
REFINEMENTS = 5


#: How far the metal built off a surface may depart from that surface's own
#: area times the thickness it was given, either way. Past this what would be
#: solved is a block rather than the drawn skin, and it is refused.
#:
#: **It is a statement about curvature, read off the result.** Offsetting a
#: surface outward by ``t`` sweeps ``area * t`` plus a term in the surface's mean
#: curvature times ``t^2``, so the departure from that product is about ``t`` over
#: the radius the surface curves through - and where the surface folds back on
#: itself the offset is trimmed and the departure runs the other way by the same
#: measure. Bounding it therefore bounds the thickness as a share of the shape's
#: own radius, which is the quantity that decides whether a skin is still a skin,
#: without this having to find that radius.
#:
#: Measuring the solid rather than testing the surface first is what makes it
#: sound. Every cheap proxy for "how wide is this surface" fails on some drawing
#: the corpus holds: the smallest side of a bounding box is as thin as the sheet
#: is tilted, and the root of the area calls a pipe wall as wide as its length
#: and a long ribbon as wide as its diagonal.
MOST_OF_A_SKIN = 0.2


#: ``BRepOffsetAPI_MakeOffsetShape``'s join mode where two offset faces meet:
#: carry both surfaces on until they cross. The kernel's own spelling, passed
#: through :meth:`Part.Shape.makeOffsetShape`.
INTERSECTING = 2


def _bounds(bound_box: Any) -> Box:
    return Box(
        lower=(float(bound_box.XMin), float(bound_box.YMin), float(bound_box.ZMin)),
        upper=(float(bound_box.XMax), float(bound_box.YMax), float(bound_box.ZMax)),
    )


def _union(boxes: Sequence[Box]) -> Box:
    return Box(
        lower=tuple(min(b.lower[d] for b in boxes) for d in range(DIMENSIONS)),  # type: ignore[arg-type]
        upper=tuple(max(b.upper[d] for b in boxes) for d in range(DIMENSIONS)),  # type: ignore[arg-type]
    )


def solid_boxes(obj: Any, skin: float | None = None) -> list[Piece]:
    """The boxes a whole document object occupies, labelled.

    :param skin: How thick to make a conductor drawn as a surface, in mm, or
        ``None`` to refuse one. Only a conductor gets this: a dielectric's
        thickness is a property of the device and nothing here may invent it.
    """
    shape = getattr(obj, "Shape", None)
    if shape is None:
        raise TranslationError(
            f"{_label(obj)!r} has no shape, so there is nothing to mesh. "
            "Material bindings must reference solid geometry"
        )
    return _boxes_of(shape, _label(obj), skin)


def _boxes_of(shape: Any, label: str, skin: float | None = None) -> list[Piece]:
    """The boxes a shape occupies - after proving a rectilinear grid holds it.

    A bounding box exists for every shape, which is exactly the problem: taking
    one from a cylinder, a fillet or a rotated block produces a box-shaped answer
    to a question that had none, and openEMS would solve it without complaint.
    So the shape's own volume is compared against its bounding box's.

    The test is on the *measurement*, not the type. ``Part::Box`` is the usual
    case but not the only honest one - a boolean result or an extruded
    rectangle that happens to be a box passes, and a ``Part::Box`` rotated 30
    degrees does not. A shape flat in one axis is judged by area instead, so a
    ground plane drawn as a face is a first-class citizen rather than an
    exception.

    Falling short of the box is not the end of it. A **flat** shape whose edges
    are all axis-aligned is held exactly by a rectilinear grid once it is cut
    into rectangles, and a layout out of DXF or the Sketcher is exactly that
    shape - so it is cut here rather than handed back to the user to redraw as
    primitives. The cut is exact, and :func:`_cut_up` proves it by area.

    A shape drawn in several separate regions is decomposed into them first,
    whether those regions are volumes or surfaces. What follows then describes
    one region, which is what makes a clearance between two of them something a
    measurement can find: they arrive as two bodies rather than as one body
    covering both, and a body is never paired with itself.
    """
    solids = list(getattr(shape, "Solids", None) or ())

    if len(solids) > 1:
        # Every face has to belong to one of the solids, or decomposing would
        # emit the solids and drop the rest without saying so. A loose face in
        # among them is a drawn sheet, and losing it is a run that
        # completes and answers about a structure with no ground plane in it.
        loose = len(shape.Faces) - sum(len(solid.Faces) for solid in solids)
        if loose:
            raise TranslationError(
                f"{label!r} holds {len(solids)} solids and {loose} more faces that "
                "belong to none of them, so it is part volume and part surface. "
                "Which of the two a face is meant to be cannot be read from the "
                "drawing. Bind them separately, or fuse them into one shape"
            )
        # Each solid becomes its own primitive rather than one surface drawn over
        # all of them. A polyhedron has to be a manifold, which lumps meeting at
        # a single vertex are not and interpenetrating lumps are not either. The
        # engine needs no surface between them: a point inside more than one is
        # resolved by priority, and these share a material and so a priority,
        # which leaves the answer the same whichever of them claims it.
        return [
            piece
            for number, solid in enumerate(solids, 1)
            for piece in _boxes_of(solid, f"{label}#{number}", skin)
        ]

    if not solids:
        surfaces = _surfaces_of(shape)
        if len(surfaces) > 1:
            return [
                piece
                for number, surface in enumerate(surfaces, 1)
                for piece in _boxes_of(surface, f"{label}#{number}", skin)
            ]

    box = _bounds(shape.BoundBox)
    extents = box.extents
    flat = [dim for dim in range(DIMENSIONS) if box.is_flat(dim)]

    if len(flat) > 1:
        raise TranslationError(
            f"{label!r} is flat in "
            f"{' and '.join(AXIS_NAMES[d] for d in flat)}, so it is a line or a "
            "point. A region needs area"
        )

    # Area or volume is the shape's own topology, not the thinness of its
    # bounding box. A solid rolled down to a foil is still a solid, and its area
    # counts both of its faces, so measuring one against a single face's extent
    # can never agree however thin it gets.
    if solids:
        # A solid wound inside out is not the shape it looks like. The kernel
        # reads it as the complement - every point in its bounding box is
        # outside it and every point in the rest of space is inside - so what it
        # describes is unbounded and no primitive can hold it. Refused rather
        # than straightened, because the two readings differ everywhere and the
        # drawing does not say which was meant.
        if float(shape.Volume) < 0.0:
            raise TranslationError(
                f"{label!r} is wound inside out: its faces point inward, so it "
                "describes everything *except* the space it appears to occupy, "
                f"and its volume reads {float(shape.Volume):g}. Nothing bounded "
                "can be built from that. Rebuild it from a shape that is not "
                "reversed - a validity check will not report this, since the "
                "surface itself is sound and only its sense is turned around"
            )
        filled = extents[0] * extents[1] * extents[2]
        if filled > 0 and math.isclose(float(shape.Volume), filled, rel_tol=BOX_TOLERANCE):
            return [Piece(label, box, shape)]
        # No thickness offered: this already has one. A solid whose surface will
        # not close is a fault in the drawing and is refused as such.
        return _triangulate(shape, label, box)

    # No volume, so this is an area. It is held exactly when it fills its own
    # bounding rectangle, and cut into rectangles when it does not.
    if flat:
        area = float(shape.Area)
        expected = math.prod(extents[d] for d in range(DIMENSIONS) if d not in flat)
        if expected > 0 and math.isclose(area, expected, rel_tol=BOX_TOLERANCE):
            return [Piece(label, box, shape)]
        return _cut_up(shape, label, box, flat[0], area)

    return _triangulate(shape, label, box, skin)


def _surfaces_of(shape: Any) -> list[Any]:
    """The separate surfaces a shape carrying no volume is drawn in.

    A shell is one surface however many faces went into it - two rectangles
    fused along an edge come back as a shell of two, and so does a face that
    has been through a file - so splitting on ``Faces`` would take an unclosed
    surface apart into pieces that are each held exactly, where the whole of it
    is a surface openEMS finds no point inside and this layer refuses. The
    regions are therefore the shells, plus every face belonging to none of them:
    a face left loose is one that was drawn, and dropping it is a run that
    completes with a conductor missing.

    Membership is by identity rather than by geometry, so two faces that occupy
    the same space are still two. The hash groups them and
    :meth:`~Part.Shape.isSame` decides, because a hash is a hash and two
    distinct faces may share one.
    """
    shells = list(getattr(shape, "Shells", None) or ())
    held: dict[int, list[Any]] = {}
    for shell in shells:
        for face in shell.Faces:
            held.setdefault(face.hashCode(), []).append(face)
    return shells + [
        face
        for face in getattr(shape, "Faces", None) or ()
        if not any(face.isSame(other) for other in held.get(face.hashCode(), ()))
    ]


def _triangulate_sheet(shape: Any, label: str, box: Box, flat: int) -> list[Piece]:
    """A flat outline that is not made of rectangles, as the triangles it covers.

    Triangles rather than the outline itself, because a contour cannot state a
    hole and an outline routinely has one - the counter of a letter, a clearance
    in a ground plane, an annulus. The kernel's own triangulation of the face has
    the holes already left out, so what is emitted is the area, not the boundary.
    """
    extents = [extent for dim, extent in enumerate(box.extents) if dim != flat]
    drawn = float(shape.Area)
    vertices, faces, covered = _tessellated(
        shape,
        DEFLECTION_OF_EXTENT * min(e for e in extents if e > 0.0),
        drawn,
        lambda points, triangles: covered_area(points, triangles, flat),
    )

    if not faces:
        raise TranslationError(
            f"{label!r} is a flat sheet that triangulates to nothing at all, so "
            "there is no area to model. Check the shape for a face the kernel "
            "could not read"
        )
    if _lost_too_much(covered, drawn):
        raise _no_longer_the_shape(label, "an area", covered, drawn)
    # Collapsed onto the plane the shape was judged flat at, the way the
    # rectangle path collapses it. A drawing is flat to within a tolerance, so a
    # triangulation of one spans a hair on that axis, and a sheet is modelled at
    # exactly one plane - carrying the hair through would offer the engine a
    # thickness it cannot use and nothing drew.
    span = _extent_of(vertices)
    elevation = box.lower[flat]
    bounds = [list(span.lower), list(span.upper)]
    bounds[0][flat] = bounds[1][flat] = elevation
    vertices = tuple(
        tuple(elevation if dim == flat else value for dim, value in enumerate(point))
        for point in vertices
    )
    return [
        Piece(
            label,
            Box(lower=tuple(bounds[0]), upper=tuple(bounds[1])),
            shape,
            vertices,
            faces,
            flat,
        )
    ]


def _tessellated(
    shape: Any,
    opening: float,
    drawn: float,
    measure: Callable[[Sequence[Vertex], Sequence[Triangle]], float],
) -> tuple[tuple[Vertex, ...], tuple[Triangle, ...], float]:
    """A triangulation of ``shape``, refined until it still measures as ``drawn``.

    Taken from a **copy** of the shape, which is what makes the answer depend on
    the drawing alone. A shape that has been displayed carries the triangulation
    its view provider drew with, and ``tessellate`` hands that back rather than
    building one - so without the copy the solved geometry follows the *Deviation*
    view property, and the same file gives different objects to two people whose
    display settings differ. A copy carries no such mesh.

    The fineness asked for is only a hint - several requests an order apart
    return the identical mesh - so what was achieved is measured against the
    shape's own figure and the request sharpened until it is met. ``measure`` is
    what the caller compares by, and it is the only thing that differs between a
    solid, judged by the volume its triangles enclose, and a flat sheet, judged
    by the area they cover.

    The measure achieved comes back with the triangles. Whether it is close
    enough is the caller's to say, because only the caller can name what was
    lost.
    """
    detached = shape.copy()
    fineness = opening
    for _ in range(REFINEMENTS):
        points, facets = detached.tessellate(fineness)
        vertices = tuple((float(p.x), float(p.y), float(p.z)) for p in points)
        faces = tuple(tuple(int(i) for i in facet) for facet in facets)
        achieved = measure(vertices, faces)
        if not _lost_too_much(achieved, drawn):
            break
        # Sharpened by a factor rather than a step, because the hint is honoured
        # so loosely that neighbouring requests return the same mesh.
        fineness /= 4.0
    return vertices, faces, achieved


def _lost_too_much(achieved: float, drawn: float) -> bool:
    """Whether a triangulation has stopped being the shape it was taken from.

    A shape the kernel measures as nothing has no figure to be held to, and its
    triangulation is judged by the checks on its topology instead.
    """
    return drawn > 0.0 and abs(achieved - drawn) > MAX_SHAPE_LOSS * drawn


def _encloses_nothing(shape: Any, box: Box) -> bool:
    """Whether the figure the kernel computes across this shape is a region.

    A surface with a boundary reports one all the same, and for a flat one that
    is rounding either side of zero - so the question is the size of the figure
    against the space the shape spans, and never its sign.
    """
    return abs(float(shape.Volume)) <= NOT_A_REGION * math.prod(box.extents)


def _no_longer_the_shape(
    label: str, measure: str, achieved: float, drawn: float
) -> TranslationError:
    return TranslationError(
        f"{label!r} triangulates to {measure} of {achieved:.6g} against a drawn "
        f"{drawn:.6g}, losing {abs(achieved - drawn) / drawn:.1%} of the shape, and "
        f"{REFINEMENTS - 1} sharper requests did not close the gap. What would be "
        "solved here is a different object from what was drawn. Check the shape "
        "for a self-intersection or a face the kernel could not triangulate"
    )


def _triangulate(shape: Any, label: str, box: Box, skin: float | None = None) -> list[Piece]:
    """A shape that is not a box, as the triangles bounding it.

    The engine holds this exactly - it answers "is this point inside?" against
    the triangles themselves - so nothing about the *shape* is approximated by
    being sent this way. What stays rectilinear is the grid, and the staircase
    is therefore the grid's, where it can be sized against the criterion and
    priced, rather than being introduced here by squaring the drawing off.

    The triangulation is then measured rather than trusted. It has to be closed,
    because an open one is read as a sheet that contains no point and takes the
    object out of the simulation; and it has to still enclose what the shape
    does.

    An open one is where a conductor drawn as a skin arrives, and ``skin`` is
    what it is given: the surface is offset into a solid and that solid is
    meshed. Tried only once the surface has been found open, so a closed one -
    a sphere, a torus, a capped shell - keeps the route it already had
    and pays nothing for a case it is not.
    """
    drawn = float(shape.Volume)
    opening = DEFLECTION_OF_EXTENT * min(e for e in box.extents if e > 0.0)
    # Where a thickness is on offer, the first question is only whether the
    # surface closes, and one triangulation answers it. Refining is aimed at the
    # shape's own volume, which an open surface has not got - the figure the
    # kernel computes across one is not a region - so every attempt would be
    # spent chasing it and the last would ask for a 256th of the deflection.
    vertices, faces, enclosed = _tessellated(
        shape, opening, 0.0 if skin is not None else drawn, enclosed_volume
    )

    fault = surface_fault(vertices, faces)
    if fault is None and skin is not None:
        # It closed, so it bounds a region after all and is meshed as one. That
        # wants the refinement the cheap pass above skipped.
        vertices, faces, enclosed = _tessellated(shape, opening, drawn, enclosed_volume)

    if fault is not None:
        if skin is not None:
            # Reached only from the one call that carries a thickness, which is
            # a conductor drawn as a shape holding no solid at all. A drawing
            # that meant a solid and failed to close reaches the same place and
            # cannot be told from a skin by its geometry - what separates them
            # is whether the offset below builds, and a fault severe enough to
            # leave the surface unusable does not survive it.
            built = _thickened(shape, label, skin)
            try:
                return [replace(piece, thickened=skin) for piece in _boxes_of(built, label)]
            except TranslationError as error:
                # The solid that failed is the one made here, so the drawing is
                # the wrong place to send a reader looking.
                raise TranslationError(
                    f"{label!r} is drawn as a surface and the solid built off it "
                    f"{skin:.4g} mm thick cannot be meshed: {error}"
                ) from error
        if _encloses_nothing(shape, box):
            raise TranslationError(
                f"{label!r} encloses no volume, so it is an area rather than a "
                "region. An area is modelled only where it is flat on one of the "
                "three axes, which this is not: openEMS lays a zero-thickness "
                "conductor as a polygon at one elevation, and there is no "
                "elevation to lay a tilted or a curved one at. Give this "
                "thickness, or bind the material to the solid the face belongs to"
            )
        raise TranslationError(
            f"{label!r} cannot be sent to openEMS as a solid: {fault}. This is "
            "the shape's own surface, so the fault is in the geometry rather "
            "than in the mesh taken from it - a shell left open, two lumps "
            "touching at a point, or faces the kernel could not knit. Run Part "
            "> Check geometry on it"
        )

    if _lost_too_much(enclosed, drawn):
        raise _no_longer_the_shape(label, "a volume", enclosed, drawn)

    return [Piece(label, _extent_of(vertices), shape, vertices, faces)]


def _thickened(shape: Any, label: str, skin: float) -> Any:
    """A conductor drawn as a surface, as the solid that surface bounds one side of.

    The field inside a conductor is zero, so what a metal skin and a metal slab
    behind it do to the problem is the same thing - which is why a thickness the
    drawing never stated can be supplied at all. What it may not do is move the
    surface that *was* drawn, so the offset runs one way only and the drawn face
    stays exactly where it is, as one wall of the result.

    ``skin`` is the thickness whose cross-section demand is the cell the metal is
    already meshed at, so the metal is thick enough that no grid can sample it
    into islands and never asks for anything finer than the model's own metal
    resolution. It is not free: a cross-section is read at points across the
    surface, so a wide skin holds the field down over its whole extent where a
    conductor drawn thicker than the mesher's reach would be read as no feature
    at all.

    Whether the thickness still suits the shape is asked twice, both against
    :data:`MOST_OF_A_SKIN`, because one question cannot reach both ways a skin
    stops being one. Across the surface it is a width, and that is answered
    before offsetting; through the surface it is a radius, and that is read off
    the solid afterwards - a flat sheet sweeps exactly its area times the
    thickness however thick it gets, so nothing about the result would say.
    """
    area = float(shape.Area)
    perimeter = float(shape.Length)
    if area <= 0.0 or perimeter <= 0.0:
        raise TranslationError(
            f"{label!r} is drawn as a surface and the kernel gives it no area to "
            "sweep, so there is no sheet here to give a thickness to. Check the "
            "shape for a face that never closed"
        )

    # Twice the area over the length of the boundary: the width of the strip
    # holding that much area inside that much rim. It is a ribbon's own width,
    # and a disc's radius. The root of the area is not it - that reads a long
    # ribbon as wide as its diagonal, and admits a thickness many times what the
    # ribbon has across it.
    width = 2.0 * area / perimeter
    if skin > MOST_OF_A_SKIN * width:
        raise TranslationError(
            f"{label!r} is drawn as a surface, so it carries no thickness, and "
            f"the {skin:.4g} mm it would be given is more than "
            f"{MOST_OF_A_SKIN:.0%} of the {width:.4g} mm it measures across. What "
            "would be solved is a bar rather than the sheet that was drawn. That "
            "thickness follows the cell the metal is meshed at, so either raise "
            "ElementsPerWavelength or EdgeRefinement until a cell is small "
            "against this shape, or draw the conductor as a solid whose thickness "
            "you chose"
        )

    args = (skin, portbox.KERNEL_TOLERANCE)
    # Where the offset of one face runs into the offset of its neighbour, take
    # where they cross. The alternatives round the corner off or build a patch
    # between the two, and both give up on a shell creased sharply enough - which
    # is a bent guide wall, or any conductor followed round a corner, and not an
    # exotic drawing. ``inter`` lets the same crossing resolve between faces that
    # are not neighbours.
    options = {"inter": True, "join": INTERSECTING, "fill": True}
    try:
        solid = shape.makeOffsetShape(*args, **options)
    except (AttributeError, TypeError):
        # Not a statement about the drawing: the kernel has no such method, or
        # not with these arguments. Left to travel as itself rather than being
        # reported as a shape to go and fix.
        raise
    except Exception as error:  # noqa: BLE001 - every kernel fault reads the same here
        raise TranslationError(
            f"{label!r} is drawn as a surface and the kernel would not offset it "
            f"into a solid {skin:.4g} mm thick: {error}. A surface can be given "
            "thickness only where it stays clear of itself over that distance, "
            "so a fold or a bend tighter than this defeats it. Draw the "
            "conductor as a solid, or run Part > Check geometry to find a "
            "surface that was never sound"
        ) from error

    if not getattr(solid, "Solids", None):
        raise TranslationError(
            f"{label!r} is drawn as a surface and offsetting it {skin:.4g} mm "
            "produced no solid. Draw the conductor as a solid instead"
        )

    departure = abs(float(solid.Volume) / (area * skin) - 1.0)
    if departure > MOST_OF_A_SKIN:
        raise TranslationError(
            f"{label!r} is drawn as a surface, so it carries no thickness, and "
            f"the {skin:.4g} mm it would be given does not stay thin against the "
            f"way it curves: the metal built off it comes to {departure:.0%} away "
            f"from its area times that thickness, against {MOST_OF_A_SKIN:.0%} "
            "allowed, which is that thickness measured against the radius this "
            "shape turns through. What would be solved is a block rather than the "
            "sheet that was drawn. The thickness follows the cell the metal is "
            "meshed at, so either raise ElementsPerWavelength or EdgeRefinement "
            "until a cell is small against this shape, or draw the conductor as a "
            "solid whose thickness you chose"
        )
    return solid


def _extent_of(vertices: Sequence[Vertex]) -> Box:
    """The triangulation's own bounds, which are what the adapter will emit.

    Not the shape's ``BoundBox``, which bounds the exact surface and so lies
    marginally outside a triangulation that chords across every curve. The
    domain and the absorber's reservation are measured against what is sent.
    """
    return Box(
        lower=tuple(min(v[d] for v in vertices) for d in range(DIMENSIONS)),  # type: ignore[arg-type]
        upper=tuple(max(v[d] for v in vertices) for d in range(DIMENSIONS)),  # type: ignore[arg-type]
    )


def _corner(point: Any, dim: int) -> float:
    return float((point.x, point.y, point.z)[dim])


def _cut_up(shape: Any, label: str, box: Box, flat: int, area: float) -> list[Piece]:
    """A flat Manhattan shape, as the rectangles it is made of.

    The outline is taken as loose edges rather than as ordered rings - every
    edge of every wire of every face, in whatever order the kernel holds them.
    A shell of coplanar faces therefore works without being a special case, and
    so does a hole, which is what a clearance in a ground plane is.

    Per *face wire*, and not ``Shape.Edges``, which is the difference between
    cutting a fused L and refusing one. Where two faces meet, the kernel holds
    the shared edge once but each face's wire runs along it, so walking the
    wires yields it twice - and twice is what makes it cancel. Even-odd counts
    an interior edge as a boundary otherwise, and the seam of a fused shape is
    exactly that: an edge with metal on both sides.

    Straightness is checked by **measuring** the edge against its own chord, not
    by asking its type. A semicircle from (0, 0) to (10, 0) has axis-aligned
    endpoints, and a check that looked only at the corners would swallow it and
    return a rectangle - which is precisely the silent staircase this module
    exists to refuse.
    """
    axes = [dim for dim in range(DIMENSIONS) if dim != flat]
    edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    curved = 0
    faces = list(getattr(shape, "Faces", ()) or ())
    walked = [edge for face in faces for wire in face.Wires for edge in wire.Edges]
    for edge in walked or list(shape.Edges):
        points = list(edge.Vertexes)
        if len(points) != 2:
            curved += 1
            continue
        start, end = (vertex.Point for vertex in points)
        chord = math.dist(
            [_corner(start, d) for d in range(DIMENSIONS)],
            [_corner(end, d) for d in range(DIMENSIONS)],
        )
        if abs(float(edge.Length) - chord) > FLATNESS:
            curved += 1
            continue
        edges.append(
            (
                (_corner(start, axes[0]), _corner(start, axes[1])),
                (_corner(end, axes[0]), _corner(end, axes[1])),
            )
        )

    try:
        if curved:
            raise RectilinearError(f"{curved} of its edges are curved")
        cut = rectangles(edges, tolerance=FLATNESS)
    except RectilinearError:
        # Not axis-aligned, so it cannot be cut into rectangles - but a sheet is
        # still an area, and openEMS takes an area as flat polygons. Cutting
        # into rectangles stays the first choice where it applies: it is exact
        # by construction and leaves the grid the fewest planes to hold.
        return _triangulate_sheet(shape, label, box, flat)

    elevation = box.lower[flat]
    boxes: list[Box] = []
    for lower_u, lower_v, upper_u, upper_v in cut:
        lower, upper = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        lower[flat] = upper[flat] = elevation
        lower[axes[0]], upper[axes[0]] = lower_u, upper_u
        lower[axes[1]], upper[axes[1]] = lower_v, upper_v
        boxes.append(Box(lower=tuple(lower), upper=tuple(upper)))  # type: ignore[arg-type]

    # What was cut has to add up to what was drawn. This is the whole guarantee
    # that the cut is exact rather than a fit: it catches an island dropped, a
    # hole not seen, a ring counted twice, and an even-odd inversion. A
    # refusal and not an assert - `assert` vanishes under -O, and what this
    # guards is the user's shape being something the kernel described
    # differently than it was read.
    laid = sum(
        (piece.upper[axes[0]] - piece.lower[axes[0]])
        * (piece.upper[axes[1]] - piece.lower[axes[1]])
        for piece in boxes
    )
    if not boxes or not math.isclose(laid, area, rel_tol=1e-6):
        # Which way it misses says whose fault it is. Short means the outline
        # encloses less than the shape claims, and pieces laid over each other
        # do exactly that - each reports its own area while the boundary counts
        # the overlap once. Over is the case with no drawing that explains it.
        if laid < area:
            why = (
                "It is short, and pieces laid over one another do that: each "
                "reports its own area while the boundary they share encloses "
                "the overlap once. Fuse them, or draw them apart"
            )
        else:
            why = (
                "It is over, which no drawing accounts for, so this is a fault "
                "in reading the shape rather than anything you drew"
            )
        raise TranslationError(
            f"{label!r} was cut into {len(boxes)} rectangles covering "
            f"{laid:.6g}, but the shape's own area is {area:.6g}. The cut is "
            f"meant to be exact. {why}"
        )

    # Every rectangle carries the shape it was cut from, which is the whole of
    # what they tile. Nothing is lost by that: each of them is a box, so the
    # grid pins its own faces and lines bound any clearance between them on
    # both sides, where a shape the grid cannot pin is one a measurement has to
    # find.
    labels = (
        [label] if len(boxes) == 1 else [f"{label}#{number}" for number in range(1, len(boxes) + 1)]
    )
    return [Piece(name, piece, shape) for name, piece in zip(labels, boxes)]


def _shape_of(obj: Any, sub: str) -> Any:
    """The shape a binding names - the whole solid, or one sub-element of it."""
    shape = getattr(obj, "Shape", None)
    if shape is None or not sub:
        return shape
    return shape.getElement(sub)


def _elements_named(reference: Any) -> list[str]:
    """The sub-element names a reference carries, or ``[""]`` for the solid."""
    return picks.named(reference)


def _reference_boxes(reference: Any, skin: float | None = None) -> tuple[Any, list[Piece]]:
    """Every region one binding reference names.

    A binding may name a whole solid, or particular faces of one. The faces are
    not a detail to drop: binding copper to the top face of a substrate is how a
    ground plane gets drawn, and taking the owning solid's box instead turns
    that sheet into a block filling the entire board. It meshes, it solves, and
    the answer is for a different structure.

    One region per named element rather than their union, because the union of
    two faces on different planes is a box that is neither of them.

    Which element a region came from is not returned beside it, because each
    :class:`Piece` carries the geometry it is - and one element yields as many
    pieces as it needs, so a name paired by position would land on whichever
    piece shared its index.
    """
    obj, sub = reference if isinstance(reference, tuple) else (reference, None)
    names = [sub] if isinstance(sub, str) else list(sub or ())
    names = [name for name in names if name]

    if not names:
        return obj, solid_boxes(obj, skin)

    shape = getattr(obj, "Shape", None)
    if shape is None:
        raise TranslationError(f"{_label(obj)!r} has no shape, so there is nothing to mesh")
    regions = []
    for name in names:
        label = f"{_label(obj)}:{name}"
        regions.extend(_boxes_of(shape.getElement(name), label, skin))
    return obj, regions


def _refinement_boxes(reference: Any, region: str) -> list[tuple[str, Box]]:
    """Every box one ``EMMeshRegion`` reference names, as ``(label, box)``.

    Bounding boxes, and deliberately **not** proved to be boxes the way
    :func:`solid_boxes` proves a material region is one. Refining around a
    cylinder or a fillet is a perfectly sensible thing to want, and the
    rectilinear answer to it is the box it sits in. That is a surprise worth
    naming rather than a refusal - so the property tooltip says "bounding
    box", and this does not complain.

    One box per named element rather than their union, for the reason
    :func:`_reference_boxes` gives: the union of two faces on different planes
    is a box that is neither of them.
    """
    obj = reference[0] if isinstance(reference, tuple) else reference
    names = [name for name in _elements_named(reference) if name]

    shape = getattr(obj, "Shape", None)
    if shape is None:
        raise TranslationError(
            f"{region!r} references {_label(obj)!r}, which has no shape, so "
            "there is nothing to refine around"
        )
    if not names:
        return [(_label(obj), _bounds(shape.BoundBox))]
    return [(f"{_label(obj)}:{name}", _bounds(shape.getElement(name).BoundBox)) for name in names]


#: What a mesh region's ``Mode`` may say. ``Objects.EMMeshRegion`` offers the
#: same words; the document layer cannot import it to ask.
REFINE, COARSEN = "Refine", "Coarsen"


def _mode(obj: Any) -> str:
    """Which way a mesh region points, for a document that may predate the ask.

    Refused rather than defaulted where the word is not one of the two. FreeCAD
    stores an enumeration's whole list in the document and restores it from
    there rather than from the class, so a file can go on offering a word this
    adapter has no behaviour for - and the reading it would fall back to
    silently is the one that leaves the region doing the opposite.
    """
    mode = str(getattr(obj, "Mode", REFINE) or REFINE)
    if mode not in (REFINE, COARSEN):
        raise TranslationError(
            f"{_label(obj)!r}: Mode is {mode!r}. A mesh region either makes the "
            f"elements around its geometry finer ({REFINE}) or lets them go "
            f"coarser ({COARSEN})"
        )
    return mode


def _across_of(obj: Any) -> int:
    """How many elements a mesh region wants across itself."""
    across = int(obj.MinElementsAcross)
    if across < 0:
        raise TranslationError(
            f"{_label(obj)!r}: MinElementsAcross is {across}; it must be 0 "
            "to inherit the global count, or a positive number"
        )
    return across


def _size_of(obj: Any) -> float:
    """The element size a mesh region asks for, whichever way it points."""
    size = _value(obj.ElementSize)
    if size <= 0:
        raise TranslationError(
            f"{_label(obj)!r} has no element size set, so it asks for "
            "nothing. Set ElementSize, or uncheck Enabled"
        )
    return size


def _targets(obj: Any, verb: str) -> list[Any]:
    """What a mesh region is aimed at, refused when it is aimed at nothing."""
    references = list(getattr(obj, "References", ()) or ())
    if not references:
        raise TranslationError(
            f"{_label(obj)!r} {verb} nothing. Select the geometry it applies to, or delete it"
        )
    return references


def _subject(obj: Any, element: str) -> tuple[str, str]:
    """What a coarsening and a material binding have to agree on to be the same.

    ``Name`` and not the label: FreeCAD keeps the name unique within a document
    and leaves the label unique only by convention, so two objects a user has
    given one label are still two things to relax separately.

    The sub-element belongs in the key. A conductor is routinely a *face* of the
    board it sits on, so the object alone does not identify what was drawn -
    relaxing by object would let a coarsened substrate take the ground plane
    down with it, which is exactly the reach this feature is built not to have.
    """
    return (getattr(obj, "Name", "") or _label(obj), element)


@dataclass(frozen=True)
class _Relaxation:
    """The size some geometry settles for, and who asked."""

    size: float
    asked_by: str


def _relaxations(refinements: Sequence[Any]) -> dict[tuple[str, str], _Relaxation]:
    """Which drawn geometry stops driving the grid, and the size it settles for.

    A coarsening names *geometry*, and that is the whole reason it is carried
    this way rather than as a box like a refinement. The grid is separable, so a
    box spends itself on a slab through the model on each axis: a refinement
    that overshoots hands out cells nobody asked for, and the opposite would
    take them off whatever else lies level with the box, anywhere in the model.

    Keyed by :func:`_subject`, so what it reaches is what a material binding
    would have had to name to be the same thing.

    Two coarsenings over one subject leave it at the finer of the two, which is
    the same way round as two refinements: the grid ends up as fine as the
    finest thing asked for, whichever direction asked.
    """
    out: dict[tuple[str, str], _Relaxation] = {}

    for obj in refinements:
        if not bool(getattr(obj, "Enabled", True)) or _mode(obj) != COARSEN:
            continue

        size = _size_of(obj)
        across = _across_of(obj)
        if across:
            raise TranslationError(
                f"{_label(obj)!r} is set to {COARSEN} and asks for {across} "
                "elements across. A count is a demand for resolution, so it "
                "would be honoured while the coarsening beside it was not. Set "
                f"MinElementsAcross to 0, or set Mode to {REFINE}"
            )

        for reference in _targets(obj, "coarsens"):
            target = reference[0] if isinstance(reference, tuple) else reference
            for element in _elements_named(reference):
                key = _subject(target, element)
                if key not in out or size < out[key].size:
                    out[key] = _Relaxation(size, _label(obj))

    return out


def _check_relaxations_were_used(
    relaxed: Mapping[tuple[str, str], _Relaxation], used: Set[tuple[str, str]]
) -> None:
    """Refuse a coarsening aimed at geometry the mesher never sizes.

    Only a material binding makes geometry something the grid has a size for, so
    a coarsening aimed anywhere else changes nothing at all. Refusing rather
    than passing over it: the region is a typed request, and one that is quietly
    dropped leaves the property editor showing a size that was never applied -
    which is the failure this whole object exists to make impossible.
    """
    for (name, element), relaxation in relaxed.items():
        if (name, element) in used:
            continue
        drawn = f"{name}:{element}" if element else name
        raise TranslationError(
            f"{relaxation.asked_by!r} coarsens {drawn!r}, which no material "
            "binding names. Only geometry a material has been bound to has an "
            "element size to settle for, so this asks for nothing. Bind a "
            "material to it, or aim the region at geometry that has one"
        )


def _sizing_regions(refinements: Sequence[Any]) -> tuple[SizingRegion, ...]:
    """``EMMeshRegion`` objects as the mesher's local refinement input.

    Refinements only. A region set to coarsen is attached to the geometry it
    names instead - see :func:`_relaxations` - and contributes no box.

    Nothing here compares the requested size against the global one. That check
    needs :class:`MeshParams`, and it lives in the mesher, where every route -
    envelope, preview, and a caller building a mesh by hand - passes through
    it exactly once.
    """
    out: list[SizingRegion] = []

    for obj in refinements:
        if not bool(getattr(obj, "Enabled", True)) or _mode(obj) == COARSEN:
            continue

        size = _size_of(obj)
        across = _across_of(obj)

        for reference in _targets(obj, "refines"):
            for label, box in _refinement_boxes(reference, _label(obj)):
                out.append(
                    SizingRegion(
                        lower=box.lower,
                        upper=box.upper,
                        size=size,
                        min_lines=across,
                        label=f"{_label(obj)} on {label}",
                    )
                )

    return tuple(out)


def _sub_box(link: Any, subject: str) -> Box:
    """A ``LinkSub`` - ``(object, ['Face3'])`` - as a box.

    Unlike :func:`solid_boxes` this does not demand the selection be a box: a face
    of a legitimately box-shaped solid is planar by construction, and callers
    check the flatness that matters to them along the axis it matters on.
    """
    if not link:
        raise TranslationError(f"{subject} is unset; select the face it should be built from")

    obj, sub = link
    names = [sub] if isinstance(sub, str) else list(sub or ())
    names = [name for name in names if name]
    shape = obj.Shape

    if not names:
        return _bounds(shape.BoundBox)
    return _union([_bounds(shape.getElement(name).BoundBox) for name in names])
