# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A drawn shape, as the regions openEMS will hold it in.

openEMS holds a region either as an axis-aligned box or as a closed triangulated
surface. This module decides which of the two a shape is, and it reads that off
the shape's own measure rather than off whatever drew it. A shape that fills its
bounding box is sent as a box. A shape bounded by planes square to the grid is
cut into the boxes it is made of, drawn flat or as a solid. Anything else is
sent as its own triangulation. Being curved, rotated or swept is not by itself a
reason to refuse a shape.

The cut is worth its own machinery for more than cost. A shape held as a box has
its faces pinned onto grid lines, and everything that sizes a cell from a
conductor's width or measures what the grid left of one reads a bounding box.
Those measurements cannot see across a shape sent whole.

Refused here, by name:

* a shape flat in two of the three axes, which is a line or a point and has no
  area to model;
* a shape that is part volume and part surface. Whether a face belonging to no
  solid was meant to be a volume or a surface cannot be read from the drawing;
* a solid wound inside out, which describes everything except the space it
  appears to occupy and so is not bounded;
* a dielectric surface that encloses no volume and lies flat on none of the
  three axes. A layer with no thickness is laid at an elevation, and a curved or
  a tilted one has no elevation. Its thickness is most of what the layer does,
  so nothing here may supply one. A conductor drawn that way is given a
  thickness instead: the field inside metal is zero either way;
* a shape whose own triangulation comes back open, or comes back having lost
  enough of the drawing to be a different object.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass, replace
from typing import Any

from ... import picks, portbox
from .lfs import Curvature, bends_through, curvature
from .model import DIMENSIONS
from .properties import AXIS_NAMES, TranslationError, _label, _value
from .rectilinear import Rect, RectilinearError, rectangles
from .regions import SizingRegion
from .surface import covered_area, enclosed_volume, surface_fault

#: Re-exported rather than redefined. See :data:`Microwave.portbox.FLATNESS` for
#: the number and the reasoning. A second copy here would be a fact that drifts.
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
class Departure:
    """How far the surface openEMS is given stands from the surface drawn.

    A polygonised curve is a chord, and a chord lies off the arc it stands for.
    This is that distance, taken over the part of the surface that curves.
    Refining the triangulation removes it. Nothing about the grid does.

    :param displaced: How far the middle of a facet stands from the surface,
        averaged over the facets by area, in mm. Signed: positive where the
        emitted surface lies inside the drawing, which is where a chord across a
        convex wall lies, and negative where it lies outside, which is where the
        same chord across a bore lies.
    :param turns_both_ways: Whether the shape curves both ways somewhere. In
        that case ``displaced`` may be a net of two opposite displacements and
        understate what either surface moved: a tube's outer wall moves in while
        its bore moves out. The reading is per sample, so one saddle-shaped face
        sets it, and it over-fires in the safe direction.

    Both come out of figures the translation already has: the drawn volume, the
    volume the triangles enclose, and what the shape's faces curve through. No
    facet is measured against the surface. That costs a kernel query apiece, on
    a path that runs at every recompute.
    """

    displaced: float
    turns_both_ways: bool

    def said_of(self, label: str) -> str:
        """This departure as one line about ``label``, or empty where there is none.

        The wording is here rather than at each place that shows it, so the task
        panel, the file written beside an envelope and anything else asking
        report one thing. The line states an average: the facet standing
        furthest from the surface is further than this, by however unevenly the
        drawing is curved.
        """
        if not self.displaced:
            return ""
        if self.turns_both_ways:
            # The line names no side. Where a wall moves in and a bore moves out
            # the two cancel, so which way the remainder points is a property of
            # the cancellation rather than of either surface.
            return (
                f"{label} curves both ways, so what a volume can say about it is "
                f"{abs(self.displaced):.4g} mm - the difference between however "
                "far its surfaces moved, which is a lower bound on each of them "
                "and says nothing about which way either went. On a hollow shape "
                "it is smaller than either wall moved by about the ratio of the "
                "shape to its own wall."
            )
        side = "inside" if self.displaced > 0.0 else "outside"
        return (
            f"{label} is solved {abs(self.displaced):.4g} mm {side} the drawing, "
            "averaged over the part of its surface that curves - a chord cuts "
            "the arc it stands for, and some facets cut deeper than others."
        )


#: What a region held as boxes departs by, which is nothing. Every face of a box
#: is a plane the mesher pins a line to, so there is no chord anywhere on it.
EXACTLY = Departure(displaced=0.0, turns_both_ways=False)


@dataclass(frozen=True)
class Piece:
    """One region a shape occupies, and how the adapter will hold it.

    ``box`` is always where the region is. ``faces`` is empty where a
    rectilinear grid holds the shape exactly and the box is the shape. Where it
    is not, the triangulated boundary is carried as well and the box is that
    triangulation's own extent.

    ``shape`` is the geometry this region is. For a compound decomposed into its
    solids that is one solid rather than the whole. It travels with the piece
    because one shape yields as many pieces as it needs, so where a piece sits
    in a list says nothing about what it was read from. A mesher handed the
    compound instead measures the distance from a shape to itself, which is zero
    and reads as touching. The field has no default: a region is always a region
    of something, and every place that builds one has to say which.

    What form the region is in is not stated here and is not a field: a piece
    with no triangles is a box, and one carrying a sheet normal is an area.
    Everything built from a piece - the :class:`~.model.Solid` the engine gets
    and the body the mesher measures - reads it off those two, so there is one
    answer and nowhere for a second to disagree with it.
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
    #: How far the emitted surface stands from the drawn one. Anything bounded
    #: by planes - a box, a shape cut into boxes, a triangulated solid with no
    #: curved face - departs by nothing and records that. ``None`` where no
    #: distance could be stated: a sheet, whose outline departs in the plane its
    #: triangles cover and this measure is blind there, or a solid whose drawn
    #: volume contradicts which side of its surface the triangulation is on.
    departure: Departure | None = None


#: How fine a triangulation is asked for, as a fraction of the solid's own
#: narrowest extent. Polygonisation error and the grid's staircase are
#: independent and add, so the triangulation is asked to be the smaller of the
#: two.
#:
#: The cell size is the quantity that matters. No grid exists when a document is
#: translated, because the grid is decided from the geometry this produces. The
#: fraction is asked against the shape instead. It is a request rather than a
#: guarantee, and FreeCAD treats it as a hint in any case. What is checked
#: afterwards is how much volume the triangulation lost.
DEFLECTION_OF_EXTENT = 0.1


#: The band a triangulation request is held inside, as shares of the radius
#: being followed: no coarser than :data:`COARSEST_OF_RADIUS` of the sharpest
#: radius on the shape, and no finer than :data:`FINEST_OF_RADIUS` of the
#: widest. They are shares rather than lengths, so a request scaled off a
#: bounding box is brought to the curve it has to follow rather than left
#: against the body's own size.
#:
#: Above the coarse edge the kernel answers each request with a mesh of its own.
#: Between the two edges one mesh comes back whatever is asked. Below the fine
#: edge it responds again and goes on responding. A request held inside the band
#: follows a feature without paying for the fine end.
#:
#: The ceiling makes a small feature come out as itself. A bead on a can, and a
#: rod drawn along a diagonal whose box is its own length on every axis, is
#: asked a deflection many times its own radius, and the kernel answers a coarse
#: request coarsely.
#:
#: The floor stops the ceiling refining a whole body for one feature on it. A
#: blend a fraction of a micron wide drives the request toward zero, and every
#: curved face then subdivides across its whole area for an answer nothing
#: needed.
#:
#: A curved outline answers the ceiling too, so one figure serves a surface and
#: an outline. The floor belongs to the surface reading alone.
#: :func:`_triangulate_sheet` says why a sheet takes the ceiling without it.
#:
#: These are bounds rather than targets. The radii they are stated against are
#: sampled, and so good only to a factor.
COARSEST_OF_RADIUS = 0.5
FINEST_OF_RADIUS = 0.005


#: How much of a shape's own measure - the volume a solid's triangles enclose,
#: the area a sheet's cover - the triangulation may lose before the drawing and
#: what gets solved are different objects. A requested deflection is only a
#: hint: measured under FreeCAD 1.1.1, two different requests come back with the
#: same mesh. So what the triangulation achieved is measured rather than
#: assumed.
MAX_SHAPE_LOSS = 0.01


#: How close a shape's own measure has to be to its bounding box's before the
#: box is the shape. It is relative, so it means the same for a foil and for a
#: waveguide.
BOX_TOLERANCE = 1e-6


#: How closely a cut into rectangles has to add back up to the shape it came
#: from. It is the tolerance above, asked of a different question. Both compare
#: a shape's own measure against one computed from its corners, so a figure of
#: its own here would be a second number to keep in step with the first. It is
#: far tighter than :data:`MAX_SHAPE_LOSS`: that one budgets for an
#: approximation, where this one proves an identity.
CUT_TOLERANCE = BOX_TOLERANCE


#: Below this fraction of the space a shape spans, the figure the kernel
#: computes across its faces is rounding rather than a region. That separates a
#: surface from a solid. It is a different question from filling a box, however
#: alike the two comparisons look. Both sides of it are measured against the
#: corpus rather than declared here.
NOT_A_REGION = 1e-12


#: How many times the shape is asked for a triangulation before it is refused.
#: The hint is honoured so loosely that a sharpening often returns the same mesh,
#: so there have to be enough attempts to leave that band. Each after the first
#: is four times finer, so the last asks for a 256th of the first.
REFINEMENTS = 5


#: How many rungs the ladder gets when it is aimed at a distance rather than
#: at the shape's own measure. The request opens inside the band where one mesh
#: comes back whatever is asked, and that band is wider than
#: :data:`REFINEMENTS` quarterings span, so a ladder of that length would refuse
#: a distance a further rung or two reaches. Volume loss needs no such room: a
#: triangulation that has lost a hundredth of the shape is nowhere near the band
#: and escapes on the first sharpening or not at all.
HELD_REFINEMENTS = 9


#: The most facets a ladder aimed at a distance will produce before it stops
#: asking, whatever it has reached.
#:
#: The structure build bounds a triangulation, not accuracy. openEMS asks its
#: containment question against the triangles at build time and never while
#: stepping, so facets are paid for once. It asks it several times a cell:
#: ``Operator::Calc_EffMatPos`` runs per node and per axis, and the quarter-cell
#: averaging it dispatches to samples the material at each shifted corner
#: (``openEMS/FDTD/operator.cpp:1437-1470``, ``:1556-1568``). This figure is
#: where the ladder stops asking rather than an accuracy anybody chose, and a
#: run that reaches it is refused with the count in the message.
#:
#: The last rung overshoots it. The count is read from a triangulation already
#: built, and a quartering multiplies it by more than four. The figure bounds
#: how far the ladder goes and not what any one rung costs.
MOST_FACETS = 400_000


#: How far the metal built off a surface may depart from that surface's own
#: area times the thickness it was given, either way. Past this what would be
#: solved is a block rather than the drawn skin, and it is refused.
#:
#: It is a statement about curvature, read off the result. Offsetting a surface
#: outward by ``t`` sweeps ``area * t`` plus a term in the surface's mean
#: curvature times ``t^2``, so the departure from that product is about ``t``
#: over the radius the surface curves through. Where the surface folds back on
#: itself the offset is trimmed and the departure runs the other way by the same
#: measure. Bounding the departure therefore bounds the thickness as a share of
#: the shape's own radius, and that share decides whether a skin is still a
#: skin. Nothing here has to find the radius.
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


def solid_boxes(obj: Any, skin: float | None = None, held_to: float = 0.0) -> list[Piece]:
    """The boxes a whole document object occupies, labelled.

    :param skin: How thick to make a conductor drawn as a surface, in mm, or
        ``None`` to refuse one. Only a conductor gets this: a dielectric's
        thickness is a property of the device and nothing here may invent it.
    :param held_to: How far a curved surface may be solved from where it was
        drawn, in mm, or zero to ask for nothing. See :func:`_within`.
    """
    shape = getattr(obj, "Shape", None)
    if shape is None:
        raise TranslationError(
            f"{_label(obj)!r} has no shape, so there is nothing to mesh. "
            "Material bindings must reference solid geometry"
        )
    return _boxes_of(shape, _label(obj), skin, held_to)


def _boxes_of(
    shape: Any, label: str, skin: float | None = None, held_to: float = 0.0
) -> list[Piece]:
    """The boxes a shape occupies, after proving a rectilinear grid holds it.

    Every shape has a bounding box, and that is the problem. Taking one from a
    cylinder, a fillet or a rotated block produces a box-shaped answer to a
    question that had none, and openEMS would solve it without complaint. So the
    shape's own volume is compared against its bounding box's.

    The test is on the measurement rather than on the type. ``Part::Box`` is the
    usual case and not the only honest one: a boolean result or an extruded
    rectangle that happens to be a box passes, and a ``Part::Box`` rotated 30
    degrees does not. A shape flat in one axis is judged by area instead, so a
    ground plane drawn as a face is handled here like any other region.

    Falling short of the box is not the end of it. A shape whose edges are all
    axis-aligned is held exactly by a rectilinear grid once it is cut into
    rectangles, and a layout out of DXF or the Sketcher is that shape. It is cut
    here rather than handed back to the user to redraw as primitives. Drawn
    flat, it is cut where it lies, and :func:`_cut_up` proves the cut by area.
    Drawn as a solid, it is cut into the slabs and rectangles it stands in, and
    :func:`_cut_up_solid` proves that by volume and by surface. The second is
    how a conductor with a thickness reaches this layer at all.

    A shape drawn in several separate regions is decomposed into them first,
    whether those regions are volumes or surfaces. What follows then describes
    one region, and a measurement can find a clearance between two of them: they
    arrive as two bodies rather than as one body covering both, and a body is
    never paired with itself.
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
        # engine needs no surface between them. A point inside more than one is
        # resolved by priority, and these share a material and so a priority, so
        # the answer is the same whichever of them claims it.
        return [
            piece
            for number, solid in enumerate(solids, 1)
            for piece in _boxes_of(solid, f"{label}#{number}", skin, held_to)
        ]

    if not solids:
        surfaces = _surfaces_of(shape)
        if len(surfaces) > 1:
            return [
                piece
                for number, surface in enumerate(surfaces, 1)
                for piece in _boxes_of(surface, f"{label}#{number}", skin, held_to)
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

    # Whether a shape is an area or a volume is its own topology rather than the
    # thinness of its bounding box. A solid rolled down to a foil is still a
    # solid, and its area counts both of its faces, so measuring that area
    # against a single face's extent can never agree however thin it gets.
    if solids:
        # A solid wound inside out is not the shape it looks like. The kernel
        # reads it as the complement - every point in its bounding box is
        # outside it and every point in the rest of space is inside - so what it
        # describes is unbounded and no primitive can hold it. It is refused
        # rather than straightened: the two readings differ everywhere, and the
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
            return [Piece(label, box, shape, departure=EXACTLY)]
        stack = _cut_up_solid(shape, label)
        if stack is not None:
            return stack
        # No thickness is offered here, since a solid already has one. A solid
        # whose surface will not close is a fault in the drawing and is refused
        # as such.
        return _triangulate(shape, label, box, held_to=held_to)

    # No volume, so this is an area. It is held exactly when it fills its own
    # bounding rectangle, and cut into rectangles when it does not.
    if flat:
        area = float(shape.Area)
        expected = math.prod(extents[d] for d in range(DIMENSIONS) if d not in flat)
        if expected > 0 and math.isclose(area, expected, rel_tol=BOX_TOLERANCE):
            return [Piece(label, box, shape, departure=EXACTLY)]
        return _cut_up(shape, label, box, flat[0], area)

    return _triangulate(shape, label, box, skin, held_to)


def _surfaces_of(shape: Any) -> list[Any]:
    """The separate surfaces a shape carrying no volume is drawn in.

    A shell is one surface however many faces went into it. Two rectangles fused
    along an edge come back as a shell of two, and so does a face that has been
    through a file. Splitting on ``Faces`` would take an unclosed surface apart
    into pieces that are each held exactly, where the whole of it is a surface
    openEMS finds no point inside and this layer refuses. The regions are
    therefore the shells, plus every face belonging to none of them. A face left
    loose is one that was drawn, and dropping it is a run that completes with a
    conductor missing.

    Membership is by identity rather than by geometry, so two faces that occupy
    the same space are still two. The hash groups them and
    :meth:`~Part.Shape.isSame` decides, since two distinct faces may share a
    hash.
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

    It emits triangles rather than the outline itself. A contour cannot state a
    hole, and an outline routinely has one: the counter of a letter, a clearance
    in a ground plane, an annulus. The kernel's own triangulation of the face
    has the holes already left out, so what is emitted is the area rather than
    the boundary.
    """
    sides = [extent for dim, extent in enumerate(box.extents) if dim != flat and extent > 0.0]
    # A share of the two extents in the sheet's own plane, held against what the
    # outline bends through rather than what a face curves through. A sheet's
    # face is flat and carries no radius, and everything the kernel is asked to
    # follow here is on the boundary. The extent alone points the wrong way,
    # since a feature is a smaller share of a larger sheet.
    #
    # It is held from above only. A boundary keeps answering a finer request
    # with a nearer polygon, so there is no place to floor it. Floored against
    # the widest curve, a via in a round ground plane would be followed as
    # coarsely as the ground is wide.
    drawn = float(shape.Area)
    vertices, faces, covered = _tessellated(
        shape,
        _no_coarser_than(DEFLECTION_OF_EXTENT * min(sides), bends_through(shape)),
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
    # exactly one plane. Carrying the hair through would offer the engine a
    # thickness it cannot use and nothing drew.
    span = _extent_of(vertices)
    elevation = box.lower[flat]
    bounds = [list(span.lower), list(span.upper)]
    bounds[0][flat] = bounds[1][flat] = elevation
    vertices = tuple(
        portbox.corner(elevation if dim == flat else value for dim, value in enumerate(point))
        for point in vertices
    )
    return [
        Piece(
            label,
            Box(lower=portbox.corner(bounds[0]), upper=portbox.corner(bounds[1])),
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
    close_enough: Callable[[float], bool] | None = None,
) -> tuple[tuple[Vertex, ...], tuple[Triangle, ...], float]:
    """A triangulation of ``shape``, refined until it still measures as ``drawn``.

    The triangulation is taken from a copy of the shape, so the answer depends
    on the drawing alone. A shape that has been displayed carries the
    triangulation its view provider drew with, and ``tessellate`` hands that
    back rather than building one. Without the copy the solved geometry would
    follow the Deviation view property, and the same file would give different
    objects to two people whose display settings differ. A copy carries no such
    mesh.

    The fineness asked for is only a hint - several requests an order apart
    return the identical mesh - so what was achieved is measured against the
    shape's own figure and the request is sharpened until it is met. ``measure``
    is what the caller compares by. It is the only thing that differs between a
    solid, judged by the volume its triangles enclose, and a flat sheet, judged
    by the area they cover.

    The measure achieved comes back with the triangles. Whether it is close
    enough is the caller's to say: only the caller can name what was lost.

    ``close_enough`` is a second stopping rule, opt-in and off wherever it is
    ``None``. The first rule asks whether the triangulation is still the shape
    at all. This one asks whether it is within a distance of the shape that the
    user named. Both have to hold before the ladder stops, so a request that
    satisfies one and not the other keeps sharpening.
    """
    detached = shape.copy()
    fineness = opening
    rungs = REFINEMENTS if close_enough is None else HELD_REFINEMENTS
    for _ in range(rungs):
        points, facets = detached.tessellate(fineness)
        vertices = tuple((float(p.x), float(p.y), float(p.z)) for p in points)
        faces = tuple((int(f[0]), int(f[1]), int(f[2])) for f in facets)
        achieved = measure(vertices, faces)
        if not _lost_too_much(achieved, drawn) and (close_enough is None or close_enough(achieved)):
            break
        if close_enough is not None and len(faces) >= MOST_FACETS:
            # Stopped on the cost rather than on the answer. The caller refuses
            # and says how many facets it took.
            break
        # Sharpened by a factor rather than a step, because the hint is honoured
        # so loosely that neighbouring requests return the same mesh.
        fineness /= 4.0
    return vertices, faces, achieved


def _opening(curved: Curvature, box: Box) -> float:
    """The deflection to open a solid's triangulation at.

    A share of the shape's own narrowest extent, held inside the band the kernel
    answers a request in, against the radii its surfaces curve through. The
    extent and the sharpest radius agree on a sphere and a cylinder, whose
    narrowest extent is twice that radius. They part company on anything
    carrying a small feature on a large body.

    ``curved`` is the caller's own reading of the shape, so the request and the
    distance the ladder aims at are taken off one walk of the surface.
    """
    sides = [e for e in box.extents if e > 0.0]
    return _held_inside_the_band(DEFLECTION_OF_EXTENT * min(sides), curved.through, max(sides))


def _held_inside_the_band(
    asked: float, curved: tuple[float, float] | None, longest: float
) -> float:
    """``asked``, brought inside the band the kernel answers a request in.

    :param asked: The request the drawing's own size implies. It is what comes
        back wherever the shape carries no radius to bound a request against.
    :param curved: The sharpest and the widest radius the shape's surfaces curve
        through, or ``None`` where none of them does.
    :param longest: The longest side of what is being triangulated, which caps
        the widest radius: a radius larger than the body does not bend within
        it.

    This applies both halves of the band, which is a solid's rule. A flat sheet
    takes the ceiling alone, and :func:`_triangulate_sheet` says why.

    The floor stops one small feature refining a whole body. The ceiling is set
    by the sharpest radius, so a blend a fraction of a micron wide would drive
    the request toward zero and triangulate the body it sits on into millions of
    facets, into vertices that collide once rounded to single precision, or into
    a surface that no longer closes. So the request is also held above the fine
    edge taken against the widest radius. Where the two cross, on a shape whose
    radii span more than the band is wide, the floor wins and the sharp feature
    is followed no better than the floor allows.

    The floor never raises a request above what was asked for. It is there to
    stop the ceiling over-refining a body for a feature on it, and a caller
    asking for a fine surface deliberately is not over-refining anything.
    """
    if curved is None:
        return asked
    widest = curved[1]
    return max(
        _no_coarser_than(asked, curved),
        min(asked, FINEST_OF_RADIUS * min(widest, longest)),
    )


def _no_coarser_than(asked: float, curved: tuple[float, float] | None) -> float:
    """``asked``, brought under the coarse edge of that band and no further.

    The ceiling on its own. It is the whole rule for a flat sheet and half of it
    for a solid.

    :param curved: The sharpest and the widest radius the shape bends through,
        or ``None`` where nothing bends. The request is then what was asked: a
        shape bounded by planes and straight runs is followed exactly whatever
        is asked. A solid states this from its surfaces and a sheet from its
        outline, and the coarse edge is the same either way.
    """
    if curved is None:
        return asked
    sharpest = curved[0]
    return min(asked, COARSEST_OF_RADIUS * sharpest)


def _lost_too_much(achieved: float, drawn: float) -> bool:
    """Whether a triangulation has stopped being the shape it was taken from.

    A shape the kernel measures as nothing has no figure to be held to, and its
    triangulation is judged by the checks on its topology instead.
    """
    return drawn > 0.0 and abs(achieved - drawn) > MAX_SHAPE_LOSS * drawn


def _encloses_nothing(shape: Any, box: Box) -> bool:
    """Whether the figure the kernel computes across this shape is a region.

    A surface with a boundary reports one all the same, and for a flat one that
    figure is rounding either side of zero. So the question is the size of the
    figure against the space the shape spans, never its sign.
    """
    return abs(float(shape.Volume)) <= NOT_A_REGION * math.prod(box.extents)


def _no_longer_the_shape(
    label: str, measure: str, achieved: float, drawn: float
) -> TranslationError:
    return TranslationError(
        f"{label!r} triangulates to {measure} of {achieved:.6g} against a drawn "
        f"{drawn:.6g}, losing {abs(achieved - drawn) / drawn:.1%} of the shape, and "
        "sharper requests did not close the gap. What would be solved here is a "
        "different object from what was drawn. Check the shape for a "
        "self-intersection or a face the kernel could not triangulate"
    )


def _departure(curved: Curvature, drawn: float, enclosed: float) -> Departure | None:
    """How far a solid's triangulation stands from the solid, without measuring a facet.

    The volume the triangles lose is an integral of the displacement over the
    surface, so dividing it by the area recovers the displacement, and the
    factor is exact rather than a fit. Over one facet the departure is a
    quadratic vanishing at the three vertices, which the surface passes through;
    the mean of such a quadratic over the triangle is ``3/4`` of its value at the
    centroid, whatever the curvature. So the mean displacement is ``4/3`` of the
    lost volume over the area.

    The lost volume is divided by the area that curves rather than by the whole
    area. A flat face is triangulated exactly and contributes nothing to the
    volume, so left in the denominator it only dilutes the result. That is worst
    where a small round feature sits on a large flat body, which is the case the
    figure exists for. A shape with no curved face at all is emitted exactly and
    departs by nothing.

    ``None`` where the two volumes disagree about which side of the surface the
    triangulation is on. A polyhedron whose vertices lie on a surface bending
    only one way is on one side of it - inside a convex wall, outside a bore -
    so a difference of the other sign is not a departure. The drawn volume is
    then not the shape's, because the kernel integrated a surface it re-fits
    rather than evaluates. A distance with the wrong sign would say the surface
    is outside the drawing when it is inside.
    """
    if curved.area <= 0.0:
        return EXACTLY
    lost = drawn - enclosed
    convex_only = curved.convex and not curved.concave
    concave_only = curved.concave and not curved.convex
    if (convex_only and lost < 0.0) or (concave_only and lost > 0.0):
        return None
    return Departure(
        displaced=(4.0 / 3.0) * lost / curved.area,
        turns_both_ways=curved.both_ways,
    )


def _within(curved: Curvature, drawn: float, held_to: float) -> Callable[[float], bool] | None:
    """Whether a triangulation stands within ``held_to`` mm of the shape, as a test.

    ``None`` where nothing was asked for, and where nothing curves: a shape
    bounded by planes is emitted exactly and has no distance left to close.

    The distance is the one :func:`_departure` reports, so the figure a user is
    shown and the figure the ladder aims at cannot come to disagree. It also
    costs nothing per rung, where measuring each facet against the surface costs
    a kernel query apiece.

    The test uses the magnitude rather than the signed figure. A bore's chords
    lie outside the metal and a wall's lie inside it, and both are the surface
    being somewhere it was not drawn.

    This never passes a shape whose surfaces may cancel. Where a wall moves in
    and a bore moves out the volumes subtract, and what is left is smaller than
    either by about the ratio of the shape to its own wall, so a thin shell
    would meet any tolerance at the first rung while both its walls stand far
    outside it. The ladder runs out and the caller refuses instead.
    """
    if held_to <= 0.0:
        return None
    if curved.area <= 0.0:
        return None

    def close_enough(enclosed: float) -> bool:
        found = _departure(curved, drawn, enclosed)
        # Nothing to hold where the volumes contradict the surface, and nothing
        # to hold where they may cancel: a rung cannot be shown to have reached
        # a distance the figure cannot state.
        if found is None or found.turns_both_ways:
            return False
        return abs(found.displaced) <= held_to

    return close_enough


def _would_not_come_close(
    label: str, held_to: float, departure: Departure | None, facets: int
) -> TranslationError:
    """The ladder ran out with the surface still further off than was asked.

    This is a refusal rather than a quiet best effort. The number is the whole
    reason the run is being made: a study asking for a wall within a distance
    and silently getting ten times it has measured something else.

    The message reports what happened. It does not report which cause applied,
    and they cannot be told apart here: a distance finer than the shape
    can be triangulated to at any price; a surface the kernel re-fits rather
    than evaluates, which answers a sharper request with a different mesh rather
    than a finer one; and a shape whose two surfaces may cancel in the figure,
    where no triangulation can demonstrate a distance was reached however good
    it is. Naming one would be wrong on the others.
    """
    if departure is None:
        reached = (
            "the volume the kernel gives the drawing stopped agreeing with "
            "which side of its surface the triangulation is on, so no distance "
            "can be stated"
        )
    elif departure.turns_both_ways:
        reached = (
            "its surfaces bend both ways, so what a volume can say about them is "
            "a net of two opposite displacements and cannot show either one "
            "reached a distance"
        )
    else:
        reached = f"reached {abs(departure.displaced):.4g} mm"
    return TranslationError(
        f"{label!r} was asked to be solved within {held_to:.4g} mm of the shape "
        f"drawn, and refining it to {facets} facets did not get there: {reached}. "
        "Ask for a distance it can reach, or redraw the surface - a shape swept, "
        "scaled unevenly or converted to a spline is re-fitted rather than "
        "evaluated, and a hollow one is measured as the difference of its two "
        "walls"
    )


def _triangulate(
    shape: Any, label: str, box: Box, skin: float | None = None, held_to: float = 0.0
) -> list[Piece]:
    """A shape that is not a box, as the triangles bounding it.

    The engine takes the triangles themselves. It answers "is this point
    inside?" against them, so the drawing is not squared onto the grid on the
    way in: the grid stays rectilinear and the staircase is the grid's, where it
    can be sized against the criterion and priced. What the route does cost is
    the triangulation's own departure from the shape, which the measures below
    hold, and the vertices arriving as three C floats, which
    :mod:`~.preflight.precision` is about.

    The triangulation is then measured rather than trusted. It has to be closed:
    an open one is read as a sheet that contains no point, and that takes the
    object out of the simulation. It also has to still enclose what the shape
    does.

    A conductor drawn as a skin arrives as an open triangulation, and ``skin``
    is the thickness it is given: the surface is offset into a solid, and that
    solid is meshed. The offset is tried only once the surface has been found
    open, so a closed surface - a sphere, a torus, a capped shell - keeps the
    route it already had and pays nothing for a case it is not.

    ``held_to`` is how far the emitted surface may stand from the drawn one, in
    mm, or zero to ask for nothing beyond the checks above. See :func:`_within`.
    """
    drawn = float(shape.Volume)
    # One reading of the surface, and every question below is answered off it:
    # the request to open at, the distance each rung is scored against, and the
    # departure reported at the end. They are the same lattice and the same
    # kernel call, and the shape does not move between them.
    curved = curvature(shape)
    opening = _opening(curved, box)
    close_enough = _within(curved, drawn, held_to)
    # Where a thickness is on offer, the first question is only whether the
    # surface closes, and one triangulation answers it. Refining is aimed at the
    # shape's own volume, which an open surface has not got, since the figure
    # the kernel computes across one is not a region. Every attempt would be
    # spent chasing it, and the last would ask for a 256th of the deflection.
    # A distance to hold is aimed at the same volume and is skipped for the same
    # reason. The pass below runs it once the surface is known to close.
    vertices, faces, enclosed = _tessellated(
        shape,
        opening,
        0.0 if skin is not None else drawn,
        enclosed_volume,
        None if skin is not None else close_enough,
    )

    fault = surface_fault(vertices, faces)
    if fault is None and skin is not None:
        # It closed, so it bounds a region after all and is meshed as one. That
        # needs the refinement the cheap pass above skipped.
        vertices, faces, enclosed = _tessellated(
            shape, opening, drawn, enclosed_volume, close_enough
        )

    if fault is not None:
        if skin is not None:
            # Reached only from the one call that carries a thickness, which is
            # a conductor drawn as a shape holding no solid at all. A drawing
            # that meant a solid and failed to close reaches the same place and
            # cannot be told from a skin by its geometry. Whether the offset
            # below builds is what separates them, and a fault severe enough to
            # leave the surface unusable does not survive it.
            built = _thickened(shape, label, skin)
            try:
                return [
                    replace(piece, thickened=skin)
                    for piece in _boxes_of(built, label, held_to=held_to)
                ]
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

    departure = _departure(curved, drawn, enclosed)
    if close_enough is not None and not close_enough(enclosed):
        raise _would_not_come_close(label, held_to, departure, len(faces))

    return [
        Piece(
            label,
            _extent_of(vertices),
            shape,
            vertices,
            faces,
            departure=departure,
        )
    ]


def _thickened(shape: Any, label: str, skin: float) -> Any:
    """A conductor drawn as a surface, as the solid that surface bounds one side of.

    The field inside a conductor is zero, so a metal skin and a metal slab
    behind it do the same thing to the problem. That is what allows a thickness
    the drawing never stated to be supplied at all. The offset may not move the
    surface that was drawn, so it runs one way only, and the drawn face stays
    exactly where it is as one wall of the result.

    ``skin`` is the thickness whose cross-section demand is the cell the metal
    is already meshed at. The metal is then thick enough that no grid can sample
    it into islands, and it never asks for anything finer than the model's own
    metal resolution. It is not free. A cross-section is read at points across
    the surface, so a wide skin holds the field down over its whole extent,
    where a conductor drawn thicker than the mesher's reach would be read as no
    feature at all.

    Whether the thickness still suits the shape is asked twice, both times
    against :data:`MOST_OF_A_SKIN`. One question cannot reach both ways a skin
    stops being one. Across the surface the thickness is a width, and that is
    answered before offsetting. Through the surface it is a radius, and that is
    read off the solid afterwards: a flat sheet sweeps exactly its area times
    the thickness however thick it gets, so nothing about the result would say.
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
    # and a disc's radius. The root of the area is a different figure. It reads
    # a long ribbon as wide as its diagonal, and admits a thickness many times
    # what the ribbon has across it.
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
    # between the two, and both give up on a shell creased sharply enough. A
    # bent guide wall, or any conductor followed round a corner, is creased that
    # sharply, and neither is an exotic drawing. ``inter`` lets the same crossing
    # resolve between faces that are not neighbours.
    options = {"inter": True, "join": INTERSECTING, "fill": True}
    try:
        solid = shape.makeOffsetShape(*args, **options)
    except (AttributeError, TypeError):
        # This says nothing about the drawing. The kernel has no such method, or
        # not with these arguments. The error travels as itself rather than
        # being reported as a shape to go and fix.
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

    These are not the shape's ``BoundBox``. That bounds the exact surface, and
    so lies marginally outside a triangulation that chords across every curve.
    The domain and the block the absorber is laid in are measured against what
    is sent.
    """
    return Box(
        lower=tuple(min(v[d] for v in vertices) for d in range(DIMENSIONS)),  # type: ignore[arg-type]
        upper=tuple(max(v[d] for v in vertices) for d in range(DIMENSIONS)),  # type: ignore[arg-type]
    )


def _corner(point: Any, dim: int) -> float:
    return float((point.x, point.y, point.z)[dim])


def _rectangles_of(edges: Sequence[Any], flat: int) -> list[Rect]:
    """The rectangles the outline closed by ``edges`` is made of, in the ``flat`` plane.

    Straightness is checked by measuring the edge against its own chord rather
    than by asking its type. A semicircle from (0, 0) to (10, 0) has
    axis-aligned endpoints, and a check that looked only at the corners would
    swallow it and return a rectangle. That is the silent staircase this module
    exists to refuse.

    Raises :class:`~.rectilinear.RectilinearError` where the outline is not made
    of axis-aligned edges. Such a shape is held another way rather than turned
    away, and which way is the caller's to say: a sheet has an area openEMS will
    take as polygons, where a solid has a surface.
    """
    axes = [dim for dim in range(DIMENSIONS) if dim != flat]
    projected: list[tuple[tuple[float, float], tuple[float, float]]] = []
    curved = 0
    for edge in edges:
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
        projected.append(
            (
                (_corner(start, axes[0]), _corner(start, axes[1])),
                (_corner(end, axes[0]), _corner(end, axes[1])),
            )
        )
    if curved:
        raise RectilinearError(f"{curved} of its edges are curved")
    return rectangles(projected, tolerance=FLATNESS)


def _outline_edges(faces: Sequence[Any]) -> list[Any]:
    """The edges bounding ``faces``, taken per face wire.

    It returns loose edges rather than ordered rings: every edge of every wire,
    in whatever order the kernel holds them. A shell of coplanar faces therefore
    works without being a special case, and so does a hole, which is what a
    clearance in a ground plane is.

    The edges come per face wire rather than from ``Shape.Edges``, which is the
    difference between cutting a fused L and refusing one. Where two faces meet,
    the kernel holds the shared edge once, but each face's wire runs along it,
    so walking the wires yields it twice and it cancels. Otherwise even-odd
    counts an interior edge as a boundary, and the seam of a fused shape is
    exactly that: an edge with metal on both sides.
    """
    return [edge for face in faces for wire in face.Wires for edge in wire.Edges]


def _standing(cut: Sequence[Rect], flat: int, low: float, high: float) -> list[Box]:
    """Each rectangle of ``cut`` as a box, spanning ``low`` to ``high`` on ``flat``.

    Passing the same value for both collapses it onto that plane, which is what
    a sheet is.
    """
    axes = [dim for dim in range(DIMENSIONS) if dim != flat]
    boxes: list[Box] = []
    for lower_u, lower_v, upper_u, upper_v in cut:
        lower, upper = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        lower[flat], upper[flat] = low, high
        lower[axes[0]], upper[axes[0]] = lower_u, upper_u
        lower[axes[1]], upper[axes[1]] = lower_v, upper_v
        boxes.append(Box(lower=tuple(lower), upper=tuple(upper)))  # type: ignore[arg-type]
    return boxes


def _laid(boxes: Sequence[Box], flat: int) -> float:
    """The area ``boxes`` cover in the ``flat`` plane."""
    axes = [dim for dim in range(DIMENSIONS) if dim != flat]
    return sum(
        (piece.upper[axes[0]] - piece.lower[axes[0]])
        * (piece.upper[axes[1]] - piece.lower[axes[1]])
        for piece in boxes
    )


def _cut_into(label: str, boxes: Sequence[Box], shape: Any) -> list[Piece]:
    """``boxes`` as pieces of ``shape``, numbered where there is more than one.

    Every rectangle carries the shape it was cut from, which is the whole of
    what they tile. Nothing is lost by that. Each of them is a box, so the grid
    pins its own faces, and lines bound any clearance between them on both
    sides. A measurement has to find a clearance only where the grid cannot pin
    the shape.
    """
    labels = (
        [label] if len(boxes) == 1 else [f"{label}#{number}" for number in range(1, len(boxes) + 1)]
    )
    return [Piece(name, piece, shape, departure=EXACTLY) for name, piece in zip(labels, boxes)]


def _flat_axis(face: Any) -> int | None:
    """The one axis ``face`` lies flat on, or ``None`` where it lies on none or two.

    A face lying in a plane square to an axis has no extent along it, to within
    the arithmetic that produced the bounding box, so this asks whether a face is
    planar and square to the grid without asking the kernel what kind of surface
    it holds. The rest of this module judges shapes by the same rule.

    One face answering ``None`` is enough to disqualify a solid, so the reading
    is cheap rather than exhaustive. Measured under FreeCAD
    1.1.1: a cylinder's two end caps are flat on the axis it stands on and its
    wall on none, a sphere's and a torus' single face on none, and a box rotated
    thirty degrees in the plane keeps the two caps square and loses the four
    sides.
    """
    at = _bounds(face.BoundBox)
    thin = [dim for dim in range(DIMENSIONS) if at.is_flat(dim)]
    return thin[0] if len(thin) == 1 else None


def _planes(square: Sequence[tuple[int, Any]], axis: int) -> list[tuple[float, list[Any]]]:
    """Where a solid's faces square to ``axis`` lie, low to high, each with its own.

    A plane is a cluster rather than a coordinate. A coordinate reached by
    arithmetic is not the one that was typed, so two faces lying in one plane
    need not report it identically. A plane split in two leaves a slab of
    nothing between them.
    """
    found = sorted(
        (_bounds(face.BoundBox).lower[axis], number)
        for number, (flat, face) in enumerate(square)
        if flat == axis
    )
    planes: list[tuple[float, list[Any]]] = []
    for where, number in found:
        if not planes or where - planes[-1][0] > FLATNESS:
            planes.append((where, []))
        planes[-1][1].append(square[number][1])
    return planes


def _stacked(square: Sequence[tuple[int, Any]], axis: int) -> list[Box] | None:
    """The solid as the slabs it stands in along ``axis``, or ``None`` where it will not cut.

    Between two neighbouring planes square to the axis nothing bounds the solid
    but walls parallel to it, so the cross-section there does not change and the
    slab is exactly the rectangles that cross-section is made of.

    Which rectangles those are is a parity. A face square to the axis is where
    metal starts or stops, so a column of the slab is inside as often as the
    faces below it have said so. Even-odd over every such outline walked so far
    is that count, so each plane's edges are added to the ones already collected
    rather than replacing them.
    """
    planes = _planes(square, axis)
    if len(planes) < 2:
        return None
    edges: list[Any] = []
    boxes: list[Box] = []
    for (low, lying), (high, _) in zip(planes, planes[1:]):
        edges += _outline_edges(lying)
        try:
            cut = _rectangles_of(edges, axis)
        except RectilinearError:
            return None
        boxes += _standing(cut, axis, low, high)
    return boxes or None


def _seams(boxes: Sequence[Box]) -> tuple[int, float]:
    """What a stack costs the grid: pieces first, then the thinnest of them.

    These are the two terms :func:`~.rectilinear._cost` weighs, for the same
    reasons: every piece is a seam the mesher has to put a line on, and the
    thinnest piece sets how close two of those lines stand, which is a cell the
    whole domain pays for.
    """
    return (len(boxes), -min(min(piece.extents) for piece in boxes))


def _cut_up_solid(shape: Any, label: str) -> list[Piece] | None:
    """A solid bounded entirely by planes square to the grid, as the boxes it is made of.

    Cut this way the solid is held exactly and every piece fills its own box.
    The grid needs that to size a cell from a conductor's width, and pre-flight
    needs it to measure what the grid left of one. Handed over as a
    triangulation instead, neither an L nor a step offers either: the box holds
    the fold and the overhang as metal, and the spans anything reads off it are
    that box's rather than any part of the metal's.

    Which axis it is sliced on decides how many pieces it comes to. The three
    are tried rather than argued about: the same step lying two ways cuts into
    different numbers of them, and no rule names the cheaper one.

    ``None`` where the shape is not one of these, and it is then triangulated
    like anything else. A proof that does not come out is not a reason to turn a
    shape away: the triangles hold it either way, and the boxes are the whole of
    what is lost.
    """
    square: list[tuple[int, Any]] = []
    for face in getattr(shape, "Faces", ()) or ():
        flat = _flat_axis(face)
        if flat is None:
            return None
        square.append((flat, face))
    if not square:
        return None
    # Proved before the choice rather than after it. An axis whose reading came
    # out wrong is one to pass over. It is not a reason to hand the whole shape
    # to the triangulator while another axis holds it exactly.
    stacks = [
        boxes
        for boxes in (_stacked(square, axis) for axis in range(DIMENSIONS))
        if boxes is not None and _adds_up(boxes, shape)
    ]
    if not stacks:
        return None
    return _cut_into(label, min(stacks, key=_seams), shape)


def _adds_up(boxes: Sequence[Box], shape: Any) -> bool:
    """Whether ``boxes`` are the shape, by what they measure and by where they are.

    The volume catches an island dropped, a hole not seen, and a plane read for
    a shape whose walls turn out not to stand square after all. It cannot catch
    a stack that holds the right amount of metal in the wrong place, and such a
    stack exists: a step's own base swept through its whole height has the
    step's volume exactly. So the surface is compared as well. The surface is
    the boundary rather than the measure, and only the stack bounded where the
    shape is answers it.
    """
    volume = math.fsum(piece.extents[0] * piece.extents[1] * piece.extents[2] for piece in boxes)
    if not math.isclose(volume, float(shape.Volume), rel_tol=CUT_TOLERANCE):
        return False
    return math.isclose(_exposed(boxes), float(shape.Area), rel_tol=CUT_TOLERANCE)


def _exposed(boxes: Sequence[Box]) -> float:
    """The surface a stack of boxes with disjoint interiors leaves outside itself.

    Where two of them meet, that much of each stops being surface, so a contact
    comes off twice.
    """
    whole = math.fsum(
        2.0 * (x * y + y * z + z * x) for x, y, z in (piece.extents for piece in boxes)
    )
    buried = math.fsum(
        _contact(boxes[i], boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))
    )
    return whole - 2.0 * buried


def _contact(one: Box, other: Box) -> float:
    """How much surface two boxes bury against each other, which is nothing unless
    they meet face to face."""
    for dim in range(DIMENSIONS):
        if (
            abs(one.upper[dim] - other.lower[dim]) <= FLATNESS
            or abs(other.upper[dim] - one.lower[dim]) <= FLATNESS
        ):
            across = [
                min(one.upper[d], other.upper[d]) - max(one.lower[d], other.lower[d])
                for d in range(DIMENSIONS)
                if d != dim
            ]
            return max(0.0, across[0]) * max(0.0, across[1])
    return 0.0


def _cut_up(shape: Any, label: str, box: Box, flat: int, area: float) -> list[Piece]:
    """A flat Manhattan shape, as the rectangles it is made of.

    A shape carrying no face at all is walked as its own edges instead. It has
    no area either, so it cannot be cut into anything that adds up, and the walk
    is here so that the proof below reports it rather than an empty list.
    """
    walked = _outline_edges(getattr(shape, "Faces", ()) or ()) or list(shape.Edges)
    try:
        cut = _rectangles_of(walked, flat)
    except RectilinearError:
        # Not axis-aligned, so it cannot be cut into rectangles. A sheet is
        # still an area, and openEMS takes an area as flat polygons. Cutting
        # into rectangles stays the first choice where it applies: it is exact
        # by construction and leaves the grid the fewest planes to hold.
        return _triangulate_sheet(shape, label, box, flat)

    elevation = box.lower[flat]
    boxes = _standing(cut, flat, elevation, elevation)

    # What was cut has to add up to what was drawn. This is the whole guarantee
    # that the cut is exact rather than a fit: it catches an island dropped, a
    # hole not seen, a ring counted twice, and an even-odd inversion. It is a
    # refusal rather than an assert, since `assert` vanishes under -O, and it
    # guards against the kernel having described the user's shape differently
    # than it was read.
    laid = _laid(boxes, flat)
    if not boxes or not math.isclose(laid, area, rel_tol=CUT_TOLERANCE):
        # Which way it misses says where the fault is. Short means the outline
        # encloses less than the shape claims, and pieces laid over each other
        # do exactly that: each reports its own area while the boundary counts
        # the overlap once. No drawing explains an over-count.
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

    return _cut_into(label, boxes, shape)


def _shape_of(obj: Any, sub: str) -> Any:
    """The shape a binding names - the whole solid, or one sub-element of it."""
    shape = getattr(obj, "Shape", None)
    if shape is None or not sub:
        return shape
    return shape.getElement(sub)


def _elements_named(reference: Any) -> list[str]:
    """The sub-element names a reference carries, or ``[""]`` for the solid."""
    return picks.named(reference)


def _reference_boxes(
    reference: Any, skin: float | None = None, held_to: float = 0.0
) -> tuple[Any, list[Piece]]:
    """Every region one binding reference names.

    A binding may name a whole solid, or particular faces of one. The faces are
    not a detail to drop: binding copper to the top face of a substrate is how a
    ground plane gets drawn, and taking the owning solid's box instead turns
    that sheet into a block filling the entire board. It meshes, it solves, and
    the answer is for a different structure.

    One region per named element rather than their union, because the union of
    two faces on different planes is a box that is neither of them.

    Which element a region came from is not returned beside it. Each
    :class:`Piece` carries the geometry it is, and one element yields as many
    pieces as it needs, so a name paired by position would land on whichever
    piece shared its index.
    """
    obj, sub = reference if isinstance(reference, tuple) else (reference, None)
    names = [sub] if isinstance(sub, str) else list(sub or ())
    names = [name for name in names if name]

    if not names:
        return obj, solid_boxes(obj, skin, held_to)

    shape = getattr(obj, "Shape", None)
    if shape is None:
        raise TranslationError(f"{_label(obj)!r} has no shape, so there is nothing to mesh")
    regions = []
    for name in names:
        label = f"{_label(obj)}:{name}"
        regions.extend(_boxes_of(shape.getElement(name), label, skin, held_to))
    return obj, regions


def _refinement_boxes(reference: Any, region: str) -> list[tuple[str, Box]]:
    """Every box one ``EMMeshRegion`` reference names, as ``(label, box)``.

    These are bounding boxes. They are not proved to be boxes the way
    :func:`solid_boxes` proves a material region is one. Refining around a
    cylinder or a fillet is a reasonable thing to want, and the rectilinear
    answer to it is the box it sits in. That is a surprise worth naming rather
    than a refusal, so the property tooltip says "bounding box" and this does
    not complain.

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

    A word that is not one of the two is refused rather than defaulted. FreeCAD
    stores an enumeration's whole list in the document and restores it from
    there rather than from the class, so a file can go on offering a word this
    adapter has no behaviour for. The reading it would fall back to silently is
    the one that leaves the region doing the opposite.
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

    The key uses ``Name`` rather than the label. FreeCAD keeps the name unique
    within a document and leaves the label unique only by convention, so two
    objects a user has given one label are still two things to relax separately.

    The sub-element belongs in the key. A conductor is routinely a face of the
    board it sits on, so the object alone does not identify what was drawn.
    Relaxing by object would let a coarsened substrate take the ground plane
    down with it, and this feature is built not to reach that far.
    """
    return (getattr(obj, "Name", "") or _label(obj), element)


@dataclass(frozen=True)
class _Relaxation:
    """The size some geometry settles for, and who asked."""

    size: float
    asked_by: str


def _relaxations(refinements: Sequence[Any]) -> dict[tuple[str, str], _Relaxation]:
    """Which drawn geometry stops driving the grid, and the size it settles for.

    A coarsening names geometry, and that is why it is carried this way rather
    than as a box like a refinement. The grid is separable, so a box spends
    itself on a slab through the model on each axis. A refinement that
    overshoots hands out cells nobody asked for, and a coarsening that
    overshoots would take them off whatever else lies level with the box,
    anywhere in the model.

    It is keyed by :func:`_subject`, so it reaches what a material binding would
    have had to name to be the same thing.

    Two coarsenings over one subject leave it at the finer of the two, the same
    way round as two refinements: the grid ends up as fine as the finest thing
    asked for, whichever direction asked.
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

    Only a material binding gives geometry a size the grid works from, so a
    coarsening aimed anywhere else changes nothing at all. It is refused rather
    than passed over. The region is a typed request, and one that is quietly
    dropped leaves the property editor showing a size that was never applied.
    This check refuses that instead.
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

    This returns refinements only. A region set to coarsen is attached to the
    geometry it names instead - see :func:`_relaxations` - and contributes no
    box.

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

    Unlike :func:`solid_boxes` this does not demand that the selection be a box.
    A face of a box-shaped solid is planar by construction, and callers check
    the flatness that matters to them along the axis it matters on.
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
