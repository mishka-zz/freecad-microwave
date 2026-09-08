# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Measuring the lengths a drawing carries, so the grid can resolve them.

Local feature size is the distance to the medial axis: at any point, the radius
of the largest ball that fits inside the material (or inside the gap) and
touches the boundary twice. This module computes no medial axis. The CAD kernel
answers the questions it needs directly.

The measurements, and what each becomes:

* **A gap between two bodies**, from the witness pair ``distToShape`` returns
  with the distance. The pair certifies a ball of that diameter touching both,
  so it is a separation demand along the direction the pair points in. The
  witness is only where the pair is closest, so the gap is also walked outward
  from samples along the run the pair holds together.
* **A curved surface's own radius**, from the sharpest principal curvature. It
  carries no direction, so it is a connection demand. It is a radius of
  curvature and not the medial radius. On a fillet the two part company badly,
  and nothing here may read it as a thickness.
* **How thick the metal is**, as the chord the body cuts from the inward normal
  at a point on its surface. It is asked directly, as one line cast at the
  triangulation the body reaches the engine as. A witness pair cannot see a
  thickness carried by a single face: a cylinder's only non-adjacent face pair
  is its two ends, so nothing witnesses its diameter.
* **How many cells span a dielectric**, which is that same chord asked as a
  count. A thin dielectric under-resolves the field varying across it rather
  than a cell failing to fit inside it.
* **Where a curved surface is**, against the radius already measured. A flat
  face is pinned as a grid line and placed exactly; a curved one is placed by
  sampling, to within about half a cell of where it was drawn.
* **Where the surface stops being smooth** - a join between two faces whose
  normals disagree. A field singularity sits there. Neither feature size nor
  curvature covers it, and the demand is made over the whole plane across the
  edge, so what the join gets does not turn on how the part was drawn.

Why a witness pair misses a cylinder's diameter, why a fillet's curvature is not
a thickness, where a join's demands and a thickness demand cross, and why the
join's demand is about a plane rather than a pair of directions, are in
docs/internals/feature-size.md.

The kernel answers each of those directly except the thickness. A gap is a
witness-pair query, and walking it outward is more of the same query; a
curvature is arithmetic on the surface's own derivatives; a join is the angle
between two normals. A thickness has no such query: it is a property of the
whole solid rather than of any face, and the nearest thing the kernel offers is
the medial axis, which is dearer to compute than the line this casts. So the
chord is read off the triangulation, in :mod:`.raycast`, where the crossings
come back sorted and the first one that leaves the material ends the run.

That is also the surface openEMS is given. A curved body reaches the engine as a
polyhedron, and the engine decides a cell's material by counting how many of
these same triangles a segment crosses (``CSPrimPolyhedron::IsInside``,
``CSXCAD/src/CSPrimPolyhedron.cpp:290-294``) - so a thickness read here is a
thickness of the structure that gets solved rather than of the drawing it
approximates.

**A face's lattice is placed against the face's own boundary.** Every
measurement off a surface is taken at stations laid across the rectangle the
surface is stated over, and a face occupies only part of that rectangle - so
each station has to be placed on the face or beside it before anything is
measured there. The kernel answers that for one parameter pair per call, at a
price that does not fall as the calls repeat on one face. Here a face is asked
nothing where it covers its whole rectangle, and read against its own boundary
as a polygon where it does not; a station the polygon puts near that boundary is
handed to the kernel, which is where a polygon and the curve it stands for can
disagree.

This module imports no FreeCAD. It is handed shapes and calls methods on them,
the way the rest of the translation layer is, so it can be exercised without a
CAD kernel present.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ...portbox import KERNEL_TOLERANCE, corner
from .raycast import Surface, chords_at, prepared
from .sizing import DIMENSIONS, Feature, fits_inside
from .spend import Spend

__all__ = ["Body", "Curvature", "bends_through", "curvature", "features"]

#: How far apart two gaps have to be before neither can affect the other,
#: as a multiple of the coarsest cell. A gap of width d asks for cells no
#: smaller than d/K, where K is the largest that ``sqrt(m_max) * sum_j sqrt(m_j)``
#: reaches over unit normals; Holder bounds that sum by 3**(3/4) and the root by
#: 1, so 3**(3/4) is safe. The body diagonal is not the worst case here. It
#: gives only sqrt(3), and culling at that drops real gaps.
SEPARATION_REACH = 3.0**0.75

#: How far the grid may misplace a curved boundary - a wall, or the rim a sheet
#: ends at - as a fraction of the radius it curves through. openEMS decides a
#: cell's material by sampling one point in it, so a boundary settles within
#: half a cell of where it was drawn on each axis; the connection form bounds
#: that displacement by half the thickness it is stated against, so stating it
#: against this times the diameter holds the boundary to this fraction of its
#: own radius. It spends the reciprocal of it, times ``sqrt(DIMENSIONS)``, in
#: cells across that diameter.
#:
#: This is a default rather than a constant. It is a share of a radius, so
#: refining the cell sizes leaves it exactly where it was, and a large smooth
#: wall keeps the same displacement however fine the rest of the grid gets. A
#: caller measuring how an answer approaches a drawing has to move it, so
#: :func:`features` takes it as an argument.
SURFACE_FIDELITY = 0.1

#: A normal disagreement at a join, beyond which the surface is not smooth
#: there and carries a field singularity. The angle is well above the tolerance
#: a kernel leaves on a join it built to be tangent, and well below anything a
#: person would draw meaning to be a corner. It therefore separates a fillet
#: from an edge, not one kind of edge from another.
SHARP_DEGREES = 15.0

#: How far under a whole cell the length behind a station count may sit and
#: still be counted as that whole one, in cells. A count is an integer read off
#: a length in millimetres, so it is a step function of a measurement, and a
#: drawing sits on a step whenever a face is a round number of the cell it is
#: sampled at. The kernel measures that length again whenever a file hands the
#: surface back, and the two readings differ in their last bits, so a count
#: taken with nothing under the step reads one lattice off the drawing and
#: another off the file.
#:
#: What it covers is rounding. A file that hands back the same surface hands
#: back the same length to its last bits, and this stands orders above that,
#: and orders below anything a drawing states on purpose - a dimension is
#: written in millimetres rather than in a rounding of them.
#:
#: What it does not cover is a surface the file re-approximates rather than
#: copies. A boolean's trimmed face comes back a different surface and its
#: length moves by the fit's own tolerance rather than by rounding, which is
#: millions of times this; so does any face far enough from the origin, the
#: fit's error being absolute while the length is not. Where such a move
#: crosses a step the lattice moves with it, and no slack reaches that.
COUNT_SLACK = 1e-9

#: The most parameter samples taken across one face in one direction. Samples
#: are spread evenly in the face's parameters, which carry no scale of their
#: own: one unit of u is a radian on a cylinder and a millimetre on a plane. So
#: the count is set from the face's size, and the spacing in space is even only
#: where the parameterisation is. A face wanting more samples than this is
#: sampled more coarsely than it asked, and the field ripples between the
#: samples by correspondingly more. :data:`MAX_EDGE_SAMPLES` bounds a rim the
#: same way, and says there why it is larger.
#:
#: The cap is per direction rather than over the face as a whole. A cap over the
#: face spends a long thin face's allowance across a width that needed none.
#: Spending it in proportion to each direction's length leaves the count
#: unsaturated on a face that would otherwise reach the cap, and an unsaturated
#: count moves by one when a re-parameterisation moves the length it was read
#: from. That shifts every sample on the face, and where the quantity sampled
#: varies quickly it changes what is asked of the grid. A face too small to
#: reach the cap is unsaturated either way. The cap settles the large faces.
MAX_SAMPLES = 32

#: How far a face's area may fall short of the area its own surface covers over
#: its parameter rectangle, and the face still be taken as covering that
#: rectangle - as a share of that area.
#:
#: The trimmed domain is a subset of the rectangle, and area is the integral of
#: a non-negative Jacobian, so the shortfall is the area of what the trim leaves
#: out. A shortfall of nothing is a trim that leaves out nothing at all.
#:
#: It is therefore a tolerance on two quadratures of one region rather than a
#: geometric slack, and it is small for that reason. What a larger one would
#: cost is a bound: an omission of area ``t`` times the patch has an inradius of
#: at most ``sqrt(t / pi)`` times the patch's own size, and one lattice step
#: covers the patch's size over the root of the stations laid on it - so a
#: station the slack lets through stands within ``sqrt(t * stations / pi)``
#: steps of the face. At this value and the most stations a face can carry, that
#: is more than two orders inside :data:`BOUNDARY_BAND`. A slack stated against
#: a lattice cell instead reaches past the band, and lets a small round hole
#: keep the station at its own centre.
UNTRIMMED_SHORTFALL = 1e-9

#: How far the polygon a trim is read as may depart from the trim itself, as a
#: share of the lattice step. It is what the parameter curves are discretised
#: to.
#:
#: The share is taken of the narrower of the two steps. A departure is bounded
#: in the face's own parameters and is read in steps, so what it becomes is the
#: departure over each step in turn - and the narrower step is the one that
#: makes it larger.
BOUNDARY_FINENESS = 1.0 / 64.0

#: How near that polygon a sample is handed to the kernel rather than answered
#: from the polygon, as a share of the lattice step.
#:
#: It is eight times :data:`BOUNDARY_FINENESS`. A deflection honoured exactly
#: would make the two equal enough: the polygon then departs from the curve by
#: no more than the band, so every sample the two could disagree about is inside
#: it. The eight is margin against a deflection that is a request rather than a
#: guarantee, which is what a deflection is everywhere else the kernel takes
#: one. Inside the band the kernel answers, so what the filter says near a
#: boundary is the kernel's own.
BOUNDARY_BAND = 1.0 / 8.0

#: How far apart two spellings of one boundary corner may stand and still be
#: one corner, in lattice steps.
#:
#: A corner reached along two edges is computed from two parameter curves, and
#: the two do not always land on the same last bits. A parity ray at exactly
#: that height then passes between the two spellings: both segments straddle it,
#: both are counted, and everything the ray crosses beyond that corner reads as
#: the other side. What separates a corner from a real turn is orders - the
#: boundary is followed to :data:`BOUNDARY_FINENESS` of a step and this is far
#: under that.
ONE_CORNER = 1e-9

#: The station-and-segment pairs the parity test builds at once. A face's whole
#: lattice is classified against its whole boundary in one array operation, and
#: this is what that array is cut into. The face with the most stations and the
#: boundary with the most segments meet here.
#:
#: It is a memory bound and not a tuning knob. Several arrays of this many pairs
#: stand at once and two of them carry a parameter pair apiece, so what a block
#: holds is a small multiple of this times what a double takes.
MOST_CROSSINGS = 1 << 19

#: How many samples a direction :func:`curvature` lays on a face, and how
#: many :func:`bends_through` lays along an edge. The count is fixed rather than
#: spaced in millimetres like :data:`MAX_SAMPLES`. Its caller triangulates
#: before any grid exists, and a count read off a cell size would make the
#: triangulation follow the mesh policy that is decided from it.
#:
#: It is a bound rather than a convergence. The sharpest radius is a minimum
#: over the samples, so it only falls as samples are added, and on a surface
#: that genuinely reaches zero - a cone with a real apex - it tracks the count
#: rather than settling. Raising it therefore drives the request toward zero on
#: such a shape, and :func:`geometry._held_inside_the_band` has a floor for
#: that. The count only has to be good enough to land inside a band orders wide.
CURVATURE_SAMPLES = 9

#: How far outside a solid a point may sit and still count as in it, in mm. It
#: is asked of points the kernel itself placed on the surface, so it has to
#: absorb the rounding in evaluating a face and nothing wider. That is the
#: kernel's own tolerance, so this is not a second length. Only the walk across
#: a gap asks it: a chord is read off the triangulation, where a crossing is
#: placed rather than tested for.
CONTAINMENT_TOLERANCE = KERNEL_TOLERANCE

#: How wide a gap has to be before it is a gap, in mm. The same length again,
#: for the same reason: it absorbs the kernel's rounding, here in bringing two
#: surfaces together rather than in evaluating one. Two solids meeting along a
#: plane answer exactly zero and need no tolerance at all. Two curved ones
#: tangent to each other answer the last bits of their own arithmetic. A demand
#: made on that figure is a cell size no grid can carry, nothing else can
#: dominate it, and it sets the timestep for the whole domain.
TOUCHING = KERNEL_TOLERANCE

#: How far to step off the surface to find out whether the sampled body lies
#: that way, in mm. The step is clear enough of :data:`CONTAINMENT_TOLERANCE`
#: that a step to the outside reads as outside, and far below any wall this
#: meshes, so a step to the inside is still inside the thinnest metal drawn in
#: practice. It settles a direction and measures nothing, so it is a length of
#: its own rather than a share of the reach. A share of the reach is set by the
#: model's coarsest cell, and would step clean through the wall it was sent to
#: find.
#:
#: The gap walk asks this because it casts at a *neighbour*, so it has to
#: establish which way is out of the body it is standing on before it can look.
#: A chord asks nothing of the kind: it is read off the body's own
#: triangulation, and there the crossing's own direction says which side the
#: material is on.
WINDING_PROBE = 10.0 * CONTAINMENT_TOLERANCE

#: How far apart the places the gap walk looks at are, as a share of the model's
#: coarsest cell. It sets the narrowest wall that can be stepped over rather
#: than struck. A wall stepped over reads the gap wide, and every proxy in this
#: module fails in that direction.
#:
#: One tenth of a cell is far below anything a grid capped there resolves, so a
#: wall this walk steps over is under-resolved either way.
#:
#: A wall thicker than this step is struck wherever it starts, and one lying
#: wholly between two of the places looked at is missed. Containment includes
#: the boundary, so a probe landing exactly on the target's near face is a
#: strike and a wall of exactly one step is struck rather than stepped over.
#:
#: A station that strikes nothing marches its whole reach, so it costs
#: :data:`SEPARATION_REACH` divided by this, rounded up. Both are shares of the
#: same cell, so that division has neither the drawing nor the mesh policy in
#: it. A station pays a winding probe on top of the march, and a body with no
#: inside is walked both ways and pays the march twice instead.
#:
#: That cost lands on a sample whose neighbour is out of reach, and on one whose
#: face points away from it. Both ask for nothing. ``tests/test_lfs.py`` holds
#: the bound.
MARCH_STEP = 0.1

#: How closely the gap's near wall is then placed inside that bracket, as a
#: fraction of the gap itself. It is relative rather than absolute. The bracket is a share
#: of the coarsest cell in the model rather than of the gap being measured, so a
#: fixed number of halvings would read a narrow clearance in a large domain to a
#: precision that has nothing to do with the clearance. A gap becomes a cell
#: size, and a hundredth is finer than a cell size is read to.
CHORD_TOLERANCE = 0.01

#: The most halvings spent reaching that, so a gap far below the bracket it was
#: found in cannot cost without bound. It is a guard rather than a working
#: limit. The bracket is one march step, the halvings needed are the log of that
#: over the gap, and this many of them is far past what a double can
#: distinguish, so nothing a kernel can hand over gets near it. Giving up here
#: leaves an error either side of the answer, where giving up in the march above
#: reads the gap wider.
MAX_HALVINGS = 32


#: The most samples taken along one edge. It is larger than the per-direction
#: bound on a face, because sampling an edge costs in proportion to its length
#: where sampling a face costs the square of its size. A rim capped below what
#: it asked for shows up as cells that ripple between the samples instead of
#: following it evenly.
MAX_EDGE_SAMPLES = 256


@dataclass(frozen=True)
class Body:
    """One drawn object, as this module needs it.

    :param label: What to call it in a message.
    :param shape: The FreeCAD shape. What is asked of it is ``BoundBox``,
        ``Faces``, ``Edges``, ``Solids``, ``distToShape`` and ``isInside``; of a
        face, ``Area``, ``ParameterRange``, ``Surface``, ``Wires``, ``Edges``,
        ``curveOnSurface``, ``valueAt``, ``normalAt``, ``curvatureAt`` and
        ``isPartOfDomain``; of a surface, ``isPlanar`` and ``parameter``; of an
        edge, ``tangentAt``, ``FirstParameter``, ``LastParameter`` and
        ``hashCode``; of a wire, ``OrderedEdges``; and of a parameter curve,
        ``discretize``. Everything else is addition and scaling on the
        points the shape hands back, so a stand-in needs no kernel. A gap is
        walked by arithmetic on a point rather than by making a line to
        intersect with, so the kernel stays out of that. The one thing built
        here is a face's own surface over its own parameter rectangle, which is
        what says whether the face covers it.
    :param vertices: The points of the triangulation this body reaches the
        engine as.
    :param faces: Its triangles, as three numbers into ``vertices``. A body
        whose own lengths are measured always has them, since what decides that
        is whether it arrived as a triangulation rather than as a box. They are
        what a chord is measured against, so the thickness read here is the
        thickness of the structure that gets solved.
    :param metal: Whether its edges carry a field singularity. A dielectric
        corner does not, so nothing is asked for one.
    :param measured: Whether this body's own lengths are read here. The mesher
        already has the face positions of a solid that reached it as a box.
        Such a body is still carried, because a gap to it is a gap.
    :param sheet: Whether the shape is an area rather than a volume. It decides
        what an edge with only one face beside it means: on a sheet that is the
        outline, where the metal stops, and on a solid it is a seam - an
        artefact of how the kernel parameterised a closed surface, with metal on
        both sides of it and nothing to resolve.
    :param relaxed_to: The finest cell this body's own lengths may ask for, or
        ``None`` to ask at what they measure. A gap to it is not relaxed. A
        separation belongs to both bodies, and one of them asking for less does
        not make the other agree.
    """

    label: str
    shape: Any
    metal: bool = False
    measured: bool = True
    sheet: bool = False
    relaxed_to: float | None = None
    vertices: Sequence[Sequence[float]] = ()
    faces: Sequence[Sequence[int]] = ()


def features(
    bodies: Sequence[Body],
    cap: float,
    edge_size: float | None = None,
    fidelity: float = SURFACE_FIDELITY,
    min_lines: int = 0,
    spend: Spend | None = None,
) -> list[Feature]:
    """Every length the drawing carries that the grid may have to resolve.

    :param cap: The coarsest cell the grid will use. Anything whose demand
        lands at or above it cannot bind, so it is never measured. The cost is
        then proportional to the geometry that is genuinely close rather than
        to all the geometry there is.
    :param edge_size: The cell size a metal edge asks for, or ``None`` to ask
        for nothing there. It floors what a curved face may ask for as well, so
        :func:`_curvatures` is handed it.
    :param fidelity: How closely a curved surface has to be followed, as a
        share of the radius it curves through. See :data:`SURFACE_FIDELITY`.
    :param min_lines: How many cells to put across a dielectric's own
        thickness, or fewer than two to ask for nothing there. One is not a
        count: it asks for the layer's whole extent. The mesher applies the
        rule to a box only above that, for the same reason.
    :param spend: Where to add what measuring this drawing cost, or ``None``
        to count nothing.

    Only an exact repeat is dropped here, by :func:`_deduplicated`. A demand
    another one covers is dropped where the axis' field is built, which is after
    every rule that decides whether a demand reaches that field at all - see
    :func:`~.sizing_field._pruned`.
    """
    found: list[Feature] = []
    found.extend(_separations(bodies, cap, spend))
    for body in bodies:
        if not body.measured:
            continue
        # What each body is asked for divides on the same line openEMS divides
        # on. openEMS decides a conductor's boundary by sampling one point, so
        # the boundary staircases and is held to a fidelity, and the
        # conductor's thickness decides whether it still conducts - an argument
        # about zeroed Yee edges sharing nodes, which is a statement about
        # metal. openEMS averages a dielectric over the cell instead. Nothing
        # about a dielectric staircases, and its thickness decides whether more
        # than one cell carries the field varying across the layer.
        mine: list[Feature] = []
        if body.metal:
            # One lattice for the two measurements that read it. They ask the
            # same faces at the same spacing, and placing a lattice on its face
            # costs a reading of the face.
            stations = _Stations(body, _spacing(cap, edge_size))
            mine.extend(_curvatures(body, cap, edge_size, stations, fidelity, spend))
            if body.sheet:
                mine.extend(_rims(body, cap, edge_size, fidelity, spend))
            mine.extend(_thicknesses(body, cap, stations, spend))
            if edge_size is not None:
                mine.extend(_sharp_joins(body, edge_size, spend))
        elif min_lines >= 2:
            mine.extend(_element_counts(body, cap, min_lines, spend))
        # Before anything reads what the demand asks for, so that a relaxed
        # demand is compared at the size it will actually ask for rather than
        # at the one it measured.
        if body.relaxed_to is not None:
            mine = [replace(feature, relaxed_to=body.relaxed_to) for feature in mine]
        found.extend(mine)
    if spend is not None:
        spend.raised += len(found)
    return _deduplicated(found)


def _deduplicated(found: Sequence[Feature]) -> list[Feature]:
    """One entry per distinct length, keeping the order they were measured in.

    A face is sampled on a lattice and answers the same curvature at the same
    place at several stations, so the same demand arrives several times over.
    They are not a repeat of a measurement standing near another one: nothing
    tells them apart at all, and whatever becomes of one becomes of the rest.

    This is the whole of what can be dropped here. Deciding that one demand
    covers another rests on that other one reaching the field, which is settled
    per axis and further down - see :func:`~.sizing_field._pruned`.
    """
    seen: set[Feature] = set()
    kept: list[Feature] = []
    for feature in found:
        if feature in seen:
            continue
        seen.add(feature)
        kept.append(feature)
    return kept


def _separations(
    bodies: Sequence[Body], cap: float, spend: Spend | None = None
) -> Iterator[Feature]:
    """Gaps between bodies, from the witness pairs that realise them, and then
    along the run the pair holds together.

    The pairing is over bodies rather than over faces. A gap is a property of
    two objects, and the kernel finds the closest points between them without
    being told which faces to look at.

    Bodies further apart than any demand could matter are never queried. The
    bound is :data:`SEPARATION_REACH`, which is where the smallest cell a gap of
    width ``d`` can ask for is worst, so a pair whose boxes are further apart
    than that multiple of the cap is provably inert. Their bounding boxes are
    compared first because that is arithmetic, where the exact query is a kernel
    call.

    The witness pair is only where the pair is closest. Two bodies running
    close along a length are witnessed at the extremum and nowhere between, and
    between point demands the sizing field climbs at its own grading slope
    rather than following the gap. On a run parallel to an axis the grid
    delivers the gap anyway, because the fine spacing one axis was asked for
    spans the whole domain on the other two. On an oblique or curved run the
    gap's own position moves across the axes, and the middle of the run is left
    to the cones from the witnesses. So the gap is also walked along the run,
    by :func:`_gap_run`.
    """
    reach = SEPARATION_REACH * cap
    for first in range(len(bodies)):
        for second in range(first + 1, len(bodies)):
            one, other = bodies[first], bodies[second]
            # Two boxes are already described to the mesher and add nothing
            # here: each pins its own faces, and the thirds rule sizes the gap
            # between them.
            if not (one.measured or other.measured):
                continue
            if _boxes_further_apart_than(one.shape, other.shape, reach):
                continue
            distance, pairs, _ = one.shape.distToShape(other.shape)
            distance = float(distance)
            # Touching is not a gap. Two solids that share a face have no
            # medial ball between them, and a zero thickness is not a length.
            # Neither is a distance the kernel arrived at by bringing two
            # curved surfaces together: that is tangency reported to the
            # precision it was computed at, not a drawn clearance.
            if distance <= TOUCHING or distance >= reach:
                continue
            source = f"the gap between {one.label!r} and {other.label!r}"
            for near, far in pairs:
                normal = corner(_xyz(far)[d] - _xyz(near)[d] for d in range(DIMENSIONS))
                for point in (near, far):
                    place = _xyz(point)
                    yield Feature(
                        thickness=distance,
                        normal=normal,
                        lower=place,
                        upper=place,
                        source=source,
                    )
            yield from _gap_run(one, other, distance, reach, cap, source, spend)


def _gap_run(
    one: Body,
    other: Body,
    distance: float,
    reach: float,
    cap: float,
    source: str,
    spend: Spend | None = None,
) -> Iterator[Feature]:
    """The same gap, asked along the run rather than only at its extremum.

    One body of the pair is sampled over its surface and the gap is walked
    outward from each sample. The walk steps until a containment probe lands
    inside the neighbour, then halves to place the crossing. A sample whose
    walk finds no neighbour within the reach asks nothing.

    The gap is probed rather than cast at a surface, because a neighbour held
    as a box carries no triangles to cast at.

    The walk follows the sampled face's own normal rather than the direction to
    the nearest point of the neighbour. One extremal query cannot answer the
    nearest direction along a run. Where the neighbour's wall is oblique to the
    sampled one, the walk over-reads the gap by the secant of the obliquity, and
    past the obliquity where secant times gap leaves the reach it strikes
    nothing and the sample asks nothing at all. Both fail toward too coarse, and
    the extremal pair bounds both: it is kept, and still pins the point where
    the gap is smallest. A pair whose walls this walk cannot read is covered by
    the extremal pair alone. On a parallel run, which is the case this exists
    for, the walk is exact.

    The face is sampled at the extremal distance. That distance sets the finest
    cell this gap can ask for anywhere, so between samples the field peaks one
    grading step above what they ask, which is the rule :func:`_sampling`
    states. A face longer than :data:`MAX_SAMPLES` times the distance is
    sampled more coarsely than that, and the field ripples between samples by
    correspondingly more. Each walked demand carries the spacing its lattice
    realised, so the report can state that residual over the finished grid
    without walking anything itself.

    The body sampled has to be one whose neighbour can be walked into.
    Containment answers nothing for an area, so a sheet is only ever reached by
    sampling it, never by marching at it. A sheet carries no solids to settle a
    winding against either, and is walked both ways instead. Where either
    ordering would do, the measured body is sampled: a box's flat walls are
    already pinned, and the grid has to find the walls of the body carried as
    geometry. Where both are measured, the body whose bounding box has the
    smaller diagonal is sampled, since it covers the shared stretch of the gap
    in fewer samples.
    A pair of sheets can be walked from neither side, and the extremal pair is
    all this measures for one.
    """
    ways = [
        (sampled, against)
        for sampled, against in ((one, other), (other, one))
        if list(getattr(against.shape, "Solids", ()) or ())
    ]
    if not ways:
        return
    sampled, against = ways[0]
    if len(ways) > 1 and (
        not sampled.measured
        or (against.measured and _diagonal(against.shape) < _diagonal(sampled.shape))
    ):
        sampled, against = ways[1]
    targets = list(getattr(against.shape, "Solids", ()) or ())
    own = list(getattr(sampled.shape, "Solids", ()) or ())
    march = MARCH_STEP * cap
    for face in _faces(sampled.shape):
        if _boxes_further_apart_than(face, against.shape, reach):
            continue
        stations, spaced = _sampling(face, distance)
        for u, v in stations:
            try:
                point, normal = face.valueAt(u, v), face.normalAt(u, v)
            except Exception:
                # A face can carry a point its own parameterisation cannot
                # answer for - a pole, a seam. One sample is not the face.
                if spend is not None:
                    spend.refused.declined(source)
                continue
            if spend is not None:
                spend.refused.answered(source)
                spend.walked += 1
            crossing = _across_gap(own, targets, point, normal, reach, march, spend)
            if crossing is None:
                continue
            width, step = crossing
            place, foot = _xyz(point), _xyz(point + step * width)
            direction = corner(far - near for near, far in zip(place, foot))
            for where in (place, foot):
                yield Feature(
                    thickness=width,
                    normal=direction,
                    lower=where,
                    upper=where,
                    source=source,
                    sampled_at=spaced,
                )


def _across_gap(
    own: Sequence[Any],
    targets: Sequence[Any],
    point: Any,
    normal: Any,
    reach: float,
    march: float,
    spend: Spend | None = None,
) -> tuple[float, Any] | None:
    """How far ``point`` sits from the neighbour along its face's normal.

    Marches away from the surface until a containment probe lands inside the
    neighbour. ``None`` where no crossing is found within ``reach``, which covers
    every stretch of surface that faces away from the neighbour or is simply
    further from it than any demand could bind.
    ``None`` as well where the face is interior to its own body: an interior
    face is not a boundary, and there is no gap to measure from it.

    A body with no solids of its own - a sheet - has no inside to settle the
    winding against, so both ways off it are walked and the nearer crossing is
    the answer. A face belonging to no solid of its own compound answers
    outside to both probes, and is walked the way its raw winding points. The
    direction is then unsettled, but whatever the walk strikes is a true
    first-crossing distance from drawn geometry, and a miss asks nothing.
    """
    length = math.sqrt(sum(value * value for value in _xyz(normal)))
    if length <= 0.0:
        return None
    step = normal * (1.0 / length)
    if own:
        if _inside(own, point + step * WINDING_PROBE, spend):
            step = step * -1.0
            if _inside(own, point + step * WINDING_PROBE, spend):
                return None
        ways: tuple[Any, ...] = (step,)
    else:
        ways = (step, step * -1.0)
    found: tuple[float, Any] | None = None
    for step in ways:
        clear, struck = 0.0, None
        for i in range(1, int(math.ceil(reach / march)) + 1):
            spot = min(march * i, reach)
            if _inside(targets, point + step * spot, spend):
                struck = spot
                break
            clear = spot
        if struck is None:
            continue
        for _ in range(MAX_HALVINGS):
            if struck - clear <= CHORD_TOLERANCE * struck:
                break
            middle = 0.5 * (clear + struck)
            if _inside(targets, point + step * middle, spend):
                struck = middle
            else:
                clear = middle
        width = 0.5 * (clear + struck)
        if found is None or width < found[0]:
            found = (width, step)
    return found


def _diagonal(shape: Any) -> float:
    box = shape.BoundBox
    return math.dist(
        (float(box.XMin), float(box.YMin), float(box.ZMin)),
        (float(box.XMax), float(box.YMax), float(box.ZMax)),
    )


def _curvatures(
    body: Body,
    cap: float,
    edge_size: float | None,
    stations: _Stations,
    fidelity: float = SURFACE_FIDELITY,
    spend: Spend | None = None,
) -> Iterator[Feature]:
    """How finely a curved face has to be followed, and where.

    The reciprocal of the larger principal curvature bounds the medial ball at
    a point, so twice it is a length the criteria can be stated against. A plane
    curves not at all and bounds nothing, which is correct: a flat face says
    nothing about the solid behind it - so a face the surface says is planar is
    skipped whole rather than sampled, and :func:`_curves` gives the bound on
    what that can cost: the demand such a face would have raised is one
    ``reach`` drops.

    That length is asked for under one of these claims, and which one holds is
    read off the radius rather than declared:

    * **Fidelity** - ``fidelity`` of the radius the surface curves through. It
      binds on a body drawn round rather than thin.
    * **The edge floor** - never finer than a metal edge asks for. A corner is
      what a fillet becomes as its radius goes to zero, and a corner asks the
      edge size over ``sqrt(2)`` across an axis-aligned edge, so a fillet asking
      for a fraction of its vanishing radius would cost unboundedly more than
      the shape it is on its way to being. The axis-aligned corner is the
      coarsest of them - :func:`sizing.edge` asks less at a turned tangent - so
      this is a bound rather than the demand of every corner a fillet could
      become.
    * **Connection** - never coarser than the length itself. That length is an
      upper bound on the medial radius rather than the metal's own
      cross-section, so a cell fitting inside it fits inside the metal. That is
      the direction that keeps a body thinner than an edge from being sampled
      into something electrically open.
    """
    reach = fits_inside(cap)
    # A corner asks the edge criterion - the edge size over sqrt(2) on each axis
    # across an axis-aligned edge - and this is the thickness whose own
    # connection demand asks exactly that.
    floor = 0.0 if edge_size is None else fits_inside(edge_size / math.sqrt(2.0))
    for number, face, pairs in stations:
        if not _curves(face):
            continue
        source = f"{body.label!r} curving on face {number}"
        for u, v in pairs:
            sharpest = _sharpest_at(face, u, v, spend, source)
            if sharpest is None:
                continue
            diameter = 2.0 / sharpest
            thickness = min(diameter, max(fidelity * diameter, floor))
            if thickness >= reach:
                continue
            place = _xyz(face.valueAt(u, v))
            yield Feature(
                thickness=thickness,
                normal=None,
                lower=place,
                upper=place,
                source=source,
            )


def _rims(
    body: Body,
    cap: float,
    edge_size: float | None,
    fidelity: float = SURFACE_FIDELITY,
    spend: Spend | None = None,
) -> Iterator[Feature]:
    """How finely a sheet's outline has to be followed, and where.

    A sheet's face is flat, so :func:`_curvatures` reads nothing off it. What
    curves on a round pad is its rim, and openEMS samples that the same way it
    samples a wall. The connection criterion is not among the claims here, and
    that is the difference between this and :func:`_curvatures`. Both criteria,
    and what a hole's rim asks, are in
    docs/internals/feature-size.md#a-rim-is-not-a-cross-section.

    A curve answers its curvature as a magnitude, where a face answers two
    signed principal ones. Measured on FreeCAD 1.1.1: reversing a circle and
    running a spline through its own inflection both leave the curvature
    non-negative. So there is no sharpest-of-two to take here, and nothing to
    take an absolute value of.
    """
    reach = fits_inside(cap)
    # The same floor a fillet gets, for the same reason - the rim of a
    # tightening pad is on its way to being an edge - and with the same caveat.
    # See :func:`_curvatures`.
    floor = 0.0 if edge_size is None else fits_inside(edge_size / math.sqrt(2.0))
    # Sampled at the metal edge size rather than at the coarser size a rim
    # usually settles on. Where a rim's curvature varies, the tighter end has to
    # be caught, and a sample spaced at the demand it produced would walk past
    # it.
    spacing = cap if edge_size is None else edge_size
    source = f"{body.label!r} rim curving"
    for edge, faces in _joins(body.shape):
        if len(faces) != 1:
            continue
        for where in _along(edge, spacing):
            try:
                curvature = float(edge.curvatureAt(where))
            except Exception:
                # A curve can carry a point its own parameterisation cannot
                # answer for - a cusp, a degenerate segment. One point is not
                # the rim.
                if spend is not None:
                    spend.refused.declined(source)
                continue
            if spend is not None:
                spend.refused.answered(source)
            if curvature <= 0.0:
                continue
            thickness = max(fidelity * 2.0 / curvature, floor)
            if thickness >= reach:
                continue
            place = _xyz(edge.valueAt(where))
            yield Feature(
                thickness=thickness,
                normal=None,
                lower=place,
                upper=place,
                source=source,
            )


def _thicknesses(
    body: Body, cap: float, stations: _Stations, spend: Spend | None = None
) -> Iterator[Feature]:
    """How thick the metal is across itself, and where.

    This is a connection demand. A cell that fits inside the metal is a cell
    the sampling cannot open it at. It is stated at its worst direction, since
    a conductor thinner than a cell fails by being sampled into a
    vertex-adjacent chain rather than in any one direction.

    The demand is spent in full, with no floor under it, unlike the fidelity
    demand in :func:`_curvatures`, and at the cost of what a tapering tip does
    to the timestep. Why a cross-section may not be floored when fidelity may,
    and what that costs, are in
    docs/internals/feature-size.md#floors-and-which-demands-may-be-given-up.
    """
    reach = fits_inside(cap)
    # Where the run begins is not read, unlike in :func:`_element_counts`. A
    # connection demand is a point constraint that ramps outward from wherever it
    # is stated, so sliding it along the normal by the sagitta of one facet moves
    # what it asks for by the grading slope times that offset. A count is stated
    # across the run instead, and there the end it starts from decides whether
    # the span lies on the material at all.
    for number, point, _, chord, _begins in _chords(body, reach, stations, spend):
        place = _xyz(point)
        yield Feature(
            thickness=chord,
            normal=None,
            lower=place,
            upper=place,
            source=f"{body.label!r} thickness on face {number}",
        )


def _element_counts(
    body: Body, cap: float, across: int, spend: Spend | None = None
) -> Iterator[Feature]:
    """How many cells span a dielectric across its own thickness, and where.

    The count a box gets, given to a shape a box cannot describe. A box states
    its thickness as an extent per axis, a triangulation states nothing, and the
    box bounding a bent board is as deep as the bend. So the chord is measured,
    and the count is stated against it along the normal it was measured on.

    Nothing here is a claim about where the boundary went. openEMS averages a
    dielectric over the cell rather than sampling it at a point, so a dielectric
    does not staircase the way a conductor does and is held to no fidelity.

    The face is sampled at the coarsest cell rather than at the size the demands
    will ask for, which is the one exception to the rule :func:`_spacing`
    states. The demand covers the chord rather than sitting at its end the way a
    cross-section does. Why each, and what the reach costs a void or a taper,
    are in docs/internals/feature-size.md#sampled-coarsely-on-purpose.
    """
    for number, point, step, chord, begins in _chords(
        body, across * cap, _Stations(body, cap), spend
    ):
        # From where the material starts rather than from the sample. The chord
        # is the run through the point, and a sample stands off its own
        # triangulation either way - outside it on a convex face and inside it
        # on a concave one - so a span laid from the sample is the layer's own
        # span slid along the normal by that much.
        near = point + step * begins
        place, far = _xyz(near), _xyz(near + step * chord)
        yield Feature(
            thickness=chord,
            # The chord as walked rather than the face's normal. A kernel winds
            # a face either way, so which side the layer lies on is settled by
            # the walk and not by the normal.
            normal=corner(end - start for start, end in zip(place, far)),
            lower=corner(min(start, end) for start, end in zip(place, far)),
            upper=corner(max(start, end) for start, end in zip(place, far)),
            across=across,
            source=f"{body.label!r} across its thickness on face {number}",
        )


def _chords(
    body: Body, reach: float, stations: _Stations, spend: Spend | None = None
) -> Iterator[tuple[int, Any, Any, float, float]]:
    """Each face sample, the way the body runs inward from it, and how far.

    Both questions a thickness answers read the same chord - whether a
    conductor still conducts across itself, and how many cells span a
    dielectric across itself - so it is walked once and they cannot come to
    disagree about how thick the body is.

    The chord is read off the triangulation the body reaches the engine as,
    which is one line cast rather than a walk: the crossings come back sorted
    and the first that leaves the material ends the run. Where a sample sits is
    still decided on the drawn face, because the sampling is spaced at the cell
    the demand will ask for and a large flat face carries two triangles and
    many samples. The kernel is asked for every sample on a face before any of
    them is cast for, and the samples of one face are walked together.

    The step and where the run begins come back as well as the length. Which
    way a chord ran is a fact about the drawing rather than about the face: a
    caller cannot tell from the normal which side of the sample the material is
    on, so the direction is settled here from the crossing that answered. And
    the run need not begin at the sample - a sample on a convex face stands
    outside its own triangulation, and one on a concave face stands inside it -
    so a caller stating a demand across the run places it on the material.

    A sheet is an area and has nothing to measure. It reaches the engine as a
    zero-thickness primitive, and no line meets a surface that encloses
    nothing.
    """
    if body.sheet:
        return
    surface = prepared(body.vertices, body.faces)
    if surface is None:
        return
    for number, face, pairs in stations:
        # Named for the walk rather than for a demand. This is walked for
        # :func:`_thicknesses` and for :func:`_element_counts`, which label what
        # they raise differently, so there is no one demand string to file a
        # lost station under.
        source = f"{body.label!r} through face {number}"
        points, normals = [], []
        for u, v in pairs:
            try:
                place, normal = face.valueAt(u, v), face.normalAt(u, v)
            except Exception:
                # A face can carry a point its own parameterisation cannot
                # answer for - a pole, a seam. One sample is not the face.
                if spend is not None:
                    spend.refused.declined(source)
                continue
            if spend is not None:
                spend.refused.answered(source)
            points.append(place)
            normals.append(normal)
        walked = _chords_at(surface, points, normals, reach, spend)
        for point, normal, found in zip(points, normals, walked, strict=True):
            if found is None:
                continue
            chord, scale, begins = found
            yield number, point, normal * scale, chord, begins


def _chords_at(
    surface: Surface,
    points: Sequence[Any],
    normals: Sequence[Any],
    reach: float,
    spend: Spend | None = None,
) -> list[tuple[float, float, float] | None]:
    """How far the body runs through each point, and which way off its own
    normal, as a multiple of that normal.

    It walks the material rather than the metal. A conductor's cross-section and
    a dielectric's element count are the same line asked for different reasons.

    The direction comes back as what to scale the normal by, so a caller holding
    the kernel's own vector gets a unit step from it. It is settled by the
    crossings rather than read off the face, so which side the material lies on
    does not turn on how the kernel wound the face, and a stand-in winding the
    other way measures the same thickness.

    ``None`` for a point where the body runs further than ``reach``, which is
    every body too thick for the answer to bind, and ``None`` as well where
    neither way off the surface is material.

    The face's samples are handed over together. A cast pays the fixed price of
    one call whatever it looks at, and a face's samples all ask the same
    surface, so what the walk spends follows the branches its samples take
    rather than the samples themselves.
    """
    lengths = [math.sqrt(sum(value * value for value in _xyz(normal))) for normal in normals]
    walked = chords_at(
        surface,
        [_xyz(point) for point in points],
        [_xyz(normal) for normal in normals],
        reach,
        spend,
    )
    return [
        None if found is None else (found[0], found[1] / length, found[2])
        for found, length in zip(walked, lengths, strict=True)
    ]


def _inside(solids: Sequence[Any], point: Any, spend: Spend | None = None) -> bool:
    """Whether any of these solids contains the point.

    Each is asked separately because a compound answers for one member only -
    a point inside its second solid and no other comes back outside.

    One place the CAD kernel is asked where a point is, and so the one place
    the gap walk's cost is counted. What is charged is the point, since that is
    what the walk decides to ask about; how many solids answer it is a property
    of the drawing.
    """
    if spend is not None:
        spend.probed += 1
    return any(solid.isInside(point, CONTAINMENT_TOLERANCE, True) for solid in solids)


def _sharp_joins(body: Body, edge_size: float, spend: Spend | None = None) -> Iterator[Feature]:
    """Edges where the surface stops being smooth, on a conductor.

    A join whose two faces share a tangent plane carries no singularity however
    tightly it curves, so this looks for the disagreement between the normals
    rather than for the curvature on either side. An edge belonging to one face
    only is a boundary of an open shell and is not a join at all.

    An edge is a curve and is followed along its whole length. A circular rim is
    one edge running right round a shape, and sampling it once refines the grid
    at a single point on the rim, which shows up as a mesh that followed the
    object on one side of it.

    Each sample asks across the edge, over the whole plane of directions square
    to it. The field there varies with distance from the edge and is constant
    along a straight one, so cells packed along its length resolve nothing. A
    demand along any one direction in that plane would hold that direction
    alone, and what the rest got would turn on how the join lies on the grid:
    two drawings of one wedge differing only by a rotation would be meshed
    apart, and the further the two faces have closed on each other, the further
    apart. Stated over the plane, the demand is the same however the part was
    turned.

    The plane is named by the line the two faces meet along, taken as the cross
    product of the two normals. Each face contains the edge, so both normals are
    square to it, and the product is best conditioned exactly where the join is
    sharpest. Two faces exactly back to back have no such line - the product
    cancels, and a solid of no thickness meets itself everywhere - and each face
    is then asked along its own normal, the one direction still known.
    """
    sharp = math.radians(SHARP_DEGREES)
    source = f"{body.label!r} edge"
    for edge, faces in _joins(body.shape):
        if len(faces) == 1 and body.sheet:
            yield from _outline(body, edge, edge_size, spend)
            continue
        if len(faces) != 2:
            continue
        for where in _along(edge, edge_size):
            try:
                # Kept as the kernel's own point type on the way to the faces,
                # and turned into numbers only for the feature. Building a
                # vector here would mean knowing which class to build. Inside
                # the guard because a point the curve will not place is the
                # same event as a normal its faces will not give, and the two
                # are asked of one station.
                meeting = edge.valueAt(where)
                normals = [_normal_at(face, meeting) for face in faces]
            except Exception:
                # A point on a join that its own faces cannot answer for is one
                # this cannot judge, and guessing sharp would refine it.
                if spend is not None:
                    spend.refused.declined(source)
                continue
            if spend is not None:
                spend.refused.answered(source)
            angle = _angle_between(*normals)
            if angle < sharp:
                continue
            place = _xyz(meeting)
            along = _cross(*normals)
            if _length(along) > 0.0:
                yield Feature(
                    thickness=edge_size,
                    normal=None,
                    lower=place,
                    upper=place,
                    source=source,
                    tangent=along,
                )
                continue
            for normal in normals:
                yield Feature(
                    thickness=edge_size,
                    normal=normal,
                    lower=place,
                    upper=place,
                    source=source,
                )


def _outline(
    body: Body, edge: Any, edge_size: float, spend: Spend | None = None
) -> Iterator[Feature]:
    """Where a sheet's metal stops, which is an edge like any other.

    A sheet is a conductor with a boundary rather than a closed surface, so the
    edge that matters has one face beside it and not two. The field singularity
    at it is the one a solid's edge carries. Without this, a sheet drawn as an
    outline reaches the engine and changes no grid line: it has no box for the
    thirds rule to work from, and no second face to disagree with.

    So the demand is the one a join makes, over the whole plane of directions
    square to the edge. A demand along any one direction in that plane holds
    that direction alone, and what the rest gets turns on how the shape lies on
    the grid. The axis out of the sheet's plane is part of that plane. The metal
    has no thickness to resolve there, but the field is not the metal and wraps
    around the edge.

    An outline is curved or diagonal wherever it reaches here at all - a flat
    conductor whose boundary runs along the axes is cut into rectangles and
    arrives as boxes - so :func:`_rims` covers the variation along the edge that
    this criterion says nothing about.
    """
    source = f"{body.label!r} outline"
    for where in _along(edge, edge_size):
        try:
            tangent = _xyz(edge.tangentAt(where))
        except Exception:
            # A point whose own curve cannot give a tangent - a cusp, a
            # degenerate segment. One point is not the outline.
            if spend is not None:
                spend.refused.declined(source)
            continue
        if spend is not None:
            spend.refused.answered(source)
        if _length(tangent) <= 0.0:
            continue
        place = _xyz(edge.valueAt(where))
        yield Feature(
            thickness=edge_size,
            normal=None,
            lower=place,
            upper=place,
            source=source,
            tangent=tangent,
        )


def _cross(
    one: tuple[float, float, float], other: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        one[1] * other[2] - one[2] * other[1],
        one[2] * other[0] - one[0] * other[2],
        one[0] * other[1] - one[1] * other[0],
    )


def _length(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


# ---------------------------------------------------------------- the kernel


def _faces(shape: Any) -> Sequence[Any]:
    return list(getattr(shape, "Faces", ()) or ())


def _edges(shape: Any) -> Sequence[Any]:
    return list(getattr(shape, "Edges", ()) or ())


def _curves(face: Any) -> bool:
    """Whether a face has a curvature to be asked for.

    The surface is asked whether it is planar, which is a question about the
    geometry and not about the class the geometry arrived as: a spline lying in
    a plane answers the same as a plane does, and a shape that came through a
    file carrying no type answers as well as one that was drawn here.

    A planar surface turns in no direction anywhere it is defined, so a face
    trimmed out of one has no radius at any point on it and nothing measured
    from a curvature there is a length of the shape.

    **What the kernel answers on such a face is a rounding rather than a zero.**
    A cap re-fitted as a spline comes back curving by about 1e-13 per mm, and
    the guards below cannot see that: each is stated against the sharpest bend
    at the same sample, so where every direction is the residue of one
    computation none of them is below the others. Skipping the face is therefore
    a different answer and not the same one reached sooner - a smaller curved
    area, and a departure reported over it that is correspondingly larger.

    **The question is a distance and not a curvature.** ``isPlanar`` asked
    without a tolerance holds a face to the kernel's own confusion tolerance,
    which is an absolute departure from a fitted plane. An analytic surface is
    settled on what it is and answers false at any radius; a free-form one - a
    ruled surface, a filled face, anything that arrived through a file - is
    measured, and a departure fixed in millimetres is a sharper radius the
    smaller the patch. So a face may be called planar and still carry a radius.

    What bounds that is the demand such a face would raise. A departure ``s``
    over a patch ``L`` across is a radius near ``L * L / (8 * s)``, and
    :func:`_curvatures` asks for :data:`SURFACE_FIDELITY` of twice that, which
    is beyond ``reach`` - and so dropped - unless ``L`` is under a few microns
    on a millimetre-sized cell. Nothing the mesher acts on moves, which is what
    ``tests/test_corpus_geometry.py::TestEveryShapeGetsAnHonestVerdict::test_the_skip_costs_the_mesher_no_demand``
    holds it to over the corpus.

    Asked once a face, because the answer is a property of the surface and not
    of any station on it. A face that cannot answer is asked.
    """
    try:
        planar = getattr(getattr(face, "Surface", None), "isPlanar", None)
        return planar is None or not planar()
    except Exception:
        return True


def _sharpest_at(
    face: Any, u: float, v: float, spend: Spend | None = None, source: str = ""
) -> float | None:
    """The sharpest principal curvature at one parameter pair, as a magnitude.

    ``None`` where the surface cannot answer for the pair - a pole, a seam - or
    where it does not curve there at all. One sample is not the face. A
    curvature answered as not-a-number goes with them: it fails a test for being
    positive, where a test for being at most zero would let it through.

    The station is counted here rather than by the caller, because those
    ``None`` are different events and only this can tell them apart. A surface
    that declined the pair read nothing there. One that answered anything read
    the place, whether it answered a curvature, a zero or a value that is not a
    number, and is counted as having answered. ``source`` is what the demand
    would have been labelled with.
    """
    try:
        curvatures = face.curvatureAt(u, v)
    except Exception:
        if spend is not None:
            spend.refused.declined(source)
        return None
    if spend is not None:
        spend.refused.answered(source)
    sharpest = max(abs(float(value)) for value in curvatures)
    return sharpest if sharpest > 0.0 else None


def _lattice(face: Any, count: int) -> Iterator[tuple[float, float]]:
    """A fixed lattice over one face's parameter range, cell-centred."""
    low_u, high_u, low_v, high_v = (float(value) for value in face.ParameterRange)
    for i in range(count):
        for j in range(count):
            yield (
                low_u + (high_u - low_u) * (i + 0.5) / count,
                low_v + (high_v - low_v) * (j + 0.5) / count,
            )


def _bend_at(edge: Any, t: float) -> float | None:
    """The radius one edge bends through at one parameter, in mm.

    ``None`` where the curve cannot answer for the parameter, where it does not
    bend there at all, or where the curvature comes back not-a-number, for the
    reason :func:`_sharpest_at` gives. One sample is not the edge.
    """
    try:
        bend = abs(float(edge.curvatureAt(t)))
    except Exception:
        return None
    return 1.0 / bend if bend > 0.0 else None


def _across(edge: Any, count: int) -> Iterator[float]:
    """``count`` parameters spanning one edge, its own ends included.

    An edge closed on itself - a whole circle, a whole ellipse - carries one
    point at both ends of its range, so it is looked at one place fewer than was
    asked for. Nothing is missed by that: the two ends are the same place.

    A count below two has no ends to space and is answered by the middle.
    """
    low, high = float(edge.FirstParameter), float(edge.LastParameter)
    if count < 2:
        yield 0.5 * (low + high)
        return
    for i in range(count):
        yield low + (high - low) * i / (count - 1)


def bends_through(shape: Any, count: int = CURVATURE_SAMPLES) -> tuple[float, float] | None:
    """The sharpest and the widest radius the shape's edges bend through, in mm.

    :func:`curvature` asked of an outline instead of a surface. It is a
    separate reading because a flat sheet's face does not curve at all while its
    boundary routinely does - a round pad, a clearance in a ground plane, the
    counter of a letter. ``None`` where nothing bends, which is every rectangle
    and every outline made of straight runs. A straight edge contributes nothing
    rather than an infinite radius, so an outline mixing straight runs with arcs
    answers about its arcs.

    The ends are sampled, where :func:`curvature` steps around a face's own
    parameter boundary. An edge is trimmed at a vertex, and a conic cut there
    carries its sharpest point exactly at that end, so a lattice held inside the
    range would answer wider than the curve. That is the costly direction: a
    request is then held against a radius the drawing does not have. An edge has
    ends that evaluate, where a face has a pole and a seam that do not.

    Both ends are still estimates and the sharpest is still the biased one, for
    the reason :attr:`Curvature.sharpest` gives, so a caller may bound a request
    with this and may not make a target of it.
    """
    sharpest = widest = None
    for edge in _edges(shape):
        for t in _across(edge, count):
            here = _bend_at(edge, t)
            if here is None:
                continue
            sharpest = here if sharpest is None else min(sharpest, here)
            widest = here if widest is None else max(widest, here)
    if sharpest is None or widest is None:
        return None
    return sharpest, widest


#: How far under the sharpest bend at one sample a principal curvature has to
#: sit before it is a direction the surface does not turn in.
#:
#: The threshold is single precision's own epsilon, ``2**-23``. A value below
#: the sharpest bend by more than that is the residue of a zero the kernel
#: computed rather than a number it found. A value above it is a direction so
#: shallow that it puts nothing into a departure whichever way it is counted.
#:
#: It is stated against the other curvature at the same point rather than
#: against a length, because those two are the only figures in hand that share a
#: scale. A bound taken against the body is wrong the other way: a blend whose
#: radius is larger than the part it is cut into still bends within it.
FLAT_AGAINST_SHARPEST = 2.0**-23

#: The kernel's own word for a face wound against the solid it bounds. A shell's
#: bore is one, and its principal curvatures come back with the sign the face
#: carries rather than the sign the metal behind it does.
_REVERSED = "Reversed"


@dataclass(frozen=True)
class Curvature:
    """What a shape's curved faces cover, and which ways they bend.

    :param area: The summed area of the faces that curve at all, in mm^2. Zero
        where nothing curves. It is the drawn area rather than the
        triangulation's. An arc is longer than the chord standing for it, so
        this is the larger of the two, and a figure divided by it is the
        smaller. The gap between the two is second order in the departure over
        the radius.
    :param convex: Whether any of them curves away from the material - a wall,
        a rod, the outside of a shell. Its centre of curvature is in the metal,
        so a chord across it lies inside the drawing.
    :param concave: Whether any curves into the material - a bore, a fillet in a
        corner, the inside of a shell. Its centre of curvature is outside the
        metal, so a chord across it lies outside the drawing.

    :param sharpest: The smallest radius any face curves through, in mm, or
        ``None`` where nothing curves - a box, a sheet, a rectilinear boolean.
        It is an estimate and it is the biased one: a minimum over samples, so
        it can only fall as samples are added, and the lattice is cell-centred,
        so a face whose sharpest point sits on its own parameter boundary is
        stepped around by construction, which is where the truncated small end
        of a cone sits. A caller may state a bound against it, wide enough to
        absorb an estimate good only to a factor, and may not make a target of
        it.
    :param widest: The largest such radius, or ``None``. An estimate as well,
        and the unbiased direction: it can only rise as samples are added.

    A quantity summed over the whole boundary cannot state the case where both
    hold: its contributions then have opposite signs. A shape bounded by planes
    has neither.
    """

    area: float
    convex: bool
    concave: bool
    sharpest: float | None
    widest: float | None

    @property
    def both_ways(self) -> bool:
        """Whether one shape carries surfaces curving each way. A bore alone
        does not qualify: a pair of opposite displacements is what nets."""
        return self.convex and self.concave

    @property
    def through(self) -> tuple[float, float] | None:
        """The two radii as a pair, or ``None`` where nothing curves.

        The pair a request is bounded against, which is what a caller sizing a
        triangulation wants and all it wants.
        """
        if self.sharpest is None or self.widest is None:
            return None
        return self.sharpest, self.widest


def curvature(shape: Any, count: int = CURVATURE_SAMPLES) -> Curvature:
    """How much of a shape curves, which ways it bends, and through what radii.

    This is the only sampling of a surface this layer has before any grid
    exists, and every question asked of it is answered off the one walk. A
    caller sizing a triangulation wants the two radii and reads
    :attr:`Curvature.through`; one measuring a departure wants the area and the
    two flags. They are one reading because they are one lattice and one kernel
    call.

    The sign is turned to face outward. ``curvatureAt`` answers in the face's
    own frame, so a face the kernel wound inward - the bore of a shell, the hole
    through a block - reports the opposite of how the material behind it curves,
    and a hollow sphere would read as curving one way.

    Which sign is which follows from the normal rather than from a convention. A
    principal curvature is signed against the surface normal, and an outward
    normal points out of the metal. So a face whose centre of curvature is in
    the metal, which is what convex means, answers negative, and a bore answers
    positive. A chord across the first therefore lies inside the drawing, and a
    chord across the second lies outside it.

    A sample the surface cannot answer for is dropped, as it is for a radius: a
    pole and a seam are places the parameterisation fails rather than places the
    shape is flat.

    A face the surface says is planar is not walked at all, and what that
    changes is in :func:`_curves`.
    """
    area = 0.0
    convex = concave = False
    sharpest = widest = None
    for face in _faces(shape):
        if not _curves(face):
            continue
        flip = -1.0 if str(getattr(face, "Orientation", "")) == _REVERSED else 1.0
        curves = False
        for u, v in _lattice(face, count):
            try:
                curvatures = face.curvatureAt(u, v)
            except Exception:
                continue
            # A direction the surface does not turn in is analytically zero,
            # and a kernel that computed it hands back the residue of its own
            # arithmetic instead - a cone's straight direction, a cylinder's
            # along its axis. So it counts only against the sharpest bend at
            # its own sample, which is the one number in hand carrying that
            # scale.
            here = max((abs(float(value)) for value in curvatures), default=0.0)
            flat = FLAT_AGAINST_SHARPEST * here
            for value in curvatures:
                bend = flip * float(value)
                # Not-a-number fails both comparisons and so counts as
                # neither. :func:`_sharpest_at` makes the same guard.
                if bend < -flat:
                    curves = convex = True
                elif bend > flat:
                    curves = concave = True
            # The radius, under the same guard :func:`_sharpest_at` applies: a
            # curvature that is not positive is a sample the surface answered
            # nothing at rather than one it bends infinitely wide at.
            if here > 0.0:
                radius = 1.0 / here
                sharpest = radius if sharpest is None else min(sharpest, radius)
                widest = radius if widest is None else max(widest, radius)
        if curves:
            area += float(face.Area)
    return Curvature(area=area, convex=convex, concave=concave, sharpest=sharpest, widest=widest)


def _joins(shape: Any) -> Iterator[tuple[Any, list[Any]]]:
    """Every edge of a shape, with the faces that meet along it.

    Built from the faces rather than asked of the shape. Asking takes ``Part``,
    and nothing here imports FreeCAD. An edge is identified by the kernel's own
    hash, so the same edge reached through two faces is one edge.
    """
    edges: dict[int, tuple[Any, list[Any]]] = {}
    for face in _faces(shape):
        for edge in getattr(face, "Edges", ()) or ():
            key = int(edge.hashCode())
            edges.setdefault(key, (edge, []))[1].append(face)
    yield from edges.values()


def _along(edge: Any, size: float) -> Iterator[float]:
    """Parameters along one edge, spaced by the cell size being asked for.

    The spacing comes out of the field rather than being chosen. Two point
    demands for cells of ``s``, a distance ``s`` apart, leave the field peaking
    at ``s(1 + g/2)`` between them, and ``g`` is ``ln(max_ratio)``, which is
    below ``max_ratio - 1`` for every ratio above one. So consecutive samples
    are one grading step apart at most.

    That decides the count. Where the samples fall is even in the edge's
    parameter, which is even in space only where the parameterisation is, so on
    a spline or an ellipse the spacing varies about it and the bound is a target
    rather than a guarantee. That is acceptable for the same reason it is
    acceptable on a face: the demand is a refinement, and being denser than
    intended in places costs constraints rather than correctness.

    The count is bounded, because an edge can be long and the size asked for
    small. A rim longer than :data:`MAX_EDGE_SAMPLES` times the cell size is
    sampled more coarsely than that, and the field ripples between the samples
    by correspondingly more. It stays a refinement of the whole rim, where
    sampling an edge once refined one point of it.
    """
    low, high = float(edge.FirstParameter), float(edge.LastParameter)
    length = float(getattr(edge, "Length", 0.0))
    steps = (
        1
        if size <= 0 or length <= 0
        else min(MAX_EDGE_SAMPLES, int(length / size + COUNT_SLACK) + 1)
    )
    for i in range(steps):
        yield low + (high - low) * (i + 0.5) / steps


def _spacing(cap: float, edge_size: float | None) -> float:
    """How far apart to sample a face, for the measurements read off its surface.

    It is the size a demand off a face will usually settle on: the field between
    two samples stays where they put it only where they are about one such cell
    apart. A body thin enough for the connection bound to take over asks for
    less than this and is sampled more coarsely than it asked, the way a face
    too large for :data:`MAX_SAMPLES` is.

    :func:`_element_counts` samples at the cap instead, and says on itself why.
    """
    return cap if edge_size is None else edge_size


class _Stations:
    """Each of a body's faces and the lattice laid across it, laid once.

    The curvature and the cross-section are read at the same places on the same
    faces, and placing a lattice on its face costs a reading of the face. So
    the lattice belongs to the body being measured rather than to either
    measurement, and neither can lay a different one.

    Laid on first reading, which is what keeps it free where nothing reads it.
    :func:`_element_counts` builds one and :func:`_chords` returns ahead of it
    on a sheet and on a body that reached the mesher without a triangulation. A
    metal body's is always read, because :func:`_curvatures` walks every face
    whatever the face answers.
    """

    def __init__(self, body: Body, spacing: float) -> None:
        self._body = body
        self._spacing = spacing
        self._laid: list[tuple[Any, list[tuple[float, float]]]] | None = None

    def __iter__(self) -> Iterator[tuple[int, Any, list[tuple[float, float]]]]:
        if self._laid is None:
            self._laid = [
                (face, _samples(face, self._spacing)) for face in _faces(self._body.shape)
            ]
        return ((number, face, pairs) for number, (face, pairs) in enumerate(self._laid))


def _samples(face: Any, spacing: float) -> list[tuple[float, float]]:
    """The sample lattice alone, for a caller with no use for its spacing."""
    return _sampling(face, spacing)[0]


def _sampling(face: Any, spacing: float) -> tuple[list[tuple[float, float]], float]:
    """Parameter pairs across one face, about ``spacing`` apart in space, and
    the coarsest spacing the lattice realises.

    The spacing has to be decided in millimetres and applied in parameters. A
    parameterisation carries no scale of its own: one unit of ``u`` is a radian
    on a cylinder and a millimetre on a plane. That costs a count split per
    direction, and it saves sampling a rod's circumference as finely as its
    length. The two differ by the aspect ratio of the face, which is large on a
    thin rod.

    The caller passes the cell size the demands will ask for. :func:`_along`
    follows the same rule, for the same reason: two point demands for cells of
    ``s``, a distance ``s`` apart, leave the field peaking one grading step
    above ``s`` between them. Sampling instead at the coarsest cell would place
    a fine demand at a few points on the face and let the field climb back to
    bulk between them, which is a surface followed in patches.

    The spacing is a target rather than a guarantee, twice over. It is even in
    the face's parameters, and so even in space only where the parameterisation
    is. And a face asking for more than :data:`MAX_SAMPLES` in a direction gets
    that many. The realised spacing comes back beside the lattice - the length
    each direction measured over the count it got, worst of the two - so a
    caller reporting what the sampling could promise reads the lattice actually
    laid rather than the target.

    This samples a curvature that varies. It does not search for the extremum:
    a crease narrower than the spacing can still be stepped over, and checking
    the finished grid is what catches that.
    """
    low_u, high_u, low_v, high_v = (float(value) for value in face.ParameterRange)
    middle_u, middle_v = 0.5 * (low_u + high_u), 0.5 * (low_v + high_v)
    across, width = _steps(face, (low_u, middle_v), (high_u, middle_v), spacing)
    along, height = _steps(face, (middle_u, low_v), (middle_u, high_v), spacing)
    pairs = [
        (
            low_u + (high_u - low_u) * (i + 0.5) / across,
            low_v + (high_v - low_v) * (j + 0.5) / along,
        )
        for i in range(across)
        for j in range(along)
    ]
    span = (low_u, high_u, low_v, high_v)
    steps = ((high_u - low_u) / across, (high_v - low_v) / along)
    on = [pair for pair, keep in zip(pairs, _on_face(face, pairs, span, steps)) if keep]
    if not on:
        # A face whose whole lattice falls outside it is a face narrower than
        # the spacing somewhere, and asking nothing there is a coarsening
        # nothing would report. Its middle is on it by construction, so the
        # face keeps one station rather than none.
        on = [(middle_u, middle_v)]
    # The spacing the lattice realises, from the stations that survived rather
    # than from the ones laid out. What carries it is the report's bound on how
    # far the field climbs between stations, and a filtered lattice leaves wider
    # spaces than the one that was asked for.
    kept_u = len({pair[0] for pair in on})
    kept_v = len({pair[1] for pair in on})
    return on, max(width / kept_u, height / kept_v)


def _on_face(
    face: Any,
    pairs: Sequence[tuple[float, float]],
    span: tuple[float, float, float, float],
    steps: tuple[float, float],
) -> list[bool]:
    """Which of a face's candidate parameter pairs land on it, taken together.

    :param span: The parameter rectangle the lattice was laid across, as the
        caller read it. Handed over rather than read again, so that a lattice is
        one reading of the face.
    :param steps: How far apart the lattice is in each parameter direction.
        Every distance below is measured in those, so a face is held to the same
        band whether one unit of its ``u`` is a radian or a millimetre.

    :func:`_on` answers this for one pair per call, at a price that does not
    fall as the calls repeat on one face - so a lattice asked pair by pair pays
    what the face's boundary costs once per station.

    Each route here is tried before the one under it. A face covering its own
    parameter rectangle keeps every pair and is asked nothing. A face that does
    not has its trim read as a polygon in the same parameters, and the whole
    lattice is classified against that at once; a pair the polygon puts within
    :data:`BOUNDARY_BAND` of the boundary goes to the kernel, that being where a
    polygon and the curve it stands for can disagree. A face that cannot supply
    a polygon is asked pair by pair.
    """
    if not pairs:
        return []
    if _covers_its_range(face, span):
        return [True] * len(pairs)
    boundary = None if min(steps) <= 0.0 else _boundary(face, steps)
    if boundary is None:
        return [_on(face, pair) for pair in pairs]
    points = np.asarray(pairs, dtype=np.float64) / np.asarray(steps, dtype=np.float64)
    inside, near = _classified(boundary, points)
    kept = [bool(value) for value in inside]
    for found in np.nonzero(near)[0]:
        place = int(found)
        kept[place] = _on(face, pairs[place])
    return kept


def _covers_its_range(face: Any, span: tuple[float, float, float, float]) -> bool:
    """Whether the face occupies the whole parameter rectangle it is stated over.

    Its own area against the area its surface covers over that rectangle. The
    trim is a subset of the rectangle and an area is the integral of a
    non-negative Jacobian, so the difference between the two is the area of what
    the trim leaves out - an identity rather than a resemblance. A difference
    under :data:`UNTRIMMED_SHORTFALL` of that area is two quadratures of one
    region rather than a region left out.

    ``False`` wherever the question cannot be put - a stand-in with no surface,
    a surface the kernel will not build a patch from, a rectangle covering no
    area. That is the direction that costs a query rather than an answer.
    """
    build = getattr(getattr(face, "Surface", None), "toShape", None)
    if build is None:
        return False
    try:
        whole = float(build(*span).Area)
    except Exception:
        return False
    if not whole > 0.0:
        return False
    # One-sided, because the two areas are quadratures of different shapes and
    # the kernel may put the patch's own below the face's. What is being tested
    # is that the trim leaves nothing out, and a shortfall below zero leaves out
    # less than nothing.
    return whole - float(face.Area) <= UNTRIMMED_SHORTFALL * whole


def _boundary(face: Any, steps: tuple[float, float]) -> np.ndarray | None:
    """A face's trim as line segments in its own parameters, measured in steps.

    Every wire of the face contributes a loop, and every loop is that wire's
    edges as parameter curves, discretised to :data:`BOUNDARY_FINENESS` of a
    step. Divided by the step so a distance across the result is a distance in
    stations.

    The edges come in the wire's own order rather than off its edge list. A wire
    running along the seam of a periodic surface uses the seam edge twice, once
    at each end of the period; the edge list holds one of the two, and a loop
    built from it is open along one side of the rectangle, which puts the whole
    face outside itself.

    ``None`` where any edge has no parameter curve or none of them yields a
    segment, which is a face this cannot answer for at all.
    """
    wires = getattr(face, "Wires", None)
    if not wires:
        return None
    fineness = BOUNDARY_FINENESS * min(steps)
    segments: list[tuple[float, float, float, float]] = []
    for wire in wires:
        ordered = getattr(wire, "OrderedEdges", None)
        if ordered is None:
            return None
        for edge in ordered:
            try:
                found = face.curveOnSurface(edge)
            except Exception:
                return None
            if found is None:
                return None
            curve, first, last = found
            walked = curve.discretize(Deflection=fineness, First=first, Last=last)
            # Keeps a curve with no run in it out of the boundary.
            if len(walked) < 2:
                continue
            for near, far in zip(walked, walked[1:]):
                segments.append((near.x, near.y, far.x, far.y))
    if not segments:
        return None
    held = np.asarray(segments, dtype=np.float64)
    held[:, 0::2] /= steps[0]
    held[:, 1::2] /= steps[1]
    return _levelled(held)


def _levelled(held: np.ndarray) -> np.ndarray:
    """The boundary with one height for each corner it turns at.

    A parity ray is horizontal, so what it can pass between is two spellings of
    one corner's height. Every height within :data:`ONE_CORNER` of the one below
    it is taken as that one, which closes the gap a ray would otherwise cross.

    Heights alone. The ray never meets two spellings of one corner's width, and
    a corner moved sideways by a rounding is a corner in the same place.
    """
    heights = held[:, 1::2].reshape(-1)
    order = np.argsort(heights, kind="stable")
    sorted_heights = heights[order]
    # A new corner wherever the step up from the one before it is more than a
    # rounding, so a run of spellings of one corner takes the first of them.
    fresh = np.empty(sorted_heights.shape, dtype=bool)
    fresh[:1] = True
    fresh[1:] = np.diff(sorted_heights) > ONE_CORNER
    heights[order] = sorted_heights[
        np.maximum.accumulate(np.where(fresh, np.arange(fresh.size), 0))
    ]
    held[:, 1::2] = heights.reshape(-1, 2)
    return held


def _classified(boundary: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Which points the boundary encloses, and which stand too near it to say.

    Enclosure is parity: a ray from the point crosses a closed curve an odd
    number of times where the point is inside it. Parity needs no wire named as
    the outer one and no segment following another, so a face's holes subtract
    themselves and a loop assembled in any order still bounds what it bounds.

    Nearness is the distance to the nearest segment. Inside
    :data:`BOUNDARY_BAND` the polygon stands for a curve it is only close to,
    and what the caller does there is ask the kernel.

    The two are read off the same block of point-and-segment pairs, cut to
    :data:`MOST_CROSSINGS`.
    """
    starts, finishes = boundary[:, 0:2], boundary[:, 2:4]
    runs = finishes - starts
    lengths = np.einsum("ij,ij->i", runs, runs)
    inside = np.zeros(points.shape[0], dtype=bool)
    near = np.zeros(points.shape[0], dtype=bool)
    span = max(1, MOST_CROSSINGS // boundary.shape[0])
    for begin in range(0, points.shape[0], span):
        block = points[begin : begin + span]
        across = block[:, 1:2]
        straddles = (starts[None, :, 1] > across) != (finishes[None, :, 1] > across)
        with np.errstate(divide="ignore", invalid="ignore"):
            where = starts[None, :, 0] + (across - starts[None, :, 1]) * (
                runs[None, :, 0] / runs[None, :, 1]
            )
        crossed = straddles & (where > block[:, 0:1])
        inside[begin : begin + span] = crossed.sum(axis=1) % 2 == 1
        offset = block[:, None, :] - starts[None, :, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            along = np.einsum("ijk,jk->ij", offset, runs) / lengths[None, :]
        along = np.clip(np.nan_to_num(along), 0.0, 1.0)
        apart = offset - along[:, :, None] * runs[None, :, :]
        near[begin : begin + span] = (
            np.einsum("ijk,ijk->ij", apart, apart).min(axis=1) <= BOUNDARY_BAND**2
        )
    return inside, near


def _on(face: Any, pair: tuple[float, float]) -> bool:
    """Whether those parameters land on the face rather than beside it.

    A face's parameter range is the rectangle its surface is trimmed out of,
    and the trimming is what makes it the face. On a plane cut to a triangle,
    or a boolean's remnant, or a sheet with a hole, a lattice laid across that
    rectangle puts a share of its points on the surface's own extension - where
    the face is not, and where nothing about the body is true. A measurement
    taken there is about no part of the drawing.

    This is the kernel's own answer, and :func:`_on_face` decides which pairs
    are worth one: which parameters a face occupies is what a trimmed surface
    is, and a stand-in that has no answer is taken to be untrimmed rather than
    empty.

    A refusal is read the same way, and which way is the decision here. A
    parameterisation that cannot answer for a point of its own - a pole, a seam
    - says nothing about whether the face is there, and a dropped sample is a
    length nobody measured and a grid that coarsens with nothing said, where a
    sample kept beside the face costs one cast that meets no material.
    """
    ask = getattr(face, "isPartOfDomain", None)
    if ask is None:
        return True
    try:
        return bool(ask(*pair))
    except Exception:
        return True


def _steps(
    face: Any, start: tuple[float, float], stop: tuple[float, float], spacing: float
) -> tuple[int, float]:
    """How many samples one parameter direction wants, from how long it is, and
    that length as well, so the caller's realised spacing divides the
    measurement the count came from.

    The length is estimated by a three-point chord rather than integrated. It
    decides a sample count, and a count short by the small amount a chord
    under-reads a curve costs one sample.

    The count is a step function of that length, so a face whose count sits at
    the cap gives the same answer however the surface was parameterised, and one
    below it would move by a sample when the length crossed a multiple of the
    spacing. A drawing sits on such a multiple whenever a face is a round number
    of cells across, which is most of them, and the kernel measures the length
    again on the far side of a file - so the step is taken with
    :data:`COUNT_SLACK` under it. A count that moved would matter most where
    what is being sampled varies quickly over the face: every sample shifts,
    and a demand set is meant to be a function of the shape.
    """
    middle = tuple(0.5 * (a + b) for a, b in zip(start, stop))
    points = [_xyz(face.valueAt(*where)) for where in (start, middle, stop)]
    length = math.dist(points[0], points[1]) + math.dist(points[1], points[2])
    if spacing <= 0:
        return 2, length
    return max(2, min(MAX_SAMPLES, int(length / spacing + COUNT_SLACK) + 1)), length


def _boxes_further_apart_than(one: Any, other: Any, reach: float) -> bool:
    """Whether two shapes' bounding boxes are certainly more than ``reach`` apart.

    The comparison is a lower bound on the true distance, so it can only decline
    to cull. It never culls a pair that mattered.
    """
    first, second = one.BoundBox, other.BoundBox
    gap = 0.0
    for low, high in (("XMin", "XMax"), ("YMin", "YMax"), ("ZMin", "ZMax")):
        along = max(
            float(getattr(first, low)) - float(getattr(second, high)),
            float(getattr(second, low)) - float(getattr(first, high)),
            0.0,
        )
        gap += along * along
    return gap >= reach * reach


def _normal_at(face: Any, point: Any) -> tuple[float, float, float]:
    u, v = face.Surface.parameter(point)
    return _xyz(face.normalAt(u, v))


def _angle_between(one: tuple[float, float, float], other: tuple[float, float, float]) -> float:
    dot = sum(a * b for a, b in zip(one, other))
    lengths = math.sqrt(sum(a * a for a in one)) * math.sqrt(sum(b * b for b in other))
    if lengths <= 0.0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, dot / lengths)))


def _xyz(point: Any) -> tuple[float, float, float]:
    return (float(point.x), float(point.y), float(point.z))
