# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Measuring the lengths a drawing carries, so the grid can resolve them.

Local feature size is the distance to the medial axis: at any point, the radius
of the largest ball that fits inside the material (or inside the gap) and
touches the boundary twice. No medial axis is computed for it, because the CAD
kernel answers the questions it is wanted for directly.

What is measured, and what each becomes:

* **A gap between two bodies**, from the witness pair ``distToShape`` returns
  with the distance. The pair certifies a ball of that diameter touching both,
  so it is a *separation* demand along the direction the pair points in.
* **A curved surface's own radius**, from the sharpest principal curvature. It
  carries no direction, so it is a *connection* demand. It is a radius of
  curvature and **not** the medial radius: on a fillet the two part company
  badly, and nothing here should read it as a thickness.
* **How thick the metal is**, as the chord the body cuts from the inward normal
  at a point on its surface. Asked directly because a witness pair cannot see a
  thickness carried by a single face - a cylinder's only non-adjacent face pair
  is its two ends, so its diameter is witnessed by nothing.
* **How many cells span a dielectric**, which is that same chord asked as a
  count, since what a thin dielectric under-resolves is the field varying across
  it rather than a cell failing to fit.
* **Where a curved surface is**, against the radius already measured. A flat
  face is pinned as a grid line and placed exactly; a curved one is placed by
  sampling, to within about half a cell of where it was drawn.
* **Where the surface stops being smooth** - a join between two faces whose
  normals disagree, which is where a field singularity sits. Neither feature
  size nor curvature covers it, and a join closed to a *knife* asks once more
  along the bisector of its two faces.

Why a witness pair misses a cylinder's diameter, why a fillet's curvature is not
a thickness, where a join's demands and a thickness demand cross, and why a
knife needs the extra one, are in docs/internals/feature-size.md.

This module imports no FreeCAD. It is handed shapes and calls methods on them,
the way the rest of the translation layer is, so it can be exercised without a
CAD kernel present.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

from ...portbox import KERNEL_TOLERANCE
from .sizing import DIMENSIONS, Feature, fits_inside

__all__ = ["Body", "features"]

#: How far apart two gaps have to be before neither can affect the other,
#: as a multiple of the coarsest cell. A gap of width d asks for cells no
#: smaller than d/K, where K is the largest that ``sqrt(m_max) * sum_j sqrt(m_j)``
#: reaches over unit normals; Holder bounds that sum by 3**(3/4) and the root by
#: 1, so 3**(3/4) is safe. The body diagonal is *not* the worst case here - it
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
#: The default, and not a constant: it is a *share of a radius*, so refining the
#: cell sizes leaves it exactly where it was and a large smooth wall keeps the
#: same displacement however fine the rest of the grid gets. A caller measuring
#: how an answer approaches a drawing has to move this, which is why
#: :func:`features` takes it.
SURFACE_FIDELITY = 0.1

#: A normal disagreement at a join, beyond which the surface is not smooth
#: there and carries a field singularity. Well above the tolerance a kernel
#: leaves on a join it built to be tangent, and well below anything a person
#: would draw meaning to be a corner - so what it separates is a fillet from an
#: edge, rather than one kind of edge from another.
#:
#: It is read from the other end as well, where a disagreement within this of
#: straight through is a knife rather than a corner. That is a second question
#: and not the same one: it asks how much of the way out between two faces they
#: have given up, which degrades smoothly and has no boundary of its own, so
#: where the line falls is a judgement about what is worth refining. One number
#: serves both because one number suffices, and moving it moves both.
SHARP_DEGREES = 15.0

#: The most parameter samples taken across one face in one direction. Samples
#: are spread evenly in the face's *parameters*, which carry no scale of their
#: own - one unit of u is a radian on a cylinder and a millimetre on a plane -
#: so the count is set from the face's size and the spacing in space is only
#: even where the parameterisation is. A face wanting more than this is sampled
#: more coarsely than it asked, and the field ripples between the samples by
#: correspondingly more - the same bound, and the same cost of reaching it,
#: that :data:`MAX_EDGE_SAMPLES` puts on a rim.
#:
#: Per direction rather than over the face as a whole, which spends a long thin
#: face's allowance across a width that needed none. Spending it in proportion
#: to each direction's length instead leaves the count *unsaturated* on a face
#: that would otherwise reach the cap, and an unsaturated count moves by one when
#: a re-parameterisation moves the length it was read from - which shifts every
#: sample on the face, and where the quantity sampled varies quickly that changes
#: what is asked of the grid. A face too small to reach the cap is unsaturated
#: either way; what the cap settles is the large ones.
MAX_SAMPLES = 32

#: How far outside a solid a point may sit and still count as in it, in mm. It
#: is asked of points the kernel itself placed on the surface, so what it has to
#: absorb is the rounding in evaluating a face and nothing wider - which is the
#: kernel's own tolerance and not a second length.
CONTAINMENT_TOLERANCE = KERNEL_TOLERANCE

#: How wide a gap has to be before it is a gap, in mm. The same length again,
#: for the same reason: it absorbs the kernel's rounding, here in bringing two
#: surfaces together rather than in evaluating one. Two solids meeting along a
#: plane answer exactly zero and need no tolerance at all, but two *curved* ones
#: tangent to each other answer the last bits of their own arithmetic - and a
#: demand made on that is a cell size no grid can carry, which nothing else can
#: dominate and which sets the timestep for the whole domain.
TOUCHING = KERNEL_TOLERANCE

#: How far to step off the surface to find out which way a face was wound, in
#: mm. Clear enough of :data:`CONTAINMENT_TOLERANCE` that a step to the outside
#: reads as outside, and far below any wall this meshes, so a step to the inside
#: is still inside the thinnest metal drawn in practice. It settles a direction and
#: measures nothing, which is why it is a length of its own and not a share of
#: the reach: a share of the reach is set by the model's coarsest cell, and
#: would step clean through the wall it was sent to find.
WINDING_PROBE = 10.0 * CONTAINMENT_TOLERANCE

#: How many places along a ray are looked at before the metal's far side is
#: bracketed. They span the whole reach, so this sets the narrowest void that
#: can be stepped over rather than crossed - and being stepped over reads the
#: chord long, which is the direction every proxy in this module fails in.
MARCH_STEPS = 16

#: How closely the crossing is then placed inside that bracket, as a fraction
#: of the chord itself. Relative rather than absolute because the bracket
#: starts at a share of the reach, which is set by the coarsest cell in the
#: model and not by the body being measured - so a fixed number of halvings
#: would read a thin wall in a large domain to a precision that has nothing to
#: do with the wall. A thickness becomes a cell size, and a hundredth is finer
#: than a cell size is read to.
CHORD_TOLERANCE = 0.01

#: The most halvings spent reaching that, so a chord far below the bracket it
#: was found in cannot cost without bound. A guard rather than a working limit:
#: the halvings needed are the log of the bracket over the chord, so reaching
#: this takes a chord some ten orders below the reach - and unlike the march
#: above it, giving up here leaves an error either side of the answer rather
#: than one that reads the metal thicker.
MAX_HALVINGS = 32


#: The most samples taken along one edge. Larger than the per-direction bound
#: on a face because an edge is a curve and a face is a surface: sampling an
#: edge costs in proportion to its length, where sampling a face costs the
#: square of its size. A rim capped below what it asked for shows up as cells
#: that ripple between the samples instead of following it evenly.
MAX_EDGE_SAMPLES = 256


@dataclass(frozen=True)
class Body:
    """One drawn object, as this module needs it.

    :param label: What to call it in a message.
    :param shape: The FreeCAD shape. Only ``BoundBox``, ``Faces``, ``Edges``,
        ``Solids``, ``distToShape`` and ``isInside`` are used, together with
        addition and scaling on the points and normals the shape hands back, so
        a stand-in needs no kernel. Nothing here *builds* geometry, which is
        what keeps the kernel out: a ray is walked by arithmetic on a point
        rather than by making a line to intersect with.
    :param metal: Whether its edges carry a field singularity. A dielectric
        corner does not, so nothing is asked for one.
    :param measured: Whether this body's own lengths are read here. A solid
        that reached the mesher as a box has already told it where its faces
        are; it is still carried, because a gap *to* it is a gap.
    :param sheet: Whether the shape is an area rather than a volume. It decides
        what an edge with only one face beside it means: on a sheet that is the
        outline, where the metal stops, and on a solid it is a seam - an
        artefact of how the kernel parameterised a closed surface, with metal on
        both sides of it and nothing to resolve.
    :param relaxed_to: The finest cell this body's own lengths may ask for, or
        ``None`` to ask at what they measure. A gap *to* it is not relaxed: a
        separation belongs to both bodies, and one of them asking for less is
        not the other agreeing.
    """

    label: str
    shape: object
    metal: bool = False
    measured: bool = True
    sheet: bool = False
    relaxed_to: float | None = None


def features(
    bodies: Sequence[Body],
    cap: float,
    edge_size: float | None = None,
    grading: float = math.log(1.3),
    fidelity: float = SURFACE_FIDELITY,
    min_lines: int = 0,
) -> list[Feature]:
    """Every length the drawing carries that the grid may have to resolve.

    :param cap: The coarsest cell the grid will use. Anything whose demand
        lands at or above it cannot bind, so it is never measured - which is
        what keeps the cost proportional to the geometry that is genuinely
        close rather than to the geometry that exists.
    :param edge_size: The cell size a metal edge asks for, or ``None`` to ask
        for nothing there. It floors what a curved face may ask for as well,
        which is why :func:`_curvatures` is handed it.
    :param grading: The sizing field's own slope, which decides when one
        measurement makes another redundant. See :func:`_pruned`.
    :param fidelity: How closely a curved surface has to be followed, as a
        share of the radius it curves through. See :data:`SURFACE_FIDELITY`.
    :param min_lines: How many cells to put across a dielectric's own
        thickness, or fewer than two to ask for nothing there. One is not a
        count - it asks for the layer's whole extent - which is the same reason
        the mesher applies the rule to a box only above that.
    """
    found: list[Feature] = []
    found.extend(_separations(bodies, cap))
    for body in bodies:
        if not body.measured:
            continue
        # What each is asked for divides on the same line openEMS divides on.
        # A conductor's boundary is decided by sampling one point, so it
        # staircases and is held to a fidelity, and its thickness decides
        # whether it still conducts - an argument about zeroed Yee edges sharing
        # nodes, which is a statement about metal. A dielectric is averaged over
        # the cell instead: nothing about it staircases, and what its thickness
        # decides is whether the field varying across the layer is carried by
        # more than one cell.
        mine: list[Feature] = []
        if body.metal:
            mine.extend(_curvatures(body, cap, edge_size, fidelity))
            if body.sheet:
                mine.extend(_rims(body, cap, edge_size, fidelity))
            mine.extend(_thicknesses(body, cap, edge_size))
            if edge_size is not None:
                mine.extend(_sharp_joins(body, edge_size))
        elif min_lines >= 2:
            mine.extend(_element_counts(body, cap, min_lines))
        # Before pruning, so that a relaxed demand is compared at the size it
        # will actually ask for rather than at the one it measured.
        if body.relaxed_to is not None:
            mine = [replace(feature, relaxed_to=body.relaxed_to) for feature in mine]
        found.extend(mine)
    return _pruned(found, grading)


def _pruned(found: Sequence[Feature], grading: float) -> list[Feature]:
    """Drop every measurement another one already covers.

    The sizing field takes a minimum of ramps: a demand for cells of ``s`` at a
    point holds down cells of ``s + g|x - y|`` at every other point ``y``. So a
    measurement asking ``s`` at ``y`` changes nothing if some other asks ``s'``
    at ``x`` with ``s' + g|x - y| <= s`` - the field is already at least that
    fine there, and carrying the second only costs a constraint.

    What that drops is a coarse measurement standing near a finer one - a bend
    beside the sharper bend it runs into, a curvature beside a gap that already
    holds the field below it. What it does **not** drop is a face sampled all
    over at one curvature: those sit at different places asking for the same
    size, and the inequality then needs zero distance, which is a repeat at one
    point rather than a second point on the surface. A surface is held down
    where each sample sits and nowhere else, so those samples are the demand and
    not a redundant restatement of it.

    That is also what bounds the work. Only a *strictly finer* demand can
    dominate, and the demands are taken in ascending size, so what has to be
    scanned is the ones already found to be finer rather than everything kept.
    A body of one curvature is then linear in its samples instead of quadratic.

    Kept in the order the demands were measured, finest first, so a body that
    asks for something no other body covers keeps it whatever else is present.
    """
    if not found:
        return []
    order = sorted(range(len(found)), key=lambda i: found[i].thickness)
    kept: list[Feature] = []
    places: list[tuple[float, float, float]] = []
    sizes: list[float] = []
    seen: set[tuple[float, tuple[float, float, float]]] = set()
    for index in order:
        feature = found[index]
        # Only an omnidirectional demand can dominate, and only another
        # omnidirectional one can be dominated: a directional demand spends
        # itself differently across the axes, so two of them are not comparable
        # by size alone.
        if feature.normal is not None:
            kept.append(feature)
            continue
        here = feature.lower
        mine = min(feature.cells())
        # The zero-distance case, which the scan below cannot reach because it
        # looks at nothing of this size.
        if (mine, here) in seen:
            continue
        finer = bisect.bisect_left(sizes, mine)
        if any(
            sizes[other] + grading * math.dist(places[other], here) <= mine
            for other in range(finer)
        ):
            continue
        kept.append(feature)
        places.append(here)
        sizes.append(mine)
        seen.add((mine, here))
    return kept


def _separations(bodies: Sequence[Body], cap: float) -> Iterator[Feature]:
    """Gaps between bodies, from the witness pairs that realise them.

    The pairing is over *bodies* and not over faces: a gap is a property of two
    objects, and the kernel finds the closest points between them without being
    told which faces to look at.

    Bodies further apart than any demand could matter are never queried. The
    smallest cell a gap of width ``d`` can ask for is ``d/sqrt(3)`` - the
    allocation's worst case, where the gap's normal points equally at all three
    axes - so a pair whose boxes are more than ``sqrt(3) * cap`` apart is
    provably inert. Their bounding boxes are compared first because that is
    arithmetic, where the exact query is a kernel call.
    """
    reach = SEPARATION_REACH * cap
    for first in range(len(bodies)):
        for second in range(first + 1, len(bodies)):
            one, other = bodies[first], bodies[second]
            # A pair of bodies that both describe themselves to the mesher
            # already - two boxes - has nothing to add here: each pins its own
            # faces and the thirds rule sizes the gap between them.
            if not (one.measured or other.measured):
                continue
            if _boxes_further_apart_than(one.shape, other.shape, reach):
                continue
            distance, pairs, _ = one.shape.distToShape(other.shape)
            distance = float(distance)
            # Touching is not a gap. Two solids that share a face have no
            # medial ball between them, and a zero thickness is not a length -
            # nor is one the kernel arrived at by bringing two curved surfaces
            # together, which is tangency reported to the precision it was
            # computed at rather than a drawn clearance.
            if distance <= TOUCHING or distance >= reach:
                continue
            for near, far in pairs:
                normal = tuple(_xyz(far)[d] - _xyz(near)[d] for d in range(DIMENSIONS))
                source = f"the gap between {one.label!r} and {other.label!r}"
                for point in (near, far):
                    place = _xyz(point)
                    yield Feature(
                        thickness=distance,
                        normal=normal,
                        lower=place,
                        upper=place,
                        source=source,
                    )


def _curvatures(
    body: Body, cap: float, edge_size: float | None, fidelity: float = SURFACE_FIDELITY
) -> Iterator[Feature]:
    """How finely a curved face has to be followed, and where.

    The reciprocal of the larger principal curvature bounds the medial ball at
    a point, so twice it is a length the criteria can be stated against - and a
    plane curves not at all and bounds nothing, which is correct, since a flat
    face says nothing about the solid behind it. What that length is asked for
    settles between these claims, and which one holds is read off the radius
    rather than declared:

    * **Fidelity**, ``fidelity`` of the radius the surface curves through, which
      is what binds on a body drawn round rather than thin.
    * **Never finer than a metal edge asks for.** A corner is what a fillet
      becomes as its radius goes to zero, and a corner asks for ``edge_size`` -
      so a fillet asking for a fraction of its vanishing radius would cost
      unboundedly more than the shape it is on its way to being.
    * **Never coarser than the length itself**, which is the connection
      criterion. That length is an upper bound on the medial radius rather than
      the metal's own cross-section, so a cell fitting inside it fits inside the
      metal - the direction that keeps a body thinner than an edge from being
      sampled into something electrically open.
    """
    reach = math.sqrt(DIMENSIONS) * cap
    # A connection demand spends its thickness as thickness/sqrt(DIMENSIONS) on
    # each axis, so this is the thickness that asks for exactly the edge size.
    floor = 0.0 if edge_size is None else math.sqrt(DIMENSIONS) * edge_size
    # How far apart to look, which is the size a demand off this face will
    # usually settle on: the field between two samples stays where they put it
    # only where they are about one such cell apart. A body thin enough for the
    # connection bound to take over asks for less than this and is sampled more
    # coarsely than it asked, the way a face too large for MAX_SAMPLES is.
    spacing = cap if edge_size is None else edge_size
    for number, face in enumerate(_faces(body.shape)):
        for u, v in _samples(face, spacing):
            try:
                curvatures = face.curvatureAt(u, v)
            except Exception:
                # A face can carry a point its own parameterisation cannot
                # answer for - a pole, a seam. One sample is not the face.
                continue
            sharpest = max(abs(float(value)) for value in curvatures)
            if sharpest <= 0.0:
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
                source=f"{body.label!r} curving on face {number}",
            )


def _rims(
    body: Body, cap: float, edge_size: float | None, fidelity: float = SURFACE_FIDELITY
) -> Iterator[Feature]:
    """How finely a sheet's outline has to be followed, and where.

    A sheet's face is flat, so :func:`_curvatures` reads nothing off it; what
    curves on a round pad is its rim, and openEMS samples that the same way it
    samples a wall. **The connection criterion is not among the claims here**,
    which is what separates this from :func:`_curvatures`. Both, and what a
    hole's rim asks, are in docs/internals/feature-size.md#a-rim-is-not-a-cross-section.

    A curve answers its curvature as a magnitude, where a face answers two signed
    principal ones - measured on FreeCAD 1.1.1, where reversing a circle and
    running a spline through its own inflection both leave it non-negative. So
    there is no sharpest-of-two to take here, and nothing to take an absolute
    value of.
    """
    reach = math.sqrt(DIMENSIONS) * cap
    floor = 0.0 if edge_size is None else math.sqrt(DIMENSIONS) * edge_size
    # Sampled at the metal edge size rather than at what a rim usually settles
    # on, which is coarser: where a rim's curvature varies it is the tighter end
    # that has to be caught, and a sample spaced at the demand it produced would
    # walk past it.
    spacing = cap if edge_size is None else edge_size
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
                continue
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
                source=f"{body.label!r} rim curving",
            )


def _thicknesses(body: Body, cap: float, edge_size: float | None) -> Iterator[Feature]:
    """How thick the metal is across itself, and where.

    A connection demand: a cell that fits inside the metal is a cell the sampling
    cannot open it at, stated at its worst direction because a conductor thinner
    than a cell fails by being sampled into a vertex-adjacent chain rather than
    in any one direction.

    **Spent in full, with no floor under it**, unlike the fidelity demand in
    :func:`_curvatures`, and at the cost of what a tapering tip does to the
    timestep. Why a cross-section may not be floored when fidelity may, and what
    that costs, are in
    docs/internals/feature-size.md#floors-and-which-demands-may-be-given-up.
    """
    reach = fits_inside(cap)
    spacing = cap if edge_size is None else edge_size
    for number, point, _, chord in _chords(body, reach, spacing):
        place = _xyz(point)
        yield Feature(
            thickness=chord,
            normal=None,
            lower=place,
            upper=place,
            source=f"{body.label!r} thickness on face {number}",
        )


def _element_counts(body: Body, cap: float, across: int) -> Iterator[Feature]:
    """How many cells span a dielectric across its own thickness, and where.

    The count a box gets, given to a shape a box cannot describe: a box states
    its thickness as an extent per axis, a triangulation states nothing, and the
    box bounding a bent board is as deep as the bend. So the chord is measured,
    and the count stated against it along the normal it was measured on.

    Nothing here is a claim about where the boundary went. openEMS averages a
    dielectric over the cell rather than sampling it at a point, so it does not
    staircase the way a conductor does and is held to no fidelity.

    **Sampled at the coarsest cell rather than the size the demands will ask
    for**, the one exception to the rule :func:`_samples` states, and **the
    demand covers the chord** rather than sitting at its end the way a
    cross-section does. Why each, and what the reach costs a void or a taper, are
    in docs/internals/feature-size.md#sampled-coarsely-on-purpose.
    """
    for number, point, step, chord in _chords(body, across * cap, cap):
        place, far = _xyz(point), _xyz(point + step * chord)
        yield Feature(
            thickness=chord,
            # The chord as walked rather than the face's normal: a kernel winds
            # a face either way, and only the walk knows which side the layer is.
            normal=tuple(end - start for start, end in zip(place, far)),
            lower=tuple(min(start, end) for start, end in zip(place, far)),
            upper=tuple(max(start, end) for start, end in zip(place, far)),
            across=across,
            source=f"{body.label!r} across its thickness on face {number}",
        )


def _chords(
    body: Body, reach: float, spacing: float
) -> Iterator[tuple[int, object, object, float]]:
    """Each face sample, the way the body runs inward from it, and how far.

    Both questions a thickness answers read the same chord - whether a conductor
    still conducts across itself, and how many cells span a dielectric across
    itself - so it is walked once and they cannot come to disagree about how
    thick the body is.

    The step comes back as well as the length, because which way a chord ran is
    a fact about the drawing rather than about the face: a kernel winds a face
    either way, so a demand that has to cover the material cannot tell from the
    normal which side of the sample that material is on. Handed over as a
    direction rather than as the far point, so the caller that wants no such
    point does not pay for one.

    A sheet is an area and has nothing to measure. It reaches the engine as a
    zero-thickness primitive, and asking a shell for containment answers
    nothing.
    """
    solids = list(getattr(body.shape, "Solids", ()) or ())
    if body.sheet or not solids:
        return
    for number, face in enumerate(_faces(body.shape)):
        for u, v in _samples(face, spacing):
            try:
                point, normal = face.valueAt(u, v), face.normalAt(u, v)
            except Exception:
                # A face can carry a point its own parameterisation cannot
                # answer for - a pole, a seam. One sample is not the face.
                continue
            walked = _chord(solids, point, normal, reach)
            if walked is None:
                continue
            chord, step = walked
            yield number, point, step, chord


def _chord(
    solids: Sequence, point: object, normal: object, reach: float
) -> tuple[float, object] | None:
    """How far the body runs from ``point`` along its own normal, and which way.

    The material rather than the metal: a conductor's cross-section and a
    dielectric's element count are the same walk asked for different reasons.

    The direction comes back as a unit step, because it is settled here and
    cannot be read off the face afterwards - a kernel winds a face either way,
    so which side of the sample the material lies on is not in the normal.

    ``None`` where it runs further than ``reach`` - every body too thick for the
    answer to bind - and where neither way off the surface is material, which a
    compound produces because its faces need not belong to any of its solids.

    Which way is inward is asked rather than derived. A face's normal points out
    of the solid or into it according to how the kernel wound the face, and one
    step to each side settles it without having to know which. Where *both* are
    inside, the face is not a boundary of the union - two solids meeting along
    it, or a face lying inside one - and the walk then goes the way the normal
    points away from, which is a side and not the nearer side. What comes back is
    one part of the body's whole run through that point, so it is short of it
    either way, and short refines rather than coarsens.

    The far side is bracketed by walking and then placed by halving. Walking is
    what the bracket costs on a body that is not convex: a ray can leave the
    material and enter it again, so the answer is the *first* crossing, and no
    ordering along the ray lets a search skip to it.
    """
    length = math.sqrt(sum(value * value for value in _xyz(normal)))
    if length <= 0.0:
        return None
    step = normal * (-1.0 / length)
    probe = WINDING_PROBE
    if not _inside(solids, point + step * probe):
        step = step * -1.0
        if not _inside(solids, point + step * probe):
            return None
    inward, outward = 0.0, None
    for i in range(1, MARCH_STEPS + 1):
        distance = reach * i / MARCH_STEPS
        if _inside(solids, point + step * distance):
            inward = distance
        else:
            outward = distance
            break
    if outward is None:
        return None
    for _ in range(MAX_HALVINGS):
        if outward - inward <= CHORD_TOLERANCE * outward:
            break
        middle = 0.5 * (inward + outward)
        if _inside(solids, point + step * middle):
            inward = middle
        else:
            outward = middle
    return 0.5 * (inward + outward), step


def _inside(solids: Sequence, point: object) -> bool:
    """Whether any of these solids contains the point.

    Each is asked separately because a compound answers for one member only -
    a point inside its second solid and no other comes back outside.
    """
    return any(solid.isInside(point, CONTAINMENT_TOLERANCE, True) for solid in solids)


def _sharp_joins(body: Body, edge_size: float) -> Iterator[Feature]:
    """Edges where the surface stops being smooth, on a conductor.

    A join whose two faces share a tangent plane carries no singularity however
    tightly it curves, so what is looked for is the disagreement between the
    normals rather than the curvature on either side. An edge belonging to one
    face only is a boundary of an open shell and is not a join at all.

    An edge is a *curve* and is followed along its whole length. A circular rim
    is one edge running right round a shape, and sampling it once refines the
    grid at a single point on the rim, which shows up as a mesh that followed the
    object on one side of it.

    Each sample asks **across** the edge, not along it: the field there varies
    with distance from the edge and is constant along a straight one, so cells
    packed along its length resolve nothing. Each of the two faces says which
    direction "across" is for it, so the demand is made twice, once per face
    normal; on a box edge that lands on exactly the two axes the thirds rule
    refines, leaving the edge's own axis alone.

    Neither is a demand along the direction *between* the two faces, and what
    that direction is left with depends on how the join lies on the grid -
    tolerably at an ordinary corner, by a margin that runs away as the faces
    close. So a join within :data:`SHARP_DEGREES` of straight through asks a
    third time, along the bisector. A near-closed *crack* reads and is treated
    the same way, its bisector pointing out of the tip rather than into it; the
    criterion is even in the normal, so which of the two it is changes nothing.
    Why that threshold is a judgement rather than a boundary in the geometry is
    in docs/internals/feature-size.md.

    Each face contains the edge, so every normal is square to it and so is the
    bisector: the third demand spends nothing along the edge either.
    """
    sharp = math.radians(SHARP_DEGREES)
    for edge, faces in _joins(body.shape):
        if len(faces) == 1 and body.sheet:
            yield from _outline(body, edge, faces[0], edge_size)
            continue
        if len(faces) != 2:
            continue
        for where in _along(edge, edge_size):
            # Kept as the kernel's own point type on the way to the faces, and
            # turned into numbers only for the feature. Building a vector here
            # would mean knowing which class to build.
            meeting = edge.valueAt(where)
            try:
                normals = [_normal_at(face, meeting) for face in faces]
            except Exception:
                # A point on a join that its own faces cannot answer for is one
                # this cannot judge, and guessing sharp would refine it.
                continue
            angle = _angle_between(*normals)
            if angle < sharp:
                continue
            place = _xyz(meeting)
            for normal in normals:
                yield Feature(
                    thickness=edge_size,
                    normal=normal,
                    lower=place,
                    upper=place,
                    source=f"{body.label!r} edge",
                )
            bisector = _bisector(*normals) if angle > math.pi - sharp else None
            if bisector is not None:
                yield Feature(
                    thickness=edge_size,
                    normal=bisector,
                    lower=place,
                    upper=place,
                    source=f"{body.label!r} knife edge",
                )


def _outline(body: Body, edge: object, face: object, edge_size: float) -> Iterator[Feature]:
    """Where a sheet's metal stops, which is an edge like any other.

    A sheet is a conductor with a boundary rather than a closed surface, so the
    edge that matters has one face beside it and not two - and the field
    singularity at it is the same one a solid's edge carries. Without this a
    sheet drawn as an outline reaches the engine and changes no grid line: it
    has no box for the thirds rule to work from, and no second face to disagree
    with.

    "Across" here is in the plane of the sheet. The edge's tangent says which way
    is *along*, the face's normal says which way is out of the plane, and their
    cross product is what is left - the direction the metal ends in.
    """
    normal = _xyz(face.normalAt(*_middle_of(face)))
    for where in _along(edge, edge_size):
        try:
            tangent = _xyz(edge.tangentAt(where))
        except Exception:
            # A point whose own curve cannot give a tangent - a cusp, a
            # degenerate segment. One point is not the outline.
            continue
        across = _cross(tangent, normal)
        if _length(across) <= 0.0:
            continue
        yield Feature(
            thickness=edge_size,
            normal=across,
            lower=_xyz(edge.valueAt(where)),
            upper=_xyz(edge.valueAt(where)),
            source=f"{body.label!r} outline",
        )


def _middle_of(face: object) -> tuple[float, float]:
    low_u, high_u, low_v, high_v = (float(value) for value in face.ParameterRange)
    return (0.5 * (low_u + high_u), 0.5 * (low_v + high_v))


def _bisector(
    one: tuple[float, float, float], other: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    """The direction halfway between two, or ``None`` where there is none.

    Halfway between two *directions*: each is scaled to unit length first, so
    how long either of them arrived has no weight in the answer. Two that point
    exactly opposite ways have nothing halfway between them, which as a join is
    a solid of no thickness at all.

    Both are known to have a length, so neither divisor is checked: a direction
    that collapsed reads as no disagreement at all, and a join with no
    disagreement is dropped before it gets here.
    """
    mine, theirs = _length(one), _length(other)
    halfway = (
        one[0] / mine + other[0] / theirs,
        one[1] / mine + other[1] / theirs,
        one[2] / mine + other[2] / theirs,
    )
    return halfway if _length(halfway) > 0.0 else None


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


def _faces(shape: object) -> Sequence:
    return list(getattr(shape, "Faces", ()) or ())


def _joins(shape: object) -> Iterator[tuple[object, list]]:
    """Every edge of a shape, with the faces that meet along it.

    Built from the faces rather than asked of the shape, because asking takes
    ``Part``, and nothing here imports FreeCAD. An edge is identified by the
    kernel's own hash, so the same edge reached through two faces is one edge.
    """
    edges: dict[int, tuple[object, list]] = {}
    for face in _faces(shape):
        for edge in getattr(face, "Edges", ()) or ():
            key = int(edge.hashCode())
            edges.setdefault(key, (edge, []))[1].append(face)
    yield from edges.values()


def _along(edge: object, size: float) -> Iterator[float]:
    """Parameters along one edge, spaced by the cell size being asked for.

    That spacing comes out of the field rather than being chosen: two point
    demands for cells of ``s``, a distance ``s`` apart, leave the field peaking
    at ``s(1 + g/2)`` between them, and ``g`` is ``ln(max_ratio)``, which is
    below ``max_ratio - 1`` for every ratio above one - one grading step, at
    most, between consecutive samples.

    The *count* is what that decides. Where the samples fall is even in the
    edge's parameter, which is even in space only where the parameterisation is,
    so on a spline or an ellipse the spacing varies about it and the bound is a
    target rather than a guarantee. What makes that acceptable is the same thing
    that makes it acceptable on a face: the demand is a refinement, and being
    denser than intended in places costs constraints rather than correctness.

    Bounded, because an edge can be long and the size asked for small. A rim
    longer than :data:`MAX_EDGE_SAMPLES` times the cell size is sampled more
    coarsely than that, and the field ripples between the samples by
    correspondingly more - it stays a refinement of the whole rim, which is the
    thing that was wrong when an edge was sampled once.
    """
    low, high = float(edge.FirstParameter), float(edge.LastParameter)
    length = float(getattr(edge, "Length", 0.0))
    steps = 1 if size <= 0 or length <= 0 else min(MAX_EDGE_SAMPLES, int(length / size) + 1)
    for i in range(steps):
        yield low + (high - low) * (i + 0.5) / steps


def _samples(face: object, spacing: float) -> Iterator[tuple[float, float]]:
    """Parameter pairs across one face, about ``spacing`` apart in space.

    The spacing has to be decided in millimetres and applied in parameters,
    because a parameterisation carries no scale of its own - one unit of ``u``
    is a radian on a cylinder and a millimetre on a plane. Splitting the count
    per direction is what that costs, and what it buys is not sampling a rod's
    circumference as finely as its length: those differ by the aspect ratio of
    the face, which on a thin rod is the whole point of the rod.

    What the caller passes is the cell size the demands will ask for, which is
    the same rule :func:`_along` follows and for the same reason: two point
    demands for cells of ``s``, a distance ``s`` apart, leave the field peaking
    one grading step above ``s`` between them. Sampling instead at the coarsest
    cell would place a fine demand at a few points on the face and let the field
    climb back to bulk between them, which is a surface followed in patches.

    A target rather than a guarantee, twice over: the spacing is even in the
    face's parameters and so in space only where the parameterisation is, and a
    face asking for more than :data:`MAX_SAMPLES` in a direction gets that many.

    It is a sampling of a curvature that varies, not a search for its extremum:
    a crease narrower than the spacing can still be stepped over, and what
    catches that is checking the finished grid.
    """
    low_u, high_u, low_v, high_v = (float(value) for value in face.ParameterRange)
    middle_u, middle_v = 0.5 * (low_u + high_u), 0.5 * (low_v + high_v)
    across = _steps(face, (low_u, middle_v), (high_u, middle_v), spacing)
    along = _steps(face, (middle_u, low_v), (middle_u, high_v), spacing)
    for i in range(across):
        for j in range(along):
            yield (
                low_u + (high_u - low_u) * (i + 0.5) / across,
                low_v + (high_v - low_v) * (j + 0.5) / along,
            )


def _steps(
    face: object, start: tuple[float, float], stop: tuple[float, float], spacing: float
) -> int:
    """How many samples one parameter direction wants, from how long it is.

    Length is estimated by a three-point chord rather than integrated: it is
    deciding a sample count, and a count that is short by the small amount a
    chord under-reads a curve costs one sample.

    The count is a step function of that length, so a face whose count sits at
    the cap gives the same answer however the surface was parameterised, and one
    below it can move by a sample when the length moves across a multiple of the
    spacing. That matters where what is being sampled varies quickly over the
    face: every sample then shifts, and a demand set is meant to be a function
    of the shape.
    """
    if spacing <= 0:
        return 2
    middle = tuple(0.5 * (a + b) for a, b in zip(start, stop))
    points = [_xyz(face.valueAt(*where)) for where in (start, middle, stop)]
    length = math.dist(points[0], points[1]) + math.dist(points[1], points[2])
    return max(2, min(MAX_SAMPLES, int(length / spacing) + 1))


def _boxes_further_apart_than(one: object, other: object, reach: float) -> bool:
    """Whether two shapes' bounding boxes are certainly more than ``reach`` apart.

    A lower bound on the true distance, so it can only decline to cull - never
    cull a pair that mattered.
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


def _normal_at(face: object, point: object) -> tuple[float, float, float]:
    u, v = face.Surface.parameter(point)
    return _xyz(face.normalAt(u, v))


def _angle_between(one: tuple[float, float, float], other: tuple[float, float, float]) -> float:
    dot = sum(a * b for a, b in zip(one, other))
    lengths = math.sqrt(sum(a * a for a in one)) * math.sqrt(sum(b * b for b in other))
    if lengths <= 0.0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, dot / lengths)))


def _xyz(point: object) -> tuple[float, float, float]:
    return (float(point.x), float(point.y), float(point.z))
