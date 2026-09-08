# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Run the corpus through the geometry layer under a real FreeCAD.

This is the half that needs the CAD kernel, and it is the only half. What it
writes is one JSON artifact per specimen holding the triangles the layer
produced, the box it measured, and what the kernel itself says the shape's volume
and area are. Everything downstream - containment, connectivity, the grid and its
invariants - reads those artifacts and needs no FreeCAD at all, which is what
keeps the checks runnable under the interpreter that owns the openEMS bindings.

Executed by ``freecadcmd``, so:

* there is no ``__main__``. The file is exec'd under a module name taken from its
  own stem, and a bare ``if __name__ == "__main__":`` guard would never fire.
* the process **segfaults on exit** once a main window has been shown, after the
  last statement has run. Everything here is durable before that, and a caller
  must judge this by its output rather than by its exit status.

The main window is shown deliberately. ``Shape.tessellate`` returns the *view
provider's* display mesh once a shape has been drawn, and a display mesh is
built to a cosmetic deviation - so the trap only exists where a view provider
does, and a probe without one would report that there is nothing to catch.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

#: Where the artifacts land. The caller names it, since a test wants them under
#: its own temporary directory rather than in the tree.
OUT = pathlib.Path(os.environ.get("CORPUS_OUT", HERE / "_corpus"))


def _document(FreeCAD):
    name = "corpus"
    if name in FreeCAD.listDocuments():
        FreeCAD.closeDocument(name)
    return FreeCAD.newDocument(name)


def _drawn(doc, shape, label):
    """The shape as a document object, recomputed and visible.

    Through a document object rather than the bare shape, because that is how a
    user's geometry arrives and it is the only way a view provider exists.
    """
    obj = doc.addObject("Part::Feature", "Specimen")
    obj.Label = label
    obj.Shape = shape
    doc.recompute()
    return obj


def _square(shape):
    """Whether every face of the shape has no extent on exactly one axis.

    Which is a shape bounded entirely by planes square to the grid, and so one a
    rectilinear grid holds with nothing left over. Read off the kernel's own
    bounding boxes rather than asked of the geometry layer, so a check selecting
    on it is not selecting by the same reading it is scoring.
    """
    from Microwave.portbox import FLATNESS

    faces = list(getattr(shape, "Faces", ()) or ())
    return bool(faces) and all(
        sum(
            1
            for low, high in (
                (face.BoundBox.XMin, face.BoundBox.XMax),
                (face.BoundBox.YMin, face.BoundBox.YMax),
                (face.BoundBox.ZMin, face.BoundBox.ZMax),
            )
            if high - low <= FLATNESS
        )
        == 1
        for face in faces
    )


def _joins_drawn(shape):
    """Every edge with two faces beside it: how far apart they point, and the
    line the edge runs along there.

    Asked of the kernel rather than read out of the measuring layer's answer, so
    a check of either is scored against the shape. The angle is arrived at the
    way the layer arrives at it - there is one way to ask a face which way it
    points - so what it is independent of is the layer's result and not its
    method. The line is not: it comes from the curve's own tangent, where the
    layer takes the cross product of the two normals, which on a join closed to
    nearly flat is the worst-conditioned quantity in the corpus.

    The ends are what let a demand be attributed to the join it was made at. The
    normals let what a demand per face would have asked be worked out from the
    drawing.

    The angle, the line and the normals are taken at the middle of the edge, so
    on a curved join they describe one place on it and not the whole of it. The
    ends are the edge's own, and a closed curve has one or none.
    """
    beside = {}
    for face in shape.Faces:
        for edge in face.Edges:
            beside.setdefault(edge.hashCode(), []).append((edge, face))
    drawn = []
    for pairs in beside.values():
        if len(pairs) != 2:
            continue
        edge = pairs[0][0]
        where = 0.5 * (edge.FirstParameter + edge.LastParameter)
        middle = edge.valueAt(where)
        try:
            normals = [face.normalAt(*face.Surface.parameter(middle)) for _, face in pairs]
            along = edge.tangentAt(where)
        except Exception:  # noqa: BLE001 - an edge its own curve cannot answer for
            continue
        drawn.append(
            {
                "disagreement": math.degrees(normals[0].getAngle(normals[1])),
                "line": [float(along.x), float(along.y), float(along.z)],
                "ends": [
                    [float(v.Point.x), float(v.Point.y), float(v.Point.z)] for v in edge.Vertexes
                ],
                "normals": [[float(n.x), float(n.y), float(n.z)] for n in normals],
            }
        )
    return drawn


def _measured(shape):
    """What the kernel says about the shape, for a downstream check to score
    the triangulation against without the kernel being present."""
    box = shape.BoundBox
    joins = _joins_drawn(shape)
    return {
        "volume": float(shape.Volume),
        "area": float(shape.Area),
        "square": _square(shape),
        "joins_drawn": joins,
        "sharpest_join": max((one["disagreement"] for one in joins), default=None),
        "shape_type": str(shape.ShapeType),
        "solids": len(shape.Solids),
        "faces": len(shape.Faces),
        "edges": len(shape.Edges),
        "closed": bool(shape.isClosed()) if shape.ShapeType in ("Shell", "Wire") else None,
        "bound_box": [
            float(box.XMin),
            float(box.YMin),
            float(box.ZMin),
            float(box.XMax),
            float(box.YMax),
            float(box.ZMax),
        ],
    }


#: Faces sampled per shape, and points per face. Chosen to cross a face count
#: in the tens while keeping the kernel queries per specimen in the hundreds; a
#: shape with more faces than this is sampled over the first of them.
FACES_SAMPLED = 32
POINTS_PER_FACE = 4

#: Points scattered through the shape's own bounding box, which is where a
#: containment fault that is not near any surface would show.
SCATTERED = 60

#: A share of the shape's size, used both for how far in from a face's own
#: outline a sample has to sit and for how far a companion is placed off the
#: surface. The outline distance is measured rather than read off the parameter
#: fractions, because a planar face carries whatever frame its surface was built
#: with and the domain is not the shape.
MARGIN = 0.02

#: How far a facet's middle may sit from the surface, as a share of the shape's
#: own size, and still be the kernel's rounding rather than a departure. Far
#: below anything a triangulation of a drawn shape produces and far above the
#: arithmetic of evaluating one face.
NOTHING_MOVED = 1e-9

#: Triangles measured when scoring how far the emitted surface departs from the
#: drawn one. Every triangle would be exact and slow on a shape with tens of
#: thousands of them, and the departure is a property of the tessellation rather
#: than of any one triangle.
TRIANGLES_MEASURED = 400


def _departure(Part, FreeCAD, shape, pieces):
    """How far the surface that will be solved sits from the surface drawn.

    The chord of a polygonised curve lies off the curve, so near the boundary
    the kernel and the engine disagree honestly and a comparison there settles
    nothing. This is the width of that band, measured: the distance from each
    emitted triangle's middle to the real boundary. A shape whose faces are
    planar is triangulated exactly and measures zero.
    """
    boundary = Part.Compound(shape.Faces)
    worst = 0.0
    for piece in pieces:
        vertices, faces = piece.vertices or (), piece.faces or ()
        if not faces:
            continue
        step = max(1, len(faces) // TRIANGLES_MEASURED)
        for face in list(faces)[::step]:
            corners = [vertices[index] for index in face]
            middle = FreeCAD.Vector(
                sum(c[0] for c in corners) / 3.0,
                sum(c[1] for c in corners) / 3.0,
                sum(c[2] for c in corners) / 3.0,
            )
            worst = max(worst, boundary.distToShape(Part.Vertex(middle))[0])
    return worst


#: Places along one edge when reading what it bends through. Far denser than the
#: translation's own sampling: what the tests score is the translation's answer
#: against a radius measured independently of it, and a reference read the same
#: way would agree with itself whatever it did.
PLACES_FOR_A_RADIUS = 400


def _bends(edge):
    """The sharpest radius one edge bends through, in mm, or ``None``.

    Measured here rather than imported so the yardstick does not come from the
    code being scored. Both ends are looked at, since a conic trimmed at its own
    vertex carries its sharpest point there.
    """
    low, high = float(edge.FirstParameter), float(edge.LastParameter)
    sharpest = None
    for i in range(PLACES_FOR_A_RADIUS):
        at = low + (high - low) * i / (PLACES_FOR_A_RADIUS - 1)
        try:
            bend = abs(float(edge.curvatureAt(at)))
        except Exception:
            continue
        if bend > 0.0:
            sharpest = 1.0 / bend if sharpest is None else min(sharpest, 1.0 / bend)
    return sharpest


def _outline_departure(Part, FreeCAD, shape, piece):
    """How far the polygon that will be solved stands from the outline drawn.

    A sheet's face is flat and is triangulated exactly, so everything the kernel
    was asked to follow is on the boundary. The polygon is taken from the
    emitted triangles rather than from the shape - the edges belonging to one
    triangle only - because that is what openEMS will be handed.

    Measured at each of the polygon's own segment middles, where a chord stands
    furthest from the arc it was cut across, and each of those is charged to the
    drawn edge nearest it and scored against that edge's own radius. Scoring a
    whole sheet against the sharpest radius anywhere on it would divide a
    departure sitting on the widest curve by a radius from the narrowest.

    Comes back as the distance in mm and as that share, and ``None`` for both
    where nothing on the outline bends.
    """
    vertices, faces = piece.vertices, piece.faces
    held = {}
    for triangle in faces:
        for one, two in zip(triangle, triangle[1:] + triangle[:1]):
            key = (min(one, two), max(one, two))
            held[key] = held.get(key, 0) + 1
    # An edge belonging to one triangle only. A closed triangulation has none,
    # which is a sheet that came back as something other than an area.
    rim = [pair for pair, count in held.items() if count == 1]
    radii = [_bends(edge) for edge in shape.Edges]
    if not rim or all(radius is None for radius in radii):
        return None, None
    worst_distance = worst_share = 0.0
    for one, two in rim:
        middle = FreeCAD.Vector(*(0.5 * (a + b) for a, b in zip(vertices[one], vertices[two])))
        vertex = Part.Vertex(middle)
        nearest = min(
            (
                (float(edge.distToShape(vertex)[0]), radius)
                for edge, radius in zip(shape.Edges, radii)
            ),
            key=lambda pair: pair[0],
        )
        distance, radius = nearest
        if radius is None:
            # The segment lies along a straight run, which is followed exactly.
            continue
        worst_distance = max(worst_distance, distance)
        worst_share = max(worst_share, distance / radius)
    return worst_distance, worst_share


def _samples(Part, FreeCAD, shape):
    """Points the kernel has an opinion about, and what that opinion is.

    A containment check needs points where the answer is known and not merely
    plausible. Points *on* the boundary are the ones that matter and the ones a
    scatter through the volume never finds: a zero-thickness sheet is hit by no
    random point at all, so a check made only of those would pass on a sheet
    that was never emitted.

    So the boundary is walked directly. Each face is sampled inside its own
    parameter domain and clear of its own outline, and around each of those
    points a pair is placed a short way along the normal - one the kernel puts
    inside the shape and one it puts outside.

    ``distance`` is measured to the shape's **faces** rather than to the shape.
    A solid contains its interior points, so it answers zero distance for all of
    them, which says nothing about how near the boundary they are.
    """
    box = shape.BoundBox
    span = max(box.XLength, box.YLength, box.ZLength)
    extents = [e for e in (box.XLength, box.YLength, box.ZLength) if e > 0.0]
    # Off the surface by a share of the *thinnest* extent, so that on a plate
    # the companion placed inward stays inside instead of crossing to the far
    # face and reporting itself outside.
    step = MARGIN * (min(extents) if extents else span)
    boundary = Part.Compound(shape.Faces) if shape.Faces else None

    placed = []
    for face in shape.Faces[:FACES_SAMPLED]:
        low_u, high_u, low_v, high_v = face.ParameterRange
        for number in range(POINTS_PER_FACE):
            across = (number + 0.5) / POINTS_PER_FACE
            along = ((number * 3 + 1) % POINTS_PER_FACE + 0.5) / POINTS_PER_FACE
            u = low_u + across * (high_u - low_u)
            v = low_v + along * (high_v - low_v)
            if not face.isPartOfDomain(u, v):
                continue
            try:
                middle = face.valueAt(u, v)
                normal = face.normalAt(u, v)
            except Exception:  # noqa: BLE001 - a face with no normal here says nothing
                continue
            # Clear of the outline by a share of *this face's* size. Against the
            # whole shape's, a face much smaller than the shape it belongs to -
            # the stroke of a letter in a line of text - has no interior left
            # and contributes no samples at all.
            near = face.BoundBox
            room = MARGIN * max(near.XLength, near.YLength, near.ZLength)
            if min(wire.distToShape(Part.Vertex(middle))[0] for wire in face.Wires) < room:
                continue
            placed.append((middle, True))
            placed.append((middle - normal * step, False))
            placed.append((middle + normal * step, False))

    seed = 12345
    for _ in range(SCATTERED):
        # A fixed sequence, so a failure is the same failure on the next run.
        shares = []
        for _ in range(3):
            seed = (1103515245 * seed + 12345) % (1 << 31)
            shares.append((seed % 10000) / 10000.0)
        placed.append(
            (
                FreeCAD.Vector(
                    box.XMin + shares[0] * box.XLength,
                    box.YMin + shares[1] * box.YLength,
                    box.ZMin + shares[2] * box.ZLength,
                ),
                False,
            )
        )

    # Each solid is asked separately. ``isInside`` on a compound does not
    # consider all of them, so a point lying in the second of two disjoint lumps
    # is answered as though the lump were not there.
    solids = list(shape.Solids)
    found = []
    for point, on_surface in placed:
        found.append(
            {
                "point": [float(point.x), float(point.y), float(point.z)],
                "inside": (
                    any(solid.isInside(point, 0.0, True) for solid in solids) if solids else None
                ),
                "on_surface": on_surface,
                "distance": (
                    float(boundary.distToShape(Part.Vertex(point))[0]) if boundary else 0.0
                ),
            }
        )
    return found


def _flatness(shape, pieces, dielectric):
    """What the skip of a planar face costs the demands, and what it drops.

    ``lfs._curves`` asks a face's surface whether it is planar and skips the
    face where it says so. Only the kernel can score that: every stand-in in
    ``tests/test_lfs.py`` answers zero for a face it was told is flat, and so
    agrees with the skip by construction.

    ``isPlanar`` is a distance from a fitted plane rather than a curvature, so a
    free-form face small enough can be called planar and still carry a radius.
    What that may cost is therefore measured rather than assumed: the demands
    :func:`lfs._curvatures` raises are read with the skip and with every face
    asked, and the two sets are written down for a comparison.

    The turn each skipped face makes across the shape's own diagonal is written
    beside them, as evidence rather than as a criterion - a radius larger than
    the body bends nothing within it.
    """
    from Microwave.Solvers.openems import document, lfs
    from tests.corpus import CELL_CAP, EDGE_SIZE

    read = {"skipped": 0, "asked": 0, "worst_skipped": 0.0, "worst_asked": 0.0, "refused": 0}
    seen = set()
    for one in [shape] + [piece.shape for piece in pieces if piece.shape is not None]:
        if id(one) in seen:
            continue
        seen.add(id(one))
        across = float(one.BoundBox.DiagonalLength)
        for face in one.Faces[:FACES_SAMPLED]:
            worst = 0.0
            for u, v in lfs._lattice(face, lfs.CURVATURE_SAMPLES):
                try:
                    curvatures = face.curvatureAt(u, v)
                except Exception:
                    read["refused"] += 1
                    continue
                worst = max(worst, max(abs(float(value)) for value in curvatures))
            side = "asked" if lfs._curves(face) else "skipped"
            read[side] += 1
            read[f"worst_{side}"] = max(read[f"worst_{side}"], worst * across)

    bodies = [document.measured_body(piece, not dielectric) for piece in pieces]

    def demands():
        """Every curvature demand the measurement layer raises, as it raises
        them - past the reach that drops one asking for more than the mesh can
        span, which is what decides whether a skipped face cost anything."""
        found = []
        for body in bodies:
            stations = lfs._Stations(body, lfs._spacing(CELL_CAP, EDGE_SIZE))
            found.extend(
                (one.thickness, one.lower, one.upper, one.source)
                for one in lfs._curvatures(body, CELL_CAP, EDGE_SIZE, stations)
            )
        return sorted(found)

    kept = demands()
    asking_every_face = lfs._curves
    lfs._curves = lambda face: True
    try:
        whole = demands()
    finally:
        lfs._curves = asking_every_face

    read["demands"] = len(kept)
    read["demands_asking_every_face"] = len(whole)
    read["demands_the_skip_drops"] = sorted(set(whole) - set(kept))[:4]
    return read


def _counts(face, spacing):
    """One face's station count per parameter direction, and the two lengths
    those counts were taken from.

    The lengths come back because a count is an integer read off one: a face
    whose length is a whole number of the spacing sits on the step the count
    takes, and a reader given the counts alone cannot see that it did.
    """
    from Microwave.Solvers.openems import lfs

    low_u, high_u, low_v, high_v = (float(value) for value in face.ParameterRange)
    middle_u, middle_v = 0.5 * (low_u + high_u), 0.5 * (low_v + high_v)
    across, width = lfs._steps(face, (low_u, middle_v), (high_u, middle_v), spacing)
    along, height = lfs._steps(face, (middle_u, low_v), (middle_u, high_v), spacing)
    return across, along, width, height


def _stations(shape):
    """Where a face's lattice is placed, read both ways round.

    ``lfs._on_face`` places a whole lattice at once, off the face's own area and
    its own boundary, and asks the kernel only where the two could disagree.
    ``lfs._on`` asks the kernel about one parameter pair. This lays the lattice
    the measurement layer lays and reads it both ways, so a face the fast route
    places differently is a specimen that says so.

    The kernel is what the stand-ins in ``tests/test_lfs.py`` cannot be. A route
    that reads a face's wires, its parameter curves and its own area is a route
    those stand-ins agree with by construction.
    """
    from Microwave.Solvers.openems import lfs
    from tests.corpus import CELL_CAP, EDGE_SIZE

    spacing = lfs._spacing(CELL_CAP, EDGE_SIZE)
    faces = walked = asked = apart = 0
    for face in shape.Faces[:FACES_SAMPLED]:
        low_u, high_u, low_v, high_v = (float(value) for value in face.ParameterRange)
        across, along, _width, _height = _counts(face, spacing)
        pairs = [
            (
                low_u + (high_u - low_u) * (i + 0.5) / across,
                low_v + (high_v - low_v) * (j + 0.5) / along,
            )
            for i in range(across)
            for j in range(along)
        ]
        steps = ((high_u - low_u) / across, (high_v - low_v) / along)
        placed = lfs._on_face(face, pairs, (low_u, high_u, low_v, high_v), steps)
        alone = [lfs._on(face, pair) for pair in pairs]
        faces += 1
        walked += len(pairs)
        asked += sum(alone)
        apart += sum(1 for one, other in zip(placed, alone) if bool(one) != bool(other))
    return {"faces": faces, "walked": walked, "kept": asked, "apart": apart}


def _piece(piece):
    departure = piece.departure
    return {
        "label": piece.label,
        "lower": [float(v) for v in piece.box.lower],
        "upper": [float(v) for v in piece.box.upper],
        "vertices": [[float(c) for c in vertex] for vertex in (piece.vertices or ())],
        "faces": [[int(i) for i in face] for face in (piece.faces or ())],
        "sheet_normal": None if piece.sheet_normal is None else int(piece.sheet_normal),
        "thickened": float(piece.thickened),
        # Named for what it is rather than "departure", which this artifact
        # already uses for the band a containment check is scored inside.
        "reported_displacement": (
            None
            if departure is None
            else {
                "displaced": float(departure.displaced),
                "turns_both_ways": bool(departure.turns_both_ways),
            }
        ),
    }


#: Facets measured one by one when scoring the reported displacement. This costs
#: one kernel query per facet where the reported figure costs none, so a budget
#: keeps a corpus run from spending its time on whichever specimen carries the
#: most facets. A piece over it is measured on none, and says so.
FACETS_SCORED = 40000


def _displacement(Part, FreeCAD, shape, vertices, faces):
    """The mean displacement of a triangulation, measured facet by facet.

    The yardstick the reported figure is scored against, by a different method
    rather than the same arithmetic twice: this asks the kernel how far each
    facet's middle stands from the surface, where the reported one divides a lost
    volume by an area and touches no facet.

    Averaged over the area that departed, because a flat face is triangulated
    exactly and averaging its zeros in would measure how much of the shape is
    flat rather than how far the curved part moved.
    """
    if len(faces) > FACETS_SCORED:
        return None
    walls = list(shape.Shells) or list(shape.Faces)
    if not walls:
        return None
    span = max(shape.BoundBox.XLength, shape.BoundBox.YLength, shape.BoundBox.ZLength) or 1.0
    base = shape.Vertexes[0]
    total = weight = worst = 0.0
    for face in faces:
        corners = [vertices[index] for index in face]
        middle = [sum(c[dim] for c in corners) / 3.0 for dim in range(3)]
        probe = base.copy()
        here = probe.Point
        probe.translate((middle[0] - here.x, middle[1] - here.y, middle[2] - here.z))
        distance = min(wall.distToShape(probe)[0] for wall in walls)
        # Below this the kernel is answering its own rounding rather than a
        # departure, and counting the area under it would dilute the mean with
        # facets that did not move.
        if distance <= NOTHING_MOVED * span:
            continue
        first, second, third = (FreeCAD.Vector(*corners[n]) for n in range(3))
        total += distance * (second - first).cross(third - first).Length / 2.0
        weight += (second - first).cross(third - first).Length / 2.0
        worst = max(worst, distance)
    return {"mean": (total / weight if weight else 0.0), "worst": worst}


def _lengths(pieces, dielectric):
    """Every length the drawing carries, as the mesher would be handed them.

    The bodies are built the way the translation layer builds them: one per
    piece, each holding the geometry that piece names, measured only where it
    reached the mesher as triangles rather than as a box.

    A specimen is a conductor unless it says otherwise, that being the material
    measured in the most ways - a dielectric is asked for nothing at its edges
    and nothing about its curvature, only that several cells span it.

    The pieces of every object the specimen drew arrive here together, so what
    is asked of a gap does not depend on whether the two sides of it were drawn
    in one operation or two.

    The count is passed whatever the material. It reaches only a dielectric, so
    handing it over unconditionally keeps one path rather than two, and a
    conductor's demands are what they were.

    One call, read several ways below: measuring a shape is the dear part of this
    probe, and the bodies come back beside the lengths so that the lattice can
    be read off the bodies that were measured rather than off bodies built a
    second time. The tally comes back with them, for the one field of it that
    is not a cost - what the kernel would not answer for, which is the only
    record of a station this drawing lost.
    """
    from Microwave.Solvers.openems import document, lfs
    from Microwave.Solvers.openems.spend import Spend
    from tests.corpus import CELL_CAP, EDGE_SIZE, ELEMENTS_ACROSS

    bodies = [document.measured_body(piece, not dielectric) for piece in pieces]
    spend = Spend()
    found = lfs.features(bodies, CELL_CAP, EDGE_SIZE, min_lines=ELEMENTS_ACROSS, spend=spend)
    return found, bodies, spend


def _lattice(bodies, dielectric):
    """The station count each measured face gets, per parameter direction, and
    the lengths those counts were taken from.

    Where every length read off a **face** is read is decided by this. A rim's
    samples are laid the same way along an edge - ``lfs._along`` - and are not
    recorded here, so two sides agreeing on every entry below have laid the same
    face lattices and not necessarily the same edge ones.

    ``lfs._steps`` takes each count from a length in space and divides by the
    spacing the demands will ask for, so a lattice is a function of the shape
    rather than of the surface's parameters - up to one step. The count is an
    integer taken from a real, so a face whose length is an exact multiple of
    the spacing sits on the step, and a file that re-measures that length a few
    ulps short lays one row fewer.

    Each entry carries the body it belongs to, because a specimen drawn beside a
    companion is measured with it and a lattice that moved on the companion is
    not a lattice that moved on the specimen. Faces are not numbered: a file may
    hand them back in any order, and what a reader needs is the face's own
    lengths and counts rather than its place in a list.

    The neighbouring ``stations`` key is a different measurement despite the
    word: it holds how far the two routes into :func:`lfs._on_face` disagree
    about which of a lattice's places lie on the face, and it is read on the
    drawn side alone.
    """
    from Microwave.Solvers.openems import lfs
    from tests.corpus import CELL_CAP, EDGE_SIZE

    spacing = CELL_CAP if dielectric else lfs._spacing(CELL_CAP, EDGE_SIZE)
    laid = []
    for body in bodies:
        # A dielectric's stations are read by :func:`lfs._element_counts` alone,
        # and it walks no sheet - an area has no thickness to count cells
        # across. A conductor's curvature is asked of every face whatever the
        # face answers.
        if not body.measured or (dielectric and body.sheet):
            continue
        for face in lfs._faces(body.shape):
            across, along, width, height = _counts(face, spacing)
            laid.append([body.label, int(across), int(along), float(width), float(height)])
    return laid


def _demands(found):
    """Those lengths projected onto the three axes, which is what a grid reads."""
    from Microwave.Solvers.openems.sizing import demands

    return [
        [[float(one.lower), float(one.upper), float(one.size)] for one in axis]
        for axis in demands(found)
    ]


def _joins(found):
    """What each demand across an edge asks for, kept whole rather than projected.

    Such a demand is stated over the whole plane of directions square to the
    edge, and the projection above keeps only how far it reaches on each axis -
    the half that changes with how the part was turned. So the line the edge runs
    along is carried here beside the cells it leaves, and a reader can ask what
    the demand spends along any direction across the edge.

    Recognised by carrying a tangent, which nothing but an edge does. A sheet's
    outline is one of these, being stated as the same demand for the same reason,
    so a shape with no join at all still writes entries here.

    One entry per distinct pair of line and cells, carrying one of the places it
    was asked at: a straight edge is sampled along its length and every sample
    asks the same thing, and an edge reached from each of its two faces gives the
    line both ways round.

    The place says which join asked. Two joins on one shape can ask for the same
    thing along the same line, so without a place a demand that stopped being
    made would be invisible behind its neighbours. A place lies strictly inside
    the edge it was taken from, the samples being cell-centred, so it names one
    join and not two.

    An axis the edge runs along is unconstrained, and comes back as ``null``
    rather than as an infinity JSON has no word for.
    """
    distinct: dict = {}
    for one in found:
        if one.tangent is not None:
            distinct.setdefault((one.tangent, one.cells()), one.lower)
    return [
        {
            "tangent": [float(c) for c in tangent],
            "cells": [float(c) if math.isfinite(c) else None for c in cells],
            "place": [float(c) for c in place],
        }
        for (tangent, cells), place in distinct.items()
    ]


def _translated(doc, shape, label, skin):
    """One drawn object through the geometry layer, and what went wrong.

    The failure comes back rather than being raised: a refusal is a verdict this
    probe is here to record, and a crash is the defect it exists to catch.
    """
    from Microwave.Solvers.openems.geometry import solid_boxes
    from Microwave.Solvers.openems.properties import TranslationError

    obj = _drawn(doc, shape, label)
    try:
        return solid_boxes(obj, skin), None
    except TranslationError as error:
        return [], {"status": "refused", "reason": str(error)}
    except Exception as error:  # noqa: BLE001 - anything else is the defect
        return [], {
            "status": "crashed",
            "reason": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    finally:
        # Dropped so the next shape can be drawn under the same label. FreeCAD
        # keeps labels unique within a document and a refusal quotes the label,
        # so a shape and its round trip would otherwise be refused in words that
        # differ where nothing about the geometry does.
        doc.removeObject(obj.Name)


def _verdict(doc, shape, label, skin=None, beside=None, dielectric=False):
    """What the geometry layer makes of one specimen, and the pieces it made.

    The pieces come back live as well as serialised, because the containment
    samples below are scored against the triangles rather than against the file.

    ``beside`` is a **second object** drawn in the same document. Only the
    mesher is shown both: what the record carries under ``pieces``, and
    everything scored against it, stays the subject's own, so a companion is
    read nowhere but in what the drawing asks the grid for.
    """
    pieces, failure = _translated(doc, shape, label, skin)
    if failure is not None:
        return failure, []

    alongside = []
    if beside is not None:
        alongside, failure = _translated(doc, beside, f"{label} beside", skin)
        if failure is not None:
            return failure, []

    found, bodies, spend = _lengths(pieces + alongside, dielectric)
    record = {
        "status": "meshed",
        "pieces": [_piece(piece) for piece in pieces],
        "demands": _demands(found),
        "joins": _joins(found),
        "lattice": _lattice(bodies, dielectric),
        # Per source: stations offered, and how many of those the kernel would
        # not answer for. Written whether or not anything was refused, so an
        # empty record and a probe that stopped filling one look different.
        "refused": {source: list(counts) for source, counts in spend.refused.stations.items()},
    }
    if beside is not None:
        record["beside"] = [_piece(piece) for piece in alongside]
    return record, pieces


def _round_tripped(Part, doc, shape, label, skin=None, beside=None, dielectric=False):
    """The same shape written to STEP and read back, put through the same layer.

    STEP carries surfaces and their trimming, and nothing of what drew them, so
    a verdict that changes here was read off something other than the geometry.
    A shape the exporter or the importer will not carry says nothing about this
    workbench and is reported as such.

    A companion goes through it too. What the invariant is read over is the
    whole drawing, and a gap whose far side never left the kernel would be
    compared against itself.
    """
    from tests.corpus import round_trip

    try:
        back = round_trip(Part, shape, "step")
        alongside = None if beside is None else round_trip(Part, beside, "step")
    except Exception as error:  # noqa: BLE001 - a file format's limit is not a verdict
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}

    record, _ = _verdict(doc, back, label, skin, alongside, dielectric)
    record["measured"] = _measured(back)
    return record


def _run_one(FreeCAD, Part, doc, specimen):
    from tests.corpus import Unavailable

    record = {"name": specimen.name, "subject": specimen.subject, "expect": specimen.expect}
    try:
        shape = specimen.build(Part)
        beside = None if specimen.beside is None else specimen.beside(Part)
    except Unavailable as error:
        record["status"] = "unavailable"
        record["reason"] = str(error)
        return record
    except Exception as error:  # noqa: BLE001 - a builder fault is not a verdict
        record["status"] = "undrawable"
        record["reason"] = f"{type(error).__name__}: {error}"
        return record

    record["measured"] = _measured(shape)
    from tests.corpus import SKIN

    skin = SKIN if specimen.skin else None
    record["skin"] = skin
    verdict, pieces = _verdict(doc, shape, specimen.name, skin, beside, specimen.dielectric)
    record.update(verdict)
    if record["status"] == "meshed":
        record["samples"] = _samples(Part, FreeCAD, shape)
        record["stations"] = _stations(shape)
        record["flatness"] = _flatness(shape, pieces, specimen.dielectric)
        record["departure"] = _departure(Part, FreeCAD, shape, pieces)
        # The yardstick the reported displacement is scored against, written
        # beside it so the two can be compared on the same triangles.
        for written, piece in zip(record["pieces"], pieces):
            written["measured_displacement"] = (
                None
                if piece.sheet_normal is not None or not piece.faces
                else _displacement(Part, FreeCAD, piece.shape, piece.vertices, piece.faces)
            )
            # A sheet is scored on its boundary rather than on its facets: its
            # face is flat and triangulated exactly, and the only thing the
            # kernel was asked to follow is the outline.
            outline = (
                (None, None)
                if piece.sheet_normal is None or not piece.faces
                else _outline_departure(Part, FreeCAD, piece.shape, piece)
            )
            written["outline_departure"], written["outline_share"] = outline
    record["round_trip"] = _round_tripped(
        Part, doc, shape, specimen.name, skin, beside, specimen.dielectric
    )
    return record


def _the_tree_this_file_is_in():
    """Refuse to score a workbench other than the one beside this probe.

    Showing the main window runs every installed addon's ``InitGui.py``, so by
    the time this file imports anything, ``Microwave`` is already in
    ``sys.modules`` - taken from wherever FreeCAD found it under ``Mod/``, with
    the ``sys.path`` entry above read by nobody. A probe run from a tree that is
    not the installed one would measure the installed one and say nothing about
    its own.

    Raised rather than reported, so no manifest is written and the caller fails
    with this on its output.
    """
    import Microwave

    loaded = pathlib.Path(Microwave.__file__).resolve().parent
    beside = (HERE.parent / "Microwave").resolve()
    if loaded != beside:
        raise SystemExit(
            f"CORPUS: this probe is in {beside}, and FreeCAD has already loaded "
            f"the workbench from {loaded} - so the run would score a tree other "
            "than its own. Install this one, or run against the installed one"
        )


def main():
    import FreeCAD
    import FreeCADGui
    import Part

    # A view provider only exists once the Gui layer is up, and the display-mesh
    # trap only exists where a view provider does.
    FreeCADGui.showMainWindow()

    _the_tree_this_file_is_in()

    from tests import corpus

    OUT.mkdir(parents=True, exist_ok=True)
    doc = _document(FreeCAD)

    written = []
    for specimen in corpus.specimens():
        record = _run_one(FreeCAD, Part, doc, specimen)
        path = OUT / f"{specimen.name}.json"
        path.write_text(json.dumps(record, indent=1, sort_keys=True))
        written.append(specimen.name)
        print(f"CORPUS: name={specimen.name} status={record['status']}")

    (OUT / "manifest.json").write_text(json.dumps({"specimens": written}, indent=1))
    print(f"CORPUS: written={len(written)} into {OUT}")
    sys.stdout.flush()


main()
