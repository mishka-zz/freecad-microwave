# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Where openEMS puts a conductor's surface, and what to hand it so it lands right.

openEMS zeroes an electric field edge when a single sample point on it reads as
metal, with no averaging and no partial fill (``Operator::CalcPEC_Range``,
``openEMS/FDTD/operator.cpp:2045``, sampled at ``:2062-2075``). That point lies
on the primary grid lines across the edge and midway along it
(``Operator::GetYeeCoords``, ``operator.cpp:183-186``, over
``Operator::GetDiscLine``, ``:152-156``).

The effect on a curved surface is measured rather than argued from the
sampling. It arrives inside where it was drawn, by a share of the cell that
``tests/test_fidelity.py`` reads off a cylinder and a sphere. The error is
one-sided, so it does not average out over the surface, and it is proportional
to the cell, so refining does not remove it.

The correction is to grow a conductor by half the cell it will be sampled on,
which puts the last line still inside it on the drawn surface. It grows on the
side the field is on, because the metal loses either way: a cavity wall opens the
cavity out, and a solid in a field comes back thin.

These boundaries are left alone:

- A **flat surface square to an axis**. The mesher gives it a lattice plane of
  its own, and it needs only to own that plane (:data:`PINNED_CLEARANCE`). Half a
  cell would move it off the drawing by all of that. It is held in its own plane
  even where its rim is grown, because a cap and the wall it closes share every
  vertex the cap has. It is held there where it is blended into a curve by a
  tangent join too, because squareness rather than the angle of the join
  recognises it. A flat surface lying oblique has no coordinate to pin and no
  line on it, so it staircases like a curved one and is grown like one.
- A **dielectric**. It reaches the operator averaged over quarter cells
  (``AverageMatQuarterCell``, ``operator.cpp:1447``) and carries no such bias.
- A **sheet's rim**. It is point-sampled and does come back cut in, by an amount
  nothing has measured. A flat conductor's charge sits at its outline where the
  grid is least accurate, so a capacitance cannot separate what the rim gave up
  from what the discretisation gained. ``tests/test_fidelity.py`` shows the two
  are the same size on a plate the grid holds exactly.

The half is established by the size of the error it leaves rather than by the
rate that error falls at. The correction does not change the rate: a
doubly-curved surface goes on meeting the grid at every phase however fine the
grid is, so what is left stays proportional to the cell. The correction changes
the coefficient, by a large factor. A correction of the wrong size leaves a share
of the cell several times larger while still looking better than none.

``tests/test_staircase_model.py`` holds both halves with no solver in it. It
solves the same rule as electrostatics on a coaxial cross-section, averaged over
where the lattice falls, since what is left is small enough that a single grid
per cell size is scatter rather than a sequence.
``tests/test_acceptance_cavity.py`` holds the size against a real solve, on every
mesh it runs, and at the meshes ``tests/cavity.py`` lists as *as drawn* it solves
the same cavity with the share set to zero. How far the engine's own reading
stands from the drawing is therefore measured on Maxwell rather than read off the
source above.

A sphere cannot establish where the rule stops, having no flat face to leave
alone. A cylinder can, and ``tests/pillbox.py`` draws one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .surface import winding

DIMENSIONS = 3

#: Grid line positions, one entry per axis. Each axis is whatever numpy will
#: make an array of - a list written out in a test, or the array a finished mesh
#: carries.
GridLines = Sequence[npt.ArrayLike]

#: Corners to a triangle. Equal to :data:`DIMENSIONS` and a different quantity:
#: one counts axes, the other counts the points of a face.
CORNERS = 3

#: What share of the cell a curved conductor is grown by.
#:
#: A choice rather than a derivation. A conductor here is only electrically
#: perfect: the rule zeroes the electric operator and leaves the magnetic one
#: alone. A resonance and an impedance are therefore read off two walls about a
#: sixth of a cell apart, and one set of vertices cannot serve both. The half
#: serves the resonance. A detuned filter is a worse failure than a line a per
#: cent mismatched.
#:
#: The envelope carries a share of its own and defaults it to this. A run is
#: built from that field rather than from this constant, so what the driver
#: builds is a function of the file it was handed. A drawing carries no such
#: setting. The correction answers for how the engine reads a surface, which is
#: the adapter's business rather than the user's.
GROWN_BY = 0.5

#: What share of the cell a flat conductor face square to an axis is grown by.
#:
#: The share is not zero. openEMS decides what a polyhedron contains by casting a
#: segment from the point to one it takes to be outside and counting the faces
#: crossed, and a point lying on a face leaves that count unsure. A line pinned
#: exactly to a flat face can therefore read as air, which leaves the tangential
#: field on the conductor's plane unzeroed and the wall not a conducting boundary
#: at all. The face is handed over displaced into the void by enough for the
#: pinned line to fall inside the metal.
#:
#: The share is bounded at both ends and set well inside them. The lower bound is
#: how far into a face containment stays unsure, which is a property of the
#: triangulation rather than a fixed tolerance;
#: ``tests/test_pinned_clearance.py`` measures it against the engine and holds
#: the clearance above it. The upper bound is half a cell, where the field edge
#: normal to the face is sampled on the void side of it
#: (``Operator::GetYeeCoords``, ``openEMS/FDTD/operator.cpp:183-186``, which puts
#: a component on the dual line of its own axis). A displacement reaching that
#: zeroes an edge belonging outside the metal and stands the wall a cell inside
#: the drawing, which is the fault this constant exists to remove, with its sign
#: turned round.
#:
#: Whether a given face was left unsure is not decided here. The segment's far
#: end is drawn from ``rand()`` when the primitive is built
#: (``CSXCAD/src/CSPrimPolyhedron.cpp:229-230``), so the answer for a point on a
#: face belongs to that ray. Two faces of one solid can answer differently, and
#: so can two primitives of one shape. A clearance of nothing therefore costs a
#: range rather than a figure. The envelope carries a share of its own, defaulted
#: to this and used to build the run, so that a pair of runs can read that range
#: on Maxwell instead of on containment.
PINNED_CLEARANCE = 1e-3

#: Below this share of the area meeting at a vertex, the triangles there cancel
#: and the vertex is left alone. There is no direction to grow along, and
#: normalising what is left would turn a rounding error into a unit vector and
#: step half a cell along it.
#:
#: A share rather than a length, because both sides of it are areas. Stated
#: against a length it would mean different things in millimetres and in metres.
TOO_FLAT = 1e-9

#: How far the faces of one surface must disagree about which way is out before
#: that surface counts as curved. Coplanar faces sum to exactly the area they
#: cover, so the ratio is one; any curvature brings it below, and a triangulated
#: curve clears this by orders of magnitude.
#:
#: A flat surface is treated separately rather than as a small case of curved. A
#: flat face square to an axis has a lattice plane on it and asks for
#: :data:`PINNED_CLEARANCE` rather than for half a cell. The one ratio bounds the
#: calculation at both ends: below :data:`TOO_FLAT` there is no direction, and
#: above this there is a direction and nowhere for the surface to travel that the
#: grid is not already holding.
PLANAR = 1e-9

#: How far two faces' normals may disagree and still describe one surface. Above
#: it they are separate faces meeting along an edge, and each answers for where
#: it is on its own.
#:
#: Asked between faces that share an edge, so it separates a facet of a curve
#: from a corner. Both ends of the range belong to the drawings rather than to
#: this rule, and the threshold is set between them:
#:
#: - **Below**, by how coarsely a kernel tessellates a curve. A threshold under a
#:   kernel's own joins cuts a swept surface into pieces, and a piece of a curve
#:   is flat, so it would pin a plane across the middle of one and hold it there.
#:   A kernel's own joins come nearer to this end than a reading of the code
#:   suggests, and nothing here measures them. Look at a swept surface first if
#:   this threshold is ever narrowed.
#: - **Above**, by the shallowest join a drawing means as an edge, which is a
#:   chamfer at forty-five degrees. A threshold over that merges a chamfer into
#:   the face it cuts.
#:
#: A join built to be tangent clears neither end, and no threshold reaches it: a
#: fillet blending into a flat face joins it at nothing at all. Squareness
#: decides that case instead, and this threshold decides the rest - see
#: :func:`_surfaces`.
SMOOTH = math.radians(30.0)

#: Below this a normal's component is zero and the face is square to the other
#: axes. One part in a million of a unit vector: a face drawn square carries only
#: the rounding of the cross product that measured it, orders below this, and a
#: face drawn a thousandth of a degree off carries far more.
SQUARE = 1e-6

#: How far one face's own normal may lean off an axis and still belong to a run
#: the grid can hold.
#:
#: Looser than :data:`SQUARE`, which is asked of a whole run's summed normal and
#: can be that tight because the sum is exact where the run is flat. A single
#: triangle of a face a drawing means as flat carries whatever the kernel left in
#: it. :data:`PLANAR` already says how much of that a run may carry and still be
#: one plane. It is an area ratio, which a lean of theta costs about theta
#: squared over two, and reading that back gives the lean below.
#:
#: Derived rather than chosen, because the two tests decide one question and must
#: not answer it differently. A face :data:`PLANAR` would have held whole, split
#: here by a tighter test, arrives as two runs, both pinned, and the mesher
#: refuses a pair of anchors closer than the cell floor.
NEARLY_SQUARE = math.sqrt(2.0 * PLANAR)

#: How much wider than what it abuts a run of coplanar faces square to an axis
#: must be before it is held in its own plane.
#:
#: Asked only of a run that runs tangentially into something else. A run bounded
#: by corners has nothing to be part of and is held whatever its width. This
#: threshold therefore decides whether a flat strip lying in the middle of a
#: curve is a face of its own.
#:
#: The lower bound is what such a strip costs when it is not a face. A curve's
#: own tessellation leaves a strip in a plane wherever the phase puts a facet
#: across a tangency - a cylinder does it along a line and a torus all the way
#: round a circle - and that plane is a chord standing inside the surface it
#: claims to be. Held, it pins a grid line at the chord, and an anchor is refused
#: outright when another lands within the cell floor of it, so a rod drawn
#: against a wall would stop meshing. Such a strip is about as wide as the facets
#: it abuts, and ``tests/test_corpus_staircase.py`` measures the ground between
#: that and a face over every drawing the corpus makes.
#:
#: Nothing forces an upper bound, so the threshold is set well clear rather than
#: close. A rejected run is grown like the curve it lies in, which costs accuracy
#: on a narrow face and never a refusal.
FACE_IS_WIDER_BY = 4.0


def _face_normals(triangles: np.ndarray) -> np.ndarray:
    """Twice each triangle's area, along its normal, so that a sum is area-weighted."""
    return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])


def _outward(points: np.ndarray, corners: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """The same normals, turned to point out of the solid they bound.

    A drawing does not settle the winding. A kernel can hand back the complement
    of a region, wound the other way and occupying the same space.
    :func:`~.surface.winding` answers for the whole set at once, so a bore's own
    negative volume does not turn its normals round on their own.
    """
    return normals * winding(points, corners)


def _cell_sizes(points: np.ndarray, grid: GridLines) -> np.ndarray:
    """The cell each point sits in, per axis.

    A point on a line belongs to the cell above it. The choice has no
    consequence: the two cells differ only where the grading changes, and by less
    than the grading ratio allows.
    """
    sizes = np.empty_like(points)
    for axis in range(DIMENSIONS):
        lines = np.asarray(grid[axis], dtype=float)
        if len(lines) < 2:
            raise ValueError(f"axis {axis} has no cell to measure, only {len(lines)} lines")
        widths = np.diff(lines)
        above = np.searchsorted(lines, points[:, axis], side="right") - 1
        sizes[:, axis] = widths[np.clip(above, 0, len(widths) - 1)]
    return sizes


def _square_axes(units: np.ndarray) -> np.ndarray:
    """Which axis each face is square to, or ``-1`` where it is square to none.

    Asked per face, so at :data:`NEARLY_SQUARE` rather than at :data:`SQUARE`.
    This decides whether two faces describe one surface, so it has to admit every
    lean the run they would make is allowed to carry.
    """
    above = np.abs(units) > NEARLY_SQUARE
    return np.where(above.sum(axis=1) == 1, np.argmax(above, axis=1), -1)


def _width(points: np.ndarray, corners: np.ndarray, faces: Sequence[int]) -> float:
    """How wide a run of faces is, as twice the area it covers over its rim.

    Twice the area over the perimeter gives an annulus' width exactly, a long
    strip's width in the limit, and a length rather than a reach for a run bent
    round a corner or pierced by a hole. A span gives none of those: the band
    round a torus' tangent circle is a fraction of a facet wide and reaches right
    across the shape. A ratio of two of these widths is what decides anything, so
    both ends of the comparison have to be about the same quantity.
    """
    triangles = points[corners[faces]]
    area = (
        float(
            np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                axis=1,
            ).sum()
        )
        / 2.0
    )
    times: dict[tuple[int, int], int] = {}
    for face in faces:
        for corner in range(CORNERS):
            ends = (int(corners[face][corner]), int(corners[face][(corner + 1) % CORNERS]))
            edge = (min(ends), max(ends))
            times[edge] = times.get(edge, 0) + 1
    rim = sum(
        float(np.linalg.norm(points[low] - points[high]))
        for (low, high), count in times.items()
        if count == 1
    )
    return 2.0 * area / rim if rim > 0.0 else 0.0


def _surfaces(
    points: np.ndarray,
    corners: np.ndarray,
    normals: np.ndarray,
    judged: list[tuple[list[int], float]] | None = None,
) -> dict[int, list[int]]:
    """Which faces make up each surface of the shape, as ``root -> faces``.

    Two faces are the same surface where they share an edge, their normals agree
    to within :data:`SMOOTH`, and either both are square to one axis or neither
    is square to any. The grouping runs over the whole shape rather than over one
    vertex. A facet of a cylinder and a triangle of a flat cap look alike from a
    vertex - both are one of a coplanar pair sharing an internal edge - and they
    differ only in what they are joined to further out.

    Squareness enters because the angle cannot reach the case that matters. A
    fillet meets the face it is blended into at nothing at all, so no threshold
    on :data:`SMOOTH` separates them. A face square to an axis is the one thing
    the grid can hold, and it answers for where it is on its own whatever it runs
    into.

    A curve's own tessellation makes such a face of its own accord: wherever the
    phase puts a facet across a tangency, that facet lies in one plane and is
    square to an axis. A run like that is a chord rather than a surface, so a run
    that runs tangentially into a curve is kept only where it is wider than what
    it runs into - see :data:`FACE_IS_WIDER_BY`.

    :param judged: A list to append every run the width test reaches to, with how
        much wider than what it runs into it turned out to be. Nothing here reads
        it. It lets a test score the ground the threshold sits in against this
        rule rather than against a copy of it.
    """
    units = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    square = _square_axes(units)
    beside: dict[tuple[int, int], list[int]] = {}
    for face, triangle in enumerate(corners):
        for corner in range(CORNERS):
            ends = (int(triangle[corner]), int(triangle[(corner + 1) % CORNERS]))
            beside.setdefault((min(ends), max(ends)), []).append(face)

    group = list(range(len(corners)))

    def root(face: int) -> int:
        while group[face] != face:
            group[face] = group[group[face]]
            face = group[face]
        return face

    def smooth(one: int, other: int) -> bool:
        return float(np.dot(units[one], units[other])) > math.cos(SMOOTH)

    def gathered() -> dict[int, list[int]]:
        found: dict[int, list[int]] = {}
        for face in range(len(corners)):
            found.setdefault(root(face), []).append(face)
        return found

    for shared in beside.values():
        for one, other in zip(shared, shared[1:]):
            if square[one] == square[other] and smooth(one, other):
                group[root(one)] = root(other)

    def running_into(members: Sequence[int]) -> dict[int, set[int]]:
        """The faces this run continues into, by the surface each belongs to.

        Only the ones it meets at less than :data:`SMOOTH`, because only a curve
        it is tangent to could be the curve it is a facet of. A run bounded by a
        corner is bounded by something the drawing put there.
        """
        mine = set(members)
        found: dict[int, set[int]] = {}
        for face in members:
            for corner in range(CORNERS):
                ends = (int(corners[face][corner]), int(corners[face][(corner + 1) % CORNERS]))
                for other in beside[(min(ends), max(ends))]:
                    if other not in mine and smooth(face, other):
                        found.setdefault(root(other), set()).add(other)
        return found

    # Decided over the grouping as it stands and applied afterwards, so folding
    # one run back cannot change what the next one is measured against. Each run
    # is measured against the widest of what it runs into, which is the cautious
    # end.
    narrow = []
    for members in gathered().values():
        if int(square[members[0]]) < 0:
            continue
        into = running_into(members)
        if not into:
            continue
        widest = max(_width(points, corners, sorted(runs)) for runs in into.values())
        ratio = _width(points, corners, members) / widest if widest > 0.0 else math.inf
        if judged is not None:
            judged.append((members, ratio))
        if ratio <= FACE_IS_WIDER_BY:
            narrow.append((members, into))

    for members, into in narrow:
        for runs in into.values():
            group[root(members[0])] = root(next(iter(runs)))

    return gathered()


def _pinned_normal(
    members: Sequence[int], normals: np.ndarray, areas: np.ndarray
) -> np.ndarray | None:
    """Which way a surface faces where the grid will hold it, else ``None``.

    Two questions in one, because the answer is only useful when both hold.
    Coplanar faces sum to exactly the area they cover, so one ratio
    (:data:`PLANAR`) separates flat from curved, and a surface whose normals
    cancel outright bounds nothing and points nowhere. Being flat is not enough
    on its own. A face gets a line of its own only where it has a coordinate to
    pin, and only a face square to an axis has one.

    Both questions are answered the same whichever way the set is wound. The
    flatness test compares the length of the sum against the summed areas, the
    squareness test reads ``abs(unit[dim])``, and :func:`_surfaces` groups on
    ``np.abs`` and on a dot product of two normals that flip together. Only the
    returned direction carries the sign, and only :func:`grown` reads it, which
    is why ``grown`` winds the set outward first and :func:`flat_planes` reads
    the triangulation as it arrived.
    """
    total = normals[members].sum(axis=0)
    length, covered = float(np.linalg.norm(total)), float(areas[members].sum())
    if length <= TOO_FLAT * covered or covered - length > PLANAR * covered:
        return None
    unit = total / length
    square = [dim for dim in range(DIMENSIONS) if abs(unit[dim]) > SQUARE]
    if len(square) != 1:
        return None
    return np.eye(DIMENSIONS)[square[0]] * math.copysign(1.0, unit[square[0]])


def grown(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    grid: GridLines,
    share: float = GROWN_BY,
    clearance: float = PINNED_CLEARANCE,
) -> tuple[tuple[float, float, float], ...]:
    """The same surface, grown along its curved parts by a share of a cell.

    :param vertices: The triangulation's points, in grid units.
    :param faces: Triangles, as indices into ``vertices``.
    :param grid: The three lists of grid lines, in the same coordinates.
    :param share: What fraction of a cell to grow by. :data:`GROWN_BY` is what
        ships. At zero a curved surface is handed over as drawn, which prices the
        correction against not making it.
    :param clearance: What fraction of a cell to displace a flat face square to
        an axis into the void by. :data:`PINNED_CLEARANCE` is what ships. It is a
        different quantity from ``share`` and does not answer for sampling.
        It makes the line pinned to that face fall in metal at all. At zero the
        face is handed over as drawn.

    The step at a vertex is that share of the cell measured along the normal
    there: the cell's own sides scaled by the direction. An axis-aligned normal
    therefore asks for the share of that axis's cell, and a cubic cell asks for
    the same step whichever way the surface faces.

    The direction comes from the surfaces the grid does not already hold, and
    each surface it does hold takes its own axis back out of that direction.
    Where one surface meets at the vertex, the direction is that surface's
    area-weighted mean normal, which follows a smooth surface exactly. Where
    several meet, the vertex is on an edge. It is stepped by what the unheld
    surfaces ask, that step is taken out of every held surface's axis, and each
    held surface then puts back the clearance it needs to own the line pinned to
    it. A flat face therefore stays where it was drawn to within
    :data:`PINNED_CLEARANCE` however its rim is grown.

    A surface is held where it is flat and square to an axis, which is the pair
    of conditions a grid line needs. A flat face lying oblique is sampled exactly
    as a curved one is and is grown the same way.

    Which surface a face belongs to is decided over the whole shape and not at
    each vertex. Lying on a flat face is not the same as being surrounded by one:
    a cylinder's cap is fanned from a point on its own rim, so every vertex the
    cap has is shared with the wall that curves, and no vertex of it looks flat.

    A curve facetted so coarsely that its own facets disagree by more than
    :data:`SMOOTH` reads as a run of separate faces rather than one surface. Each
    face is still grown along its own normal, so :data:`SMOOTH` decides how the
    step is shared at a vertex rather than whether there is one.

    A flat face blended into a curve by a tangent join is held like any other. It
    is recognised by squareness rather than by the angle it meets the fillet at,
    which is nothing at all - see :func:`_surfaces`. That leaves a flat strip
    lying in the middle of a curve and not much wider than the curve's own
    facets. Such a strip is grown, and :data:`FACE_IS_WIDER_BY` is where the line
    falls.
    """
    points = np.asarray(vertices, dtype=float)
    corners = np.asarray(faces, dtype=int)
    if not len(corners):
        return tuple((float(p[0]), float(p[1]), float(p[2])) for p in points)
    triangles = points[corners]
    normals = _outward(points, corners, _face_normals(triangles))
    areas = np.linalg.norm(normals, axis=1)
    surfaces = _surfaces(points, corners, normals)
    pinned = {root: _pinned_normal(members, normals, areas) for root, members in surfaces.items()}
    belongs = {face: root for root, members in surfaces.items() for face in members}

    meeting: list[dict[int, list[int]]] = [{} for _ in points]
    for face, triangle in enumerate(corners):
        for corner in triangle:
            meeting[int(corner)].setdefault(belongs[face], []).append(face)

    sizes = _cell_sizes(points, grid)
    moved = np.array(points, dtype=float)
    for vertex, here in enumerate(meeting):
        wanted = np.zeros(DIMENSIONS)
        held: list[np.ndarray] = []
        for root, group in here.items():
            settled = pinned[root]
            if settled is not None:
                held.append(settled)
                continue
            # The surface's own normal at this vertex rather than over the whole
            # of it. A closed wall's normals cancel taken together, leaving no
            # direction that is out anywhere on it.
            total = normals[group].sum(axis=0)
            length = float(np.linalg.norm(total))
            if length > TOO_FLAT * float(areas[group].sum()):
                wanted += total / length
        step = np.zeros(DIMENSIONS)
        along = float(np.linalg.norm(wanted))
        if along > 0.0:
            wanted /= along
            step = share * float(np.linalg.norm(wanted * sizes[vertex])) * wanted
            # Each held direction is an axis, so taking them out one at a time
            # is the projection rather than an approximation of it.
            for direction in held:
                step -= float(np.dot(step, direction)) * direction
        # After the projection, so a held face keeps its own clearance and
        # nothing else adds to it.
        for direction in held:
            step += clearance * float(np.abs(direction) @ sizes[vertex]) * direction
        moved[vertex] += step
    return tuple((float(p[0]), float(p[1]), float(p[2])) for p in moved)


def flat_planes(
    vertices: Sequence[Sequence[float]], faces: Sequence[Sequence[int]]
) -> tuple[tuple[int, float], ...]:
    """Where this surface is flat and square to an axis, as ``(axis, position)``.

    :param vertices: The triangulation's points, in grid units.
    :param faces: Triangles, as indices into ``vertices``.

    No correction recovers a flat conductor face. A curved one is
    sampled half a cell in and grown half a cell out to meet it. A flat one is
    grown by nothing, because growing it would move it bodily, so the conductor
    ends up wherever the cell falls. A line on the face settles that, and these
    are the positions to ask for.

    Read off the triangles rather than off the solid's box. A shell's box is its
    outer surface, and the faces bounding the void inside it are nowhere on that
    box.

    Faces that are flat but oblique are not here. They have no single coordinate
    to pin, so the criterion is squareness rather than flatness alone.

    These are exactly the faces :func:`grown` holds, both reading them off
    :func:`_surfaces`. The two have to agree. A line pinned to a face that was
    grown anyway leaves the face half a cell out, and a face held with no line
    pinned to it leaves the wall wherever the grading put it. They agree across
    a difference: ``grown`` winds the set outward first and this does not, and
    every test between here and the grouping is even in that sign.
    """
    points = np.asarray(vertices, dtype=float)
    corners = np.asarray(faces, dtype=int)
    if not len(corners):
        return ()
    normals = _face_normals(points[corners])
    areas = np.linalg.norm(normals, axis=1)

    found: set[tuple[int, float]] = set()
    for members in _surfaces(points, corners, normals).values():
        normal = _pinned_normal(members, normals, areas)
        if normal is not None:
            axis = int(np.argmax(np.abs(normal)))
            found.add((axis, float(points[corners[members]][..., axis].mean())))
    return tuple(sorted(found))
