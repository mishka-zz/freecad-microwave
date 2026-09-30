# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A size at each place a demand states, and what the mesh holds there.

Each place becomes one Gmsh size field, and every field goes into one ``Min``
field set as the background, so the size at a point is the finest any place
asks for there. The coarsest size bounds them all as a Gmsh option. A size
field refines and never coarsens, which is why a place at or above the coarsest
size is refused before this is reached.

* :class:`~.vocabulary.Near` and :class:`~.vocabulary.AtRim` are a ``Distance``
  field over the place's entities read through a ``Threshold``: the place's
  size at them, rising at the growth to the coarsest size. One ``Distance``
  over each class of a place's entities rather than one each, since Gmsh
  evaluates a ``Min`` over every field at each point it asks - see the sampling
  below for what a class is.
* :class:`~.vocabulary.Within` is a ``Constant`` field on the label's entities,
  their boundary included. Without the boundary it sizes the interior alone, and
  the boundary mesh built first bounds that interior at the coarsest size.
* :class:`~.vocabulary.Near` at the filled dimension is both: the ``Constant``
  of a :class:`~.vocabulary.Within` on the entities, and the ``Distance`` of an
  :class:`~.vocabulary.AtRim` on the rim round them. Inside a volume there is
  nothing a distance is measured from.

A place names a label or a mark, and a mark's entities are sized exactly as a
label's are. A mark's point or curve standing inside a volume is not a
constraint on the volume's mesh - see :func:`~.labels.release` - so the
elements at it are the ones holding its nodes rather than the ones with a
corner on it.

A place that stands on no entity lays no field. That is the rim of a label
closed on itself, and the answer says so. Where no place lays a field, the mesh
is the one the two bounds give.

A ``Distance`` field measures from points it lays along its entities, and a
point between two of them is further from the place than the place is. So the
points are laid no further apart than the place's size. Along a curve Gmsh
spaces them evenly in the curve's parameter, and the parameter of a curve such
as a B-spline covers more length per unit along one stretch than along another.
So a curve's sampling is taken off the length it would have if it ran
everywhere as fast as along its fastest stretch. Across a surface they are laid
over triangles covering it, spaced by the triangles' length rather than by the
surface's parameter, at a spacing set by the diagonal of the surface's bounding
box over the sampling and up to twice that where a triangle is not a whole
number of spacings long. So a surface's sampling is taken off twice that
diagonal.

The sampling is one number per field, and Gmsh lays it on every entity of the
field whatever that entity's size (``src/mesh/Field.cpp:2602`` along a curve,
``src/mesh/Field.cpp:2620`` across a surface). A surface is given points of the
order of the square of it, whatever its own size, so a field whose sampling is
taken off its largest surface spends that many on each of its smallest. A place
cut into many small faces beside one large one would spend the large face's
count on every piece. So a place's surfaces are split into classes, each
spanning no more than :data:`CLASS` in the length its sampling is taken off, and
each class is a ``Distance`` of its own with the sampling its longest surface
needs. A surface is then sampled at most :data:`CLASS` times as finely along
each direction as it needs, which is at most the square of that in points. A
curve is given the sampling itself, so a place of curves costs its count of
curves times the sampling and stays one field.

What the answer reports at a place is read off the elements there. Along a
place of curves it is the longest element edge lying on them, on a place of
surfaces the longest edge lying on it, and beside either the median of those
edges; at a place of points the longest edge leaving them; at a mark's point or
curve inside a volume the longest edge of the elements holding it; and
throughout a label or a mark of the filled dimension the median mean edge of the
elements in it. The elements standing on a place are coarser than those along it by more
the steeper the growth, and a growth steep enough leaves the place itself
coarser than asked. The figures say how far the mesh stands from the size at
each place, and nothing here is checked against that size.

This module is the Gmsh half of the package, as :mod:`.mesh` is.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import gmsh
import numpy as np
import numpy.typing as npt

from . import labels, wedges
from .vocabulary import (
    HALF_TURN,
    TURNING,
    AtRim,
    Demand,
    Edges,
    Label,
    Near,
    Place,
    Reached,
    Refused,
    Within,
    transition,
)

__all__ = ["Resolved", "edges", "install", "reached", "resolve"]

#: Which list of a ``Distance`` field holds entities of each dimension.
DISTANCE_LISTS = {0: "PointsList", 1: "CurvesList", 2: "SurfacesList"}

#: Which list of a ``Constant`` field holds entities of each dimension.
CONSTANT_LISTS = {1: "CurvesList", 2: "SurfacesList", 3: "VolumesList"}

#: The fewest points a ``Distance`` field is asked to lay along a curve: its two
#: ends, which Gmsh does not lay, and one between them, which it does.
LEAST_SAMPLING = 3

#: At how many points along its parameter a curve is asked how fast the
#: parameter runs there.
PROBES = 257

#: How far apart in length the surfaces of one ``Distance`` field may be: the
#: longest length a class holds over the shortest. Two keeps a surface's points
#: within four times its own need, and the classes a place is split into grow as
#: the logarithm of how far its lengths spread rather than with its count of
#: surfaces.
CLASS = 2.0

#: The Gmsh options that size a mesh from something other than the fields, each
#: set to zero wherever a place lays a field. Each takes a size from the drawing
#: rather than from the demand: from the elements laid on the curves round a
#: face and on the faces round a volume, and from the size stored at each point.
#: Where one is on, the size an element is meshed to is the least of that
#: source's and the field's, so the growth a field lays away from a place is cut
#: short wherever the source is finer. The size from how sharply the drawing
#: bends is the demand's, :attr:`~.vocabulary.Demand.per_turn`, and is set
#: whether or not a place lays a field; where it is set, these are off too.
OTHER_SOURCES = (
    "Mesh.MeshSizeExtendFromBoundary",
    "Mesh.MeshSizeFromPoints",
)


@dataclass(frozen=True)
class Resolved:
    """One place, as the entities of the fragmented model it stands on.

    :param place: the place as it was asked for.
    :param dimension: the dimension of ``entities``.
    :param entities: what the size is laid at: a label's or a mark's entities,
        or the rim round them. Empty for the rim of a label closed on itself,
        which has none, and for a mark standing wholly outside the model;
        nothing is laid at either.
    :param filled: whether the size is laid throughout ``entities``, which are
        of the filled dimension: a :class:`~.vocabulary.Within`, or a
        :class:`~.vocabulary.Near` at that dimension.
    :param rim: for a :class:`~.vocabulary.Near` at the filled dimension, the
        rim round its entities, which the size grows away from. Empty otherwise.
    :param left: for a ``reentrant`` :class:`~.vocabulary.AtRim`, how many
        curves bounding its label the room turns round by no more than half a
        turn, which are not in ``entities``.
    """

    place: Place
    dimension: int
    entities: tuple[int, ...]
    filled: bool = False
    rim: tuple[int, ...] = ()
    left: int = 0


def resolve(
    places: Sequence[Place],
    given: Mapping[str, Label],
    top: int,
    marked: Mapping[str, tuple[int, tuple[int, ...]]] = MappingProxyType({}),
    leaves: Collection[str] = (),
    walls: Collection[str] = (),
    mirrors: Collection[str] = (),
) -> list[Resolved]:
    """Each place as entities of the fragmented model, refused where its label holds none.

    A label holds nothing where it is no group of the mesh. A drawn label the
    fragmenting or the priorities left holding nothing is refused before this is
    asked, so that is a remainder where the drawn labels claim every entity where
    the model ends. Gmsh would take a field over no entity and size nothing
    without a word. A label closed on itself holds entities and has no rim,
    which is a drawing rather than a mistake: its rim is resolved to no entity,
    nothing is laid there, and the answer says so. So is a mark standing wholly
    outside the model, which marks nothing the mesh holds.

    A ``reentrant`` rim is every curve bounding the label's faces but a seam,
    kept where the room turns round it past half a turn as :mod:`.wedges`
    reads it against the faces ``walls`` and ``mirrors`` hold. The room is read
    once, for the first such rim.

    :param marked: per mark, the dimension it was drawn at and the entities it
        stands on in the model that is meshed.
    :param leaves: the labels that left the model, whose rim is every entity
        bounding what they hold - see :class:`~.vocabulary.AtRim`.
    :param walls: the labels :attr:`~.vocabulary.Demand.walls` names.
    :param mirrors: the labels :attr:`~.vocabulary.Demand.mirrors` names.
    """
    held = {name: (label.dimension, label.entities) for name, label in given.items()}
    held.update(marked)
    found = []
    said = []
    faces: tuple[set[int], set[int], wedges.Room] | None = None
    for place in places:
        standing = held.get(place.label)
        if standing is None:
            said.append(
                f"the place {place.name!r} names the label {place.label!r}, which holds "
                f"nothing once the drawing is cut up"
            )
            continue
        dimension, entities = standing
        if isinstance(place, Within) and dimension != top:
            said.append(
                f"the place {place.name!r} asks for a size throughout {place.label!r}, which "
                f"stands inside what leaves the model and holds no element of dimension {top}"
            )
            continue
        if isinstance(place, AtRim) and place.reentrant:
            if faces is None:
                faces = _faces(given, walls), _faces(given, mirrors), wedges.room()
            candidates = labels.bounding(dimension, entities)
            kept = [
                curve for curve in candidates if wedges.opening(curve, *faces) > HALF_TURN + TURNING
            ]
            found.append(
                Resolved(
                    place=place,
                    dimension=dimension - 1,
                    entities=tuple(kept),
                    left=len(candidates) - len(kept),
                )
            )
        elif isinstance(place, AtRim):
            rim = (labels.bounding if place.label in leaves else labels.rim)(dimension, entities)
            found.append(Resolved(place=place, dimension=dimension - 1, entities=tuple(rim)))
        elif isinstance(place, Near) and dimension == top:
            found.append(
                Resolved(
                    place=place,
                    dimension=dimension,
                    entities=entities,
                    filled=True,
                    rim=tuple(labels.rim(dimension, entities)),
                )
            )
        else:
            found.append(
                Resolved(
                    place=place,
                    dimension=dimension,
                    entities=entities,
                    filled=isinstance(place, Within),
                )
            )
    if said:
        raise Refused(said)
    return found


def _faces(given: Mapping[str, Label], names: Collection[str]) -> set[int]:
    """The faces the labels of these names hold."""
    return {
        tag
        for name in names
        if name in given and given[name].dimension == 2
        for tag in given[name].entities
    }


def install(demand: Demand, resolved: Sequence[Resolved]) -> None:
    """Lay one field per place and set their minimum as the background size.

    The other sources are turned off where a place lays a field, and where the
    demand sizes a surface of the drawing that bends. Carried from the boundary
    into a volume, the size a curve asks for fills the volume at the size of its
    sharpest point, and at the apex of a cone that is the floor.
    """
    laying = [one for one in resolved if _lays(one)]
    if not laying:
        if demand.per_turn and _bends():
            for option in OTHER_SOURCES:
                gmsh.option.setNumber(option, 0)
        return
    field = gmsh.model.mesh.field
    made = []
    for one in laying:
        place = one.place
        if one.filled:
            made.append(_throughout(one.dimension, one.entities, place.size, demand))
            if one.rim:
                made.extend(_away(one.dimension - 1, one.rim, place.size, demand))
        else:
            made.extend(_away(one.dimension, one.entities, place.size, demand))
    least = field.add("Min")
    field.setNumbers(least, "FieldsList", made)
    field.setAsBackgroundMesh(least)
    for option in OTHER_SOURCES:
        gmsh.option.setNumber(option, 0)


def _bends() -> bool:
    """Whether any surface of the model is other than a plane."""
    return any(gmsh.model.getType(2, tag) != "Plane" for _, tag in gmsh.model.getEntities(2))


def _throughout(dimension: int, entities: Sequence[int], size: float, demand: Demand) -> int:
    """A ``Constant`` field of ``size`` on these entities and their boundary."""
    field = gmsh.model.mesh.field
    constant = field.add("Constant")
    field.setNumbers(constant, CONSTANT_LISTS[dimension], list(entities))
    field.setNumber(constant, "VIn", size)
    field.setNumber(constant, "VOut", demand.coarsest)
    field.setNumber(constant, "IncludeBoundary", 1)
    return constant


def _away(dimension: int, entities: Sequence[int], size: float, demand: Demand) -> list[int]:
    """``size`` at these entities, growing to the coarsest: one ``Threshold`` over a
    ``Distance`` per class of them, as the module gives."""
    field = gmsh.model.mesh.field
    made = []
    for sampled, members in _classes(dimension, entities, size):
        distance = field.add("Distance")
        field.setNumbers(distance, DISTANCE_LISTS[dimension], members)
        if dimension:
            field.setNumber(distance, "Sampling", sampled)
        threshold = field.add("Threshold")
        field.setNumber(threshold, "InField", distance)
        field.setNumber(threshold, "SizeMin", size)
        field.setNumber(threshold, "SizeMax", demand.coarsest)
        field.setNumber(threshold, "DistMin", 0.0)
        field.setNumber(threshold, "DistMax", transition(demand.coarsest, size, demand.growth))
        made.append(threshold)
    return made


def _lays(one: Resolved) -> bool:
    """Whether a place lays a field: where it stands on an entity."""
    return bool(one.entities)


def _classes(dimension: int, entities: Sequence[int], size: float) -> list[tuple[int, list[int]]]:
    """The entities of one place split into classes, each with its sampling.

    Surfaces only, longest first: a class takes every surface still left whose
    length is more than the longest one's over :data:`CLASS`, so no class spans
    more than that. Curves are one class, sampled for the longest of them: a
    curve carries the sampling once where a surface carries about its square,
    so a place of many short curves beside one long one costs its count of
    curves times the long one's need, and splitting it would save that linear
    term and change the field every conductor's rim is laid with. Points are
    one class, since Gmsh lays no sampling on them.
    """
    if not dimension:
        return [(0, list(entities))]
    if dimension == 1:
        return [(_sampling(max(_length(1, tag) for tag in entities), size), list(entities))]
    lengths = sorted(((_length(dimension, tag), tag) for tag in entities), reverse=True)
    found = []
    while lengths:
        longest = lengths[0][0]
        members = [tag for length, tag in lengths if length * CLASS > longest]
        lengths = [(length, tag) for length, tag in lengths if length * CLASS <= longest]
        found.append((_sampling(longest, size), sorted(members)))
    return found


def _length(dimension: int, tag: int) -> float:
    """The length an entity's sampling is taken off: along a curve
    :func:`_stretched`, and across a surface twice the diagonal of its bounding
    box, for the reasons the module gives."""
    if dimension == 1:
        return _stretched(tag)
    low = gmsh.model.getBoundingBox(2, tag)
    return 2.0 * math.dist(low[:3], low[3:])


def _sampling(longest: float, size: float) -> int:
    """How many points a ``Distance`` field lays along each entity whose length,
    as :func:`_length` takes it, is at most ``longest``.

    Along a curve Gmsh lays the points at even steps of the parameter, counts
    both ends in the number and lays a point at neither, so each end is one step
    from the nearest point, and a number under :data:`LEAST_SAMPLING` lays no
    point at all (Gmsh 4.15.2, ``src/mesh/Field.cpp:2602``). A step covers no
    more length than the size wherever the parameter runs at most as fast as
    along its fastest stretch. A curve no longer than the size would otherwise
    be asked for two.
    """
    return max(math.ceil(longest / size) + 1, LEAST_SAMPLING)


def _stretched(tag: int) -> float:
    """The length a curve would have if its parameter ran everywhere as fast as
    where it runs fastest, in millimetres.

    The curve is asked at :data:`PROBES` points along its parameter how much
    length a unit of parameter covers there, which is the length of the
    derivative, and the greatest of those is taken over the parameter's span.
    Where the parameter runs at one speed along the curve, as along a line or a
    circle, that is the curve's length.
    """
    (low,), (high,) = gmsh.model.getParametrizationBounds(1, tag)
    speeds = np.asarray(
        gmsh.model.getDerivative(1, tag, list(np.linspace(low, high, PROBES))), dtype=float
    ).reshape(-1, 3)
    return float(np.linalg.norm(speeds, axis=1).max()) * (high - low)


class _Nodes:
    """Where every node is, looked up by tag."""

    def __init__(self) -> None:
        tags, flat, _ = gmsh.model.mesh.getNodes()
        tags = np.asarray(tags, dtype=np.int64)
        self.coordinates = np.asarray(flat, dtype=float).reshape(-1, 3)
        self.index = np.zeros(int(tags.max()) + 1 if len(tags) else 1, dtype=np.int64)
        self.index[tags] = np.arange(len(tags))

    def at(self, tags: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        return self.coordinates[self.index[tags]]


def _lengths(nodes: _Nodes, pairs: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """The length of each edge given as a pair of node tags."""
    ends = nodes.at(pairs[..., 1]) - nodes.at(pairs[..., 0])
    squared = (
        ends[..., 0] * ends[..., 0] + ends[..., 1] * ends[..., 1] + ends[..., 2] * ends[..., 2]
    )
    return np.sqrt(squared)


@dataclass(frozen=True)
class _Elements:
    """Some elements of one dimension: their corners and the lengths of their edges."""

    corners: list[npt.NDArray[np.int64]]
    longest: npt.NDArray[np.float64]
    mean: npt.NDArray[np.float64]
    shortest: npt.NDArray[np.float64]
    tags: npt.NDArray[np.int64]


def _elements(nodes: _Nodes, dimension: int, entities: Sequence[int]) -> _Elements:
    """The elements of these entities, an element type at a time.

    The edges are Gmsh's own list for each type, between corners only: a curved
    element carries nodes on the drawn surface, and the distance from a corner
    to one of those is a part of an edge rather than an edge.
    """
    corners: list[npt.NDArray[np.int64]] = []
    longest, mean, shortest, numbered = [], [], [], []
    for tag in entities:
        for kind in gmsh.model.mesh.getElementTypes(dimension, tag):
            properties = gmsh.model.mesh.getElementProperties(kind)
            per, primary = int(properties[3]), int(properties[5])
            elements, tags = gmsh.model.mesh.getElementsByType(kind, tag)
            listed = np.asarray(tags, dtype=np.int64).reshape(-1, per)
            if not len(listed):
                continue
            pairs = np.asarray(
                gmsh.model.mesh.getElementEdgeNodes(kind, tag, primary=True), dtype=np.int64
            ).reshape(len(listed), -1, 2)
            lengths = _lengths(nodes, pairs)
            corners.append(listed[:, :primary])
            numbered.append(np.asarray(elements, dtype=np.int64))
            longest.append(lengths.max(axis=1))
            mean.append(lengths.mean(axis=1))
            shortest.append(lengths.min(axis=1))
    empty = np.zeros(0)
    return _Elements(
        corners=corners,
        longest=np.concatenate(longest) if longest else empty,
        mean=np.concatenate(mean) if mean else empty,
        shortest=np.concatenate(shortest) if shortest else empty,
        tags=np.concatenate(numbered) if numbered else np.zeros(0, dtype=np.int64),
    )


def edges(given: Mapping[str, Label]) -> dict[str, tuple[Edges | None, int]]:
    """The shortest and the longest edge among each label's own elements, and
    how many of those elements there are.

    The edges are ``None`` for a label of points, and for one whose entities
    carry no element with an edge.
    """
    nodes = _Nodes()
    found: dict[str, tuple[Edges | None, int]] = {}
    for name, label in given.items():
        if label.dimension < 1:
            found[name] = (None, 0)
            continue
        held = _elements(nodes, label.dimension, label.entities)
        found[name] = (
            Edges(shortest=float(held.shortest.min()), longest=float(held.longest.max()))
            if len(held.longest)
            else None,
            len(held.longest),
        )
    return found


def reached(
    resolved: Sequence[Resolved], top: int, free: Collection[tuple[int, int]] = ()
) -> dict[str, Reached]:
    """What the mesh holds at each place, under the place's name.

    :param free: the points and curves the mesh of dimension ``top`` was not
        made to conform to, which no element has a corner on - see
        :func:`~.labels.release`. The elements at such an entity are those
        holding a node of it.
    """
    if not resolved:
        return {}
    nodes = _Nodes()
    filled = _elements(nodes, top, [tag for _, tag in gmsh.model.getEntities(top)])
    found = {}
    for one in resolved:
        place = one.place
        if not _lays(one):
            found[place.name] = Reached(
                asked=place.size,
                reached=None,
                standing=None,
                elements=0,
                dimension=one.dimension,
                laid=False,
                left=one.left,
            )
        elif one.filled:
            held = _elements(nodes, top, one.entities)
            found[place.name] = replace(
                _figures(place.size, _median(held.mean), held.longest, top),
                extent=_extent(top, one.entities),
            )
        else:
            bound = [tag for tag in one.entities if (one.dimension, tag) not in free]
            loose = [tag for tag in one.entities if (one.dimension, tag) in free]
            on = np.zeros(len(nodes.index), dtype=bool)
            for tag in bound:
                tags, _, _ = gmsh.model.mesh.getNodes(one.dimension, tag, includeBoundary=True)
                on[np.asarray(tags, dtype=np.int64)] = True
            holding = np.isin(filled.tags, _holding(one.dimension, loose, top))
            touching = _touching(on, filled) | holding
            median = None
            if one.dimension:
                lying = _elements(nodes, one.dimension, bound)
                along = float(lying.longest.max()) if len(lying.longest) else None
                median = _median(lying.longest)
            else:
                along = _leaving(nodes, on, filled)
            if holding.any():
                most = float(filled.longest[holding].max())
                along = most if along is None else max(along, most)
            found[place.name] = replace(
                _figures(place.size, along, filled.longest[touching], one.dimension),
                along=median,
                extent=_extent(one.dimension, one.entities) if one.dimension else None,
                left=one.left,
            )
    return found


def _holding(dimension: int, entities: Sequence[int], top: int) -> list[int]:
    """The elements of dimension ``top`` holding a node of these entities, which
    none of them has a corner on. A node on a face or an edge of an element is
    held by every element meeting there."""
    found: set[int] = set()
    for tag in entities:
        _, flat, _ = gmsh.model.mesh.getNodes(dimension, tag, includeBoundary=True)
        for point in np.asarray(flat, dtype=float).reshape(-1, 3):
            found.update(
                int(held) for held in gmsh.model.mesh.getElementsByCoordinates(*point, top)
            )
    return sorted(found)


def _extent(dimension: int, entities: Sequence[int]) -> float:
    """The length, area or volume of these entities, the geometry's."""
    return float(sum(abs(gmsh.model.occ.getMass(dimension, tag)) for tag in entities))


def _touching(on: npt.NDArray[np.bool_], elements: _Elements) -> npt.NDArray[np.bool_]:
    """Which elements have a corner on a node marked ``on``."""
    if not elements.corners:
        return np.zeros(0, dtype=bool)
    return np.concatenate([on[listed].any(axis=1) for listed in elements.corners])


def _leaving(nodes: _Nodes, on: npt.NDArray[np.bool_], elements: _Elements) -> float | None:
    """The longest edge from a node marked ``on`` to another corner of an element.

    Every two corners of a triangle or a tetrahedron are joined by an edge.
    ``None`` where no element has a corner on a marked node.
    """
    longest = None
    for listed in elements.corners:
        for corner in range(listed.shape[1]):
            rows = listed[on[listed[:, corner]]]
            if not len(rows):
                continue
            pairs = np.stack([np.broadcast_to(rows[:, [corner]], rows.shape), rows], axis=-1)
            found = float(_lengths(nodes, pairs).max())
            longest = found if longest is None else max(longest, found)
    return longest


def _median(values: npt.NDArray[np.float64]) -> float | None:
    return float(np.median(values)) if len(values) else None


def _figures(
    asked: float, along: float | None, longest: npt.NDArray[np.float64], dimension: int
) -> Reached:
    return Reached(
        asked=asked,
        reached=along,
        standing=_median(longest),
        elements=int(len(longest)),
        dimension=dimension,
    )
