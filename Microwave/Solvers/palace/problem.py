# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What this adapter runs, once the document has been read.

Each kind below is a label the mesher will carry through the fragmenting. A
region is a volume the field lives in and is what a material fills. A feed is a
face a wave leaves through, which stands on the outside of the model - because
it is a face of a region, or because the mesher leaves out what stands behind
it. A lumped feed is a resistor across a gap, as one face per element, and a
face a lumped element lies in carries the rest of that face as a label of its
own. A conductor is a sheet of metal the user drew, which carries a perfect
condition wherever it stands - on the boundary of the model or inside it, where
the region runs round it on both sides. A lossy conductor is the same sheet
carrying the surface impedance of a metal of finite conductivity instead.

There is no other kind and no flag that turns one into another, so a region with
a port's properties or a wall with a permittivity cannot be built.

A study whose drawing leaves room in the box round it adds labels nobody drew:
the room, a region of the study's medium, and in an open study the sides of it
left open, which carry an absorbing condition. See :class:`Reserved`.

The wall is not one of them. Metal is modelled here as a condition on a face,
and where none was drawn the wall is a perfect conductor and a name handed to the
mesher for everything at the boundary the rest of the labels left. So the faces
it holds
are read off the fragmented model rather than off the drawing: a face between
two bodies that touch bounds two volumes and is not in it, where a caller
listing a solid's own faces has no way to tell that face from the outside of the
model.

What the wall rests on is therefore the fragmenting, and it is no stronger than
that. Where the kernel leaves a body standing inside another instead of cutting
it out, every face of the inner body bounds one volume and reads as the boundary
- and a perfect conductor lands on a shell in the middle of the region the field
is in. Such a body shares no face with the region around it, so it is a part of
its own, and the profile's demand for one region refuses it by name.

A refinement region is not a label. What it names is handed over as a mark,
which says where a size is laid and claims nothing: see :class:`RegionMark`.

A shape here is whatever the document handed over. Nothing in this module looks
at one: they are carried so that the stage owning a directory can write them
out, one file per label, which is what makes the label map exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...Gmsh.vocabulary import Demand, Mark, Piece, Profile
from ...portbox import AXIS_NAMES
from .config import Filling, Sweep

__all__ = [
    "OPEN",
    "SPACE",
    "Bare",
    "Beside",
    "Coarsening",
    "Conductor",
    "CountAcross",
    "Creases",
    "Element",
    "Floored",
    "Feed",
    "Joined",
    "LossyConductor",
    "LumpedFeed",
    "Reserved",
    "Sealed",
    "Plane",
    "Problem",
    "Refinement",
    "Region",
    "RegionCount",
    "RegionMark",
    "Relaxed",
    "Unlaid",
    "Unrefining",
    "Unfilled",
    "Unreserved",
    "Unwalled",
]

#: A volume, and the faces bounding it. The mesher takes a label's dimension
#: because a file arrives with its whole hierarchy, and a solid's own faces come
#: back beside it.
VOLUME = 3
SURFACE = 2

#: The priority the rest of a face a lumped element lies in is handed over at,
#: below every drawn label's. The label is drawn as the whole face, and the
#: fragmenting cuts the elements out of it, so every piece of the face that
#: nothing else was drawn over is its own and nothing of it is left for the wall.
PLANE_PRIORITY = -1

#: The priority the reserved air is handed over at, below every drawn region's.
#: It is drawn as the box round the structure less the room sealed off, and every bound body stands
#: inside it, so each piece a body covers goes to the body and the rest is air.
#: At the drawn regions' own priority the mesher refuses each of those pieces as
#: two regions claiming one.
BACKGROUND_PRIORITY = -1

#: The priority a label is handed over at where nothing below names one: every
#: drawn region's, among others.
DEFAULT_PRIORITY = 0

#: The priority a body of metal is handed over at, above every drawn region's,
#: so a piece a body and a region were both drawn over leaves the model with the
#: body.
METAL_PRIORITY = DEFAULT_PRIORITY + 1

#: What the reserved air is called in the mesh, and what its open sides are
#: called. Each is one label, because one filling and one condition cover it.
SPACE = "free space"
OPEN = "open boundary"


@dataclass(frozen=True)
class Region:
    """A volume the field lives in, and what fills it.

    :param material: the name of the material that fills it, which a result
        states the loss by. Empty where the caller names none, and the label
        stands for it.
    """

    label: str
    shapes: tuple[Any, ...]
    filling: Filling
    material: str = ""


@dataclass(frozen=True)
class Conductor:
    """Perfect conductor, as the sheets or the bodies it was drawn as.

    :param solid: whether it was drawn as bodies. A body's inside is not part of
        the region, so the mesher takes it out and the faces it leaves behind
        carry the condition.
    :param joined: the other bindings whose bodies meet these or each other,
        drawn under this label with their bodies among ``shapes``. Every body
        carries one condition, so the space two of them share is metal
        whichever fills it.
    """

    label: str
    shapes: tuple[Any, ...]
    solid: bool = False
    joined: tuple[str, ...] = ()


@dataclass(frozen=True)
class LossyConductor:
    """A sheet of metal of finite conductivity, as the faces it was drawn as.

    :param conductivity: in siemens per metre.
    :param permeability: the metal's own, relative.
    :param thickness: in millimetres, and zero where the metal is thick against
        its skin depth across the whole band, which is where the thickness
        changes nothing Palace computes.
    :param material: as a region's.
    """

    label: str
    shapes: tuple[Any, ...]
    conductivity: float
    permeability: float
    thickness: float
    material: str = ""


@dataclass(frozen=True)
class Feed:
    """A face a wave leaves through, and how it is driven.

    :param number: the port's number, which is the document's and indexes every
        table the run writes.
    :param behind: a point on the model's side of the face, in millimetres,
        which says which way the power through the face is measured.
    :param mode: which mode of the cross-section, ranked by decreasing wave
        number rather than named.
    :param offset: how far the reference plane stands in from the face, in
        millimetres. The result is de-embedded by that distance.
    :param excited: whether this port is driven.
    :param inward: the direction from the face into the model. The mesher
        leaves out what stands on the other side of it, and refuses a face
        whose region stands that way.
    """

    label: str
    shapes: tuple[Any, ...]
    number: int
    behind: tuple[float, float, float]
    mode: int = 1
    offset: float = 0.0
    excited: bool = False
    inward: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class Element:
    """One face of a lumped port, and the way it is driven across.

    :param direction: the axis the element is driven along, signed from the
        source to the reference, as Palace names one: ``+X`` to ``-Z``.
    :param facing: for an element lying in no face of the region, what stands
        nearest it on each side of its plane: the side, as ``-X`` to ``+Z``, what
        stands there, and how far away, in millimetres.
    """

    label: str
    shapes: tuple[Any, ...]
    direction: str
    facing: tuple[tuple[str, str, float], ...] = ()


@dataclass(frozen=True)
class LumpedFeed:
    """A resistor across a gap, as the elements it is laid on.

    The elements stand in parallel, so the resistance is the port's own and not
    an element's.

    :param number: the port's number, which is the document's and indexes every
        table the run writes.
    :param resistance: in ohms, and positive.
    :param excited: whether this port is driven.
    :param planes: the labels of the faces its elements lie in, which carry the
        rest of each of those faces. Empty where an element lies in no face of a
        region.
    """

    label: str
    number: int
    elements: tuple[Element, ...]
    resistance: float
    excited: bool = False
    planes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plane:
    """The faces of the regions one or more lumped elements lie in.

    Drawn as the whole of each face, at :data:`PLANE_PRIORITY`, so what it holds
    in the mesh is the rest of the face once the elements and any metal drawn
    over it took their pieces. Where the model ends there, it carries a magnetic
    wall: a perfect conductor beside an element shorts the element along its
    side.

    :param ports: the labels of the lumped ports whose elements lie in it.
    """

    label: str
    shapes: tuple[Any, ...]
    ports: tuple[str, ...]


@dataclass(frozen=True)
class RegionMark:
    """One thing a refinement region names, as the mesher is handed it.

    The shape itself rather than a box round it, handed over as a mark: it is
    cut up with the drawing and claims nothing of it, and the region's size is
    laid at what it became - along a face, an edge or a point and growing away
    from it, or throughout a body and growing away from its faces. A box round
    a curved shape would size the space the shape goes round as well.

    :param name: what the mesh reports the place under, and the mark's name.
    :param dimension: what the shape is, read off what it holds: three where it
        holds a solid, two a face, one an edge, and nought only points.
    :param shape: the shape, where the drawing shows it.
    :param thinnest: the smallest side of the box round the shape, in
        millimetres, which a count across the region is said against.
    """

    name: str
    dimension: int
    shape: Any
    thinnest: float


@dataclass(frozen=True)
class Refinement:
    """A mesh refinement region, as the marks its references are handed over as.

    :param label: the region's own label.
    :param coarsens: whether it asks what it names to settle for its size rather
        than to be refined to it.
    :param size: its element size, in millimetres.
    :param across: the count of elements across it the region asks for, and zero
        for none.
    :param marks: one per element a reference names, or per whole shape. Empty
        for a coarsening, which is laid, where it is, at the rims of what it
        names.
    :param part: for a coarsening, the metal it names some faces of, and not
        all, where no other coarsening names the rest.
    :param bodies: for a coarsening, the regions it names geometry of.
    :param joined: for a coarsening, the bindings drawn under each label in
        ``part`` besides its own, in the same order - see
        :attr:`Conductor.joined`.
    """

    label: str
    coarsens: bool
    size: float
    across: int
    marks: tuple[RegionMark, ...] = ()
    part: tuple[str, ...] = ()
    bodies: tuple[str, ...] = ()
    joined: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class CountAcross:
    """The mesh policy's count of elements across each region, which no size
    here expresses.

    :param subject: the mesh policy's label.
    :param count: the count asked for.
    :param labels: the regions it is asked of.
    """

    subject: str
    count: int
    labels: tuple[str, ...]


@dataclass(frozen=True)
class RegionCount:
    """A refinement region's count of elements across itself, which no size here
    expresses.

    :param subject: the region's label.
    :param count: the count asked for.
    :param places: each thing the region names, by the name the mesh reports it
        under, with the smallest side of the box round it in millimetres.
    """

    subject: str
    count: int
    places: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class Coarsening:
    """A refinement region asking what it names to settle for a coarser size,
    where some of what it names is not coarsened here.

    A coarsening is laid at the edges of metal whose label it names whole -
    see :class:`Relaxed`. Nothing else here has a size of its own to
    settle for.

    :param subject: the region's label.
    :param size: the size it settles for, in millimetres.
    :param part: the metal it names some faces of and not all.
    :param bodies: the regions it names geometry of.
    :param joined: the bindings drawn under each label in ``part`` besides its
        own, in the same order.
    """

    subject: str
    size: float
    part: tuple[str, ...] = ()
    bodies: tuple[str, ...] = ()
    joined: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class Relaxed:
    """The edges of metal whose bindings a region set to coarsen names whole,
    laid at the size it settles for where the room turns round them past half a
    turn.

    A region names a label of metal whole where it names every face each binding
    drawn under the label names, alone or with other regions, as the other
    backend relaxes a reference only where every element it names is
    coarsened. The other backend keeps a gap
    between the coarsened metal and another conductor at its own size, and this
    one does not, so the rim's nearest approach to another conductor is stated.

    :param rim: the label of metal, which is its rim's place.
    :param size: the size it settles for, in millimetres: the finest of the
        coarsenings naming it.
    :param asked_by: the label of the region that asked for that size.
    :param length: the length of the metal's edges as drawn, in millimetres:
        every edge of the label's faces but a seam, once.
    :param nearest: the label of the conductor the rim comes nearest, and how
        near in millimetres, or ``None`` where there is no other conductor.
    :param edge: the size ``EdgeRefinement`` lays at a rim otherwise, in
        millimetres, or ``None`` where it lays none.
    :param refined: each region set to refine that names some of the same
        metal at a size finer than ``size``, with that size. Where it lays its
        size the finer one is laid.
    """

    rim: str
    size: float
    asked_by: str
    length: float
    nearest: tuple[str, float] | None
    edge: float | None = None
    refined: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Unrefining:
    """A refinement region whose size is no finer than the size everywhere, so it
    lays nothing.

    :param subject: the region's label.
    :param size: its element size, in millimetres.
    :param recipe: the label of the object the size everywhere is asked on.
    :param coarsest: the size everywhere, in millimetres.
    :param count: the count of elements across it the region asks for, and zero
        for none.
    """

    subject: str
    size: float
    recipe: str
    coarsest: float
    count: int


@dataclass(frozen=True)
class Floored:
    """A refinement region whose size is below the policy's floor, laid at the
    floor.

    :param subject: the region's label.
    :param size: its element size, in millimetres.
    :param policy: the mesh policy's label.
    :param floor: the policy's floor, in millimetres.
    :param places: the name of each thing the region names, as the mesh reports
        it.
    """

    subject: str
    size: float
    policy: str
    floor: float
    places: tuple[str, ...]


@dataclass(frozen=True)
class Creases:
    """Where the Gmsh mesh's refinement at a conductor's edges is laid, and
    where it is not.

    :param subject: the Gmsh mesh's label.
    :param refinement: how many times finer than the bulk the edges are asked
        to be.
    :param rims: the labels of the metal whose edges carry it.
    :param wall: the rest of the boundary, whose creases carry none of it: the
        sides of a box, which the room stands inside.
    :param size: the size the refinement asks for at a rim, in millimetres.
    :param floor: the policy's floor, which a rim is laid at where ``size`` is
        below it.
    :param coarsened: the labels of the metal whose edges carry none of it because a
        coarsening settles them at or above the size everywhere. Each is said
        on a line of its own as well.
    """

    subject: str
    refinement: float
    rims: tuple[str, ...]
    wall: str
    size: float
    floor: float
    coarsened: tuple[str, ...] = ()


@dataclass(frozen=True)
class Unreserved:
    """A clearance stated on a study with no ``Air`` face, which reserves no
    free space for it to measure.

    :param subject: the mesh policy's label.
    :param clearance: the length stated, in millimetres.
    """

    subject: str
    clearance: float


@dataclass(frozen=True)
class Unfilled:
    """A medium linked on a study whose drawing leaves it no room: every room in
    the domain is drawn and bound, or sealed off by metal.

    :param subject: the mesh policy's label.
    :param medium: the medium's name.
    """

    subject: str
    medium: str


#: A size the document states that this backend does not lay as stated, said
#: once the mesh can answer it with a figure.
Unlaid = (
    CountAcross
    | RegionCount
    | Coarsening
    | Relaxed
    | Unrefining
    | Floored
    | Creases
    | Unreserved
    | Unfilled
)


@dataclass(frozen=True)
class Beside:
    """The part of a side of the reserved box that no bound body stands on.

    The box is flush with the structure on every side the policy does not open,
    so what the drawing does not cover there is reserved air. That air carries
    the condition the side carries: a perfect wall on an ``Ends`` side, and a
    magnetic wall on a ``Through`` side, where it is the rest of a lumped port's
    plane.

    :param side: the side, as ``{axis}{side}``.
    :param area: the air's area, in square millimetres.
    :param whole: the whole side's area, in square millimetres.
    """

    side: str
    area: float
    whole: float


@dataclass(frozen=True)
class Sealed:
    """Room in the reserved box that the drawing seals off from the model.

    Metal and the back of a waveguide port's face stand on its boundary, and
    the sides of the box where it reaches them. No lumped element and no other
    material stands in it or on it, no port faces into it, and it reaches no
    open side, so no field reaches it, and the box leaves it out. Where it was,
    the model ends on the metal and the box's sides. Room between a housing and
    the box, the inside of a shell of metal and a guide's stretch behind its
    port are such room.

    :param least: its least corner, in millimetres.
    :param most: its greatest corner.
    :param volume: in cubic millimetres.
    :param bounded_by: the label of each metal and each port standing on it,
        and of each body standing on it where it ``stands_off``.
    :param sides: each side of the box it reaches, as ``{axis}{side}``.
    :param stands_off: whether it is room between the box built on the bound
        and the drawing's own extent, on a side the domain ends on the
        structure. The drawing leaves no room there; the bound does.
    """

    least: tuple[float, float, float]
    most: tuple[float, float, float]
    volume: float
    bounded_by: tuple[str, ...]
    sides: tuple[str, ...] = ()
    stands_off: bool = False


@dataclass(frozen=True)
class Joined:
    """Room thinner than a slip that a body of the medium's own material takes.

    The room is the medium, so the body grown over it is the same device, and
    the mesh lays no elements as thin as the gap. See
    :func:`~Microwave.drawn.joined`.

    :param least: its least corner, in millimetres.
    :param most: its greatest corner.
    :param thickness: twice its volume over its surface, in millimetres.
    :param into: the label of the body that takes it.
    :param across: the label of each other body standing on it.
    """

    least: tuple[float, float, float]
    most: tuple[float, float, float]
    thickness: float
    into: str
    across: tuple[str, ...] = ()


@dataclass(frozen=True)
class Reserved:
    """The room reserved round the structure, and the sides of it that are open.

    The room is a box round everything bound to a material, grown by the
    clearance on each face the mesh policy says ``Air`` on and flush with the
    structure on every other. A study is open where it has such a face. Every
    part of the box no bound body stands in is the study's medium, handed over
    as the one region :data:`SPACE` at :data:`BACKGROUND_PRIORITY`, except the
    room metal seals off from the model. Each ``Air`` side of the box is a
    rectangle drawn from the corners of the part of the box the model lies in,
    which stops where the drawing reaches on each side the domain ends on, and
    the rectangles together are
    :data:`OPEN`, which carries the absorbing condition. No face of the box is
    picked out of the kernel's own listing of it.

    A closed study reserves the box only where its drawing leaves room in it
    that metal does not seal off. A drawing that fills its box is the model as it
    stands.

    :param shapes: one rectangle per open side.
    :param faces: the open sides, as ``{axis}{side}``, and none in a closed
        study.
    :param clearance: how far each open side stands from the structure, in
        millimetres, and nothing in a closed study.
    :param lower: the least corner of the box, in millimetres.
    :param upper: the greatest corner.
    :param thinnest: the drawn region standing in the reserved air whose body is
        thinnest, and the smallest extent of that body, in millimetres, or
        ``None`` where no region stands in it. The field of a line falls off over
        a distance its cross-section sets, and the run states the clearance
        against it - so the region taken is one the air reaches, and not a puck
        buried inside another body.
    :param walled: each side the policy ends the domain on where the drawing
        does not cover the whole of it, so the wall stands in the reserved air.
    :param magnetic: each ``Through`` side a lumped port's plane lies in where
        the reserved air stands beside the faces drawn there. The plane is the
        whole side, so that air is a magnetic wall as the plane is, and a wave
        radiated onto it is reflected.
    :param overstated: the bound shape whose body the kernel's bounding box
        stands furthest off, and how far, in millimetres, or ``None`` where no
        shape's box stands off its body. The box is built on the kernel's bound,
        which is exact for a planar extreme and stands off a curved one, so a
        clearance is measured from that bound rather than from the body.
    :param sealed: each room metal seals off from the model, which the box
        leaves out.
    :param filling: the study's medium, which fills the box.
    :param medium: the medium's name, and empty for vacuum.
    """

    shapes: tuple[Any, ...]
    faces: tuple[str, ...]
    clearance: float
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    thinnest: tuple[str, float] | None
    walled: tuple[Beside, ...] = ()
    magnetic: tuple[Beside, ...] = ()
    overstated: tuple[str, float] | None = None
    sealed: tuple[Sealed, ...] = ()
    label: str = OPEN
    filling: Filling = Filling()
    medium: str = ""

    @property
    def structure(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """The box round the structure, in millimetres: the reserved box less the
        clearance on each open side."""
        lower, upper = list(self.lower), list(self.upper)
        for face in self.faces:
            dim = AXIS_NAMES.index(face[0])
            if face.endswith("Min"):
                lower[dim] += self.clearance
            else:
                upper[dim] -= self.clearance
        return (lower[0], lower[1], lower[2]), (upper[0], upper[1], upper[2])


@dataclass(frozen=True)
class Bare:
    """One face of a region standing in the reserved air.

    :param where: where the face stands, as the axis it is flat across and its
        coordinate, or its least and greatest corner where it is flat across
        none.
    :param area: the part of it no metal covers and no other body stands on, in
        square millimetres.
    :param side: the side of the study the face lies toward, as
        ``{axis}{side}``, where it stands at the structure's own extreme along
        that axis, and empty otherwise. Setting that side to ``Ends`` keeps the
        face walled.
    """

    where: str
    area: float
    side: str = ""


@dataclass(frozen=True)
class Unwalled:
    """A region's faces that stand in the reserved air with no metal drawn on
    them.

    Each such face is a dielectric-air interface. A region drawn as the inside of
    a shield, a cavity or a guide has lost its walls that way, and a board's
    substrate meets the air that way on purpose. Both are stated and neither is
    refused, because the drawing does not say which was meant.

    The decision is per face. A face metal covers whole carries the condition the
    metal carries, a face lying in a side the policy does not open is still where
    the model ends, and a face another body stands on meets no air - and each of
    those can be true of one face of a region while the next face stands bare.

    :param region: the region's label.
    :param faces: each face of it standing in the air, with its area.
    """

    region: str
    faces: tuple[Bare, ...]

    @property
    def area(self) -> float:
        """The whole area the region stands in the air over, in square
        millimetres."""
        return sum(face.area for face in self.faces)


@dataclass(frozen=True)
class Problem:
    """One run, as this adapter describes it before anything is written."""

    regions: tuple[Region, ...]
    wall: str
    feeds: tuple[Feed, ...]
    sweep: Sweep
    demand: Demand
    profile: Profile
    order: int
    conductors: tuple[Conductor, ...] = ()
    lossy: tuple[LossyConductor, ...] = ()
    lumped: tuple[LumpedFeed, ...] = ()
    planes: tuple[Plane, ...] = ()
    unlaid: tuple[Unlaid, ...] = ()
    reserved: Reserved | None = None
    unwalled: tuple[Unwalled, ...] = ()
    marks: tuple[RegionMark, ...] = ()
    joined: tuple[Joined, ...] = ()
    #: The smallest response the study reads, as a magnitude in S: one is full
    #: scale. What an adaptive sweep's model is held to is a share of it.
    smallest: float = 1.0

    @property
    def opened(self) -> Reserved | None:
        """The reserved air where some side of it is open to free space, and
        ``None`` in a closed study."""
        return self.reserved if self.reserved is not None and self.reserved.faces else None

    @property
    def ports(self) -> tuple[Feed | LumpedFeed, ...]:
        """Every port, of either kind, in the order of their numbers."""
        return tuple(sorted((*self.feeds, *self.lumped), key=lambda port: port.number))

    @property
    def labelled(self) -> tuple[tuple[str, int, tuple[Any, ...]], ...]:
        """Every drawn label, its dimension and its shapes.

        The stage that owns a directory turns each shape into a file and each
        row into a ``Piece``. Reported in one place so that what a label means
        and what dimension it is declared at are stated together.

        The wall is not here. It is drawn as nothing, so it has no file and no
        dimension of its own to declare.

        A body of metal is declared at the region's dimension, and the mesher
        takes its inside out and leaves the label holding its faces.

        No priority is stated here, and one label has one: a plane at
        :data:`PLANE_PRIORITY`, the reserved air at :data:`BACKGROUND_PRIORITY`,
        a body of metal at :data:`METAL_PRIORITY`, and every other label, a
        drawn region among them, at :data:`DEFAULT_PRIORITY`. A priority
        separates two labels drawn over one piece.
        The wall is drawn as nothing, so it takes only what no label claimed and
        contests nothing, and a region is a volume where a port and a sheet of
        metal are faces, so a region meets neither over a piece. A body of metal
        stands above every region, so each piece both were drawn over leaves the
        model with the body, and a face it leaves behind that a port or a sheet
        was drawn over is theirs. A plane is drawn over every piece of its face
        and gives each up to whatever else was drawn there, and the reserved air
        does the same with every piece of its box. Two drawn regions are
        refused by the translation where they share a volume, and two bindings
        whose bodies of metal meet are drawn under one label. What can meet
        at one priority is two regions over a piece thinner than that
        measures, or two faces - a port's or an element's and
        a sheet of metal, or two sheets - drawn over one piece,
        and each of those is refused by the mesher naming both labels and where
        the piece is. The open sides stand clear of every bound shape by the
        clearance, so no other face is drawn over them.
        """
        rows: list[tuple[str, int, tuple[Any, ...]]] = [
            (region.label, VOLUME, region.shapes) for region in self.regions
        ]
        rows += [(feed.label, SURFACE, feed.shapes) for feed in self.feeds]
        rows += [
            (element.label, SURFACE, element.shapes)
            for feed in self.lumped
            for element in feed.elements
        ]
        rows += [(plane.label, SURFACE, plane.shapes) for plane in self.planes]
        rows += [
            (conductor.label, VOLUME if conductor.solid else SURFACE, conductor.shapes)
            for conductor in self.conductors
        ]
        rows += [(sheet.label, SURFACE, sheet.shapes) for sheet in self.lossy]
        if self.opened is not None:
            rows.append((self.opened.label, SURFACE, self.opened.shapes))
        return tuple(rows)

    @property
    def dividing(self) -> frozenset[str]:
        """Every label whose faces no field crosses, so the mesher reports what
        each of them leaves standing on its own.

        Palace duplicates the boundary elements of every attribute a condition
        names but a lumped port's, so the two sides of such a face are two
        problems: a perfect conductor constrains the field on each side, and a
        surface impedance is answered on each face from its own side. A lumped
        element is what drives a field across its face, so it joins rather than
        divides.

        A body of metal is here with the sheets. Its faces are where the model
        ends once it has gone, so they join nothing whether it is named or not,
        and naming it lets a part it closes off be told by its name.

        Three labels of this backend's carry a condition only where the model
        ends, and there a face bounds one region and joins nothing to begin
        with: the wall, which is built out of the frontier; the open sides of
        the reserved air; and the rest of a face a lumped element lies in,
        which is a magnetic wall where the model ends and nothing at all inside
        it. None of them is here, and naming one would state a cut across faces
        the field crosses freely.
        """
        return frozenset(
            (
                *(conductor.label for conductor in self.conductors),
                *(sheet.label for sheet in self.lossy),
            )
        )

    def pieces(self, files: dict[str, tuple[str, ...]]) -> tuple[Piece, ...]:
        """The mesher's request, given where each label's shapes were written.

        ``files`` holds one path per shape, under the label it was drawn for, in
        the order :attr:`labelled` reported them.
        """
        inward = {feed.label: feed.inward for feed in self.feeds}
        priority = {plane.label: PLANE_PRIORITY for plane in self.planes}
        priority[SPACE] = BACKGROUND_PRIORITY
        bodies = {conductor.label for conductor in self.conductors if conductor.solid}
        priority.update({label: METAL_PRIORITY for label in bodies})
        cuts = self.dividing
        made: list[Piece] = []
        for label, dimension, shapes in self.labelled:
            written = files[label]
            if len(written) != len(shapes):
                raise ValueError(
                    f"the label {label} was drawn over {len(shapes)} shapes and "
                    f"{len(written)} files were written for it"
                )
            made.extend(
                Piece(
                    label=label,
                    dimension=dimension,
                    file=path,
                    priority=priority.get(label, DEFAULT_PRIORITY),
                    divides=label in cuts,
                    leaves=label in bodies,
                    inward=inward.get(label),
                )
                for path in written
            )
        return tuple(made)

    def marked(self, files: dict[str, str]) -> tuple[Mark, ...]:
        """The mesher's marks, given the one file each mark's shape was written to."""
        return tuple(
            Mark(name=mark.name, dimension=mark.dimension, file=files[mark.name])
            for mark in self.marks
        )
