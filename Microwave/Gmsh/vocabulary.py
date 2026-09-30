# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a caller hands the mesher, and what comes back.

Millimetres throughout, because that is what a drawing is in. Nothing here
scales anything and no length here carries a unit of its own: a solver working
in metres is told the scale factor by its own adapter, and the mesh file stays
in the units the shapes were drawn in.

There is no word here for a port, a material or a boundary, and there is not
meant to be. A label is a string the caller chose, and what it means is the
caller's business.

This module imports Gmsh nowhere, so a request can be built and an answer read
on a machine that has no mesher.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "BOTH",
    "HALF_TURN",
    "TURNING",
    "FRONTIER",
    "DIMENSIONS",
    "FORMATS",
    "INTERIOR",
    "AtRim",
    "Format",
    "Demand",
    "Edges",
    "Label",
    "LeftOut",
    "Mark",
    "Mesh",
    "Near",
    "Part",
    "Piece",
    "Place",
    "Profile",
    "Reached",
    "Refused",
    "Settled",
    "Trimmed",
    "Uncut",
    "Unmeshed",
    "Within",
    "transition",
]

#: Where a labelled entity of the dimension below the filled one lies, read off
#: how many of its sides the filled dimension stands on. One is where the model
#: ends, which is not the same as the outside of it - the wall of a cavity
#: bounds one volume and is enclosed by the body. More than one is inside the model, and a
#: label can hold some of each.
#:
#: The words are the topology's rather than the caller's: a piece on the
#: frontier is one where the meshed region ends, and what that is for is the
#: caller's to know - a face dividing a volume is exactly what a port is, and it
#: is also a perfect conductor cutting a cavity in half.
#:
#: A piece the fragmenting left inside a volume without dividing it is inside
#: the model too: the volume stands on both sides of it and is meshed to it.
#:
#: An entity bounding none of the filled dimension and standing inside none of
#: it is neither, and is refused rather than reported: nothing in the mesh
#: would lie on it.
FRONTIER = "frontier"
INTERIOR = "interior"
BOTH = "both"

#: The dimensions Gmsh works in. A profile asking to fill anything else is
#: asking for a mesh of nothing.
DIMENSIONS = (1, 2, 3)


@dataclass(frozen=True)
class Format:
    """One format this package writes, and what it holds of what was asked for.

    A writer that cannot hold something does not say so. It writes what it can
    and reports no error, so each field below records one way a file can
    disagree with the request and still look like an answer to it.

    :param suffix: what the file is called. Gmsh picks its writer by the path's
        suffix, so the mesher builds the path from this rather than taking one.
    :param revision: which revision to ask for, where the format has more than
        one, and ``None`` where it has one.
    :param forbids: characters this writer will not carry in a group name. It
        cuts the name at one or rewrites it, and two labels then arrive as one
        under a name neither of them was given.
    :param holds_order: the highest element order this writer keeps, and
        ``None`` where it keeps whatever Gmsh builds. Asked for more, it writes
        the elements it has a record for and says nothing - so a request for a
        curved mesh comes back as a straight-sided file, and every figure taken
        off the elements in the session is about elements the file does not
        hold.
    :param name_length: how many bytes of a group name this writer keeps.
        Bytes and not characters, which is what a name outside ASCII turns on:
        past this the Gmsh formats truncate and UNV comes back with the group
        holding no name at all, so two labels alike that far arrive as one, or
        as none.
    """

    suffix: str
    revision: float | None
    forbids: str
    holds_order: int | None
    name_length: int


#: The formats this package writes, by the name a profile states. What each one
#: forbids, holds and keeps is measured by writing a mesh through it and reading
#: the file back.
FORMATS: dict[str, Format] = {
    "msh22": Format(suffix="msh", revision=2.2, forbids='"\n\r', holds_order=None, name_length=128),
    "msh41": Format(suffix="msh", revision=4.1, forbids='"\n\r', holds_order=None, name_length=128),
    "unv": Format(suffix="unv", revision=None, forbids='"\n\r\t ', holds_order=2, name_length=254),
}


@dataclass(frozen=True)
class Piece:
    """One shape the caller drew, in a file of its own, under the label it gave it.

    One file rather than one model, because that is what makes the label map
    exact: the entities an import returns are that file's, so which of them
    belong to this label needs nothing matched.

    :param label: the caller's name for what this shape is. Several pieces may
        carry one label, and the mesher unions what they became.
    :param dimension: which of the shapes in the file this label declares. The
        import returns the whole hierarchy of what the file held, and a solid's
        own faces come back beside it, so the declaration is what separates a
        label's shapes from their boundaries. A file holding shapes of two
        dimensions that are both meant is two pieces under two labels: a loose
        face is indistinguishable from a solid's own by dimension alone.
    :param file: a BREP, STEP or IGES file, by path.
    :param priority: which label owns a region two of them cover. Where
        shapes overlap, the fragmenting cuts the overlap into pieces of its own
        and hands each of them to every label that was drawn over it. One piece
        can carry one label, so somebody has to say which, and the caller is the
        only one who knows: a part standing inside a domain is meant, and two
        dielectrics standing inside each other is a mistyped dimension, and the
        two look identical to anything reading the drawing. The higher number
        takes the piece. Equal is a refusal naming the labels that tie and the
        place, which is what every label gets until the caller states
        otherwise.

        Several pieces under one label state one priority between them, since
        it is a fact about the label rather than about the shape.
    :param divides: for a label at the dimension below the filled one, whether
        nothing of what is filled crosses its faces. The mesher takes no shape
        apart for it and refuses no drawing over the parts it leaves: it asks
        what is filled a second time with those faces as cuts, and reports what
        comes of it as :attr:`Mesh.parted`. A caller whose condition decouples the
        two sides of a face - a perfect conductor, a magnetic wall, an
        impedance a solver duplicates the elements of - states it, and reads
        off the answer which parts of its model stand on their own. Whether a
        part standing on its own is a fault is the caller's, since only the
        caller knows what reaches one. One per label, as a priority is.
    :param leaves: for a label at the filled dimension, whether what it fills
        leaves the model. The pieces it keeps once the priority has settled
        every contest over them are taken out, with everything that stands on
        nothing else. The label then holds the faces they leave behind, at the
        dimension below. It is for a body whose skin carries a condition and
        whose inside is not meshed. A face among those that another label was
        drawn over is that label's and not this one's, so nothing is contested
        a second time. What another label was drawn over inside the pieces
        leaves with them and is reported as :attr:`Mesh.trimmed`. One per
        label, as a priority is.
    :param inward: for a label at the dimension below the filled one, the
        direction from its faces into the model, in the coordinates the shapes
        were drawn in. What the fragmenting leaves on the other side of those
        faces is left out of the mesh, and the faces are then where the model
        ends. Refused wherever what would be left out is not the region in front
        of the faces run on past them - see :mod:`.ends` - and ``None`` leaves
        the model whole. One per label, as a priority is.
    """

    label: str
    dimension: int
    file: str
    priority: int = 0
    divides: bool = False
    leaves: bool = False
    inward: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class Mark:
    """A shape that says where a place stands, and claims nothing.

    It is cut up with every piece, so where it meets the model it stands on the
    model's own entities, and a place naming it is laid at what it became. It
    takes no piece of the model, joins no contest over one and makes no group:
    whatever label was drawn over a piece keeps it. What it became where no
    label was drawn is taken out of the model before anything is meshed, so a
    mark reaching past the model marks nothing there.

    Where it stands inside what a label holds without matching a boundary of
    it, the model is cut there: a face is meshed to on both sides, and a body
    divides the label's region in two, both parts still the label's. A face the
    mark crosses is cut into pieces, and the label holding that face holds
    every piece.

    A mark of the filled dimension standing wholly inside what a label that
    leaves the model takes out stands on the faces the body it stood in leaves
    behind, since a size asked of a body the model no longer holds is asked of
    its skin. One that stands partly outside it keeps what is still in the
    model, and those faces bound it.

    A point or a curve it became standing inside a body, in a mesh filling
    volumes, is not meshed round: the body's elements have no corner on it, and
    the size a place asks there is measured from where it stands. Gmsh's
    Delaunay mesher sizes the elements round a node or a chain of edges it is
    made to keep inside a body from sizes held at those nodes rather than from
    the size asked - see :func:`~.labels.release` - so a point or a curve kept
    there would be sized nowhere but along itself. On a face either stays in the
    face's mesh, which is sized first.

    :param name: what a place names it by. No label carries the same name.
        Several marks may carry one name, and then they are drawn at one
        dimension between them.
    :param dimension: as :attr:`Piece.dimension`.
    :param file: as :attr:`Piece.file`.
    """

    name: str
    dimension: int
    file: str


@dataclass(frozen=True)
class Near:
    """An element size at what a label or a mark holds, growing to the coarsest
    size away from it.

    A distance is measured from points, curves and surfaces. So at a label of
    the filled dimension the size is laid throughout it, as :class:`Within`
    lays it, and grows away from the rim round it, as :class:`AtRim` grows it.

    :param name: what the answer reports this place under. The caller's to
        choose, and one to a place.
    :param label: the label or the mark whose entities the size is asked at, of
        the filled dimension or below it.
    :param size: the element size at those entities, in millimetres. Finer than
        :attr:`Demand.coarsest` and not below :attr:`Demand.finest`.
    """

    name: str
    label: str
    size: float


@dataclass(frozen=True)
class AtRim:
    """An element size along the rim of what a label holds, growing to the
    coarsest size away from it.

    The rim is what bounds the label's entities once the fragmenting has cut
    them up, one dimension below the label: the curves round the edge of a set
    of faces, or the faces round a set of volumes. An entity two of the label's
    own entities share is inside the set rather than round it, so a face the
    fragmenting cut in two keeps the same rim it was drawn with.

    A rim on the model's frontier is a rim like any other. The label is of the
    filled dimension or the one below it, and never of points, which nothing
    bounds.

    An entity bounds the set only where it bounds one of the label's entities.
    So neither a fold of the label's faces nor a curve three of them meet at is
    on its rim, and a label closed on itself has no rim at all. Nothing is laid
    there, and :attr:`Reached.laid` says so.

    A label that leaves the model is the exception. What it holds is the skin
    of a body, which closes, so its rim is every curve on that skin but a seam:
    the body's edges, and the curves the fragmenting cut into the skin where
    something else meets it.

    A rim asked ``reentrant`` is the label's edges where the room turns round
    them past half a turn, and no others. Every curve bounding the label's
    faces but a seam is a candidate, a body's skin, a sheet's free edge and a
    fold of it alike, and each is kept where the room it stands in opens round
    it wider than :data:`HALF_TURN` by more than :data:`TURNING`, as
    :func:`~.wedges.opening` reads it against :attr:`Demand.walls` and
    :attr:`Demand.mirrors`. So a free edge, a step and the outside of a fold
    are kept, and a corner the room stands in, a flat seam and a sheet's edge
    against a wall are not. The label is of faces in a model filling volumes,
    since the opening is read round a curve.

    :param name: as :class:`Near`'s.
    :param label: the label or the mark whose rim the size is asked at.
    :param size: as :class:`Near`'s.
    :param reentrant: whether only the edges the room turns round past half a
        turn are the rim.
    """

    name: str
    label: str
    size: float
    reentrant: bool = False


#: Half a turn, in radians: how far the room opens round a flat seam.
HALF_TURN = math.pi

#: How far past :data:`HALF_TURN` the room turns round an edge a ``reentrant``
#: rim keeps: one degree. Two faces meeting tangentially, a fillet's edge or a
#: joint a file exchange re-approximated, read half a turn give or take the
#: kernel's error, and the field at an edge opening one degree past half a turn
#: is singular as the distance to the power of minus one in one hundred and
#: eighty-one.
TURNING = math.radians(1.0)


@dataclass(frozen=True)
class Within:
    """An element size throughout the entities a label holds, at the filled
    dimension.

    No growth is laid round it. The size stops at the label's boundary, and the
    elements beside it grow at whatever rate the mesher reaches on its own.

    :param name: as :class:`Near`'s.
    :param label: a label or a mark of the filled dimension.
    :param size: as :class:`Near`'s.
    """

    name: str
    label: str
    size: float


#: A size asked for at one place in the drawing.
Place = Near | AtRim | Within


def transition(coarsest: float, size: float, growth: float) -> float:
    """How far from a place the size reaches the coarsest one, in millimetres.

    The size grows by ``growth - 1`` millimetres for each millimetre away from
    the place, so an element of the place's size standing on it ends where the
    size is ``growth`` times the place's.
    """
    return (coarsest - size) / (growth - 1.0)


@dataclass(frozen=True)
class Demand:
    """The element size asked for, in millimetres.

    ``coarsest`` bounds every element in the model and ``finest`` is a floor
    under all of them. Where the drawing carries a feature finer than
    ``coarsest``, the elements at the feature are shaped by the drawing rather
    than by the request. Where no place lays a field, the size of the elements
    laid round a face or a volume is also carried into it, so a finer
    ``coarsest`` can come back as the same mesh. Where a place lays a field or
    ``per_turn`` sizes curved surfaces, that is turned off, and away from the
    drawing's features the size is the field's or ``coarsest``.

    ``finest`` of zero asks for no floor, which is the ordinary case. Both are
    checked before Gmsh is asked: a ceiling of nothing or less stops Gmsh with
    a message about an internal quantity, and a floor above the ceiling is
    taken silently and meshed to the ceiling.

    :param places: sizes asked for at places in the drawing, each finer than
        ``coarsest``. A size here refines and never coarsens: the coarsest size
        is a ceiling over every place. Where none is stated the mesh is the one
        the two bounds give.
    :param growth: how fast the size grows away from a place: an element of the
        place's size standing on it ends where the size is ``growth`` times the
        place's. Above one wherever a place grows away from itself, which is
        every kind but :class:`Within`, and read nowhere else.
    :param walls: the labels whose faces end the room on each side of them, for
        a ``reentrant`` :class:`AtRim` to read an opening against. Read nowhere
        else.
    :param mirrors: the labels whose faces the room is reflected in where the
        model ends, which an opening ending on a wall doubles. Read where
        ``walls`` is.
    :param per_turn: how many elements stand round a full turn of each curved
        curve and surface of the drawing, at the sharpest curvature at each
        point, wherever that is finer than what else sizes the elements there.
        Zero asks for nothing. A curved element follows its surface only
        across a limited arc, and a curve tighter than the element asked for
        otherwise turns elements inside out. On a surface the size is one
        length at each point (``src/mesh/BackgroundMeshTools.cpp:222``), so it
        holds along a curved surface's straight direction as well. ``finest``
        bounds it where the curvature grows without bound, as at the apex of a
        cone.
    """

    coarsest: float
    finest: float
    growth: float = 0.0
    places: tuple[Place, ...] = ()
    walls: tuple[str, ...] = ()
    mirrors: tuple[str, ...] = ()
    per_turn: int = 0


@dataclass(frozen=True)
class Profile:
    """What the mesh is to be, which the caller's backend decides and a user
    never sees.

    It has no unit anybody could check and nothing in it is a preference. A
    straight-sided mesh into a volume method is legal where the same mesh into a
    surface method is a refusal, and on a drawing with a curve in it the
    difference between straight-sided and curved is the difference between an
    answer and a wrong one.

    :param top: the dimension to fill with elements, and one of
        :data:`DIMENSIONS`. A method working on the
        surface of a closed body asks for that surface and gets nothing inside
        it; one working on a volume asks for the volume. What else the file
        carries follows from the labels, since an entity reaches the file by
        being in a group.
    :param element_order: the geometric element order. One is straight-sided.
    :param curved: whether the nodes an order above one are placed on the drawn
        surface rather than on the chord between the corners. It does nothing
        at element order one, where there are no such nodes.
    :param written: which of :data:`FORMATS` to write.
    :param connected: whether what is filled has to be one region, each part
        of it reached from every other through entities of the dimension below
        that bound two parts at once - a face between two volumes, and not an
        edge or a point where they touch. Asked for, a drawing that comes apart
        is refused before an element is built, naming the labels on each part
        and where it is. The kernel joins two shapes only closer than its own
        tolerance, so bodies drawn to meet and left a hair apart come apart
        here, and nothing else in the mesh says so.

        Which parts are joined is topology and nothing more. A caller that
        puts a condition on an entity inside what is filled has divided it in
        a way this cannot see.
    """

    top: int
    element_order: int = 1
    curved: bool = False
    written: str = "msh22"
    connected: bool = False


@dataclass(frozen=True)
class Label:
    """Where one of the caller's labels went in the mesh.

    :param dimension: the dimension of everything in the group.
    :param tag: the physical tag, unique across the whole mesh. Gmsh itself
        keeps a tag unique only within its own dimension, and would let a
        surface group and a volume group both hold tag one; one of the formats
        written here has a single namespace for group numbers, and the second
        group to arrive would take the first one's name. So the tags are
        counted once for the model, and a caller still states the dimension
        because a Gmsh model it did not write may not.
    :param entities: the entities the label was left holding, after the
        fragmenting cut them up and after any piece two labels covered went to
        the one the priority named. They name that model, which is gone by
        the time this is returned, so they are what the mesh was built from
        rather than something to ask about.

        A remainder label was drawn as nothing, so none of this happened to it:
        its entities are the ones the fragmented model had left at the dimension
        below the filled one where the model ends, and no shape of the caller's
        was ever cut up under it.
    :param sits: :data:`FRONTIER`, :data:`INTERIOR` or :data:`BOTH` for a label
        of the dimension below the filled one, and ``None`` for any other.
    :param lower: the least corner of the box round the label's nodes as
        meshed, in the drawing's units. The nodes rather than the geometry,
        whose box the kernel pads by its tolerance.
    :param upper: the greatest corner of that box.
    :param size: the length, area or volume of what the label holds, the
        geometry's. With the box, it says whether a face fills the rectangle
        round it.
    :param edges: the shortest and the longest edge among the label's own
        elements, and ``None`` for a label of points, whose elements have none.
    :param elements: how many elements of its own dimension the label holds.
    :param beside: for a label of the dimension below the filled one, the labels
        of the filled dimension its entities bound or stand inside, in name
        order, each once. Read off the fragmented model rather than off the
        drawing, so a face the fragmenting cut into pieces names every label on
        either side of any piece, and a face cut by a mark alone names the same
        labels it would have named uncut. Empty for a label of any other
        dimension.
    """

    dimension: int
    tag: int
    entities: tuple[int, ...]
    sits: str | None
    lower: tuple[float, float, float] | None = None
    upper: tuple[float, float, float] | None = None
    size: float | None = None
    edges: Edges | None = None
    elements: int = 0
    beside: tuple[str, ...] = ()


@dataclass(frozen=True)
class Settled:
    """One piece more than one label was drawn over, and who ended up with it.

    A caller reads this to check that what it drew over what is what it meant.
    Nothing here says the drawing is wrong - an ordinary drawing puts a part
    inside a domain and every piece of the part is settled this way - and
    nothing else in the answer records that a label was drawn over a region it
    did not keep.

    :param dimension: the dimension of the piece.
    :param tag: the piece, in the fragmented model. A piece a label that leaves
        the model took is not in the mesh.
    :param took: the label the priority gave it to.
    :param gave_up: the other labels drawn over it, in name order.
    :param place: where the piece is, in the coordinates the caller drew in.
    """

    dimension: int
    tag: int
    took: str
    gave_up: tuple[str, ...]
    place: str


@dataclass(frozen=True)
class LeftOut:
    """One part of the drawing left out of the mesh, behind a face that ends the model.

    Nothing about it is a fault: it is what a caller that handed a label a way
    into the model asked for. It is reported because it is a region somebody
    drew and the mesh does not hold, which nothing else in the answer shows.

    :param behind: the label whose faces it stood behind.
    :param place: where it was, in the coordinates the caller drew in.
    :param held: the labels drawn over it, in name order.
    """

    behind: str
    place: str
    held: tuple[str, ...]


@dataclass(frozen=True)
class Trimmed:
    """What a label was drawn over inside a label that leaves the model.

    It leaves with the pieces it stands in, and nothing else in the answer shows
    it. Whether losing that part changes what the label means is the caller's
    to judge.

    :param label: the label that lost it.
    :param by: every label that leaves the model and whose pieces it stood on,
        in name order.
    :param place: where it was, in the coordinates the caller drew in.
    """

    label: str
    by: tuple[str, ...]
    place: str


@dataclass(frozen=True)
class Part:
    """One part of what is filled, once every face under a dividing label is a cut.

    A region a condition closes off is one region to :attr:`Profile.connected`,
    which asks what the faces join and knows nothing of what a label means. So
    the parts are asked again with the faces the caller said nothing crosses
    taken out of the joining, and what comes of it is reported rather than
    refused: a part standing on its own is a cavity where the caller can reach
    it and a mistake where it cannot, and only the caller knows which.

    Reported only where a label said it divides, and a model that stayed whole
    under the cuts is reported as the one part it is. Where no label said so,
    :attr:`Mesh.parted` is empty rather than one part: nothing was asked.

    :param labels: every label of the filled dimension holding an entity of
        this part, and every label of the dimension below standing on one of
        them, in name order - what fills it and what bounds it, the dividing
        labels included. A label lower than either is on no part.
    :param place: where it is, in the coordinates the shapes were drawn in.
    """

    labels: tuple[str, ...]
    place: str


@dataclass(frozen=True)
class Edges:
    """The shortest and the longest edge among the elements of the meshed dimension.

    In the units the shapes were drawn in, which is what a :class:`Demand` is
    stated in, so what was asked for and what came back are the same quantity
    and can be compared. A size asked for is a target at a point and not a
    bound on an element, so a longest edge above ``coarsest`` is ordinary
    rather than a fault - which is the reason to report the pair instead of
    checking it.

    Measured between the corners, so a curved element is measured along the
    chord its corners span rather than along the surface its middle nodes sit
    on.

    Taken off the model rather than off the written file. The two differ by
    what the writer keeps, and a drawing far from the origin spends digits on
    the offset.

    :param shortest: the shortest edge, in the units the shapes were drawn in.
    :param longest: the longest.
    """

    shortest: float
    longest: float


@dataclass(frozen=True)
class Reached:
    """What the mesh holds at one place, in millimetres.

    Nothing here is checked against the size asked for. A size is a target at a
    point rather than a bound on an element, so an element above it is the
    ordinary case, and the figures are reported for the caller to read.

    :param asked: the size the place states.
    :param reached: the figure to set against ``asked``. At a place of curves or
        surfaces, the longest edge among the elements lying on them, which is
        the size along the place itself. At a place of points, the longest edge
        of the filled dimension leaving them. At a mark's point or curve standing
        inside a body, which no element has a corner on, the longest edge of the
        elements holding a node of it. Throughout a label, and near a
        label or a mark of the filled dimension, the median of the mean edge of
        the elements of the filled dimension in it. A steep growth leaves even
        this coarser than ``asked``.
    :param standing: the median of the longest edge of the elements of the
        filled dimension at the place: those with a corner on it, those holding
        a node of a mark's point or curve inside a body, or those in it where
        the place is of the filled dimension. Beside a place of curves
        or surfaces this is what the growth leaves next to it, which is larger
        than the size along it by more the steeper the growth. ``None`` where no
        element is at the place.
    :param elements: how many elements of the filled dimension are at the place.
    :param laid: whether a size was laid there. ``False`` at the rim of a label
        closed on itself, which has none, and at a mark standing wholly outside
        the model; then no element is at the place.
    :param along: at a place of curves or surfaces, the median of the longest
        edge of the elements lying on them. ``reached`` is the greatest of the
        same edges, one element's out of every one on the place, and this is
        what most of them are. ``None`` at any other place, and at a mark's
        curve standing inside a body, on which no element lies.
    :param dimension: the dimension of what the place stands on: points,
        curves, surfaces, or the filled dimension for a place sized throughout.
    :param extent: the length, area or volume of what the place stands on, the
        geometry's: a rim's length, a face's area, a body's volume. ``None`` at
        a place of points and where nothing is laid.
    :param left: at a ``reentrant`` rim, how many of the label's edges the room
        turns round by no more than half a turn, which nothing is laid at.
    """

    asked: float
    reached: float | None
    standing: float | None
    elements: int
    dimension: int
    laid: bool = True
    along: float | None = None
    extent: float | None = None
    left: int = 0


@dataclass(frozen=True)
class Mesh:
    """The mesh that was written, and what built it.

    :param path: the file, in the units the shapes were drawn in.
    :param labels: what each label the caller gave became, and the remainder
        where one was named and anything was left over for it. A remainder is
        not among the caller's shapes, so it
        is here and nowhere else: what it holds is not knowable before the
        fragmenting.
    :param worst_quality: per dimension carrying elements whose shape can be
        measured, the worst quality among the elements the file holds. Taken
        over the physical groups rather than the model, because an entity in no
        group is meshed and not written, and taken after any high-order pass,
        because the quality of a straight-sided element says nothing about the
        curved one that replaced it.

        Small and positive is reported and never refused on: it is the only
        number available before a solve that separates a mesh carrying a sliver
        from its siblings, which the residual of the run does not at every
        solver order, and a thin conductor is a body people draw. Nothing at or
        below zero comes back at all: the measure is signed, so such an element
        is flat or has its corners in the wrong order, and it lies partly
        outside the drawing.
    :param edges: the shortest and the longest element edge, over the dimension
        the profile filled. The demand is a length and the mesh that answers it
        is a length, and without this nothing states the second. Where the
        drawing carries a feature finer than the size asked for, the drawing
        decides the element instead, so a band of requested sizes produces one
        mesh - and a caller turning that number has nothing else that says so.
    :param settled: every piece more than one label was drawn over, and which
        label the priority gave it to. Empty where no two shapes overlapped,
        which is what a caller that subtracted before handing over will see.
    :param version: the Gmsh that built it.
    :param algorithm: per dimension, the meshing algorithm asked for, in Gmsh's
        own numbering. Read back off the option rather than out of the mesh, so
        it says what was requested and not what Gmsh fell back to. Here at all
        because it is a choice made below the caller and it decides the
        elements, so two runs alike in every stated thing can differ in it.
    :param rims: what stands on each entity a label at the dimension below the
        filled one holds, remainder included, a dimension at a time, each named
        by its dimension and its tag. Under that dimension, each such entity,
        the entities of the dimension below bounding it, and whatever stands
        embedded in it; under the dimension below that, each entity of that
        dimension listed and the entities bounding it - a face's curves and the
        points left on it, and each curve's end points, in a volume mesh. The
        tags name the model the mesh was built from, so what they answer is
        which labels meet where: two entities listing one entity touch there. A
        dimension is listed only where there is one below it.
    :param bounds: the box round each entity :attr:`rims` lists at the dimension
        two below the filled one, per dimension and tag, as its least and its
        greatest corner - each curve of a labelled face, in a volume mesh. Taken
        over points on the curve rather than off the kernel's box, which is
        padded by the curve's tolerance, so a curve across an axis spans
        nothing along it whatever tolerance the shape carries.
    :param numbered: the second file, where one was asked for, holding the mesh
        :attr:`path` holds with each group named by its :attr:`Label.tag`.
        Empty where none was.
    :param left_out: every part of the drawing left out behind a face a label
        was handed a way into the model for. Empty where no label was.
    :param reached: what the mesh holds at each place the demand states, under
        the place's name. Empty where none was stated.
    :param elements: how many elements of the filled dimension the model holds,
        over every label of that dimension, which is what a place's share of
        the model is counted against.
    :param trimmed: everything a label lost inside a label that leaves the
        model - see :class:`Trimmed`. Empty where no label leaves it.
    :param parted: what is filled, split where a label said nothing crosses it -
        see :class:`Part`. Empty where no label said so, and one entry where
        the model stayed whole.
    """

    path: str
    labels: dict[str, Label]
    worst_quality: dict[int, float]
    edges: Edges
    settled: tuple[Settled, ...]
    version: str
    algorithm: dict[int, int]
    rims: dict[int, dict[int, tuple[tuple[int, int], ...]]]
    numbered: str = ""
    left_out: tuple[LeftOut, ...] = ()
    bounds: dict[int, dict[int, tuple[tuple[float, float, float], tuple[float, float, float]]]] = (
        field(default_factory=dict)
    )
    reached: dict[str, Reached] = field(default_factory=dict)
    elements: int = 0
    trimmed: tuple[Trimmed, ...] = ()
    parted: tuple[Part, ...] = ()


class Refused(Exception):
    """The request, the drawing or its labels do not describe a mesh.

    Nothing has been meshed when this is raised: every question behind it is
    answered before Gmsh is asked for an element.

    Each complaint names something the caller wrote - a label, a size, an order,
    a format - or where in the drawing the fault is, because the layer that
    knows both the drawing and the label is the only one that can. A backend
    handed the mesh instead stops on an internal face number and two integers,
    or on the name of an indexing operator, and neither reaches anything a user
    drew. A piece nothing claims has no label to be named by, and is named by
    the coordinates it was drawn in.
    """

    def __init__(self, complaints: Sequence[str]) -> None:
        super().__init__("; ".join(complaints))
        #: Each thing wrong with the request or the labelling, one sentence
        #: apiece.
        self.complaints: tuple[str, ...] = tuple(complaints)


class Unmeshed(Exception):
    """Gmsh was asked for this mesh and there is no mesh of the drawing.

    A caller acts the same way on each of the things that raise it. The kernel
    fails to cut the drawn shapes against each other, which is :class:`Uncut`.
    Gmsh stops and says why, carrying the error it logged. Gmsh finishes,
    reports no error, and leaves a region of the filled dimension with nothing
    in it. Gmsh finishes and returns elements turned inside out, which lie
    partly outside the drawing, and the labels holding them are named.

    The last two are the ones nothing else would notice. Every label reached its
    entities and every group is one Gmsh would write, and what is wrong is the
    space the elements cover: a region of it carries none of them, or some of
    them reach outside it. No file is written for any of them - the checks run
    before the write - so a caller that finds a file has a mesh.

    :param complaints: what is missing, one sentence apiece.
    :param said: the error and warning lines Gmsh logged while it worked. Gmsh
        writes them to a terminal, which a caller running the mesher beside its
        document does not have. Empty in a session the caller opened, where the
        logger is the caller's and holds everything since the caller started it.
    """

    def __init__(self, complaints: Sequence[str], said: Sequence[str] = ()) -> None:
        super().__init__("; ".join(complaints))
        self.complaints: tuple[str, ...] = tuple(complaints)
        self.said: tuple[str, ...] = tuple(said)


class Uncut(Unmeshed):
    """The kernel could not cut the drawn shapes against each other, so Gmsh was
    never asked for an element.

    Each shape can be sound and the cut still fail. A curve standing inside a
    solid that stands inside another fails this way where the file marks the
    inner solid as part of another shape, as a solid once put in a compound is
    written, and where Gmsh has copied the solid or cut something else against
    it. The same drawing is cut where the file marks the solid free. The
    complaint names the labels whose shapes were being cut. Finer elements
    change nothing here, so a caller advising on the element size leaves this
    one out.
    """
