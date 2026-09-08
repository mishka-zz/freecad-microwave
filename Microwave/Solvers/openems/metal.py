# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Where a piece of metal is, and what its faces ask of the grid.

The questions come in two kinds. One is about the drawing, and a box cannot
answer it: which boxes are one piece of metal, where that piece ends, which
planes are its faces, whether it has a width on an axis at all.
:func:`_grouped_into_conductors` asks all of those once for the whole drawing,
and :class:`Conductor` is where the answers are carried and why.

The other is about one face: the cell the grid is built from there, the pair of
lines that straddle it, and whether the face is an isolated edge at all.
:func:`_edge_pair` is that question, and both the pinning and the edge demand
put it.

The arithmetic is comparison of box corners, and the pair of lines a face
would be meshed with. Nothing here reads a grid: a face is answered from the
drawing and the policy alone, and :mod:`~.mesh` decides which of the answers
becomes a line.

docs/internals/conductor-width.md works out why a face is sized from the width
behind it, and what a policy that ignores the width costs the metal openEMS
builds.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from ...portbox import FLATNESS, Box
from .regions import DIMENSIONS, MaterialClass, MeshParams, Region

#: How near a conductor's drawn width the grid has to bring the metal openEMS
#: builds, as a fraction of that width.
#:
#: openEMS reads a cell's material at one sample point, so a face arrives on
#: whichever of the two lines straddling it is the nearer.
#: :data:`~.regions.EDGE_LINE_INSIDE` decides which that is, and the face therefore moves
#: by ``min(share, 1 - share)`` of a cell whatever the policy asked for. On a
#: narrow trace that is a large part of the metal. The mesher holds the width to
#: this fraction instead.
#:
#: Placed by measurement, on a microstrip read as the propagation constant
#: between two lengths of one line, and set just under the least share measured
#: to answer acceptably rather than in the middle of the range. See
#: ``docs/internals/conductor-width.md``, which carries the figures and the
#: method. That measurement is of metal the grid lost. The same tolerance is
#: applied to metal it gains, which is not measured and is the looser half of
#: this figure.
CONDUCTOR_WIDTH_KEPT = 0.95


#: Which way the void lies at a conductor's lower and upper face on an axis.
_OUTWARD = (-1.0, 1.0)


@dataclass(frozen=True)
class ConductorFace:
    """A plane one piece of metal has a face at, and how deep the metal is there.

    ``depths`` rather than a depth, because one plane is a face of the same metal
    in several places at once - a comb's teeth all end on one line - and how deep
    the metal is behind each is a different number. Which of them decides a cell
    is a question about the policy, so they are kept whole here and narrowed
    where the policy is known. See :func:`_face_depth`.
    """

    plane: float
    at_high: bool
    depths: tuple[float, ...]


@dataclass(frozen=True, kw_only=True)
class Conductor(Region):
    """A metal region, with the piece of metal it is one of worked out.

    A solid reaches the mesher as the boxes the translation cut it into, and
    almost everything the grid wants to know about metal is a question about the
    solid rather than about one box: where the metal ends, which faces are faces
    of it, whether it has a width on an axis at all. None of it can be answered
    from a box alone, so it is answered once for the whole drawing by
    :func:`_grouped_into_conductors` and carried here.

    A metal region that has not been through it is a :class:`Region`, so what
    grouping worked out either arrives whole or the object is a different type,
    and no reader has to decide what an absent field stands for.
    """

    #: The run of metal through this box: the corners it reaches everywhere
    #: across its own cross-section. Read to decide whether the metal carries on
    #: past a face, which the box's own corners cannot answer - they stop where
    #: the cut did.
    run: Box
    #: Which connected piece of metal this region is part of.
    #:
    #: Not derivable from :attr:`run`, which is a box. An L's box holds space its
    #: metal does not, so two conductors whose boxes overlap need not be one
    #: piece, and two boxes of one piece need not touch each other. A chain of
    #: boxes that do touch makes them one piece, and no pair of boxes can answer
    #: that on its own.
    piece: int
    #: The faces that piece of metal has, one tuple per axis.
    #:
    #: Not derivable from :attr:`run` either, which holds only what the metal
    #: reaches everywhere across this box. A face is open over part of a
    #: cross-section and covered over the rest, and how deep the metal is behind
    #: the open part is a question a box cannot be asked. Two cuts of one solid
    #: answer it differently. Read wherever a face is the subject; whether the
    #: axis is a width at all is :attr:`piece_box`'s question.
    faces: tuple[tuple[ConductorFace, ...], ...]
    #: The corners of that piece of metal.
    #:
    #: A third box and a third question. :attr:`run` is this box's run, which the
    #: cut moves. This is the piece's own extent, which the cut cannot move. It
    #: says whether the metal has a width on an axis at all: a wire is a wire
    #: whichever rectangles it was cut into. See :func:`width_axes`.
    piece_box: Box


@dataclass(frozen=True)
class Grouped:
    """Every region of one drawing, each conductor knowing its piece of metal.

    The order is the order they arrived in, because a demand's place in the list
    decides which box a coincident anchor is named after and how the field's
    bends are summed. Metal is a :class:`Conductor` and everything else a
    :class:`Region`, so ``isinstance`` is what a reader about to take one of the
    fields grouping filled in asks.

    It carries nothing else. A function taking one of these is being told the
    question about the whole drawing has already been asked of every region in
    it, which is the only thing a bare list cannot say.
    """

    regions: tuple[Region, ...]


def _meets(box: Box, lower: Sequence[float], upper: Sequence[float], axis: int) -> bool:
    """Whether ``box`` lies over this cross-section on ``axis``.

    Strict where the cross-section has width, so a box merely butted against its
    side is beside it rather than over it and answers for nothing there. Closed
    where it has none, so a conductor drawn as a surface passes: its
    cross-section is a plane, and metal resting on that plane does lie over it.

    For :func:`conductor_run` this is only a cost filter. A box failing it covers
    no point of the cross-section, and its faces mark no change in what does, so
    dropping it changes no answer. For :func:`_met_by_metal` it is the rule. What
    the metal past a face is made of is a question about metal actually over that
    face.
    """
    if upper[axis] - lower[axis] <= FLATNESS:
        return box[0][axis] <= upper[axis] + FLATNESS and box[1][axis] >= lower[axis] - FLATNESS
    return box[0][axis] < upper[axis] - FLATNESS and box[1][axis] > lower[axis] + FLATNESS


def _bands(low: float, high: float, cuts: Iterable[float]) -> list[tuple[float, float]]:
    """``low`` to ``high``, divided at every cut that falls strictly inside it.

    A span of no width comes back as one band of no width rather than none at
    all, so that a conductor drawn as a surface can still be answered about. Over
    no bands the shallowest run is unbounded, so every sheet would report itself
    arbitrarily wide and quietly lose the edge treatment on every face.

    A band of no width inside a span is never wanted. A decomposed solid puts
    several boxes' faces at one coordinate, so the cuts arrive repeated. Taken at
    face value they divide the span there twice and leave a band between the two.
    A band stands for the coverage at its middle, and that middle is the plane
    itself, where the metal on both sides answers and the run comes back deeper
    than any column the metal has.
    """
    inside: list[float] = []
    for cut in sorted(cuts):
        if low + FLATNESS < cut < high - FLATNESS and (not inside or cut - inside[-1] > FLATNESS):
            inside.append(cut)
    walk = [low, *inside, high]
    return list(zip(walk, walk[1:]))


def _band_runs(
    lower: Sequence[float], upper: Sequence[float], others: Iterable[Box], dim: int
) -> Iterator[tuple[float, float]]:
    """The run of metal containing this box, over each band of its cross-section.

    The cross-section is divided into bands at every face that crosses it, and
    within a band every meeting box's coverage is constant - the only places it
    changes are that box's own faces, and those are cuts. One point therefore
    decides a band, and the run there is the span of the union containing the
    box's own.

    Both questions a conductor's width raises are read off this one sweep, so
    they cannot drift apart. :func:`conductor_run` takes the shallowest band,
    which is how far the metal reaches everywhere across this box.
    :func:`conductor_faces` takes each band's two ends, which are the planes the
    metal has a face at and how deep it is behind each.

    One axis at a time, and never two at once. Growing on two would change the
    cross-section as it went, and an L would reach different metal depending on
    which arm was followed first.
    """
    boxes = list(others)
    across = [axis for axis in range(DIMENSIONS) if axis != dim]
    meeting = [box for box in boxes if all(_meets(box, lower, upper, axis) for axis in across)]
    for first, second in itertools.product(
        *(
            _bands(lower[axis], upper[axis], (end[axis] for box in meeting for end in box))
            for axis in across
        )
    ):
        middle = ((first[0] + first[1]) / 2.0, (second[0] + second[1]) / 2.0)
        spans = sorted(
            (box[0][dim], box[1][dim])
            for box in meeting
            if all(
                box[0][across[n]] <= middle[n] + FLATNESS
                and box[1][across[n]] >= middle[n] - FLATNESS
                for n in range(len(across))
            )
        )
        # Swept in order of where each box starts, so one pass reaches the end of
        # the run: a box that extends it can only be met after every box that
        # reaches it. Repeated passes over an unsorted list find the same answer
        # and turn meshing a long chain of rectangles into an interactive wait.
        band_low, band_high = lower[dim], upper[dim]
        for start, end in spans:
            if start > band_high + FLATNESS:
                break
            band_high = max(band_high, end)
        for start, end in sorted(spans, key=lambda span: span[1], reverse=True):
            if end < band_low - FLATNESS:
                break
            band_low = min(band_low, start)
        yield band_low, band_high


def conductor_run(
    lower: Sequence[float], upper: Sequence[float], others: Iterable[Box], dim: int
) -> tuple[float, float]:
    """How far the metal this box is in reaches either way along ``dim``, everywhere.

    Two conductors butted face to face are one piece of metal. The field
    penetrates neither, so the seam between them is not a boundary of anything. A
    piece of a strip is therefore not a narrower strip, and this function reports
    how wide the strip actually is. The translation cuts a drawn outline into
    rectangles by itself, so most conductors reach the grid in pieces nobody
    drew, and a width read off one of those is a width of nothing.

    The metal has to cover the box's whole cross-section for the run to carry on,
    and that is asked of the boxes together rather than of any one of them.
    Asking it of one makes the answer depend on the cut: a box whose continuation
    is split across its own cross-section - a plate with two pads standing on it
    - then reports a run shorter than any run the metal has.

    The cross-section is therefore divided into bands - see :func:`_band_runs` -
    and the answer is the shallowest of them. That says where the metal is
    everywhere across the box, so a box no deeper than this is inside metal
    throughout. It is not what a face asks. A face is open over only some of
    those bands, and how deep the metal is behind the open part is
    :func:`conductor_faces`.
    """
    # The shallowest band decides, so this starts where nothing can be shallower
    # and closes in. Every band answers at least the box's own interval, which
    # stops an empty cross-section from returning the unbounded pair.
    low, high = -math.inf, math.inf
    for band_low, band_high in _band_runs(lower, upper, others, dim):
        low, high = max(low, band_low), min(high, band_high)
    return low, high


def conductor_extents(lower: Sequence[float], upper: Sequence[float], others: Iterable[Box]) -> Box:
    """How far the metal reaches everywhere across the box ``lower`` to ``upper``.

    Three runs, one per axis - see :func:`conductor_run`, which is where what a
    run is and why it is asked that way are written down.

    ``others`` are the surrounding boxes of the same metal, this one included or
    not as the caller finds convenient.

    This is a width and never what a face is worth. A face may open over only
    part of a box's cross-section. The mesher's demand and pre-flight's complaint
    share :func:`conductor_faces`. This function is read where the question is
    how far the metal reaches everywhere across a box.
    """
    boxes = list(others)
    runs = [conductor_run(lower, upper, boxes, dim) for dim in range(DIMENSIONS)]
    return tuple(run[0] for run in runs), tuple(run[1] for run in runs)


def conductor_faces(
    subjects: Sequence[Box], others: Iterable[Box], dim: int
) -> tuple[ConductorFace, ...]:
    """Every plane one piece of metal has a face at along ``dim``.

    ``subjects`` are that piece's boxes; ``others`` the surrounding metal a run
    is measured through, this piece included or not as the caller finds
    convenient.

    A face belongs to the metal rather than to the cut, so it is asked of the
    piece rather than of each box in turn. Every plane the metal ends at is the
    end of some band's run - metal stops there, so some box has a face there, and
    coverage is constant across that box's band. Walking every box's bands
    therefore reaches every face of the piece and nothing that is not one, and
    two tilings of one piece build the same table.

    A plane appears here exactly as some box named it, so two boxes cut apart can
    leave two entries for one face of the metal. They are not gathered.
    Everything reading the table asks for the planes within :data:`FLATNESS` of
    the one it names, and gathering them first would answer that a second time.
    """
    seen: dict[tuple[bool, float], set[float]] = {}
    boxes = list(others)
    for lower, upper in subjects:
        for low, high in _band_runs(lower, upper, boxes, dim):
            for at_high, plane in enumerate((low, high)):
                seen.setdefault((bool(at_high), plane), set()).add(high - low)
    return tuple(
        ConductorFace(plane, at_high, tuple(sorted(depths)))
        for (at_high, plane), depths in seen.items()
    )


def width_axes(lower: Sequence[float], upper: Sequence[float], cell: float) -> tuple[int, ...]:
    """The axes across which a conductor has a width the grid resolves.

    Every axis it spans more than one ``cell`` of, where ``cell`` is the size the
    policy asks for at metal.

    An axis it spans less than that is a thickness, and it is excluded because
    nothing there is at stake. The mesher does not apply the thirds rule across a
    thickness; it pins both faces plainly instead. A conductor whose faces are on
    lines arrives exactly as drawn, so there is no share of it to hold. A foil
    deliberately gets that treatment: a single cell through it, and its own
    thickness handed to openEMS as a material property.

    Anisotropy would be the alternative to reading the width off the cell, and
    it is worse. A rule picking the shortest axis excludes one of a cube's three
    arbitrarily, and the grid then holds two of its faces to a share and the
    third to nothing, for a shape with no thin direction at all.

    A shape with extent on fewer than two axes has no width at all, whatever its
    one span is. It is a line, it encloses nothing, and a share of it describes
    nothing.

    Here rather than beside either caller, so the mesher and pre-flight cannot
    disagree about which axes a demand was spent on and which one a complaint is
    about.
    """
    live = tuple(dim for dim in range(DIMENSIONS) if upper[dim] - lower[dim] > 0.0)
    if len(live) < 2:
        return ()
    return tuple(dim for dim in live if upper[dim] - lower[dim] > cell)


def conductor_pieces(boxes: Sequence[Box]) -> list[int]:
    """Which connected piece of metal each box is in, as an index per box.

    Touching counts, because two conductors butted face to face are one piece,
    which is the reading the rest of this module takes. Connectedness is not a
    pairwise question, so it is answered by walking chains rather than by any
    test two boxes can apply to each other.
    """
    piece = list(range(len(boxes)))

    def root(n: int) -> int:
        while piece[n] != n:
            piece[n] = piece[piece[n]]
            n = piece[n]
        return n

    for one in range(len(boxes)):
        for other in range(one + 1, len(boxes)):
            if root(one) != root(other) and all(
                boxes[one][0][axis] <= boxes[other][1][axis] + FLATNESS
                and boxes[other][0][axis] <= boxes[one][1][axis] + FLATNESS
                for axis in range(DIMENSIONS)
            ):
                piece[root(one)] = root(other)
    return [root(n) for n in range(len(boxes))]


def _grouped_into_conductors(regions: Sequence[Region]) -> Grouped:
    """Every region of the drawing, each conductor carrying the piece it is one of.

    Done once here rather than wherever a width is wanted. It is a question about
    every other region, its callers want all three axes, and asking it again per
    axis per caller made meshing a decomposed outline quadratic in the rectangles
    it was cut into.

    Asked by class rather than by material name, matching :func:`_met_by_metal`.
    The metal either side of a copper-against-PEC seam is still metal, and a
    width is a question about where the metal ends.

    Which connected piece each region is in is settled here too, and so is the
    table of faces that piece has.

    The table is built per piece and the run per box, so the bands are swept
    twice. One sweep answering both would be a second fold of :func:`_band_runs`
    written out here, and two implementations of a run disagree the day one of
    them is corrected.
    """
    where = [n for n, region in enumerate(regions) if region.material is MaterialClass.METAL]
    if not where:
        return Grouped(tuple(regions))
    boxes = [(regions[n].lower, regions[n].upper) for n in where]
    pieces = conductor_pieces(boxes)
    mine = {piece: [box for box, at in zip(boxes, pieces) if at == piece] for piece in set(pieces)}
    faces = {
        piece: tuple(conductor_faces(theirs, boxes, dim) for dim in range(DIMENSIONS))
        for piece, theirs in mine.items()
    }
    extent = {
        piece: (
            tuple(min(box[0][dim] for box in theirs) for dim in range(DIMENSIONS)),
            tuple(max(box[1][dim] for box in theirs) for dim in range(DIMENSIONS)),
        )
        for piece, theirs in mine.items()
    }
    settled: list[Region] = list(regions)
    for slot, piece in zip(where, pieces):
        region = regions[slot]
        settled[slot] = Conductor(
            lower=region.lower,
            upper=region.upper,
            material=region.material,
            label=region.label,
            material_name=region.material_name,
            size=region.size,
            continuous=region.continuous,
            drawn=region.drawn,
            relaxed_to=region.relaxed_to,
            run=conductor_extents(region.lower, region.upper, boxes),
            piece=piece,
            faces=faces[piece],
            piece_box=extent[piece],
        )
    return Grouped(tuple(settled))


def _met_by_metal(region: Conductor, grouped: Grouped, dim: int, at_high: bool) -> Conductor | None:
    """The conductor continuing this region's face along ``dim``, if any.

    Two conductors butted in plane are one piece of metal. The field penetrates
    neither, so the seam carries no edge singularity, and the thirds rule - a
    treatment for an isolated edge - has nothing to resolve there. Without this
    test both sides claim the seam as an edge and lay a pair of lines each, which
    is four lines across continuous metal.

    The face has to be covered rather than merely touched, and that separates a
    seam from a T-junction. A stem meeting the side of a bar ends where the bar
    begins, but the bar's own face at that plane runs past the stem and is still
    exposed for most of its length.

    Asked by class rather than by name, because the singularity is what is being
    ruled out and the metal either side of a copper-against-PEC seam is still
    metal. Whether the seam needs a line is a different question, and the caller
    asks it of the region returned here.

    Whether the metal carries on is read off the run rather than worked out
    again. Two implementations of one predicate disagree the day one of them is
    corrected. The answer is therefore that the piece of metal reaches past this
    face, and the region handed back is only which one to name.

    Where the metal past a face is several regions, one of a different material
    is returned in preference. The caller pins a plain line for that and nothing
    for the other, and a property boundary anywhere on the face moves by up to a
    cell if no line holds it.
    """
    run = region.run
    plane = region.upper[dim] if at_high else region.lower[dim]
    if not (run[1][dim] > plane + FLATNESS if at_high else run[0][dim] < plane - FLATNESS):
        return None
    past = [
        other
        for other in grouped.regions
        if other is not region
        and isinstance(other, Conductor)
        and (
            other.upper[dim] > plane + FLATNESS if at_high else other.lower[dim] < plane - FLATNESS
        )
        and all(
            _meets((other.lower, other.upper), region.lower, region.upper, axis)
            for axis in range(DIMENSIONS)
            if axis != dim
        )
    ]
    if not past:
        return None
    return next((other for other in past if other.material_name != region.material_name), past[0])


def _edge_to_resolve(
    region: Conductor,
    grouped: Grouped,
    dim: int,
    at_high: bool,
    outside: float,
    domain_lower: float,
    domain_upper: float,
) -> bool:
    """Whether this conductor face is an isolated edge, with a singularity on it.

    It is not an isolated edge wherever the metal carries on past the face: the
    region declares the axis continuous, another conductor covers the face, or
    the face sits at the domain wall and the conductor runs on into the absorber.
    Each is the same absence in different words, and each gets the same answer.

    Both the thirds rule and the refinement constraints ask, and where both ask
    they have to agree about a plane. If they disagree, one treats as an edge
    what the other declines to, and the cells there come out several times finer
    than anything asked for, at a pitch the absorber then copies. ``outside`` is
    therefore built from the policy's own resolution at both call sites and never
    from a region's relaxed size. The two must put the same question.

    A relaxed region is the one case where the thirds rule does not ask at all,
    having declined the treatment. The constraint still asks, at the relaxed
    size. Agreement is not at stake there, because only one of them is asking.

    ``outside`` is where the outer thirds line would fall, which decides whether
    the edge has room to be resolved inside the domain at all.
    """
    if dim in region.continuous:
        return False
    if _met_by_metal(region, grouped, dim, at_high) is not None:
        return False
    return domain_lower < outside < domain_upper


def _face_depth(region: Conductor, dim: int, at_high: bool, params: MeshParams) -> float | None:
    """How deep the metal is behind one of ``region``'s faces, or ``None``.

    ``None`` says the face gets no thirds pair. Either the axis is a thickness
    rather than a width - see :func:`width_axes` - or nothing at that plane is a
    face of metal wide enough to hold a share of.

    The depth is used and never :attr:`Conductor.run`. A run stops at the first
    band of the cross-section where the metal does not carry on, and a face is a
    face only over the bands where it does not carry on. Counting a seam band
    into the run asks a finer cell than the metal needs, and asks a different one
    of two cuts of the same solid.

    Only depths a width could be read off count. One piece of metal can have a
    thin fin ending on the same plane a deep arm does. The fin gets plain lines
    of its own, and letting its depth decide the plane would size the arm's pair
    from metal that is not asking for one.

    Whether the axis is a width is asked of the piece rather than of this box's
    run. A run is an intersection over the box's cross-section, and it can be
    narrower than every column of the metal. A face whose axis was declined that
    way gets a plain line, which is a line on the conductor.
    """
    if dim not in width_axes(region.piece_box[0], region.piece_box[1], params.metal_res):
        return None
    plane = region.upper[dim] if at_high else region.lower[dim]
    depths = [
        depth
        for face in region.faces[dim]
        if face.at_high is at_high and abs(face.plane - plane) < FLATNESS
        for depth in face.depths
        if depth > params.metal_res
    ]
    return min(depths) if depths else None


def _cell_holding(depth: float | None, dim: int, params: MeshParams, res: float) -> float:
    """``res``, or the coarsest cell holding ``depth`` where that is finer.

    :func:`_edge_size`'s arithmetic without working the depth out, so a caller
    wanting both the depth and the cell asks for the depth once.
    :func:`_edge_size` and :func:`_face_depth` say what the two quantities are.
    """
    if depth is None:
        return res
    # The rule inverted. A face lands on whichever member of its pair is
    # nearer, so it moves by `min(share, 1 - share)` of a cell whichever way it
    # goes - inward while the inner line is the nearer, outward once the outer
    # one is. A width `w` therefore departs by `2 * moved * cell / w`, and the
    # coarsest cell holding that inside `1 - K` is `(1 - K) * w / (2 * moved)`.
    moved = min(params.edge_line_inside, 1.0 - params.edge_line_inside)
    if moved <= 0.0:
        # A line on the face conducts, so the conductor arrives whole and there
        # is nothing to size the cell from.
        return res
    holding = 0.5 / moved * (1.0 - CONDUCTOR_WIDTH_KEPT) * depth
    # A demand the floor would dominate is not a demand. Held to within one
    # graded step of `min_cell`, the field around the face is the floor rather
    # than the size asked for, while the thirds rule still pins a pair that close
    # together. Between them they make a grid that cannot be built, on geometry
    # the user is entitled to mesh. The demand is therefore dropped rather than
    # clamped. The conductor is meshed at the size the policy asked for, and
    # pre-flight reports what that left of it.
    if holding <= params.floor * params.max_ratio[dim]:
        return res
    return min(res, holding)


def _edge_size(region: Conductor, dim: int, at_high: bool, params: MeshParams) -> float:
    """The cell the grid is built from at one of ``region``'s faces on ``dim``.

    The policy's own size for metal, except at a face with metal behind it,
    where it is instead the coarsest cell that still leaves
    :data:`CONDUCTOR_WIDTH_KEPT` of that depth conducting.

    That second case is the only control the grid has over how much of a
    conductor survives. openEMS rounds each face to the nearer of the pair
    straddling it, the pair is registered a share of a cell inside the metal, and
    both of those hold whatever the policy asked for. A width is therefore held
    by sizing the cell from the width, and by nothing else. The policy's size
    resolves the field singularity at an edge, which is a different quantity, so
    the two are combined by taking the finer rather than one replacing the other.

    Asked per face rather than per region, and the depth is the metal's rather
    than this box's. The translation cuts a drawn outline into rectangles by
    itself, and two boxes of one piece against one plane are one face. See
    :func:`_face_depth`.

    The thickness is left at the policy's size, because the mesher deliberately
    lays a single cell through a foil. See :func:`width_axes`.

    Both the thirds rule and the edge constraint are built from this, and they
    have to be. They place lines around one edge between them, and a disagreement
    about the size puts them at cross purposes. Neither passes ``relaxed_to`` in.
    That coarsens what a region asks for, and where the outer line falls is a
    question about the drawing. See :func:`_edge_to_resolve`.
    """
    res = params.resolution(region.material)
    return _cell_holding(_face_depth(region, dim, at_high, params), dim, params, res)


def _touching(one: Conductor, other: Conductor) -> bool:
    """Whether two conductors are the same piece of metal.

    Read only to decide whether one box's demand at a face may stand for
    another's, which it may exactly when the two are the same metal.

    Read from :attr:`Conductor.piece`, worked out by walking chains, and never
    from the conductor boxes. An L's box holds space its metal does not, so a
    pair of unrelated conductors inside one L's box would answer for each other.
    """
    return one.piece == other.piece


def _one_pair_at_each_face(
    candidates: Sequence[tuple[float, float, float, Conductor]],
) -> list[tuple[float, float, float, Conductor]]:
    """``candidates``, one thirds pair per face of a conductor rather than per box.

    A solid reaches the mesher as the boxes it was cut into, and every box
    against a plane asks about it. They all ask for the same cell, because the
    depth is read off the piece's own table of faces rather than off each box's
    run, so only one of them has to be kept.

    Kept per face, which is a plane and a side. Two blocks sharing a plane, one
    below it and one beside it above, have an edge each at that coordinate
    looking opposite ways, and the void one of them faces is where the other's
    metal is. Those are two edges, and keeping one would leave the other
    unresolved.

    Kept per piece of metal, because a demand is only answerable by lines the
    same metal asked for. An unrelated object with a face at the same coordinate
    would otherwise stand in for this one.

    Coordinates are gathered at :data:`FLATNESS`, and against the pair already
    held rather than against a running position. Two boxes cut apart and measured
    separately do not name one plane with the same bits, and a comparison that
    moved with each match would let a cluster walk.

    The pairs come back in the order they arrived in, which decides which box a
    coincident anchor is named after.
    """
    kept: list[int] = []
    for n, (edge, inside, outside, region) in enumerate(candidates):
        side = outside > inside
        if any(
            (candidates[held][2] > candidates[held][1]) == side
            and abs(candidates[held][0] - edge) < FLATNESS
            and _touching(candidates[held][3], region)
            for held in kept
        ):
            continue
        kept.append(n)
    return [candidates[n] for n in kept]


def edge_lines(edge: float, outward: float, cell: float, inside: float) -> tuple[float, float]:
    """The pair of lines the mesher puts about a conductor face, inner one first.

    ``outward`` is the direction the void lies in, ``-1.0`` or ``+1.0``. The two
    are one cell apart and carry the face between them, the inner one ``inside``
    of a cell within the metal. An ``inside`` of nothing therefore puts a line on
    the face and the other a whole cell clear of it.

    Where the lines go and how much metal that leaves are one question, so
    everything asking either comes here: the pinning, the demand inverted from it
    in :func:`_edge_size`, and the probe :func:`_edge_to_resolve` sends out to
    ask whether the outer line has room to land at all.
    """
    within = edge - outward * inside * cell
    return within, within + outward * cell


@dataclass(frozen=True)
class _EdgePair:
    """The pair of lines one conductor face would be meshed with on one axis.

    The pinning and the edge demand both put this question, and they have to put
    the same one: they place lines about a single edge between them, and a
    disagreement about the cell or about whether the face is an edge at all puts
    them at cross purposes. So the question has a name, and each caller reads
    what it needs off the answer.

    ``resolve`` is whether the thirds rule applies here at all - see
    :func:`_edge_to_resolve`. It is the metal's own answer. A caller may hold the
    face to more than that, and :func:`~.mesh._fixed_positions` does.
    """

    #: Where the face is.
    position: float
    #: The cell the grid is built from at this face. See :func:`_edge_size`.
    cell: float
    #: The line of the pair inside the metal.
    inside: float
    #: The line of the pair outside it.
    outside: float
    #: Whether this face is an isolated edge, with a singularity on it.
    resolve: bool


def _edge_pair(
    region: Conductor,
    grouped: Grouped,
    dim: int,
    at_high: bool,
    params: MeshParams,
    domain_lower: float,
    domain_upper: float,
) -> _EdgePair:
    """What the thirds rule would do at one face of ``region`` on ``dim``."""
    outward = _OUTWARD[at_high]
    position = region.upper[dim] if at_high else region.lower[dim]
    cell = _edge_size(region, dim, at_high, params)
    inside, outside = edge_lines(position, outward, cell, params.edge_line_inside)
    resolve = _edge_to_resolve(region, grouped, dim, at_high, outside, domain_lower, domain_upper)
    return _EdgePair(position, cell, inside, outside, resolve)
