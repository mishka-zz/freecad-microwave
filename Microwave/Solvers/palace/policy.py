# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the run is, as against what the device is, and what the mesher is asked for.

The study says what band a device is characterised over, what the device asks of
any mesh, and what this pipeline lays that as. This module turns those into what
the layers below the adapter take: the sweep Palace is given, and the demand and
the profile the tetrahedral mesher is given.

It also refuses, by name, what the study asks for and this backend cannot do,
and it states what an open study's outer surface does to a radiated wave before
the run starts. There is no absorbing layer here. An ``Open`` face is met by air
the adapter reserves round the structure, with an absorbing condition on its
outer sides, and that condition reflects a share of a wave the structure
radiates which the distance, the band and the angle the wave meets it at decide.

The profile is not read off anything. It has no unit a user could check and
nothing in it is a preference, so it is this adapter's own statement of what
kind of mesh a finite element method in three dimensions needs.

Every property is reached by duck typing, so this module imports no FreeCAD and
is testable against plain attribute bags.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import Any

from ... import units
from ...Gmsh.vocabulary import AtRim, Demand, Edges, Mesh, Near, Place, Profile, Within
from ...portbox import AXIS_NAMES, FLATNESS
from ..errors import TranslationError
from ..medium import Medium
from ..medium import said as medium_said
from ..properties import label, value
from .config import Sweep
from .problem import (
    FLOORED,
    SETTLED,
    UNREFINED,
    Coarsening,
    Conductor,
    CountAcross,
    Creases,
    Floored,
    Joined,
    Narrow,
    Narrowed,
    Refinement,
    RegionCount,
    Relaxed,
    Reserved,
    Unfilled,
    Unlaid,
    Unrefining,
    Unreserved,
    Unwalled,
)

__all__ = [
    "ACROSS",
    "OPEN_FACE",
    "DISCRETE",
    "FACES",
    "FLOOR",
    "PROFILE",
    "SWEEPS",
    "THROUGH",
    "band",
    "demand",
    "element_size",
    "floor",
    "joining",
    "one_metal",
    "reserving",
    "opens",
    "percent",
    "reflection",
    "refusals",
    "slanted",
    "steepest",
    "said",
    "sweeping",
    "unlaid",
]

#: The two faces of the domain each axis has, spelling the halves of the
#: property names the mesh policy carries.
SIDES = ("Min", "Max")

#: Every face of the domain, spelt the way a per-face property name spells it.
FACES = tuple(f"{axis}{side}" for axis in AXIS_NAMES for side in SIDES)

#: The padding values. ``Open`` says there is free space beyond the structure on
#: that face, and a study saying it on any face is open. ``Through`` says the
#: structure runs out through the absorber, and ``PEC Wall`` that the domain ends
#: where the structure does.
#:
#: Stated again here rather than imported, because ``Objects/mesh.py`` sits
#: behind ``import FreeCAD``. A test holds the copies together.
OPEN_FACE = "Open"
THROUGH = "Through"
PEC_WALL_FACE = "PEC Wall"

#: How a sweep reaches the points across the band, as the solver object offers
#: them. Stated again here for the reason the padding words above are.
SWEEPS = ("Adaptive", "Discrete")
DISCRETE = "Discrete"

#: What kind of mesh this backend needs.
#:
#: Three dimensions because the field lives in a volume. Element order above one
#: with the nodes on the drawn surface, because a straight-sided element meets a
#: curved boundary along the chord between its corners and no number of them
#: removes that: the error is the surface's, not the mesh's. The older mesh
#: revision, because it is the one the backend's mesh reader takes. And one
#: region, because a part no face joins to the rest is closed off by the wall
#: where it ends: no wave crosses from one part to another, so a port sees
#: nothing of any part but its own, and a part holding no port is a sealed
#: cavity the run never reaches. Nothing in the run says so.
PROFILE = Profile(top=3, element_order=2, curved=True, written="msh22", connected=True)

#: How many times finer than the size in the slowest region the floor under
#: every element is, where ``MinElementSize`` is zero. The size from curvature
#: grows finer without bound where the curvature does, at the apex of a cone and
#: along the edge of a knife, and the floor ends it there. It stands far enough
#: below the size everywhere to leave a round wire, a pin or a via a user draws
#: its elements. Where a curve is tighter than the floor allows, the mesher
#: names the labels holding the elements it turned inside out.
FLOOR = 100.0

#: How many elements are laid between two edges of a sheet of metal that face
#: each other across it, or across a slot in it, where the size at its edges
#: would lay fewer. The field is singular along both edges, and a strip's
#: impedance is set by the field across its width, so a strip that the size at
#: its edges spans in one element is a line the mesh does not resolve. The
#: published low-pass gate on Palace holds the count: it doubles it and asks that
#: the corner move less than the measurement is read to. A lower order leaves
#: more to the count, and a Mesh Refinement at the edges lays a finer size where
#: wanted.
ACROSS = 4


def band(analysis: Any) -> Sweep:
    """The band the study is characterised over, and how many points across it.

    Each point is a full solve here unless the sweep is adaptive, which the
    solver decides, not the study. What is refused here is a count that
    describes no sweep at all.
    """
    start = value(analysis.FrequencyStart)
    stop = value(analysis.FrequencyStop)
    points = int(analysis.FrequencyPoints)
    try:
        return Sweep(start=start, stop=stop, points=points)
    except ValueError as error:
        raise TranslationError(f"{label(analysis)!r}: {error}") from error


def sweeping(sweep: Sweep) -> str:
    """How the band is swept, in one sentence with no full stop."""
    if sweep.samples == 1:
        return "The one frequency is solved in full"
    if sweep.adaptive is None:
        return f"Each of the {sweep.samples} points is solved in full"
    return (
        f"The {sweep.samples} points are answered by a reduced model built to a field "
        f"error of {sweep.adaptive.tolerance:g} from at most {sweep.adaptive.solves} full "
        "solves for each driven port, and checked against full solves at up to two points"
    )


def opens(settings: Any) -> tuple[str, ...]:
    """The faces the mesh policy opens to free space, by ``{axis}{side}``. A study
    is open where there is one."""
    return tuple(face for face in FACES if str(getattr(settings, f"Boundary{face}")) == OPEN_FACE)


def floor(recipe: Any, bulk: float) -> float:
    """The floor under every element, in millimetres: ``MinElementSize`` on the
    recipe, or :data:`FLOOR` times finer than ``bulk``, the size in the slowest
    region, where that is zero."""
    return value(recipe.MinElementSize) or bulk / FLOOR


def element_size(per_wavelength: float, stop: float, slowing: float) -> float:
    """The element size ``per_wavelength`` elements across a wavelength at
    ``stop`` make, in a medium slowing a wave by the root of ``slowing``, in
    millimetres."""
    return units.SPEED_OF_LIGHT / stop / math.sqrt(slowing) * units.MM_PER_M / per_wavelength


def refusals(
    settings: Any,
    recipe: Any,
    stop: float,
    slowest: float,
    bounded: Sequence[str] = (),
    reserving: bool = False,
) -> float:
    """Everything the two objects state that this backend refuses, and the count
    of elements across a wavelength once none of it is wrong.

    Called twice on the way to a run. :func:`demand` calls it before it sizes
    anything, and in an open study the translation calls it before it asks the
    CAD kernel for the box the study reserves - so a policy this backend will not honour is
    refused by name rather than reaching the kernel, which answers a box it was
    given no room for with a message about a box.

    :param bounded: the faces of the domain a port's own plane bounds. The
        translation knows fewer of them before the box exists than after, so it
        passes every face a port's plane may bound, and the call from
        :func:`demand` passes the ones it does.
    """
    asked_on = label(recipe)
    per_wavelength = value(recipe.ElementsPerWavelength)
    if per_wavelength <= 0:
        raise TranslationError(
            f"{asked_on!r}: ElementsPerWavelength is {per_wavelength:.4g}. Enter a value above 0"
        )
    bulk = element_size(per_wavelength, stop, slowest)
    finest = value(recipe.MinElementSize)
    if finest < 0:
        raise TranslationError(
            f"{asked_on!r}: MinElementSize is {finest:.4g} mm. Enter 0 to derive it, or a "
            "positive length"
        )
    # Raising ElementsPerWavelength lowers the largest element, so only a lower
    # MinElementSize lifts this.
    if finest and finest >= bulk:
        raise TranslationError(
            f"{asked_on!r}: MinElementSize is {finest:.4g} mm, at or above the {bulk:.4g} mm "
            "largest element the band and ElementsPerWavelength allow, so every element "
            "would be the largest one. Lower MinElementSize"
        )
    per_turn = int(recipe.ElementsPerTurn)
    if per_turn < 0:
        raise TranslationError(
            f"{asked_on!r}: ElementsPerTurn is {per_turn}. Enter 0 for none, or a count of "
            "elements round a full turn of a curved surface"
        )
    _check_through_is_bounded(settings, bounded, reserving)
    _check_the_surface_is_not_held_to_a_distance(settings)
    return per_wavelength


def demand(
    settings: Any,
    recipe: Any,
    stop: float,
    slowest: float,
    rims: Sequence[str] = (),
    refinements: Sequence[Refinement] = (),
    bounded: Sequence[str] = (),
    within: Sequence[tuple[str, float]] = (),
    reserving: bool = False,
    coarsened: Mapping[str, float] = MappingProxyType({}),
    medium: float = 1.0,
    wall: str = "",
    mirrors: Sequence[str] = (),
    narrow: Sequence[Narrow] = (),
) -> Demand:
    """The element size asked for, in millimetres, from both objects that say.

    The Gmsh mesh states the element. ``ElementsPerWavelength`` counts elements
    across the wavelength in the slowest material at the top of the band, and
    never a length: a remembered millimetre under-resolves the moment anything
    raises the permittivity or the frequency. A wave slows by the square root of
    the product of permittivity and permeability, and meshing to the vacuum
    wavelength under-resolves a material by exactly that factor.

    ``ElementsPerTurn`` sizes each curved surface by how sharply it bends,
    wherever that asks for finer elements than the rest: a second-order element
    follows a curved surface only across a limited arc, and one laid across a
    tighter curve is turned inside out.

    The mesh policy states what the device asks whatever lays it.
    ``MinElementSize`` is a floor against a sliver a boolean left behind, and
    against the size from curvature where the curvature grows without bound. A
    floor above the ceiling is taken silently and meshed to the ceiling, so it
    is refused instead. Zero derives the floor, :data:`FLOOR` times finer than
    the size in the slowest region.

    ``EdgeRefinement`` sizes each edge of metal the field is singular along,
    where the bulk size leaves the impedance of a line wrong: an edge of a sheet
    or a body in ``rims`` the room turns round past half a turn. The field at the
    edge of a perfectly conducting wedge the room opens round by an angle goes as
    the distance to the power of half a turn over that angle, less one, which is
    singular past half a turn. So a
    sheet's free edge, a step and an iris's edge are refined, and a corner the
    room stands in, a flat seam and metal meeting metal are not. A wedge ending
    on a magnetic wall, one of the ``mirrors``, is the room and its reflection,
    and opens twice as wide. Each refinement region set to refine, and finer
    than the bulk, sizes each thing it names where that thing is.
    Where two edges of a sheet face each other across it, or across a slot in
    it, closer than :data:`ACROSS` times the size at the rim, each of the two is
    laid at their distance apart over :data:`ACROSS`, each of the ``narrow``.
    ``MaxGrowthRatio`` is how fast either grows back to the bulk size.

    :param settings: the mesh policy.
    :param recipe: the Gmsh mesh.
    :param rims: the labels of the metal whose rims are refined.
    :param refinements: the enabled refinement regions.
    A study that reserves room round the structure asks for a size per region.
    The reserved air is the study's medium, so the size everywhere is the
    medium's, and each drawn region slower than the medium is laid at its own
    size throughout. The rims and the floor are held to the slowest region's
    size, as in a study that reserves nothing, where that is the size
    everywhere.

    :param bounded: the faces of the domain a port's own plane bounds, by
        ``{axis}{side}``. See :func:`_check_through_is_bounded`.
    :param within: for a study that reserves room, each drawn region's label and
        the product a wave slows by in it, and empty for one that does not.
    :param reserving: whether the study reserves room round the structure, which
        decides the size everywhere and what a ``Through`` face may be honoured
        by.
    :param coarsened: per label of metal in ``rims`` whose bindings a region set
        to coarsen names whole, the size it settles for.
    :param medium: the product a wave slows by the root of in the medium the
        reserved air is.
    :param wall: the label of the rest of the boundary, a perfect conductor that
        ends the room as metal does. It is the sides of the box the study
        reserves or its drawing fills, so no crease of it is one the room turns
        round past half a turn, and no size is asked there.
    :param mirrors: the labels of the faces that are magnetic walls where the
        model ends.
    """
    asked_on = label(recipe)
    per_wavelength = refusals(settings, recipe, stop, slowest, bounded, reserving=reserving)

    def size(slowing: float) -> float:
        return element_size(per_wavelength, stop, slowing)

    bulk = size(slowest)
    coarsest = size(medium) if reserving else bulk
    finest = floor(recipe, bulk)
    per_turn = int(recipe.ElementsPerTurn)
    places = [
        *_at_rims(recipe, bulk, finest, rims, coarsened),
        *(
            Within(name=name, label=name, size=size(slowing))
            for name, slowing in within
            if size(slowing) < coarsest
        ),
        *_at_marks(coarsest, finest, refinements),
        *_across(recipe, bulk, finest, narrow, coarsened),
    ]
    if not places:
        return Demand(coarsest=coarsest, finest=finest, per_turn=per_turn)
    growth = value(recipe.MaxGrowthRatio)
    if not (math.isfinite(growth) and growth > 1.0):
        raise TranslationError(
            f"{asked_on!r}: MaxGrowthRatio is {growth:.4g}, so a finer size at an edge or in "
            "a refinement region never grows back to the bulk size. Raise MaxGrowthRatio "
            "above 1"
        )
    return Demand(
        coarsest=coarsest,
        finest=finest,
        growth=growth,
        places=tuple(places),
        walls=(*rims, *([wall] if wall else [])),
        mirrors=tuple(mirrors),
        per_turn=per_turn,
    )


def _at_rims(
    recipe: Any,
    coarsest: float,
    finest: float,
    rims: Sequence[str],
    coarsened: Mapping[str, float] = MappingProxyType({}),
) -> list[Place]:
    """One place at the rim of each label of metal, ``EdgeRefinement`` times finer
    than the bulk, and never below ``MinElementSize``, laid only where the room
    turns round an edge past half a turn - see :func:`demand`.

    A refinement of one asks for the bulk size, which is no place at all, and
    one below it asks for edges coarser than open space - which the other backend
    refuses as well. A rim finer than the floor is laid at the floor,
    which is as fine as the policy lets any element be, and :func:`unlaid`
    states it.

    A rim a coarsening names whole settles for the coarsening's size where that
    is coarser, as a body the other backend relaxes asks for no finer than it,
    and a rim settling at or above the bulk is no place at all.
    """
    subject = label(recipe)
    refinement = value(recipe.EdgeRefinement)
    # Edges carry the field singularity that sets a line's impedance, so they
    # need the finer mesh and never the coarser one.
    if not refinement >= 1:
        raise TranslationError(
            f"{subject!r}: EdgeRefinement is {refinement:g}, so conductor edges are meshed "
            "more coarsely than open space. Enter 1 or more, where 1 means no refinement"
        )
    if refinement == 1 or not rims:
        return []
    size = max(coarsest / refinement, finest)
    places: list[Place] = []
    for rim in rims:
        settled = max(size, coarsened.get(rim, 0.0))
        if settled < coarsest:
            places.append(AtRim(name=rim, label=rim, size=settled, reentrant=True))
    return places


def _across(
    recipe: Any,
    bulk: float,
    finest: float,
    narrow: Sequence[Narrow],
    coarsened: Mapping[str, float] = MappingProxyType({}),
) -> list[Place]:
    """One place at each edge of a sheet of metal facing another closer than
    :data:`ACROSS` times the size at the rim, at their distance apart over
    :data:`ACROSS`, and never below ``MinElementSize``.

    It is a refinement of the edges of metal, so a refinement of one lays none of
    it, as it lays no rim. A coarsening that names a metal whole settles these
    edges as it settles its rim: none is laid finer than the size it settles
    for, and where that is no finer than the rim, none is laid.
    """
    refinement = value(recipe.EdgeRefinement)
    if not refinement > 1:
        return []
    rim = max(bulk / refinement, finest)
    places: list[Place] = []
    for one in narrow:
        settled = coarsened.get(one.rim, 0.0)
        size = max(one.apart / ACROSS, finest, settled)
        if size < rim:
            places.append(Near(name=one.mark.name, label=one.mark.name, size=size))
    return places


def _at_marks(coarsest: float, finest: float, refinements: Sequence[Refinement]) -> list[Place]:
    """One place at each thing a refinement region set to refine names.

    A tetrahedral mesh sizes a shape where it is: along a face, an edge or a
    point and growing away from it, or throughout a body and growing away from
    its faces. A box round the shape would size the space round it as well,
    which round a ring is the whole of its inside.

    A size at or above the bulk is met everywhere already, and a size field
    refines and never coarsens, so no place is laid for it. A size below
    ``MinElementSize`` is laid at the floor. :func:`unlaid` states both.
    """
    places: list[Place] = []
    for region in refinements:
        if region.coarsens or region.size >= coarsest:
            continue
        places.extend(
            Near(name=mark.name, label=mark.name, size=max(region.size, finest))
            for mark in region.marks
        )
    return places


def unlaid(
    settings: Any,
    recipe: Any,
    asked: Demand,
    regions: Sequence[str],
    refinements: Sequence[Refinement],
    rims: Sequence[str],
    wall: str,
    bulk: float | None = None,
    relaxed: Sequence[Relaxed] = (),
    opened: bool = False,
    unfilled: str = "",
    narrow: Sequence[Narrow] = (),
    bodies: Sequence[tuple[str, float]] = (),
) -> tuple[Unlaid, ...]:
    """What the document states about element sizes that this backend does not lay
    as stated.

    Nothing here is refused. A count across a body and a coarsening each have
    an expression on a rectilinear grid and none here: an element size on
    tetrahedra is the same along every axis, and a size field refines and never
    coarsens. A region no finer than the bulk asks for what is met everywhere,
    and a size below ``MinElementSize`` is laid at the floor. Each changes how
    finely the model is meshed and not which model it is, so each is said once
    the mesh can answer it with a figure.

    :param asked: the demand :func:`demand` made of the same two objects.
    :param bulk: the size in the slowest region, which a rim is refined from,
        where it is not the size everywhere: in a study that reserves air.
        ``None`` says it reserves none.
    :param relaxed: each rim a coarsening names whole, stated with the size a
        rim is otherwise laid at.
    :param opened: whether a face of the study is open to free space, which is
        what ``Clearance`` measures from. A closed study lays none of it.
    :param unfilled: the medium's name where the study links one and its drawing
        leaves it no room, and empty otherwise.
    :param narrow: each edge of a sheet of metal facing another closer than
        :data:`ACROSS` times the size at the rim, or the size in the slowest
        region where no rim is laid.
    :param bodies: each label of metal drawn as bodies, with the middle side of
        the box round it.
    """
    subject = label(settings)
    asked_on = label(recipe)
    found: list[Unlaid] = []
    stated = value(settings.Clearance)
    if not opened and stated > 0:
        found.append(Unreserved(subject=subject, clearance=stated))
    if unfilled:
        found.append(Unfilled(subject=subject, medium=unfilled))
    count = int(recipe.MinElementsAcross)
    if count > 0 and regions:
        found.append(CountAcross(subject=asked_on, count=count, labels=tuple(regions)))
    refinement = value(recipe.EdgeRefinement)
    edge = (asked.coarsest if bulk is None else bulk) / refinement
    laid = {place.name for place in asked.places}
    if refinement > 1:
        away = {one.rim for one in relaxed if one.rim not in laid}
        found.append(
            Creases(
                subject=asked_on,
                refinement=refinement,
                rims=tuple(rim for rim in rims if rim not in away),
                wall=wall,
                size=edge,
                floor=asked.finest,
                coarsened=tuple(rim for rim in rims if rim in away),
            )
        )
    found.extend(
        replace(one, edge=max(edge, asked.finest) if refinement > 1 else None) for one in relaxed
    )
    narrowed = _narrowed_record(asked_on, refinement, edge, asked, narrow, bodies, relaxed)
    if narrowed is not None:
        found.append(narrowed)
    for region in refinements:
        if region.coarsens:
            if region.part or region.bodies:
                found.append(
                    Coarsening(
                        subject=region.label,
                        size=region.size,
                        part=region.part,
                        bodies=region.bodies,
                        joined=region.joined,
                    )
                )
            continue
        if region.size >= asked.coarsest:
            found.append(
                Unrefining(
                    subject=region.label,
                    size=region.size,
                    recipe=asked_on,
                    coarsest=asked.coarsest,
                    count=region.across,
                )
            )
            continue
        if region.size < asked.finest:
            found.append(
                Floored(
                    subject=region.label,
                    size=region.size,
                    settings=asked_on,
                    floor=asked.finest,
                    places=tuple(mark.name for mark in region.marks),
                )
            )
        if region.across > 0:
            found.append(
                RegionCount(
                    subject=region.label,
                    count=region.across,
                    places=tuple((mark.name, mark.thinnest) for mark in region.marks),
                )
            )
    return tuple(found)


def _narrowed_record(
    subject: str,
    refinement: float,
    edge: float,
    asked: Demand,
    narrow: Sequence[Narrow],
    bodies: Sequence[tuple[str, float]],
    relaxed: Sequence[Relaxed],
) -> Narrowed | None:
    """Where the count across a narrow sheet is laid, where it is not and why,
    and the metal drawn as bodies narrow enough to want it, or nothing where none
    of those is."""
    rim = max(edge, asked.finest) if refinement > 1 else edge
    sizes = {place.name: place.size for place in asked.places}
    settled = {one.rim: one.size for one in relaxed}
    least: dict[tuple[str, str], tuple[float, float]] = {}
    for one in narrow:
        if one.mark.name in sizes:
            continue
        if refinement <= 1:
            why, size = UNREFINED, rim
        elif one.rim in settled:
            why, size = SETTLED, max(settled[one.rim], rim)
        else:
            why, size = FLOORED, rim
        if one.apart >= ACROSS * size:
            continue
        key = (one.drawn, why)
        if key not in least or one.apart < least[key][0]:
            least[key] = (one.apart, size)
    narrowed = Narrowed(
        subject=subject,
        refinement=refinement,
        across=ACROSS,
        rim=rim,
        laid=tuple(
            (one.mark.name, one.apart, sizes[one.mark.name])
            for one in narrow
            if one.mark.name in sizes
        ),
        unlaid=tuple((drawn, apart, size, why) for (drawn, why), (apart, size) in least.items()),
        settled=tuple(
            (rim_label, size)
            for rim_label, size in settled.items()
            if any(one.rim == rim_label for one in narrow)
        )
        if refinement > 1
        else (),
        bodies=tuple((name, side) for name, side in bodies if side < ACROSS * rim),
    )
    if not (narrowed.laid or narrowed.unlaid or narrowed.bodies):
        return None
    return narrowed


def said(records: Sequence[Unlaid], mesh: Mesh) -> list[str]:
    """Each size the document states and this backend does not lay as stated,
    with the figure the mesh answers it with."""
    lines = []
    for record in records:
        if isinstance(record, CountAcross):
            lines.extend(_count_across(record, mesh))
        elif isinstance(record, RegionCount):
            lines.extend(_region_count(record, mesh))
        elif isinstance(record, Coarsening):
            lines.append(_coarsening(record, mesh))
        elif isinstance(record, Relaxed):
            lines.append(_relaxed(record, mesh))
        elif isinstance(record, Unrefining):
            lines.append(_unrefining(record, mesh))
        elif isinstance(record, Floored):
            lines.append(_floored(record, mesh))
        elif isinstance(record, Unreserved):
            lines.append(
                f"{record.subject!r}: Clearance {record.clearance:.4g} mm has no effect, since "
                f"no Boundary is {OPEN_FACE}: every face of the domain is a perfect wall flush "
                f"with the structure. Set a Boundary to {OPEN_FACE} to reserve that space"
            )
        elif isinstance(record, Narrowed):
            lines.extend(_narrowed(record, mesh))
        elif isinstance(record, Unfilled):
            lines.append(
                f"{record.subject!r}: Medium {record.medium!r} fills nothing, since every room "
                "in the domain is drawn and bound or closed off by metal"
            )
        else:
            lines.extend(_creases(record, mesh))
    return lines


def _coarsening(record: Coarsening, mesh: Mesh) -> str:
    """A coarsening where some of what it names is not coarsened, said with what
    that is and the elements the mesh holds."""
    parts = []
    if record.part:
        named = [
            repr(label)
            + (f" (drawn with {_listed([repr(one) for one in others])})" if others else "")
            for label, others in zip(record.part, record.joined, strict=True)
        ]
        # The rim of metal settles for a coarser size only where every face each
        # binding drawn under its label names is coarsened.
        parts.append(
            f"it names only some faces of {_listed(named)}, and metal is coarsened only "
            "where every face of each of its bindings is named"
        )
    if record.bodies:
        parts.append(
            f"it names what {_listed([repr(one) for one in record.bodies])} is made of, "
            "which is meshed at the size everywhere"
        )
    return (
        f"{record.subject!r}: Mode Coarsen at {record.size:.4g} mm has no effect where "
        f"{'; and where '.join(parts)}, so what it names there is meshed as though the "
        f"region were absent. The elements run {_run(mesh.edges)}"
    )


def _relaxed(record: Relaxed, mesh: Mesh) -> str:
    """Metal a coarsening names whole, said with the size it settled for, how long
    its edges and the rim laid along them are, and how near they come to another
    conductor.

    The other backend keeps a gap between coarsened metal and another conductor
    at the gap's own size. This one lays the rim's size along the whole rim, so
    the nearest approach is what says whether that gap was resolved. The edges
    are the drawing's and the rim is the mesh's: the edges the room turns round
    past half a turn.
    """
    subject = f"{record.asked_by!r}: Mode Coarsen at {record.size:.4g} mm"
    reached = mesh.reached.get(record.rim)
    if not record.length:
        return f"{subject} lays nothing at {record.rim!r}, which has no edge but a seam"
    if record.edge is None:
        said = (
            f"{subject} lays nothing at the rim of {record.rim!r}, which EdgeRefinement "
            "leaves at the size everywhere"
        )
    elif reached is None:
        said = (
            f"{subject} lays no size at the rim of {record.rim!r} rather than "
            f"{record.edge:.4g} mm, since that is no finer than the size everywhere"
        )
    elif not reached.laid:
        return (
            f"{subject} lays nothing at {record.rim!r}, since none of its edges is a free or "
            "re-entrant edge"
        )
    elif reached.asked > record.edge:
        said = (
            f"{subject} lays the rim of {record.rim!r} at {reached.asked:.4g} mm rather "
            f"than {record.edge:.4g} mm"
        )
    else:
        said = (
            f"{subject} leaves the rim of {record.rim!r} at {record.edge:.4g} mm, which is "
            "coarser already"
        )
    if record.refined:
        said += (
            f". {_listed([f'{name!r} at {size:.4g} mm' for name, size in record.refined])} "
            f"{'refines' if len(record.refined) == 1 else 'refine'} some of the same metal "
            "finer, and the finer size holds where it is laid"
        )
    said += f". Its edges are {record.length:.4g} mm long"
    if reached is not None and reached.laid and reached.extent:
        said += f", the rim laid along them {reached.extent:.4g} mm"
    if record.nearest is not None:
        other, gap = record.nearest
        if gap <= FLATNESS:
            said += f", and they touch {other!r}"
        else:
            said += (
                f", and they come within {gap:.4g} mm of {other!r}. A gap between them is "
                "meshed at the rim's size where the rim is laid, and openEMS keeps it at its "
                "own size"
            )
    return said


def _run(edges: Edges) -> str:
    return f"from {edges.shortest:.4g} mm to {edges.longest:.4g} mm"


def _count_across(record: CountAcross, mesh: Mesh) -> list[str]:
    """A line for each region the mesh settings' count is asked of."""
    lines = []
    for name in record.labels:
        held = mesh.labels.get(name)
        if held is None:
            continue
        edges = held.edges or mesh.edges
        # An element size here is the same along every axis, so a count across
        # the body's thinnest extent would fill the whole body at that size.
        line = (
            f"{record.subject!r}: MinElementsAcross {record.count} is not laid in {name!r}, "
            f"since a Palace element is the same size along every axis. Its elements run "
            f"{_run(edges)}"
        )
        if held.lower is not None and held.upper is not None:
            side = min(high - low for low, high in zip(held.lower, held.upper, strict=True))
            line += f", and the smallest side of the box round it is {side:.4g} mm"
        lines.append(
            line + ". A Mesh Refinement on the body lays its ElementSize throughout the body"
        )
    return lines


def _measured(name: str, mesh: Mesh) -> str:
    """What the mesh holds at one thing a region names: throughout a body, and
    along anything else."""
    reached = mesh.reached.get(name)
    if reached is None or reached.reached is None:
        return f"no element is measured at {name!r}"
    if reached.dimension == PROFILE.top:
        return (
            f"the median mean edge of the {reached.elements} elements in {name!r} is "
            f"{reached.reached:.4g} mm"
        )
    if reached.along is None:
        return f"the edges leaving {name!r} reach {reached.reached:.4g} mm"
    return f"the median longest edge of the elements along {name!r} is {reached.along:.4g} mm"


def _region_count(record: RegionCount, mesh: Mesh) -> list[str]:
    """A line for each thing a refinement region's count is asked of."""
    return [
        f"{record.subject!r}: MinElementsAcross {record.count} is not laid, since a Palace "
        "element is the same size along every axis, and its ElementSize is laid instead: "
        f"{_measured(name, mesh)}, and the smallest side of the box round it is {side:.4g} mm"
        for name, side in record.places
    ]


def _unrefining(record: Unrefining, mesh: Mesh) -> str:
    """A refinement region no finer than the size everywhere, said with the elements
    the mesh holds."""
    line = (
        f"{record.subject!r}: ElementSize {record.size:.4g} mm is not laid, since "
        f"ElementsPerWavelength on {record.recipe!r} already asks for "
        f"{record.coarsest:.4g} mm everywhere and a refinement never coarsens. The "
        f"elements run {_run(mesh.edges)}"
    )
    if record.count:
        line += f". Its MinElementsAcross {record.count} is not laid either"
    return line


def _floored(record: Floored, mesh: Mesh) -> str:
    """A refinement region finer than the floor, said with what the mesh holds at
    each thing it names."""
    return (
        f"{record.subject!r}: ElementSize {record.size:.4g} mm is laid at "
        f"{record.floor:.4g} mm, the floor MinElementSize on {record.settings!r} sets, or "
        "derives where it is 0: "
        f"{'; '.join(_measured(name, mesh) for name in record.places)}"
    )


def _narrowed(record: Narrowed, mesh: Mesh) -> list[str]:
    """A line for the edges laid at the count across, one for each reason others
    carry no finer size, and one for each body of metal as narrow."""
    lines = []
    # The edge an edge faces may be its own reflection in a lumped port's
    # plane, which is a magnetic wall.
    if record.laid:
        lines.append(
            f"{record.subject!r}: EdgeRefinement {record.refinement:g} lays "
            f"{record.rim:.4g} mm at the edges of metal, which leaves fewer than "
            f"{record.across} elements across a sheet or a gap in it narrower than "
            f"{record.across * record.rim:.4g} mm. Each such edge is laid at its distance "
            f"to the edge it faces over {record.across}, or at the floor or a coarsening's "
            "size where that is coarser: "
            + "; ".join(
                f"{name!r} stands {apart:.4g} mm from the edge it faces and is laid at "
                f"{size:.4g} mm, and {_measured(name, mesh)}"
                for name, apart, size in record.laid
            )
        )
    across = (
        f"lay a Mesh Refinement at those edges with an ElementSize of their distance apart "
        f"over {record.across}"
    )
    because = {
        UNREFINED: (
            f"EdgeRefinement {record.refinement:g} lays no size at the edges of metal",
            f"Set EdgeRefinement above 1, or {across}",
        ),
        SETTLED: (
            "a coarsening settles "
            + ", ".join(f"{rim!r} at {size:.4g} mm" for rim, size in record.settled),
            across[0].upper() + across[1:],
        ),
        FLOORED: (
            "the edges of metal are laid at the floor MinElementSize sets, or derives "
            "where it is 0",
            "Set MinElementSize in the Palace mesh settings below each distance over "
            f"{record.across}",
        ),
    }
    for why, (said, remedy) in because.items():
        held = [
            (drawn, apart, size) for drawn, apart, size, reason in record.unlaid if reason == why
        ]
        if not held:
            continue
        lines.append(
            f"{record.subject!r}: fewer than {record.across} elements lie across a sheet or "
            f"a gap in it, since {said}: "
            + "; ".join(
                f"{drawn!r} {apart:.4g} mm across, {apart / size:.2g} times {size:.4g} mm"
                for drawn, apart, size in held
            )
            + f". {remedy}"
        )
    lines.extend(
        f"{record.subject!r}: {name!r} is drawn as bodies {side:.4g} mm across, about "
        f"{side / record.rim:.2g} elements at the {record.rim:.4g} mm laid at its edges. The "
        f"{record.across} elements across are laid on sheets only; lay a Mesh Refinement at "
        "the body's edges for a finer size there"
        for name, side in record.bodies
    )
    return lines


def _creases(record: Creases, mesh: Mesh) -> list[str]:
    """Where the refinement at the edges of metal is laid, said where anything
    about it is other than stated: a size laid at the floor, metal the room
    turns round no edge of past half a turn, or no metal at all.

    The mesh's own line for each place says how many edges carry the size and
    how many do not.
    """

    def laid(name: str) -> bool:
        return name not in mesh.reached or mesh.reached[name].laid

    refined = [rim for rim in record.rims if laid(rim)]
    bare = [rim for rim in record.rims if not laid(rim)]
    floored = record.size < record.floor
    if record.rims and not bare and not floored:
        return []
    parts = []
    if floored:
        parts.append(
            f"asks for {record.size:.4g} mm at the edges of metal and is laid at "
            f"{record.floor:.4g} mm, the floor MinElementSize sets, or derives where it is 0"
        )
    if not record.rims and record.coarsened:
        parts.append(
            f"is laid at no binding, since a coarsening settles the edges of "
            f"{', '.join(repr(rim) for rim in record.coarsened)} at the size everywhere"
        )
    elif not record.rims:
        parts.append("is laid at no binding, since no sheet or body is bound to a conductor")
    elif refined:
        parts.append(f"is laid at the edges of {', '.join(repr(rim) for rim in refined)}")
    if bare:
        parts.append(
            f"is laid at no edge of {', '.join(repr(rim) for rim in bare)}, since none of "
            "them is a free or re-entrant edge"
        )
    if not record.rims:
        parts.append(f"the elements run {_run(mesh.edges)}")
    # A free or re-entrant edge is one the room turns round by more than half a
    # turn, where the field is singular.
    return [
        f"{record.subject!r}: EdgeRefinement {record.refinement:g} {', and '.join(parts)}. "
        "It refines only edges where the field is singular: the free edge of a sheet, a "
        "step, the edge of an iris, and not an inside corner, a flat seam or metal meeting "
        "metal. A Mesh Refinement set to Coarsen that names a metal whole at the size "
        "everywhere lays nothing at its edges"
    ]


def _check_through_is_bounded(
    settings: Any, bounded: Sequence[str], reserving: bool = False
) -> None:
    """Refuse a face the structure is said to run out through where nothing ends it
    but the wall.

    There is no perfectly matched layer and no absorber made of cells. ``Through``
    says the structure runs out through the absorber, so the wave meets no end at
    all - and against a perfect wall it meets the hardest end there is.

    ``Through`` is honoured on the faces where it is true here, and the two kinds
    of study honour different ones. In either, a wave port's own plane ends the
    model on that face: the mesher leaves out what stands behind the plane. In a
    study that reserves room a lumped port's plane also ends it, on a side of the
    reserved air the plane lies in - the plane is that whole side, and outside
    the port's elements it is a magnetic wall, so a line's field meets the port's
    resistor there rather than the wall. A study whose drawing fills its box
    reserves no room and has no such side, so a lumped port in one honours
    nothing, and the message says so rather than asking for a port that already
    stands there. ``bounded`` is which faces
    are bounded, and it is read off the study's own ports.

    ``Open`` is not asked here. The adapter reserves free space beyond each
    ``Open`` face, and nothing is read off another pipeline's recipe, so a study
    losing the other pipeline's object does not change which problem this one
    solves.
    """
    through = [
        face
        for face in FACES
        if str(getattr(settings, f"Boundary{face}")) == THROUGH and face not in bounded
    ]
    if not through:
        return
    named = ", ".join(f"Boundary{face}" for face in through)
    # Open first, because it is the one answer the other backend takes as well:
    # openEMS refuses a lumped port lying in a Through face. Open reserves free
    # space beyond the face on both solvers, and in a study that reserves no
    # air it also lets a lumped port's plane in that side end the model.
    remedy = (
        f"Set each to {OPEN_FACE}, lay a lumped port whose elements lie in that side of "
        "the reserved air"
        if reserving
        else f"Set each to {OPEN_FACE}, stand a waveguide port across that face"
    )
    raise TranslationError(
        f"{label(settings)!r}: {named} {'is' if len(through) == 1 else 'are'} {THROUGH}, "
        "and Palace builds no absorber, so the perfect wall where the model ends reflects "
        f"the whole wave there. {remedy}, or set each to {PEC_WALL_FACE} to end the "
        "domain at the structure"
    )


def _check_the_surface_is_not_held_to_a_distance(settings: Any) -> None:
    """Refuse a curve tolerance, which reaches nothing here.

    The other backend sends a curved surface as flat facets and holds them to a
    distance from the drawing. This one meshes the surface itself and represents
    it by where the nodes above an element's corners are placed, which is the
    profile above and is not a length anybody states.
    """
    tolerance = value(settings.CurveTolerance)
    if tolerance:
        raise TranslationError(
            f"{label(settings)!r}: CurveTolerance is {tolerance:.4g} mm, which Palace does "
            "not use, since it meshes curved surfaces directly. Set CurveTolerance to 0"
        )


def reflection(frequency: float, distance: float, slowing: float = 1.0) -> float:
    """What a first-order absorbing condition ``distance`` millimetres from a
    radiator reflects of the dipole mode it radiates at ``frequency``, as a share
    of the wave's amplitude, in a medium slowing a wave by the root of
    ``slowing``.

    On a sphere of radius ``a`` about a dipole the condition's reflection is
    ``1 / sqrt(4 (k a)^4 + 1)``: the condition matches the outgoing wave's
    impedance only in the far field, and the dipole's field carries terms in
    ``1 / (k r)`` that it does not match. It says nothing about a wave meeting a
    flat side at a slant: see :func:`slanted`. See
    ``docs/internals/palace-open-boundary.md``.
    """
    wavenumber = 2.0 * math.pi * frequency * math.sqrt(slowing) / units.SPEED_OF_LIGHT
    ka = wavenumber * distance / units.MM_PER_M
    return 1.0 / math.sqrt(4.0 * ka**4 + 1.0)


def steepest(reserved: Reserved) -> float:
    """The steepest angle from square, in radians, at which a straight line from
    the structure meets an open side.

    Read off the box round the structure and the reserved box alone. A line from
    the near face of the structure's box to the far corner of an open side
    crosses the clearance and, along the side, the most either end of the side
    reaches past the structure's box on the other side of it.
    """
    inner_lower, inner_upper = reserved.structure
    widest = 0.0
    for face in reserved.faces:
        dim = AXIS_NAMES.index(face[0])
        along = math.hypot(
            *(
                max(
                    reserved.upper[other] - inner_lower[other],
                    inner_upper[other] - reserved.lower[other],
                )
                for other in range(len(AXIS_NAMES))
                if other != dim
            )
        )
        widest = max(widest, math.atan2(along, reserved.clearance))
    return widest


def slanted(angle: float) -> float:
    """What a first-order absorbing condition on a flat side reflects of a plane
    wave meeting it ``angle`` radians from square, as a share of the wave's
    amplitude: ``(1 - cos(angle)) / (1 + cos(angle))``, whatever the frequency.

    The condition holds the tangential fields on the side to the impedance of a
    wave meeting it squarely. At a slant the tangential fields of a plane wave
    stand in that impedance times the cosine of the angle, or over it, as the
    wave is polarised, and either mismatch reflects the same share.
    """
    cosine = math.cos(angle)
    return (1.0 - cosine) / (1.0 + cosine)


def percent(share: float, figures: int = 3) -> str:
    """A share as a percentage to ``figures`` significant figures, written out in
    full: a share near one reads ``100%`` rather than ``1e+02%``, and a share
    near the bar is not rounded to nothing."""
    value = share * 100.0
    if value == 0.0 or not math.isfinite(value):
        return f"{value:g}%"
    places = max(figures - 1 - math.floor(math.log10(abs(value))), 0)
    written = f"{value:.{places}f}"
    if "." in written:
        written = written.rstrip("0").rstrip(".")
    return f"{written}%"


def one_metal(conductors: Sequence[Conductor]) -> list[str]:
    """What the run states before it starts of each group of bindings of perfect
    conductor whose bodies meet: the label the mesh holds them under."""
    return [
        f"{_listed([repr(one) for one in (metal.label, *metal.joined)])} bind perfect "
        "conductors whose bodies meet, directly or through one another, so the mesh "
        f"holds them as one metal under {metal.label!r}"
        for metal in conductors
        if metal.joined
    ]


def joining(joined: Sequence[Joined]) -> list[str]:
    """What the run states before it starts of each slip a body of the medium's
    own material takes: where the room is, how thin, and which body it is solved
    as part of."""
    return [
        f"The room from {_point(room.least)} to {_point(room.most)} mm, "
        f"{room.thickness:.3g} mm thick between "
        f"{_listed([repr(one) for one in (room.into, *room.across)])}, "
        f"is solved as part of {room.into!r}, which is of the study's medium, so no "
        "element as thin as the gap is laid"
        for room in joined
    ]


def reserving(reserved: Reserved, sweep: Sweep, unwalled: Sequence[Unwalled] = ()) -> list[str]:
    """What a study that reserves room round the structure states before the
    run: where the open surface stands and what it reflects, how far the bound
    the box is built on stands off the drawing, what each side the domain ends
    on carries where the drawing does not cover it, which room metal seals off,
    and each face of a region standing in the reserved air.

    Every one of these is a statement and not a warning. A wall the drawing does
    not cover is the domain's own truncation, and a face standing in the air is
    what reserving it means; what the drawing does not say is which of them was
    meant.

    Two figures are stated at the bottom of the band, and neither bounds what the
    surface does. The first is what the condition reflects of the dipole mode,
    which is largest at the bottom of the band, where the surface stands the
    fewest wavelengths away. The second is what a flat side reflects of a plane
    wave meeting it at the steepest angle a straight line from the structure
    meets an open side at, which no distance lowers. The run takes the larger as
    the reflection its estimate after the solve is formed from. Both are closed
    forms in the drawing and the band, so both are stated before anything is
    meshed.

    Neither says anything about the field a line holds to its cross-section,
    which falls off over a distance the cross-section sets rather than the
    wavelength, so the clearance is stated against the thinnest drawn body as
    well.
    """
    lines = medium_said(
        Medium(
            name=reserved.medium,
            permittivity=reserved.filling.permittivity,
            permeability=reserved.filling.permeability,
            loss_tangent=reserved.filling.loss_tangent,
            conductivity=reserved.filling.conductivity,
        ),
        sweep.start,
        sweep.stop,
        reserved.clearance,
        bool(reserved.faces),
    )
    if reserved.faces:
        lines.append(_surface(reserved, sweep))
    if reserved.overstated is not None:
        shape, stands = reserved.overstated
        # The bound is exact for a face square to an axis and stands off a
        # curved surface.
        lines.append(
            f"The reserved air is measured from FreeCAD's bounding box of each bound shape, "
            f"which stands up to {stands:.4g} mm off the body of {shape!r}, so an open side "
            "stands up to that much farther from the body than the clearance"
        )
    # The field a line holds in the air meets a Through side squarely, and a
    # magnetic wall ends the line there as the port's plane does.
    for one in reserved.magnetic:
        lines.append(
            f"The air beside the lumped ports' faces in {one.side} is a magnetic wall, as "
            f"each port's plane is: {one.area:.4g} mm^2 of that {one.whole:.4g} mm^2 side. It "
            "reflects any wave radiated onto it, and no figure stated here counts that"
        )
    for one in reserved.walled:
        lines.append(
            f"{one.side} of the reserved air is a perfect conductor over {one.area:.4g} mm^2 "
            f"of its {one.whole:.4g} mm^2 that the drawing does not cover, because "
            f"Boundary{one.side} ends the domain there. Draw the conductor out to the edge of "
            f"the air, or set Boundary{one.side} to {OPEN_FACE}"
        )
    for room in reserved.sealed:
        if room.stands_off:
            # The box is built on a bound that stands off a curved face, on a side
            # the domain ends on the structure.
            lines.append(
                f"The room from {_point(room.least)} to {_point(room.most)} mm, "
                f"{room.volume:.4g} mm^3, between "
                f"{_listed([repr(one) for one in room.bounded_by] or ['the drawing'])} and "
                "the bounding box of a curved face, is left out of the model. The model ends "
                "on the drawing there, which carries the wall"
            )
            continue
        closing = [repr(one) for one in room.bounded_by]
        if room.sides:
            sides = "side" if len(room.sides) == 1 else "sides"
            closing.append(f"the domain's {sides} {_listed(room.sides)}")
        lines.append(
            f"The room from {_point(room.least)} to {_point(room.most)} mm, {room.volume:.4g} "
            f"mm^3, is left out of the model: it is closed off by {_listed(closing)}, and no "
            "port, lumped element, other material or open side reaches it, so no field does. "
            "The model ends on its faces, which carry the wall"
        )
    beyond = f"the medium {reserved.medium!r}" if reserved.medium else "vacuum"
    for bare in unwalled:
        faces = "; ".join(
            f"{face.where}, {face.area:.4g} mm^2"
            + (f", toward Boundary{face.side}" if face.side else "")
            for face in bare.faces
        )
        lines.append(
            f"{bare.region!r} stands in the reserved air over {bare.area:.4g} mm^2, on "
            f"{faces}. No metal is drawn on those faces, so each borders {beyond} and none is "
            "a wall. Where one is meant as a wall, draw metal on it, or set the Boundary it "
            f"stands toward to {PEC_WALL_FACE}"
        )
    return lines


def _surface(reserved: Reserved, sweep: Sweep) -> str:
    """Where the open surface stands, and what it reflects at the bottom of the
    band. See :func:`reserving`."""
    distance = reserved.clearance
    slowing = reserved.filling.slowing
    edges = [sweep.start] if sweep.highest == sweep.start else [sweep.start, sweep.highest]
    where = f"wavelength in {reserved.medium!r}" if reserved.medium else "free-space wavelength"
    wavelengths = [
        f"{distance / (units.SPEED_OF_LIGHT / frequency / math.sqrt(slowing) * units.MM_PER_M):.2g}"
        f" of the {where} at {frequency / 1e9:.4g} GHz"
        for frequency in edges
    ]
    thinnest = ""
    if reserved.thinnest is not None:
        region, extent = reserved.thinnest
        thinnest = (
            f", and {distance / extent:.3g} times the {extent:.4g} mm smallest extent of {region!r}"
        )
    angle = steepest(reserved)
    dipole, slant = reflection(sweep.start, distance, slowing), slanted(angle)
    larger = "dipole mode's" if dipole >= slant else "slanted wave's"
    # The field a line holds to its cross-section falls off over a distance the
    # cross-section sets rather than the wavelength, so neither figure covers it.
    return (
        f"The open surface is {_listed(reserved.faces)} of the reserved air, "
        f"{distance:.4g} mm from the structure: {' and '.join(wavelengths)}{thinnest}. "
        "It carries a first-order absorbing condition. At the bottom of the band, "
        f"{sweep.start / 1e9:.4g} GHz, it reflects {percent(dipole, 2)} of the dipole mode "
        f"radiated from the structure, and {percent(slant, 2)} of a plane wave meeting a "
        f"side {math.degrees(angle):.3g} degrees from square, the steepest angle at which a "
        f"line from the structure meets an open side. The {larger} is the larger, and sets "
        "the estimate after the run. "
        "Neither figure is a bound, and neither covers the field a line holds close to "
        "its cross-section"
    )


def _point(point: Sequence[float]) -> str:
    """A point in a sentence, its coordinates in millimetres."""
    return "(" + ", ".join(f"{at:.4g}" for at in point) + ")"


def _listed(names: Sequence[str]) -> str:
    """Names in a sentence: ``A``, ``A and B``, ``A, B and C``."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
