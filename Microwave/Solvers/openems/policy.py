# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the solver object says about the run, as the adapter's own settings.

This module reads the band, the excitation, the condition on each face of the
domain, how much air to leave around the structure, the mesh policy and the Yee
grid: everything that describes the run rather than the device. The device is
:mod:`~.geometry`, :mod:`~.materials` and :mod:`~.ports`.

Lengths arrive in millimetres and frequencies in hertz, which is what the
envelope holds.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from ... import units
from ...portbox import AXIS_NAMES, FLATNESS
from .model import (
    DIMENSIONS,
    SPEED_OF_LIGHT,
    THROUGH,
    Frequency,
    Material,
    Port,
    Solid,
    Termination,
    check_timestep_factor,
)
from .plan import structure_bounds
from .properties import TranslationError, _label, _model_fault, _value, clearance
from .regions import Faces, MeshParams
from .sizing import fits_inside

#: How a face that absorbs does it, as the solver object offers it. Stated
#: again here rather than imported, because ``Objects/solver.py`` sits behind
#: ``import FreeCAD``. A test holds the two lists together.
ABSORBERS = ("PML", "Mur")

#: What openEMS calls a wall of perfect conductor.
WALL = "PEC"

#: What a face of the domain may say lies beyond the structure. Stated again
#: here rather than imported, because ``Objects/mesh.py`` sits behind
#: ``import FreeCAD`` and this module reaches no FreeCAD at all. A test holds
#: the two lists together.
OPEN_FACE = "Open"
THROUGH_FACE = "Through"
PEC_WALL_FACE = "PEC Wall"


# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------


def _frequency(analysis: Any) -> Frequency:
    start = _value(analysis.FrequencyStart)
    stop = _value(analysis.FrequencyStop)
    if start <= 0 or stop <= start:
        raise TranslationError(
            f"{_label(analysis)!r}: FrequencyStart is {start:.4g} Hz and FrequencyStop is "
            f"{stop:.4g} Hz. Set FrequencyStart above 0 and below FrequencyStop"
        )
    with _model_fault(_label(analysis)):
        return Frequency(start=start, stop=stop, points=int(analysis.FrequencyPoints))


#: The one excitation this adapter has. ``driver`` builds a single pulse and
#: nothing selects between shapes. It is stated again here rather than imported,
#: because ``Objects/solver.WAVEFORMS`` sits behind ``import FreeCAD``. A test
#: holds the two together, and that test also catches a value offered in the
#: property editor before there is a call to honour it.
GAUSSIAN = "Gaussian"


def _waveform(solver: Any) -> None:
    """Refuse a waveform this adapter cannot produce.

    The document object offers one value, so a document built through the GUI
    cannot fail this check. A file can. FreeCAD stores an enumeration's whole
    list in the document and restores it from there rather than from the class,
    so a document written with a longer list goes on offering that list, and
    what it offers would otherwise be solved as a Gaussian with no message.
    """
    declared = str(solver.Waveform)
    if declared != GAUSSIAN:
        raise TranslationError(
            f"{_label(solver)!r}: Waveform is {declared!r}, and openEMS is driven only "
            f"with a {GAUSSIAN} pulse here. Set Waveform to {GAUSSIAN}"
        )


def _absorbing(solver: Any, opened: Sequence[str] = ()) -> str:
    """The word openEMS takes for a face that absorbs, as the solver says it does.

    ``opened`` names each ``PEC Wall`` face that absorbs because waveguide ports
    cover it. Mur's condition there is refused: the port's source lies on the
    face, and openEMS holds a Mur face shut until the source on it has finished,
    which for this adapter's drive is never, so the face would reflect for the
    whole run.
    """
    kind = str(solver.Absorber)
    if kind == "Mur":
        if opened:
            raise TranslationError(
                f"{_label(solver)!r}: Absorber is Mur, and waveguide ports stand on "
                f"{' and '.join(opened)}, so the face under each port reflects for the "
                "whole run. Set Absorber to PML"
            )
        return "MUR"
    if kind != "PML":
        raise TranslationError(
            f"{_label(solver)!r}: Absorber is {kind!r}, which openEMS does not offer here. "
            f"Set Absorber to {' or '.join(ABSORBERS)}"
        )
    cells = int(solver.PMLCells)
    if cells < 1:
        raise TranslationError(
            f"{_label(solver)!r}: Absorber is PML and PMLCells is {cells}, so the absorber "
            "has no depth. Set PMLCells to 1 or more; 8 is the usual depth"
        )
    return f"PML_{cells}"


def _boundary(
    solver: Any, settings: Any, solids: Sequence[Solid], ports: Sequence[Port]
) -> tuple[str, str, str, str, str, str]:
    """The condition on each face of the domain, read off what the mesh policy
    says lies beyond it.

    ``Open`` and ``Through`` absorb: beyond an ``Open`` face the medium runs on, and
    through a ``Through`` face the structure runs on without end. ``PEC Wall`` says
    the domain stops where the structure does, and what stops it there is a
    perfect wall - which is what a backend that builds no absorber puts on every
    face it does not open. The one exception is a ``PEC Wall`` face waveguide ports
    stand across, whole: the wave leaves through the port, and a wall behind it
    would send the wave back. openEMS states one condition for a whole face, so
    a face the ports cover in part is refused rather than given either.

    A port whose plane lies in a face the wall is laid on is refused as well,
    since the wall shorts it, and so is a lumped port lying in a face the
    absorber is laid over. :func:`_check_no_port_lies_in` says which. How the face absorbs
    is the solver's own: a perfectly matched layer ``PMLCells`` deep, or Mur's
    condition.
    """
    lower, upper = structure_bounds(solids, ports)
    absorbs = []
    opened = []
    for dim, axis in enumerate(AXIS_NAMES):
        for side, name in enumerate(("Min", "Max")):
            face = f"{axis}{name}"
            stated = str(getattr(settings, f"Boundary{face}"))
            at = float((lower, upper)[side][dim])
            _check_no_port_lies_in(face, stated, dim, at, ports)
            guided = stated == PEC_WALL_FACE and _left_through_a_guide(
                settings, face, dim, at, lower, upper, ports
            )
            absorbs.append(stated != PEC_WALL_FACE or guided)
            if guided:
                opened.append(face)
    # Asked only where a face absorbs, so a study walled on every face states
    # no absorber and is not refused over one.
    absorbing = _absorbing(solver, opened) if any(absorbs) else WALL
    words = [absorbing if absorb else WALL for absorb in absorbs]
    _check_no_guide_runs_on_behind_its_port(settings, words, ports)
    x_min, x_max, y_min, y_max, z_min, z_max = words
    return (x_min, x_max, y_min, y_max, z_min, z_max)


def _check_no_guide_runs_on_behind_its_port(
    settings: Any, words: Sequence[str], ports: Sequence[Port]
) -> None:
    """Refuse a waveguide port standing inside the guide with a wall behind it.

    openEMS solves what lies behind a waveguide port's plane: the guide runs on
    from the port to the face behind it, and where that face is a wall the guide
    behind the port is a stub shorted there, which reflects. A solver that ends
    the model on the port's plane cuts that stub away. The drawing answers only
    one of the two, so it is refused rather than solved as either. A port on the
    face itself has opened that face, and a face the guide runs out through is
    the absorber.
    """
    for port in ports:
        if port.kind != "rect_waveguide":
            continue
        dim = port.propagation_axis
        side = 0 if port.direction > 0 else 1
        if words[2 * dim + side] != WALL:
            continue
        face = f"{AXIS_NAMES[dim]}{('Min', 'Max')[side]}"
        # A solver that ends the model on the port's plane cuts the stub away,
        # so the drawing does not decide between the two answers.
        raise TranslationError(
            f"waveguide port {port.name!r} stands inside the guide, and the face behind "
            f"it, {face}, is {PEC_WALL_FACE}, so openEMS solves the guide behind the port "
            f"as a stub shorted at that face. Set Boundary{face} to {THROUGH_FACE} on "
            f"{_label(settings)!r}, or move the port onto the guide's end face"
        )


def _check_no_port_lies_in(
    face: str, stated: str, dim: int, at: float, ports: Sequence[Port]
) -> None:
    """Refuse a port whose plane lies in a face of the domain the wall or the
    absorber is laid on, where it cannot be driven there.

    On a ``PEC Wall`` face openEMS lays a perfect wall, and a wall across a port's
    plane shorts it: a lumped port lying flat in the face, or a microstrip or
    coaxial port whose feed stands on it. Only rectangular waveguide ports
    standing across the whole face are left to :func:`_left_through_a_guide`,
    which opens the face for them. On a ``Through`` face a lumped port is
    refused as well, since the line it drives runs on into the absorber rather
    than ending on the port; a microstrip or coaxial port there is the line
    running out through the absorber, which is what the face says. An ``Open``
    face stands a clearance off the structure, so no port lies in it.
    """
    if stated == OPEN_FACE:
        return
    for port in ports:
        if port.kind == "rect_waveguide":
            continue
        flat = abs(port.start[dim] - at) <= FLATNESS and abs(port.stop[dim] - at) <= FLATNESS
        fed = (
            port.propagation_axis == dim
            and abs(port.start[dim] + port.direction * port.feed_shift - at) <= FLATNESS
        )
        if port.kind == "lumped":
            if not flat:
                continue
        elif stated != PEC_WALL_FACE or not fed:
            continue
        if stated == THROUGH_FACE:
            what = "the line the port drives runs on into the absorber past the port"
            remedy = f"Set Boundary{face} to {OPEN_FACE}, or draw the port inside the structure"
        else:
            what = "openEMS lays a perfect wall across the port's plane and shorts it"
            remedy = (
                f"Set Boundary{face} to {OPEN_FACE}, or draw the port inside the structure"
                if port.kind == "lumped"
                else f"Set Boundary{face} to {THROUGH_FACE} or {OPEN_FACE}, or raise FeedOffset "
                "on the port so the feed stands inside the structure"
            )
        raise TranslationError(
            f"{port.kind.replace('_', ' ')} port {port.name!r} lies in the {face} face of "
            f"the domain, and Boundary{face} is {stated}, so {what}. {remedy}"
        )


def _left_through_a_guide(
    settings: Any,
    face: str,
    dim: int,
    at: float,
    lower: Any,
    upper: Any,
    ports: Sequence[Port],
) -> bool:
    """Whether waveguide ports stand across the whole of one face of the domain.

    A port stands on the face where its plane lies within :data:`FLATNESS` of
    it. The ports' cross-sections are taken together, so two guides side by side
    that fill the face between them cover it. Refused where they cover part of
    it and not the rest.
    """
    standing = [
        port
        for port in ports
        if port.kind == "rect_waveguide"
        and port.propagation_axis == dim
        and abs(port.start[dim] - at) <= FLATNESS
    ]
    if not standing:
        return False
    across = [other for other in range(DIMENSIONS) if other != dim]
    spans = [float(upper[other]) - float(lower[other]) for other in across]
    whole = spans[0] * spans[1]

    def span(port: Port, other: int) -> tuple[float, float]:
        """Where the port's cross-section runs along ``other``, held to the face."""
        return (
            max(min(port.start[other], port.stop[other]), float(lower[other])),
            min(max(port.start[other], port.stop[other]), float(upper[other])),
        )

    covered = _covered([(span(port, across[0]), span(port, across[1])) for port in standing])
    if whole - covered <= FLATNESS * 2.0 * (spans[0] + spans[1]):
        return True
    names = " and ".join(repr(port.name) for port in standing)
    # openEMS states one condition for a whole face. An absorber there takes
    # away the wall beside the port, and a wall there stands behind the port.
    raise TranslationError(
        f"{names} {'stands' if len(standing) == 1 else 'stand'} on the {face} face of "
        f"the domain and {'leaves' if len(standing) == 1 else 'leave'} "
        f"{whole - covered:.4g} mm^2 of its {whole:.4g} mm^2 uncovered. Boundary{face} is "
        f"{PEC_WALL_FACE}, and openEMS applies one condition to a whole face, so the face "
        "cannot be a wall beside the port and open under it. Draw the guide as the air "
        f"inside it alone, so the {PEC_WALL_FACE} faces are its walls and the port covers "
        f"its end face, or set Boundary{face} to {THROUGH_FACE} on {_label(settings)!r} and "
        "move the port to a plane inside the guide"
    )


def _covered(rectangles: Sequence[tuple[tuple[float, float], tuple[float, float]]]) -> float:
    """The area a set of rectangles covers together, each counted once."""
    firsts = sorted({edge for (first, _) in rectangles for edge in first})
    seconds = sorted({edge for (_, second) in rectangles for edge in second})
    area = 0.0
    for low, high in zip(firsts, firsts[1:]):
        for bottom, top in zip(seconds, seconds[1:]):
            if any(
                first[0] <= low and high <= first[1] and second[0] <= bottom and top <= second[1]
                for first, second in rectangles
            ):
                area += (high - low) * (top - bottom)
    return area


#: Every face of the domain, spelt the way a per-face property name spells it.
_FACES = tuple(f"{axis}{side}" for axis in AXIS_NAMES for side in ("Min", "Max"))


def _absorber_cells(boundary: Sequence[str], depth: int) -> Faces:
    """Absorber depth at each face, for the mesher.

    A face laying a perfectly matched layer gets ``depth`` uniform cells,
    because a graded absorber reflects. A face walled by a conductor gets none,
    and that matters twice. openEMS lays the wall on the outermost line, so a
    cell beyond the drawing moves the wall off it: a waveguide meshed as though
    it absorbed sideways grows past its own walls and its cutoff moves, and the
    short that ends a one-port guide stands a block past its drawn end. Mur's
    condition sits on the face itself and takes no cells either.
    """
    return tuple(  # type: ignore[return-value]
        tuple(depth if boundary[2 * dim + side].startswith("PML") else 0 for side in (0, 1))
        for dim in range(DIMENSIONS)
    )


def _padding(
    settings: Any, frequency: Frequency, slowing: float = 1.0
) -> tuple[tuple[Any, Any], ...]:
    """Per-face domain padding in millimetres, in the form :func:`~.plan.domain` wants.

    What lies beyond the structure on each face is the policy's, because it is a
    statement about the problem, and so is how far: ``Open`` is padded by the
    policy's ``Clearance``, and :func:`~...properties.clearance` derives it
    where it is zero, in the medium ``slowing`` names. The length stated is the
    length laid, whatever the grid's density.

    ``Through`` does not add air. It ends the grid on the structure and takes
    the absorber out of the structure's own extent. A transmission line padded
    that way is infinite. A line given air at its ends radiates off an
    open circuit, and the reflection contaminates every impedance read from it.
    The choice is therefore made per face by the user rather than inferred.

    ``PEC Wall`` leaves no room past the face. The domain stops on the structure,
    and an absorber a face carries is added beyond it rather than taken out of
    it, which is what a padding of nought expresses to :func:`~.plan.domain`. It is what a face
    walled by a conductor wants: nothing is measured outside the wall, so cells
    spent there are cells spent where the field does not go.
    """
    faces = []
    for axis in AXIS_NAMES:
        pair: list[Any] = []
        for name in ("Min", "Max"):
            mode = str(getattr(settings, f"Boundary{axis}{name}"))
            if mode == THROUGH_FACE:
                pair.append(THROUGH)
            elif mode == PEC_WALL_FACE:
                pair.append(0)
            elif mode == OPEN_FACE:
                pair.append(clearance(settings.Clearance, frequency.stop, settings, slowing))
            else:
                raise TranslationError(
                    f"{_label(settings)!r}: Boundary{axis}{name} is {mode!r}. Set it to "
                    f"{OPEN_FACE}, {THROUGH_FACE} or {PEC_WALL_FACE}"
                )
        faces.append((pair[0], pair[1]))
    return tuple(faces)


def _wavelength(materials: Iterable[Material], frequency: Frequency) -> float:
    """Wavelength in millimetres in the slowest material, at the top of the band.

    The top of the band is where cells have to be smallest. The slowest
    material is the one that matters, because a wave slows by sqrt(epsilon * mu)
    inside it, and meshing to the vacuum wavelength under-resolves it by that
    factor.

    Permeability belongs in that product. ``mu`` is settable in the GUI and is
    passed to CSXCAD, so leaving it out solves a ferrite as a magnetic material
    and meshes it as a non-magnetic one: cells too coarse by sqrt(mu_r),
    dispersion, and a resonance in the wrong place with no warning.
    """
    slowest = max([material.epsilon * material.mu for material in materials] or [1.0])
    return SPEED_OF_LIGHT / frequency.stop / math.sqrt(slowest) * units.MM_PER_M


def _mesh_params(
    recipe: Any,
    materials: Iterable[Material],
    frequency: Frequency,
    absorber: Faces,
    medium: float = 1.0,
) -> MeshParams:
    """What the grid is laid to, in millimetres, from the Yee grid.

    The Yee grid states the cell: ``ElementsPerWavelength`` counts cells across
    the wavelength in the slowest material in the model at the top of the band,
    and never a length. A remembered millimetre value under-resolves the moment
    anything raises the permittivity or the frequency. It states as well a count
    across a dielectric, and a floor against a sliver.

    ``medium`` is the product a wave slows by the root of in the study's medium.
    The medium fills every cell no solid asks a size of, so its bulk size is the
    coarsest cell anywhere, and it is one of ``materials`` as well.

    ``EdgeRefinement`` sizes the cell at a conductor edge, and it is the error
    there that looks like ordinary discretisation error rather than like a
    conductor meshed wrongly. It is not the whole of what the grid spends on a
    conductor: :func:`~.metal._edge_size` also sizes the cell from the metal's
    own width, and takes the finer of the two wherever that second demand is
    above the grid's floor. So on a narrow trace a coarse refinement is
    overridden rather than obeyed. The defaults follow openEMS' own
    ``MSL_Losses.m``: bulk at lambda/20, edges six times finer.
    """
    per_wavelength, refinement = _check_yee_grid(recipe)
    wavelength = _wavelength(materials, frequency)
    bulk = wavelength / per_wavelength
    ratio = float(recipe.MaxGrowthRatio)
    # Zero means "derive it", which MeshParams spells as None.
    floor = _value(recipe.MinElementSize) or None
    _check_the_floor_leaves_the_edge_cell_room(recipe, bulk / refinement, floor)
    return MeshParams(
        metal_res=bulk / refinement,
        dielectric_res=bulk,
        max_ratio=(ratio, ratio, ratio),
        min_lines=int(recipe.MinElementsAcross),
        pml_cells=absorber,
        min_cell=floor,
        # The bulk size in vacuum. Every dielectric asks for its own
        # sqrt(epsilon) finer over its own span, and the medium, which fills
        # what asks for nothing, is the ceiling. What taking one lambda from the
        # slowest material in the model instead would cost is on
        # regions.Region.size.
        cap=_coarsest_cell(recipe, frequency),
        medium=medium,
    )


def _check_the_floor_leaves_the_edge_cell_room(
    recipe: Any, edge_cell: float, floor: float | None
) -> None:
    """Refuse a floor that stands above the cell a conductor's edge is meshed at.

    The mesher holds the same bound and states it in its own words -
    ``min_cell`` against ``metal_res``, neither of which is a property anybody
    can find. This says it in the units the properties are stated in and names
    the object that carries them, so the message is one the user can act on, and
    it fires before the mesher is reached.

    The bound is the edge cell rather than the bulk one. The floor is a guard
    against a sliver a boolean left, so a floor above the finest cell the grid
    means to lay would override the resolution it is there to protect.
    """
    if floor is None or floor <= edge_cell:
        return
    raise TranslationError(
        f"{_label(recipe)!r}: MinElementSize is {floor:.4g} mm, above the {edge_cell:.4g} mm "
        "cell at conductor edges, so it coarsens the edges. Lower MinElementSize below "
        f"{edge_cell:.4g} mm"
    )


def _coarsest_cell(recipe: Any, frequency: Frequency) -> float:
    """The largest cell the grid will use anywhere, in mm.

    It is read off the vacuum wavelength at the top of the band, so it needs no
    material and can be asked before the geometry has been read. A conductor
    drawn without a thickness can therefore be given one measured against the
    grid that will hold it.
    """
    per_wavelength, _ = _check_yee_grid(recipe)
    return SPEED_OF_LIGHT / frequency.stop * units.MM_PER_M / per_wavelength


def _skin(recipe: Any, frequency: Frequency) -> float:
    """How thick to make a conductor drawn with no thickness at all, in mm.

    A cell has to fit inside the metal, or openEMS samples it into islands, and
    the cell a conductor's own edges are sized at is the metal resolution. The
    thickness is therefore the one whose connection demand is exactly that: the
    cell of that size has the thickness for its body diagonal, which is the
    worst direction a slab can be crossed in and so the direction the criterion
    is stated at.

    It is measured in vacuum, where that resolution is at its coarsest. A
    dielectric in the model only refines it, so this thickness stays resolved
    whatever else is drawn. It also bounds what the skin can cost: its demand is
    the vacuum metal resolution, which no model's own metal resolution is
    coarser than, so a skin is never the finest thing asking.

    The skin is not free. A cross-section is read at points across a surface
    rather than over a span, so a wide skin holds the field down across its
    whole extent where it would otherwise have relaxed toward the coarsest cell.
    A conductor drawn thicker than the mesher's reach is read as no feature at
    all and costs less. That is what supplying the thickness here costs.
    """
    _, refinement = _check_yee_grid(recipe)
    return fits_inside(_coarsest_cell(recipe, frequency) / refinement)


def _curve_tolerance(settings: Any) -> float:
    """How far a curved surface may be solved from where it was drawn, in mm.

    Zero asks for nothing, and is what the property carries by default. Leaving
    the kernel's own band costs time, and the shape's own unevenness decides how
    much, so it is set per drawing, on the models where a radius decides the
    answer, rather than as a constant for every model.

    It is read through :func:`~.properties._value` like every other length, so a
    quantity typed with a unit and a bare float are one number here. The
    property is absent on a document that predates it, and zero is what such a
    document was solved at.
    """
    return max(0.0, _value(getattr(settings, "CurveTolerance", 0.0)))


def _check_yee_grid(recipe: Any) -> tuple[float, float]:
    """Validate the Yee grid's settings before anything reads them.

    :func:`~.document.contents` calls it as soon as it has found the grid, so
    the check does not wait for :func:`_mesh_params`. ``_Context`` divides by
    ``ElementsPerWavelength`` to size ports, and it is built before mesh
    parameters exist, so validating inside ``_mesh_params`` would let a zero
    through to a ``ZeroDivisionError`` two functions before the message meant
    to explain it. These checks belong at the one place every route passes
    through, for the reason the driver re-runs pre-flight.

    It returns the two numbers it had to parse anyway, and each function here
    that reads them calls it again for that reason.
    """
    if not hasattr(recipe, "ElementsPerWavelength"):
        raise TranslationError(
            f"{_label(recipe)!r} has no ElementsPerWavelength, so it cannot size the openEMS mesh"
        )

    per_wavelength = float(recipe.ElementsPerWavelength)
    if per_wavelength <= 0:
        raise TranslationError(
            f"{_label(recipe)!r}: ElementsPerWavelength is {per_wavelength:g}. Enter a "
            "value above 0"
        )

    refinement = float(recipe.EdgeRefinement)
    if refinement < 1:
        raise TranslationError(
            f"{_label(recipe)!r}: EdgeRefinement is {refinement:g}, so conductor edges "
            "are meshed more coarsely than open space. Enter 1 or more, where 1 means no "
            "refinement"
        )

    # The mesher holds the same bound as ``min_lines``, which is no property
    # anybody can find. A count of nought also reads as "ask for nothing" - it is
    # what the same-named property on a refinement region means - so it is a
    # value a user reaches honestly, and the value that asks for nothing here is
    # one: the mesher spends a count only from two up
    # (``mesh._dielectric_spans`` and ``lfs.features``), and a dielectric the
    # grid spans is spanned by one cell whatever this says.
    across = int(recipe.MinElementsAcross)
    if across < 1:
        raise TranslationError(
            f"{_label(recipe)!r}: MinElementsAcross is {across}. Enter 1 or more; 1 asks "
            "for no minimum"
        )
    return per_wavelength, refinement


def _termination(solver: Any) -> Termination:
    """When to stop stepping.

    ``EnergyDecay`` is a level below the energy's peak, so it is negative, and
    zero means "do not stop early", which is the default. It reads like a
    disabled feature, and it is the only reproducible setting. openEMS
    re-evaluates its energy criterion inside a branch gated on four seconds of
    wall clock (``openEMS/openems.cpp:1445``), so an energy-terminated run stops
    at a step count that depends on machine load. Pre-flight warns when this is
    switched on.

    A positive value is refused. It would be a level above the peak, which the
    energy never falls to, so the run would take every step while the property
    asked for fewer. So is a negative level whose share of the peak energy, the
    number openEMS is handed, rounds to one or to zero: one is the peak itself,
    and zero is the value that switches the criterion off.
    """
    decay = float(solver.EnergyDecay)
    if not math.isfinite(decay):
        raise TranslationError(
            f"{_label(solver)!r}: EnergyDecay is {decay:g}. Enter a level in dB below the "
            "energy peak, such as -40, or 0 to run every step"
        )
    if decay > 0:
        advice = "Enter a negative level"
        if _held(-decay):
            advice += f", such as {_exactly(-decay)} for {_exactly(decay)} dB down"
        raise TranslationError(
            f"{_label(solver)!r}: EnergyDecay is {decay:g} dB, above the energy peak, so "
            f"the run never stops early. {advice}, or 0 to run every step"
        )
    # A share of zero switches the criterion off, and a share of one is the
    # peak itself.
    if decay < 0 and not _held(decay):
        raise TranslationError(
            f"{_label(solver)!r}: EnergyDecay is {decay:g} dB, a share of "
            f"{_share(decay):g} of the peak energy, and openEMS needs a share above 0 and "
            "below 1. Enter a level such as -40, or 0 to run every step"
        )
    steps = int(solver.MaxTimesteps)
    if steps < 1:
        raise TranslationError(f"{_label(solver)!r}: MaxTimesteps is {steps}. Enter 1 or more")
    with _model_fault(_label(solver)):
        return Termination(
            max_timesteps=steps,
            end_criteria=0.0 if decay >= 0 else _share(decay),
        )


def _share(decay: float) -> float:
    """The share of the peak energy a level in dB below it stands for."""
    return float(10 ** (decay / 10.0))


def _held(decay: float) -> bool:
    """Whether openEMS can be handed ``decay`` as a share it stops on."""
    return 0.0 < _share(decay) < 1.0


def _exactly(level: float) -> str:
    """``level`` as a message prints it, in as many figures as typing it back
    needs to give the same number."""
    short = f"{level:g}"
    return short if float(short) == level else repr(level)


def _threads(solver: Any) -> int:
    """How many threads openEMS may use. Zero means "as many as it likes".

    It is read through its own function for the reason :func:`_model_fault`
    records. It is the one value a user can type that lands in ``Problem``, and
    ``Problem``'s construction is the one place that context manager must not
    wrap. Checking the property before it gets there costs nothing and catches
    only itself.
    """
    threads = int(solver.Threads)
    if threads < 0:
        raise TranslationError(
            f"{_label(solver)!r}: Threads is {threads}. Enter 0 to let openEMS "
            "choose, or a positive count"
        )
    return threads


def timestep_factor(solver: Any) -> float:
    """The stability factor, refused here if openEMS would not act on it.

    It is read through a function rather than inline because two paths want it
    and only one of them builds a :class:`Problem`. The mesh preview scales its
    reported timestep by the factor without constructing an envelope, so the
    envelope's own invariant does not stand between the user and the number on
    screen. Without this check, typing 2 into ``TimestepFactor`` and pressing
    Mesh reports a timestep at twice the vacuum CFL bound, captioned as
    an estimate of it and in green, and only Run refuses it. The two paths would
    then disagree about whether a value is legal, with the permissive one
    drawing the picture.

    It raises a :class:`TranslationError` rather than the
    :class:`EnvelopeError` underneath, through :func:`_model_fault`, where that
    rule is written down. Typing 2 into a property field is not a crash. The
    bound itself stays in ``model.check_timestep_factor``, stated once.
    """
    with _model_fault(_label(solver)):
        return check_timestep_factor(float(solver.TimestepFactor), "TimestepFactor")
