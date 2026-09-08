# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the solver object says about the run, as the adapter's own settings.

This module reads the band, the excitation, the boundaries, how much air to
leave around the structure, and the mesh policy: everything that describes the
run rather than the device. The device is :mod:`~.geometry`, :mod:`~.materials`
and :mod:`~.ports`.

Lengths arrive in millimetres and frequencies in hertz, which is what the
envelope holds.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from ... import units
from ...portbox import AXIS_NAMES
from .model import (
    DIMENSIONS,
    SPEED_OF_LIGHT,
    THROUGH,
    Frequency,
    Material,
    Termination,
    check_timestep_factor,
)
from .properties import TranslationError, _label, _model_fault, _value
from .regions import MeshParams
from .sizing import fits_inside

#: Document boundary names to openEMS'. ``PML`` takes its depth from
#: ``PMLCells`` and is spelled per-face.
_BOUNDARY_WORDS = {"PEC": "PEC", "PMC": "PMC", "Mur": "MUR"}


# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------


def _frequency(analysis: Any) -> Frequency:
    start = _value(analysis.FrequencyStart)
    stop = _value(analysis.FrequencyStop)
    if start <= 0 or stop <= start:
        raise TranslationError(
            f"{_label(analysis)!r}: the band runs {start:.4g} to {stop:.4g} Hz. "
            "FrequencyStart must be above zero and below FrequencyStop"
        )
    with _model_fault(_label(analysis)):
        return Frequency(start=start, stop=stop, points=int(analysis.NumFrequencyPoints))


#: The one excitation this adapter has. ``driver`` builds a single pulse and
#: nothing selects between shapes. It is stated again here rather than imported,
#: because ``Objects/analysis.GAUSSIAN`` sits behind ``import FreeCAD``. A test
#: holds the two together, and that test also catches a value offered in the
#: property editor before there is a call to honour it.
GAUSSIAN = "Gaussian"


def _waveform(analysis: Any) -> None:
    """Refuse a waveform this adapter cannot produce.

    The document object offers one value, so a document built through the GUI
    cannot fail this check. A file can. FreeCAD stores an enumeration's whole
    list in the document and restores it from there rather than from the class,
    so a document written with a longer list goes on offering that list, and
    what it offers would otherwise be solved as a Gaussian with no message.
    """
    declared = str(analysis.Waveform)
    if declared != GAUSSIAN:
        raise TranslationError(
            f"{_label(analysis)!r}: Waveform is {declared!r}. openEMS is driven "
            f"here with a {GAUSSIAN} pulse covering the whole band, and this "
            f"adapter has no other excitation to offer. Set it to {GAUSSIAN}"
        )


def _smallest_response(analysis: Any) -> float:
    """The study's reading floor, converted from decibels to a magnitude in S.

    The conversion is in amplitude rather than power, because the floor is
    weighed against an error in S: twenty decibels a decade, the same decibels
    the response is plotted in.

    Zero is full scale, and it is what a study that has not considered the
    question says, so the ordinary run keeps the absolute bar. A value above
    zero is refused rather than clamped. A passive device answers no more than
    one, so a positive value is a typing slip, and a slip quietly taken as "full
    scale" would leave the property reading as though it had been honoured.
    """
    declared = float(analysis.SmallestResponse)
    if not math.isfinite(declared) or declared > 0.0:
        raise TranslationError(
            f"{_label(analysis)!r}: SmallestResponse is {declared:g} dB. It is "
            "how far down the response is read, so it is zero for full scale "
            "and negative below it"
        )
    return 10.0 ** (declared / 20.0)


def _boundary(solver: Any) -> tuple[str, str, str, str, str, str]:
    cells = int(solver.PMLCells)
    words = []
    for axis in ("X", "Y", "Z"):
        for side in ("Min", "Max"):
            name = f"Boundary{axis}{side}"
            declared = str(getattr(solver, name))
            if declared == "PML":
                if cells < 1:
                    raise TranslationError(
                        f"{_label(solver)!r}: {name} is PML but PMLCells is "
                        f"{cells}. An absorber needs depth; eight cells is the "
                        "usual choice"
                    )
                words.append(f"PML_{cells}")
                continue
            word = _BOUNDARY_WORDS.get(declared)
            if word is None:
                raise TranslationError(
                    f"{_label(solver)!r}: {name} is {declared!r}, which the "
                    "openEMS adapter does not support. Use PML, PEC, PMC or Mur"
                )
            words.append(word)
    # Unpacked because the six are a fixed set, one per face, which the loop
    # above only implies.
    x_min, x_max, y_min, y_max, z_min, z_max = words
    return (x_min, x_max, y_min, y_max, z_min, z_max)


def _absorber_cells(boundary: Sequence[str], depth: int) -> tuple[int, int, int]:
    """Absorber depth per axis, for the mesher.

    The mesher lays uniform cells at both ends of any axis that absorbs,
    because a graded absorber reflects. An axis walled by a conductor gets none.
    That matters: a waveguide meshed as though it absorbed sideways would grow
    past its own walls, and its cutoff frequency would move.

    An axis absorbing on one face only still gets uniform cells on both. The
    cost is a few cells of grid in a corner of the model. The alternative is a
    per-face parameter the mesher does not have.
    """
    return tuple(  # type: ignore[return-value]
        depth if any(boundary[2 * dim + side].startswith("PML") for side in (0, 1)) else 0
        for dim in range(DIMENSIONS)
    )


def _padding(settings: Any) -> tuple[tuple[Any, Any], ...]:
    """Per-face domain padding, in the form :func:`~.plan.domain` wants.

    ``Through`` does not add air. It ends the grid on the structure and takes
    the absorber out of the structure's own extent. A transmission line padded
    that way is infinite. A line given air at its ends radiates off an
    open circuit, and the reflection contaminates every impedance read from it.
    The choice is therefore made per face by the user rather than inferred.
    """
    faces = []
    for axis in AXIS_NAMES:
        pair: list[Any] = []
        for name in ("Min", "Max"):
            mode = str(getattr(settings, f"Padding{axis}{name}"))
            if mode == "Through":
                pair.append(THROUGH)
            elif mode == "Air":
                buffer = f"AirCells{axis}{name}"
                count = int(getattr(settings, buffer))
                if count < 0:
                    raise TranslationError(
                        f"{_label(settings)!r}: {buffer} is {count}. Set that "
                        "face's Padding to Through to put the absorber on the "
                        "structure; a negative count is not the way to ask for it"
                    )
                pair.append(count)
            else:
                raise TranslationError(
                    f"{_label(settings)!r}: Padding{axis}{name} is "
                    f"{mode!r}; expected 'Air' or 'Through'"
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
    settings: Any,
    materials: Iterable[Material],
    frequency: Frequency,
    absorber: tuple[int, int, int],
) -> MeshParams:
    """Mesh policy in millimetres, from the document's wavelength fractions.

    ``ElementsPerWavelength`` counts elements across the wavelength in the
    slowest material in the model at the top of the band, and never a length. A
    remembered millimetre value under-resolves the moment anything raises the
    permittivity or the frequency.

    ``EdgeRefinement`` sizes the cell at a conductor edge, and it is the error
    there that looks like ordinary discretisation error rather than like a
    conductor meshed wrongly. It is not the whole of what the grid spends on a
    conductor: :func:`~.metal._edge_size` also sizes the cell from the metal's
    own width, and takes the finer of the two wherever that second demand is
    above the grid's floor. So on a narrow trace a coarse refinement is
    overridden rather than obeyed. The defaults follow openEMS' own
    ``MSL_Losses.m``: bulk at lambda/20, edges six times finer.
    """
    per_wavelength, refinement = _check_mesh_policy(settings)
    wavelength = _wavelength(materials, frequency)
    bulk = wavelength / per_wavelength
    ratio = float(settings.MaxGrowthRatio)
    # Zero means "derive it", which MeshParams spells as None.
    floor = _value(settings.MinElementSize) or None
    return MeshParams(
        metal_res=bulk / refinement,
        dielectric_res=bulk,
        max_ratio=(ratio, ratio, ratio),
        min_lines=int(settings.MinElementsAcross),
        pml_cells=absorber,
        min_cell=floor,
        # The coarsest cell anywhere: the bulk size in vacuum. Every
        # dielectric then asks for its own sqrt(epsilon) finer over its own
        # span, and air, which asks for nothing, gets this. What taking one
        # lambda from the slowest material in the model instead would cost is on
        # regions.Region.size.
        #
        # It sizes the air faces of the domain as well as the sizing field, air
        # being what relaxes to it. A THROUGH face is sized by nothing here: the
        # domain ends on the drawing, and the absorber comes out of it at the
        # pitch the mesher lays there.
        cap=_coarsest_cell(settings, frequency),
    )


def _coarsest_cell(settings: Any, frequency: Frequency) -> float:
    """The largest cell the grid will use anywhere, in mm.

    It is read off the vacuum wavelength at the top of the band, so it needs no
    material and can be asked before the geometry has been read. A conductor
    drawn without a thickness can therefore be given one measured against the
    grid that will hold it.
    """
    per_wavelength, _ = _check_mesh_policy(settings)
    return SPEED_OF_LIGHT / frequency.stop * units.MM_PER_M / per_wavelength


def _skin(settings: Any, frequency: Frequency) -> float:
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
    _, refinement = _check_mesh_policy(settings)
    return fits_inside(_coarsest_cell(settings, frequency) / refinement)


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


def _check_mesh_policy(settings: Any) -> tuple[float, float]:
    """Validate the mesh policy once, before anything reads it.

    It is called from :func:`contents` rather than only from
    :func:`_mesh_params`. ``_Context`` divides by ``ElementsPerWavelength`` to
    size ports, and it is built before mesh parameters exist, so validating
    inside ``_mesh_params`` would let a zero through to a ``ZeroDivisionError``
    two functions before the message meant to explain it. Policy checks belong
    at the one place every route passes through, for the reason the driver
    re-runs pre-flight.

    It returns the two numbers it had to parse anyway.
    """
    if not hasattr(settings, "ElementsPerWavelength"):
        raise TranslationError(
            f"{_label(settings)!r} has no ElementsPerWavelength, so it does not "
            "look like a mesh policy object; attach an EMMeshPolicy"
        )

    per_wavelength = float(settings.ElementsPerWavelength)
    if per_wavelength <= 0:
        raise TranslationError(
            f"{_label(settings)!r}: ElementsPerWavelength is {per_wavelength}; it "
            "counts elements across a wavelength and must be above zero"
        )

    refinement = float(settings.EdgeRefinement)
    if refinement < 1:
        raise TranslationError(
            f"{_label(settings)!r}: EdgeRefinement is {refinement}, so conductor "
            "edges would be meshed more coarsely than open space. Edges carry the "
            "field singularity that sets a line's impedance; they need the finer "
            "grid, not the coarser one. Use 1 for no refinement"
        )
    return per_wavelength, refinement


def _termination(solver: Any) -> Termination:
    """When to stop stepping.

    ``EnergyDecay`` at or above zero means "do not stop early", and that is the
    default. It reads like a disabled feature, and it is the only reproducible
    setting. openEMS re-evaluates its energy criterion inside a branch gated on
    four seconds of wall clock (``openEMS/openems.cpp:1445``), so an
    energy-terminated run stops at a step count that depends on machine load.
    Pre-flight warns when this is switched on.
    """
    decay = float(solver.EnergyDecay)
    with _model_fault(_label(solver)):
        return Termination(
            max_timesteps=int(solver.MaxTimesteps),
            end_criteria=0.0 if decay >= 0 else 10 ** (decay / 10.0),
        )


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
            f"{_label(solver)!r}: Threads is {threads}. Use 0 to let openEMS "
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
    Update Mesh reports a timestep at twice the vacuum CFL bound, captioned as
    an estimate of it and in green, and only Run refuses it. The two paths would
    then disagree about whether a value is legal, with the permissive one
    drawing the picture.

    It raises a :class:`TranslationError` rather than the
    :class:`EnvelopeError` underneath, through :func:`_model_fault`, where that
    rule is written down. Typing 2 into a property field is not a crash. The
    bound itself stays in ``model.check_timestep_factor``, stated once.
    """
    with _model_fault(_label(solver)):
        return check_timestep_factor(float(solver.TimestepFactor))
