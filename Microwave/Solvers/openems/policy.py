# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the solver object says about the run, as the adapter's own settings.

The band, the excitation, the boundaries, how much air to leave around the
structure, and the mesh policy - everything that describes the *run* rather than
the device. The device is :mod:`~.geometry`, :mod:`~.materials` and
:mod:`~.ports`.

Lengths arrive in millimetres and frequencies in hertz, which is what the
envelope holds.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from ... import units
from ...portbox import AXIS_NAMES
from .mesh import MeshParams
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


#: The one excitation this adapter has: ``driver`` calls ``SetGaussExcite`` and
#: nothing selects between calls. Stated again here rather than imported, because
#: ``Objects/analysis.GAUSSIAN`` sits behind ``import FreeCAD``; the two are held
#: together by a test, which is also what would catch a value being offered in
#: the property editor before there is a call to honour it.
GAUSSIAN = "Gaussian"


def _waveform(analysis: Any) -> None:
    """Refuse a waveform this adapter cannot produce.

    The document object offers one value, so a document built through the GUI
    cannot fail this. A file can: FreeCAD stores an enumeration's whole list in
    the document and restores it from there rather than from the class, so a
    document written with a longer list goes on offering it - and what it
    offers would otherwise be solved as a Gaussian without a word.
    """
    declared = str(analysis.Waveform)
    if declared != GAUSSIAN:
        raise TranslationError(
            f"{_label(analysis)!r}: Waveform is {declared!r}. openEMS is driven "
            f"here with a {GAUSSIAN} pulse covering the whole band, and this "
            f"adapter has no other excitation to offer. Set it to {GAUSSIAN}"
        )


def _smallest_response(analysis: Any) -> float:
    """The study's reading floor, from decibels to the magnitude in S it is.

    Amplitude and not power, because what it is weighed against is an error in
    S: twenty decibels a decade, the same decibels the response is plotted in.

    Zero is full scale and is what a study that has not thought about it says,
    so the ordinary run keeps the absolute bar. Above zero is refused rather
    than clamped - a passive device answers no more than one, so it is a typing
    slip, and a slip that silently became "full scale" would leave the property
    reading as though it had been honoured.
    """
    declared = float(analysis.SmallestResponse)
    if not math.isfinite(declared) or declared > 0.0:
        raise TranslationError(
            f"{_label(analysis)!r}: SmallestResponse is {declared:g} dB. It is "
            "how far down the response is read, so it is zero for full scale "
            "and negative below it"
        )
    return 10.0 ** (declared / 20.0)


def _boundary(solver: Any) -> tuple[str, ...]:
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
    return tuple(words)


def _absorber_cells(boundary: Sequence[str], depth: int) -> tuple[int, int, int]:
    """Absorber depth per axis, for the mesher.

    The mesher lays uniform cells at both ends of any axis that absorbs, because
    a graded absorber reflects. An axis walled by a conductor gets none - which
    matters: a waveguide meshed as though it absorbed sideways would grow past
    its own walls and its cutoff frequency would move.

    An axis absorbing on one face only still gets uniform cells on both. The cost
    is a few cells of grid in a corner of the model; the alternative is a
    per-face parameter the mesher does not have.
    """
    return tuple(  # type: ignore[return-value]
        depth if any(boundary[2 * dim + side].startswith("PML") for side in (0, 1)) else 0
        for dim in range(DIMENSIONS)
    )


def _padding(settings: Any) -> tuple[tuple[Any, Any], ...]:
    """Per-face domain padding, in the form :func:`~.write.domain` wants.

    ``Through`` is not "more air". It pulls the domain *in* so the absorber lands
    on the structure, which is what makes a transmission line infinite. Give a
    line air at its ends instead and it radiates off an open circuit, and every
    impedance read from it is contaminated by the reflection - so this is a
    per-face choice the user has to make, not something to infer.
    """
    faces = []
    for axis in AXIS_NAMES:
        pair = []
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
                        "face's Padding to Through to pull the domain in; a "
                        "negative count is not the way to ask for it"
                    )
                pair.append(count)
            else:
                raise TranslationError(
                    f"{_label(settings)!r}: Padding{axis}{name} is "
                    f"{mode!r}; expected 'Air' or 'Through'"
                )
        faces.append(tuple(pair))
    return tuple(faces)


def _wavelength(materials: Iterable[Material], frequency: Frequency) -> float:
    """Wavelength in millimetres in the slowest material, at the top of the band.

    The top of the band because that is where cells have to be smallest, and the
    slowest material because a wave slows by sqrt(epsilon * mu) inside it -
    meshing to the vacuum wavelength under-resolves it by exactly that factor.

    Permeability belongs in that product. ``mu`` is settable in the GUI and is
    passed to CSXCAD, so leaving it out solves a ferrite as a magnetic material
    and meshes it as a non-magnetic one - cells too coarse by sqrt(mu_r),
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

    ``ElementsPerWavelength`` counts elements across the wavelength *in the
    slowest material in the model* at the top of the band, never a length. A
    remembered millimetre value silently under-resolves the moment anything raises
    the permittivity or the frequency.

    What matters most is ``EdgeRefinement``, and it is easy to get wrong because
    the error it causes looks like ordinary discretisation error. At the same
    cell count, refining the conductor edge by 2 instead of 6 moves the
    extracted impedance by more than the microstrip gate's tolerance allows. The
    defaults follow openEMS' own ``MSL_Losses.m``: bulk at lambda/20, edges six
    times finer.
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
        # The coarsest cell anywhere: the bulk size in vacuum. Every dielectric
        # then asks for its own sqrt(epsilon) finer over its own span, and air
        # - which asks for nothing - gets this. What taking one lambda from
        # the slowest material in the model instead would cost is on
        # mesh.Region.size.
        #
        # It sizes the air faces of the domain as well as the sizing field -
        # air is what relaxes to it. A THROUGH face is sized by the material
        # that will be at the wall instead, because that is what the absorber
        # gets laid in; see write.domain.
        cap=_coarsest_cell(settings, frequency),
    )


def _coarsest_cell(settings: Any, frequency: Frequency) -> float:
    """The largest cell the grid will use anywhere, in mm.

    Read off the vacuum wavelength at the top of the band, so it needs no
    material and can be asked before the geometry has been read - which is what
    lets a conductor drawn without one be given a thickness measured against the
    grid that will hold it.
    """
    per_wavelength, _ = _check_mesh_policy(settings)
    return SPEED_OF_LIGHT / frequency.stop * units.MM_PER_M / per_wavelength


def _skin(settings: Any, frequency: Frequency) -> float:
    """How thick to make a conductor drawn with no thickness at all, in mm.

    A cell has to fit inside the metal or openEMS samples it into islands, and
    the cell a conductor's own edges are sized at is the metal resolution - so
    the thickness is the one whose *connection* demand is exactly that: the cell
    of that size has the thickness for its body diagonal, which is the worst
    direction a slab can be crossed in and so the one the criterion is stated at.

    Measured in vacuum, where that resolution is at its coarsest. A dielectric in
    the model only refines it, so this is the thickness that stays resolved
    whatever else is drawn - and it bounds what the skin can cost: its demand is
    the vacuum metal resolution, which no model's own metal resolution is coarser
    than, so a skin is never the finest thing asking.

    What it is not is free. A cross-section is read at points across a surface
    rather than over a span, so a wide skin holds the field down across its whole
    extent where it would otherwise have relaxed toward the coarsest cell. A
    conductor drawn thicker than the mesher's reach is read as no feature at all
    and costs less; that is the price of the thickness being invented here.
    """
    _, refinement = _check_mesh_policy(settings)
    return fits_inside(_coarsest_cell(settings, frequency) / refinement)


def _check_mesh_policy(settings: Any) -> tuple[float, float]:
    """Validate the mesh policy once, before anything reads it.

    Called from :func:`contents`, not only from :func:`_mesh_params`, and that
    placement is the point. ``_Context`` divides by ``ElementsPerWavelength`` to
    size ports, and it is built *before* mesh parameters exist - so validating
    inside ``_mesh_params`` would let a zero through to a ``ZeroDivisionError``
    two functions before the message meant to explain it. Policy checks belong
    at the one place every route passes through, for the same reason the driver
    re-runs pre-flight.

    Returns the two numbers it had to parse anyway.
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

    ``EnergyDecay`` at or above zero means "do not stop early", and that is
    the default. It reads like a disabled feature and it is the only reproducible
    setting: openEMS re-evaluates its energy criterion inside a branch gated on
    four seconds of wall clock, so an energy-terminated run stops at a step count
    that depends on machine load. Pre-flight warns when this is switched on.
    """
    decay = float(solver.EnergyDecay)
    with _model_fault(_label(solver)):
        return Termination(
            max_timesteps=int(solver.MaxTimesteps),
            end_criteria=0.0 if decay >= 0 else 10 ** (decay / 10.0),
        )


def _threads(solver: Any) -> int:
    """How many threads openEMS may use. Zero means "as many as it likes".

    Read through its own function for the reason :func:`_model_fault` records:
    it is the one value a user can type that lands in ``Problem``, whose
    construction is the one place that context manager must not be wrapped
    around. Checking the property before it gets there costs nothing and catches
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

    Read through a function rather than inline because two paths want it and
    only one of them builds a :class:`Problem`: the mesh preview scales its
    reported timestep by the factor without ever constructing an envelope, so
    the envelope's own invariant does not stand between the user and the number
    on screen. Without this, typing 2 into ``TimestepFactor`` and pressing
    Update Mesh reports a CFL bound at twice the real one - captioned as the
    bound, in green - and only Run refuses it. Two paths disagreeing about
    whether a value is legal, with the permissive one drawing the picture.

    A :class:`TranslationError` and not the :class:`EnvelopeError` underneath,
    through :func:`_model_fault`, which is where that rule is written down.
    Typing 2 into a property field is not a crash. The bound itself stays in
    ``model.check_timestep_factor``, stated once.
    """
    with _model_fault(_label(solver)):
        return check_timestep_factor(float(solver.TimestepFactor))
