# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checks about what the model is made of.

Whether a material is one this adapter can build, whether a loss figure quoted
at one frequency still stands in over this band, and whether two solids claim
the same space.

A conducting sheet is asked two separate questions, and the pair is easy to read
as one: whether it is thin against the *cell* that has to hold it, and whether
openEMS has fitted surface-impedance coefficients for it at the top of the band.
Neither implies the other, and only the first is about the mesh.
"""

from __future__ import annotations

import math

import numpy as np

from ..capabilities import Capabilities
from ..model import Problem, Solid
from .finding import _ON_THE_GRID, REFUSE, WARN, Finding, hz


def _check_materials(problem: Problem, caps: Capabilities) -> list[Finding]:
    findings = []
    for material in problem.materials:
        if not caps.supports_material(material.kind):
            findings.append(
                Finding(
                    REFUSE,
                    material.name,
                    f"material kind {material.kind!r} is not supported; "
                    f"this adapter handles {sorted(caps.materials)}",
                )
            )
        if material.kind == "conducting_sheet" and material.thickness <= 0:
            findings.append(
                Finding(
                    WARN,
                    material.name,
                    "a conducting sheet with zero thickness has no surface "
                    "impedance, so it will behave as a perfect conductor",
                )
            )
    return findings


def _check_the_loss_was_measured_in_this_band(problem: Problem) -> list[Finding]:
    """A conductivity built from a loss tangent that was quoted somewhere else.

    ``kappa`` is one number for the whole run, so it reproduces the loss tangent
    it came from at exactly one frequency and falls away as 1/f either side.
    Nothing downstream can tell: the envelope hands the engine a conductivity
    and the engine uses it. ``measured_at`` is the only record of where that
    number was true, and this is what reads it.

    Warns, never refuses. An engineer who knows their laminate is flat across
    the band is right, so this says its piece and stops.
    """
    centre = problem.frequency.center
    findings = []
    for material in problem.materials:
        # The term, not the kind. They agree - ``Material`` refuses a kappa on
        # any kind the engine would not be handed it for - and it is the term
        # that carries the approximation.
        if material.kappa <= 0:
            continue
        if material.measured_at <= 0:
            findings.append(
                Finding(
                    WARN,
                    material.name,
                    f"a conductivity of {material.kappa:.4g} S/m with no record "
                    "of the frequency the loss tangent behind it was quoted at. "
                    f"It is held fixed, so it is that loss tangent only at "
                    f"{hz(centre)}; record where it was measured",
                )
            )
            continue
        ratio = max(material.measured_at, centre) / min(material.measured_at, centre)
        if ratio >= FAR:
            findings.append(
                Finding(
                    WARN,
                    material.name,
                    f"its loss is quoted at {hz(material.measured_at)}, and this "
                    f"study is centred at {hz(centre)}, where openEMS fixes the "
                    "conductivity it builds from that number",
                )
            )
    return findings


def _same_space(one: Solid, other: Solid) -> bool:
    """Whether two boxes stand in the same place, to ``_ON_THE_GRID``.

    Not equality. Two coordinates a part in 1e16 apart - what a stackup
    measured from the top on one layer and from the bottom on the other
    produces - build the structure identical numbers build, because every
    cell centre inside one box is inside the other. A tolerance below any
    length this workbench meshes is what makes the check about the structure
    rather than about the arithmetic that reached it.
    """
    return all(
        abs(a - b) <= _ON_THE_GRID
        for corners in ((one.lower, other.lower), (one.upper, other.upper))
        for a, b in zip(*corners)
    )


def _check_coincident_solids(problem: Problem) -> list[Finding]:
    """Two solids in the same place, of one material or of two.

    Both come back from openEMS as the same sentence: the box that gets no
    cells is reported as ``Warning: Unused primitive (type: Box) detected in
    property: Copper!`` (``CSProperties::WarnUnusedPrimitves``, called for
    every run at ``openEMS/openems.cpp:1335``). A CSXCAD property is a
    *material*, so the only name in that line is the material's, and which
    object it meant appears nowhere. It is also the same line a healthy run
    prints for a benign reason, since ``Unused primitive`` also has one. The
    engine cannot tell these apart. This can, and that is the whole point of
    it.

    **One material** is a drawing to tidy up rather than a fault: the same
    conductor drawn twice is one conductor, and the run is right. So it warns,
    and names the two objects the engine's line does not.

    **Two materials** is a wrong structure, so it refuses. One space cannot be
    made of two materials, and the engine resolves the contradiction rather
    than reporting it: ``ContinuousStructure::GetPropertyByCoordPriority``
    (``CSXCAD/src/ContinuousStructure.cpp:276``) gives every cell to the
    highest priority covering it, and to the property added first where they
    tie. A board bound to both a laminate and a copper is meshed as solid
    copper and answers like a board. Binding geometry twice is one selection
    away, so this names the pair.
    """
    findings = []
    seen: list[Solid] = []
    for solid in problem.solids:
        first = next((other for other in seen if _same_space(other, solid)), None)
        seen.append(solid)
        if first is None:
            continue
        subject = solid.label or solid.material
        occupies = f"occupies the same space as {first.label or first.material!r}"
        if first.material == solid.material:
            findings.append(
                Finding(
                    WARN,
                    subject,
                    # No numeral, here or below: identical sentences are merged
                    # into one line naming every object that said them, and
                    # "both" would be false the moment a third solid shares the
                    # space.
                    f"{occupies}, which is also {solid.material!r}. openEMS "
                    "discretises one box in a shared space and drops the rest "
                    "with an 'Unused primitive' warning that names only the "
                    "material, so this says which objects it meant. Delete the "
                    "spare, or move it if they were meant to be different",
                )
            )
            continue
        # Labels are unique in a FreeCAD document, so two solids carrying one
        # is one object bound twice rather than two objects in a heap - and
        # "Substrate occupies the same space as 'Substrate'" is no help to the
        # person who did it.
        clash = (
            f"is bound to both {first.material!r} and {solid.material!r}"
            if solid.label and solid.label == first.label
            else f"{occupies}, and the materials differ: this is "
            f"{solid.material!r}, that is {first.material!r}"
        )
        findings.append(
            Finding(
                REFUSE,
                subject,
                f"{clash}. Every cell there goes to one of them - the higher "
                "priority, or the one written first where they tie - so the "
                "run would describe a structure made of that one throughout, "
                "and say so only as an 'Unused primitive' line naming the "
                "loser's material. Bind the geometry once",
            )
        )
    return findings


def _check_sheet_thickness(problem: Problem) -> list[Finding]:
    """A conducting sheet thicker than a cell is not a sheet.

    Worth a check of its own because openEMS will not stop for it. Its
    surface-impedance model fits a rational approximation whose validity scales
    as ``1/thickness**2``; outside that range it prints a warning to stderr,
    clamps to its last tabulated coefficients and keeps going. The result looks
    like a solve and is wrong. A unit slip - CSXCAD wants this field in metres
    while everything around it is in grid units - lands exactly there, and
    this is what catches it.
    """
    findings = []
    for material in problem.materials:
        if material.kind != "conducting_sheet":
            continue

        # Compared against the cells the sheet actually lies in, along the axis
        # it is thin in - not against the finest cell anywhere in the model.
        # The global minimum makes this check strictly stronger, so one fine
        # feature elsewhere on the board would refuse a perfectly valid sheet.
        smallest = _cells_at_sheet(problem, material.name)
        if smallest is None:
            continue
        if material.thickness > smallest:
            findings.append(
                Finding(
                    REFUSE,
                    material.name,
                    f"its modelled thickness ({material.thickness:g}) exceeds "
                    f"the smallest cell in the grid ({smallest:g}), so it is "
                    "not thin compared to the mesh and the surface-impedance "
                    "model does not apply. openEMS will not refuse this - it "
                    "clamps its fit and returns a plausible-looking wrong "
                    "answer. Model it as a solid, or check the units",
                )
            )
    return findings


#: How far a loss tangent's quoted frequency may sit from band centre, as a
#: ratio either way, before the fixed conductivity built from it stops standing
#: in for it. Two octaves: inside that, this approximation's error and the
#: laminate's own drift are both small next to the spread between one sheet of
#: FR-4 and the next - and a warning that fires on ordinary models is one
#: people learn to ignore.
#:
#: Read by ``Gui.material_picker`` too, which shows the same thing about a
#: catalog entry before there is a model to translate.
FAR = 4.0


#: The largest ``Omega = w/w0`` openEMS holds fitted coefficients for: the last
#: entry of ``omega_stop`` in ``FDTD/extensions/cond_sheet_parameter.h``, a
#: generated file its own header marks *"Do not change"*.
_SHEET_OMEGA_MAX = 2556680.79


#: Vacuum permeability, for the skin depth. SI, unlike everything around it.
_MU0 = 4.0e-7 * math.pi


def _check_sheet_fits_the_surface_impedance_model(problem: Problem) -> list[Finding]:
    """A sheet openEMS has no fitted coefficients for at the top of the band.

    ``operator_ext_conductingsheet.cpp`` forms ``w0 = 8 / (sigma t^2 mu0)`` and
    takes the first tabulated ``omega_stop`` above ``Omega = 2 pi f_max / w0``.
    Past the last entry it prints *"conductor thickness, conductivity or max.
    simulation frequency of interest is too high"* and **clamps to the last
    coefficient set** - a solve-shaped wrong answer, which is the same way the
    check above fails and the reason both exist.

    ``Omega`` is the skin depth in disguise. With ``delta^2 = 2/(w mu sigma)``
    it is exactly ``(t / 2 delta)^2``, so the bound is ``t <= 3198 delta`` at
    the top of the band - which is why the neighbouring guard, comparing
    thickness against the *cell*, could never have caught this. Ordinary foil
    at ordinary frequencies sits inside the fit with room to spare, so a
    complaint from it reads like a silent failure and is openEMS being right.
    """
    findings = []
    top = float(problem.frequency.stop)
    if top <= 0:
        return findings

    for material in problem.materials:
        if material.kind != "conducting_sheet":
            continue
        if material.thickness <= 0 or material.conductivity <= 0:
            continue

        thickness = material.thickness * problem.length_unit
        omega = 2.0 * math.pi * top * _MU0 * material.conductivity * thickness**2 / 8.0
        if omega <= _SHEET_OMEGA_MAX:
            continue

        skin = math.sqrt(2.0 / (2.0 * math.pi * top * _MU0 * material.conductivity))
        limit = 2.0 * skin * math.sqrt(_SHEET_OMEGA_MAX) / problem.length_unit
        findings.append(
            Finding(
                REFUSE,
                material.name,
                f"at {top / 1e9:g} GHz it is {thickness / skin:.0f} skin depths "
                f"thick, past the last coefficient set openEMS has fitted for a "
                f"conducting sheet. It does not refuse this - it clamps to its "
                f"last row and keeps going. The limit here is "
                f"{limit:.3g} in grid units; model it as a solid, or lower the "
                f"top of the band",
            )
        )
    return findings


def _cells_at_sheet(problem: Problem, material: str) -> float | None:
    """Smallest cell straddling a sheet of ``material``, along its thin axis.

    ``None`` when nothing in the model is made of it, in which case there is
    nothing to judge.
    """
    sizes = []
    for solid in problem.solids:
        if solid.material != material:
            continue
        for dim in range(3):
            if solid.lower[dim] != solid.upper[dim]:
                continue
            lines = problem.grid[dim]
            index = int(np.argmin(np.abs(lines - solid.lower[dim])))
            neighbours = [
                float(lines[j] - lines[j - 1]) for j in (index, index + 1) if 0 < j < len(lines)
            ]
            if neighbours:
                sizes.append(min(neighbours))

    for port in problem.ports:
        if port.metal != material or not port.lays_conductor():
            continue
        low, high = port.trace_region()
        for dim in range(3):
            if low[dim] != high[dim]:
                continue
            lines = problem.grid[dim]
            index = int(np.argmin(np.abs(lines - low[dim])))
            neighbours = [
                float(lines[j] - lines[j - 1]) for j in (index, index + 1) if 0 < j < len(lines)
            ]
            if neighbours:
                sizes.append(min(neighbours))

    return min(sizes) if sizes else None
