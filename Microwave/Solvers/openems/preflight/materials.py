# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checks about what the model is made of.

These checks ask whether a material is one this adapter can build, whether a
loss figure quoted at one frequency still stands in over this band, whether any
one figure could, whether two solids claim the same space, and whether a
conductor is standing on a thickness the drawing never carried.

A conducting sheet is asked separate questions that are easy to read as one:
whether the shape it is bound to spans a surface at all, whether it is thin
against the cell that has to hold it, and whether openEMS has fitted
surface-impedance coefficients for it at the top of the band. They run in that
order because the first decides whether the others reach the object. The cell
measurement reaches only a solid that is flat on an axis, so a sheet given a
volume is invisible to it.
"""

from __future__ import annotations

import math

import numpy as np

from ....units import VACUUM_PERMEABILITY
from ..capabilities import Capabilities
from ..model import Problem, Solid
from .finding import _ON_THE_GRID, REFUSE, SUBSTITUTE, WARN, Finding, hz


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
    Nothing downstream sees this: the envelope hands the engine a conductivity
    and the engine uses it. ``measured_at`` is the only record of where that
    number was true, and this check is what reads it.

    This warns and never refuses. An engineer who knows their laminate is flat
    across the band is right, so the check states its point and stops.
    """
    centre = problem.frequency.center
    findings = []
    for material in problem.materials:
        # Tested on the term rather than on the kind. The two agree, since
        # ``Material`` refuses a kappa on any kind the engine would not be
        # handed it for, and the term is what carries the approximation.
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


def _check_the_band_fits_one_conductivity(problem: Problem) -> list[Finding]:
    """A band no single conductivity covers, however well the number was quoted.

    The check above asks whether a loss tangent was quoted where it is being
    used. This one asks what survives a yes. ``kappa`` is fixed for the run, so
    the loss tangent it amounts to is the declared one scaled by
    ``f_centre / f``, and a wide enough band sets its own ends far from centre.

    The scaling is that frequency ratio alone, since the permittivity and the
    loss tangent both divide out. Every lossy material in a model is therefore
    off by the same factor, and one sentence merges to name all of them.

    Only the bottom of the band is looked at, and only the bottom can be. The
    centre is the mean of the two ends, so the top is never a factor of two from
    it and cannot reach ``FAR``. The bottom has no such bound and runs away as
    the band widens.

    That bounds the ratio rather than the damage. Attenuation from a fixed
    conductivity does not vary with frequency at all, while a low-loss
    dielectric's rises in proportion to it, and the two are made equal at the
    centre. The band's two ends are therefore wrong by the same amount of loss,
    over-stated below the centre and under-stated above it. The message states
    both, because the top is where a dielectric's loss is largest and so where
    that same amount is the smaller share of the answer.

    This warns and never refuses. Narrowing the band changes the question being
    asked rather than answering it. What would answer it is a Debye fit, whose
    poles openEMS integrates itself, and this adapter does not write one.
    """
    frequency = problem.frequency
    overstated = frequency.center / frequency.start
    if overstated < FAR:
        return []
    message = (
        f"a fixed conductivity is the loss tangent it was built from at "
        f"{hz(frequency.center)} alone, and models one that goes as 1/f either "
        f"side. This study runs down to {hz(frequency.start)}, where the "
        f"modelled loss tangent is {overstated:.3g} times the declared one - "
        f"and roughly half the declared one at the top of the band. Solve a "
        f"narrower band, or split this one into studies"
    )
    return [
        Finding(WARN, material.name, message)
        for material in problem.materials
        if material.kappa > 0
    ]


def _same_space(one: Solid, other: Solid) -> bool:
    """Whether two boxes stand in the same place, to ``_ON_THE_GRID``.

    The test is not equality. Two coordinates a part in 1e16 apart, which is
    what a stackup measured from the top on one layer and from the bottom on the
    other produces, build the structure identical numbers build, because every
    cell centre inside one box is inside the other. A tolerance below any length
    this workbench meshes keeps the check about the structure rather than about
    the arithmetic that reached it.

    A triangulated solid answers no. Its corners bound the shape rather than
    being it, so equal corners are not two objects in one place. A coil and the
    former it is wound on share a bounding box exactly and touch nowhere.
    Refusing that pair, which is what this finding does for two materials, would
    refuse an ordinary model to catch a fault there is no evidence of.
    """
    if one.is_mesh or other.is_mesh:
        return False
    return all(
        abs(a - b) <= _ON_THE_GRID
        for corners in ((one.lower, other.lower), (one.upper, other.upper))
        for a, b in zip(*corners)
    )


def _check_coincident_solids(problem: Problem) -> list[Finding]:
    """Two solids in the same place, of one material or of two.

    openEMS reports both cases with the same sentence: the box that gets no
    cells is reported as ``Warning: Unused primitive (type: Box) detected in
    property: Copper!`` (``CSProperties::WarnUnusedPrimitves``, called for
    every run at ``openEMS/openems.cpp:1335``). A CSXCAD property is a
    material, so the only name in that line is the material's, and the object
    it meant appears nowhere. A healthy run prints the same line for a benign
    reason, since ``Unused primitive`` also has one. The engine cannot tell
    these apart. This check can.

    Two solids of one material are a drawing to tidy up rather than a fault:
    the same conductor drawn twice is one conductor, and the run is right. So
    this warns, and names the two objects the engine's line does not.

    Two solids of different materials are a wrong structure, so this refuses.
    One space cannot be made of two materials, and the engine resolves the
    contradiction rather than reporting it:
    ``ContinuousStructure::GetPropertyByCoordPriority``
    (``CSXCAD/src/ContinuousStructure.cpp:276``) gives every cell to the
    highest priority covering it, and to the property added first where they
    tie. A board bound to both a laminate and a copper is meshed as solid
    copper, and the run completes and reports numbers for it. Binding geometry
    twice is one selection away, so this names the pair.
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
                    # No numeral here or below. Identical sentences are merged
                    # into one line naming every object that said them, so
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
        # label are one object bound twice rather than two objects in a heap.
        # "Substrate occupies the same space as 'Substrate'" would not help the
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


def _check_a_thickness_was_invented(problem: Problem) -> list[Finding]:
    """A conductor drawn as a surface, and the metal built off it.

    This is reported even where the drawing meant a surface. The thickness
    follows the cell the metal is meshed at, so it moves with the band and the
    mesh policy, and the document carries that length nowhere a reader could
    look.

    A conducting sheet is left out. Its length is not free, being what carries
    the shape out of the surface-impedance model, and the remedy offered here -
    close the shape around its drawn thickness - produces the shape the adapter
    refuses. :func:`_check_a_conducting_sheet_spans_a_surface` names that object
    instead.
    """
    sheets = {
        material.name for material in problem.materials if material.kind == "conducting_sheet"
    }
    return [
        Finding(
            SUBSTITUTE,
            solid.name,
            f"it is drawn as a surface carrying no thickness, and {solid.thickened:.4g} "
            "mm of metal has been built off it. The field inside a conductor is "
            "zero, so a skin and a slab solve alike and the length is free - but "
            "a surface that was meant to close and did not looks exactly the "
            "same here. If this one was meant as a solid, close it and give it "
            "the thickness you drew",
        )
        for solid in problem.solids
        if solid.thickened and solid.material not in sheets
    ]


def _check_a_conducting_sheet_spans_a_surface(problem: Problem) -> list[Finding]:
    """A conducting sheet that is not a surface is solved as a perfect conductor.

    openEMS applies the surface-impedance model only where the primitive
    carrying the property spans exactly two axes. The test is what the shape
    spans rather than how thin it is, so a volume fails it, and so do a line and
    a point. Everything else is written as a perfect conductor, with the
    conductivity and the thickness discarded together, and the run finishes.

    A sheet gets a volume either from the mesher, where a conductor is drawn as
    a surface it cannot lay flat, or from the user, by binding a sheet material
    to something drawn solid. The material alone distinguishes neither case from
    a working sheet. The geometry does.

    A port's own trace never reaches this check. It is flattened onto its
    excitation coordinate as it is built, so it spans two axes by construction.
    """
    sheets = {
        material.name for material in problem.materials if material.kind == "conducting_sheet"
    }
    findings = []
    for solid in problem.solids:
        if solid.material not in sheets:
            continue
        spanned = sum(1 for dim in range(3) if solid.lower[dim] != solid.upper[dim])
        if spanned == 2:
            continue
        if spanned < 2:
            fault = f"it spans {spanned} of the three axes, so it has no area to carry a surface"
            remedy = "Draw the conductor as the face it is meant to be"
        elif solid.thickened:
            fault = (
                "the drawing carried no surface this could be laid flat on, so "
                f"{solid.thickened:.4g} mm of metal was built off it to close it"
            )
            remedy = (
                "A conductor that curves out of one plane cannot carry a surface "
                "impedance; bind it to a perfect conductor and take the answer as "
                "lossless"
            )
        else:
            fault = "it is drawn with thickness on every axis"
            remedy = (
                "Draw the conductor as a flat face - a conducting sheet carries "
                "its thickness as a loss property rather than as geometry - or "
                "bind it to a perfect conductor"
            )
        findings.append(
            Finding(
                REFUSE,
                solid.name,
                f"it is bound to {solid.material!r}, a conducting sheet, and "
                f"{fault}. openEMS applies its surface-impedance model only to a "
                "shape spanning two axes; this one it would write as a perfect "
                "conductor, discarding the conductivity and the thickness both, "
                f"and say so only in its own output. {remedy}",
            )
        )
    return findings


def _check_sheet_thickness(problem: Problem) -> list[Finding]:
    """A conducting sheet thicker than a cell is not a sheet.

    This has a check of its own because openEMS will not stop for it. Its
    surface-impedance model fits a rational approximation whose validity scales
    as ``1/thickness**2``. Outside that range openEMS prints a warning to
    stderr, clamps to its last tabulated coefficients and keeps going. The
    result looks like a solve and is wrong. A unit slip lands exactly there,
    since CSXCAD wants this field in metres while everything around it is in
    grid units, and this check is what catches it.
    """
    findings = []
    for material in problem.materials:
        if material.kind != "conducting_sheet":
            continue

        # Compared against the cells the sheet lies in, along the axis it is
        # thin in, rather than against the finest cell anywhere in the model.
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


#: How far a frequency may sit from band centre before the fixed conductivity
#: built from a loss tangent stops standing in for it. Two octaves.
#:
#: Two frequencies are held to it, for reasons of different weight. The
#: frequency the loss tangent was quoted at is a question about the material:
#: two octaves away, the material's own drift is small next to the spread
#: between one sheet of FR-4 and the next, so inside that the number still
#: describes the laminate. The bottom of the band is a question about the
#: conversion, and there the ratio is the error itself - at this threshold the
#: model carries four times the declared loss tangent, which no laminate's
#: spread excuses.
#:
#: The band question is therefore the more loosely held of the two, and it is
#: held to this figure anyway. A second constant would be a second thing to tune
#: with no better argument behind its value, and one number that moves both
#: states plainly that this is a judgement about how far is too far. Below it
#: the band check is silent while the model runs up to fourfold lossy at the
#: bottom.
#:
#: The same figure lets that check look only downwards. The top of a band is
#: never a factor of two from centre, and a threshold of two octaves is out of
#: its reach.
#:
#: ``Gui.material_picker`` reads this too, and shows the same thing about a
#: catalog entry before there is a model to translate.
FAR = 4.0


#: The largest ``Omega = w/w0`` openEMS holds fitted coefficients for: the last
#: entry of ``omega_stop`` in ``FDTD/extensions/cond_sheet_parameter.h``, a
#: generated file its own header marks *"Do not change"*.
_SHEET_OMEGA_MAX = 2556680.79


def _check_sheet_fits_the_surface_impedance_model(problem: Problem) -> list[Finding]:
    """A sheet openEMS has no fitted coefficients for at the top of the band.

    ``operator_ext_conductingsheet.cpp`` forms ``w0 = 8 / (sigma t^2 mu0)`` and
    takes the first tabulated ``omega_stop`` above ``Omega = 2 pi f_max / w0``.
    Past the last entry it prints *"conductor thickness, conductivity or max.
    simulation frequency of interest is too high"* and clamps to the last
    coefficient set, returning a wrong answer in the shape of a solve. The
    check above guards a failure of the same shape against a different bound.

    ``Omega`` is the skin depth in another form. With
    ``delta^2 = 2/(w mu sigma)`` it is exactly ``(t / 2 delta)^2``, so the bound
    is ``t <= 3198 delta`` at the top of the band. The neighbouring guard
    compares thickness against the cell and so could never have caught this.
    Ordinary foil at ordinary frequencies sits inside the fit with room to
    spare, so a complaint from it reads like a silent failure and is openEMS
    being right.
    """
    findings: list[Finding] = []
    top = float(problem.frequency.stop)
    for material in problem.materials:
        if material.kind != "conducting_sheet":
            continue
        if material.thickness <= 0 or material.conductivity <= 0:
            continue

        thickness = material.thickness * problem.length_unit
        omega = (
            2.0 * math.pi * top * VACUUM_PERMEABILITY * material.conductivity * thickness**2 / 8.0
        )
        if omega <= _SHEET_OMEGA_MAX:
            continue

        skin = math.sqrt(2.0 / (2.0 * math.pi * top * VACUUM_PERMEABILITY * material.conductivity))
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
