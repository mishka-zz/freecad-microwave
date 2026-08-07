# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Read a FreeCAD document and build this adapter's :class:`Problem`.

This is the openEMS adapter's front door. Everything upstream of it is
solver-neutral - materials, ports, mesh policy, all expressed as document
objects that know nothing about FDTD - and everything downstream is openEMS'
own. The translation belongs to the adapter, not to a shared bridge: NEC2 will
read the same document and ask completely different questions of it.

**Imports no FreeCAD.** Document objects are attribute bags, and every property
this module reads is reachable by duck typing. So the whole translation is
unit-testable against plain fakes, with no CAD kernel and no solver anywhere
near it - and the adapter stays importable on a machine that has neither, which
is what makes "which solvers could run this model?" a question the workbench
can answer offline.

What it refuses
---------------

Loudly, and naming the object, per the workbench's capability rules:

* geometry that is not an axis-aligned box - verified against the shape's own
  volume, not assumed from its type;
* a port kind this adapter has no builder for - reachable the day another
  adapter grows one, and refused by name here rather than mid-build;
* dispersive materials, which need a fitted pole set this adapter cannot write;
* a trace whose conductor material the document never states.

A silent substitution is never an option. The failure mode this guards against
is not a crash, it is a plausible number: a rotated box staircased without
comment, or a port laid on a dielectric, both solve happily and both are wrong.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

from ... import portbox

# ``X``, ``Y``, ``Z`` - the *document's* spelling, from portbox, not the
# envelope's lowercase one from model. Everything this module names an axis for
# is document-facing: a message about PropagationAxis, whose value the user sees
# as "X", or a property name like PaddingXMin. Take model's copy instead and the
# messages spell an axis differently from the property they name. The lowercase
# copy stays in model, where it spells envelope keys.
from ...portbox import AXIS_NAMES

# MeshError is re-exported deliberately. Most refusals a user can cause are
# raised here as TranslationError, but the ones that need MeshParams to decide
# - geometry outside the domain, anchors below the cell floor, a refinement
# region asking to coarsen - are raised by the mesher and reach the caller
# through this module. They are the same thing to whoever drew the model: the
# model is refused and the message names the object. Callers that handle one
# should handle both, or a modelling mistake is reported as an internal error.
from .mesh import MeshError as MeshError
from .mesh import MeshParams, SizingRegion
from .model import (
    CONDUCTOR_KINDS,
    DIMENSIONS,
    SPEED_OF_LIGHT,
    THROUGH,
    EnvelopeError,
    Frequency,
    Material,
    Port,
    Problem,
    Solid,
    Termination,
    canonical,
    check_mode,
    check_timestep_factor,
)
from .rectilinear import RectilinearError, rectangles
from .write import grid_from, plan_grid, plan_mesh, structure_bounds

VACUUM_PERMITTIVITY = 8.854_187_812_8e-12

#: Re-exported, not redefined: see :data:`Microwave.portbox.FLATNESS` for the
#: number and the reasoning. A second copy here is a fact that drifts.
FLATNESS = portbox.FLATNESS

#: Layering. Dielectrics underlay metals, and a port's own conductor (priority
#: 10, from :class:`~.model.Port`) sits over both - ``MSLPort`` lays its strip
#: across the same span the user's trace occupies, and the strip must win.
DIELECTRIC_PRIORITY = 0
METAL_PRIORITY = 5

#: Document ``MaterialType`` to the adapter's material kinds. Absent entries are
#: refusals, not defaults.
_MATERIAL_KINDS = {
    "Dielectric": "dielectric",
    "PEC": "pec",
    "ConductingSheet": "conducting_sheet",
}

#: The value of ``EMPort.ReferencedTo`` that means "against the port's own
#: impedance". Spelled here as well as in ``Objects/ports.py`` because this
#: module imports no FreeCAD and that one is a document object - the same
#: reason ``_MATERIAL_KINDS`` restates an enumeration above.
_PORT_IMPEDANCE = "Port impedance"

#: Document boundary names to openEMS'. ``PML`` takes its depth from
#: ``PMLCells`` and is spelled per-face.
_BOUNDARY_WORDS = {"PEC": "PEC", "PMC": "PMC", "Mur": "MUR"}

_AXES = {
    "X": (0, 1),
    "-X": (0, -1),
    "Y": (1, 1),
    "-Y": (1, -1),
    "Z": (2, 1),
    "-Z": (2, -1),
}


class TranslationError(Exception):
    """The document does not describe something this adapter can run."""


@contextmanager
def _model_fault(label: str = "") -> Iterator[None]:
    """Re-raise the envelope's own validation as the document's problem.

    Everything the envelope validates arrived from a property field, so an
    :class:`EnvelopeError` raised while translating is something the user
    typed, not a crash - and the panel splits on exactly that, showing a
    :class:`TranslationError` as the model's problem and routing anything else
    through a catch-all that prints "Internal error" and a traceback. A
    traceback reads as a bug in the workbench, and a message about something
    the user typed must not look like one.

    **Only around a constructor whose every field is a property**, which is
    ``Material``, ``Termination`` and ``Frequency``. Not ``Problem``, which
    spans ``plan_grid`` and every structural invariant, so wrapping it would
    report a meshing fault as something the user typed. Not ``Port`` either:
    its ``__post_init__`` validates geometry that ``portbox`` computed. The
    values a user can put into a port are read through
    :func:`_reference_impedance`, :func:`_resistance` and
    :func:`~.model.check_mode` instead, and :func:`_threads` does the same for
    what lands in ``Problem``.

    The catch stays narrow: widening it to ``Exception`` turns every
    ``AttributeError`` inside a builder into "cannot translate this model",
    with the traceback thrown away.

    ``label`` is omitted where the envelope's own message already names the
    subject, which :class:`~.model.Material`'s do. Prefixing those gives
    *"'FR4': material 'FR4': relative permittivity 0.5 is below 1"*.
    """
    try:
        yield
    except EnvelopeError as error:
        prefix = f"{label!r}: " if label else ""
        raise TranslationError(f"{prefix}{error}") from error


# ---------------------------------------------------------------------------
# Reading properties off document objects
# ---------------------------------------------------------------------------


def _kind(obj: Any) -> str:
    """The document object's class, by its Python proxy.

    FreeCAD's ``App::FeaturePython`` gives every workbench object the same
    ``TypeId``, so the proxy class is the only thing that distinguishes an
    ``EMPortMicrostrip`` from an ``EMSolverOpenEMS``.

    A copy of ``Objects.kinds.kind_of``, and the only one. It cannot import that
    one, because ``Objects/__init__.py`` imports FreeCAD and this module's whole
    claim is that it does not.
    """
    proxy = getattr(obj, "Proxy", None)
    return type(proxy).__name__ if proxy is not None else ""


def _label(obj: Any) -> str:
    return str(getattr(obj, "Label", None) or getattr(obj, "Name", "?"))


def _value(quantity: Any) -> float:
    """A property as a plain float, whether or not it carries a unit."""
    return float(getattr(quantity, "Value", quantity))


def _axis(name: str, subject: str) -> tuple[int, int]:
    """``'-Z'`` to ``(2, -1)``: an axis index and which way along it."""
    try:
        return _AXES[name]
    except KeyError:
        raise TranslationError(
            f"{subject}: {name!r} is not an axis; expected one of {sorted(_AXES)}"
        ) from None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in millimetres, corners sorted."""

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    @property
    def extents(self) -> tuple[float, float, float]:
        return tuple(b - a for a, b in zip(self.lower, self.upper))  # type: ignore[return-value]

    def middle(self, dim: int) -> float:
        return 0.5 * (self.lower[dim] + self.upper[dim])

    def nearest(self, dim: int, to: float) -> float:
        """Whichever face along ``dim`` lies closer to ``to``."""
        low, high = self.lower[dim], self.upper[dim]
        return low if abs(low - to) <= abs(high - to) else high

    def is_flat(self, dim: int) -> bool:
        return self.extents[dim] <= FLATNESS

    def as_pair(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The form ``Microwave.portbox`` takes: two corner triples."""
        return (self.lower, self.upper)


def _bounds(bound_box: Any) -> Box:
    return Box(
        lower=(float(bound_box.XMin), float(bound_box.YMin), float(bound_box.ZMin)),
        upper=(float(bound_box.XMax), float(bound_box.YMax), float(bound_box.ZMax)),
    )


def _union(boxes: Sequence[Box]) -> Box:
    return Box(
        lower=tuple(min(b.lower[d] for b in boxes) for d in range(DIMENSIONS)),  # type: ignore[arg-type]
        upper=tuple(max(b.upper[d] for b in boxes) for d in range(DIMENSIONS)),  # type: ignore[arg-type]
    )


def solid_boxes(obj: Any) -> list[tuple[str, Box]]:
    """The boxes a whole document object occupies, labelled."""
    shape = getattr(obj, "Shape", None)
    if shape is None:
        raise TranslationError(
            f"{_label(obj)!r} has no shape, so there is nothing to mesh. "
            "Material bindings must reference solid geometry"
        )
    return _boxes_of(shape, _label(obj))


def _boxes_of(shape: Any, label: str) -> list[tuple[str, Box]]:
    """The boxes a shape occupies - after proving a rectilinear grid holds it.

    A bounding box exists for every shape, which is exactly the problem: taking
    one from a cylinder, a fillet or a rotated block produces a box-shaped answer
    to a question that had none, and openEMS would solve it without complaint.
    So the shape's own volume is compared against its bounding box's.

    The test is on the *measurement*, not the type. ``Part::Box`` is the usual
    case but not the only honest one - a boolean result or an extruded
    rectangle that happens to be a box passes, and a ``Part::Box`` rotated 30
    degrees does not. A shape flat in one axis is judged by area instead, so a
    ground plane drawn as a face is a first-class citizen rather than an
    exception.

    Falling short of the box is not the end of it. A **flat** shape whose edges
    are all axis-aligned is held exactly by a rectilinear grid once it is cut
    into rectangles, and a layout out of DXF or the Sketcher is exactly that
    shape - so it is cut here rather than handed back to the user to redraw as
    primitives. The cut is exact, and :func:`_cut_up` proves it by area.
    """
    box = _bounds(shape.BoundBox)
    extents = box.extents
    flat = [dim for dim in range(DIMENSIONS) if box.is_flat(dim)]

    if len(flat) > 1:
        raise TranslationError(
            f"{label!r} is flat in "
            f"{' and '.join(AXIS_NAMES[d] for d in flat)}, so it is a line or a "
            "point. A region needs area"
        )

    if flat:
        measured = float(shape.Area)
        expected = math.prod(extents[d] for d in range(DIMENSIONS) if d not in flat)
        what = "area"
    else:
        measured = float(shape.Volume)
        expected = extents[0] * extents[1] * extents[2]
        what = "volume"

    if expected > 0 and math.isclose(measured, expected, rel_tol=1e-6):
        return [(label, box)]

    if flat:
        return _cut_up(shape, label, box, flat[0], measured)

    filled = measured / expected if expected > 0 else 0.0
    raise TranslationError(
        f"{label!r} is not an axis-aligned box: its {what} is "
        f"{measured:.6g}, but its bounding box's is {expected:.6g} "
        f"({filled:.1%}). The Yee grid is rectilinear, so openEMS would "
        "staircase this shape without saying so. A rotation and a curve fail "
        "this, and so does an L or a notch cut by a boolean. Draw it as several "
        "boxes bound to one material, or as a flat sheet - a sheet whose edges "
        "are all axis-aligned is cut into rectangles for you"
    )


def _corner(point: Any, dim: int) -> float:
    return float((point.x, point.y, point.z)[dim])


def _cut_up(shape: Any, label: str, box: Box, flat: int, area: float) -> list[tuple[str, Box]]:
    """A flat Manhattan shape, as the rectangles it is made of.

    The outline is taken as loose edges rather than as ordered rings - every
    edge of every wire of every face, in whatever order the kernel holds them.
    A compound of coplanar faces therefore works without being a special case,
    and so does a hole, which is what a clearance in a ground plane is.

    Per *face wire*, and not ``Shape.Edges``, which is the difference between
    cutting a fused L and refusing one. Where two faces meet, the kernel holds
    the shared edge once but each face's wire runs along it, so walking the
    wires yields it twice - and twice is what makes it cancel. Even-odd counts
    an interior edge as a boundary otherwise, and the seam of a fused shape is
    exactly that: an edge with metal on both sides.

    Straightness is checked by **measuring** the edge against its own chord, not
    by asking its type. A semicircle from (0, 0) to (10, 0) has axis-aligned
    endpoints, and a check that looked only at the corners would swallow it and
    return a rectangle - which is precisely the silent staircase this module
    exists to refuse.
    """
    axes = [dim for dim in range(DIMENSIONS) if dim != flat]
    edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    curved = 0
    faces = list(getattr(shape, "Faces", ()) or ())
    walked = [edge for face in faces for wire in face.Wires for edge in wire.Edges]
    for edge in walked or list(shape.Edges):
        points = list(edge.Vertexes)
        if len(points) != 2:
            curved += 1
            continue
        start, end = (vertex.Point for vertex in points)
        chord = math.dist(
            [_corner(start, d) for d in range(DIMENSIONS)],
            [_corner(end, d) for d in range(DIMENSIONS)],
        )
        if abs(float(edge.Length) - chord) > FLATNESS:
            curved += 1
            continue
        edges.append(
            (
                (_corner(start, axes[0]), _corner(start, axes[1])),
                (_corner(end, axes[0]), _corner(end, axes[1])),
            )
        )

    try:
        if curved:
            raise RectilinearError(f"{curved} of its edges are curved")
        cut = rectangles(edges, tolerance=FLATNESS)
    except RectilinearError as error:
        raise TranslationError(
            f"{label!r} does not fill its bounding box, and cannot be cut into "
            f"rectangles either: {error}. The Yee grid is rectilinear, so "
            "openEMS would staircase this shape without saying so. Redraw the "
            "outline with axis-aligned edges, or pick a solver with a "
            "conforming mesh"
        ) from error

    elevation = box.lower[flat]
    boxes: list[Box] = []
    for lower_u, lower_v, upper_u, upper_v in cut:
        lower, upper = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        lower[flat] = upper[flat] = elevation
        lower[axes[0]], upper[axes[0]] = lower_u, upper_u
        lower[axes[1]], upper[axes[1]] = lower_v, upper_v
        boxes.append(Box(lower=tuple(lower), upper=tuple(upper)))  # type: ignore[arg-type]

    # What was cut has to add up to what was drawn. This is the whole guarantee
    # that the cut is exact rather than a fit: it catches an island dropped, a
    # hole not seen, a ring counted twice, and an even-odd inversion. A
    # refusal and not an assert - `assert` vanishes under -O, and what this
    # guards is the user's shape being something the kernel described
    # differently than it was read.
    laid = sum(
        (piece.upper[axes[0]] - piece.lower[axes[0]])
        * (piece.upper[axes[1]] - piece.lower[axes[1]])
        for piece in boxes
    )
    if not boxes or not math.isclose(laid, area, rel_tol=1e-6):
        # Which way it misses says whose fault it is. Short means the outline
        # encloses less than the shape claims, and pieces laid over each other
        # do exactly that - each reports its own area while the boundary counts
        # the overlap once. Over is the case with no drawing that explains it.
        if laid < area:
            why = (
                "It is short, and pieces laid over one another do that: each "
                "reports its own area while the boundary they share encloses "
                "the overlap once. Fuse them, or draw them apart"
            )
        else:
            why = (
                "It is over, which no drawing accounts for, so this is a fault "
                "in reading the shape rather than anything you drew"
            )
        raise TranslationError(
            f"{label!r} was cut into {len(boxes)} rectangles covering "
            f"{laid:.6g}, but the shape's own area is {area:.6g}. The cut is "
            f"meant to be exact. {why}"
        )

    if len(boxes) == 1:
        return [(label, boxes[0])]
    return [(f"{label}#{number}", piece) for number, piece in enumerate(boxes, 1)]


def _elements_named(reference: Any) -> list[str]:
    """The sub-element names a reference carries, or ``[""]`` for the solid."""
    _, sub = reference if isinstance(reference, tuple) else (reference, None)
    names = [sub] if isinstance(sub, str) else list(sub or ())
    return [name for name in names if name] or [""]


def _reference_boxes(reference: Any) -> tuple[Any, list[tuple[str, Box]]]:
    """Every region one binding reference names, as ``(label, box)`` pairs.

    A binding may name a whole solid, or particular faces of one. The faces are
    not a detail to drop: binding copper to the top face of a substrate is how a
    ground plane gets drawn, and taking the owning solid's box instead turns
    that sheet into a block filling the entire board. It meshes, it solves, and
    the answer is for a different structure.

    One region per named element rather than their union, because the union of
    two faces on different planes is a box that is neither of them.
    """
    obj, sub = reference if isinstance(reference, tuple) else (reference, None)
    names = [sub] if isinstance(sub, str) else list(sub or ())
    names = [name for name in names if name]

    if not names:
        return obj, solid_boxes(obj)

    shape = getattr(obj, "Shape", None)
    if shape is None:
        raise TranslationError(f"{_label(obj)!r} has no shape, so there is nothing to mesh")
    regions = []
    for name in names:
        label = f"{_label(obj)}:{name}"
        regions.extend(_boxes_of(shape.getElement(name), label))
    return obj, regions


def _refinement_boxes(reference: Any, region: str) -> list[tuple[str, Box]]:
    """Every box one ``EMMeshRegion`` reference names, as ``(label, box)``.

    Bounding boxes, and deliberately **not** proved to be boxes the way
    :func:`solid_boxes` proves a material region is one. Refining around a
    cylinder or a fillet is a perfectly sensible thing to want, and the
    rectilinear answer to it is the box it sits in. That is a surprise worth
    naming rather than a refusal - so the property tooltip says "bounding
    box", and this does not complain.

    One box per named element rather than their union, for the reason
    :func:`_reference_boxes` gives: the union of two faces on different planes
    is a box that is neither of them.
    """
    obj = reference[0] if isinstance(reference, tuple) else reference
    names = [name for name in _elements_named(reference) if name]

    shape = getattr(obj, "Shape", None)
    if shape is None:
        raise TranslationError(
            f"{region!r} references {_label(obj)!r}, which has no shape, so "
            "there is nothing to refine around"
        )
    if not names:
        return [(_label(obj), _bounds(shape.BoundBox))]
    return [(f"{_label(obj)}:{name}", _bounds(shape.getElement(name).BoundBox)) for name in names]


def _sizing_regions(refinements: Sequence[Any]) -> tuple[SizingRegion, ...]:
    """``EMMeshRegion`` objects as the mesher's local refinement input.

    Nothing here compares the requested size against the global one. That check
    needs :class:`MeshParams`, and it lives in the mesher, where every route -
    envelope, preview, and a caller building a mesh by hand - passes through
    it exactly once.
    """
    out: list[SizingRegion] = []

    for obj in refinements:
        if not bool(getattr(obj, "Enabled", True)):
            continue

        size = _value(obj.ElementSize)
        if size <= 0:
            raise TranslationError(
                f"{_label(obj)!r} has no element size set, so it asks for "
                "nothing. Set ElementSize, or uncheck Enabled"
            )

        across = int(obj.MinElementsAcross)
        if across < 0:
            raise TranslationError(
                f"{_label(obj)!r}: MinElementsAcross is {across}; it must be 0 "
                "to inherit the global count, or a positive number"
            )

        references = list(getattr(obj, "References", ()) or ())
        if not references:
            raise TranslationError(
                f"{_label(obj)!r} refines nothing. Select the geometry it applies to, or delete it"
            )

        for reference in references:
            for label, box in _refinement_boxes(reference, _label(obj)):
                out.append(
                    SizingRegion(
                        lower=box.lower,
                        upper=box.upper,
                        size=size,
                        min_lines=across,
                        label=f"{_label(obj)} on {label}",
                    )
                )

    return tuple(out)


def _sub_box(link: Any, subject: str) -> Box:
    """A ``LinkSub`` - ``(object, ['Face3'])`` - as a box.

    Unlike :func:`solid_boxes` this does not demand the selection be a box: a face
    of a legitimately box-shaped solid is planar by construction, and callers
    check the flatness that matters to them along the axis it matters on.
    """
    if not link:
        raise TranslationError(f"{subject} is unset; select the face it should be built from")

    obj, sub = link
    names = [sub] if isinstance(sub, str) else list(sub or ())
    names = [name for name in names if name]
    shape = obj.Shape

    if not names:
        return _bounds(shape.BoundBox)
    return _union([_bounds(shape.getElement(name).BoundBox) for name in names])


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def _material(obj: Any, center_hz: float) -> Material:
    """One ``EMMaterial`` as the adapter sees it.

    Loss tangent becomes a conductivity, ``kappa = 2*pi*f*eps0*eps_r*tand``,
    evaluated at the centre of the band. That is openEMS' own convention and it
    is an approximation with a name: a fixed kappa gives a loss tangent that
    falls as 1/f, so the model is exact at band centre and drifts either side of
    it. Wideband accuracy needs a dispersive fit, which is why
    ``FrequencyDependentDielectric`` is refused rather than quietly flattened.

    ``MeasuredAt`` travels beside the kappa it was folded into, because that is
    the only number that says how good the approximation is, and pre-flight -
    which sees the envelope and never the document - is what compares the two.
    """
    declared = str(obj.MaterialType)
    kind = _MATERIAL_KINDS.get(declared)
    if kind is None:
        raise TranslationError(
            f"material {_label(obj)!r}: {declared!r} is not supported by the "
            "openEMS adapter. A dispersive material needs a fitted Debye or "
            "Lorentz pole set, which this adapter does not write; use a "
            "Dielectric with the loss tangent at your band centre instead"
        )

    epsilon = float(obj.Permittivity)
    loss_tangent = float(obj.LossTangent)
    # Checked here because this is the last place the value exists: below it
    # becomes kappa, and `> 0` sends anything else down the lossless branch as
    # a clean 0.0, which every guard downstream then passes.
    if not math.isfinite(loss_tangent) or loss_tangent < 0:
        raise TranslationError(
            f"{_label(obj)!r}: loss tangent is {loss_tangent:g}. It must be a "
            "finite number and cannot be negative, which would be a material "
            "that supplies energy. Use 0 for a lossless dielectric"
        )
    # Only a dielectric's loss tangent becomes anything. A conductor's loss is
    # its conductivity and its thickness, so a loss tangent on one reaches
    # neither the envelope nor the engine - and dropping a number the user
    # typed is the silent no-op 4.2 forbids, whatever it would have meant.
    if loss_tangent > 0 and kind != "dielectric":
        raise TranslationError(
            f"{_label(obj)!r}: a {declared} carries a loss tangent of "
            f"{loss_tangent:g}. openEMS takes a conductor's loss from its "
            "conductivity and thickness, so this number would reach nothing. "
            "Clear it, or make this a Dielectric"
        )
    kappa = 0.0
    if kind == "dielectric" and loss_tangent > 0:
        kind = "lossy_dielectric"
        kappa = 2 * math.pi * center_hz * VACUUM_PERMITTIVITY * epsilon * loss_tangent

    with _model_fault():
        return Material(
            name=_label(obj),
            kind=kind,
            epsilon=epsilon,
            mu=float(obj.Permeability),
            kappa=kappa,
            conductivity=float(obj.Conductivity),
            thickness=_value(obj.Thickness),
            measured_at=_value(obj.MeasuredAt),
        )


def _is_metal(material: Material) -> bool:
    return material.kind in CONDUCTOR_KINDS


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Context:
    """What a port needs to know that is not on the port itself."""

    frequency: Frequency
    #: ``(object Name, sub-element name)`` to the material bound to it, with an
    #: empty element name for a whole-solid binding. A microstrip port takes its
    #: conductor from whatever its trace is made of, so the two can never
    #: disagree - and the element has to be part of the key, because one solid
    #: may carry different materials on different faces.
    conductor_of: dict[tuple[str, str], str]
    materials: dict[str, Material]
    #: Bulk cell size in millimetres. A waveguide port's default depth is a
    #: fixed number of cells, so it has to be known before the grid is built.
    resolution: float


def _measurement_distance(obj: Any, ctx: _Context, subject: str) -> float:
    """Where the probes sit, refusing zero by naming the number it should be.

    Zero is what a port made outside the command gets - by a script, or by
    hand in the property editor - and it is the one value the geometry cannot
    stand in for. Defaulting silently here would be worse than refusing: the
    drawn box has no band to derive the same number from, so the picture and
    the solve would part company, and the whole reason to draw a port box is
    that they cannot.
    """
    distance = _value(obj.MeasurementDistance)
    if distance > 0:
        return distance
    wanted = portbox.clearance(ctx.frequency.start)
    raise TranslationError(
        f"{subject}: MeasurementDistance is 0, so the probes would sit on the "
        "source and read its near field rather than the line's mode. For this "
        f"band it wants to be at least {wanted:g} mm - {portbox.CLEARANCE:g} "
        f"free-space wavelengths at {ctx.frequency.start / 1e9:.4g} GHz"
    )


def _box(build, *args, **keywords):
    """Call the shared box builder, in this layer's exception vocabulary.

    The geometry lives in ``Microwave.portbox`` so the document object can draw
    exactly what the solver is given - one answer, used twice, with nothing to
    drift. Only the exception type is translated: a caller catching
    ``TranslationError`` must not have to know where the arithmetic lives.
    """
    try:
        return build(*args, **keywords)
    except portbox.BoxError as error:
        raise TranslationError(str(error)) from error


def _check_reaches_inward(
    subject: str, link: Any, face: Box, axis: int, direction: int, what: str
) -> None:
    """Refuse a propagation direction that points out of the structure.

    The port box has to reach *into* the model from the face it starts on. Point
    it the other way and it hangs in the air outside, where openEMS will happily
    launch a mode into nothing and report an S-matrix for it. Nothing downstream
    can tell that apart from a real answer, so it is checked here, against the
    body the face belongs to rather than against the user's word for it.
    """
    owner = link[0]
    body = _bounds(owner.Shape.BoundBox)
    inward = body.middle(axis) - face.middle(axis)
    if inward * direction < 0:
        raise TranslationError(
            f"{subject}: PropagationAxis is {AXIS_NAMES[axis]}"
            f"{'+' if direction > 0 else '-'} but {_label(owner)!r} lies the "
            f"other way from its {what}. The port would reach out of the "
            "structure into open space, and would measure it"
        )


def _shared(obj: Any, number: int, kind: str, label: str) -> dict[str, Any]:
    return {
        "number": number,
        "kind": kind,
        "label": label,
        "reference_impedance": _reference_impedance(obj),
    }


def _reference_impedance(obj: Any) -> float | None:
    """What the S-parameters are reported against. Positive, or nothing works.

    ``None`` when the port is referenced to its own impedance, which is what the
    envelope's ``None`` means and what the result layer resolves against the
    impedance each run reported. The number is then neither read nor checked:
    the editor hides it in that mode, and refusing a hidden field would name a
    property the user cannot see.

    Read through its own function for the reason :func:`_model_fault` records,
    and *checked* here rather than left to the envelope for a second reason: the
    property editor spells it ``ReferenceImpedance`` and the envelope spells it
    ``reference_impedance``, and a message naming the second sends a user
    looking for a property that does not exist - the reason the ``AXIS_NAMES``
    import above takes the document's spelling.

    Zero is the case worth naming. It reads as "unset" and is not: renormalising
    divides by it, so an ideal matched line comes back showing gain and
    non-reciprocity, finite enough to reach a Touchstone file.
    """
    if str(obj.ReferencedTo) == _PORT_IMPEDANCE:
        return None
    value = _value(obj.ReferenceImpedance)
    if not math.isfinite(value) or value <= 0:
        raise TranslationError(
            f"{_label(obj)!r}: ReferenceImpedance is {value:g}. The S-parameters "
            "are reported against it, so it has to be a positive impedance"
        )
    return value


def _resistance(obj: Any, name: str) -> float:
    """One of the two resistances, refusing a negative.

    openEMS does not refuse it: ``LumpedPort`` binds its resistive element only
    in the ``R > 0`` and ``R == 0`` branches, so a negative R falls through to an
    ``UnboundLocalError`` several frames inside the bindings.

    ``not >= 0`` rather than ``< 0`` because NaN is False for both comparisons,
    and the envelope's own ``_finite`` would then catch it one layer too late,
    as an "internal error" with a traceback.
    """
    resistance = _value(getattr(obj, name))
    if not resistance >= 0 or not math.isfinite(resistance):
        raise TranslationError(
            f"{_label(obj)!r}: {name} is {resistance}. A negative resistance is "
            "not a thing openEMS can build; set it to zero for a short, or "
            "leave it positive"
        )
    return resistance


def _feed_resistance(obj: Any) -> float | None:
    """A microstrip's damping resistor, or ``None`` for a bare source.

    A matched series resistance damps the reflection off the feed, so the port
    settles in far fewer timesteps. Zero means a bare voltage source, which is
    what the microstrip acceptance case uses - with the line run out through
    the absorber there is nothing to reflect off, and the undamped source gives a
    cleaner incident wave.

    Zero has to become ``None`` here, not travel as a number. ``MSLPort`` spells
    "no resistor" as an infinite ``Feed_R`` and reserves ``Feed_R == 0`` for a
    metal short across the feed - so passing the user's zero through would
    build the opposite of what they asked for.
    """
    return _resistance(obj, "FeedResistance") or None


def _conductor_for(link: Any, ctx: _Context, subject: str) -> str:
    """The material a port's conductor is made of, from the document's bindings.

    Resolved per sub-element, not per object. A binding may name particular
    faces, and one solid may legitimately carry different materials on
    different faces - keying this by the object alone lets the second binding
    overwrite the first, so a port would be laid in whichever material happened
    to be processed last. Falls back to a whole-solid binding when the port's
    face is not individually bound, which is the ordinary case.
    """
    owner = link[0]
    found = {
        ctx.conductor_of[(owner.Name, element)]
        for element in _elements_named(link)
        if (owner.Name, element) in ctx.conductor_of
    }
    if not found and (owner.Name, "") in ctx.conductor_of:
        found = {ctx.conductor_of[(owner.Name, "")]}

    if not found:
        raise TranslationError(
            f"{subject}: {_label(owner)!r} has no material bound to it, so the "
            "adapter cannot tell what the conductor is made of. Bind one before "
            "running"
        )
    if len(found) > 1:
        raise TranslationError(
            f"{subject}: {_label(owner)!r} carries more than one material where "
            f"this port sits ({', '.join(sorted(found))}). The port lays its "
            "conductor in one of them and the document does not say which"
        )
    return found.pop()


def _microstrip(obj: Any, number: int, ctx: _Context) -> Port:
    """A microstrip port: a strip over a ground plane, fed across the substrate.

    The corner ordering is the whole point. ``start`` sits on the trace and
    ``stop`` on the ground plane, because ``MSLPort`` integrates the voltage from
    one to the other and the direction of that integration is the sign of the
    excitation. A sorted bounding box would lose it, and the port would be driven
    backwards - which produces a perfectly clean-looking solve with the phase
    inverted.
    """
    label = _label(obj)
    subject = f"microstrip port {label!r}"

    prop_axis, direction = _axis(obj.PropagationAxis, f"{subject}: PropagationAxis")
    exc_axis, exc_direction = _axis(obj.ExcitationAxis, f"{subject}: ExcitationAxis")
    if prop_axis == exc_axis:
        raise TranslationError(
            f"{subject}: PropagationAxis and ExcitationAxis are both "
            f"{AXIS_NAMES[prop_axis]}. The wave travels along the trace and the "
            "field points across to the ground plane; they cannot be the same"
        )
    width_axis = portbox.third_axis(prop_axis, exc_axis)

    trace = _sub_box(obj.TraceEnd, f"{subject}: TraceEnd")
    ground = _sub_box(obj.GroundReference, f"{subject}: GroundReference")

    if not trace.is_flat(prop_axis):
        raise TranslationError(
            f"{subject}: TraceEnd spans {trace.extents[prop_axis]:.4g} mm along "
            f"{AXIS_NAMES[prop_axis]}, the propagation axis. Select the end face "
            "of the trace - the one the wave enters through - not a face "
            "running along it"
        )
    if trace.is_flat(width_axis):
        raise TranslationError(
            f"{subject}: TraceEnd has no width along {AXIS_NAMES[width_axis]}, "
            "so there is no strip to excite"
        )

    _check_reaches_inward(subject, obj.TraceEnd, trace, prop_axis, direction, "end face")

    conductor = _conductor_for(obj.TraceEnd, ctx, subject)
    if not _is_metal(ctx.materials[conductor]):
        raise TranslationError(
            f"{subject}: the trace is bound to {conductor!r}, a "
            f"{ctx.materials[conductor].kind}. A microstrip port lays its strip "
            "in that material, and a dielectric strip carries no current"
        )

    box = _box(
        portbox.microstrip,
        trace.as_pair(),
        ground.as_pair(),
        propagation_axis=prop_axis,
        direction=direction,
        excitation_axis=exc_axis,
        excitation_direction=exc_direction,
        feed_offset=_value(obj.FeedOffset),
        measurement_distance=_measurement_distance(obj, ctx, subject),
        stated_length=_value(obj.Length),
        subject=subject,
    )

    return Port(
        start=box.start,
        stop=box.stop,
        propagation_axis=prop_axis,
        excitation_axis=exc_axis,
        metal=conductor,
        feed_shift=box.feed,
        measurement_shift=box.measurement,
        feed_resistance=_feed_resistance(obj),
        **_shared(obj, number, "microstrip", label),
    )


def _lumped(obj: Any, number: int, ctx: _Context) -> Port:
    """A lumped port: a resistor across a gap, driven along one axis.

    ``start`` on the source entity and ``stop`` on the reference, for the same
    reason as the microstrip - the ordering is the sign.

    openEMS' lumped port has no propagation direction; it is a circuit element,
    not a transmission line. The adapter's model still wants an axis, and uses it
    only to check the port is clear of the absorber, so the wider of the two
    transverse extents is taken. Nothing measured depends on the choice.
    """
    label = _label(obj)
    subject = f"lumped port {label!r}"

    exc_axis, _ = _axis(obj.ExcitationAxis, f"{subject}: ExcitationAxis")
    source = _sub_box(obj.SourceEntity, f"{subject}: SourceEntity")
    reference = _sub_box(obj.ReferenceEntity, f"{subject}: ReferenceEntity")

    for name, box in (("SourceEntity", source), ("ReferenceEntity", reference)):
        if not box.is_flat(exc_axis):
            raise TranslationError(
                f"{subject}: {name} spans {box.extents[exc_axis]:.4g} mm along "
                f"{AXIS_NAMES[exc_axis]}, the axis the port drives across. That "
                "is a solid, not the surface bounding the gap - the port would "
                "be built from its outer face and reach through the conductor. "
                "Select the face at the gap"
            )

    box = _box(
        portbox.lumped,
        source.as_pair(),
        reference.as_pair(),
        excitation_axis=exc_axis,
        subject=subject,
    )

    return Port(
        start=box.start,
        stop=box.stop,
        propagation_axis=box.propagation_axis,
        excitation_axis=exc_axis,
        # Never None. A lumped port's resistance *is* the element, so zero is a
        # short - openEMS lays metal across the gap for it - and the envelope
        # refuses a lumped port that states no resistance at all, there being
        # nothing sensible to build.
        feed_resistance=_resistance(obj, "Resistance"),
        **_shared(obj, number, "lumped", label),
    )


def _rect_waveguide(obj: Any, number: int, ctx: _Context) -> Port:
    """A rectangular waveguide port: a mode launched over a cross-section.

    No excitation axis and no shifts - the excitation sits on the near face of
    the port box and the probes on the far one, and that is not adjustable. The
    model refuses both rather than accepting numbers it would ignore.
    """
    label = _label(obj)
    subject = f"waveguide port {label!r}"

    prop_axis, direction = _axis(obj.PropagationAxis, f"{subject}: PropagationAxis")
    # CrossSection is a LinkSub, so the reachable planes are the outer faces of
    # whatever solid the guide is drawn as - a port *inside* the guide cannot
    # be pointed at.
    face = _sub_box(obj.CrossSection, f"{subject}: CrossSection")

    if not face.is_flat(prop_axis):
        raise TranslationError(
            f"{subject}: CrossSection spans "
            f"{face.extents[prop_axis]:.4g} mm along {AXIS_NAMES[prop_axis]}, the "
            "propagation axis. Select the guide's cross-section, not a wall"
        )
    for dim in range(DIMENSIONS):
        if dim != prop_axis and face.is_flat(dim):
            raise TranslationError(
                f"{subject}: CrossSection has no extent along "
                f"{AXIS_NAMES[dim]}; a waveguide port must span the whole "
                "cross-section of the guide"
            )

    _check_reaches_inward(subject, obj.CrossSection, face, prop_axis, direction, "cross-section")

    # Before the box, not after: an unrunnable mode costs nothing to spot and
    # the geometry checks above have already named anything worse.
    with _model_fault(_label(obj)):
        check_mode(str(obj.Mode), "Mode")

    # Five cells, not the guide's length: a box spanning the whole guide would
    # put the probes at the opposite end. openEMS' own examples and this
    # project's WR-42 gate both use a box a few cells deep. It is a *depth*, not
    # a reach, so there is no geometry to fall back on.
    box = _box(
        portbox.rect_waveguide,
        face.as_pair(),
        propagation_axis=prop_axis,
        direction=direction,
        stated_length=_value(obj.Length),
        fallback=5 * ctx.resolution,
        subject=subject,
    )

    return Port(
        start=box.start,
        stop=box.stop,
        propagation_axis=prop_axis,
        mode=str(obj.Mode),
        **_shared(obj, number, "rect_waveguide", label),
    )


_PORT_BUILDERS = {
    "EMPortMicrostrip": _microstrip,
    "EMPortLumped": _lumped,
    "EMPortRectWaveguide": _rect_waveguide,
}


def _port_numbers(ports: Sequence[Any]) -> list[int]:
    """Each port's number, refusing anything ambiguous.

    Assignment happens where ports are created, not here: renumbering during a
    run would mean the S-matrix a user reads back is indexed differently from the
    tree they are looking at.
    """
    numbers = [int(port.Number) for port in ports]

    unset = [_label(p) for p, n in zip(ports, numbers) if n < 1]
    if unset:
        raise TranslationError(
            f"{', '.join(unset)}: port number is unset. Every port needs a "
            "number - it indexes the S-matrix"
        )

    seen: dict[int, str] = {}
    for port, number in zip(ports, numbers):
        if number in seen:
            raise TranslationError(
                f"ports {seen[number]!r} and {_label(port)!r} are both numbered "
                f"{number}; S{number}{number} would be ambiguous"
            )
        seen[number] = _label(port)

    return numbers


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
    return SPEED_OF_LIGHT / frequency.stop / math.sqrt(slowest) * 1e3


def _mesh_params(
    settings: Any,
    materials: Iterable[Material],
    frequency: Frequency,
    absorber: tuple[int, int, int],
) -> MeshParams:
    """Mesh policy in millimetres, from the document's wavelength fractions.

    ``ElementsPerWavelength`` counts elements across the wavelength *in the
    slowest material in the model* at the top of the band, never a length. A
    remembered millimetre value silently under-resolves the moment anyone raises
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
        cap=SPEED_OF_LIGHT / frequency.stop * 1e3 / per_wavelength,
    )


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


# ---------------------------------------------------------------------------
# The front door
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Contents:
    """Everything one analysis owns."""

    analysis: Any
    solver: Any
    settings: Any
    bindings: tuple[Any, ...]
    ports: tuple[Any, ...]
    # Not "regions": that word already means the material boxes the mesher
    # resolves. These are the EMMeshRegion objects asking for finer elements.
    refinements: tuple[Any, ...]


def contents(analysis: Any) -> Contents:
    """Find what an analysis owns, by membership rather than by scan.

    Scanning ``document.Objects`` for every binding, port and region instead
    makes a second simulation in one document unresolvable: it would silently
    inherit everything, so two would have to be refused outright - and one board
    marked up twice is an ordinary thing to want.

    Membership answers it. ``EMAnalysis.Group`` is FreeCAD's own ownership,
    maintained by FreeCAD, and reading it here is duck typing like everything
    else in this module - nothing below imports FreeCAD.

    Nested groups are followed. Sorting twenty ports into a subgroup is
    ordinary housekeeping and must not quietly drop them from the study.
    """
    _check_is_an_analysis(analysis)

    members = _members(analysis)

    solvers = [obj for obj in members if _kind(obj) == "EMSolverOpenEMS"]
    if not solvers:
        raise TranslationError(
            f"{_label(analysis)!r} holds no solver, so there is nothing to run "
            "it with. Add an openEMS solver to the analysis"
        )
    if len(solvers) > 1:
        names = ", ".join(_label(obj) for obj in solvers)
        raise TranslationError(
            f"{_label(analysis)!r} holds {len(solvers)} openEMS solvers "
            f"({names}), and nothing says which one runs. Keep one, or move the "
            "others to their own analysis"
        )

    settings = [obj for obj in members if _kind(obj) == "EMMeshPolicy"]
    if not settings:
        raise TranslationError(
            f"{_label(analysis)!r} holds no mesh policy, so there is nothing to "
            "mesh by. Add an EMMeshPolicy object to the analysis"
        )
    if len(settings) > 1:
        names = ", ".join(_label(obj) for obj in settings)
        raise TranslationError(
            f"{_label(analysis)!r} holds {len(settings)} mesh policies ({names}). "
            "One study is meshed one way; keep one"
        )
    _check_mesh_policy(settings[0])

    return Contents(
        analysis=analysis,
        solver=solvers[0],
        settings=settings[0],
        bindings=tuple(obj for obj in members if _kind(obj) == "EMMaterialBinding"),
        ports=tuple(obj for obj in members if _kind(obj).startswith("EMPort")),
        refinements=tuple(obj for obj in members if _kind(obj) == "EMMeshRegion"),
    )


def _members(group: Any, seen: set[int] | None = None) -> list[Any]:
    """Everything in a group, following nested groups once each.

    ``seen`` guards against a cycle. FreeCAD will not normally build one, but a
    document is a file a user can edit and an infinite loop here would hang the
    GUI with no message.
    """
    seen = set() if seen is None else seen
    found: list[Any] = []
    for member in getattr(group, "Group", None) or ():
        if id(member) in seen:
            continue
        seen.add(id(member))
        found.append(member)
        if getattr(member, "Group", None) is not None:
            found.extend(_members(member, seen))
    return found


def _check_is_an_analysis(analysis: Any) -> None:
    """A run starts from a study, and says so rather than dying on ``Group``.

    The solver gets its own sentence because selecting it and pressing Run is an
    ordinary mistake, not a stale document: it is the object called "openEMS", it
    is where the run settings are, and the band and the ownership of ports live
    one level up.
    """
    kind = _kind(analysis)
    if kind == "EMAnalysis":
        return
    if kind == "EMSolverOpenEMS":
        raise TranslationError(
            f"{_label(analysis)!r} is a solver, not a study. The band, the ports "
            "and the geometry belong to the EMAnalysis it sits in, and that is "
            "what a run starts from"
        )
    raise TranslationError(
        f"{_label(analysis)!r} is not an EM analysis, so there is no study to "
        "read. A run starts from an EMAnalysis"
    )


def _geometry(
    bindings: Sequence[Any], center_hz: float
) -> tuple[tuple[Material, ...], tuple[Solid, ...], dict[str, str]]:
    """Bindings to materials, solids, and a trace-to-conductor lookup."""
    materials: dict[str, Material] = {}
    solids: list[Solid] = []
    conductor_of: dict[str, str] = {}

    for binding in bindings:
        material_obj = getattr(binding, "Material", None)
        if material_obj is None:
            raise TranslationError(
                f"{_label(binding)!r} has no material assigned. Pick one, or delete the binding"
            )

        material = _material(material_obj, center_hz)
        existing = materials.get(material.name)
        if existing is not None and existing != material:
            raise TranslationError(
                f"two materials are both labelled {material.name!r} and differ. "
                "Labels are how bindings name materials, so they must be unique"
            )
        materials[material.name] = material

        references = list(getattr(binding, "References", ()) or ())
        if not references:
            raise TranslationError(
                f"{_label(binding)!r} binds {material.name!r} to nothing. Select "
                "the geometry it applies to, or delete the binding"
            )

        for reference in references:
            obj, regions = _reference_boxes(reference)
            for element in _elements_named(reference):
                conductor_of[(obj.Name, element)] = material.name
            for label, box in regions:
                solids.append(
                    Solid(
                        material=material.name,
                        lower=box.lower,
                        upper=box.upper,
                        priority=(METAL_PRIORITY if _is_metal(material) else DIELECTRIC_PRIORITY),
                        label=label,
                    )
                )

    return tuple(materials.values()), tuple(solids), conductor_of


def _ports(found: Sequence[Any], ctx: _Context) -> tuple[Port, ...]:
    numbers = _port_numbers(found)
    ports = []
    for obj, number in zip(found, numbers):
        builder = _PORT_BUILDERS.get(_kind(obj))
        if builder is None:
            # Reachable, and the message is the whole point: the document can
            # hold a port kind this adapter cannot build, which is what
            # declaring capabilities and refusing loudly is for - and it is
            # what the workbench will say the day a second adapter grows a port
            # kind openEMS does not have.
            raise TranslationError(f"{_label(obj)!r}: no openEMS builder for {_kind(obj)}")
        ports.append(builder(obj, number, ctx))
    return tuple(ports)


def _active(ports: Sequence[Any]) -> list[int]:
    """Numbers of the ports the user marked as sources, in ascending order."""
    numbers = _port_numbers(ports)
    return sorted(number for obj, number in zip(ports, numbers) if bool(obj.Excitation))


@dataclass(frozen=True)
class _Translated:
    """Everything read out of the document, before an excitation is chosen."""

    found: Contents
    frequency: Frequency
    materials: tuple[Material, ...]
    solids: tuple[Solid, ...]
    ports: tuple[Port, ...]
    params: MeshParams
    padding: tuple
    boundary: tuple[str, ...]
    sizing: tuple[SizingRegion, ...]


def _translate(analysis: Any) -> _Translated:
    """The whole read of a document, shared by :func:`problem` and :func:`mesh`.

    One function, because two would drift. The preview and the envelope have to
    describe the same grid or the preview is worse than nothing - it would be
    a picture of a mesh nobody solves.
    """
    found = contents(analysis)
    frequency = _frequency(found.analysis)
    materials, solids, conductor_of = _geometry(found.bindings, frequency.center)

    if not found.ports:
        raise TranslationError(
            f"{_label(analysis)!r} has no ports, so there is nothing to "
            "excite and nothing to measure"
        )

    ctx = _Context(
        frequency=frequency,
        conductor_of=conductor_of,
        materials={material.name: material for material in materials},
        resolution=(
            _wavelength(materials, frequency) / float(found.settings.ElementsPerWavelength)
        ),
    )
    ports = _ports(found.ports, ctx)

    boundary = _boundary(found.solver)
    absorber = _absorber_cells(boundary, int(found.solver.PMLCells))

    return _Translated(
        found=found,
        frequency=frequency,
        materials=materials,
        solids=solids,
        ports=ports,
        params=_mesh_params(found.settings, materials, frequency, absorber),
        padding=_padding(found.settings),
        boundary=boundary,
        sizing=_sizing_regions(found.refinements),
    )


@dataclass(frozen=True)
class MeshPlan:
    """A grid, with everything needed to draw and describe it.

    Not part of the envelope: ``lines`` carries why each pinned line exists, and
    provenance is not solver input. It is what the preview draws and what the
    report reads.
    """

    lines: Any
    regions: tuple
    params: MeshParams
    grid: Any
    #: ``(lower, upper)`` of the box the user drew, before the absorber moved
    #: the domain off it. Left as plain tuples rather than a ``report.Extent``
    #: so this module does not depend on the one that describes it.
    structure: tuple | None = None

    def digest(self) -> str:
        """The finished grid's content hash. Provenance for a drawing."""
        return self.grid.digest()


def mesh(analysis: Any) -> MeshPlan:
    """Mesh a document without choosing an excitation.

    A preview does not care which port is driven - every run in a sweep shares
    one grid by construction - so this stops short of the envelope. It goes
    through the same :func:`_translate` as :func:`problem`, which is what makes
    the picture and the solve the same mesh rather than two that agree today.
    """
    read = _translate(analysis)
    lines, regions, bounds = plan_mesh(
        read.solids,
        read.ports,
        read.materials,
        read.params,
        read.padding,
        read.sizing,
    )
    drawn_lower, drawn_upper = structure_bounds(read.solids, read.ports)
    return MeshPlan(
        lines=lines,
        regions=regions,
        params=read.params,
        grid=grid_from(lines, read.params, bounds, read.padding),
        structure=(tuple(drawn_lower), tuple(drawn_upper)),
    )


def grid_inputs_digest(analysis: Any) -> str:
    """Hash everything the grid is computed from, without computing it.

    The mesher is a pure, deterministic function of exactly these arguments -
    ``plan_mesh(solids, ports, materials, params, padding, sizing)`` - so identical
    inputs give an identical grid, and this is a complete staleness key rather
    than a cheap approximation of one.

    Cheap is the point. Deciding "is the drawing still right?" costs one
    translation and no mesh, and the mesher is the expensive half. That is what
    makes it affordable on every recompute, which is what makes the preview's
    out-of-date badge *precise*: raising ``MaxTimesteps`` moves the envelope and
    appears nowhere below, so the badge stays quiet, and a badge that cries wolf
    is a badge people learn to ignore.
    """
    read = _translate(analysis)
    payload = {
        "materials": [material.to_dict() for material in read.materials],
        "solids": [solid.to_dict() for solid in read.solids],
        "ports": [port.to_dict() for port in read.ports],
        "params": {
            "metal_res": read.params.metal_res,
            "dielectric_res": read.params.dielectric_res,
            "max_ratio": list(read.params.max_ratio),
            "min_lines": read.params.min_lines,
            "pml_cells": list(read.params.pml_cells),
            "min_cell": read.params.min_cell,
            "cap": read.params.cap,
        },
        "padding": [[str(face) for face in axis] for axis in read.padding],
        "sizing": [
            {
                "lower": list(region.lower),
                "upper": list(region.upper),
                "size": region.size,
                "min_lines": region.min_lines,
            }
            for region in read.sizing
        ],
    }
    serialised = json.dumps(canonical(payload), sort_keys=True)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def problem(analysis: Any, exciting: int | None = None) -> Problem:
    """Build the envelope for one solve.

    One :class:`~.model.Problem` drives one port, because that is what one
    openEMS run does. ``exciting`` picks which; by default it is the
    lowest-numbered port the user marked as a source. :func:`sweep` produces the
    whole set.
    """
    read = _translate(analysis)
    found, ports = read.found, read.ports

    active = _active(found.ports)
    if not active:
        raise TranslationError(
            "no port has Excitation set, so nothing would drive the model. Mark "
            "at least one port as a source"
        )
    if exciting is None:
        exciting = active[0]
    elif exciting not in {port.number for port in ports}:
        raise TranslationError(f"there is no port numbered {exciting}")

    ports = tuple(replace(port, excite=(port.number == exciting)) for port in ports)

    # Read before the grid is planned, not inside the call below. Keyword
    # arguments evaluate in source order, so leaving it there spends a full mesh
    # on a document that is about to be refused for a property that costs
    # nothing to look at.
    factor = timestep_factor(found.solver)
    # Here rather than in `_translate`, which the mesh preview shares: what
    # drives the model is a question about a solve, and a preview drawn from a
    # document with a stale waveform is still the grid that would be solved.
    _waveform(found.analysis)

    return Problem(
        title=_label(found.analysis),
        frequency=read.frequency,
        grid=plan_grid(
            read.solids,
            ports,
            read.materials,
            read.params,
            padding=read.padding,
            sizing=read.sizing,
        ),
        materials=read.materials,
        solids=read.solids,
        ports=ports,
        boundary=read.boundary,
        termination=_termination(found.solver),
        threads=_threads(found.solver),
        timestep_factor=factor,
    )


def sweep(analysis: Any) -> list[Problem]:
    """One problem per active port - the runs an N-port S-matrix needs.

    Built by re-exciting a single translation rather than translating N times, so
    every run in the set is guaranteed to share one geometry and one grid. Two
    translations of a document that changed underneath would produce an S-matrix
    whose columns describe different structures.
    """
    base = problem(analysis)
    return [base.exciting(number) for number in _active(contents(analysis).ports)]
