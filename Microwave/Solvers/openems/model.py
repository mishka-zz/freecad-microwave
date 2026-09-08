# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The openEMS adapter's private input description.

The envelope crosses the process boundary. The workbench builds a
:class:`Problem` in FreeCAD's Python and serialises it. A separate interpreter
that owns the openEMS bindings reads it back and solves it.

The envelope is private to this adapter. No other adapter reads it, nothing
outside ``Solvers/openems/`` constructs it, and it is not an interchange format.
It does two jobs: it crosses the process boundary, and it can be attached to a
bug report. It is therefore JSON rather than a pickle, and it carries provenance
the run does not need.

This module imports numpy and the standard library only. Both sides of the
process boundary read it: FreeCAD's interpreter, which has no openEMS, writes
it, and the solver's interpreter, which has no FreeCAD, executes it.

Units
-----

Lengths are in millimetres and frequencies in Hertz throughout, with no
per-field overrides. ``length_unit`` is written into the file so that the file
describes itself; it is not a setting to vary. openEMS works in whatever unit
its grid is told to use, and the driver sets that unit from this field. The one
conversion past that is a conducting sheet's ``thickness``, which CSXCAD takes
in metres and :mod:`.driver` converts at the boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ... import units
from .staircase import GROWN_BY, PINNED_CLEARANCE, grown
from .surface import sheet_fault, surface_fault

#: Bump this whenever :meth:`Problem.to_dict` changes shape. The digest is
#: computed over the re-serialised form. An envelope written under an older
#: version comes back from this adapter carrying keys it was written without, so
#: it digests differently: in a sim directory whose ``envelope.sha256`` and
#: ``results.json`` agree with each other, ``Results.matches()`` would still deny
#: that the results came from the envelope. This adapter therefore refuses an
#: unknown version by name.
SCHEMA_VERSION = 6

AXIS_NAMES = ("x", "y", "z")

#: How far off its own plane a sheet's vertex may sit, in millimetres. A drawing
#: is flat to within a kernel tolerance rather than exactly. This value is three
#: orders above FreeCAD's own 1e-7 mm: loose enough for anything the kernel calls
#: planar, and tight enough to catch a face that is genuinely not flat rather
#: than flatten it.
SHEET_FLATNESS = 1e-4

#: Re-exported rather than redefined. See
#: :data:`Microwave.units.SPEED_OF_LIGHT` for the value. The name is repeated
#: here because every other module in the adapter already imports this one. The
#: figure is not repeated: a constant of the vacuum written down twice can
#: drift.
SPEED_OF_LIGHT = units.SPEED_OF_LIGHT
DIMENSIONS = len(AXIS_NAMES)

#: Significant digits kept when the envelope is written.
#:
#: The envelope is canonicalised on the way out so that one problem has one
#: serialisation. Without that it would not: measured under FreeCAD 1.1.1,
#: ``App::PropertyFloat`` reaches the document with about 14 significant digits,
#: short of the 17 an IEEE-754 double needs to survive intact, so a property
#: holding 1/120 comes back from a save as 0.0083333333333333.
#: PropertyQuantity, PropertyLength and PropertyPrecision all behave
#: identically, so choosing a different one does not avoid it.
#:
#: The model is unaffected. The same document before and after a save meshes and
#: solves the same. Provenance is affected: ``digest()`` claims that a result
#: whose digest differs came from a different input, and without this rounding a
#: save and reopen breaks that claim.
#:
#: Twelve digits is 0.1 picometres over a 100 mm domain, far below any cell the
#: mesher will lay, and two digits clear of where FreeCAD truncates. It applies
#: to the whole envelope rather than to the properties that trip it today. Any
#: user-entered dimension can trip it: a substrate 1.6/3 mm thick truncates the
#: same way.
CANONICAL_DIGITS = 12

#: Timesteps a run takes when nothing says otherwise. See :class:`Termination`
#: for why this is a run length rather than a ceiling.
#:
#: The run has to be long enough for the excitation to decay into numerical
#: noise, so that truncating the series stops moving the DFT. That length is a
#: property of the model rather than a constant. Each acceptance gate sets its
#: own, found by lengthening the run until the extracted figure stopped
#: changing. This default sits above all of them with room to spare, and the
#: cost of the margin is linear in the step count.
#:
#: Nothing yet checks that a given model decayed within it. Until something
#: does, a run that is too long is the safer error.
DEFAULT_TIMESTEPS = 30000


def canonical(value: Any) -> Any:
    """Round every float to :data:`CANONICAL_DIGITS`.

    One problem then has one serialisation. The function is idempotent, so
    serialising a round-tripped envelope reproduces it exactly. Integers and
    booleans are left alone. They are exact already, and coercing them to float
    would change the JSON they produce.
    """
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, bool) or not isinstance(value, (float, np.floating)):
        return value
    number = float(value)
    if not np.isfinite(number) or number == 0.0:
        return number
    magnitude = math.floor(math.log10(abs(number)))
    return round(number, CANONICAL_DIGITS - 1 - magnitude)


#: Padding sentinel. The structure continues out through the absorber rather
#: than ending inside the domain. A transmission line padded this way is
#: infinite: substrate and trace run into the PML, so the line has no end to
#: reflect off. See :mod:`.plan` for the effect on the domain.
THROUGH = "through"

MATERIAL_KINDS = frozenset({"dielectric", "lossy_dielectric", "pec", "conducting_sheet"})
PORT_KINDS = frozenset({"microstrip", "lumped", "rect_waveguide", "coaxial"})

#: Material kinds openEMS models as conductors. ``driver._add_material`` hands
#: a ``pec`` its name alone, and a ``conducting_sheet`` its conductivity and
#: thickness. Neither is given ``epsilon`` or ``mu``, so for these two kinds
#: those fields never reach the solver. They do reach the mesher and the
#: pre-flight checks, so :class:`Material` refuses them rather than leaving them
#: set. See ``Material.__post_init__``.
CONDUCTOR_KINDS = frozenset({"pec", "conducting_sheet"})

#: Port kinds that integrate a voltage across a gap. Each has to be told which
#: way to integrate. A waveguide port excites a mode over its whole
#: cross-section and has no such axis.
_NEEDS_EXCITATION_AXIS = frozenset({"microstrip", "lumped"})

#: Mode names openEMS understands, such as TE10. TE modes only, and not TE00.
#:
#: Upstream ``RectWGPort`` raises "Currently only TE-modes are supported!" for
#: anything else (openEMS `python/openEMS/ports.py:434`), so a TM mode reaches
#: the driver as a crash rather than a named refusal. TE00 fails differently: it
#: builds cleanly with ``kc = 0`` and mode functions that are identically zero
#: (ports.py:450-458), so the excitation is zero and the run returns 0/0 after
#: its full runtime, with no error and no refusal.
_MODE_PATTERN = re.compile(r"^TE(\d)(\d)$")

#: Port kinds that lay down a conductor of their own, which the mesh has to
#: resolve. A waveguide's walls are boundary conditions rather than geometry.
_LAYS_CONDUCTOR = frozenset({"microstrip"})

#: Port kinds whose planes openEMS moves onto the grid rather than leaving where
#: they were asked for. ``MSLPort`` takes an ``argmin`` over the propagation
#: lines (``openEMS/python/openEMS/ports.py``:291, :333), and a lumped port is
#: snapped by ``SnapBox2Mesh``.
#: A coaxial port snaps the same way this adapter's builder does, by an
#: ``argmin`` over the same lines. A ``rect_waveguide`` plane is not moved at
#: all. With no line on it nothing is discretised and the run returns 0/0 after
#: its full time, so :meth:`required_lines` pins those two coordinates instead.
_SNAPS_TO_THE_GRID = frozenset({"microstrip", "lumped", "coaxial"})

#: Port kinds that extract by differencing three probes across the measurement
#: plane. The other kinds integrate a mode over one plane and have no difference
#: to take. This set lives here rather than in :mod:`.preflight` so that
#: :meth:`Port.wanted_lines` can ask the same question. Preflight imports this
#: module, so the dependency runs one way only.
_USES_PROBE_TRIPLET = frozenset({"microstrip", "coaxial"})

#: Port kinds whose excitation is a uniform field imposed across the gap. Such
#: an excitation launches the line's mode plus the evanescent content needed to
#: reconcile the imposed shape with the real one, and the probes have to stand
#: clear of that content. A coaxial port is not in this set: its excitation
#: carries the ``1/r`` radial profile of the mode itself, so the shape it imposes
#: is the shape of the mode.
_EXCITES_A_UNIFORM_GAP = frozenset({"microstrip"})

#: Port kinds whose transverse cross-section is a circle rather than a
#: rectangle. The box corners bound a bore, and the port carries a radius the
#: box cannot express.
_IS_ROUND = frozenset({"coaxial"})

#: How far a round port's two transverse extents may disagree, relatively.
#:
#: The two extents are one diameter read twice off one bounding box, so only
#: that box's own rounding separates them. A bore drawn oval by any amount a
#: drawing can express is orders above this tolerance. The test therefore says
#: that the box bounds a circle rather than only a square.
_ROUND_TOLERANCE = 1e-6


class EnvelopeError(Exception):
    """The envelope is malformed, or describes something openEMS cannot run."""


def _axis(value: Any, what: str) -> int:
    """Accept ``0/1/2`` or ``'x'/'y'/'z'``. The stored value is always an index."""
    if isinstance(value, str):
        if value not in AXIS_NAMES:
            raise EnvelopeError(f"{what}: {value!r} is not one of {AXIS_NAMES}")
        return AXIS_NAMES.index(value)
    if value not in (0, 1, 2):
        raise EnvelopeError(f"{what}: {value!r} is not an axis index")
    return int(value)


def _finite(
    value: float,
    what: str,
    *,
    low: float | None = None,
    high: float | None = None,
    strict: bool = False,
    high_strict: bool = False,
) -> float:
    """Check one numeric field for finiteness, then against its bounds.

    A ``<`` comparison is ``False`` for NaN, so a range guard written as one
    passes a NaN silently and hands it to the engine. Finiteness is therefore
    checked first, and every numeric field comes through here.

    The two ends open independently, because a field that wants an open end
    wants it at one end only. Full scale is a legitimate response to look for
    and zero is not a legitimate length unit, while a clearance is accepted at
    nothing and refused at the cell it would swallow.
    """
    number = float(value)
    if not math.isfinite(number):
        raise EnvelopeError(f"{what}: {number} is not a finite number")
    if low is not None and (number <= low if strict else number < low):
        raise EnvelopeError(
            f"{what}: {number:g} must be {'above' if strict else 'at least'} {low:g}"
        )
    if high is not None and (number >= high if high_strict else number > high):
        raise EnvelopeError(
            f"{what}: {number:g} must be {'below' if high_strict else 'at most'} {high:g}"
        )
    return number


def _point(value: Sequence[float], what: str) -> tuple[float, float, float]:
    values = tuple(float(v) for v in value)
    if len(values) != 3:
        raise EnvelopeError(f"{what}: expected 3 coordinates, got {len(values)}")
    if not all(np.isfinite(values)):
        raise EnvelopeError(f"{what}: coordinates must be finite, got {values}")
    return values


def _shifted(
    point: Sequence[float], offset: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(float(p) + float(d) for p, d in zip(point, offset))  # type: ignore[return-value]


def origin_offset(
    solids: Sequence[Solid] = (),
    ports: Sequence[Port] = (),
    grid: MeshGrid | None = None,
) -> tuple[float, float, float]:
    """The translation that puts a structure's minimum corner at the origin.

    openEMS is sound only in non-negative coordinates.
    ``CSPrimPolyhedron::IsInside`` counts how many faces a segment crosses on
    its way to a point it takes to be outside, and it builds that point by
    scaling the primitive's own maximum corner away from the origin
    (``CSXCAD/src/CSPrimPolyhedron.cpp:229``). Scaling a negative coordinate
    also moves it further from the origin, which is toward the solid rather than
    away from it. For a solid whose every maximum is at or below the origin the
    endpoint lands inside the shape and the parity inverts: the object reads as
    hollow, and the space around it as filled.

    The engine is therefore handed a structure with no negative coordinate in
    it, rather than one checked for the corner where that particular ray goes
    wrong. The translation is applied unconditionally, so every structure is put
    in the same place, every run takes the same path, and the property holds by
    construction. A conditional would encode another project's defect in a
    branch that runs on almost no model and is trusted on all of them.

    A translation is a symmetry of Maxwell's equations. The whole structure
    moves together, so every length, every gap and every cell is what it was.
    No solid is moved to suit the engine on its own: where a solid sits relative
    to the rest of the model is the device itself.
    """
    # Every vertex, rather than the box said to bound them. openEMS recomputes
    # the bounding box the ray endpoint is built from out of the vertices
    # themselves, so a vertex outside the box it was given would be outside the
    # guarantee too.
    corners = [solid.lower for solid in solids]
    corners += [vertex for solid in solids for vertex in solid.vertices]
    corners += [port.start for port in ports] + [port.stop for port in ports]
    if grid is not None:
        corners.append((float(grid[0][0]), float(grid[1][0]), float(grid[2][0])))
    if not corners:
        return (0.0, 0.0, 0.0)
    return tuple(  # type: ignore[return-value]
        -min(corner[dim] for corner in corners) for dim in range(DIMENSIONS)
    )


@dataclass(frozen=True)
class Material:
    """A material property, in openEMS' terms.

    :param kind: ``dielectric`` (lossless), ``lossy_dielectric`` (adds
        ``kappa``), ``pec`` (a perfect conductor), or ``conducting_sheet``
        (a zero-thickness conductor with a surface-impedance loss model).
    :param thickness: For ``conducting_sheet`` only, in the same length unit as
        everything else here. It is a loss parameter and feeds the surface
        impedance. The sheet stays geometrically flat, so this never enters the
        mesh. CSXCAD wants this one field in metres, and :mod:`.driver` converts
        it at the boundary.
    :param measured_at: The frequency, in Hz, at which the loss that ``kappa``
        was built from was quoted; zero when nothing recorded it. The solver
        never sees this field, because openEMS is handed ``kappa`` and nothing
        else. It travels anyway: ``kappa`` is fixed for the whole run, so it
        stands for that loss at one frequency only. Only this field can say whether
        that frequency lies in this band, and pre-flight asks on every route,
        including the one that is handed an envelope.
    """

    name: str
    kind: str
    epsilon: float = 1.0
    mu: float = 1.0
    kappa: float = 0.0
    conductivity: float = 0.0
    thickness: float = 0.0
    measured_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise EnvelopeError("material name must not be empty")
        if self.kind not in MATERIAL_KINDS:
            raise EnvelopeError(
                f"material {self.name!r}: unknown kind {self.kind!r}; "
                f"expected one of {sorted(MATERIAL_KINDS)}"
            )
        subject = f"material {self.name!r}"
        if not math.isfinite(float(self.epsilon)) or self.epsilon < 1.0:
            raise EnvelopeError(
                f"{subject}: relative permittivity {self.epsilon} "
                "is below 1 or not a number; openEMS cannot represent it and "
                "the timestep estimate would be wrong"
            )
        # Zero reaches plan.plan_mesh as `cap / sqrt(eps * mu)` and raises
        # ZeroDivisionError; negative raises a math domain error; NaN goes all
        # the way to the engine.
        _finite(self.mu, f"{subject}: relative permeability", low=0.0, strict=True)
        _finite(self.kappa, f"{subject}: kappa", low=0.0)
        _finite(self.conductivity, f"{subject}: conductivity", low=0.0)
        _finite(self.thickness, f"{subject}: thickness", low=0.0)
        _finite(self.measured_at, f"{subject}: measured_at", low=0.0)
        # After the range checks, so a conductor at epsilon NaN is told it is
        # not a number rather than that it is not 1. The more specific complaint
        # is the useful one, and this check only makes sense about a value that
        # is otherwise legal.
        if self.kind in CONDUCTOR_KINDS and (self.epsilon != 1.0 or self.mu != 1.0):
            # Refused rather than ignored, and refused here rather than in the
            # document layer, because `python -m ...driver openems.json` is a
            # documented entry point that never passes through that layer.
            #
            # Ignoring the two fields would be the worse fault. Neither reaches
            # the solver (``driver._add_material`` passes neither), but both
            # reach `policy._wavelength`, which takes `max(epsilon * mu)` over
            # every material and sizes the whole grid from it, and both reach
            # two pre-flight thresholds computed the same way. A permittivity
            # left on a conductor therefore moves the cell count and the
            # clearance thresholds from a value openEMS never sees.
            raise EnvelopeError(
                f"{subject}: a {self.kind} conductor carries relative "
                f"permittivity {self.epsilon:g} and permeability {self.mu:g}. "
                "openEMS is given a conductor's conductivity and nothing else, "
                "so neither reaches the solver - but both reach the mesher, "
                "where they resize every cell in the model. Leave them at 1, or "
                "make this a dielectric"
            )
        # The same fault as above, in a sharper form. ``driver._add_material``
        # passes kappa in its ``lossy_dielectric`` branch and in no other, so
        # anywhere else the loss is dropped between here and the engine and the
        # run comes back lossless: a plausible answer to a question that was not
        # asked. Pre-flight reads this field to decide whether a material's loss
        # was quoted in this band, and that reading applies only to materials
        # whose loss arrives.
        if self.kappa > 0 and self.kind != "lossy_dielectric":
            raise EnvelopeError(
                f"{subject}: a {self.kind} carries kappa {self.kappa:g}. "
                "openEMS is handed a conductivity for a lossy_dielectric and "
                "for nothing else, so this one would be dropped on the way to "
                "the engine and the run would come back lossless. Make it a "
                "lossy_dielectric, or leave kappa at 0"
            )
        if self.kind == "conducting_sheet" and self.conductivity <= 0:
            raise EnvelopeError(
                f"material {self.name!r}: a conducting sheet needs a positive "
                "conductivity; use kind 'pec' for a lossless conductor"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "epsilon": self.epsilon,
            "mu": self.mu,
            "kappa": self.kappa,
            "conductivity": self.conductivity,
            "thickness": self.thickness,
            "measured_at": self.measured_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Material:
        return cls(
            name=data["name"],
            kind=data["kind"],
            epsilon=float(data.get("epsilon", 1.0)),
            mu=float(data.get("mu", 1.0)),
            kappa=float(data.get("kappa", 0.0)),
            conductivity=float(data.get("conductivity", 0.0)),
            thickness=float(data.get("thickness", 0.0)),
            measured_at=float(data.get("measured_at", 0.0)),
        )


@dataclass(frozen=True)
class Solid:
    """A region of one material, and how it is drawn.

    A solid takes one of two shapes. Both carry ``lower`` and ``upper``, so
    anything that only needs to know where a solid is - the domain, the
    absorber's reservation, a pre-flight check - reads those two fields and does
    not consult the kind.

    Without ``faces`` the solid is an axis-aligned box, and equal corners make it
    a sheet. With ``faces`` it is the triangulated boundary of whatever was
    drawn, and the corners are that triangulation's own extent. The grid stays
    rectilinear either way. A triangulation holds the shape exactly, so the
    staircase is the grid's alone and can be priced, rather than being introduced
    silently by squaring the drawing off first.

    A triangulation is checked here, on the way in, rather than trusted. See
    :mod:`~.surface`. The engine reads an open surface as a sheet, that sheet
    contains no point at all, and the object leaves the simulation with no
    message that can be told apart from a benign one.
    """

    material: str
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    priority: int = 0
    label: str = ""
    #: The boundary as triangles, and the vertices they index. Empty for a box.
    vertices: tuple[tuple[float, float, float], ...] = ()
    faces: tuple[tuple[int, int, int], ...] = ()
    #: The axis a sheet is flat on, or ``None`` for a solid. A sheet's triangles
    #: cover its area rather than bounding a volume, so the same two fields mean
    #: something different here: openEMS takes them as flat polygons at one
    #: elevation, and the closedness test a solid is held to would refuse every
    #: one of them.
    sheet_normal: int | None = None
    #: The thickness this conductor was given because the drawing carried none,
    #: in mm, or zero where the drawing carried its own. Nothing is built from
    #: it; the vertices already bound the metal. It crosses the envelope so that
    #: pre-flight can name a length the drawing did not carry.
    thickened: float = 0.0
    #: The finest cell this solid's own demands may ask the mesher for, in mm,
    #: or zero to ask at the policy's sizes. A mesh region set to coarsen names
    #: the object rather than a box. The grid is separable, so a box would spend
    #: itself on a slab through the model on each axis.
    relaxed_to: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower", _point(self.lower, f"solid {self.name!r}"))
        object.__setattr__(self, "upper", _point(self.upper, f"solid {self.name!r}"))
        object.__setattr__(
            self, "vertices", tuple(tuple(float(v) for v in p) for p in self.vertices)
        )
        object.__setattr__(self, "faces", tuple(tuple(int(i) for i in f) for f in self.faces))
        object.__setattr__(
            self,
            "thickened",
            _finite(self.thickened, f"solid {self.name!r}: supplied thickness", low=0.0),
        )
        object.__setattr__(
            self,
            "relaxed_to",
            _finite(self.relaxed_to, f"solid {self.name!r}: relaxed cell size", low=0.0),
        )
        for dim in range(3):
            if self.upper[dim] < self.lower[dim]:
                raise EnvelopeError(
                    f"solid {self.name!r}: upper corner is below lower corner in "
                    f"{AXIS_NAMES[dim]} ({self.upper[dim]} < {self.lower[dim]})"
                )
        axis = self.sheet_normal
        if axis is not None:
            if not 0 <= axis < 3:
                raise EnvelopeError(f"solid {self.name!r}: {axis} is not an axis")
            if self.lower[axis] != self.upper[axis]:
                raise EnvelopeError(
                    f"sheet {self.name!r} has thickness {self.upper[axis] - self.lower[axis]} "
                    f"in {AXIS_NAMES[axis]}, the axis it is declared flat on. A sheet is "
                    "modelled at one plane and would be laid at the wrong one"
                )
            fault = sheet_fault(self.vertices, self.faces, axis, self.lower[axis], SHEET_FLATNESS)
            if fault is not None:
                raise EnvelopeError(f"sheet {self.name!r} cannot be modelled: {fault}")
        elif self.faces or self.vertices:
            # Every question here is about the triangle set alone, and the
            # answers are the same wherever the set was drawn. Where a solid
            # sits is settled for the whole structure at once by
            # :func:`origin_offset`, which makes the translation unconditional
            # rather than asking anything. What that placement decides and
            # nothing here can - whether two corners the kernel held apart
            # survive the engine's single precision - is asked of the placed
            # coordinates, in the pre-flight check for it.
            fault = surface_fault(self.vertices, self.faces)
            if fault is not None:
                raise EnvelopeError(
                    f"solid {self.name!r} is not a closed surface: {fault}. Either "
                    "way openEMS solves something that is not this shape and the "
                    "run completes - a surface it cannot close holds no point at "
                    "all, and one whose faces it rejects it keeps as a solid with "
                    "holes in it"
                )

    @property
    def name(self) -> str:
        return self.label or self.material

    def moved(self, offset: tuple[float, float, float]) -> Solid:
        """The same solid, translated. Every coordinate it carries is moved."""
        return replace(
            self,
            lower=_shifted(self.lower, offset),
            upper=_shifted(self.upper, offset),
            vertices=tuple(_shifted(point, offset) for point in self.vertices),
        )

    @property
    def is_mesh(self) -> bool:
        """True where the shape is held as triangles rather than as its box."""
        return bool(self.faces)

    @property
    def is_sheet(self) -> bool:
        """True where the triangles cover an area rather than bound a volume."""
        return self.sheet_normal is not None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "material": self.material,
            "lower": list(self.lower),
            "upper": list(self.upper),
            "priority": self.priority,
            "label": self.label,
        }
        if self.is_mesh:
            data["vertices"] = [list(point) for point in self.vertices]
            data["faces"] = [list(face) for face in self.faces]
        if self.is_sheet:
            data["sheet_normal"] = self.sheet_normal
        if self.thickened:
            data["thickened"] = self.thickened
        if self.relaxed_to:
            data["relaxed_to"] = self.relaxed_to
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Solid:
        return cls(
            material=data["material"],
            lower=tuple(data["lower"]),
            upper=tuple(data["upper"]),
            priority=int(data.get("priority", 0)),
            label=data.get("label", ""),
            vertices=tuple(tuple(p) for p in data.get("vertices", ())),
            faces=tuple(tuple(f) for f in data.get("faces", ())),
            sheet_normal=data.get("sheet_normal"),
            thickened=float(data.get("thickened", 0.0)),
            relaxed_to=float(data.get("relaxed_to", 0.0)),
        )


@dataclass(frozen=True)
class Port:
    """A port, in the diagonal-corner convention openEMS' own ports use.

    ``start`` and ``stop`` are opposite corners of the port box rather than a
    sorted bounding box. Their difference along ``excitation_axis`` sets which
    way the excitation points, and their difference along ``propagation_axis``
    sets which way is into the structure. For a microstrip, ``start`` sits on the
    trace and ``stop`` on the ground plane. Sorting them would reverse the port
    without saying so.

    :param feed_shift: Distance from ``start`` along the propagation axis to the
        feed, in length units. Keep it clear of the absorber.
    :param measurement_shift: Distance from ``start`` to the measurement plane,
        where the three voltage probes go. The adapter checks at pre-flight that
        the grid is locally uniform there - see :mod:`.preflight`.
    :param feed_resistance: Series resistance at the feed. A matched value damps
        the reflection off the feed; ``None`` means a bare voltage source.
    """

    number: int
    kind: str
    start: tuple[float, float, float]
    stop: tuple[float, float, float]
    propagation_axis: int
    #: Which way the voltage is integrated. Required by the kinds in
    #: :data:`_NEEDS_EXCITATION_AXIS`, and refused for every other.
    excitation_axis: int | None = None
    #: The conductor this port lays down. Microstrip only. A waveguide's walls
    #: are boundary conditions rather than geometry.
    metal: str = ""
    #: ``TE10`` and the like. Waveguide ports only, in the textbook convention:
    #: the first digit counts half-waves across the broad wall, so ``TE10`` is
    #: the dominant mode however the guide is drawn.
    #: :meth:`waveguide_arguments` renumbers it onto openEMS' axes.
    mode: str = ""
    #: The inner conductor's radius, for the kinds in :data:`_IS_ROUND` and
    #: refused for every other. The outer radius is not a field. It is half the
    #: box's transverse extent, so the bore the probes reach across cannot
    #: disagree with the volume the port claims. A waveguide port reads ``a`` and
    #: ``b`` off its box for the same reason.
    inner_radius: float = 0.0
    excite: bool = False
    feed_shift: float = 0.0
    measurement_shift: float = 0.0
    feed_resistance: float | None = None
    #: What the S-parameters are reported against. ``None`` means the port
    #: itself, at whatever impedance it turns out to have at each frequency.
    #: ``None`` is an answer rather than a gap to fill with a default. This
    #: adapter renormalises nothing, so an unstated reference is the basis the
    #: probes already measure in, and it is the only expressible answer for a
    #: dispersive guide.
    reference_impedance: float | None = None
    priority: int = 10
    label: str = ""
    #: True where ``propagation_axis`` rests on nothing the drawing said. The
    #: translation normally reads which way the body lies from the picked
    #: cross-section, and refuses an axis pointing the other way. Some drawings
    #: cannot say: a conductor drawn as a surface has no volume for a pick to be
    #: on one side of, and the axis is then whatever was declared. A launch
    #: turned around solves cleanly with the phase inverted, so the doubt travels
    #: to pre-flight. It is False for an envelope written by hand, because there
    #: is no drawing to disagree with.
    direction_unchecked: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _point(self.start, f"port {self.number}"))
        object.__setattr__(self, "stop", _point(self.stop, f"port {self.number}"))
        object.__setattr__(
            self,
            "propagation_axis",
            _axis(self.propagation_axis, f"port {self.number} propagation_axis"),
        )
        if self.kind not in PORT_KINDS:
            raise EnvelopeError(
                f"port {self.number}: unknown kind {self.kind!r}; "
                f"expected one of {sorted(PORT_KINDS)}"
            )
        if self.length <= 0:
            raise EnvelopeError(
                f"port {self.number}: zero extent along the propagation axis "
                f"({AXIS_NAMES[self.propagation_axis]})"
            )

        self._settle_excitation_axis()
        self._check_the_kind_has_what_it_needs()
        self._check_waveguide_box()
        self._check_round_box()
        self._check_shifts()
        self._check_resistances()

    def _settle_excitation_axis(self) -> None:
        """Coerce the excitation axis, or refuse a kind that has no use for one."""
        if self.kind not in _NEEDS_EXCITATION_AXIS:
            if self.excitation_axis is not None:
                raise EnvelopeError(
                    f"port {self.number}: a {self.kind} port excites a mode over its "
                    "whole cross-section and has no excitation axis; leave it unset "
                    "rather than picking one that does nothing"
                )
            return

        if self.excitation_axis is None:
            raise EnvelopeError(
                f"port {self.number}: a {self.kind} port integrates a voltage "
                "across a gap and must say along which axis"
            )
        object.__setattr__(
            self,
            "excitation_axis",
            _axis(self.excitation_axis, f"port {self.number} excitation_axis"),
        )
        if self.propagation_axis == self.excitation_axis:
            raise EnvelopeError(
                f"port {self.number}: propagation and excitation axes are both "
                f"{AXIS_NAMES[self.propagation_axis]}; they must differ"
            )
        if self.start[self.excitation_axis] == self.stop[self.excitation_axis]:
            raise EnvelopeError(
                f"port {self.number}: zero extent along the excitation axis "
                f"({AXIS_NAMES[self.excitation_axis]}); there is nothing to "
                "integrate the voltage across"
            )

    def _check_the_kind_has_what_it_needs(self) -> None:
        """Fields that are optional in general and mandatory for one kind."""
        if self.kind in _LAYS_CONDUCTOR and not self.metal:
            raise EnvelopeError(
                f"port {self.number}: a {self.kind} port lays down a conductor "
                "and must name the material to use"
            )

        if self.kind == "lumped" and self.feed_resistance is None:
            # A lumped port is its resistance. openEMS builds a resistive
            # sheet across the gap, so this field is structure rather than
            # reporting. It cannot fall back to reference_impedance, which says
            # what the answer is reported against and is unset exactly when the
            # port is reported against itself. Falling back would let an
            # unrelated question decide how much metal is laid across the gap.
            raise EnvelopeError(
                f"port {self.number}: a lumped port is a resistance across a gap "
                "and must state it; 0 asks for a short"
            )

    def _check_waveguide_box(self) -> None:
        """Everything a rect_waveguide port asks of its box and its mode.

        Neither shift is accepted. ``AddRectWaveGuidePort`` takes none: the
        excitation sits on the box's near face and the probes on its far one.
        Accepting a shift would be harmful. Pre-flight validates the feed
        position it computes from ``feed_shift``, so a nonzero value moves the
        checked position while the real excitation stays put, and the absorber
        check can then be walked straight past.

        The mode has to be one this adapter can name, and the box has to span
        the guide. :meth:`waveguide_arguments` reads ``a`` and ``b`` off the
        box's transverse extents rather than carrying them as fields, so the
        mode is computed from the cross-section the user drew. A box flat in a
        transverse axis therefore describes no guide at all.
        """
        if self.kind != "rect_waveguide":
            return
        for name, value in (
            ("feed_shift", self.feed_shift),
            ("measurement_shift", self.measurement_shift),
        ):
            if value:
                raise EnvelopeError(
                    f"port {self.number}: a rect_waveguide port has no "
                    f"{name} - its planes are the faces of the port box. "
                    f"Setting it to {value} would move where this adapter "
                    "thinks the port is without moving the port"
                )
        check_mode(self.mode, f"port {self.number}")
        for axis in self.transverse_axes:
            if self.start[axis] == self.stop[axis]:
                raise EnvelopeError(
                    f"port {self.number}: the port box has no extent in "
                    f"{AXIS_NAMES[axis]}; a waveguide port must span the "
                    "whole guide cross-section"
                )

    def _check_round_box(self) -> None:
        """A round port's box bounds its bore, and holds one radius inside it.

        The box is the bore's bounding box, so its two transverse extents are
        the same diameter read twice and :attr:`outer_radius` is half of it. The
        outer radius is therefore derived from the box, and nothing the user
        writes can contradict it. The inner radius has nothing to be read off
        and is carried.

        A kind that is not round is refused a radius rather than having it
        ignored, for the reason :meth:`_settle_excitation_axis` refuses an axis:
        a field that reaches the solver nowhere still reaches the editor.
        """
        if self.kind not in _IS_ROUND:
            if self.inner_radius:
                raise EnvelopeError(
                    f"port {self.number}: a {self.kind} port has no bore, so "
                    f"inner_radius {self.inner_radius:g} describes nothing; "
                    "leave it unset"
                )
            return

        first, second = (abs(self.stop[axis] - self.start[axis]) for axis in self.transverse_axes)
        if not math.isclose(first, second, rel_tol=_ROUND_TOLERANCE, abs_tol=0.0):
            raise EnvelopeError(
                f"port {self.number}: the port box spans {first:g} by {second:g} "
                f"across {AXIS_NAMES[self.propagation_axis]}, so it does not "
                "bound a circle. A coaxial port is built on a round bore"
            )
        _finite(self.inner_radius, f"port {self.number}: inner_radius", low=0.0, strict=True)
        if self.inner_radius >= self.outer_radius:
            raise EnvelopeError(
                f"port {self.number}: inner_radius {self.inner_radius:g} is not "
                f"inside the bore, which has radius {self.outer_radius:g}. There "
                "is no annulus between the conductors to drive across"
            )

    def _check_shifts(self) -> None:
        """The feed and measurement planes lie within the port box.

        The far face is inside the port, so a measurement plane sitting exactly
        on it is legal. That is the ordinary case: an unset Length ends the box
        there. ``length`` is reconstructed from the corners, so it equals the
        shift only to within the rounding of ``start + direction * length``. The
        tolerance is there to stop a box refusing its own measurement plane, and
        it applies to the reconstruction rather than to the physics.
        """
        for name, value in (
            ("feed_shift", self.feed_shift),
            ("measurement_shift", self.measurement_shift),
        ):
            if not 0.0 <= value <= self.length and not math.isclose(
                value, self.length, rel_tol=1e-9, abs_tol=0.0
            ):
                raise EnvelopeError(
                    f"port {self.number}: {name} of {value} is outside the port, "
                    f"which spans {self.length} along "
                    f"{AXIS_NAMES[self.propagation_axis]}"
                )

    def _check_resistances(self) -> None:
        """The two resistances, checked separately.

        Zero means opposite things to them, so they cannot share a check. A
        lumped port's zero lays metal across the gap and is the only way to ask
        for a short. A zero reference impedance renormalises an ideal matched
        line to ``|S12|`` above one: gain, non-reciprocal, and finite enough to
        reach a Touchstone file. See
        ``TestZeroMeansOppositeThingsToTheTwoResistances``.
        """
        if self.feed_resistance is not None:
            _finite(
                self.feed_resistance,
                f"port {self.number}: feed_resistance",
                low=0.0,
            )
        if self.reference_impedance is not None:
            _finite(
                self.reference_impedance,
                f"port {self.number}: reference_impedance",
                low=0.0,
                strict=True,
            )

    @property
    def name(self) -> str:
        return self.label or f"port {self.number}"

    @property
    def length(self) -> float:
        """Extent along the propagation axis."""
        axis = self.propagation_axis
        return abs(self.stop[axis] - self.start[axis])

    @property
    def direction(self) -> int:
        """``+1`` if the port runs along increasing propagation coordinate."""
        axis = self.propagation_axis
        return 1 if self.stop[axis] >= self.start[axis] else -1

    @property
    def transverse_axes(self) -> tuple[int, int]:
        """The two axes across the propagation direction, in ascending order.

        For propagation along y that is not the order openEMS pairs them in; see
        :attr:`mode_axes`. Use this property only where the pair is a set.
        """
        first, second = (a for a in range(DIMENSIONS) if a != self.propagation_axis)
        return (first, second)

    @property
    def outer_radius(self) -> float:
        """Half the bore's diameter, read off the box rather than off a field.

        The value is the mean of the two transverse half-extents, so it does not
        depend on which axis is asked first. :attr:`transverse_axes` and
        :attr:`mode_axes` disagree about that order, and only one of them is the
        pair openEMS binds.
        """
        extents = [abs(self.stop[axis] - self.start[axis]) for axis in self.transverse_axes]
        return sum(extents) / (2 * len(extents))

    @property
    def bore_centre(self) -> tuple[float, float, float]:
        """Where the line's axis runs, with the propagation coordinate at ``start``."""
        centre = [(a + b) / 2.0 for a, b in zip(self.start, self.stop)]
        centre[self.propagation_axis] = self.start[self.propagation_axis]
        return (centre[0], centre[1], centre[2])

    @property
    def mode_axes(self) -> tuple[int, int]:
        """The transverse axes in the order ``RectWGPort`` binds them.

        ``ny_P = (p+1) % 3`` and ``ny_PP = (p+2) % 3``
        (``openEMS/python/openEMS/ports.py``:580-581).
        The pairing is cyclic, so for propagation along y it is ``(z, x)``, the
        reverse of :attr:`transverse_axes`.
        """
        axis = self.propagation_axis
        return (axis + 1) % DIMENSIONS, (axis + 2) % DIMENSIONS

    def waveguide_arguments(self, length_unit: float) -> tuple[float, float, str]:
        """``(a, b, mode)`` exactly as ``AddRectWaveGuidePort`` takes them.

        The pair is positional rather than sorted. openEMS binds ``a`` to
        :attr:`mode_axes`\\ ``[0]`` and ``b`` to ``[1]``, both in ``kc`` and in
        the mode functions, so ``a`` is the extent along the first of those axes
        whether or not it is the broad wall. ``mode`` arrives here in the
        textbook convention the document uses, where the first digit counts
        half-waves across the broad wall, and is renumbered onto the axes.

        Handing openEMS a sorted pair instead lands the broad wall's length on
        whichever axis comes first, and the field it then launches is not a mode
        of the guide that was drawn. The symptoms are total reflection, no
        transmission, more power out than in, and measurement planes nowhere
        near the geometry. Cutoff and ``Z_ref`` still look right, because both
        are analytic in ``a`` and ``b`` alone. Through this method the same guide
        drawn on two axes agrees to every digit the gate prints, and the
        waveguide gate asserts both drawings.

        ``a`` and ``b`` are derived from the port box rather than carried as
        fields, so the geometry cannot disagree with the numbers the mode is
        computed from.
        """
        first, second = self.mode_axes
        extents = tuple(
            abs(self.stop[axis] - self.start[axis]) * length_unit for axis in (first, second)
        )
        match = _MODE_PATTERN.match(self.mode)
        if match is None:
            # A rect_waveguide port cannot reach this branch: __post_init__
            # refuses a mode this pattern does not match. Any other kind can
            # reach it, by being asked a question it has no answer to.
            raise EnvelopeError(
                f"port {self.number}: a {self.kind} port has no waveguide mode "
                f"({self.mode!r}), so it has no (a, b) to bind to axes"
            )
        broad, narrow = match.group(1), match.group(2)
        wide, thin = extents
        if wide >= thin:
            return wide, thin, f"TE{broad}{narrow}"
        return wide, thin, f"TE{narrow}{broad}"

    @property
    def excite_sign(self) -> int:
        """openEMS' excitation sign.

        ``MSLPort`` integrates the voltage from ``start`` to ``stop`` along the
        excitation axis. Where that runs downward, from trace to ground, which
        is the ordinary microstrip case, the excitation has to be negated.
        openEMS' own MSL tutorial negates it the same way.

        A waveguide port takes no sign. Its direction comes from the ordering of
        ``start`` and ``stop`` along the propagation axis, so both ports of a
        through-line point inward and are excited with 1.
        """
        if not self.excite:
            return 0
        if self.excitation_axis is None:
            return 1
        axis = self.excitation_axis
        return 1 if self.stop[axis] >= self.start[axis] else -1

    def measurement_position(self) -> float:
        """Absolute coordinate of the measurement plane on the propagation axis.

        A waveguide port takes no shift. Its probes sit on the port box's far
        face (``ports.py``: ``m_start[exc_ny] = m_stop[exc_ny]``), so ``stop``
        is the measurement plane.
        """
        axis = self.propagation_axis
        if self.kind == "rect_waveguide":
            return self.stop[axis]
        return self.start[axis] + self.direction * self.measurement_shift

    def moved(self, offset: tuple[float, float, float]) -> Port:
        """The same port, translated. The corners are positions and move; the
        shifts along the axes are lengths and stay as they are."""
        return replace(self, start=_shifted(self.start, offset), stop=_shifted(self.stop, offset))

    def lays_conductor(self) -> bool:
        """Whether this port adds metal the mesh has to resolve."""
        return self.kind in _LAYS_CONDUCTOR

    def snaps_to_the_grid(self) -> bool:
        """Whether openEMS moves this port's planes onto the nearest grid line."""
        return self.kind in _SNAPS_TO_THE_GRID

    def required_lines(self) -> tuple[list[float], list[float], list[float]]:
        """Positions, per axis, where this port needs a grid line to exist.

        Every kind that asks does so for one reason: an excitation is not
        snapped. openEMS discretises an excitation by walking the grid and
        asking the geometry what is at each coordinate, so a box that contains
        no coordinate excites nothing. It reports that as ``Unused primitive``,
        one line among thousands, and the run then completes without driving
        anything, with every S-parameter at 0/0.

        A ``rect_waveguide`` port puts its excitation on a zero-thickness plane
        at ``start`` and its probes on one at ``stop``. openEMS' own tutorial
        adds those lines by hand before meshing. Here the port asks for them, so
        no caller can forget.

        A lumped port that excites asks for a line on every axis it is flat
        across. Flat is the ordinary shape: the canonical lumped port is fed
        from a trace's end face, which has no extent along the line. The
        resistor and the probes both survive it, because openEMS snaps those.
        The excitation is the one primitive that is not snapped, so a plane
        lying between two lines drives nothing at all.

        An axis the box has extent on asks for nothing here. A line inside the
        box is the mesher's to place or not, rather than a position anything can
        name. Whether one landed there is a question about the grid, and
        :mod:`.preflight` asks it where the grid exists. Where the box's own
        faces should fall is a separate question, and :meth:`element_lines` asks
        that one.

        ``MSLPort`` snaps both planes to the nearest existing line rather than
        discretising nothing, so it asks through :meth:`wanted_lines` instead.
        """
        lines: tuple[list[float], list[float], list[float]] = ([], [], [])
        if self.kind == "rect_waveguide":
            axis = self.propagation_axis
            lines[axis].append(self.start[axis])
            lines[axis].append(self.stop[axis])
        elif self.kind == "lumped" and self.excite_sign:
            for axis in range(DIMENSIONS):
                if self.start[axis] == self.stop[axis]:
                    lines[axis].append(self.start[axis])
        return lines

    def wanted_lines(self) -> tuple[list[float], list[float], list[float]]:
        """Positions where a line would put a plane exactly where it was asked for.

        ``MSLPort`` snaps. The source goes to the grid line nearest
        ``start + feed_shift``, and the probe triplet to the three nearest the
        measurement plane. Nothing pins either, so both land wherever the
        grading happened to put a line, up to half a cell from where they were
        asked for. The engineer set those distances deliberately, and the grid
        rounds them without reporting it.

        These lines are wanted rather than required. A missing line here costs a
        fraction of a cell of reference-plane offset, where a waveguide port's
        two faces cost a run of 0/0. A position outside the meshed domain is
        therefore dropped here and reported by :mod:`.preflight`, which also
        reports why it is outside. A source in the absorber is driving a field
        that is being attenuated on purpose, which is more use than "no line
        here".
        """
        lines: tuple[list[float], list[float], list[float]] = ([], [], [])
        if self.kind in _USES_PROBE_TRIPLET:
            axis = self.propagation_axis
            lines[axis].append(self.start[axis] + self.direction * self.feed_shift)
            lines[axis].append(self.measurement_position())
        return lines

    def element_lines(self) -> tuple[list[float], list[float], list[float]]:
        """Positions, per axis, where a lumped element's own box stands.

        The element is that box, and openEMS builds it from the box snapped.
        Each face is moved to the grid line nearest it: ``Calc_LumpedElements``
        snaps with the default method (``openEMS/FDTD/operator.cpp``:1625,
        ``openEMS/FDTD/operator.h``:202), and ``SnapToMeshLine``
        (``openEMS/FDTD/operator.cpp``:253) answers with the line whose dual
        cell holds the coordinate. The resistance is then integrated over what
        the snapping left. An element whose faces the grid does not hold is
        built with a gap the excitation drives across, and a cross-section the
        resistance spreads over, that the model never stated.

        Both faces are asked for, on every axis, for a termination as much as
        for a source. openEMS snaps a resistor whether or not anything is
        exciting it.

        These positions are kept apart from :meth:`wanted_lines` because a solid
        outranks them. A port drawn flush against a conductor should span what
        that conductor spans, and the mesher settles a conductor's edge with the
        thirds rule, which puts lines either side of the edge and never on it. A
        line laid here would override that for one edge of a trace and leave the
        rest of it meshed the other way. :func:`plan.plan_mesh` holds both
        requests and applies the precedence.
        """
        lines: tuple[list[float], list[float], list[float]] = ([], [], [])
        if self.kind == "lumped":
            for axis in range(DIMENSIONS):
                lines[axis].append(self.start[axis])
                if self.stop[axis] != self.start[axis]:
                    lines[axis].append(self.stop[axis])
        return lines

    def trace_region(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """The conductor this port lays down, as a sorted box.

        ``MSLPort`` builds the strip itself, flattened onto ``start``'s
        excitation coordinate. The mesher has to know about that strip, which is
        the finest metal in the model, so the adapter reconstructs it here
        rather than discovering it afterwards.
        """
        axis = self.excitation_axis
        if axis is None:
            raise EnvelopeError(
                f"port {self.number}: a {self.kind} port lays no conductor down, "
                "having no excitation axis to flatten one onto"
            )
        stop = list(self.stop)
        stop[axis] = self.start[axis]
        lower = tuple(min(a, b) for a, b in zip(self.start, stop))
        upper = tuple(max(a, b) for a, b in zip(self.start, stop))
        return lower, upper  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        data = {
            "number": self.number,
            "kind": self.kind,
            "metal": self.metal,
            "mode": self.mode,
            "inner_radius": self.inner_radius,
            "start": list(self.start),
            "stop": list(self.stop),
            "propagation_axis": self.propagation_axis,
            "excitation_axis": self.excitation_axis,
            "excite": self.excite,
            "feed_shift": self.feed_shift,
            "measurement_shift": self.measurement_shift,
            "feed_resistance": self.feed_resistance,
            "reference_impedance": self.reference_impedance,
            "priority": self.priority,
            "label": self.label,
        }
        # Written only where there is something to report, as the supplied
        # thickness on a solid is. A key on every port would change every
        # envelope this adapter produces and say nothing on most of them.
        if self.direction_unchecked:
            data["direction_unchecked"] = True
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Port:
        return cls(
            number=int(data["number"]),
            kind=data["kind"],
            metal=data.get("metal", ""),
            mode=data.get("mode", ""),
            inner_radius=float(data.get("inner_radius", 0.0)),
            start=tuple(data["start"]),
            stop=tuple(data["stop"]),
            propagation_axis=data["propagation_axis"],
            excitation_axis=data.get("excitation_axis"),
            excite=bool(data.get("excite", False)),
            feed_shift=float(data.get("feed_shift", 0.0)),
            measurement_shift=float(data.get("measurement_shift", 0.0)),
            feed_resistance=data.get("feed_resistance"),
            reference_impedance=data.get("reference_impedance"),
            priority=int(data.get("priority", 10)),
            label=data.get("label", ""),
            direction_unchecked=bool(data.get("direction_unchecked", False)),
        )


@dataclass(frozen=True)
class Frequency:
    """The band of interest. openEMS excites it all at once with one pulse."""

    start: float
    stop: float
    points: int = 201

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.start))
            or not math.isfinite(float(self.stop))
            or self.start <= 0
            or self.stop <= self.start
        ):
            raise EnvelopeError(
                f"frequency range {self.start} to {self.stop} Hz is not an "
                "ascending positive band of finite numbers"
            )
        if self.points < 2:
            raise EnvelopeError("a frequency sweep needs at least 2 points")

    @property
    def center(self) -> float:
        return 0.5 * (self.start + self.stop)

    @property
    def half_bandwidth(self) -> float:
        return 0.5 * (self.stop - self.start)

    def values(self) -> np.ndarray:
        return np.linspace(self.start, self.stop, self.points)

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "stop": self.stop, "points": self.points}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Frequency:
        return cls(
            start=float(data["start"]),
            stop=float(data["stop"]),
            points=int(data.get("points", 201)),
        )


@dataclass(frozen=True)
class Termination:
    """When to stop stepping.

    :param max_timesteps: Step count. With ``end_criteria`` at zero this is the
        run length rather than a ceiling: every step is taken, and runtime is
        linear in it. Upstream's 30000
        (``Tutorials/Simple_Patch_Antenna.py:55``, ``Conical_Horn_Antenna.m:51``)
        is a safety net behind ``EndCriteria=1e-4``, and this project cannot use
        that net. :data:`DEFAULT_TIMESTEPS` says what the number rests on.
    :param end_criteria: Stop early once the residual energy falls this far
        below its peak. Zero disables it, and zero is the only reproducible
        setting. openEMS re-evaluates this criterion inside a branch gated on
        four seconds of wall clock (``openEMS/openems.cpp:1445``, against the
        loop's own test at ``:1426``), so an energy-terminated run stops at a
        timestep that depends on machine load. That truncates the recorded time
        series differently and moves every extracted number.
    """

    max_timesteps: int = DEFAULT_TIMESTEPS
    end_criteria: float = 0.0

    def __post_init__(self) -> None:
        if self.max_timesteps < 1:
            raise EnvelopeError("max_timesteps must be at least 1")
        if not 0.0 <= self.end_criteria < 1.0:
            raise EnvelopeError("end_criteria must be in [0, 1)")

    @property
    def reproducible(self) -> bool:
        return self.end_criteria == 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_timesteps": self.max_timesteps,
            "end_criteria": self.end_criteria,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Termination:
        return cls(
            max_timesteps=int(data.get("max_timesteps", DEFAULT_TIMESTEPS)),
            end_criteria=float(data.get("end_criteria", 0.0)),
        )


@dataclass(frozen=True)
class MeshGrid:
    """The grid, as explicit line positions.

    The lines are computed on the FreeCAD side and carried here verbatim. The
    driver never meshes, so the array a mesh preview showed is the array that
    was solved rather than a prediction of it.

    :param params: The policy the lines came from. Provenance only. Re-running
        the policy is not the driver's job, and would break the guarantee above.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for dim, axis in enumerate(AXIS_NAMES):
            lines = np.asarray(self[dim], dtype=float)
            object.__setattr__(self, axis, lines)
            if lines.ndim != 1 or lines.size < 5:
                raise EnvelopeError(
                    f"grid axis {axis} has {lines.size} lines; openEMS' ports "
                    "require at least 5 along any axis they measure on"
                )
            if not np.all(np.diff(lines) > 0):
                raise EnvelopeError(f"grid axis {axis} is not strictly increasing")

    def __getitem__(self, dim: int) -> np.ndarray:
        return (self.x, self.y, self.z)[dim]

    def moved(self, offset: tuple[float, float, float]) -> MeshGrid:
        """The same grid, translated. Every spacing in it is a difference and is
        untouched, so this is a change of coordinates rather than a different
        mesh."""
        return replace(
            self,
            x=self.x + offset[0],
            y=self.y + offset[1],
            z=self.z + offset[2],
        )

    @property
    def cell_count(self) -> int:
        """Lines rather than intervals. See ``grid.MeshLines.cell_count``."""
        return int(np.prod([len(self[d]) for d in range(3)]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "z": self.z.tolist(),
            "params": self.params,
        }

    def digest(self) -> str:
        """Content hash of the grid alone.

        This is not the envelope's digest. Raising ``MaxTimesteps`` moves the
        envelope and changes nothing about the mesh, and a preview reported
        stale for that would soon be ignored. An envelope digest also needs an
        excitation chosen, which a preview does not have and does not need:
        every run in a sweep shares one grid.

        The grid is canonicalised the same way the envelope is, so a grid that
        survived a save and reload hashes to what it hashed before.
        """
        payload = json.dumps(canonical(self.to_dict()), indent=2, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeshGrid:
        return cls(
            x=np.asarray(data["x"], dtype=float),
            y=np.asarray(data["y"], dtype=float),
            z=np.asarray(data["z"], dtype=float),
            params=data.get("params", {}),
        )


def check_mode(mode: str, subject: str) -> str:
    """Return the waveguide mode name, or raise.

    Two callers ask. :class:`Port` asks on the way in. The document layer asks
    before building a port, so a mode typed into a property field is refused as
    a fault in the model rather than reaching the panel's catch-all as a
    traceback. The rule is stated here once; a second copy in the translator
    would be a second rule.
    """
    match = _MODE_PATTERN.match(mode)
    if match is None:
        raise EnvelopeError(
            f"{subject}: {mode!r} is not a mode this adapter can run; openEMS' "
            "RectWGPort supports TE modes only (ports.py:434), so use TE10, "
            "TE01, TE20 ..."
        )
    if match.group(1) == match.group(2) == "0":
        raise EnvelopeError(
            f"{subject}: TE00 has no field. Its cutoff is zero and its mode "
            "functions are identically zero, so the run would excite nothing "
            "and return 0/0 after its full runtime"
        )
    return mode


def check_timestep_factor(value: float) -> float:
    """Return ``value`` if openEMS would act on it, and raise otherwise.

    The bound is stated here rather than at each of the two call sites, which
    are the envelope's own invariant and the document layer reading a property
    the user typed. The bound and its reason then exist once.

    The value is refused rather than clamped, because openEMS' answer to a
    factor outside ``(0, 1]`` is not an error. Against v0.0.36-157-gc4ce357:

    * at 0 or below, ``Operator::SetTimestepFactor`` prints "invalid timestep
      factor, skipping!" and the run steps at full size;
    * above 1 nothing is printed at all. ``openEMS::SetupFDTD`` guards the call
      with ``if (m_TS_fac<1)``, so the routine that would complain is never
      reached. The run steps at full size while ``openEMS::Write2XML`` writes
      ``TimeStepFactor="2"`` into the XML, which is the file a bug report
      carries.

    Either way a user who reduced the step to stop a run diverging gets the
    diverging run back, and the workbench reports nothing. That is a silent
    no-op, so the value never leaves here.
    """
    if not 0.0 < value <= 1.0:
        raise EnvelopeError(
            f"timestep_factor must be above 0 and at most 1, got {value:g}. "
            "openEMS scales its own timestep by it, and ignores anything else"
        )
    return value


@dataclass(frozen=True)
class Problem:
    """Everything the driver needs to run one excitation.

    One :class:`Problem` is one solve. An N-port S-matrix is N of these, each
    exciting a different port. The loop belongs to the adapter rather than to
    the envelope, so that a single run stays reproducible on its own.

    :param timestep_factor: Scales the timestep openEMS calculates for itself,
        in ``(0, 1]``. See :func:`check_timestep_factor`. One asks for the
        engine's own step, which is what openEMS uses anyway: it applies the
        factor only when the factor is below one. A factor below one buys
        stability on a grid the CFL bound alone does not settle, and costs
        simulated time, since the same ``max_timesteps`` then covers
        proportionally less of it. Pre-flight reports that.

        A new field bumps ``SCHEMA_VERSION`` even where its absence would
        default correctly. The key is absent only in the file: :meth:`to_dict`
        emits it either way and :meth:`digest` runs over the re-serialised form,
        so an older envelope digests to something its own ``envelope.sha256``
        never said, and :meth:`Results.matches` then denies that a results file
        came from the envelope beside it.

    :param smallest_response: The smallest magnitude in S this study reads, in
        ``(0, 1]``, where one is full scale. Nothing in the run is solved
        differently for it. ``residual.unfinished`` weighs the leakage of a
        truncated record against this value: a leak that is negligible beside a
        response of one is the whole of a stopband. The value is declared and
        never inferred, because a sweep cannot tell a term that is the point of
        the exercise from one that is a rounding error.

    :param grown_by: What share of its cell a curved conductor is grown by on
        the way to the engine, in ``[0, 0.5]``. See :mod:`.staircase`, whose
        ``GROWN_BY`` the translation writes here.

        The share is carried in the file rather than read off that module, so
        that the structure the driver builds is a function of the file. An
        envelope attached to a bug report and re-run against a different
        constant would build a different conductor and report nothing about it.
        The digest would not notice either, because it is taken over the
        envelope, which does not hold the constant.

        Zero hands a curved surface over as drawn, which is the one way to price
        the correction against not making it. Half is the ceiling, half a cell
        being what the sampling gives up: a boundary rounds down to the last
        lattice plane inside the metal, and over a surface meeting the grid at
        every phase the mean of that is half a cell. Growing by more stands the
        surface the engine builds outside the drawing rather than on it, which
        is the fault the growth exists to remove, with its sign turned round.

    :param pinned_clearance: What share of its cell a flat conductor face square
        to an axis is displaced into the void by, in ``[0, 0.5)``. See
        :mod:`.staircase`, whose ``PINNED_CLEARANCE`` the translation writes
        here.

        It is carried for the reason ``grown_by`` is, and it answers a different
        question. ``grown_by`` says where a sampled boundary lands; this says
        whether there is a boundary at all. The mesher pins a line to such a
        face, and a point lying on a face leaves openEMS' containment segment
        with nothing to be sure about. Without the displacement the pinned line
        can therefore read as air, the tangential field on the conductor's plane
        is never zeroed, and the wall the drawing states is not built.

        Zero hands the face over as drawn, which is what prices it. Half a cell
        is where the field edge normal to the face is sampled on the void side,
        so a clearance reaching it zeroes an edge belonging outside the metal
        and stands the wall a cell inside the drawing, which is the same fault
        with its sign turned round.
    """

    frequency: Frequency
    grid: MeshGrid
    materials: tuple[Material, ...]
    solids: tuple[Solid, ...]
    ports: tuple[Port, ...]
    boundary: tuple[str, str, str, str, str, str] = ("PML_8",) * 6
    termination: Termination = field(default_factory=Termination)
    length_unit: float = 1e-3
    threads: int = 0
    timestep_factor: float = 1.0
    smallest_response: float = 1.0
    grown_by: float = GROWN_BY
    pinned_clearance: float = PINNED_CLEARANCE
    title: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "materials", tuple(self.materials))
        object.__setattr__(self, "solids", tuple(self.solids))
        object.__setattr__(self, "ports", tuple(self.ports))
        object.__setattr__(self, "boundary", tuple(self.boundary))

        check_timestep_factor(self.timestep_factor)
        # length_unit=0 makes grid.SetDeltaUnit(0) and divides by zero in
        # preflight; a negative thread count goes straight to numThreads. Both
        # are scale factors on everything else in the file.
        _finite(self.length_unit, "length_unit", low=0.0, strict=True)
        _finite(self.threads, "threads", low=0.0)
        _finite(self.smallest_response, "smallest_response", low=0.0, high=1.0, strict=True)
        # Half a cell is the mean of what the sampling gives up, so it is all
        # there is to answer for. The shipped share sitting at that ceiling is a
        # separate fact about the shipped share.
        _finite(self.grown_by, "grown_by", low=0.0, high=0.5)
        # Zero hands the face over as drawn, and prices the displacement
        # against not making it. Half a cell is where the normal field edge on
        # the void side is sampled, and a clearance reaching it zeroes that edge.
        _finite(self.pinned_clearance, "pinned_clearance", low=0.0, high=0.5, high_strict=True)
        if len(self.boundary) != 6:
            raise EnvelopeError(
                f"boundary needs 6 entries (xmin xmax ymin ymax zmin zmax), "
                f"got {len(self.boundary)}"
            )
        if not self.ports:
            raise EnvelopeError("a problem with no ports has nothing to measure")

        known = {material.name for material in self.materials}
        if len(known) != len(self.materials):
            raise EnvelopeError("two materials share a name")
        for solid in self.solids:
            if solid.material not in known:
                raise EnvelopeError(
                    f"solid {solid.name!r} references material "
                    f"{solid.material!r}, which is not defined"
                )
        for port in self.ports:
            if port.metal and port.metal not in known:
                raise EnvelopeError(
                    f"{port.name} references metal {port.metal!r}, which is not defined"
                )

        numbers = [port.number for port in self.ports]
        if len(set(numbers)) != len(numbers):
            raise EnvelopeError(f"port numbers are not unique: {sorted(numbers)}")
        excited = [port.number for port in self.ports if port.excite]
        if len(excited) != 1:
            raise EnvelopeError(
                f"exactly one port must be excited per run, but {len(excited)} "
                f"are ({excited or 'none'}); an S-matrix is one run per port"
            )

    @property
    def excited_port(self) -> Port:
        return next(port for port in self.ports if port.excite)

    def as_given(self, solid: Solid) -> tuple[tuple[float, float, float], ...] | None:
        """The triangulation this solid reaches the engine with, where it differs.

        A conductor's surface is not handed over as it was drawn. openEMS
        decides a point-sampled material by one sample per field edge, so the
        surface is grown by the share of a cell this problem carries, and a flat
        face square to an axis is displaced by its clearance. See
        :mod:`.staircase`. Asking the drawing instead answers about a shape no
        run builds.

        The decision lives here rather than at each caller. Several callers ask
        it: what the driver builds, whether a port's plane is in metal, whether
        the grid still holds a conductor whole. A second copy would stop
        agreeing with this one.

        The answer is ``None`` where the drawn surface is what is handed over: a
        box, which is pinned on every face and rounded nowhere; a sheet, which is
        modelled at its plane and whose rim is a separate question nothing has
        measured; and a dielectric, which is averaged over quarter cells rather
        than sampled and so carries no such bias.
        """
        rounded = {material.name for material in self.materials if material.kind in CONDUCTOR_KINDS}
        if solid.is_sheet or not solid.is_mesh or solid.material not in rounded:
            return None
        lines = tuple(self.grid[dim] for dim in range(len(AXIS_NAMES)))
        return grown(solid.vertices, solid.faces, lines, self.grown_by, self.pinned_clearance)

    def exciting(self, number: int) -> Problem:
        """The same problem with a different port driven."""
        if number not in {port.number for port in self.ports}:
            raise EnvelopeError(f"no port numbered {number}")
        return replace(
            self,
            ports=tuple(replace(p, excite=(p.number == number)) for p in self.ports),
        )

    def at_the_origin(self) -> tuple[Problem, tuple[float, float, float]]:
        """This problem with its minimum corner at the origin, and the offset.

        The offset is returned because the structure the engine is handed is
        not the one the user drew, and the XML beside it is in these
        coordinates. A reader of that file has to be told the offset, and
        anything read back off the engine by position would have to subtract it.
        Nothing this adapter reads back is a position, and an S-matrix is not
        one, so the offset is provenance rather than a correction to apply.

        The translation is applied where the envelope is turned into a structure
        and nowhere earlier, so the mesh preview, a pre-flight message and the
        envelope on disk all stay in the coordinates the user drew in. See
        :func:`origin_offset` for why the engine is given no other choice.
        """
        offset = origin_offset(self.solids, self.ports, self.grid)
        return (
            replace(
                self,
                grid=self.grid.moved(offset),
                solids=tuple(solid.moved(offset) for solid in self.solids),
                ports=tuple(port.moved(offset) for port in self.ports),
            ),
            offset,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "title": self.title,
            "length_unit": self.length_unit,
            "threads": self.threads,
            "timestep_factor": self.timestep_factor,
            "smallest_response": self.smallest_response,
            "grown_by": self.grown_by,
            "pinned_clearance": self.pinned_clearance,
            "frequency": self.frequency.to_dict(),
            "termination": self.termination.to_dict(),
            "boundary": list(self.boundary),
            "materials": [m.to_dict() for m in self.materials],
            "solids": [s.to_dict() for s in self.solids],
            "ports": [p.to_dict() for p in self.ports],
            "grid": self.grid.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Problem:
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise EnvelopeError(
                f"envelope schema version {version!r} cannot be read by this "
                f"adapter, which speaks version {SCHEMA_VERSION}"
            )
        return cls(
            frequency=Frequency.from_dict(data["frequency"]),
            grid=MeshGrid.from_dict(data["grid"]),
            materials=tuple(Material.from_dict(m) for m in data["materials"]),
            solids=tuple(Solid.from_dict(s) for s in data["solids"]),
            ports=tuple(Port.from_dict(p) for p in data["ports"]),
            boundary=tuple(data.get("boundary", ("PML_8",) * 6)),
            termination=Termination.from_dict(data.get("termination", {})),
            length_unit=float(data.get("length_unit", 1e-3)),
            threads=int(data.get("threads", 0)),
            timestep_factor=float(data.get("timestep_factor", 1.0)),
            smallest_response=float(data.get("smallest_response", 1.0)),
            grown_by=float(data.get("grown_by", GROWN_BY)),
            pinned_clearance=float(data.get("pinned_clearance", PINNED_CLEARANCE)),
            title=data.get("title", ""),
        )

    def to_json(self) -> str:
        """The envelope as the driver receives it, canonicalised.

        The rounding happens here rather than at the digest, so the bytes
        hashed are the bytes written. ``sha256sum openems.json`` then agrees
        with :meth:`digest`; otherwise the provenance could only be checked by
        code that reimplements the rounding. See :data:`CANONICAL_DIGITS`.

        ``allow_nan=False`` is set because Python's default writes the bare
        tokens ``NaN`` and ``Infinity``, which RFC 8259 does not permit. A file
        holding one could not be read by ``jq``, by a browser, or by any
        non-Python parser, and the envelope has to be attachable to a bug
        report. The fields are validated on the way in, and this is the backstop
        that turns a leak into an exception rather than a corrupt artefact.
        """
        return json.dumps(canonical(self.to_dict()), indent=2, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> Problem:
        return cls.from_dict(json.loads(text))

    def digest(self) -> str:
        """Content hash of the envelope, for provenance.

        The hash is taken over the serialised form, so it changes if and only
        if what the driver actually receives changes. A result whose digest does
        not match its envelope was produced by a different input.
        """
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
