# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The openEMS adapter's private input description.

This is the envelope that crosses the process boundary: the workbench builds a
:class:`Problem` in FreeCAD's Python, serialises it, and a separate interpreter
that owns the openEMS bindings reads it back and solves it.

**It is private to this adapter.** No other adapter reads it, nothing outside
``Solvers/openems/`` constructs it, and it is not an interchange format. Its two
jobs are crossing the process boundary and being attachable to a bug report -
which is why it is JSON and not a pickle, and why it carries provenance it does
not strictly need to run.

Imports numpy and the standard library only, because both sides of the process
boundary have to read it: FreeCAD's interpreter (no openEMS) to write it, and
the solver's interpreter (no FreeCAD) to execute it.

Units
-----

Lengths are in millimetres and frequencies in Hertz, everywhere, with no
exceptions and no per-field overrides - ``length_unit`` exists to be *written
into the file* so it is self-describing, not to be varied. openEMS works in
whatever unit its grid is told to use; the driver sets that from this field and
nothing else converts.
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
from .surface import sheet_fault, surface_fault

#: Bump whenever :meth:`Problem.to_dict` changes shape. The digest is computed
#: over the re-serialised form, so an envelope written under an older version
#: comes back from this adapter carrying keys it was written without, and
#: digests differently - a sim directory whose ``envelope.sha256`` and
#: ``results.json`` agree with each other would still have ``Results.matches()``
#: deny that the results came from the envelope. Refusing an unknown version by
#: name is the honest answer.
SCHEMA_VERSION = 4

AXIS_NAMES = ("x", "y", "z")

#: How far off its own plane a sheet's vertex may sit, in millimetres. A drawing
#: is flat to within a kernel tolerance rather than exactly, and this is three
#: orders above FreeCAD's own 1e-7 mm - loose enough for anything the kernel
#: calls planar, tight enough that a face which is genuinely not flat is caught
#: rather than flattened.
SHEET_FLATNESS = 1e-4

#: Re-exported, not redefined: see :data:`Microwave.units.SPEED_OF_LIGHT`. The
#: name is here because this is the one module every other one in the adapter
#: already imports, and the figure is not, because a constant of the vacuum
#: written down twice is a fact that can drift.
SPEED_OF_LIGHT = units.SPEED_OF_LIGHT
DIMENSIONS = len(AXIS_NAMES)

#: Significant digits kept when the envelope is written.
#:
#: The envelope is canonicalised on the way out so that one problem has one
#: serialisation. It does not otherwise: FreeCAD writes ``App::PropertyFloat``
#: to its document with about 14 significant digits, short of the 17 an IEEE-754
#: double needs to survive intact, so a property holding 1/120 comes back from a
#: save as 0.0083333333333333. PropertyQuantity, PropertyLength and
#: PropertyPrecision all behave identically, so it cannot be dodged by choosing
#: a different one.
#:
#: The effect on the model is nil - the same document before and after a save
#: meshes and solves the same. The effect on *provenance* is not: ``digest()``
#: claims a result whose digest differs came from a different input, and without
#: this a save and reopen is enough to break that claim.
#:
#: Twelve digits is 0.1 picometres over a 100 mm domain, twenty orders of
#: magnitude below anything the mesher resolves, and two digits clear of where
#: FreeCAD truncates. It applies to the whole envelope rather than to the
#: properties that happen to trip it today, because any user-entered dimension
#: can: a substrate 1.6/3 mm thick truncates exactly the same way.
CANONICAL_DIGITS = 12

#: Timesteps a run takes when nothing says otherwise. See
#: :class:`Termination` for why this is a run length rather than a ceiling.
#:
#: What it has to be long enough for is the excitation decaying into numerical
#: noise, so that where the series is truncated stops moving the DFT. That is a
#: property of the model, not a constant: each acceptance gate sets its own,
#: found by lengthening the run until the extracted figure stopped changing.
#: This default sits above all of them with room to spare, and the price of the
#: margin is linear in the step count.
#:
#: Nothing yet checks that a given model decayed within it. Until something
#: does, too long is the safe direction to be wrong in.
DEFAULT_TIMESTEPS = 30000


def canonical(value: Any) -> Any:
    """One problem, one serialisation: floats to :data:`CANONICAL_DIGITS`.

    Idempotent, so serialising a round-tripped envelope reproduces it exactly.
    Integers and booleans are left alone - they are exact already, and coercing
    them to float would change the JSON they produce.
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


#: Padding sentinel: the structure continues out *through* the absorber rather
#: than ending inside the domain. This is what makes a transmission line
#: infinite - substrate and trace run into the PML, so the line never sees an
#: end and never reflects off one. See :mod:`.write` for what it does to the
#: domain.
THROUGH = "through"

MATERIAL_KINDS = frozenset({"dielectric", "lossy_dielectric", "pec", "conducting_sheet"})
PORT_KINDS = frozenset({"microstrip", "lumped", "rect_waveguide", "coaxial"})

#: Material kinds openEMS models as conductors. ``driver.build_material`` hands
#: a ``pec`` its name alone and a ``conducting_sheet`` its conductivity and
#: thickness - neither is given ``epsilon`` or ``mu``, so for these two kinds
#: those fields reach the solver nowhere. :class:`Material` refuses them rather
#: than letting them sit there inert, because they are not inert: they reach the
#: mesher and the pre-flight checks. See ``Material.__post_init__``.
CONDUCTOR_KINDS = frozenset({"pec", "conducting_sheet"})

#: Port kinds that integrate a voltage across a gap, and so need to be told
#: which way. A waveguide port excites a *mode* over its whole cross-section and
#: has no such axis.
_NEEDS_EXCITATION_AXIS = frozenset({"microstrip", "lumped"})

#: Mode names openEMS understands, e.g. TE10. TE only, and not TE00.
#:
#: Upstream ``RectWGPort`` raises "Currently only TE-modes are supported!" for
#: anything else (openEMS `python/openEMS/ports.py:434`), so a TM mode
#: reached the driver as a crash rather than a named refusal. TE00 is worse: it
#: builds cleanly with ``kc = 0`` and mode functions that are identically zero
#: (ports.py:450-458), so the excitation is nothing at all and the run returns
#: 0/0 after its full runtime - the worst failure class there is, with no
#: error and no refusal.
_MODE_PATTERN = re.compile(r"^TE(\d)(\d)$")

#: Port kinds that lay down a conductor of their own, which the mesh must
#: resolve. A waveguide's walls are boundary conditions, not geometry.
_LAYS_CONDUCTOR = frozenset({"microstrip"})

#: Port kinds whose planes openEMS moves onto the grid rather than leaving where
#: they were asked for. ``MSLPort`` takes an ``argmin`` over the propagation
#: lines (ports.py:255, :296) and a lumped port is snapped by ``SnapBox2Mesh``.
#: A coaxial port snaps the same way this adapter's builder does it, by an
#: ``argmin`` over the same lines. A ``rect_waveguide`` plane is not moved at
#: all - with no line on it nothing is discretised, and the run returns 0/0
#: after its full time, which is why :meth:`required_lines` pins those two
#: coordinates instead.
_SNAPS_TO_THE_GRID = frozenset({"microstrip", "lumped", "coaxial"})

#: Port kinds that extract by differencing three probes across the measurement
#: plane. Others integrate a mode over one plane and have no difference to take.
#: Here rather than in :mod:`.preflight` so that :meth:`Port.wanted_lines` can
#: ask the same question - preflight imports this module, so the dependency
#: only runs one way.
_USES_PROBE_TRIPLET = frozenset({"microstrip", "coaxial"})

#: Port kinds whose excitation is a uniform field imposed across the gap, so
#: what it launches is the line's mode *plus* the evanescent content needed to
#: square that shape with the real one - which the probes have to stand clear
#: of. A coaxial port is not one: its excitation carries the ``1/r`` radial
#: profile of the mode itself, so the shape it imposes is the shape it wants.
_EXCITES_A_UNIFORM_GAP = frozenset({"microstrip"})

#: Port kinds whose transverse cross-section is a circle rather than a
#: rectangle, so the box corners bound a bore and the port carries a radius
#: the box cannot express.
_IS_ROUND = frozenset({"coaxial"})

#: How far a round port's two transverse extents may disagree, relatively.
#:
#: They are one diameter read twice off one bounding box, so what separates
#: them is that box's own rounding. A bore drawn oval by any amount a drawing
#: can express is orders above this, which is what makes the test worth making:
#: it says the box bounds a circle rather than merely a square.
_ROUND_TOLERANCE = 1e-6


class EnvelopeError(Exception):
    """The envelope is malformed, or describes something openEMS cannot run."""


def _axis(value: Any, what: str) -> int:
    """Accept ``0/1/2`` or ``'x'/'y'/'z'``, always store an index."""
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
) -> float:
    """One numeric field, checked for being a number before being checked at all.

    Every range guard in this module was a ``<`` comparison, and every ``<``
    comparison is ``False`` for NaN - so ``epsilon=nan``, ``mu=0``,
    ``length_unit=0`` and a NaN frequency all passed validation and went to the
    engine. ``mu`` was never checked at all, and zero reached
    ``write.plan_mesh`` as a ``ZeroDivisionError``.

    Order matters: finiteness first, because a NaN that reaches a bound test
    silently passes it, which is the whole defect.
    """
    number = float(value)
    if not math.isfinite(number):
        raise EnvelopeError(f"{what}: {number} is not a finite number")
    if low is not None and (number <= low if strict else number < low):
        raise EnvelopeError(
            f"{what}: {number:g} must be {'above' if strict else 'at least'} {low:g}"
        )
    if high is not None and number > high:
        raise EnvelopeError(f"{what}: {number:g} must be at most {high:g}")
    return number


def _point(value: Sequence[float], what: str) -> tuple[float, float, float]:
    values = tuple(float(v) for v in value)
    if len(values) != 3:
        raise EnvelopeError(f"{what}: expected 3 coordinates, got {len(values)}")
    if not all(np.isfinite(values)):
        raise EnvelopeError(f"{what}: coordinates must be finite, got {values}")
    return values  # type: ignore[return-value]


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

    **openEMS is only sound in non-negative coordinates.**
    ``CSPrimPolyhedron::IsInside`` counts how many faces a segment crosses on
    its way to a point it takes to be outside, and it builds that point by
    scaling the primitive's own maximum corner away from the origin
    (``CSXCAD/src/CSPrimPolyhedron.cpp:229``). Scaling a *negative* coordinate
    moves it further from the origin too, which is toward the solid rather than
    away from it - so for a solid whose every maximum is at or below the origin
    the endpoint lands inside the shape, and the parity inverts: the object
    reads as hollow, and the space around it as filled.

    The engine is therefore handed a structure that has no negative coordinate
    in it at all, rather than one checked for the corner where that particular
    ray goes wrong. Unconditionally, so it is a placement and not a repair:
    every structure is put in the same place, so the path is the one every run
    takes and the property holds by construction. A conditional would be a
    branch that encodes another project's defect, run on almost no model, and
    trusted on all of them.

    It costs nothing to be right about, because a translation is a symmetry of
    Maxwell's equations: the whole structure moves together, so every length,
    every gap and every cell is what it was. What it does *not* do is move one
    solid to suit the engine - where a solid sits relative to the rest of the
    model is the device itself.
    """
    # Every vertex, and not the box that is said to bound them: the ray endpoint
    # is built from a bounding box openEMS recomputes from the vertices
    # themselves, so a vertex outside the box it was given would be outside the
    # guarantee too.
    corners = [solid.lower for solid in solids]
    corners += [vertex for solid in solids for vertex in solid.vertices]
    corners += [port.start for port in ports] + [port.stop for port in ports]
    if grid is not None:
        corners.append(tuple(float(grid[dim][0]) for dim in range(DIMENSIONS)))
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
        everything else here, and a *loss* parameter - it feeds the surface
        impedance. The sheet stays geometrically flat, so this never enters the
        mesh. (CSXCAD wants this one field in metres; :mod:`.driver` converts at
        the boundary.)
    :param measured_at: The frequency, in Hz, that the loss ``kappa`` was built
        from was quoted at; zero when nothing recorded it. The solver never sees
        it - openEMS is handed ``kappa`` and nothing else - and it travels
        anyway, because ``kappa`` is fixed for the whole run and therefore
        stands for that loss exactly at one frequency. Whether that frequency is
        this band's is a question only this field can answer, and pre-flight is
        where every route asks it, including the one that is handed an envelope.
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
        # Zero reaches write.plan_mesh as `cap / sqrt(eps * mu)` and raises
        # ZeroDivisionError; negative raises a math domain error; NaN goes all
        # the way to the engine.
        _finite(self.mu, f"{subject}: relative permeability", low=0.0, strict=True)
        _finite(self.kappa, f"{subject}: kappa", low=0.0)
        _finite(self.conductivity, f"{subject}: conductivity", low=0.0)
        _finite(self.thickness, f"{subject}: thickness", low=0.0)
        _finite(self.measured_at, f"{subject}: measured_at", low=0.0)
        # After the range checks, so a conductor at epsilon nan is told it is not
        # a number rather than that it is not 1 - the more specific complaint
        # wins, and this one only makes sense about a value that is otherwise
        # legal.
        if self.kind in CONDUCTOR_KINDS and (self.epsilon != 1.0 or self.mu != 1.0):
            # Refused rather than ignored, and refused *here* rather than in the
            # document layer, because `python -m ...driver openems.json` is a
            # documented entry point that never passes through it.
            #
            # Ignoring would be the worse fault: these two fields reach the
            # solver nowhere (``driver._add_material`` passes neither), but they
            # do reach `policy._wavelength`, which takes `max(epsilon * mu)`
            # over every material and sizes the whole grid from it, and two
            # pre-flight thresholds computed the same way. A permittivity left
            # on a conductor therefore moves the cell count and the clearance
            # thresholds, on a value openEMS never sees.
            raise EnvelopeError(
                f"{subject}: a {self.kind} conductor carries relative "
                f"permittivity {self.epsilon:g} and permeability {self.mu:g}. "
                "openEMS is given a conductor's conductivity and nothing else, "
                "so neither reaches the solver - but both reach the mesher, "
                "where they resize every cell in the model. Leave them at 1, or "
                "make this a dielectric"
            )
        # The same fault as the one above, and the sharper case:
        # ``driver._add_material`` passes kappa in its ``lossy_dielectric``
        # branch and in no other, so anywhere else the loss is dropped between
        # here and the engine and the run comes back lossless - a plausible
        # answer to a question that was not asked. Pre-flight reads this field to
        # decide whether a material's loss was quoted in this band, and that
        # reading is only about materials whose loss arrives.
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

    Two shapes, both carrying ``lower`` and ``upper``, so that everything which
    only wants to know *where* a solid is - the domain, the absorber's
    reservation, a pre-flight check - reads one field and does not care which
    kind it has.

    Without ``faces`` it is an axis-aligned box, and equal corners give a sheet.
    With them it is the triangulated boundary of whatever was drawn, and the
    corners are that triangulation's own extent. The grid stays rectilinear
    either way: what a triangulation buys is that the *shape* is held exactly,
    so the staircase is the grid's and can be priced, rather than being
    introduced silently by squaring the drawing off first.

    A triangulation is checked here, on the way in, rather than trusted. See
    :mod:`~.surface`: an open one is read by the engine as a sheet, contains no
    point at all, and takes the object out of the simulation without any message
    that can be told from a benign one.
    """

    material: str
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    priority: int = 0
    label: str = ""
    #: The boundary as triangles, and the vertices they index. Empty for a box.
    vertices: tuple[tuple[float, float, float], ...] = ()
    faces: tuple[tuple[int, int, int], ...] = ()
    #: The axis a *sheet* is flat on, or ``None`` for a solid. A sheet's
    #: triangles cover its area rather than bounding a volume, so they are a
    #: different kind of thing from the same fields on a solid: openEMS takes
    #: them as flat polygons at one elevation, and the closedness a solid is
    #: held to would refuse every one of them.
    sheet_normal: int | None = None
    #: The thickness this conductor was given because the drawing carried none,
    #: in mm, or zero where the drawing carried its own. Nothing is built from
    #: it - the vertices already bound the metal - and it crosses the envelope
    #: only so that pre-flight can name a length the drawing did not carry.
    thickened: float = 0.0
    #: The finest cell this solid's own demands may ask the mesher for, in mm,
    #: or zero to ask at the policy's sizes. A mesh region set to coarsen names
    #: the object rather than a box, because the grid is separable and a box
    #: spends itself on a slab through the model on each axis.
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
        if self.is_sheet:
            axis = self.sheet_normal
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
            # Where a solid sits relative to the origin is not asked here. It
            # decides whether openEMS reads the shape or its complement, and
            # :func:`origin_offset` is what settles it - for the whole
            # structure at once, which is the only level at which the answer is
            # a translation rather than a distortion.
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
        """The same solid, translated. Every coordinate it carries, or none."""
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

    ``start`` and ``stop`` are *opposite corners of the port box*, not a sorted
    bounding box: their difference along ``excitation_axis`` sets which way the
    excitation points, and along ``propagation_axis`` which way is "into" the
    structure. For a microstrip that means ``start`` sits on the trace and
    ``stop`` on the ground plane. Sorting them would silently reverse the port.

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
    #: The conductor this port lays down. Microstrip only - a waveguide's
    #: walls are boundary conditions, not geometry.
    metal: str = ""
    #: ``TE10`` and the like. Waveguide ports only, and in the textbook
    #: convention: the first digit counts half-waves across the **broad** wall,
    #: so ``TE10`` is the dominant mode however the guide is drawn.
    #: :meth:`waveguide_arguments` renumbers it onto openEMS' axes.
    mode: str = ""
    #: The inner conductor's radius, for the kinds in :data:`_IS_ROUND` and
    #: refused for every other. The *outer* radius is not a field: it is half
    #: the box's transverse extent, so the bore the probes reach across cannot
    #: disagree with the volume the port claims - the same reason a waveguide
    #: port reads ``a`` and ``b`` off its box rather than carrying them.
    inner_radius: float = 0.0
    excite: bool = False
    feed_shift: float = 0.0
    measurement_shift: float = 0.0
    feed_resistance: float | None = None
    #: What the S-parameters are to be reported against. ``None`` means the port
    #: itself - whatever impedance it turns out to have, per frequency. That is
    #: not a gap to be filled with a default: this adapter renormalises nothing,
    #: so an unstated reference is exactly the basis the probes already measure
    #: in, and it is the only expressible answer for a dispersive guide.
    reference_impedance: float | None = None
    priority: int = 10
    label: str = ""

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
        """Coerce the excitation axis, or refuse a kind that has no business with one."""
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
            # A lumped port *is* its resistance: openEMS builds a resistive sheet
            # across the gap, so this one is structure rather than reporting.
            # It cannot fall back to reference_impedance, which says what the
            # answer is reported against and is unset precisely when the port is
            # reported against itself - a value that would then decide how much
            # metal is laid across a gap by way of an unrelated question.
            raise EnvelopeError(
                f"port {self.number}: a lumped port is a resistance across a gap "
                "and must state it; 0 asks for a short"
            )

    def _check_waveguide_box(self) -> None:
        """Everything a rect_waveguide port asks of its box and its mode.

        **Neither shift.** ``AddRectWaveGuidePort`` takes none: the excitation
        sits on the box's near face and the probes on its far one, full stop.
        Accepting one would be worse than useless - pre-flight validates the
        feed position it computes *from* ``feed_shift``, so a nonzero value moves
        the checked position while the real excitation stays put, and the
        absorber check can be walked straight past.

        **A mode this adapter can name, and a box that spans the guide.** The two
        are one question: :meth:`waveguide_arguments` reads ``a`` and ``b`` off
        the box's transverse extents rather than carrying them as fields, so the
        cross-section the user drew is what the mode is computed from. A box flat
        in a transverse axis therefore describes no guide at all.
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
        the same diameter read twice and :attr:`outer_radius` is half of it.
        That is what makes the *outer* radius underivable from anything the user
        can contradict; the inner one has nothing to be read off and is carried.

        A kind that is not round is refused a radius rather than ignoring it,
        for the reason :meth:`_settle_excitation_axis` refuses an axis: a field
        that reaches the solver nowhere still reaches the editor.
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

        The far face is *inside* the port, so a measurement plane sitting exactly
        on it is legal - and that is the ordinary case, since an unset Length
        ends the box there. ``length`` is reconstructed from the corners, so it
        equals the shift only to within the rounding of
        ``start + direction * length``, and a box that refuses its own
        measurement plane is the failure that buys the tolerance. It is a
        tolerance on the reconstruction, not on the physics.
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
        """The two resistances, separately.

        Zero divides them, and it divides them the opposite way round, which is
        why they are not checked together: a lumped port's zero lays metal across
        the gap and is the only way to ask for one, while a zero reference
        impedance renormalises an ideal matched line to ``|S12|`` above one -
        gain, non-reciprocal, and finite enough to reach a Touchstone file. See
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
        """The two axes across the propagation direction, ascending.

        Ascending, which for propagation along **y** is not the order openEMS
        pairs them in - see :attr:`mode_axes`. Use this only where the pair is
        a set.
        """
        return tuple(a for a in range(DIMENSIONS) if a != self.propagation_axis)

    @property
    def outer_radius(self) -> float:
        """Half the bore's diameter, off the box rather than off a field.

        The mean of the two transverse half-extents, so the answer does not
        depend on which axis is asked first - :attr:`transverse_axes` and
        :attr:`mode_axes` disagree about that order, and only one of them is
        the pair openEMS binds.
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

        ``ny_P = (p+1) % 3`` and ``ny_PP = (p+2) % 3`` (``ports.py``:416-417).
        Cyclic, so for propagation along y this is ``(z, x)`` - the reverse of
        :attr:`transverse_axes`.
        """
        axis = self.propagation_axis
        return (axis + 1) % DIMENSIONS, (axis + 2) % DIMENSIONS

    def waveguide_arguments(self, length_unit: float) -> tuple[float, float, str]:
        """``(a, b, mode)`` exactly as ``AddRectWaveGuidePort`` takes them.

        The pair is **positional, not sorted**: openEMS binds ``a`` to
        :attr:`mode_axes`\\ ``[0]`` and ``b`` to ``[1]``, in ``kc`` *and* in the
        mode functions, so ``a`` is the extent along the first of those axes
        whether or not it is the broad wall. ``mode`` here arrives in the
        textbook convention the document uses - the first digit counts
        half-waves across the broad wall - and is renumbered onto the axes.

        Handing openEMS a sorted pair instead lands the broad wall's *length* on
        whichever axis happens to come first, and the field it then launches is
        not a mode of the guide that was drawn: total reflection, no
        transmission, more power out than in, and measurement planes nowhere
        near the geometry - while cutoff and ``Z_ref`` still look right, both
        being analytic in ``a`` and ``b`` alone. Through this method the same
        guide drawn on two axes agrees to every digit the gate prints, and the
        waveguide gate asserts both drawings.

        Derived from the port box rather than carried as fields, so the geometry
        cannot disagree with the numbers the mode is computed from.
        """
        first, second = self.mode_axes
        extents = tuple(
            abs(self.stop[axis] - self.start[axis]) * length_unit for axis in (first, second)
        )
        match = _MODE_PATTERN.match(self.mode)
        if match is None:
            # A rect_waveguide port cannot get here - __post_init__ refuses a
            # mode this pattern does not match. Any other kind can, by being
            # asked a question it has no answer to.
            raise EnvelopeError(
                f"port {self.number}: a {self.kind} port has no waveguide mode "
                f"({self.mode!r}), so it has no (a, b) to bind to axes"
            )
        broad, narrow = match.group(1), match.group(2)
        if extents[0] >= extents[1]:
            return *extents, f"TE{broad}{narrow}"
        return *extents, f"TE{narrow}{broad}"

    @property
    def excite_sign(self) -> int:
        """openEMS' excitation sign.

        ``MSLPort`` integrates the voltage from ``start`` to ``stop`` along the
        excitation axis. When that runs *downward* - trace to ground, the
        normal microstrip case - the excitation must be negated, which is what
        openEMS' own MSL tutorial does.

        A waveguide port takes no sign: its direction comes from the ordering of
        ``start`` and ``stop`` along the propagation axis, so both ports of a
        through-line point inward and are excited with a plain 1.
        """
        if not self.excite:
            return 0
        if self.excitation_axis is None:
            return 1
        axis = self.excitation_axis
        return 1 if self.stop[axis] >= self.start[axis] else -1

    def measurement_position(self) -> float:
        """Absolute coordinate of the measurement plane on the propagation axis.

        A waveguide port does not shift: its probes sit on the port box's far
        face (``ports.py``: ``m_start[exc_ny] = m_stop[exc_ny]``), so ``stop``
        *is* the measurement plane.
        """
        axis = self.propagation_axis
        if self.kind == "rect_waveguide":
            return self.stop[axis]
        return self.start[axis] + self.direction * self.measurement_shift

    def moved(self, offset: tuple[float, float, float]) -> Port:
        """The same port, translated. The shifts along the axes are lengths and
        stay as they are; the corners are places and move."""
        return replace(self, start=_shifted(self.start, offset), stop=_shifted(self.stop, offset))

    def lays_conductor(self) -> bool:
        """Whether this port adds metal the mesh has to resolve."""
        return self.kind in _LAYS_CONDUCTOR

    def snaps_to_the_grid(self) -> bool:
        """Whether openEMS moves this port's planes onto the nearest grid line."""
        return self.kind in _SNAPS_TO_THE_GRID

    def required_lines(self) -> tuple[list[float], list[float], list[float]]:
        """Positions, per axis, where this port needs a grid line to exist.

        The kinds that ask all ask for one reason: **an excitation is not
        snapped.** openEMS discretises one by walking the grid and asking the
        geometry what is at each coordinate, so a box that contains no
        coordinate excites nothing - and says so only as ``Unused primitive``,
        one line among thousands, after which the run completes having driven
        nothing and every S-parameter comes back 0/0.

        A ``rect_waveguide`` port puts its excitation on a zero-thickness plane
        at ``start`` and its probes on one at ``stop``. openEMS' own tutorial
        adds those lines by hand before meshing; here the port asks, so no
        caller can forget.

        A **lumped** port that excites asks for a line on every axis it is flat
        across. Flat is the ordinary shape - the canonical lumped port is fed
        from a trace's end face, which has no extent along the line - and the
        resistor and the probes both survive it, because openEMS snaps those.
        The excitation is the one primitive that does not, so a plane lying
        between two lines drives nothing at all.

        An axis the box has *extent* on asks for nothing here, a line inside it
        being the mesher's to place or not rather than a position anything can
        name. Whether one landed there is a question about the grid, and
        :mod:`.preflight` is where the grid exists to be asked.

        ``MSLPort`` is not a cliff - it snaps both planes to the nearest
        existing line rather than discretising nothing - so it asks through
        :meth:`wanted_lines` instead.
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

        ``MSLPort`` snaps: the source goes to the grid line nearest
        ``start + feed_shift`` and the probe triplet to the three nearest the
        measurement plane. Nothing pins either, so both land wherever the
        grading happens to have put a line - measured on the acceptance line,
        the source asked for x = -30 and got -30.167, a sixth of a millimetre
        away, and a probe asked for x = 0 and got -0.355, half a cell. The
        engineer set those distances deliberately and the grid quietly rounded
        them.

        Wanted rather than *required*: unlike a waveguide port's two faces, a
        missing line here costs a fraction of a cell of reference-plane offset,
        not a run of 0/0. So a position outside the meshed domain is dropped
        here and reported by :mod:`.preflight`, which says *why* it is outside
        - a source in the absorber is driving a field that is being
        attenuated on purpose, and that is worth more than "no line here".
        """
        lines: tuple[list[float], list[float], list[float]] = ([], [], [])
        if self.kind in _USES_PROBE_TRIPLET:
            axis = self.propagation_axis
            lines[axis].append(self.start[axis] + self.direction * self.feed_shift)
            lines[axis].append(self.measurement_position())
        return lines

    def trace_region(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """The conductor this port lays down, as a sorted box.

        ``MSLPort`` builds the strip itself, flattened onto ``start``'s
        excitation coordinate. The mesher has to know about it - it is the
        finest metal in the model - so the adapter reconstructs it here rather
        than discovering it after the fact.
        """
        axis = self.excitation_axis
        stop = list(self.stop)
        stop[axis] = self.start[axis]
        lower = tuple(min(a, b) for a, b in zip(self.start, stop))
        upper = tuple(max(a, b) for a, b in zip(self.start, stop))
        return lower, upper  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
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

    :param max_timesteps: Step count. With ``end_criteria`` at zero this is not
        a ceiling but the **run length** - every step is taken, and runtime is
        linear in it. That is worth saying because upstream's 30000
        (``Tutorials/Simple_Patch_Antenna.py:55``, ``Conical_Horn_Antenna.m:44``)
        is a safety net behind ``EndCriteria=1e-4``, and this project cannot use
        that net. :data:`DEFAULT_TIMESTEPS` says what the number rests on.
    :param end_criteria: Stop early once the residual energy falls this far
        below its peak. **Zero disables it, and zero is the only reproducible
        setting.** openEMS re-evaluates this criterion inside a branch gated on
        four seconds of wall clock, so an energy-terminated run stops at a
        machine-load-dependent timestep, which truncates the recorded time
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

    Computed on the FreeCAD side and carried here verbatim: the driver never
    meshes. That is what makes a mesh *preview* trustworthy - what the user
    saw is the array that was solved, not a prediction of it.

    :param params: The policy the lines came from. Provenance only; re-running
        it is not the driver's job and would defeat the point.
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
        """The same grid, translated. Every spacing in it is a difference and
        so is untouched, which is what makes this a change of coordinates
        rather than a different mesh."""
        return replace(
            self,
            x=self.x + offset[0],
            y=self.y + offset[1],
            z=self.z + offset[2],
        )

    @property
    def cell_count(self) -> int:
        """Lines, not intervals. See ``mesh.MeshLines.cell_count``."""
        return int(np.prod([len(self[d]) for d in range(3)]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "z": self.z.tolist(),
            "params": self.params,
        }

    def digest(self) -> str:
        """Content hash of the grid alone. What a mesh preview goes stale against.

        Deliberately not the envelope's digest. Raising ``MaxTimesteps`` moves
        the envelope and changes nothing about the mesh, and a preview that
        cried stale for that would train people to ignore it. An envelope digest
        also needs an excitation chosen, which a preview does not have and does
        not need - every run in a sweep shares one grid.

        Canonicalised the same way the envelope is, so a grid that survived a
        save and reload hashes to what it hashed before.
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
    """The waveguide mode name, or raise. Stated once, asked twice.

    :class:`Port` asks on the way in; the document layer asks *before* building
    a port, so a mode typed into a property field is refused as the model's
    problem rather than reaching the panel's catch-all as a traceback. Same rule
    either way - a second copy in the translator would be a second rule.
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
    """``value``, if openEMS would act on it. Raises otherwise.

    Stated here rather than at each of its two call sites - the envelope's own
    invariant, and the document layer reading the property a user typed - so
    that the bound and its reason exist once.

    Refused rather than clamped, and the reason is that openEMS' answer to a
    factor outside ``(0, 1]`` is not an error. Against v0.0.36-157-gc4ce357:

    * at 0 or below, ``Operator::SetTimestepFactor`` prints "invalid timestep
      factor, skipping!" and the run steps at full size;
    * **above 1, nothing is printed at all** - ``openEMS::SetupFDTD`` guards
      the call with ``if (m_TS_fac<1)``, so the routine that would complain is
      never reached - and the run steps at full size while
      ``openEMS::Write2XML`` writes ``TimeStepFactor="2"`` into the XML, which
      is the file a bug report carries.

    Either way a user who reduced the step to stop a run diverging gets the
    diverging run back, with the workbench none the wiser. That is the silent
    no-op section 4.2 forbids, so the value never leaves here.
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
    exciting a different port - the loop belongs to the adapter, not to the
    envelope, so that a single run stays reproducible on its own.

    :param timestep_factor: Scales the timestep openEMS calculates for itself,
        in ``(0, 1]`` - see :func:`check_timestep_factor`. One means "use the
        engine's own step", which is what openEMS does anyway, applying the
        factor only when it is below one. Below one it buys stability on a grid
        the CFL bound alone does not settle, and costs simulated time: the same
        ``max_timesteps`` then covers proportionally less of it. Pre-flight says
        so.

        Adding it is what took ``SCHEMA_VERSION`` to 2. Leaving the version
        alone looked safe - absent means one, which is what those runs did -
        but the key is absent only in the *file*: :meth:`to_dict` emits it
        either way, and :meth:`digest` runs over the re-serialised form, so an
        untouched version 1 envelope digests to something its own
        ``envelope.sha256`` never said - and :meth:`Results.matches` then
        denies that a results file came from the envelope beside it.

    :param smallest_response: The smallest magnitude in S this study reads, in
        ``(0, 1]``, of which one is full scale. Nothing in the run is solved
        differently for it: it is what ``residual.unfinished`` weighs the
        leakage of a truncated record against, since a leak that is negligible
        beside a response of one is the whole of a stopband. Declared and never
        inferred - a sweep cannot tell a term that is the point of the exercise
        from one that is a rounding error.
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
    title: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "materials", tuple(self.materials))
        object.__setattr__(self, "solids", tuple(self.solids))
        object.__setattr__(self, "ports", tuple(self.ports))
        object.__setattr__(self, "boundary", tuple(self.boundary))

        check_timestep_factor(self.timestep_factor)
        # Neither was checked. length_unit=0 makes grid.SetDeltaUnit(0) and
        # divides by zero in preflight; a negative thread count goes straight to
        # numThreads. Both are scale factors on everything else in the file.
        _finite(self.length_unit, "length_unit", low=0.0, strict=True)
        _finite(self.threads, "threads", low=0.0)
        _finite(self.smallest_response, "smallest_response", low=0.0, high=1.0, strict=True)
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

        The offset comes back because the structure the engine is handed is
        then not the one the user drew, and the XML beside it is in these
        coordinates: a reader of that file has to be told, and anything
        read back off the engine by *position* would have to subtract it.
        Nothing this adapter reads back is a position - an S-matrix is not - so
        it is provenance rather than a correction to apply.

        Applied where the envelope is turned into a structure and nowhere
        earlier, so what the user is shown - the mesh preview, a pre-flight
        message, the envelope on disk - stays in the coordinates they drew in.
        See :func:`origin_offset` for why the engine is given no other choice.
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
            title=data.get("title", ""),
        )

    def to_json(self) -> str:
        """The envelope as the driver receives it, canonicalised.

        Rounding happens here rather than at the digest, so the bytes hashed are
        the bytes written: ``sha256sum openems.json`` has to agree with
        :meth:`digest`, or the provenance can only be checked by code that
        reimplements the rounding. See :data:`CANONICAL_DIGITS`.

        ``allow_nan=False`` because Python's default writes the bare tokens
        ``NaN`` and ``Infinity``, which RFC 8259 does not permit - so a file
        holding one could not be read by ``jq``, a browser, or any non-Python
        parser, and the envelope's second job is being attachable to a bug
        report. The fields are validated on the way in; this is the backstop
        that turns a leak into an exception rather than a corrupt artefact.
        """
        return json.dumps(canonical(self.to_dict()), indent=2, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> Problem:
        return cls.from_dict(json.loads(text))

    def digest(self) -> str:
        """Content hash of the envelope, for provenance.

        Over the serialised form, so it changes if and only if what the driver
        actually receives changes. A result whose digest does not match its
        envelope was produced by a different input.
        """
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
