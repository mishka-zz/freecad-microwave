# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The solver side of the process boundary. Runs under openEMS' interpreter.

This module may import openEMS and CSXCAD. It must never import FreeCAD. It
runs in a different Python, launched by :mod:`.run`, because FreeCAD's
interpreter does not have the solver bindings.

Invoked as a module::

    python -m Microwave.Solvers.openems.driver <envelope.json> [--build-only]

It reports progress by writing line-oriented markers to stdout, which
:mod:`.run` parses. The markers are a contract. Keep them stable, one per line,
prefixed ``OPENEMS:``.

===========================  ==========================================
Marker                       Meaning
===========================  ==========================================
``STARTED``                  process is alive, envelope not yet read
``ENVELOPE digest=…``        envelope parsed and hashed
``CHECK severity=… …``       a pre-flight finding, or one about the solve
``PLACED offset=…``          where the structure was put, against the drawing
``GRID cells=… lines=…``     grid installed
``GROWN solids=… most=…``    conductors grown for where openEMS samples them
``BUILT``                    geometry and ports constructed
``SOLVER_STARTED``           handed off to openEMS
``SOLVER_FINISHED``          time stepping done
``RESULTS <path>``           results written
``DONE``                     clean exit
``ERROR kind=… message=…``   giving up
===========================  ==========================================

Exit codes: ``0`` success, ``1`` the envelope is bad or pre-flight refused it,
``2`` anything else.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ... import units
from . import balance, excitation, nearfield, portreading, residual
from .capabilities import ADAPTER_VERSION
from .model import (
    AXIS_NAMES,
    LOSSY_KINDS,
    MEDIUM_PRIORITY,
    THROUGH,
    EnvelopeError,
    Material,
    Port,
    Problem,
    Solid,
)
from .preflight.absorber import _absorber_cells
from .preflight.finding import WARN
from .write import read_envelope

MARKER = "OPENEMS:"

RESULTS_NAME = "results.json"
STRUCTURE_NAME = "structure.xml"
RUN_DIRNAME = "run"

#: What the probes a waveguide port is read at again inside its box are named
#: with, ahead of the name openEMS gives a port's own: at the deeper plane
#: :func:`.nearfield.inside` places, and at the shallower.
INSIDE_PREFIXES = ("inside_", "halfway_")

#: A waveguide port's number, to what reads it again at each plane inside its
#: box, the deeper first: the port as it is read there, the cells in from its
#: plane, and the openEMS port.
Inside = dict[int, list[tuple[Port, int, Any]]]

#: Each body's label, the property it is laid in and the primitives it was laid
#: as: one for a box or a surface, one for each triangle of a sheet.
Bodies = list[tuple[str, Any, list[Any]]]

#: Where the run's provenance says which impedance each waveguide port states.
#: A guide has several, and a matrix referenced to its own mode is the same
#: whichever is stated, but renormalising it is not, so a file that carries
#: the number has to say which it is.
IMPEDANCE_STATED = "impedance_stated"

#: Where the run's provenance keeps, per waveguide port, how far its waves were
#: moved from where openEMS read them to its reference plane, in length units,
#: positive towards the face the mode is launched on.
REFERRED = "referred"

#: What openEMS states for a ``RectWGPort``: ``self.ZL = k * Z0 / self.beta``,
#: with the cutoff of the port's box as drawn.
WAVE_IMPEDANCE = "the wave impedance of its mode in the guide as drawn, k Z0 / beta"
#: What a port read again by :mod:`.portreading` states: the same as the grid
#: propagates the mode in the guide as built, over the share of a wave's power
#: the grid carries where the waves are split.
READ_IMPEDANCE = (
    "the wave impedance of its mode as the grid propagates it in the guide as "
    "built, k Z0 / beta at the grid's own wavenumber and cutoff, over the share "
    "of a wave's power the grid carries where the waves are split"
)


def marker(name: str, detail: str = "") -> None:
    print(f"{MARKER}{name}{' ' + detail if detail else ''}", flush=True)


def _sanitise(text: object) -> str:
    """Fold a message onto one line. Markers are line-oriented, so a message
    may not contain newlines.

    It takes anything. Callers hand it a finding, an exception or a string, and
    each wants the same single line out.
    """
    return " ".join(str(text).split())


def everything_wrong(problem: Problem) -> list:
    """The checks a run makes, in the order their findings are reported.

    They ask different questions. Pre-flight asks whether this adapter can
    do what the envelope describes, and is cheap enough for the task panel to
    run on every keystroke. The other two ask whether the grid in the envelope
    still holds the conductors it was built for, and whether openEMS reads each
    dielectric anywhere. Those are questions about the finished grid and cost a
    triangle for every point they sample, which is affordable beside a solve and
    not beside a keystroke, so they run here and nowhere else.
    """
    from . import conductors, dielectrics, preflight

    return preflight.check(problem) + conductors.check(problem) + dielectrics.check(problem)


def build(problem: Problem, sim_dir: Path) -> tuple[Any, Any, dict[int, Any], Inside, Bodies]:
    """Construct the CSX structure and the FDTD object. No solving.

    Each rectangular waveguide port is also read again inside its own box, by
    probes :mod:`.nearfield` places, so the ports come back twice: each by its
    number, and what reads a waveguide port again at each plane by the same
    number.

    The order of the calls matters. ``MSLPort`` reads the grid at construction
    time to place its probes, and raises unless its propagation axis carries
    more than five lines (``openEMS/python/openEMS/ports.py:289``), so the grid
    is installed before any port is added.
    """
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS

    # Before anything is placed, so that every coordinate below is in one
    # system and none of them is negative. The offset is reported rather than
    # hidden. The XML written at the end of this function is in these
    # coordinates, and nothing else the user sees is.
    problem, offset = problem.at_the_origin()
    marker("PLACED", "offset=" + ",".join(f"{d:.6g}" for d in offset))

    csx = ContinuousStructure()
    grid = csx.GetGrid()
    grid.SetDeltaUnit(problem.length_unit)
    for dim, axis in enumerate(AXIS_NAMES):
        grid.AddLine(axis, problem.grid[dim].tolist())
    # The marker carries the line shape beside the cell count. openEMS prints
    # its own ``FDTD simulation size: <nx>x<ny>x<nz> --> <n> FDTD cells`` a few
    # lines later, and the two have to be readable as one grid.
    shape = "x".join(str(len(problem.grid[dim])) for dim in range(3))
    marker("GRID", f"cells={problem.grid.cell_count} lines={shape}")

    # ``TimeStepFactor`` is passed unconditionally, including at 1.0. openEMS
    # applies it only when it is below one (``openEMS::SetupFDTD``), so one is
    # the engine's own step by the engine's own arithmetic rather than by this
    # adapter skipping the call.
    #
    # It is also the one setting here that leaves no usable trace in any file
    # openEMS writes. ``openEMS::Write2XML`` records the attribute only when the
    # factor is above one, which is the range that is never applied, so the XML
    # says nothing for every value that did something and states the one value
    # that did not. The provenance below is the only record.
    fdtd = openEMS(
        NrTS=problem.termination.max_timesteps,
        EndCriteria=problem.termination.end_criteria,
        TimeStepFactor=problem.timestep_factor,
    )
    # openEMS' own Gaussian pulse, with its carrier turned a quarter period so
    # that it carries nothing at zero frequency. See ``excitation``. The band's
    # top goes in twice because the two arguments do not do what their names
    # suggest: ``CalcCustomExcitation`` overwrites the maximum frequency with
    # the second one and takes the probes' sampling rate from it, while the
    # third reaches only the conducting-sheet model, which is built before the
    # signal is.
    fdtd.SetCustomExcite(
        excitation.expression(problem.frequency.center, problem.frequency.half_bandwidth),
        problem.frequency.stop,
        problem.frequency.stop,
    )
    fdtd.SetBoundaryCond(list(problem.boundary))
    fdtd.SetCSX(csx)

    properties = {
        material.name: _add_material(csx, material, problem.length_unit)
        for material in problem.materials
    }

    grew = []
    bodies: Bodies = []
    for solid in problem.solids:
        prop = properties[solid.material]
        # A property appends each primitive it is given, so the body's are the
        # ones past the count it held before.
        before = prop.GetQtyPrimitives()
        moved = add_solid(prop, solid, problem.as_given(solid))
        laid = [prop.GetPrimitive(index) for index in range(before, prop.GetQtyPrimitives())]
        bodies.append((solid.label or solid.material, prop, laid))
        if moved is not None:
            grew.append(moved)
    if grew:
        marker("GROWN", f"solids={len(grew)} most={max(grew):.6g}")
    if problem.medium:
        _add_medium(properties[problem.medium], problem)

    ports = {}
    inside: Inside = {}
    for port in problem.ports:
        metal = properties[port.metal] if port.metal else None
        ports[port.number] = _add_port(
            fdtd, csx, metal, port, problem.length_unit, problem.grown_by
        )
        if port.kind == "rect_waveguide":
            again = _add_inside(fdtd, problem, port)
            if again:
                inside[port.number] = again

    csx.Write2XML(str(sim_dir / STRUCTURE_NAME))
    marker("BUILT")
    return fdtd, csx, ports, inside, bodies


class SetupFailed(Exception):
    """openEMS refused to set the run up, and solved nothing."""


#: How far one bounding box may stand past another and still be held by it, as
#: a share of the larger's span: CSXCAD keeps a surface's vertices in single
#: precision (``CSPrimPolyhedron.h``), and a box in double.
HELD_WITHIN = 1e-6


def _box_holds(outer: Any, inner: Any) -> bool:
    """Whether one bounding box, as CSXCAD states it, holds another."""
    low, high = np.asarray(outer, dtype=float)
    slack = HELD_WITHIN * max(float(np.max(high - low)), 1.0)
    return bool(np.all(low - slack <= inner[0]) and np.all(inner[1] <= high + slack))


def _unlaid(bodies: Bodies, csx: Any) -> list[str]:
    """Each body none of whose primitives openEMS gave a cell, and what spans it.

    CSXCAD gives each sample of the grid to the first primitive holding it in
    order of priority and marks that one used
    (``ContinuousStructure::GetPropertyByCoordPriority``), and openEMS reports
    one never marked by its property's name alone. A body none of whose
    primitives was marked lost every sample to another primitive - a
    microstrip port's own strip laid over the trace, a second copy of a sheet,
    a body of an alike metal in the same place - or the grid sampled none of
    it, and then it is not in the model solved. Which is not known here, so each
    is named with whatever laid primitive's bounding box spans it. A sheet some
    of whose triangles were marked is laid, the others holding no sample.
    """
    owner = {primitive.GetID(): label for label, _, laid in bodies for primitive in laid}
    every = csx.GetAllProperties()
    said = []
    for label, prop, laid in bodies:
        if not laid or any(piece.GetPrimitiveUsed() for piece in laid):
            continue
        corners = [np.asarray(piece.GetBoundBox(), dtype=float) for piece in laid]
        extent = (np.min([c[0] for c in corners], axis=0), np.max([c[1] for c in corners], axis=0))
        # The sample goes to the highest priority holding it, so of the laid
        # primitives spanning the body that is the one to name.
        spans = max(
            (
                (other, holder)
                for holder in every
                for other in holder.GetAllPrimitives()
                if other.GetPrimitiveUsed() and _box_holds(other.GetBoundBox(), extent)
            ),
            key=lambda found: found[0].GetPriority(),
            default=None,
        )
        if spans is None:
            said.append(
                f"{label}: openEMS gave this {prop.GetName()} body no cell, and no primitive "
                "it laid spans it, so unless bodies around it took its cells the model solved "
                "is without it - a sheet off every grid line or a surface enclosing no sample "
                "is lost this way. Draw it where the mesh lays a line, or refine across it"
            )
            continue
        other, holder = spans
        by = owner.get(other.GetID(), f"a {other.GetTypeName().lower()} a port lays")
        said.append(
            f"{label}: openEMS gave this {prop.GetName()} body no cell, and {by}, of "
            f"{holder.GetName()}, spans it and was laid, so where it took these cells they "
            f"are {holder.GetName()}"
        )
    return said


def _add_medium(prop: Any, problem: Problem) -> None:
    """The medium as one box below every solid, reaching past the grid.

    It reaches a whole span of the grid past each end, so no cell at the grid's
    edge, the absorber's included, is left to the vacuum openEMS fills an
    uncovered cell with. Beyond an ``Air`` face that absorber stands in the
    medium, and beyond a ``Through`` face the structure runs on through it.

    A face the domain ends on and the absorber stands beyond is one waveguide
    ports cover, and the absorber there carries on the guide behind the ports'
    plane. openEMS reads a guide as empty, so that absorber is left to vacuum and
    the box stops at the domain's face.
    """
    padding = list(problem.grid.params.get("padding") or ())
    lower, upper = [], []
    for dim in range(len(AXIS_NAMES)):
        lines = problem.grid[dim]
        span = float(lines[-1] - lines[0])
        low, high = float(lines[0]) - span, float(lines[-1]) + span
        cells = _absorber_cells(problem.grid, dim)
        stated = padding[dim] if dim < len(padding) else ()
        for side, face in enumerate(stated):
            if str(face) == THROUGH or float(face) > 0.0 or not cells[side]:
                continue
            if side == 0:
                low = float(lines[cells[0]])
            else:
                high = float(lines[-1 - cells[1]])
        lower.append(low)
        upper.append(high)
    prop.AddBox(lower, upper, priority=MEDIUM_PRIORITY)


def _add_inside(fdtd: Any, problem: Problem, port: Port) -> list[tuple[Port, int, Any]]:
    """Probes reading ``port``'s mode again at the planes
    :func:`.nearfield.inside` places in its box, the deeper first; none where
    the box has no room for one.

    Each is a waveguide port with no excitation, which adds its two probes and
    nothing else, named apart from the port's own.
    """
    axis = port.propagation_axis
    where = nearfield.inside(problem.grid[axis], port.start[axis], port.stop[axis])
    if where is None:
        return []
    added = []
    for (plane, cells), prefix in zip(
        (at for at in where if at is not None), INSIDE_PREFIXES, strict=False
    ):
        stop = list(port.stop)
        stop[axis] = plane
        again = replace(port, stop=(stop[0], stop[1], stop[2]), excite=False)
        a, b, mode = again.waveguide_arguments(problem.length_unit)
        probes = fdtd.AddRectWaveGuidePort(
            port.number,
            list(again.start),
            list(again.stop),
            axis,
            a,
            b,
            mode,
            excite=0,
            PortNamePrefix=prefix,
        )
        added.append((again, cells, probes))
    return added


def add_solid(
    prop: Any, solid: Solid, given: Sequence[Sequence[float]] | None = None
) -> float | None:
    """One envelope solid as whichever primitive holds it.

    The forms a solid arrives in and the primitive each becomes are stated in
    one place, so anything wanting to know what openEMS will hold asks this
    function rather than reproducing the choice.

    :param given: The triangulation to hand over, from
        :meth:`~.model.Problem.as_given`, or ``None`` for the one that was
        drawn. Which solids get one, and what it corrects for, is that method's
        business: the surface openEMS is given depends on the whole envelope,
        which is more than the driver asks about.
    :returns: How far the surface moved to answer for it, or ``None`` where
        nothing moved.

    A sheet is never corrected, and the claim is narrow. The mesher pins a
    line at the plane a sheet lies in, so nothing rounds it across its
    thickness. Its outline is sampled like any other conductor boundary and is
    cut back like one. What that costs has not been measured on anything that
    solves Maxwell, and correcting it is a different operation: a sheet arrives
    as a triangulation of its area, and its interior points have no outward
    direction in the plane to be moved along.
    """
    if solid.is_sheet:
        _add_sheet(prop, solid)
        return None
    if solid.is_mesh:
        vertices = solid.vertices if given is None else given
        _add_polyhedron(prop, solid, vertices)
        moved = max(math.dist(was, now) for was, now in zip(solid.vertices, vertices))
        # How far it went, and nothing where it did not move. A share of zero
        # returns the surface it was given rather than the same object, so the
        # question to ask is whether the points moved. Otherwise ``GROWN``
        # reports a structure that is the size it was drawn.
        return moved if moved > 0.0 else None
    prop.AddBox(list(solid.lower), list(solid.upper), priority=solid.priority)
    return None


def _add_sheet(prop: Any, solid: Solid) -> None:
    """One flat outline as coplanar polygons, a triangle each.

    CSXCAD's polygon takes a single closed contour, so it cannot state a hole,
    and a letter with a counter, a clearance in a plane, and an annulus are all
    holes. The sheet's own triangles can state one: they are what the outline
    encloses, with the holes already left out, and each is convex and
    unambiguous.

    A polygon is the right primitive rather than a flattened solid. A sheet has
    no volume to bound, so a closed surface cannot describe it, and openEMS
    models a zero-thickness conductor by finding it at one plane. ``elevation``
    here states that plane, and the mesher pins a line there.
    """
    axis = solid.sheet_normal
    assert axis is not None  # only a sheet reaches here, and a sheet has a normal
    # CSXCAD reads the first coordinate list as axis (n+1)%3 and the second as
    # (n+2)%3, which is a cycle rather than the remaining axes in order. The two
    # agree for a sheet normal to x or z and swap for one normal to y, so a
    # vertical sheet laid the obvious way arrives mirrored about x = z, with no
    # message: the primitive is used either way.
    across = [(axis + 1) % 3, (axis + 2) % 3]
    for first, second, third in solid.faces:
        corners = [solid.vertices[index] for index in (first, second, third)]
        prop.AddPolygon(
            [[corner[across[0]] for corner in corners], [corner[across[1]] for corner in corners]],
            norm_dir=axis,
            elevation=solid.lower[axis],
            priority=solid.priority,
        )


def _add_polyhedron(prop: Any, solid: Solid, vertices: Sequence[Sequence[float]]) -> None:
    """One triangulated solid as a CSXCAD polyhedron.

    The vertices are passed rather than read off the solid, because what
    reaches the engine may have been grown to answer for where openEMS puts a
    conductor's surface. See :mod:`staircase`. The faces are the same either
    way: growing a surface moves its points and not how they are joined.

    The primitive takes no geometry when it is created. Vertices and faces are
    added to it one call at a time. That is cheap enough not to need the
    file-reading variant, because it is a Python loop over an array the mesher
    already holds, and the solve dominates it.

    Nothing is read back. CSXCAD's per-face verdict is written by CGAL's
    builder, and the builder stops at the first construction error, leaving
    every later face's flag at whatever the memory held. On the path a check
    would be for, the flags are therefore indeterminate. What the surface is has
    already been settled combinatorially on the way into the envelope, where the
    answer is deterministic and where a refusal can still name the object.
    """
    primitive = prop.AddPolyhedron(priority=solid.priority)
    for point in vertices:
        primitive.AddVertex(*point)
    for face in solid.faces:
        primitive.AddFace(list(face))


def _add_material(csx: Any, material: Material, length_unit: float) -> Any:
    """One envelope material to one CSXCAD property.

    ``length_unit`` is here for one field. Every length in the CSX tree is in
    grid units except a conducting sheet's thickness, which CSXCAD takes in
    metres, alongside conductivity in S/m. Getting it wrong does not fail:
    openEMS finds the resulting surface-impedance fit out of range, prints to
    stderr, clamps to its last tabulated coefficients and carries on with a
    wrong answer.
    """
    if material.kind == "pec":
        return csx.AddMetal(material.name)
    if material.kind == "conducting_sheet":
        return csx.AddConductingSheet(
            material.name,
            conductivity=material.conductivity,
            thickness=material.thickness * length_unit,
        )
    if material.kind == "lossy_dielectric":
        # kappa is openEMS' electric conductivity
        # (``CSXCAD/python/CSXCAD/CSProperties.pyx:542``), so the one a loss
        # tangent became and the one the material states add.
        return csx.AddMaterial(
            material.name,
            epsilon=material.epsilon,
            mue=material.mu,
            kappa=material.kappa + material.conductivity,
        )
    if material.kind == "dielectric":
        return csx.AddMaterial(material.name, epsilon=material.epsilon, mue=material.mu)

    raise EnvelopeError(f"{material.name}: no builder for material kind {material.kind!r}")


def _add_port(
    fdtd: Any, csx: Any, metal_prop: Any, port: Port, length_unit: float, grown_by: float
) -> Any:
    """One envelope port to one openEMS port object.

    An upstream port class is used rather than reimplemented wherever one
    exists. The impedance an ``MSLPort`` extracts, ``sqrt(Et*dEt / (Ht*dHt))``,
    is the definition this adapter reports. The coaxial kind is the exception,
    and :mod:`.coaxial` says why: the released bindings carry no such class,
    and the one on openEMS' master branch lays the line it measures, which the
    drawing here already supplies.

    A ``RectWGPort`` differs in kind. It computes its reference impedance
    analytically from the mode
    and the guide dimensions rather than measuring it (``ports.py``:
    ``self.ZL = k * Z0 / self.beta``), so comparing its ``Z_ref`` against the
    closed form would be circular. A waveguide port measures the phase of the
    wave that crossed it.
    """
    if port.kind == "microstrip":
        keywords: dict[str, Any] = {
            "FeedShift": port.feed_shift,
            "MeasPlaneShift": port.measurement_shift,
            "priority": port.priority,
        }
        if port.feed_resistance is not None:
            keywords["Feed_R"] = port.feed_resistance
        return fdtd.AddMSLPort(
            port.number,
            metal_prop,
            list(port.start),
            list(port.stop),
            port.propagation_axis,
            port.excitation_axis,
            excite=port.excite_sign,
            **keywords,
        )

    if port.kind == "rect_waveguide":
        # a and b are in metres, while start and stop are in grid units. Both
        # belong to an axis rather than to a wall, and the mode name is numbered
        # to match. The model derives all three, so they cannot disagree.
        a, b, mode = port.waveguide_arguments(length_unit)
        return fdtd.AddRectWaveGuidePort(
            port.number,
            list(port.start),
            list(port.stop),
            port.propagation_axis,
            a,
            b,
            mode,
            excite=port.excite_sign,
            priority=port.priority,
        )

    if port.kind == "coaxial":
        # Imported here for the reason ``build`` imports the engine here. This
        # module is reachable without openEMS installed, and only the paths that
        # construct a structure need it.
        from .coaxial import CoaxialPort

        # On the axis rather than on the corners. A round port reaches out to
        # its own radii from the line, and the box the envelope carries bounds
        # that reach rather than defining it.
        axis_start = list(port.bore_centre)
        axis_stop = list(port.bore_centre)
        axis_stop[port.propagation_axis] = port.stop[port.propagation_axis]
        return CoaxialPort(
            csx,
            port.number,
            axis_start,
            axis_stop,
            port.propagation_axis,
            port.inner_radius,
            port.outer_radius,
            excite=port.excite_sign,
            grown_by=grown_by,
            FeedShift=port.feed_shift,
            MeasPlaneShift=port.measurement_shift,
            priority=port.priority,
        )

    if port.kind == "lumped":
        return fdtd.AddLumpedPort(
            port.number,
            port.feed_resistance,
            list(port.start),
            list(port.stop),
            port.excitation_axis,
            excite=port.excite_sign,
            priority=port.priority,
        )

    raise EnvelopeError(f"{port.name}: no builder for port kind {port.kind!r}")


def solve(problem: Problem, directory: Path, build_only: bool = False) -> Path | None:
    """Build, run, extract. Returns the results path, or ``None`` if build-only."""
    sim_dir = directory / RUN_DIRNAME
    sim_dir.mkdir(parents=True, exist_ok=True)

    fdtd, csx, ports, inside, bodies = build(problem, directory)
    if build_only:
        return None

    started = time.monotonic()
    marker("SOLVER_STARTED", f"threads={problem.threads} dir={sim_dir}")
    failed = fdtd.Run(str(sim_dir), cleanup=True, numThreads=problem.threads)
    if failed:
        # ``openEMS.Run`` returns the code ``openEMS::SetupFDTD`` refused with,
        # and solves nothing.
        raise SetupFailed(f"openEMS could not set the run up, and returned code {failed}")
    elapsed = time.monotonic() - started
    marker("SOLVER_FINISHED", f"seconds={elapsed:.1f}")
    for unlaid in _unlaid(bodies, csx):
        marker("CHECK", f"severity={WARN} {_sanitise(unlaid)}")

    results = _extract(problem, ports, inside, sim_dir, elapsed, csx)

    # Reported as well as recorded, and here rather than in pre-flight. None of
    # these can be known before the solve, and by the time one exists the
    # minutes are spent, so each is a warning and never a refusal. The numbers
    # are in the provenance either way, and each warning is made from there, so
    # anything reading the results says what the run said.
    provenance = results["provenance"]
    ringing = residual.unfinished(provenance["tail_share"], problem.smallest_response)
    for warning in (
        ringing,
        portreading.unread(provenance),
        nearfield.said(provenance, finished=ringing is None),
        balance.said(provenance),
    ):
        if warning:
            marker("CHECK", f"severity={WARN} {_sanitise(warning)}")

    path = directory / RESULTS_NAME
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


def _complex_pair(values: npt.ArrayLike, points: int) -> dict[str, list[float]]:
    """Always a list of ``points`` floats.

    A lumped port's reference impedance is a scalar rather than a per-frequency
    array. Written out as it stands it would produce a bare number where every
    reader expects a sequence. It is broadcast here rather than special-cased on
    the way back in.
    """
    array = np.broadcast_to(np.asarray(values, dtype=complex), (points,))
    return {"re": array.real.tolist(), "im": array.imag.tolist()}


def _power(wave: npt.ArrayLike, reference: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """The power a modal voltage carries, as ``Port.CalcPort`` states it:
    half the real part of the voltage times the conjugate of its current."""
    voltage = np.asarray(wave, dtype=complex)
    return np.asarray(0.5 * np.real(voltage * np.conj(voltage / np.asarray(reference))))


def _below_cutoff(
    port: Any, frequency: np.ndarray
) -> tuple[npt.NDArray[np.complex128], ...] | None:
    """openEMS' own split of a ``RectWGPort``, stated below its mode's cutoff
    as well, or ``None`` where openEMS states it at every point.

    ``WaveguidePort.CalcPort`` in ``openEMS/python/openEMS/ports.py`` takes
    ``beta`` as the real square root of ``k^2 - kc^2``, which is ``nan`` below
    cutoff, and so is every wave it splits there. Below cutoff
    ``beta`` is imaginary, the impedance ``k Z0 / beta`` has no real part, and
    the port carries no power. This states that impedance where openEMS states
    none and splits the waves at it as openEMS does, ``uf_inc = (uf_tot +
    if_tot Z) / 2``, so the matrix names those points rather than refusing the
    run. Every other point is openEMS' own. The cutoff is the vacuum one, as
    openEMS' is: ``WaveguidePort`` fixes ``ref_index = 1``.

    ``k^2 - kc^2`` is openEMS' own expression over the port's own ``kc``, so a
    point is below cutoff here exactly where it is for openEMS. A point exactly
    on the cutoff has ``beta = 0`` for both, and its impedance has no bound.
    """
    points = np.asarray(frequency).size
    reference = np.broadcast_to(np.asarray(port.Z_ref, dtype=complex), (points,))
    missing = ~np.isfinite(reference)
    if not missing.any():
        return None
    wavenumber = 2.0 * np.pi * np.asarray(frequency, dtype=float) / units.SPEED_OF_LIGHT
    beta = np.sqrt((wavenumber**2 - float(port.kc) ** 2).astype(complex))
    total = np.broadcast_to(np.asarray(port.uf_tot, dtype=complex), (points,))
    current = np.broadcast_to(np.asarray(port.if_tot, dtype=complex), (points,))
    # A point on the cutoff divides by a beta of zero, and _check_the_impedances
    # refuses it.
    with np.errstate(divide="ignore", invalid="ignore"):
        reference = np.where(
            missing, wavenumber * portreading.FREE_SPACE_IMPEDANCE / beta, reference
        )
        split = 0.5 * (total + current * reference)
        incident = np.where(missing, split, np.asarray(port.uf_inc, dtype=complex))
        reflected = np.where(missing, total - split, np.asarray(port.uf_ref, dtype=complex))
        power_incident = np.where(missing, _power(incident, reference), np.real(port.P_inc))
        power_reflected = np.where(missing, _power(reflected, reference), np.real(port.P_ref))
    return reference, incident, reflected, power_incident, power_reflected


class NotANumber(Exception):
    """A number every S-parameter of a port depends on is not a number."""


def _check_the_records(ports: dict[int, Any]) -> None:
    """Refuse a run where a port's probe recorded a sample that is not a number.

    openEMS' transform, ``utilities.DFT_time2freq`` in
    ``openEMS/python/openEMS/utilities.py``, sums every sample at every
    frequency, and a port's waves and a microstrip port's impedance are made of
    what its probes transform to, so one such sample leaves the port nothing.
    Each probe is read as ``UI_data`` in ``openEMS/python/openEMS/ports.py``
    holds it.
    """
    for number, port in sorted(ports.items()):
        for quantity, data in (("voltage", port.u_data), ("current", port.i_data)):
            for name, times, values in zip(data.fns, data.ui_time, data.ui_val):
                bad = np.flatnonzero(~np.isfinite(np.asarray(values, dtype=float)))
                if bad.size:
                    raise NotANumber(
                        f"port {number}'s {quantity} probe {name} recorded a value that "
                        f"is not a number at {bad.size} of {np.size(values)} samples, the "
                        f"first at {float(np.asarray(times)[bad[0]]) * 1e9:.6g} ns. Every "
                        "frequency point of what a port reads is a sum over every sample "
                        "its probes recorded, so this run has no S-parameters to return"
                    )


def _check_the_impedances(
    per_port: dict[str, Any], models: dict[int, Port], frequency: np.ndarray
) -> None:
    """Refuse a run where a port states no finite reference impedance.

    Every wave at a port is normalised by its impedance, so no S-parameter at
    such a point is a number. A rectangular waveguide port states one below its
    mode's cutoff as well, and has none only on the cutoff, which one point of a
    band can lie on. A microstrip port measured in a run a lumped port drives has
    none at any point, and a port that has one at some points is a different
    case.
    """
    lumped = any(model.kind == "lumped" and model.excite for model in models.values())
    for number, model in sorted(models.items()):
        z0 = per_port[str(number)]["z0"]
        missing = ~np.isfinite(np.asarray(z0["re"]) + 1j * np.asarray(z0["im"]))
        if not missing.any():
            continue
        where = np.asarray(frequency, dtype=float)[missing] / 1e9
        at = (
            f"at {where[0]:.9g} GHz"
            if where.size == 1
            else f"at {where.size} frequency points from {where[0]:.6g} to {where[-1]:.6g} GHz"
        )
        said = (
            f"port {number} states no finite reference impedance {at}, so no "
            "S-parameter there can be normalised"
        )
        if model.kind == "rect_waveguide" and where.size == 1:
            which = (
                "the cutoff of the guide the grid holds"
                if "raw" in per_port[str(number)]
                else f"its {model.mode} cutoff"
            )
            said += (
                f". That point lies exactly on {which}, where beta is zero and the "
                "impedance has no bound. Move the band so no point falls on it"
            )
        elif model.kind == "microstrip" and lumped and missing.all():
            said += (
                ". A microstrip port measured in a run a lumped port drives states "
                "none at any point on this engine. Drive the microstrip port instead, "
                "or measure both ports the same way"
            )
        raise NotANumber(said)


def _read_waveguide(
    problem: Problem, model: Port, port: Any, csx: Any, sim_dir: Path, frequency: np.ndarray
) -> portreading.Reading:
    """How a ``RectWGPort`` read the waves passing it, in ``problem``'s coordinates.

    The node ranges come from the headers of its two probe files, and the metal
    from ``csx`` after the run, where the engine built the structure from the
    same primitives in this process. The timestep is twice the delay of the
    current's first sample after the voltage's, since the engine samples the
    magnetic field half a step after the electric; ``port`` has been
    calculated, so its records are loaded.
    """
    timestep = 2.0 * (float(port.i_data.ui_time[0][0]) - float(port.u_data.ui_time[0][0]))
    if not (math.isfinite(timestep) and timestep > 0):
        raise portreading.NotRead("its probes' first samples do not give the engine's timestep")
    ranges = []
    for name in (port.U_filenames[0], port.I_filenames[0]):
        try:
            with open(sim_dir / name, encoding="utf-8") as record:
                header = "".join(record.readline() for _ in range(4))
        except OSError as error:
            raise portreading.NotRead(f"its probe file {name} cannot be read: {error}") from error
        ranges.append(portreading.nodes(header))
    a, b, _ = model.waveguide_arguments(1.0)
    axis = model.propagation_axis
    return portreading.read(
        [problem.grid[dim] for dim in range(3)],
        axis,
        _orders(model),
        (a, b),
        (model.start, model.stop),
        ranges[0],
        ranges[1],
        1 if model.stop[axis] > model.start[axis] else -1,
        _conducts(csx, problem),
        frequency,
        problem.length_unit,
        timestep,
    )


def _cells(count: int) -> str:
    """A count of cells in words, as a message reads it."""
    return f"{count} cell" if count == 1 else f"{count} cells"


def _orders(model: Port) -> tuple[int, int]:
    """A waveguide port's ``(M, N)`` as openEMS is given them."""
    _, _, mode = model.waveguide_arguments(1.0)
    return int(mode[2]), int(mode[3])


def _conducts(csx: Any, problem: Problem) -> portreading.Conducts:
    """The engine's answer to whether the electric field at a point is tied.

    ``Operator::CalcPEC_Range`` asks the structure for the material or metal of
    highest priority at the point and ties the field where that is a plain
    metal, which is what is asked here. The engine ties the field lying in every
    face of the domain as well, and a face that absorbs or is a magnetic wall
    does more than that; only a PEC face is answered as tied, so a guide walled
    by anything else is not read.

    A point holding anything but metal or empty space stops the reading. The
    mode :mod:`.portreading` reads is the mode of an empty guide walled by tied
    edges, and a dielectric in the guide, or a wall of conducting sheet, which
    the engine models as a surface rather than by tying edges, makes a guide it
    does not describe.
    """
    from CSXCAD.CSProperties import PropertyType

    kinds = int(PropertyType.MATERIAL) | int(PropertyType.METAL)
    metal = int(PropertyType.METAL)
    ends = [(float(problem.grid[dim][0]), float(problem.grid[dim][-1])) for dim in range(3)]
    walls = [word == "PEC" for word in problem.boundary]
    materials = {material.name: material for material in problem.materials}

    def conducts(point: portreading.Point) -> bool:
        for dim, (low, high) in enumerate(ends):
            if point[dim] == low and walls[2 * dim]:
                return True
            if point[dim] == high and walls[2 * dim + 1]:
                return True
        found = csx.GetPropertyByCoordPriority(list(point), kinds)
        if found is None:
            return False
        if found.GetType() == metal:
            return True
        material = materials.get(found.GetName())
        if material is None or (material.kind, material.epsilon, material.mu) != (
            "dielectric",
            1.0,
            1.0,
        ):
            raise portreading.NotRead(
                f"the material {found.GetName()!r} lies where it reads its guide, and "
                "only an empty guide walled by metal is modelled"
            )
        return False

    return conducts


def _extract(
    problem: Problem, ports: dict, inside: Inside, sim_dir: Path, elapsed: float, csx: Any
) -> dict:
    """Probe files to numbers.

    ``CalcPort`` is called without a reference impedance on purpose. For an
    ``MSLPort`` a reference would overwrite the impedance it measured from the
    field, and that measured impedance is the quantity wanted. It is also what a
    truncated record corrupts, so each port's tail is weighed here, off the same
    probes and the same axes the transform was taken over.

    A rectangular waveguide port's waves are taken apart again from what it
    recorded, at the scales :mod:`.portreading` says it read them at, and
    openEMS' own split is kept beside them under ``raw``. Where one port's
    reading cannot be modelled, every waveguide port keeps openEMS' split and
    the run names the one. openEMS reads every port short of its wave, so its
    errors in a transmission partly cancel; undoing them at one port of the two
    leaves the other's whole.

    Where the run has a waveguide port, :func:`_near_field` weighs what each
    one read besides its mode, and :func:`_balance` how much of the power
    driven in the ports account for together.
    """
    frequency = problem.frequency.values()
    points = frequency.size
    excited = problem.excited_port
    # The engine is asked about positions, and it holds the structure where
    # build() put it.
    placed, _ = problem.at_the_origin()
    models = {port.number: port for port in placed.ports}

    per_port: dict[str, Any] = {}
    records: dict[str, Any] = {"recorded_samples": {}}
    stated: dict[str, str] = {}
    readings: dict[int, dict[str, Any]] = {}
    waves: dict[int, tuple[Any, Any]] = {}
    moved: dict[str, float] = {}
    # Each probe on the axis it was written on. A Yee scheme staggers the two
    # by half a step in time, and openEMS transforms each against its own
    # column. Both are taken, because the leakage a truncated record costs is an
    # error in the S-parameters below, and those are made of waves rather than
    # volts. A corrected waveguide port is weighed on openEMS' own split: the
    # tail is a share of the record, and the correction moves that share by a
    # share of its own.
    recorded = {}
    for port in ports.values():
        port.CalcPort(str(sim_dir), frequency)
    _check_the_records(ports)
    read_as: dict[int, portreading.Reading] = {}
    for number, port in sorted(ports.items()):
        if models[number].kind != "rect_waveguide":
            continue
        try:
            read_as[number] = _read_waveguide(placed, models[number], port, csx, sim_dir, frequency)
        except portreading.NotRead as why:
            readings[number] = {"corrected": False, "reason": str(why), "refused": True}
    if readings:
        refused = min(readings)
        for number in read_as:
            readings[number] = {
                "corrected": False,
                "reason": f"port {refused} of this run could not be read",
            }
        read_as = {}

    for number, port in sorted(ports.items()):
        records["recorded_samples"][number] = int(np.size(port.ut_tot))
        entry: dict[str, Any] = {
            "z0": _complex_pair(port.Z_ref, points),
            "incident": _complex_pair(port.uf_inc, points),
            "reflected": _complex_pair(port.uf_ref, points),
            "power_incident": np.broadcast_to(
                np.asarray(port.P_inc, dtype=float), (points,)
            ).tolist(),
            "power_reflected": np.broadcast_to(
                np.asarray(port.P_ref, dtype=float), (points,)
            ).tolist(),
        }
        waves[number] = (port.uf_inc, port.uf_ref)
        if models[number].kind == "rect_waveguide" and number not in read_as:
            below = _below_cutoff(port, frequency)
            if below is not None:
                reference, incident, reflected, power_incident, power_reflected = below
                entry.update(
                    z0=_complex_pair(reference, points),
                    incident=_complex_pair(incident, points),
                    reflected=_complex_pair(reflected, points),
                    power_incident=power_incident.tolist(),
                    power_reflected=power_reflected.tolist(),
                )
                waves[number] = (incident, reflected)
        if number in read_as:
            reading = read_as[number]
            incident, reflected = reading.waves(port.uf_tot, port.if_tot)
            reference = reading.impedance
            # A point on the cutoff has no impedance, and _check_the_impedances
            # refuses it.
            with np.errstate(invalid="ignore"):
                going, coming = _power(incident, reference), _power(reflected, reference)
            entry = {
                "z0": _complex_pair(reference, points),
                "incident": _complex_pair(incident, points),
                "reflected": _complex_pair(reflected, points),
                "power_incident": going.tolist(),
                "power_reflected": coming.tolist(),
                "raw": {key: entry[key] for key in ("z0", "incident", "reflected")},
            }
            waves[number] = (incident, reflected)
            readings[number] = {
                "corrected": True,
                "voltage_scale": reading.voltage,
                "current_scale": reading.current,
                "guide": list(reading.guide),
                "carried": reading.carried,
                "timestep": reading.timestep,
            }
        if models[number].kind == "rect_waveguide":
            stated[str(number)] = READ_IMPEDANCE if number in read_as else WAVE_IMPEDANCE
            turn = referred(
                models[number],
                placed.grid[models[number].propagation_axis],
                frequency,
                placed.length_unit,
                _fill(csx, placed, models[number]),
            )
            waves[number] = (waves[number][0] * turn, waves[number][1] / turn)
            _refer(entry, turn, points)
            moved[str(number)] = _moved(
                models[number], placed.grid[models[number].propagation_axis]
            )
        per_port[str(number)] = entry
        recorded[number] = residual.Record(
            voltage=residual.Probe(port.u_data.ui_time[0], port.ut_tot),
            current=residual.Probe(port.i_data.ui_time[0], port.it_tot),
            reference=port.Z_ref,
        )
    _check_the_impedances(per_port, models, frequency)
    if stated:
        records[IMPEDANCE_STATED] = stated
    if moved:
        records[REFERRED] = moved
    if readings:
        records[portreading.KEY] = readings
        driven = np.asarray(per_port[str(excited.number)]["power_incident"], dtype=float)
        records[nearfield.KEY] = _near_field(
            placed, models, ports, inside, read_as, csx, sim_dir, frequency, driven
        )

    incident = np.asarray(waves[excited.number][0], dtype=complex)
    s_parameters = {}
    for number in sorted(ports):
        reflected = np.asarray(waves[number][1], dtype=complex)
        s_parameters[f"S{number}{excited.number}"] = _complex_pair(reflected / incident, points)
    # Every port at once, so the record that drove the run is transformed once
    # for the device rather than once for each port weighed against it.
    records["tail_share"] = residual.tail_shares(recorded, excited.number, frequency)
    if readings:
        records[balance.KEY] = _balance(
            placed, models, ports, read_as, csx, frequency, driven, records["tail_share"]
        )

    return {
        "frequency": frequency.tolist(),
        "excited_port": excited.number,
        "ports": per_port,
        "s_parameters": s_parameters,
        "provenance": _provenance(problem, elapsed, records),
    }


def _moved(model: Port, lines: npt.ArrayLike) -> float:
    """How far a waveguide port's waves are moved to its reference plane, in
    length units, on a grid whose lines along the port's axis are ``lines``:
    positive where the plane stands before the probes, towards the face the mode
    is launched on."""
    axis = model.propagation_axis
    direction = 1 if model.stop[axis] > model.start[axis] else -1
    reference = model.start[axis] + direction * model.reference_depth
    return direction * (model.probe_plane(lines) - reference)


def referred(
    model: Port,
    lines: npt.ArrayLike,
    frequency: npt.ArrayLike,
    length_unit: float,
    fill: tuple[float, float, float] = (1.0, 1.0, 0.0),
) -> npt.NDArray[np.complex128]:
    """What a waveguide port's incoming wave is multiplied by, and its outgoing
    wave divided by, to move both from where openEMS read them to the port's
    reference plane.

    A wave travelling into the structure is ``exp(-j beta z)`` of itself a
    distance ``z`` further in, so at a plane ``s`` before the probes it was
    ``exp(j beta s)`` of what they read, and the wave travelling out is
    ``exp(-j beta s)`` of it there. ``beta`` is the propagation constant of the
    port's mode in the guide's fill, ``sqrt(k^2 eps mu - kc^2)`` with the fill's
    conductivity in the permittivity, taken on the root whose imaginary part is
    not positive: a lossy guide attenuates a wave as it travels, and below
    cutoff the mode decays. Only a guide uniform over ``s`` is moved exactly,
    which the translation holds the drawing to.

    :param fill: the guide's relative permittivity, relative permeability and
        conductivity in S/m.
    """
    epsilon, mu, kappa = fill
    angular = 2.0 * np.pi * np.asarray(frequency, dtype=float)
    wavenumber = angular / units.SPEED_OF_LIGHT
    a, b, mode = model.waveguide_arguments(length_unit)
    first, second = int(mode[2]), int(mode[3])
    cutoff = (first * np.pi / a) ** 2 + (second * np.pi / b) ** 2
    permittivity = epsilon - 1j * kappa / (angular * units.VACUUM_PERMITTIVITY)
    squared = wavenumber**2 * permittivity * mu - cutoff
    root = np.sqrt(squared.astype(complex))
    # The principal root has a positive imaginary part below a lossless
    # guide's cutoff, and that root grows along the guide.
    beta = np.where(root.imag > 0.0, -root, root)
    distance = _moved(model, lines) * length_unit
    return np.asarray(np.exp(1j * beta * distance), dtype=complex)


def _refer(entry: dict[str, Any], turn: npt.NDArray[np.complex128], points: int) -> None:
    """Move one port's record of its waves to its reference plane, in place.

    The power a wave carries moves with the square of its magnitude, which is
    one wherever the guide is lossless. What openEMS read is kept under
    ``raw`` where the port was read again, and is left where it was read.
    """
    incident = np.asarray(entry["incident"]["re"]) + 1j * np.asarray(entry["incident"]["im"])
    reflected = np.asarray(entry["reflected"]["re"]) + 1j * np.asarray(entry["reflected"]["im"])
    entry["incident"] = _complex_pair(incident * turn, points)
    entry["reflected"] = _complex_pair(reflected / turn, points)
    gain = np.abs(turn) ** 2
    entry["power_incident"] = (np.asarray(entry["power_incident"]) * gain).tolist()
    entry["power_reflected"] = (np.asarray(entry["power_reflected"]) / gain).tolist()


def _fill(csx: Any, problem: Problem, model: Port) -> tuple[float, float, float]:
    """What fills a waveguide port's guide, as openEMS holds it: relative
    permittivity, relative permeability and conductivity in S/m.

    The engine is asked at the middle of the port's cross-section, halfway
    between the face the mode is launched on and the deeper of the probes and
    the reference plane. The translation holds the guide uniform over that
    length, so one point answers for it. Empty space is vacuum.
    """
    from CSXCAD.CSProperties import PropertyType

    axis = model.propagation_axis
    direction = 1 if model.stop[axis] > model.start[axis] else -1
    reach = max(model.length, model.reference_depth)
    point = [0.5 * (model.start[dim] + model.stop[dim]) for dim in range(3)]
    point[axis] = model.start[axis] + direction * reach / 2.0
    found = csx.GetPropertyByCoordPriority(point, int(PropertyType.MATERIAL))
    if found is None:
        return (1.0, 1.0, 0.0)
    material = {one.name: one for one in problem.materials}.get(found.GetName())
    if material is None:
        return (1.0, 1.0, 0.0)
    return (material.epsilon, material.mu, material.kappa + material.conductivity)


def _near_field(
    problem: Problem,
    models: dict[int, Port],
    ports: dict[int, Any],
    inside: Inside,
    read_as: dict[int, portreading.Reading],
    csx: Any,
    sim_dir: Path,
    frequency: npt.NDArray[np.float64],
    driven: npt.NDArray[np.float64],
) -> dict[str, Any]:
    """How far the waves each waveguide port read differ from those its planes
    inside predict, as :mod:`.nearfield` records it, and why a port is not
    weighed where it is not.

    A plane inside is read as the port's own is, so it is weighed only where the
    port's own reading was modelled. Its probe files are read after the port's
    own and the matrix does not depend on them, so a file that cannot be read,
    or that reads as nothing, is a reason recorded against the port and never a
    failure of the run.

    :param driven: Per frequency, the power driven into the run.
    """
    weighed: dict[int, nearfield.Weighed] = {}
    unchecked: dict[int, str] = {}
    missed: dict[int, str] = {}
    for number, model in sorted(models.items()):
        if model.kind != "rect_waveguide":
            unchecked[number] = "only a rectangular waveguide port is read again"
            continue
        if number not in read_as:
            unchecked[number] = "its reading is left as openEMS returned it"
            continue
        if number not in inside:
            missed[number] = "its box holds no grid line between its source and its plane"
            continue
        here = read_as[number]
        axis = model.propagation_axis
        read = here.waves(ports[number].uf_tot, ports[number].if_tot)
        planes: list[tuple[float, npt.NDArray[np.float64]]] = []
        for again, cells, probes in inside[number]:
            depth = abs(again.stop[axis] - model.stop[axis])
            try:
                # What Port.CalcPort raises on a probe file that is missing, empty
                # or malformed, and what the reading raises on a guide it does not
                # model.
                probes.CalcPort(str(sim_dir), frequency)
                there = _read_waveguide(problem, again, probes, csx, sim_dir, frequency)
            except (portreading.NotRead, OSError, ValueError, IndexError) as why:
                # The deeper plane bounds the field alone, so a shallower plane
                # that cannot be read leaves the port weighed at one plane.
                if not planes:
                    missed[number] = f"it is not read {_cells(cells)} into its box: {why}"
                break
            expected = portreading.predicted(
                here,
                there,
                problem.grid[axis],
                problem.length_unit,
                probes.uf_tot,
                probes.if_tot,
            )
            apart = sum(
                _power(got - want, here.impedance) for got, want in zip(read, expected, strict=True)
            )
            kept = here.propagates & there.propagates & (driven > 0)
            planes.append((depth, np.where(kept, apart / np.where(kept, driven, 1.0), 0.0)))
        if number in missed:
            continue
        (depth, differ), shallower = planes[0], planes[1:]
        across, along = here.cutoffs
        rate = nearfield.decay(
            here.angular / units.SPEED_OF_LIGHT, across, along, max(_orders(model))
        )
        weighed[number] = nearfield.Weighed(
            depth=depth,
            differ=differ,
            decay=rate * problem.length_unit,
            shallow=shallower[0][0] if shallower else None,
            nearer=shallower[0][1] if shallower else None,
        )
    return nearfield.record(weighed, unchecked, missed, frequency)


def _balance(
    problem: Problem,
    models: dict[int, Port],
    ports: dict[int, Any],
    read_as: dict[int, portreading.Reading],
    csx: Any,
    frequency: npt.NDArray[np.float64],
    driven: npt.NDArray[np.float64],
    tail: dict[int, float],
) -> dict[str, Any]:
    """How much of the power driven in the ports account for, as
    :mod:`.balance` records it, or why it is not weighed.

    Each port's net power is read as the port's own reading reads it, so the
    balance is weighed only where every port is a waveguide port whose reading
    was modelled.

    :param driven: Per frequency, the power driven into the run.
    """
    if any(model.kind != "rect_waveguide" for model in models.values()):
        return balance.unweighed("a port of another kind is in the run")
    if set(read_as) != set(models):
        return balance.unweighed("the ports' readings are left as openEMS returned them")
    grid = [problem.grid[dim] for dim in range(3)]
    apart = balance.apart(grid, problem.boundary, list(models.values()))
    if apart:
        return balance.unweighed(apart)
    moved = balance.moved_by(tail)
    if not moved < balance.BAR / 2:
        return balance.unweighed(f"the record's tail can move it by {moved:.3g}")
    propagates = np.logical_and.reduce([read_as[number].propagates for number in sorted(ports)])
    values = balance.share(
        [_net(read_as[number], ports[number]) for number in sorted(ports)], driven, propagates
    )
    excited = problem.excited_port.number
    sent = balance.share(
        [
            _incident(read_as[number], ports[number])
            for number in sorted(ports)
            if number != excited
        ],
        driven,
        propagates,
    )
    dissipates = any(material.kind in LOSSY_KINDS for material in problem.materials)
    leaks = None
    if not dissipates:
        engine = _conducts(csx, problem)

        def conducts(point: portreading.Point) -> bool:
            # A point holding a material stops a port's reading, and here it is
            # a point the field can reach.
            try:
                return engine(point)
            except portreading.NotRead:
                return False

        leaks = balance.closed(grid, problem.boundary, list(models.values()), conducts)
    return balance.record(values, sent, frequency, dissipates, leaks)


def _net(reading: portreading.Reading, port: Any) -> npt.NDArray[np.float64]:
    """Per frequency, the power travelling in less the power travelling out,
    from what ``port`` recorded and how ``reading`` says it read it."""
    incident, reflected = reading.waves(port.uf_tot, port.if_tot)
    return _power(incident, reading.impedance) - _power(reflected, reading.impedance)


def _incident(reading: portreading.Reading, port: Any) -> npt.NDArray[np.float64]:
    """Per frequency, the power travelling in, read as :func:`_net` reads it."""
    incident, _ = reading.waves(port.uf_tot, port.if_tot)
    return _power(incident, reading.impedance)


def _provenance(problem: Problem, elapsed: float, records: dict) -> dict:
    """Enough to falsify the result later.

    Without the envelope digest a results file cannot be tied to the input that
    produced it, and a stale file is indistinguishable from a fresh one.

    ``records`` is what :func:`_extract` measured off the recorded signals, per
    port rather than per run. The two ports of a sweep stop ringing at different
    times, so the per-port figure is the useful one.
    """
    try:
        import openEMS as engine

        version = getattr(engine, "__version__", "unknown")
    except Exception:
        # Everything else here is arithmetic over the envelope, so the record
        # is written whether or not the bindings are importable. A solve has
        # them loaded long before this point.
        version = "unknown"

    return {
        # The study's own name, so a result can say what it is of.
        # SParameters.network() and the plot both read it. Without it every
        # chart and every Touchstone network is unnamed.
        "title": problem.title,
        "solver": "openEMS",
        "solver_version": version,
        "adapter_version": ADAPTER_VERSION,
        "envelope_digest": problem.digest(),
        "wall_seconds": round(elapsed, 3),
        # The bar the tail shares above were judged against, so that anything
        # reading this file later judges them the same way rather than at full
        # scale. The judgement itself is not stored. It is one line of
        # arithmetic, and a stored verdict goes stale when the bar moves.
        "smallest_response": problem.smallest_response,
        "max_timesteps": problem.termination.max_timesteps,
        "end_criteria": problem.termination.end_criteria,
        "reproducible": problem.termination.reproducible,
        **records,
        "cells": problem.grid.cell_count,
        "modelled": modelled(problem),
        "threads": problem.threads,
        "timestep_factor": problem.timestep_factor,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def modelled(problem: Problem) -> list[dict[str, Any]]:
    """How openEMS is given each lossy material, one record to a material.

    The records are the ones ``Results/modelled.py`` puts into words. A lossy
    dielectric is handed over as one conductivity, fixed for the run: the one a
    loss tangent became at the centre of the band added to the material's own,
    which is what :func:`_add_material` gives the engine. A conducting sheet is
    one plane carrying the current through it as a whole
    (``openEMS/FDTD/extensions/operator_ext_conductingsheet.cpp``).

    One record of the model's outside goes with them, where any face of the
    domain carries an absorbing layer: how deep the layer is in cells, which
    faces the structure stands clear of it on and how far, which faces it runs
    out through. It is written so that two runs of one drawing on two backends
    can be told apart by what each held open and how far its boundary stood,
    which is what both of them state about an outside. A record of the medium
    goes with them where it is not vacuum.
    """
    records: list[dict[str, Any]] = []
    for material in problem.materials:
        if material.kind == "lossy_dielectric":
            record: dict[str, Any] = {
                "material": material.name,
                "held": "conductivity",
                "conductivity": material.kappa + material.conductivity,
            }
            if material.kappa:
                record.update(folded=material.kappa, at=problem.frequency.center)
            records.append(record)
        elif material.kind == "conducting_sheet":
            records.append(
                {
                    "material": material.name,
                    "sheet": "net current",
                    "conductivity": material.conductivity,
                    "thickness": material.thickness,
                }
            )
    outside = _outside(problem)
    if outside is not None:
        records.append(outside)
    if problem.medium:
        records.append({"medium": problem.medium})
    return records


#: The boundary conditions that carry a wave out of the domain. Everything else
#: openEMS takes on a face reflects it: a perfect electric or magnetic wall.
_ABSORBING = ("PML", "MUR")


def _outside(problem: Problem) -> dict[str, Any] | None:
    """The model's outside, as one record, or ``None`` where every face of the
    domain is a wall.

    What the envelope holds is the condition on each face and the padding the
    document asked for, in the order ``{axis}{side}``. A padding that is a length
    is air the domain reserves on that face, and the absorber stands beyond it, so
    that length is how far the boundary stands from the structure. A padding of
    ``through`` ends the domain on the structure and takes the absorber out of the
    structure's own extent, so the boundary stands nowhere clear of it. A padding
    of zero is the face the domain ends on.

    The distance recorded is the smallest of the lengths, which is the nearest the
    boundary stood to the structure. A face that absorbs with a padding of zero is
    one waveguide ports cover: the domain ends on the structure there, and the
    layer stands beyond the ports' plane. It is recorded in ``ends``, because it is
    held open to the port's guide rather than to free space.

    The depth is recorded per face, in the order ``{axis}{side}``. The envelope
    holds one count for an axis whose faces agree and a pair for one whose faces
    differ, and a wall face has none.
    """
    padding = list(problem.grid.params.get("padding") or ())
    if not padding:
        return None
    faces: list[str] = []
    through: list[str] = []
    ends: list[str] = []
    distances: list[float] = []
    for dim, axis in enumerate(AXIS_NAMES):
        for half, side in enumerate(("Min", "Max")):
            if not str(problem.boundary[dim * 2 + half]).upper().startswith(_ABSORBING):
                continue
            # Spelt as the mesh policy's own property spells a face, which is
            # what the record is read against.
            name = f"{axis.upper()}{side}"
            stated = padding[dim][half]
            if str(stated) == THROUGH:
                through.append(name)
            elif float(stated) > 0.0:
                faces.append(name)
                distances.append(float(stated))
            else:
                ends.append(name)
    if not faces and not through and not ends:
        return None
    return {
        "boundary": "absorbing layer",
        "cells": [cells for dim in range(3) for cells in _absorber_cells(problem.grid, dim)],
        "clearance": min(distances) if distances else 0.0,
        "faces": faces,
        "through": through,
        "ends": ends,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("envelope", help="path to the envelope JSON")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="construct the structure and write the XML, but do not solve",
    )
    args = parser.parse_args(argv)

    marker("STARTED")
    try:
        problem = read_envelope(args.envelope)
    except (OSError, ValueError, EnvelopeError) as error:
        marker("ERROR", f"kind=envelope message={_sanitise(error)}")
        return 1

    marker("ENVELOPE", f"digest={problem.digest()[:16]} ports={len(problem.ports)}")

    # Checked here rather than only by whoever built the envelope. This is the
    # last point before the solver, and every route passes through it: the
    # workbench, a script, a bug report replayed by hand. Left to the caller,
    # every guard would be opt-in, and for a defect that ends in a plausible
    # wrong number rather than a crash an opt-in guard is no guard.
    from . import preflight

    findings = everything_wrong(problem)
    for finding in findings:
        marker("CHECK", f"severity={finding.severity} {_sanitise(finding)}")
    blocking = preflight.refusals(findings)
    if blocking:
        marker("ERROR", f"kind=UnsupportedModel message={_sanitise(blocking[0])}")
        return 1

    try:
        path = solve(problem, Path(args.envelope).parent, build_only=args.build_only)
    except Exception as error:  # noqa: BLE001 - the boundary reports everything
        marker("ERROR", f"kind={type(error).__name__} message={_sanitise(error)}")
        return 2

    if path is not None:
        marker("RESULTS", str(path))
    marker("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
