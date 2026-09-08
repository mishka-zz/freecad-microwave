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
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from . import excitation, residual
from .capabilities import ADAPTER_VERSION
from .model import AXIS_NAMES, EnvelopeError, Material, Port, Problem, Solid
from .preflight.finding import WARN
from .write import read_envelope

MARKER = "OPENEMS:"

RESULTS_NAME = "results.json"
STRUCTURE_NAME = "structure.xml"
RUN_DIRNAME = "run"


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
    run on every keystroke. The second asks whether the grid in the envelope
    still holds the conductors it was built for. That is a question about the
    finished grid and costs a triangle for every point it samples, which is
    affordable beside a solve and not beside a keystroke, so it runs here and
    nowhere else.
    """
    from . import conductors, preflight

    return preflight.check(problem) + conductors.check(problem)


def build(problem: Problem, sim_dir: Path) -> tuple[Any, Any, dict[int, Any]]:
    """Construct the CSX structure and the FDTD object. No solving.

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
    # its own
    # ``Dimensions: <nx>x<ny>x<nz> = <n> Cells`` a few lines later, and the two
    # have to be readable as one grid.
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
    for solid in problem.solids:
        moved = add_solid(properties[solid.material], solid, problem.as_given(solid))
        if moved is not None:
            grew.append(moved)
    if grew:
        marker("GROWN", f"solids={len(grew)} most={max(grew):.6g}")

    ports = {}
    for port in problem.ports:
        metal = properties[port.metal] if port.metal else None
        ports[port.number] = _add_port(
            fdtd, csx, metal, port, problem.length_unit, problem.grown_by
        )

    csx.Write2XML(str(sim_dir / STRUCTURE_NAME))
    marker("BUILT")
    return fdtd, csx, ports


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
        return csx.AddMaterial(
            material.name,
            epsilon=material.epsilon,
            mue=material.mu,
            kappa=material.kappa,
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

    fdtd, _csx, ports = build(problem, directory)
    if build_only:
        return None

    started = time.monotonic()
    marker("SOLVER_STARTED", f"threads={problem.threads} dir={sim_dir}")
    fdtd.Run(str(sim_dir), cleanup=True, numThreads=problem.threads)
    elapsed = time.monotonic() - started
    marker("SOLVER_FINISHED", f"seconds={elapsed:.1f}")

    results = _extract(problem, ports, sim_dir, elapsed)

    # Reported as well as recorded, and here rather than in pre-flight. A
    # residual cannot be known before the solve, and by the time one exists the
    # minutes are spent, so this is a warning and never a refusal. The number is
    # in the provenance either way.
    warning = residual.unfinished(results["provenance"]["tail_share"], problem.smallest_response)
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


def _extract(problem: Problem, ports: dict, sim_dir: Path, elapsed: float) -> dict:
    """Probe files to numbers.

    ``CalcPort`` is called without a reference impedance on purpose. For an
    ``MSLPort`` a reference would overwrite the impedance it measured from the
    field, and that measured impedance is the quantity wanted. It is also what a
    truncated record corrupts, so each port's tail is weighed here, off the same
    probes and the same axes the transform was taken over.
    """
    frequency = problem.frequency.values()
    points = frequency.size
    excited = problem.excited_port

    per_port: dict[str, Any] = {}
    records: dict[str, dict[int, Any]] = {"recorded_samples": {}}
    for number, port in sorted(ports.items()):
        port.CalcPort(str(sim_dir), frequency)
        records["recorded_samples"][number] = int(np.size(port.ut_tot))
        per_port[str(number)] = {
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

    incident = np.asarray(ports[excited.number].uf_inc, dtype=complex)
    s_parameters = {}
    # Each probe on the axis it was written on. A Yee scheme staggers the two
    # by half a step in time, and openEMS transforms each against its own
    # column. Both are taken, because the leakage a truncated record costs is an
    # error in the S-parameters above, and those are made of waves rather than
    # volts.
    recorded = {
        number: residual.Record(
            voltage=residual.Probe(port.u_data.ui_time[0], port.ut_tot),
            current=residual.Probe(port.i_data.ui_time[0], port.it_tot),
            reference=port.Z_ref,
        )
        for number, port in sorted(ports.items())
    }
    for number, port in sorted(ports.items()):
        reflected = np.asarray(port.uf_ref, dtype=complex)
        s_parameters[f"S{number}{excited.number}"] = _complex_pair(reflected / incident, points)
    # Every port at once, so the record that drove the run is transformed once
    # for the device rather than once for each port weighed against it.
    records["tail_share"] = residual.tail_shares(recorded, excited.number, frequency)

    return {
        "frequency": frequency.tolist(),
        "excited_port": excited.number,
        "ports": per_port,
        "s_parameters": s_parameters,
        "provenance": _provenance(problem, elapsed, records),
    }


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
        "threads": problem.threads,
        "timestep_factor": problem.timestep_factor,
        "python": platform.python_version(),
        "platform": platform.platform(),
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
