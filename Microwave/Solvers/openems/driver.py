# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The solver side of the process boundary. Runs under openEMS' interpreter.

May import openEMS and CSXCAD. **Must never import FreeCAD** - it runs in a
different Python, launched by :mod:`.run`, precisely because FreeCAD's
interpreter does not have the solver bindings.

Invoked as a module::

    python -m Microwave.Solvers.openems.driver <envelope.json> [--build-only]

It reports progress by writing line-oriented markers to stdout, which
:mod:`.run` parses. Markers are a contract: keep them stable, one per line,
prefixed ``OPENEMS:``.

===========================  ==========================================
Marker                       Meaning
===========================  ==========================================
``STARTED``                  process is alive, envelope not yet read
``ENVELOPE digest=…``        envelope parsed and hashed
``CHECK severity=… …``       a pre-flight finding, or one about the solve
``GRID cells=… lines=…``     grid installed
``BUILT``                    geometry and ports constructed
``SOLVER_STARTED``           handed off to openEMS
``SOLVER_FINISHED``          time stepping done
``RESULTS <path>``           results written
``DONE``                     clean exit
``ERROR kind=… message=…``   giving up
===========================  ==========================================

Exit codes: ``0`` success, ``1`` the envelope is bad, ``2`` anything else.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import residual
from .capabilities import ADAPTER_VERSION
from .model import AXIS_NAMES, EnvelopeError, Problem
from .preflight.finding import WARN
from .write import read_envelope

MARKER = "OPENEMS:"

RESULTS_NAME = "results.json"
STRUCTURE_NAME = "structure.xml"
RUN_DIRNAME = "run"


def marker(name: str, detail: str = "") -> None:
    print(f"{MARKER}{name}{' ' + detail if detail else ''}", flush=True)


def _sanitise(text: str) -> str:
    """Markers are line-oriented, so a message may not contain newlines."""
    return " ".join(str(text).split())


def build(problem: Problem, sim_dir: Path):
    """Construct the CSX structure and the FDTD object. No solving.

    Ordering is not free here. ``MSLPort`` reads the grid at construction time
    to place its probes and raises if fewer than five lines exist on its
    propagation axis, so the grid must be installed *before* any port is added.
    """
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS

    csx = ContinuousStructure()
    grid = csx.GetGrid()
    grid.SetDeltaUnit(problem.length_unit)
    for dim, axis in enumerate(AXIS_NAMES):
        grid.AddLine(axis, problem.grid[dim].tolist())
    # The shape travels with the count: openEMS prints its own
    # ``Dimensions: 141x117x57 = 940329 Cells`` a few lines later, and the two
    # have to be readable as one grid.
    shape = "x".join(str(len(problem.grid[dim])) for dim in range(3))
    marker("GRID", f"cells={problem.grid.cell_count} lines={shape}")

    # ``TimeStepFactor`` is passed unconditionally, including at 1.0. openEMS
    # applies it only when it is below one (``openEMS::SetupFDTD``), so one is
    # the engine's own step by the engine's own arithmetic, not by ours skipping
    # the call.
    #
    # It is also the one setting here that leaves no usable trace in any file
    # openEMS writes: ``openEMS::Write2XML`` records the attribute only when the
    # factor is *above* one, which is the range that is never applied, so the
    # XML says nothing for every value that did something and lies for the one
    # that did not. The provenance below is the only record.
    fdtd = openEMS(
        NrTS=problem.termination.max_timesteps,
        EndCriteria=problem.termination.end_criteria,
        TimeStepFactor=problem.timestep_factor,
    )
    fdtd.SetGaussExcite(problem.frequency.center, problem.frequency.half_bandwidth)
    fdtd.SetBoundaryCond(list(problem.boundary))
    fdtd.SetCSX(csx)

    properties = {
        material.name: _add_material(csx, material, problem.length_unit)
        for material in problem.materials
    }

    for solid in problem.solids:
        properties[solid.material].AddBox(
            list(solid.lower), list(solid.upper), priority=solid.priority
        )

    ports = {}
    for port in problem.ports:
        metal = properties[port.metal] if port.metal else None
        ports[port.number] = _add_port(fdtd, metal, port, problem.length_unit)

    csx.Write2XML(str(sim_dir / STRUCTURE_NAME))
    marker("BUILT")
    return fdtd, csx, ports


def _add_material(csx, material, length_unit: float):
    """One envelope material to one CSXCAD property.

    ``length_unit`` exists for one field. Every length in the CSX tree is in
    grid units - except a conducting sheet's thickness, which CSXCAD takes in
    **metres**, alongside conductivity in S/m. Getting it wrong does not fail:
    openEMS notices the resulting surface-impedance fit is out of range, prints
    to stderr, clamps to its last tabulated coefficients and carries on with a
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


def _add_port(fdtd, metal_prop, port, length_unit: float):
    """One envelope port to one upstream openEMS port object.

    Upstream port classes are used rather than reimplemented. The impedance an ``MSLPort`` extracts,
    ``sqrt(Et*dEt / (Ht*dHt))``, is the definition this adapter reports.

    A ``RectWGPort`` is different in kind and worth knowing about before
    trusting its numbers: it *computes* its reference impedance analytically
    from the mode and the guide dimensions rather than measuring it
    (``ports.py``: ``self.ZL = k * Z0 / self.beta``). Comparing its ``Z_ref``
    against the closed form would therefore be circular. What a waveguide
    genuinely measures is the phase of the wave that crossed it.
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
        # a and b in metres, while start/stop are in grid units; both belong to
        # an axis rather than to a wall, and the mode name is numbered to match.
        # Derived by the model so the three cannot disagree.
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

    # Said out loud as well as recorded, and here rather than in pre-flight:
    # nobody can know a residual before the solve, and by the time one exists
    # the minutes are spent, so this is a warning and never a refusal. The
    # number is in the provenance either way.
    warning = residual.unfinished(results["provenance"]["tail_share"])
    if warning:
        marker("CHECK", f"severity={WARN} {_sanitise(warning)}")

    path = directory / RESULTS_NAME
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


def _complex_pair(values, points: int) -> dict[str, list[float]]:
    """Always a list of ``points`` floats.

    A lumped port's reference impedance is a scalar, not a per-frequency array;
    written out as-is it would produce a bare number where every reader expects
    a sequence. Broadcast here rather than special-casing on the way back in.
    """
    array = np.broadcast_to(np.asarray(values, dtype=complex), (points,))
    return {"re": array.real.tolist(), "im": array.imag.tolist()}


def _extract(problem: Problem, ports: dict, sim_dir: Path, elapsed: float) -> dict:
    """Probe files to numbers.

    ``CalcPort`` is called without a reference impedance on purpose: for an
    ``MSLPort`` that would *overwrite* the impedance it just measured from the
    field, which is the quantity we want. That measured impedance is also what a
    truncated record corrupts, which is why each port's tail is weighed here,
    off the same arrays the transform was taken over.
    """
    frequency = problem.frequency.values()
    points = frequency.size
    excited = problem.excited_port

    per_port: dict[str, Any] = {}
    records: dict[str, dict[int, Any]] = {"tail_share": {}, "recorded_samples": {}}
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
    for number, port in sorted(ports.items()):
        reflected = np.asarray(port.uf_ref, dtype=complex)
        s_parameters[f"S{number}{excited.number}"] = _complex_pair(reflected / incident, points)
        # Every probe of a run is dumped on one interval, so any of their time
        # axes is the record's; the summing in ``Port.ReadUIData`` relies on the
        # same thing. Against ``incident``, because what a truncated record
        # costs is an error in the S-parameters above, not in the volts.
        records["tail_share"][number] = residual.tail_share(
            port.u_data.ui_time[0], port.ut_tot, frequency, incident
        )

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
    produced it, and a stale one is indistinguishable from a fresh one.

    ``records`` is what :func:`_extract` measured off the recorded signals, per
    port. Per port and not per run: a two-port sweep's ports do not stop ringing
    together, and which one is still going is the useful half.
    """
    try:
        import openEMS as engine

        version = getattr(engine, "__version__", "unknown")
    except Exception:  # pragma: no cover - only if the bindings vanish mid-run
        version = "unknown"

    return {
        # The study's own name, so a result can say what it is of.
        # SParameters.network() and the plot both read it; without it every
        # chart and every Touchstone network is unnamed.
        "title": problem.title,
        "solver": "openEMS",
        "solver_version": version,
        "adapter_version": ADAPTER_VERSION,
        "envelope_digest": problem.digest(),
        "wall_seconds": round(elapsed, 3),
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

    # Checked here, and not only by whoever built the envelope: this is the last
    # point before the solver and the one every route passes through - the
    # workbench, a script, a bug report replayed by hand. Left to the caller,
    # every guard is opt-in, which for a defect that ends in a plausible wrong
    # number rather than a crash is the same as not having it.
    from . import preflight

    findings = preflight.check(problem)
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
