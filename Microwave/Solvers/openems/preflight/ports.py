# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checks about the ports, one port at a time.

:func:`_check_ports` is the loop, and everything below it asks one question of
one port - that the adapter can build its kind, that it survives snapping to
the grid, that a guide is empty and its mode propagates, that its box lies inside
the grid and clear of the absorber, and that the lines it needs exist.

:func:`_check_lumped_excitation_beside_a_measured_line` is the exception, and
sits above the loop: it is about the *combination* of ports in one run, which no
port can answer on its own.
"""

from __future__ import annotations

import numpy as np

from ..capabilities import Capabilities
from ..model import AXIS_NAMES, DIMENSIONS, SPEED_OF_LIGHT, MeshGrid, Port, Problem
from .absorber import _absorber_bounds, _absorber_cells, _absorber_depth
from .finding import _ON_THE_GRID, REFUSE, WARN, Finding


def _check_lumped_excitation_beside_a_measured_line(problem: Problem) -> list[Finding]:
    """A run a lumped port drives, with a microstrip port left to measure.

    In such a run the ``MSLPort`` reports ``z0 = nan`` at every frequency
    point, while the same two ports in the same document are finite in the run
    the microstrip port drives. Not a resistance effect - the table is
    identical at 0 ohms - and not the near-null indeterminacy, which blanks
    some points rather than all of them. The control, two ``MSLPort``s, is
    finite in both runs.

    ``Z_ref`` per port is what the whole matrix is normalised in, so this run
    yields nothing, and it yields nothing after taking its full wall time.

    A warning and not a refusal. The consequence is certain on what has been
    measured, but the *mechanism* is not established, so refusing would block
    a geometry nobody has tried on the strength of one combination. Saying it
    before the minutes are spent is the part that has value; the assembly
    refuses by name afterwards either way.
    """
    driven = [port for port in problem.ports if port.excite]
    if not any(port.kind == "lumped" for port in driven):
        return []
    measured = [port for port in problem.ports if port.kind == "microstrip" and not port.excite]
    if not measured:
        return []

    lumped = ", ".join(str(port.number) for port in driven if port.kind == "lumped")
    return [
        Finding(
            WARN,
            # `.name`, as every other check spells a port: a labelled port
            # named two ways is one object counted twice.
            measured[0].name,
            f"this run is driven by lumped port {lumped}, and a microstrip "
            "port measured in a lumped-driven run reports a non-finite "
            "reference impedance at every frequency on this engine. The run "
            "will complete, take its full time and produce no S-matrix. Drive "
            "the microstrip port instead, or measure both ports the same way",
        )
    ]


def _check_ports(problem: Problem, caps: Capabilities) -> list[Finding]:
    findings = []
    for port in problem.ports:
        if not caps.supports_port(port.kind):
            findings.append(
                Finding(
                    REFUSE,
                    port.name,
                    f"port type {port.kind!r} is not supported; this adapter "
                    f"handles {sorted(caps.port_types)}",
                )
            )
        findings += _check_port_inside_the_grid(port, problem.grid)
        findings += _check_required_lines_exist(port, problem.grid)
        findings += _check_port_clear_of_absorber(port, problem.grid)
        findings += _check_the_guide_is_empty(port, problem)
        findings += _check_the_mode_propagates(port, problem)
        findings += _check_the_element_survives_snapping(port, problem.grid)
    return findings


def _check_the_element_survives_snapping(port: Port, grid: MeshGrid) -> list[Finding]:
    """A lumped gap thinner than a cell snaps shut, and the resistor is dropped.

    ``Operator::Calc_LumpedElements`` snaps the element's box to the mesh and
    then, if both ends land on the *same* line, prints a warning and lays no
    element at all (``operator.cpp``:1637-1652; the RLC extension repeats it at
    ``operator_ext_lumpedRLC.cpp``:262-277). A gap wider than a cell builds
    silently; one thinner prints

        Warning: Lumped Element with zero (snapped) length is invalid! skipping.

    and the run finishes with every S-parameter **NaN** - the port has an
    excitation and probes across a gap with nothing in it. The warning is merged
    into the run log rather than lost, but it is one line among thousands of
    progress lines and nothing acts on it; what the user meets is the results
    layer refusing a matrix that holds no numbers, which cannot say why.

    ``model.Port`` already refuses a gap of exactly zero, which is openEMS' own
    Python-side check (``openEMS/ports.py``:169). That one runs before snapping, so it
    passes everything the mesh is about to close.

    Snapping is nearest-line: ``SnapToMeshLine`` returns the first line whose
    dual node is not below the coordinate, and a dual node is the midpoint
    between two lines (``operator.cpp``:144-157), so ``argmin`` reproduces it,
    ties included.
    """
    if port.kind != "lumped" or port.excitation_axis is None:
        return []

    axis = port.excitation_axis
    lines = np.asarray(grid[axis], dtype=float)
    start, stop = sorted((port.start[axis], port.stop[axis]))
    if stop < lines[0] or start > lines[-1]:
        return []  # entirely off this axis: _check_port_inside_the_grid owns it

    first, last = (int(np.argmin(np.abs(lines - value))) for value in (start, stop))
    if first != last:
        return []

    return [
        Finding(
            REFUSE,
            port.name,
            f"its gap spans {AXIS_NAMES[axis]}={start:.4g} to {stop:.4g}, which "
            f"is shorter than the cell there: both ends snap to the single grid "
            f"line at {lines[first]:.4g}. openEMS snaps a lumped element to the "
            f"mesh and skips one whose snapped length is zero, so the resistor "
            f"would not be laid at all and the run would return NaN. Refine the "
            f"mesh across {AXIS_NAMES[axis]}, or widen the gap",
        )
    ]


def _check_the_guide_is_empty(port: Port, problem: Problem) -> list[Finding]:
    """This adapter drives openEMS' waveguide port at its vacuum default.

    ``WaveguidePort.__init__`` sets ``ref_index = 1`` unconditionally and never
    reads a material (``openEMS/ports.py``:355). Everything the port reports is
    built on it: the phase constant, and through it the mode impedance the
    S-parameters are referenced to.

    A filled guide asked for over a band where it genuinely propagates comes
    back with nan S-parameters across the lower half, a reference impedance
    several times theory, and more power out than in.

    The engine is not the limit, and the message must not claim it is. Setting
    ``ref_index`` from the fill makes ``beta`` exact and leaves the whole error
    in ``ZL``, which is built from free-space ``Z0`` where the medium's
    ``Z0 / n`` belongs. The other half is
    ``CalcPort(..., ref_impedance=k * Z0 * mu_r / (n * beta))``, since ``ZL``
    only *defaults* ``Z_ref`` and the wave decomposition reads ``Z_ref`` alone
    (``openEMS/ports.py``:136-139). What stops that being two lines is
    ``driver._extract``, which passes no reference impedance on purpose so an
    ``MSLPort`` keeps the impedance it measured - so the override has to be per
    port kind, and needs a gate of its own. Until then this refuses by name.

    The fill is the solids that overlap the port box, not the largest
    permittivity in the model, which is usually somewhere else entirely.
    """
    if port.kind != "rect_waveguide":
        return []

    epsilon = {material.name: material for material in problem.materials}
    lower = tuple(min(a, b) for a, b in zip(port.start, port.stop))
    upper = tuple(max(a, b) for a, b in zip(port.start, port.stop))

    findings = []
    for solid in problem.solids:
        if any(
            solid.upper[axis] <= lower[axis] or solid.lower[axis] >= upper[axis]
            for axis in range(DIMENSIONS)
        ):
            continue
        material = epsilon.get(solid.material)
        if material is None or (material.epsilon == 1.0 and material.mu == 1.0):
            continue
        findings.append(
            Finding(
                REFUSE,
                port.name,
                f"the guide is filled with {solid.name!r} (relative "
                f"permittivity {material.epsilon:g}, permeability "
                f"{material.mu:g}), and this adapter does not yet drive a "
                f"filled waveguide port: openEMS would report the cutoff, the "
                f"phase constant and the mode impedance as if the guide were "
                f"empty. Model the guide empty, or use a lumped or microstrip "
                f"port",
            )
        )
    return findings


def _check_the_mode_propagates(port: Port, problem: Problem) -> list[Finding]:
    """A waveguide mode below cutoff carries nothing, and says nothing.

    Below cutoff the phase constant is imaginary: the field decays instead of
    travelling, so the run completes its full step count and every S-parameter
    comes back at the noise floor. Asking for ``TE01`` on WR-42 over 20-26 GHz
    does exactly that - cutoff 34.9 GHz, nothing propagates, no error.

    The cutoff is the **vacuum** one, because that is the one openEMS uses:
    ``WaveguidePort`` fixes ``ref_index = 1`` and ``CalcPort`` builds
    ``beta = sqrt(k^2 - kc^2)`` from it (``openEMS/ports.py``:393-395). Dividing by
    ``sqrt(epsilon)`` here instead made this check disagree with the engine it
    is checking - it concluded a filled guide propagated while the engine was
    computing that the mode did not exist. A guide that is not empty is refused
    by :func:`_check_the_guide_is_empty` before this runs, so the two agree.
    """
    if port.kind != "rect_waveguide":
        return []

    a, b, mode = port.waveguide_arguments(problem.length_unit)
    first, second = (int(digit) for digit in mode[2:])
    kc = float(np.hypot(first * np.pi / a, second * np.pi / b))
    cutoff = SPEED_OF_LIGHT * kc / (2 * np.pi)

    if cutoff >= problem.frequency.stop:
        return [
            Finding(
                REFUSE,
                port.name,
                f"{port.mode} cuts off at {cutoff / 1e9:.3g} GHz in this guide, "
                f"above the top of the band ({problem.frequency.stop / 1e9:.3g} "
                f"GHz). Nothing would propagate: the run would take its full "
                f"length and return an S-matrix of noise. The guide measures "
                f"{a * 1e3:.4g} x {b * 1e3:.4g} mm",
            )
        ]
    if cutoff > problem.frequency.start:
        return [
            Finding(
                WARN,
                port.name,
                f"{port.mode} cuts off at {cutoff / 1e9:.3g} GHz, inside the "
                f"band. Below that the mode decays rather than travels, so the "
                f"results are only meaningful above it",
            )
        ]
    return []


#: Port kinds that re-derive their geometry from the lines they land on instead
#: of being laid as the box they were given, so an overhang costs them nothing.
#:
#: ``MSLPort`` picks its excitation plane and its three voltage probes by
#: ``argmin`` over the propagation axis' lines (``openEMS/ports.py``:258-264, :297) -
#: and *only* over that axis. The strip, the probe boxes' transverse extent and
#: the current probes all take the box verbatim (``openEMS/ports.py``:246-249, :273-274,
#: :284-286), so the exemption this set grants is per axis as well as per kind:
#: see :func:`_check_port_inside_the_grid`.
#:
#: A kind absent from here gets the strict rule, which is the safe direction -
#: a refusal is loud where a clamp is silent. Distinct from
#: :meth:`model.Port.snaps_to_the_grid`, which asks whether openEMS *moves* a
#: port's planes onto the grid; a lumped port's box is snapped that way and is
#: still laid as a box, so it answers yes there and is absent here.
_REBUILT_FROM_THE_GRID = frozenset({"microstrip"})


def _check_port_inside_the_grid(port: Port, grid: MeshGrid) -> list[Finding]:
    """How much of a port's box has to be on the grid, which is per kind and axis.

    ``_check_grid_covers_the_model`` looks only at solids and at the strip a
    microstrip port lays, so nothing else sees a *port box* at all.

    openEMS never rejects a box for hanging off the grid; it clamps it to the
    edge in silence. What the clamp costs depends on how the port uses its box:

    - ``MSLPort`` rebuilds itself from the lines it lands on along the
      propagation axis, so there the clamp is the operation it was going to
      perform anyway, and its box legitimately overhangs. On the other two axes
      it is laid as a box like anything else, and the voltage probes'
      integration path is the box's excitation-axis extent verbatim.
    - A *lumped* port's box is the gap the element sits across, and its
      conductance is integrated over the snapped index range
      (``operator.cpp``:1654-1673). ``kappa`` is normalised by that, so the
      resistance survives the clamp and the *geometry* does not: the element is
      laid across a shorter path than was drawn, and the run returns a
      plausible reflection for a termination nobody asked for.
    - A *waveguide* port's ``kc`` - and through it ``beta`` and the ``ZL``
      everything is referenced to - comes from the ``a`` and ``b`` the port was
      *told* (``openEMS/ports.py``:437-444), while the excitation is an analytic
      mode profile anchored at the box's own start (:448-458). Clamped, that
      profile is laid over fewer cells than it was written for and no longer
      falls to zero at the wall, so what is launched is not the mode.

    Per axis in every case: a box outside the grid on any single axis has no
    intersection with it, whatever the other two do.

    Landing on a line is a separate question from lying inside the grid, and
    :func:`_check_required_lines_exist` asks it.
    """
    findings = []
    for dim in range(DIMENSIONS):
        rebuilt = dim == port.propagation_axis and port.kind in _REBUILT_FROM_THE_GRID
        lines = grid[dim]
        low, high = float(lines[0]), float(lines[-1])
        start, stop = sorted((port.start[dim], port.stop[dim]))
        span = f"its box spans {AXIS_NAMES[dim]}={start:.4g} to {stop:.4g}"
        runs = f"the grid, which runs {low:.4g} to {high:.4g}"

        if stop < low - _ON_THE_GRID or start > high + _ON_THE_GRID:
            findings.append(
                Finding(
                    REFUSE,
                    port.name,
                    f"{span}, entirely outside {runs}. openEMS clips it away "
                    "without comment, so the port excites nothing and the run "
                    "reports zeros after its full time. Extend the mesh to "
                    "cover the port, or move the port onto the structure",
                )
            )
            continue
        if rebuilt:
            continue

        # Each end named separately, as _check_the_absorber_leaves_a_model does
        # and for the same reason: a box can hang off both, and one number for
        # the pair is not a distance anything can be moved by.
        past = [
            f"{gap:.4g} mm past {AXIS_NAMES[dim]}={end}"
            for end, gap in (("min", low - start), ("max", stop - high))
            if gap > _ON_THE_GRID
        ]
        if past:
            findings.append(
                Finding(
                    REFUSE,
                    port.name,
                    f"{span}, hanging {' and '.join(past)} of {runs}. A "
                    f"{port.kind} port is laid as that box on this axis and "
                    "openEMS clamps a box to the grid edge rather than refusing "
                    "it, so what is built is the part that fits and the run "
                    "returns plausible numbers for geometry that was never "
                    "drawn. Draw the port inside the grid, or - if the grid "
                    "stops short because the absorber ate the domain - raise "
                    "ElementsPerWavelength or lower PMLCells",
                )
            )
    return findings


def _check_required_lines_exist(port: Port, grid: MeshGrid) -> list[Finding]:
    """A plane openEMS will not move needs a grid line on it, and may not have one.

    :meth:`model.Port.required_lines` names those planes, and
    :func:`write.plan_grid` hands them to the mesher as anchors, so on the route
    that meshes they are there by construction. That is not every route: the
    driver is handed a finished envelope with its grid already in it and re-runs
    pre-flight over that, which is the whole reason the guards live here rather
    than beside the mesher. Nothing between a replayed envelope and the solver
    would otherwise look.

    Driven by what the port *declares* rather than by its kind, so a kind that
    grows a pinned plane is covered by saying so once.
    """
    findings = []
    for dim, positions in enumerate(port.required_lines()):
        if not positions:
            continue
        lines = np.asarray(grid[dim], dtype=float)
        for position in positions:
            gaps = np.abs(lines - position)
            if float(gaps.min()) <= _ON_THE_GRID:
                continue
            findings.append(
                Finding(
                    REFUSE,
                    port.name,
                    f"it needs a grid line at {AXIS_NAMES[dim]}={position:.6g} "
                    f"and the nearest is {float(lines[gaps.argmin()]):.6g}. "
                    "openEMS discretises an excitation by walking the grid and "
                    "asking what is at each coordinate, and it does not move "
                    "one onto the grid the way it moves a resistor or a probe - "
                    "so a plane with no line on it drives nothing, and the run "
                    "completes having excited nothing and returning 0/0. "
                    "Re-mesh the model, or move the port onto a line",
                )
            )
    return findings


def _check_port_clear_of_absorber(port: Port, grid: MeshGrid) -> list[Finding]:
    """A probe or feed inside the absorber measures a field that is being eaten.

    The absorber's depth, the shortfall and the end are all named here rather
    than left to :func:`~.absorber._check_the_absorber_leaves_a_model`, which is
    a warning with a threshold and stays silent in exactly the case that
    produces this refusal - and an interior quoted on its own gives the symptom
    with no route to the cause.

    A point past the grid edge is a different fault and is said differently:
    "inside the absorber" is not true of a waveguide plane, which is simply not
    discretised, and the run returns 0/0.
    :func:`_check_port_inside_the_grid` covers neither, testing the port *box*,
    which legitimately overhangs on a THROUGH face.

    What is quoted is how far the *grid* falls short, not how far the plane
    would travel: ``openEMS/ports.py``:257-260 clamps the probe triplet's centre
    into ``[1, len-2]``, so a plane past the edge lands one edge cell beyond it.
    The shortfall is the number a user acts on and is the same for every kind.

    The closing advice holds only where the face was pulled *in*. On an
    outward-padded face the interior wall is the domain wall, so lowering
    PMLCells does not move it. No branch for that: ``model.py`` bounds both
    shifts inside the port box and ``structure_bounds`` includes ports, so on a
    padded axis this check cannot fire.
    """
    dim = port.propagation_axis
    interior = _absorber_bounds(grid, dim)
    if interior is None:
        return []

    low, high = interior
    lines = grid[dim]
    edges = float(lines[0]), float(lines[-1])
    cells = _absorber_cells(grid, dim)
    depths = _absorber_depth(grid, dim)
    axis = AXIS_NAMES[dim]

    findings = []
    positions = {
        "measurement plane": port.measurement_position(),
        "feed": port.start[dim] + port.direction * port.feed_shift,
    }
    for what, position in positions.items():
        if low <= position <= high:
            continue
        face = 0 if position < low else 1
        end = "min" if face == 0 else "max"
        beyond = (edges[0] - position) if face == 0 else (position - edges[1])

        if beyond > 0:
            where = (
                f"outside the grid, which runs {edges[0]:.4g} to {edges[1]:.4g} "
                f"and stops {beyond:.4g} mm short of it"
            )
            why = (
                "openEMS moves it onto the nearest line it has, deep in the "
                "absorber, and reports nothing about the move"
                if port.snaps_to_the_grid()
                else f"openEMS does not move a {port.kind} plane onto the grid: "
                f"with no line there it discretises nothing, and the run "
                f"completes having measured nothing and returns 0/0"
            )
        else:
            inside = (low - position) if face == 0 else (position - high)
            where = f"inside the absorber (the interior runs {low:.4g} to {high:.4g})"
            why = (
                f"the field there is being attenuated on purpose and anything "
                f"measured in it is meaningless. The absorber is {cells} cells "
                f"and {depths[face]:.4g} mm deep at {axis}={end}, so this sits "
                f"{inside:.4g} mm inside it"
            )

        findings.append(
            Finding(
                REFUSE,
                port.name,
                f"its {what} sits at {axis}={position:.4g}, {where}; {why}. "
                f"Move it further in, or raise ElementsPerWavelength or lower "
                f"PMLCells to make the absorber thinner",
            )
        )
    return findings
