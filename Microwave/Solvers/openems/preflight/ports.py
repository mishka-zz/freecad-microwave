# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checks about the ports, one port at a time.

:func:`_check_ports` is the loop, and everything below it asks one question of
one port - that the adapter can build its kind, that it survives snapping to
the grid, that a guide is empty and its mode propagates, that its box lies inside
the grid and clear of the absorber, that the lines it needs exist, that its
excitation catches one, and that its element still meets the metal it was drawn
against.

:func:`_check_lumped_excitation_beside_a_measured_line` is the exception, and
sits above the loop: it is about the *combination* of ports in one run, which no
port can answer on its own.
"""

from __future__ import annotations

import numpy as np

from ....units import MM_PER_M
from .. import staircase
from ..capabilities import Capabilities
from ..containment import contains
from ..model import (
    AXIS_NAMES,
    CONDUCTOR_KINDS,
    DIMENSIONS,
    SPEED_OF_LIGHT,
    MeshGrid,
    Port,
    Problem,
)
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
    an untried geometry on the strength of one combination. Saying it
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
        findings += _check_the_excitation_is_sampled(port, problem.grid)
        findings += _check_port_clear_of_absorber(port, problem.grid)
        findings += _check_the_guide_is_empty(port, problem)
        findings += _check_the_mode_propagates(port, problem)
        findings += _check_the_element_survives_snapping(port, problem.grid)
        findings += _check_the_element_meets_its_metal(port, problem)
        findings += _check_the_annulus_is_resolved(port, problem.grid)
    return findings


#: How many grid lines a coaxial port's annulus must carry for its probes to
#: read anything. A probe integrates the field along the edges between the lines
#: its box covers, so two lines is one edge and one edge is the least that is an
#: integral at all. This is the floor at which the port is not broken, not the
#: resolution at which its answer is right - what an under-resolved annulus
#: costs in impedance is the acceptance gate's subject, not a threshold's.
_ANNULUS_LINES = 2


def _check_the_annulus_is_resolved(port: Port, grid: MeshGrid) -> list[Finding]:
    """A coaxial port whose gap falls inside one cell measures nothing.

    Every primitive this port kind places lives in the annulus: three voltage
    probes running radially across it, two current loops just outside the inner
    conductor, and an excitation shell filling it. openEMS discretises each by
    the grid lines its box covers, so an annulus that no line falls inside is a
    port with no field in it - and the run completes, having driven and read
    nothing.

    Both transverse axes, because the current loop spans both while the voltage
    probe runs along one.
    """
    if port.kind != "coaxial":
        return []

    centre = port.bore_centre
    findings = []
    for axis in port.transverse_axes:
        lines = np.asarray(grid[axis], dtype=float)
        low = centre[axis] + port.inner_radius
        high = centre[axis] + port.outer_radius
        inside = int(np.count_nonzero((lines >= low) & (lines <= high)))
        if inside >= _ANNULUS_LINES:
            continue
        findings.append(
            Finding(
                REFUSE,
                port.name,
                f"its annulus runs from radius {port.inner_radius:.4g} to "
                f"{port.outer_radius:.4g}, and only {inside} grid line(s) fall "
                f"across it along {AXIS_NAMES[axis]}. The port's probes and its "
                f"excitation all live in that gap, so the run would drive "
                f"nothing and read nothing. Refine the mesh across "
                f"{AXIS_NAMES[axis]}",
            )
        )
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


def _in_the_metal(problem: Problem, points: np.ndarray) -> np.ndarray:
    """Which points sit in a conductor, as the engine will be given them.

    Given rather than drawn: a curved conductor is handed over grown by half the
    cell it will be sampled on, so the metal that conducts reaches past the
    surface that was drawn, and asking the drawing would answer a question about
    a shape the run never sees. :func:`driver.add_solid` decides the same thing
    the same way, so what is asked here is what is built.

    A port that lays metal of its own lays it as a box, which is the one shape
    nothing rounds.
    """
    conducting = {
        material.name for material in problem.materials if material.kind in CONDUCTOR_KINDS
    }
    lines = tuple(problem.grid[dim] for dim in range(DIMENSIONS))

    held = np.zeros(len(points), dtype=bool)
    for solid in problem.solids:
        if solid.material not in conducting:
            continue
        grown = (
            staircase.grown(solid.vertices, solid.faces, lines)
            if solid.is_mesh and not solid.is_sheet
            else None
        )
        held |= contains(solid, points, vertices=grown)
    for port in problem.ports:
        if port.lays_conductor():
            lower, upper = port.trace_region()
            held |= np.all((points >= lower) & (points <= upper), axis=1)
    return held


def _check_the_element_meets_its_metal(port: Port, problem: Problem) -> list[Finding]:
    """An element drawn against a conductor, and snapped where nothing conducts.

    ``SnapBox2Mesh`` moves the element's box onto the mesh
    (``operator.cpp``:1625) and the element is laid on the edges between the
    lines it lands on (:1680-1701), so its terminals are the nodes at either end
    of that run. The conductor is a separate rasterisation of the same grid:
    ``CalcPEC_Range`` zeroes an edge whose own sample point the shape holds
    (:2029, :2046-2055), and that point sits on the grid lines across the edge
    and at the cell's midpoint along it (``GetYeeCoords``, :182-186).

    So a terminal is bonded to the metal in either of two ways, and needs only
    one:

    - **The node is in the conductor.** That is how a conductor with no
      thickness conducts at all - it holds no midpoint, so the only edges it
      zeroes are the ones tangential to it, on the single line it lies on.
    - **The edge running out of the terminal is zeroed**, which is how a
      conductor with thickness conducts. ``SnapToMeshLine`` returns the first
      line whose dual node is not below the coordinate (:253-282), and that dual
      node is the outward edge's own sample point - so for an end that snapped
      *into* the gap it lands at or past where the end was drawn, on the metal's
      side, and never more than half a cell past it. Half a cell of metal beyond
      the drawing is therefore enough, and a solid has it.

    What is left is the conductor with neither: a plane, on a grid holding no
    line where it lies. The element ends a cell away from it with a live edge in
    between, in series with the resistance the port declares, and the run
    completes.

    What is asked is the *pair*: the end as drawn lies in the metal, and neither
    answer holds once it is snapped. An element meeting no metal as drawn is left
    alone, that being a model a user means - a short element in the middle of a
    cavity is a dipole probe. The fault is the grid moving a terminal off the
    conductor it was drawn against, and only the grid can be asked about that.

    The ends are asked at the box's transverse centre. openEMS lays the element
    across every transverse line pair the snapped box covers (:1682-1684), so
    that one sample stands for all of them and is the one the element's own
    geometry names. An end the grid did not move is not asked at all: it cannot
    have been moved off anything.
    """
    if port.kind != "lumped" or port.excitation_axis is None:
        return []

    axis = port.excitation_axis
    lines = np.asarray(problem.grid[axis], dtype=float)
    drawn = sorted((port.start[axis], port.stop[axis]))
    landed = [int(np.argmin(np.abs(lines - end))) for end in drawn]
    if landed[0] == landed[1]:
        # The element snaps shut, which _check_the_element_survives_snapping
        # says. A box entirely off this axis lands here too, both ends on the
        # edge line, and _check_port_inside_the_grid owns that one.
        return []

    ends = []
    # A gap between two conductors has its metal outward from each end: below
    # the lower one and above the upper.
    for end, index, outward in zip(drawn, landed, (-1, 1)):
        node = float(lines[index])
        # The same slack a line is called present within, so that a plane the
        # mesher pinned and rounded is one the element still ends on.
        if abs(node - end) <= _ON_THE_GRID:
            continue
        beyond = index + outward
        edge = 0.5 * (node + float(lines[beyond])) if 0 <= beyond < len(lines) else node
        ends.append((end, node, edge))
    if not ends:
        return []

    centre = np.asarray([0.5 * (a + b) for a, b in zip(port.start, port.stop)], dtype=float)
    places = list(zip(*ends))  # the ends as drawn, then the nodes, then the edges
    points = np.tile(centre, (len(places) * len(ends), 1))
    points[:, axis] = [value for column in places for value in column]

    answers = _in_the_metal(problem, points).reshape(len(places), len(ends))
    was, at_node, at_edge = answers

    findings = []
    for (end, node, _), before, on_node, on_edge in zip(ends, was, at_node, at_edge):
        if on_node or on_edge or not before:
            continue
        findings.append(
            Finding(
                REFUSE,
                port.name,
                f"its {AXIS_NAMES[axis]}={end:.4g} end is drawn on a conductor "
                f"and snaps to {node:.4g}, where nothing conducts. openEMS makes "
                f"a conductor by zeroing the Yee edges whose own sample point "
                f"the shape holds, and neither the line this end landed on nor "
                f"the edge running out of it into the metal is one of them - so "
                f"the element would be laid with a live edge between it and the "
                f"conductor, in series with the resistance this port declares. "
                f"Re-mesh the model, which pins a line to each of a conductor's "
                f"faces, or give the conductor enough thickness across "
                f"{AXIS_NAMES[axis]} for the grid to hold an edge inside it",
            )
        )
    return findings


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
        # A triangulated solid's corners bound its shape rather than being it,
        # so an overlap between boxes is no evidence that any material is in the
        # guide - a curved part passing beside a port clips its box while its
        # surface stays clear. Refusing on that would refuse a legal model.
        if solid.is_mesh:
            continue
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
                f"{a * MM_PER_M:.4g} x {b * MM_PER_M:.4g} mm",
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
#: A *coaxial* port is here for a stronger reason than a microstrip's: it lays
#: nothing at all. Along the line its box supplies only the origin the two
#: shifts are measured from, and both land on grid lines by ``argmin`` like the
#: microstrip's. Across the line it supplies the bore's centre and radius, which
#: every probe and the excitation shell are built from - so there the box is
#: taken verbatim and the strict rule holds.
#:
#: A kind absent from here gets the strict rule, which is the safe direction -
#: a refusal is loud where a clamp is silent. Distinct from
#: :meth:`model.Port.snaps_to_the_grid`, which asks whether openEMS *moves* a
#: port's planes onto the grid; a lumped port's box is snapped that way and is
#: still laid as a box, so it answers yes there and is absent here.
_REBUILT_FROM_THE_GRID = frozenset({"microstrip", "coaxial"})


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
      plausible reflection for a termination that was not asked for.
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


#: The kinds whose excitation is a box driving a single field component, which
#: is what lets the rule below be read one axis at a time. A waveguide port and
#: a coaxial one drive both transverse components, so each of their axes carries
#: a demand from each - a different statement, and not one this makes. Both
#: place their excitation on a grid line along the propagation axis, through
#: :meth:`model.Port.required_lines` and through :mod:`.coaxial` respectively,
#: and the plane the rest of their primitives occupy is where the annulus check
#: and the guide's own cross-section answer for them.
_EXCITES_ONE_COMPONENT = frozenset({"lumped", "microstrip"})


def _check_the_excitation_is_sampled(port: Port, grid: MeshGrid) -> list[Finding]:
    """A box wide enough to span cells, and placed so it holds no sample at all.

    An excitation is laid by walking the grid and asking the geometry what is at
    each coordinate (``operator_ext_excitation.cpp``:158-166); nothing moves it
    onto the grid the way a resistor and a probe are moved. A box holding no
    coordinate is dropped as ``Unused primitive``, one line in a log of
    thousands, after which the run completes having driven nothing and every
    S-parameter comes back 0/0.

    Which coordinates those are is staggered. A field component is sampled on
    the dual grid along the axis it points down and on the ordinary grid along
    the other two (``operator.cpp``:183-187), so the excitation axis wants a
    cell centre inside the box and the axes either side of it want a line.

    What is tested here is the axis that asks for nothing: one the box has
    *extent* on, where the position a line would have to take is the mesher's to
    choose and so is nothing a port can name in advance. Whether one landed
    inside is a question about the grid, and this is where the grid is.

    Each of the others is left to what already answers for it:

    - A **flat** axis. :meth:`model.Port.required_lines` names the plane, the
      mesher pins it, and :func:`_check_required_lines_exist` says so when a
      replayed envelope arrives without it.
    - A **lumped** port's excitation axis. A box holding no cell centre is one
      whose ends snap to a single line, which
      :func:`_check_the_element_survives_snapping` refuses from the resistor's
      end. That one refuses a shade more - a box whose face lands exactly on a
      cell centre holds it and the element still snaps shut - so the axis is
      covered rather than shared.
    - A **microstrip** port's propagation axis, which openEMS puts on the
      nearest existing line (``openEMS/ports.py``:297-301). That axis alone: the
      two either side of it are the port's own box, copied unchanged.
    """
    if port.kind not in _EXCITES_ONE_COMPONENT or not port.excite:
        return []

    findings = []
    for axis in range(DIMENSIONS):
        low, high = sorted((port.start[axis], port.stop[axis]))
        if high == low:
            continue  # flat: required_lines names the plane, and it is checked above
        if port.kind == "microstrip" and axis == port.propagation_axis:
            continue  # openEMS places this coordinate on the nearest line itself
        if port.kind == "lumped" and axis == port.excitation_axis:
            continue  # the resistor snaps shut on the same boxes, and is refused there

        lines = np.asarray(grid[axis], dtype=float)
        if high < lines[0] or low > lines[-1]:
            continue  # entirely off this axis: _check_port_inside_the_grid owns it

        centres = axis == port.excitation_axis
        samples = 0.5 * (lines[:-1] + lines[1:]) if centres else lines
        # The slack is the mesher's arithmetic, not openEMS' box test, which
        # takes a coordinate on the face exactly (``CSPrimitives.cpp``:69-72).
        if np.any((samples >= low - _ON_THE_GRID) & (samples <= high + _ON_THE_GRID)):
            continue

        what = "cell centre" if centres else "grid line"
        gaps = np.minimum(np.abs(samples - low), np.abs(samples - high))
        findings.append(
            Finding(
                REFUSE,
                port.name,
                f"its excitation box spans {AXIS_NAMES[axis]}={low:.4g} to "
                f"{high:.4g}, and the nearest {what} is at "
                f"{float(samples[gaps.argmin()]):.4g}, outside it. openEMS "
                f"samples the field at {what}s along {AXIS_NAMES[axis]} and "
                "lays an excitation only where its box holds one. Unlike a "
                "resistor or a probe it is moved onto no line of its own, so "
                "this port drives nothing and the run returns 0/0 after taking "
                f"its full time. Refine the mesh across {AXIS_NAMES[axis]}, or "
                "add a mesh refinement region over the port",
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
