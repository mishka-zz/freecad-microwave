# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Draw a pillbox cavity under a real CAD kernel and write it as an envelope.

Run by ``test_pillbox_drawing``, under ``freecadcmd`` and not under the
interpreter that owns the openEMS bindings - drawing this needs a real kernel and
solving it needs a real engine, and no interpreter here has both. The envelope is
the handoff: this side draws two solids, triangulates them and plans a grid; the
other side would read the file and solve it verbatim. Nothing solves it yet, and
:mod:`tests.pillbox` says why.

The can is a cylinder with a shorter cylinder taken out of it, so it arrives as a
compound with an enclosed void, and nothing here reaches openEMS as a box.

It writes ``manifest.json`` naming what it wrote, and one envelope per case. The
manifest is written last, so its existence is what says this finished - see
``tests/conftest.py``'s ``probe_manifest`` for why the exit status is not the
thing to read.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402
import Part  # noqa: E402

from Microwave.Solvers.openems import document, geometry, lfs, plan  # noqa: E402
from Microwave.Solvers.openems.model import (  # noqa: E402
    Frequency,
    Material,
    Port,
    Problem,
    Solid,
    Termination,
)
from Microwave.Solvers.openems.regions import MeshParams
from tests.pillbox import (  # noqa: E402
    BAND,
    CASES,
    EPS_R,
    KAPPA,
    LOSS_MEASURED_AT,
    POINTS,
    PORT_IMPEDANCE,
    PROBE_WIDTH,
    RADIUS,
    SHELL_WALL,
    WALL_PHASE,
    WALL_PHASE_PASSES,
    cell_size,
    timesteps,
    wall_cell,
    wall_phase,
)

#: Above the priority a port gets, so the can wins wherever they meet and the
#: cavity is bounded by metal rather than by whichever was added last.
SHELL_PRIORITY = 20

#: Below it, so the probe's element occupies the gap it is drawn in rather than
#: being overwritten by the fill it sits inside.
FILL_PRIORITY = 5

#: How many cells stand between the can and the edge of the domain. Nothing
#: radiates out of a sealed can, so this is room for the grading to end in
#: rather than space anything is solved in.
PADDING = ((2, 2), (2, 2), (2, 2))

#: Fewest cells the mesher will leave across a dielectric region. Both heights
#: here span more than this at every cell solved, so it never binds - which is
#: what keeps the two heights meshed alike across their cross-section.
MIN_LINES = 9


def _shapes(height):
    """The can and the fill, both cylinders on the ``z`` axis about the origin.

    Centred rather than based at the origin so that the probe sits at the middle
    of the cavity by construction, and so that the two heights differ in nothing
    but their extent.
    """
    axis = FreeCAD.Vector(0.0, 0.0, 1.0)

    def can(radius, span):
        return Part.makeCylinder(radius, span, FreeCAD.Vector(0.0, 0.0, -span / 2.0), axis)

    bore = can(RADIUS, height)
    return (
        ("Shell", "Copper", can(RADIUS + SHELL_WALL, height + 2 * SHELL_WALL).cut(bore)),
        ("Fill", "Vacuum", bore),
    )


def _geometry(doc, height):
    """Triangulate both solids, and measure what the drawing asks of the grid.

    At the fineness the translation uses for everything else. A gate that asked
    for its own would be scoring a surface no user is given.
    """
    solids, bodies = [], []
    for label, material, shape in _shapes(height):
        obj = doc.addObject("Part::Feature", label)
        obj.Label = label
        obj.Shape = shape
        metal = material == "Copper"
        for piece in geometry.solid_boxes(obj):
            solid = Solid(
                material=material,
                lower=piece.box.lower,
                upper=piece.box.upper,
                priority=SHELL_PRIORITY if metal else FILL_PRIORITY,
                label=piece.label,
                vertices=piece.vertices,
                faces=piece.faces,
                sheet_normal=piece.sheet_normal,
                thickened=piece.thickened,
            )
            solids.append(solid)
            # Asked the way the translation asks it, so this cannot come to
            # disagree with what gets emitted.
            bodies.append(document.measured_body(piece, metal))
        doc.removeObject(obj.Name)
    return tuple(solids), tuple(bodies)


def _port(case):
    """A short element along the axis, where the case says to stand it.

    The dominant mode has its electric field along the axis, uniform down it and
    greatest on it, so an element at the centre couples to that mode as strongly
    as anything can. What it does *not* couple to is the rest of the spectrum: a
    mode with azimuthal variation has a node on the axis, and one with an odd
    number of half-waves along it has a node at mid-height. Moving it off both is
    what the spectrum case is.

    **With extent on every axis.** An excitation is not snapped: openEMS walks
    the grid and asks the geometry what is at each coordinate, so a box holding
    no coordinate drives nothing and says so only as ``Unused primitive``. A
    lumped port asks for a grid line on every axis it is flat across, and in the
    middle of a cavity there is no surface for the mesher to pin one against.
    """
    radial, axial = case.probe_at
    # Out along one axis rather than on a diagonal, so that of the pair a
    # circular cavity is degenerate in, the probe stands on the crest of one.
    across, down = radial * RADIUS, axial * case.height
    return Port(
        number=1,
        kind="lumped",
        start=(across - PROBE_WIDTH / 2, -PROBE_WIDTH / 2, down - case.probe / 2),
        stop=(across + PROBE_WIDTH / 2, PROBE_WIDTH / 2, down + case.probe / 2),
        # A lumped port is a circuit element and has no propagation axis of its
        # own - the adapter wants one only to check the port is clear of the
        # absorber - but it must have extent along whichever axis is named.
        propagation_axis=1,
        excitation_axis=2,
        excite=True,
        feed_resistance=PORT_IMPEDANCE,
        reference_impedance=PORT_IMPEDANCE,
        label="Probe",
    )


def _slid(grid, ports, offset):
    """The mesh moved by ``offset`` mm, and the probe carried along with it.

    Every spacing in a grid is a difference, so translating it leaves the mesh
    exactly the mesh it was and changes only whereabouts in a cell a surface
    falls. That is the free variable a single solve per resolution never
    records: two grids of the same cell can sample a curved conductor at
    different points of it and answer differently.

    The probe travels with the mesh so that an offset changes where the lines fall
    against the *wall* rather than what the probe is. openEMS builds a lumped
    element from its box snapped to the grid, so a probe left standing while the
    lines move is rebuilt at whatever size they leave it, and the case would be
    measuring that too.

    It does move the probe off the cavity's axis, by up to a cell and a half once
    the slides are added up, and that displacement differs per mesh. The mode read
    here is stationary at the axis - its field goes as ``J0`` of the radius, whose
    slope at zero is nothing - so the leading term in that displacement is
    quadratic. It is not measured.
    """
    return grid.moved(offset), tuple(port.moved(offset) for port in ports)


def _offset(grid, phase) -> tuple[float, float, float]:
    """How far to slide, in mm, for a ``phase`` given in cells at the wall -
    that being the surface the resonance is a measurement of.

    Per axis, because the mesher does not give a cylinder on the ``z`` axis the
    same grid on ``x`` as on ``y``.
    """
    return (phase[0] * wall_cell(grid.x), phase[1] * wall_cell(grid.y), 0.0)


def _hold_the_wall(grid, ports):
    """Slide until the wall stands at :data:`~tests.pillbox.WALL_PHASE`."""
    for _ in range(WALL_PHASE_PASSES):
        wanted = (wall_phase(grid.x) - WALL_PHASE, wall_phase(grid.y) - WALL_PHASE)
        grid, ports = _slid(grid, ports, _offset(grid, wanted))
    return grid, ports


def _problem(doc, case, name) -> Problem:
    solids, bodies = _geometry(doc, case.height)
    materials = (
        Material(name="Copper", kind="pec"),
        Material(
            name="Vacuum",
            kind="lossy_dielectric",
            epsilon=EPS_R,
            kappa=KAPPA,
            measured_at=LOSS_MEASURED_AT,
        ),
    )
    ports = (_port(case),)
    resolution = cell_size(case.divisor)
    params = MeshParams(
        metal_res=resolution,
        dielectric_res=resolution,
        max_ratio=(1.3, 1.3, 1.3),
        min_lines=MIN_LINES,
        # Nothing radiates out of a sealed can, so there is nothing for an
        # absorber to absorb and the domain stops just outside the metal.
        pml_cells=0,
        cap=resolution,
    )
    grid = plan.plan_grid(
        solids,
        ports,
        materials,
        params,
        padding=PADDING,
        measured=lfs.features(bodies, params.cap, params.metal_res, min_lines=params.min_lines),
    )
    grid, ports = _hold_the_wall(grid, ports)
    if any(case.phase):
        grid, ports = _slid(grid, ports, _offset(grid, case.phase))
    return Problem(
        title=f"pillbox cavity, {name}",
        frequency=Frequency(start=BAND[0], stop=BAND[1], points=POINTS),
        grid=grid,
        materials=materials,
        solids=solids,
        ports=ports,
        boundary=("PEC",) * 6,
        termination=Termination(max_timesteps=timesteps(case.divisor), end_criteria=0.0),
        pinned_clearance=case.clearance,
    )


def main(out):
    os.makedirs(out, exist_ok=True)
    written = []
    for name, case in sorted(CASES.items()):
        doc = FreeCAD.newDocument(f"pillbox_{name}")
        problem = _problem(doc, case, name)
        grid = problem.grid
        print(
            f"{name}: {sum(len(s.faces) for s in problem.solids)} triangles, "
            f"wall cell {wall_cell(grid.x):.4f} mm, {grid.cell_count:,} cells, "
            f"lines {len(grid.x)} x {len(grid.y)} x {len(grid.z)}, "
            f"{problem.termination.max_timesteps} timesteps"
        )
        os.makedirs(os.path.join(out, name), exist_ok=True)
        with open(os.path.join(out, name, "openems.json"), "w") as handle:
            json.dump(problem.to_dict(), handle)
        written.append(name)
        FreeCAD.closeDocument(doc.Name)

    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump({"cases": written}, handle)
    print(f"wrote {len(written)} envelopes to {out}")


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "pillbox_probe"):
    main(os.environ.get("PILLBOX_OUT", "."))
