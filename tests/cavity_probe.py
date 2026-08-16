# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Draw a spherical cavity under a real CAD kernel and write it as an envelope.

Run by ``test_acceptance_cavity``, under ``freecadcmd`` and not under the
interpreter that owns the openEMS bindings - the gate needs a real kernel to
make the geometry and a real solver to solve it, and no interpreter here has
both. The envelope is the handoff: this side draws two curved solids,
triangulates them and plans a grid; the other side reads the file and solves it
verbatim.

Nothing here fills its bounding box and nothing reaches openEMS as a box.

It writes ``manifest.json`` naming what it wrote, and one envelope per case. The
gate judges the run by that manifest: ``freecadcmd`` segfaults in Qt's teardown
after everything has been written, so its exit status says nothing.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402
import Part  # noqa: E402

from Microwave.Solvers.openems import geometry, lfs, write  # noqa: E402
from Microwave.Solvers.openems.mesh import MeshParams  # noqa: E402
from Microwave.Solvers.openems.model import (  # noqa: E402
    Frequency,
    Material,
    Port,
    Problem,
    Solid,
    Termination,
)
from tests.cavity import (  # noqa: E402
    BAND,
    DIVISORS,
    EPS_R,
    KAPPA,
    LOSS_MEASURED_AT,
    POINTS,
    POLE_AXES,
    PORT_IMPEDANCE,
    PROBE_LENGTH,
    PROBE_WIDTH,
    RADIUS,
    SHELL_WALL,
    cell_size,
    timesteps,
)

#: Above the priority a port gets, so the shell wins wherever they meet and the
#: cavity is bounded by metal rather than by whichever was added last.
SHELL_PRIORITY = 20

#: Below it, so the probe's element occupies the gap it is drawn in rather than
#: being overwritten by the fill it sits inside.
FILL_PRIORITY = 5


def _shapes(pole):
    """The shell and the fill, with the sphere's poles turned onto ``pole``.

    A sphere is the same shape whichever way its parameterisation runs, and the
    triangulation is not: the seam and the poles carry the densest vertices, so
    turning them puts a different polyhedron on the same surface and a different
    staircase on the same grid.
    """
    centre = FreeCAD.Vector(0.0, 0.0, 0.0)
    turn = FreeCAD.Rotation(FreeCAD.Vector(0.0, 0.0, 1.0), FreeCAD.Vector(*pole))

    def ball(radius):
        shape = Part.makeSphere(radius, centre)
        shape.Placement = FreeCAD.Placement(centre, turn)
        return shape

    bore = ball(RADIUS)
    return (("Shell", "Copper", ball(RADIUS + SHELL_WALL).cut(bore)), ("Fill", "Vacuum", bore))


def _geometry(doc, pole):
    solids, bodies = [], []
    for label, material, shape in _shapes(pole):
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
            bodies.append(lfs.Body(piece.label, piece.shape, metal, solid.is_mesh, solid.is_sheet))
        doc.removeObject(obj.Name)
    return tuple(solids), tuple(bodies)


def _port():
    """A short element at the centre, along the axis the dominant mode's field is.

    TM(1,0,1) is a dipole pattern with its electric field through the middle, so
    an element there couples to it and to nothing that has a node there.

    **Flat on one axis only.** An excitation is not snapped: openEMS walks the
    grid and asks the geometry what is at each coordinate, so a box holding no
    coordinate drives nothing and says so only as ``Unused primitive``. A lumped
    port asks for a grid line on every axis it is flat across, and inside a
    sphere there is no surface for the mesher to pin one against.
    """
    return Port(
        number=1,
        kind="lumped",
        start=(-PROBE_WIDTH / 2, 0.0, -PROBE_LENGTH / 2),
        stop=(PROBE_WIDTH / 2, PROBE_WIDTH, PROBE_LENGTH / 2),
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


def _problem(doc, divisor, pole, name) -> Problem:
    solids, bodies = _geometry(doc, pole)
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
    ports = (_port(),)
    resolution = cell_size(divisor)
    params = MeshParams(
        metal_res=resolution,
        dielectric_res=resolution,
        max_ratio=(1.3, 1.3, 1.3),
        min_lines=9,
        # Nothing radiates out of a sealed shell, so there is nothing for an
        # absorber to absorb and the domain stops just outside the metal.
        pml_cells=0,
        cap=resolution,
    )
    grid = write.plan_grid(
        solids,
        ports,
        materials,
        params,
        padding=((2, 2), (2, 2), (2, 2)),
        measured=lfs.features(bodies, params.cap, params.metal_res, min_lines=params.min_lines),
    )
    return Problem(
        title=f"spherical cavity, {name}",
        frequency=Frequency(start=BAND[0], stop=BAND[1], points=POINTS),
        grid=grid,
        materials=materials,
        solids=solids,
        ports=ports,
        boundary=("PEC",) * 6,
        termination=Termination(max_timesteps=timesteps(divisor), end_criteria=0.0),
    )


def cases():
    """Every case the gate solves, as ``name -> (divisor, pole axis)``.

    The sequence the extrapolation runs over is one orientation at each cell
    size. The turned sphere is solved at the coarsest of them, where the
    staircase is largest and so a triangulation the answer depended on would
    show up most.
    """
    found = {f"upright-{divisor}": (divisor, POLE_AXES["upright"]) for divisor in DIVISORS}
    found[f"turned-{min(DIVISORS)}"] = (min(DIVISORS), POLE_AXES["turned"])
    return found


def main(out):
    os.makedirs(out, exist_ok=True)
    written = []
    for name, (divisor, pole) in sorted(cases().items()):
        doc = FreeCAD.newDocument(f"cavity_{name}")
        problem = _problem(doc, divisor, pole, name)
        grid = problem.grid
        print(
            f"{name}: {sum(len(s.faces) for s in problem.solids)} triangles, "
            f"{grid.cell_count:,} cells, "
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
if __name__ in ("__main__", "cavity_probe"):
    main(os.environ.get("CAVITY_OUT", "."))
