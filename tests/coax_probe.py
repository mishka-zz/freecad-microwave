# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Draw a coaxial line under a real CAD kernel and write it as an envelope.

Run by ``test_acceptance_coax``, under ``freecadcmd`` and not under the
interpreter that owns the openEMS bindings - the gate needs a real kernel to
make the geometry and a real solver to solve it, and no interpreter here has
both. The envelope is the handoff, which is what it is for: this side draws
three curved solids, triangulates them, measures the lengths they carry and
plans a grid; the other side reads the file and solves it verbatim.

Every conductor here is a *cylinder*, so nothing in the model fills its
bounding box and nothing reaches openEMS as a box. That is the whole point of
the gate - the impedance of a coaxial line depends on the two radii through
``ln(b/a)``, so a surface the engine reads as the wrong size answers with the
wrong impedance, and the closed form says by how much.

The same line is written at several cell sizes across the annulus, at two
triangulation finenesses, and at each end of those cell sizes at several
alignments against its own grid. Refining the cell says how the discretisation
approaches the drawing, which one answer at one mesh cannot; refining the
triangulation says the drawing is the shape rather than the polygon - chords
across a circle are inscribed, so a coarse triangulation is a *smaller* inner
conductor and a *larger* outer one, both of which push the impedance the same
way. Moving the grid says how much of a difference between two cell sizes is the
cell size, the rest of it being where the cells happened to fall.

Only the mesh moves across the sequence. The line as drawn, the ports as drawn
and the record's length in seconds are the same in every case, because a device
that refines along with its grid leaves nothing to read a rate from. Where the
ports' planes *land* is not the same: openEMS snaps the source and the probe
triplet to the nearest grid lines, so which lines those are moves with the mesh.
That is part of the discretisation being measured rather than part of the
drawing.

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

from Microwave.Solvers.openems import document, geometry, lfs, plan  # noqa: E402
from Microwave.Solvers.openems.model import (  # noqa: E402
    THROUGH,
    Frequency,
    Material,
    Port,
    Problem,
    Solid,
    Termination,
)
from Microwave.Solvers.openems.regions import MeshParams
from tests.coax import (  # noqa: E402
    BAND_TOP,
    CAP,
    DIELECTRIC_RES,
    EPS_R,
    FEED_SHIFT,
    INNER,
    INNER_RADIUS,
    LENGTH,
    MEASUREMENT_SHIFT,
    OUTER_RADIUS,
    POINTS,
    SHIELD,
    SHIELD_WALL,
    SURFACE_FIDELITY,
    annulus_cell,
    cases,
    conductor_res,
    timesteps,
)

#: Above the priority the adapter gives a port, so that where the feeding
#: element reaches into a conductor the metal wins and the element occupies
#: exactly the gap. Below that it would carve its own material out of the
#: conductor it is trying to connect to.
METAL_PRIORITY = 20
DIELECTRIC_PRIORITY = 5


def _shapes():
    """The three solids, as the kernel builds them.

    The shield and the dielectric are each a cylinder with a cylinder taken out
    of it, so both arrive as a compound rather than as a solid - which is the
    ordinary result of a boolean and is why nothing here may read a shape's
    type to decide what it is.
    """
    axis = FreeCAD.Vector(0.0, 0.0, 1.0)
    origin = FreeCAD.Vector(0.0, 0.0, 0.0)

    def tube(inner, outer):
        return Part.makeCylinder(outer, LENGTH, origin, axis).cut(
            Part.makeCylinder(inner, LENGTH, origin, axis)
        )

    return (
        (INNER, "Copper", Part.makeCylinder(INNER_RADIUS, LENGTH, origin, axis)),
        ("Insulator", "PTFE", tube(INNER_RADIUS, OUTER_RADIUS)),
        (SHIELD, "Copper", tube(OUTER_RADIUS, OUTER_RADIUS + SHIELD_WALL)),
    )


def _drawn(doc, label, shape):
    obj = doc.addObject("Part::Feature", label)
    obj.Label = label
    obj.Shape = shape
    return obj


def _geometry(doc, fineness):
    """Triangulate every solid, and measure what the drawing asks of the grid.

    ``DEFLECTION_OF_EXTENT`` is what the translation opens its request at, and
    setting it is how this asks for a coarser or a finer surface. There is no
    parameter for it: the fineness is a property of the layer rather than of a
    call, and a gate that varies it is the only caller that has ever wanted to.
    """
    geometry.DEFLECTION_OF_EXTENT = fineness

    # How far each conductor's emitted surface stands from the drawing, in mm,
    # keyed by the name the envelope carries. Written out beside the envelope so
    # a gate can score what the translation reports against what the solve did
    # with it.
    departures = {}
    solids, bodies = [], []
    for label, material, shape in _shapes():
        obj = _drawn(doc, label, shape)
        metal = material == "Copper"
        for piece in geometry.solid_boxes(obj):
            solid = Solid(
                material=material,
                lower=piece.box.lower,
                upper=piece.box.upper,
                priority=METAL_PRIORITY if metal else DIELECTRIC_PRIORITY,
                label=piece.label,
                vertices=piece.vertices,
                faces=piece.faces,
                sheet_normal=piece.sheet_normal,
                thickened=piece.thickened,
            )
            solids.append(solid)
            # The solid is asked what form it is in, the way the translation
            # asks it, so this cannot come to disagree with what gets emitted.
            bodies.append(document.measured_body(piece, metal))
            if piece.departure is not None and piece.departure.displaced:
                departures[solid.name] = piece.departure.displaced
        doc.removeObject(obj.Name)
    return tuple(solids), tuple(bodies), departures


def _ports():
    """One coaxial port spanning the line, reading it across its own annulus.

    The port lays nothing. Its excitation is a shell carrying the mode's own
    ``1/r`` radial profile, and its probes read the voltage across the annulus
    and the current around the inner conductor - so what it measures is the line
    drawn above it and not an idealisation beside it.

    One port and not two, because the line is made *infinite* rather than
    terminated: it runs out through the absorber at both ends, so there is no
    end to reflect off and nothing for a second port to measure. That is the
    same arrangement the microstrip gate uses, and for the same reason - an
    impedance read off a line with an open end is the line plus the reflection.

    Reported against the port itself rather than against a number: the whole
    quantity under test is the impedance the line turns out to have, and
    renormalising to anything would be reporting it against an answer.
    """
    return (
        Port(
            number=1,
            kind="coaxial",
            # Opposite corners of the bore's bounding box: transverse they are
            # the square around the shield's inner surface, and along the line
            # they are its two ends.
            start=(-OUTER_RADIUS, -OUTER_RADIUS, 0.0),
            stop=(OUTER_RADIUS, OUTER_RADIUS, LENGTH),
            propagation_axis=2,
            inner_radius=INNER_RADIUS,
            excite=True,
            feed_shift=FEED_SHIFT,
            measurement_shift=MEASUREMENT_SHIFT,
            label="Port 1",
        ),
    )


def _aligned(grid, offset):
    """The same mesh, moved against the drawing by ``offset`` cells.

    Every spacing in a grid is a difference, so translating it leaves the mesh
    exactly the mesh it was and changes only whereabouts in a cell each wall
    falls. That is the free variable a single solve per resolution never
    records: two grids of the same cell can sample a curved conductor at
    different points of it and answer differently.

    In cells at the annulus, because that gap is what the impedance is read
    across and what a rate here is fitted against.
    """
    if not any(offset):
        return grid
    return grid.moved((offset[0] * annulus_cell(grid.x), offset[1] * annulus_cell(grid.y), 0.0))


def _problem(doc, fineness, steps, offset) -> tuple[Problem, dict]:
    """The envelope for one case, and how far its conductors' surfaces stand
    from the shapes they were drawn as."""
    solids, bodies, departures = _geometry(doc, fineness)
    materials = (
        Material(name="Copper", kind="pec"),
        Material(name="PTFE", kind="dielectric", epsilon=EPS_R),
    )
    ports = _ports()
    params = MeshParams(
        # The conductor cell is the only one moved across the sequence. The bulk
        # cells come from the wavelength and have nothing to do with the
        # staircase.
        metal_res=conductor_res(steps),
        dielectric_res=DIELECTRIC_RES,
        max_ratio=(1.3, 1.3, 1.3),
        min_lines=9,
        pml_cells=8,
        cap=CAP,
    )
    # Pinned across the sequence, and below what the mesher would choose. A
    # curved wall is followed to a share of its own radius, which no cell size
    # moves; below the floor that share is clamped at, the cell is what binds
    # and one number describes the mesh. See tests.coax.SURFACE_FIDELITY.
    measured = lfs.features(
        bodies,
        params.cap,
        params.metal_res,
        fidelity=SURFACE_FIDELITY,
        min_lines=params.min_lines,
    )
    grid = plan.plan_grid(
        solids,
        ports,
        materials,
        params,
        padding=((8, 8), (8, 8), (THROUGH, THROUGH)),
        measured=measured,
    )
    grid = _aligned(grid, offset)
    problem = Problem(
        title=(
            f"coaxial line, conductor cell one {steps}th of the annulus, "
            f"triangulated at {fineness:g} of its extent, lattice at "
            f"({offset[0]:g}, {offset[1]:g}) cells"
        ),
        frequency=Frequency(start=BAND_TOP / POINTS, stop=BAND_TOP, points=POINTS),
        grid=grid,
        materials=materials,
        solids=solids,
        ports=ports,
        boundary=("PML_8",) * 6,
        termination=Termination(max_timesteps=timesteps(grid), end_criteria=0.0),
    )
    return problem, departures


def main(out):
    os.makedirs(out, exist_ok=True)
    written = []
    for name, (fineness, steps, offset) in sorted(cases().items()):
        doc = FreeCAD.newDocument(f"coax_{name}")
        problem, departures = _problem(doc, fineness, steps, offset)
        grid = problem.grid
        triangles = sum(len(solid.faces) for solid in problem.solids)
        print(
            f"{name}: fineness {fineness:g}, {triangles} triangles, "
            f"cell {conductor_res(steps):.4f} mm, lattice at "
            f"({offset[0]:g}, {offset[1]:g}) cells, {grid.cell_count:,} cells, "
            f"lines {len(grid.x)} x {len(grid.y)} x {len(grid.z)}, "
            f"{problem.termination.max_timesteps} timesteps"
        )
        os.makedirs(os.path.join(out, name), exist_ok=True)
        with open(os.path.join(out, name, "openems.json"), "w") as handle:
            json.dump(problem.to_dict(), handle)
        with open(os.path.join(out, name, "departures.json"), "w") as handle:
            json.dump(departures, handle)
        written.append(name)
        FreeCAD.closeDocument(doc.Name)

    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump({"cases": written}, handle)
    print(f"wrote {len(written)} envelopes to {out}")


# freecadcmd execs a script under a module name taken from the file stem rather
# than "__main__", so a bare guard never fires and the script silently does
# nothing.
if __name__ in ("__main__", "coax_probe"):
    main(os.environ.get("COAX_OUT", "."))
