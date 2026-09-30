# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Build ``waveguide_wr42.FCStd``: a hollow WR-42 guide, as a real document.

The guide ``tests/test_acceptance_openems_waveguide.py`` builds by hand -
WR-42 over 20-26 GHz, a band single-mode by construction - drawn with ``Part``
primitives and marked up with the document objects a user would use. The
dimensions are that gate's, so the cutoff and phase constant it is scored
against hold here unchanged. The port planes are placed at a length rather
than at a count of cells, so the separation between them is this document's
own.

It is the enclosed case, and it is drawn the way an enclosed problem has to be:
the solid is the air inside the pipe, the four guide walls are boundary
conditions rather than metal somebody drew, and the two ends are where the wave
leaves. Nothing here is a transmission line over a ground plane, so none of the
other examples' markup applies.

**The ports do not sit on the guide's end faces.** Those ends run out through
the absorber, so a plane on one of them measures a field being attenuated on
purpose, and pre-flight refuses it by name. What a port is pointed at here is a
plane drawn inside the air, which is what a reference plane is. The cost is two
warnings the run cannot resolve: a plane has structure on both sides, so
nothing in the drawing says which way the port faces, and each is set here and
checked by nobody. Reversed, the run finishes and reports a plausible matrix
with the phase inverted.

Run it with FreeCAD's own interpreter, which is not the one that owns the
openEMS bindings::

    freecadcmd examples/waveguide_wr42.py

``freecadcmd`` ships inside the FreeCAD installation. On macOS it is
``FreeCAD.app/Contents/Resources/bin/freecadcmd``.

It writes beside itself. Set ``OUT`` to put the document somewhere else.
"""

import os
import sys

import FreeCAD

from Microwave.Objects.analysis import createEMAnalysis, solver_of
from Microwave.Objects.materials import createEMMaterial, createEMMaterialBinding
from Microwave.Objects.ports import PORT_IMPEDANCE, createEMPortRectWaveguide
from Microwave.Solvers.openems import document, preflight

# Beside this script, so the shared stamp can be imported. freecadcmd runs a
# script without putting its directory on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _visibility import stamp_visibility  # noqa: E402

# WR-42, in mm. The standard band is 18 to 26.5 GHz; TE10 cuts off at 14.0 GHz
# and TE20 at 28.0 GHz, so 20 to 26 GHz carries one mode and no other.
BROAD = 10.7
NARROW = 4.3
LENGTH = 50.0

FREQ_START = "20 GHz"
FREQ_STOP = "26 GHz"

#: The gate meshes to a thirtieth of the free-space wavelength at the top of the
#: band. Free space rather than guide wavelength: the guide wavelength is the
#: longer of the two, so this is the conservative choice.
ELEMENTS_PER_WAVELENGTH = 30

MODE = "TE10"

#: How far in from each end the port planes stand, in mm. They cannot sit on
#: the guide's end faces: those ends run out through the absorber, and a plane
#: inside it measures a field being attenuated on purpose. This clears the
#: absorber at the resolution above with room to spare, and pre-flight says by
#: how much if either ever stops being true.
PORT_INSET = 5.0


def geometry(doc):
    """The air inside the pipe, and a cross-section plane at each port.

    A hollow guide has no conductor to draw. Its walls are perfect and are
    stated as boundary conditions, so drawing them as metal would put a second
    description of the same wall in the document.

    The two planes are drawn because a port has to be pointed at a face and
    the guide's own end faces are unusable - they run out through the absorber.
    A plane standing inside the air is what a reference plane is, and drawing
    it puts a grid line on it.
    """
    air = doc.addObject("Part::Box", "GuideAir")
    air.Length, air.Width, air.Height = LENGTH, BROAD, NARROW

    planes = []
    for name, at in (("PlanePort1", PORT_INSET), ("PlanePort2", LENGTH - PORT_INSET)):
        plane = doc.addObject("Part::Plane", name)
        plane.Length, plane.Width = BROAD, NARROW
        # A Part::Plane is drawn in its own xy, so it is turned to stand across
        # the guide: its length onto y, its width onto z.
        plane.Placement = FreeCAD.Placement(
            FreeCAD.Vector(at, 0.0, 0.0),
            FreeCAD.Rotation(FreeCAD.Vector(0, 0, 1), 90).multiply(
                FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), 90)
            ),
        )
        planes.append(plane)

    doc.recompute()

    # The placement above is arithmetic nobody re-derives, and a plane turned
    # the wrong way is a port box hanging outside the grid. Ask the drawing.
    for plane in planes:
        box = plane.Shape.BoundBox
        spans = (
            abs(box.XLength) < 1e-6,
            abs(box.YLength - BROAD) < 1e-6,
            abs(box.ZLength - NARROW) < 1e-6,
        )
        if not all(spans):
            raise RuntimeError(f"{plane.Name} does not span the guide's cross-section")

    return air, planes


def markup(analysis, air, planes):
    """The air the guide encloses, and a port on each end face facing inward.

    The air is bound to a material even though it is vacuum. A binding is how
    a drawn solid enters the problem at all - the translation gathers solids
    through them and through nothing else - so an unbound body is a body the
    run does not have.
    """
    vacuum = createEMMaterial("Vacuum")
    vacuum.Label = "Vacuum"
    vacuum.MaterialType = "Dielectric"
    vacuum.Permittivity = 1.0

    binding = createEMMaterialBinding("GuideAirBinding")
    binding.Label = "GuideAirBinding"
    binding.Material = vacuum
    # NOT [(air, [])]. FreeCAD's PropertyLinkSubList drops any entry whose
    # sub-element list is empty, so that form reads back as [] and the binding
    # silently disappears. [""] is how "the whole solid" is spelt.
    binding.References = [(air, [""])]
    analysis.addObject(binding)

    ports = []
    for number, (plane, axis) in enumerate(zip(planes, ("X", "-X")), start=1):
        port = createEMPortRectWaveguide(f"Port{number}")
        port.Label = f"Port{number}"
        port.Number = number
        # Both ports are driven, one run each, so the whole matrix is measured
        # rather than half of it assumed by reciprocity.
        port.Excitation = True
        port.Mode = MODE
        port.PropagationAxis = axis
        port.CrossSection = (plane, ["Face1"])
        # The S-parameters are referred to the plane the port stands on.
        port.ReferenceDepth = 0.0
        # A guide's S-parameters are reported against its own wave impedance.
        # A fixed fifty ohms would be a number about a cable somewhere else.
        port.ReferencedTo = PORT_IMPEDANCE
        analysis.addObject(port)
        ports.append(port)
    return ports


def study(doc):
    """The analysis, its solver and its mesh policy, all settings applied."""
    analysis = createEMAnalysis(doc)
    analysis.Label = "WR-42 waveguide"
    analysis.FrequencyStart = FREQ_START
    analysis.FrequencyStop = FREQ_STOP
    analysis.NumFrequencyPoints = 201

    solver_of(analysis).MaxTimesteps = 12000

    found = document.contents(analysis)
    settings, grid = found.settings, found.recipe
    grid.ElementsPerWavelength = ELEMENTS_PER_WAVELENGTH
    # The guide runs out through the absorber at both ends, and the domain ends
    # on the drawing everywhere else, where each face is a perfect wall. The four
    # walls are the whole of what makes this an enclosed problem, and it is why
    # the domain must not grow sideways: air outside a wall moves the wall, and
    # the cutoff moves with it.
    settings.PaddingXMin = "Through"
    settings.PaddingXMax = "Through"
    for axis, side in (("Y", "Min"), ("Y", "Max"), ("Z", "Min"), ("Z", "Max")):
        setattr(settings, f"Padding{axis}{side}", "Ends")
    return analysis


def main(out):
    doc = FreeCAD.newDocument("waveguide_wr42")
    air, planes = geometry(doc)
    analysis = study(doc)
    markup(analysis, air, planes)
    doc.recompute()

    problem = document.problem(analysis)
    grid = problem.grid
    print(f"cells: {grid.cell_count:,}  lines: {len(grid.x)} x {len(grid.y)} x {len(grid.z)}")
    print(f"digest: {problem.digest()[:16]}")
    for finding in preflight.check(problem):
        print(f"  {finding}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    doc.saveAs(out)
    # Every object, not only the solids. Anything the stamp does not name comes
    # back hidden - which greys it out in the tree and, for a port, means its
    # arrow never draws.
    stamp_visibility(out, [obj.Name for obj in doc.Objects])
    print(f"saved: {out}")


# freecadcmd execs a script under a module name taken from the file stem, not
# "__main__", so the usual guard alone never fires and the script does nothing.
if __name__ in ("__main__", "waveguide_wr42"):
    # Beside this script, not in the working directory. A bare relative name
    # writes a second copy wherever the documented command was run from.
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waveguide_wr42.FCStd")
    main(os.environ.get("OUT", default))
