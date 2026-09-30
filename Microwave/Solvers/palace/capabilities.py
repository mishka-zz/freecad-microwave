# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What this adapter can and cannot express.

Pure data, and the standard library only: the workbench has to say which solvers
could answer a model on a machine where no engine is installed, so a declaration
that needs its engine to be read cannot help choose one.

The list is kept to what the adapter has been driven against. A name here that
nothing has run turns a refusal a user can act on into a failure they cannot.
"""

from __future__ import annotations

from ..capabilities import Capabilities

#: Every frequency in a Palace configuration is in gigahertz, and every
#: frequency in this workbench is in Hertz.
HZ_PER_GHZ = 1.0e9

#: ``Model.L0``, the metres one unit of the mesh stands for. The mesh is written
#: in the units the shapes were drawn in, and this workbench draws in
#: millimetres.
METRES_PER_UNIT = 1.0e-3


def capabilities() -> Capabilities:
    """This adapter's declaration."""
    return Capabilities(
        solver="Palace",
        port_types=frozenset({"lumped", "rect_waveguide"}),
        materials=frozenset({"dielectric", "lossy_dielectric", "pec", "conducting_sheet"}),
        domains=frozenset({"enclosed", "open"}),
        outputs=frozenset({"s_parameters"}),
        excitations=frozenset({"port"}),
        geometry=frozenset({"arbitrary solid", "arbitrary sheet"}),
        notes={
            "conforming_mesh": "the mesh follows the drawn surface rather than "
            "squaring it onto a grid, so a curve costs no staircase - and the "
            "element order the geometry is written at decides how closely, "
            "which is a solver setting rather than anything a drawing states",
            "solves_per_frequency": "a frequency-domain method answers one "
            "frequency per full solve. An adaptive sweep solves a few in full and "
            "answers every point from a reduced model built of them. Where the "
            "points are many the cost is those solves, and at each point a solve "
            "of each wave port's mode on its face and the model's own small one. A "
            "broadband answer from a single pulse is the other method's, not "
            "this one's",
            "no_absorbing_layer": "there is no perfectly matched layer. A "
            "problem open to free space is bounded by a surface carrying a "
            "first-order absorbing condition. The workbench reserves air round "
            "everything bound to a material, Clearance deep on each face Padding "
            "says Air and flush on every other, and the condition stands on the "
            "open sides of it",
            "wave_port_beside_open": "a study open to free space is driven through "
            "lumped ports. A wave port beside the absorbing surface is refused by "
            "name: it is not offered yet",
            "wall_is_a_condition": "metal is a condition on a face. A sheet "
            "bound to a perfect conductor carries it where it was drawn, on the "
            "boundary of the region or inside it, and every face of the boundary "
            "nothing else was drawn on carries it too. A body bound to a perfect "
            "conductor leaves the region, and the faces it leaves carry the "
            "condition. A body of finite conductivity is refused",
            "sheet_is_two_faces": "a conducting sheet is the surface impedance of "
            "its metal at each frequency, from the skin depth its conductivity and "
            "permeability give there, and its thickness where that is thin enough "
            "to matter. Inside the region each of its two faces carries it and "
            "nothing passes between them: a metal that stops the field, where the "
            "other backend takes the current through the sheet as a whole",
            "metal_closes_a_part_off": "no field crosses a face carrying metal, so "
            "a skin of it closed round a space cuts that space out of the problem "
            "the rest of the region poses. A part no port stands on is driven by "
            "nothing and is refused by name, so a post drawn as the surface of a "
            "body is refused where a cavity with a port in it is solved",
            "port_ends_the_model": "a wave port is measured on a face where the "
            "model ends. A port on a plane drawn across the region leaves out "
            "what stands behind it, on the side its PropagationAxis points away "
            "from, so a guide drawn running on past its port planes is solved "
            "between them",
            "lumped_port_is_a_face": "a lumped port is a resistor laid on one flat "
            "rectangle for each face its reference names, where the source faces "
            "that face across the gap, and the rectangles stand in parallel. The "
            "rest of a face a rectangle lies in, where the model ends, is a "
            "magnetic wall rather than the metal wall, which would short it. A "
            "study drives its ports through one kind or the other",
        },
    )
