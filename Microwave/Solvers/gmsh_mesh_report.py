# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a Gmsh mesh came out as, said in the units it was asked for.

A caller states an element size as a length and gets a file, so without this
the length reaches the mesher and is answered nowhere.

Here rather than under an adapter, for the reason :mod:`.gmsh_meshing`
gives: the mesher belongs to no backend, and a second one reading a Gmsh
mesh would otherwise import the first to say these lines.

Nothing here refuses. A size asked for is a target at a point rather than a
bound on an element, so elements above it are the ordinary case; an element of
poor shape is a fact about the drawing as often as about the mesher; and a piece
two labels were drawn over is what an ordinary drawing does. Each is reported,
and whoever reads it judges it.
"""

from __future__ import annotations

from ..Gmsh.vocabulary import AtRim, Demand, Mesh, Place, Within, transition

__all__ = ["describe"]


def describe(mesh: Mesh, demand: Demand) -> list[str]:
    """The mesh's own answers, one line each: its size, the size of each label's
    elements, what each place holds, its shape, its overlaps, what it left out,
    and the parts a dividing label cut it into.

    The parts are said only where there is more than one: a model that stayed
    whole under the cuts is the ordinary answer, and saying so on every mesh
    would bury the line that matters.
    """
    asked = f"asked for elements around {demand.coarsest:.4g} mm"
    if demand.finest:
        asked += f", and none below {demand.finest:.4g} mm"
    lines = [
        f"Mesh: elements from {mesh.edges.shortest:.4g} mm to {mesh.edges.longest:.4g} mm, {asked}",
        _filling(mesh),
    ]
    for name in sorted(mesh.labels):
        edges = mesh.labels[name].edges
        if edges is not None:
            lines.append(
                f"Mesh: {name!r} elements from {edges.shortest:.4g} mm to {edges.longest:.4g} mm"
            )
    for place in demand.places:
        if place.name in mesh.reached:
            lines.append(_place(place, mesh, demand))
    for dimension in sorted(mesh.worst_quality):
        lines.append(
            f"Mesh: worst element quality at dimension {dimension} is "
            f"{mesh.worst_quality[dimension]:.4g}"
        )
    for one in mesh.settled:
        lines.append(
            f"Mesh: {one.took!r} took a piece also drawn over by "
            f"{', '.join(repr(name) for name in one.gave_up)}, at {one.place}"
        )
    for gone in mesh.left_out:
        lines.append(
            f"Mesh: left out what stands behind {gone.behind!r}, at {gone.place}, drawn "
            f"as {', '.join(repr(name) for name in gone.held)}"
        )
    for cut in mesh.trimmed:
        lines.append(
            f"Mesh: the part of {cut.label!r} inside "
            f"{', '.join(repr(name) for name in cut.by)}, at {cut.place}, leaves the model "
            f"with it"
        )
    if len(mesh.parted) > 1:
        lines.append("Mesh: no field crosses from one part of the model to another. Each part:")
        for part in mesh.parted:
            lines.append(
                f"Mesh: a part at {part.place}, on {', '.join(repr(name) for name in part.labels)}"
            )
    return lines


def _filling(mesh: Mesh) -> str:
    """How many elements fill the model, and how many of them each label holds."""
    top = _filled(mesh)
    held = ", ".join(
        f"{name!r} {label.elements} ({_share(label.elements, mesh)})"
        for name, label in sorted(mesh.labels.items())
        if label.dimension == top
    )
    return f"Mesh: {mesh.elements} elements fill the model: {held}"


def _share(count: int, mesh: Mesh) -> str:
    """A count of elements as a share of the model's."""
    return f"{100.0 * count / mesh.elements:.3g}%" if mesh.elements else "none"


#: What a place's extent is said as, by the dimension it stands on.
EXTENT = {1: "along its {0:.4g} mm", 2: "over its {0:.4g} square mm"}


def _place(place: Place, mesh: Mesh, demand: Demand) -> str:
    """What one place holds, beside the size it asked for.

    The figure set against the size asked for comes first: along a place of
    curves or surfaces the median longest edge of the elements lying on it, and
    the longest of them, which is one element's out of every one there. Beside
    it the elements touching the place, which are larger than those along it and
    larger the steeper the growth, and their share of the model. That share is
    the layer of elements on the place and not what the place costs: the growth
    away from it adds elements past that layer, and an element two places touch
    is counted at each. What a place costs is the difference between the mesh
    made with it and the mesh made without it.
    """
    reached = mesh.reached[place.name]
    if not reached.laid and isinstance(place, AtRim) and place.reentrant:
        if not reached.left:
            return (
                f"Mesh: at {place.name!r} asked {place.size:.4g} mm: nothing is laid, since "
                f"{place.label!r} holds no curve but a seam"
            )
        return (
            f"Mesh: at {place.name!r} asked {place.size:.4g} mm: nothing is laid, since the "
            f"room turns round no curve of {place.label!r} by more than half a turn: "
            f"{reached.left} in all"
        )
    if not reached.laid:
        if isinstance(place, AtRim):
            return (
                f"Mesh: at {place.name!r} asked {place.size:.4g} mm: nothing is laid, since "
                f"{place.label!r} is closed on itself and has no rim"
            )
        return (
            f"Mesh: at {place.name!r} asked {place.size:.4g} mm: nothing is laid, since "
            f"{place.label!r} stands outside the model"
        )
    if isinstance(place, Within):
        opening = f"Mesh: throughout {place.name!r} asked {place.size:.4g} mm"
    else:
        ramp = transition(demand.coarsest, place.size, demand.growth)
        opening = (
            f"Mesh: at {place.name!r} asked {place.size:.4g} mm, growing to "
            f"{demand.coarsest:.4g} mm over {ramp:.4g} mm"
        )
    if reached.standing is None:
        return f"{opening}: no element of the dimension filled is at it"
    share = _share(reached.elements, mesh)
    if reached.dimension == _filled(mesh):
        return (
            f"{opening}: the {reached.elements} elements in it are {share} of the model, "
            f"their median mean edge is {reached.reached:.4g} mm and their median longest "
            f"edge {reached.standing:.4g} mm"
        )
    standing = (
        f"the {reached.elements} elements touching it are {share} of the model, and "
        f"their median longest edge is {reached.standing:.4g} mm"
    )
    if reached.along is None:
        if reached.reached is None:
            return f"{opening}: {standing}"
        return (
            f"{opening}: the longest edge of the elements at it is {reached.reached:.4g} mm; "
            f"{standing}"
        )
    where = EXTENT[reached.dimension].format(reached.extent) if reached.extent else ""
    line = (
        f"{opening}: {where + ' ' if where else ''}the median longest edge of the elements "
        f"lying on it is {reached.along:.4g} mm and the longest {reached.reached:.4g} mm; "
        f"{standing}"
    )
    if reached.left:
        line += (
            f". Nothing is laid at the curves of {place.label!r} the room turns round by "
            f"no more than half a turn: {reached.left} in all"
        )
    return line


def _filled(mesh: Mesh) -> int:
    """The dimension the mesh fills, which is the highest any label stands at."""
    return max(label.dimension for label in mesh.labels.values())
