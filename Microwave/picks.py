# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the user picked, beyond the box it reduces to.

A port is built from bounding boxes. A bounding box does not report whether the
pick encloses an area, nor which side of the pick the body is on. The end edge
of a trace running off a grid axis has a box with depth on both axes across the
port, and so does a small pad: the boxes are the same six numbers and the ports
are not. :func:`Microwave.portbox.lumped` is given the answer rather than
guessing it.

This module sits beside :mod:`Microwave.portbox` rather than under ``Objects``
for the same mechanical reason as :mod:`Microwave.annulus`: ``Objects/__init__``
imports FreeCAD, and the openEMS adapter asks this about the same pick and has
to run without it. One module keeps the drawing and the envelope from answering
differently about one port.

Nothing here imports FreeCAD. It asks a shape for ``getElement``, ``Faces``,
``Edges``, ``Vertexes``, ``Solids``, ``Volume``, ``common`` and ``translated``.
``translated`` takes a coordinate triple as readily as a vector. Measured under
1.1.1: an edge answers no faces, and a face answers itself.
"""

from __future__ import annotations

from typing import Any

from .portbox import DIMENSIONS, KERNEL_TOLERANCE


def named(link: Any) -> list[str]:
    """The sub-element names a ``PropertyLinkSub`` carries, or ``[""]``.

    ``(object, ['Face3'])`` is the ordinary form, ``(object, [''])`` is the
    whole shape, and a bare object is what a link with nothing selected under it
    reduces to. The return is one empty name rather than an empty list, so a
    caller loops the same way over either.
    """
    _, sub = link if isinstance(link, tuple) else (link, None)
    names = [sub] if isinstance(sub, str) else list(sub or ())
    return [name for name in names if name] or [""]


def body(link: Any) -> Any:
    """The whole shape a pick was made on, or ``None`` where it has none."""
    obj = link[0] if isinstance(link, tuple) else link
    return getattr(obj, "Shape", None)


def shapes(link: Any) -> list[Any]:
    """Every shape a ``PropertyLinkSub`` names, or the whole one."""
    shape = body(link)
    if shape is None:
        return []
    return [shape if not name else shape.getElement(name) for name in named(link)]


def is_outline(link: Any) -> bool:
    """Whether everything a pick names encloses no area.

    Every name is read, and not the first alone. A pick naming an edge beside a
    face is not an outline, and reading one name would let the drawing and the
    envelope reach different answers about the same port.

    The question is asked of the shape's contents rather than of ``ShapeType``,
    which names how a shape is built and not what it covers. A pick that
    resolves to nothing is not an outline. The caller has no shape to build from
    either, and reports that first.
    """
    picked = shapes(link)
    return bool(picked) and not any(shape.Faces for shape in picked)


#: How far the pick is stepped to find out which side of it the body is on, in
#: mm. The step stands clear enough of
#: :data:`~Microwave.portbox.KERNEL_TOLERANCE` that a step out reads as out and
#: a step in reads as in. A boolean resolves anything closer than the kernel's
#: own tolerance as touching, so a pick moved by one tolerance meets the body on
#: both sides of itself, and where the surface beside the pick is oblique - a
#: horn's throat, a cone's mouth - it meets the body on neither side. The step
#: is far below the thinnest metal drawn in practice, so a step in stays inside
#: the material it was sent to find. It settles a direction and measures
#: nothing, so it is a length of its own rather than the tolerance restated.
PROBE = 10.0 * KERNEL_TOLERANCE


def _meets(whole: Any, picked: Any, axis: int, sign: int) -> bool | None:
    """Whether the body holds anything where the pick has been stepped to.

    The answer is ``None`` where the kernel would not answer at all, which is
    one of the cases where the shape settles nothing. Both exceptions are the
    kernel's own: a boolean that fails raises ``Part.OCCError``, which derives
    from ``RuntimeError``, and a shape with nothing in it - an object whose
    recompute failed - raises ``ValueError`` before the boolean is reached.
    """
    step = [0.0] * DIMENSIONS
    step[axis] = sign * PROBE
    try:
        met = whole.common(picked.translated(tuple(step)))
    except (RuntimeError, ValueError):
        return None
    return bool(met.Faces or met.Edges or met.Vertexes)


def inward(link: Any, axis: int) -> int | None:
    """Which way along ``axis`` the body lies from the pick, or ``None``.

    This is the sign a port launches along. A port is built from its
    cross-section into the structure, never out of it into the absorber. A
    launch turned around solves cleanly with the phase inverted.

    The question is asked of the material beside the pick rather than of the box
    around it. A conductor folded back on itself puts the centre of its own box
    past the end of a shorter arm, so the box describes the whole shape while
    the wave goes into whatever is behind the one face it starts on.

    The pick itself is stepped either way and met with the body. An empty
    meeting is air, and anything at all is material. Stepping the pick rather
    than probing a point in it lets one rule serve a solid's end face, a sheet's
    end edge and a coaxial annulus alike. The annulus is the case whose own box
    centre lies in the hole, in air, on neither side of anything.

    The answer is ``None`` wherever it would be a guess, and the caller decides
    what that is worth. A pick with material on both sides of it gets ``None``,
    and so does one with material on neither. So do two names pointing opposite
    ways. So does a body wound inside out: the kernel reads it as the complement
    of the space it appears to occupy, so every meeting comes back the wrong way
    round.

    A body drawn as a shell answers only sometimes, and the wall beside the pick
    decides whether it does. A boolean keeps a meeting only where one of the two
    shapes has volume, and a shell has none, so the only meeting left is the
    moved pick landing back on the surface it came from. That happens where the
    surface runs parallel to the step. A pipe's end ring answers. The same pipe
    flared into a horn does not, and neither does a shell picked on a face.
    ``Shape.section`` is not a way round that. It asks whether the body's skin
    crosses the moved pick, which is a different answer the moment a wall
    slopes.

    The inside-out reading is taken off the solids and never off the whole
    shape, because a shape's volume is not a sum over what it holds: measured
    under 1.1.1, a compound of a zero-thickness face and a solid beside it
    reports a negative volume with every solid in it wound the right way.
    """
    whole = body(link)
    if whole is None or any(float(solid.Volume) < 0.0 for solid in whole.Solids):
        return None
    sides: set[int | None] = set()
    for picked in shapes(link):
        plus = _meets(whole, picked, axis, 1)
        minus = _meets(whole, picked, axis, -1)
        if plus is None or minus is None or plus == minus:
            sides.add(None)
        else:
            sides.add(1 if plus else -1)
    return sides.pop() if len(sides) == 1 else None
