# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the user picked, beyond the box it reduces to.

A port is built from bounding boxes, and one thing they cannot say is whether
the pick encloses an area. The end edge of a trace running off a grid axis has a
box with depth on both axes across the port, and so does a small pad; the boxes
are the same six numbers and the ports are not.
:func:`Microwave.portbox.lumped` is given the answer rather than guessing it.

Beside :mod:`Microwave.portbox` rather than under ``Objects`` for the same
mechanical reason as :mod:`Microwave.annulus`: ``Objects/__init__`` imports
FreeCAD, and the openEMS adapter asks this about the same pick and has to run
without it. One module so that the drawing and the envelope cannot answer
differently about one port, which is what the shared box vocabulary is for.

Nothing here imports FreeCAD. What it asks of a shape is ``getElement`` and
``Faces``, measured under 1.1.1: an edge answers no faces, and a face answers
itself.
"""

from __future__ import annotations

from typing import Any


def named(link: Any) -> list[str]:
    """The sub-element names a ``PropertyLinkSub`` carries, or ``[""]``.

    ``(object, ['Face3'])`` is the ordinary form, ``(object, [''])`` is the
    whole shape, and a bare object is what a link with nothing selected under it
    reduces to. One empty name rather than an empty list, so a caller loops the
    same way over either.
    """
    _, sub = link if isinstance(link, tuple) else (link, None)
    names = [sub] if isinstance(sub, str) else list(sub or ())
    return [name for name in names if name] or [""]


def shapes(link: Any) -> list[Any]:
    """Every shape a ``PropertyLinkSub`` names, or the whole one."""
    obj = link[0] if isinstance(link, tuple) else link
    shape = getattr(obj, "Shape", None)
    if shape is None:
        return []
    return [shape if not name else shape.getElement(name) for name in named(link)]


def is_outline(link: Any) -> bool:
    """Whether everything a pick names encloses no area.

    Every name and not the first, because a pick naming an edge beside a face is
    not an outline, and reading one name would let the drawing and the envelope
    reach different answers about the same port.

    Asked of the contents rather than of ``ShapeType``, which names how a shape
    is built and not what it covers. A pick that resolves to nothing is not an
    outline: the caller has no shape to build from either, and says so first.
    """
    picked = shapes(link)
    return bool(picked) and not any(shape.Faces for shape in picked)
