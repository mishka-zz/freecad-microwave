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
``Edges``, ``Vertexes``, ``Solids``, ``Volume``, ``common``, ``translated``,
``transformGeometry`` and ``isNull``, and a document object for ``Shape``,
``Placement``, ``Parents``, ``getSubObject``, ``getSubObjectList``,
``LinkedObject``, ``ElementList`` and ``Label``.
``translated`` takes a coordinate triple as readily as a vector. Measured under
1.1.1: an edge answers no faces, and a face answers itself.

Every shape read off a document object here is read where FreeCAD shows it.
An object's ``Shape`` is in the coordinates of the container it stands in - an
``App::Part``, a ``PartDesign::Body``, an ``App::LinkGroup`` - and the placement
of each container above it is left out. :func:`placed` asks the tree instead.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, NamedTuple

from .portbox import DIMENSIONS, KERNEL_TOLERANCE


class Unplaced(ValueError):
    """A shape this workbench cannot put where FreeCAD shows it."""


def _label(obj: Any) -> str:
    return str(getattr(obj, "Label", None) or getattr(obj, "Name", "?"))


def placed(obj: Any) -> Any:
    """``obj``'s whole shape where FreeCAD shows it, or ``None`` where it has none.

    The top container holding the object hands back its shape along the path
    down to it with every placement on the way applied, which is where FreeCAD
    draws it through nested parts, a body and a link group alike - measured under
    1.1.1, where ``getGlobalPlacement`` leaves a link group's placement out. An
    object standing in no container is shown where its own ``Shape`` is.

    A container answers no shape along a path that ends at it, measured under
    1.1.1. Its own ``Shape`` is then moved by every container above it, as
    :func:`local` moves the other way.

    :raises Unplaced: see :func:`_place`.
    """
    shape = getattr(obj, "Shape", None)
    if shape is None:
        return None
    place = _place(obj)
    if place is None:
        return shape
    root, path = place
    shown = root.getSubObject(path)
    if shown is not None:
        return shown
    above = root.getSubObject(path, retType=3).multiply(obj.Placement.inverse())
    return shape.transformGeometry(above.toMatrix())


def local(obj: Any, shape: Any) -> Any:
    """``shape``, given where it is to be shown, in ``obj``'s own coordinates.

    What a document object of this workbench draws - a port's box, a mesh
    preview - is worked out where FreeCAD shows the shapes it comes from, and
    FreeCAD shows the object's own ``Shape`` moved by every container above it.
    So the shape is moved back by those containers before it is assigned. The
    accumulated placement a path answers includes the object's own, which is
    taken off again. The move is made in the geometry rather than in the shape's
    placement: a recompute puts a feature's ``Placement`` back on its shape
    after ``execute``, so a placement given to the shape does not last.

    :raises Unplaced: see :func:`_place`.
    """
    place = _place(obj)
    if place is None or shape.isNull():
        return shape
    root, path = place
    above = root.getSubObject(path, retType=3).multiply(obj.Placement.inverse())
    return shape.transformGeometry(above.inverse().toMatrix())


def _place(obj: Any) -> tuple[Any, str] | None:
    """The top container ``obj`` stands in and the path down, or ``None``.

    ``Parents`` names every such pair, and not all of them are places:

    * a path through a link - an ``App::Link``, a link group's member shown by
      another link - is a copy the link shows, and the object stands where it
      was drawn. An element of a link array is the array's own, and the path
      from the array to it is where it stands;
    * a root in another open document is that document's drawing;
    * a body lists a feature once for each way it claims it, along one path, so
      a place is the root and the path rather than the entry, and the root is
      the object rather than its label, which two documents can share.

    :raises Unplaced: two containers hold the object, so where it is meant is
        not something the drawing says.
    """
    home = getattr(obj, "Document", None)
    places = {
        (id(root), path): (root, path)
        for root, path in getattr(obj, "Parents", None) or ()
        if getattr(root, "Document", None) is home and not _through_a_link(root, path)
    }
    if not places:
        return None
    if len(places) > 1:
        shown = sorted(f"{_label(root)}.{path}" for root, path in places.values())
        raise Unplaced(
            f"{_label(obj)!r} stands in more than one container ({', '.join(shown)}), so "
            "its place is ambiguous. Keep it in one"
        )
    ((root, path),) = places.values()
    return root, path


def _through_a_link(root: Any, path: str) -> bool:
    """Whether the path from ``root`` passes through the copy a link shows on
    its way down, as :func:`_through_a_copy` reads it.

    A link is what carries ``LinkedObject``, which a link group, a part and a
    body do not.
    """
    along = getattr(root, "getSubObjectList", None)
    if along is None:
        return getattr(root, "LinkedObject", None) is not None
    return _through_a_copy(list(along(path)))


def _through_a_copy(chain: list[Any]) -> bool:
    """Whether a chain of objects from a root down steps from a link into the
    copy it shows.

    An array link lists its own elements in ``ElementList``, measured under
    1.1.1, and a step from it to one of those is not into a copy.
    """
    return any(
        getattr(step, "LinkedObject", None) is not None
        and not any(below is each for each in getattr(step, "ElementList", None) or ())
        for step, below in zip(chain, chain[1:], strict=False)
    )


class Selected(NamedTuple):
    """One pick, on the object that owns it, with the element it names or none
    for the whole object - the form a ``Gui.Selection`` entry takes."""

    Object: Any
    SubElementNames: tuple[str, ...]


def resolved(
    selection: Iterable[Any], document: str | None = None
) -> tuple[list[Selected], list[str]]:
    """Each pick on the object that owns it, and each pick made inside a copy.

    ``selection`` is ``Gui.Selection.getCompleteSelection(0)``. It leaves each
    pick on the top object it was made through, with the path down to it, one
    entry to a pick in the order picked, measured under 1.1.1. The default
    resolution drops the path, and ``getSelectionEx`` gathers the picks under
    each top object, which loses the order between objects. A port reads its
    faces off that order, so each pick here is its own entry, in the order
    picked.

    The selection spans every open document (``Gui/Selection/Selection.cpp``,
    ``getCompleteSelection``), and a link cannot reach into another, so only
    the picks in ``document`` are kept where it is named.

    A pick made inside the copy a link shows resolves to the original, where it
    was drawn, so a link to it would name geometry the user did not pick. Each
    such pick is named rather than kept, as the original's label inside the top
    object's. Elsewhere the pick lands on the last object along the path, with
    the element at the path's end, which is what the default gives. An object
    that cannot list the path keeps it whole, which :func:`element` refuses by
    name.
    """
    picked: list[Selected] = []
    copies: list[str] = []
    for entry in selection:
        if document is not None and getattr(entry, "DocumentName", document) != document:
            continue
        top = getattr(entry, "Object", entry)
        for sub in tuple(getattr(entry, "SubElementNames", ()) or ()) or ("",):
            along = getattr(top, "getSubObjectList", None)
            if not sub or along is None:
                picked.append(Selected(top, (sub,) if sub else ()))
                continue
            chain = list(along(sub))
            if _through_a_copy(chain):
                copies.append(f"{_label(chain[-1])!r} in {_label(top)!r}")
                continue
            name = sub.rsplit(".", 1)[-1]
            picked.append(Selected(chain[-1], (name,) if name else ()))
    return picked, copies


def copy_note(labels: Iterable[str]) -> str:
    """What a command says of the picks :func:`resolved` names."""
    labels = list(dict.fromkeys(labels))
    named = ", ".join(labels)
    was = "was" if len(labels) == 1 else "were"
    return (
        f"{named} {was} picked inside the copy a link shows and left out, since such a "
        "pick names the original rather than the copy. Pick the original, or the link "
        "itself"
    )


def element(obj: Any, name: str = "") -> Any:
    """What ``name`` names on ``obj``, where FreeCAD shows it, or the whole shape.

    :raises Unplaced: ``name`` reaches the element through a container, as a
        path such as ``Part.Box.Face1`` does, or :func:`placed` does.
    """
    if "." in name:
        raise Unplaced(
            f"{_label(obj)!r} is picked at {name}, an element named through a "
            "container it holds. Pick it on the object that owns it"
        )
    shape = placed(obj)
    if shape is None or not name:
        return shape
    return shape.getElement(name)


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
    return placed(link[0] if isinstance(link, tuple) else link)


def shapes(link: Any) -> list[Any]:
    """Every shape a ``PropertyLinkSub`` names, or the whole one."""
    obj = link[0] if isinstance(link, tuple) else link
    if body(link) is None:
        return []
    return [element(obj, name) for name in named(link)]


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
