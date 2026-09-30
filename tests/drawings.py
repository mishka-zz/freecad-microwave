# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The drawings the tetrahedral mesher is run over, each cut into labelled pieces.

A separate corpus from ``tests/openems_corpus.py``, which is one shape per
specimen and asks what the geometry layer measures off it. Here a specimen is
several shapes under labels a user chose, and what is asked is what the label
map says about them - so the two share nothing but the idea of drawing the
awkward case rather than describing it.

Drawn by FreeCAD and not by Gmsh's own kernel, deliberately. The map these
exercise was measured over shapes FreeCAD wrote, and a corpus drawn the other
way would test the mechanism on specimens it was never measured on. What
FreeCAD puts in a file is its own: a boolean comes back a compound, and a
compound holds whatever it was given, so a piece here can carry a shape of a
dimension its label did not declare - which is a drawing a user makes and the
other kernel does not.

One file per piece, because that is the mechanism: each drawn shape is imported
on its own, and what came back belongs to that shape's label with nothing
matched.

This module is imported by the probe that runs under FreeCAD and by the tests
that read what the probe wrote, so the kernel is imported lazily: under a plain
Python there is none, and the names still have to be enumerable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["COAX_RADIUS", "Drawing", "MESHED", "REFUSED", "UNMESHED", "drawings", "ends"]

#: The mesher is required to write a mesh for this drawing.
MESHED = "meshed"

#: The mesher is required to refuse it before Gmsh is asked for an element,
#: naming the labels the drawing used or where in the drawing the fault is.
REFUSED = "refused"

#: Gmsh is asked and there is no mesh of the drawing. The mesher is required to
#: say so rather than hand back the file that was written anyway.
UNMESHED = "unmeshed"

#: How much of a curved body a straight-sided mesh of it is short of, where the
#: element asked for is the order of the body's own radius. Every element lies
#: on chords of the surface and therefore inside it, and the bound is generous:
#: what is held is that the shortfall stays the order of the discretisation and
#: does not become a region the meshing left out.
CHORDS = 0.05

#: The outer radius of the coaxial specimen, in mm. Named because a case that
#: asks whether a node lies on the drawn surface has to know where that surface
#: is, and reading it off the drawing is what keeps the two in step.
COAX_RADIUS = 10.0

#: How far one specimen's port face stands past the solid it was drawn against,
#: in mm. Far enough that the kernel leaves the overhang standing: fragmenting
#: closes a difference below an absolute length, and what that length is belongs
#: to the kernel rather than to the drawing.
AN_OVERHANG = 0.001


@dataclass(frozen=True)
class Drawing:
    """One drawing, cut into the pieces a caller would hand over.

    :param name: names the files the probe writes and the case that reads them,
        so it is not renamed casually.
    :param build: takes the ``Part`` module and returns ``(label, dimension,
        shape)`` per piece. Anything else the kernel offers is imported inside
        the builder that wants it, so this module needs no kernel.
    :param subject: what this drawing is here to say, in a few words. It goes
        into the failure message.
    :param expect: :data:`MESHED`, :data:`REFUSED` or :data:`UNMESHED`.
    :param complaint: a word the refusal has to contain, where the drawing is
        here for one particular refusal. A drawing can be wrong in more ways
        than one, and a case that only asks "was it refused" passes on the
        wrong complaint.
    :param declares: the label and dimension of each piece, in the order the
        builder returns them. Written out here as well as built, because what
        the probe wrote is checked in and nothing else compares it with the
        builder it came from. What it catches is a piece renamed, added,
        dropped or re-declared without the probe being run again; a builder
        whose geometry moved under unchanged labels it cannot see.
    :param top: the dimension a mesher is asked to fill for this drawing.
    :param element_order: the geometric element order to ask for.
    :param curved: whether to ask for the added nodes on the drawn surface.
        Both are here because a drawing can be sound at one order and not at
        another, and the pair belongs with the shape it is about.
    :param coarsest: the largest element asked for, in mm.
    :param finest: the smallest, in mm. Both are here rather than in the tests
        because a drawing four orders below the element size is a specimen
        about the element size, and the pair belongs with the shape it is about.
    :param per_turn: the elements asked round a full turn of a curved surface,
        for the same reason.
    :param loses: the largest share of the drawing's own measure the finished
        mesh may be short of. Zero to rounding for a flat-sided body, since a
        straight element fills a straight region exactly; a curved body is
        approximated by chords and stands inside what it was drawn as, and the
        share is what that costs at this element size and this order.
    :param round_trip: how far a figure read back out of the written file may
        stand from the one the mesher reported, relative. Rounding for a
        drawing near the origin. A file holds each coordinate in a fixed number
        of significant digits, so a drawing far from the origin spends most of
        them on the distance and leaves the shape of an element the rest.
    :param priority: per label, which of two labels drawn over one region
        owns it. Absent for a label is nothing stated, which is what a drawing
        with no overlap in it needs and what makes an overlap a refusal.
    :param apart: whether what the drawing fills comes apart into parts no
        entity of the dimension below joins. Written out rather than read off
        the drawing, because the mesher's answer is what it is compared with.
    :param inward: per label, the way from its faces into the model. What the
        mesher leaves on the other side is left out, so a drawing stating one
        is meshed short of what it measures, and is in :func:`ends` rather than
        in :func:`drawings`.
    :param divides: the labels nothing of what is filled crosses. Nothing of
        the mesh moves for it: what changes is that the answer says which parts
        those faces leave.
    :param parts: how many parts the labels that divide leave what is filled
        in. One where none was stated, since the answer then carries nothing.
    :param leaves: the labels whose pieces leave the model, each ending as the
        faces its pieces leave behind. The drawing's measure is what is left.
    :param tags: coarse groupings a test can select on.
    """

    name: str
    build: Callable[[Any], Sequence[tuple[str, int, Any]]]
    subject: str
    declares: tuple[tuple[str, int], ...]
    expect: str = MESHED
    complaint: str = ""
    top: int = 3
    element_order: int = 1
    curved: bool = False
    coarsest: float = 8.0
    finest: float = 2.0
    per_turn: int = 0
    loses: float = 1e-12
    round_trip: float = 1e-9
    priority: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    apart: bool = False
    inward: tuple[tuple[str, tuple[float, float, float]], ...] = field(default_factory=tuple)
    divides: tuple[str, ...] = field(default_factory=tuple)
    parts: int = 1
    leaves: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)


def _vector(part: Any, x: float, y: float, z: float) -> Any:
    """A kernel vector. ``Part`` does not carry one, so it comes from the app."""
    import FreeCAD

    return FreeCAD.Vector(x, y, z)


def _face_at_x(part: Any, x: float, y0: float, y1: float, z0: float, z1: float) -> Any:
    """A rectangle standing in the plane of constant x, drawn from its corners."""
    corners = [
        _vector(part, x, y0, z0),
        _vector(part, x, y1, z0),
        _vector(part, x, y1, z1),
        _vector(part, x, y0, z1),
        _vector(part, x, y0, z0),
    ]
    return part.Face(part.Wire(part.makePolygon(corners)))


# ------------------------------------------------------------- what meshes


def _plain_box(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [("air", 3, box), ("walls", 2, part.Compound(box.Faces))]


def _split_cavity(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [
        ("air", 3, box),
        ("walls", 2, part.Compound(box.Faces)),
        ("port", 2, _face_at_x(part, 15.0, 0.0, 20.0, 0.0, 10.0)),
    ]


def _shared_interface(part: Any) -> Sequence[tuple[str, int, Any]]:
    lower = part.makeBox(40.0, 40.0, 10.0)
    upper = part.makeBox(40.0, 40.0, 6.0, _vector(part, 0, 0, 10))
    return [
        ("substrate", 3, lower),
        ("air", 3, upper),
        ("walls", 2, part.Compound(lower.Faces + upper.Faces)),
    ]


def _crossing_port(part: Any) -> Sequence[tuple[str, int, Any]]:
    lower = part.makeBox(40.0, 40.0, 10.0)
    upper = part.makeBox(40.0, 40.0, 6.0, _vector(part, 0, 0, 10))
    return [
        ("substrate", 3, lower),
        ("air", 3, upper),
        ("walls", 2, part.Compound(lower.Faces + upper.Faces)),
        ("port", 2, _face_at_x(part, 20.0, 0.0, 40.0, 0.0, 16.0)),
    ]


def _coax(part: Any) -> Sequence[tuple[str, int, Any]]:
    outer = part.makeCylinder(COAX_RADIUS, 50.0).cut(part.makeCylinder(4.0, 50.0))
    pin = part.makeCylinder(4.0, 50.0)
    return [
        ("dielectric", 3, outer),
        ("conductor", 3, pin),
        ("walls", 2, part.Compound(outer.Faces + pin.Faces)),
    ]


def _post_in_the_box(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A box of air with a solid post standing clear of every wall."""
    box = part.makeBox(30.0, 20.0, 10.0)
    post = part.makeBox(8.0, 6.0, 4.0, _vector(part, 10.0, 7.0, 3.0))
    return [("air", 3, box), ("post", 3, post), ("walls", 2, part.Compound(box.Faces))]


def _cone_on_its_apex(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A box of air with a cone of radius 1 and height 2 standing on its apex
    clear of every wall."""
    box = part.makeBox(30.0, 20.0, 10.0)
    cone = part.makeCone(0.0, 1.0, 2.0, _vector(part, 15.0, 10.0, 4.0))
    return [("air", 3, box), ("cone", 3, cone), ("walls", 2, part.Compound(box.Faces))]


def _curve_inside_a_post(part: Any) -> Sequence[tuple[str, int, Any]]:
    """The post in the box, with a curve standing wholly inside the post. The
    post has been made part of another shape, as a shape out of a document
    often has, so the file marks it as no longer free."""
    box = part.makeBox(30.0, 20.0, 10.0)
    post = part.makeBox(8.0, 6.0, 4.0, _vector(part, 10.0, 7.0, 3.0))
    part.Compound([post])
    curve = part.makeLine(_vector(part, 12.0, 10.0, 5.0), _vector(part, 16.0, 10.0, 5.0))
    return [
        ("air", 3, box),
        ("post", 3, post),
        ("curve", 1, curve),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _plate_standing_in_the_box(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A box of air with a solid plate standing its full height, the air running
    round both ends of it. Two of the plate's faces lie on two of the box's."""
    box = part.makeBox(30.0, 20.0, 10.0)
    plate = part.makeBox(10.0, 1.0, 10.0, _vector(part, 10.0, 9.5, 0.0))
    return [("air", 3, box), ("plate", 3, plate), ("walls", 2, part.Compound(box.Faces))]


def _plate_across_the_box(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A box of air with a solid plate filling its whole cross-section."""
    box = part.makeBox(30.0, 20.0, 10.0)
    plate = part.makeBox(2.0, 20.0, 10.0, _vector(part, 14.0, 0.0, 0.0))
    return [("air", 3, box), ("plate", 3, plate), ("walls", 2, part.Compound(box.Faces))]


def _post_through_a_sheet(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A box of air with a solid post, and a sheet standing in the air that
    passes through the post."""
    box = part.makeBox(30.0, 20.0, 10.0)
    post = part.makeBox(8.0, 6.0, 4.0, _vector(part, 10.0, 7.0, 3.0))
    sheet = _face_at_x(part, 14.0, 2.0, 18.0, 1.0, 9.0)
    return [
        ("air", 3, box),
        ("post", 3, post),
        ("sheet", 2, sheet),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _sheet_inside_a_post(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A box of air with a solid post, and a sheet standing wholly inside the post."""
    box = part.makeBox(30.0, 20.0, 10.0)
    post = part.makeBox(8.0, 6.0, 4.0, _vector(part, 10.0, 7.0, 3.0))
    sheet = _face_at_x(part, 14.0, 8.0, 12.0, 4.0, 6.0)
    return [
        ("air", 3, box),
        ("post", 3, post),
        ("sheet", 2, sheet),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _block_inside_a_post(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A box of air with a solid post, and a block drawn wholly inside the post."""
    box = part.makeBox(30.0, 20.0, 10.0)
    post = part.makeBox(8.0, 6.0, 4.0, _vector(part, 10.0, 7.0, 3.0))
    block = part.makeBox(2.0, 2.0, 2.0, _vector(part, 13.0, 9.0, 4.0))
    return [
        ("air", 3, box),
        ("post", 3, post),
        ("block", 3, block),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _post_under_a_higher_block(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A box of air with a solid post, and a block overlapping one end of it."""
    box = part.makeBox(30.0, 20.0, 10.0)
    post = part.makeBox(8.0, 6.0, 4.0, _vector(part, 10.0, 7.0, 3.0))
    block = part.makeBox(4.0, 6.0, 4.0, _vector(part, 16.0, 7.0, 3.0))
    return [
        ("air", 3, box),
        ("post", 3, post),
        ("block", 3, block),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _sheet_on_a_post_face(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A box of air with a solid post, and a sheet drawn as the post's top face."""
    box = part.makeBox(30.0, 20.0, 10.0)
    post = part.makeBox(8.0, 6.0, 4.0, _vector(part, 10.0, 7.0, 3.0))
    top = max(post.Faces, key=lambda face: face.CenterOfMass.z)
    return [
        ("air", 3, box),
        ("post", 3, post),
        ("sheet", 2, top.copy()),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _post_in_its_own_skin(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A box of air with a solid post, and its every face drawn as a skin."""
    box = part.makeBox(30.0, 20.0, 10.0)
    post = part.makeBox(8.0, 6.0, 4.0, _vector(part, 10.0, 7.0, 3.0))
    return [
        ("air", 3, box),
        ("post", 3, post),
        ("skin", 2, part.Compound(post.Faces)),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _post_beside_the_box(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A box of air, and a solid post standing clear of it."""
    box = part.makeBox(30.0, 20.0, 10.0)
    post = part.makeBox(8.0, 6.0, 4.0, _vector(part, 40.0, 7.0, 3.0))
    return [("air", 3, box), ("post", 3, post), ("walls", 2, part.Compound(box.Faces))]


def _post_through_the_wall(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A box of air with a solid post passing out through one of its walls."""
    box = part.makeBox(30.0, 20.0, 10.0)
    post = part.makeBox(8.0, 6.0, 4.0, _vector(part, 26.0, 7.0, 3.0))
    return [("air", 3, box), ("post", 3, post), ("walls", 2, part.Compound(box.Faces))]


def _two_posts_across_a_sheet(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A box of air with two solid posts side by side, touching on a face, and a
    sheet standing across both."""
    box = part.makeBox(30.0, 20.0, 10.0)
    left = part.makeBox(4.0, 6.0, 4.0, _vector(part, 10.0, 7.0, 3.0))
    right = part.makeBox(4.0, 6.0, 4.0, _vector(part, 14.0, 7.0, 3.0))
    sheet = part.Face(
        part.makePolygon(
            [
                _vector(part, 8.0, 10.0, 1.0),
                _vector(part, 20.0, 10.0, 1.0),
                _vector(part, 20.0, 10.0, 9.0),
                _vector(part, 8.0, 10.0, 9.0),
                _vector(part, 8.0, 10.0, 1.0),
            ]
        )
    )
    return [
        ("air", 3, box),
        ("left", 3, left),
        ("right", 3, right),
        ("sheet", 2, sheet),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _two_posts_overlapping(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A box of air with two solid posts overlapping each other."""
    box = part.makeBox(30.0, 20.0, 10.0)
    left = part.makeBox(6.0, 6.0, 4.0, _vector(part, 10.0, 7.0, 3.0))
    right = part.makeBox(6.0, 6.0, 4.0, _vector(part, 13.0, 7.0, 3.0))
    return [
        ("air", 3, box),
        ("left", 3, left),
        ("right", 3, right),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _closed_shell(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [("skin", 2, part.Compound(box.Faces))]


def _skin_sealing_a_cavity(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A box of air with a closed skin standing inside it, sealing what it holds.

    What metal drawn as the surface of a post looks like, and what the wall of
    a cavity somebody meant looks like. The two are one drawing.
    """
    box = part.makeBox(30.0, 20.0, 10.0)
    inner = part.makeBox(8.0, 6.0, 4.0, _vector(part, 10.0, 7.0, 3.0))
    return [
        ("air", 3, box),
        ("skin", 2, part.Compound(inner.Faces)),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _skin_across_the_box(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A box of air with a skin standing in it that it wraps round.

    The skin divides nothing: the air reaches past both its ends, so the parts
    come back as one however the skin is read.
    """
    box = part.makeBox(30.0, 20.0, 10.0)
    sheet = _face_at_x(part, 15.0, 4.0, 16.0, 0.0, 10.0)
    return [
        ("air", 3, box),
        ("skin", 2, sheet),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _sphere_skin(part: Any) -> Sequence[tuple[str, int, Any]]:
    sphere = part.makeSphere(10.0)
    return [("skin", 2, part.Compound(sphere.Faces))]


def _sphere_seam(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A sphere's surface with the curve it closes on labelled as well."""
    sphere = part.makeSphere(10.0)
    face = sphere.Faces[0]
    seam = max(face.Edges, key=lambda edge: edge.Length)
    return [("skin", 2, part.Compound([face])), ("seam", 1, part.Compound([seam]))]


def _cavity(part: Any) -> Sequence[tuple[str, int, Any]]:
    body = part.makeBox(20.0, 20.0, 20.0).cut(
        part.makeBox(10.0, 10.0, 10.0, _vector(part, 5, 5, 5))
    )
    return [("air", 3, body), ("walls", 2, part.Compound(body.Faces))]


# -------------------------------------------- what a mesher is asked to break on


def _touching_at_an_edge(part: Any) -> Sequence[tuple[str, int, Any]]:
    both = part.makeBox(10.0, 10.0, 10.0).fuse(
        part.makeBox(10.0, 10.0, 10.0, _vector(part, 10.0, 10.0, 0.0))
    )
    return [("air", 3, both), ("walls", 2, part.Compound(both.Faces))]


def _touching_at_a_vertex(part: Any) -> Sequence[tuple[str, int, Any]]:
    both = part.makeBox(10.0, 10.0, 10.0).fuse(
        part.makeBox(10.0, 10.0, 10.0, _vector(part, 10.0, 10.0, 10.0))
    )
    return [("air", 3, both), ("walls", 2, part.Compound(both.Faces))]


def _sliver_slab(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 1.0e-3, 10.0)
    return [("air", 3, box), ("walls", 2, part.Compound(box.Faces))]


def _sliver_wedge(part: Any) -> Sequence[tuple[str, int, Any]]:
    corners = [
        _vector(part, 0.0, 0.0, 0.0),
        _vector(part, 30.0, 0.0, 0.0),
        _vector(part, 30.0, 1.0e-4, 0.0),
        _vector(part, 0.0, 10.0, 0.0),
        _vector(part, 0.0, 0.0, 0.0),
    ]
    face = part.Face(part.Wire(part.makePolygon(corners)))
    solid = face.extrude(_vector(part, 0.0, 0.0, 10.0))
    return [("air", 3, solid), ("walls", 2, part.Compound(solid.Faces))]


def _small_and_far_apart(part: Any) -> Sequence[tuple[str, int, Any]]:
    near = part.makeBox(30.0, 20.0, 10.0)
    far = part.makeBox(0.1, 0.1, 0.1, _vector(part, 1.0e6, 0.0, 0.0))
    return [
        ("air", 3, near),
        ("air", 3, far),
        ("walls", 2, part.Compound(near.Faces + far.Faces)),
    ]


def _far_from_the_origin(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0, _vector(part, 1.0e9, 1.0e9, 1.0e9))
    return [("air", 3, box), ("walls", 2, part.Compound(box.Faces))]


def _below_the_element(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(1.0e-4, 1.0e-4, 1.0e-4)
    return [("air", 3, box), ("walls", 2, part.Compound(box.Faces))]


def _apart_below_tolerance(part: Any) -> Sequence[tuple[str, int, Any]]:
    one = part.makeBox(10.0, 10.0, 10.0)
    two = part.makeBox(10.0, 10.0, 10.0, _vector(part, 10.0 + 1.0e-6, 0.0, 0.0))
    return [
        ("air", 3, one),
        ("air", 3, two),
        ("walls", 2, part.Compound(one.Faces + two.Faces)),
    ]


def _curved_body(part: Any) -> Sequence[tuple[str, int, Any]]:
    pipe = part.makeCylinder(10.0, 30.0)
    return [("air", 3, pipe), ("walls", 2, part.Compound(pipe.Faces))]


def _self_intersecting(part: Any) -> Sequence[tuple[str, int, Any]]:
    corners = [
        _vector(part, 0.0, 0.0, 0.0),
        _vector(part, 20.0, 20.0, 0.0),
        _vector(part, 20.0, 0.0, 0.0),
        _vector(part, 0.0, 20.0, 0.0),
        _vector(part, 0.0, 0.0, 0.0),
    ]
    face = part.Face(part.Wire(part.makePolygon(corners)))
    solid = face.extrude(_vector(part, 0.0, 0.0, 10.0))
    return [("air", 3, solid), ("walls", 2, part.Compound(solid.Faces))]


def _partly_unmeshed(part: Any) -> Sequence[tuple[str, int, Any]]:
    sound = part.makeBox(30.0, 20.0, 10.0)
    corners = [
        _vector(part, 60.0, 0.0, 0.0),
        _vector(part, 80.0, 20.0, 0.0),
        _vector(part, 80.0, 0.0, 0.0),
        _vector(part, 60.0, 20.0, 0.0),
        _vector(part, 60.0, 0.0, 0.0),
    ]
    crossed = part.Face(part.Wire(part.makePolygon(corners))).extrude(_vector(part, 0.0, 0.0, 10.0))
    return [
        ("air", 3, sound),
        ("air", 3, crossed),
        ("walls", 2, part.Compound(sound.Faces + crossed.Faces)),
    ]


def _rim_far_off(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A small open shell and its rim, with a large body a long way off."""
    small = part.makeBox(1.0e-4, 1.0e-4, 1.0e-4)
    skin = part.Compound(small.Faces[1:])
    rim = part.Compound(small.Faces[0].Edges)
    far = part.makeBox(1.0, 1.0, 1.0, _vector(part, 1.0e9, 0.0, 0.0))
    return [("skin", 2, skin), ("rim", 1, rim), ("far", 2, part.Compound(far.Faces))]


def _hidden_interface(part: Any) -> Sequence[tuple[str, int, Any]]:
    """Two solids on a shared face, with only the outside labelled."""
    lower = part.makeBox(30.0, 20.0, 10.0)
    upper = part.makeBox(30.0, 20.0, 6.0, _vector(part, 0, 0, 10))
    outer = [face for face in lower.Faces + upper.Faces if abs(face.CenterOfMass.z - 10.0) > 1e-9]
    return [
        ("substrate", 3, lower),
        ("air", 3, upper),
        ("walls", 2, part.Compound(outer)),
    ]


def _curving_turns_elements_over(part: Any) -> Sequence[tuple[str, int, Any]]:
    ring = part.makeTorus(20.0, 5.0)
    return [("air", 3, ring), ("walls", 2, part.Compound(ring.Faces))]


def _curving_pass_fails(part: Any) -> Sequence[tuple[str, int, Any]]:
    """The ring above with a thinner tube.

    Gmsh's curving pass does not answer the same way on every run, so a change
    to this ring is checked over repeated runs.
    """
    ring = part.makeTorus(20.0, 3.0)
    return [("air", 3, ring), ("walls", 2, part.Compound(ring.Faces))]


def _unsewn_shell(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    faces = list(box.Faces)
    faces[0] = faces[0].translated(_vector(part, 0.01, 0.0, 0.0))
    return [("air", 3, part.Solid(part.Shell(faces))), ("walls", 2, part.Compound(faces))]


def _open_shell(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [("skin", 2, part.Compound(box.Faces[1:]))]


# ------------------------------------------------- what stands inside what


def _part_in_a_box(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    inner = part.makeBox(8.0, 6.0, 4.0, _vector(part, 10.0, 7.0, 3.0))
    return [
        ("air", 3, box),
        ("part", 3, inner),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _part_flush_with_a_wall(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    inner = part.makeBox(8.0, 6.0, 4.0, _vector(part, 0.0, 7.0, 3.0))
    return [
        ("air", 3, box),
        ("part", 3, inner),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _nested_three_deep(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    case = part.makeBox(20.0, 14.0, 7.0, _vector(part, 5.0, 3.0, 1.5))
    inner = part.makeBox(8.0, 6.0, 3.0, _vector(part, 11.0, 7.0, 3.0))
    return [
        ("air", 3, box),
        ("case", 3, case),
        ("part", 3, inner),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _round_part_in_a_box(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    rod = part.makeCylinder(3.0, 6.0, _vector(part, 15.0, 10.0, 2.0))
    return [
        ("air", 3, box),
        ("rod", 3, rod),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _two_parts_in_one_box(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    near = part.makeBox(6.0, 6.0, 4.0, _vector(part, 4.0, 7.0, 3.0))
    far = part.makeBox(6.0, 6.0, 4.0, _vector(part, 20.0, 7.0, 3.0))
    return [
        ("air", 3, box),
        ("near", 3, near),
        ("far", 3, far),
        ("walls", 2, part.Compound(box.Faces)),
    ]


#: The side of a box-shaped body small enough that the kernel leaves it standing
#: inside the one holding it rather than cutting it out, in mm, and one just
#: large enough that it cuts it. The kernel's and not the drawing's: the figure
#: is absolute, and a round body of the smaller size is cut.
SPECK = 1.0e-5
CUT_SPECK = 1.05e-5

#: Where each small body stands in the 30 x 20 x 10 domain, clear of the middle
#: so that no face or corner of it falls on a point the domain is symmetric
#: about.
SPECK_AT = (13.0, 8.0, 4.0)

#: How far one small body stands from each of three walls, in mm, in a corner of
#: the domain. Clear of them by more than the kernel's tolerance, so it stands
#: free and is left uncut, and as far from them as it is across.
CLEAR_OF_THE_WALLS = 1e-5

#: A gap in mm the fragmenting leaves open: wider than the tolerance the
#: kernel joins two faces within, and narrower than the slab it stands beside.
HAIR = 5e-7


def _speck(part: Any, side: float, at: Sequence[float] = SPECK_AT) -> Any:
    return part.makeBox(side, side, side, _vector(part, *at))


def _in_the_domain(part: Any, body: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [
        ("air", 3, box),
        ("speck", 3, body),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _speck_in_a_box(part: Any) -> Sequence[tuple[str, int, Any]]:
    return _in_the_domain(part, _speck(part, SPECK))


def _speck_cut_out(part: Any) -> Sequence[tuple[str, int, Any]]:
    return _in_the_domain(part, _speck(part, CUT_SPECK))


def _speck_cut_out_at_the_middle(part: Any) -> Sequence[tuple[str, int, Any]]:
    """The body just large enough to be cut out, with a corner at the middle of
    the domain, which is where the centre of the domain less the body falls."""
    return _in_the_domain(part, _speck(part, CUT_SPECK, (15.0, 10.0, 5.0)))


def _speck_against_a_wall(part: Any) -> Sequence[tuple[str, int, Any]]:
    """Against the top wall, where the face on the wall bounds the body alone
    and every point the kernel accepts on it is on the domain's rim round it."""
    return _in_the_domain(part, _speck(part, SPECK, (*SPECK_AT[:2], 10.0 - SPECK)))


def _speck_in_a_corner(part: Any) -> Sequence[tuple[str, int, Any]]:
    return _in_the_domain(part, _speck(part, SPECK, (CLEAR_OF_THE_WALLS,) * 3))


def _rod_speck(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A cylinder a tenth of the small body's side tall, which the kernel cuts
    out."""
    return _in_the_domain(part, part.makeCylinder(SPECK / 20, SPECK / 10, _vector(part, *SPECK_AT)))


def _plug_in_a_pocket(part: Any) -> Sequence[tuple[str, int, Any]]:
    """The small body at the bottom of a pocket three times as deep as it, in the
    domain's top face. Its sides are the pocket's, and every point the kernel
    accepts on its top face is on the pocket's walls."""
    at = (*SPECK_AT[:2], 10.0 - 3 * SPECK)
    block = part.makeBox(30.0, 20.0, 10.0).cut(
        part.makeBox(SPECK, SPECK, 3 * SPECK, _vector(part, *at))
    )
    plug = _speck(part, SPECK, at)
    return [
        ("block", 3, block),
        ("plug", 3, plug),
        ("walls", 2, part.Compound(block.Faces + plug.Faces)),
    ]


def _ring_on_a_post(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A ring round a post that stands on a base, the post and the base one body.
    The ring's centre is inside the post."""
    base = part.makeBox(10.0, 10.0, 1.0)
    post = part.makeCylinder(1.5, 6.0, _vector(part, 5.0, 5.0, 0.0))
    holder = base.fuse(post)
    ring = part.makeCylinder(4.0, 2.0, _vector(part, 5.0, 5.0, 1.0)).cut(post)
    return [
        ("holder", 3, holder),
        ("ring", 3, ring),
        ("walls", 2, part.Compound(holder.Faces + ring.Faces)),
    ]


def _solid_dropped_inside(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A curve label whose file also carries a solid standing inside the domain.
    The declaration keeps the curves, and the solid stays in the model uncut."""
    box = part.makeBox(30.0, 20.0, 10.0)
    stray = part.makeBox(1.0, 1.0, 1.0, _vector(part, 10.0, 5.0, 3.0))
    return [
        ("air", 3, box),
        ("walls", 2, part.Compound(box.Faces)),
        ("wire", 1, part.Compound([stray])),
    ]


#: A copper trace's thickness in mm, on a substrate and cut out of the air above
#: it, as thick metal is drawn.
COPPER = 0.035


def _cut_out_of_the_air(part: Any, trace: Any) -> Sequence[tuple[str, int, Any]]:
    substrate = part.makeBox(30.0, 20.0, 1.5)
    air = part.makeBox(30.0, 20.0, 8.5, _vector(part, 0.0, 0.0, 1.5)).cut(trace)
    return [
        ("substrate", 3, substrate),
        ("air", 3, air),
        ("walls", 2, part.Compound(substrate.Faces + air.Faces)),
    ]


def _bent_trace(part: Any) -> Sequence[tuple[str, int, Any]]:
    """An L of trace a millimetre wide. The face the substrate shows under the
    bend is an L, and a coarse grid over it finds points on its rim only."""
    run = part.makeBox(12.0, 1.0, COPPER, _vector(part, 5.0, 5.0, 1.5))
    arm = part.makeBox(1.0, 10.0, COPPER, _vector(part, 5.0, 5.0, 1.5))
    return _cut_out_of_the_air(part, run.fuse(arm).removeSplitter())


#: The arms of an L of unit width whose centre falls on its inner corner: the
#: square of the golden ratio, from the centre's two coordinates each set to the
#: width.
GOLDEN_ARM = ((1.0 + 5.0**0.5) / 2.0) ** 2


def _copper_bend(part: Any, along: float = 4.0, up: float = 3.0) -> Sequence[tuple[str, int, Any]]:
    """An L of copper a millimetre wide from the domain's wall, drawn as a body
    of its own. At 4 mm along and 3 mm up, the centre of the L falls on its own
    face that the air shares."""
    run = part.makeBox(along, 1.0, COPPER, _vector(part, 0.0, 5.0, 1.5))
    arm = part.makeBox(1.0, up, COPPER, _vector(part, 0.0, 5.0, 1.5))
    trace = run.fuse(arm).removeSplitter()
    substrate = part.makeBox(30.0, 20.0, 1.5)
    air = part.makeBox(30.0, 20.0, 8.5, _vector(part, 0.0, 0.0, 1.5)).cut(trace)
    return [
        ("substrate", 3, substrate),
        ("air", 3, air),
        ("copper", 3, trace),
        ("walls", 2, part.Compound(substrate.Faces + air.Faces + trace.Faces)),
    ]


def _copper_bend_on_its_edge(part: Any) -> Sequence[tuple[str, int, Any]]:
    """The same L with both arms :data:`GOLDEN_ARM` long, whose centre falls on
    the edge of its inner corner, which the air shares."""
    return _copper_bend(part, GOLDEN_ARM, GOLDEN_ARM)


def _speck_at_the_middle_of_a_ball(part: Any) -> Sequence[tuple[str, int, Any]]:
    """The small body at the centre of a ball of air, which the kernel leaves
    uncut; the ball's centre is then inside the body."""
    ball = part.makeSphere(10.0)
    speck = _speck(part, SPECK, (-SPECK / 2,) * 3)
    return [("air", 3, ball), ("speck", 3, speck), ("walls", 2, part.Compound(ball.Faces))]


def _lid_over_a_groove(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A lid on a block, over a ring groove cut in the block's top. The lid's
    face over the groove is a ring 0.4 mm wide with the block round both rims."""
    groove = part.makeCylinder(5.0, 2.0, _vector(part, 15.0, 10.0, 9.0)).cut(
        part.makeCylinder(4.6, 4.0, _vector(part, 15.0, 10.0, 8.0))
    )
    block = part.makeBox(30.0, 20.0, 10.0).cut(groove)
    lid = part.makeBox(14.0, 14.0, 2.0, _vector(part, 8.0, 3.0, 10.0))
    return [
        ("block", 3, block),
        ("lid", 3, lid),
        ("walls", 2, part.Compound(block.Faces + lid.Faces)),
    ]


def _slab_in_a_slot(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A slab standing free in a slot through the domain, a hair from the slot's
    floor and roof."""
    slot = part.makeBox(
        25.0, 22.0, SPECK + 2 * HAIR, _vector(part, 5.0, -1.0, 5.0 - SPECK / 2 - HAIR)
    )
    block = part.makeBox(30.0, 20.0, 10.0).cut(slot)
    slab = part.makeBox(15.0, 10.0, SPECK, _vector(part, 10.0, 5.0, 5.0 - SPECK / 2))
    return [
        ("block", 3, block),
        ("slab", 3, slab),
        ("walls", 2, part.Compound(block.Faces + slab.Faces)),
    ]


def _post_through_a_slab(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A post filling a hole through a slab and standing out of it, the hole's
    edges on lines of the grid a point on a face is looked for on."""
    slab = part.makeBox(10.0, 10.0, 2.0).cut(
        part.makeBox(5.0, 5.0, 2.0, _vector(part, 2.5, 2.5, 0.0))
    )
    post = part.makeBox(5.0, 5.0, 4.0, _vector(part, 2.5, 2.5, -1.0))
    return [
        ("substrate", 3, slab),
        ("post", 3, post),
        ("walls", 2, part.Compound(slab.Faces + post.Faces)),
    ]


def _block_in_a_notch(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A slab with one quarter cut away, and a taller block standing in it."""
    slab = part.makeBox(10.0, 10.0, 2.0).cut(
        part.makeBox(5.0, 5.0, 2.0, _vector(part, 5.0, 5.0, 0.0))
    )
    block = part.makeBox(5.0, 5.0, 4.0, _vector(part, 5.0, 5.0, 0.0))
    return [
        ("substrate", 3, slab),
        ("block", 3, block),
        ("walls", 2, part.Compound(slab.Faces + block.Faces)),
    ]


def _speck_in_a_part(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [
        ("air", 3, box),
        ("part", 3, part.makeBox(8.0, 6.0, 4.0, _vector(part, 10.0, 7.0, 3.0))),
        ("speck", 3, _speck(part, SPECK)),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _speck_in_a_rod(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [
        ("air", 3, box),
        ("rod", 3, part.makeCylinder(2.0, 6.0, _vector(part, SPECK_AT[0], SPECK_AT[1], 1.0))),
        ("speck", 3, _speck(part, SPECK)),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _beside(part: Any) -> Any:
    """The small body again, against the side of the first one away from the origin."""
    return _speck(part, SPECK, (SPECK_AT[0] + SPECK, *SPECK_AT[1:]))


def _specks_touching(part: Any) -> Sequence[tuple[str, int, Any]]:
    return _in_the_domain(part, part.Compound([_speck(part, SPECK), _beside(part)]))


def _specks_touching_under_two_labels(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [
        ("air", 3, box),
        ("one", 3, _speck(part, SPECK)),
        ("two", 3, _beside(part)),
        ("walls", 2, part.Compound(box.Faces)),
    ]


#: A needle's length and radius in mm, laid along the diagonal of its own box. A
#: rod across its box fills a share of it of 3 * sqrt(3) * pi * (r / L) ** 2,
#: which here is below the mesher's ROUNDING, while its volume against its area
#: times its box's largest side is sqrt(3) * r / 2L and far above it.
NEEDLE_LENGTH = 1000.0
NEEDLE_RADIUS = 1e-4


def _needle(part: Any, speck: bool, reverse: bool) -> Sequence[tuple[str, int, Any]]:
    """A needle along the diagonal of a domain round it, with the small body at
    its middle or not, drawn the right way round or inside out."""
    side = NEEDLE_LENGTH / 3.0**0.5
    box = part.makeBox(side + 2.0, side + 2.0, side + 2.0, _vector(part, -1.0, -1.0, -1.0))
    needle = part.makeCylinder(
        NEEDLE_RADIUS, NEEDLE_LENGTH, _vector(part, 0.0, 0.0, 0.0), _vector(part, 1.0, 1.0, 1.0)
    )
    pieces = [("air", 3, box), ("needle", 3, needle.reversed() if reverse else needle)]
    if speck:
        pieces.append(("speck", 3, _speck(part, SPECK, (side / 2.0,) * 3)))
    return [*pieces, ("walls", 2, part.Compound(box.Faces))]


def _speck_in_a_needle(part: Any) -> Sequence[tuple[str, int, Any]]:
    return _needle(part, speck=True, reverse=False)


def _needle_inside_out(part: Any) -> Sequence[tuple[str, int, Any]]:
    return _needle(part, speck=False, reverse=True)


def _part_inside_out(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [
        ("air", 3, box),
        ("part", 3, part.makeBox(8.0, 6.0, 4.0, _vector(part, 10.0, 7.0, 3.0)).reversed()),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _domain_inside_out(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [("air", 3, box.reversed()), ("walls", 2, part.Compound(box.Faces))]


def _round_speck(part: Any) -> Sequence[tuple[str, int, Any]]:
    return _in_the_domain(part, part.makeSphere(SPECK / 2, _vector(part, *SPECK_AT)))


def _speck_in_one_compound(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A stray solid carried in the domain's own compound, as an imported part
    brings one along."""
    box = part.makeBox(30.0, 20.0, 10.0)
    return [
        ("air", 3, part.Compound([box, _speck(part, SPECK)])),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _blocks_along_an_edge(part: Any) -> Sequence[tuple[str, int, Any]]:
    one = part.makeBox(10.0, 10.0, 10.0)
    two = part.makeBox(10.0, 10.0, 10.0, _vector(part, 10.0 - SPECK, 10.0 - SPECK, 0.0))
    return [
        ("one", 3, one),
        ("two", 3, two),
        ("walls", 2, part.Compound(one.Faces + two.Faces)),
    ]


def _edge_drawn_twice(part: Any) -> Sequence[tuple[str, int, Any]]:
    """One edge of the box under two labels, which stands below both dimensions
    the coverage questions are asked in."""
    box = part.makeBox(30.0, 20.0, 10.0)
    edge = part.makeLine(_vector(part, 0.0, 0.0, 0.0), _vector(part, 0.0, 0.0, 10.0))
    return [
        ("air", 3, box),
        ("walls", 2, part.Compound(box.Faces)),
        ("wire", 1, part.Compound([edge])),
        ("probe", 1, part.Compound([edge])),
    ]


def _part_out_through_a_wall(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    through = part.makeBox(8.0, 6.0, 4.0, _vector(part, 26.0, 7.0, 3.0))
    return [
        ("air", 3, box),
        ("part", 3, through),
        ("walls", 2, part.Compound(box.Faces + through.Faces)),
    ]


def _parts_crossing(part: Any) -> Sequence[tuple[str, int, Any]]:
    one = part.makeBox(20.0, 20.0, 10.0)
    two = part.makeBox(20.0, 20.0, 10.0, _vector(part, 10.0, 10.0, 5.0))
    return [
        ("one", 3, one),
        ("two", 3, two),
        ("walls", 2, part.Compound(one.Faces + two.Faces)),
    ]


def _one_shape_drawn_twice(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [
        ("one", 3, box),
        ("two", 3, part.makeBox(30.0, 20.0, 10.0)),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _parts_overlapping_in_a_box(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    near = part.makeBox(10.0, 8.0, 5.0, _vector(part, 5.0, 4.0, 2.0))
    far = part.makeBox(10.0, 8.0, 5.0, _vector(part, 12.0, 8.0, 4.0))
    return [
        ("air", 3, box),
        ("near", 3, near),
        ("far", 3, far),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _air_filled_by_its_parts(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    lower = part.makeBox(30.0, 20.0, 4.0)
    upper = part.makeBox(30.0, 20.0, 6.0, _vector(part, 0.0, 0.0, 4.0))
    return [
        ("air", 3, box),
        ("substrate", 3, lower),
        ("cover", 3, upper),
        ("walls", 2, part.Compound(box.Faces)),
    ]


# ------------------------------------------------------------ what is refused


def _bare_walls(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [("air", 3, box), ("walls", 2, part.Compound(box.Faces[:3]))]


def _lone_box(part: Any) -> Sequence[tuple[str, int, Any]]:
    return [("air", 3, part.makeBox(30.0, 20.0, 10.0))]


def _boundary_port(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [
        ("air", 3, box),
        ("walls", 2, part.Compound(box.Faces)),
        ("port", 2, _face_at_x(part, 30.0, 0.0, 20.0, 0.0, 10.0)),
    ]


def _patch_port(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [
        ("air", 3, box),
        ("walls", 2, part.Compound(box.Faces)),
        ("port", 2, _face_at_x(part, 30.0, 5.0, 15.0, 2.0, 8.0)),
    ]


def _whisker_out(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [
        ("air", 3, box),
        ("walls", 2, part.Compound(box.Faces)),
        ("port", 2, _face_at_x(part, 15.0, 0.0, 20.0, 0.0, 10.0 + AN_OVERHANG)),
    ]


def _mixed_compound(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    loose = _face_at_x(part, 15.0, 0.0, 20.0, 0.0, 10.0)
    return [
        ("air", 3, part.Compound([box, loose])),
        ("walls", 2, part.Compound(box.Faces)),
    ]


def _embedded_port(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [
        ("air", 3, box),
        ("walls", 2, part.Compound(box.Faces)),
        ("port", 2, _face_at_x(part, 15.0, 0.0, 20.0, 0.0, 5.0)),
    ]


def _unlabelled_region(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [("walls", 2, part.Compound([*box.Faces, box]))]


def _compound_at_a_face(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [
        ("air", 3, box),
        ("walls", 2, part.Compound([*box.Faces, box])),
    ]


def _one_label_two_dimensions(part: Any) -> Sequence[tuple[str, int, Any]]:
    box = part.makeBox(30.0, 20.0, 10.0)
    return [("region", 3, box), ("region", 2, part.Compound(box.Faces))]


# ------------------------------------------ where a face handed a direction ends it

#: The guide the shipped WR-42 example draws, in mm: its length, its broad and
#: its narrow side, and how far in from each end its port planes stand.
GUIDE = (50.0, 10.7, 4.3)
GUIDE_INSET = 5.0

#: The directions a face is handed into the model, along x.
FORWARD = (1.0, 0.0, 0.0)
BACKWARD = (-1.0, 0.0, 0.0)


def _guide(part: Any) -> Any:
    length, broad, narrow = GUIDE
    return part.makeBox(length, broad, narrow)


def _across_the_guide(part: Any, x: float, inset: float = 0.0) -> Any:
    _, broad, narrow = GUIDE
    return _face_at_x(part, x, inset, broad - inset, inset, narrow - inset)


def _guide_run_on(part: Any) -> Sequence[tuple[str, int, Any]]:
    length = GUIDE[0]
    return [
        ("air", 3, _guide(part)),
        ("port1", 2, _across_the_guide(part, GUIDE_INSET)),
        ("port2", 2, _across_the_guide(part, length - GUIDE_INSET)),
    ]


def _plane_short_of_the_guide(part: Any) -> Sequence[tuple[str, int, Any]]:
    length = GUIDE[0]
    return [
        ("air", 3, _guide(part)),
        ("port1", 2, _across_the_guide(part, GUIDE_INSET, inset=1.0)),
        ("port2", 2, _across_the_guide(part, length - GUIDE_INSET)),
    ]


def _guide_ends(part: Any) -> Sequence[tuple[str, int, Any]]:
    length = GUIDE[0]
    return [
        ("air", 3, _guide(part)),
        ("port1", 2, _across_the_guide(part, 0.0)),
        ("port2", 2, _across_the_guide(part, length)),
    ]


def _port_where_two_bodies_meet(part: Any) -> Sequence[tuple[str, int, Any]]:
    length, broad, narrow = GUIDE
    half = length / 2.0
    return [
        ("air", 3, part.makeBox(half, broad, narrow)),
        ("air", 3, part.makeBox(half, broad, narrow, _vector(part, half, 0.0, 0.0))),
        ("port1", 2, _across_the_guide(part, half)),
        ("port2", 2, _across_the_guide(part, length)),
    ]


def _run_on_holds_a_slab(part: Any) -> Sequence[tuple[str, int, Any]]:
    length, broad, narrow = GUIDE
    slab = part.makeBox(2.0, broad, narrow, _vector(part, 1.0, 0.0, 0.0))
    return [
        ("air", 3, _guide(part)),
        ("slab", 3, slab),
        ("port1", 2, _across_the_guide(part, GUIDE_INSET)),
        ("port2", 2, _across_the_guide(part, length - GUIDE_INSET)),
    ]


def _walls_on_the_run_on(part: Any) -> Sequence[tuple[str, int, Any]]:
    length = GUIDE[0]
    guide = _guide(part)
    return [
        ("air", 3, guide),
        ("walls", 2, part.Compound(guide.Faces)),
        ("port1", 2, _across_the_guide(part, GUIDE_INSET)),
        ("port2", 2, _across_the_guide(part, length - GUIDE_INSET)),
    ]


def _loaded_guide_run_on(part: Any) -> Sequence[tuple[str, int, Any]]:
    """The guide with a slab filling half its width along the whole of it."""
    length, broad, narrow = GUIDE
    return [
        ("air", 3, _guide(part)),
        ("slab", 3, part.makeBox(length, broad / 2.0, narrow)),
        ("port1", 2, _across_the_guide(part, GUIDE_INSET)),
        ("port2", 2, _across_the_guide(part, length - GUIDE_INSET)),
    ]


def _iris_behind_a_reversed_port(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A guide narrowed by an iris halfway along, with its one port facing out."""
    length, broad, narrow = GUIDE
    iris = part.makeBox(1.0, broad / 3.0, narrow, _vector(part, length / 2.0, 0.0, 0.0))
    return [
        ("air", 3, _guide(part).cut(iris)),
        ("port1", 2, _across_the_guide(part, GUIDE_INSET)),
    ]


def _jog_in_the_run_on(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A run-on of the port's own cross-section, set sideways: its volume is the
    face's area times its depth, and its shape is not the face carried on."""
    model = part.makeBox(40.0, 20.0, 10.0)
    run_on = part.makeBox(30.0, 20.0, 10.0, _vector(part, -30.0, 3.0, 0.0))
    return [
        ("air", 3, model.fuse(run_on).removeSplitter()),
        ("port1", 2, _face_at_x(part, 0.0, 0.0, 20.0, 0.0, 10.0)),
    ]


def _slab_stops_in_the_run_on(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A slab loading half the guide that stops short of its ends, so each
    run-on is slab and then air where the guide in front is slab throughout."""
    length, broad, narrow = GUIDE
    return [
        ("air", 3, _guide(part)),
        ("slab", 3, part.makeBox(length - 4.0, broad / 2.0, narrow, _vector(part, 2.0, 0.0, 0.0))),
        ("port1", 2, _across_the_guide(part, GUIDE_INSET)),
        ("port2", 2, _across_the_guide(part, length - GUIDE_INSET)),
    ]


def _straight_guide_behind_a_reversed_port(part: Any) -> Sequence[tuple[str, int, Any]]:
    return [
        ("air", 3, _guide(part)),
        ("port1", 2, _across_the_guide(part, GUIDE_INSET)),
    ]


def _point_in_the_run_on(part: Any) -> Sequence[tuple[str, int, Any]]:
    length = GUIDE[0]
    return [
        ("air", 3, _guide(part)),
        ("probe", 0, part.Vertex(_vector(part, 2.0, 2.0, 2.0))),
        ("port1", 2, _across_the_guide(part, GUIDE_INSET)),
        ("port2", 2, _across_the_guide(part, length - GUIDE_INSET)),
    ]


def _frame_round_the_plane(part: Any) -> Sequence[tuple[str, int, Any]]:
    """A square ring, cut across one side. The two sides of the plane are one
    region, joined round the ring."""
    frame = part.makeBox(30.0, 30.0, 10.0).cut(
        part.makeBox(10.0, 10.0, 10.0, _vector(part, 10.0, 10.0, 0.0))
    )
    corners = [
        _vector(part, 0.0, 15.0, 0.0),
        _vector(part, 10.0, 15.0, 0.0),
        _vector(part, 10.0, 15.0, 10.0),
        _vector(part, 0.0, 15.0, 10.0),
        _vector(part, 0.0, 15.0, 0.0),
    ]
    return [
        ("air", 3, frame),
        ("port1", 2, part.Face(part.Wire(part.makePolygon(corners)))),
    ]


#: The coaxial line's radii and length, in mm, and where along it the port
#: rings stand.
COAX_LINE = (3.5, 1.5, 30.0)
COAX_INSET = 4.0


def _ring(part: Any, z: float) -> Any:
    outer, inner, _ = COAX_LINE
    at = _vector(part, 0.0, 0.0, z)
    along = _vector(part, 0.0, 0.0, 1.0)
    disk = part.Face(part.Wire(part.makeCircle(outer, at, along)))
    hole = part.Face(part.Wire(part.makeCircle(inner, at, along)))
    return disk.cut(hole)


def _coax_run_on(part: Any) -> Sequence[tuple[str, int, Any]]:
    outer, inner, length = COAX_LINE
    line = part.makeCylinder(outer, length).cut(part.makeCylinder(inner, length))
    return [
        ("dielectric", 3, line),
        ("port1", 2, _ring(part, COAX_INSET)),
        ("port2", 2, _ring(part, length - COAX_INSET)),
    ]


def drawings() -> tuple[Drawing, ...]:
    """The corpus. What meshes first, then one drawing per refusal."""
    return (
        Drawing(
            "plain_box",
            _plain_box,
            "a wall label drawn as the solid's own faces, with nothing cutting them",
            declares=(("air", 3), ("walls", 2)),
            tags=("whole",),
        ),
        Drawing(
            "split_cavity",
            _split_cavity,
            "a face cutting the volume in two, so both labels follow into more pieces "
            "than were drawn",
            declares=(("air", 3), ("walls", 2), ("port", 2)),
            tags=("cut",),
        ),
        Drawing(
            "shared_interface",
            _shared_interface,
            "two solids meeting on a face, which a volume label cannot claim and the "
            "wall label must not",
            declares=(("substrate", 3), ("air", 3), ("walls", 2)),
            tags=("cut",),
        ),
        Drawing(
            "crossing_port",
            _crossing_port,
            "one drawn face crossing two solids, arriving as a piece in each",
            declares=(("substrate", 3), ("air", 3), ("walls", 2), ("port", 2)),
            tags=("cut",),
        ),
        Drawing(
            "coax",
            _coax,
            "a curved interface between two solids, where the wall label is drawn from "
            "the faces of both",
            declares=(("dielectric", 3), ("conductor", 3), ("walls", 2)),
            loses=CHORDS,
            tags=("curved",),
        ),
        Drawing(
            "skin_sealing_a_cavity",
            _skin_sealing_a_cavity,
            "a closed skin standing inside what is filled, which the faces still "
            "join and a condition on them does not",
            declares=(("air", 3), ("skin", 2), ("walls", 2)),
            divides=("skin",),
            parts=2,
            tags=("parts",),
        ),
        Drawing(
            "skin_across_the_box",
            _skin_across_the_box,
            "a skin standing in what is filled that the region reaches past, so "
            "dividing on it leaves the parts as they were",
            declares=(("air", 3), ("skin", 2), ("walls", 2)),
            divides=("skin",),
            parts=1,
            tags=("parts",),
        ),
        Drawing(
            "post_in_the_box",
            _post_in_the_box,
            "a body that leaves the model standing clear of every wall, so every "
            "face it leaves behind is where the model ends",
            declares=(("air", 3), ("post", 3), ("walls", 2)),
            priority=(("post", 1),),
            leaves=("post",),
            tags=("leaves",),
        ),
        Drawing(
            "post_at_the_regions_priority",
            _post_in_the_box,
            "a body that leaves the model stated at the priority of the region it "
            "stands in, so nothing says which of them keeps what they share",
            declares=(("air", 3), ("post", 3), ("walls", 2)),
            expect=REFUSED,
            complaint="claim one piece",
            leaves=("post",),
            tags=("leaves",),
        ),
        Drawing(
            "plate_standing_in_the_box",
            _plate_standing_in_the_box,
            "a body that leaves the model lying on two walls, so what the walls were "
            "drawn over under it leaves with it",
            declares=(("air", 3), ("plate", 3), ("walls", 2)),
            priority=(("plate", 1),),
            leaves=("plate",),
            tags=("leaves",),
        ),
        Drawing(
            "plate_across_the_box",
            _plate_across_the_box,
            "a body that leaves the model and fills the cross-section, so the region "
            "is two once it has gone and the drawing still joined them through it",
            declares=(("air", 3), ("plate", 3), ("walls", 2)),
            priority=(("plate", 1),),
            leaves=("plate",),
            divides=("plate",),
            parts=2,
            tags=("leaves", "parts"),
        ),
        Drawing(
            "cone_on_its_apex",
            _cone_on_its_apex,
            "a cone standing on its apex, whose surface Gmsh fails to mesh near the "
            "apex at an element far coarser than the cone, sized round each turn",
            declares=(("air", 3), ("cone", 3), ("walls", 2)),
            expect=UNMESHED,
            complaint="It stopped at a surface of 'air', 'cone' at (",
            priority=(("cone", 1),),
            element_order=2,
            curved=True,
            coarsest=14.3,
            finest=0.084,
            per_turn=6,
            tags=("ordeal", "curved"),
        ),
        Drawing(
            "curve_inside_a_post",
            _curve_inside_a_post,
            "a curve wholly inside a body that leaves the model, which the kernel "
            "fails to cut when the file marks the body as part of another shape",
            declares=(("air", 3), ("post", 3), ("curve", 1), ("walls", 2)),
            expect=UNMESHED,
            complaint="could not cut the shapes drawn for 'air', 'curve', 'post', 'walls'",
            priority=(("post", 1),),
            leaves=("post",),
            tags=("leaves", "ordeal", "kernel_fails"),
        ),
        Drawing(
            "post_through_a_sheet",
            _post_through_a_sheet,
            "a sheet passing through a body that leaves the model, whose part inside "
            "the body leaves with it",
            declares=(("air", 3), ("post", 3), ("sheet", 2), ("walls", 2)),
            priority=(("post", 1),),
            leaves=("post",),
            tags=("leaves",),
        ),
        Drawing(
            "sheet_inside_a_post",
            _sheet_inside_a_post,
            "a sheet standing wholly inside a body that leaves the model, so nothing "
            "of it is left to carry its label",
            declares=(("air", 3), ("post", 3), ("sheet", 2), ("walls", 2)),
            expect=REFUSED,
            complaint="wholly inside",
            priority=(("post", 1),),
            leaves=("post",),
            tags=("leaves",),
        ),
        Drawing(
            "block_inside_a_post",
            _block_inside_a_post,
            "a region drawn wholly inside a body that leaves the model, which keeps none of it",
            declares=(("air", 3), ("post", 3), ("block", 3), ("walls", 2)),
            expect=REFUSED,
            complaint="kept nothing of what it was drawn over",
            priority=(("post", 1),),
            leaves=("post",),
            tags=("leaves",),
        ),
        Drawing(
            "post_under_a_higher_block",
            _post_under_a_higher_block,
            "a body that leaves the model overlapped by a region stated above it, "
            "which keeps what they share, and the air under both",
            declares=(("air", 3), ("post", 3), ("block", 3), ("walls", 2)),
            priority=(("post", 1), ("block", 2)),
            leaves=("post",),
            tags=("leaves",),
        ),
        Drawing(
            "sheet_on_a_post_face",
            _sheet_on_a_post_face,
            "a sheet drawn on a face of a body that leaves the model, which the sheet keeps",
            declares=(("air", 3), ("post", 3), ("sheet", 2), ("walls", 2)),
            priority=(("post", 1),),
            leaves=("post",),
            tags=("leaves",),
        ),
        Drawing(
            "post_in_its_own_skin",
            _post_in_its_own_skin,
            "a body that leaves the model with every face of it drawn under another "
            "label, so none is left for it",
            declares=(("air", 3), ("post", 3), ("skin", 2), ("walls", 2)),
            expect=REFUSED,
            complaint="was drawn under another label",
            priority=(("post", 1),),
            leaves=("post",),
            tags=("leaves",),
        ),
        Drawing(
            "post_beside_the_box",
            _post_beside_the_box,
            "a body that leaves the model standing clear of what is filled, so it "
            "leaves no face behind",
            declares=(("air", 3), ("post", 3), ("walls", 2)),
            expect=REFUSED,
            complaint="stands against nothing that stays",
            apart=True,
            leaves=("post",),
            tags=("leaves",),
        ),
        Drawing(
            "post_through_the_wall",
            _post_through_the_wall,
            "a body that leaves the model passing out through a wall, so what stood "
            "outside it goes and the wall under it goes with it",
            declares=(("air", 3), ("post", 3), ("walls", 2)),
            priority=(("post", 1),),
            leaves=("post",),
            tags=("leaves",),
        ),
        Drawing(
            "two_posts_across_a_sheet",
            _two_posts_across_a_sheet,
            "two bodies that leave the model touching on a face, with a sheet across "
            "both, so the face between them goes and the sheet loses a part to each",
            declares=(("air", 3), ("left", 3), ("right", 3), ("sheet", 2), ("walls", 2)),
            priority=(("left", 1), ("right", 1)),
            leaves=("left", "right"),
            tags=("leaves",),
        ),
        Drawing(
            "two_posts_overlapping",
            _two_posts_overlapping,
            "two bodies that leave the model drawn over each other at one priority, "
            "so nothing says which keeps what they share",
            declares=(("air", 3), ("left", 3), ("right", 3), ("walls", 2)),
            expect=REFUSED,
            complaint="claim one piece",
            priority=(("left", 1), ("right", 1)),
            leaves=("left", "right"),
            tags=("leaves",),
        ),
        Drawing(
            "closed_shell",
            _closed_shell,
            "the surface of a closed body and nothing inside it, which is what a "
            "method that never fills a volume asks for",
            declares=(("skin", 2),),
            top=2,
            tags=("surface",),
        ),
        Drawing(
            "sphere_skin",
            _sphere_skin,
            "a closed surface bounded by two curves of no length, which are listed "
            "once apiece and read exactly like a curve where the model ends",
            declares=(("skin", 2),),
            top=2,
            loses=CHORDS,
            tags=("surface", "curved"),
        ),
        Drawing(
            "sphere_seam",
            _sphere_seam,
            "a closed surface with the curve it closes on labelled, which the surface "
            "stands on both sides of",
            declares=(("skin", 2), ("seam", 1)),
            top=2,
            loses=CHORDS,
            tags=("surface", "curved"),
        ),
        Drawing(
            "cavity",
            _cavity,
            "a void enclosed by the body, whose walls bound one volume each and are "
            "not on the outside of anything",
            declares=(("air", 3), ("walls", 2)),
            tags=("whole",),
        ),
        Drawing(
            "part_in_a_box",
            _part_in_a_box,
            "a part standing inside the domain that holds it, which is the shape of an "
            "everyday drawing and where both labels are drawn over the part's space",
            declares=(("air", 3), ("part", 3), ("walls", 2)),
            priority=(("part", 1),),
            tags=("inside",),
        ),
        Drawing(
            "part_flush_with_a_wall",
            _part_flush_with_a_wall,
            "the same part moved against a wall, so the two shapes share a face as well "
            "as a region",
            declares=(("air", 3), ("part", 3), ("walls", 2)),
            priority=(("part", 1),),
            tags=("inside",),
        ),
        Drawing(
            "nested_three_deep",
            _nested_three_deep,
            "a part inside a case inside the domain, where the innermost region is drawn "
            "over by all three labels",
            declares=(("air", 3), ("case", 3), ("part", 3), ("walls", 2)),
            priority=(("case", 1), ("part", 2)),
            tags=("inside",),
        ),
        Drawing(
            "round_part_in_a_box",
            _round_part_in_a_box,
            "a curved part inside the domain, whose interface the mesh approximates by "
            "chords and whose shortfall moves from one label to the other",
            declares=(("air", 3), ("rod", 3), ("walls", 2)),
            priority=(("rod", 1),),
            loses=CHORDS,
            tags=("inside", "curved"),
        ),
        Drawing(
            "two_parts_in_one_box",
            _two_parts_in_one_box,
            "two parts apart inside one domain, which share no region with each other "
            "and so state one priority between them",
            declares=(("air", 3), ("near", 3), ("far", 3), ("walls", 2)),
            priority=(("near", 1), ("far", 1)),
            tags=("inside",),
        ),
        Drawing(
            "speck_in_a_box",
            _speck_in_a_box,
            "a box-shaped body below the size the kernel cuts, standing free inside the "
            "domain, which the fragmenting leaves where it was drawn",
            declares=(("air", 3), ("speck", 3), ("walls", 2)),
            priority=(("speck", 1),),
            expect=REFUSED,
            complaint="did not cut it out",
            apart=True,
            tags=("inside", "speck", "uncut"),
        ),
        Drawing(
            "speck_in_one_compound",
            _speck_in_one_compound,
            "the same body carried in the domain's own compound, where no second label "
            "is drawn over it",
            declares=(("air", 3), ("walls", 2)),
            expect=REFUSED,
            complaint="did not cut it out",
            apart=True,
            tags=("inside", "speck", "uncut"),
        ),
        Drawing(
            "speck_in_a_corner",
            _speck_in_a_corner,
            "the free-standing body in a corner of the domain, as far from three walls "
            "as it is across, which is left uncut like the rest",
            declares=(("air", 3), ("speck", 3), ("walls", 2)),
            priority=(("speck", 1),),
            expect=REFUSED,
            complaint="did not cut it out",
            apart=True,
            tags=("inside", "speck", "uncut"),
        ),
        Drawing(
            "rod_speck",
            _rod_speck,
            "a cylinder a nanometre tall, which the kernel cuts out",
            declares=(("air", 3), ("speck", 3), ("walls", 2)),
            priority=(("speck", 1),),
            loses=CHORDS,
            # Small against its distance from the origin, which the file's digits go
            # on, as they go on the distance in the drawing far from it.
            round_trip=1e-5,
            tags=("inside", "speck", "curved"),
        ),
        Drawing(
            "slab_in_a_slot",
            _slab_in_a_slot,
            "a slab standing free in a slot a hair from its floor and roof, where each "
            "broad face of the slab has the block close beyond it",
            declares=(("block", 3), ("slab", 3), ("walls", 2)),
            apart=True,
            # Thin against its height above the origin, which the file's digits go
            # on, as they go on the distance in the drawing far from it.
            round_trip=1e-5,
            tags=("inside", "sliver"),
        ),
        Drawing(
            "plug_in_a_pocket",
            _plug_in_a_pocket,
            "the small body at the bottom of a pocket, sharing its sides with the block, "
            "with its top face on the outside of the model and ringed by the pocket's walls",
            declares=(("block", 3), ("plug", 3), ("walls", 2)),
            tags=("inside", "speck"),
        ),
        Drawing(
            "ring_on_a_post",
            _ring_on_a_post,
            "a ring round a post on a base, whose centre is inside the post",
            declares=(("holder", 3), ("ring", 3), ("walls", 2)),
            loses=CHORDS,
            tags=("inside", "curved"),
        ),
        Drawing(
            "bent_trace",
            _bent_trace,
            "an L of thick trace cut out of the air on a substrate, where the face the "
            "substrate shows under the bend is an L",
            declares=(("substrate", 3), ("air", 3), ("walls", 2)),
            tags=("inside", "sliver"),
        ),
        Drawing(
            "copper_bend",
            _copper_bend,
            "an L of copper drawn as a body, whose centre falls on its own face at the "
            "whole millimetres it is drawn to",
            declares=(("substrate", 3), ("air", 3), ("copper", 3), ("walls", 2)),
            tags=("inside", "sliver"),
        ),
        Drawing(
            "copper_bend_on_its_edge",
            _copper_bend_on_its_edge,
            "an L of copper drawn as a body, whose centre falls on the edge of its inner corner",
            declares=(("substrate", 3), ("air", 3), ("copper", 3), ("walls", 2)),
            tags=("inside", "sliver"),
        ),
        Drawing(
            "speck_at_the_middle_of_a_ball",
            _speck_at_the_middle_of_a_ball,
            "the small body left uncut at the centre of a ball of air, whose own centre "
            "is inside the body",
            declares=(("air", 3), ("speck", 3), ("walls", 2)),
            priority=(("speck", 1),),
            expect=REFUSED,
            complaint="did not cut it out",
            apart=True,
            tags=("inside", "speck", "uncut"),
        ),
        Drawing(
            "speck_in_a_part",
            _speck_in_a_part,
            "the small body inside a part inside the domain, where the kernel returns "
            "the domain whole beside the part and a copy of the part turned inside out",
            declares=(("air", 3), ("part", 3), ("speck", 3), ("walls", 2)),
            priority=(("part", 1), ("speck", 2)),
            expect=REFUSED,
            complaint="a piece turned inside out there",
            apart=True,
            tags=("inside", "speck", "turned"),
        ),
        Drawing(
            "speck_in_a_rod",
            _speck_in_a_rod,
            "the same body inside a cylinder standing in the domain",
            declares=(("air", 3), ("rod", 3), ("speck", 3), ("walls", 2)),
            priority=(("rod", 1), ("speck", 2)),
            expect=REFUSED,
            complaint="a piece turned inside out there",
            apart=True,
            tags=("inside", "speck", "turned", "curved"),
        ),
        Drawing(
            "specks_touching",
            _specks_touching,
            "two small bodies touching face to face under one label, standing free in "
            "the domain, which come back beside a copy of their space turned inside out",
            declares=(("air", 3), ("speck", 3), ("walls", 2)),
            priority=(("speck", 1),),
            expect=REFUSED,
            complaint="a piece turned inside out there",
            apart=True,
            tags=("inside", "speck", "turned"),
        ),
        Drawing(
            "specks_touching_under_two_labels",
            _specks_touching_under_two_labels,
            "the same two bodies, each under a label of its own",
            declares=(("air", 3), ("one", 3), ("two", 3), ("walls", 2)),
            priority=(("one", 1), ("two", 2)),
            expect=REFUSED,
            complaint="a piece turned inside out there",
            apart=True,
            tags=("inside", "speck", "turned"),
        ),
        Drawing(
            "speck_in_a_needle",
            _speck_in_a_needle,
            "the small body at the middle of a needle laid across its box, whose piece "
            "turned inside out fills almost none of the box",
            declares=(("air", 3), ("needle", 3), ("speck", 3), ("walls", 2)),
            priority=(("needle", 1), ("speck", 2)),
            expect=REFUSED,
            complaint="a piece turned inside out there",
            apart=True,
            tags=("inside", "speck", "turned", "curved"),
        ),
        Drawing(
            "needle_inside_out",
            _needle_inside_out,
            "the needle drawn inside out, which fills almost none of its box",
            declares=(("air", 3), ("needle", 3), ("walls", 2)),
            priority=(("needle", 1),),
            expect=REFUSED,
            complaint="is wound inside out",
            tags=("inside_out",),
        ),
        Drawing(
            "part_inside_out",
            _part_inside_out,
            "a part inside the domain drawn with its faces pointing inward, which the "
            "kernel reads as the space outside it",
            declares=(("air", 3), ("part", 3), ("walls", 2)),
            priority=(("part", 1),),
            expect=REFUSED,
            complaint="is wound inside out",
            tags=("inside_out",),
        ),
        Drawing(
            "domain_inside_out",
            _domain_inside_out,
            "the domain alone drawn with its faces pointing inward",
            declares=(("air", 3), ("walls", 2)),
            expect=REFUSED,
            complaint="is wound inside out",
            tags=("inside_out",),
        ),
        Drawing(
            "lid_over_a_groove",
            _lid_over_a_groove,
            "a lid over a ring groove, whose face over the groove is a ring 0.4 mm wide "
            "with the block round both its rims",
            declares=(("block", 3), ("lid", 3), ("walls", 2)),
            loses=CHORDS,
            tags=("inside", "curved"),
        ),
        Drawing(
            "post_through_a_slab",
            _post_through_a_slab,
            "a post filling a hole through a slab, where a point on the rim of a face "
            "of one is on the other",
            declares=(("substrate", 3), ("post", 3), ("walls", 2)),
            tags=("inside",),
        ),
        Drawing(
            "block_in_a_notch",
            _block_in_a_notch,
            "a block standing in a notch cut out of a slab, where a point on the rim of "
            "a face of one is on the other",
            declares=(("substrate", 3), ("block", 3), ("walls", 2)),
            tags=("inside",),
        ),
        Drawing(
            "speck_cut_out",
            _speck_cut_out,
            "the body just large enough for the kernel to cut it out",
            declares=(("air", 3), ("speck", 3), ("walls", 2)),
            priority=(("speck", 1),),
            tags=("inside", "speck"),
        ),
        Drawing(
            "speck_cut_out_at_the_middle",
            _speck_cut_out_at_the_middle,
            "the body just large enough to be cut out, with a corner at the middle of the "
            "domain, on which the centre of the domain less the body falls",
            declares=(("air", 3), ("speck", 3), ("walls", 2)),
            priority=(("speck", 1),),
            tags=("inside", "speck"),
        ),
        Drawing(
            "speck_against_a_wall",
            _speck_against_a_wall,
            "the small body moved against a wall, where the kernel cuts it out and one "
            "face of it is on the outside of the model",
            declares=(("air", 3), ("speck", 3), ("walls", 2)),
            priority=(("speck", 1),),
            tags=("inside", "speck"),
        ),
        Drawing(
            "round_speck",
            _round_speck,
            "a ball as small as the body, which the kernel cuts out",
            declares=(("air", 3), ("speck", 3), ("walls", 2)),
            priority=(("speck", 1),),
            loses=CHORDS,
            tags=("inside", "speck", "curved"),
        ),
        Drawing(
            "blocks_along_an_edge",
            _blocks_along_an_edge,
            "two blocks overlapping along an edge by as little as the small body is "
            "across, which the kernel cuts into a needle both are drawn over",
            declares=(("one", 3), ("two", 3), ("walls", 2)),
            priority=(("one", 1),),
            tags=("inside", "sliver"),
        ),
        Drawing(
            "air_filled_by_its_parts",
            _air_filled_by_its_parts,
            "a domain label its own contents leave nothing of, which is a group with no "
            "element in it and a caller finding nothing under its own name",
            declares=(("air", 3), ("substrate", 3), ("cover", 3), ("walls", 2)),
            priority=(("substrate", 1), ("cover", 1)),
            expect=REFUSED,
            complaint="kept nothing",
            tags=("inside",),
        ),
        Drawing(
            "bare_walls",
            _bare_walls,
            "part of the outside of the model left in no label",
            declares=(("air", 3), ("walls", 2)),
            expect=REFUSED,
            complaint="the model ends",
            tags=("coverage",),
        ),
        Drawing(
            "unlabelled_region",
            _unlabelled_region,
            "a file whose solid the declaration drops, leaving a region in the model "
            "that no label covers",
            declares=(("walls", 2),),
            expect=REFUSED,
            complaint="no label covers",
            tags=("coverage",),
        ),
        Drawing(
            "lone_box",
            _lone_box,
            "one drawn shape, which is the drawing the fragmenting returns no map for",
            declares=(("air", 3),),
            expect=REFUSED,
            complaint="the model ends",
            tags=("coverage",),
        ),
        Drawing(
            "boundary_port",
            _boundary_port,
            "a port face on a wall the wall label covers too, where the order the caller "
            "states leaves the wall label the other five",
            declares=(("air", 3), ("walls", 2), ("port", 2)),
            priority=(("port", 1),),
            tags=("contest",),
        ),
        Drawing(
            "patch_port",
            _patch_port,
            "the same over part of a wall, where the wall becomes the patch and the "
            "remainder and the wall label keeps the remainder",
            declares=(("air", 3), ("walls", 2), ("port", 2)),
            priority=(("port", 1),),
            tags=("contest",),
        ),
        Drawing(
            "edge_drawn_twice",
            _edge_drawn_twice,
            "one curve under two labels, standing below the dimension the coverage "
            "questions are asked in, where a tag alone reaches nothing the user drew",
            declares=(("air", 3), ("walls", 2), ("wire", 1), ("probe", 1)),
            expect=REFUSED,
            complaint="at dimension 1",
            tags=("contest",),
        ),
        Drawing(
            "part_out_through_a_wall",
            _part_out_through_a_wall,
            "a part crossing a wall of the box it stands in, so each of the two labels "
            "holds a piece the other does not and neither region is inside the other",
            declares=(("air", 3), ("part", 3), ("walls", 2)),
            expect=REFUSED,
            complaint="the regions cross",
            tags=("contest",),
        ),
        Drawing(
            "parts_crossing",
            _parts_crossing,
            "two bodies overlapping in a corner, neither of them inside the other",
            declares=(("one", 3), ("two", 3), ("walls", 2)),
            expect=REFUSED,
            complaint="the regions cross",
            tags=("contest",),
        ),
        Drawing(
            "one_shape_drawn_twice",
            _one_shape_drawn_twice,
            "one region under two labels, where each stands inside the other and "
            "neither is the inner one",
            declares=(("one", 3), ("two", 3), ("walls", 2)),
            expect=REFUSED,
            complaint="one region is drawn under every one of them",
            tags=("contest",),
        ),
        Drawing(
            "parts_overlapping_in_a_box",
            _parts_overlapping_in_a_box,
            "two parts in one box that also overlap each other, where the order settles "
            "each against the domain and leaves the piece they share",
            declares=(("air", 3), ("near", 3), ("far", 3), ("walls", 2)),
            priority=(("near", 1), ("far", 1)),
            expect=REFUSED,
            complaint="far, near all claim",
            tags=("contest",),
        ),
        Drawing(
            "whisker_out",
            _whisker_out,
            "a port drawn taller than the solid, so a piece of it stands outside and "
            "bounds nothing",
            declares=(("air", 3), ("walls", 2), ("port", 2)),
            expect=REFUSED,
            complaint="bound nothing",
            tags=("outside",),
        ),
        Drawing(
            "embedded_port",
            _embedded_port,
            "a face inside the solid that stops short of dividing it, which the kernel "
            "embeds in the solid rather than listing in its boundary, and which the "
            "solid is meshed to on both sides",
            declares=(("air", 3), ("walls", 2), ("port", 2)),
            tags=("inside",),
        ),
        Drawing(
            "mixed_compound",
            _mixed_compound,
            "one file holding a solid and a loose face, where the face is filtered out "
            "by the declared dimension and would go without a word",
            declares=(("air", 3), ("walls", 2)),
            expect=REFUSED,
            complaint="does not declare",
            tags=("dropped",),
        ),
        Drawing(
            "compound_at_a_face",
            _compound_at_a_face,
            "the same loss the other way up: a file declared at the lower dimension "
            "whose solid the declaration drops",
            declares=(("air", 3), ("walls", 2)),
            expect=REFUSED,
            complaint="does not declare",
            tags=("dropped",),
        ),
        Drawing(
            "solid_dropped_inside",
            _solid_dropped_inside,
            "a curve label's file carrying a solid inside the domain, which the "
            "declaration drops and the fragmenting never sees",
            declares=(("air", 3), ("walls", 2), ("wire", 1)),
            expect=REFUSED,
            complaint="does not declare",
            apart=True,
            tags=("dropped",),
        ),
        Drawing(
            "one_label_two_dimensions",
            _one_label_two_dimensions,
            "one label put on a solid and on its faces, which one group cannot hold",
            declares=(("region", 3), ("region", 2)),
            expect=REFUSED,
            complaint="two labels",
            tags=("dimension",),
        ),
        Drawing(
            "touching_at_an_edge",
            _touching_at_an_edge,
            "a body touching itself along one edge and nowhere else, which no face of it separates",
            declares=(("air", 3), ("walls", 2)),
            apart=True,
            tags=("ordeal", "non-manifold"),
        ),
        Drawing(
            "touching_at_a_vertex",
            _touching_at_a_vertex,
            "the same contact reduced to a point, where the two halves meet at one "
            "node of the mesh",
            declares=(("air", 3), ("walls", 2)),
            apart=True,
            tags=("ordeal", "non-manifold"),
        ),
        Drawing(
            "sliver_slab",
            _sliver_slab,
            "a slab four orders thinner than it is wide, so the element that fills it "
            "cannot be well shaped at any size",
            declares=(("air", 3), ("walls", 2)),
            tags=("ordeal", "sliver"),
        ),
        Drawing(
            "sliver_wedge",
            _sliver_wedge,
            "a body closing to a near-zero edge, so its faces run out to slivers "
            "wherever the element size is put",
            declares=(("air", 3), ("walls", 2)),
            tags=("ordeal", "sliver"),
        ),
        Drawing(
            "small_and_far_apart",
            _small_and_far_apart,
            "a body a tenth of a millimetre across a million millimetres from another, "
            "whose faces stand below the rounding on the model's own extent",
            declares=(("air", 3), ("air", 3), ("walls", 2)),
            finest=0.0,
            apart=True,
            tags=("ordeal", "far", "small"),
        ),
        Drawing(
            "far_from_the_origin",
            _far_from_the_origin,
            "a body a thousand kilometres of millimetres out, where every coordinate "
            "carries eight fewer digits of the shape",
            declares=(("air", 3), ("walls", 2)),
            round_trip=1e-5,
            tags=("ordeal", "far"),
        ),
        Drawing(
            "below_the_element",
            _below_the_element,
            "a body four orders below the element asked for, which the drawing bounds "
            "and the request cannot",
            declares=(("air", 3), ("walls", 2)),
            finest=0.0,
            tags=("ordeal", "small"),
        ),
        Drawing(
            "apart_below_tolerance",
            _apart_below_tolerance,
            "two bodies a millionth of a millimetre apart under one label, which the "
            "fragmenting neither joins nor drops",
            declares=(("air", 3), ("air", 3), ("walls", 2)),
            apart=True,
            tags=("ordeal", "near"),
        ),
        Drawing(
            "curved_body",
            _curved_body,
            "a cylinder at an element size near its own radius, which is what a "
            "straight-sided mesh of a curved body costs",
            declares=(("air", 3), ("walls", 2)),
            loses=CHORDS,
            tags=("ordeal", "curved"),
        ),
        Drawing(
            "self_intersecting",
            _self_intersecting,
            "a body that passes through itself, which fragments with a warning and "
            "meshes to a surface and no tetrahedra at all",
            declares=(("air", 3), ("walls", 2)),
            expect=UNMESHED,
            complaint="no element in them",
            tags=("ordeal", "invalid"),
        ),
        Drawing(
            "partly_unmeshed",
            _partly_unmeshed,
            "a sound body beside one that passes through itself, where the mesh is "
            "of half the drawing and the model as a whole is not empty",
            declares=(("air", 3), ("air", 3), ("walls", 2)),
            expect=UNMESHED,
            complaint="no element in them",
            apart=True,
            tags=("ordeal", "invalid"),
        ),
        Drawing(
            "rim_far_off",
            _rim_far_off,
            "a small open shell whose rim is real and shorter than the rounding on "
            "the whole model, because something else stands a long way off",
            declares=(("skin", 2), ("rim", 1), ("far", 2)),
            top=2,
            coarsest=1.0,
            finest=0.0,
            apart=True,
            tags=("ordeal", "far", "small", "open"),
        ),
        Drawing(
            "hidden_interface",
            _hidden_interface,
            "two solids on a shared face with only the outside labelled, so the "
            "interface is meshed and not written",
            declares=(("substrate", 3), ("air", 3), ("walls", 2)),
            tags=("ordeal", "cut"),
        ),
        Drawing(
            "curving_turns_elements_over",
            _curving_turns_elements_over,
            "a ring at an element size near its own thickness, where placing the "
            "added nodes on the drawn surface turns elements inside out",
            declares=(("air", 3), ("walls", 2)),
            expect=UNMESHED,
            complaint="turned inside out",
            element_order=2,
            curved=True,
            coarsest=30.0,
            finest=0.0,
            tags=("ordeal", "curved"),
        ),
        Drawing(
            "curving_pass_fails",
            _curving_pass_fails,
            "a thinner ring, where the pass that pulls back what curving turned "
            "over fails and says so",
            declares=(("air", 3), ("walls", 2)),
            expect=UNMESHED,
            complaint="critical value",
            element_order=2,
            curved=True,
            coarsest=30.0,
            finest=0.0,
            tags=("ordeal", "curved"),
        ),
        Drawing(
            "unsewn_shell",
            _unsewn_shell,
            "a solid made from faces that do not meet, whose walls bound nothing "
            "because there is nothing for them to bound",
            declares=(("air", 3), ("walls", 2)),
            expect=REFUSED,
            complaint="bound nothing",
            tags=("ordeal", "invalid"),
        ),
        Drawing(
            "open_shell",
            _open_shell,
            "a shell that does not close, refused at the rim where the surface ends "
            "and no label claims it",
            declares=(("skin", 2),),
            expect=REFUSED,
            complaint="the model ends",
            top=2,
            tags=("ordeal", "open"),
        ),
    )


#: The directions a coaxial line's rings are handed, along z.
UP = (0.0, 0.0, 1.0)
DOWN = (0.0, 0.0, -1.0)


def ends() -> tuple[Drawing, ...]:
    """Drawings that hand a label a way into the model.

    Kept apart from :func:`drawings`, whose every meshed drawing is held to the
    mesh filling what was drawn: here what stands behind a face is left out on
    purpose, so each is meshed short of its own measure. Every one names a
    remainder for what the model ends at, since what bounds a run-on is left out
    with it.
    """
    guide = (("air", 3), ("port1", 2), ("port2", 2))
    return (
        Drawing(
            "guide_run_on",
            _guide_run_on,
            "a guide running on past each port plane, as a method measuring inside "
            "the model draws one, meshed between the planes",
            declares=guide,
            inward=(("port1", FORWARD), ("port2", BACKWARD)),
        ),
        Drawing(
            "coax_run_on",
            _coax_run_on,
            "a coaxial line running on past each port ring, whose rings have their "
            "middle in the hole",
            declares=(("dielectric", 3), ("port1", 2), ("port2", 2)),
            inward=(("port1", UP), ("port2", DOWN)),
            loses=CHORDS,
            tags=("curved",),
        ),
        Drawing(
            "loaded_guide_run_on",
            _loaded_guide_run_on,
            "a guide loaded with a slab along its whole length, whose run-ons are the "
            "two regions carried on piece by piece of each port plane",
            declares=(("air", 3), ("slab", 3), ("port1", 2), ("port2", 2)),
            priority=(("slab", 1),),
            inward=(("port1", FORWARD), ("port2", BACKWARD)),
        ),
        Drawing(
            "straight_guide_behind_a_reversed_port",
            _straight_guide_behind_a_reversed_port,
            "a straight guide with its one port facing out, which is the same shapes "
            "as a shorter guide run on past its port, and is meshed as one",
            declares=(("air", 3), ("port1", 2)),
            inward=(("port1", BACKWARD),),
        ),
        Drawing(
            "guide_ends",
            _guide_ends,
            "ports on the guide's own end faces, each facing into it, where nothing "
            "stands behind either",
            declares=guide,
            inward=(("port1", FORWARD), ("port2", BACKWARD)),
        ),
        Drawing(
            "port_faces_out",
            _guide_ends,
            "a port on an end face handed the way out of the model",
            declares=guide,
            expect=REFUSED,
            complaint="faces out of the model",
            inward=(("port1", BACKWARD), ("port2", BACKWARD)),
        ),
        Drawing(
            "plane_short_of_the_guide",
            _plane_short_of_the_guide,
            "a port plane drawn smaller than the cross-section, which cuts nothing: "
            "the guide runs round it on both sides",
            declares=guide,
            expect=REFUSED,
            complaint="runs round it",
            inward=(("port1", FORWARD), ("port2", BACKWARD)),
        ),
        Drawing(
            "both_ports_reversed",
            _guide_run_on,
            "both port planes facing out, which would leave out the guide between "
            "them and keep the two ends",
            declares=guide,
            expect=REFUSED,
            complaint="at once",
            inward=(("port1", BACKWARD), ("port2", FORWARD)),
        ),
        Drawing(
            "one_port_reversed",
            _guide_run_on,
            "one port plane facing the same way as the other, so the guide between "
            "them stands in front of one and behind the other",
            declares=guide,
            expect=REFUSED,
            complaint="stands in front of port1 and behind port2",
            inward=(("port1", FORWARD), ("port2", FORWARD)),
        ),
        Drawing(
            "port_where_two_bodies_meet",
            _port_where_two_bodies_meet,
            "a port pointed at the face where two bodies of one region meet, which "
            "would leave out the whole of one body",
            declares=(("air", 3), ("air", 3), ("port1", 2), ("port2", 2)),
            expect=REFUSED,
            complaint="the whole of a shape",
            inward=(("port1", FORWARD), ("port2", BACKWARD)),
        ),
        Drawing(
            "run_on_holds_a_slab",
            _run_on_holds_a_slab,
            "a slab of another region standing in a run-on, which is not the guide "
            "in front of the plane carried on",
            declares=(("air", 3), ("slab", 3), ("port1", 2), ("port2", 2)),
            expect=REFUSED,
            complaint="holds slab",
            priority=(("slab", 1),),
            inward=(("port1", FORWARD), ("port2", BACKWARD)),
        ),
        Drawing(
            "iris_behind_a_reversed_port",
            _iris_behind_a_reversed_port,
            "a guide with an iris in it and its one port facing out, so what would be "
            "left out is the device and not the port plane carried on",
            declares=(("air", 3), ("port1", 2)),
            expect=REFUSED,
            complaint="not the face carried straight on",
            inward=(("port1", BACKWARD),),
        ),
        Drawing(
            "jog_in_the_run_on",
            _jog_in_the_run_on,
            "a run-on of the port's cross-section set sideways, whose volume is the "
            "face's area times its depth",
            declares=(("air", 3), ("port1", 2)),
            expect=REFUSED,
            complaint="not the face carried straight on",
            inward=(("port1", FORWARD),),
        ),
        Drawing(
            "slab_stops_in_the_run_on",
            _slab_stops_in_the_run_on,
            "a slab that stops short in each run-on, which holds the same regions as "
            "the guide in front of it in a different arrangement",
            declares=(("air", 3), ("slab", 3), ("port1", 2), ("port2", 2)),
            expect=REFUSED,
            complaint="not the face carried straight on",
            priority=(("slab", 1),),
            inward=(("port1", FORWARD), ("port2", BACKWARD)),
        ),
        Drawing(
            "point_in_the_run_on",
            _point_in_the_run_on,
            "a labelled point standing inside a run-on, which the fragmenting embeds "
            "and no boundary lists",
            declares=(("air", 3), ("probe", 0), ("port1", 2), ("port2", 2)),
            expect=REFUSED,
            complaint="would be left out with it",
            inward=(("port1", FORWARD), ("port2", BACKWARD)),
        ),
        Drawing(
            "walls_on_the_run_on",
            _walls_on_the_run_on,
            "a wall label drawn over the whole guide, run-ons included, which would "
            "be left out with them",
            declares=(("air", 3), ("walls", 2), ("port1", 2), ("port2", 2)),
            expect=REFUSED,
            complaint="would be left out with it",
            inward=(("port1", FORWARD), ("port2", BACKWARD)),
        ),
        Drawing(
            "frame_round_the_plane",
            _frame_round_the_plane,
            "a ring cut across one side, which one face does not divide: the kernel "
            "leaves it standing inside the ring, with the ring on both sides of it",
            declares=(("air", 3), ("port1", 2)),
            expect=REFUSED,
            complaint="runs round it",
            inward=(("port1", (0.0, 1.0, 0.0)),),
        ),
    )
