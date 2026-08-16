# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Run the corpus through the geometry layer under a real FreeCAD.

This is the half that needs the CAD kernel, and it is the only half. What it
writes is one JSON artifact per specimen holding the triangles the layer
produced, the box it measured, and what the kernel itself says the shape's volume
and area are. Everything downstream - containment, connectivity, the grid and its
invariants - reads those artifacts and needs no FreeCAD at all, which is what
keeps the checks runnable under the interpreter that owns the openEMS bindings.

Executed by ``freecadcmd``, so:

* there is no ``__main__``. The file is exec'd under a module name taken from its
  own stem, and a bare ``if __name__ == "__main__":`` guard would never fire.
* the process **segfaults on exit** once a main window has been shown, after the
  last statement has run. Everything here is durable before that, and a caller
  must judge this by its output rather than by its exit status.

The main window is shown deliberately. ``Shape.tessellate`` returns the *view
provider's* display mesh once a shape has been drawn, and a display mesh is
built to a cosmetic deviation - so the trap only exists where a view provider
does, and a probe without one would report that there is nothing to catch.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

#: Where the artifacts land. The caller names it, since a test wants them under
#: its own temporary directory rather than in the tree.
OUT = pathlib.Path(os.environ.get("CORPUS_OUT", HERE / "_corpus"))


def _document(FreeCAD):
    name = "corpus"
    if name in FreeCAD.listDocuments():
        FreeCAD.closeDocument(name)
    return FreeCAD.newDocument(name)


def _drawn(doc, shape, label):
    """The shape as a document object, recomputed and visible.

    Through a document object rather than the bare shape, because that is how a
    user's geometry arrives and it is the only way a view provider exists.
    """
    obj = doc.addObject("Part::Feature", "Specimen")
    obj.Label = label
    obj.Shape = shape
    doc.recompute()
    return obj


def _measured(shape):
    """What the kernel says about the shape, for a downstream check to score
    the triangulation against without the kernel being present."""
    box = shape.BoundBox
    return {
        "volume": float(shape.Volume),
        "area": float(shape.Area),
        "shape_type": str(shape.ShapeType),
        "solids": len(shape.Solids),
        "faces": len(shape.Faces),
        "edges": len(shape.Edges),
        "closed": bool(shape.isClosed()) if shape.ShapeType in ("Shell", "Wire") else None,
        "bound_box": [
            float(box.XMin),
            float(box.YMin),
            float(box.ZMin),
            float(box.XMax),
            float(box.YMax),
            float(box.ZMax),
        ],
    }


#: Faces sampled per shape, and points per face. Chosen to cross a face count
#: in the tens while keeping the kernel queries per specimen in the hundreds; a
#: shape with more faces than this is sampled over the first of them.
FACES_SAMPLED = 32
POINTS_PER_FACE = 4

#: Points scattered through the shape's own bounding box, which is where a
#: containment fault that is not near any surface would show.
SCATTERED = 60

#: A share of the shape's size, used both for how far in from a face's own
#: outline a sample has to sit and for how far a companion is placed off the
#: surface. The outline distance is measured rather than read off the parameter
#: fractions, because a planar face carries whatever frame its surface was built
#: with and the domain is not the shape.
MARGIN = 0.02

#: Triangles measured when scoring how far the emitted surface departs from the
#: drawn one. Every triangle would be exact and slow on a shape with tens of
#: thousands of them, and the departure is a property of the tessellation rather
#: than of any one triangle.
TRIANGLES_MEASURED = 400


def _departure(Part, FreeCAD, shape, pieces):
    """How far the surface that will be solved sits from the surface drawn.

    The chord of a polygonised curve lies off the curve, so near the boundary
    the kernel and the engine disagree honestly and a comparison there settles
    nothing. This is the width of that band, measured: the distance from each
    emitted triangle's middle to the real boundary. A shape whose faces are
    planar is triangulated exactly and measures zero.
    """
    boundary = Part.Compound(shape.Faces)
    worst = 0.0
    for piece in pieces:
        vertices, faces = piece.vertices or (), piece.faces or ()
        if not faces:
            continue
        step = max(1, len(faces) // TRIANGLES_MEASURED)
        for face in list(faces)[::step]:
            corners = [vertices[index] for index in face]
            middle = FreeCAD.Vector(
                sum(c[0] for c in corners) / 3.0,
                sum(c[1] for c in corners) / 3.0,
                sum(c[2] for c in corners) / 3.0,
            )
            worst = max(worst, boundary.distToShape(Part.Vertex(middle))[0])
    return worst


def _samples(Part, FreeCAD, shape):
    """Points the kernel has an opinion about, and what that opinion is.

    A containment check needs points where the answer is known and not merely
    plausible. Points *on* the boundary are the ones that matter and the ones a
    scatter through the volume never finds: a zero-thickness sheet is hit by no
    random point at all, so a check made only of those would pass on a sheet
    that was never emitted.

    So the boundary is walked directly. Each face is sampled inside its own
    parameter domain and clear of its own outline, and around each of those
    points a pair is placed a short way along the normal - one the kernel puts
    inside the shape and one it puts outside.

    ``distance`` is measured to the shape's **faces** rather than to the shape.
    A solid contains its interior points, so it answers zero distance for all of
    them, which says nothing about how near the boundary they are.
    """
    box = shape.BoundBox
    span = max(box.XLength, box.YLength, box.ZLength)
    extents = [e for e in (box.XLength, box.YLength, box.ZLength) if e > 0.0]
    # Off the surface by a share of the *thinnest* extent, so that on a plate
    # the companion placed inward stays inside instead of crossing to the far
    # face and reporting itself outside.
    step = MARGIN * (min(extents) if extents else span)
    boundary = Part.Compound(shape.Faces) if shape.Faces else None

    placed = []
    for face in shape.Faces[:FACES_SAMPLED]:
        low_u, high_u, low_v, high_v = face.ParameterRange
        for number in range(POINTS_PER_FACE):
            across = (number + 0.5) / POINTS_PER_FACE
            along = ((number * 3 + 1) % POINTS_PER_FACE + 0.5) / POINTS_PER_FACE
            u = low_u + across * (high_u - low_u)
            v = low_v + along * (high_v - low_v)
            if not face.isPartOfDomain(u, v):
                continue
            try:
                middle = face.valueAt(u, v)
                normal = face.normalAt(u, v)
            except Exception:  # noqa: BLE001 - a face with no normal here says nothing
                continue
            # Clear of the outline by a share of *this face's* size. Against the
            # whole shape's, a face much smaller than the shape it belongs to -
            # the stroke of a letter in a line of text - has no interior left
            # and contributes no samples at all.
            near = face.BoundBox
            room = MARGIN * max(near.XLength, near.YLength, near.ZLength)
            if min(wire.distToShape(Part.Vertex(middle))[0] for wire in face.Wires) < room:
                continue
            placed.append((middle, True))
            placed.append((middle - normal * step, False))
            placed.append((middle + normal * step, False))

    seed = 12345
    for _ in range(SCATTERED):
        # A fixed sequence, so a failure is the same failure on the next run.
        shares = []
        for _ in range(3):
            seed = (1103515245 * seed + 12345) % (1 << 31)
            shares.append((seed % 10000) / 10000.0)
        placed.append(
            (
                FreeCAD.Vector(
                    box.XMin + shares[0] * box.XLength,
                    box.YMin + shares[1] * box.YLength,
                    box.ZMin + shares[2] * box.ZLength,
                ),
                False,
            )
        )

    # Each solid is asked separately. ``isInside`` on a compound does not
    # consider all of them, so a point lying in the second of two disjoint lumps
    # is answered as though the lump were not there.
    solids = list(shape.Solids)
    found = []
    for point, on_surface in placed:
        found.append(
            {
                "point": [float(point.x), float(point.y), float(point.z)],
                "inside": (
                    any(solid.isInside(point, 0.0, True) for solid in solids) if solids else None
                ),
                "on_surface": on_surface,
                "distance": (
                    float(boundary.distToShape(Part.Vertex(point))[0]) if boundary else 0.0
                ),
            }
        )
    return found


def _piece(piece):
    return {
        "label": piece.label,
        "lower": [float(v) for v in piece.box.lower],
        "upper": [float(v) for v in piece.box.upper],
        "vertices": [[float(c) for c in vertex] for vertex in (piece.vertices or ())],
        "faces": [[int(i) for i in face] for face in (piece.faces or ())],
        "sheet_normal": None if piece.sheet_normal is None else int(piece.sheet_normal),
        "thickened": float(piece.thickened),
    }


def _demands(pieces):
    """Every length the drawing carries, projected onto the three axes.

    The bodies are built the way the translation layer builds them: one per
    piece, each holding the geometry that piece names, measured only where it
    reached the mesher as triangles rather than as a box.

    Every specimen is declared a conductor, which is the case that measures the
    most - a dielectric is asked for nothing at its edges.

    A specimen drawn as one compound therefore has the gaps *between its lumps*
    measured here, which is the whole of what a witness pair needs. What is
    still out of reach is a gap to a second drawn object, since a specimen is
    one object and the corpus draws them one at a time.
    """
    from Microwave.Solvers.openems import lfs
    from Microwave.Solvers.openems.sizing import demands
    from tests.corpus import CELL_CAP, EDGE_SIZE

    bodies = [
        lfs.Body(
            piece.label,
            piece.shape,
            metal=True,
            measured=bool(piece.faces),
            sheet=piece.sheet_normal is not None,
        )
        for piece in pieces
    ]
    found = lfs.features(bodies, CELL_CAP, EDGE_SIZE)
    return [
        [[float(one.lower), float(one.upper), float(one.size)] for one in axis]
        for axis in demands(found)
    ]


def _verdict(doc, shape, label, skin=None):
    """What the geometry layer makes of one shape, and the pieces it made.

    The pieces come back live as well as serialised, because the containment
    samples below are scored against the triangles rather than against the file.
    """
    from Microwave.Solvers.openems.geometry import solid_boxes
    from Microwave.Solvers.openems.properties import TranslationError

    obj = _drawn(doc, shape, label)
    try:
        pieces = solid_boxes(obj, skin)
    except TranslationError as error:
        return {"status": "refused", "reason": str(error)}, []
    except Exception as error:  # noqa: BLE001 - anything else is the defect
        return {
            "status": "crashed",
            "reason": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }, []
    finally:
        # Dropped so the next shape can be drawn under the same label. FreeCAD
        # keeps labels unique within a document and a refusal quotes the label,
        # so a shape and its round trip would otherwise be refused in words that
        # differ where nothing about the geometry does.
        doc.removeObject(obj.Name)

    return {
        "status": "meshed",
        "pieces": [_piece(piece) for piece in pieces],
        "demands": _demands(pieces),
    }, pieces


def _round_tripped(Part, doc, shape, label, skin=None):
    """The same shape written to STEP and read back, put through the same layer.

    STEP carries surfaces and their trimming, and nothing of what drew them, so
    a verdict that changes here was read off something other than the geometry.
    A shape the exporter or the importer will not carry says nothing about this
    workbench and is reported as such.
    """
    from tests.corpus import round_trip

    try:
        back = round_trip(Part, shape, "step")
    except Exception as error:  # noqa: BLE001 - a file format's limit is not a verdict
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}

    record, _ = _verdict(doc, back, label, skin)
    record["measured"] = _measured(back)
    return record


def _run_one(FreeCAD, Part, doc, specimen):
    from tests.corpus import Unavailable

    record = {"name": specimen.name, "subject": specimen.subject, "expect": specimen.expect}
    try:
        shape = specimen.build(Part)
    except Unavailable as error:
        record["status"] = "unavailable"
        record["reason"] = str(error)
        return record
    except Exception as error:  # noqa: BLE001 - a builder fault is not a verdict
        record["status"] = "undrawable"
        record["reason"] = f"{type(error).__name__}: {error}"
        return record

    record["measured"] = _measured(shape)
    from tests.corpus import SKIN

    skin = SKIN if specimen.skin else None
    record["skin"] = skin
    verdict, pieces = _verdict(doc, shape, specimen.name, skin)
    record.update(verdict)
    if record["status"] == "meshed":
        record["samples"] = _samples(Part, FreeCAD, shape)
        record["departure"] = _departure(Part, FreeCAD, shape, pieces)
    record["round_trip"] = _round_tripped(Part, doc, shape, specimen.name, skin)
    return record


def main():
    import FreeCAD
    import FreeCADGui
    import Part

    # A view provider only exists once the Gui layer is up, and the display-mesh
    # trap only exists where a view provider does.
    FreeCADGui.showMainWindow()

    from tests import corpus

    OUT.mkdir(parents=True, exist_ok=True)
    doc = _document(FreeCAD)

    written = []
    for specimen in corpus.specimens():
        record = _run_one(FreeCAD, Part, doc, specimen)
        path = OUT / f"{specimen.name}.json"
        path.write_text(json.dumps(record, indent=1, sort_keys=True))
        written.append(specimen.name)
        print(f"CORPUS: name={specimen.name} status={record['status']}")

    (OUT / "manifest.json").write_text(json.dumps({"specimens": written}, indent=1))
    print(f"CORPUS: written={len(written)} into {OUT}")
    sys.stdout.flush()


main()
