# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Draw the labelled corpus under a real FreeCAD, one file per piece.

This is the half that needs the CAD kernel, and it is the only half. What it
writes is one BREP per drawn piece and a manifest naming which file carries
which label, which is the shape a caller hands the mesher. Everything
downstream reads those files and needs no FreeCAD, which is what keeps the
checks runnable under any interpreter that has Gmsh.

Executed by ``freecadcmd``, so:

* there is no ``__main__``. The file is exec'd under a module name taken from
  its own stem, and a bare ``if __name__ == "__main__":`` guard would never
  fire.
* a script that raises is run twice, so output that looks duplicated has
  raised. The *"Exception while processing file"* line on stderr is unbuffered
  and arrives nowhere near the output it belongs to.
* an extra argument on the command line is opened as a document, so where the
  files land is taken from the environment.

Run it from the workbench root::

    DRAWINGS_OUT=tests/_drawings freecadcmd tests/drawings_probe.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

#: Where the files land. The caller names it, since a test wants them under its
#: own temporary directory as readily as in the tree.
OUT = pathlib.Path(os.environ.get("DRAWINGS_OUT", HERE / "_drawings"))


def measure(shape, dimension: int) -> float:
    """What a shape is worth at that dimension - a volume, or an area.

    Written here rather than derived downstream because it is the kernel's own
    answer for the shape that was drawn, and a mesh of that shape is compared
    against it. Magnitudes, since a reversed solid occupies the space it was
    drawn in and reports its volume negative.
    """
    if dimension == 3:
        return sum(abs(solid.Volume) for solid in shape.Solids)
    if dimension == 2:
        return sum(abs(face.Area) for face in shape.Faces)
    return 0.0


def whole(shapes, dimension: int, boundary: bool = False, less=()) -> float:
    """What the drawing is worth at that dimension, counting a region once.

    Summed piece by piece a drawing with an overlap in it is counted twice
    where two shapes were drawn over one region, and a mesh of it is then short
    of a figure that describes no drawing. So the shapes are fused first, which
    is the one place a boolean belongs: here the question is what was drawn,
    and nothing of what is fused is handed to the mesher.

    Asked for the boundary, it answers what the frontier of that region
    measures - the faces of the fused solid, a cavity's wall among them, which
    is where the region ends rather than the outside of anything - and nothing
    where two of the shapes meet. A mesh whose regions were not joined has that
    interface in it twice and comes out over.

    ``less`` is what a label that leaves the model was drawn as, and is cut out
    of the region once it is fused: the mesh holds what is left.
    """
    if not shapes:
        return 0.0
    joined = shapes[0]
    for other in shapes[1:]:
        joined = joined.fuse(other)
    for gone in less:
        joined = joined.cut(gone)
    if not boundary:
        return measure(joined, dimension)
    if dimension != 3:
        return 0.0
    return sum(abs(face.Area) for solid in joined.Solids for face in solid.Faces)


def _less(shape, others):
    """The shape less every one of ``others``."""
    for other in others:
        shape = shape.cut(other)
    return shape


def draw() -> None:
    """Write every drawing, and say what each piece came out as."""
    import Part

    from tests.drawings import drawings, ends

    OUT.mkdir(parents=True, exist_ok=True)
    for drawing in drawings() + ends():
        stated = dict(drawing.priority)
        stated.update((label, 0) for label, _ in drawing.declares if label not in stated)
        facing = dict(drawing.inward)
        cutting = set(drawing.divides)
        going = set(drawing.leaves)
        pieces = []
        topmost = []
        gone = []
        for number, (label, dimension, shape) in enumerate(drawing.build(Part)):
            name = f"{drawing.name}.{number}.brep"
            shape.exportBrep(str(OUT / name))
            pieces.append(
                {
                    "label": label,
                    "dim": dimension,
                    "file": name,
                    "priority": stated.get(label, 0),
                    **({"divides": True} if label in cutting else {}),
                    **({"leaves": True} if label in going else {}),
                    **({"inward": list(facing[label])} if label in facing else {}),
                }
            )
            if dimension == drawing.top:
                (gone if label in going else topmost).append((label, shape))
        # What leaves is what the leaving label keeps: a label above it keeps
        # what they share.
        gone = [
            _less(shape, [kept for name, kept in topmost if stated.get(name, 0) > stated[label]])
            for label, shape in gone
        ]
        topmost = [shape for _, shape in topmost]
        drawn = whole(topmost, drawing.top, less=gone)
        around = whole(topmost, drawing.top, boundary=True, less=gone)
        built = tuple((piece["label"], piece["dim"]) for piece in pieces)
        if built != drawing.declares:
            raise AssertionError(f"{drawing.name} builds {built} and declares {drawing.declares}")
        (OUT / f"{drawing.name}.manifest.json").write_text(
            json.dumps({"pieces": pieces, "measure": drawn, "outside": around}, indent=2)
        )
        print(f"{drawing.name}: {drawing.expect}, measures {drawn:.6g}")
        for piece in pieces:
            print(f"    {piece['label']:12s} dim {piece['dim']}  {piece['file']}")
    sys.stdout.flush()


draw()
