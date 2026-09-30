# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Every bound the workbench takes round a shape is taken in one place.

FreeCAD's view meshes each shape it shows, and the kernel then answers
``BoundBox`` and ``optimalBoundingBox`` from the mesh's nodes, whose chords cut
inside a curved extreme. :func:`Microwave.drawn.bound` asks a copy, which
carries no mesh. A module asking the shape itself answers differently with the
view up than without it, and only a run that raises the view sees the
difference. So this holds the rule against the tree.
"""

from __future__ import annotations

import ast

from tests.repo import ROOT

#: The attributes that answer from a mesh once the view has laid one. The
#: ``FreeCAD.BoundBox`` type is built from points and asks no shape.
MESHED = ("BoundBox", "optimalBoundingBox")

#: Where each may be asked, as ``(module, function)``: on a copy, and nowhere else.
ASKED = {
    ("Microwave/drawn.py", "_edge_bound"),
    ("Microwave/drawn.py", "_face_bound"),
    ("Microwave/drawn.py", "_held"),
    ("Microwave/drawn.py", "reached"),
    ("Microwave/drawn.py", "tightest"),
    ("Microwave/drawn.py", "touching"),
}


def _asked(path):
    """Each place in one module asking a meshed attribute, as the innermost
    function round it, the line and the attribute."""
    found = []

    def visit(node, function):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
                continue
            if (
                isinstance(child, ast.Attribute)
                and child.attr in MESHED
                and not (isinstance(child.value, ast.Name) and child.value.id == "FreeCAD")
            ):
                found.append((function, child.lineno, child.attr))
            visit(child, function)

    visit(ast.parse(path.read_text(), filename=str(path)), "<module>")
    return found


def test_no_module_asks_a_shape_for_its_bound_but_through_drawn():
    wrong = []
    for path in sorted((ROOT / "Microwave").rglob("*.py")):
        module = path.relative_to(ROOT).as_posix()
        for function, line, attr in _asked(path):
            if (module, function) not in ASKED:
                wrong.append(f"{module}:{line} {function} asks {attr}")
    assert not wrong, (
        "these ask a shape for a bound directly, which the view's mesh answers "
        "once the shape is shown. Ask Microwave.drawn.bound instead:\n" + "\n".join(wrong)
    )


def test_the_places_allowed_are_the_places_that_ask():
    """A place left on the list after it stopped asking lets the next one in
    unseen."""
    asking = {
        (path.relative_to(ROOT).as_posix(), function)
        for path in (ROOT / "Microwave").rglob("*.py")
        for function, _, _ in _asked(path)
    }
    assert asking >= ASKED, f"no longer asking: {sorted(ASKED - asking)}"
