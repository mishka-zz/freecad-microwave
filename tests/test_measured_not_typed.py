# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Nothing here decides anything from what kind of object FreeCAD says it is.

A shape's type says less about it than its contents do, and every plausible
reading of it is wrong somewhere. A boolean, a fillet and a chamfer all come back
as a ``Compound`` rather than a ``Solid``. An open ``Shell`` reports a nonzero
volume. A ``Part::Box`` rotated thirty degrees is no longer held by any
rectilinear grid, and a boolean whose result happens to be a box is. A shape that
arrived through STEP has no FreeCAD type at all - which is most shapes, once a
user is working with somebody else's CAD.

So the rule is that a shape is judged by measuring it. This states the rule in a
form that fails: a type identifier that names a *shape* is refused, and so is
asking an object which class it is.

What stays is the handful of identifiers by which an object is **created**.
``Part::Feature`` is a container for a shape and says nothing about which shape,
and a workbench that could not name it could not put geometry in a document at
all. ``App::`` is property and document vocabulary and never geometry.

Read off the syntax tree, so a comment or a docstring may say ``Part::Box`` as
freely as this one does - the geometry layer's own comments explain the rule by
naming what it does not do, and that is prose about the code rather than the
code.

The corpus round trip is the other half of this rule and the stronger one: it
puts every shape through a file that strips its type, and requires the same
answer back. This catches what no specimen happens to reach; that catches what no
list of names anticipates.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from tests import repo
from tests.repo import ROOT

#: The modules a shape's class comes out of. ``App`` is deliberately absent: it
#: is properties, documents and groups, and none of those is a claim about a
#: shape.
GEOMETRY = r"(?:Part|PartDesign|Sketcher|Draft|Mesh|Points)"

#: A FreeCAD type identifier that names geometry.
GEOMETRY_TYPE = re.compile(rf"\b{GEOMETRY}::\w+")

#: The same modules as a bare name, for the class arguments of an
#: ``isinstance``.
GEOMETRY_MODULE = re.compile(rf"\b{GEOMETRY}\b")

#: The identifiers an object cannot be created without. Each is a *container*
#: for a shape rather than a statement about one, which is what distinguishes
#: them from every other name the pattern matches.
CONTAINERS = frozenset({"Part::Feature", "Part::FeaturePython"})

#: Asking an object what class it is, by either of the names FreeCAD offers, and
#: asking a shape what kind of topology it is. The last is the one that reads as
#: harmless: ``Solids``, ``Shells`` and ``Faces`` answer what a shape *holds*,
#: which is the question, and ``ShapeType`` answers how it was assembled.
INTERROGATIONS = frozenset({"TypeId", "isDerivedFrom", "ShapeType"})

#: What ``ShapeType`` answers. Forbidden as the answer as well as as the
#: question, because the question has more spellings than can be enumerated -
#: ``type(shape).__name__`` reaches the same word without asking any of them,
#: and that one cannot be forbidden outright: a document object's proxy class is
#: read the same way, and it is the only thing distinguishing one of this
#: workbench's objects from another.
#:
#: Only the words that can be nothing else. ``Face``, ``Edge`` and ``Vertex``
#: are how a *selection* names a sub-element - a reference arrives as
#: ``(object, ['Face3'])`` - so code that reads one is reading what the user
#: picked rather than asking what kind of shape it is. Those the round trip
#: covers instead: a face written to STEP comes back a shell, so anything
#: branching on either is caught by answering differently about one shape.
TOPOLOGY = frozenset({"Compound", "CompSolid", "Solid", "Shell", "Wire"})


#: The package, and only the package. Everything outside it - the tests, the
#: examples, the script that draws the figures for the manual - *draws* shapes
#: rather than judging them, and drawing one means naming the primitive that
#: makes it, exactly as a user does. The stub the suite runs against models
#: ``TypeId`` and ``isDerivedFrom`` for the same reason: FreeCAD has them.
PACKAGE = "Microwave"


def sources() -> list[pathlib.Path]:
    """The workbench's own modules, which are the ones that judge geometry."""
    return [path for path in repo.sources(".py") if path.relative_to(ROOT).parts[0] == PACKAGE]


def _docstrings(tree: ast.AST) -> set[int]:
    """Every string that is a docstring, by identity.

    Comments never reach the tree at all; a docstring does, and it is prose held
    to the same standard as this file's own.
    """
    found = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(getattr(first.value, "value", None), str):
            found.add(id(first.value))
    return found


def offences(source: str, name: str = "<source>") -> list[str]:
    """Every place in one module that decides something from a FreeCAD type."""
    tree = ast.parse(source)
    prose = _docstrings(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in prose:
                continue
            # As a string as well as as an attribute, because ``getattr`` reaches
            # the same property by spelling its name instead of writing it.
            if node.value in INTERROGATIONS:
                found.append(f"{name}:{node.lineno}: asks an object for its {node.value}")
            if node.value in TOPOLOGY:
                found.append(f"{name}:{node.lineno}: names the topology {node.value!r}")
            for named in GEOMETRY_TYPE.findall(node.value):
                if named not in CONTAINERS:
                    found.append(f"{name}:{node.lineno}: names the type {named!r}")
        elif isinstance(node, ast.Attribute) and node.attr in INTERROGATIONS:
            found.append(f"{name}:{node.lineno}: asks an object for its {node.attr}")
        elif _is_a_class_test(node):
            asked = ast.unparse(node.args[1])
            found.append(f"{name}:{node.lineno}: tests the class against {asked}")
    return found


def _is_a_class_test(node: ast.AST) -> bool:
    """An ``isinstance`` against something out of a geometry module.

    The other way to ask what a shape is, and the one a type identifier never
    appears in - ``Part.Compound`` is a class rather than a name.
    """
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return False
    if node.func.id not in ("isinstance", "issubclass") or len(node.args) != 2:
        return False
    return bool(GEOMETRY_MODULE.search(ast.unparse(node.args[1])))


@pytest.mark.parametrize("path", sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_module_decides_anything_from_a_freecad_type(path):
    assert offences(path.read_text(encoding="utf-8"), path.name) == []


class TestTheCheckItself:
    """A lint that cannot fire is a rule nobody is keeping."""

    def test_it_finds_a_branch_on_a_primitive(self):
        found = offences('if obj.TypeId == "Part::Box":\n    pass\n')
        assert any("TypeId" in one for one in found), found
        assert any("Part::Box" in one for one in found), found

    def test_it_finds_the_topology_question(self):
        assert offences("kind = shape.ShapeType\n")

    def test_it_finds_the_question_spelled_rather_than_written(self):
        """``getattr`` reaches the same property by naming it, so a check that
        read attribute access alone would see nothing here at all."""
        assert offences('if getattr(shape, "ShapeType", "") == "Solid":\n    pass\n')

    def test_it_finds_the_answer_without_the_question(self):
        """``type(x).__name__`` gives the same word and asks nothing, which is
        why what it is compared against is forbidden as well."""
        assert offences('if type(shape).__name__ == "Compound":\n    pass\n')

    def test_it_finds_the_class_asked_for_directly(self):
        """A class is not a type identifier and carries no ``::`` to match."""
        assert offences("if isinstance(shape, Part.Compound):\n    pass\n")

    def test_it_finds_the_derived_form(self):
        assert offences('if obj.isDerivedFrom("Part::Cylinder"):\n    pass\n')

    def test_it_lets_an_object_be_created(self):
        assert offences('doc.addObject("Part::FeaturePython", "EMMeshPreview")\n') == []

    def test_it_lets_a_property_be_declared(self):
        assert offences('obj.addProperty("App::PropertyLength", "Width")\n') == []

    def test_it_lets_the_shape_itself_be_read(self):
        """The property the geometry arrives in, and a face named by a selection.
        Neither is a claim about what kind of shape it is."""
        assert offences('shape = getattr(obj, "Shape", None)\n') == []
        assert offences('if sub.startswith("Face"):\n    pass\n') == []

    def test_it_lets_our_own_proxy_class_be_read(self):
        """Every object this workbench puts in a document carries the same
        FreeCAD type, so its proxy's class name is the only thing telling a port
        from a solver - and it is read exactly as a shape's class would be."""
        assert offences("return type(obj.Proxy).__name__\n") == []

    def test_it_lets_prose_say_what_the_code_may_not(self):
        """The geometry layer explains itself by naming what it does not do, and
        a rule that forbade the explanation would be kept by deleting it."""
        assert offences('"""A Part::Box is the usual case and not the only one."""\n') == []

    def test_it_reads_a_message_as_code(self):
        """A refusal telling the user to redraw it as a primitive ties them to a
        constructor just as firmly as a branch would."""
        assert offences('raise Error("draw it as a Part::Box instead")\n')

    def test_it_looks_past_the_first_statement(self):
        """A docstring is exempt because it is the first statement, so a check
        that exempted every string would pass everything after one."""
        assert offences('"""Prose."""\nkind = shape.ShapeType\n')
