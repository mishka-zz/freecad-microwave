# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A constant of the vacuum is written down once, and this is what says so.

Copies of a defined constant agree until somebody corrects one of them, and then
every module goes on being convinced it has the right figure. Nothing has to be
wrong with any of the copies for that to happen, which is what makes it worth a
rule rather than a habit.

So the rule is stated in a form that fails. Whatever ``Microwave.units`` declares
to enough figures to be a physical constant may appear nowhere else in the
workbench, in any spelling: the check is on the *value*, so ``2.99792458e11``,
``299_792_458.0`` and ``4.0e-7 * math.pi`` are each recognised as the constant
they are and not as the characters they were typed as. A module wanting one in
other units derives it, and a derivation cannot disagree.

The test corpus is deliberately outside this. ``tests/analytic/reference.py``
carries its own speed of light because a closed form scored against the code
that produced it is scored against nothing.
"""

from __future__ import annotations

import ast
import math
import operator
import pathlib

import pytest

from Microwave import units
from tests import repo

#: Where the constants live, and the one file the rule does not apply to.
UNITS = pathlib.Path(units.__file__).resolve()

#: Significant digits at which a number stops being a count, a share or a
#: tolerance and starts being a measurement of the world. Nothing this workbench
#: chooses for itself is quoted anywhere near this precisely; every constant of
#: the vacuum is.
FIGURES = 6


def digits(value: float) -> str:
    """The significant digits of ``value``, with its scale and sign removed.

    What identifies a physical constant across spellings. Metres per second and
    millimetres per second differ by a power of ten, which is exactly what this
    throws away - so a module holding the same constant in its own units is
    caught, and that is the copy most likely to drift unnoticed.
    """
    written = repr(abs(float(value)))
    mantissa = written.split("e")[0].replace(".", "")
    return mantissa.strip("0")


#: What a constant may be reached through besides a figure. Both spell the same
#: floats, and either would do for writing a permeability out as a product.
LIBRARIES = ("math", "np", "numpy")

#: Arithmetic a constant can be assembled out of without ceasing to be one.
ARITHMETIC = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def number(node) -> float | None:
    """What that expression is worth on its own, or ``None`` if it needs the
    program around it.

    A constant is not always typed as one figure - the permeability of the
    vacuum is conventionally written as a product with pi - so an expression
    standing entirely on figures and on :data:`LIBRARIES` is folded and judged by
    what it comes to. An expression mentioning anything else is not a constant
    and is left alone, which is what keeps ``2.0 * math.pi * frequency`` out of
    this.
    """
    if isinstance(node, ast.Constant):
        return float(node.value) if type(node.value) in (int, float) else None
    if isinstance(node, ast.Attribute) and getattr(node.value, "id", None) in LIBRARIES:
        found = getattr(math, node.attr, None)
        return float(found) if isinstance(found, float) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = number(node.operand)
        return None if inner is None else (inner if isinstance(node.op, ast.UAdd) else -inner)
    if isinstance(node, ast.BinOp) and type(node.op) in ARITHMETIC:
        left, right = number(node.left), number(node.right)
        if left is None or right is None:
            return None
        try:
            return float(ARITHMETIC[type(node.op)](left, right))
        except (ArithmeticError, OverflowError):
            return None
    return None


def literals(path: pathlib.Path):
    """Every number that file states, with the line it is on.

    Sub-expressions count. A copy buried in the middle of a formula is still a
    copy, and it is the one nobody would think to look for.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = ((number(node), getattr(node, "lineno", 0)) for node in ast.walk(tree))
    return [(value, line) for value, line in found if value is not None]


@pytest.fixture(scope="module")
def declared():
    """``{digits: name}`` for the constants in ``units`` quoted precisely enough.

    Read off the module rather than listed here: a constant added to it joins
    this rule without anybody remembering to say so, which is the only way a
    rule of this kind stays true.
    """
    found = {}
    for name, value in vars(units).items():
        if name.startswith("_") or not isinstance(value, float):
            continue
        spelling = digits(value)
        if len(spelling) >= FIGURES:
            found[spelling] = name
    return found


def test_every_constant_is_quoted_precisely_enough_to_be_recognised(declared):
    """Otherwise the sweep below passes by testing nothing at all."""
    assert set(declared.values()) == {
        "SPEED_OF_LIGHT",
        "VACUUM_PERMITTIVITY",
        "VACUUM_PERMEABILITY",
    }


@pytest.mark.parametrize(
    "written",
    ["299_792_458.0", "2.99792458e11", "4.0e-7 * math.pi", "-8.854_187_812_8e-12"],
)
def test_a_constant_is_recognised_however_it_is_written(declared, written):
    """The spellings a copy would actually arrive in: the same figure with the
    digits grouped differently, the same constant in this workbench's own
    millimetres, one assembled out of pi, and one carrying a sign."""
    assert digits(number(ast.parse(written, mode="eval").body)) in declared


@pytest.mark.parametrize("written", ["2.0 * math.pi * top", "top / 8.0", "math.pi"])
def test_and_an_expression_that_is_not_one_is_left_alone(declared, written):
    """A formula mentioning a variable is not a constant however it is built,
    and pi on its own is not a constant of the vacuum."""
    value = number(ast.parse(written, mode="eval").body)
    assert value is None or digits(value) not in declared


def test_the_three_constants_of_the_vacuum_agree_with_each_other():
    """``eps0 mu0 c^2 = 1``, which is the only check on these figures that is
    not a restatement of them.

    It fails on a mistyped digit in any of them, and it is the reason each is
    declared here rather than left to a comment beside the others. What it does not
    reach is exact: since the 2019 redefinition the permeability is measured,
    and the conventional ``4 pi x 1e-7`` departs from it by about a part in two
    billion. The tolerance is the round number just above that departure, so it
    is set by the physics rather than opened until the assertion passed.
    """
    identity = units.SPEED_OF_LIGHT**2 * units.VACUUM_PERMITTIVITY * units.VACUUM_PERMEABILITY
    assert identity == pytest.approx(1.0, rel=1e-9, abs=0.0)


def test_no_other_module_writes_a_constant_of_the_vacuum_down(declared):
    copies = []
    for path in repo.sources(".py"):
        if path == UNITS or "tests" in path.relative_to(repo.ROOT).parts:
            continue
        for value, line in literals(path):
            name = declared.get(digits(value))
            if name is not None:
                copies.append(f"{path.relative_to(repo.ROOT)}:{line} restates {name}")
    assert not copies, "\n".join(copies) + "\n\nImport it from Microwave.units instead"


def test_a_derived_spelling_is_recognised_as_the_same_constant(declared):
    """The case the sweep exists for, and the one a text search would miss:
    ``portbox`` needs millimetres per second and would once have said so in
    figures of its own."""
    assert digits(units.SPEED_OF_LIGHT * units.MM_PER_M) in declared


@pytest.mark.parametrize("value", [0.1, 1e3, 1e-6, 15.0, 4096, 2556680.79])
def test_a_number_this_project_chose_is_not_mistaken_for_one(declared, value):
    """A share, a conversion, a tolerance, an angle, a budget, and a figure read
    off somebody else's table - none of them is a constant of the vacuum, and a
    rule that swept them up would be turned off rather than obeyed."""
    assert digits(value) not in declared
