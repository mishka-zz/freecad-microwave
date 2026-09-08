# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The oldest FreeCAD and the oldest Python this workbench loads on.

``Init.py`` prints the refusal at startup and ``InitGui.py`` then registers
nothing, so an unsupported host is turned away by name at one point rather than
failing at whichever call it reaches first.

The two floors are independent, and both are asked. A FreeCAD release is built
against whatever Python its packager chose, so the release floor does not settle
the Python. One refusal carries whichever answers apply, so the composition is
decided here rather than at each entry point, and neither floor's answer can be
lost to the other's.

Nothing here imports FreeCAD. The caller passes ``FreeCAD.Version()`` and
``sys.version_info`` in. This is the one module that has to import on a host too
old to run the rest, so it needs nothing newer than the floors it rejects, and
its annotations are deferred for that reason: a ``str | None`` evaluated at
import time raises on such a Python, taking down the refusal this module exists
to print.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Major and minor of the oldest supported FreeCAD release. Everything in this
#: workbench is measured on 1.1.
FREECAD_MINIMUM = (1, 0)

#: Major and minor of the oldest supported Python. The material catalogs read
#: ``tomllib`` and the Touchstone header is dated with ``datetime.UTC``, and
#: both arrived in 3.11. ``pyproject.toml`` restates this as ``requires-python``
#: for tooling that reads that file without importing this package.
PYTHON_MINIMUM = (3, 11)


def parse_freecad(version: Sequence[str]) -> tuple[int, int] | None:
    """``(major, minor)`` out of ``FreeCAD.Version()``, or ``None``.

    ``Version()`` yields strings, and its fields past the patch number are build
    metadata whose shape changes between releases. Only the first two are read.
    """
    try:
        return (int(version[0]), int(version[1]))
    except (TypeError, ValueError, IndexError):
        return None


def parse_python(version: Sequence[int]) -> tuple[int, int] | None:
    """``(major, minor)`` out of ``sys.version_info``, or ``None``.

    The fields past the minor are the micro, the release level and its serial,
    and none of them decides whether a module is there to import.
    """
    try:
        return (int(version[0]), int(version[1]))
    except (TypeError, ValueError, IndexError):
        return None


def refusal(freecad: Sequence[str], python: Sequence[int]) -> str | None:
    """What to tell the user, or ``None`` if this host meets both floors.

    A version this cannot read draws no complaint, on either floor. Such a host
    is one this workbench was not written against, and refusing to load on a
    guess costs a working user more than the unpredictable failure the guess
    would have prevented. The unreadable floor is forgiven on its own, so the
    other one is still answered.
    """
    said = []
    release = parse_freecad(freecad)
    if release is not None and release < FREECAD_MINIMUM:
        said.append(
            f"needs FreeCAD {FREECAD_MINIMUM[0]}.{FREECAD_MINIMUM[1]} or newer, "
            f"and this is {release[0]}.{release[1]}."
        )
    embedded = parse_python(python)
    old_python = embedded is not None and embedded < PYTHON_MINIMUM
    if old_python:
        assert embedded is not None  # `old_python` is false where it is not
        said.append(
            f"needs Python {PYTHON_MINIMUM[0]}.{PYTHON_MINIMUM[1]} or newer, and "
            f"this FreeCAD embeds {embedded[0]}.{embedded[1]}."
        )
    if not said:
        return None
    sentences = [f"Microwave {clause}" for clause in said]
    sentences.append("The workbench will not be loaded.")
    if old_python:
        # The Python is the FreeCAD build's, not the user's, so the remedy is a
        # different build rather than a different interpreter.
        sentences.append(
            f"Install a FreeCAD built against Python "
            f"{PYTHON_MINIMUM[0]}.{PYTHON_MINIMUM[1]} or newer."
        )
    return " ".join(sentences)


def supported(freecad: Sequence[str], python: Sequence[int]) -> bool:
    """Whether this host meets both floors.

    Derived from `refusal` so that the message and the decision cannot disagree
    about what is supported.
    """
    return refusal(freecad, python) is None
