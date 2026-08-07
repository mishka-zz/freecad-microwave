# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The oldest FreeCAD this workbench loads on, stated once and checked.

``Init.py`` prints the refusal at startup and ``InitGui.py`` then registers
nothing, so an older FreeCAD is turned away by name at one point instead of
failing at whichever call it reaches first.

Nothing here imports FreeCAD - the caller passes ``FreeCAD.Version()`` in -
and nothing here needs a Python newer than the floor it is rejecting. This is
the one module that has to import on a FreeCAD too old to run the rest.
"""

#: Major and minor of the oldest supported release. Everything in this
#: workbench is measured on 1.1, and a FreeCAD whose embedded Python is older
#: than ``pyproject.toml``'s ``requires-python`` cannot import it at all.
MINIMUM = (1, 0)


def parse(version):
    """``(major, minor)`` out of ``FreeCAD.Version()``, or ``None``.

    ``Version()`` yields strings, and its fields past the patch number are build
    metadata whose shape changes between releases. Only the first two are read.
    """
    try:
        return (int(version[0]), int(version[1]))
    except (TypeError, ValueError, IndexError):
        return None


def supported(version):
    """Whether ``FreeCAD.Version()`` names a release at or above `MINIMUM`.

    A version that will not parse counts as supported. It is a FreeCAD this was
    not written against, and refusing to load on a guess costs a working user
    more than the unpredictable failure the guess would have prevented.
    """
    found = parse(version)
    return found is None or found >= MINIMUM


def refusal(version):
    """What to tell the user, or ``None`` if this FreeCAD is new enough."""
    found = parse(version)
    if supported(version):
        return None
    return (
        f"Microwave needs FreeCAD {MINIMUM[0]}.{MINIMUM[1]} or newer, and this is "
        f"{found[0]}.{found[1]}. The workbench will not be loaded."
    )
