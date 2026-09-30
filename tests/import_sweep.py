# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Every adapter module, imported where neither FreeCAD nor a solver exists.

Run it from the workbench root::

    python -m tests.import_sweep

CI runs it on a runner that has numpy and nothing else, which is the
environment the rule is about: there, a module reaching for anything else fails
outright, and no list of what may not be imported has to be kept up to date.
The suite asks the same question of the same modules in
``tests/test_openems_adapter.py``, against such a list, because the machine a
test runs on has the runner's own dependencies installed. Both reach the module
list through :func:`swept`, so the two cannot come to sweep different sets.

One child per module, so that each is imported into an interpreter holding
nothing else of this workbench. That is what the claim is about: a module
imported after the others is judged on a machine they have already furnished,
and the run says nothing about importing it on its own.

Stdlib only. It is imported by the job whose whole point is that nothing else
is installed.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

__all__ = ["PACKAGE", "PACKAGES", "ROOT", "SOLVER_SIDE", "main", "swept"]

#: The adapter package, as an import writes it. The one the suite asks about by
#: default, being the one whose modules the tests below it name.
PACKAGE = "Microwave.Solvers.openems"

#: Every adapter package swept. A rule stated for one adapter and checked on one
#: adapter is a rule the second one is only promised to keep.
PACKAGES = (PACKAGE, "Microwave.Solvers.palace")

#: The workbench root, which is where an import of ``Microwave`` resolves from.
ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The modules that exist to load the engine, and so are the ones no import
#: sweep may hold to importing without it. ``driver`` is the solver-side entry
#: point; ``coaxial`` is a port class built out of openEMS' own, which the
#: bindings do not ship. Both are reached only from a path that has already
#: loaded the engine - ``driver`` imports ``coaxial`` inside the branch that
#: builds one. Nothing under the Palace adapter is on this side: Palace is a
#: program rather than a set of bindings, and the mesher it asks for stands
#: outside every adapter.
SOLVER_SIDE = frozenset({"driver", "coaxial"})


def swept(package: str = PACKAGE) -> list[str]:
    """Adapter modules that must import with neither FreeCAD nor openEMS.

    Discovered rather than listed, because a hand-maintained list is how a new
    module goes silently uncovered. ``tests/test_openems_adapter.py`` names a
    floor of them, which is what says this discovery still finds anything.

    Sub-packages count as one name each. Importing the package runs its
    ``__init__``, which is where a package that eagerly imports its own modules
    pulls all of them in, so the sweep still judges every file underneath.
    """
    where = ROOT / package.replace(".", "/")
    return sorted(
        path.stem
        for path in where.iterdir()
        if (path.suffix == ".py" or (path / "__init__.py").is_file())
        and path.stem not in {"__init__", *SOLVER_SIDE}
    )


def main() -> int:
    """Import each in its own interpreter, and stop at the first that will not."""
    for package in PACKAGES:
        for name in swept(package):
            print(f"- {package}.{name}", flush=True)
            done = subprocess.run([sys.executable, "-c", f"import {package}.{name}"], cwd=ROOT)
            if done.returncode:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
