# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What "a file of this project" means, for the tests that police the tree.

A filesystem walk answers a different question: it finds whatever the tools left
behind as well. ``.pytest_cache/README.md`` is pytest's own prose, held to none
of the conventions here, and it exists only after a run - so a walk also makes
the collected test count depend on whether the suite has been run before.

Asking git is exact, and it keeps no list of caches to fall behind. The list is
what drifted: two callers each carried the same one, and neither had grown the
entry for pytest's cache.
"""

from __future__ import annotations

import pathlib
import subprocess

import Microwave

ROOT = pathlib.Path(Microwave.__file__).resolve().parent.parent


def sources(*suffixes: str) -> list[pathlib.Path]:
    """Every file this project wrote, with one of these suffixes.

    Vendored code is somebody else's text and is not included.
    """
    found = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        capture_output=True,
        text=True,
    )
    if found.returncode != 0:
        # Refusing beats returning nothing: an empty list collects zero tests,
        # which reads exactly like a rule that holds everywhere.
        raise RuntimeError(
            f"cannot list this repository's files - git said: {found.stderr.strip()}"
        )

    paths = (ROOT / name for name in found.stdout.split("\0") if name)
    return sorted(p for p in paths if p.suffix in suffixes and "_vendor" not in p.parts)
