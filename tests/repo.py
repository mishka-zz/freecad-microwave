# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What "a file of this project" means, for the tests that police the tree.

A filesystem walk answers a different question: it finds whatever the tools left
behind as well - ``.pytest_cache/README.md`` is pytest's own prose, held to none
of the conventions here. Telling that from our text takes a list of what to
skip, and the one place a list like that is already kept up is git's own.

Tracked and untracked alike: a guard reading the index alone is silent about a
file until somebody stages it, which is the whole of the time its author is the
one who could act on a complaint about it.

The cost is that "untracked" also covers a scratch, a download, an environment
and a build's output. Those are named where git reads them: ``.gitignore`` for
what any checkout would grow, ``.git/info/exclude`` for what is only yours, and
a machine's own ``core.excludesFile``, whose names go unpoliced here until they
are staged.
"""

from __future__ import annotations

import pathlib
import subprocess

import Microwave

ROOT = pathlib.Path(Microwave.__file__).resolve().parent.parent


def sources(*suffixes: str, root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    """Every file this project wrote, with one of these suffixes.

    Vendored code is somebody else's text and is not included. Neither is a
    repository cloned into the tree, which git marks with a trailing slash. That
    mark is what excludes it, and not the working tree having a directory there:
    a tracked symlink to a directory is a directory on disk and is ours.

    A path that is here in name only comes back like any other, and the guard
    reading it fails on it, named by the path it failed on. Dropping it would
    leave every caller policing a smaller tree and saying nothing.
    """
    found = run_git("ls-files", "-z", "--cached", "--others", "--exclude-standard", at=root)
    named = {root / name for name in found.split("\0") if name and not name.endswith("/")}
    return sorted(
        path
        for path in named
        if path.suffix in suffixes and "_vendor" not in path.relative_to(root).parts
    )


def run_git(*arguments: str, at: pathlib.Path) -> str:
    """What git said, or a refusal naming what went wrong.

    Refusing beats returning nothing: an empty list collects zero tests, which
    reads exactly like a rule that holds everywhere.
    """
    try:
        answered = subprocess.run(
            ["git", "-C", str(at), *arguments], capture_output=True, text=True
        )
    except OSError as why:
        raise RuntimeError(f"cannot list this repository's files - no git to ask: {why}") from why
    if answered.returncode != 0:
        raise RuntimeError(
            f"cannot list this repository's files - git said: {answered.stderr.strip()}"
        )
    return answered.stdout
