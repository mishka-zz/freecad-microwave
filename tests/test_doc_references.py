# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A page this project cites is a page it still has.

Reasoning too long to sit beside the code it governs lives in
``docs/internals``, and the code says which page. That pointer is the whole of
what holds the two together, and it is exactly the kind of thing that breaks
without a sound: the page is renamed or a section retitled, the citation goes on
reading as though it were still true, and the next person to follow it finds
nothing. A derivation nobody can reach has been deleted - slowly, and without
anybody deciding to.

So the pointer is checked. Every ``docs/internals/...`` reference names a file
that is there, and a ``#section`` on the end of one names a heading in that file.
A page linking to a sibling is held to the same thing: that link is spelled
relatively, so it looks nothing like the citations the code makes and would
otherwise be the one pointer here that nothing watches.

This is the same bargain the self-containment rule makes. A convention nobody
checks is a convention that drifts, and the answer is to state it in a form that
fails.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from tests import repo
from tests.repo import ROOT

SUFFIXES = (".py", ".md", ".toml")

#: This file, which has to spell out a broken citation in order to forbid one.
SELF = pathlib.Path(__file__).resolve()

#: Where the pages live.
INTERNALS = ROOT / "docs" / "internals"

#: A citation: the page, and optionally the section of it. Bounded on the left
#: so that a longer path ending in these characters is not read as one.
REFERENCE = re.compile(r"(?<![\w/])docs/internals/([\w.-]+\.md)(?:#([\w-]+))?")

#: A page linking to a sibling, which is how the pages cross-reference each
#: other and how the index names them. It is spelled relatively - ``(page.md)``
#: rather than the full path - so the pattern above never sees it, and a rename
#: would break exactly the link this file exists to keep.
SIBLING = re.compile(r"\]\(([\w.-]+\.md)(?:#([\w-]+))?\)")


def anchor(title: str) -> str:
    """A heading written the way a link to it is written.

    The spelling Markdown viewers agree on: lowercased, spaces to hyphens, and
    anything that is neither a letter, a digit nor a hyphen dropped - which is
    what becomes of the backticks a heading here tends to carry.
    """
    kept = "".join(c for c in title.lower() if c.isalnum() or c in " -")
    return kept.strip().replace(" ", "-")


def anchors(page: pathlib.Path) -> set[str]:
    """Every section of a page that can be linked to."""
    return {
        anchor(line.lstrip("#").strip())
        for line in page.read_text(encoding="utf-8").split("\n")
        if line.startswith("#")
    }


def offences(path: pathlib.Path, internals: pathlib.Path = INTERNALS) -> list[str]:
    """Every citation in one file that no longer arrives anywhere."""
    found = []
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        cited = list(REFERENCE.findall(line))
        if path.parent == internals:
            cited += SIBLING.findall(line)
        for name, section in cited:
            page = internals / name
            if not page.is_file():
                found.append(f"{path.name}:{number}: no page {name}")
            elif section and section not in anchors(page):
                found.append(f"{path.name}:{number}: {name} has no section {section}")
    return found


def sources():
    return [path for path in repo.sources(*SUFFIXES) if path != SELF]


@pytest.mark.parametrize("path", list(sources()), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_page_this_file_cites_is_still_there(path):
    assert offences(path) == []


def test_the_check_knows_a_live_citation_from_a_dead_one(tmp_path):
    internals = tmp_path / "internals"
    internals.mkdir()
    (internals / "sizing.md").write_text(
        "# Spending a thickness\n\n## Why not the cheaper objective\n", encoding="utf-8"
    )

    kept = tmp_path / "kept.py"
    kept.write_text(
        "# The working is in docs/internals/sizing.md.\n"
        "# A section of it: docs/internals/sizing.md#spending-a-thickness.\n"
        "# And one whose title carried punctuation and a backtick:\n"
        "#   docs/internals/sizing.md#why-not-the-cheaper-objective\n"
        "# Somebody else's path is not one of these: Mod/docs/internals/x.md\n",
        encoding="utf-8",
    )
    assert offences(kept, internals) == []

    # A page linking to a sibling, which only counts inside the directory.
    sibling = internals / "index.md"
    sibling.write_text(
        "See [the working](sizing.md) and [one part](sizing.md#spending-a-thickness).\n",
        encoding="utf-8",
    )
    assert offences(sibling, internals) == []

    broken = internals / "stale.md"
    broken.write_text("Renamed: [gone](allocation.md).\n", encoding="utf-8")
    assert [o.split(":")[1] for o in offences(broken, internals)] == ["1"]

    outside = tmp_path / "elsewhere.md"
    outside.write_text("A link to [something](allocation.md) not in internals.\n", encoding="utf-8")
    assert offences(outside, internals) == []

    caught = tmp_path / "caught.py"
    caught.write_text(
        "# Renamed out from under it: docs/internals/allocation.md\n"
        "# Section retitled: docs/internals/sizing.md#spending-the-budget\n",
        encoding="utf-8",
    )
    # By line, rather than by how many: which citation died is the thing a
    # reader has to act on, and a tally is the same either way.
    assert [offence.split(":")[1] for offence in offences(caught, internals)] == ["1", "2"]
