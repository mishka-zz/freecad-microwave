# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Nothing here points at a file that is not here.

This workbench is developed inside a harness that also holds the architecture
document, the open-work list and the notes on openEMS' behaviour. It is
published on its own, so a pointer to any of those is a pointer to nothing: the
reader who most needs it is the one who cannot follow it.

The fix is never a relative path that climbs out. Where the reference carried a
fact, the fact is written where it was needed; where it carried only provenance,
it goes. A citation of somebody else's *source* is different and stays - openEMS
is public, and ``openEMS/FDTD/operator.cpp:517`` resolves for anyone who has it.
So is a clause of somebody else's *standard*, which is spelled exactly like an
item's identifier and is told apart by the unit named beside it.

A convention nobody checks is a convention that drifts, so this states the rule
in a form that fails.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from tests import repo
from tests.repo import ROOT

SUFFIXES = (".py", ".md", ".toml", ".yml", ".cfg")

#: This file, which has to name what it forbids in order to forbid it.
SELF = pathlib.Path(__file__).resolve()

#: The harness around this repository: its documents, and the read-only upstream
#: sources it keeps beside them. A bare ``§4.2`` is on the list too - it cites
#: the same architecture document without even saying which document it is, and
#: so does a bare ``M2``, which names one of that document's milestones. Where a
#: milestone carried a fact, say the fact: "a second solver in one document" is
#: what the reader needed, and it goes on being true after the schedule moves.
#: The harness's own documents are named here without their directory too. A
#: bare ``debt.md`` reaches just as far outside as ``plans/debt.md`` does, and
#: reads as a file the workbench ships - which is worse, since a reader will
#: look for it here first. So does an item's bare identifier: ``D-17`` names a
#: line of a list this repository does not have, and the fix is the same one -
#: say what the item said. A part number keeps its letter, because the boundary
#: on the left refuses a citation that follows one, as ``WR-42`` does.
OUTSIDE = re.compile(
    r"""
      (?<![\w/])(?:specification|AGENT|CLAUDE|GEMINI)\.md
    | (?<![\w/])(?:debt|status|decisions|ideas|use_cases|ui_ux_strategy)\.md
    | (?<![\w/])(?:engine-quirks|literature)\.md
    | (?<![\w/])(?:plans|notes|archive|references)/
    | §\s*\d
    | (?<![\w.])M[0-9](?![\w.])
    | (?<![\w-])[A-Z]-\d+(?![\w-])
    """,
    re.VERBOSE,
)

#: An appendix of a published standard numbers its subsections the way this
#: project once numbered its open items, so ``C-5`` is both a clause anyone can
#: buy and a line of a list only the harness has. What tells them apart is on
#: the left: a clause is introduced by the unit it is a clause of, and an
#: identifier stands bare. Adjacency is the whole test - a line that mentions an
#: appendix elsewhere in it exempts nothing.
CITED = re.compile(
    r"(?i:(?:sub)?section|appendix|annex|clause|para\.|table|fig(?:ure|\.))\s+[A-Z]-\d+(?![\w-])"
)


def sources():
    return [path for path in repo.sources(*SUFFIXES) if path != SELF]


def offences(path: pathlib.Path) -> list[str]:
    """Every line of one file that reaches outside this repository."""
    found = []
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if OUTSIDE.search(CITED.sub(" ", line)):
            found.append(f"{path.name}:{number}: {line.strip()[:80]}")
    return found


@pytest.mark.parametrize("path", list(sources()), ids=lambda p: str(p.relative_to(ROOT)))
def test_nothing_points_outside_this_repository(path):
    assert offences(path) == []


def test_the_check_knows_a_citation_from_a_dangling_pointer(tmp_path):
    kept = tmp_path / "kept.py"
    kept.write_text(
        "# openEMS `FDTD/operator.cpp:517`, and `openEMS/ports.py`:355.\n"
        "# A FreeCAD path like Mod/Plot, and a parameter group Mod/Microwave.\n"
        "# Section 4.2 spelled out in words is prose, not a pointer.\n"
        "# A mode name carries one: TM01.\n"
        "# A part number is not an item: WR-42, and a coax type RG-58.\n"
        "# A clause names its unit: subsection C-7, Appendix C-6, Fig. B-4.\n",
        encoding="utf-8",
    )
    assert offences(kept) == []

    caught = tmp_path / "caught.py"
    caught.write_text(
        "# See specification.md and ``AGENT.md``.\n"
        "# ``plans/debt.md`` M-2, notes/engine-quirks.md, references/openEMS/x.\n"
        "# What §4.2 forbids.\n"
        "# What M2 wanted, and what M1 was reviewed for.\n"
        "# ``debt.md`` T-6 names it without the directory, and so does status.md.\n"
        "# ui_ux_strategy.md is one of them too.\n"
        "# D-17, which names a line of a list this repository does not ship.\n"
        "# An appendix mentioned somewhere on the line exempts nothing: C-7.\n",
        encoding="utf-8",
    )
    # By line, rather than by how many there are: a count says nothing about
    # *which* rule stopped working, and the line a rule is on is what a change
    # to the pattern moves.
    assert [offence.split(":")[1] for offence in offences(caught)] == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
    ]
