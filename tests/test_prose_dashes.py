# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""One dash in prose, and it is a plain ASCII hyphen.

Comments, docstrings, messages and Markdown all use ``-``. An em dash, an en
dash, a minus sign, ``--`` and ``---`` are mojibake or noise in the places this
text is actually read: a terminal, FreeCAD's report view, a ``git log``. Three
spellings of one mark also make it ungreppable.

A convention nobody checks is a convention that drifts, and this one spans every
file in the tree, so a single careless line would start it back. Hence a test
rather than a paragraph: this states the rule in a form that fails.

What is *not* punctuation, and stays: a drawn rule of four or more hyphens, a
``--flag``, an ``-->`` arrow, and anything quoted verbatim from somebody else,
which in Markdown means everything inside a fence or backticks. Quoting openEMS'
own docstring or a ``git log -- path`` is not this project's prose to normalise.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from tests import repo
from tests.repo import ROOT

SUFFIXES = (".py", ".md", ".toml", ".yml")

#: This file, which is the one place every spelling the rule forbids has to
#: appear in order to be forbidden.
SELF = pathlib.Path(__file__).resolve()

#: Never syntax anywhere, so never allowed anywhere: em dash, en dash, minus
#: sign. The superscript minus of ``10⁻⁵`` and the ``±`` of a tolerance are not
#: dashes and are not listed.
TYPOGRAPHIC = re.compile(r"[—–−]")

#: A run of two or three used as punctuation, in each of the shapes it takes.
#:
#: **Standing alone**: opening the line or after whitespace, with none of a word
#: character, ``>`` or a further hyphen after it - which is what excludes
#: ``--build-only``, ``-->`` and ``<--``. The line start counts, a wrapped
#: sentence putting its dash in column 0 as readily as mid-line.
#:
#: **Opening a continued string literal**: a quote before and whitespace after,
#: which is where a message split across two literals puts its dash. The
#: whitespace is what tells it from a ``"--"`` that is a value - matplotlib
#: spells a dashed linestyle that way.
#:
#: **Closed up between two words**: ``1.5--5.5``, ``Hammerstad--Bekkadal``. It
#: can be nothing else there, a flag never having a word character before it.
PROSE = re.compile(r"(?:^|(?<=\s))-{2,3}(?![\w>-])|(?<=[\"'])-{2,3}(?=\s)|(?<=\w)-{2,3}(?=\w)")

#: A drawn rule: a section banner, a table separator, an RST underline.
RULE = re.compile(r"-{4,}")
BLANKISH = re.compile(r"[\s|:+\-]*\Z")
FENCE = re.compile(r"\s*(```|~~~)")
CODE = re.compile(r"`[^`]*`")


def sources():
    return [path for path in repo.sources(*SUFFIXES) if path != SELF]


def offences(path: pathlib.Path) -> list[str]:
    """Every line of one file that spells a dash some other way."""
    markdown = path.suffix == ".md"
    found, fenced = [], False
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        where = f"{path.name}:{number}: {line.strip()[:80]}"
        if TYPOGRAPHIC.search(line):
            found.append(where)
        if markdown and FENCE.match(line):
            fenced = not fenced
            continue
        if fenced or BLANKISH.fullmatch(line) or RULE.search(line):
            continue
        # A dash inside a code span belongs to the code, so spans are blanked
        # rather than split out: splitting loses the columns, and the rule turns
        # on one of them - a run in column 0 is prose too.
        probe = CODE.sub(lambda span: "\0" * len(span.group()), line) if markdown else line
        if PROSE.search(probe):
            found.append(where)
    return found


@pytest.mark.parametrize("path", list(sources()), ids=lambda p: str(p.relative_to(ROOT)))
def test_prose_spells_a_dash_as_one_ascii_hyphen(path):
    assert offences(path) == []


def test_the_check_can_tell_punctuation_from_the_things_that_look_like_it(tmp_path):
    """The exceptions are the whole difficulty, so they are asserted, not assumed."""
    kept = tmp_path / "kept.md"
    kept.write_text(
        "Run `pytest --collect-only -q`, see `git log -- path`.\n"
        "\n"
        "| gate | bound |\n"
        "|---|---|\n"
        "\n"
        "```\n"
        "size: 30000 ( --> 13.7 * Excitation)\n"
        "--- A a=0.010668 b=0.004318\n"
        "prose --- inside a fence is quoted output\n"
        "```\n"
        "\n"
        "# ------------------------------------------------- steps\n",
        encoding="utf-8",
    )
    assert offences(kept) == []

    caught = tmp_path / "caught.md"
    caught.write_text(
        "an em — dash\nan en – dash\na minus −1\nand --- this\n--- and this, in column 0\n",
        encoding="utf-8",
    )
    assert len(offences(caught)) == 5
