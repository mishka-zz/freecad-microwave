# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Every default the manual states is the default the object has.

A property table is the one place documentation carries figures, and a figure
in prose is the one thing nothing re-runs. So the tables name the object they
describe, in a comment the reader never sees, and this reads the value back off
a freshly created one.

What it can check is what the cell claims. A default written as a number, a
word from an enumeration or a boolean is compared; a cell saying where the
value comes from - ``inferred``, ``from the band`` - claims no figure and is
passed over. A stale number and a stale choice are therefore caught, which is
the drift that happens.

``<axis>`` and ``<side>`` in a property name stand for the six faces, and each
one is checked.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import Microwave
from Microwave.Objects.analysis import createEMAnalysis
from Microwave.Objects.mesh import (
    createEMGmshMesh,
    createEMMeshPolicy,
    createEMMeshRegion,
    createEMYeeGrid,
)
from Microwave.Objects.ports import (
    createEMPortCoaxial,
    createEMPortLumped,
    createEMPortMicrostrip,
    createEMPortRectWaveguide,
)
from Microwave.Objects.solver import createEMSolverOpenEMS, createEMSolverPalace

DOCS = pathlib.Path(Microwave.__file__).resolve().parent.parent / "docs"

#: What each documented object is made by. A table names one of these keys.
FACTORIES = {
    "EMAnalysis": createEMAnalysis,
    "EMSolverOpenEMS": createEMSolverOpenEMS,
    "EMSolverPalace": createEMSolverPalace,
    "EMMeshPolicy": createEMMeshPolicy,
    "EMMeshRegion": createEMMeshRegion,
    "EMYeeGrid": createEMYeeGrid,
    "EMGmshMesh": createEMGmshMesh,
    "EMPortMicrostrip": lambda doc: createEMPortMicrostrip(doc=doc),
    "EMPortLumped": lambda doc: createEMPortLumped(doc=doc),
    "EMPortRectWaveguide": lambda doc: createEMPortRectWaveguide(doc=doc),
    "EMPortCoaxial": lambda doc: createEMPortCoaxial(doc=doc),
}

#: The comment that ties a table to an object, invisible where the page renders.
MARKER = re.compile(r"<!--\s*defaults:\s*([\w, ]+?)\s*-->")

_ROW = re.compile(r"^\|(.*)\|\s*$")
_TEMPLATE = re.compile(r"<axis>|<side>")


def tables():
    """``(page, object name, property, stated default)`` for every marked row."""
    found = []
    for page in sorted(DOCS.glob("*.md")):
        lines = page.read_text(encoding="utf-8").split("\n")
        for number, line in enumerate(lines):
            marked = MARKER.search(line)
            if not marked:
                continue
            for name in (part.strip() for part in marked.group(1).split(",")):
                found += [
                    (page.name, name, prop, stated)
                    for prop, stated in _rows(lines[number + 1 :], page.name, number)
                ]
    return found


def _rows(lines, page, number):
    """The property and default column of the table starting after a marker."""
    body = []
    for line in lines:
        row = _ROW.match(line)
        if row is None:
            break
        body.append([cell.strip() for cell in row.group(1).split("|")])
    if len(body) < 3:
        raise AssertionError(f"{page}:{number + 1}: the marker is not above a table")
    for cells in body[2:]:
        yield cells[0].strip("`"), cells[1]


def _expand(name):
    """One property name per face, where the name is written as a template."""
    if not _TEMPLATE.search(name):
        return [name]
    return [
        name.replace("<axis>", axis).replace("<side>", side)
        for axis in ("X", "Y", "Z")
        for side in ("Min", "Max")
    ]


def _stale(stated, value, choices=()):
    """Whether this cell states a default the object does not have.

    A cell that agrees - as a word or as a number - holds. A *number* that
    disagrees has gone stale, and so has one of the property's own
    ``choices``. Anything else is prose about where the value comes from,
    ``inferred`` or ``assigned``, and claims nothing to check.
    """
    if stated.lower() == str(value).lower():
        return False
    if stated.lower() in (choice.lower() for choice in choices):
        return True
    try:
        return float(stated) != float(value)
    except (TypeError, ValueError):
        return False


def _states_wrongly(created, name, stated):
    """Whether the cell ``stated`` is a default ``created.<name>`` does not have."""
    try:
        choices = created.getEnumerationsOfProperty(name) or ()
    except AttributeError:
        choices = ()
    return _stale(stated, getattr(created, name), choices)


@pytest.mark.parametrize(
    "page,obj,prop,stated",
    tables(),
    ids=lambda part: str(part).replace(" ", "-"),
)
def test_the_manual_states_the_default_the_object_has(page, obj, prop, stated, doc):
    assert obj in FACTORIES, f"{page}: no factory for {obj!r}"
    created = FACTORIES[obj](doc)
    for name in _expand(prop):
        assert hasattr(created, name), f"{page}: {obj} has no property {name!r}"
        assert not _states_wrongly(created, name, stated), (
            f"{page}: {obj}.{name} defaults to {getattr(created, name)!r}, not {stated!r}"
        )


class TestWhatACellClaims:
    """Every cell the manual holds today agrees, so the check over the pages
    says nothing about whether a disagreeing one would be caught. These hold a
    cell of each kind against a port whose value each test sets itself, so
    none of them moves with a default."""

    def test_another_of_the_propertys_own_choices_is_stale(self, doc):
        port = createEMPortRectWaveguide(doc=doc)
        port.ReferencedTo = "Port impedance"
        assert _states_wrongly(port, "ReferencedTo", "Fixed impedance")
        assert _states_wrongly(port, "ReferencedTo", "fixed impedance")

    def test_the_value_among_them_holds(self, doc):
        port = createEMPortRectWaveguide(doc=doc)
        port.ReferencedTo = "Fixed impedance"
        assert not _states_wrongly(port, "ReferencedTo", "Fixed impedance")

    def test_prose_about_where_the_value_comes_from_claims_nothing(self, doc):
        port = createEMPortRectWaveguide(doc=doc)
        assert not _states_wrongly(port, "PropagationAxis", "inferred")

    def test_another_number_is_stale(self, doc):
        port = createEMPortRectWaveguide(doc=doc)
        port.ReferenceImpedance = 50.0
        assert _states_wrongly(port, "ReferenceImpedance", "75.0")


def test_every_page_the_index_offers_is_there():
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    for link in re.findall(r"\]\(([\w.]+\.md)\)", index):
        assert (DOCS / link).exists(), f"the index links to {link}, which is not there"


def test_every_figure_a_page_shows_is_there():
    """A figure is drawn by ``docs/images/build.py`` and can go missing.

    It is written by a script nothing else runs, under an interpreter the suite
    does not have, so the picture and the page that shows it drift apart in the
    one direction a reader notices: a broken image.
    """
    for page in sorted(DOCS.glob("*.md")):
        for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", page.read_text(encoding="utf-8")):
            assert (DOCS / link).exists(), f"{page.name} shows {link}, which is not there"


def test_every_page_is_reachable_from_the_index():
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    linked = set(re.findall(r"\]\(([\w.]+\.md)\)", index))
    for page in DOCS.glob("*.md"):
        if page.name != "README.md":
            assert page.name in linked, f"{page.name} is not on the index"
