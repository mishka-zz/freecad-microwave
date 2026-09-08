# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A recompute asks the document nothing, and what moves no cell fires none.

Deriving the staleness key translates the document's geometry, which is the
whole cost of a recompute on a real board. So the preview marks the drawing out
of date rather than asking, and each class that owns properties keeps the ones
that move no cell off the dependency graph - FreeCAD leaves an object up to
date when the property assigned to it is marked ``Output``. The study cannot be
reached that way and marks the drawing itself.

The fast suite cannot see any of this. Whether an assignment touches its object
is FreeCAD's decision, and the suite's stubs model it; a stub asserting the
answer would be asserting itself. The probe therefore counts under a real
FreeCAD, and this reads what it counted.
"""

import os

import pytest

from Microwave.Objects.analysis import EMAnalysis

from .conftest import probe_manifest

PROBE = os.path.join(os.path.dirname(__file__), "preview_recompute_probe.py")

#: Edits that must reach the preview's ``execute`` on no document and leave the
#: badge where it was: the display properties, which are a request to look at
#: the same grid differently; the solver's run length, threads and where its
#: interpreter is; whether a port drives the run; how finely the answer is
#: reported; and a result object arriving in the study after a solve.
#:
#: Written out rather than read off the classes under test. Taking the list
#: from ``MOVES_NO_CELL`` would make a declaration that stopped covering a
#: property stop being tested for it, which is the one failure this file is
#: here to catch.
MOVES_NO_CELL = (
    "Display",
    "EnergyDecay",
    "Excitation",
    "MaxTimesteps",
    "NumFrequencyPoints",
    "ShowSliceX",
    "ShowSliceY",
    "ShowSliceZ",
    "SimDir",
    "SliceX",
    "SliceY",
    "SliceZ",
    "SolverPython",
    "Threads",
    "TimestepFactor",
    "a result arriving",
    "a material moved into the study",
)

#: Changes that move a cell. Each has to leave the drawing reported as no
#: longer describing the document, and none may translate the document to say
#: so.
#:
#: ``a bound solid`` makes the example corpus a contract: every shipped
#: document with a study has a bound solid carrying a ``Length`` to edit. A
#: specimen drawn some other way wants its own edit here rather than a skip,
#: since a case that is skipped passes by not being counted.
#:
#: ``a bound solid touched`` is here rather than among the quiet cases. A touch
#: says only that something might have moved, and the badge now believes it -
#: the drawing is reported stale for a solid edited and put back, and what that
#: costs is one press of Update Mesh.
MOVES_A_CELL = (
    "the mesh policy",
    "a bound solid",
    "an edit behind a slice nudge",
    "the frequency band",
    "a bound solid touched",
    "a region arriving",
    "a region arriving in a subgroup",
)


#: What the sweep is allowed to find the badge quiet on while the key moves.
#: A label is in the payload the key is taken over, so renaming a port moves it
#: - and no class declares a label a grid input, because a rename moves no
#: cell. The key is what is wrong here, and the panel reports a renamed port as
#: a changed model today. Correcting it moves every stored digest, so it is
#: separate work.
THE_KEY_OVER_REPORTS = ("port.Label",)


@pytest.fixture(scope="session")
def probed(tmp_path_factory):
    out = tmp_path_factory.mktemp("preview_recompute")
    return probe_manifest(PROBE, out, "PREVIEW_RECOMPUTE_OUT", key=None)


@pytest.fixture(scope="session")
def counted(probed):
    return probed["counted"]


@pytest.fixture(scope="session")
def declared(probed):
    return probed["declared"]


def test_every_shipped_document_was_counted(counted):
    """A run that counted nothing passes every assertion below it."""
    assert counted, "the probe opened no document with a study in it"


@pytest.mark.parametrize("name", MOVES_NO_CELL)
def test_a_change_that_moves_no_cell_is_not_recomputed_at_all(counted, name):
    """The declaration is what does this. An edit to a marked property touches
    nothing, so FreeCAD's graph never reaches the preview."""
    ran = {
        document: case[name]["executes"]
        for document, case in counted.items()
        if name in case and case[name]["executes"]
    }
    assert not ran, ran


@pytest.mark.parametrize("name", MOVES_NO_CELL)
def test_a_change_that_moves_no_cell_leaves_the_preview_up_to_date(counted, name):
    touched = [
        document
        for document, case in counted.items()
        if name in case and "Touched" in case[name]["state"]
    ]
    assert not touched, touched


@pytest.mark.parametrize("name", MOVES_NO_CELL)
def test_a_change_that_moves_no_cell_leaves_the_verdict_where_it_was(counted, name):
    """Nothing about the grid moved, so the badge must not have."""
    moved = [
        document
        for document, case in counted.items()
        if name in case and case[name]["badge"] != "Current"
    ]
    assert not moved, moved


@pytest.mark.parametrize("name", MOVES_NO_CELL)
def test_every_document_exercised_each_quiet_change(counted, name):
    """A case skipped for want of a property to edit would pass the three
    above by never being counted."""
    missing = [document for document, case in counted.items() if name not in case]
    assert not missing, missing


@pytest.mark.parametrize("name", MOVES_A_CELL)
def test_a_change_to_what_the_grid_is_laid_from_moves_the_verdict(counted, name):
    wrong = {
        document: case[name]["badge"]
        for document, case in counted.items()
        if name in case and case[name]["badge"] != "Out of date"
    }
    assert not wrong, wrong


@pytest.mark.parametrize("name", MOVES_A_CELL)
def test_every_document_exercised_each_change(counted, name):
    """A case skipped for want of a solid to edit would pass the one above by
    never being counted."""
    missing = [document for document, case in counted.items() if name not in case]
    assert not missing, missing


@pytest.mark.parametrize("name", MOVES_A_CELL + MOVES_NO_CELL)
def test_no_edit_translates_the_document(counted, name):
    """The point of the whole change. The key is derived where somebody reads
    it - the panel, Run and Check - and on no edit."""
    asked = {
        document: case[name]["asks"]
        for document, case in counted.items()
        if name in case and case[name]["asks"]
    }
    assert not asked, asked


def test_the_band_reaches_the_badge_without_a_link(counted):
    """The study cannot be linked: it holds the preview in its group, and a
    link back would close a cycle. It marks the drawing itself, which is why
    ``the frequency band`` is a case at all."""
    assert "FrequencyStop" not in EMAnalysis.MOVES_NO_CELL
    quiet = [
        document
        for document, case in counted.items()
        if case["the frequency band"]["badge"] != "Out of date"
    ]
    assert not quiet, quiet


def test_a_recompute_with_nothing_to_do_reaches_no_preview(counted):
    """The control on the counts above. A zero in the quiet cases means
    something only because a recompute is what would have made it one."""
    ran = {
        document: case["nothing at all"]["executes"]
        for document, case in counted.items()
        if case["nothing at all"]["executes"]
    }
    assert not ran, ran


def test_the_badge_fires_wherever_the_key_moves(declared):
    """The declaration is a complement, so a property nobody thought about is
    an input. This is what says the complement is the right way round: over
    every property of every object a study owns, the badge is allowed to be
    louder than the key and never quieter.
    """
    quiet = {
        f"{document} {name}": entry["badges"]
        for document, swept in declared.items()
        for name, entry in swept.items()
        if entry["grid"] and "Out of date" not in entry["badges"]
    }
    allowed = {f"{document} {name}" for document in declared for name in THE_KEY_OVER_REPORTS}
    assert set(quiet) <= allowed, quiet


def test_every_document_was_swept(declared):
    """A sweep that perturbed nothing passes the assertion above it. Every kind
    a study owns has to have contributed, or a whole class could stop being
    swept without anything saying so."""
    assert declared
    wanted = {"study", "solver", "policy", "binding", "port"}
    thin = {
        document: sorted(wanted - {name.split(".")[0] for name in swept})
        for document, swept in declared.items()
        if wanted - {name.split(".")[0] for name in swept}
    }
    assert not thin, thin
