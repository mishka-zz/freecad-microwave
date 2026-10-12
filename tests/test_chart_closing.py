# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A chart tab closes with the first document it draws from, whatever route
opened it, and a chart drawn from no document stays.

``Mod/Plot`` puts a tab in the MDI area with no document behind it, so FreeCAD
closes it with none. The stubs cannot say what FreeCAD's own close does to a
tab; ``chart_closing_probe.py`` drives it under a real FreeCAD with its main
window up.
"""

import os

import pytest

from .conftest import probe_manifest

PROBE = os.path.join(os.path.dirname(__file__), "chart_closing_probe.py")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    return probe_manifest(PROBE, tmp_path_factory.mktemp("closing"), "CLOSING_OUT", key=None)


def test_every_chart_was_open_but_the_one_closed_by_hand(made):
    assert made["before"] == ["both", "first", "loose", "panel", "second", "traced"]


def test_each_chart_drawn_from_the_document_closed_closes_with_it(made):
    assert made["after"] == ["loose", "second"]


def test_a_tab_closed_and_one_closed_with_its_document_are_forgotten(made):
    assert made["held"] == 1
