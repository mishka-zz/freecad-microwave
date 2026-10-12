# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The icon a chart's tab and its window show. The stubs cannot say it, since
the tab is Qt's. ``chart_icon_probe.py`` reads it under a real FreeCAD with its
main window up."""

import os

import pytest

from .conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "chart_icon_probe.py")


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    return probe_manifest(PROBE, tmp_path_factory.mktemp("icon"), "CHART_ICON_OUT", key=None)


def test_the_set_s_icon_is_not_the_application_s(made):
    """Otherwise the two tests below could not tell the two apart."""
    assert made["distinct"]


def test_a_marked_chart_shows_the_mark_on_its_tab_and_its_window(made):
    for where in ("window", "subwindow", "tab", "alone"):
        assert made["marked"][where] == {"mark": True, "application": False}, where


def test_a_chart_marked_with_nothing_shows_the_application_s(made):
    for where in ("window", "subwindow", "tab", "alone"):
        assert made["unmarked"][where] == {"mark": False, "application": True}, where
