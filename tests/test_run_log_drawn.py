# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The colour each line of a log is drawn in, in a panel's log and in a kept
run's log window, and which run Show Run Log opens.

The stubs cannot say it, since the colour is Qt's text format. ``run_log_probe.py``
reads it under a real FreeCAD with its main window up.
"""

import os

import pytest

from .conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "run_log_probe.py")


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    return probe_manifest(PROBE, tmp_path_factory.mktemp("log"), "RUN_LOG_OUT", key=None)


def test_a_warning_is_orange_a_failure_red_and_the_rest_its_own(made):
    """The colours are the status line's for a warning and a failure. A line
    after a coloured one is in the view's own colour, and a failure of two
    lines is red on both."""
    assert made["panel"] == [
        ["plain", None],
        ["WARNING: warned", "#ffa500"],
        ["after the warning", None],
        ["failed", "#ff0000"],
        ["and its second line", "#ff0000"],
        ["after the failure", None],
    ]


def test_a_kept_runs_window_draws_each_line_it_warned_of_orange(made):
    assert made["window"] == [
        ["Run 1 begins", None],
        ["[warn] Port1: a warning", "#ffa500"],
        ["of two lines", "#ffa500"],
        ["Run 1 ends", None],
    ]


def test_show_run_log_opens_the_run_picked_and_is_greyed_with_none(made):
    first, _ = made["labels"]
    assert made["active picked"] and made["picked"] == [f"Log of {first}"]
    assert not made["active unpicked"] and made["unpicked"] == []


def test_a_log_window_closes_with_its_document(made):
    """It belongs to the main window, so FreeCAD alone would leave it open over
    a run that is gone."""
    assert made["before closing"] == [f"Log of {made['labels'][1]}"]
    assert made["after closing"] == []
