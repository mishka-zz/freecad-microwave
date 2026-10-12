# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A shipped document is meshed once, whichever port is driven.

A sweep plans one grid and re-excites it for each driven port, and the preview
draws that grid without choosing a port at all. Both rest on the grid not
depending on which port is the source. A grid pinning only the driven port's
plane breaks that: the run driving the next port is refused for want of a line
on its plane, the preview shows a grid that is not the one solved on, and a
mirrored two-port is meshed off its mirror because one end is pinned and the
other is not.

``tests/openems_one_grid_probe.py`` translates each shipped document under
``freecadcmd`` once per port, with every port marked a source, and this holds
what it wrote down.
"""

import os

import pytest

from .conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "openems_one_grid_probe.py")


@pytest.fixture(scope="session")
def read(tmp_path_factory):
    out = tmp_path_factory.mktemp("one_grid")
    return probe_manifest(PROBE, out, "ONE_GRID_OUT", key="read")


def two_ports(read):
    return {name: record for name, record in read.items() if record["ports"] == 2}


def test_the_measured_low_pass_is_among_what_was_read(read):
    """Its two lumped ports are each flat across the line, which is the port
    whose plane a run needs only while it drives."""
    assert "lumped" in two_ports(read)["stepped_lowpass_measured.FCStd"]["kinds"], sorted(read)


def test_every_port_driven_lays_the_same_grid(read):
    apart = {
        name: sorted({run["grid"] for run in record["runs"].values()})
        for name, record in read.items()
        if len({run["grid"] for run in record["runs"].values()}) > 1
    }
    assert not apart, apart


def test_the_preview_is_the_grid_solved_on(read):
    apart = [name for name, record in read.items() if record["preview"] != record["shipped"]]
    assert not apart, apart


def test_no_run_is_refused(read):
    refused = {
        f"{name} driving port {number}": run["refused"]
        for name, record in read.items()
        for number, run in record["runs"].items()
        if run["refused"]
    }
    assert not refused, refused


def test_a_shipped_two_port_draws_no_mirror_warning(read):
    """Every shipped two-port is drawn as its own mirror image between its
    ports. A two-port that is not belongs to a list of its own here."""
    warned = {
        name: record["mirror"] for name, record in two_ports(read).items() if record["mirror"]
    }
    assert not warned, warned
