# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A body standing in another of a different material is refused on openEMS in
either binding order, whatever the kernel drew it as.

openEMS fills a space two bodies share with one of them, by the order the
materials were bound. ``tests/openems_overlaps_probe.py`` draws a board with a
part standing in it under ``freecadcmd``, binds the two in both orders and
translates each. Where the part is cut out of the board, or is the board's own
material under another name, or is metal, the order decides nothing and the
drawing is taken. Metal stands above the board, which is handed over whole.

A lid sunk into the board by less than FLATNESS shares a face with it and is
taken. The same lid with a peg sunk from it into the board is refused, however
much of the lid lies in a skin beside the peg.
"""

from __future__ import annotations

import os

import pytest

from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "openems_overlaps_probe.py")

#: Drawings refused, and a phrase of the refusal.
REFUSED = {
    "box": "'Part' stands inside 'Board'",
    "closed_shell": "'Part' stands inside 'Board'",
    "sphere": "'Part' stands inside 'Board'",
    "invalid_solid": "'Part' is not a valid solid",
    "box_proud_of_the_board": "'Part' stands inside 'Board'",
    "lid_with_a_peg_in_the_board": "Cut one out of the other",
}

#: Drawings taken.
TAKEN = (
    "box_in_its_pocket",
    "sphere_in_its_pocket",
    "same_values_another_name",
    "pec",
    "lid_on_the_board",
)


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    return probe_manifest(PROBE, tmp_path_factory.mktemp("overlaps"), "OVERLAPS_OUT", key=None)


@pytest.fixture(scope="module")
def said(manifest):
    return manifest["cases"]


def test_the_probe_drew_every_case(said):
    assert set(said) == set(REFUSED) | set(TAKEN)


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_a_part_standing_in_the_board_is_refused_in_either_order(said, case):
    for order, answer in said[case].items():
        assert REFUSED[case] in answer, f"{case}, {order}: {answer}"


@pytest.mark.parametrize("case", TAKEN)
def test_a_drawing_the_order_cannot_change_is_taken(said, case):
    assert set(said[case].values()) == {"taken"}, said[case]


def test_metal_in_an_uncut_board_stands_above_it_and_the_board_is_handed_over_whole(manifest):
    board = manifest["board"]
    for order, handed in manifest["priorities"]["pec"].items():
        ((dielectric,), (metal,)) = handed["Board"], handed["Part"]
        assert metal > dielectric, f"{order}: {handed}"
    for order, filled in manifest["filled"]["pec"].items():
        assert sum(filled["Board"]) == pytest.approx(board, rel=1e-3), f"{order}: {filled}"


def test_metal_drawn_as_nested_surfaces_is_read_as_openems_fills_it(manifest):
    """The can's cavity is metal to openEMS, so copper standing in it stands in
    the can, and the two metals differ."""
    assert "'Block' stands inside 'Can'" in manifest["can"], manifest["can"]
