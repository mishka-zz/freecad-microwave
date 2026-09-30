# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A drawing standing in containers is translated where FreeCAD shows it.

An object's ``Shape`` is in the coordinates of the container it stands in, and
the placement of each container above it is not in it. So a study whose metal
stands in an ``App::Part`` carrying a placement was a study of the metal
somewhere else, with every guard passing. Each case here is one drawing twice:
in its containers, and flat with every placement already applied, and each
adapter has to hand on the same thing for both. No reference and no solve: two
drawings of one device that must agree.

Why a subprocess
----------------

For the reason ``tests/test_acceptance_palace_waveguide.py`` gives: the drawing
needs the CAD kernel, and ``tests/placement_probe.py`` runs under ``freecadcmd``
to draw and translate it.
"""

from __future__ import annotations

import os

import pytest

from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "placement_probe.py")

CASES = [
    "part",
    "nested",
    "septum_in_part",
    "link_group",
    "linked_septum",
    "body",
    "study_in_part",
    "linked_array",
]

#: How far a coordinate either adapter hands on may sit from its flat twin, in
#: mm. A quarter turn composed in the tree and composed by hand differ in the
#: last bits, and nothing here is near a distance that means anything.
SAME_PLACE = 1e-9


@pytest.fixture(scope="module")
def translated(tmp_path_factory):
    out = tmp_path_factory.mktemp("placement")
    manifest = probe_manifest(PROBE, out, "PLACEMENT_OUT", key=None)
    assert manifest["cases"] == CASES, f"the probe stopped after {manifest['cases']}"
    return manifest


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("backend", ["openEMS", "Palace"])
def test_a_drawing_in_containers_is_handed_on_where_it_is_shown(translated, case, backend):
    contained, flat = translated[case]["contained"][backend], translated[case]["flat"][backend]
    assert "said" not in contained and "said" not in flat, (contained, flat)
    assert contained.keys() == flat.keys()
    for kind in flat:
        assert _numbers(contained[kind]) == pytest.approx(_numbers(flat[kind]), abs=SAME_PLACE), (
            kind
        )
        assert _names(contained[kind]) == _names(flat[kind]), kind


@pytest.mark.parametrize("backend", ["openEMS", "Palace"])
def test_another_document_showing_the_container_changes_nothing(translated, backend):
    here, flat = translated["elsewhere"][backend], translated["part"]["flat"][backend]
    assert "said" not in here, here
    for kind in flat:
        assert _numbers(here[kind]) == pytest.approx(_numbers(flat[kind]), abs=SAME_PLACE), kind


@pytest.mark.parametrize("backend", ["openEMS", "Palace"])
def test_an_object_in_two_containers_is_refused_naming_both(translated, backend):
    said = translated["twice"][backend].get("said", "")
    assert said.startswith("'Septum' stands in more than one container"), said
    assert "One.Septum." in said and "Other.Septum." in said


@pytest.mark.parametrize("backend", ["openEMS", "Palace"])
def test_a_container_shown_again_by_a_link_is_read_where_it_stands(translated, backend):
    """A link shows a copy, and what the study binds is the original."""
    here, flat = translated["instance"][backend], translated["part"]["flat"][backend]
    assert "said" not in here, here
    for kind in flat:
        assert _numbers(here[kind]) == pytest.approx(_numbers(flat[kind]), abs=SAME_PLACE), kind


@pytest.mark.parametrize("case", CASES)
def test_each_port_is_drawn_where_the_face_it_stands_on_is_shown(translated, case):
    """A port's box is worked out where the drawing is shown, and the port is
    shown moved by the containers it stands in itself - a study dragged into
    the part its drawing stands in is moved twice unless that is taken off."""
    contained, flat = translated[case]["contained"]["drawn"], translated[case]["flat"]["drawn"]
    assert len(flat) == 2 and all(abs(value) < 1e6 for value in _numbers(flat)), flat
    assert _numbers(contained) == pytest.approx(_numbers(flat), abs=SAME_PLACE)


@pytest.mark.parametrize("case", CASES)
def test_the_mesh_preview_is_drawn_where_the_drawing_is_shown(translated, case):
    contained = translated[case]["contained"]["preview"]
    flat = translated[case]["flat"]["preview"]
    assert contained == pytest.approx(flat, abs=SAME_PLACE)


def _numbers(rows):
    return [value for row in _flat(rows) for value in row if not isinstance(value, str)]


def _names(rows):
    return [value for row in _flat(rows) for value in row if isinstance(value, str)]


def _flat(rows):
    """Rows of numbers and names, whether the value is one row or a list of them."""
    if rows and not isinstance(rows[0], list):
        return [rows]
    return rows
