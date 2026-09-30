# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Bodies of perfect conductor under different bindings that meet, on Palace.

The mesher refuses a piece two labels of one priority were both drawn over, and
every body of metal stands at one. Two perfect conductors carry one condition,
so the translation draws bindings whose bodies touch or share a volume under
one label, the first binding's, and the space they share is metal whichever
fills it. A film thinner than FLATNESS is such a piece to the mesher and no
volume to the translation, so bodies meeting across one are joined as well.

Each drawing that meets is meshed as one metal, and those drawn again under a
single binding are compared with it: the metal holds as much surface, and the
region as much volume. Bodies standing apart stay two conductors.

Why a subprocess
----------------

For the reason ``tests/test_acceptance_palace_waveguide.py`` gives: the
drawing needs the CAD kernel, and ``tests/palace_metal_bodies_probe.py`` runs
under ``freecadcmd`` to draw it and mesh it. A machine without Palace or Gmsh
makes the probe say which, and the test skips.
"""

from __future__ import annotations

import os

import pytest

from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_metal_bodies_probe.py")

#: Each drawing whose bodies meet, as the label its metal is drawn
#: under and the bindings joined to it, in the order they were made.
JOINED = {
    "via_through_block": ("BlockMetal", ["ViaMetal"]),
    "via_buried_in_block": ("BlockMetal", ["ViaMetal"]),
    "via_buried_in_a_notched_block": ("BlockMetal", ["ViaMetal"]),
    "pad_and_barrel": ("PadMetal", ["BarrelMetal"]),
    "one_place": ("OneMetal", ["TwoMetal"]),
    "covered_by_two": ("LeftMetal", ["MiddleMetal", "RightMetal"]),
    "sharing_a_face": ("OneMetal", ["TwoMetal"]),
    "sharing_a_film": ("OneMetal", ["TwoMetal"]),
    "a_film_apart": ("OneMetal", ["TwoMetal"]),
}

#: Each drawing under several bindings, as the same bodies under one binding
#: and that binding's label.
ONE_BINDING = {
    "via_through_block": ("via_through_block_one_binding", "BlockViaMetal"),
    "pad_and_barrel": ("pad_and_barrel_one_binding", "PadBarrelMetal"),
    "sharing_a_film": ("sharing_a_film_one_binding", "OneTwoMetal"),
}

#: The region every drawing binds, and what its sizes are compared to.
REGION = "AirBinding"
SAME = 1e-9


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    """Every drawing, meshed under a real FreeCAD, as the probe left them."""
    out = tmp_path_factory.mktemp("palace_metal_bodies")
    read = probe_manifest(PROBE, out, "METAL_BODIES_OUT", key=None)
    if read.get("missing"):
        pytest.skip(f"this machine cannot drive Palace: {read['missing']}")
    return read


@pytest.mark.parametrize("case", sorted(JOINED))
def test_bodies_that_meet_are_meshed_as_one_metal(manifest, case):
    run = manifest[case]
    label, joined = JOINED[case]
    assert run["said"] == "", run["said"]
    assert run["conductors"] == [[label, joined]]
    assert run["sizes"][label] > 0.0
    assert not set(joined) & set(run["sizes"]), run["sizes"]


@pytest.mark.parametrize("case", sorted(ONE_BINDING))
def test_the_joined_metal_is_what_one_binding_draws(manifest, case):
    twin, alone = ONE_BINDING[case]
    label, _ = JOINED[case]
    joined, bound = manifest[case]["sizes"], manifest[twin]["sizes"]
    assert manifest[twin]["said"] == "", manifest[twin]["said"]
    assert joined[label] == pytest.approx(bound[alone], rel=SAME, abs=0.0)
    assert joined[REGION] == pytest.approx(bound[REGION], rel=SAME, abs=0.0)


@pytest.mark.parametrize(
    "case, labels",
    [
        ("apart", ("OneMetal", "TwoMetal")),
        ("in_the_corner_of_an_angle", ("AngleMetal", "PostMetal")),
    ],
)
def test_bodies_standing_apart_stay_two_conductors(manifest, case, labels):
    """The post stands inside the angle's box and clear of the angle itself."""
    run = manifest[case]
    assert run["said"] == "", run["said"]
    assert run["conductors"] == [[label, []] for label in labels]
    assert all(run["sizes"][label] > 0.0 for label in labels)
