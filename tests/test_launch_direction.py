# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Which way a port launches, scored on conductors a real kernel drew.

The rule itself is checked against boxes in ``test_picks.py``, and boxes are
where it is easy. These are the shapes that are not boxes: a trace narrowing to
a point, a line running into a circular patch, a hairpin turned through an arc,
a coaxial annulus whose own centre is in the hole. Each of them breaks a rule
that reads a shape's boundary planes instead of the material beside the pick,
and the last one breaks any rule that probes a point read off the pick's box.

What each answer should be is arithmetic on the dimensions in ``launches.py``,
written there beside the drawing and never read off a run.
"""

import os

import pytest

from .conftest import probe_manifest
from .launches import LAUNCHES, SILENT, SURFACES, TWINS

PROBE = os.path.join(os.path.dirname(__file__), "launch_probe.py")


def _named(name):
    return next(launch for launch in LAUNCHES + SILENT if launch.name == name)


@pytest.fixture(scope="session")
def read(tmp_path_factory):
    """Draw every conductor under a real FreeCAD once, and hand back what the
    rule answered on each."""
    out = tmp_path_factory.mktemp("launches")
    return probe_manifest(PROBE, out, "LAUNCH_OUT", key="read")


def _answer(read, launch):
    assert launch.name in read, (
        f"{launch.name!r} was not read at all, so the probe stopped before "
        "reaching it. A specimen that is not run is not a specimen that passed"
    )
    record = read[launch.name]
    if "unavailable" in record:
        pytest.skip(f"{launch.name} would not draw here: {record['unavailable']}")
    return record


@pytest.mark.parametrize("launch", LAUNCHES, ids=lambda launch: launch.name)
def test_the_wave_goes_into_the_metal_behind_the_face_it_starts_on(read, launch):
    record = _answer(read, launch)
    assert record["inward"] == launch.inward, (
        f"{launch.name}: {launch.why}. The metal runs "
        f"{'+' if launch.inward > 0 else '-'} along axis {launch.axis} from the "
        f"plane at {launch.at:g}, and the pick {record['element']} answered "
        f"{record['inward']}"
    )


@pytest.mark.parametrize("launch", SILENT, ids=lambda launch: launch.name)
def test_a_shape_that_does_not_say_is_not_guessed_at(read, launch):
    record = _answer(read, launch)
    assert record["inward"] is None, f"{launch.name}: {launch.why}"


@pytest.mark.parametrize("launch", SURFACES, ids=lambda launch: launch.name)
def test_a_conductor_drawn_as_a_shell_answers_the_same_way_or_not_at_all(read, launch):
    """A surface has no volume for a pick to be on a side of.

    So the rule is allowed to decline here and is not allowed to answer the
    other way round: a decline leaves the axis to the user and says so, while a
    reversed answer refuses the true declaration and accepts its opposite.

    It is the property and not the answers because which of these speak is a
    fact about the kernel rather than about the workbench - a wall that runs
    parallel to the step keeps the moved pick on the surface it came from, and
    a wall that slopes does not. Pinning the ones that answer today would pin
    that, and it would fail on a legitimate change to the probe length.

    It does not choose between two rules. Every specimen here is one drawn
    conductor, and a rule reading the body's boundary rather than its material
    answers all of them correctly - it comes apart on a drawing in two lumps,
    where a second surface lies flush against the pick on the air side. There is
    no such specimen here, and one would have to state a truth that the drawing
    genuinely leaves open.
    """
    record = _answer(read, launch)
    assert record["inward"] in (launch.inward, None), (
        f"{launch.name}: {launch.why}. The metal runs "
        f"{'+' if launch.inward > 0 else '-'} along axis {launch.axis}, and the "
        f"pick {record['element']} answered {record['inward']} - which is the "
        "other way, so a port declared correctly would be refused and its "
        "reverse accepted"
    )


def test_a_sheet_is_picked_by_an_edge_and_a_solid_by_a_face(read):
    """The two ways a cross-section arrives, and one rule serving both.

    A conductor drawn as a surface has no face square to the axis it runs
    along, so what a user picks at its end is an edge. Nothing above this knows
    the difference, and this is the assertion that says so.
    """
    assert _answer(read, _named("fold_sheet"))["element"].startswith("Edge")
    assert _answer(read, _named("fold_extruded"))["element"].startswith("Face")


def test_the_same_fold_answers_the_same_however_it_was_built(read):
    """Drawn as one extrusion or fused from three boxes - which arrive a solid
    and a compound - it is one conductor and one launch."""
    answers = {_answer(read, _named(name))["inward"] for name in ("fold_extruded", "fold_fused")}
    assert answers == {-1}


def test_every_specimen_was_drawn(read):
    """A probe that died part-way leaves the rest looking untested rather than
    failing, so what was read is checked against what was asked for."""
    assert set(read) == {launch.name for launch in LAUNCHES + SILENT + SURFACES}


def test_every_conductor_is_drawn_a_second_time_as_a_surface():
    """What the check above cannot see, because it compares two derived sets.

    A conductor that lost its shell would take its own name out of both sides
    at once and the manifest would still agree with itself. An empty
    parametrisation does not fail either - pytest skips it - so the shelled
    half could go missing entirely and every test here would stay green.
    """
    assert {launch.name for launch in TWINS} == {f"{one.name}_shell" for one in LAUNCHES}


def test_and_the_kernel_actually_drew_them(read):
    """The list existing is not the drawing happening.

    A twin is built by sewing a conductor's faces into one surface, and a
    conductor the kernel will not sew records itself as unavailable - which
    every test over it then *skips*. So the whole shelled half can fail to draw
    and the run stays green, one step past the check above, which reads names
    and never asks whether any of them was answered.

    Every one of them sews today, a drawing in two separate lumps included -
    ``Part.Shell`` takes disjoint faces without complaint - so all of them are
    asked for rather than a share, which a threshold would swallow.
    """
    unsewn = [one.name for one in TWINS if "unavailable" in read[one.name]]
    assert unsewn == [], (
        "these conductors no longer draw as a surface, so the property they "
        f"carry is being skipped rather than tested: {unsewn}"
    )
