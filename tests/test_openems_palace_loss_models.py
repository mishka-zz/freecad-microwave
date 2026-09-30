# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""One lossy document, solved on both backends, says it answers two models.

A loss tangent is held across the band by Palace and folded into one
conductivity at the centre of it by openEMS, and a conducting sheet inside the
region is two faces carrying a surface impedance to Palace and a plane carrying
the net current to openEMS. The matrices cannot show either. So what is held
here is what a user reads: each stored result states how its backend modelled
each lossy material, and the answer filed second names each material the two
modelled differently.

Why a subprocess
----------------

For the reason ``tests/test_acceptance_palace_waveguide.py`` gives: the
drawing needs the CAD kernel, and
``tests/openems_palace_loss_models_probe.py`` runs under ``freecadcmd`` to
draw it and drive both runs. A machine without one of the backends makes
the probe say which, and the test skips.
"""

from __future__ import annotations

import os

import pytest

from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "openems_palace_loss_models_probe.py")


@pytest.fixture(scope="module")
def filed(tmp_path_factory, interpreter):
    """Both answers, as the document holds them once the two runs are filed."""
    out = tmp_path_factory.mktemp("loss_models")
    # The probe runs openEMS the way the workbench does, from FreeCAD's own
    # interpreter, which finds the engine's by this first.
    with pytest.MonkeyPatch.context() as patched:
        patched.setenv("MICROWAVE_OPENEMS_PYTHON", str(interpreter))
        manifest = probe_manifest(PROBE, out, "LOSS_MODELS_OUT", key=None)
    if manifest.get("missing"):
        pytest.skip(f"this machine cannot drive both backends: {manifest['missing']}")
    return manifest


@pytest.mark.parametrize("backend", ["openEMS", "Palace"])
def test_each_result_is_filed_under_its_backend(filed, backend):
    assert filed[backend]["label"] == f"S-Parameters ({backend})"


def test_openems_states_the_loss_tangent_folded_at_the_centre_of_the_band(filed):
    (line,) = [line for line in filed["openEMS"]["modelled"] if line.startswith("'LossyFill'")]
    assert "a loss tangent folded at 23 GHz into a conductivity" in line


def test_openems_states_the_sheet_carries_its_net_current(filed):
    (line,) = [line for line in filed["openEMS"]["modelled"] if line.startswith("'Brass'")]
    assert line.endswith("carrying the net current through it")


def test_palace_states_the_loss_tangent_held_across_the_band(filed):
    assert (
        "'LossyFill': a loss tangent of 0.001, held across the band" in filed["Palace"]["modelled"]
    )


def test_palace_states_the_sheet_stands_inside_on_two_faces(filed):
    (line,) = [line for line in filed["Palace"]["modelled"] if line.startswith("'Brass'")]
    assert line.endswith("on each of its two faces, passing nothing between them")


def test_what_is_stated_is_what_the_run_recorded(filed):
    """The words are the record's, so a result whose words and record parted
    would be stating a model nobody solved."""
    from Microwave.Results import modelled

    for backend in ("openEMS", "Palace"):
        assert filed[backend]["modelled"] == modelled.said(filed[backend]["provenance"])


def test_the_first_filed_has_nothing_to_compare_with(filed):
    assert filed["openEMS"]["beside"] == []


def test_the_second_filed_names_both_materials_the_two_modelled_otherwise(filed):
    said = [line for line in filed["Palace"]["beside"] if " is not one model" in line]
    assert sorted(line.split(" is not one model")[0] for line in said) == ["'Brass'", "'LossyFill'"]
    assert all(": Palace solved " in line and ", and openEMS " in line for line in said)


def test_both_answers_are_filed_as_solved_from_one_drawing(filed):
    assert filed["openEMS"]["drawing"]
    assert filed["openEMS"]["drawing"] == filed["Palace"]["drawing"]


def test_the_second_filed_states_how_far_apart_every_term_stands(filed):
    """Every term of a two-port both backends drove, compared rather than
    refused, whatever the figure is."""
    said = [line for line in filed["Palace"]["beside"] if " differs by at most " in line]
    assert sorted(line.split(" differs")[0] for line in said) == ["S11", "S12", "S21", "S22"]
    assert not any(line.startswith("Not compared") for line in filed["Palace"]["beside"])
    assert not any("not known to be of one drawing" in line for line in filed["Palace"]["beside"])
