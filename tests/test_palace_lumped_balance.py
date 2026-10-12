# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a run through lumped ports says of its balance, on the examples.

A lumped port's face is not read: the power through it is formed from the
field's curl, the matrix from the field averaged along the port, and the two
disagree on a model that loses nothing. So the share a run through lumped ports
leaves is stated as information, and the line warns only where the matrix and
the open surface account for more power than went in.

Each shipped example driven through lumped ports is solved by
``tests/palace_lumped_balance_probe.py`` as shipped and drawn lossless, on a few
points up to the top of its band. What is held:

- as shipped, none warns of its balance;
- drawn lossless, none warns, and the line says the share is the solver's;
- the gate's line states what each leaves drawn lossless, which is the size of
  the solver's share on the boards the workbench ships.

Why a subprocess
----------------

For the reason ``tests/test_acceptance_palace_waveguide.py`` gives: the
documents need the CAD kernel, and the probe runs under ``freecadcmd`` to open
and solve them. A machine without Palace or Gmsh makes the probe say which, and
the test skips.

Each solve is an example drawn otherwise than shipped, so the module is held
back to the release run.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from Microwave.Solvers.palace import balance
from Microwave.Solvers.palace.read import Scattering
from tests.conftest import probe_manifest

pytestmark = [pytest.mark.slow, pytest.mark.release]

PROBE = os.path.join(os.path.dirname(__file__), "palace_lumped_balance_probe.py")


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    out = tmp_path_factory.mktemp("lumped_balance")
    found = probe_manifest(PROBE, out, "LUMPED_BALANCE_OUT", key=None)
    if found.get("missing"):
        pytest.skip(f"this machine cannot drive Palace: {found['missing']}")
    assert found["examples"], "no shipped example is solved on Palace through lumped ports"
    wanted = [f"{name}_{way}" for name in found["examples"] for way in ("shipped", "lossless")]
    assert found["cases"] == wanted, (
        f"the probe wrote {found['cases']} rather than {wanted}, so it stopped before "
        "every run finished"
    )
    return found


def answer(case: dict) -> Scattering:
    return Scattering(
        frequency=np.asarray(case["frequency"]),
        out=tuple(case["out"]),
        driven=tuple(case["driven"]),
        matrix=np.asarray(case["real"]) + 1j * np.asarray(case["imaginary"]),
        flux=np.asarray(case["flux"]),
        dissipates=case["dissipates"],
        radiated=None if case["radiated"] is None else np.asarray(case["radiated"]),
        stated={int(number): meaning for number, meaning in case["stated"].items()},
    )


#: What only the balance line of a run through lumped ports says, in each of its
#: forms.
BALANCE = ("does not come back", "more power than went in")


def said_of_balance(case: dict) -> list[str]:
    return [line for line in case["said"] if any(words in line for words in BALANCE)]


def test_drawn_lossless_no_example_warns_and_the_share_is_the_solvers(manifest):
    shares = {}
    for name in manifest["examples"]:
        case = manifest[f"{name}_lossless"]
        assert case["changed"] and not case["dissipates"], case["changed"]
        for short in balance.shortfalls(answer(case)):
            assert short.lumped
            shares[f"{name} driven from port {short.excitation}"] = short
        said = said_of_balance(case)
        assert said and not [line for line in said if line.startswith(balance.WARNING)], said
        kept = [line for line in said if "does not come back" in line]
        assert all("is the solver's error" in line for line in kept), said
    print(
        "GATE lumped-port balance on Palace, drawn lossless: "
        + ", ".join(
            f"{name} {short.share:.3g} at {short.frequency / 1e9:.3g} GHz"
            for name, short in shares.items()
        )
    )


def test_as_shipped_no_example_warns_of_its_balance(manifest):
    for name in manifest["examples"]:
        said = said_of_balance(manifest[f"{name}_shipped"])
        assert said, f"{name} said nothing of its balance"
        assert not [line for line in said if line.startswith(balance.WARNING)], said
