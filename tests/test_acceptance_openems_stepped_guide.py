# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Acceptance gate: WR-42 with a step in it, driven from each end, against reciprocity.

A guide whose two ports end alike reads both at one scale, and the ratio between
them is right whatever that scale is. Here they end differently. The first half
is WR-42 whole; in the second a metal block takes away half the height, so one
port has both broad walls on the domain and the other has one on drawn metal.
openEMS' ``RectWGPort`` reads each at a scale the cells at its own walls set,
and the matrix it returns is scaled by the ratio: not reciprocal, with one column
giving out more power than the structure took and the other less.
:mod:`.portreading` undoes that.

Nothing here needs a reference, because both properties are exact for a lossless
reciprocal structure: ``S21 = S12``, and each column carries out the power that
went in. The power is held to the bar the straight guide's gate sets on its own.
openEMS' own reading of the same runs, kept beside the corrected one, fails the
reciprocity bar, which is what stops the correction being removable without a
gate saying so.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from Microwave.Results.sparameters import MAGNIFICATION_TOLERANCE, MAGNIFIED, SParameters
from Microwave.Solvers.openems import (
    balance,
    driver,
    nearfield,
    plan,
    preflight,
    read,
    residual,
    run,
    write,
)
from Microwave.Solvers.openems.model import (
    THROUGH,
    Frequency,
    Material,
    Port,
    Problem,
    Solid,
    Termination,
)
from Microwave.Solvers.openems.regions import MeshParams
from tests.test_acceptance_openems_waveguide import (
    FREQ_MAX,
    FREQ_MIN,
    GUIDE_BROAD,
    GUIDE_NARROW,
    POWER_TOLERANCE,
    RESOLUTION,
    TIMESTEPS,
)

pytestmark = pytest.mark.slow

GUIDE_LENGTH = 60.0
PORT_INSET = 10 * RESOLUTION
PORT_DEPTH = 5 * RESOLUTION

#: What the second half keeps of the guide, as ``(broad, narrow)``.
KEPT = (GUIDE_BROAD, GUIDE_NARROW / 2)

#: How far from one ``|S21| / |S12|`` may sit. Reciprocity is exact, so this
#: bounds the reading alone. openEMS reads a port's mode short by about half the
#: share of its height its end cells take, and that share is twice as large in a
#: guide half the height, so the ratio of the two ports' readings is out by it;
#: the third test holds openEMS' own reading of this specimen to missing the bar.
RECIPROCITY_TOLERANCE = 0.01


def _problem(driven: int, step: float = GUIDE_LENGTH / 2, end: float = GUIDE_LENGTH) -> Problem:
    """The guide driven from one port, with the second half's block starting at
    ``step`` along it and ending at ``end``."""
    broad, narrow = KEPT
    materials = (
        Material(name="Air", kind="dielectric", epsilon=1.0),
        Material(name="Block", kind="pec"),
    )
    solids = (
        Solid(
            material="Air",
            lower=(0.0, 0.0, 0.0),
            upper=(GUIDE_BROAD, GUIDE_NARROW, GUIDE_LENGTH),
            priority=0,
            label="Guide",
        ),
        Solid(
            material="Block",
            lower=(0.0, narrow, step),
            upper=(GUIDE_BROAD, GUIDE_NARROW, end),
            priority=10,
            label="Step",
        ),
    )
    ports = (
        Port(
            number=1,
            kind="rect_waveguide",
            mode="TE10",
            start=(0.0, 0.0, PORT_INSET),
            stop=(GUIDE_BROAD, GUIDE_NARROW, PORT_INSET + PORT_DEPTH),
            propagation_axis=2,
            excite=driven == 1,
            label="Port 1",
        ),
        Port(
            number=2,
            kind="rect_waveguide",
            mode="TE10",
            start=(0.0, 0.0, GUIDE_LENGTH - PORT_INSET),
            stop=(broad, narrow, GUIDE_LENGTH - PORT_INSET - PORT_DEPTH),
            propagation_axis=2,
            excite=driven == 2,
            label="Port 2",
        ),
    )
    params = MeshParams(
        metal_res=RESOLUTION,
        dielectric_res=RESOLUTION,
        max_ratio=(1.4, 1.4, 1.4),
        min_lines=10,
        pml_cells=(0, 0, 8),
    )
    grid = plan.plan_grid(
        solids, ports, materials, params, padding=((0, 0), (0, 0), (THROUGH, THROUGH))
    )
    return Problem(
        title=f"WR-42 with a step in its broad wall, port {driven} driven",
        frequency=Frequency(start=FREQ_MIN, stop=FREQ_MAX, points=201),
        grid=grid,
        materials=materials,
        solids=solids,
        ports=ports,
        boundary=("PEC", "PEC", "PEC", "PEC", "PML_8", "PML_8"),
        termination=Termination(max_timesteps=TIMESTEPS, end_criteria=0.0),
    )


def _solve(problem: Problem, directory: Path, interpreter) -> list[str]:
    """Solve ``problem`` into ``directory``, and return the warnings the run made."""
    envelope = write.write(problem, directory)
    preflight.refuse_if_blocked(preflight.check(problem))
    said: list[str] = []

    def heard(item: run.Marker | str) -> None:
        if isinstance(item, run.Marker) and item.name == "CHECK":
            said.append(item.detail)

    run.run(envelope, interpreter=interpreter, on_output=heard)
    return said


@pytest.fixture(scope="module")
def columns(interpreter, tmp_path_factory) -> dict[int, Path]:
    """Where each column was solved: the run driving port 1 and the one driving 2."""
    solved = {}
    for driven in (1, 2):
        directory = tmp_path_factory.mktemp(f"stepped-{driven}")
        _solve(_problem(driven), directory, interpreter)
        solved[driven] = directory
    return solved


def _matrix(columns: dict[int, Path]) -> np.ndarray:
    """The matrix as the result layer assembles it, each port against its own
    reference, so the power waves of two guides of different size meet."""
    runs = [read.read(columns[driven]) for driven in (1, 2)]
    return SParameters.from_runs(runs, reference=None).s


def _raw(ports: dict, number: int, key: str) -> np.ndarray:
    raw = ports[str(number)]["raw"][key]
    return np.asarray(raw["re"]) + 1j * np.asarray(raw["im"])


def _raw_matrix(columns: dict[int, Path]) -> np.ndarray:
    """The same, from the waves openEMS split itself, kept beside the corrected
    ones in each results file."""
    matrix = None
    for driven, directory in columns.items():
        ports = json.loads((directory / driver.RESULTS_NAME).read_text())["ports"]
        power = {n: 1 / np.sqrt(_raw(ports, n, "z0")) for n in (1, 2)}
        inward = _raw(ports, driven, "incident") * power[driven]
        if matrix is None:
            matrix = np.empty((inward.size, 2, 2), dtype=complex)
        for n in (1, 2):
            matrix[:, n - 1, driven - 1] = _raw(ports, n, "reflected") * power[n] / inward
    assert matrix is not None
    return matrix


def _reciprocity(matrix: np.ndarray) -> np.ndarray:
    return np.abs(matrix[:, 1, 0]) / np.abs(matrix[:, 0, 1])


def _departure(matrix: np.ndarray) -> float:
    """How far the worst column of a lossless structure is from unit power."""
    return float(np.max(np.abs(np.sum(np.abs(matrix) ** 2, axis=1) - 1)))


def test_the_matrix_is_reciprocal(columns):
    ratio = _reciprocity(_matrix(columns))
    worst = float(np.max(np.abs(ratio - 1)))
    print(
        f"\nGATE stepped WR-42 |S21|/|S12| = {ratio.min():.4f} to {ratio.max():.4f}, "
        f"bound {RECIPROCITY_TOLERANCE:.0e}"
    )
    assert worst < RECIPROCITY_TOLERANCE, (
        f"|S21|/|S12| reaches {1 + worst:.4f} on a reciprocal structure: the two ports "
        "are read at different scales"
    )


def test_each_column_carries_out_the_power_that_went_in(columns):
    matrix = _matrix(columns)
    departure = np.abs(matrix) ** 2
    worst = float(np.max(np.abs(departure.sum(axis=1) - 1)))
    print(f"\nGATE stepped WR-42 worst power departure = {worst:.3e}, bound {POWER_TOLERANCE:.0e}")
    assert worst < POWER_TOLERANCE, (
        f"a column of a lossless structure departs from unit power by {worst:.3e}"
    )


def test_against_a_constant_it_says_how_far_the_solve_is_magnified(columns):
    """Both ports are moved to 50 ohm, far under their own impedance.

    An error ``H`` in the solve comes out of the renormalisation as ``A H B``,
    with ``A = (I - G^2)^(1/2) (I - S G)^-1`` and
    ``B = (I - G S)^-1 (I - G^2)^(1/2)``, ``G`` the reflection of 50 ohm against
    each port's own impedance. The record is ``|A| |B|`` where it is largest,
    and the panel says it.
    """
    runs = [read.read(columns[driven]) for driven in (1, 2)]
    as_solved = SParameters.from_runs(runs, reference=None)
    at_fifty = SParameters.from_runs(runs, reference=50.0)
    record = at_fifty.provenance[MAGNIFIED]
    at = int(np.flatnonzero(at_fifty.frequency == record["frequency"])[0])

    s = as_solved.s[at]
    z = as_solved.reference[at].real
    g = np.diag((50.0 - z) / (50.0 + z))
    root = np.sqrt(np.eye(2) - g @ g)
    a = root @ np.linalg.inv(np.eye(2) - s @ g)
    b = np.linalg.inv(np.eye(2) - g @ s) @ root
    expected = float(np.linalg.norm(a, 2) * np.linalg.norm(b, 2))
    vswr = float(np.max((1 + np.abs(np.diag(g))) / (1 - np.abs(np.diag(g)))))

    print(
        f"\nGATE stepped WR-42 at 50 ohm magnifies the solve's error up to "
        f"{record['factor']:.2f} times at {record['frequency'] / 1e9:.2f} GHz, "
        f"closed form {expected:.2f}, largest VSWR {vswr:.2f}"
    )
    assert record["factor"] == pytest.approx(expected, rel=1e-4, abs=0.0)
    assert record["factor"] > MAGNIFICATION_TOLERANCE
    assert at_fifty.magnified().startswith("Ports 1 and 2 are reported against 50 ohm")
    assert MAGNIFIED not in as_solved.provenance


def test_openems_own_reading_of_the_same_runs_is_not_reciprocal(columns):
    """What the gate above would pass on if the correction were taken out, and
    why it has to be there: the specimen is one whose ports end differently
    enough for the engine's own reading to fail the same bar."""
    ratio = _reciprocity(_raw_matrix(columns))
    assert float(np.max(np.abs(ratio - 1))) > RECIPROCITY_TOLERANCE


def _near_field(directory: Path) -> dict:
    return read.read(directory).provenance[nearfield.KEY]


def test_ports_far_from_the_step_read_their_mode_alone(columns):
    """The step stands many cells from each port, so the field it leaves has
    died away before either plane: the waves each port read are those its
    planes inside predict, and the bound on what a field costs the matrix is
    far under the bar."""
    bound = nearfield.BAR / 10
    worst = max(_near_field(columns[driven])["share"] for driven in columns)
    print(
        f"\nGATE stepped WR-42 bound on the field near the ports = {worst:.2e}, bound {bound:.0e}"
    )
    assert worst < bound
    for driven in columns:
        ports = _near_field(columns[driven])["ports"]
        assert set(ports) == {"1", "2"}
        for port in ports.values():
            assert 0 < port["shallow"] < port["depth"] < PORT_DEPTH


def _balance(directory: Path) -> dict:
    return read.read(directory).provenance[balance.KEY]


def test_the_ports_account_for_the_power_driven_in(columns):
    """Nothing dissipates, and both ends absorb behind a port whose guide the
    step's metal closes in, so every watt driven in leaves through a port."""
    bound = balance.BAR / 10
    found = [_balance(columns[driven]) for driven in columns]
    worst = max(abs(record[side]["share"]) for record in found for side in ("below", "above"))
    print(
        f"\nGATE stepped WR-42 power the ports do not account for = {worst:.2e}, bound {bound:.0e}"
    )
    assert worst < bound
    assert not any("below_only" in record for record in found)


#: How far from port 1's plane the step stands in the run that puts it on the
#: full-height side: near enough that the column is off past the bar.
FULL_HEIGHT_NEAR = 8 * RESOLUTION


def test_a_step_in_front_of_a_port_is_warned_of_by_the_field_and_by_the_power(
    interpreter, tmp_path
):
    """The field the step leaves dies away slowly, and a plane inside port 1's
    box still holds much of it; the bound takes that share into account. The
    ports' power is read at their own planes, where the field is whole."""
    said = _solve(
        _problem(1, step=PORT_INSET + PORT_DEPTH + FULL_HEIGHT_NEAR), tmp_path, interpreter
    )
    near, power = _near_field(tmp_path), _balance(tmp_path)
    short = power["above"]["share"]
    print(
        f"\nGATE stepped WR-42 with the step 8 cells from port 1: power not accounted for "
        f"{short:.2e}, bound on the field near the ports {near['share']:.2e} growing "
        f"{near['growth']:.3g} of the depths, bar {balance.BAR:.0e}"
    )
    assert near["share"] > nearfield.BAR and near["port"] == 1 and near["side"] == "front"
    assert short > balance.BAR and "below_only" not in power
    [field] = [line for line in said if "planes inside their boxes predict" in line]
    assert "may be right" not in field
    assert len([line for line in said if "less power than was driven in" in line]) == 1


#: How far from port 2's plane the step stands in the run that puts it there:
#: near enough for the field the step leaves to move the ports' net power past
#: the bar, which the test holds it to.
NEAR = 2 * RESOLUTION


def test_a_step_near_the_driven_port_is_warned_of_by_name(interpreter, tmp_path):
    """The port the check has to reach as well as any other: a study that
    drives one port reads that port's reflection with the field in it."""
    plane = GUIDE_LENGTH - PORT_INSET - PORT_DEPTH
    said = _solve(_problem(2, step=plane - NEAR), tmp_path, interpreter)
    near = _near_field(tmp_path)
    warned = [line for line in said if "planes inside their boxes predict" in line]
    assert len(warned) == 1 and "the largest part of it at port 2" in warned[0]
    assert near["port"] == 2 and near["share"] > nearfield.BAR
    assert near["ports"]["1"]["share"] < nearfield.BAR / 10


#: Where the block ends in the run that changes the guide behind port 2: seven
#: cells behind its source, toward the absorber. Past port 2's box the guide
#: opens back to full height, which sends back part of every wave reaching it.
#: Far enough behind that the field the change leaves has died before port 2's
#: plane, which the test holds by the run warning of nothing.
BEHIND = GUIDE_LENGTH - PORT_INSET + 7 * RESOLUTION


@pytest.fixture(scope="module")
def behind(interpreter, tmp_path_factory) -> dict[int, tuple[Path, list[str]]]:
    """Each column of the guide changing behind port 2, and what its run said."""
    solved = {}
    for driven in (1, 2):
        directory = tmp_path_factory.mktemp(f"behind-{driven}")
        solved[driven] = (directory, _solve(_problem(driven, end=BEHIND), directory, interpreter))
    return solved


def test_a_guide_changing_behind_a_port_is_counted_out(behind):
    """Port 2 sends back much of what reaches it, and a column read alone
    carries that wave. Both columns counted together carry none of it: the
    matrix is the device between the planes, whatever lies behind them."""
    runs = [read.read(behind[driven][0]) for driven in (1, 2)]
    matrix = SParameters.from_runs(runs, reference=None).s
    alone = SParameters.from_runs(runs[:1], reference=None).s[:, :, :1]
    ratio = _reciprocity(matrix)
    print(
        f"\nGATE stepped WR-42 changing behind port 2: power departure {_departure(matrix):.2e}, "
        f"|S21|/|S12| = {ratio.min():.5f} to {ratio.max():.5f}; column 1 read alone "
        f"departs by {_departure(alone):.2e}"
    )
    # The planes inside port 2's box stand nearer the change than the port's
    # own, so each column warns of the field near port 2, finds it growing
    # toward port 2's source, and says the matrix may be right; nothing else
    # is said.
    for directory, said in behind.values():
        near = _near_field(directory)
        assert near["share"] > nearfield.BAR, near
        assert near["port"] == 2 and near["side"] == "behind", near
        [field] = said
        assert "planes inside their boxes predict" in field and "may then be right" in field
    assert _departure(alone) > POWER_TOLERANCE
    assert _departure(matrix) < POWER_TOLERANCE
    assert float(np.max(np.abs(ratio - 1))) < RECIPROCITY_TOLERANCE


def test_a_column_read_alone_says_what_its_undriven_port_sent_back(behind, columns):
    """Port 2 is warned of where the guide changes behind it, and neither
    port is where the guide runs on uniform into the absorber."""
    changing = SParameters.from_runs([read.read(behind[1][0])], reference=None)
    said = changing.sent_back(residual.WANTED)
    assert said is not None and said.startswith("A term of column 1") and "port 2 most" in said
    for driven in (1, 2):
        uniform = SParameters.from_runs([read.read(columns[driven])], reference=None)
        assert uniform.sent_back(residual.WANTED) is None
