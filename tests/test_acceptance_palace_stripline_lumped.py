# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Acceptance gate: the shipped stripline driven through lumped ports on Palace.

The line ``examples/stripline_50ohm.py`` draws, marked up the way a board is
marked up: a lumped port at each end, driven from the strip's end edge to both
ground faces of the fill, at a resistance. Its impedance is exact. A symmetric
stripline is filled with one dielectric, so its mode is TEM and conformal
mapping gives its impedance in closed form (``tests/analytic/reference.py``).

What this gate exercises
------------------------

The port every board drives its trace through, on the second backend: the
elements laid across the gap from the picks, their directions, the magnetic wall
on the rest of the face they lie in, and the resistance each port's matrix is
referenced to.

A lumped port is not where the line starts. The current turns from the elements
into the strip, and the field fringes round their edges, so each end adds a
network of its own, and a matrix read at one length is the line and both
networks together. The line is therefore solved at two lengths and the networks
are taken off by ``tests/deembed.py``. It fits the line's impedance and one
network of a series impedance and a shunt admittance per frequency over both
lengths, with the line's electrical length held at the drawn length at the
wavenumber of the material the translation read for the fill.

The network is what the port adds, sized by two numbers at each frequency. The
chain matrix of the whole two-port does not depend on the resistance its ports
are referenced to, and neither does the impedance fitted to it. How the network
is split between its two numbers does move: the fit weighs the four terms by the
reference, and the weaker of the two numbers follows that weighting. So the two
are printed as an inductance and a capacitance and not held to being either, and
what is held of the network is its size.

What is held, at each mesh:

- the impedance is the same at every frequency;
- the network is small against the reference;
- the network is the same at both ports, where the runs can see a difference;
- the two ports drive the wave through the line the same way.

At the mesh solved before a commit, and again at unequal resistances:

- each port states as its reference the resistance Palace says it configured;
- the line at unequal resistances de-embeds to the impedance it does at equal
  ones, which a matrix referenced to anything but each port's own resistance
  does not;
- the two solves agree about the transmission, which a normalisation that
  differs between the ports parts.

Against the closed forms. Before a commit, one mesh holds the impedance and the
phase constant under bounds. Before a release, a refinement sequence is carried
through ``tests/convergence.py``: the closed form of each has to be inside the
uncertainty the procedure computes for the finest mesh, and each bound has to
still cover the coarsest mesh. The phase constant's closed form is the fill's
wavenumber, since a TEM line's wave travels at the speed of its medium.

Every one of those meshes is refined in the bulk alone, with the mesh policy's
edge refinement at one. The field of a strip is singular along its rim, and the
size there is what the edge refinement lays, so before a release the line is
solved once more on the mesh of the commit's run with the edge refinement a new
mesh policy carries, and once on a mesh refined in the bulk alone to no fewer
unknowns. The impedance with the rim refined has to stand nearer the closed form
than the one refined in the bulk does, by more than one mesh of the sequence
stands off the sequence's own trend.

Why one network serves both ports
---------------------------------

To first order, an inductance moved from one port's network to the other's,
with a capacitance moved the same way, changes no term of either run. A network
fitted per port to runs with any misfit in them wanders along that direction by
far more than either network is worth. What a difference between the ports does
reach is the difference of the two reflections, and that is what is held.

The power the matrix leaves unaccounted for
-------------------------------------------

The model is lossless, and a column of the matrix can still carry less power
than went in. Its cause is not known, so it is printed for each length on the
gate's line and not held to a bar.

Refusals
--------

The refusals a lumped port on Palace makes are held on drawings of their own
rather than here. In ``TestWhatALumpedPortRefusesBeforeAMesh`` of
``tests/test_palace_lumped.py``:

- metal along an element's side in its drive direction:
  ``test_a_sheet_of_metal_along_a_side_of_the_element`` and
  ``test_the_edge_of_the_region_along_a_side_of_the_element``;
- no resistance: ``test_no_resistance``;
- picks that do not face each other: ``test_picks_that_do_not_face_each_other``.

And with the CAD kernel answering, in ``tests/test_palace_lumped_port.py``:
``test_metal_along_part_of_a_side_is_refused_naming_the_port``.

On the other backend the same port is referenced to its resistance as well,
held in
``tests/test_openems_driver_translation.py::TestALumpedPortIsReferencedToItsResistance``.
This document is not solved there: the openEMS adapter refuses a lumped port
naming more than one reference face.

Why a subprocess
----------------

For the reason ``tests/test_acceptance_palace_waveguide.py`` gives: the
drawing needs the CAD kernel, and ``tests/palace_stripline_lumped_probe.py``
runs under ``freecadcmd`` to draw it and drive the run. A machine without
Palace or Gmsh makes the probe say which, and the gate skips.

The refinement sequence is held back to the release run. Its tests reach
``refinements``, which ``tests/conftest.py`` names as a study, and conftest
marks a test only where it reaches the engine through ``interpreter`` as well,
which a Palace solve does not pass through. So each test that reads the
sequence carries the marker itself.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from Microwave.Solvers.palace import read
from Microwave.units import MM_PER_M
from tests import convergence, deembed, openems_stripline
from tests.analytic import reference
from tests.conftest import probe_manifest

pytestmark = pytest.mark.slow

PROBE = os.path.join(os.path.dirname(__file__), "palace_stripline_lumped_probe.py")

#: The mesh solved before a commit, in elements per wavelength at the top of the
#: band. The coarsest of the sequence, so the release run measures it again as
#: the sequence's first point.
ONE_MESH = 50.0

#: The meshes the release run adds to it. The procedure in ``tests/convergence.py``
#: reads four grids at least.
FINER = (60.0, 72.0, 86.0)

#: The mesh refined in the bulk alone that the rim-refined run is held against,
#: in elements per wavelength at the top of the band. Chosen to solve no fewer
#: unknowns than the line on :data:`ONE_MESH` with the rim refined, and the
#: release test asserts that it does: what the two are compared on is where the
#: unknowns go, not how many there are.
#:
#: So this number follows the edge refinement and growth a new Gmsh mesh
#: carries, and has to be measured again when either moves: a rim refined more
#: cheaply leaves this mesh spending far more than it is held against, and the
#: comparison stops being a comparison of where the unknowns went. Re-derive it
#: by meshing the line at this refinement with no rim over a range of sizes and
#: taking the smallest that clears the rim-refined run at both lengths.
BULK_MATCHED = 94.0

#: Each port's resistance, in ohms: the same at both ports, and a quarter and
#: four times that at port 1 and port 2.
EQUAL = (50.0, 50.0)
UNEQUAL = (25.0, 100.0)

#: How far the impedance may range across the band, as a share of its mean.
#:
#: A chosen value. A TEM line has no dispersion, so the impedance fitted at each
#: frequency differs by the misfit alone, and the misfit comes from the mesh
#: rather than from chance, so no standard error bounds it. What the bar catches
#: is a matrix that is not a line between two small networks: whatever of it the
#: model cannot represent is taken up by the impedance in a different share at
#: each frequency.
FLAT = 5e-3

#: How large the port's network may be: the magnitude of its series impedance
#: over the reference, and of its shunt admittance times it.
#:
#: A chosen value. The network is a correction to the line. What the bar catches
#: is a matrix the model cannot represent taken up by the network rather than
#: left in the residual, which leaves the impedance read off a network grown to
#: hold it.
NETWORK_AT_MOST = 0.1

#: How far apart the two reflections of one run may be, as a share of the
#: network's size - the larger of its series impedance over the reference and
#: its shunt admittance times it, as :func:`test_the_network_is_small` reads it -
#: carried through the line.
#:
#: The two ends are meshed independently, so their networks differ by the
#: discretisation. Two networks reflecting ``G1`` and ``G2`` on their own part
#: ``S11`` from ``S22`` by ``(G1 - G2) (1 - exp(-2j k L))`` to first order, and a
#: network's reflection is no more than its size. The size rather than the
#: reflection, since a network's reflection vanishes where its inductance over its
#: capacitance is the reference squared while the two networks can still differ.
#: Half is a chosen value.
PORTS_ALIKE = 0.5

#: How many times the misfit of the line at equal resistances the line at
#: unequal ones may leave.
#:
#: A choice. Both are meshed alike and leave a misfit alike; two stands above
#: that and far below what a matrix read against the wrong references leaves.
MISFIT_ALIKE = 2.0

#: The edge refinement every mesh of the sequence is solved at, and the word the
#: probe takes for the value a new mesh policy carries.
PINNED = "1"
POLICY = "policy"

#: How far the impedance at :data:`ONE_MESH` may stand from the closed form, and
#: the phase constant from the fill's wavenumber, each as a share of it.
#:
#: The error the release sequence measures at that mesh, with a chosen margin
#: added. The bounds guard the pre-commit run against drift; the release run
#: holds each to still covering that mesh.
ONE_MESH_BOUND = 0.025
PHASE_BOUND = 2e-3


def case(length: float, mesh: float, resistances: tuple[float, float], edges: str) -> str:
    """The name the probe writes a run under."""
    return (
        f"line_{length:g}mm_{mesh:g}_per_wavelength_{resistances[0]:g}_{resistances[1]:g}_ohm"
        f"_edges_{edges}"
    )


def solved(meshes, pairs, out, edges=PINNED) -> dict:
    """The probe's manifest for ``meshes`` at ``pairs`` and at the edge refinement
    ``edges``, every run in it finished."""
    manifest = probe_manifest(
        PROBE,
        out,
        "STRIPLINE_LUMPED_PALACE_OUT",
        key=None,
        environment={
            "STRIPLINE_LUMPED_PALACE_MESHES": ",".join(f"{mesh:g}" for mesh in meshes),
            "STRIPLINE_LUMPED_PALACE_RESISTANCES": ",".join(
                f"{one:g}/{other:g}" for one, other in pairs
            ),
            "STRIPLINE_LUMPED_PALACE_EDGES": edges,
        },
    )
    if manifest.get("missing"):
        pytest.skip(f"this machine cannot drive Palace: {manifest['missing']}")
    wanted = [
        case(length, mesh, pair, edges)
        for mesh in meshes
        for pair in pairs
        for length in manifest["lengths"]
    ]
    assert manifest["cases"] == wanted, (
        f"the probe wrote {manifest['cases']} rather than {wanted}, so it stopped "
        "before every run finished. A gate that is not run is not a gate that passed"
    )
    return manifest


@pytest.fixture(scope="module")
def one_mesh(tmp_path_factory):
    """The line at both lengths on :data:`ONE_MESH`, at equal and unequal
    resistances."""
    return solved([ONE_MESH], [EQUAL, UNEQUAL], tmp_path_factory.mktemp("palace_stripline_lumped"))


@pytest.fixture(scope="module")
def refinements(tmp_path_factory):
    """The line at both lengths on every mesh of :data:`FINER`."""
    return solved(FINER, [EQUAL], tmp_path_factory.mktemp("palace_stripline_lumped_finer"))


@pytest.fixture(scope="module")
def bulk_matched(tmp_path_factory):
    """The line at both lengths on :data:`BULK_MATCHED` at equal resistances."""
    return solved(
        [BULK_MATCHED], [EQUAL], tmp_path_factory.mktemp("palace_stripline_lumped_bulk_matched")
    )


@pytest.fixture(scope="module")
def rim_refined(tmp_path_factory):
    """The line at both lengths on :data:`ONE_MESH` at equal resistances, with the
    edge refinement a new mesh policy carries."""
    return solved(
        [ONE_MESH], [EQUAL], tmp_path_factory.mktemp("palace_stripline_lumped_rim"), POLICY
    )


class Line:
    """Both lengths of the line on one mesh at one pair of resistances, and what
    they de-embed to."""

    def __init__(
        self,
        manifest: dict,
        mesh: float,
        resistances: tuple[float, float] = EQUAL,
        edges: str = PINNED,
    ):
        self.mesh = mesh
        self.resistances = resistances
        self.runs = [
            manifest[case(length, mesh, resistances, edges)] for length in manifest["lengths"]
        ]
        self.lengths = np.array([run["length"] for run in self.runs]) / MM_PER_M
        self.frequency = np.asarray(self.runs[0]["frequency"], dtype=float)
        for run in self.runs:
            assert np.array_equal(np.asarray(run["frequency"], dtype=float), self.frequency)
            # Indexed by position below, so the order is a claim.
            assert (run["out"], run["driven"]) == ([1, 2], [1, 2])
            run["s"] = np.asarray(run["real"]) + 1j * np.asarray(run["imaginary"])
        self.width, self.separation = manifest["width"], manifest["separation"]
        self.shield = manifest["shield"]
        self.permittivity = self.runs[0]["permittivity"]
        self.permeability = self.runs[0]["permeability"]
        self.element = self.runs[0]["element"]
        self.wavenumber = np.array(
            [deembed.wavenumber(f, self.permittivity, self.permeability) for f in self.frequency]
        )
        self.angular = 2 * np.pi * self.frequency
        self.reference = float(np.sqrt(np.prod(resistances)))
        self.fitted = [
            deembed.fit(self.chains(index), self.lengths * k, self.reference, deembed.SERIES_FIRST)
            for index, k in enumerate(self.wavenumber)
        ]
        self.impedance = np.array([one.impedance for one in self.fitted])

    def chains(self, index: int) -> list[np.ndarray]:
        return [deembed.abcd(run["s"][index], self.resistances) for run in self.runs]

    @property
    def closed_form(self) -> float:
        return reference.stripline_impedance(self.width, self.separation, self.permittivity)

    @property
    def phase_constant(self) -> np.ndarray:
        """The phase constant across the two lengths, over the fill's wavenumber."""
        difference = self.lengths[1] - self.lengths[0]
        return np.array(
            [
                deembed.phase_across(*self.chains(index)) / (k * difference)
                for index, k in enumerate(self.wavenumber)
            ]
        )

    def shortfall(self, run: dict) -> float:
        """The largest share of a watt driven in that no term of a column carries."""
        return float(np.max(1.0 - np.sum(np.abs(run["s"]) ** 2, axis=1)))


@pytest.fixture(
    scope="module",
    params=[ONE_MESH, *(pytest.param(mesh, marks=pytest.mark.release) for mesh in FINER)],
    ids=lambda mesh: f"{mesh:g}_per_wavelength",
)
def line(request):
    """One mesh of the line at equal resistances. The finer ones are the release
    run's."""
    if request.param == ONE_MESH:
        return Line(request.getfixturevalue("one_mesh"), ONE_MESH)
    return Line(request.getfixturevalue("refinements"), request.param)


def test_the_impedance_is_the_same_at_every_frequency(line):
    """A TEM line has no dispersion, so once the network is off the impedance is
    one number."""
    spread = float(np.ptp(line.impedance) / np.mean(line.impedance))
    error = line.impedance / line.closed_form - 1.0
    runs = line.runs
    print(
        f"\nGATE stripline through lumped ports on Palace (order {runs[0]['order']}, "
        f"{line.mesh:g} elements per wavelength asked {line.element:.4g} mm, edge refinement "
        f"{runs[0]['edge_refinement']:g}, elements from "
        f"{min(run['shortest'] for run in runs):.4g} to "
        f"{max(run['longest'] for run in runs):.4g} mm, unknowns "
        f"{' and '.join(str(run['unknowns']) for run in runs)}): Zc = "
        + ", ".join(
            f"{value:.4f} ohm ({100 * off:+.3f} %) at {f / 1e9:g} GHz"
            for value, off, f in zip(line.impedance, error, line.frequency)
        )
        + f" against {line.closed_form:.4f} ohm, ranging over {spread:.2e} of its mean; "
        "beta/k - 1 = "
        + ", ".join(f"{value - 1:+.2e}" for value in line.phase_constant)
        + "; power unaccounted for up to "
        + " and ".join(
            f"{line.shortfall(run):.2e} on the {run['length']:g} mm line" for run in runs
        )
    )
    assert spread < FLAT, (
        f"the de-embedded impedance ranges over {spread:.3g} of its mean across the band: "
        f"{line.impedance} ohm"
    )


def test_the_network_is_small(line):
    """Printed as the inductance and the capacitance its two numbers come to at
    each frequency, which are not held."""
    size = [
        max(abs(fit.series) / line.reference, abs(fit.shunt) * line.reference)
        for fit in line.fitted
    ]
    print(
        f"\nGATE stripline through lumped ports on Palace ({line.mesh:g} per wavelength): "
        "each port's network reads as L = "
        + ", ".join(f"{1e12 * f.inductance(w):.2f} pH" for f, w in zip(line.fitted, line.angular))
        + " and C = "
        + ", ".join(f"{1e15 * f.capacitance(w):.3f} fF" for f, w in zip(line.fitted, line.angular))
        + f" over {line.frequency[0] / 1e9:g} to {line.frequency[-1] / 1e9:g} GHz, "
        f"largest term over the reference {max(size):.2e}"
    )
    assert max(size) < NETWORK_AT_MOST, (
        f"the port's network reaches {max(size):.3g} of the reference, so it is not a "
        f"correction to the line: series {[f.series for f in line.fitted]} ohm, shunt "
        f"{[f.shunt for f in line.fitted]} S"
    )


def test_the_network_is_the_same_at_both_ports(line):
    for index, fit in enumerate(line.fitted):
        size = max(abs(fit.series) / line.reference, abs(fit.shunt) * line.reference)
        for run, length in zip(line.runs, line.lengths):
            s = run["s"][index]
            parted = abs(s[0, 0] - s[1, 1])
            unlike = size * abs(1 - np.exp(-2j * line.wavenumber[index] * length))
            assert parted < PORTS_ALIKE * unlike, (
                f"at {line.frequency[index] / 1e9:g} GHz on the {run['length']:g} mm line "
                f"the two reflections part by {parted:.3g}, and two networks the fitted "
                f"one's size and wholly unlike part them by up to {unlike:.3g}"
            )


def test_the_two_ports_drive_the_wave_the_same_way(line):
    """Whether the ports agree, and not which way either drives: both turned
    round leaves every term of the matrix as it was. At the bottom of the band
    neither line is near a half wavelength, so a transmission of the wave's own
    phase is told from one turned half a turn by which side of a quarter turn is
    left over."""
    assert np.all(line.wavenumber[0] * line.lengths < np.pi / 2)
    for run, length in zip(line.runs, line.lengths):
        for driven, out in ((0, 1), (1, 0)):
            left = run["s"][0, out, driven] * np.exp(1j * line.wavenumber[0] * length)
            assert left.real > 0.0, (
                f"S{out + 1}{driven + 1} on the {run['length']:g} mm line leaves "
                f"{math.degrees(np.angle(left)):.1f} degrees beyond -k L"
            )


def test_each_port_is_referenced_to_the_resistance_palace_configured(one_mesh):
    """Stated, and then shown by the physics.

    Each port states as its reference the resistance Palace says in its own log
    it configured on that port. The line at a quarter of the resistance at one
    end and four times it at the other de-embeds to the same impedance as at
    equal resistances.

    Both lines are meshed alike, since the resistance changes nothing the mesher
    is given, so the misfit the mesh leaves is common to both, and both bars are
    taken from the line at equal resistances.

    What each bar sees. A reference wrong by a factor ``1 + d`` at one port moves
    the impedance by about ``d / 2``, and the impedances are held within the
    equal line's standard error taken twice in quadrature, so a ``d`` past about
    three standard errors over the impedance is seen there. Wrong at both ports
    by factors whose product is one - the matrix read against the geometric mean
    of the two resistances - the moves cancel in the impedance, and the matrix is
    no longer a line between two alike networks: the misfit is held to
    :data:`MISFIT_ALIKE` times the equal line's.
    """
    equal, unequal = (Line(one_mesh, ONE_MESH, pair) for pair in (EQUAL, UNEQUAL))
    for one in (equal, unequal):
        for run in one.runs:
            configured = [run["configured"][str(port)] for port in (1, 2)]
            assert configured == list(one.resistances), run["configured"]
            assert run["stated"] == {"1": read.RESISTANCE, "2": read.RESISTANCE}, run["stated"]
            np.testing.assert_array_equal(
                np.asarray(run["impedance"]), np.tile(configured, (one.frequency.size, 1))
            )
    apart = np.abs(unequal.impedance - equal.impedance)
    allowed = np.sqrt(2.0) * np.array([fit.impedance_error for fit in equal.fitted])
    misfit = np.array([fit.residual for fit in unequal.fitted]) / np.array(
        [fit.residual for fit in equal.fitted]
    )
    print(
        f"\nGATE stripline through lumped ports on Palace ({ONE_MESH:g} per wavelength): "
        f"at {UNEQUAL[0]:g} and {UNEQUAL[1]:g} ohm Zc = "
        + ", ".join(f"{value:.4f}" for value in unequal.impedance)
        + " ohm, apart from equal resistances by "
        + ", ".join(f"{value:.2e}" for value in apart)
        + " ohm against standard errors of "
        + ", ".join(f"{value:.2e}" for value in allowed)
        + "; misfit over the equal line's "
        + ", ".join(f"{value:.3f}" for value in misfit)
    )
    assert np.all(apart <= allowed), (
        f"the line de-embeds to {unequal.impedance} ohm at {UNEQUAL} and to "
        f"{equal.impedance} ohm at {EQUAL}"
    )
    assert np.all(misfit < MISFIT_ALIKE), (
        f"the line at {UNEQUAL} ohm leaves {misfit} times the misfit it leaves at "
        f"{EQUAL} ohm, so its matrix is not a line and a network read against each "
        "port's own resistance"
    )


def test_the_two_solves_agree_about_the_transmission(one_mesh):
    """Each off-diagonal term comes from the solve driving its column.

    What it catches is a normalisation that differs between the ports: waves at
    one port scaled against the other's by the root of ``1 + d`` part the two
    terms by ``|S21| d``. An error common to both ports leaves them equal, and so
    does a reference misread at either port, which moves the impedance instead.

    The bar is chosen and not derived. It holds ``d / 2`` below the impedance's
    standard error over itself, which is the share a reference wrong by ``1 + d``
    at one port moves the impedance by; tying the two is the choice.
    """
    gaps = {}
    for pair in (EQUAL, UNEQUAL):
        one = Line(one_mesh, ONE_MESH, pair)
        for run in one.runs:
            gap = np.abs(run["s"][:, 1, 0] - run["s"][:, 0, 1])
            share = gap / np.abs(run["s"][:, 1, 0]) / 2.0
            resolved = np.array([fit.impedance_error / fit.impedance for fit in one.fitted])
            gaps[(pair, run["length"])] = float(np.max(gap))
            assert np.all(share < resolved), (
                f"at {pair} ohm on the {run['length']:g} mm line S21 and S12 part by "
                f"{2 * share} of the transmission, against the impedance resolved to "
                f"{resolved}"
            )
    print(
        f"\nGATE stripline through lumped ports on Palace ({ONE_MESH:g} per wavelength): "
        "worst gap between S21 and S12 "
        + ", ".join(
            f"{gap:.2e} at {pair[0]:g}/{pair[1]:g} ohm on {length:g} mm"
            for (pair, length), gap in gaps.items()
        )
    )


def test_one_mesh_is_inside_its_bounds_on_the_closed_forms(one_mesh):
    """The pre-commit half of the comparison. The release run holds the bounds."""
    single = Line(one_mesh, ONE_MESH)
    error = abs(single.impedance.mean() / single.closed_form - 1.0)
    phase = float(np.max(np.abs(single.phase_constant - 1.0)))
    assert error < ONE_MESH_BOUND, (
        f"the line de-embeds to {single.impedance.mean():.4f} ohm against "
        f"{single.closed_form:.4f} ohm, {100 * error:.2f} % away"
    )
    assert phase < PHASE_BOUND, (
        f"the phase constant across the lengths stands {phase:.3g} from the fill's wavenumber"
    )


@pytest.mark.release
def test_refinement_carries_the_line_to_its_closed_forms(one_mesh, refinements):
    """The closed form is inside the uncertainty the procedure computes for the
    finest mesh, with no uncertainty on the reference - the comparison
    ``tests/test_acceptance_openems_stripline.py`` builds, where a pass is an
    error no larger than that band.

    The shield biases the closed form, which is of two infinite planes;
    ``tests/test_openems_stripline_fixture.py`` holds that bias below
    ``openems_stripline.REFERENCE_WALLS`` for this cross-section, and it is held
    here to stand inside the band, so the comparison does not read the shield.

    Whether the sequence can be read at all is the procedure's own test, its
    scatter below its data range. The error is not also held to falling at every
    step: the meshes are not nested, and a step between two of them is as much
    where the elements fell as how large they are.
    """
    lines = [Line(one_mesh, ONE_MESH), *(Line(refinements, mesh) for mesh in FINER)]
    want = lines[0].closed_form
    assert (lines[0].width, lines[0].separation, lines[0].shield) == (
        openems_stripline.WIDTH,
        openems_stripline.SEPARATION,
        openems_stripline.SHIELD,
    ), "the shield's bias is held for another cross-section than the one solved"
    cells = [one.element for one in lines]
    values = [float(one.impedance.mean()) for one in lines]
    phases = [float(one.phase_constant.mean()) for one in lines]
    estimate = convergence.uncertainty_of(cells, values)
    carried = convergence.uncertainty_of(cells, phases)
    print(
        "\nGATE stripline through lumped ports on Palace, refined: Zc "
        + ", ".join(
            f"{value:.4f} ohm ({100 * (value / want - 1):+.3f} %) at {cell:.4g} mm"
            for value, cell in zip(values, cells)
        )
        + f"; finest {estimate.finest:.4f} ohm +/- "
        f"{100 * estimate.uncertainty / want:.3f} % at order {estimate.order:.2f} by the "
        f"{estimate.expansion} expansion, heading for {estimate.limit:.4f} ohm, against "
        f"{want:.4f} ohm. beta/k - 1 "
        + ", ".join(f"{value - 1:+.2e}" for value in phases)
        + f"; finest {carried.finest - 1:+.2e} +/- {carried.uncertainty:.2e} at order "
        f"{carried.order:.2f}"
    )
    for name, fit in (("impedance", estimate), ("phase constant", carried)):
        assert fit.readable, (
            f"the {name} fit scatters by {fit.scatter:.4g} against a data range of "
            f"{fit.data_range:.4g}, so where it heads is not a limit"
        )
    assert openems_stripline.REFERENCE_WALLS * want < estimate.uncertainty, (
        "the band is narrower than what the shield may bias the closed form by, so the "
        "comparison would read the shield"
    )
    assert estimate.covers(want), (
        f"the finest mesh answers {estimate.finest:.4f} ohm +/- {estimate.uncertainty:.4f} "
        f"and the closed form is {want:.4f} ohm"
    )
    assert carried.covers(1.0), (
        f"the finest mesh's phase constant is {carried.finest:.6f} of the fill's "
        f"wavenumber +/- {carried.uncertainty:.3g}"
    )
    assert abs(values[0] / want - 1.0) < ONE_MESH_BOUND, (
        f"the coarsest mesh stands {100 * abs(values[0] / want - 1.0):.3f} % from the "
        f"closed form, past the {100 * ONE_MESH_BOUND:g} % the pre-commit run holds it to"
    )
    worst_phase = float(np.max(np.abs(lines[0].phase_constant - 1.0)))
    assert worst_phase < PHASE_BOUND, (
        f"the coarsest mesh's phase constant stands {worst_phase:.3g} from the fill's "
        f"wavenumber, past the {PHASE_BOUND:g} the pre-commit run holds it to"
    )


@pytest.mark.release
def test_refining_the_rim_carries_the_impedance_toward_its_closed_form(
    one_mesh, refinements, rim_refined, bulk_matched
):
    """The line with the size at the strip's rim refined, against the line refined
    in the bulk alone to no fewer unknowns.

    The field of a strip is singular along its rim, and the error it leaves in the
    impedance falls as the elements there shrink rather than as the bulk does. So
    the impedance with the rim refined has to stand nearer the closed form than
    the impedance on a mesh that spends at least as many unknowns everywhere.
    Against the bulk mesh the rim run started from, any refinement of that cost
    would pass.

    Nearer by more than the scatter of the sequence. Two meshes of one size part
    by where their elements fell as well as by how large they are, and the
    standard deviation of the fit :func:`tests.convergence.uncertainty_of` takes
    over the sequence refined in the bulk is how far one mesh stands off that
    trend. A difference no larger than it is another mesh rather than a finer
    rim.
    """
    bulk = [Line(one_mesh, ONE_MESH), *(Line(refinements, mesh) for mesh in FINER)]
    plain = bulk[0]
    rim = Line(rim_refined, ONE_MESH, edges=POLICY)
    matched = Line(bulk_matched, BULK_MATCHED)
    want = plain.closed_form
    estimate = convergence.uncertainty_of(
        [one.element for one in bulk], [float(one.impedance.mean()) for one in bulk]
    )
    impedance = {
        "plain": float(plain.impedance.mean()),
        "rim": float(rim.impedance.mean()),
        "matched": float(matched.impedance.mean()),
    }
    nearer = abs(impedance["matched"] - want) - abs(impedance["rim"] - want)

    def stated(line, name):
        return (
            f"{impedance[name]:.4f} ohm ({100 * (impedance[name] / want - 1):+.3f} %) at "
            f"{line.mesh:g} per wavelength and edge refinement "
            f"{line.runs[0]['edge_refinement']:g} on unknowns "
            f"{' and '.join(str(run['unknowns']) for run in line.runs)}"
        )

    print(
        f"\nGATE stripline through lumped ports on Palace, rim refined: Zc "
        f"{stated(plain, 'plain')}; {stated(rim, 'rim')}; {stated(matched, 'matched')}; "
        f"against {want:.4f} ohm; the rim nearer than the bulk of no fewer unknowns by "
        f"{nearer:.4f} ohm against a scatter of {estimate.scatter:.4f} ohm; "
        + "; ".join(
            f"at {name!r} on the {run['length']:g} mm line asked {place['asked']:.4g} mm, "
            f"elements along it up to {place['reached']:.4g} mm, standing on it "
            f"{place['standing']:.4g} mm"
            for run in rim.runs
            for name, place in run["reached"].items()
        )
    )
    for line in (plain, matched):
        for run in line.runs:
            assert not run["reached"], (
                f"the {run['length']:g} mm line at {line.mesh:g} per wavelength and edge "
                f"refinement {run['edge_refinement']:g} sized {sorted(run['reached'])}, so "
                "it is not refined in the bulk alone"
            )
    for run in rim.runs:
        assert run["reached"], (
            f"the {run['length']:g} mm line at edge refinement {run['edge_refinement']:g} "
            "sized no place, so what is compared is two meshes of the bulk"
        )
    for with_rim, without in zip(rim.runs, matched.runs, strict=True):
        assert without["unknowns"] >= with_rim["unknowns"], (
            f"the {without['length']:g} mm line refined in the bulk alone solves "
            f"{without['unknowns']} unknowns and with the rim refined "
            f"{with_rim['unknowns']}, so the comparison is between meshes of different "
            "cost. Raise BULK_MATCHED"
        )
    assert nearer > estimate.scatter, (
        f"with the rim refined the impedance is {impedance['rim']:.4f} ohm and refined in "
        f"the bulk alone to no fewer unknowns {impedance['matched']:.4f} ohm, against "
        f"{want:.4f} ohm: the rim stands nearer by {nearer:.4f} ohm, and one mesh stands "
        f"off the sequence by {estimate.scatter:.4f} ohm"
    )
