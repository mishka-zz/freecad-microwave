# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checking a declared mirror symmetry against what was drawn.

The policy under test is "trust the engineer, warn where it is cheap to look":
nothing here refuses a run, so what has to be right is *which* mismatches get
mentioned and which do not. A warning on a model that is fine is as bad as
silence on one that is not - both teach people to stop reading.

Built on the real ``Problem`` and the real mesher, not on stubs, because three
of the five checks read the grid the mesher produced and the whole point of the
grid check is that geometry and grid can disagree.
"""

import numpy as np

from Microwave.Gui.symmetry import mirror_warnings
from Microwave.Solvers.openems import plan
from Microwave.Solvers.openems.model import (
    Frequency,
    Material,
    Port,
    Problem,
    Solid,
)
from Microwave.Solvers.openems.regions import MeshParams

AIR = Material(name="Air", kind="dielectric", epsilon=1.0)
METAL = Material(name="Metal", kind="pec")


def line(port_overrides=(), solids=None, span=(-20.0, 20.0)):
    """A uniform two-port line: symmetric geometry, mirrored lumped ports."""
    low, high = span
    materials = (AIR, METAL)
    solids = (
        solids
        if solids is not None
        else (
            Solid(material="Air", lower=(low, -6, 0), upper=(high, 6, 3), label="Air"),
            Solid(material="Metal", lower=(low, -6, 0), upper=(high, 6, 0), label="Ground"),
        )
    )
    ports = []
    for number in (1, 2):
        edge = low if number == 1 else high
        inner = edge + 1.0 if number == 1 else edge - 1.0
        settings = dict(
            number=number,
            kind="lumped",
            start=(min(edge, inner), -1.0, 0.0),
            stop=(max(edge, inner), 1.0, 3.0),
            propagation_axis=0,
            excitation_axis=2,
            excite=(number == 1),
            feed_resistance=50.0,
        )
        settings.update(dict(port_overrides).get(number, {}))
        ports.append(Port(**settings))
    ports = tuple(ports)

    params = MeshParams(metal_res=1.0, dielectric_res=2.0, min_lines=4, pml_cells=8)
    grid = plan.plan_grid(solids, ports, materials, params, ((8, 8), (8, 8), (8, 8)))
    return Problem(
        title="line",
        frequency=Frequency(1e9, 10e9, 51),
        grid=grid,
        materials=materials,
        solids=solids,
        ports=ports,
        boundary=("PML_8",) * 6,
    )


class TestASymmetricModelIsLeftAlone:
    """The false-positive case, and the one that decides whether anyone reads
    these warnings at all."""

    def test_a_uniform_line_with_matched_ports_says_nothing(self):
        assert mirror_warnings(line()) == []


class TestThePortsMustBeMirrorsToo:
    def test_unequal_port_impedances_are_called_out(self):
        """The check that matters most and costs least.

        Completion copies S11 into S22 in the basis the solver measured in, and
        that basis is defined by the port impedances. A 25 ohm port facing a
        100 ohm one is not a mirror however symmetric the structure between
        them is - and this is exactly the acceptance two-port.
        """
        problem = line(port_overrides={1: {"feed_resistance": 25.0}, 2: {"feed_resistance": 100.0}})
        found = mirror_warnings(problem)
        assert any("25" in w and "100" in w for w in found), found

    def test_a_microstrip_port_pair_is_compared_by_reference_impedance(self):
        """A check reading ``feed_resistance`` and then a ``resistance``
        attribute ``Port`` does not have - so for a microstrip port, whose
        feed is a bare voltage source, it never ran at all. Two ports at 25 and
        100 ohm produced silence."""
        microstrip = {"kind": "microstrip", "metal": "Metal", "feed_resistance": None}
        problem = line(
            port_overrides={
                1: dict(microstrip, reference_impedance=25.0),
                2: dict(microstrip, reference_impedance=100.0),
            }
        )
        found = mirror_warnings(problem)
        assert any("25" in w and "100" in w for w in found), found

    def test_one_port_referenced_to_itself_and_one_to_a_number_is_called_out(self):
        """The two ports must be referenced the same *way*, not merely close.

        A copy taken in one basis and reported in another is the one error
        ``_derive_mirror`` cannot absorb, because the final renormalisation
        moves the two diagonal terms by different amounts.
        """
        problem = line(
            port_overrides={
                1: {"reference_impedance": 50.0},
                2: {"reference_impedance": None},
            }
        )
        found = mirror_warnings(problem)
        assert any("fixed impedance" in w and "its own" in w for w in found), found

    def test_different_port_kinds_are_called_out(self):
        problem = line(port_overrides={2: {"kind": "microstrip", "metal": "Metal"}})
        assert any("not mirror images" in w for w in mirror_warnings(problem))

    def test_different_measurement_planes_are_called_out(self):
        """Different shifts break S22 = S11 on a model that is otherwise
        perfectly symmetric: the reference planes are not mirrored."""
        problem = line(port_overrides={2: {"measurement_shift": 0.5}})
        assert any("measurement shift" in w for w in mirror_warnings(problem))


class TestTheDrawingAndTheGrid:
    def test_an_offset_feature_breaks_the_solid_mirror(self):
        """A stub on one side only. The geometry check is what sees it -
        nothing about the ports has changed."""
        solids = (
            Solid(material="Air", lower=(-20, -6, 0), upper=(20, 6, 3), label="Air"),
            Solid(material="Metal", lower=(-20, -6, 0), upper=(20, 6, 0), label="Ground"),
            Solid(material="Metal", lower=(-12, -1, 3), upper=(-8, 1, 3), label="Stub"),
        )
        found = mirror_warnings(line(solids=solids))
        assert any("not a mirror image about the middle" in w for w in found), found

    def test_a_partner_of_a_different_material_does_not_count(self):
        """Copper on one side, substrate on the other, is not a mirror.

        The position check alone accepted it: two boxes in mirrored places
        satisfy everything except being the same thing.
        """
        solids = (
            Solid(material="Air", lower=(-20, -6, 0), upper=(20, 6, 3), label="Air"),
            Solid(material="Metal", lower=(-20, -6, 0), upper=(20, 6, 0), label="Ground"),
            Solid(material="Metal", lower=(-12, -1, 3), upper=(-8, 1, 3), label="Stub"),
            Solid(material="Air", lower=(8, -1, 3), upper=(12, 1, 3), label="Pocket"),
        )
        found = mirror_warnings(line(solids=solids))
        assert any("not a mirror image about the middle" in w for w in found), found

    def test_a_partner_offset_across_the_other_axes_does_not_count(self):
        """Mirrored in x, but one stub sits at +y and the other at -y. Only the
        propagation axis is mirrored; everything else has to match outright."""
        solids = (
            Solid(material="Air", lower=(-20, -6, 0), upper=(20, 6, 3), label="Air"),
            Solid(material="Metal", lower=(-20, -6, 0), upper=(20, 6, 0), label="Ground"),
            Solid(material="Metal", lower=(-12, 1, 3), upper=(-8, 3, 3), label="StubA"),
            Solid(material="Metal", lower=(8, -3, 3), upper=(12, -1, 3), label="StubB"),
        )
        found = mirror_warnings(line(solids=solids))
        assert any("not a mirror image about the middle" in w for w in found), found

    def test_a_mirrored_pair_of_features_is_accepted(self):
        """Two stubs, one each side. Same object count, same materials - the
        difference from the case above is only *where*, which is the whole
        thing the check exists to see."""
        solids = (
            Solid(material="Air", lower=(-20, -6, 0), upper=(20, 6, 3), label="Air"),
            Solid(material="Metal", lower=(-20, -6, 0), upper=(20, 6, 0), label="Ground"),
            Solid(material="Metal", lower=(-12, -1, 3), upper=(-8, 1, 3), label="StubA"),
            Solid(material="Metal", lower=(8, -1, 3), upper=(12, 1, 3), label="StubB"),
        )
        assert mirror_warnings(line(solids=solids)) == []

    def test_an_asymmetric_grid_is_called_out(self):
        """The sneaky one: geometry perfectly symmetric, mesh lines not.

        Injected rather than provoked through the mesher, because *how* a grid
        comes out lopsided (snapping, a one-sided refinement, unequal air
        buffers) is the mesher's business and what matters here is that a
        lopsided one is noticed at all.
        """
        problem = line()
        lopsided = np.sort(np.append(np.asarray(problem.grid.x, dtype=float), -19.37))
        object.__setattr__(problem.grid, "x", lopsided)

        found = mirror_warnings(problem)
        assert any("mesh lines" in w for w in found), found


class TestWhereMirrorSymmetryDoesNotApply:
    def test_a_one_port_study_is_told_nothing_will_be_derived(self):
        problem = line()
        object.__setattr__(problem, "ports", problem.ports[:1])
        found = mirror_warnings(problem)
        assert len(found) == 1
        assert "1 port(s)" in found[0]
        assert "nothing will be derived" in found[0]
