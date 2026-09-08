# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the driver actually hands to openEMS, checked without openEMS.

``driver.build`` is the only code that imports openEMS, so the fast tests skip
it and the acceptance tests reach it only through a number at the far end of a
solve. Without this file the layer is structurally untested: hardcoding the
waveguide mode, dropping every solid's priority, ignoring the requested boundary
conditions or dropping the conversion of conducting-sheet thickness to metres
all pass the entire suite otherwise, and that last one puts the sheet's
resistance out by the millimetre-to-metre factor.

The instrument is a recording fake, injected over ``sys.modules``. That is not
"testing the mock": for an adapter, **the calls it makes are its output**. What
is asserted here is arithmetic we performed - 0.035 mm becoming 3.5e-5 m -
not openEMS' response to it. The fakes deliberately mirror the real binding
signatures, so a call the real library would reject fails here too:

    ContinuousStructure.AddConductingSheet(name, **kw)
    openEMS.AddRectWaveGuidePort(port_nr, start, stop, p_dir, a, b, mode_name, excite=0, **kw)
    openEMS.AddMSLPort(port_nr, metal_prop, start, stop, prop_dir, exc_dir, excite=0, **kw)

Keep them matching if the bindings move.
"""

from __future__ import annotations

import json
import math
import sys
import types
from dataclasses import replace

import numpy as np
import pytest

from Microwave.Solvers.openems import excitation, plan, staircase
from Microwave.Solvers.openems.model import (
    THROUGH,
    Frequency,
    Material,
    MeshGrid,
    Port,
    Problem,
    Solid,
    Termination,
)
from Microwave.Solvers.openems.regions import MeshParams


class FakePolyhedron:
    """A CSXCAD polyhedron. Records the points and triangles handed to it."""

    def __init__(self, priority: int):
        self.priority = priority
        self.vertices: list[tuple] = []
        self.faces: list[tuple] = []

    def AddVertex(self, *point):
        self.vertices.append(tuple(point))

    def AddFace(self, indices):
        self.faces.append(tuple(indices))


class FakeProperty:
    """A CSXCAD property. Records the primitives added to it."""

    def __init__(self, kind: str, name: str, **kw):
        self.kind = kind
        self.name = name
        self.kw = kw
        self.boxes: list[tuple] = []
        self.polyhedra: list[FakePolyhedron] = []

    def AddBox(self, start, stop, priority=0, **kw):
        self.boxes.append((tuple(start), tuple(stop), priority))

    def AddPolyhedron(self, priority=0, **kw):
        self.polyhedra.append(FakePolyhedron(priority))
        return self.polyhedra[-1]


class FakeGrid:
    def __init__(self):
        self.delta_unit = None
        self.lines: dict[str, list[float]] = {}

    def SetDeltaUnit(self, unit):
        self.delta_unit = unit

    def AddLine(self, ny, lines):
        self.lines.setdefault(ny, []).extend(np.atleast_1d(lines).tolist())


class FakeCSX:
    def __init__(self):
        self.grid = FakeGrid()
        self.properties: list[FakeProperty] = []
        self.written_to = None

    def GetGrid(self):
        return self.grid

    def Write2XML(self, path):
        self.written_to = path

    def _add(self, kind, name, **kw):
        prop = FakeProperty(kind, name, **kw)
        self.properties.append(prop)
        return prop

    def AddMetal(self, name, **kw):
        return self._add("metal", name, **kw)

    def AddMaterial(self, name, **kw):
        return self._add("material", name, **kw)

    def AddConductingSheet(self, name, **kw):
        return self._add("conducting_sheet", name, **kw)

    def AddLumpedElement(self, name, **kw):
        return self._add("lumped", name, **kw)


class FakePort:
    """A port, and after ``CalcPort`` the arrays the driver reads off one.

    The record is a drive that finishes well inside it and a ringing mode cut off
    at :attr:`record_decay` of its peak, so a test can say how truncated the run
    was and nothing else has to change. The current carries the same two with the
    outgoing one reversed, which is what gives the record an incident and a
    reflected wave to be split into.

    **Only the port that was excited has the drive in it.** A passive port's
    record is what came out of the device and nothing that went in, so its own
    incident wave is nothing at all - and a residual that divided a port by
    itself rather than by the port that drove would say so loudly.
    """

    #: Where the record was cut, as a share of its own peak. The default is a
    #: run that went on long enough; a test about truncation replaces it.
    record_decay = 1e-4

    #: Long enough to hold many cycles of the bands the problems here use.
    RECORD_SECONDS = 8e-9

    #: When the drive arrives, as a share of the record. One sample wide, so its
    #: spectrum is flat across whatever band a problem here asks for and
    #: truncating the record leaves it alone: what moves is the reflected wave.
    DRIVE_AT = 0.05

    #: One volt-nanosecond, the scale a transform of a one-volt pulse comes out
    #: at - so a share of it reads as a share rather than as an exponent. The
    #: drive in the record is scaled to transform to exactly this, so what the
    #: fake declares and what it recorded are the same wave.
    INCIDENT = 1e-9

    #: What the fake's waves are referenced to, in ohms.
    REFERENCE = 50.0

    def __init__(self, kind, **recorded):
        self.kind = kind
        self.recorded = recorded

    def CalcPort(self, sim_path, freq):
        self.calculated_in = sim_path
        freq = np.asarray(freq, dtype=float)
        points = freq.size
        times = np.linspace(0.0, self.RECORD_SECONDS, 4096)
        span = times[-1]
        drive = np.zeros_like(times)
        if self.recorded.get("excite"):
            drive[int(times.size * self.DRIVE_AT)] = self.INCIDENT / (2 * (times[1] - times[0]))
        envelope = self.record_decay ** (times / span)
        ringing = envelope * np.cos(2 * np.pi * freq.mean() * times)
        # Three voltage probes sharing one time axis, as an MSLPort has, and the
        # current on its own - a Yee scheme staggers the two by half a step.
        self.u_data = types.SimpleNamespace(ui_time=[times, times, times])
        self.i_data = types.SimpleNamespace(ui_time=[times + 0.5 * (times[1] - times[0])])
        self.ut_tot = drive + ringing
        self.it_tot = (drive - ringing) / self.REFERENCE
        self.Z_ref = np.full(points, self.REFERENCE)
        self.uf_inc = np.full(points, self.INCIDENT, dtype=complex)
        self.uf_ref = np.full(points, 0.1 * self.INCIDENT, dtype=complex)
        self.P_inc = np.ones(points)
        self.P_ref = np.full(points, 0.01)


class FakeFDTD:
    def __init__(self, NrTS=None, EndCriteria=None, TimeStepFactor=None, **kw):
        self.nr_ts = NrTS
        self.end_criteria = EndCriteria
        # Named, not swept into **kw, so that dropping the argument at the call
        # site is visible here. The real binding takes it the same way -
        # openEMS.__cinit__ pops 'TimeStepFactor' and calls SetTimeStepFactor.
        self.timestep_factor = TimeStepFactor
        self.excite = None
        self.boundary = None
        self.csx = None
        self.ports: list[FakePort] = []
        self.run_args = None

    def SetCustomExcite(self, _str, f0, fmax):
        self.excite = (_str, f0, fmax)

    def SetBoundaryCond(self, BC):
        self.boundary = list(BC)

    def SetCSX(self, CSX):
        self.csx = CSX

    def AddMSLPort(self, port_nr, metal_prop, start, stop, prop_dir, exc_dir, excite=0, **kw):
        port = FakePort(
            "msl",
            port_nr=port_nr,
            metal=metal_prop,
            start=tuple(start),
            stop=tuple(stop),
            prop_dir=prop_dir,
            exc_dir=exc_dir,
            excite=excite,
            **kw,
        )
        self.ports.append(port)
        return port

    def AddRectWaveGuidePort(self, port_nr, start, stop, p_dir, a, b, mode_name, excite=0, **kw):
        port = FakePort(
            "waveguide",
            port_nr=port_nr,
            start=tuple(start),
            stop=tuple(stop),
            p_dir=p_dir,
            a=a,
            b=b,
            mode_name=mode_name,
            excite=excite,
            **kw,
        )
        self.ports.append(port)
        return port

    def AddLumpedPort(self, port_nr, R, start, stop, p_dir, excite=0, **kw):
        port = FakePort(
            "lumped",
            port_nr=port_nr,
            R=R,
            start=tuple(start),
            stop=tuple(stop),
            p_dir=p_dir,
            excite=excite,
            **kw,
        )
        self.ports.append(port)
        return port

    def Run(self, sim_path, cleanup=False, **kw):
        self.run_args = (sim_path, cleanup, kw)


@pytest.fixture
def fake_engine(monkeypatch):
    """Stand in for CSXCAD and openEMS for the duration of one test."""
    csxcad = types.ModuleType("CSXCAD")
    csxcad.ContinuousStructure = FakeCSX
    engine = types.ModuleType("openEMS")
    engine.openEMS = FakeFDTD
    engine.__version__ = "fake"
    monkeypatch.setitem(sys.modules, "CSXCAD", csxcad)
    monkeypatch.setitem(sys.modules, "openEMS", engine)
    return csxcad, engine


COPPER_THICKNESS_MM = 0.035
COPPER_CONDUCTIVITY = 5.8e7


def _microstrip() -> Problem:
    materials = (
        Material(name="FR4", kind="dielectric", epsilon=4.4),
        Material(name="GroundPlane", kind="pec"),
        Material(
            name="Trace",
            kind="conducting_sheet",
            conductivity=COPPER_CONDUCTIVITY,
            thickness=COPPER_THICKNESS_MM,
        ),
    )
    solids = (
        Solid(
            material="FR4", lower=(-50, -10, 0), upper=(50, 10, 1.6), priority=3, label="Substrate"
        ),
        Solid(
            material="GroundPlane",
            lower=(-50, -10, 0),
            upper=(50, 10, 0),
            priority=7,
            label="Ground",
        ),
    )
    ports = (
        Port(
            number=1,
            kind="microstrip",
            metal="Trace",
            start=(-50, -1.5, 1.6),
            stop=(50, 1.5, 0.0),
            propagation_axis=0,
            excitation_axis=2,
            excite=True,
            feed_shift=20.0,
            measurement_shift=50.0,
        ),
    )
    params = MeshParams(metal_res=0.5, dielectric_res=1.0, min_lines=6, pml_cells=8)
    grid = plan.plan_grid(solids, ports, materials, params, ((THROUGH, THROUGH), (8, 8), (8, 8)))
    return Problem(
        frequency=Frequency(1e9, 10e9, 51),
        grid=grid,
        materials=materials,
        solids=solids,
        ports=ports,
        boundary=("PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"),
        termination=Termination(max_timesteps=1234, end_criteria=0.0),
    )


def _two_port() -> Problem:
    """The same line with a second port on its far end, undriven."""
    driven = _microstrip()
    far = replace(
        driven.ports[0],
        number=2,
        start=(50, -1.5, 1.6),
        stop=(-50, 1.5, 0.0),
        excite=False,
    )
    return replace(driven, ports=(*driven.ports, far))


def _waveguide(broad_axis: int = 0) -> Problem:
    section = [4.3, 4.3]
    section[broad_axis] = 10.7
    width, height = section

    materials = (Material(name="Air", kind="dielectric", epsilon=1.0),)
    solids = (Solid(material="Air", lower=(0, 0, 0), upper=(width, height, 50.0)),)
    ports = (
        Port(
            number=1,
            kind="rect_waveguide",
            mode="TE20",
            start=(0.0, 0.0, 4.0),
            stop=(width, height, 6.0),
            propagation_axis=2,
            excite=True,
        ),
    )
    params = MeshParams(metal_res=0.4, dielectric_res=0.4, min_lines=10, pml_cells=(0, 0, 8))
    grid = plan.plan_grid(solids, ports, materials, params, ((0, 0), (0, 0), (THROUGH, THROUGH)))
    return Problem(
        frequency=Frequency(20e9, 26e9, 51),
        grid=grid,
        materials=materials,
        solids=solids,
        ports=ports,
        boundary=("PEC", "PEC", "PEC", "PEC", "PML_8", "PML_8"),
    )


def _build(problem, tmp_path):
    from Microwave.Solvers.openems import driver

    return driver.build(problem, tmp_path)


class TestAKindWithNoBuilderIsRefused:
    """The far end of the capability declaration, where "never a silent no-op"
    is either honoured or not.

    Widening ``MATERIAL_KINDS`` or ``PORT_KINDS`` is how a kind gets this far:
    the envelope's ``__post_init__`` is the only gate above the driver, so a
    vocabulary that grows without a builder to match arrives here. A material
    reached the fall-through and was built as a plain dielectric - epsilon and
    mu of whatever the new kind happened to carry, no message, a full run of
    numbers. Ports already refused; both do now.
    """

    def _grown(self, monkeypatch, name, extra):
        from Microwave.Solvers.openems import model

        monkeypatch.setattr(model, name, getattr(model, name) | {extra})

    def test_a_material_kind_with_no_builder(self, monkeypatch, fake_engine, tmp_path):
        from Microwave.Solvers.openems.model import EnvelopeError

        self._grown(monkeypatch, "MATERIAL_KINDS", "magnetic")
        problem = _microstrip()
        grown = replace(
            problem, materials=(Material(name="Ferrite", kind="magnetic"),) + problem.materials
        )

        with pytest.raises(EnvelopeError, match="no builder for material kind 'magnetic'"):
            _build(grown, tmp_path)

    def test_a_port_kind_with_no_builder(self, monkeypatch, fake_engine, tmp_path):
        from Microwave.Solvers.openems.model import EnvelopeError

        # No excitation axis: the envelope asks for one only from the kinds that
        # integrate a voltage across a gap, and refuses it from the rest - so a
        # kind it has never heard of has to arrive without one to get this far.
        self._grown(monkeypatch, "PORT_KINDS", "differential")
        problem = _microstrip()
        grown = replace(
            problem,
            ports=(replace(problem.ports[0], kind="differential", excitation_axis=None),)
            + problem.ports[1:],
        )

        with pytest.raises(EnvelopeError, match="no builder for port kind 'differential'"):
            _build(grown, tmp_path)


class TestUnitsCrossingTheBoundary:
    """The traps live here. Every other length is in grid units; some are not."""

    def test_sheet_thickness_is_converted_to_metres(self, fake_engine, tmp_path):
        """CSXCAD wants this one field in SI while its neighbours are in mm.

        Getting it wrong does not fail: openEMS notices the surface-impedance
        fit is out of range, clamps to its last tabulated coefficients and
        completes. A 50 ohm line came back as 10.7 kilohms.
        """
        _, csx, _ = _build(_microstrip(), tmp_path)
        sheet = next(p for p in csx.properties if p.kind == "conducting_sheet")
        assert sheet.kw["thickness"] == pytest.approx(COPPER_THICKNESS_MM * 1e-3)
        assert sheet.kw["thickness"] != pytest.approx(COPPER_THICKNESS_MM)

    def test_conductivity_is_not_converted(self, fake_engine, tmp_path):
        """It is already SI. Converting it too would be the symmetric mistake."""
        _, csx, _ = _build(_microstrip(), tmp_path)
        sheet = next(p for p in csx.properties if p.kind == "conducting_sheet")
        assert sheet.kw["conductivity"] == pytest.approx(COPPER_CONDUCTIVITY)

    def test_the_grid_keeps_its_own_unit(self, fake_engine, tmp_path):
        problem = _microstrip()
        _, csx, _ = _build(problem, tmp_path)
        assert csx.grid.delta_unit == problem.length_unit

    def test_geometry_stays_in_grid_units(self, fake_engine, tmp_path):
        """Boxes are millimetres; only the sheet thickness is not.

        Read as an extent, which the placement at the origin cannot change and
        a conversion to metres could not survive.
        """
        problem = _microstrip()
        _, csx, _ = _build(problem, tmp_path)
        substrate = next(p for p in csx.properties if p.name == "FR4")
        drawn = next(solid for solid in problem.solids if solid.material == "FR4")
        lower, upper, _ = substrate.boxes[0]
        assert [b - a for a, b in zip(lower, upper)] == pytest.approx(
            [b - a for a, b in zip(drawn.lower, drawn.upper)]
        )

    def test_waveguide_dimensions_are_converted_to_metres(self, fake_engine, tmp_path):
        """a and b are SI while the box that defines them is in grid units."""
        fdtd, _, _ = _build(_waveguide(), tmp_path)
        port = fdtd.ports[0]
        assert port.recorded["a"] == pytest.approx(10.7e-3)
        assert port.recorded["b"] == pytest.approx(4.3e-3)


class TestNothingIsDroppedOnTheWay:
    """Each of these was a surviving mutant: the value was silently ignored."""

    def test_the_requested_boundary_conditions_are_passed(self, fake_engine, tmp_path):
        problem = _microstrip()
        fdtd, _, _ = _build(problem, tmp_path)
        assert fdtd.boundary == list(problem.boundary)

    def test_solid_priorities_are_passed(self, fake_engine, tmp_path):
        """Priority decides which material wins where two overlap.

        Flattening them to a constant makes a ground plane vanish into the
        substrate it sits on, with no error.
        """
        _, csx, _ = _build(_microstrip(), tmp_path)
        priorities = {p.name: p.boxes[0][2] for p in csx.properties if p.boxes}
        assert priorities["FR4"] == 3
        assert priorities["GroundPlane"] == 7

    def test_the_waveguide_mode_is_passed(self, fake_engine, tmp_path):
        """Hardcoding TE10 would pass every other test in the suite."""
        fdtd, _, _ = _build(_waveguide(), tmp_path)
        port = fdtd.ports[0].recorded
        assert port["mode_name"] == "TE20"
        assert (port["a"], port["b"]) == pytest.approx((0.0107, 0.0043))

    def test_a_rotated_guide_arrives_with_its_digits_on_the_same_axes(self, fake_engine, tmp_path):
        """``a`` goes to the first transverse axis whether or not it is broad.

        This is the call site of the fault behind ``DEFERRED.md`` item 3: the pair
        sorted broad-first put 10.7 mm on a 4.3 mm axis, and the guide launched a
        field that is not one of its modes. Asserted here as well as in the model
        because passing them in the wrong order is a mistake available only here.
        """
        fdtd, _, _ = _build(_waveguide(broad_axis=1), tmp_path)
        port = fdtd.ports[0].recorded
        assert (port["a"], port["b"]) == pytest.approx((0.0043, 0.0107))
        assert port["mode_name"] == "TE02"

    def test_termination_is_passed(self, fake_engine, tmp_path):
        """A run that ignores its step count is not reproducible."""
        fdtd, _, _ = _build(_microstrip(), tmp_path)
        assert fdtd.nr_ts == 1234
        assert fdtd.end_criteria == 0.0

    def test_the_timestep_factor_is_passed(self, fake_engine, tmp_path):
        """Nothing about a solve says whether the factor arrived: openEMS writes
        the attribute into its XML only when it is *above* one, which is the one
        range it refuses to apply."""
        problem = replace(_microstrip(), timestep_factor=0.8)
        fdtd, _, _ = _build(problem, tmp_path)
        assert fdtd.timestep_factor == 0.8

    def test_the_default_factor_is_passed_rather_than_skipped(self, fake_engine, tmp_path):
        """One is the engine's own step *by the engine's arithmetic* - openEMS
        applies the factor only below one - so there is nothing to gain from
        the call site second-guessing it, and a conditional there is a branch
        that can be got wrong."""
        fdtd, _, _ = _build(_microstrip(), tmp_path)
        assert fdtd.timestep_factor == 1.0

    def test_the_excitation_spectrum_follows_the_band(self, fake_engine, tmp_path):
        problem = _microstrip()
        fdtd, _, _ = _build(problem, tmp_path)
        assert fdtd.excite[0] == excitation.expression(5.5e9, 4.5e9)

    def test_and_both_frequencies_it_is_given_are_the_top_of_that_band(self, fake_engine, tmp_path):
        """Neither argument is what its name says. openEMS takes the probes'
        sampling rate from the second, overwriting whatever the third said, and
        builds the conducting-sheet model off the third before the signal
        exists - so a centre frequency in either place undersamples the record
        or ages the sheet at the wrong frequency.
        """
        fdtd, _, _ = _build(_microstrip(), tmp_path)
        assert fdtd.excite[1:] == (10e9, 10e9)

    def test_the_grid_reaches_the_solver_as_drawn_apart_from_where_it_is(self):
        """The envelope's lines are what gets solved - that is what makes a
        mesh preview honest. The engine is handed them at the origin rather than
        where they were drawn, and a translation is the one change that leaves
        every spacing in a mesh alone.
        """
        problem = _microstrip()
        placed, offset = problem.at_the_origin()
        for dim in range(3):
            assert placed.grid[dim].tolist() == pytest.approx(
                (problem.grid[dim] + offset[dim]).tolist()
            )
            assert np.diff(placed.grid[dim]).tolist() == pytest.approx(
                np.diff(problem.grid[dim]).tolist()
            )

    def test_and_the_grid_the_geometry_and_the_ports_are_in_one_system(self, fake_engine, tmp_path):
        """The way a translation can go wrong that is worse than not doing one:
        a structure part of which moved.

        Read off what the engine was handed against what the envelope holds, and
        never against what the placement says those should be - those two agree
        by construction even when the placement is wrong, so a check made that
        way passes on a model whose grid moved and whose metal stayed put.
        """
        problem = _microstrip()
        fdtd, csx, _ = _build(problem, tmp_path)

        moved: dict[str, tuple] = {
            "the grid": tuple(
                csx.grid.lines[axis][0] - float(problem.grid[dim][0])
                for dim, axis in enumerate("xyz")
            )
        }
        for material in {solid.material for solid in problem.solids}:
            drawn = [solid for solid in problem.solids if solid.material == material]
            boxes = next(p for p in csx.properties if p.name == material).boxes
            assert len(boxes) == len(drawn), f"{material} reached the engine as other solids"
            for solid, box in zip(drawn, boxes):
                moved[f"solid {solid.name!r}"] = tuple(
                    here - there for here, there in zip(box[0], solid.lower)
                )
        for number, port in enumerate(problem.ports):
            moved[f"port {port.number}"] = tuple(
                here - there
                for here, there in zip(fdtd.ports[number].recorded["start"], port.start)
            )

        # Not equality: recovering an offset by subtraction lands on a different
        # last bit for each pair of operands it is recovered from. A structure
        # part of which did not move disagrees by the offset itself.
        theirs = moved.pop("the grid")
        for what, ours in moved.items():
            assert ours == pytest.approx(theirs, rel=1e-9), (
                f"{what} did not move with the grid, which went to {theirs}"
            )

    def test_the_microstrip_excitation_sign_is_passed(self, fake_engine, tmp_path):
        """Trace-to-ground integration runs downward, so it must be negated."""
        fdtd, _, _ = _build(_microstrip(), tmp_path)
        assert fdtd.ports[0].recorded["excite"] == -1

    def test_port_shifts_are_passed(self, fake_engine, tmp_path):
        fdtd, _, _ = _build(_microstrip(), tmp_path)
        recorded = fdtd.ports[0].recorded
        assert recorded["FeedShift"] == 20.0
        assert recorded["MeasPlaneShift"] == 50.0


class TestOrderingTheEngineRequires:
    def test_the_grid_exists_before_any_port_is_built(self, fake_engine, tmp_path):
        """MSLPort and RectWGPort both read the grid at construction. MSLPort
        counts the lines on its axis and raises where there are too few;
        RectWGPort takes the grid's unit off it and counts nothing. Either way
        the grid has to be in place first."""
        problem = _waveguide()

        seen: list[str] = []

        class Watchful(FakeFDTD):
            def AddRectWaveGuidePort(self, *args, **kw):
                seen.append(f"port:{len(self.csx.grid.lines)}")
                return super().AddRectWaveGuidePort(*args, **kw)

        sys.modules["openEMS"].openEMS = Watchful
        _build(problem, tmp_path)
        assert seen == ["port:3"], "the port was built before all three axes existed"

    def test_the_structure_is_written_for_debugging(self, fake_engine, tmp_path):
        """Opening the exact geometry that was solved is worth a lot in a bug
        report, so the XML is not optional."""
        _, csx, _ = _build(_microstrip(), tmp_path)
        assert csx.written_to is not None and str(csx.written_to).endswith(".xml")


class TestTheRecordAResultCarries:
    """Provenance is what makes a result two weeks old falsifiable.

    Tested here and not through a solve: ``_provenance`` is pure arithmetic over
    the envelope, and every field of it can revert silently - ``title`` did,
    and shipped, leaving every chart and every Touchstone network unnamed until
    somebody noticed the axis label was blank.
    """

    def provenance(self, problem, records=None):
        from Microwave.Solvers.openems import driver

        return driver._provenance(problem, elapsed=1.5, records=records or {})

    def test_it_carries_the_study_s_name(self):
        """``SParameters.network()`` and the S-parameter plot both read it, and
        neither has anything else to fall back on."""
        problem = replace(_microstrip(), title="Microstrip 50 ohm")
        assert self.provenance(problem)["title"] == "Microstrip 50 ohm"

    def test_it_ties_the_result_to_the_envelope_that_produced_it(self):
        problem = _microstrip()
        assert self.provenance(problem)["envelope_digest"] == problem.digest()

    def test_it_records_the_termination_that_was_asked_for(self):
        """A run that stopped on energy decay is not reproducible, and anything
        comparing two results has to be able to tell."""
        problem = _microstrip()
        record = self.provenance(problem)
        assert record["max_timesteps"] == 1234
        assert record["reproducible"] is True
        assert record["cells"] == problem.grid.cell_count

    def test_it_records_the_timestep_factor(self):
        """The only record of it. openEMS' own XML omits the factor for every
        value it accepts, so a result solved at 0.5 and one solved at 1.0 are
        otherwise indistinguishable after the fact - and they do not agree."""
        problem = replace(_microstrip(), timestep_factor=0.5)
        assert self.provenance(problem)["timestep_factor"] == 0.5

    def test_it_records_what_the_tail_shares_were_judged_against(self):
        """They are shares of the drive, and what makes one of them a verdict
        is the response the study said it reads. Anything reading this file
        later has to be able to reach the same verdict, and the tail shares
        alone do not say which one it was."""
        problem = replace(_microstrip(), smallest_response=0.01)
        assert self.provenance(problem)["smallest_response"] == 0.01

    def test_what_was_measured_off_the_records_goes_in_whole(self):
        """Per port, and under the names ``_extract`` gave them. A merge that
        renamed or flattened them would leave ``tail_share`` describing one port
        of a sweep that had several."""
        records = {"tail_share": {1: 1e-4, 2: 0.2}, "recorded_samples": {1: 181, 2: 181}}
        assert self.provenance(_microstrip(), records)["tail_share"] == {1: 1e-4, 2: 0.2}
        assert self.provenance(_microstrip(), records)["recorded_samples"] == {1: 181, 2: 181}


class TestWhatASolveSaysAboutItsOwnRecords:
    """The one route every solve passes through, run end to end against fakes.

    ``_extract`` is where the tail is weighed, and it is also the only place the
    port objects exist at all - so a measurement that silently stopped being
    taken, or a warning that stopped being emitted, has nowhere else to show up
    until somebody solves a resonator by hand.
    """

    def solved(self, tmp_path, problem=None):
        from Microwave.Solvers.openems import driver

        path = driver.solve(problem or _microstrip(), tmp_path)
        return json.loads(path.read_text())

    def test_a_run_that_went_on_long_enough_is_under_the_bar(self, fake_engine, tmp_path):
        from Microwave.Solvers.openems import residual

        assert self.solved(tmp_path)["provenance"]["tail_share"]["1"] < residual.WANTED

    def test_every_port_is_weighed_and_not_only_the_driven_one(self, fake_engine, tmp_path):
        """A passive port watches a different part of the device, and on a
        filter it is the one still ringing."""
        provenance = self.solved(tmp_path, _two_port())["provenance"]
        assert sorted(provenance["tail_share"]) == ["1", "2"]
        assert sorted(provenance["recorded_samples"]) == ["1", "2"]

    def test_and_each_is_weighed_against_the_port_that_drove(self, fake_engine, tmp_path):
        """An S-parameter is one port's reflection over the *driven* port's
        incident wave, so a passive port's share is not its own record divided by
        itself.

        Both ports ring the same way here and only one of them was driven, so
        the wave coming back out is the same at both and the wave it is weighed
        against is the same one twice: the two shares are one number. A measure
        asking each port what drove it would ask the passive one about a drive it
        never saw.
        """
        from Microwave.Solvers.openems import residual

        share = self.solved(tmp_path, _two_port())["provenance"]["tail_share"]
        # Not to the last bit: the current probe sits half a step behind the
        # voltage probe, so what cancels out of one port's split is a phase away
        # from cancelling out of the other's.
        assert share["2"] == pytest.approx(share["1"], rel=1e-3, abs=0.0)
        assert 0.0 < share["2"] < residual.WANTED

    def test_and_off_the_probes_and_the_axes_the_run_wrote(self, fake_engine, tmp_path):
        """The wiring between the port objects and the measurement, which no
        figure downstream would show as wrong - a residual computed off the wrong
        axis, or against the wrong reference, is a plausible small number."""
        from Microwave.Solvers.openems import residual

        problem = _microstrip()
        reported = self.solved(tmp_path, problem)["provenance"]["tail_share"]["1"]
        port = FakePort("lumped", excite=1)
        port.CalcPort(str(tmp_path), problem.frequency.values())
        record = residual.Record(
            voltage=residual.Probe(port.u_data.ui_time[0], port.ut_tot),
            current=residual.Probe(port.i_data.ui_time[0], port.it_tot),
            reference=port.Z_ref,
        )
        assert reported == pytest.approx(
            residual.tail_shares({1: record}, 1, problem.frequency.values())[1],
            rel=1e-12,
            abs=0.0,
        )

    def test_a_run_that_stopped_while_it_was_ringing_says_so(
        self, fake_engine, tmp_path, monkeypatch, capsys
    ):
        """On a marker, which is what the headless route has. The number is in
        provenance whether or not this fires; the marker is what makes it
        arrive without being asked for."""
        monkeypatch.setattr(FakePort, "record_decay", 0.5)

        self.solved(tmp_path)

        checks = [line for line in capsys.readouterr().out.splitlines() if "CHECK" in line]
        assert len(checks) == 1
        assert "severity=warn" in checks[0] and "stopped before the response" in checks[0]

    def test_a_run_that_finished_says_nothing(self, fake_engine, tmp_path, capsys):
        self.solved(tmp_path)
        assert "CHECK" not in capsys.readouterr().out

    def test_the_same_run_is_called_short_by_a_study_reading_deeper(
        self, fake_engine, tmp_path, capsys
    ):
        """The one route with no human on it, so the declaration has to reach
        the marker rather than only the panel: an unattended search over a
        stopband would otherwise be handed leakage as a result."""
        deeper = replace(_microstrip(), smallest_response=1e-4)
        self.solved(tmp_path, deeper)

        checks = [line for line in capsys.readouterr().out.splitlines() if "CHECK" in line]
        assert len(checks) == 1
        assert "severity=warn" in checks[0] and "-80 dB" in checks[0]


# ---------------------------------------------------------------------------
# Which surfaces are grown before the engine samples them
# ---------------------------------------------------------------------------

_OCTAHEDRON_FACES = (
    (0, 1, 4),
    (1, 2, 4),
    (2, 3, 4),
    (3, 0, 4),
    (1, 0, 5),
    (2, 1, 5),
    (3, 2, 5),
    (0, 3, 5),
)


def _octahedron(centre, radius):
    """A closed surface wound outward, whose every vertex normal is an axis.

    Which is what makes it worth the fixture: the step at each point is then one
    axis's own cell rather than a mixture of three, so what reached the engine
    can be read against the grid without trigonometry.
    """
    points = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            points.append(tuple(c + sign * radius * (dim == axis) for dim, c in enumerate(centre)))
    order = (0, 2, 1, 3, 4, 5)
    return tuple(points[index] for index in order), _OCTAHEDRON_FACES


def _prism(centre, radius, height, facets=16):
    """A closed prism on ``z``, capped by fans from its own rim.

    An octahedron cannot pose the question the clearance answers: every one of
    its faces lies oblique, so the mesher pins a line to none of them and there
    is nothing for a displacement to make conduct. This one has two faces square
    to an axis and a wall that is square to none, so both corrections are legible
    in one shape and in different directions.
    """
    ring = [
        (
            centre[0] + radius * math.cos(2 * math.pi * step / facets),
            centre[1] + radius * math.sin(2 * math.pi * step / facets),
        )
        for step in range(facets)
    ]
    vertices = [(x, y, centre[2] + height / 2) for x, y in ring]
    vertices += [(x, y, centre[2] - height / 2) for x, y in ring]
    faces = []
    for step in range(facets):
        here, ahead = step, (step + 1) % facets
        faces.append((here, facets + here, facets + ahead))
        faces.append((here, facets + ahead, ahead))
    for step in range(1, facets - 1):
        faces.append((0, step, step + 1))
        faces.append((facets, facets + step + 1, facets + step))
    return tuple(vertices), tuple(faces)


def _with_a_curved_solid(material_kind: str, shape=None) -> Problem:
    vertices, faces = _octahedron((10.0, 10.0, 10.0), 4.0) if shape is None else shape
    lower = tuple(min(point[dim] for point in vertices) for dim in range(3))
    upper = tuple(max(point[dim] for point in vertices) for dim in range(3))
    return Problem(
        title="one curved solid",
        frequency=Frequency(start=1e9, stop=2e9, points=11),
        grid=MeshGrid(
            x=np.arange(0.0, 20.5, 0.5),
            y=np.arange(0.0, 20.5, 0.5),
            z=np.arange(0.0, 20.5, 0.5),
        ),
        materials=(
            Material(name="Body", kind=material_kind, epsilon=2.0)
            if material_kind == "dielectric"
            else Material(name="Body", kind=material_kind),
        ),
        solids=(
            Solid(
                material="Body",
                lower=lower,
                upper=upper,
                priority=10,
                label="Ball",
                vertices=vertices,
                faces=faces,
            ),
        ),
        ports=(
            Port(
                number=1,
                kind="lumped",
                start=(2.0, 2.0, 2.0),
                stop=(3.0, 3.0, 3.0),
                propagation_axis=1,
                excitation_axis=2,
                excite=True,
                feed_resistance=50.0,
                reference_impedance=50.0,
                label="Probe",
            ),
        ),
        boundary=("PEC",) * 6,
        termination=Termination(max_timesteps=10, end_criteria=0.0),
    )


class TestWhichSurfacesTheEnvelopeAnswersFor:
    """``as_given`` says what openEMS is handed, and for most solids that is what
    was drawn: it returns nothing rather than a copy.

    Asked of the method rather than of what the engine got, because most of these
    are shapes the driver already declines to correct by another route - a box
    never reaches the polyhedron builder, and a sheet's own primitive ignores the
    surface offered to it. A growth made here for one of them would be invisible
    in the structure and still be there for the next caller that trusts it.
    """

    def _one(self, problem: Problem):
        return problem.as_given(problem.solids[0])

    def test_a_curved_conductor_is_answered_for(self):
        problem = _with_a_curved_solid("pec")
        assert self._one(problem) != problem.solids[0].vertices

    def test_a_dielectric_is_not(self):
        """openEMS averages it over quarter cells rather than sampling a point,
        so it carries no rounding and growing it would introduce one."""
        assert self._one(_with_a_curved_solid("dielectric")) is None

    def test_a_box_is_not(self):
        """The mesher pins a line to each of its faces, so there is nothing left
        to round."""
        problem = _with_a_curved_solid("pec")
        box = replace(problem.solids[0], vertices=(), faces=())
        assert problem.as_given(box) is None

    def test_a_sheet_is_not(self):
        """It is modelled at the plane it lies in, where there is no outward
        direction to grow along; its rim is a separate question."""
        problem = _with_a_curved_solid("pec")
        corners = ((6.0, 6.0, 10.0), (14.0, 6.0, 10.0), (14.0, 14.0, 10.0), (6.0, 14.0, 10.0))
        sheet = replace(
            problem.solids[0],
            lower=(6.0, 6.0, 10.0),
            upper=(14.0, 14.0, 10.0),
            vertices=corners,
            faces=((0, 1, 2), (0, 2, 3)),
            sheet_normal=2,
        )
        assert problem.as_given(sheet) is None


class TestAConductorIsGrownBeforeItIsSampled:
    """openEMS decides a metal edge on one point and so builds a conductor's
    surface at the last grid line still inside the drawing. The adapter answers for
    that by handing over a surface grown by half a cell, and this is where that
    decision is read off what the engine was actually given - the alternative
    being a two-minute solve, which is where it was only covered before."""

    def _handed(self, problem, tmp_path):
        _, csx, _ = _build(problem, tmp_path)
        primitive = next(p for p in csx.properties if p.polyhedra).polyhedra[0]
        return np.asarray(primitive.vertices), primitive

    def test_a_metal_surface_arrives_half_a_cell_out(self, fake_engine, tmp_path):
        problem = _with_a_curved_solid("pec")
        drawn = np.asarray(problem.solids[0].vertices)
        handed, _ = self._handed(problem, tmp_path)
        centre = drawn.mean(axis=0)
        was = np.linalg.norm(drawn - centre, axis=1)
        # Read against the placed drawing rather than the original: the
        # structure is moved to the origin first, and both went together.
        now = np.linalg.norm(handed - handed.mean(axis=0), axis=1)
        assert now == pytest.approx(was + 0.25, rel=1e-9, abs=0.0), (
            "a conductor did not reach the engine grown by half a cell"
        )

    def test_a_dielectric_surface_arrives_as_it_was_drawn(self, fake_engine, tmp_path):
        """openEMS averages a dielectric over the cell rather than sampling a
        point, so it carries no such rounding and growing it would introduce
        one."""
        problem = _with_a_curved_solid("dielectric")
        drawn = np.asarray(problem.solids[0].vertices)
        handed, _ = self._handed(problem, tmp_path)
        was = np.linalg.norm(drawn - drawn.mean(axis=0), axis=1)
        now = np.linalg.norm(handed - handed.mean(axis=0), axis=1)
        assert now == pytest.approx(was, rel=1e-9, abs=0.0), (
            "a dielectric was moved, and nothing rounds it"
        )

    def test_the_engine_gets_exactly_the_surface_the_envelope_says_it_will(
        self, fake_engine, tmp_path
    ):
        """Everything that asks where a conductor's metal is asks
        :meth:`~Microwave.Solvers.openems.model.Problem.as_given`, so its answer
        has to be the surface the engine gets - point for point, not a size or a
        distance two different surfaces could share.
        """
        problem = _with_a_curved_solid("pec")
        placed, _ = problem.at_the_origin()
        handed, _ = self._handed(problem, tmp_path)
        assert handed.tolist() == [list(point) for point in placed.as_given(placed.solids[0])]

    def test_the_triangles_are_handed_over_unchanged(self, fake_engine, tmp_path):
        """Growing a surface moves its points. How they are joined is what makes
        it the same surface."""
        problem = _with_a_curved_solid("pec")
        _, primitive = self._handed(problem, tmp_path)
        assert primitive.faces == [tuple(face) for face in problem.solids[0].faces]

    def test_a_box_is_not_a_surface_and_is_not_moved(self, fake_engine, tmp_path):
        """A box arrives exactly, because the mesher pins a line to each of its
        faces - there is no rounding to answer for and moving it would be one."""
        problem = _microstrip()
        _, csx, _ = _build(problem, tmp_path)
        assert not any(prop.polyhedra for prop in csx.properties)

    def test_a_share_of_nothing_hands_the_metal_over_as_it_was_drawn(self, fake_engine, tmp_path):
        """Which is the case that prices the correction against not making it -
        see :class:`~Microwave.Solvers.openems.model.Problem`."""
        problem = replace(_with_a_curved_solid("pec"), grown_by=0.0)
        drawn = np.asarray(problem.solids[0].vertices)
        handed, _ = self._handed(problem, tmp_path)
        was = np.linalg.norm(drawn - drawn.mean(axis=0), axis=1)
        now = np.linalg.norm(handed - handed.mean(axis=0), axis=1)
        assert now == pytest.approx(was, rel=1e-9, abs=0.0), (
            "a conductor was moved on a run that asked for no correction"
        )

    def test_and_says_nothing_about_having_grown_it(self, fake_engine, tmp_path, capsys):
        """``GROWN`` is the marker that says the structure openEMS was given is
        not the size it was drawn. On this run it is, so the run has nothing to
        report and a marker reading zero would be the opposite of the truth."""
        _build(replace(_with_a_curved_solid("pec"), grown_by=0.0), tmp_path)
        assert "GROWN" not in capsys.readouterr().out

    def test_where_a_run_that_grew_it_does(self, fake_engine, tmp_path, capsys):
        _build(_with_a_curved_solid("pec"), tmp_path)
        assert "GROWN" in capsys.readouterr().out


class TestAFlatFaceIsDisplacedSoTheLinePinnedToItConducts:
    """The other correction the same call makes, and it is not a share of the
    same thing. A flat conductor face square to an axis gets a grid line of its
    own from the mesher, and the line lands *on* the face - where openEMS'
    containment ray has nothing to be sure about, so the face can read as air and
    never zero the tangential field standing on it. The face is handed over
    displaced into the void by enough for that line to fall in metal.

    Read here off what the engine was given, and off a prism rather than an
    octahedron, because a shape with no face square to an axis cannot show it.
    """

    def _capped(self, **fields) -> Problem:
        shape = _prism((10.0, 10.0, 10.0), 4.0, 6.0)
        return replace(_with_a_curved_solid("pec", shape=shape), **fields)

    def _standing_on_a_cap(self, **fields) -> Problem:
        """The same prism with the element drawn just off its top face.

        Off it rather than on it: an end that landed on a grid line did not move,
        so the check has nothing to ask about it, and an element meeting no metal
        at all is a probe in free space and is left alone. Both ends here snap,
        which is the case where what conducts has to be decided.
        """
        port = replace(
            self._capped().ports[0],
            start=(9.5, 9.5, 13.2),
            stop=(10.5, 10.5, 14.2),
            propagation_axis=0,
            excitation_axis=2,
        )
        return self._capped(ports=(port,), **fields)

    def _cell(self, problem) -> float:
        """The grid's own pitch, which both corrections are shares of."""
        return float(np.diff(problem.grid.z)[0])

    def _reach(self, problem, tmp_path):
        """How far the caps and the wall each ended up from where they were drawn.

        Measured about each set's own centre, so that the translation onto the
        origin drops out. Both corrections push a closed surface outward, so
        neither moves that centre and neither is hidden by taking it out.
        """
        drawn = np.asarray(problem.solids[0].vertices)
        _, csx, _ = _build(problem, tmp_path)
        handed = np.asarray(next(p for p in csx.properties if p.polyhedra).polyhedra[0].vertices)
        step = (handed - handed.mean(axis=0)) - (drawn - drawn.mean(axis=0))
        return float(np.abs(step[:, 2]).max()), float(np.hypot(step[:, 0], step[:, 1]).max())

    def test_the_clearance_the_run_carries_reaches_the_surface(self, fake_engine, tmp_path):
        """The envelope's field and not the module's constant, so a run
        reproduced from a file builds the wall that file describes."""
        problem = self._capped(pinned_clearance=0.01)
        along, _ = self._reach(problem, tmp_path)
        assert along == pytest.approx(0.01 * self._cell(problem), rel=1e-9, abs=0.0)

    def test_a_clearance_of_nothing_hands_the_cap_over_as_it_was_drawn(self, fake_engine, tmp_path):
        """Which is the case that prices the displacement against not making it,
        and the one where openEMS may decide the wall is not there at all."""
        problem = self._capped(pinned_clearance=0.0)
        along, across = self._reach(problem, tmp_path)
        assert along == pytest.approx(0.0, rel=0.0, abs=0.0), (
            "a flat face was displaced on a run that asked for no clearance"
        )
        assert across == pytest.approx(
            staircase.GROWN_BY * self._cell(problem), rel=1e-9, abs=0.0
        ), "the curved wall stopped being grown, and the clearance is a separate correction"

    def test_and_the_two_corrections_are_asked_for_separately(self, fake_engine, tmp_path):
        """A share of nothing is about where a sampled boundary lands and says
        nothing about whether one is built, so it leaves the caps displaced."""
        problem = self._capped(grown_by=0.0)
        along, across = self._reach(problem, tmp_path)
        assert along == pytest.approx(
            staircase.PINNED_CLEARANCE * self._cell(problem), rel=1e-9, abs=0.0
        )
        assert across == pytest.approx(0.0, rel=0.0, abs=1e-15)

    def test_every_route_that_asks_what_conducts_asks_the_envelope(
        self, fake_engine, tmp_path, monkeypatch
    ):
        """Each of these routes asks where a conductor's metal is - what the
        driver builds, whether a port's plane stands in it, whether the grid
        still holds it whole - and an answer about a conductor the run will not
        build is worse than none: it names a fault in a shape nobody solved, or
        passes one that is there. That is what
        :meth:`~Microwave.Solvers.openems.model.Problem.as_given` is for, and a
        route reaching past it is what this keeps closed.

        Asserted over the calls rather than over a shape, because a route that
        stopped asking would still answer plausibly about the drawing, and a
        fixture where the two surfaces differ enough to notice is built around
        one route's arithmetic.
        """
        from Microwave.Solvers.openems import conductors, preflight

        problem = self._standing_on_a_cap(pinned_clearance=0.01, grown_by=0.2)
        asked = []
        answered = Problem.as_given

        def recording(self, solid):
            asked.append(solid.name)
            return answered(self, solid)

        monkeypatch.setattr(Problem, "as_given", recording)
        _build(problem, tmp_path)
        driver_asked = list(asked)
        preflight.ports._check_the_element_meets_its_metal(problem.ports[0], problem)
        ports_asked = asked[len(driver_asked) :]
        conductors.check(problem)
        conductors_asked = asked[len(driver_asked) + len(ports_asked) :]

        metal = problem.solids[0].name
        assert metal in driver_asked, "the driver stopped asking, and builds its own surface"
        assert metal in ports_asked, "a port's plane is checked against the drawing"
        assert metal in conductors_asked, "connectivity is checked against the drawing"
