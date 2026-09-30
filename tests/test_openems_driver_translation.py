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

from Microwave.Solvers.openems import excitation, plan, portreading, staircase
from Microwave.Solvers.openems.model import (
    THROUGH,
    EnvelopeError,
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


class FakePiece:
    """What a CSXCAD property says of one primitive it holds: every one used."""

    def __init__(self, number: int):
        self.number = number

    def GetID(self):
        return self.number

    def GetPrimitiveUsed(self):
        return True

    def GetBoundBox(self):
        return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    def GetTypeName(self):
        return "Box"

    def GetPriority(self):
        return 0


class FakeProperty:
    """A CSXCAD property. Records the primitives added to it."""

    def __init__(self, kind: str, name: str, **kw):
        self.kind = kind
        self.name = name
        self.kw = kw
        self.boxes: list[tuple] = []
        self.polyhedra: list[FakePolyhedron] = []
        self.pieces: list[FakePiece] = []

    def AddBox(self, start, stop, priority=0, **kw):
        self.boxes.append((tuple(start), tuple(stop), priority))
        self.pieces.append(FakePiece(len(self.pieces)))

    def AddPolyhedron(self, priority=0, **kw):
        self.polyhedra.append(FakePolyhedron(priority))
        self.pieces.append(FakePiece(len(self.pieces)))
        return self.polyhedra[-1]

    def GetName(self):
        return self.name

    def GetQtyPrimitives(self):
        return len(self.pieces)

    def GetPrimitive(self, index):
        return self.pieces[index]

    def GetAllPrimitives(self):
        return list(self.pieces)


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

    def GetAllProperties(self):
        return list(self.properties)

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

    def GetPropertyByCoordPriority(self, coord, prop_type=None, markFoundAsUsed=False):
        """The metal or material of highest priority whose box holds the point,
        faces included, as a property answering ``GetType`` and ``GetName``."""
        kinds = {
            "metal": PROPERTY_TYPE.METAL,
            "material": PROPERTY_TYPE.MATERIAL,
            "conducting_sheet": PROPERTY_TYPE.CONDUCTINGSHEET | PROPERTY_TYPE.METAL,
        }
        found = None
        for prop in self.properties:
            if prop.kind not in kinds:
                continue
            for start, stop, priority in prop.boxes:
                inside = all(min(a, b) <= c <= max(a, b) for a, b, c in zip(start, stop, coord))
                if inside and (found is None or priority > found[0]):
                    found = (priority, prop)
        if found is None:
            return None
        prop = found[1]
        return types.SimpleNamespace(GetType=lambda: kinds[prop.kind], GetName=lambda: prop.name)


#: ``CSXCAD.CSProperties.PropertyType``, with the values the binding carries.
PROPERTY_TYPE = types.SimpleNamespace(MATERIAL=2, METAL=4, CONDUCTINGSHEET=16384)


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
    #: Samples in each record, one a timestep.
    SAMPLES = 4096

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
        times = np.linspace(0.0, self.RECORD_SECONDS, self.SAMPLES)
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
        # What each probe file held, as ``UI_data`` keeps it.
        number = self.recorded.get("port_nr")
        self.u_data.fns = [f"port_ut{number}{letter}" for letter in "ABC"]
        self.u_data.ui_val = [self.ut_tot.copy() for _ in self.u_data.fns]
        self.i_data.fns = [f"port_it{number}"]
        self.i_data.ui_val = [self.it_tot.copy()]
        self.Z_ref = np.full(points, self.REFERENCE)
        self.uf_inc = np.full(points, self.INCIDENT, dtype=complex)
        self.uf_ref = np.full(points, 0.1 * self.INCIDENT, dtype=complex)
        self.uf_tot = self.uf_inc + self.uf_ref
        self.if_tot = (self.uf_inc - self.uf_ref) / self.Z_ref
        self.P_inc = np.ones(points)
        self.P_ref = np.full(points, 0.01)


#: The fake engine's timestep, which its current samples trail its voltage
#: samples by half of.
FAKE_TIMESTEP = FakePort.RECORD_SECONDS / (FakePort.SAMPLES - 1)


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
        # Where ``WaveguidePort`` has the engine write its two probes, after the
        # prefix ``Port`` puts ahead of every name it gives.
        prefix = kw.get("PortNamePrefix", "")
        port.U_filenames = [f"{prefix}port_ut_{port_nr}"]
        port.I_filenames = [f"{prefix}port_it_{port_nr}"]
        # ``RectWGPort``'s own expression, over a and b in metres.
        first, second = (float(digit) for digit in mode_name[2:])
        port.kc = np.sqrt((first * np.pi / a) ** 2 + (second * np.pi / b) ** 2)
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
    properties = types.ModuleType("CSXCAD.CSProperties")
    properties.PropertyType = PROPERTY_TYPE
    csxcad.CSProperties = properties
    engine = types.ModuleType("openEMS")
    engine.openEMS = FakeFDTD
    engine.__version__ = "fake"
    monkeypatch.setitem(sys.modules, "CSXCAD", csxcad)
    monkeypatch.setitem(sys.modules, "CSXCAD.CSProperties", properties)
    monkeypatch.setitem(sys.modules, "openEMS", engine)
    # The fake records one field at both planes of a waveguide port, with no
    # guide between them, so what the plane inside predicts at the port's plane
    # is the port's own reading of it. A test about the prediction puts the
    # real one back.
    monkeypatch.setattr(portreading, "predicted", _at_the_plane)
    return csxcad, engine


def _at_the_plane(here, there, x, length_unit, voltage, current):
    return here.waves(voltage, current)


#: The prediction the fake engine replaces, for the tests that hold it.
PREDICTED = portreading.predicted


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
    # Eight millimetres of air each side, in millimetres as every padding is.
    grid = plan.plan_grid(
        solids, ports, materials, params, ((THROUGH, THROUGH), (8.0, 8.0), (8.0, 8.0))
    )
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

    fdtd, csx, ports, _, _ = driver.build(problem, tmp_path)
    return fdtd, csx, ports


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

    def test_a_dielectrics_two_conductivities_reach_the_engine_as_one(self, fake_engine, tmp_path):
        """openEMS takes one electric conductivity for a material: the one a loss
        tangent became and the one the material states are added."""
        problem = _microstrip()
        lossy = tuple(
            replace(material, kind="lossy_dielectric", kappa=0.05, conductivity=0.02)
            if material.name == "FR4"
            else material
            for material in problem.materials
        )
        _, csx, _ = _build(replace(problem, materials=lossy), tmp_path)
        (fr4,) = [prop for prop in csx.properties if prop.name == "FR4"]
        assert fr4.kw["kappa"] == pytest.approx(0.07, abs=0.0)

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
        assert seen and set(seen) == {"port:3"}, "a port was built before all three axes existed"

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


class TestHowTheLossIsModelled:
    """What the record says the engine was given is what the engine was given."""

    def lossy(self, kappa, conductivity):
        problem = _microstrip()
        return replace(
            problem,
            materials=tuple(
                replace(material, kind="lossy_dielectric", kappa=kappa, conductivity=conductivity)
                if material.name == "FR4"
                else material
                for material in problem.materials
            ),
        )

    def recorded(self, problem, name):
        from Microwave.Solvers.openems import driver

        (record,) = [
            record
            for record in driver._provenance(problem, elapsed=1.5, records={})["modelled"]
            if record.get("material") == name
        ]
        return record

    def test_a_dielectric_holds_the_conductivity_the_engine_is_handed(self, fake_engine, tmp_path):
        problem = self.lossy(0.05, 0.02)
        _, csx, _ = _build(problem, tmp_path)
        (fr4,) = [prop for prop in csx.properties if prop.name == "FR4"]
        record = self.recorded(problem, "FR4")
        assert record["held"] == "conductivity"
        assert record["conductivity"] == fr4.kw["kappa"]

    def test_the_share_a_loss_tangent_became_is_stated_with_where_it_was_folded(self):
        problem = self.lossy(0.05, 0.02)
        record = self.recorded(problem, "FR4")
        assert record["folded"] == 0.05
        assert record["at"] == problem.frequency.center

    def test_a_conductivity_alone_folds_nothing(self):
        record = self.recorded(self.lossy(0.0, 0.02), "FR4")
        assert "folded" not in record and "at" not in record

    def test_a_sheet_is_one_plane_carrying_its_current(self, fake_engine, tmp_path):
        problem = _microstrip()
        _, csx, _ = _build(problem, tmp_path)
        (trace,) = [prop for prop in csx.properties if prop.kind == "conducting_sheet"]
        record = self.recorded(problem, "Trace")
        assert record["sheet"] == "net current"
        assert record["conductivity"] == trace.kw["conductivity"]
        assert record["thickness"] == COPPER_THICKNESS_MM

    def test_the_outside_is_recorded_with_what_it_holds_open_and_how_far(self):
        """The line runs out through X, and the domain reserves eight millimetres of
        air on the Y and Z faces the boundary absorbs at. ZMin is a perfect wall and
        is no part of the outside."""
        from Microwave.Solvers.openems import driver

        (record,) = [one for one in driver.modelled(_microstrip()) if "boundary" in one]
        assert record["boundary"] == "absorbing layer"
        assert record["faces"] == ["YMin", "YMax", "ZMax"]
        assert record["through"] == ["XMin", "XMax"]
        assert record["clearance"] == 8.0

    def test_a_domain_walled_on_every_face_records_no_outside(self):
        from Microwave.Solvers.openems import driver

        walled = replace(_microstrip(), boundary=("PEC",) * 6)
        assert not any("boundary" in one for one in driver.modelled(walled))

    def test_the_nearest_the_boundary_stood_is_what_is_recorded(self):
        """One clearance feeds every padded face today, and the record states the
        distance a comparison is made on, which is the smallest."""
        from Microwave.Solvers.openems import driver
        from Microwave.Solvers.openems.model import THROUGH

        problem = _microstrip()
        params = dict(problem.grid.params, padding=[[THROUGH, THROUGH], [3.0, 8.0], [8.0, 8.0]])
        near = replace(problem, grid=replace(problem.grid, params=params))
        (record,) = [one for one in driver.modelled(near) if "boundary" in one]
        assert record["clearance"] == 3.0

    def test_an_axis_absorbing_on_one_face_is_recorded_face_by_face(self):
        """The envelope holds a pair of counts for an axis whose faces differ,
        and the record states the depth of each face, which the result layer
        reads as it is stored."""
        from Microwave.Results import modelled
        from Microwave.Solvers.openems import driver

        problem = _microstrip()
        params = dict(problem.grid.params, pml_cells=[[8, 0], 8, [0, 8]])
        boundary = ("PML_8", "PEC", "PML_8", "PML_8", "PEC", "PML_8")
        asymmetric = replace(problem, boundary=boundary, grid=replace(problem.grid, params=params))
        (record,) = [one for one in driver.modelled(asymmetric) if "boundary" in one]
        assert record["cells"] == [8, 0, 8, 8, 0, 8]
        (line,) = modelled.said([record])
        assert "an absorbing layer 8/0/8/8/0/8 cells deep" in line

    def test_a_face_the_ports_open_is_recorded_as_absorbing(self):
        """An Ends face waveguide ports cover absorbs beyond the ports' plane, so
        a domain whose only absorbing faces are those has an outside."""
        from Microwave.Results import modelled
        from Microwave.Solvers.openems import driver

        problem = _microstrip()
        params = dict(
            problem.grid.params, padding=[[0, 0], [0, 0], [0, 0]], pml_cells=[[8, 0], 0, 0]
        )
        boundary = ("PML_8",) + ("PEC",) * 5
        guided = replace(problem, boundary=boundary, grid=replace(problem.grid, params=params))
        (record,) = [one for one in driver.modelled(guided) if "boundary" in one]
        assert record["ends"] == ["XMin"]
        assert record["faces"] == [] and record["through"] == []
        (line,) = modelled.said([record])
        assert "beyond the waveguide ports covering XMin" in line

    def test_what_every_backend_models_alike_is_not_recorded(self):
        from Microwave.Solvers.openems import driver

        named = {
            record["material"] for record in driver.modelled(_microstrip()) if "material" in record
        }
        assert named == {"Trace"}


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

    def test_a_body_openems_gave_no_cell_is_named_beside_the_answer(
        self, fake_engine, tmp_path, monkeypatch, capsys
    ):
        from Microwave.Solvers.openems import driver

        monkeypatch.setattr(FakePiece, "GetPrimitiveUsed", lambda piece: False)
        problem = _microstrip()
        assert driver.solve(problem, tmp_path).is_file()
        said = [
            line
            for line in capsys.readouterr().out.splitlines()
            if line.startswith(f"{driver.MARKER}CHECK")
        ]
        assert any(f" {problem.solids[0].label}: openEMS gave this " in line for line in said)

    def test_a_run_openems_could_not_set_up_is_refused(self, fake_engine, tmp_path, monkeypatch):
        """``openEMS.Run`` prints the code ``SetupFDTD`` refused with and returns
        it, solving nothing."""
        from Microwave.Solvers.openems import driver

        monkeypatch.setattr(FakeFDTD, "Run", lambda self, *args, **kw: 2)
        with pytest.raises(driver.SetupFailed, match="returned code 2"):
            driver.solve(_microstrip(), tmp_path)
        assert not (tmp_path / driver.RESULTS_NAME).exists()

    def test_every_port_is_weighed_and_not_only_the_driven_one(self, fake_engine, tmp_path):
        """A passive port watches a different part of the device, and on a
        filter it is the one still ringing."""
        provenance = self.solved(tmp_path, _two_port())["provenance"]
        assert sorted(provenance["tail_share"]) == ["1", "2"]
        assert sorted(provenance["recorded_samples"]) == ["1", "2"]

    def test_a_line_states_no_choice_of_impedance(self, fake_engine, tmp_path):
        """A guide has several impedances and says which it states. A
        microstrip's is the one it measures, and there is nothing to choose."""
        from Microwave.Results.sparameters import IMPEDANCE_STATED

        assert IMPEDANCE_STATED not in self.solved(tmp_path, _two_port())["provenance"]

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


class TestANumberEveryWaveNeedsThatIsNotANumberRefusesTheRun:
    """openEMS sums every sample of a record at every frequency point, and
    every wave at a port is normalised by the port's impedance. Where either is
    not a number the port's column holds none anywhere, and the run says which
    before a results file exists."""

    @staticmethod
    def poisoned(monkeypatch, number, change):
        """The fake, with ``change`` applied to port ``number`` after its
        ``CalcPort``."""
        calculate = FakePort.CalcPort

        def calculated(self, sim_path, freq):
            calculate(self, sim_path, freq)
            if self.recorded.get("port_nr") == number:
                change(self)

        monkeypatch.setattr(FakePort, "CalcPort", calculated)

    @pytest.mark.parametrize("number", [1, 2])
    @pytest.mark.parametrize("index", [0, 2048, -1])
    @pytest.mark.parametrize(
        ("data", "probe", "quantity"),
        [("u_data", 0, "voltage"), ("u_data", 2, "voltage"), ("i_data", 0, "current")],
    )
    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_a_probe_holding_a_sample_that_is_not_a_number(
        self, fake_engine, tmp_path, monkeypatch, capsys, number, index, data, probe, quantity, bad
    ):
        """At the driven port and at the other, first sample, middle and last,
        in a probe a microstrip port takes its impedance from as well as in the
        one it takes its waves from, on the axis that probe recorded. It is not
        weighed as a run cut short, which a longer record would not change."""
        from Microwave.Solvers.openems import driver

        def change(port):
            held = getattr(port, data)
            held.ui_val[probe][index] = bad
            self.named = held.fns[probe]
            self.at = float(held.ui_time[probe][index])

        self.poisoned(monkeypatch, number, change)
        with pytest.raises(driver.NotANumber) as raised:
            driver.solve(_two_port(), tmp_path)

        assert str(raised.value).startswith(
            f"port {number}'s {quantity} probe {self.named} recorded a value that is not "
            f"a number at 1 of 4096 samples, the first at {self.at * 1e9:.6g} ns."
        )
        assert not (tmp_path / driver.RESULTS_NAME).exists()
        assert "max_timesteps" not in capsys.readouterr().out

    def test_the_first_sample_of_several_is_named(self, fake_engine, tmp_path, monkeypatch):
        from Microwave.Solvers.openems import driver

        def change(port):
            held = port.u_data.ui_val
            held[1] = np.where(np.arange(held[1].size) >= 3000, math.nan, held[1])
            self.at = float(port.u_data.ui_time[1][3000])

        self.poisoned(monkeypatch, 2, change)
        with pytest.raises(driver.NotANumber) as raised:
            driver.solve(_two_port(), tmp_path)
        assert f"at 1096 of 4096 samples, the first at {self.at * 1e9:.6g} ns" in str(raised.value)

    def test_a_microstrip_port_with_no_impedance_is_named_with_no_cause(
        self, fake_engine, tmp_path, monkeypatch
    ):
        """Nothing established puts a nan or an infinity there in a run a
        microstrip port drives, so the message states what the port held and
        guesses nothing."""
        from Microwave.Solvers.openems import driver

        def change(port):
            where = np.arange(port.Z_ref.size) % 20
            port.Z_ref = np.where(where == 3, math.nan, np.where(where == 13, math.inf, port.Z_ref))

        self.poisoned(monkeypatch, 2, change)
        with pytest.raises(driver.NotANumber) as raised:
            driver.solve(_two_port(), tmp_path)

        assert str(raised.value) == (
            "port 2 states no finite reference impedance at 5 frequency points from "
            "1.54 to 8.74 GHz, so no S-parameter there can be normalised"
        )
        assert not (tmp_path / driver.RESULTS_NAME).exists()

    def test_in_a_run_a_lumped_port_drives_it_says_so(self, fake_engine, tmp_path, monkeypatch):
        """``MSLPort`` states ``nan`` at every point in a run a lumped port
        drives, and the pre-flight warns of it before the solve."""
        from Microwave.Solvers.openems import driver

        line = _two_port()
        feed = replace(line.ports[0], kind="lumped", feed_resistance=50.0)
        problem = replace(line, ports=(feed, line.ports[1]))

        def change(port):
            port.Z_ref = np.full(port.Z_ref.size, math.nan)

        self.poisoned(monkeypatch, 2, change)
        with pytest.raises(driver.NotANumber) as raised:
            driver.solve(problem, tmp_path)

        said = str(raised.value)
        assert said.startswith("port 2 states no finite reference impedance at 51 frequency")
        assert "a lumped port drives states none at any point" in said

    def test_but_not_where_it_states_one_at_some_points(self, fake_engine, tmp_path, monkeypatch):
        """What was measured is none at every point, so a port missing some is
        not put down to the lumped port."""
        from Microwave.Solvers.openems import driver

        line = _two_port()
        feed = replace(line.ports[0], kind="lumped", feed_resistance=50.0)
        problem = replace(line, ports=(feed, line.ports[1]))

        def change(port):
            port.Z_ref = np.where(np.arange(port.Z_ref.size) == 7, math.nan, port.Z_ref)

        self.poisoned(monkeypatch, 2, change)
        with pytest.raises(driver.NotANumber) as raised:
            driver.solve(problem, tmp_path)
        assert "lumped" not in str(raised.value)

    def test_an_undriven_lumped_port_is_no_such_run(self, fake_engine, tmp_path, monkeypatch):
        from Microwave.Solvers.openems import driver

        line = _two_port()
        far = replace(line.ports[1], kind="lumped", feed_resistance=50.0)
        problem = replace(line, ports=(line.ports[0], far))

        def change(port):
            port.Z_ref = np.full(port.Z_ref.size, math.nan)

        self.poisoned(monkeypatch, 1, change)
        with pytest.raises(driver.NotANumber) as raised:
            driver.solve(problem, tmp_path)
        assert "lumped" not in str(raised.value)


class TestALumpedPortIsReferencedToItsResistance:
    """openEMS references a lumped port's waves to its resistance when the port
    is asked for no reference of its own (``openEMS/python/openEMS/ports.py:240-242``).
    The driver asks for none, writes what the port states, and a matrix
    referenced to each port's own impedance is then referenced to the
    resistance. The fake port takes its reference as openEMS' lumped port does,
    at a resistance other than the one every other fake port states."""

    RESISTANCE = 25.0

    def test_the_result_states_the_resistance_as_the_reference(
        self, fake_engine, tmp_path, monkeypatch
    ):
        from Microwave.Results.sparameters import SParameters
        from Microwave.Solvers.openems import driver, read

        assert self.RESISTANCE != FakePort.REFERENCE
        calculate = FakePort.CalcPort

        def calculated(self, sim_path, freq, ref_impedance=None, ref_plane_shift=None):
            calculate(self, sim_path, freq)
            if self.kind == "lumped" and ref_impedance is None:
                self.Z_ref = np.full(self.Z_ref.size, self.recorded["R"])
                self.if_tot = (self.uf_inc - self.uf_ref) / self.Z_ref

        monkeypatch.setattr(FakePort, "CalcPort", calculated)
        line = _two_port()
        feed = replace(line.ports[0], kind="lumped", feed_resistance=self.RESISTANCE)
        driver.solve(replace(line, ports=(feed, line.ports[1])), tmp_path)

        solved = read.read(tmp_path)
        assert np.all(solved.port(1).z0 == self.RESISTANCE)
        matrix = SParameters.from_runs([solved], reference=None)
        assert np.all(np.asarray(matrix.reference)[:, 0] == self.RESISTANCE)


class TestAWaveguidePortIsReadAtItsOwnScales:
    """``_extract`` takes a ``RectWGPort``'s waves apart again at the scales
    :mod:`.portreading` gives them, from the engine's own answers: the nodes in
    the probe files' headers, and the metal the structure holds after the run."""

    def headers(self, problem, sim_dir, numbers=(1,), inside=()):
        """What the engine writes at the top of each port's two probe files, for
        a box over the whole cross-section, excluded at the domain's lines; and
        for the ports in ``inside``, of the probes reading it again in its box."""
        from Microwave.Solvers.openems import driver, nearfield

        placed, _ = problem.at_the_origin()
        sim_dir.mkdir(parents=True, exist_ok=True)
        for port in placed.ports:
            axis = port.propagation_axis
            lines = np.asarray(placed.grid[axis])
            planes = []
            if port.number in numbers:
                planes.append(("", port.stop[axis]))
            if port.number in inside:
                where = nearfield.inside(lines, port.start[axis], port.stop[axis])
                assert where is not None
                for prefix, plane in zip(driver.INSIDE_PREFIXES, where, strict=True):
                    if plane is not None:
                        planes.append((prefix, plane[0]))
            for prefix, at in planes:
                plane = int(np.argmin(np.abs(lines - at)))
                start, stop = [plane] * 3, [plane] * 3
                for dim in port.mode_axes:
                    start[dim], stop[dim] = 1, len(placed.grid[dim]) - 2
                for probe, quantity in (("ut", "voltage"), ("it", "current")):
                    (sim_dir / f"{prefix}port_{probe}_{port.number}").write_text(
                        f"% time-domain {quantity} mode matching by openEMS\n"
                        f"% start-coordinates: (0,0,0) m -> [{start[0]},{start[1]},{start[2]}]\n"
                        f"% stop-coordinates: (0,0,0) m -> [{stop[0]},{stop[1]},{stop[2]}]\n"
                        f"% t/s\t{quantity}\tmode_purity\n"
                    )

    @staticmethod
    def guide(second=False, fill=(1.0, 1.0), referred_at_the_probes=True):
        """The fake's waveguide in its dominant mode, which carries power across
        the whole band, with a second port facing the first where asked.

        Each port is referred to the plane its probes read unless asked
        otherwise, so the waves the run states are the ones the port read."""
        guide = _waveguide()
        first = replace(guide.ports[0], mode="TE10")
        if referred_at_the_probes:
            first = replace(first, reference_depth=first.length)
        ports = (first,)
        if second:
            ports += (
                replace(
                    first, number=2, start=(0.0, 0.0, 12.0), stop=(10.7, 4.3, 10.0), excite=False
                ),
            )
        air = replace(guide.materials[0], epsilon=fill[0], mu=fill[1])
        return replace(guide, ports=ports, materials=(air,))

    def solved(self, tmp_path, headers=(1,), problem=None, inside=None):
        """Solved on the fake, with the probe headers written for the ports in
        ``headers``, and for the planes inside the boxes of those in ``inside``,
        which are the same ports unless it says otherwise."""
        from Microwave.Solvers.openems import driver

        problem = problem or self.guide()
        inside = headers if inside is None else inside
        self.headers(problem, tmp_path / driver.RUN_DIRNAME, headers, inside)
        return json.loads(driver.solve(problem, tmp_path).read_text())

    def test_its_waves_are_taken_apart_again_and_openems_own_kept_beside_them(
        self, fake_engine, tmp_path, capsys
    ):
        results = self.solved(tmp_path)
        port = results["ports"]["1"]
        raw = port["raw"]["incident"]["re"][0] + 1j * port["raw"]["incident"]["im"][0]
        incident = port["incident"]["re"][0] + 1j * port["incident"]["im"][0]
        reflected = port["reflected"]["re"][0] + 1j * port["reflected"]["im"][0]
        s11 = results["s_parameters"]["S11"]["re"][0] + 1j * results["s_parameters"]["S11"]["im"][0]

        assert raw == FakePort.INCIDENT
        # The waves read back are the ones that give what the port recorded.
        reading = results["provenance"]["port_reading"]["1"]
        recorded = FakePort.INCIDENT * 1.1
        assert reading["voltage_scale"] * (incident + reflected) == pytest.approx(
            recorded, rel=1e-12, abs=0.0
        )
        assert s11 == pytest.approx(reflected / incident, rel=1e-12, abs=0.0)
        # They are referred to the mode impedance of the guide the grid holds,
        # rather than to the one openEMS split them at: the grid's wavenumber
        # at the fake's timestep and its cutoff on the lines across, over
        # cos(theta) of the half cell into the guide, where the cells either
        # side of the plane here are alike.
        z0 = port["z0"]["re"][0] + 1j * port["z0"]["im"][0]
        k = 2 / FAKE_TIMESTEP * math.sin(math.pi * 20e9 * FAKE_TIMESTEP) / 299_792_458.0
        across = np.asarray(self.guide().at_the_origin()[0].grid.x) * 1e-3
        beta = math.sqrt(k**2 - portreading._mode(across, 1)[0] ** 2)
        z = _waveguide().grid.z
        plane = int(np.argmin(np.abs(z - 6.0)))
        cells = np.diff(z[plane - 2 : plane + 3])
        assert np.allclose(cells, cells[0], rtol=1e-9, atol=0.0)
        carried = math.cos(math.asin(beta * cells[0] * 1e-3 / 2))
        assert z0 == pytest.approx(k * 4e-7 * math.pi * 299_792_458.0 / beta / carried, rel=1e-12)
        # The powers are the ones those waves carry, as Port.CalcPort states
        # them, and not the ones openEMS computed from its own split.
        carried = 0.5 * np.real(incident * np.conj(incident / z0))
        assert port["power_incident"][0] == pytest.approx(carried, rel=1e-12, abs=0.0)
        assert port["power_reflected"][0] / port["power_incident"][0] == pytest.approx(
            abs(reflected / incident) ** 2, rel=1e-12, abs=0.0
        )
        assert reading["corrected"] is True
        # A file carrying that number says which of the guide's impedances it
        # is, since renormalising the matrix depends on it.
        from Microwave.Results.sparameters import IMPEDANCE_STATED
        from Microwave.Solvers.openems import driver

        assert results["provenance"][IMPEDANCE_STATED] == {"1": driver.READ_IMPEDANCE}
        # The domain is the guide and its four sides are PEC.
        assert reading["guide"] == [pytest.approx(10.7), pytest.approx(4.3)]
        assert "CHECK" not in capsys.readouterr().out

    def test_one_port_left_as_read_leaves_every_other_as_read_too(
        self, fake_engine, tmp_path, capsys
    ):
        """Two ports the engine read alike have a right ratio between them, and
        correcting one of them alone would break it."""
        results = self.solved(tmp_path, headers=(1,), problem=self.guide(second=True))
        readings = results["provenance"]["port_reading"]

        assert "raw" not in results["ports"]["1"] and "raw" not in results["ports"]["2"]
        assert readings["2"]["corrected"] is False and "cannot be read" in readings["2"]["reason"]
        assert readings["1"] == {
            "corrected": False,
            "reason": "port 2 of this run could not be read",
        }
        # Each then states the impedance openEMS split it at.
        from Microwave.Results.sparameters import IMPEDANCE_STATED
        from Microwave.Solvers.openems import driver

        assert results["provenance"][IMPEDANCE_STATED] == {
            "1": driver.WAVE_IMPEDANCE,
            "2": driver.WAVE_IMPEDANCE,
        }
        checks = [line for line in capsys.readouterr().out.splitlines() if "CHECK" in line]
        assert len(checks) == 1 and "port 2:" in checks[0]

    def test_a_port_left_as_read_states_its_impedance_below_cutoff(
        self, fake_engine, tmp_path, monkeypatch
    ):
        """openEMS takes ``beta`` as a real square root, ``nan`` below cutoff,
        and so is every wave it splits there; handed on so, the assembly refused
        the whole run over the points it blanks. Below cutoff the port states
        ``k Z0 / beta`` with ``beta`` imaginary and splits its waves at it, and
        every other point is openEMS' own."""
        cutoff = 299_792_458.0 / (2 * 10.7e-3)
        calculate = FakePort.CalcPort

        def as_openems(self, sim_path, freq):
            calculate(self, sim_path, freq)
            if self.kind == "waveguide":
                below = np.asarray(freq) < cutoff
                for name in ("Z_ref", "uf_inc", "uf_ref"):
                    values = np.array(getattr(self, name), dtype=complex)
                    values[below] = np.nan
                    setattr(self, name, values)

        monkeypatch.setattr(FakePort, "CalcPort", as_openems)
        problem = replace(self.guide(), frequency=Frequency(12e9, 20e9, 41))
        results = self.solved(tmp_path, headers=(), problem=problem)
        assert results["provenance"]["port_reading"]["1"]["corrected"] is False

        port = results["ports"]["1"]
        pair = {
            key: np.array(port[key]["re"]) + 1j * np.array(port[key]["im"])
            for key in ("z0", "incident", "reflected")
        }
        frequency = np.asarray(results["frequency"])
        below = frequency < cutoff
        assert below.any() and not below.all()
        k = 2 * np.pi * frequency / 299_792_458.0
        beta = np.sqrt(k**2 - (np.pi / 10.7e-3) ** 2 + 0j)
        stated = k * 4e-7 * np.pi * 299_792_458.0 / beta
        assert np.all(pair["z0"][below].real == 0.0)
        np.testing.assert_allclose(pair["z0"][below], stated[below], rtol=1e-12, atol=0.0)
        total = FakePort.INCIDENT * 1.1
        current = 0.9 * FakePort.INCIDENT / FakePort.REFERENCE
        np.testing.assert_allclose(
            pair["incident"][below], 0.5 * (total + current * stated[below]), rtol=1e-12, atol=0.0
        )
        np.testing.assert_allclose(
            pair["incident"][below] + pair["reflected"][below], total, rtol=1e-12, atol=0.0
        )
        assert np.all(pair["z0"][~below] == FakePort.REFERENCE)
        assert np.all(pair["incident"][~below] == FakePort.INCIDENT)
        assert np.all(pair["reflected"][~below] == 0.1 * FakePort.INCIDENT)
        assert port["power_incident"][-1] == 1.0

    @staticmethod
    def on_the_cutoff(wavenumber, cutoff, near=None):
        """The frequency whose wavenumber squared is the cutoff's, to the bit,
        searched for from ``near``, the continuous guide's where not given."""
        near = cutoff * 299_792_458.0 / (2 * np.pi) if near is None else near
        for direction in (np.inf, -np.inf):
            found = near
            for _ in range(1000):
                if wavenumber(found) ** 2 - cutoff**2 == 0.0:
                    return found
                found = np.nextafter(found, direction)
        raise AssertionError("no frequency lies on the cutoff to the bit")

    def test_a_point_exactly_on_the_cutoff_is_refused_by_name(
        self, fake_engine, tmp_path, monkeypatch
    ):
        """openEMS' ``beta`` is zero there and its impedance has no bound, and
        the port's own ``kc`` puts the point on the cutoff here as well. The
        next point above it is answered."""
        from Microwave.Solvers.openems import driver

        problem = self.guide()
        a, b, mode = problem.ports[0].waveguide_arguments(problem.length_unit)
        cutoff = np.sqrt((float(mode[2]) * np.pi / a) ** 2 + (float(mode[3]) * np.pi / b) ** 2)

        def wavenumber(frequency):
            return 2.0 * np.pi * frequency / 299_792_458.0

        calculate = FakePort.CalcPort

        def as_openems(port, sim_path, freq):
            calculate(port, sim_path, freq)
            if port.kind == "waveguide":
                k = wavenumber(np.asarray(freq))
                with np.errstate(divide="ignore", invalid="ignore"):
                    port.Z_ref = k * 4e-7 * np.pi * 299_792_458.0 / np.sqrt(k**2 - port.kc**2)
                unset = ~np.isfinite(port.Z_ref)
                for name in ("uf_inc", "uf_ref"):
                    values = np.array(getattr(port, name), dtype=complex)
                    values[unset] = np.nan
                    setattr(port, name, values)

        monkeypatch.setattr(FakePort, "CalcPort", as_openems)
        on = self.on_the_cutoff(wavenumber, cutoff)
        with pytest.raises(driver.NotANumber) as raised:
            self.solved(
                tmp_path, headers=(), problem=replace(problem, frequency=Frequency(on, 20e9, 41))
            )
        assert str(raised.value) == (
            f"port 1 states no finite reference impedance at {on / 1e9:.9g} GHz, so no "
            "S-parameter there can be normalised. That point lies exactly on its TE10 "
            "cutoff, where beta is zero and the impedance has no bound. Move the band so "
            "no point falls on it"
        )

        above = np.nextafter(on, np.inf)
        while wavenumber(above) ** 2 - cutoff**2 <= 0.0:
            above = np.nextafter(above, np.inf)
        results = self.solved(
            tmp_path / "above",
            headers=(),
            problem=replace(problem, frequency=Frequency(above, 20e9, 41)),
        )
        z0 = results["ports"]["1"]["z0"]
        assert np.all(np.isfinite(z0["re"])) and z0["re"][0] > 0.0

    def test_and_so_is_one_on_the_cutoff_of_a_port_read_at_its_own_scales(
        self, fake_engine, tmp_path
    ):
        from Microwave import units
        from Microwave.Solvers.openems import driver

        problem = self.guide()
        cutoff = portreading._mode(np.asarray(problem.at_the_origin()[0].grid.x) * 1e-3, 1)[0]
        step = FAKE_TIMESTEP

        def wavenumber(frequency):
            return 2 / step * np.sin(math.pi * frequency * step) / units.SPEED_OF_LIGHT

        near = math.asin(cutoff * units.SPEED_OF_LIGHT * step / 2) / (math.pi * step)
        on = self.on_the_cutoff(wavenumber, cutoff, near)
        with pytest.raises(driver.NotANumber) as raised:
            self.solved(tmp_path, problem=replace(problem, frequency=Frequency(on, 20e9, 41)))
        assert str(raised.value).startswith(
            f"port 1 states no finite reference impedance at {on / 1e9:.9g} GHz"
        )
        assert "exactly on the cutoff of the guide the grid holds" in str(raised.value)

    def test_every_port_left_as_read_names_the_one_that_could_not_be(self, fake_engine, tmp_path):
        guide = self.guide(second=True)
        third = replace(guide.ports[0], number=3, start=(0.0, 0.0, 20.0), stop=(10.7, 4.3, 22.0))
        problem = replace(guide, ports=(*guide.ports, replace(third, excite=False)))
        readings = self.solved(tmp_path, headers=(1, 2), problem=problem)["provenance"]
        reasons = {
            number: reading["reason"] for number, reading in readings["port_reading"].items()
        }
        assert reasons["1"] == reasons["2"] == "port 3 of this run could not be read"

    @pytest.mark.parametrize("fill", [(4.0, 1.0), (1.0, 2.0)], ids=["dielectric", "magnetic"])
    def test_a_guide_holding_a_material_at_the_plane_is_left_as_read(
        self, fake_engine, tmp_path, fill
    ):
        """The mode read is the mode of an empty guide."""
        results = self.solved(tmp_path, problem=self.guide(fill=fill))
        reading = results["provenance"]["port_reading"]["1"]
        assert reading["corrected"] is False and "'Air' lies" in reading["reason"]

    def test_a_conducting_sheet_is_not_answered_as_a_tied_wall(self, fake_engine):
        """The engine models a sheet as a surface and ties none of its edges, so
        a guide walled by one is not a guide the reading describes."""
        from Microwave.Solvers.openems import driver, portreading

        guide = _waveguide()
        sheet = Material(name="Foil", kind="conducting_sheet", conductivity=5.8e7, thickness=0.035)
        problem = replace(guide, materials=(*guide.materials, sheet))
        csx = FakeCSX()
        csx.AddConductingSheet("Foil").AddBox((0.0, 2.0, 0.0), (10.7, 2.0, 50.0), priority=20)
        conducts = driver._conducts(csx, problem)
        with pytest.raises(portreading.NotRead, match="'Foil' lies"):
            conducts((5.0, 2.0, 6.0))

    @staticmethod
    def recorded(voltage_at: float, current_at: float):
        """A port whose probe files are the fake's, and whose first samples
        were taken at these times."""
        return types.SimpleNamespace(
            U_filenames=["port_ut_1"],
            I_filenames=["port_it_1"],
            u_data=types.SimpleNamespace(ui_time=[np.array([voltage_at, voltage_at + 1e-11])]),
            i_data=types.SimpleNamespace(ui_time=[np.array([current_at, current_at + 1e-11])]),
        )

    @pytest.mark.parametrize("step", [5e-13, 2e-12])
    def test_the_engines_timestep_is_twice_the_currents_first_delay(
        self, fake_engine, tmp_path, step
    ):
        """The engine samples the magnetic field half a step after the
        electric, and only the first sample of each is read. The waves are
        referred to the grid's wavenumber at that step."""
        from Microwave import units
        from Microwave.Solvers.openems import driver

        problem = self.guide()
        self.headers(problem, tmp_path)
        frequency = problem.frequency.values()
        reading = driver._read_waveguide(
            problem, problem.ports[0], self.recorded(0.0, step / 2), FakeCSX(), tmp_path, frequency
        )
        k = 2 / step * np.sin(np.pi * frequency * step) / units.SPEED_OF_LIGHT
        beta = np.sqrt(k**2 - portreading._mode(np.asarray(problem.grid.x) * 1e-3, 1)[0] ** 2)
        z = np.asarray(problem.grid.z)
        plane = int(np.argmin(np.abs(z - 6.0)))
        cell = (z[plane + 1] - z[plane]) * 1e-3
        share = np.cos(np.arcsin(beta * cell / 2))
        expected = k * portreading.FREE_SPACE_IMPEDANCE / beta / share
        np.testing.assert_allclose(reading.impedance, expected, rtol=1e-12, atol=0.0)

    @pytest.mark.parametrize("current_at", [0.0, -1e-13, math.nan])
    def test_records_that_give_no_timestep_leave_the_port_unread(
        self, fake_engine, tmp_path, current_at
    ):
        from Microwave.Solvers.openems import driver

        problem = self.guide()
        self.headers(problem, tmp_path)
        with pytest.raises(portreading.NotRead, match="timestep"):
            driver._read_waveguide(
                problem,
                problem.ports[0],
                self.recorded(0.0, current_at),
                FakeCSX(),
                tmp_path,
                problem.frequency.values(),
            )

    def test_the_engine_is_asked_about_each_face_of_the_domain_by_its_own_boundary(
        self, fake_engine
    ):
        """A PEC face ties the field lying in it, and an absorbing one does not,
        whichever end of the axis it is."""
        from Microwave.Solvers.openems import driver

        guide = _waveguide()
        problem = replace(guide, boundary=("PEC", "PML_8", "PML_8", "PEC", "PML_8", "PML_8"))
        conducts = driver._conducts(FakeCSX(), problem)
        middle = [float(np.mean(problem.grid[dim])) for dim in range(3)]
        for dim in range(3):
            for end, index in ((0, 0), (1, -1)):
                point = list(middle)
                point[dim] = float(problem.grid[dim][index])
                assert conducts(tuple(point)) == (problem.boundary[2 * dim + end] == "PEC")
        assert not conducts(tuple(middle))

    def test_its_waves_are_moved_from_the_probes_to_its_reference_plane(
        self, fake_engine, tmp_path
    ):
        """A wave travelling in is ``exp(-j beta z)`` of itself ``z`` further on,
        so referred ``s`` before the probes it is ``exp(j beta s)`` of what they
        read, and the wave travelling out ``exp(-j beta s)`` of it, with ``beta``
        the guide's own ``sqrt(k^2 - (pi / a)^2)``. The fake's probes read ``2``
        in from the face."""
        from Microwave.Solvers.openems import driver

        at_probes = self.solved(tmp_path / "probes")
        at_face = self.solved(tmp_path / "face", problem=self.guide(referred_at_the_probes=False))
        frequency = np.asarray(at_face["frequency"])
        k = 2 * np.pi * frequency / 299_792_458.0
        beta = np.sqrt(k**2 - (np.pi / 10.7e-3) ** 2)
        turn = np.exp(1j * beta * 2e-3)

        def wave(results, key):
            port = results["ports"]["1"]
            return np.array(port[key]["re"]) + 1j * np.array(port[key]["im"])

        np.testing.assert_allclose(
            wave(at_face, "incident"), wave(at_probes, "incident") * turn, rtol=1e-9, atol=0.0
        )
        np.testing.assert_allclose(
            wave(at_face, "reflected"), wave(at_probes, "reflected") / turn, rtol=1e-9, atol=0.0
        )
        s11 = {
            name: np.array(one["s_parameters"]["S11"]["re"])
            + 1j * np.array(one["s_parameters"]["S11"]["im"])
            for name, one in (("face", at_face), ("probes", at_probes))
        }
        np.testing.assert_allclose(s11["face"], s11["probes"] / turn**2, rtol=1e-9, atol=0.0)
        assert at_face["provenance"][driver.REFERRED] == {"1": pytest.approx(2.0, abs=1e-12)}
        assert at_probes["provenance"][driver.REFERRED] == {"1": 0.0}

    def test_the_waves_are_moved_at_the_fill_the_engine_holds(self, fake_engine, tmp_path):
        """The propagation constant is the fill's, ``sqrt(k^2 eps - (pi / a)^2)``,
        asked of the structure the engine built."""
        fill = (2.0, 1.0)

        def incident(problem, where):
            port = self.solved(tmp_path / where, problem=problem)["ports"]["1"]
            return np.array(port["incident"]["re"]) + 1j * np.array(port["incident"]["im"])

        at_probes = incident(self.guide(fill=fill), "probes")
        at_face = incident(self.guide(fill=fill, referred_at_the_probes=False), "face")
        frequency = self.guide().frequency.values()
        k = 2 * np.pi * frequency / 299_792_458.0
        beta = np.sqrt(k**2 * fill[0] - (np.pi / 10.7e-3) ** 2)
        np.testing.assert_allclose(
            at_face, at_probes * np.exp(1j * beta * 2e-3), rtol=1e-9, atol=0.0
        )

    def test_the_power_a_moved_wave_carries_moves_with_its_magnitude(self):
        """Referred before the probes in a lossy guide, a wave travelling in
        carries more power and one travelling out less, each by the square of
        how much its magnitude moved."""
        from Microwave.Solvers.openems import driver

        entry = {
            "incident": {"re": [1.0], "im": [0.0]},
            "reflected": {"re": [0.5], "im": [0.0]},
            "power_incident": [2.0],
            "power_reflected": [0.5],
        }
        driver._refer(entry, np.array([2.0 * np.exp(0.3j)]), 1)
        assert entry["incident"]["re"][0] + 1j * entry["incident"]["im"][0] == pytest.approx(
            2.0 * np.exp(0.3j), rel=1e-12, abs=0.0
        )
        assert entry["reflected"]["re"][0] + 1j * entry["reflected"]["im"][0] == pytest.approx(
            0.25 * np.exp(-0.3j), rel=1e-12, abs=0.0
        )
        assert entry["power_incident"] == [pytest.approx(8.0, rel=1e-12, abs=0.0)]
        assert entry["power_reflected"] == [pytest.approx(0.125, rel=1e-12, abs=0.0)]

    def test_a_lossy_fill_moves_the_waves_by_its_attenuation_too(self):
        """Referred before the probes, a wave travelling in was larger by the
        loss it met on the way and the wave travelling out smaller; a wave below
        cutoff is the same arithmetic on a mode that decays."""
        from Microwave.Solvers.openems import driver

        guide = self.guide(referred_at_the_probes=False)
        placed, _ = guide.at_the_origin()
        port = placed.ports[0]
        lines = placed.grid[port.propagation_axis]
        frequency = np.array([10e9, 20e9, 26e9])
        lossless = driver.referred(port, lines, frequency, 1e-3)
        lossy = driver.referred(port, lines, frequency, 1e-3, (1.0, 1.0, 0.5))
        assert np.abs(lossless[1:]) == pytest.approx(1.0, abs=1e-12)
        assert np.all(np.abs(lossy) > 1.0)
        k = 2 * np.pi * frequency / 299_792_458.0
        decay = np.sqrt((np.pi / 10.7e-3) ** 2 - k[0] ** 2)
        assert lossless[0] == pytest.approx(np.exp(decay * 2e-3), rel=1e-12, abs=0.0)
        # A wave travelling in loses phase and power alike as it goes, so the
        # constant is beta - j alpha with both positive.
        permittivity = 1.0 - 1j * 0.5 / (2 * np.pi * frequency * 8.8541878128e-12)
        beta = np.sqrt(k**2 * permittivity - (np.pi / 10.7e-3) ** 2)
        assert np.all(beta.real > 0.0) and np.all(beta.imag < 0.0)
        np.testing.assert_allclose(lossy, np.exp(1j * beta * 2e-3), rtol=1e-9, atol=0.0)

    def test_a_port_facing_down_its_axis_is_moved_towards_its_own_face(self):
        """The distance is along the way the wave travels in, so a port facing
        down its axis is moved up it, back to its face."""
        from Microwave.Solvers.openems import driver

        guide = self.guide(second=True, referred_at_the_probes=False)
        placed, _ = guide.at_the_origin()
        facing_down = placed.ports[1]
        assert facing_down.stop[2] < facing_down.start[2]
        lines = placed.grid[2]
        assert driver._moved(facing_down, lines) == pytest.approx(2.0, abs=1e-12)
        assert driver._moved(replace(facing_down, reference_depth=3.0), lines) == pytest.approx(
            -1.0, abs=1e-12
        )

    def test_the_probes_are_read_on_the_line_nearest_the_box_s_far_face(self):
        """openEMS lays a probe box flat across a line, so a far face between
        two lines is read on the nearer one, and the wave is moved from there."""
        from Microwave.Solvers.openems import driver

        port = self.guide(referred_at_the_probes=False).ports[0]
        lines = np.array([0.0, 4.0, 5.2, 6.3, 8.0])
        assert port.probe_plane(lines) == 6.3
        assert driver._moved(port, lines) == pytest.approx(2.3, abs=1e-12)

    def test_the_fill_is_read_with_its_conductivity(self, fake_engine, tmp_path):
        """A lossy dielectric reaches the engine with its two conductivities
        added, and the wave is attenuated by both."""
        from Microwave.Solvers.openems import driver

        guide = self.guide()
        lossy = replace(guide.materials[0], kind="lossy_dielectric", kappa=0.25, conductivity=0.5)
        problem = replace(guide, materials=(lossy,))
        placed, _ = problem.at_the_origin()
        _, csx, _, _, _ = driver.build(placed, tmp_path)
        assert driver._fill(csx, placed, placed.ports[0]) == (1.0, 1.0, 0.75)

    def test_a_material_the_envelope_does_not_hold_is_read_as_vacuum(self, fake_engine):
        from Microwave.Solvers.openems import driver

        class Stranger:
            def GetPropertyByCoordPriority(self, coord, prop_type=None, markFoundAsUsed=False):
                class Found:
                    def GetName(self):
                        return "Unheard"

                return Found()

        placed, _ = self.guide(fill=(2.0, 1.0)).at_the_origin()
        assert driver._fill(Stranger(), placed, placed.ports[0]) == (1.0, 1.0, 0.0)

    def test_a_guide_the_engine_finds_nothing_in_is_vacuum(self, fake_engine):
        from Microwave.Solvers.openems import driver

        class Empty:
            def GetPropertyByCoordPriority(self, coord, prop_type=None, markFoundAsUsed=False):
                return None

        placed, _ = self.guide(fill=(2.0, 1.0)).at_the_origin()
        assert driver._fill(Empty(), placed, placed.ports[0]) == (1.0, 1.0, 0.0)

    def test_one_whose_probes_do_not_say_what_they_summed_is_left_as_read_and_named(
        self, fake_engine, tmp_path, capsys
    ):
        results = self.solved(tmp_path, headers=())
        port = results["ports"]["1"]

        assert port["incident"]["re"][0] == FakePort.INCIDENT
        assert "raw" not in port
        reading = results["provenance"]["port_reading"]["1"]
        assert reading["corrected"] is False and "cannot be read" in reading["reason"]
        checks = [line for line in capsys.readouterr().out.splitlines() if "CHECK" in line]
        assert len(checks) == 1
        assert "severity=warn port 1:" in checks[0] and "left as openEMS returned" in checks[0]


class TestAWaveguidePortIsReadAgainInsideItsBox:
    """Each waveguide port's mode is read again at two planes in its box, and how
    far the waves it read differ from those the planes predict is recorded and,
    past the bar, warned of by port."""

    reading = TestAWaveguidePortIsReadAtItsOwnScales()

    #: The fake's own, taken before any test replaces it.
    added = staticmethod(FakeFDTD.AddRectWaveGuidePort)

    def recording(self, monkeypatch, reflected, halfway=None):
        """The probes inside each box record the port's own incident wave, and
        its reflected wave times ``reflected``; those at the shallower plane
        times ``halfway`` where it is given."""
        from Microwave.Solvers.openems import driver

        added = self.added
        factors = dict(zip(driver.INSIDE_PREFIXES, (reflected, halfway or reflected), strict=True))

        def add(self, *args, **kw):
            port = added(self, *args, **kw)
            if kw.get("PortNamePrefix"):
                calculate = port.CalcPort
                factor = factors[kw["PortNamePrefix"]]

                def again(sim_path, freq):
                    calculate(sim_path, freq)
                    port.uf_ref = port.uf_ref * factor
                    port.uf_tot = port.uf_inc + port.uf_ref
                    port.if_tot = (port.uf_inc - port.uf_ref) / port.Z_ref

                port.CalcPort = again
            return port

        monkeypatch.setattr(FakeFDTD, "AddRectWaveGuidePort", add)

    @staticmethod
    def checks(capsys):
        return [line for line in capsys.readouterr().out.splitlines() if "CHECK" in line]

    def test_each_is_read_again_a_line_short_of_its_source_and_halfway_there(
        self, fake_engine, tmp_path
    ):
        """Facing either way along the guide, with no excitation of its own. The
        box is five cells deep, so the deeper plane is four cells in and the
        shallower two."""
        from Microwave.Solvers.openems import driver

        problem = self.reading.guide(second=True)
        fdtd, _, _ = _build(problem, tmp_path)
        z = np.asarray(problem.grid.z)
        for port, toward in ((problem.ports[0], -1), (problem.ports[1], 1)):
            again = {
                p.recorded["PortNamePrefix"]: p.recorded
                for p in fdtd.ports
                if "PortNamePrefix" in p.recorded and p.recorded["port_nr"] == port.number
            }
            plane = int(np.argmin(np.abs(z - port.stop[2])))
            assert set(again) == set(driver.INSIDE_PREFIXES)
            for prefix, cells in zip(driver.INSIDE_PREFIXES, (4, 2), strict=True):
                probes = again[prefix]
                assert probes["excite"] == 0
                assert probes["start"] == port.start
                assert probes["stop"][:2] == port.stop[:2]
                assert probes["stop"][2] == z[plane + cells * toward]

    def test_a_port_of_another_kind_is_not(self, fake_engine, tmp_path):
        fdtd, _, _ = _build(_microstrip(), tmp_path)
        assert not any("PortNamePrefix" in p.recorded for p in fdtd.ports)

    def test_a_port_reading_its_mode_alone_is_stated_and_not_warned_of(
        self, fake_engine, tmp_path, capsys
    ):
        results = self.reading.solved(tmp_path, inside=(1,))
        near = results["provenance"]["near_field"]
        assert near["ports"]["1"]["depth"] == pytest.approx(1.6, abs=1e-9)
        assert near["ports"]["1"]["shallow"] == pytest.approx(0.8, abs=1e-9)
        # The next mode is TE20 across the broad wall, per millimetre, and the
        # grid holds its cutoff a little under the continuous guide's.
        k = 2 * math.pi * near["ports"]["1"]["frequency"] / 299792458.0 * 1e-3
        continuous = math.sqrt((2 * math.pi / 10.7) ** 2 - k**2)
        assert near["ports"]["1"]["decay"] == pytest.approx(continuous, rel=0.02, abs=0.0)
        assert near["ports"]["1"]["decay"] < continuous
        assert near["share"] == pytest.approx(0.0, abs=1e-12)
        assert near["port"] == 1
        assert not self.checks(capsys)

    def test_field_besides_its_mode_is_warned_of_naming_the_port(
        self, fake_engine, tmp_path, capsys, monkeypatch
    ):
        """The planes inside record the port's reflected wave ``f`` times over,
        so the waves differ by ``f - 1`` of what that wave reads as at each
        plane: twice as much at ``f = 3`` as at ``2``, and alike at both planes,
        which the deeper lies twice as deep as."""
        shares = []
        for twice, factor in enumerate((2.0, 3.0)):
            self.recording(monkeypatch, factor)
            found = self.reading.solved(tmp_path / str(twice), inside=(1,))["provenance"]
            shares.append(found["near_field"]["share"])
            assert found["near_field"]["growth"] == pytest.approx(0.5, rel=1e-9, abs=0.0)
        assert shares[1] == pytest.approx(2 * shares[0], rel=1e-9, abs=0.0)
        checks = self.checks(capsys)
        assert len(checks) == 2
        assert all("severity=warn" in c and "largest part of it at port 1" in c for c in checks)
        assert all("Draw the port farther" in c and "may be right" not in c for c in checks)

    def test_a_difference_growing_toward_the_source_says_the_matrix_may_be_right(
        self, fake_engine, tmp_path, capsys, monkeypatch
    ):
        """The deeper plane is twice as deep, and its difference here is three
        times the shallower one's."""
        self.recording(monkeypatch, 4.0, halfway=2.0)
        results = self.reading.solved(tmp_path, inside=(1,))
        assert results["provenance"]["near_field"]["growth"] == pytest.approx(
            1.5, rel=1e-9, abs=0.0
        )
        assert results["provenance"]["near_field"]["side"] == "behind"
        [check] = self.checks(capsys)
        assert "grows with depth faster" in check and "may then be right" in check

    def test_a_port_not_driven_is_weighed_against_the_power_driven_in(
        self, fake_engine, tmp_path, monkeypatch
    ):
        from Microwave.Solvers.openems import nearfield

        self.recording(monkeypatch, 3.0)
        problem = self.reading.guide(second=True)
        results = self.reading.solved(tmp_path, headers=(1, 2), problem=problem)
        ports = results["provenance"]["near_field"]["ports"]
        assert ports["2"]["share"] > nearfield.BAR

    def test_a_run_whose_record_did_not_finish_says_that_and_not_this(
        self, fake_engine, tmp_path, capsys, monkeypatch
    ):
        """A record cut short moves the planes' readings apart by itself, so the
        port is not blamed; the share is kept in the record either way."""
        from Microwave.Solvers.openems import nearfield

        monkeypatch.setattr(FakePort, "record_decay", 0.5)
        self.recording(monkeypatch, 3.0)
        results = self.reading.solved(tmp_path, inside=(1,))
        assert results["provenance"]["near_field"]["share"] > nearfield.BAR
        checks = self.checks(capsys)
        assert len(checks) == 1 and "stopped before the response" in checks[0]

    def test_the_waves_are_carried_along_the_guide_between_the_planes(
        self, fake_engine, tmp_path, monkeypatch
    ):
        """The fake records one field at every plane. Taken as the port's own,
        the planes agree; carried along a guide, which would turn the wave
        between the planes, they differ, and the run keeps by how much and
        where."""
        from Microwave.Solvers.openems import nearfield

        same = self.reading.solved(tmp_path / "same", inside=(1,))["provenance"]["near_field"]
        assert same["share"] == pytest.approx(0.0, abs=1e-12)
        monkeypatch.setattr(portreading, "predicted", PREDICTED)
        turned = self.reading.solved(tmp_path / "turned", inside=(1,))["provenance"]
        found = turned["near_field"]
        assert found["port"] == 1 and found["share"] > nearfield.BAR

    def test_the_waves_are_weighed_only_where_the_mode_propagates(
        self, fake_engine, tmp_path, monkeypatch
    ):
        """A band starting below the guide's cutoff: the waves there decay
        rather than travel, and the figure leaves them out."""
        from Microwave import units

        monkeypatch.setattr(portreading, "predicted", PREDICTED)
        problem = replace(self.reading.guide(), frequency=Frequency(10e9, 20e9, 41))
        near = self.reading.solved(tmp_path, problem=problem, inside=(1,))["provenance"]
        cutoff = units.SPEED_OF_LIGHT / (2 * 10.7e-3)
        assert near["near_field"]["frequency"] > cutoff

    def test_a_port_facing_down_the_axis_is_carried_as_one_facing_up(
        self, fake_engine, tmp_path, monkeypatch
    ):
        """The same guide drawn the other way along its axis carries the planes
        inside the other way, and the two differ by as much."""
        monkeypatch.setattr(portreading, "predicted", PREDICTED)
        up = self.reading.guide()
        top = float(np.asarray(up.grid.z)[-1])
        [port] = up.ports
        down = replace(
            up,
            grid=MeshGrid(up.grid.x, up.grid.y, top - np.asarray(up.grid.z)[::-1]),
            solids=tuple(
                replace(
                    s,
                    lower=(*s.lower[:2], top - s.upper[2]),
                    upper=(*s.upper[:2], top - s.lower[2]),
                )
                for s in up.solids
            ),
            ports=(
                replace(
                    port,
                    start=(*port.start[:2], top - port.start[2]),
                    stop=(*port.stop[:2], top - port.stop[2]),
                ),
            ),
        )
        found = [
            self.reading.solved(tmp_path / name, problem=problem, inside=(1,))["provenance"][
                "near_field"
            ]
            for name, problem in (("up", up), ("down", down))
        ]
        assert found[0]["share"] > 0
        assert found[1]["share"] == pytest.approx(found[0]["share"], rel=1e-9, abs=0.0)
        assert found[1]["growth"] == pytest.approx(found[0]["growth"], rel=1e-9, abs=0.0)

    def test_a_port_left_as_read_is_not_weighed_and_says_so(self, fake_engine, tmp_path):
        results = self.reading.solved(tmp_path, headers=(), inside=(1,))
        near = results["provenance"]["near_field"]
        assert near["ports"] == {"1": {"reason": "its reading is left as openEMS returned it"}}
        assert "share" not in near

    def test_a_plane_inside_that_cannot_be_read_says_why(self, fake_engine, tmp_path, capsys):
        results = self.reading.solved(tmp_path, inside=())
        port = results["provenance"]["near_field"]["ports"]["1"]
        assert port["missed"] is True
        assert port["reason"].startswith("it is not read 4 cells into its box: its probe file")
        checks = self.checks(capsys)
        assert len(checks) == 1 and "port 1 is not weighed" in checks[0]

    @staticmethod
    def unreadable(monkeypatch, prefix):
        """The probes named with ``prefix`` raise as the engine's own reader
        does on a file that is missing or empty."""
        added = TestAWaveguidePortIsReadAgainInsideItsBox.added

        def add(self, *args, **kw):
            port = added(self, *args, **kw)
            if kw.get("PortNamePrefix") == prefix:

                def unreadable(sim_path, freq):
                    raise FileNotFoundError(f"{prefix}port_ut_1 not found")

                port.CalcPort = unreadable
            return port

        monkeypatch.setattr(FakeFDTD, "AddRectWaveGuidePort", add)

    def test_probes_that_recorded_nothing_cost_the_run_nothing(
        self, fake_engine, tmp_path, capsys, monkeypatch
    ):
        """The matrix does not depend on these probes, so the run keeps its
        answer and names the port."""
        from Microwave.Solvers.openems import driver

        self.unreadable(monkeypatch, driver.INSIDE_PREFIXES[0])
        results = self.reading.solved(tmp_path, inside=(1,))
        assert "S11" in results["s_parameters"]
        port = results["provenance"]["near_field"]["ports"]["1"]
        assert port == {
            "reason": "it is not read 4 cells into its box: inside_port_ut_1 not found",
            "missed": True,
        }
        assert any("port 1 is not weighed" in c for c in self.checks(capsys))

    def test_a_shallower_plane_that_recorded_nothing_leaves_the_port_weighed_at_one(
        self, fake_engine, tmp_path, monkeypatch
    ):
        from Microwave.Solvers.openems import driver

        self.unreadable(monkeypatch, driver.INSIDE_PREFIXES[1])
        port = self.reading.solved(tmp_path, inside=(1,))["provenance"]["near_field"]["ports"]["1"]
        assert "missed" not in port
        assert port["depth"] == pytest.approx(1.6, abs=1e-9) and port["shallow"] is None

    def test_a_box_with_no_line_between_its_source_and_its_plane_says_so(
        self, fake_engine, tmp_path, capsys
    ):
        guide = self.reading.guide()
        z = np.asarray(guide.grid.z)
        plane = int(np.argmin(np.abs(z - 6.0)))
        port = replace(guide.ports[0], start=(0.0, 0.0, float(z[plane - 1])))
        results = self.reading.solved(tmp_path, problem=replace(guide, ports=(port,)), inside=())
        assert results["provenance"]["near_field"]["ports"]["1"] == {
            "reason": "its box holds no grid line between its source and its plane",
            "missed": True,
        }
        checks = self.checks(capsys)
        assert len(checks) == 1 and "port 1 is not weighed" in checks[0]

    def test_a_box_two_cells_deep_is_read_one_cell_in_and_cannot_place_the_change(
        self, fake_engine, tmp_path, capsys, monkeypatch
    ):
        guide = self.reading.guide()
        z = np.asarray(guide.grid.z)
        plane = int(np.argmin(np.abs(z - 6.0)))
        port = replace(guide.ports[0], start=(0.0, 0.0, float(z[plane - 2])))
        problem = replace(guide, ports=(port,))
        results = self.reading.solved(tmp_path, problem=problem, inside=(1,))
        near = results["provenance"]["near_field"]
        assert near["ports"]["1"]["depth"] == pytest.approx(0.4, abs=1e-9)
        assert near["ports"]["1"]["shallow"] is None
        assert not self.checks(capsys)
        self.recording(monkeypatch, 3.0)
        self.reading.solved(tmp_path / "again", problem=problem, inside=(1,))
        [check] = self.checks(capsys)
        assert "Port 1 is read at one plane inside its box, which cannot tell" in check

    def test_a_port_of_another_kind_in_the_run_is_named(self, fake_engine, tmp_path):
        from Microwave.Solvers.openems import driver

        problem = self.reading.guide()
        found = driver._near_field(
            problem,
            {2: types.SimpleNamespace(kind="lumped")},
            {},
            {},
            {},
            None,
            tmp_path,
            problem.frequency.values(),
            np.ones(problem.frequency.points),
        )
        assert found["ports"] == {
            "2": {"reason": "only a rectangular waveguide port is read again"}
        }


class TestTheWaveguidePortsAccountForThePowerDrivenIn:
    """Every waveguide port's net power is summed over the power driven in, and
    the run says where that is past the bar and sound."""

    reading = TestAWaveguidePortIsReadAtItsOwnScales()

    @staticmethod
    def expected(results):
        """Each side of the share, from each port's two powers as the results
        file keeps them."""
        ports = results["ports"]
        excited = str(results["excited_port"])
        driven = np.asarray(ports[excited]["power_incident"])
        net = sum(
            np.asarray(port["power_incident"]) - np.asarray(port["power_reflected"])
            for port in ports.values()
        )
        sent = sum(
            np.asarray(port["power_incident"])
            for number, port in ports.items()
            if number != excited
        )
        share, sent = net / driven, np.broadcast_to(sent / driven, driven.shape)

        def side(at):
            return {
                "share": pytest.approx(float(share[at]), rel=1e-12, abs=0.0),
                "sent": pytest.approx(float(sent[at]), rel=1e-12, abs=0.0),
                "frequency": results["frequency"][at],
            }

        return {"below": side(int(np.argmin(share))), "above": side(int(np.argmax(share)))}

    @staticmethod
    def reflecting(monkeypatch, factor):
        """Each port's own probes record a reflected wave ``factor`` times the
        incident one."""
        added = FakeFDTD.AddRectWaveGuidePort

        def add(self, *args, **kw):
            port = added(self, *args, **kw)
            if not kw.get("PortNamePrefix"):
                calculate = port.CalcPort

                def again(sim_path, freq):
                    calculate(sim_path, freq)
                    port.uf_ref = port.uf_inc * factor
                    port.uf_tot = port.uf_inc + port.uf_ref
                    port.if_tot = (port.uf_inc - port.uf_ref) / port.Z_ref

                port.CalcPort = again
            return port

        monkeypatch.setattr(FakeFDTD, "AddRectWaveGuidePort", add)

    @staticmethod
    def balance_checks(capsys):
        lines = capsys.readouterr().out.splitlines()
        return [line for line in lines if "CHECK" in line and "power than was" in line]

    def test_a_closed_guide_whose_ports_account_for_less_is_warned_of(
        self, fake_engine, tmp_path, capsys
    ):
        """Both ends absorb behind a port filling the guide, so power leaves
        only through the ports; the fake's two ports each send back a tenth."""
        problem = self.reading.guide(second=True)
        results = self.reading.solved(tmp_path, headers=(1, 2), problem=problem)
        found = results["provenance"]["power_balance"]
        assert found == {"bar": 0.001, **self.expected(results)}
        assert found["above"]["share"] > 0.001 and found["below"]["share"] > 0.0
        checks = self.balance_checks(capsys)
        assert len(checks) == 1 and "account for" in checks[0]

    def test_a_guide_open_at_its_far_end_is_held_below_zero_only(
        self, fake_engine, tmp_path, capsys
    ):
        results = self.reading.solved(tmp_path)
        found = results["provenance"]["power_balance"]
        assert found["above"]["share"] > 0.001
        assert found["below_only"] == (
            "the ZMax face absorbs, and field can reach it outside every waveguide port's guide"
        )
        assert not self.balance_checks(capsys)

    def test_more_power_out_than_in_is_warned_of_in_any_model(
        self, fake_engine, tmp_path, capsys, monkeypatch
    ):
        self.reflecting(monkeypatch, 1.1)
        results = self.reading.solved(tmp_path)
        found = results["provenance"]["power_balance"]
        assert "below_only" in found and found["below"]["share"] < -0.001
        assert found["below"] == self.expected(results)["below"]
        checks = self.balance_checks(capsys)
        assert len(checks) == 1 and "give out" in checks[0]

    def test_a_material_that_dissipates_holds_it_below_zero_only(
        self, fake_engine, tmp_path, capsys
    ):
        guide = self.reading.guide(second=True)
        lossy = Material(name="Lossy", kind="lossy_dielectric", epsilon=2.2, conductivity=0.5)
        block = Solid(material="Lossy", lower=(0.0, 0.0, 20.0), upper=(10.7, 4.3, 30.0))
        problem = replace(guide, materials=(*guide.materials, lossy), solids=(*guide.solids, block))
        results = self.reading.solved(tmp_path, headers=(1, 2), problem=problem)
        found = results["provenance"]["power_balance"]
        assert found["below_only"] == "a material in the model turns power into heat"
        assert found["above"]["share"] > 0.001
        assert not self.balance_checks(capsys)

    def test_a_record_whose_tail_can_move_it_by_half_the_bar_is_not_weighed(
        self, fake_engine, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(FakePort, "record_decay", 0.5)
        results = self.reading.solved(tmp_path, headers=(1, 2), problem=self.reading.guide(True))
        found = results["provenance"]["power_balance"]
        assert set(found) == {"bar", "reason"}
        assert found["reason"].startswith("the record's tail can move it by ")

    @staticmethod
    def weighed(problem, ports, tail):
        """What ``_balance`` says of ``ports`` before it asks any for a reading."""
        from Microwave.Solvers.openems import driver

        frequency = problem.frequency.values()
        models = {port.number: port for port in ports}
        return driver._balance(
            problem, models, {}, dict.fromkeys(models), None, frequency, frequency, tail
        )

    def test_a_tail_just_past_half_the_bar_is_enough(self, fake_engine):
        """A tail of 2.6e-4 moves it by up to 5.2e-4, past half the bar."""
        problem = self.reading.guide()
        found = self.weighed(problem, problem.ports, {1: 2.6e-4})
        assert found == {"bar": 0.001, "reason": "the record's tail can move it by 0.00052"}

    def test_ports_facing_one_face_on_different_planes_are_not_weighed(self, fake_engine):
        problem = self.reading.guide()
        first = problem.ports[0]
        deeper = replace(first, number=2, stop=(10.7, 4.3, 8.0), excite=False)
        found = self.weighed(problem, (first, deeper), {1: 0.0})
        assert found["reason"].startswith("the waveguide ports facing the ZMin face read on")

    def test_the_power_driven_in_is_the_driven_port_s(self, fake_engine, tmp_path, monkeypatch):
        """Port 2 is driven and reads its waves twice as strong as port 1."""
        added = FakeFDTD.AddRectWaveGuidePort

        def add(self, *args, **kw):
            port = added(self, *args, **kw)
            if args[0] == 2 and not kw.get("PortNamePrefix"):
                calculate = port.CalcPort

                def again(sim_path, freq):
                    calculate(sim_path, freq)
                    for name in ("uf_inc", "uf_ref", "uf_tot", "if_tot"):
                        setattr(port, name, getattr(port, name) * 2.0)

                port.CalcPort = again
            return port

        monkeypatch.setattr(FakeFDTD, "AddRectWaveGuidePort", add)
        guide = self.reading.guide(second=True)
        first, second = guide.ports
        problem = replace(guide, ports=(replace(first, excite=False), replace(second, excite=True)))
        results = self.reading.solved(tmp_path, headers=(1, 2), problem=problem)
        assert results["excited_port"] == 2
        found = results["provenance"]["power_balance"]
        assert found == {"bar": 0.001, **self.expected(results)}

    def test_a_window_in_the_metal_before_an_absorber_opens_its_face(self, fake_engine, tmp_path):
        """The far end is shorted by metal through the absorber's inner plane,
        which closes it, until a dielectric window runs from the guide through
        the metal and across that plane. A pocket of the same glass sealed
        inside the metal crosses the plane too, and nothing reaches it."""
        guide = self.reading.guide()
        # The absorber's inner line, eight cells in from the far end.
        inner = float(np.asarray(guide.grid.z)[-9])
        block = Material(name="Block", kind="pec")
        glass = Material(name="Glass", kind="dielectric", epsilon=2.2)
        short = Solid(
            material="Block", lower=(0.0, 0.0, 40.0), upper=(10.7, 4.3, 50.0), priority=10
        )
        window = Solid(
            material="Glass",
            lower=(0.0, 0.0, 39.0),
            upper=(2.0, 4.3, inner + 0.5),
            priority=20,
        )
        pocket = replace(window, lower=(4.0, 1.0, inner - 0.5), upper=(6.0, 3.0, inner + 0.5))
        found = {}
        for name, solids in (
            ("shorted", (short,)),
            ("window", (short, window)),
            ("pocket", (short, pocket)),
        ):
            problem = replace(
                guide,
                materials=(*guide.materials, block, glass),
                solids=(*guide.solids, *solids),
            )
            results = self.reading.solved(tmp_path / name, problem=problem)
            found[name] = results["provenance"]["power_balance"]
        assert "below_only" not in found["shorted"]
        assert found["window"]["below_only"] == (
            "the ZMax face absorbs, and field can reach it outside every waveguide port's guide"
        )
        assert "below_only" not in found["pocket"]

    def test_a_kind_dissipates_exactly_where_the_engine_is_handed_a_conductivity(self):
        from Microwave.Solvers.openems import driver
        from Microwave.Solvers.openems.model import LOSSY_KINDS, MATERIAL_KINDS

        lossy = {
            "pec": {},
            "dielectric": {"epsilon": 2.2},
            "lossy_dielectric": {"epsilon": 2.2, "conductivity": 0.5},
            "conducting_sheet": {"conductivity": 5.8e7, "thickness": 0.035},
        }
        assert set(lossy) == MATERIAL_KINDS
        for kind, fields in lossy.items():
            prop = driver._add_material(FakeCSX(), Material(name="M", kind=kind, **fields), 1e-3)
            handed = prop.kw.get("kappa", 0.0) > 0 or prop.kw.get("conductivity", 0.0) > 0
            assert handed == (kind in LOSSY_KINDS), kind

    def test_ports_left_as_openems_read_them_are_not_weighed(self, fake_engine, tmp_path):
        results = self.reading.solved(tmp_path, headers=(), inside=())
        assert results["provenance"]["power_balance"] == {
            "bar": 0.001,
            "reason": "the ports' readings are left as openEMS returned them",
        }

    def test_a_port_of_another_kind_in_the_run_is_not_weighed(self, fake_engine):
        problem = self.reading.guide()
        lumped = types.SimpleNamespace(number=2, kind="lumped")
        found = self.weighed(problem, (problem.ports[0], lumped), {})
        assert found == {"bar": 0.001, "reason": "a port of another kind is in the run"}


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


def _in_ptfe(problem: Problem) -> Problem:
    """``problem`` in a medium of PTFE, which fills every cell no solid covers."""
    ptfe = Material(name="PTFE", kind="dielectric", epsilon=2.1)
    return replace(problem, materials=(*problem.materials, ptfe), medium="PTFE")


class TestTheMediumFillsEveryCellNoSolidCovers:
    """One box of the medium's material under every solid, reaching past the
    grid so the absorber's cells are the medium too."""

    def test_it_is_one_box_below_every_solid_reaching_past_the_grid(self, fake_engine, tmp_path):
        from Microwave.Solvers.openems.model import MEDIUM_PRIORITY

        problem = _in_ptfe(_microstrip())
        _, csx, _ = _build(problem, tmp_path)
        (medium,) = [prop for prop in csx.properties if prop.name == "PTFE"]
        assert medium.kw["epsilon"] == 2.1
        ((start, stop, priority),) = medium.boxes
        assert priority == MEDIUM_PRIORITY
        assert all(solid.priority > MEDIUM_PRIORITY for solid in problem.solids)
        placed, _ = problem.at_the_origin()
        for dim in range(3):
            lines = placed.grid[dim]
            assert start[dim] < lines[0] and stop[dim] > lines[-1]

    def test_it_stops_at_a_face_the_domain_ends_on_where_ports_open_the_absorber(
        self, fake_engine, tmp_path
    ):
        """The absorber beyond an Ends face carries on the guide behind the
        ports' plane, which openEMS reads as empty, so it is left to vacuum."""
        problem = _in_ptfe(_waveguide())
        params = dict(problem.grid.params, padding=[["0", "0"], ["0", "0"], ["0", "0"]])
        ended = replace(problem, grid=replace(problem.grid, params=params))
        _, csx, _ = _build(ended, tmp_path)
        ((start, stop, _),) = next(prop for prop in csx.properties if prop.name == "PTFE").boxes
        placed, _ = ended.at_the_origin()
        lines = placed.grid[2]
        assert (start[2], stop[2]) == (float(lines[8]), float(lines[-9]))
        assert start[0] < placed.grid[0][0] and stop[0] > placed.grid[0][-1]

    def test_a_study_in_vacuum_lays_nothing_for_it(self, fake_engine, tmp_path):
        _, csx, _ = _build(_microstrip(), tmp_path)
        assert [prop.name for prop in csx.properties if prop.kind == "material"] == ["FR4"]

    def test_it_travels_in_the_envelope(self):
        problem = _in_ptfe(_microstrip())
        assert Problem.from_json(problem.to_json()).medium == "PTFE"
        assert json.loads(_microstrip().to_json())["medium"] == ""

    def test_a_medium_the_envelope_does_not_define_is_refused(self):
        with pytest.raises(EnvelopeError, match="the medium 'PTFE' is not a defined material"):
            replace(_microstrip(), medium="PTFE")

    def test_a_metal_is_no_medium(self):
        with pytest.raises(EnvelopeError, match="the medium 'GroundPlane' is a pec"):
            replace(_microstrip(), medium="GroundPlane")

    def test_a_solid_the_medium_would_cover_is_refused(self):
        from Microwave.Solvers.openems.model import MEDIUM_PRIORITY

        problem = _in_ptfe(_microstrip())
        sunk = replace(problem.solids[0], priority=MEDIUM_PRIORITY)
        with pytest.raises(EnvelopeError, match="where the medium would cover it"):
            replace(problem, solids=(sunk, *problem.solids[1:]))

    def test_the_run_records_it_apart_from_its_outside(self):
        from Microwave.Results import modelled
        from Microwave.Solvers.openems import driver

        records = driver.modelled(_in_ptfe(_microstrip()))
        assert {"medium": "PTFE"} in records
        assert modelled.said([{"medium": "PTFE"}]) == [
            "The medium: 'PTFE' fills every space no bound body fills"
        ]
        assert not any(modelled.medium(one) for one in driver.modelled(_microstrip()))

    def test_a_closed_run_records_it_too(self):
        from Microwave.Solvers.openems import driver

        walled = replace(_in_ptfe(_microstrip()), boundary=("PEC",) * 6)
        records = driver.modelled(walled)
        assert {"medium": "PTFE"} in records
        assert not any("boundary" in one for one in records)


class TestAWaveguidePortInAMedium:
    """openEMS' waveguide port reads its guide as empty, so a medium filling the
    guide is refused, and a guide drawn as a box of vacuum inside it is taken."""

    def _findings(self, problem):
        from Microwave.Solvers.openems.preflight import ports

        return ports._check_the_guide_is_empty(problem.ports[0], problem)

    def test_a_guide_drawn_as_its_vacuum_is_taken(self):
        assert self._findings(_in_ptfe(_waveguide())) == []

    def test_a_guide_the_medium_fills_is_refused_naming_it(self):
        problem = _in_ptfe(_waveguide())
        walls = (Material(name="Walls", kind="pec"),)
        emptied = replace(problem, materials=(*problem.materials, *walls), solids=())
        (finding,) = self._findings(emptied)
        assert "filled with the study's medium 'PTFE'" in finding.message

    def test_a_guide_the_vacuum_box_covers_in_part_is_refused(self):
        problem = _in_ptfe(_waveguide())
        (air,) = problem.solids
        short = replace(air, upper=(air.upper[0], air.upper[1], 5.0))
        (finding,) = self._findings(replace(problem, solids=(short,)))
        assert "'PTFE'" in finding.message

    def test_a_guide_drawn_as_a_triangulated_body_of_vacuum_is_taken(self):
        """A body with a curved part cut from it reaches the envelope as a
        triangulation, and the port's box inside it is held as a box would be."""
        problem = _in_ptfe(_waveguide())
        (air,) = problem.solids
        (x0, y0, z0), (x1, y1, z1) = air.lower, air.upper
        corners = (
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        )  # fmt: skip
        faces = (
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
        )  # fmt: skip
        meshed = replace(air, vertices=corners, faces=faces)
        assert meshed.is_mesh
        assert self._findings(replace(problem, solids=(meshed,))) == []

    def test_a_lossy_medium_of_vacuum_s_permittivity_fills_the_guide_too(self):
        problem = _in_ptfe(_waveguide())
        lossy = Material(name="PTFE", kind="lossy_dielectric", epsilon=1.0, conductivity=0.1)
        emptied = replace(problem, materials=(*problem.materials[:-1], lossy), solids=())
        (finding,) = self._findings(emptied)
        assert "'PTFE'" in finding.message

    def test_a_guide_refused_for_its_drawn_fill_is_not_refused_again(self):
        problem = _in_ptfe(_waveguide())
        filled = replace(problem.materials[0], epsilon=2.0)
        (finding,) = self._findings(replace(problem, materials=(filled, *problem.materials[1:])))
        assert "study's medium" not in finding.message


def test_a_medium_slower_than_nothing_is_still_a_medium():
    """A relative permeability below one slows a wave by less than vacuum does,
    and the grid is laid in it all the same."""
    params = MeshParams(metal_res=0.1, dielectric_res=1.0, cap=1.0, medium=0.9)
    assert params.ceiling == pytest.approx(1.0 / math.sqrt(0.9), rel=1e-12)
