# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the coaxial port hands openEMS, and what it reads back.

This is the one port kind the adapter builds itself, so it is the one whose
*calls* are its output - there is no upstream class behind it whose behaviour
could be taken on trust. What is asserted here is therefore the same thing
``test_driver_translation`` asserts about the driver: which primitives were
placed, where, and with what weights.

The fakes record rather than compute. ``CoaxialPort`` subclasses openEMS' own
``Port``, so the bindings have to be importable - but nothing here solves, and a
``ContinuousStructure`` is stood in for by an object that answers three
questions.

``ReadUIData`` is exercised against probe files written by hand, carrying a
travelling wave whose impedance and propagation constant are known in closed
form, so what comes back can be compared with what went in.

That leaves the sign, and where it has to be caught is not obvious: the
impedance is a ratio in which the current appears twice and cancels, so it is
blind to an inverted loop. What an inverted loop moves is the incident and
reflected split, and so every S-parameter. So the loop's sign is asserted where
the probe is *made* rather than in what it reads.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("CSXCAD", reason="the openEMS bindings are not on this interpreter")

from Microwave.Solvers.openems.coaxial import CoaxialPort  # noqa: E402

#: The line every case here is built on: a 1 mm conductor in a 3.5 mm bore.
INNER = 1.0
OUTER = 3.5

#: A uniform grid, so a probe's placement can be read off it by eye and the
#: differences telescope exactly.
CELL = 1.0
LINES = np.arange(0.0, 41.0, CELL)


class Recorder:
    """One probe or excitation, remembering how it was made and what it holds."""

    def __init__(self, kind, name, **keywords):
        self.kind = kind
        self.name = name
        self.keywords = keywords
        self.boxes = []
        self.shells = []
        self.weight_function = None

    def AddBox(self, start, stop, priority=0, **keywords):  # noqa: N802 - upstream's name
        self.boxes.append((tuple(start), tuple(stop)))

    def AddCylindricalShell(self, start, stop, radius, thickness, priority=0):  # noqa: N802
        self.shells.append((tuple(start), tuple(stop), radius, thickness, priority))

    def SetWeightFunction(self, functions):  # noqa: N802 - upstream's name
        self.weight_function = list(functions)


class Grid:
    def __init__(self, lines=LINES, unit=1e-3):
        self._lines = lines
        self._unit = unit

    def GetLines(self, axis):  # noqa: N802 - upstream's name
        return self._lines

    def GetDeltaUnit(self):  # noqa: N802 - upstream's name
        return self._unit


class Structure:
    """Everything ``CoaxialPort`` asks of a ``ContinuousStructure``."""

    def __init__(self, lines=LINES, unit=1e-3):
        self.grid = Grid(lines, unit)
        self.probes = []
        self.excitations = []

    def GetGrid(self):  # noqa: N802 - upstream's name
        return self.grid

    def AddProbe(self, name, p_type, weight=1, norm_dir=None, **keywords):  # noqa: N802
        made = Recorder("probe", name, p_type=p_type, weight=weight, norm_dir=norm_dir)
        self.probes.append(made)
        return made

    def AddExcitation(self, name, exc_type, exc_val, delay=0):  # noqa: N802
        made = Recorder("excitation", name, exc_type=exc_type, exc_val=list(exc_val))
        self.excitations.append(made)
        return made


def built(csx=None, start=(0.0, 0.0, 0.0), stop=(0.0, 0.0, 40.0), **keywords):
    """A port on the z axis, with everything else at its default."""
    csx = Structure() if csx is None else csx
    settings = {"MeasPlaneShift": 20.0, **keywords}
    port = CoaxialPort(csx, 1, list(start), list(stop), 2, INNER, OUTER, **settings)
    return csx, port


class TestWhatIsRefused:
    def test_a_port_with_no_length_along_the_line(self):
        with pytest.raises(ValueError, match="no length along its propagation axis"):
            built(stop=(0.0, 0.0, 0.0))

    @pytest.mark.parametrize("radii", [(3.5, 1.0), (0.0, 3.5), (-1.0, 3.5)])
    def test_radii_that_are_not_an_annulus(self, radii):
        csx = Structure()
        with pytest.raises(ValueError, match="0 < r_inner < r_outer"):
            CoaxialPort(csx, 1, [0, 0, 0], [0, 0, 40], 2, *radii)


class TestTheVoltageProbes:
    """Three of them, on the three grid lines around the measurement plane."""

    def test_they_land_on_the_lines_around_the_plane_that_was_asked_for(self):
        csx, port = built(MeasPlaneShift=20.0)
        voltage = [p for p in csx.probes if p.keywords["p_type"] == 0]

        assert [p.boxes[0][0][2] for p in voltage] == [19.0, 20.0, 21.0]

    def test_the_plane_that_was_read_is_reported_rather_than_the_one_asked_for(self):
        """openEMS snaps, so where the reference plane ended up is a fact about
        the grid. Asked for a coordinate a third of a cell off a line, the port
        reports the line."""
        csx, port = built(MeasPlaneShift=20.3)

        assert port.measplane_shift == 20.0

    def test_each_probe_runs_radially_across_the_annulus(self):
        csx, port = built()
        first = [p for p in csx.probes if p.keywords["p_type"] == 0][0]
        start, stop = first.boxes[0]

        assert (start[0], stop[0]) == (INNER, OUTER)
        assert start[1] == stop[1] == 0.0

    def test_a_line_drawn_off_the_origin_puts_them_on_its_own_axis(self):
        csx, port = built(start=(10.0, -4.0, 0.0), stop=(10.0, -4.0, 40.0))
        first = [p for p in csx.probes if p.keywords["p_type"] == 0][0]
        start, stop = first.boxes[0]

        assert (start[0], stop[0]) == (10.0 + INNER, 10.0 + OUTER)
        assert start[1] == stop[1] == -4.0

    def test_the_spacings_are_the_cells_the_probes_landed_on(self):
        csx, port = built()

        assert list(port.U_delta) == [CELL, CELL]


class TestTheCurrentProbes:
    """Two loops, encircling the inner conductor and nothing else."""

    def test_they_sit_between_the_voltage_planes(self):
        csx, port = built()
        current = [p for p in csx.probes if p.keywords["p_type"] == 1]

        assert [p.boxes[0][0][2] for p in current] == [19.5, 20.5]

    def test_each_loop_clears_the_inner_conductor_and_stays_inside_the_shield(self):
        csx, port = built()
        start, stop = [p for p in csx.probes if p.keywords["p_type"] == 1][0].boxes[0]

        for axis in (0, 1):
            assert start[axis] < -INNER and stop[axis] > INNER
            assert start[axis] > -OUTER and stop[axis] < OUTER

    def test_the_loop_clears_the_metal_as_the_engine_has_it_not_as_it_was_drawn(self):
        """A curved conductor reaches openEMS grown outward by half the cell it
        is sampled on. A loop placed a fixed share of the gap out therefore sits
        *inside* the conductor on a coarse mesh, and a contour inside the metal
        encircles part of the current rather than all of it - too little current,
        so too high an impedance, and nothing says so."""
        coarse = Structure(lines=np.arange(0.0, 41.0, 2.0))
        csx, port = built(csx=coarse, MeasPlaneShift=20.0)
        start, stop = [p for p in csx.probes if p.keywords["p_type"] == 1][0].boxes[0]

        assert stop[0] - INNER >= 1.0, "half of a 2 mm cell of growth is not cleared"
        assert stop[0] < OUTER

    def test_a_fine_mesh_leaves_the_loop_where_the_gap_puts_it(self):
        """The growth term must not take over on a mesh where it is negligible,
        or the loop drifts out towards the shield for no reason."""
        fine = Structure(lines=np.arange(0.0, 41.0, 0.05))
        csx, port = built(csx=fine, MeasPlaneShift=20.0)
        start, stop = [p for p in csx.probes if p.keywords["p_type"] == 1][0].boxes[0]

        assert stop[0] == pytest.approx(INNER + 0.1 * (OUTER - INNER), rel=1e-12, abs=0.0)

    def test_a_coarse_absorber_far_away_does_not_move_the_loop(self):
        """The cell that decides how far the metal grew is the one *at the
        conductor*. Taking the largest anywhere on the axis instead lets the
        absorber's cells - many times the size, and nowhere near the line -
        push the loop out towards the shield."""
        near = np.arange(0.0, 6.0, 0.05)
        far = np.arange(8.0, 60.0, 4.0)
        lopsided = Structure(lines=np.concatenate([near, far]))
        csx, port = built(csx=lopsided, MeasPlaneShift=4.0)
        start, stop = [p for p in csx.probes if p.keywords["p_type"] == 1][0].boxes[0]

        assert stop[0] == pytest.approx(INNER + 0.1 * (OUTER - INNER), rel=1e-12, abs=0.0)

    def test_the_loop_never_wanders_past_the_middle_of_the_annulus(self):
        """On a mesh too coarse for the port at all, the loop is clamped rather
        than allowed out to the shield, where it would start enclosing the
        return current. Such a mesh is refused by pre-flight; this is what the
        builder does if it is reached anyway."""
        crude = Structure(lines=np.arange(0.0, 41.0, 20.0))
        csx, port = built(csx=crude, MeasPlaneShift=20.0)
        start, stop = [p for p in csx.probes if p.keywords["p_type"] == 1][0].boxes[0]

        assert stop[0] == pytest.approx(INNER + 0.5 * (OUTER - INNER), rel=1e-12, abs=0.0)

    def test_the_loop_is_normal_to_the_line_and_signed_by_its_direction(self):
        """Without ``norm_dir`` openEMS integrates the wrong component, and
        without the sign the current comes back inverted - which reads as a
        negative impedance rather than as an error."""
        csx, port = built()
        current = [p for p in csx.probes if p.keywords["p_type"] == 1]

        assert {p.keywords["norm_dir"] for p in current} == {2}
        assert {p.keywords["weight"] for p in current} == {1}

    def test_a_port_pointing_back_down_the_line_inverts_the_loop(self):
        csx, port = built(start=(0.0, 0.0, 40.0), stop=(0.0, 0.0, 0.0), MeasPlaneShift=20.0)
        current = [p for p in csx.probes if p.keywords["p_type"] == 1]

        assert port.direction == -1
        assert {p.keywords["weight"] for p in current} == {-1}


class TestTheExcitation:
    def test_a_passive_port_places_none(self):
        csx, port = built(excite=0)

        assert csx.excitations == []

    def test_it_sits_on_a_grid_line_at_the_feed(self):
        """A surface lying between two lines is sampled nowhere and drives
        nothing, and openEMS says so only as ``Unused primitive``."""
        csx, port = built(excite=1, FeedShift=8.4)
        (start, stop, radius, thickness, _) = csx.excitations[0].shells[0]

        assert start[2] < 8.0 < stop[2]
        assert stop[2] - start[2] < CELL

    def test_the_shell_fills_the_annulus_and_no_more(self):
        csx, port = built(excite=1)
        (_, _, radius, thickness, _) = csx.excitations[0].shells[0]

        assert radius == (INNER + OUTER) / 2
        assert thickness == OUTER - INNER

    def test_it_drives_across_the_line_and_never_along_it(self):
        """A TEM mode has no field along the line, so a component there would be
        driving something the line does not carry."""
        csx, port = built(excite=1)

        assert csx.excitations[0].keywords["exc_val"][2] == 0.0

    def test_the_amplitude_reaches_the_engine(self):
        csx, port = built(excite=3.0)

        assert csx.excitations[0].keywords["exc_val"][:2] == [3.0, 3.0]

    def test_the_weight_carries_the_modes_own_radial_profile(self):
        """Evaluated as openEMS would: the two transverse components must come
        out as a unit vector pointing away from the axis, scaled by ``1/r``.
        Exciting a uniform field instead launches the mode plus a spray of
        higher-order ones that then have to die before the probes."""
        csx, port = built(excite=1)
        weight = csx.excitations[0].weight_function

        assert weight[2] == "0"
        for x, y in ((2.0, 0.0), (0.0, 2.0), (1.5, 1.5)):
            radius = math.hypot(x, y)
            found = [eval(part, {"x": x, "y": y, "z": 0.0, "sqrt": math.sqrt}) for part in weight]
            assert found[0] == pytest.approx(x / radius**2, rel=1e-12, abs=0.0)
            assert found[1] == pytest.approx(y / radius**2, rel=1e-12, abs=0.0)
            assert math.hypot(found[0], found[1]) == pytest.approx(1 / radius, rel=1e-12, abs=0.0)

    def test_the_weight_is_zero_outside_the_annulus(self):
        csx, port = built(excite=1)
        weight = csx.excitations[0].weight_function

        for x in (0.5, 4.0):
            found = [eval(part, {"x": x, "y": 0.0, "z": 0.0, "sqrt": math.sqrt}) for part in weight]
            assert found == [0.0, 0.0, 0.0]

    def test_a_line_off_the_origin_weights_about_its_own_axis(self):
        csx, port = built(start=(10.0, -4.0, 0.0), stop=(10.0, -4.0, 40.0), excite=1)
        weight = csx.excitations[0].weight_function

        at_centre = [eval(p, {"x": 12.0, "y": -4.0, "z": 0.0, "sqrt": math.sqrt}) for p in weight]
        assert at_centre[0] == pytest.approx(0.5, rel=1e-12, abs=0.0)
        assert at_centre[1] == 0.0


class TestReadingTheLineBack:
    """The arithmetic that turns five probe records into an impedance.

    Fed a travelling wave ``V(z) = V0 exp(-j beta z)`` and ``I = V / Z``, the
    extraction has to return that ``Z`` and that ``beta``. Both are chosen to be
    nothing like each other or like any radius here, so a transposition shows up.
    """

    IMPEDANCE = 51.83
    BETA = 120.0  # radians per metre

    def _read(self, monkeypatch, port, drawn=(19.0, 20.0, 21.0), unit=1e-3, invert_current=False):
        """Fill this port's probes with a travelling wave and let it read them.

        ``drawn`` is where the voltage planes sit in *drawing* units; the wave
        is built at the metres those become, which is the conversion the
        extraction has to make for itself.
        """
        import Microwave.Solvers.openems.coaxial as module

        planes = np.array(drawn) * unit
        loops = planes[:2] + np.diff(planes) / 2.0
        voltage = np.exp(-1j * self.BETA * planes)
        current = np.exp(-1j * self.BETA * loops) / self.IMPEDANCE
        if invert_current:
            current = -current

        class Data:
            def __init__(self, values):
                self.ui_f_val = [np.array([v]) for v in values]
                self.ui_val = [np.array([v.real]) for v in values]
                self.ui_time = [np.zeros(1)]

        order = {tuple(port.U_filenames): voltage, tuple(port.I_filenames): current}
        monkeypatch.setattr(module, "UI_data", lambda names, *rest: Data(order[tuple(names)]))
        port.ReadUIData("unused", np.array([1e9]))

    def test_the_impedance_comes_back(self, monkeypatch):
        csx, port = built()
        self._read(monkeypatch, port)

        assert port.Z_ref[0].real == pytest.approx(self.IMPEDANCE, rel=1e-3, abs=0.0)
        assert abs(port.Z_ref[0].imag) < 1e-6 * self.IMPEDANCE

    def test_the_propagation_constant_comes_back(self, monkeypatch):
        csx, port = built()
        self._read(monkeypatch, port)

        assert port.beta[0].real == pytest.approx(self.BETA, rel=1e-3, abs=0.0)

    def test_the_impedance_is_blind_to_a_global_current_sign(self, monkeypatch):
        """The current appears twice in ``sqrt(V dV / (I dI))`` - as ``I`` and
        inside ``dI`` - so inverting both loops together cancels out of it
        exactly. What an inverted loop moves is the incident and reflected
        split below, and so the S-parameters.

        Which is why the loop's sign is asserted where the probe is *made*,
        in ``TestTheCurrentProbes``: by the time the impedance is read there is
        nothing left to notice it."""
        csx, upright = built()
        self._read(monkeypatch, upright)
        upright_current = upright.if_tot[0]

        csx, inverted = built()
        self._read(monkeypatch, inverted, invert_current=True)

        assert inverted.Z_ref[0].real == pytest.approx(upright.Z_ref[0].real, rel=1e-12, abs=0.0)
        assert inverted.if_tot[0] == pytest.approx(-upright_current, rel=1e-12, abs=0.0)

    def test_one_line_described_in_two_units_answers_the_same(self, monkeypatch):
        """The probe spacings are in drawing units and the derivative is per
        metre, so the grid's own unit is what converts them. Drop it and beta
        comes out scaled by a thousand while the impedance - a ratio in which
        the conversion cancels - goes on looking right.

        The same physical line is described here in metres rather than
        millimetres, so every coordinate is a thousandth of the case above and
        the unit is a thousand times larger."""
        metres = Structure(lines=LINES / 1000.0, unit=1.0)
        csx, port = built(csx=metres, stop=(0.0, 0.0, 0.040), MeasPlaneShift=0.020)
        self._read(monkeypatch, port, drawn=(0.019, 0.020, 0.021), unit=1.0)

        assert port.beta[0].real == pytest.approx(self.BETA, rel=1e-3, abs=0.0)
        assert port.Z_ref[0].real == pytest.approx(self.IMPEDANCE, rel=1e-3, abs=0.0)

    def test_the_reading_is_the_middle_probe_rather_than_their_sum(self, monkeypatch):
        """The base class sums every probe, which for a triplet spanning two
        cells of a travelling wave is three samples of it added together."""
        csx, port = built()
        self._read(monkeypatch, port)

        assert abs(port.uf_tot[0]) == pytest.approx(1.0, rel=1e-9, abs=0.0)
