# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A coaxial port built over geometry that is already in the structure.

Every other port kind this adapter builds is one call to an upstream openEMS
port class. This one is not.

The bindings carry no coaxial class. There is none in ``openEMS.ports`` to call.

The upstream class draws the line. The one on openEMS' master branch takes a
metal property and a fill property and lays a cylinder, a shell and the
dielectric between them itself. In this workbench the user has already drawn all
three and bound their materials, so calling it would put a second, analytic copy
of every conductor into a structure that already holds the drawn one: two
representations of one shield, meshed differently, resolved against each other
by priority. What a coaxial port is wanted for here is the half that is not
geometry - three voltage probes, two current probes, and a radially weighted
excitation across the annulus.

This class therefore lays no metal, no fill and no termination. It reads a line
built elsewhere, and so composes with a drawing.

The measurement is openEMS' own. Differencing a probe triplet along the line
gives the propagation constant and the characteristic impedance, which is the
arithmetic ``MSLPort.ReadUIData`` runs. Only the probes differ: they are shaped
for a round cross-section rather than for a strip over a plane.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from CSXCAD.Utilities import CheckNyDir
from openEMS.ports import Port, UI_data

from .staircase import GROWN_BY

#: How far outside the inner conductor the current loop runs, as a share of the
#: annulus. The loop has to encircle the inner conductor and nothing else, so it
#: must clear the metal without reaching the shield.
#:
#: A share of the gap alone does not clear the metal, because the metal is not
#: where it was drawn. A curved conductor is handed to openEMS grown outward by
#: the share of a cell the run carries, and how big that cell is against the gap
#: is the mesh's business rather than this constant's. The margin is therefore
#: the larger of this share and that growth, and :data:`_LOOP_CLEARANCE` says
#: what happens when even that does not fit.
_LOOP_MARGIN = 0.1

#: The share of the annulus the loop may not pass once it has cleared the metal.
#: Beyond here it is closer to the shield than to the conductor it is meant to
#: encircle, and what it integrates starts to include the return current.
_LOOP_CLEARANCE = 0.5

#: Half-thickness of the excitation shell, as a share of the smallest cell on
#: the propagation axis. The excitation is a surface and openEMS discretises a
#: primitive by sampling grid lines, so the shell has to be thin enough to catch
#: one line and no more. It is a share of the cell rather than an absolute
#: length, because the cell is what must not be spanned.
_EXCITE_HALF_CELL = 0.01


def _cell_at(lines: npt.ArrayLike, position: float) -> float:
    """The width of the cell ``position`` falls in, in drawing units."""
    edges = np.asarray(lines, dtype=float)
    index = int(np.searchsorted(edges, position, side="right")) - 1
    index = max(0, min(index, len(edges) - 2))
    return float(edges[index + 1] - edges[index])


class CoaxialPort(Port):
    """A TEM port on a coaxial line the structure already contains.

    ``start`` and ``stop`` are points on the line's axis rather than corners of
    a box. A round port has no corners, and the two radii say how far out it
    reaches. Their order along ``prop_dir`` is the direction into the structure.

    There is no excitation axis and no sign to get wrong, as there is on an
    ``MSLPort``. The field of a coaxial line is radial, so the two radii decide
    which way it points, and the weight function below carries that.

    A feed resistance is not offered. It would be metal or a lumped element laid
    across the line's mouth, and every conductor in this structure is one the
    user drew. A port that added another would model something that is not in
    the drawing.
    """

    def __init__(
        self,
        csx: Any,
        port_nr: int,
        start: Sequence[float],
        stop: Sequence[float],
        prop_dir: int | str,
        r_inner: float,
        r_outer: float,
        excite: float = 0,
        grown_by: float = GROWN_BY,
        **keywords: Any,
    ) -> None:
        super().__init__(csx, port_nr=port_nr, start=start, stop=stop, excite=excite, **keywords)

        self.prop_ny = CheckNyDir(prop_dir)
        self.ny_P = (self.prop_ny + 1) % 3
        self.ny_PP = (self.prop_ny + 2) % 3
        self.direction = int(np.sign(self.stop[self.prop_ny] - self.start[self.prop_ny]))
        if self.direction == 0:
            raise ValueError("a coaxial port has no length along its propagation axis")
        if not 0 < r_inner < r_outer:
            raise ValueError(
                f"a coaxial port needs 0 < r_inner < r_outer, got {r_inner} and {r_outer}"
            )
        self.r_inner = float(r_inner)
        self.r_outer = float(r_outer)
        # Named rather than swept into the keywords, which go on to openEMS'
        # own Port and would not know it.
        self.grown_by = float(grown_by)

        lines = np.asarray(csx.GetGrid().GetLines(self.prop_ny), dtype=float)
        triplet = self._probe_lines(lines, keywords.get("MeasPlaneShift"))
        self._add_voltage_probes(csx, triplet)
        self._add_current_probes(csx, triplet)
        if excite:
            self._add_excitation(csx, lines, float(keywords.get("FeedShift", 0.0)), excite)

    # ------------------------------------------------------------------ probes

    def _probe_lines(self, lines: np.ndarray, measurement_shift: float | None) -> np.ndarray:
        """The three grid lines the port differences across, in travel order.

        The measurement plane is snapped to the nearest line, and the two lines
        beside it are taken with it, so the plane the user asked for is not in
        general the plane that was read. :attr:`measplane_shift` is where it
        landed. The index is clamped off both ends of the grid, because the
        triplet needs a neighbour on each side.
        """
        span = abs(self.stop[self.prop_ny] - self.start[self.prop_ny])
        shift = 0.5 * span if measurement_shift is None else float(measurement_shift)
        wanted = self.start[self.prop_ny] + self.direction * shift

        index = int(np.argmin(np.abs(lines - wanted)))
        index = max(1, min(index, len(lines) - 2))
        triplet = lines[index - 1 : index + 2]
        if self.direction < 0:
            triplet = triplet[::-1]

        self.measplane_shift = abs(triplet[1] - self.start[self.prop_ny])
        self.U_delta = np.diff(triplet)
        midpoints = triplet[:2] + np.diff(triplet) / 2.0
        self.I_delta = np.diff(midpoints)
        self._current_positions = midpoints
        return triplet

    def _radial_probe_box(self, position: float) -> tuple[np.ndarray, np.ndarray]:
        """A line from the inner conductor's surface out to the shield's bore.

        Along one ray from the axis, at ``position`` on the propagation axis.
        Which ray does not matter: the field is radial, so the voltage between
        the conductors is the same integral whichever way out it is taken.
        """
        start = np.array(self.start, dtype=float)
        start[self.prop_ny] = position
        start[self.ny_P] = self.start[self.ny_P] + self.r_inner
        stop = start.copy()
        stop[self.ny_P] = self.start[self.ny_P] + self.r_outer
        return start, stop

    def _add_voltage_probes(self, csx: Any, triplet: np.ndarray) -> None:
        self.U_filenames = []
        for suffix, position in zip("ABC", triplet):
            name = self.lbl_temp.format("ut") + suffix
            self.U_filenames.append(name)
            probe = csx.AddProbe(name, p_type=0, weight=1)
            probe.AddBox(*self._radial_probe_box(position))
            self.port_props.append(probe)

    def _add_current_probes(self, csx: Any, triplet: np.ndarray) -> None:
        """Two loops around the inner conductor, between the voltage planes.

        A current probe integrates the magnetic field around the box it is
        given, so it measures whatever that box encircles. The box is square
        rather than round because the integral is a contour and its shape does
        not enter the answer. Only what is inside does, and that is the inner
        conductor alone.
        """
        reach = self._loop_reach(csx)
        self.I_filenames = []
        for suffix, position in zip("AB", self._current_positions):
            start = np.zeros(3)
            stop = np.zeros(3)
            start[self.prop_ny] = stop[self.prop_ny] = position
            for axis in (self.ny_P, self.ny_PP):
                start[axis] = self.start[axis] - reach
                stop[axis] = self.start[axis] + reach

            name = self.lbl_temp.format("it") + suffix
            self.I_filenames.append(name)
            probe = csx.AddProbe(name, p_type=1, weight=self.direction, norm_dir=self.prop_ny)
            probe.AddBox(start, stop)
            self.port_props.append(probe)

    def _loop_reach(self, csx: Any) -> float:
        """How far from the axis the current loop runs, in drawing units.

        The loop clears the metal as openEMS has it rather than as it was
        drawn. A curved conductor arrives grown outward by half the cell it is
        sampled on, so a loop placed a fixed share of the gap out can sit inside
        the conductor on a coarse mesh. A contour inside the metal encircles
        part of the current rather than all of it, which reads as too little
        current and so as too high an impedance, with nothing to report it.

        The growth reaches the share the run is being built at. That share is
        read off the envelope rather than off :data:`~.staircase.GROWN_BY`, so a
        run given the metal as drawn does not clear a growth it never made. The
        cell that matters is the one at the conductor's surface rather than the
        largest anywhere on the axis. Out in the absorber the largest is many
        times it, and it would push the loop away for a reason that has nothing
        to do with the metal.

        Where even that leaves no room the loop is put at the middle of the
        annulus. The mesh is too coarse for this port either way, and
        ``preflight.ports`` refuses an annulus the grid does not reach into.
        """
        grid = csx.GetGrid()
        cell = max(
            _cell_at(grid.GetLines(axis), self.start[axis] + self.r_inner)
            for axis in (self.ny_P, self.ny_PP)
        )
        gap = self.r_outer - self.r_inner
        margin = max(_LOOP_MARGIN * gap, self.grown_by * cell)
        return self.r_inner + min(margin, _LOOP_CLEARANCE * gap)

    # -------------------------------------------------------------- excitation

    def _add_excitation(
        self, csx: Any, lines: np.ndarray, feed_shift: float, amplitude: float
    ) -> None:
        """The line's own mode, as a field profile rather than a voltage.

        A coaxial line's TEM field falls as one over the radius, and the weight
        function states that. The two transverse components are ``dx/r^2`` and
        ``dy/r^2``, whose magnitude is ``1/r`` and whose direction is radially
        outward. Exciting a uniform field across the annulus instead would
        launch the mode plus a spray of higher-order ones that then have to die
        out before the probes.

        The mask multiplies the weight to zero outside the annulus. That is a
        second guard beside the cylindrical shell the excitation is drawn on,
        and it costs nothing: openEMS evaluates the function only where the
        primitive already is.

        The shell is placed on a grid line rather than where the feed was asked
        for, because openEMS discretises a primitive by asking the geometry what
        sits on each line. A surface lying between two lines is sampled nowhere
        and drives nothing, and it reports that only as ``Unused primitive`` in
        a log of thousands.
        """
        wanted = self.start[self.prop_ny] + feed_shift * self.direction
        index = int(np.argmin(np.abs(lines - wanted)))
        half = _EXCITE_HALF_CELL * float(np.min(np.diff(lines)))
        start = np.array(self.start, dtype=float)
        start[self.prop_ny] = lines[index] - half
        stop = start.copy()
        stop[self.prop_ny] = lines[index] + half

        names = "xyz"
        offsets = [f"({names[axis]}-{self.start[axis]:.15g})" for axis in (self.ny_P, self.ny_PP)]
        squared = "({0}*{0}+{1}*{1})".format(*offsets)
        mask = f"*(sqrt({squared})<{self.r_outer:.15g})*(sqrt({squared})>{self.r_inner:.15g})"
        weight = ["0", "0", "0"]
        weight[self.ny_P] = f"{offsets[0]}/{squared}{mask}"
        weight[self.ny_PP] = f"{offsets[1]}/{squared}{mask}"

        # Transverse only: a TEM mode has no field along the line.
        value = np.full(3, float(amplitude))
        value[self.prop_ny] = 0.0

        # Priority against the drawn conductors is not a question this has to
        # answer. openEMS resolves an overlap among the primitives of one
        # property type, ``GetPropertyByCoordPriority`` filtering on the type
        # before it compares at all
        # (``CSXCAD/src/ContinuousStructure.cpp``:276-289), so an excitation
        # never competes with a material, whatever either says.
        excitation = csx.AddExcitation(
            self.lbl_temp.format("excite"), exc_type=0, exc_val=value, delay=self.delay
        )
        excitation.SetWeightFunction(weight)
        excitation.AddCylindricalShell(
            start,
            stop,
            0.5 * (self.r_inner + self.r_outer),
            self.r_outer - self.r_inner,
            priority=self.priority,
        )
        self.port_props.append(excitation)

    # ------------------------------------------------------------------ reading

    def ReadUIData(  # noqa: N802 - upstream's name
        self, sim_path: str, freq: npt.ArrayLike, signal_type: str = "pulse"
    ) -> None:
        """Impedance and propagation constant, by differencing along the line.

        The middle probe is the reading. The outer two give its derivative,
        and the two current loops give theirs. ``sqrt(E * dE / (H * dH))`` is
        then the characteristic impedance and ``sqrt(-dE * dH / (H * E))`` the
        propagation constant, both per frequency. ``MSLPort`` uses the same
        method, and the impedance it defines is the one this adapter reports.

        The grid's own unit converts the probe spacings, which are in drawing
        units, into the metres the derivative is per.
        """
        self.u_data = UI_data(self.U_filenames, sim_path, freq, signal_type)
        self.uf_tot = self.u_data.ui_f_val[1]
        self.ut_tot = self.u_data.ui_val[1]

        self.i_data = UI_data(self.I_filenames, sim_path, freq, signal_type)
        # Averaged so that the current lands at the same place along the line
        # as the voltage. The loops sit between the voltage planes rather than
        # on them.
        self.if_tot = 0.5 * (self.i_data.ui_f_val[0] + self.i_data.ui_f_val[1])
        self.it_tot = 0.5 * (self.i_data.ui_val[0] + self.i_data.ui_val[1])

        unit = self.CSX.GetGrid().GetDeltaUnit()
        voltage = self.u_data.ui_f_val[1]
        d_voltage = (self.u_data.ui_f_val[2] - self.u_data.ui_f_val[0]) / (
            np.sum(np.abs(self.U_delta)) * unit
        )
        current = self.if_tot
        d_current = (self.i_data.ui_f_val[1] - self.i_data.ui_f_val[0]) / (
            np.abs(self.I_delta[0]) * unit
        )

        beta = np.sqrt(-d_voltage * d_current / (current * voltage))
        beta[np.real(beta) < 0] *= -1
        self.beta = beta
        self.Z_ref = np.sqrt(voltage * d_voltage / (current * d_current))
