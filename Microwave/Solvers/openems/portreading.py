# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What openEMS' rectangular waveguide port reads of a wave, and the wave it read.

``RectWGPort`` measures a voltage and a current by mode matching
(``openEMS/Common/processmodematch.cpp``). Each is the field on the port's
plane weighted by the mode, summed over the grid's nodes with each node's area,
and divided by the mode's own norm over the same nodes. The sum is not the
integral it stands for:

- it leaves out the nodes the engine excludes at the edges of the box;
- it reads a node beside a wall as the average of an edge in the guide and an
  edge in the metal (``FDTD/engine_interface_fdtd.cpp``, the node
  interpolation);
- it reads the current from the four magnetic edges around the node, two of
  them half a cell up and down the guide, where a travelling wave has another
  phase;
- it weights the current with the mode taken on the dual lines, half a cell
  from where the interpolated field sits.

Each port therefore reads the wave passing it at scales of its own: one for the
voltage, and for the current one for the wave travelling in and another for the
wave travelling out. The cells at the port's walls and either side of its plane
set them. Two ports whose cells are alike read alike, and the ratio between
them is right. Otherwise the matrix is scaled by the ratio and is not
reciprocal, one column gives out more power than a lossless structure took and
the other less, and a reflection shows where the voltage and the current scales
differ.

This module runs the engine's arithmetic on the mode of the guide the engine
built, and undoes it. A drawn face arrives on a grid line rather than where it
was drawn, so the guide as built is read off which electric edges the engine
ties to a conductor. The mode is the scheme's own on the lines the guide was
built on, so nothing in the correction is fitted. The reading is modelled only
where the engine built a rectangular guide of empty space bounded by lines it
ties, and where that guide is the one the port's box was drawn on. Anywhere else
:class:`NotRead` says why, and the port is left as the engine read it.

Along the guide the waves are split as the grid propagates the mode rather than
as the continuous guide would, since a split at the wrong impedance reads part
of a wave as travelling the other way. A mode varying across one wall is exactly
a ladder along the guide, and the ladder's constants are the grid's own:

- the wavenumber ``(2 / c dt) sin(w dt / 2)``, since the scheme steps in time by
  ``dt``. The engine samples the mode-matched current half a step after the
  voltage (``openems.cpp:572-573`` makes that probe dual-time, and
  ``FDTD/engine_interface_fdtd.h``'s ``GetTime`` adds the half step), so the
  timestep is twice the delay between their first samples;
- the cutoff, the eigenvalue of the scheme's second difference on the lines
  across the guide as built;
- a propagation constant from the two, a cell ``h`` turning a wave by ``2
  theta`` with ``sin(theta) = beta h / 2``.

Across the guide the field is the scheme's mode too: the eigenvector of that
second difference, zero at the walls, which is the sampled sine on equal cells
and departs from it where the cells across are graded. It is normalised over
the nodes' widths, the sum the scheme's power through the plane is, and openEMS
still weights it with the continuous sine it was given.

A port's plane usually sits where the cells change size, and there the change
itself reflects. The reading is carried along the ladder to the first node
whose two cells are alike, split there, and turned back to the plane. The guide
has to be the same rectangle over every node crossed, and the carry reaches no
further from the plane than the port's box is deep, which is the length of
guide the port was drawn to stand in. Where no such node comes first, the waves
are split at the plane from the side whose cells run on, with the grid's
propagation constant, and :attr:`Reading.carried` says so.

The node ranges come from the engine rather than from a model of its
exclusions. Each probe file's header states the range the probe summed over,
and versions of openEMS exclude different nodes: the source this adapter cites
drops the last node of the current's box (``processmodematch.cpp:102-105``),
and the engine the workbench was measured against keeps it.

The module uses numpy alone: no FreeCAD, no openEMS.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ... import units

#: Ohms. The wave impedance of free space, from the two constants it is made of.
FREE_SPACE_IMPEDANCE = units.VACUUM_PERMEABILITY * units.SPEED_OF_LIGHT

#: Where the run's provenance keeps how each waveguide port was read: per port,
#: ``corrected``, and where it is false the ``reason``, with ``refused`` on the
#: port whose reading could not be modelled.
KEY = "port_reading"

Point = tuple[float, float, float]

#: Whether the engine ties the electric field at a point to a conductor. The
#: points asked are the positions of electric edges, where
#: ``Operator::CalcPEC_Range`` asks the same question.
Conducts = Callable[[Point], bool]

_CORNER = re.compile(r"^% (start|stop)-coordinates: .* -> \[(\d+),(\d+),(\d+)\]\s*$")


class NotRead(Exception):
    """The port's reading cannot be modelled. The message says why."""


def unread(provenance: Mapping[str, Any]) -> str | None:
    """The warning a run makes where it left its waveguide ports as openEMS read
    them, naming each port that could not be read; ``None`` where it did not."""
    refused = [
        f"port {number}: its waveguide reading is left as openEMS returned it, because "
        f"{reading['reason']}"
        for number, reading in (provenance.get(KEY) or {}).items()
        if reading.get("refused")
    ]
    if not refused:
        return None
    return (
        "; ".join(refused) + ". So is every other waveguide port's in this run. The "
        "voltage and current are then read at scales the cells at each port's walls set, "
        "so a transmission between ports whose walls end differently is read at the "
        "ratio of the two, and a port whose voltage and current scales differ shows a "
        "reflection that is not there"
    )


@dataclass(frozen=True)
class Nodes:
    """The nodes one probe summed over, by index, as the engine reported them."""

    start: tuple[int, int, int]
    stop: tuple[int, int, int]


def nodes(header: str) -> Nodes:
    """The node range stated in the header of a probe file openEMS wrote.

    ``ProcessIntegral::InitProcess`` writes the range after the probe has made
    its exclusions, so it is the range the sum ran over.
    """
    corners: dict[str, tuple[int, int, int]] = {}
    for line in header.splitlines():
        found = _CORNER.match(line)
        if found:
            i, j, k = (int(found.group(n)) for n in (2, 3, 4))
            corners[found.group(1)] = (i, j, k)
    if set(corners) != {"start", "stop"}:
        raise NotRead("its probe file does not state the nodes the probe summed over")
    return Nodes(corners["start"], corners["stop"])


@dataclass(frozen=True)
class Reading:
    """How one port read the wave passing it.

    The port recorded ``voltage_in * a + voltage_out * b`` as its voltage and
    ``(inward * a - outward * b) / impedance`` as its current, where ``a`` is
    the modal voltage of the wave travelling into the structure and ``b`` that
    of the wave travelling out, both normalised over the guide as built and
    taken at the port's plane.
    """

    #: The voltage's scale across the plane.
    voltage: float
    #: The current's scale across the plane, before the half cells along it.
    current: float
    #: Per frequency, the current's scale for the wave travelling in.
    inward: npt.NDArray[np.complex128]
    #: Per frequency, the current's scale for the wave travelling out.
    outward: npt.NDArray[np.complex128]
    #: Per frequency, in ohms, what the waves are referred to: the mode
    #: impedance of the guide as built, over the share of a wave's power the
    #: grid carries where the waves are split, so that ``|a|^2 / (2 Re
    #: impedance)`` is the power the scheme carries in the wave ``a``.
    impedance: npt.NDArray[np.complex128]
    #: The guide as built along the two mode axes, in drawing units.
    guide: tuple[float, float]
    #: Per frequency, whether the mode propagates in the guide the grid holds.
    propagates: npt.NDArray[np.bool_]
    #: Per frequency, the voltage's scale for the wave travelling in, which is
    #: the voltage's scale itself where the waves are split at the plane.
    voltage_in: npt.NDArray[np.complex128]
    #: Per frequency, the voltage's scale for the wave travelling out.
    voltage_out: npt.NDArray[np.complex128]
    #: How many cells along its guide the reading was carried to be split
    #: where the cells are alike, or ``None`` where it was split at the plane
    #: across a change of cell.
    carried: int | None
    #: The engine's timestep the grid's wavenumber was taken at, in seconds.
    timestep: float
    #: The index of the line on the propagation axis the port read on.
    plane: int
    #: Per frequency, the grid's angular frequency at that timestep.
    angular: npt.NDArray[np.float64]
    #: Per frequency, the mode's propagation constant as the grid carries it.
    beta: npt.NDArray[np.complex128]
    #: ``+1`` where the wave into the structure moves up the axis, else ``-1``.
    direction: int
    #: The cutoff wavenumbers the grid holds across the guide as built, on the
    #: axis the mode varies on, and across its uniform axis, per metre, lowest
    #: first. A mode varying ``p`` times on the one and ``q`` on the other is cut
    #: off at the root of the sum of their squares, since the scheme's
    #: difference across the rectangle separates.
    cutoffs: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]

    def waves(
        self, voltage: npt.ArrayLike, current: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        """The modal voltages travelling in and out, from what the port recorded.

        With every scale one this is openEMS' own split,
        ``(u + i Z) / 2`` in and ``u`` less that out.
        """
        v = np.asarray(voltage, dtype=complex)
        zi = np.asarray(current, dtype=complex) * self.impedance
        into, out = self.voltage_in, self.voltage_out
        # A point exactly on the cutoff has no scales, which the driver refuses
        # by name.
        with np.errstate(divide="ignore", invalid="ignore"):
            incident = (zi * out + self.outward * v) / (self.inward * out + self.outward * into)
            return incident, (v - into * incident) / out


def read(
    lines: Sequence[npt.ArrayLike],
    axis: int,
    mode: tuple[int, int],
    size: tuple[float, float],
    box: tuple[Point, Point],
    voltage: Nodes,
    current: Nodes,
    direction: int,
    conducts: Conducts,
    frequency: npt.ArrayLike,
    length_unit: float,
    timestep: float,
) -> Reading:
    """How a ``RectWGPort`` read the wave passing it.

    :param lines: The grid lines on each axis, in drawing units.
    :param axis: The propagation axis.
    :param mode: ``(M, N)`` as openEMS was given them, bound to the axes
        ``axis + 1`` and ``axis + 2``.
    :param size: The ``a`` and ``b`` openEMS was given, in drawing units. The
        mode it weights the field with is built from these.
    :param box: The port's ``start`` and ``stop`` corners, in drawing units. The
        mode's coordinates are counted from ``start``.
    :param voltage: The nodes the voltage probe summed over.
    :param current: The nodes the current probe summed over.
    :param direction: ``+1`` where the wave travelling into the structure moves
        up ``axis``, ``-1`` where it moves down.
    :param conducts: The engine's answer at an electric edge.
    :param frequency: In hertz.
    :param length_unit: Metres per drawing unit.
    :param timestep: The engine's timestep, in seconds.
    :raises NotRead: Where the guide at the plane is not one this models.
    """
    grid = [np.asarray(line, dtype=float) for line in lines]
    m, n = mode
    if (m == 0) == (n == 0):
        raise NotRead(
            f"its mode TE{m}{n} has a field varying across both walls, and only a "
            "mode varying across one is modelled"
        )
    first, second = (axis + 1) % 3, (axis + 2) % 3
    # The field varies across one axis and lies along the other, on which it is
    # uniform.
    across, uniform = (first, second) if m else (second, first)
    order = m or n
    drawn = size[0] if m else size[1]

    plane = voltage.start[axis]
    if {voltage.stop[axis], current.start[axis], current.stop[axis]} != {plane}:
        raise NotRead("its probes did not sum over one plane")
    x = grid[axis]
    if not 1 < plane < len(x) - 2:
        raise NotRead("its plane lies within two lines of the edge of the grid")

    s, u = grid[across], grid[uniform]
    centre = (
        (voltage.start[across] + voltage.stop[across]) // 2,
        (voltage.start[uniform] + voltage.stop[uniform]) // 2,
    )
    left, right, bottom, top = _walls(grid, axis, across, uniform, plane, centre, conducts)
    # A box drawn on the guide has each side within a cell of the wall the
    # engine built for it: a drawn face arrives on the nearer line along the
    # uniform axis and on the first line inside the metal across it. A side
    # further in draws a mode the guide does not have, which the port also
    # launches, and a side further out belongs to another guide.
    origin = box[0]
    for dim, walls in ((across, (left, right)), (uniform, (bottom, top))):
        line = grid[dim]
        for side, wall in zip(sorted((box[0][dim], box[1][dim])), walls, strict=True):
            near = line[max(wall - 1, 0)], line[min(wall + 1, len(line) - 1)]
            if not near[0] <= side <= near[1]:
                raise NotRead(
                    "the guide the engine built at its plane is not the one its box was drawn "
                    "on, which is also what a wall drawn as a sheet across the field gives, "
                    "since a sheet ties no edge crossing it"
                )
    wide, high = s[right] - s[left], u[top] - u[bottom]

    def shape(at: npt.NDArray[np.float64], start: float, length: float) -> npt.NDArray[np.float64]:
        return np.sin(order * math.pi * (at - start) / length)

    # The scheme's mode across the guide, on the lines from wall to wall, and
    # its cutoff. Both the voltage and the node's current lie on those lines.
    cutoff, mode_across = _mode(s[left : right + 1] * length_unit, order)
    cutoffs = (
        _modes(s[left : right + 1] * length_unit)[0],
        _modes(u[bottom : top + 1] * length_unit)[0],
    )

    def across_the_guide(js: npt.NDArray[np.intp]) -> npt.NDArray[np.float64]:
        inside = (js >= left) & (js <= right)
        return np.where(inside, mode_across[np.clip(js - left, 0, right - left)], 0.0)

    # The mode of a unit modal voltage is the field over this, the square root
    # of the field's own sum over the guide as built, each node weighted by its
    # width.
    cells_across = np.diff(s[left : right + 1])
    widths_across = (cells_across[:-1] + cells_across[1:]) / 2
    unit = math.sqrt(float(np.sum(mode_across[1:-1] ** 2 * widths_across)) * high)

    # The voltage: the uniform component on the primary lines across the guide,
    # interpolated along its own axis between the edge below each node and the
    # one above it.
    js = _span(voltage, across)
    ks = _span(voltage, uniform)
    field = across_the_guide(js)
    weight = shape(s[js], origin[across], drawn)
    width = _node_width(s, js, dual=False)
    height = _node_width(u, ks, dual=False)
    edges = _edges_inside(u, bottom, top)
    along = np.array([_interpolated(u, edges, k) for k in ks])
    reads_v = np.sum(field * weight * width) * np.sum(along * height)
    norm_v = math.sqrt(np.sum(weight**2 * width) * np.sum(height))
    scale_v = float(reads_v / norm_v / unit)

    # The current: the magnetic component across the guide, on the primary
    # lines of its own axis and averaged over the half cells either side along
    # the other two. The mode is taken on the dual lines.
    js = _span(current, across)
    ks = _span(current, uniform)
    field = across_the_guide(js)
    weight = shape(_dual(s, js), origin[across], drawn)
    width = _node_width(s, js, dual=True)
    height = _node_width(u, ks, dual=True)
    cells = np.append(edges, 0.0)
    along = np.array([0.0 if k in (0, len(u) - 1) else 0.5 * (cells[k] + cells[k - 1]) for k in ks])
    reads_i = np.sum(field * weight * width) * np.sum(along * height)
    norm_i = math.sqrt(np.sum(weight**2 * width) * np.sum(height))
    scale_i = float(reads_i / norm_i / unit)

    # The grid's own wavenumber and propagation constant, which the module's
    # opening says how each is taken.
    angular = 2 / timestep * np.sin(math.pi * np.asarray(frequency, dtype=float) * timestep)
    wavenumber = angular / units.SPEED_OF_LIGHT
    beta = np.sqrt(wavenumber.astype(complex) ** 2 - cutoff**2)
    guide = (wide, high) if m else (high, wide)
    built = (left, right, bottom, top)

    def same_guide(node: int) -> bool:
        try:
            return _walls(grid, axis, across, uniform, node, centre, conducts) == built
        except NotRead:
            return False

    # The port's box runs from its source to its plane, and the carry reaches
    # no further than that from the plane.
    source = int(np.argmin(np.abs(x - box[0][axis])))
    settled = _alike(x, plane, direction, same_guide, abs(plane - source))
    # A point exactly on the cutoff has beta zero and no impedance, which the
    # driver refuses by name.
    with np.errstate(divide="ignore", invalid="ignore"):
        mode_impedance = (wavenumber * FREE_SPACE_IMPEDANCE / beta).astype(complex)
        if settled is None:
            inward, outward, share = _along(x, plane, direction, beta, length_unit)
            ones = np.full(beta.shape, scale_v, dtype=complex)
            return Reading(
                voltage=scale_v,
                current=scale_i,
                inward=scale_i * inward / share,
                outward=scale_i * outward / share,
                impedance=mode_impedance / share,
                guide=guide,
                propagates=wavenumber > cutoff,
                voltage_in=ones,
                voltage_out=ones,
                carried=None,
                timestep=timestep,
                plane=plane,
                angular=angular,
                beta=beta,
                direction=direction,
                cutoffs=cutoffs,
            )
        scales = _carried(x, plane, settled, direction, beta, angular, mode_impedance, length_unit)
        voltage_in, voltage_out, inward, outward, share = scales
        return Reading(
            voltage=scale_v,
            current=scale_i,
            inward=scale_i * inward,
            outward=scale_i * outward,
            impedance=mode_impedance / share,
            guide=guide,
            propagates=wavenumber > cutoff,
            voltage_in=scale_v * voltage_in,
            voltage_out=scale_v * voltage_out,
            carried=abs(settled - plane),
            timestep=timestep,
            plane=plane,
            angular=angular,
            beta=beta,
            direction=direction,
            cutoffs=cutoffs,
        )


#: Two cells whose sizes differ by a share this small are alike. The envelope
#: holds each line to twelve significant figures (``model.CANONICAL_DIGITS``),
#: so a cell's size is good to about ``1e-12`` of the coordinates at its ends,
#: which stays under this share while a coordinate lies fewer than a hundred
#: thousand cells from the origin. A split at a node whose cells differ by this
#: share reads a wave out of a wave in by less than the share itself.
_ALIKE = 1e-6


def _along(
    x: npt.NDArray[np.float64],
    plane: int,
    direction: int,
    beta: npt.NDArray[np.complex128],
    length_unit: float,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """The current's scale for the wave travelling in and out, from the half
    cells either side of the plane, and the share of a wave's power the grid
    carries.

    The node's magnetic field is the mean of the edges half a cell up and down
    the guide. Where the two cells differ, the change of cell reflects a little,
    and the waves either side of the plane are not one wave, so the mean is not
    the mean of one wave taken at two points. It is written instead from the
    waves on one side: Faraday's law over the cell on that side gives the edge
    there, ``exp(-j theta)`` of the wave with ``sin(theta) = beta h / 2`` for a
    cell ``h``, and Ampere's law at the node gives the edge on the other side
    from it, ``j beta h*`` of the node's field further on, ``h*`` being the node's
    own width. The two edges differ by ``j beta h*`` of the node's field, and
    the mean takes half of that. The side taken is the one whose cells run on
    unchanged, where the waves are those of a uniform guide. On equal cells the
    two sides agree, and each wave reads ``cos(theta)`` of itself.

    The power the scheme carries through the plane is the real part of the
    node's field against the conjugate of either edge, and so of their mean. The
    two scales are each other's conjugates, and a wave's power comes to
    ``cos(theta)`` of what the mode impedance alone gives. That share is
    returned third, and the impedance the waves are referred to is divided by
    it, so that two ports on cells of different size meet in one matrix.
    """
    below = (x[plane] - x[plane - 1], x[plane - 1] - x[plane - 2])
    above = (x[plane + 1] - x[plane], x[plane + 2] - x[plane + 1])
    structure, outside = (above, below) if direction > 0 else (below, above)

    def uneven(cells: tuple[float, float]) -> float:
        return abs(math.log(cells[0] / cells[1]))

    if uneven(structure) <= uneven(outside) + _ALIKE:
        side, sign = structure, 1
    else:
        side, sign = outside, -1
    theta = np.arcsin(beta * side[0] * length_unit / 2)
    width = beta * (structure[0] + outside[0]) * length_unit / 4
    inward = np.exp(-1j * sign * theta) + 1j * sign * width
    outward = np.exp(1j * sign * theta) - 1j * sign * width
    return inward, outward, np.cos(theta)


def _modes(
    lines: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Every mode the scheme holds across the lines in metres from one wall to
    the other, lowest first: the cutoff wavenumbers, and one row per mode of its
    field on each line, zero at the walls and of the sign of the continuous
    mode's sine.

    Both come from the scheme's second difference with the field tied at both
    walls: a node's field against its neighbours' over the cells between, over
    the node's own width, half of each cell beside it. A cutoff is the square
    root of an eigenvalue and the field its eigenvector. The operator is
    symmetric once each node is weighted by the square root of its width, and it
    is solved in that form. On equal cells the cutoff of order ``n`` is ``(2 /
    h) sin(n pi h / 2 a)``, tending to ``n pi / a``, and the field is the sine
    sampled on the lines.
    """
    cells = np.diff(lines)
    widths = (cells[:-1] + cells[1:]) / 2
    root = np.sqrt(widths)
    coupling = 1 / (cells[1:-1] * root[:-1] * root[1:])
    operator = (
        np.diag((1 / cells[:-1] + 1 / cells[1:]) / widths)
        - np.diag(coupling, 1)
        - np.diag(coupling, -1)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    fields = np.zeros((widths.size, len(lines)))
    fields[:, 1:-1] = (eigenvectors / root[:, None]).T
    orders = np.arange(1, widths.size + 1)[:, None]
    sines = np.sin(orders * math.pi * (lines - lines[0]) / (lines[-1] - lines[0]))
    fields *= np.where(np.sum(fields * sines, axis=1) < 0, -1.0, 1.0)[:, None]
    return np.sqrt(eigenvalues), fields


def _mode(lines: npt.NDArray[np.float64], order: int) -> tuple[float, npt.NDArray[np.float64]]:
    """The cutoff and the field of the mode of ``order`` :func:`_modes` gives."""
    cutoffs, fields = _modes(lines)
    if order > cutoffs.size:
        raise NotRead(
            f"its guide holds {len(lines) - 1} cells across, and a mode of order {order} needs more"
        )
    return float(cutoffs[order - 1]), fields[order - 1]


def _alike(
    x: npt.NDArray[np.float64],
    plane: int,
    direction: int,
    same_guide: Callable[[int], bool],
    reach: int,
) -> int | None:
    """The first node from ``plane`` toward the structure whose two cells are
    alike, no more than ``reach`` cells on and crossing only nodes where the
    guide is the one at the plane, or ``None`` where the grid, the guide or the
    reach ends first."""
    node = plane
    while 1 <= node < len(x) - 1 and abs(node - plane) <= reach:
        if node != plane and not same_guide(node):
            return None
        cells = (x[node] - x[node - 1], x[node + 1] - x[node])
        if abs(math.log(cells[0] / cells[1])) <= _ALIKE:
            return node
        node += direction
    return None


def predicted(
    here: Reading,
    there: Reading,
    x: npt.ArrayLike,
    length_unit: float,
    voltage: npt.ArrayLike,
    current: npt.ArrayLike,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """The waves ``here`` would read at its plane where the guide between the
    two planes carries the mode alone: what ``there``'s probes recorded, over
    its scales, carried along the :func:`ladder` to ``here``'s plane, put on
    ``here``'s scales and split as ``here`` splits.

    :param x: The grid lines on the propagation axis, in drawing units.
    """
    lines = np.asarray(x, dtype=float)
    direction = here.direction
    v = np.asarray(np.asarray(voltage) / there.voltage, dtype=complex)
    mean = np.asarray(direction * np.asarray(current) / there.current, dtype=complex)
    v, mean = ladder(lines, there.plane, here.plane, here.beta, here.angular, length_unit, v, mean)
    return here.waves(here.voltage * v, here.current * direction * mean)


def ladder(
    x: npt.NDArray[np.float64],
    start: int,
    stop: int,
    beta: npt.NDArray[np.complex128],
    angular: npt.NDArray[np.float64],
    length_unit: float,
    voltage: npt.NDArray[np.complex128],
    current: npt.NDArray[np.complex128],
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """A node's voltage and the mean of the currents either side of it, up the
    axis, carried from node ``start`` to node ``stop`` along the guide.

    Along the guide the mode is a ladder: a cell ``h`` is a series ``j w mu h``
    and a node a shunt ``j beta^2 h* / (w mu)``, ``h*`` the node's width, with
    ``w`` and ``beta`` the grid's own - Faraday's law over each cell and
    Ampere's at each node, the scheme's own equations for the mode.

    :param x: The grid lines on the propagation axis, in drawing units.
    """
    cells = np.diff(x) * length_unit
    series = 1j * angular * units.VACUUM_PERMEABILITY
    shunt = 1j * beta**2 / (angular * units.VACUUM_PERMEABILITY)

    def admittance(node: int) -> npt.NDArray[np.complex128]:
        return shunt * (cells[node - 1] + cells[node]) / 2

    v = np.asarray(voltage, dtype=complex)
    mean = np.asarray(current, dtype=complex)
    # The currents on the edges below and above the node, along the axis.
    below = mean + admittance(start) * v / 2
    above = mean - admittance(start) * v / 2
    node = start
    while node != stop:
        if stop > node:
            v = v - series * cells[node] * above
            node += 1
            below, above = above, above - admittance(node) * v
        else:
            v = v + series * cells[node - 1] * below
            node -= 1
            below, above = below + admittance(node) * v, below
    return v, (below + above) / 2


def _carried(
    x: npt.NDArray[np.float64],
    plane: int,
    settled: int,
    direction: int,
    beta: npt.NDArray[np.complex128],
    angular: npt.NDArray[np.float64],
    impedance: npt.NDArray[np.complex128],
    length_unit: float,
) -> tuple[npt.NDArray[np.complex128], ...]:
    """The voltage's scales for the wave in and out, the current's, and the
    share of a wave's power the grid carries, for a reading carried from
    ``plane`` to ``settled`` and split there.

    The node's voltage and the mean of the currents either side of it, which is
    what the port recorded, are carried along the :func:`ladder` to the node
    whose cells are alike. There each wave reads
    ``cos(theta)`` of itself, ``sin(theta) = beta h / 2``, and the waves are
    turned back to the plane by ``2 theta`` a cell. Every step is linear, so the
    reading at the plane is two scales on the voltage and two on the current.
    """
    cells = np.diff(x) * length_unit

    def carry(voltage: complex, current: complex) -> tuple[npt.NDArray[np.complex128], ...]:
        """The node voltage and the mean current into the structure at
        ``settled``, from those at ``plane``."""
        v = np.full(beta.shape, voltage, dtype=complex)
        mean = np.full(beta.shape, direction * current, dtype=complex)
        v, mean = ladder(x, plane, settled, beta, angular, length_unit, v, mean)
        return v, direction * mean

    # The carry of a unit voltage and of a unit current, as the columns of the
    # matrix taking the plane's voltage and current to the settled node's.
    (v_v, i_v), (v_i, i_i) = carry(1.0, 0.0), carry(0.0, 1.0)
    lo, hi = sorted((plane, settled))
    turn = np.exp(1j * sum((2 * np.arcsin(beta * cells[k] / 2) for k in range(lo, hi)), 0j))
    share = np.cos(np.arcsin(beta * cells[settled] / 2))
    # At the settled node a wave's voltage is a + b and its current
    # share * (a - b) / impedance; the plane's voltage and current follow by
    # the inverse of the carry.
    det = v_v * i_i - v_i * i_v
    n_vv, n_vi, n_iv, n_ii = i_i / det, -v_i / det, -i_v / det, v_v / det
    ratio = share / impedance
    voltage_in = (n_vv + n_vi * ratio) / turn
    voltage_out = (n_vv - n_vi * ratio) * turn
    inward = (n_iv / ratio + n_ii) / turn
    outward = (n_ii - n_iv / ratio) * turn
    return voltage_in, voltage_out, inward, outward, share


def _walls(
    grid: list[npt.NDArray[np.float64]],
    axis: int,
    across: int,
    uniform: int,
    plane: int,
    centre: tuple[int, int],
    conducts: Conducts,
) -> tuple[int, int, int, int]:
    """The line indices of the guide's four walls as the engine tied them.

    The mode's electric field lies along the uniform axis alone, so the edges
    along that axis decide where it can be. A wall across the guide is the first
    line out from the centre on which those edges are tied, and the field lying
    in that line is then tied. Along the uniform axis the guide is the run of
    free edges through the centre, and each end of the run has to be a line on
    which the engine ties the field lying in it: the edges across the guide.
    Where a drawn face falls nearer the line outside the metal than the line
    inside, the edge crossing it is tied and the line outside is not, so the
    guide ends between two lines. The magnetic field there drives the field in
    the free line, and the guide the engine solves has no wall this models.

    The rectangle is then checked on this plane and on the plane either side,
    which is what the current's average reaches: every edge along the uniform
    axis on a wall or past an end tied, every one between them free, and every
    edge across the guide on an end tied.
    """
    s, u = grid[across], grid[uniform]

    def tied(o: int, across_at: float, uniform_at: float) -> bool:
        point = [0.0, 0.0, 0.0]
        point[axis] = float(grid[axis][plane + o])
        point[across] = across_at
        point[uniform] = uniform_at
        return conducts((point[0], point[1], point[2]))

    def edge(o: int, j: int, k: int) -> bool:
        """Whether the edge along the uniform axis from line ``k`` is tied."""
        return tied(o, float(s[j]), float(_dual(u, k)))

    j0, k0 = centre
    if not (0 < j0 < len(s) - 1 and 0 <= k0 < len(u) - 1):
        raise NotRead("its probe box has no interior on this grid")
    # The walls are looked for out from an edge the field is free on, so they
    # bound a guide of some width and height. From a tied one they would bound
    # nothing, and every check below would pass over an empty range.
    if edge(0, j0, k0):
        raise NotRead(_NOT_A_RECTANGLE)

    def outward(start: int, last: int, step: int) -> int:
        at = start
        while not edge(0, at, k0):
            if at == last:
                raise NotRead(_NOT_A_CONDUCTOR)
            at += step
        return at

    left = outward(j0, 0, -1)
    right = outward(j0, len(s) - 1, 1)
    bottom, top = k0, k0 + 1
    while bottom > 0 and not edge(0, j0, bottom - 1):
        bottom -= 1
    while top < len(u) - 1 and not edge(0, j0, top):
        top += 1

    def across_tied(o: int, j: int, k: int) -> bool:
        """Whether the edge across the guide from line ``j``, lying in line
        ``k`` of the uniform axis, is tied."""
        return tied(o, float(_dual(s, j)), float(u[k]))

    for end in (bottom, top):
        if not across_tied(0, j0, end):
            raise NotRead(
                _NOT_A_CONDUCTOR
                if end in (0, len(u) - 1)
                else "its guide ends between two grid lines rather than on one the engine ties"
            )

    for o in (-1, 0, 1):
        for j in range(left, right + 1):
            wall = j in (left, right)
            for k in range(bottom, top):
                if edge(o, j, k) != wall:
                    raise NotRead(_NOT_A_RECTANGLE)
            for k in (bottom - 1, top):
                if 0 <= k < len(u) - 1 and not wall and not edge(o, j, k):
                    raise NotRead(_NOT_A_RECTANGLE)
            if j < right and not (across_tied(o, j, bottom) and across_tied(o, j, top)):
                raise NotRead(_NOT_A_RECTANGLE)
    return left, right, bottom, top


_NOT_A_CONDUCTOR = "a side of its guide is not a conductor"


_NOT_A_RECTANGLE = (
    "the metal the engine built at its plane, or within a cell of it, is not a rectangular guide"
)


def _span(box: Nodes, dim: int) -> npt.NDArray[np.intp]:
    return np.arange(box.start[dim], box.stop[dim] + 1)


def _dual(line: npt.NDArray[np.float64], k: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """``Operator::GetDiscLine`` on the dual mesh: the middle of each cell, and
    half a cell past the last line for the last."""
    k = np.asarray(k)
    last = len(line) - 1
    inner = np.minimum(k, last - 1)
    middle = 0.5 * (line[inner] + line[inner + 1])
    beyond = line[last] + 0.5 * (line[last] - line[last - 1])
    return np.where(k < last, middle, beyond)


def _node_width(
    line: npt.NDArray[np.float64], k: npt.NDArray[np.intp], dual: bool
) -> npt.NDArray[np.float64]:
    """``Operator::GetNodeWidth``: the dual cell around a primary node, and the
    primary cell above a dual one (``operator.h:174``, through
    ``GetDiscDelta``)."""
    last = len(line) - 1
    if dual:
        inner = np.minimum(k, last - 1)
        return np.where(k < last, line[inner + 1] - line[inner], line[last] - line[last - 1])
    below = _dual(line, np.maximum(k - 1, 0))
    return np.where(k > 0, _dual(line, k) - below, line[1] - line[0])


def _edges_inside(line: npt.NDArray[np.float64], bottom: int, top: int) -> npt.NDArray[np.float64]:
    """One for each edge between two lines inside the guide, and zero for the
    rest: an edge on the uniform axis carries the field only there."""
    index = np.arange(len(line) - 1)
    return ((index >= bottom) & (index < top)).astype(float)


def _interpolated(line: npt.NDArray[np.float64], edges: npt.NDArray[np.float64], k: int) -> float:
    """The node value ``Engine_Interface_FDTD`` gives an electric component
    along its own axis: the edge below and the edge above, each weighted by
    the length of the other, and one of them alone at either end of the grid."""
    last = len(line) - 1
    if k == last:
        return float(edges[k - 1])
    if k == 0:
        return float(edges[0])
    up, down = line[k + 1] - line[k], line[k] - line[k - 1]
    share = up / (up + down)
    return float(edges[k] * (1 - share) + edges[k - 1] * share)
