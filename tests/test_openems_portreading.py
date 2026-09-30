# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a rectangular waveguide port reads, run on guides whose answer is arithmetic.

No solver: each case is a grid and a set of tied edges, and the property held is
one the engine's arithmetic gives in closed form. A reading solved and scored
against reciprocity and power is the stepped-guide gate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Microwave import units
from Microwave.Solvers.openems import portreading
from Microwave.Solvers.openems.portreading import Nodes, NotRead

BROAD, NARROW = 10.7, 4.3
LENGTH_UNIT = 1e-3
FREQUENCY = np.linspace(20e9, 26e9, 7)
#: The plane's index on the propagation axis.
PLANE = 20
#: Seconds. The engine's timestep on a grid of these cells, near its own bound.
TIMESTEP = 5e-13


def along(below: float = 0.5, above: float = 0.5) -> np.ndarray:
    """Lines on the propagation axis, with the two cells either side of the
    plane as given."""
    lower = PLANE * below - np.arange(PLANE, -1, -1) * below
    upper = PLANE * below + np.arange(1, PLANE + 1) * above
    return np.concatenate([lower, upper])


def walls_of(lines: list[np.ndarray], *boxes, open_side: int | None = None):
    """The engine's answer on a domain whose four sides across the guide are PEC,
    with metal boxes in it. A box contains its own faces, as a box primitive
    does."""

    def conducts(point) -> bool:
        for dim in (1, 2):
            for end, line in ((0, lines[dim][0]), (1, lines[dim][-1])):
                if point[dim] == line and open_side != 2 * dim + end:
                    return True
        return any(all(lo[d] <= point[d] <= hi[d] for d in range(3)) for lo, hi in boxes)

    return conducts


def read(
    lines,
    voltage=None,
    current=None,
    conducts=None,
    direction=1,
    mode=(1, 0),
    size=None,
    frequency=FREQUENCY,
    source=0.0,
    plane=PLANE,
):
    """A ``RectWGPort`` on the whole cross-section, excluded at the domain's own
    lines as every release of the engine does, reading on line ``plane``, its
    box starting at ``source`` along the guide."""
    across, uniform = len(lines[1]), len(lines[2])
    size = size or (float(lines[1][-1]), float(lines[2][-1]))
    full = Nodes((plane, 1, 1), (plane, across - 2, uniform - 2))
    return portreading.read(
        lines,
        0,
        mode,
        size,
        ((source, 0.0, 0.0), (1.0, *size)),
        voltage or full,
        current or full,
        direction,
        conducts or walls_of(lines),
        frequency,
        LENGTH_UNIT,
        TIMESTEP,
    )


def guide(narrow_cells: int = 12, broad_cells: int = 30) -> list[np.ndarray]:
    return [
        along(),
        np.linspace(0.0, BROAD, broad_cells + 1),
        np.linspace(0.0, NARROW, narrow_cells + 1),
    ]


class TestTheNodesComeFromTheEngine:
    #: The head of a probe file openEMS wrote for a guide's second port.
    HEADER = (
        "% time-domain current mode matching by openEMS v0.0.36-157-gc4ce357"
        " @Sun Sep 13 12:58:05 2026\n"
        "% start-coordinates: (0.0430783,0.000374242,0.000354474) m -> [123,1,1]\n"
        "% stop-coordinates: (0.0430783,0.00802135,0.00394553) m -> [123,26,11]\n"
        "% t/s\tcurrent\tmode_purity\n"
    )

    def test_the_range_is_read_off_the_header(self):
        assert portreading.nodes(self.HEADER) == Nodes((123, 1, 1), (123, 26, 11))

    def test_a_header_that_does_not_state_it_is_not_read(self):
        with pytest.raises(NotRead, match="does not state the nodes"):
            portreading.nodes(self.HEADER.replace("stop-coordinates", "stop"))


class TestTheWaves:
    def test_with_every_scale_one_they_are_openems_own_split(self):
        """``Port.CalcPort``: ``(u + i Z) / 2`` in, ``u`` less that out."""
        z = np.full(3, 500.0 + 0j)
        ones = np.ones(3, dtype=complex)
        reading = portreading.Reading(
            1.0,
            1.0,
            ones,
            ones,
            z,
            (BROAD, NARROW),
            ones.real > 0,
            ones,
            ones,
            0,
            TIMESTEP,
            PLANE,
            ones.real,
            ones,
            1,
            (np.empty(0), np.empty(0)),
        )
        u = np.array([1.0 + 0.5j, -0.3j, 2.0])
        i = np.array([0.002 - 0.001j, 0.004j, -0.001])
        incident, reflected = reading.waves(u, i)
        np.testing.assert_allclose(incident, 0.5 * (u + i * z), rtol=1e-15, atol=0.0)
        np.testing.assert_allclose(reflected, u - 0.5 * (u + i * z), rtol=1e-15, atol=0.0)

    def test_what_the_port_recorded_of_them_gives_them_back(self):
        """The reading's own statement of what was recorded, inverted. Unequal
        cells either side of the plane, so the wave in and the wave out are
        read at different scales."""
        lines = guide()
        lines[0] = along(below=0.4, above=0.6)
        reading = read(lines)
        a = np.linspace(1.0, 2.0, FREQUENCY.size) * np.exp(0.3j)
        b = np.linspace(0.2, 0.1, FREQUENCY.size) * np.exp(-1.1j)
        u = reading.voltage * (a + b)
        i = (reading.inward * a - reading.outward * b) / reading.impedance
        incident, reflected = reading.waves(u, i)
        np.testing.assert_allclose(incident, a, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(reflected, b, rtol=1e-12, atol=0.0)


class TestAcrossTheNarrowWall:
    """The field is uniform along the narrow wall, so a node there reads the
    height it covers. With ``N`` equal cells and the two nodes on the domain's
    walls left out, the nodes cover ``N - 1`` of them, and a sum over a share
    ``s`` of the height reads ``sqrt(s)`` of the mode."""

    def test_the_voltage_reads_the_share_of_the_height_its_nodes_cover(self):
        coarse, fine = read(guide(narrow_cells=8)), read(guide(narrow_cells=24))
        expected = math.sqrt((1 - 1 / 8) / (1 - 1 / 24))
        assert coarse.voltage / fine.voltage == pytest.approx(expected, rel=1e-12, abs=0.0)

    def test_the_current_follows_the_range_the_engine_reported(self):
        """The source this adapter cites drops the last node of the current's
        box, and the engine it was measured against keeps it; the reading takes
        whichever the header states."""
        lines = guide(narrow_cells=12)
        kept = read(lines)
        dropped = read(lines, current=Nodes((PLANE, 1, 1), (PLANE, 29, 10)))
        expected = math.sqrt((12 - 1) / (12 - 2))
        assert kept.current / dropped.current == pytest.approx(expected, rel=1e-12, abs=0.0)
        assert kept.voltage == dropped.voltage


class TestAtADrawnWall:
    def test_the_node_on_it_reads_the_share_its_two_edges_give(self):
        """``N`` equal cells ``h`` up to a line inside the metal, and a cell of
        ``2h`` above it. The node on that line reads its edge below at the
        share ``2h / 3h`` and the one above, which is tied, not at all, over a
        width of ``1.5h``; its magnetic field is half the cell below over the
        cell above. So the voltage covers ``N h`` of a norm of ``(N + 1/2) h``,
        and the current ``N h`` of ``(N + 1) h``, against a guide ``N h`` high.
        A guide on the domain's walls with ``M`` cells reads ``sqrt(1 - 1/M)``
        of both, which is the reference the drawn one is taken against."""
        cell, n, m = 0.25, 8, 12
        wall = n * cell
        z = np.concatenate([np.arange(n) * cell, [wall], wall + 2 * cell * np.arange(1, 6)])
        lines = [along(), np.linspace(0.0, BROAD, 31), z]
        metal = ((-1.0, -1.0, wall - cell / 3), (99.0, 99.0, 99.0))
        box = Nodes((PLANE, 1, 1), (PLANE, 29, n))
        drawn = read(lines, box, box, walls_of(lines, metal), size=(BROAD, wall - cell / 3))
        domain = read(guide(narrow_cells=m))
        below = math.sqrt(1 - 1 / m)
        assert drawn.guide[1] == pytest.approx(wall, abs=1e-12)
        assert drawn.voltage / domain.voltage == pytest.approx(
            math.sqrt(n / (n + 0.5)) / below, rel=1e-12, abs=0.0
        )
        assert drawn.current / domain.current == pytest.approx(
            math.sqrt(n / (n + 1)) / below, rel=1e-12, abs=0.0
        )


class TestAcrossTheBroadWall:
    def test_the_current_takes_the_mode_half_a_cell_from_where_it_reads_the_field(self):
        """On ``N`` equal cells the voltage sums ``sin(pi j / N)`` against itself,
        which comes to ``N / 2``. The current sums the same field against
        ``sin(pi (j + 1/2) / N)``, which comes to ``(N / 2) cos(pi / 2N)``, and
        normalises that over ``(N - 1 + cos(pi / N)) / 2``. Along the narrow
        wall the two read alike, so the ratio of the two scales is the ratio
        across the broad one."""
        n = 30
        reading = read(guide(broad_cells=n))
        expected = math.cos(math.pi / (2 * n)) * math.sqrt(n / (n - 1 + math.cos(math.pi / n)))
        assert reading.current / reading.voltage == pytest.approx(expected, rel=1e-12, abs=0.0)


def graded(cells: int = 30, ratio: float = 1.1) -> np.ndarray:
    """Lines across the broad wall, each cell ``ratio`` of the one before."""
    lines = np.concatenate([[0.0], np.cumsum(ratio ** np.arange(cells))])
    lines *= BROAD / lines[-1]
    lines[-1] = BROAD
    return lines


def marched(lines: np.ndarray, cutoff: float) -> np.ndarray:
    """The field the scheme's own equation gives at ``cutoff``, tied at the
    first wall and marched across node by node."""
    cells = np.diff(lines)
    field = [0.0, 1.0]
    for node in range(1, len(lines) - 1):
        width = (cells[node - 1] + cells[node]) / 2
        here, before = field[-1], field[-2]
        flux = (here - before) / cells[node - 1] - cutoff**2 * width * here
        field.append(here + flux * cells[node])
    return np.array(field)


class TestAcrossAGradedGuide:
    """Across the guide the field a port reads is the grid's own mode, and
    openEMS weights it with the continuous sine it was given."""

    def test_the_cutoffs_across_both_axes_are_the_schemes_on_the_guide_as_built(self):
        """On equal cells each axis holds ``(2 / h) sin(n pi h / 2 a)`` for every
        order its nodes allow, the broad wall's on the axis the mode varies on
        and the narrow wall's on the other."""
        reading = read(guide(narrow_cells=12, broad_cells=30))
        for cutoffs, wall, cells in (
            (reading.cutoffs[0], BROAD, 30),
            (reading.cutoffs[1], NARROW, 12),
        ):
            h = wall / cells * LENGTH_UNIT
            orders = np.arange(1, cells)
            expected = 2 / h * np.sin(orders * math.pi * h / (2 * wall * LENGTH_UNIT))
            np.testing.assert_allclose(cutoffs, expected, rtol=1e-9, atol=0.0)

    def test_a_guide_one_cell_high_holds_no_mode_across_that_axis(self):
        cutoffs, fields = portreading._modes(np.array([0.0, NARROW]) * LENGTH_UNIT)
        assert cutoffs.size == 0 and fields.shape == (0, 2)

    def test_on_equal_cells_the_field_is_the_sampled_sine(self):
        lines = np.linspace(0.0, BROAD, 31) * LENGTH_UNIT
        _, field = portreading._mode(lines, 2)
        sine = np.sin(2 * math.pi * lines / lines[-1])
        np.testing.assert_allclose(field / field[1], sine / sine[1], rtol=1e-9, atol=1e-12)

    def test_on_graded_cells_the_field_is_the_one_the_scheme_marches(self):
        lines = graded() * LENGTH_UNIT
        cutoff, field = portreading._mode(lines, 1)
        expected = marched(lines, cutoff)
        np.testing.assert_allclose(field / field[1], expected, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize("cells", [12, 20, 30])
    @pytest.mark.parametrize("ratio", [1.1, 1.2])
    def test_the_field_has_the_sign_of_the_sine(self, cells, ratio):
        """An eigenvector comes with either sign, and these guides and orders
        hold both."""
        lines = graded(cells, ratio) * LENGTH_UNIT
        for order in range(1, 6):
            _, field = portreading._mode(lines, order)
            sine = np.sin(order * math.pi * lines / lines[-1])
            assert np.sum(field * sine) > 0

    def test_each_scale_is_the_share_of_the_sine_in_the_mode_the_grid_holds(self):
        """The voltage sums the field against the sine on the lines, and the
        current against the sine half a cell on; each over the norm of the sine
        and of the field, taken over the widths of the nodes. The lines along
        the narrow wall are one guide's, so their part is the same in both, and
        the two guides are of different widths, so the norm is not."""

        def shares(across: np.ndarray, field: np.ndarray) -> tuple[float, float]:
            cells = np.diff(across)
            nodes = slice(1, -1)
            widths = (cells[:-1] + cells[1:]) / 2
            wide = across[-1]
            sine = np.sin(math.pi * across / wide)
            half = np.sin(math.pi * (across[:-1] + cells / 2) / wide)[1:]
            norm = math.sqrt(np.sum(field[nodes] ** 2 * widths))
            voltage = np.sum(field[nodes] * sine[nodes] * widths)
            voltage /= math.sqrt(np.sum(sine[nodes] ** 2 * widths)) * norm
            current = np.sum(field[nodes] * half * cells[1:])
            current /= math.sqrt(np.sum(half**2 * cells[1:])) * norm
            return voltage, current

        steep = graded(ratio=1.2)
        equal = np.linspace(0.0, 8.0, 31)
        lines = guide()
        one = read([lines[0], steep, lines[2]])
        other = read([lines[0], equal, lines[2]])
        field = marched(steep * LENGTH_UNIT, portreading._mode(steep * LENGTH_UNIT, 1)[0])
        voltage, current = shares(steep, field)
        voltage_equal, current_equal = shares(equal, np.sin(math.pi * equal / equal[-1]))
        # The sine itself on the same lines reads apart from the grid's mode by
        # far more than the tolerance below.
        as_sine = shares(steep, np.sin(math.pi * steep / BROAD))
        assert abs(as_sine[0] / voltage - 1) > 1e-6
        assert abs(as_sine[1] / current - 1) > 1e-6
        assert one.voltage / other.voltage == pytest.approx(
            voltage / voltage_equal, rel=1e-9, abs=0.0
        )
        assert one.current / other.current == pytest.approx(
            current / current_equal, rel=1e-9, abs=0.0
        )


class TestTheGuideAsBuilt:
    def test_a_drawn_face_arrives_on_the_line_inside_the_metal(self):
        """Lines two thirds of a cell outside a face and one third inside it, as
        the mesher lays them. The edge between them has its middle outside the
        metal and carries the field, so the guide is taller than drawn."""
        face, cell = 2.15, 0.06
        outside, inside = face - 2 * cell / 3, face + cell / 3
        z = np.concatenate(
            [np.linspace(0.0, outside, 12), [inside], np.linspace(inside + cell, NARROW, 8)]
        )
        lines = [along(), np.linspace(0.0, BROAD, 31), z]
        top = int(np.searchsorted(z, inside))
        metal = ((-1.0, -1.0, face), (99.0, 99.0, 99.0))
        reading = read(
            lines,
            voltage=Nodes((PLANE, 1, 1), (PLANE, 29, top)),
            current=Nodes((PLANE, 1, 1), (PLANE, 29, top)),
            conducts=walls_of(lines, metal),
            size=(BROAD, face),
        )
        assert reading.guide == (pytest.approx(BROAD, abs=1e-12), pytest.approx(inside, abs=1e-12))

    def test_a_side_wall_drawn_in_metal_sets_the_width_the_mode_is_cut_off_at(self):
        """The same pair of lines at a narrow wall. The waves are referred to the
        mode impedance of the guide the grid holds, ``k Z0 / beta`` with ``beta``
        cut off on the lines up to the one inside the metal, over ``cos(theta)``
        of the half cell along the guide."""
        face, cell = 8.0, 0.09
        outside, inside = face - 2 * cell / 3, face + cell / 3
        y = np.concatenate(
            [np.linspace(0.0, outside, 24), [inside], np.linspace(inside + cell, BROAD, 8)]
        )
        lines = [along(), y, np.linspace(0.0, NARROW, 13)]
        right = int(np.searchsorted(y, inside))
        metal = ((-1.0, face, -1.0), (99.0, 99.0, 99.0))
        box = Nodes((PLANE, 1, 1), (PLANE, right, 11))
        reading = read(lines, box, box, walls_of(lines, metal), size=(face, NARROW))
        k = grid_wavenumber()
        cutoff = portreading._mode(y[: right + 1] * LENGTH_UNIT, 1)[0]
        beta = np.sqrt(k**2 - cutoff**2)
        carried = np.sqrt(1 - (beta * 0.5 * LENGTH_UNIT / 2) ** 2)
        assert reading.guide[0] == pytest.approx(inside, abs=1e-12)
        np.testing.assert_allclose(
            reading.impedance,
            k * portreading.FREE_SPACE_IMPEDANCE / beta / carried,
            rtol=1e-12,
            atol=0.0,
        )

    def test_a_probe_reaching_a_line_past_the_wall_reads_no_field_there(self):
        """The node past the wall lies in the metal, where the field is zero, so
        it adds nothing to what the probe sums and only its weight to the norm."""
        face, cell = 8.0, 0.09
        outside, inside = face - 2 * cell / 3, face + cell / 3
        y = np.concatenate(
            [np.linspace(0.0, outside, 24), [inside], np.linspace(inside + cell, BROAD, 8)]
        )
        lines = [along(), y, np.linspace(0.0, NARROW, 13)]
        right = int(np.searchsorted(y, inside))
        metal = ((-1.0, face, -1.0), (99.0, 99.0, 99.0))
        box = Nodes((PLANE, 1, 1), (PLANE, right, 11))
        past = Nodes((PLANE, 1, 1), (PLANE, right + 1, 11))
        conducts = walls_of(lines, metal)
        one = read(lines, box, box, conducts, size=(face, NARROW))
        other = read(lines, past, box, conducts, size=(face, NARROW))

        def norm(stop: int) -> float:
            js = np.arange(1, stop + 1)
            width = portreading._node_width(y, js, dual=False)
            return math.sqrt(np.sum(np.sin(math.pi * y[js] / face) ** 2 * width))

        assert other.voltage / one.voltage == pytest.approx(
            norm(right) / norm(right + 1), rel=1e-12, abs=0.0
        )
        assert other.current == one.current

    def test_a_side_wall_whose_face_is_nearer_the_line_outside_is_read(self):
        """Across the guide the wall is the first line inside the metal, where
        the field lying in it is tied, whichever line the face is nearer; the
        edge crossing the face carries none of this mode's field."""
        face, cell = 8.0, 0.09
        outside, inside = face - cell / 3, face + 2 * cell / 3
        y = np.concatenate(
            [np.linspace(0.0, outside, 24), [inside], np.linspace(inside + cell, BROAD, 8)]
        )
        lines = [along(), y, np.linspace(0.0, NARROW, 13)]
        box = Nodes((PLANE, 1, 1), (PLANE, int(np.searchsorted(y, outside)), 11))
        metal = ((-1.0, face, -1.0), (99.0, 99.0, 99.0))
        reading = read(lines, box, box, walls_of(lines, metal), size=(face, NARROW))
        assert reading.guide[0] == pytest.approx(inside, abs=1e-12)

    def test_the_same_guide_drawn_across_the_other_axis_reads_the_same(self):
        """openEMS binds ``a`` to the first mode axis, so the broad wall on the
        second is TE01 to it."""
        lines = guide()
        turned = [lines[0], lines[2], lines[1]]
        across = Nodes((PLANE, 1, 1), (PLANE, 11, 29))
        one = read(lines)
        other = read(turned, voltage=across, current=across, mode=(0, 1), size=(NARROW, BROAD))
        assert other.voltage == pytest.approx(one.voltage, rel=1e-12, abs=0.0)
        assert other.current == pytest.approx(one.current, rel=1e-12, abs=0.0)
        assert other.guide == one.guide[::-1]


def grid_wavenumber(frequency: np.ndarray = FREQUENCY) -> np.ndarray:
    """What the scheme steps for ``2 pi f / c``, a timestep at a time."""
    return 2 / TIMESTEP * np.sin(math.pi * frequency * TIMESTEP) / units.SPEED_OF_LIGHT


def grid_beta(broad_cells: int = 30) -> np.ndarray:
    """TE10's propagation constant on :func:`guide`'s equal cells across, with
    the cutoff in the scheme's closed form for them."""
    h = BROAD / broad_cells * LENGTH_UNIT
    cutoff = 2 / h * math.sin(math.pi * h / (2 * BROAD * LENGTH_UNIT))
    return np.sqrt(grid_wavenumber().astype(complex) ** 2 - cutoff**2)


def laid(x, direction, start, away, travelling, to=PLANE):
    """A wave alone on the equal cells running on from node ``start`` away from
    the plane, carried to the plane by the scheme's own equations.

    Faraday's law over a cell ``h``, ``V(k+1) - V(k) = -j w mu h I(k+1/2)``, and
    Ampere's law at a node of width ``h*``, ``I(k+1/2) - I(k-1/2) = -j (beta^2 /
    w mu) h* V(k)``, with the grid's own ``w`` and ``beta``.

    :returns: The plane's node voltage, the mean of its two edge currents into
        the structure, the turn of the cells between the plane and ``start``,
        and the power the wave carries into the structure.
    """
    cells = np.diff(x) * LENGTH_UNIT
    w = grid_wavenumber() * units.SPEED_OF_LIGHT
    beta = grid_beta()
    series = 1j * w * units.VACUUM_PERMEABILITY
    shunt = 1j * beta**2 / (w * units.VACUUM_PERMEABILITY)
    width = lambda node: (cells[node - 1] + cells[node]) / 2  # noqa: E731
    up = direction if travelling == "in" else -direction
    h = cells[start] if away > 0 else cells[start - 1]
    step = np.exp(-1j * up * away * 2 * np.arcsin(beta * h / 2))
    v = np.ones_like(beta)
    if away > 0:
        edge = (v - step) / (series * h)
        power = 0.5 * np.real(v * np.conj(edge)) * direction
        node, above = start, edge
        while node != to:
            below = above + shunt * width(node) * v
            v = v + series * cells[node - 1] * below
            node -= 1
            above = below
        below = above + shunt * width(node) * v
    else:
        edge = (step - v) / (series * h)
        power = 0.5 * np.real(v * np.conj(edge)) * direction
        node, below = start, edge
        while node != to:
            above = below - shunt * width(node) * v
            v = v - series * cells[node] * above
            node += 1
            below = above
        above = below - shunt * width(node) * v
    lo, hi = sorted((to, start))
    turn = np.exp(1j * sum((2 * np.arcsin(beta * cells[k] / 2) for k in range(lo, hi)), 0j))
    return v, direction * (below + above) / 2, turn, power


def mirrored(x: np.ndarray) -> np.ndarray:
    return x[-1] - x[::-1]


class TestAlongTheGuide:
    """The grid carries the mode as a ladder along the guide, and a wave the
    ladder carries alone is read back alone, with the power it carries.

    ``laid`` writes the ladder out afresh rather than asking the module."""

    def test_on_equal_cells_the_cutoff_is_the_schemes_closed_form(self):
        h = BROAD / 30 * LENGTH_UNIT
        cutoff = portreading._mode(np.linspace(0.0, BROAD, 31) * LENGTH_UNIT, 1)[0]
        assert cutoff == pytest.approx(
            2 / h * math.sin(math.pi * h / (2 * BROAD * LENGTH_UNIT)), rel=1e-12, abs=0.0
        )

    def test_on_graded_cells_the_cutoff_ties_the_field_at_both_walls(self):
        """The field tied at one wall and marched across by the scheme's own
        equation at the cutoff comes to rest at the other."""
        lines = np.concatenate([[0.0], np.cumsum(0.2 * 1.08 ** np.arange(30))]) * LENGTH_UNIT
        cutoff = portreading._mode(lines, 1)[0]
        cells = np.diff(lines)
        field = [0.0, 1.0]
        for node in range(1, len(lines) - 1):
            width = (cells[node - 1] + cells[node]) / 2
            here, before = field[-1], field[-2]
            flux = (here - before) / cells[node - 1] - cutoff**2 * width * here
            field.append(here + flux * cells[node])
        assert abs(field[-1]) < 1e-9 * max(abs(value) for value in field)

    def test_on_equal_cells_the_half_cells_are_the_impedance_of_the_grid(self):
        """The current is the mean of two edges half a cell up and down the
        guide, a half-cell phase ``theta`` either way with ``sin(theta) =
        beta h / 2``. That mean is ``cos(theta)`` of a wave whichever way it
        travels, and a wave carries ``cos(theta)`` of the power the mode
        impedance gives, so each is read at the current's scale alone against
        ``Z / cos(theta)``."""
        cell = 0.5
        reading = read(guide())
        beta = grid_beta()
        half = beta * cell * LENGTH_UNIT / 2
        z = grid_wavenumber() * portreading.FREE_SPACE_IMPEDANCE / beta
        assert reading.carried == 0
        np.testing.assert_allclose(reading.inward, reading.current, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(reading.outward, reading.current, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(reading.voltage_in, reading.voltage, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(reading.voltage_out, reading.voltage, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(reading.impedance, z / np.sqrt(1 - half**2), rtol=1e-12)

    @staticmethod
    def settling() -> np.ndarray:
        """Cells of 0.5 behind the plane, then 0.3 and 0.5 on into the
        structure: the plane and the node after it sit on a change of cell,
        and the node two cells in is the first whose cells are alike."""
        x = along()
        x[PLANE + 1 :] -= 0.2
        return x

    @pytest.mark.parametrize("travelling", ["in", "out"])
    @pytest.mark.parametrize("direction", [1, -1], ids=["up", "down"])
    def test_a_wave_the_ladder_carries_is_read_back_across_a_change_of_cell(
        self, direction, travelling
    ):
        """The wave is laid where the cells are alike and carried to the plane
        by the ladder; the reading gives it back alone, turned to the plane,
        with the power it carries."""
        x = self.settling() if direction > 0 else mirrored(self.settling())
        lines = guide()
        lines[0] = x
        reading = read(lines, direction=direction)
        settled = PLANE + 2 * direction
        v, mean, turn, power = laid(x, direction, settled, direction, travelling)
        incident, reflected = reading.waves(reading.voltage * v, reading.current * mean)
        assert reading.carried == 2
        if travelling == "in":
            wave, other, expected = incident, reflected, turn
        else:
            wave, other, expected = reflected, incident, 1 / turn
        np.testing.assert_allclose(wave, expected, rtol=1e-9, atol=0.0)
        np.testing.assert_allclose(other, 0.0, rtol=0.0, atol=1e-9)
        sign = 1 if travelling == "in" else -1
        carried = sign * np.abs(wave) ** 2 / (2 * np.real(reading.impedance))
        np.testing.assert_allclose(carried, power, rtol=1e-9, atol=0.0)

    @pytest.mark.parametrize("direction", [1, -1], ids=["up", "down"])
    def test_where_the_structures_cells_never_settle_the_side_that_runs_on_is_read(self, direction):
        """Cells of 0.4 running on behind the plane, and cells growing by a
        tenth each into the structure, so no node there has two alike. The
        waves are split at the plane from the side behind it, and a wave laid
        there comes back alone."""
        x = along(below=0.4)
        x[PLANE + 1 :] = x[PLANE] + np.cumsum(0.4 * 1.1 ** np.arange(1, PLANE + 1))
        x = x if direction > 0 else mirrored(x)
        lines = guide()
        lines[0] = x
        reading = read(lines, direction=direction)
        v, mean, _, power = laid(x, direction, PLANE, -direction, "in")
        incident, reflected = reading.waves(reading.voltage * v, reading.current * mean)
        assert reading.carried is None
        np.testing.assert_allclose(incident, 1.0, rtol=1e-9, atol=0.0)
        np.testing.assert_allclose(reflected, 0.0, rtol=0.0, atol=1e-9)
        carried = np.abs(incident) ** 2 / (2 * np.real(reading.impedance))
        np.testing.assert_allclose(carried, power, rtol=1e-9, atol=0.0)

    @pytest.mark.parametrize("travelling", ["in", "out"])
    @pytest.mark.parametrize("direction", [1, -1], ids=["up", "down"])
    def test_a_plane_inside_predicts_what_the_port_reads_of_its_mode_alone(
        self, direction, travelling
    ):
        """One wave laid alone and read at the port's plane, on a change of
        cell, and at a plane three cells into the box: what the plane inside
        recorded, carried along the ladder, gives the waves the port read."""
        x = self.settling() if direction > 0 else mirrored(self.settling())
        lines = guide()
        lines[0] = x
        inside = PLANE - 3 * direction
        here = read(lines, direction=direction)
        there = read(lines, plane=inside, direction=direction)
        settled = PLANE + 2 * direction
        at_plane = laid(x, direction, settled, direction, travelling)
        at_inside = laid(x, direction, settled, direction, travelling, to=inside)
        read_here = here.waves(here.voltage * at_plane[0], here.current * at_plane[1])
        predicted = portreading.predicted(
            here,
            there,
            x,
            LENGTH_UNIT,
            there.voltage * at_inside[0],
            there.current * at_inside[1],
        )
        for got, want in zip(predicted, read_here, strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)

    def test_a_guide_changing_before_the_cells_settle_is_not_carried_through(self):
        """Cells of 0.3, 0.4 and then 0.5 into the structure, so the node three
        cells in is the first with two alike; a metal wall narrowing the guide
        from the node two cells in stops the carry short of it, and the waves
        are split at the plane."""
        x = along()
        x[PLANE + 1 :] = x[PLANE] + np.cumsum([0.3, 0.4, *([0.5] * (PLANE - 2))])
        lines = guide()
        lines[0] = x
        assert read(lines).carried == 3
        metal = ((x[PLANE + 2], 8.0, -1.0), (99.0, 99.0, 99.0))
        assert read(lines, conducts=walls_of(lines, metal)).carried is None

    def test_the_carry_reaches_no_further_than_the_ports_box_is_deep(self):
        """The same cells, and a box two cells deep: the node three cells in
        is out of reach."""
        x = along()
        x[PLANE + 1 :] = x[PLANE] + np.cumsum([0.3, 0.4, *([0.5] * (PLANE - 2))])
        lines = guide()
        lines[0] = x
        assert read(lines, source=x[PLANE - 3]).carried == 3
        assert read(lines, source=x[PLANE - 2]).carried is None

    def test_cells_alike_to_the_rounding_of_their_lines_are_alike(self):
        """The envelope holds a line to twelve significant figures, so two equal
        cells can come back a part in a hundred million apart."""
        x = along()
        x[PLANE + 1 :] += 0.5e-8
        lines = guide()
        lines[0] = x
        assert read(lines).carried == 0

    def test_a_port_facing_the_other_way_on_the_mirrored_grid_reads_the_same(self):
        """Which way is up the axis is a convention of the grid's."""
        lines, flipped = guide(), guide()
        lines[0], flipped[0] = self.settling(), mirrored(self.settling())
        forward, backward = read(lines, direction=1), read(flipped, direction=-1)
        for field in ("inward", "outward", "voltage_in", "voltage_out", "impedance"):
            np.testing.assert_allclose(
                getattr(backward, field), getattr(forward, field), rtol=1e-12, atol=0.0
            )

    def test_the_mode_propagates_above_the_cutoff_the_grid_holds(self):
        """The grid's cutoff lies a little under the continuous guide's, so a
        point between the two propagates in the guide the port reads."""
        wide = 12.0
        cells = 30
        lines = [along(), np.linspace(0.0, wide, cells + 1), np.linspace(0.0, NARROW, 13)]
        h = wide / cells * LENGTH_UNIT
        grid = 2 / h * math.sin(2 * math.pi * h / (2 * wide * LENGTH_UNIT))
        continuous = 2 * math.pi / (wide * LENGTH_UNIT)
        between = np.array([(grid + continuous) / 2 * units.SPEED_OF_LIGHT / (2 * math.pi)])
        reading = read(lines, mode=(2, 0), frequency=between)
        assert grid < grid_wavenumber(between)[0] < continuous
        assert reading.propagates.tolist() == [True]


class TestWhatIsNotRead:
    def test_a_mode_varying_across_both_walls(self):
        with pytest.raises(NotRead, match="across both walls"):
            read(guide(), mode=(1, 1))

    def test_a_mode_of_more_orders_than_the_guide_has_lines_across(self):
        with pytest.raises(NotRead, match="3 cells across"):
            read(guide(broad_cells=3), mode=(3, 0))
        assert read(guide(broad_cells=3), mode=(2, 0)).cutoffs[0].size == 2

    @pytest.mark.parametrize("side", [3, 5], ids=["across", "along"])
    def test_a_guide_one_side_of_which_does_not_conduct(self, side):
        """A face of the domain that absorbs, across the broad wall and across
        the narrow one."""
        lines = guide()
        with pytest.raises(NotRead, match="not a conductor"):
            read(lines, conducts=walls_of(lines, open_side=side))

    @pytest.mark.parametrize(
        "lower, upper",
        [((9.0, 5.0, 0.0), (11.0, 5.5, NARROW)), ((9.0, 2.0, 0.0), (11.0, 2.5, 2.0))],
        ids=["centre", "aside"],
    )
    def test_a_post_standing_in_the_plane(self, lower, upper):
        """One on the line the walls are looked for from, and one beside it and
        short of the far wall. A post beside it and meeting both walls would
        be a wall, and the guide it leaves a rectangle."""
        lines = guide()
        with pytest.raises(NotRead, match="not a rectangular guide"):
            read(lines, conducts=walls_of(lines, (lower, upper)))

    def test_a_ridge_hanging_from_the_top_wall(self):
        """Over the middle of the broad wall the guide ends at the ridge, and
        either side of it at the top wall, so no one height holds across it."""
        lines = guide()
        ridge = ((9.0, 5.0, float(lines[2][8])), (11.0, 5.7, NARROW))
        with pytest.raises(NotRead, match="not a rectangular guide"):
            read(lines, conducts=walls_of(lines, ridge))

    def test_a_roof_of_metal_over_the_middle_and_of_sheet_either_side(self):
        """The line under the roof is tied all across, and the field crosses the
        sheet where no metal stands over it."""
        lines = guide()
        roof = float(lines[2][8])
        block = ((-1.0, 4.0, roof), (99.0, 7.0, 99.0))
        sheet = ((-1.0, -1.0, roof), (99.0, 99.0, roof))
        box = Nodes((PLANE, 1, 1), (PLANE, 29, 8))
        with pytest.raises(NotRead, match="not a rectangular guide"):
            read(lines, box, box, walls_of(lines, block, sheet), size=(BROAD, roof))

    def test_a_roof_ending_on_a_line_over_part_of_the_width_alone(self):
        """Over the middle the roof's face stands on the line, and to one side
        a little above it, where the edge up from the line is tied and the line
        is not."""
        lines = guide()
        roof = float(lines[2][8])
        cell = float(lines[2][9]) - roof
        middle = ((-1.0, -1.0, roof), (99.0, 7.0, 99.0))
        aside = ((-1.0, 7.0, roof + cell / 4), (99.0, 99.0, 99.0))
        box = Nodes((PLANE, 1, 1), (PLANE, 29, 8))
        with pytest.raises(NotRead, match="not a rectangular guide"):
            read(lines, box, box, walls_of(lines, middle, aside), size=(BROAD, roof))

    def test_a_face_nearer_the_line_outside_the_metal(self):
        """Lines a third of a cell outside a face and two thirds inside it. The
        edge between them is tied and the line outside is not, so the guide
        ends between the two, where no wall stands."""
        face, cell = 2.15, 0.06
        outside, inside = face - cell / 3, face + 2 * cell / 3
        z = np.concatenate(
            [np.linspace(0.0, outside, 12), [inside], np.linspace(inside + cell, NARROW, 8)]
        )
        lines = [along(), np.linspace(0.0, BROAD, 31), z]
        box = Nodes((PLANE, 1, 1), (PLANE, 29, int(np.searchsorted(z, outside))))
        metal = ((-1.0, -1.0, face), (99.0, 99.0, 99.0))
        with pytest.raises(NotRead, match="between two grid lines"):
            read(lines, box, box, walls_of(lines, metal), size=(BROAD, face))

    def test_a_sheet_over_the_box_across_the_field(self):
        """A metal sheet ties the field lying in it and none crossing it, so the
        mode of the whole height passes it and the box drawn under it holds
        part of a guide."""
        lines = guide()
        roof = float(lines[2][6])
        sheet = ((-1.0, -1.0, roof), (99.0, 99.0, roof))
        box = Nodes((PLANE, 1, 1), (PLANE, 29, 6))
        with pytest.raises(NotRead, match="not the one its box was drawn on"):
            read(lines, box, box, walls_of(lines, sheet), size=(BROAD, roof))

    def test_a_box_narrower_than_the_guide(self):
        lines = guide()
        box = Nodes((PLANE, 1, 1), (PLANE, 20, 11))
        with pytest.raises(NotRead, match="not the one its box was drawn on"):
            read(lines, box, box, size=(float(lines[1][20]), NARROW))

    def test_a_box_short_of_a_domain_wall_by_more_than_a_cell(self):
        """It snaps to the line next to the wall, which is also the first node a
        box on the wall keeps once the engine leaves the wall out, so only the
        box as drawn tells the two apart."""
        lines = guide()
        inset = 1.4 * lines[1][1]
        box = Nodes((PLANE, 1, 1), (PLANE, 29, 11))
        with pytest.raises(NotRead, match="not the one its box was drawn on"):
            portreading.read(
                lines,
                0,
                (1, 0),
                (BROAD - inset, NARROW),
                ((0.0, inset, 0.0), (1.0, BROAD, NARROW)),
                box,
                box,
                1,
                walls_of(lines),
                FREQUENCY,
                LENGTH_UNIT,
                TIMESTEP,
            )

    @pytest.mark.parametrize("offset", [1, -1], ids=["ahead", "behind"])
    def test_a_guide_changing_within_a_cell_of_the_plane(self, offset):
        """The current's edges half a cell away sit in the next section."""
        lines = guide()
        x = lines[0][PLANE + offset]
        step = (
            ((x, -1.0, 3.0), (99.0, 99.0, 99.0))
            if offset > 0
            else ((-99.0, -1.0, 3.0), (x, 99.0, 99.0))
        )
        with pytest.raises(NotRead, match="within a cell"):
            read(lines, conducts=walls_of(lines, step))

    def test_a_plane_on_the_edge_of_the_grid(self):
        lines = guide()
        edge = Nodes((1, 1, 1), (1, 29, 11))
        with pytest.raises(NotRead, match="edge of the grid"):
            read(lines, voltage=edge, current=edge)


class TestWhereTheModePropagates:
    def test_above_the_cutoff_of_the_guide_as_built(self):
        """TE20 in a guide twelve millimetres wide is cut off inside the band."""
        wide = 12.0
        lines = [along(), np.linspace(0.0, wide, 31), np.linspace(0.0, NARROW, 13)]
        reading = read(lines, mode=(2, 0))
        h = wide / 30 * LENGTH_UNIT
        cutoff = 2 / h * math.sin(2 * math.pi * h / (2 * wide * LENGTH_UNIT))
        assert 0 < np.count_nonzero(reading.propagates) < FREQUENCY.size
        np.testing.assert_array_equal(reading.propagates, cutoff < grid_wavenumber())


class TestWhatARunLeftAsReadSays:
    def test_nothing_where_every_port_was_read(self):
        record = {"1": {"corrected": True}, "2": {"corrected": True}}
        assert portreading.unread({portreading.KEY: record}) is None
        assert portreading.unread({}) is None

    def test_each_port_that_could_not_be_read_and_why(self):
        record = {
            "1": {"corrected": False, "reason": "port 2 of this run could not be read"},
            "2": {"corrected": False, "reason": "R2", "refused": True},
            "3": {"corrected": False, "reason": "R3", "refused": True},
        }
        said = portreading.unread({portreading.KEY: record})
        assert said is not None
        assert "port 2: its waveguide reading is left as openEMS returned it, because R2" in said
        assert "port 3: its waveguide reading is left as openEMS returned it, because R3" in said
        assert "port 1:" not in said
        assert "every other waveguide port" in said
