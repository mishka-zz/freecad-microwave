# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Several S-parameter tables on one set of axes. Solver-neutral.

A table is a matrix and the name it is drawn under: a run of either backend, or
a matrix standing on its own. Each is drawn on its own grid, so nothing is
interpolated to draw it; only a difference of two needs shared points.

What is drawn of one term:

- its magnitude in decibels;
- its phase in degrees, unwrapped along each stretch of samples that hold a
  number, and moved by whole turns onto the first table's at the lowest
  frequency both hold a number at, so tables over different bands are not drawn
  whole turns apart;
- its group delay, the slope of that phase against angular frequency taken
  across one aperture in hertz for every table, within one stretch, so two
  grids are not smoothed differently and a hole is not read across;
- the share of the power sent into a port that none of the ports received,
  ``1 - sum |Si,j|^2`` over the column of port ``j``, which is lost to loss or
  to radiation.

The difference of two tables is ``|Sa - Sb|`` of the complex terms: a
difference of decibels deep in a null is large and means nothing, and a port
whose mode's sign a solver chose by a rule of its own is turned back where that
brings the two closer. It is taken at the
distinct frequencies of the coarser of the two grids inside the band both
cover, with the other table interpolated there, linearly in its real and
imaginary parts. A feature narrower than the coarser step is not compared. Two
tables of different ports are not of one device and are not differenced. Two
referencing a port to different impedances are ratios against different waves:
they are differenced, and the difference names each such port, since it holds
the change of reference as well. Two ports each at an own impedance that moves
with frequency are one basis, each matched to its own line, unless one is a
guide's mode and the other is not known to be one. A difference states its
largest value, the frequency it falls at, and the magnitude of the term in each
table there, and how far a table whose column was read alone can be off in a
term of that column.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .compared import SIGN_BY_RULE, written_difference
from .sparameters import (
    IMPEDANCE_STATED,
    LUMPED_RESISTANCE,
    SENT_BACK,
    SParameters,
    _engineering_hz,
    _magnitude,
    _term,
    describe_reference,
)

__all__ = [
    "DB",
    "DELAY",
    "LOST",
    "PHASE",
    "QUANTITIES",
    "Apart",
    "Table",
    "Unlike",
    "aperture",
    "delay",
    "difference",
    "grid",
    "lost",
    "phases",
    "terms",
    "driven",
    "reference",
]

#: What one term is drawn as, by the name the chart offers it under.
DB = "Magnitude (dB)"
PHASE = "Phase (deg)"
DELAY = "Group delay (ns)"
LOST = "Power lost (%)"
QUANTITIES = (DB, PHASE, DELAY, LOST)


class Unlike(ValueError):
    """Two tables that cannot be differenced, with the reason."""


@dataclass(frozen=True)
class Table:
    """One matrix and the name it is drawn under."""

    result: SParameters
    name: str
    #: What the run that solved it was asked, where it was a run.
    snapshot: dict = field(default_factory=dict)
    #: The name of the document it was read from, or nothing for a table no
    #: document holds. A chart closes with each document it draws from.
    document: str = ""


def terms(tables: list[Table]) -> list[tuple[int, int]]:
    """Every term some table measured, reflections first, each in port order."""
    found: set[tuple[int, int]] = set()
    for table in tables:
        result = table.result
        for receiving in result.port_numbers:
            for driving in result.port_numbers:
                if np.any(np.isfinite(result.parameter(receiving, driving))):
                    found.add((receiving, driving))
    return sorted(found, key=lambda term: (term[0] != term[1], term[1], term[0]))


def driven(tables: list[Table]) -> list[int]:
    """Every port some table holds a whole column of, in order."""
    found = set()
    for table in tables:
        for number in table.result.port_numbers:
            if np.any(np.isfinite(lost(table.result, number))):
                found.add(number)
    return sorted(found)


def grid(result: SParameters) -> str:
    """The grid a table holds, as a reader is told it."""
    frequency = np.asarray(result.frequency, dtype=float)
    if not frequency.size:
        return "no point"
    if frequency.size == 1:
        return f"one point at {_engineering_hz(float(frequency[0]))}"
    step = float(np.max(np.diff(np.sort(frequency))))
    return (
        f"{frequency.size} points from {_engineering_hz(float(frequency.min()))} to "
        f"{_engineering_hz(float(frequency.max()))}, step {_engineering_hz(step)}"
        + ("" if np.allclose(np.diff(np.sort(frequency)), step, rtol=1e-6) else " at most")
    )


def reference(result: SParameters) -> str:
    """What a table is referenced to, as a reader is told it."""
    return describe_reference(result.reference, result.port_numbers, result.self_referenced)


def _unwrapped(frequency: np.ndarray, term: np.ndarray) -> np.ndarray:
    """The phase of ``term`` in radians, unwrapped along each stretch of
    samples that hold a number, in order of frequency."""
    order = np.argsort(frequency, kind="stable")
    angle = np.full(term.shape, np.nan)
    held = np.isfinite(term[order])
    stretch: list[int] = []
    for index, keep in zip(order, held, strict=True):
        if keep:
            stretch.append(int(index))
            continue
        if stretch:
            angle[stretch] = np.unwrap(np.angle(term[stretch]))
        stretch = []
    if stretch:
        angle[stretch] = np.unwrap(np.angle(term[stretch]))
    return angle


def _stretches(frequency: np.ndarray, term: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each stretch of neighbouring samples, in order of frequency, that hold a
    number: its distinct frequencies, rising, and its phase there in radians,
    unwrapped along the stretch alone."""
    order = np.argsort(frequency, kind="stable")
    f, t = frequency[order], term[order]
    held = np.isfinite(t)
    found = []
    start = None
    for index in range(f.size + 1):
        if index < f.size and held[index]:
            start = index if start is None else start
            continue
        if start is not None:
            distinct, first = np.unique(f[start:index], return_index=True)
            found.append((distinct, np.unwrap(np.angle(t[start:index][first]))))
            start = None
    return found


def _in(stretches: list[tuple[np.ndarray, np.ndarray]], at: float) -> float | None:
    """The phase ``stretches`` hold at ``at``, interpolated within the stretch
    standing there, or ``None`` where none does."""
    for f, phi in stretches:
        if f[0] <= at <= f[-1]:
            return float(np.interp(at, f, phi))
    return None


def phases(
    tables: list[Table], receiving: int, driving: int
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Each table's phase of one term in degrees, what was done to draw it, and
    where the curve drawn may not be the phase.

    Each table after the first is moved by whole turns onto the first table
    that holds the term, at the lowest frequency it holds that stands inside a
    stretch of that table. A step between neighbouring samples of more than a
    quarter turn is named among the doubts, since the unwrapping cannot tell it
    from a turn the other way.
    """
    drawn: list[np.ndarray] = []
    said: list[str] = []
    doubted: list[str] = []
    anchor: tuple[list[tuple[np.ndarray, np.ndarray]], str] | None = None
    for table in tables:
        result = table.result
        if receiving not in result.port_numbers or driving not in result.port_numbers:
            drawn.append(np.array([]))
            continue
        frequency = np.asarray(result.frequency, dtype=float)
        term = np.asarray(result.parameter(receiving, driving))
        angle = _unwrapped(frequency, term)
        held = np.isfinite(angle)
        if not held.any():
            drawn.append(np.degrees(angle))
            continue
        if anchor is None:
            anchor = (_stretches(frequency, term), table.name)
        else:
            for index in np.flatnonzero(held)[np.argsort(frequency[held], kind="stable")]:
                theirs = _in(anchor[0], float(frequency[index]))
                if theirs is None:
                    continue
                turns = round((theirs - float(angle[index])) / (2 * math.pi))
                if turns:
                    angle = angle + 2 * math.pi * turns
                    said.append(
                        f"the phase of {table.name} is moved by {turns * 360:+d} degrees onto "
                        f"{anchor[1]}'s at {_engineering_hz(float(frequency[index]))}"
                    )
                break
        for _, phi in _stretches(frequency, term):
            if np.any(np.abs(np.diff(phi)) > math.pi / 2):
                doubted.append(
                    f"the phase of {table.name} turns by more than 90 degrees between "
                    "neighbouring samples, where unwrapping cannot tell which way it turned"
                )
                break
        drawn.append(np.degrees(angle))
    return drawn, said, doubted


def aperture(tables: list[Table], receiving: int = 0, driving: int = 0) -> float:
    """The aperture group delay is taken across, in hertz: twice the coarsest
    step of the tables holding the term, so every table's slope spans the same
    band. Every table is asked where no term is named."""
    steps = []
    for table in tables:
        result = table.result
        if receiving and (
            receiving not in result.port_numbers or driving not in result.port_numbers
        ):
            continue
        frequency = np.unique(np.asarray(result.frequency, dtype=float))
        if frequency.size > 1:
            steps.append(float(np.max(np.diff(frequency))))
    return 2 * max(steps) if steps else 0.0


def delay(table: Table, receiving: int, driving: int, across: float) -> np.ndarray:
    """Group delay of one term in nanoseconds, across ``across`` hertz.

    The slope is taken between the unwrapped phase half the aperture below a
    frequency and half above it, within the stretch of samples holding a
    number that the frequency stands in, so a frequency closer to either end of
    its stretch than half the aperture holds no number, and a table without
    either port holds none.
    """
    result = table.result
    frequency = np.asarray(result.frequency, dtype=float)
    value = np.full(frequency.shape, np.nan)
    if across <= 0.0 or not {receiving, driving} <= set(result.port_numbers):
        return value
    term = np.asarray(result.parameter(receiving, driving))
    for f, phi in _stretches(frequency, term):
        low, high = frequency - across / 2, frequency + across / 2
        inside = np.isfinite(term) & (low >= f[0]) & (high <= f[-1])
        if f.size < 2 or not inside.any():
            continue
        rise = np.interp(high[inside], f, phi) - np.interp(low[inside], f, phi)
        value[inside] = -rise / (2 * math.pi * across) * 1e9
    return value


def lost(result: SParameters, driving: int) -> np.ndarray:
    """``1 - sum |Si,j|^2`` over the column of port ``driving``: the share of
    the power sent into that port that no port received. No number where a
    term of the column holds none."""
    if driving not in result.port_numbers:
        return np.full(np.asarray(result.frequency).shape, np.nan)
    column = np.asarray(result.s)[:, :, result.index_of(driving)]
    return 1.0 - np.sum(np.abs(column) ** 2, axis=1)


@dataclass(frozen=True)
class Apart:
    """``|Sa - Sb|`` of one term at the frequencies of the coarser grid."""

    frequency: np.ndarray
    value: np.ndarray
    #: What was done to take it, a line each.
    said: tuple[str, ...]
    #: The term's name, and the names of the table differenced against and of
    #: the table differenced.
    name: str = ""
    by: tuple[str, str] = ("", "")
    #: The term of each of the two at :attr:`frequency`, as differenced.
    first: np.ndarray = field(default_factory=lambda: np.array([]))
    second: np.ndarray = field(default_factory=lambda: np.array([]))
    #: Each port the two reference to different impedances, a line each, so
    #: the difference holds the change of reference as well.
    unalike: tuple[str, ...] = ()
    #: For each of the two whose column of the term was read alone: its name,
    #: the port the column is driven from, how far a term of that column can be
    #: off from what the ports not driven sent back at worst over the band it
    #: holds, the frequency that worst falls at, and the port that sent back
    #: most there.
    sent_back: tuple[tuple[str, int, float, float, int], ...] = ()

    def largest(self) -> float:
        """The largest value the difference holds, or nothing where no point
        holds one."""
        held = self.value[np.isfinite(self.value)]
        return float(held.max()) if held.size else math.nan

    def carried(self) -> list[str]:
        """How far each table whose column was read alone can be off in a term
        of that column, a line each."""
        return [
            f"A term of {table} driven from port {column} can be off by up to about "
            f"{_magnitude(bound)} at {_engineering_hz(at)}: port {port}, not driven, sent "
            "back part of what reached it"
            for table, column, bound, at, port in self.sent_back
        ]

    def swamped(self) -> list[tuple[str, int]]:
        """Each table whose bound falls inside the band differenced and is not
        under the largest difference, which it can then account for whole, and
        the port that sent back most."""
        largest = self.largest()
        return [
            (table, port)
            for table, _, bound, at, port in self.sent_back
            if self.frequency.min() <= at <= self.frequency.max() and bound >= largest
        ]

    def line(self) -> str:
        """How far apart the two stand at worst, and how large the term is in
        each there, or nothing where no point holds a number in both."""
        held = np.isfinite(self.value)
        if not held.any():
            return ""
        at = int(np.flatnonzero(held)[np.argmax(self.value[held])])
        first, second = self.by
        where = _engineering_hz(float(self.frequency[at]))
        return (
            f"{self.name} of {second} differs from {first} by at most "
            f"{written_difference(float(self.value[at]))} at {where}, "
            f"where |{self.name}| is {abs(self.first[at]):.3g} in {first} and "
            f"{abs(self.second[at]):.3g} in {second}"
        )


def difference(first: Table, second: Table, receiving: int, driving: int) -> Apart:
    """How far ``second`` stands from ``first`` in one term.

    A port whose mode's sign a solver chose by a rule of its own, as a matrix
    lists under :data:`~.compared.SIGN_BY_RULE`, can come back turned half a
    cycle against the other table, which negates a transmission through it; the
    term is turned back where that brings the two closer, and the difference
    says so, as the backend cross-check does.

    A port referenced differently in the two is named in
    :attr:`Apart.unalike`, and the difference is taken all the same.

    :raises Unlike: the two hold different ports, the bands do not overlap, or
        a table holds no number of the term.
    """
    name = _term(receiving, driving)
    a, b = first.result, second.result
    if sorted(a.port_numbers) != sorted(b.port_numbers):
        raise Unlike(f"{first.name} and {second.name} hold different ports")
    for table in (first, second):
        result = table.result
        if receiving not in result.port_numbers or driving not in result.port_numbers:
            raise Unlike(f"{table.name} holds no {name}")
        if not np.any(np.isfinite(result.parameter(receiving, driving))):
            raise Unlike(f"{table.name} did not measure {name}")
    fa = np.unique(np.asarray(a.frequency, dtype=float))
    fb = np.unique(np.asarray(b.frequency, dtype=float))
    low, high = max(fa.min(), fb.min()), min(fa.max(), fb.max())
    if low > high:
        raise Unlike(f"{first.name} and {second.name} share no band")
    inside_a = fa[(fa >= low) & (fa <= high)]
    inside_b = fb[(fb >= low) & (fb <= high)]
    # A grid holding no point inside the shared band, as one point between two
    # of the other grid's does, is interpolated at the other's points.
    first_coarser = bool(inside_a.size) and (inside_a.size <= inside_b.size or not inside_b.size)
    coarse, fine = (first, second) if first_coarser else (second, first)
    at = inside_a if coarse is first else inside_b
    unalike = [line for number in a.port_numbers if (line := _unalike(coarse, fine, number, at))]
    mine = _at(a.parameter(receiving, driving), np.asarray(a.frequency, dtype=float), at)
    theirs = _at(b.parameter(receiving, driving), np.asarray(b.frequency, dtype=float), at)
    said = [
        f"|{name}| of the difference of {first.name} and {second.name} is taken of the "
        f"complex terms at the {at.size} points of {coarse.name} from {_engineering_hz(low)} "
        f"to {_engineering_hz(high)}, with {fine.name} interpolated there. A feature narrower "
        "than that step is not compared, and a reference plane in another place shows as a "
        "difference even where the magnitudes agree"
    ]
    signed = sorted((_signed(a) | _signed(b)) & {receiving, driving})
    if receiving != driving and signed:
        held = np.isfinite(mine) & np.isfinite(theirs)
        if np.sum(np.abs(mine[held] + theirs[held]) ** 2) < np.sum(
            np.abs(mine[held] - theirs[held]) ** 2
        ):
            theirs = -theirs
            said.append(
                f"{name} of {second.name} is turned half a cycle, because a solver chose the "
                f"sign of the mode at port {signed[0]} by a rule of its own"
            )
    return Apart(
        frequency=at,
        value=np.abs(mine - theirs),
        said=tuple(said),
        name=name,
        by=(first.name, second.name),
        first=mine,
        second=theirs,
        unalike=tuple(unalike),
        sent_back=tuple(
            (
                table.name,
                driving,
                float(record["bound"]),
                float(record["frequency"]),
                int(record["port"]),
            )
            for table in (first, second)
            if (record := (table.result.provenance.get(SENT_BACK) or {}).get(str(driving)))
        ),
    )


def _signed(result: SParameters) -> set[int]:
    listed = result.provenance.get(SIGN_BY_RULE)
    return set() if listed is None else {int(number) for number in listed}


def _at(values: np.ndarray, frequency: np.ndarray, at: np.ndarray) -> np.ndarray:
    """``values`` over ``frequency``, interpolated at ``at`` linearly in their
    real and imaginary parts."""
    distinct, first = np.unique(frequency, return_index=True)
    held = np.asarray(values)[first]
    return np.interp(at, distinct, held.real) + 1j * np.interp(at, distinct, held.imag)


def _unalike(coarse: Table, fine: Table, number: int, at: np.ndarray) -> str:
    """Why the two tables reference port ``number`` to different impedances,
    or nothing where they reference it alike.

    Two fixed references are compared at the frequencies both tables hold, and
    interpolated onto the coarser grid only where they share none. A lumped port
    referenced to its own resistance is referenced to that number, so it is
    compared as one. Two own impedances that move with frequency are alike, each
    matched to its own line, unless one table states a guide's mode there and
    the other states nothing.
    """
    own = [not _fixed(table, number) for table in (coarse, fine)]
    if all(own):
        moded = [_stated(table, number) for table in (coarse, fine)]
        if moded[0] == moded[1]:
            return ""
        return (
            f"port {number} is at its own {'mode' if moded[0] else 'impedance'} in "
            f"{coarse.name}, its own {'mode' if moded[1] else 'impedance'} in {fine.name}"
        )
    if not any(own):
        stated = [
            np.asarray(table.result.reference)[:, table.result.index_of(number)]
            for table in (coarse, fine)
        ]
        frequency = [np.asarray(table.result.frequency, dtype=float) for table in (coarse, fine)]
        shared = np.intersect1d(frequency[0], frequency[1])
        where = shared if shared.size else at
        mine = _at(stated[0], frequency[0], where)
        theirs = _at(stated[1], frequency[1], where)
        # A point a table holds no reference at says nothing either way.
        held = np.isfinite(mine) & np.isfinite(theirs)
        if np.allclose(mine[held], theirs[held], rtol=1e-6, atol=0.0):
            return ""
    return _why_apart(coarse, fine, number)


def _why_apart(coarse: Table, fine: Table, number: int) -> str:
    """Why port ``number`` holds the two tables against different waves, in
    one short line."""
    if all(_fixed(table, number) for table in (coarse, fine)):
        return (
            f"port {number} is {_said(coarse, number)} in {coarse.name}, "
            f"{_said(fine, number)} in {fine.name}"
        )
    own = coarse if not _fixed(coarse, number) else fine
    if number in own.result.unmeasured:
        return f"port {number} is not driven in {own.name}"
    if str(number) in (own.result.provenance.get(IMPEDANCE_STATED) or {}):
        return f"port {number} is at its own mode in {own.name}"
    return f"port {number} is at its own impedance in {own.name}"


def _fixed(table: Table, number: int) -> bool:
    """Whether ``table`` references port ``number`` to a number: one it was
    asked for, the resistance of a lumped element it is referenced to as its
    own, or an own impedance that holds one value across the band, as the
    basis a chart states reads it. A guide referenced to its own mode, which the
    backend states, and an own impedance that moves with frequency are
    referenced to no number."""
    result = table.result
    if number not in result.self_referenced:
        return True
    stated = result.provenance.get(IMPEDANCE_STATED) or {}
    if str(number) in stated:
        return stated[str(number)] == LUMPED_RESISTANCE
    column = np.asarray(result.reference)[:, result.index_of(number)]
    held = column[np.isfinite(column)]
    return bool(held.size) and bool(np.allclose(held, held[0], rtol=1e-6, atol=0.0))


def _stated(table: Table, number: int) -> bool:
    """Whether ``table`` states which impedance of a guide's mode port
    ``number`` is at. Asked of a port at its own impedance alone, which a
    lumped port is not. The adapters state nothing for a line's own impedance,
    and a file read in states nothing."""
    return str(number) in (table.result.provenance.get(IMPEDANCE_STATED) or {})


def _said(table: Table, number: int) -> str:
    result = table.result
    if not _fixed(table, number):
        return "its own impedance"
    index = result.index_of(number)
    return describe_reference(np.asarray(result.reference)[:, index : index + 1], (number,), ())
