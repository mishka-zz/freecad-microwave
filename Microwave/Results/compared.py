# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How far two S-matrices of one drawing stand apart. Solver-neutral.

Two backends answering one drawing give two matrices, and the difference between
them measures neither: it bounds how far the two could be trusted to agree, and
no tolerance here says whether that is close enough. So this states, for each
term both measured, the largest ``|Sa - Sb|`` over the frequencies both solved,
where it falls, and how large the term itself is there. The difference is taken
of the complex terms, never of their decibels or of their phases: a difference
of decibels between two terms deep in a null is large and means nothing, and a
phase is undefined where a term vanishes.

Two matrices are compared only where they are ratios of the same waves:

- the same ports, by number;
- the same drawing, where both carry the key :data:`DRAWING` of what they were
  solved from;
- each port referenced alike - to its own impedance in both, or to the same
  number at every frequency both solved. Two backends that reference a port to
  its own mode state different numbers for one mode, a wave impedance and a
  power-voltage impedance, and the matrix is the same either way;
- at least one frequency in common.

A port whose mode's sign the solver chose by a rule of its own, listed under
:data:`SIGN_BY_RULE`, can come back turned half a cycle against the other
solve, which negates every transmission through it and leaves its reflection
as it was. Such a port is turned back where turning it brings the two matrices
closer over the whole band, and the comparison says which ports it turned and
in which matrix. Turning a set of ports and turning every other port give one
matrix, so where both sets are listed the smaller is the one named. A matrix
that does not say what drawing it was solved from is compared, and the
comparison says so.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .sparameters import SParameters, _engineering_hz, _magnitude, _ports, _term, describe_reference

__all__ = ["DRAWING", "SIGN_BY_RULE", "Comparison", "Incomparable", "Term", "compare"]

#: The provenance key under which a stored matrix records a digest of the
#: drawing it was solved from: the geometry, materials and ports, and nothing
#: that belongs to one backend's mesh.
DRAWING = "drawing"

#: The provenance key under which a matrix lists the ports whose mode's sign
#: the solver chose by a rule of its own rather than by a direction the drawing
#: fixes.
SIGN_BY_RULE = "sign_by_rule"

#: How close two frequencies are and still one frequency, as a share of either.
#: Both backends compute the band from one expression, and a frequency read back
#: from Palace's tables carries the nine figures it is written with there, so it
#: parts from the computed one by up to half a unit in the ninth figure. Each
#: frequency is paired with the nearest of the other band, once.
SAME_FREQUENCY = 1e-8

#: How close two fixed references are and still one, as a share of either. A
#: difference below the figures a message prints them to would be refused with a
#: reason that reads as none.
SAME_IMPEDANCE = 1e-6


class Incomparable(ValueError):
    """Two matrices that are not ratios of the same waves, with the reason."""


@dataclass(frozen=True)
class Term:
    """The largest difference one term shows over the shared band.

    :param receiving: and ``driving``: the term, by port number.
    :param worst: the largest ``|Sa - Sb|``.
    :param at: the frequency it falls at, in hertz.
    :param first: and ``second``: ``|Sa|`` and ``|Sb|`` there.
    """

    receiving: int
    driving: int
    worst: float
    at: float
    first: float
    second: float

    @property
    def name(self) -> str:
        return _term(self.receiving, self.driving)


@dataclass(frozen=True)
class Comparison:
    """Two matrices on the frequencies both solved, and how far apart they stand.

    ``first`` and ``second`` are indexed as :attr:`SParameters.s` is, over
    :attr:`frequency`, with ``second`` already turned where :attr:`turned` says.
    """

    by: tuple[str, str]
    frequency: np.ndarray
    port_numbers: tuple[int, ...]
    first: np.ndarray
    second: np.ndarray
    terms: tuple[Term, ...]
    #: Each port turned half a cycle, and what solved the matrix it was turned in.
    turned: tuple[tuple[int, str], ...]
    #: Frequencies only one of the two solved, which nothing is said about.
    unshared: int
    #: What solved each matrix that does not say what drawing it was solved
    #: from, so that the two could not be held to one drawing.
    unchecked: tuple[str, ...] = ()

    def lines(self) -> list[str]:
        """What a reader is told: one line per term, then what was done to make
        the two comparable."""
        return [self.line(term) for term in self.terms] + self.notes()

    def line(self, term: Term) -> str:
        """How far one term stands apart, and how large it is there."""
        first, second = self.by
        return (
            f"{term.name} differs by at most {_difference(term.worst)} at "
            f"{_engineering_hz(term.at)}, where |{term.name}| is {term.first:.3g} in what "
            f"{first} solved and {term.second:.3g} in what {second} solved"
        )

    def notes(self) -> list[str]:
        """What was done to the two to compare them, and what could not be
        checked, a line each."""
        said = []
        for by in dict.fromkeys(owner for _, owner in self.turned):
            ports = [number for number, owner in self.turned if owner == by]
            said.append(
                f"{_ports(ports)} of what {by} solved "
                f"{'was' if len(ports) == 1 else 'were'} turned half a cycle, because "
                f"{by} chose the sign of the mode there by a rule of its own"
            )
        if self.unshared:
            said.append(
                f"{_frequencies(self.unshared)} solved by one of the two alone "
                f"{'is' if self.unshared == 1 else 'are'} not compared"
            )
        for by in self.unchecked:
            said.append(
                f"what {by} solved does not say what drawing it was solved from, so the "
                "two are not known to be of one drawing"
            )
        return said

    def difference(self, receiving: int, driving: int) -> np.ndarray:
        """``|Sa - Sb|`` of one term at each shared frequency."""
        row, column = self.port_numbers.index(receiving), self.port_numbers.index(driving)
        return np.asarray(np.abs(self.first[:, row, column] - self.second[:, row, column]))


def compare(first: SParameters, first_by: str, second: SParameters, second_by: str) -> Comparison:
    """How far ``second`` stands from ``first``, term by term.

    :param first_by: and ``second_by``: what solved each, as its provenance
        names it, which is how every line tells the reader which is which.
    :raises Incomparable: where the two are not ratios of the same waves.
    """
    if sorted(first.port_numbers) != sorted(second.port_numbers):
        raise Incomparable(
            f"what {first_by} solved measured {_ports(first.port_numbers)} and what "
            f"{second_by} solved measured {_ports(second.port_numbers)}, so the two "
            "matrices are not of one device. Give the study one set of ports and run "
            "both again"
        )
    drawn, other = first.provenance.get(DRAWING), second.provenance.get(DRAWING)
    if drawn and other and drawn != other:
        raise Incomparable(
            f"what {first_by} solved and what {second_by} solved are of two different "
            "drawings: the geometry, a material or a port changed between the two runs. "
            "Run the one solved before the change again"
        )
    rows, columns = _shared(first.frequency, second.frequency)
    if not rows.size:
        raise Incomparable(
            f"what {first_by} solved and what {second_by} solved share no frequency. "
            "Solve both over one band"
        )
    ports = tuple(int(number) for number in first.port_numbers)
    # The second matrix's ports, in the order the first holds them.
    order = [list(second.port_numbers).index(number) for number in ports]
    for index, number in enumerate(ports):
        _require_one_reference(
            np.asarray(first.reference)[rows, index],
            number in first.self_referenced,
            np.asarray(second.reference)[columns, order[index]],
            number in second.self_referenced,
            number,
            first_by,
            second_by,
        )

    a = np.asarray(first.s)[rows]
    b = np.asarray(second.s)[columns][:, order][:, :, order]
    # Turning a port in either matrix brings the two equally close, so the
    # choice is made on one, and each port is turned in the matrix whose solver
    # chose its sign.
    theirs = _signed(second)
    mine = _signed(first) - theirs
    turned = _turned(a, b, ports, sorted((mine | theirs) & set(ports)))
    a = _turn(a, ports, [number for number in turned if number in mine])
    b = _turn(b, ports, [number for number in turned if number in theirs])

    frequency = np.asarray(first.frequency, dtype=float)[rows]
    terms = tuple(
        term
        for row, receiving in enumerate(ports)
        for column, driving in enumerate(ports)
        if (term := _term_apart(a, b, frequency, row, column, receiving, driving)) is not None
    )
    if not terms:
        raise Incomparable(
            f"no term was measured by both {first_by} and {second_by}: each drove "
            "different ports. Drive one port in both and run again"
        )
    return Comparison(
        by=(first_by, second_by),
        frequency=frequency,
        port_numbers=ports,
        first=a,
        second=b,
        terms=terms,
        turned=tuple((number, first_by if number in mine else second_by) for number in turned),
        unshared=int(
            np.unique(first.frequency).size + np.unique(second.frequency).size - 2 * rows.size
        ),
        unchecked=tuple(by for by, known in ((first_by, drawn), (second_by, other)) if not known),
    )


def _signed(result: SParameters) -> set[int]:
    listed = result.provenance.get(SIGN_BY_RULE)
    return set() if listed is None else {int(number) for number in listed}


def _turn(s: np.ndarray, ports: Sequence[int], turned: Sequence[int]) -> np.ndarray:
    """``s`` with each port of ``turned`` taken the other way round."""
    signs = np.array([-1.0 if number in turned else 1.0 for number in ports])
    return s * signs[None, :, None] * signs[None, None, :]


def _shared(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The indices into each band of the frequencies both hold, rising.

    Each frequency of the first is paired with the nearest of the second where
    that is within :data:`SAME_FREQUENCY`, and no frequency is paired twice: a
    frequency a band holds twice is one frequency.
    """
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    rows: list[int] = []
    columns: list[int] = []
    for row in np.argsort(a, kind="stable"):
        nearest = int(np.argmin(np.abs(b - a[row]))) if b.size else -1
        if nearest < 0 or nearest in columns or (rows and a[rows[-1]] == a[row]):
            continue
        if abs(b[nearest] - a[row]) <= SAME_FREQUENCY * abs(a[row]):
            rows.append(int(row))
            columns.append(nearest)
    return np.array(rows, dtype=int), np.array(columns, dtype=int)


def _require_one_reference(
    mine: np.ndarray,
    mine_own: bool,
    theirs: np.ndarray,
    theirs_own: bool,
    number: int,
    first_by: str,
    second_by: str,
) -> None:
    """Refuse a port the two matrices reference to different impedances.

    :param mine: and ``theirs``: the port's reference at each shared frequency.
    :param mine_own: and ``theirs_own``: whether each references the port to its
        own impedance.
    """
    if (mine_own and theirs_own) or np.allclose(mine, theirs, rtol=SAME_IMPEDANCE, atol=0.0):
        return
    said = describe_reference(mine[:, None], (number,), (number,) if mine_own else ())
    other = describe_reference(theirs[:, None], (number,), (number,) if theirs_own else ())
    raise Incomparable(
        f"port {number} is referenced to {said} in what {first_by} solved and to "
        f"{other} in what {second_by} solved, so the two matrices are ratios against "
        "different impedances. Reference the port to its own impedance, or to one "
        "number in both, and run again"
    )


def _turned(a: np.ndarray, b: np.ndarray, ports: Sequence[int], free: Sequence[int]) -> list[int]:
    """The ports of ``free`` to turn half a cycle, to bring ``b`` closest to ``a``.

    Every way of turning them is tried, and the one leaving the smallest sum of
    squared differences over every term and frequency both hold is kept. None is
    turned where turning brings the two no closer.
    """
    known = np.isfinite(a) & np.isfinite(b)
    best: tuple[float, list[int]] = (_apart(a, b, known), [])
    for count in range(1, len(free) + 1):
        for chosen in itertools.combinations(free, count):
            spread = _apart(a, _turn(b, ports, chosen), known)
            if spread < best[0]:
                best = (spread, list(chosen))
    return best[1]


def _apart(a: np.ndarray, b: np.ndarray, known: np.ndarray) -> float:
    return float(np.sum(np.abs(a[known] - b[known]) ** 2))


def _term_apart(
    a: np.ndarray,
    b: np.ndarray,
    frequency: np.ndarray,
    row: int,
    column: int,
    receiving: int,
    driving: int,
) -> Term | None:
    """One term's largest difference, or ``None`` where no frequency holds it in
    both."""
    difference = np.abs(a[:, row, column] - b[:, row, column])
    known = np.isfinite(difference)
    if not known.any():
        return None
    at = int(np.flatnonzero(known)[np.argmax(difference[known])])
    return Term(
        receiving=receiving,
        driving=driving,
        worst=float(difference[at]),
        at=float(frequency[at]),
        first=float(abs(a[at, row, column])),
        second=float(abs(b[at, row, column])),
    )


def _difference(value: float) -> str:
    """``0.014 (-37.1 dB)``, and no difference at all as ``0``."""
    return _magnitude(value) if value > 0.0 else "0"


def _frequencies(count: int) -> str:
    return "one frequency" if count == 1 else f"{count} frequencies"
