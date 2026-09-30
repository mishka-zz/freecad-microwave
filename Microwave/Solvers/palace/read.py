# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Read the scattering table a driven run wrote, and the power through its ports.

One file holds the whole matrix: a frequency and, per driven port, a magnitude
in decibels and a phase in degrees at each of the ports. A second holds, per
driven port, the power that left the model through each port's face, in watts
against the watt the driven port sends in. For a lumped port the table holds the
power the port's elements give the model, which is the same figure with the
sign turned, and it is turned here. In an open run it also holds the power
through the open surface, measured two-sided as a lumped port's is, and turned
the same way. A third, where a port was given a
line to read its voltage along, holds its power-voltage impedance. A lumped port
states its resistance, which is in the configuration rather than in a table.
What comes back here is the complex matrix, those powers and impedances and the
port numbers their rows and columns stand for, and nothing else - assembling a result object
is the neutral layer's, and this adapter never reaches it.

What Palace does with the file decides the refusals. It rewrites every table
whole at each step and marks none of them partial, so a run stopped part way
leaves a well-formed file holding fewer samples than the band asked for. And a
driven run whose excitations do not each drive exactly one port writes no
scattering table at all and says nothing about it, so the file being absent is
an answer about the run rather than about the disk.

A column is named by the port it was measured at and the excitation it was driven
from, and Palace requires an excitation past the first to carry the number of the
port it drives - so both halves of a column's name are port numbers.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .capabilities import HZ_PER_GHZ
from .config import Driven, LumpedPort, WavePort

__all__ = [
    "FLUX_TABLE",
    "IMPEDANCE_TABLE",
    "MODES_TABLE",
    "POWER_VOLTAGE",
    "RESISTANCE",
    "TABLE",
    "ResultsError",
    "Scattering",
    "modes",
    "scattering",
]

#: What the table is called in the output directory.
TABLE = "port-S.csv"

#: What the table of power through each port's face is called, and one of its
#: columns: the port it was measured at, then the excitation in brackets where
#: the run has more than one and nothing where it has one
#: (``palace/models/postoperatorcsv.cpp:613, 646-649``).
FLUX_TABLE = "surface-F.csv"
_FLUX = "Φ_pow[{out}]{excitation} (W)"

#: What the table of the ports' impedance is called, and the column of one
#: port's power-voltage impedance, in ohms. Palace writes it where a port was
#: given a line to read its voltage along
#: (``palace/models/postoperatorcsv.cpp:1254-1269``).
IMPEDANCE_TABLE = "port-Z.csv"
_IMPEDANCE = "Re{{Z_PV[{port}]}} (Ohm)"

#: Which impedance a port the run read a voltage at states, as the result's
#: provenance says it.
POWER_VOLTAGE = (
    "the power-voltage impedance of its mode, |V|^2 / 2P with P the power the mode "
    "carries, V read along the line across its narrow side at the middle of its broad side"
)

#: What a lumped port states. Palace normalises a lumped port's incident and
#: reflected waves to its resistance (``LumpedPortData::GetExcitationRefResistance``
#: in ``palace/models/lumpedportoperator.hpp``), so the matrix is referenced to it.
RESISTANCE = "resistance"

#: What a mode run's table of propagation constants is called, and the columns
#: of each one's real and imaginary part, in inverse metres.
MODES_TABLE = "mode-kn.csv"
_REAL = "Re{kn} (1/m)"
_IMAGINARY = "Im{kn} (1/m)"

#: The column naming the frequency, in gigahertz, which is what every frequency
#: in a Palace file is written in.
FREQUENCY = "f (GHz)"

#: A magnitude column and the phase column that goes with it. The first number
#: is the port the wave was measured at and the second is the excitation it was
#: driven from, which Palace requires to be the driving port's own number.
_MAGNITUDE = "|S[{out}][{port}]| (dB)"
_PHASE = "arg(S[{out}][{port}]) (deg.)"

#: What a decade of amplitude is worth in decibels. Twenty rather than ten,
#: because a scattering parameter is a wave amplitude and not a power.
_DECIBELS = 20.0


class ResultsError(Exception):
    """The run wrote no table this adapter can read as its answer."""


@dataclass(frozen=True)
class Scattering:
    """The matrix a driven run measured.

    :param frequency: the band, in Hertz, one entry per sample.
    :param out: the port each row was measured at.
    :param driven: the port each column was driven from, in the order the columns
        stand in. Shorter than ``out`` where the run drove some of the ports
        rather than all of them.
    :param matrix: complex, indexed by sample, then row, then column.
    :param flux: the power that left the model through each port's face, in
        watts against the watt each driven port sends in, indexed as
        ``matrix`` is. Negative where more came in than went out, which the
        driven port's own face is.
    :param dissipates: whether anything in the model the run solved turns power
        into heat.
    :param modelled: how the run was given each lossy material, as
        ``pipeline.modelled`` records it. Empty as read, since the tables say
        nothing about it.
    :param impedance: each port's impedance in ohms, indexed by sample, then by
        the port as ``out`` orders them: a wave port's power-voltage impedance,
        a lumped port's resistance, and ``nan`` for a wave port the run read no
        voltage at. ``None`` where no port states one.
    :param stated: for each port ``impedance`` holds a figure for, by number,
        which impedance that is - :data:`POWER_VOLTAGE` or :data:`RESISTANCE`.
    :param radiated: the power that left the model through the open surface, in
        watts against the watt each driven port sends in, indexed by sample and
        then by column as ``matrix`` is. ``None`` where the run has no open
        surface.
    :param sign_by_rule: each wave port given no line to read a voltage along,
        whose mode Palace signed by a rule of its own over a quarter of the
        face (``palace/models/waveportoperator.cpp:650-699``) rather than by
        the direction the line runs.
    """

    frequency: Any
    out: tuple[int, ...]
    driven: tuple[int, ...]
    matrix: Any
    flux: Any
    dissipates: bool
    modelled: tuple[dict[str, Any], ...] = ()
    impedance: Any = None
    stated: dict[int, str] = field(default_factory=dict)
    radiated: Any = None
    sign_by_rule: tuple[int, ...] = ()


def scattering(directory: str | Path, run: Driven) -> Scattering:
    """The matrix in ``directory``, read against the run that asked for it."""
    path = Path(directory) / TABLE
    if not path.is_file():
        raise ResultsError(
            f"there is no scattering table at {path}. Palace writes one only where each "
            "excitation drives exactly one port, and says nothing where it does not"
        )

    head, body = _table(path)
    if len(body) < run.sweep.samples:
        raise ResultsError(
            f"{path} holds {len(body)} of the {run.sweep.samples} samples the band asked "
            "for. Every table is rewritten whole at each step and none of them is marked "
            "partial, so this is a run that stopped rather than a run that finished"
        )

    driven = run.excitations
    out = tuple(sorted(port.index for port in run.ports))

    frequency = body[:, _column(head, FREQUENCY, path)] * HZ_PER_GHZ
    matrix = np.empty((len(body), len(out), len(driven)), dtype=complex)
    for row, measured in enumerate(out):
        for column, port in enumerate(driven):
            decibels = body[:, _column(head, _MAGNITUDE.format(out=measured, port=port), path)]
            degrees = body[:, _column(head, _PHASE.format(out=measured, port=port), path)]
            matrix[:, row, column] = 10.0 ** (decibels / _DECIBELS) * np.exp(
                1j * np.radians(degrees)
            )

    unfinished = ~np.isfinite(matrix).all(axis=(1, 2))
    if unfinished.any():
        # Palace writes a solve that produced no number as nan, finishes, and
        # exits cleanly. A linear solve that diverged also warns, which the log
        # is read for; this is the table read on its own terms.
        raise ResultsError(
            f"{path} holds a term that is not a finite number at "
            f"{', '.join(f'{f:.6g}' for f in body[unfinished, _column(head, FREQUENCY, path)])} "
            "GHz, so the solve there produced no answer"
        )

    band = body[:, _column(head, FREQUENCY, path)]
    impedance = _impedance(Path(directory) / IMPEDANCE_TABLE, run, out, band)
    flux, radiated = _flux(Path(directory) / FLUX_TABLE, run, out, band)
    return Scattering(
        frequency=frequency,
        out=out,
        driven=driven,
        matrix=matrix,
        flux=flux,
        radiated=radiated,
        dissipates=run.dissipates,
        impedance=impedance,
        stated={
            port.index: RESISTANCE if isinstance(port, LumpedPort) else POWER_VOLTAGE
            for port in run.ports
            if impedance is not None and np.all(np.isfinite(impedance[:, out.index(port.index)]))
        },
        sign_by_rule=tuple(
            sorted(
                port.index
                for port in run.ports
                if isinstance(port, WavePort) and port.voltage is None
            )
        ),
    )


def _impedance(path: Path, run: Driven, out: tuple[int, ...], band: Any) -> Any:
    """Each port's impedance, read against the run that asked for it, or
    ``None`` where no port states one.

    A lumped port states its resistance at every sample. A wave port states its
    power-voltage impedance where it was given a line to read a voltage along,
    and Palace writes the table only where one was. A figure there that is not a
    positive number is a line that read no voltage: Palace reads zero where a
    point of the line stands outside the mesh (``palace/fem/interpolator.cpp:327``),
    so a line that missed the face reads zero.
    """
    stated = [
        port.index for port in run.ports if isinstance(port, WavePort) and port.voltage is not None
    ]
    resisted = {port.index: port.resistance for port in run.ports if isinstance(port, LumpedPort)}
    if not stated and not resisted:
        return None
    impedance = np.full((len(band), len(out)), np.nan)
    for index, resistance in resisted.items():
        impedance[:, out.index(index)] = resistance
    if not stated:
        return impedance
    if not path.is_file():
        raise ResultsError(
            f"there is no table of the ports' impedance at {path}, and the run asked for "
            f"one at port {', '.join(str(index) for index in stated)}"
        )
    head, body = _table(path)
    written = body[:, _column(head, FREQUENCY, path)]
    if written.shape != band.shape or not np.array_equal(written, band):
        raise ResultsError(
            f"{path} holds the frequencies {written.tolist()} GHz where the scattering "
            f"table holds {band.tolist()}, so the two are not one run's"
        )
    for index in stated:
        figure = body[:, _column(head, _IMPEDANCE.format(port=index), path)]
        if not np.all(np.isfinite(figure) & (figure > 0)):
            raise ResultsError(
                f"port {index} read an impedance of {figure.tolist()} ohm along the line "
                "across its face, and a port's impedance is positive. The line read no "
                "voltage, which is a line that missed the face"
            )
        impedance[:, out.index(index)] = figure
    return impedance


def _flux(path: Path, run: Driven, out: tuple[int, ...], band: Any) -> tuple[Any, Any]:
    """The power through each port's face, and through the open surface where the
    run has one, read against the run that asked for it and against the
    frequencies, in gigahertz, the scattering table holds.

    Asked of every driven run, so a table missing is a run that did not answer
    the configuration it was given, as the scattering table's is.
    """
    if not path.is_file():
        raise ResultsError(f"there is no table of the power through the ports at {path}")
    head, body = _table(path)
    if len(body) < run.sweep.samples:
        raise ResultsError(
            f"{path} holds {len(body)} of the {run.sweep.samples} samples the band asked "
            "for, so the run stopped before it finished"
        )
    written = body[:, _column(head, FREQUENCY, path)]
    if written.shape != band.shape or not np.array_equal(written, band):
        raise ResultsError(
            f"{path} holds the frequencies {written.tolist()} GHz where the scattering "
            f"table holds {band.tolist()}, so the two are not one run's"
        )
    driven = run.excitations
    # A lumped port's flux is measured as what it gives the model, where every
    # other figure here is what leaves it: see ``config.LumpedPort.to_flux``.
    given = {port.index for port in run.ports if isinstance(port, LumpedPort)}
    flux = np.empty((len(body), len(out), len(driven)), dtype=float)
    radiated = None if run.radiated is None else np.empty((len(body), len(driven)))
    for column, port in enumerate(driven):
        excitation = "" if len(driven) == 1 else f"[{port}]"
        for row, measured in enumerate(out):
            name = _FLUX.format(out=measured, excitation=excitation)
            sign = -1.0 if measured in given else 1.0
            flux[:, row, column] = sign * body[:, _column(head, name, path)]
        if radiated is not None:
            # Measured two-sided, as a lumped port's is, so what is written is
            # the power going in: see ``config.Driven.to_radiated_flux``.
            name = _FLUX.format(out=run.radiated, excitation=excitation)
            radiated[:, column] = -body[:, _column(head, name, path)]
    return flux, radiated


def modes(directory: str | Path) -> tuple[complex, ...]:
    """The propagation constants a mode run found, in inverse metres, in the
    order Palace wrote them."""
    path = Path(directory) / MODES_TABLE
    if not path.is_file():
        raise ResultsError(f"the mode run left no table at {path}")
    head, body = _table(path)
    real, imaginary = _column(head, _REAL, path), _column(head, _IMAGINARY, path)
    return tuple(complex(row[real], row[imaginary]) for row in body)


def _table(path: Path) -> tuple[list[str], Any]:
    """The header and the numbers, with the padding Palace writes taken off."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ResultsError(f"{path} is empty")
    head = [cell.strip() for cell in rows[0]]
    try:
        body = np.array([[float(cell) for cell in row] for row in rows[1:]], dtype=float)
    except ValueError as error:
        raise ResultsError(f"{path} carries something that is not a number: {error}") from error
    if body.size == 0:
        raise ResultsError(f"{path} holds a header and no samples")
    return head, body


def _column(head: list[str], name: str, path: Path) -> int:
    try:
        return head.index(name)
    except ValueError:
        raise ResultsError(
            f"{path} carries no column {name!r}, so the run measured something other "
            "than what it was asked for"
        ) from None
