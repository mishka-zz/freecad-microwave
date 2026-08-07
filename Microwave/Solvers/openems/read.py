# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Native output back into results. Runs in FreeCAD; numpy and stdlib only.

The objects here are the adapter's side of the boundary: still openEMS-shaped
but no longer openEMS-dependent. Turning them into the neutral result objects -
``EMSParameters`` on an ``skrf.Network`` substrate, and friends - is the
document layer's job, and needs skrf, which this does not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .driver import RESULTS_NAME


class ResultsError(Exception):
    """The results file is missing, unreadable, or does not match its input."""


def _complex(data: dict[str, list[float]]) -> np.ndarray:
    return np.asarray(data["re"], dtype=float) + 1j * np.asarray(data["im"], dtype=float)


@dataclass(frozen=True)
class PortResult:
    """What one port measured."""

    number: int
    z0: np.ndarray
    incident: np.ndarray
    reflected: np.ndarray

    @property
    def impedance(self) -> np.ndarray:
        """Characteristic impedance magnitude, in ohms.

        This is openEMS' ``sqrt(Et*dEt / (Ht*dHt))`` - the voltage-current
        definition, sampled at one measurement plane. It is genuinely
        frequency-dependent for a microstrip, so comparing it against a
        quasi-static closed form is only valid at the bottom of the band.
        """
        return np.abs(self.z0)


@dataclass(frozen=True)
class Results:
    """One solve, parsed."""

    frequency: np.ndarray
    excited_port: int
    ports: dict[int, PortResult]
    s_parameters: dict[str, np.ndarray]
    provenance: dict[str, Any] = field(default_factory=dict)

    def port(self, number: int = 1) -> PortResult:
        if number not in self.ports:
            raise ResultsError(f"no results for port {number}; this run has {sorted(self.ports)}")
        return self.ports[number]

    def s(self, receiving: int, driving: int | None = None) -> np.ndarray:
        """One S-parameter. ``driving`` defaults to the port this run excited."""
        driving = self.excited_port if driving is None else driving
        key = f"S{receiving}{driving}"
        if key not in self.s_parameters:
            raise ResultsError(
                f"{key} was not measured; this run excited port "
                f"{self.excited_port} and has {sorted(self.s_parameters)}"
            )
        return self.s_parameters[key]

    def band(self, low: float, high: float) -> np.ndarray:
        """Boolean mask selecting frequency bins in ``[low, high]`` Hz."""
        return (self.frequency >= low) & (self.frequency <= high)

    @property
    def reproducible(self) -> bool:
        """Whether this run would give the same numbers again.

        False when energy termination was used - openEMS evaluates that
        criterion on a wall-clock timer, so the run length depends on machine
        load. Anything comparing two results has to check this first.
        """
        return bool(self.provenance.get("reproducible", False))

    @property
    def tail_share(self) -> dict[int, float]:
        """Per port, what the end of its record was still worth in this run's S.

        JSON has no integer keys, so they come back as strings and are put back
        the way :func:`read` puts the port numbers back.
        """
        found = self.provenance.get("tail_share") or {}
        return {int(number): float(value) for number, value in found.items()}

    def matches(self, digest: str) -> bool:
        """Whether these results came from the envelope with this digest."""
        return self.provenance.get("envelope_digest") == digest


def read(directory: str | Path) -> Results:
    """Load results written by the driver."""
    directory = Path(directory)
    path = directory if directory.is_file() else directory / RESULTS_NAME
    if not path.is_file():
        raise ResultsError(f"no results at {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ResultsError(f"{path} is not readable JSON: {error}") from error

    try:
        ports = {
            int(number): PortResult(
                number=int(number),
                z0=_complex(values["z0"]),
                incident=_complex(values["incident"]),
                reflected=_complex(values["reflected"]),
            )
            for number, values in data["ports"].items()
        }
        return Results(
            frequency=np.asarray(data["frequency"], dtype=float),
            excited_port=int(data["excited_port"]),
            ports=ports,
            s_parameters={key: _complex(value) for key, value in data["s_parameters"].items()},
            provenance=data.get("provenance", {}),
        )
    except KeyError as error:
        raise ResultsError(f"{path} is missing {error}") from error
    except (TypeError, ValueError, AttributeError) as error:
        # A key that is present but holds the wrong thing - a port numbered
        # 'one', a frequency of 'dc', a list where the ports belong. Catching
        # only the absent key leaves a type fault arriving as a traceback, or as
        # an "is missing" that is not true.
        raise ResultsError(f"{path} is not the results of a run: {error}") from error
