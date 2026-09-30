# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Take the ports off a uniform line solved at several lengths.

A port does not launch its wave at the plane it is drawn in. What stands
between the plane and the line - the current turning from the element into the
strip, the field fringing round the element's edges - is a small network of its
own, the same whatever the line's length. Each run's matrix is therefore that
network, the line, and the far port's network turned round::

    T(L) = X . M(Zc, k L) . reverse(X)

in the chain (ABCD) form, where ``M`` is an ideal TEM line of impedance ``Zc``
and electrical length ``k L``, ``k`` being the wavenumber in the material that
fills the line (:func:`wavenumber`), and ``X`` is a port's network from its
plane inward. A run at one length cannot tell the network from the line. Runs at
two lengths share the network and differ in the line, so the unknowns are fitted
per frequency over every length at once.

The network is modelled as a series impedance and a shunt admittance, in either
order, at each frequency on its own. That is a model of whatever the port adds
between its plane and the line, sized by two numbers, and not a claim that the
port is an inductance and a capacitance: nothing here holds the two numbers to a
dependence on frequency. The chain matrix of the whole two-port does not depend
on the reference its scattering matrix is stated against. How the fitted network
is split between its two numbers does: the fit weighs the four terms by the
reference it is given, and the weaker term follows that weighting.

**What the runs cannot separate**, each exactly or nearly invisible in every
term of every run:

- An ideal transformer of ratio ``n`` in the network turns the line into one of
  ``n**2 Zc`` and changes nothing else. So the impedance fitted here is
  conditional on the network holding none.
- The two orders are that same transformer. A series impedance ``Z`` then a
  shunt admittance ``Y`` is a shunt admittance ``Y / (1 + Z Y)``, a series
  impedance ``Z (1 + Z Y)`` and a transformer of ratio ``1 + Z Y``. Where
  ``Z Y`` is real, as it is for an inductance and a capacitance, the two fits
  reproduce the runs alike and their impedances stand in the ratio
  ``(1 + Z Y)**2``. How far apart they are says how large the network is, and
  nothing about which model is right.
- A network of each port's own is not identifiable, which is why one network
  serves both ports here. To first order in the networks, an inductance ``L``
  moved from one port's network to the other's, together with a capacitance
  ``L / Zc**2`` moved the same way, moves no term of any run. What a
  difference between the two ports does reach is the difference of the two
  reflections, ``S11 - S22``, which a caller reads off the runs directly.

The line's electrical length is not fitted: it is the drawn length at the
wavenumber given, so a line that is not the drawing, or a fill that is not the
one the wavenumber was taken from, leaves a residual rather than a fitted length.

**The wave's own phase** comes from no model of the network at all. Over two
lengths ``T(L2) T(L1)^-1 = X M(Zc, k (L2 - L1)) X^-1``, a similarity transform,
so its eigenvalues are the line's own, ``exp(+-j k (L2 - L1))``, whatever ``X``
is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from Microwave.units import SPEED_OF_LIGHT

#: The two orders a port's network is modelled in, from the port's plane inward.
SERIES_FIRST = "series first"
SHUNT_FIRST = "shunt first"
TOPOLOGIES = (SERIES_FIRST, SHUNT_FIRST)


def _references(references: float | Sequence[float]) -> np.ndarray:
    """Each port's reference, from one figure for both or one per port."""
    stated = np.broadcast_to(np.asarray(references, dtype=float), (2,))
    if not np.all(np.isfinite(stated) & (stated > 0)):
        raise ValueError(f"a reference impedance is a positive number of ohms, got {stated}")
    return stated


def abcd(s: np.ndarray, references: float | Sequence[float]) -> np.ndarray:
    """The chain matrix of a two-port whose scattering matrix ``s`` is referenced
    to ``references`` - one real impedance for both ports, or one each.

    ``s`` is indexed ``[out - 1][driven - 1]``, as a matrix is written. Taken
    through the impedance matrix, ``Z = G (I - S)^-1 (I + S) G`` with ``G`` the
    root of each reference on the diagonal, which is the power waves' definition
    for real references and reduces to the familiar closed form where the two
    are equal.
    """
    root = np.diag(np.sqrt(_references(references)))
    unit = np.eye(2)
    z = root @ np.linalg.inv(unit - s) @ (unit + s) @ root
    return np.array(
        [
            [z[0, 0] / z[1, 0], (z[0, 0] * z[1, 1] - z[0, 1] * z[1, 0]) / z[1, 0]],
            [1 / z[1, 0], z[1, 1] / z[1, 0]],
        ]
    )


def scattering(t: np.ndarray, references: float | Sequence[float]) -> np.ndarray:
    """The scattering matrix of the chain matrix ``t``, referenced to
    ``references``: the inverse of :func:`abcd`."""
    stated = _references(references)
    a, b, c, d = t[0, 0], t[0, 1], t[1, 0], t[1, 1]
    z = np.array([[a / c, (a * d - b * c) / c], [1 / c, d / c]])
    root = np.diag(np.sqrt(stated))
    return np.linalg.inv(root) @ (z - np.diag(stated)) @ np.linalg.inv(z + np.diag(stated)) @ root


def wavenumber(frequency: float, permittivity: float, permeability: float) -> float:
    """The wavenumber of a TEM wave in a medium, in radians per metre: the vacuum
    one times the root of the relative permittivity and permeability."""
    return float(2 * np.pi * frequency / SPEED_OF_LIGHT * np.sqrt(permittivity * permeability))


def line(impedance: float, electrical_length: float) -> np.ndarray:
    """An ideal TEM line of that impedance and that length in radians."""
    cos, sin = np.cos(electrical_length), np.sin(electrical_length)
    return np.array([[cos, 1j * impedance * sin], [1j * sin / impedance, cos]])


def network(series: complex, shunt: complex, topology: str) -> np.ndarray:
    """A port's network from its plane inward: a series impedance and a shunt
    admittance, in the order ``topology`` names."""
    across = np.array([[1, series], [0, 1]], dtype=complex)
    down = np.array([[1, 0], [shunt, 1]], dtype=complex)
    if topology == SERIES_FIRST:
        return across @ down
    if topology == SHUNT_FIRST:
        return down @ across
    raise ValueError(f"no topology {topology!r}; there are {TOPOLOGIES}")


def reverse(t: np.ndarray) -> np.ndarray:
    """The same reciprocal two-port seen from its other end."""
    return np.array([[t[1, 1], t[0, 1]], [t[1, 0], t[0, 0]]])


def transformer(ratio: float) -> np.ndarray:
    """An ideal transformer, ``ratio`` turns on the port's side to one inside."""
    return np.array([[ratio, 0], [0, 1 / ratio]], dtype=complex)


@dataclass(frozen=True)
class Fit:
    """What one frequency's runs de-embed to.

    :param impedance: the line's, in ohms.
    :param series: the impedance in each port's network, in ohms.
    :param shunt: the admittance in each port's network, in siemens.
    :param residual: the root mean square of what the model leaves of the
        measured chain matrices, each entry made dimensionless by the reference,
        so a fit that reproduced the runs exactly leaves nothing.
    :param impedance_error: the standard error of ``impedance``, in ohms.
    :param reactance_error: of the imaginary part of ``series``, in ohms.
    :param susceptance_error: of the imaginary part of ``shunt``, in siemens.

    Each standard error is what the fit leaves, carried through how strongly the
    runs depend on that unknown: the residual variance over the degrees of
    freedom, times the diagonal of the inverse normal matrix. A figure the runs
    barely reach has a wide one, whatever the residual.
    """

    impedance: float
    series: complex
    shunt: complex
    residual: float
    impedance_error: float
    reactance_error: float
    susceptance_error: float

    def inductance(self, angular: float) -> float:
        """The series reactance as an inductance, in henries."""
        return float(self.series.imag / angular)

    def capacitance(self, angular: float) -> float:
        """The shunt susceptance as a capacitance, in farads."""
        return float(self.shunt.imag / angular)

    def inductance_error(self, angular: float) -> float:
        """The standard error of :meth:`inductance`, in henries."""
        return self.reactance_error / angular

    def capacitance_error(self, angular: float) -> float:
        """The standard error of :meth:`capacitance`, in farads."""
        return self.susceptance_error / angular


def fit(
    measured: Sequence[np.ndarray],
    electrical_lengths: Sequence[float],
    reference: float,
    topology: str,
) -> Fit:
    """The line's impedance and the ports' network, fitted to runs at several
    lengths of one line at one frequency.

    :param measured: each run's chain matrix.
    :param electrical_lengths: each run's line length times the wavenumber, in
        radians, in the same order.
    :param reference: an impedance of the order of the line's, which the entries
        are scaled by so that the four weigh alike.
    """
    if len(measured) != len(electrical_lengths) or len(measured) < 2:
        raise ValueError(
            f"{len(measured)} runs and {len(electrical_lengths)} lengths: a fit wants one "
            "length a run, and two runs at least, a single run being three unknowns short"
        )
    scale = np.array([[1.0, 1.0 / reference], [reference, 1.0]])

    def unpack(p: np.ndarray) -> tuple[float, complex, complex]:
        return (
            float(p[0]) * reference,
            complex(p[1], p[2]) * reference,
            complex(p[3], p[4]) / reference,
        )

    def model(p: np.ndarray) -> np.ndarray:
        impedance, series, shunt = unpack(p)
        x = network(series, shunt, topology)
        left = np.concatenate(
            [
                ((x @ line(impedance, theta) @ reverse(x) - t) * scale).ravel()
                for t, theta in zip(measured, electrical_lengths)
            ]
        )
        return np.concatenate([left.real, left.imag])

    # Started from the impedance the run whose line is furthest from a whole
    # number of half wavelengths reads with no network taken off at all, which
    # is where that reading is least sensitive to the network.
    start = max(zip(measured, electrical_lengths), key=lambda pair: abs(np.sin(pair[1])))[0]
    guess = np.zeros(5)
    guess[0] = abs(np.sqrt(start[0, 1] / start[1, 0])) / reference
    best = least_squares(model, guess, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
    impedance, series, shunt = unpack(best.x)
    freedom = best.fun.size - best.x.size
    variance = float(np.sum(np.square(best.fun))) / freedom
    deviation = np.sqrt(variance * np.diag(np.linalg.inv(best.jac.T @ best.jac)))
    return Fit(
        impedance=impedance,
        series=series,
        shunt=shunt,
        residual=float(np.sqrt(np.mean(np.square(best.fun)))),
        impedance_error=float(deviation[0]) * reference,
        reactance_error=float(deviation[2]) * reference,
        susceptance_error=float(deviation[4]) / reference,
    )


def phase_across(shorter: np.ndarray, longer: np.ndarray) -> float:
    """The line's phase over the difference of two lengths, in radians, from
    their chain matrices and nothing else.

    The eigenvalues of ``longer . shorter^-1`` are ``exp(+-j phase)``. The angle
    is read in ``[0, pi]``, so a difference of more than half a wavelength comes
    back folded, and the caller keeps the difference below that.
    """
    eigenvalues = np.linalg.eigvals(longer @ np.linalg.inv(shorter))
    return float(np.mean(np.abs(np.angle(eigenvalues))))
