# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Closed-form references for microstrip transmission line.

The formulas are Hammerstad's quasi-static approximations [1], for a conductor
of zero thickness. They are accurate to roughly 1% against full-wave solutions
for ``0.05 <= W/h <= 20`` and ``1 <= eps_r <= 16``, well below the cutoff
frequency of the first higher-order mode.

Quasi-static means *no dispersion*: the returned impedance is the low-frequency
limit. Compare against simulation near the bottom of the band, or the reference
itself becomes the error term. On 1.6 mm FR4 the quasi-static assumption is good
to well under 1% at 1 GHz and visibly wrong by 10 GHz.

All lengths are in consistent units (their ratio is what matters); impedance is
in ohms.

[1] E. Hammerstad, "Equations for Microstrip Circuit Design", Proc. 5th European
    Microwave Conference, 1975, pp. 268-272.
"""

from __future__ import annotations

import numpy as np

# Free-space wave impedance. Both Hammerstad branches take it from here: the
# narrow one used the textbook literal 60.0 (i.e. 120*pi/2pi) against the wide
# one's mu0*c, a step in the reference itself comparable to the margin the
# microstrip gate reports against it.
#
# Hammerstad's two branches remain discontinuous at W/h = 1 by 0.388 % - the
# fits were derived over separate ranges and never made to meet. Anything gated
# near W/h = 1 is comparing against a reference with a step in it wider than the
# agreement being claimed.
Z_FREE_SPACE = 4.0e-7 * np.pi * 299792458.0


def effective_permittivity(width: float, height: float, eps_r: float) -> float:
    """Effective relative permittivity of a microstrip line.

    The field is split between the substrate and the air above it, so the line
    behaves as if embedded in a uniform medium of this permittivity. Always
    between 1 and ``eps_r``.

    :param width: Strip width.
    :param height: Substrate thickness.
    :param eps_r: Substrate relative permittivity.
    """
    _check_positive(width=width, height=height)
    if eps_r < 1.0:
        raise ValueError(f"eps_r must be >= 1, got {eps_r}")

    u = width / height
    mean = (eps_r + 1.0) / 2.0
    half_diff = (eps_r - 1.0) / 2.0
    fill = (1.0 + 12.0 / u) ** -0.5

    if u <= 1.0:
        # Narrow strips need the extra term; it vanishes at u == 1.
        return mean + half_diff * (fill + 0.04 * (1.0 - u) ** 2)
    return mean + half_diff * fill


def characteristic_impedance(width: float, height: float, eps_r: float) -> float:
    """Characteristic impedance of a microstrip line, in ohms.

    Zero conductor thickness. A finite-thickness correction does not belong
    here unless a gate uses it: the obvious form to reach for is Wheeler's
    coefficient-free one, which is not the published Hammerstad--Bekkadal dW,
    and the difference is worth several times the agreement the gate reports.
    The one
    test that pinned its magnitude allowed a 6:1 window that admitted both. The
    gate argues for the zero-thickness form anyway: comparing a solver that
    models 35 um copper against the thickness-corrected reference shifts the
    target ~1.1% in the direction that makes a healthy solver look broken.
    """
    _check_positive(width=width, height=height)
    u = width / height
    sqrt_eps = np.sqrt(effective_permittivity(width, height, eps_r))

    if u <= 1.0:
        # eta/2pi, not the literal 60.0 the textbooks print - see Z_FREE_SPACE.
        return float(Z_FREE_SPACE / (2.0 * np.pi) / sqrt_eps * np.log(8.0 / u + u / 4.0))
    return float(Z_FREE_SPACE / sqrt_eps / (u + 1.393 + 0.667 * np.log(u + 1.444)))


def parallel_plate_impedance(width: float, height: float, eps_r: float) -> float:
    """Impedance of an ideal parallel-plate line, in ohms.

    Exact in the limit ``width >> height``, where fringing is negligible. This
    is not a microstrip model - it exists so the microstrip formula can be
    checked against a result derived independently of it.
    """
    _check_positive(width=width, height=height)
    return float(Z_FREE_SPACE / np.sqrt(eps_r) * height / width)


def _check_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0, got {value}")


# ---------------------------------------------------------------------------
# Rectangular waveguide
#
# Unlike the microstrip formulas above, these are not fits. They fall directly
# out of separating Maxwell's equations in a hollow rectangular pipe of perfect
# conductor, and they are exact - there is no accuracy figure to quote and no
# error bar for a discretisation mistake to hide inside. That is what makes
# them worth having: a microstrip mesh that was four times too coarse at the
# conductor edge sat comfortably inside Hammerstad's own 1% uncertainty for a
# whole day. It could not have hidden here.
#
# Following Pozar, *Microwave Engineering*, as openEMS' own RectWGPort does.
# ---------------------------------------------------------------------------

SPEED_OF_LIGHT = 299_792_458.0

# VACUUM_IMPEDANCE = 376.730313668 was here: a third copy of Z_FREE_SPACE, as a
# rounded literal, disagreeing with it in the tenth digit. Only the two deleted
# functions above reached it. Anything needing it takes Z_FREE_SPACE.


def cutoff_wavenumber(width: float, height: float, m: int = 1, n: int = 0) -> float:
    """Transverse wavenumber of mode TE/TM(m,n), in rad per unit length.

    :param width: Broad wall dimension ``a``.
    :param height: Narrow wall dimension ``b``, in the same units.
    """
    if width <= 0 or height <= 0:
        raise ValueError("waveguide dimensions must be positive")
    if m < 0 or n < 0 or (m == 0 and n == 0):
        raise ValueError(f"TE{m}{n} is not a propagating mode")
    return np.sqrt((m * np.pi / width) ** 2 + (n * np.pi / height) ** 2)


def cutoff_frequency(
    width: float, height: float, m: int = 1, n: int = 0, epsilon_r: float = 1.0
) -> float:
    """Cutoff frequency in Hz, for dimensions given in metres.

    Below this the mode does not propagate: the phase constant turns imaginary
    and the field decays exponentially instead of travelling. For the dominant
    TE10 mode this reduces to ``c / 2a``, independent of the narrow wall.
    """
    kc = cutoff_wavenumber(width, height, m, n)
    return SPEED_OF_LIGHT * kc / (2 * np.pi * np.sqrt(epsilon_r))


def phase_constant(
    frequency: np.ndarray,
    width: float,
    height: float,
    m: int = 1,
    n: int = 0,
    epsilon_r: float = 1.0,
) -> np.ndarray:
    """Guide phase constant beta, in rad per metre. Dimensions in metres.

    Real above cutoff, imaginary below it. Returned as a complex array so the
    evanescent region is representable rather than silently NaN.
    """
    frequency = np.asarray(frequency, dtype=float)
    k = 2 * np.pi * frequency * np.sqrt(epsilon_r) / SPEED_OF_LIGHT
    kc = cutoff_wavenumber(width, height, m, n)
    return np.sqrt(np.asarray(k**2 - kc**2, dtype=complex))


# This is the module the project's correctness rests on, and an untested closed
# form in it is worse than none: a guide_wavelength dividing by np.real(beta),
# which is 0 below cutoff, carries a reachable division by zero nothing catches.
# Add such helpers back
# with the gate that needs them.
