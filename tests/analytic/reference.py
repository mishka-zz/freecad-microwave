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
from scipy.optimize import brentq
from scipy.special import spherical_jn

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


# ---------------------------------------------------------------------------
# Coaxial line
#
# Exact, like the waveguide formulas above and unlike Hammerstad. The field
# between two concentric perfect conductors is purely TEM, the potential
# problem is Laplace's equation in one variable, and the answer falls out of it
# in closed form with no fit and no range of validity. There is no accuracy
# figure to quote, and so nowhere for a discretisation mistake to hide.
#
# It holds below the first higher-order mode, whose cutoff is where the mean
# circumference reaches a wavelength - so a line worked well under
# `higher_mode_cutoff` carries one mode and one impedance.
# ---------------------------------------------------------------------------


def coaxial_impedance(inner: float, outer: float, eps_r: float = 1.0) -> float:
    """Characteristic impedance of a coaxial line, in ohms.

    :param inner: Outer radius of the inner conductor.
    :param outer: Inner radius of the outer conductor, in the same units. Only
        the ratio matters, so any consistent unit does.

    The capacitance per length of two concentric cylinders is
    ``2 pi eps / ln(outer/inner)`` and the inductance per length is
    ``mu ln(outer/inner) / 2 pi``; ``sqrt(L/C)`` is what this returns. Both
    come from the same logarithm, which is why the geometry enters only as that
    ratio and why the answer is exact rather than fitted.
    """
    _check_positive(inner=inner, eps_r=eps_r)
    if outer <= inner:
        raise ValueError(f"outer radius must exceed inner, got {outer} <= {inner}")
    return float(Z_FREE_SPACE / (2 * np.pi * np.sqrt(eps_r)) * np.log(outer / inner))


def higher_mode_cutoff(inner: float, outer: float, eps_r: float = 1.0) -> float:
    """Where the coaxial TE11 mode starts to propagate, in Hz. Radii in metres.

    The standard approximation, exact in the limit of a thin annulus and a few
    per cent high for a wide one: the mode appears when a wavelength in the
    dielectric fits into the mean circumference. Its purpose is to keep a
    measurement of :func:`coaxial_impedance` below it, where the line carries
    one mode and therefore has one impedance at all - so an approximation on
    the safe side of a band edge is what is wanted, not a root of a Bessel
    determinant.
    """
    _check_positive(inner=inner, eps_r=eps_r)
    if outer <= inner:
        raise ValueError(f"outer radius must exceed inner, got {outer} <= {inner}")
    return float(SPEED_OF_LIGHT / (np.pi * (inner + outer) * np.sqrt(eps_r)))


# ---------------------------------------------------------------------------
# Spherical cavity
#
# Exact, like the waveguide and coaxial forms above: separating Maxwell's
# equations in a sphere of perfect conductor gives spherical Bessel functions in
# the radius, and a resonance is where one of them meets the wall condition.
# There is no fit, no range of validity and no accuracy figure to quote.
#
# What it is for is a shape a rectilinear grid cannot hold. Every other exact
# reference here describes a box, so the grid holds it and the geometry reaches
# the solver as what was drawn. A sphere never does, and its resonance depends
# on the radius its staircased surface actually has - so the closed form prices
# what the discretisation cost.
#
# A resonance rather than an impedance, because a frequency is read off *where*
# a feature sits. A driven line's impedance has to be read through its feed, and
# a lumped element bridging curved conductors carries a resistance and an
# inductance the grid decides; those move a coupling and the depth of a dip, and
# they do not move where a cavity resonates.
# ---------------------------------------------------------------------------

#: The mode families a sphere carries, each as the function of ``ka`` that
#: vanishes at a resonance. ``TE`` has no radial electric field and so needs the
#: tangential ``E`` it does have to vanish at the wall, which is ``j_n(ka) = 0``;
#: ``TM`` has no radial magnetic field and needs the tangential ``H`` to vanish,
#: which is the Riccati-Bessel derivative ``[x j_n(x)]' = 0``.
_CAVITY_CONDITIONS = {
    "TE": lambda n, x: spherical_jn(n, x),
    "TM": lambda n, x: spherical_jn(n, x) + x * spherical_jn(n, x, derivative=True),
}


def spherical_cavity_root(n: int = 1, p: int = 1, kind: str = "TM") -> float:
    """The ``p``-th root of the wall condition for polar order ``n``, as ``ka``.

    :param n: Polar order. The azimuthal index does not appear: modes differing
        only in it are degenerate, which is why a sphere answers with a few
        strong lines rather than many.
    :param p: Which root, counting outward from the origin. It is the number of
        radial half-variations of the field.
    :param kind: ``TM`` or ``TE``. The dominant mode of a sphere is TM at
        ``n = p = 1``; the lowest TE sits well above it.

    Dimensionless, so it is the whole geometry dependence: a sphere's spectrum
    is this set of numbers divided by its radius.
    """
    if kind not in _CAVITY_CONDITIONS:
        raise ValueError(f"mode family must be one of {sorted(_CAVITY_CONDITIONS)}, got {kind!r}")
    if n < 1:
        # n = 0 has no angular variation, which would need a radial field with
        # nothing curling it. Both conditions do have roots there, and neither
        # is a mode.
        raise ValueError(f"a spherical cavity carries no {kind}(0) mode, got n = {n}")
    if p < 1:
        raise ValueError(f"root index must be >= 1, got {p}")

    condition = _CAVITY_CONDITIONS[kind]
    # Both conditions leave the origin as x**n and neither turns back before the
    # polar order, so a scan from just above zero finds the roots in order and
    # misses none. They run more than pi apart, so a step well inside that
    # cannot put two of them in one interval and a bracket is enough to place
    # each exactly. The first root sits beyond n, so this span reaches the p-th
    # with room to spare, and finding fewer is a bug rather than an answer.
    span = n + (p + 2) * np.pi
    x = np.linspace(1e-6, span, int(span / (np.pi / 8)) + 2)
    y = condition(n, x)
    crossings = np.nonzero(np.signbit(y[1:]) != np.signbit(y[:-1]))[0]
    if len(crossings) < p:
        raise RuntimeError(f"found {len(crossings)} roots of {kind}({n}) looking for {p}")
    edge = crossings[p - 1]
    return float(brentq(lambda value: condition(n, value), x[edge], x[edge + 1]))


def spherical_cavity_frequency(
    radius: float, n: int = 1, p: int = 1, kind: str = "TM", eps_r: float = 1.0
) -> float:
    """Resonant frequency of a spherical cavity in Hz, for a radius in metres.

    The wall is a perfect conductor and the fill is uniform. Nothing else enters
    - not a feed, not a wall thickness, and for the dominant mode not any
    dimension but the one. That is what makes it a measurement of where a
    curved surface reached the solver.
    """
    _check_positive(radius=radius, eps_r=eps_r)
    root = spherical_cavity_root(n=n, p=p, kind=kind)
    return float(SPEED_OF_LIGHT * root / (2 * np.pi * radius * np.sqrt(eps_r)))
