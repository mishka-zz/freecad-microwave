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
from scipy.special import ellipkm1, jn_zeros, jnp_zeros, spherical_jn

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
    coefficient-free one, which is not the published Hammerstad-Bekkadal dW, and
    the difference is worth several times the agreement the gate reports. The
    gate argues for the zero-thickness form anyway: comparing a solver that
    models 35 um copper against the thickness-corrected reference shifts the
    target in the direction that makes a healthy solver look broken.
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
# them worth having: a microstrip mesh several times too coarse at the conductor
# edge sits comfortably inside Hammerstad's own accuracy. It could not hide
# here.
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
# Symmetric stripline
#
# Exact, and it is the one exact impedance for a structure a rectilinear grid
# holds exactly. A stripline is filled with one dielectric throughout, so unlike
# a microstrip its mode is genuinely TEM rather than hybrid: the cross-section is
# a potential problem, and conformal mapping solves that one in closed form. A
# microstrip has no exact impedance to compare against at all - the field is
# split between substrate and air, the mode is hybrid, and every published
# expression for it is a fit.
#
# Zero conductor thickness, which is also what the solver is handed: a conducting
# sheet is geometrically flat and its thickness feeds only the surface-impedance
# loss model. The finite-thickness forms in circulation are approximations, so
# taking one would put a fit back in the reference.
# ---------------------------------------------------------------------------


def stripline_impedance(width: float, separation: float, eps_r: float = 1.0) -> float:
    """Characteristic impedance of a symmetric stripline, in ohms.

    :param width: Width of the centre strip.
    :param separation: Distance between the ground planes, in the same units.
        Only the ratio matters. The strip is centred between them.

    The mapping takes the half-plane between the ground planes onto a rectangle
    whose aspect ratio is the ratio of complete elliptic integrals below, and the
    capacitance of a parallel-plate region is that ratio directly - so the
    geometry enters only through ``k``, the way a coaxial line's enters only
    through a ratio of radii.

    Both squared moduli are formed, each by the expression that is exact where
    it is small, and neither is ever obtained from the other by subtraction.
    ``tanh**2`` is accurate for a narrow strip and saturates to 1 for a wide
    one; the folded exponential is accurate for a wide strip and saturates to 1
    for a narrow one. Each saturation is the right answer in the place it
    happens, because ``ellipkm1`` of 1 is ``K(0)``, which is what the other end
    of the mapping degenerates to.

    Writing either as ``1 - `` the other is what fails, and it fails silently at
    whichever end is not being looked at: the subtraction returns 0 where the
    modulus is merely small, ``K`` of the result is then infinite or ``pi/2``
    rather than the value wanted, and the quotient is a confident zero at the
    wide end or a wrong finite number at the narrow one. Neither raises.

    The exponential rather than ``cosh`` for the same reason once more: ``cosh``
    overflows on a very wide strip and hands back an infinity that reads as a
    modulus of zero, where the ratio here underflows toward zero as it should.
    What remains at either extreme is a strip whose mapping has no representable
    modulus at all, and that is refused rather than answered.
    """
    _check_positive(width=width, separation=separation, eps_r=eps_r)
    if eps_r < 1.0:
        raise ValueError(f"eps_r must be >= 1, got {eps_r}")

    quarter = np.pi * width / (2.0 * separation)
    folded = np.exp(-2.0 * quarter)
    modulus = np.tanh(quarter) ** 2
    complement = 4.0 * folded / (1.0 + folded) ** 2
    if modulus == 0.0 or complement == 0.0:
        raise ValueError(
            f"a strip {width / separation:g} separations wide leaves one of the mapping's "
            f"moduli below what a float holds, so the impedance is not computable here"
        )
    return float(Z_FREE_SPACE / (4.0 * np.sqrt(eps_r)) * ellipkm1(modulus) / ellipkm1(complement))


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


# ---------------------------------------------------------------------------
# Circular cylindrical cavity
#
# Exact, and it separates what a sphere holds together. Maxwell's equations in
# a cylinder of perfect conductor separate into a Bessel function across the
# radius and a half-wave count along the axis, so a resonance is a root of the
# first and an integer in the second, and the two add in quadrature.
#
# What the axial term makes possible is a mode that does not use it. A TM mode
# with no axial variation has the same frequency at every height - it is the
# radius alone, through one Bessel root - so a cavity drawn at two heights has
# one resonance between them, and the flat ends contribute nothing to it. A
# rectilinear grid holds those ends exactly in any case, which leaves the curved
# wall as the only surface the discretisation can misplace.
#
# The cylinder is also *ruled*: it is straight along its axis, so a
# triangulation of it is exact in that direction and approximates only around
# the circumference. A sphere has curvature in two directions at once and no
# such split.
# ---------------------------------------------------------------------------

#: The mode families a cylinder carries, each as the Bessel zeros the radial
#: wavenumber is quantised by. ``TM`` has an axial electric field that must
#: vanish on the wall, which is ``J_n(p) = 0``; ``TE`` has an axial magnetic
#: field whose radial derivative must vanish there, which is ``J_n'(p) = 0``.
_CIRCULAR_ZEROS = {"TM": jn_zeros, "TE": jnp_zeros}


def circular_cavity_root(order: int = 0, root: int = 1, kind: str = "TM") -> float:
    """The ``root``-th zero fixing the radial wavenumber, as ``k_c a``.

    :param order: Azimuthal order - how many periods the field runs through
        around the axis. Zero is the axially symmetric case.
    :param root: Which zero, counting outward. It is the number of radial
        half-variations of the field.
    :param kind: ``TM`` or ``TE``.

    Dimensionless, so it is the whole of the cross-section's contribution: a
    cylinder's transverse wavenumber is this number divided by the radius,
    whatever the height.
    """
    if kind not in _CIRCULAR_ZEROS:
        raise ValueError(f"mode family must be one of {sorted(_CIRCULAR_ZEROS)}, got {kind!r}")
    if order < 0:
        raise ValueError(f"azimuthal order must be >= 0, got {order}")
    if root < 1:
        raise ValueError(f"root index must be >= 1, got {root}")
    return float(_CIRCULAR_ZEROS[kind](order, root)[root - 1])


def circular_cavity_frequency(
    radius: float,
    height: float,
    order: int = 0,
    root: int = 1,
    axial: int = 0,
    kind: str = "TM",
    eps_r: float = 1.0,
) -> float:
    """Resonant frequency of a right circular cavity in Hz, dimensions in metres.

    :param axial: How many half-waves stand along the axis.

    The wall is a perfect conductor and the fill is uniform. The transverse and
    axial wavenumbers add in quadrature, so ``axial = 0`` leaves the height out
    of the answer entirely - that is the mode a gate reads, and the height is
    still asked for because it is what says which mode is the lowest.

    A ``TE`` mode's axial magnetic field varies as ``sin(axial pi z / d)``, so
    there is no such mode without at least one half-wave along the axis; a
    ``TM`` mode's axial electric field varies as the cosine and so has one.
    """
    _check_positive(radius=radius, height=height, eps_r=eps_r)
    if axial < 0:
        raise ValueError(f"axial half-waves must be >= 0, got {axial}")
    if kind == "TE" and axial == 0:
        raise ValueError("a TE mode with no axial variation has no field at all, got axial = 0")
    transverse = circular_cavity_root(order=order, root=root, kind=kind) / radius
    return float(
        SPEED_OF_LIGHT / (2 * np.pi * np.sqrt(eps_r)) * np.hypot(transverse, axial * np.pi / height)
    )
