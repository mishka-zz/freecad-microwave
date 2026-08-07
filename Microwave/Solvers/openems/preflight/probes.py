# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checks about where the measurement planes sit.

openEMS extracts an impedance from three voltage and two current probes around
the measurement plane, and the differences telescope, so the extraction is exact
on a matched line at any grading. Both checks here are about what breaks that
match: an uneven pair of cells matters only against a reflection at the plane,
and a plane too close to the feed is still in the feed's evanescent field.
"""

from __future__ import annotations

import numpy as np

from ..model import _USES_PROBE_TRIPLET, SPEED_OF_LIGHT, Problem
from .finding import WARN, Finding

#: Fractional difference between the two cells flanking a measurement plane
#: above which the probe triple's error stops being negligible.
#:
#: openEMS places the three voltage and two current probes so that the
#: differences telescope, which makes the extraction *exact* on a matched line
#: at any grading. Error appears only against a reflection at the plane, and
#: then, measured against the closed form, |dZ|/Z0 ~= |Gamma| * pi * asymmetry
#: / N cells per guide wavelength. At the mesh density this project works at,
#: that puts the probe's own error below the margin the microstrip gate reports
#: against Hammerstad - past here it stops being the accurate part of the
#: measurement. See TestTheProbeAsymmetryLimit, which derives it and holds it.
_PROBE_ASYMMETRY_LIMIT = 0.02


#: Fraction of the longest wavelength in the band that the measurement plane
#: should stand clear of the feed by. Bracketed by two measurements on the
#: acceptance line rather than derived - see :func:`_check_probes_clear_of_the_feed`.
_FEED_CLEARANCE = 0.1


def _check_probes_clear_of_the_feed(problem: Problem) -> list[Finding]:
    """The measurement plane has to stand back from the source.

    A feed launches evanescent higher-order modes as well as the one being
    measured. They decay within a fraction of a wavelength, but a probe placed
    inside that region reads them as part of the field and the extracted
    impedance comes out high. Nothing downstream can see this: the curve is
    smooth, the run is converged, and the number is simply wrong.

    The threshold is bracketed by two solves on the microstrip acceptance line,
    changing nothing but where the feed and the probes sit; the pair is tabulated
    once, at :data:`portbox.CLEARANCE`. It sits between them and is not otherwise
    derived, which is why this warns rather than refuses: the boundary is
    bracketed, not known.
    The wavelength is taken at the *bottom* of the band, where it is longest and
    the constraint is tightest, and in the slowest material in the model.

    **The slowest material here and free space in ``portbox.CLEARANCE``, and the
    difference is what each one knows.** The quantity that governs the decay is
    the line's own eps_eff. This runs against a translated problem, so its
    materials are the ones the study actually binds and sqrt(eps_r) is a usable
    proxy for sqrt(eps_eff); ``portbox.clearance`` runs when a port is created,
    before there is a line, a substrate or a binding, so it has no material to
    be a proxy for and takes the conservative bound instead. A default port
    therefore clears this threshold by sqrt(eps_r) - never less than once, and
    exactly once for a model with no dielectric in it, which is the one case
    where the two rules are the same rule.
    """
    epsilon = max([material.epsilon for material in problem.materials] or [1.0])
    wavelength = SPEED_OF_LIGHT / problem.frequency.start / np.sqrt(epsilon) / problem.length_unit
    limit = _FEED_CLEARANCE * wavelength

    findings = []
    for port in problem.ports:
        if port.kind not in _USES_PROBE_TRIPLET:
            continue
        dim = port.propagation_axis
        feed = port.start[dim] + port.direction * port.feed_shift
        separation = abs(port.measurement_position() - feed)
        if separation < limit:
            findings.append(
                Finding(
                    WARN,
                    port.name,
                    f"its measurement plane is {separation:.4g} from its feed, "
                    f"which is {separation / wavelength:.3f} of a wavelength at "
                    f"{problem.frequency.start / 1e9:.3g} GHz. The feed's "
                    f"near field has not decayed there, and the impedance will "
                    f"read high - measurably so: at 0.04 wavelengths this line "
                    f"came out 1.7% above its closed form, at 0.21 it came out "
                    f"0.07% above. Increase the port's length, or move the "
                    f"measurement shift away from the feed shift",
                )
            )
    return findings


def _check_probes(problem: Problem) -> list[Finding]:
    """Predict where openEMS will put each port's probes, and judge the spacing.

    ``MSLPort`` snaps three voltage probes to the grid lines nearest the
    measurement plane and differentiates across them, and puts the two current
    probes at the midpoints between them. That placement is not incidental: with
    A and B the half-cell phase factors either side, the voltage difference goes
    as A^2 - B^2 and ``Ht * dHt`` as (A + B)(A - B), so they cancel and the
    extraction is exact on a matched line however uneven the grid is. What
    survives is the part of the field that does not share the travelling wave's
    form - a reflection at the plane - and there the unevenness enters at
    first order. See :data:`_PROBE_ASYMMETRY_LIMIT` for the coefficient.

    Only that scheme is checked. A waveguide port integrates mode functions over
    a single plane and has no difference to take, so applying this to one
    produces a warning about an inaccuracy that cannot occur - and a warning
    that is wrong is worse than no warning, because it teaches people to skip
    reading them.

    The tempting fix - moving grid lines after meshing to force the two
    spacings equal - is worse than the problem: it voids every guarantee the
    mesher just validated. So
    this measures instead of mutating, and says what it found. Feeding the
    measurement plane to the mesher as a constraint is the real fix, and it
    belongs upstream of here.
    """
    findings = []
    for port in problem.ports:
        if port.kind not in _USES_PROBE_TRIPLET:
            continue
        dim = port.propagation_axis
        lines = problem.grid[dim]
        index = int(np.argmin(np.abs(lines - port.measurement_position())))
        index = min(max(index, 1), len(lines) - 2)

        below = float(lines[index] - lines[index - 1])
        above = float(lines[index + 1] - lines[index])
        asymmetry = abs(above - below) / max(above, below)

        if asymmetry > _PROBE_ASYMMETRY_LIMIT:
            findings.append(
                Finding(
                    WARN,
                    port.name,
                    f"the grid is uneven at its measurement plane: the cells "
                    f"either side are {below:.4g} and {above:.4g} "
                    f"({asymmetry * 100:.1f}% apart), which puts "
                    f"{asymmetry * np.pi / 20 * 100:.3f}% of Z0 into the "
                    f"extracted impedance for every unit of reflection there",
                )
            )
    return findings
