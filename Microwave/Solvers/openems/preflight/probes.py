# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checks about where the measurement planes sit.

openEMS extracts an impedance from three voltage and two current probes around
the measurement plane. The differences telescope, so the extraction is exact on
a matched line at any grading. Both checks here are about what breaks that
match. An uneven pair of cells matters only against a reflection at the plane. A
plane too close to the feed is still in the feed's evanescent field.

Every kind that differences a triplet is subject to the first check. The second
is asked of every such kind too, but against a different length. What separates
the two lengths is what has been measured about each structure rather than what
kind of source is on it.

A round port's clearance is its own circumference, because a coaxial line's
first mode that is not TEM is cut off below about that and decays over a length
of that order. A shielded line meshed as a strip between planes behaves the same
way, over the shield's width divided by pi
(``tests/test_acceptance_stripline.py``, ``GATE stripline clearance``). So does
an open microstrip, which is what the uniform-gap check below mostly runs on: a
ladder of measurement planes read off one run gives the distance there as the
cross-section's too (``tests/test_acceptance_microstrip.py``, ``GATE microstrip
clearance``), even though such a line radiates and carries a surface wave its
ground plane never cuts off.

The check below is stated as a share of the wavelength nonetheless. The open
ladder gives the quantity that governs and no rule to compute it with, and the
placement default is filled in with no cross-section in hand at all.
"""

from __future__ import annotations

import numpy as np

from ....portbox import CLEARANCE
from ..model import (
    _EXCITES_A_UNIFORM_GAP,
    _IS_ROUND,
    _USES_PROBE_TRIPLET,
    SPEED_OF_LIGHT,
    Problem,
)
from .finding import WARN, Finding

#: Fractional difference between the two cells flanking a measurement plane
#: above which the probe triple's error stops being negligible.
#:
#: openEMS places the three voltage and two current probes so that the
#: differences telescope, which makes the extraction exact on a matched line at
#: any grading. Error appears only against a reflection at the plane, and then,
#: measured against the closed form, |dZ|/Z0 ~= |Gamma| * pi * asymmetry / N
#: cells per guide wavelength. At the mesh density this project works at, that
#: puts the probe's own error below the margin the microstrip gate reports
#: against Hammerstad. Past this limit the probe stops being the accurate part
#: of the measurement. See TestTheProbeAsymmetryLimit, which derives the limit
#: and holds it.
_PROBE_ASYMMETRY_LIMIT = 0.02

#: Cells per guide wavelength the ``N`` above is read at, where the message
#: below states what the asymmetry costs. It is the density this project meshes
#: at rather than a reading off the grid in hand: the guide wavelength at the
#: measurement plane needs the mode's own index, which pre-flight does not
#: have. The message names the assumption, so a reader can scale it.
#: ``TestTheProbeAsymmetryLimit`` scores the law at this density and either side
#: of it.
_CELLS_PER_WAVELENGTH_ASSUMED = 20.0


#: Mean circumferences a round port's probes should stand clear of its source
#: by. A coaxial line's first mode that is not TEM cannot propagate until its
#: mean circumference is about a wavelength, so below that it decays over a
#: length of that order rather than over a fraction of the band's wavelength.
#: For a line a few millimetres across, over a band reaching down towards DC,
#: those two lengths are completely different, and a threshold set by the
#: wavelength would fire on every model ever drawn.
_BORE_CLEARANCE = 1.0


def _check_probes_clear_of_the_feed(problem: Problem) -> list[Finding]:
    """The measurement plane has to stand back from the source.

    A feed launches evanescent higher-order modes as well as the one being
    measured, and a probe placed inside that region reads them as part of the
    field, so the extracted impedance comes out high. Nothing downstream shows
    this: the curve is smooth, the run is converged, and the number is simply
    wrong.

    The cross-section rather than the band sets how far those modes survive, and
    that has been measured on both kinds of structure. On a shielded line the
    excess falls off over the length the shield's own cutoff gives, with no
    frequency entering it. On an open line, which is what this threshold mostly
    judges, a ladder of measurement planes read off one run gives the same answer
    from the other side: moving the band barely shifts the contamination at a
    given distance, while changing the cross-section shifts it a great deal. The
    gates are ``tests/test_acceptance_stripline.py`` and
    ``tests/test_acceptance_microstrip.py``, on their ``clearance`` lines.

    That contradicts the shape of this threshold and offers no length to put in
    its place. On the open ladder the contamination and the distance the reading
    settles over do not move together, so no length rule can be read off it,
    where in the shielded case one dimension carries the whole length. The
    threshold therefore stays a share of a wavelength, which is where its figure
    came from rather than what governs it.

    The threshold is :data:`Microwave.portbox.CLEARANCE`, the fraction that
    places the probes as well, so one bracket is read in both places. Two solves
    on the microstrip acceptance line bracket it, changing nothing but where the
    feed and the probes sit, and the message below quotes that pair. The value
    sits between them and is not otherwise derived, so this warns rather than
    refusing: the boundary is bracketed rather than known.

    Sharing the fraction with the placement stops the two drifting apart, at the
    cost that this check cannot judge the placement default itself: lower that
    default and the threshold follows it down. What it judges is a port as it
    now stands, which is what a translated problem can show and what a user can
    change. The ladder judges the default, by reading where the settling happens
    rather than comparing one plane against a formula.

    The wavelength is taken in free space, and at the bottom of the band, where
    it is longest and the constraint tightest. ``portbox.clearance`` takes the
    free-space bound as well, so a default port clears this threshold by
    whatever its rounding up leaves rather than by a margin that depends on
    which materials the model happens to bind. Dividing it by a material's
    sqrt(eps_r) corrects for nothing, since the distance is the cross-section's
    and has no frequency in it, and it errs short, which is the direction that
    returns a plausible impedance and no complaint.
    """
    wavelength = SPEED_OF_LIGHT / problem.frequency.start / problem.length_unit
    limit = CLEARANCE * wavelength

    findings = []
    for port in problem.ports:
        if port.kind not in _EXCITES_A_UNIFORM_GAP:
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
                    f"read high. Increase the port's length, or move the "
                    f"measurement shift away from the feed shift",
                )
            )
    return findings + _check_probes_clear_of_the_bore(problem)


def _check_probes_clear_of_the_bore(problem: Problem) -> list[Finding]:
    """The same requirement on a round port, against the length that governs it.

    A coaxial port's source carries the mode's own radial profile, so what it
    launches beside the mode is the line's higher-order modes, and one cell of
    source along the line is enough to excite a little of them. The first of
    those is TE11, which cannot propagate until the mean circumference is about
    a wavelength. Below its cutoff it decays over a length of that order.

    The clearance a round port needs is therefore set by its own cross-section
    rather than by the band. That holds for every enclosed line and is not a
    property of round ones. It is separated here because a bore is the one
    cross-section this adapter can name, and no such length has been measured
    for the open structures above. On a line a few millimetres across, over a
    band reaching down towards DC, the wavelength rule asks for metres.
    """
    findings = []
    for port in problem.ports:
        if port.kind not in _IS_ROUND:
            continue
        circumference = np.pi * (port.inner_radius + port.outer_radius)
        dim = port.propagation_axis
        feed = port.start[dim] + port.direction * port.feed_shift
        separation = abs(port.measurement_position() - feed)
        if separation >= _BORE_CLEARANCE * circumference:
            continue
        findings.append(
            Finding(
                WARN,
                port.name,
                f"its measurement plane is {separation:.4g} from its feed, and "
                f"the mean circumference of its bore is {circumference:.4g}. A "
                f"mode that is not TEM is cut off below about that length and "
                f"decays over it, so some of what the probes difference there "
                f"is not the line's own wave and the impedance will read high. "
                f"Increase the port's length, or move the measurement shift "
                f"away from the feed shift",
            )
        )
    return findings


def _check_probes(problem: Problem) -> list[Finding]:
    """Predict where openEMS will put each port's probes, and judge the spacing.

    ``MSLPort`` snaps three voltage probes to the grid lines nearest the
    measurement plane and differentiates across them, and puts the two current
    probes at the midpoints between them. That placement is chosen: with A and B
    the half-cell phase factors either side, the voltage difference goes as
    A^2 - B^2 and ``Ht * dHt`` as (A + B)(A - B), so they cancel and the
    extraction is exact on a matched line however uneven the grid is. What
    survives is the part of the field that does not share the travelling wave's
    form, which is a reflection at the plane, and there the unevenness enters at
    first order. See :data:`_PROBE_ASYMMETRY_LIMIT` for the coefficient.

    Only that scheme is checked. A waveguide port integrates mode functions over
    a single plane and has no difference to take, so applying this check to one
    produces a warning about an inaccuracy that cannot occur. A warning that is
    wrong is worse than no warning, because a reader who meets one learns to
    skip the rest.

    Moving grid lines after meshing to force the two spacings equal is tempting
    and is worse than the problem: it voids every guarantee the mesher has just
    validated. This check therefore measures rather than mutating, and reports
    what it found. Feeding the measurement plane to the mesher as a constraint
    is the real fix, and it belongs upstream of here.
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
                    f"{asymmetry * np.pi / _CELLS_PER_WAVELENGTH_ASSUMED * 100:.3f}% "
                    f"of Z0 into the extracted impedance for every unit of "
                    f"reflection there, at "
                    f"{_CELLS_PER_WAVELENGTH_ASSUMED:g} cells per guide wavelength",
                )
            )
    return findings
