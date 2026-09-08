# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Where a port's box is, and where its two planes sit inside it.

The openEMS adapter builds its envelope from this module, and the document
object draws its own shape from it. Both read one answer, so the envelope and
the drawing cannot disagree. The box is drawn so that the user sees what the
solver gets.

This module is at the package root rather than under ``Objects``, for a
mechanical reason. ``Microwave.Objects.__init__`` imports FreeCAD, so an adapter
importing anything from that package would break the rule that adapters run
without it, and a test enforces that rule in a clean interpreter.
``Microwave.__init__`` imports nothing but ``sys.path``.

Everything here is arithmetic on boxes. A box is ``(lower, upper)``, two triples
of millimetres, exactly as in ``Objects.port_setup``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from . import units

DIMENSIONS = 3
AXIS_NAMES = ("X", "Y", "Z")

#: A corner: one coordinate per axis, in the order the axes are numbered.
Corner = tuple[float, float, float]

#: A box: its two corners. They are sequences rather than :data:`Corner`,
#: because callers build them a list at a time and a bounding box arrives as
#: whatever the CAD kernel handed over.
Box = tuple[Sequence[float], Sequence[float]]

#: Below this a difference in millimetres is arithmetic rather than drawing.
#:
#: One nanometre. It is a margin rather than a reading of the kernel. What was
#: measured is the smallest box OCC will build at all, which
#: ``Objects.port_shape._FLAT_BOX`` records and which is itself strictly above
#: ``Precision::Confusion()``. This constant stands a decade clear of that
#: measurement. On the model side it is four orders under the thinnest conductor
#: drawn in practice: 35 um of plated copper is far above it.
#:
#: There is one length under one name, because the questions asked at this size
#: share an answer: whether an extent is flat, whether a point lies inside,
#: whether two surfaces touch. Each question has a constant named for it, below
#: or where the question is asked, and none of those carries a figure of its own.
#: They therefore move together or not at all. That is the trade: two copies a
#: decade apart would agree about every model that does not sit between them,
#: and one length means a change made for the drawing side lands on the meshing
#: side too. What a demand at this size costs there is at
#: :data:`~Microwave.Solvers.openems.lfs.TOUCHING`.
KERNEL_TOLERANCE = 1e-6

#: Below this an extent is a plane rather than a solid, in mm. This is the
#: tolerance above, asked of a thickness. A selected face is exactly flat, but a
#: face read back through a bounding box is only flat to within the arithmetic
#: that produced it, so the test cannot be against zero.
FLATNESS = KERNEL_TOLERANCE

#: In millimetres per second, because every length here is in millimetres.
SPEED_OF_LIGHT = units.SPEED_OF_LIGHT * units.MM_PER_M

#: How far the probes must sit from the source, as a fraction of the free-space
#: wavelength at the bottom of the band, where the wavelength is longest.
#:
#: This number decides whether an extracted impedance is right. The source is a
#: sheet of current across the strip, and it radiates a near field that is not
#: the transmission-line mode. The probes have to sit far enough downstream for
#: that field to have decayed.
#:
#: A fraction of a wavelength is not the law the decay follows. A set of
#: evanescent modes has to die away, and a mode below its cutoff dies over a
#: length its own cross-section sets - on a shielded line, the shield's width
#: over pi. A plane standing more wavelengths clear at the top of a band
#: therefore reads further out rather than nearer. An open line behaves the same
#: way, although it radiates and carries a surface wave nothing cuts off. The
#: gates are ``tests/test_acceptance_stripline.py`` and
#: ``tests/test_acceptance_microstrip.py``, on their ``clearance`` lines.
#:
#: The figure stays a fraction because no cross-section is available at the
#: moment it is needed. A port is created before there is a strip, a substrate
#: or a binding to read one off, and no permittivity is available either. Free
#: space is the conservative bound of the two that can be had, since eps_eff is
#: never below one. The longest wavelength errs long. Too much clearance costs
#: feed line the engineer can see; too little costs an extracted impedance with
#: no symptom.
#:
#: The figure is good to about one significant figure.
#: ``tests/test_acceptance_stripline.py`` solves a ladder of clearances and
#: measures what each buys.
CLEARANCE = 0.1


class BoxError(ValueError):
    """The geometry does not describe a port. The message names the reason."""


@dataclass(frozen=True)
class PortBox:
    """The volume openEMS builds the port in, and the planes inside it.

    ``start`` and ``stop`` are corners in the order the solver wants them, and
    are not sorted. For a microstrip, ``start`` is on the trace and ``stop`` on
    the ground: ``MSLPort`` integrates the voltage from one to the other, and the
    direction of that integration is the sign of the excitation.

    ``feed`` and ``measurement`` are distances from ``start`` along the
    propagation axis, in millimetres. They are not fractions of the box length.
    A fraction would make the two distances an engineer has to get right depend
    on a third they mostly do not care about: 0.5 puts the probes 50 mm along a
    100 mm line and 10 mm along a 20 mm one, and only one of those is far enough
    from the source.
    """

    start: tuple[float, float, float]
    stop: tuple[float, float, float]
    propagation_axis: int
    feed: float = 0.0
    measurement: float = 0.0
    #: Distance from ``start`` along the propagation axis to where openEMS reads
    #: this port's numbers, in millimetres. The port kind decides which plane
    #: that is, and nothing else does, so each constructor below sets it. It is
    #: separate from :attr:`measurement`. A microstrip, a coaxial line and a
    #: waveguide set both to the same distance; a lumped port sets only this
    #: one, having no measurement plane.
    #:
    #: It is named for the probes rather than for the reference plane the user
    #: knows it as, because ``reference`` already names an entity in
    #: :func:`lumped` and an impedance everywhere in the result layer.
    #:
    #: This is where the plane is asked for. A microstrip's probes land on the
    #: nearest grid line instead, which is a sub-cell move and so does not reach
    #: a distance axis, but it does mean this is not the coordinate openEMS
    #: used.
    probe: float = 0.0

    @property
    def length(self) -> float:
        axis = self.propagation_axis
        return abs(self.stop[axis] - self.start[axis])

    @property
    def direction(self) -> int:
        axis = self.propagation_axis
        return 1 if self.stop[axis] >= self.start[axis] else -1

    def plane_at(self, distance: float) -> float:
        """The propagation-axis coordinate ``distance`` into the box."""
        return self.start[self.propagation_axis] + self.direction * distance

    def probe_point(self) -> tuple[float, float, float]:
        """Where in the model this port's numbers are read, in millimetres.

        On the propagation axis the point is :attr:`probe`. Across the other two
        axes it is the middle of the box, where every kind's probes sit. The
        distance between two ports' reference points is the length a velocity
        divides into, and that distance is a straight line. On a line with a
        bend in it, the velocity that comes back is an average over the real
        path and the drawn one together. That is the number a velocity factor
        calibrated against a ruler would give.
        """
        axis = self.propagation_axis
        point = [(a + b) / 2.0 for a, b in zip(self.start, self.stop)]
        point[axis] = self.plane_at(self.probe)
        return (point[0], point[1], point[2])

    def corners(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The same volume with its corners sorted, for drawing."""
        lower = tuple(min(a, b) for a, b in zip(self.start, self.stop))
        upper = tuple(max(a, b) for a, b in zip(self.start, self.stop))
        return lower, upper


# The box vocabulary, defined once here and imported by everything that speaks
# it. The import runs in this direction only, because ``Objects/__init__.py``
# imports FreeCAD and this module must stay importable without it.
def corner(values: Iterable[float]) -> Corner:
    """Three coordinates as the fixed-width triple a corner is.

    The values are unpacked rather than cast. A comprehension over the axes
    cannot state how wide it is, and unpacking refuses anything that is not
    three values, where a cast would only assert it.
    """
    x, y, z = values
    return (x, y, z)


def middle(box: Box, axis: int) -> float:
    return (box[0][axis] + box[1][axis]) / 2.0


def extent(box: Box, axis: int) -> float:
    return box[1][axis] - box[0][axis]


def _nearest(box: Box, axis: int, to: float) -> float:
    """Whichever of the box's two faces on ``axis`` is closer to ``to``."""
    low, high = box[0][axis], box[1][axis]
    return low if abs(low - to) <= abs(high - to) else high


def third_axis(first: int, second: int) -> int:
    return DIMENSIONS - first - second


def clearance(frequency: float) -> float:
    """How far the probes must sit from the source, in millimetres.

    ``frequency`` is the bottom of the band, where the wavelength is longest and
    the requirement is tightest. See :data:`CLEARANCE` for why no permittivity
    comes into it.

    The result is rounded up to the next whole millimetre. :data:`CLEARANCE` is
    known to about one significant figure, and a default carrying three decimals
    claims a precision that does not exist. Rounding up rather than to nearest
    keeps the requirement from being shaved below the distance actually
    measured. At millimetre wavelengths this is a coarse step, and the result is
    a starting value an engineer edits.

    Returns ``0.0`` when there is no band to derive it from. A document with no
    study yet is an ordinary state rather than an error, and the caller writes
    the number into the port, where the engineer can see it and change it.
    """
    if frequency <= 0:
        return 0.0
    return float(math.ceil(CLEARANCE * SPEED_OF_LIGHT / frequency))


def length_of(stated: float, fallback: float, subject: str) -> float:
    """The port's extent: what was asked for, or what the geometry offers."""
    if stated < 0:
        raise BoxError(
            f"{subject}: Length is {stated}. A negative length would put the "
            "port's far face behind its near one, which reverses the port "
            "rather than shortening it"
        )
    if stated > 0:
        return stated
    if fallback <= FLATNESS:
        raise BoxError(
            f"{subject}: Length is unset and the geometry it sits on has no "
            "extent to fall back on. Set Length explicitly"
        )
    return fallback


def microstrip(
    trace_face: Box,
    ground: Box,
    *,
    propagation_axis: int,
    direction: int,
    excitation_axis: int,
    excitation_direction: int,
    feed_offset: float,
    measurement_distance: float,
    stated_length: float,
    subject: str = "microstrip port",
) -> PortBox:
    """A strip over a ground plane, fed across the substrate.

    The box states the port's requirement rather than reading the geometry. It
    runs from the picked face inward as far as the measurement plane has to be,
    whether or not the trace under it is that long. A box that ends where the
    copper ends always looks as though it fits, and an engineer needs to be
    shown a short feed line.

    The distances from the picked face, all in millimetres and all absolute:

    ``feed_offset``
        where the source sits. Zero is the ordinary case: the trace ends at the
        port and the source sits on that end face. It is not zero when the line
        deliberately runs out through the absorber, where the field is
        attenuated on purpose and a source in it excites nothing.
    ``measurement_distance``
        how far downstream of the source the probes sit. It is measured from the
        source and not from the picked face. The source is a sheet of current
        and the probes have to be clear of its near field, which does not depend
        on where the face is. The two distances are equal only while the feed
        sits at zero. See :data:`CLEARANCE`.
    ``stated_length``
        how far the box reaches. Zero ends the box at the measurement plane.
        Everything past that plane is strip openEMS lays and no probe ever
        reads.

    Both planes are the conductor surfaces facing each other, and never the
    mid-planes of the boxes they were read from. A ground plane drawn with real
    thickness has its middle inside the metal, and a port built to that middle
    would drive across a gap half a conductor too long.
    """
    width_axis = third_axis(propagation_axis, excitation_axis)

    ground_plane = _nearest(ground, excitation_axis, middle(trace_face, excitation_axis))
    trace_plane = _nearest(trace_face, excitation_axis, ground_plane)
    if abs(trace_plane - ground_plane) <= FLATNESS:
        raise BoxError(
            f"{subject}: the trace and the ground reference are both at "
            f"{AXIS_NAMES[excitation_axis]}={trace_plane:.4g}, so there is no "
            "gap to drive across"
        )

    # The direction the field points follows from the model rather than from a
    # setting, so it is measured here and compared against what the user stated.
    # The sign is the sign of the excitation, and inverting it produces a
    # perfectly clean-looking solve with the phase reversed. The refusal is here
    # rather than in the adapter, so a port that would solve backwards does not
    # draw either.
    measured = 1 if ground_plane > trace_plane else -1
    if excitation_direction != measured:
        raise BoxError(
            f"{subject}: the ground reference is at "
            f"{AXIS_NAMES[excitation_axis]}={ground_plane:.4g} and the trace at "
            f"{trace_plane:.4g}, so the field points the other way. Use "
            f"{'' if measured > 0 else '-'}{AXIS_NAMES[excitation_axis]}"
        )

    length, measurement = _line_length(feed_offset, measurement_distance, stated_length, subject)

    start, stop = [0.0] * DIMENSIONS, [0.0] * DIMENSIONS
    start[propagation_axis] = middle(trace_face, propagation_axis)
    stop[propagation_axis] = start[propagation_axis] + direction * length
    start[excitation_axis], stop[excitation_axis] = trace_plane, ground_plane
    start[width_axis] = trace_face[0][width_axis]
    stop[width_axis] = trace_face[1][width_axis]

    return PortBox(
        corner(start),
        corner(stop),
        propagation_axis,
        feed=feed_offset,
        measurement=measurement,
        probe=measurement,
    )


def _line_length(
    feed_offset: float,
    measurement_distance: float,
    stated_length: float,
    subject: str,
) -> tuple[float, float]:
    """``(box length, measurement plane)``, both from the picked face.

    Every kind that runs a source down a line and reads it back from a probe
    triplet downstream shares this. Every way those distances can disagree is
    refused here, so a port that would solve wrongly does not draw either.
    """
    if feed_offset < 0:
        raise BoxError(
            f"{subject}: FeedOffset is {feed_offset:g}. It is measured inward "
            "from the face you picked, so a negative value puts the source "
            "outside the structure"
        )
    if measurement_distance <= 0:
        raise BoxError(
            f"{subject}: MeasurementDistance is {measurement_distance:g}, so "
            "the probes would sit on the source. This port measures its own "
            "impedance from three probes downstream of the source, clear of "
            f"its near field - set MeasurementDistance to about {CLEARANCE:g} "
            "wavelengths"
        )
    measurement = feed_offset + measurement_distance
    if stated_length < 0:
        raise BoxError(
            f"{subject}: Length is {stated_length:g}. A negative length would "
            "put the port's far face behind its near one, which reverses the "
            "port rather than shortening it"
        )
    if stated_length == 0:
        return measurement, measurement
    if stated_length < measurement:
        raise BoxError(
            f"{subject}: Length is {stated_length:g} mm, but the measurement "
            f"plane sits {measurement:g} mm in (FeedOffset {feed_offset:g} plus "
            f"MeasurementDistance {measurement_distance:g}), so the box would "
            "stop short of it. Leave Length at 0 to end the box there"
        )
    return stated_length, measurement


def coaxial(
    annulus: Box,
    *,
    propagation_axis: int,
    direction: int,
    feed_offset: float,
    measurement_distance: float,
    stated_length: float,
    subject: str = "coaxial port",
) -> PortBox:
    """A TEM line between two concentric conductors, driven across the gap.

    The box spans the bore rather than the annulus. Across the two transverse
    axes it spans the picked ring's own bounding box, which for a ring centred
    on the line is the square circumscribing the shield's inner surface. The box
    therefore holds the outer radius and the centre, and the inner radius is the
    one number that has to travel beside it.

    The distances along the line are the microstrip's and mean the same things.
    ``feed_offset`` places the source, ``measurement_distance`` places the probes
    downstream of it, and ``stated_length`` states how far the box reaches. A
    coaxial line reads its impedance the same way a microstrip does, by
    differencing three probes, so the source needs the same near-field
    clearance.

    A coaxial port has no excitation axis, unlike a microstrip. The field is
    radial, so its direction follows from the two radii and not from an axis
    anybody could name.
    """
    length, measurement = _line_length(feed_offset, measurement_distance, stated_length, subject)

    start, stop = list(annulus[0]), list(annulus[1])
    start[propagation_axis] = middle(annulus, propagation_axis)
    stop[propagation_axis] = start[propagation_axis] + direction * length

    return PortBox(
        corner(start),
        corner(stop),
        propagation_axis,
        feed=feed_offset,
        measurement=measurement,
        probe=measurement,
    )


def lumped(
    source: Box,
    reference: Box,
    *,
    excitation_axis: int,
    outline: Box | None,
    subject: str = "lumped port",
) -> PortBox:
    """A resistor across a gap, driven along one axis.

    Across the excitation axis the port spans where the two entities actually
    face each other, which is their overlap. It is not their union, and not
    either entity alone. A union is wrong the moment the reference is a ground
    plane covering the whole footprint: the port becomes a sheet resistor
    spanning the model, and it solves. Taking the source alone leaves an
    asymmetry, so picking the ground as the source brings the same fault
    straight back. An overlap has no preferred end.

    Fed from a trace end face, the overlap has zero extent across one axis and
    the box is a plane. That is legitimate and needs no grid line of its own.
    openEMS snaps a lumped element's box to the mesh (``operator.cpp``:1637), so
    the plane lands on the nearest line whatever sub-cell offset it was drawn
    at. The lumped gate holds that invariance.

    ``outline`` is the body a curve source was picked off - the trace behind its
    own end edge, which is how a planar layout offers a cross-section - and it is
    ``None`` where the source is an area instead. openEMS has no rotated port:
    its lumped element takes a plain box and reads the corners raw, with no
    transform (``operator.cpp``:1601 and :1636), and the current probe is a
    surface with an axis normal. So the end of a trace running off a grid axis
    arrives as a bounding box whose diagonal is the trace end, with half its area
    past the end of the conductor, driving air above the reference. The box is
    the wrong shape rather than a coarse one, and no cell size reaches it.

    The port is therefore flattened onto the end of that depth the body lies
    behind, and not onto its middle. The middle is where the plane crosses the
    diagonal, so it leaves half the element off the metal exactly as the box did.
    It is worse once the conductor is rasterised, since it stands the whole port
    on the staircase, where rounding decides which side a cell falls. Taken to
    the inward end, the plane clears the diagonal entirely and the port lies
    within the conductor's own outline.

    Flattening does not recover everything. The port sits that depth inside the
    drawn end. The conductor also reaches the grid rasterised, so a port on its
    outline is on the edge of what conducts however the plane was chosen.
    Sampling a curve on a lattice puts that edge on the staircase, and a
    bounding box carries nothing that says where.

    An area source keeps both extents, since a pad driven across its own face
    genuinely spans them. The caller decides which it has, because a bounding
    box does not separate them: a tilted edge's box and a small pad's box are
    the same six numbers.

    Snapping does not survive a gap thinner than a cell along the excitation
    axis: both ends land on one line and openEMS drops the element.
    ``preflight.ports._check_the_element_survives_snapping`` refuses that.
    """
    transverse = [dim for dim in range(DIMENSIONS) if dim != excitation_axis]
    start, stop = [0.0] * DIMENSIONS, [0.0] * DIMENSIONS
    for dim in transverse:
        low = max(source[0][dim], reference[0][dim])
        high = min(source[1][dim], reference[1][dim])
        if high < low - FLATNESS:
            raise BoxError(
                f"{subject}: SourceEntity and ReferenceEntity do not overlap "
                f"along {AXIS_NAMES[dim]} - one spans {source[0][dim]:.9g} to "
                f"{source[1][dim]:.9g} and the other {reference[0][dim]:.9g} to "
                f"{reference[1][dim]:.9g}. A lumped port drives between two "
                "conductors facing each other"
            )
        # The comparison is against the tolerance and not against zero, and the
        # result is then clamped, because these two numbers come from different
        # shapes. A sketch-derived end and a box-derived ground meeting at one
        # coordinate are not bit-identical, so which of them compares larger
        # is arbitrary. Untouched, that is a port that refuses at one end of a
        # line and builds at the other, over a distance no drawing can express.
        # A fixture is arithmetically exact and a kernel's output is not.
        start[dim], stop[dim] = low, max(high, low)

    start[excitation_axis] = middle(source, excitation_axis)
    stop[excitation_axis] = middle(reference, excitation_axis)
    if abs(start[excitation_axis] - stop[excitation_axis]) <= FLATNESS:
        raise BoxError(
            f"{subject}: SourceEntity and ReferenceEntity are both at "
            f"{AXIS_NAMES[excitation_axis]}={start[excitation_axis]:.4g}. A "
            "lumped port drives a voltage across a gap; along the excitation "
            "axis there is none"
        )

    # A lumped port is a circuit element rather than a transmission line, so it
    # has no propagation axis of its own. The adapter uses the axis only to
    # check the port is clear of the absorber, so the wider transverse extent is
    # taken. That is also the projection a flattened outline keeps, the two
    # agreeing because the wider extent is the conductor and the narrower is the
    # tilt.
    propagation_axis = max(transverse, key=lambda d: stop[d] - start[d])
    if stop[propagation_axis] - start[propagation_axis] <= FLATNESS:
        raise BoxError(
            f"{subject}: the port is a line, not a box - where the two "
            f"entities overlap has extent only along "
            f"{AXIS_NAMES[excitation_axis]}. Select faces, or edges that span "
            "the conductor's width"
        )
    if outline is not None:
        across = third_axis(excitation_axis, propagation_axis)
        start[across] = stop[across] = _nearest((start, stop), across, middle(outline, across))
    # Half the box, because openEMS reads a lumped port at its centre. Which
    # transverse axis was called the propagation one moves nothing here, the
    # centre being the centre either way. It does decide which extent a
    # flattened outline keeps, and the extent to keep is the wider one.
    return PortBox(
        corner(start),
        corner(stop),
        propagation_axis,
        probe=abs(stop[propagation_axis] - start[propagation_axis]) / 2.0,
    )


def rect_waveguide(
    face: Box,
    *,
    propagation_axis: int,
    direction: int,
    stated_length: float,
    fallback: float,
    subject: str = "waveguide port",
) -> PortBox:
    """A mode launched over a cross-section.

    The excitation goes on the near face of this box and the probes on the far
    one, so the length is where the measurement plane sits. Neither shift
    applies, and the envelope refuses one.
    """
    length = length_of(stated_length, fallback, subject)
    start, stop = list(face[0]), list(face[1])
    start[propagation_axis] = middle(face, propagation_axis)
    stop[propagation_axis] = start[propagation_axis] + direction * length
    return PortBox(corner(start), corner(stop), propagation_axis, measurement=length, probe=length)
