# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""S-parameters, solver-neutral.

One N×N matrix against frequency, referenced to the impedance the user asked
for, with the provenance to say where it came from. An adapter reads its own
output into :meth:`SParameters.from_runs`. Plots, Touchstone and extracted
scalars then work the same whichever backend ran.

Imports numpy and the standard library at module scope. scikit-rf is reached
through :mod:`._skrf`, and only when a matrix is assembled or written.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from . import _skrf, modelled


class ResultError(ValueError):
    """The runs handed in cannot be assembled into one S-matrix."""


#: The structure is its own mirror image about the plane between its two ports,
#: and the ports are identical. Then S22 = S11 and, by reciprocity, S12 = S21,
#: so one solve determines the whole matrix and the second is redundant.
#:
#: The user declares this, and nothing infers it. Whether a drawing is symmetric
#: is a question about intent as much as geometry. A board that is symmetric to
#: within a via's placement is symmetric for this purpose, and no geometric test
#: will say so.
MIRROR = "mirror"

#: How far the runs of one sweep may disagree about a port's reference
#: impedance, per frequency point, before that point is unusable.
#:
#: The runs disagree because a microstrip port measures its own Z0 from
#: ``sqrt(Et*dEt / (Ht*dHt))``, evaluated as finite differences across three
#: grid lines. That form is exact in exact arithmetic however strong the
#: reflection. Near a standing-wave voltage null, ``V*dV`` and ``I*dI`` both
#: collapse toward zero at once and the ratio becomes indeterminate. Where the
#: nulls fall depends on which end is driven, so the runs of one sweep disagree
#: exactly where the extraction stopped working.
#:
#: A uniform two-port line agrees at every point. Moving a resonant open stub
#: near the probes makes the runs disagree around resonance, and putting the
#: stub under the probes makes the extraction return a reactive impedance for a
#: real line.
#:
#: This check catches disagreement rather than error. Two runs whose extraction
#: fails the same way agree perfectly and pass, which is the third case above:
#: both ports' probes inside the stub junction, both runs equally wrong, and no
#: spread between them. A single run has nothing to disagree with, and driving
#: one port is the common case. So this refuses a class of bad results and
#: certifies nothing about the rest. Only a measurement plane on clean feed
#: line makes the extraction sound, and no check after the solve can move
#: one.
IMPEDANCE_TOLERANCE = 0.10

#: The provenance key under which a matrix not every column of which is known
#: records, per driven column, how much of the wave driven into it the undriven
#: ports sent back.
SENT_BACK = "sent_back"

#: The provenance key under which a matrix reported against anything but its
#: ports' own impedance records how far that renormalisation can magnify an
#: error in the solve.
MAGNIFIED = "magnified"

#: The provenance key under which an adapter says, per port number, which of a
#: guide's impedances that port states. The openEMS adapter writes it and
#: spells it itself, since it imports nothing from this layer.
IMPEDANCE_STATED = "impedance_stated"

#: What a lumped port states under :data:`IMPEDANCE_STATED`: its resistance,
#: which its S-parameters are referenced to. A port stating it is no guide.
LUMPED_RESISTANCE = "the resistance of its lumped element"

#: How many times a renormalisation may magnify the solve's error before the
#: panel says so. A renormalisation past two can add more error than the solve
#: left.
MAGNIFICATION_TOLERANCE = 2.0

#: Why a frequency point holds no numbers: a port carries no power there, the
#: runs disagreed there about port impedance, or neither and the solve returned
#: no field.
_UNPOWERED, _DISCARDED, _EMPTY = "unpowered", "discarded", "empty"

#: The provenance key under which :meth:`SParameters.usable` records why the
#: points it dropped held no numbers, a sentence per cause, beside
#: ``discarded_points``, which lists them.
DROPPED_BECAUSE = "dropped_because"


def _span(hz: np.ndarray) -> str:
    """Where a set of frequencies lies, without implying it is contiguous.

    The phrasing is "lowest ... highest" rather than "between ... and ...". The
    points one resonance spoils are contiguous, but nothing makes them so. A
    structure with two resonances puts holes at both ends of a sweep, and
    "between 1 and 10 GHz" would report nine sound gigahertz as dead.
    """
    hz = np.asarray(hz, dtype=float)
    if hz.size == 1:
        return f"at {hz[0] / 1e9:.4g} GHz"
    return f"lowest {hz.min() / 1e9:.4g} GHz, highest {hz.max() / 1e9:.4g} GHz"


def _engineering_hz(hz: float) -> str:
    """One frequency, in the unit an engineer would have said it in.

    A Touchstone body is in hertz because the format says so. A header is read
    by a person, and 1e+10 is not how 10 GHz is written.
    """
    for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if abs(hz) >= scale:
            return f"{hz / scale:.6g} {unit}"
    return f"{hz:.6g} Hz"


def _band(hz: np.ndarray) -> str:
    """The span and how densely it was sampled."""
    hz = np.asarray(hz, dtype=float)
    if hz.size == 1:
        return _engineering_hz(float(hz[0]))
    return (
        f"{_engineering_hz(float(hz.min()))} to {_engineering_hz(float(hz.max()))}, "
        f"{hz.size} points"
    )


def _reference(z: np.ndarray) -> str:
    """What the S-parameters are referenced to.

    The value is always one real number. :meth:`SParameters._reference_lines`
    calls this only where :attr:`SParameters.one_reference` holds, and states
    any other reference itself. So this reports the number rather than
    surveying for one.
    """
    # Take ``.real`` first. A reference impedance is complex in general, and
    # casting the array whole would warn and discard rather than take what was
    # wanted.
    return f"{float(np.asarray(z).real.flat[0]):.6g} ohm"


#: Longest a caption may be. A header line is read at a glance beside the
#: others, and a FreeCAD ``Label`` has no length limit at all.
_CAPTION_LIMIT = 60


def _comment_text(value: object) -> str:
    """One line of external text, made safe to put in a comment.

    A Touchstone comment is not inert. The reader dispatches on what follows the
    ``!``. ``! gamma`` and ``! port impedance`` are HFSS extensions that consume
    the following lines as floats, so a study named "Gamma sweep" written bare
    would consume the option line and the file would no longer parse. The caller
    keeps user text off the front of the line. This function makes the text
    itself a single printable ASCII line.

    Whitespace collapses because an embedded newline splits one comment into a
    second line with no ``!`` on it, which the reader reads as data. Non-ASCII
    characters go because the format predates any encoding assumption and the
    parsers are old, and a degree sign in a label should not decide whether a
    file opens. The length is capped because a caption is read at a glance.
    """
    text = _comment_text_whole(value)
    if len(text) > _CAPTION_LIMIT:
        text = text[: _CAPTION_LIMIT - 3].rstrip() + "..."
    return text


def _comment_text_whole(value: object) -> str:
    """:func:`_comment_text` without the cap, for a caller wrapping the text.

    The reader also takes the wave definition from any comment holding
    ``S-parameter uses the ... definition``, wherever it stands, and the file
    names its own, so that phrase loses its hyphen here.
    """
    text = " ".join(str(value).split()).replace("S-parameter uses the", "S parameter uses the")
    return text.encode("ascii", "replace").decode("ascii")


#: What one port's reference reads as when there is no number to give it. Each
#: phrase follows "referenced to" and sits after "port 3:" equally well, because
#: a label is used in either position depending on what the other ports turn out
#: to say.
_OWN = "its own impedance"
_DISPERSIVE = "an impedance that varies with frequency"


def describe_reference(
    reference: np.ndarray, port_numbers: Sequence[int], at_own: Sequence[int] = ()
) -> str:
    """The reference impedance as a line to read, e.g. ``"50 ohm"``.

    This returns a string rather than a float, because the reference is per port
    per frequency and a single number could only be right by luck. One value
    across the whole array reads as one value. Ports that differ are named. A
    reference that varies with frequency is reported as varying rather than as
    its first bin.

    ``at_own`` names the ports referenced to their own impedance: a port the
    user pointed at itself, and an undriven column. It decides only what to say
    where there is no number. See :func:`_own_or`. A port at its own 50 ohm
    reads as 50 ohm. A guide at its own dispersive impedance reads as its own
    impedance, because folding that together with a requested 50 would read as
    "varies with frequency" and say nothing about either.

    The caller passes ``at_own`` rather than this function inferring it. The
    array cannot answer it for a derived mirror whose common impedance
    disperses: each port then reads as varying against its own measurement, and
    the pair sits at one impedance that is neither port's.
    :meth:`SParameters.from_runs` is the one place that knows.

    Every form has to read as English after the words "referenced to", which the
    chart's footnote, the Touchstone refusals and caveat, the file's header and
    the task panel's log line all prepend.
    """
    reference = np.atleast_2d(np.asarray(reference, dtype=complex))
    if reference.size == 0:
        return ""

    own = set(at_own)
    labels = [
        _own_or(_one_impedance(reference[:, column]), number in own)
        for column, number in enumerate(port_numbers)
    ]
    if len(set(labels)) == 1:
        if labels[0] == _OWN and len(labels) > 1:
            return "each port's own impedance"
        return labels[0]
    return ", ".join(
        f"port {number}: {label}" for number, label in zip(port_numbers, labels, strict=True)
    )


def _own_or(label: str, self_referenced: bool) -> str:
    """``its own impedance``, only where there is nothing else to say.

    A port at its own impedance still reads as the number when it has one, and a
    lumped port always has one: its own impedance is the resistance that was
    typed. Reporting the reason instead captions the commonest study in the
    workbench, two lumped ports at 50 ohm with one of them undriven, as though
    its two curves sat on different bases. The undriven column's basis makes no
    difference to any term drawn from the driven one.

    :attr:`SParameters.one_reference` already judges by the number rather than
    by what was declared, to decide what a Touchstone option line can hold. The
    format's limit is on the numbers, so a guide and a 50-ohm line that agree across the band
    are one reference. The two predicates differ: that one compares exactly, and
    this one to a tolerance.

    This reports what the terms are against, and not whether that is what was
    asked for. An undriven port keeps its own impedance whatever was requested,
    so where the two differ this reports the one that was got. Nothing states
    that difference to the user.

    The case the phrase was written for survives. A guide's own impedance
    disperses, so there is no number, and saying it varies with frequency would
    answer a different question than the one asked.
    """
    return _OWN if self_referenced and label == _DISPERSIVE else label


def _one_impedance(column: np.ndarray) -> str:
    """One port's reference across the band, as a number or as its absence."""
    if np.allclose(column, column.flat[0]):
        return _ohms(column.flat[0])
    return _DISPERSIVE


def _ohms(value: complex) -> str:
    """``"50 ohm"`` for a real reference, ``"49.7 - 0.8j ohm"`` for a complex one."""
    value = complex(value)
    if value.imag == 0.0:
        return f"{value.real:g} ohm"
    sign = "+" if value.imag >= 0 else "-"
    return f"{value.real:g} {sign} {abs(value.imag):g}j ohm"


def _ohms_to(value: complex, digits: int) -> str:
    """:func:`_ohms` for an impedance a run computed, to ``digits`` figures."""
    if value.imag == 0.0:
        return f"{_to(value.real, digits)} ohm"
    sign = "+" if value.imag >= 0 else "-"
    return f"{_to(value.real, digits)} {sign} {_to(abs(value.imag), digits)}j ohm"


def _to(value: float, digits: int) -> str:
    """``value`` to ``digits`` significant figures, without an exponent:
    ``1460``, not ``1.46e+03``."""
    return np.format_float_positional(
        value, precision=digits, unique=False, fractional=False, trim="-"
    )


def _up_to(bound: float, digits: int) -> str:
    """:func:`_to` for a bound, rounded up so that it still bounds: 2.03 is
    ``2.1`` and not ``2``."""
    unit = 10.0 ** (np.floor(np.log10(bound)) - digits + 1)
    return _to(np.ceil(bound / unit) * unit, digits)


def _normalisation(z: np.ndarray) -> np.ndarray:
    """Pseudo-wave normalisation ``sqrt(Re z) / |z|``, per port per frequency.

    openEMS reports wave amplitudes in volts. Its own
    ``openEMS/python/openEMS/ports.py`` computes
    ``uf_inc = (uf_tot + if_tot * Z_ref) / 2`` and ``uf_ref = uf_tot - uf_inc``,
    with no impedance normalisation at all. Every S-parameter definition in the
    literature normalises, so the ratio ``uf_ref_i / uf_inc_j`` that the driver
    writes is not S_ij unless the two ports share a reference impedance.

    scikit-rf does not fix this on the way in. Its ``s_def`` conversion returns
    early when every port impedance is real, because all three of its
    definitions already carry the normalisation. Its own docstring says "results
    are the same for real-valued characteristic impedances". Handing it raw
    volts would mislabel them.

    Between two ports of differing impedance, uncorrected S21 is not S21.
    ``test_sparameters`` holds both forms against a series resistor, where the
    closed form is exact.

    The pseudo-wave form (Marks & Williams) is the one to match, because openEMS
    decomposes with ``Z_ref`` rather than its conjugate, and a microstrip's
    ``Z_ref`` is genuinely complex. For real impedances it collapses to
    ``1 / sqrt(z)`` and the correction to ``sqrt(z_j / z_i)``.
    """
    return np.sqrt(z.real) / np.abs(z)


def _powered(z: np.ndarray) -> np.ndarray:
    """Boolean mask over frequency: points where every port's impedance has a
    positive real part.

    A port whose impedance has none carries no power, and no wave at it can be
    normalised: :func:`_normalisation` is zero there, and every term it divides
    is 0/0. A waveguide port's impedance below its mode's cutoff is ``k Z0 /
    beta`` with ``beta`` imaginary, which is such a point. A number that is not
    there is not asked about: an impedance a solver did not state is ``nan``,
    and :func:`_measured_impedance` refuses one a run returned.
    """
    return ~np.any(np.asarray(z, dtype=complex).real <= 0.0, axis=1)


def _start_past(frequency: np.ndarray, unpowered: np.ndarray, powered: np.ndarray) -> str:
    """Where to start a band so it leaves out the points ``unpowered`` marks.

    The first point past them where every port carries power is named rather
    than the last one where a port carries none, because the cutoff lies
    somewhere between the two, and a band started just past the second can
    still start below it.
    """
    top = float(frequency[unpowered].max())
    past = frequency[(frequency > top) & powered]
    if past.size:
        return (
            f"Start the band at {_engineering_hz(float(past.min()))} or above, where every "
            "port carries power"
        )
    return (
        f"The band holds no point above them, so where a port starts to carry power is "
        f"not known from here: start the band above the cutoff of each port's mode, "
        f"which lies above {_engineering_hz(top)}"
    )


def _unpowered_where(
    z: np.ndarray, frequency: np.ndarray, numbers: Sequence[int], where: str
) -> str:
    """Which ports carry no power at the points ``z`` holds, as a sentence.

    Where every port named lacks power at the same points they are named
    together. Where they differ, as the ports of a junction whose arms are cut
    off at different frequencies do, each is named with its own points.
    """
    lacking = [
        (n, z[:, i].real <= 0.0) for i, n in enumerate(numbers) if np.any(z[:, i].real <= 0.0)
    ]
    cause = "as a waveguide port's has below its mode's cutoff"
    if all(np.array_equal(mask, lacking[0][1]) for _, mask in lacking):
        ports = [n for n, _ in lacking]
        one = len(ports) == 1
        return (
            f"{_ports(ports).capitalize()} carr{'ies' if one else 'y'} no power {where}: "
            f"{'its' if one else 'their'} impedance has no real part, {cause}"
        )
    each = "; ".join(
        f"port {n} at {int(mask.sum())} ({_span(frequency[mask])})" for n, mask in lacking
    )
    lead = "At each of them" if where == "there" else f"A{where[1:]}"
    return (
        f"{lead} a port carries no power: {each}. A port's impedance has no real part "
        f"where it carries none, {cause}"
    )


@dataclass(frozen=True)
class SParameters:
    """One S-matrix against frequency, at a stated reference impedance.

    ``s`` is indexed ``[frequency, receiving, driving]``, and ``port_numbers``
    gives the document's port number for each index, in the same order. The two
    are read together, because a document whose ports are numbered 2 and 5 still
    produces a 2×2 matrix.

    The matrix may be incomplete, and it records that it is. FDTD drives one
    port per run, so column *j* exists only if port *j* was driven. ``driven``
    names the ports that were driven, and every other column is ``nan``. A
    time-domain solve produces exactly this, and so does a one-path VNA (a
    LiteVNA has one source and two receivers): S11 and S21, with S12 and S22 not
    measured. The gap is marked ``nan`` rather than zero because a zero is a
    number that will be plotted.

    Such a column is the device with every other port terminated as the run
    left it, and a port that sends back part of what reaches it is not
    terminated in its own reference impedance. A matrix every column of which
    is known carries none of that, because :meth:`from_runs` counts every wave.
    """

    frequency: np.ndarray
    s: np.ndarray
    port_numbers: tuple[int, ...]
    reference: np.ndarray
    measured_impedance: np.ndarray
    #: Document numbers of the ports that were driven, which are the columns
    #: that exist. ``None`` means all of them, which is what a full sweep
    #: produces and what any caller building a complete matrix by hand means.
    driven: tuple[int, ...] | None = None
    #: Undriven columns that were filled from a symmetry the user declared.
    #: They are present in the matrix and are kept distinct from the measured
    #: columns: the plot draws them dashed and labels them, a Touchstone header
    #: names them and quotes how far the two ports measured apart, the document
    #: stores them apart, and provenance records which symmetry was claimed.
    derived: tuple[int, ...] = ()
    #: Indices into :attr:`frequency` where every term is ``nan`` because the
    #: runs disagreed about port impedance by more than
    #: :data:`IMPEDANCE_TOLERANCE`. These are a hole in the band, and
    #: :attr:`unmeasured` is a hole in the matrix: those columns were never
    #: solved for, and these points were solved for and came back
    #: unnormalisable. A point is also solved for and comes back blank where a
    #: port carries no power, which :attr:`unpowered` names, and where the solve
    #: returned no field.
    #:
    #: They are blanked rather than dropped so a plot breaks its line where the
    #: numbers stop. Removing the points instead would join the two sides of the
    #: gap into one smooth curve, which fails the same way as writing a zero.
    discarded: tuple[int, ...] = ()
    #: Ports referenced to their own impedance rather than to a number: the ones
    #: the caller asked for nothing at, and the undriven columns.
    #:
    #: This is recorded rather than re-derived. :attr:`reference` and
    #: :attr:`measured_impedance` agree at such a port, but they also agree at a
    #: 50 ohm lumped port asked for 50, and they disagree for both ports of a
    #: derived mirror, which sit at the pair's one common impedance. Only
    #: :meth:`from_runs` knows which it did, so only it may say.
    self_referenced: tuple[int, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.driven is None:
            object.__setattr__(self, "driven", tuple(self.port_numbers))
        else:
            object.__setattr__(self, "driven", tuple(sorted(self.driven)))
        object.__setattr__(self, "derived", tuple(sorted(self.derived)))
        object.__setattr__(self, "discarded", tuple(sorted({int(i) for i in self.discarded})))
        # An undriven column is at its own impedance whatever the caller asked
        # for, so it belongs here however this object was built. This is not the
        # inference the field exists to avoid. :meth:`from_runs` states and
        # enforces the same rule, and it is restated here where it cannot be
        # forgotten.
        object.__setattr__(
            self,
            "self_referenced",
            tuple(sorted(set(self.self_referenced) | set(self.unmeasured))),
        )

    @property
    def ports(self) -> int:
        return len(self.port_numbers)

    @property
    def drivers(self) -> tuple[int, ...]:
        """The ports that were driven, after ``__post_init__`` has settled it.

        The field is declared as what a caller may pass, which includes leaving
        it out to mean every port. ``__post_init__`` fills that in.
        """
        assert self.driven is not None  # __post_init__ settles it
        return self.driven

    @property
    def known(self) -> frozenset:
        """Ports whose columns hold numbers, measured or derived."""
        return frozenset(self.drivers) | frozenset(self.derived)

    @property
    def complete(self) -> bool:
        """Whether every column holds numbers."""
        return self.known >= frozenset(self.port_numbers)

    @property
    def unmeasured(self) -> tuple[int, ...]:
        """Ports whose columns are ``nan`` - neither driven nor derived."""
        return tuple(n for n in self.port_numbers if n not in self.known)

    def _require_complete(self, action: str) -> None:
        missing = self.unmeasured
        if missing:
            raise ResultError(
                f"cannot {action}: port(s) {list(missing)} were never driven, so "
                f"{len(missing)} column(s) of this {self.ports}-port matrix are "
                "not measured. Mark them as excitation sources and solve again, "
                "or declare the study symmetric so the rest can be derived"
            )

    @property
    def unpowered(self) -> tuple[int, ...]:
        """Indices into :attr:`frequency` where some port carries no power, so
        no term there can be normalised. :func:`_powered` says which those are.

        This is read off :attr:`measured_impedance` rather than recorded, so a
        result reopened from the document says it as the one assembled did.
        """
        return tuple(int(i) for i in np.flatnonzero(~_powered(self.measured_impedance)))

    @property
    def kept(self) -> np.ndarray:
        """Boolean mask over :attr:`frequency`: points whose terms hold numbers.

        This reads the array rather than the bookkeeping alone.
        :attr:`discarded` and :attr:`unpowered` name the points assembly
        blanked, and that is not the whole test. A run can come back all
        ``nan`` with nothing having disagreed. A waveguide port whose plane
        misses the grid excites nothing, ``|uf_inc|`` is zero, every term is
        ``nan``, and a single run has nothing to disagree with, so ``discarded``
        stays empty. Such a result passes every refusal here and writes a
        Touchstone file of ``nan`` tokens.

        Only the columns that are supposed to hold numbers are read. An undriven
        column is legitimately ``nan``, and :meth:`_require_complete` covers
        that.
        """
        mask = _powered(self.measured_impedance)
        if self.discarded:
            mask[list(self.discarded)] = False
        columns = [i for i, n in enumerate(self.port_numbers) if n in self.known]
        if columns:
            mask &= np.isfinite(np.asarray(self.s)[:, :, columns]).all(axis=(1, 2))
        return mask

    def blank_span(self) -> str:
        """Where the points that hold no numbers are, as a phrase for a message."""
        blank = np.flatnonzero(~self.kept)
        if blank.size == 0:
            return ""
        return _span(np.asarray(self.frequency, dtype=float)[blank])

    def _blank_by_cause(self) -> list[tuple[str, np.ndarray]]:
        """The points that hold no numbers, gathered by why they hold none, as
        one of :data:`_UNPOWERED`, :data:`_DISCARDED` and :data:`_EMPTY` and a
        mask over :attr:`frequency`.

        A point goes under the first cause it has. A port that carries no power
        comes first: the runs are not asked to agree about a number no wave
        there could be normalised by.
        """
        blank = ~self.kept
        unpowered = blank & ~_powered(self.measured_impedance)
        discarded = np.zeros_like(blank)
        discarded[list(self.discarded)] = True
        discarded &= blank & ~unpowered
        empty = blank & ~unpowered & ~discarded
        causes = ((_UNPOWERED, unpowered), (_DISCARDED, discarded), (_EMPTY, empty))
        return [(cause, mask) for cause, mask in causes if mask.any()]

    def _why_blank(self) -> list[tuple[str, str]]:
        """Why the points that hold no numbers hold none, and what to do about
        it, one pair of sentences per cause.

        Where one cause covers every blank point it is said of them all; where
        more do, each says which points are its own.
        """
        gathered = self._blank_by_cause()
        frequency = np.asarray(self.frequency, dtype=float)
        said = []
        for cause, mask in gathered:
            where = (
                "there"
                if len(gathered) == 1
                else f"at {int(mask.sum())} of them ({_span(frequency[mask])})"
            )
            if cause == _UNPOWERED:
                said.append(
                    (
                        _unpowered_where(
                            np.asarray(self.measured_impedance, dtype=complex)[mask],
                            frequency[mask],
                            self.port_numbers,
                            where,
                        ),
                        _start_past(frequency, mask, _powered(self.measured_impedance)),
                    )
                )
            elif cause == _DISCARDED:
                said.append(
                    (
                        f"The runs disagreed about port impedance {where}",
                        "Move the measurement planes onto clean feed line so the "
                        "extraction works across the whole band",
                    )
                )
            else:
                said.append(
                    (
                        f"Nothing disagreed about port impedance {where} and every port "
                        "carried power: the solve itself returned no field",
                        "Check the port planes landed on grid lines",
                    )
                )
        return said

    def blank_causes(self) -> list[str]:
        """Why the points that hold no numbers hold none, a sentence per cause
        saying what to do about it. Empty where every point holds numbers."""
        return [f"{why}. {remedy}" for why, remedy in self._why_blank()]

    def _require_usable(self, action: str) -> None:
        keep = self.kept
        if keep.all():
            return
        blank = int((~keep).sum())
        # The way out is offered only when there is something to keep. An
        # all-blank solve has nothing to keep, and sending the user to a method
        # that can only hand back the same object is worse than saying nothing.
        way_out = (
            "Call usable() for the points that do"
            if keep.any()
            else "Not one point holds a number, so there is nothing to keep"
        )
        raise ResultError(
            f"cannot {action}: {blank} of {self.frequency.size} frequency "
            f"points hold no numbers ({self.blank_span()}). "
            + ". ".join([*self.blank_causes(), way_out])
        )

    def reference_description(self) -> str:
        """What this matrix is referenced to, as a line to read."""
        return describe_reference(self.reference, self.port_numbers, self.self_referenced)

    @property
    def one_reference(self) -> bool:
        """True when one real impedance covers every port at every frequency.

        That is all a Touchstone option line can state. This property is public
        so a caller can ask before opening a save dialog it would have to
        abandon. :attr:`unmeasured` and :meth:`_require_complete` divide the
        same way.
        """
        z = np.asarray(self.reference, dtype=complex)
        return not z.size or bool(np.all(z == z.flat[0]) and z.flat[0].imag == 0.0)

    @property
    def real_reference(self) -> bool:
        """True when every port's reference is a positive real number at every
        frequency.

        That is what a Touchstone file can state point by point. A complex one
        is not, because the wave definitions differ there, and scikit-rf reads a
        file in the per-point form as the one it takes HFSS to use unless the
        file names another. A number that is not there, where the solver stated
        none, is not either, and nor is one no port can have.
        """
        z = np.asarray(self.reference, dtype=complex)
        return bool(np.all(np.isfinite(z)) and np.all(z.imag == 0.0) and np.all(z.real > 0.0))

    def _require_statable(self, action: str, per_point: bool) -> None:
        """A Touchstone file states the reference once, or at every point.

        The option line holds one real number, shared by every port and every
        frequency. The HFSS form scikit-rf writes puts a reference that moves in
        a comment after each point's data, ``! Port Impedance`` and each port's
        value, and a reader that knows the form takes those instead. A reader
        that does not has only the option line's bare ``R``, which scikit-rf
        itself reads as 50 ohm and another reader may refuse. So that form is
        written only when the caller asks for it.
        Without this refusal, scikit-rf raises "Network has unequal port
        impedances but reference impedance for renormalization 'r_ref' is not
        specified" from inside a vendored library, after the save dialog, naming
        nothing about the model.

        This asks the array rather than what was declared, because the format's
        limit is on the numbers. A guide and a 50-ohm line that happen to agree
        across the band are one reference, and two ports declared identically
        whose measurements differ in the sixth digit are not. The message
        reports what was asked for, which is the part a user can act on.
        """
        if self.one_reference or (per_point and self.real_reference):
            return
        described = self.reference_description()
        if self.real_reference:
            raise ResultError(
                f"cannot {action}: this matrix is referenced to {described}, and a "
                "Touchstone option line states one real reference impedance for every "
                "port at every frequency. Pass per_point=True to state each port's at "
                "each point, which a reader that ignores that form reads with no "
                "reference, or reference every port to the same fixed impedance and "
                "assemble again"
            )
        raise ResultError(
            f"cannot {action}: this matrix is referenced to {described}, and a "
            "Touchstone file states a reference as a real impedance, once or at each "
            "point. Reference every port to the same fixed impedance and assemble "
            "again, or keep the result in the document, where the reference it was "
            "measured against is preserved"
        )

    def usable(self) -> SParameters:
        """This result restricted to the frequency points that hold numbers.

        The band comes back shorter and with a gap in it, so nothing does this
        implicitly. Touchstone and ``skrf.Network`` both need it, because a
        ``nan`` in either propagates through every later operation and arrives
        at the far end as a plot of nothing.

        This asks :attr:`kept`, for the same reason :meth:`_require_usable`
        does. An all-``nan`` solve can have nothing discarded, as the waveguide
        port whose plane missed the grid does, so guarding on :attr:`discarded`
        would hand back the identical object under a refusal telling the caller
        to ask for the points that do.
        """
        keep = self.kept
        if keep.all():
            return self
        provenance = dict(self.provenance)
        provenance["discarded_points"] = [float(value) for value in self.frequency[~keep]]
        provenance[DROPPED_BECAUSE] = [why for why, _ in self._why_blank()]
        return SParameters(
            frequency=self.frequency[keep],
            s=self.s[keep],
            port_numbers=self.port_numbers,
            reference=self.reference[keep],
            measured_impedance=self.measured_impedance[keep],
            driven=self.driven,
            derived=self.derived,
            self_referenced=self.self_referenced,
            provenance=provenance,
        )

    def sent_back(self, bar: float) -> str | None:
        """Which columns read alone can be off S by more than ``bar`` from what
        the ports not driven in their runs sent back, or ``None``.

        A matrix every column of which is known counted those waves out, and
        records nothing to say.
        """
        said = []
        records = self.provenance.get(SENT_BACK, {})
        for column, record in sorted(records.items(), key=lambda item: int(item[0])):
            if not record["bound"] > bar:
                continue
            said.append(
                f"A term of column {column} can be off S by up to about "
                f"{_magnitude(record['bound'])} at {_engineering_hz(record['frequency'])}, "
                f"against a bar of {_magnitude(bar)}: the ports not driven in the runs "
                f"behind it sent back part of what reached them, port {record['port']} "
                "most. That is an error in S itself, so a term much smaller than it is "
                "off by more than its own size"
            )
        if not said:
            return None
        remedy = f"Drive {_ports(self.unmeasured)} as well and every wave is counted out"
        if self.ports == 2:
            remedy += ", or declare the mirror if the device is one"
        return ". ".join([*said, remedy])

    def magnified(self) -> str | None:
        """Which ports are reported against a reference far enough from their
        own impedance that renormalising there can magnify an error in the
        solve past :data:`MAGNIFICATION_TOLERANCE` times, or ``None``.

        The ports named are the ones whose own VSWR against their reference
        passes the tolerance, or the one furthest out where none does alone.
        """
        record = self.provenance.get(MAGNIFIED)
        if not record or not record["factor"] > MAGNIFICATION_TOLERANCE:
            return None
        ports = record["ports"]
        named = sorted(
            (n for n in ports if ports[n]["vswr"] > MAGNIFICATION_TOLERANCE), key=int
        ) or [max(ports, key=lambda n: ports[n]["vswr"])]
        references = [_ohms(complex(*ports[n]["reference"])) for n in named]
        owns = [_ohms_to(complex(*ports[n]["own"]), 3) for n in named]
        one = len(named) == 1
        if one:
            own = f"its own impedance is {owns[0]}"
        elif len(set(owns)) == 1:
            own = f"their own impedance is {owns[0]}"
        else:
            own = f"their own impedances are {_listed(owns)}"
        against = references[0] if len(set(references)) == 1 else _listed(references)
        return (
            f"{_ports([int(n) for n in named]).capitalize()} {'is' if one else 'are'} "
            f"reported against {against} and {own} at "
            f"{_engineering_hz(record['frequency'])}, so renormalising there can magnify an "
            f"error in the solve by up to {_up_to(record['factor'], 2)} times. Reference "
            f"{'it to its' if one else 'them to their'} own impedance, where the error is "
            "not magnified"
        )

    def nonreciprocal(self, bar: float) -> str | None:
        """Where a matrix every column of which was measured has two terms
        reciprocity makes equal differ by more than ``bar``, or ``None``.

        Every material a model can hold is reciprocal, so the device's S is
        symmetric in power waves at any reference, and ``|S_ij - S_ji|`` is an
        error of the matrix. At least one of the two terms is off by half of it
        or more, so this is a floor on the error and never a bound: an error
        common to both terms leaves them equal. The departure itself is held to
        the bar, not the half of it that is the floor, which errs toward
        saying. The section
        docs/internals/s-matrix-from-runs.md#a-matrix-says-how-far-it-departs-from-reciprocity
        says what departs.

        Any other matrix says nothing. A column a declared mirror derived is
        symmetric with its source by construction, and the columns of a matrix
        not every column of which was driven carry what the undriven ports
        sent back, which :meth:`sent_back` says.
        """
        if not self.complete or self.derived:
            return None
        worst = (0.0, 0, 0, 0)
        for i in range(self.ports):
            for j in range(i + 1, self.ports):
                gap = np.abs(self.s[:, i, j] - self.s[:, j, i])
                if not np.any(np.isfinite(gap)):
                    continue
                point = int(np.nanargmax(gap))
                if gap[point] > worst[0]:
                    worst = (float(gap[point]), point, j, i)
        gap, point, row, column = worst
        if not gap > bar:
            return None
        one = _term(self.port_numbers[row], self.port_numbers[column])
        other = _term(self.port_numbers[column], self.port_numbers[row])
        return (
            f"{one} and {other} differ by up to {_magnitude(gap, 3)} at "
            f"{_engineering_hz(float(self.frequency[point]))}, against a bar of "
            f"{_magnitude(bar, 3)}. Every material a model holds is reciprocal, so the "
            f"device's two are equal, and at least one of them is off by "
            f"{_magnitude(gap / 2, 3)} or more there"
        )

    def mirror_disagreement(self) -> float:
        """How far a fully measured two-port misses being mirror-symmetric.

        The largest of ``|S22 - S11|`` and ``|S12 - S21|``, relative to the
        largest term in the matrix. Zero for a perfectly symmetric structure on
        a perfectly symmetric grid.

        This makes a declared symmetry falsifiable. Solve both ports once and
        read this: a small value means the declaration is safe and every later
        run can be half the cost. It is meaningless on a matrix that was itself
        derived from the declaration, so it refuses one.

        It answers the first of the two things the declaration needs, that the
        network is its own mirror image at a common reference. The second, that
        the two ports are mirror images of each other so their measured
        impedances agree, is not visible here, because renormalising both ports
        to 50 ohm removes it. A series element is its own mirror however unequal
        the impedances feeding it, so this metric reads zero for a two-port fed
        at 25 and 100 ohm. That check costs no solve and belongs in pre-flight.

        Both questions are asked at a common reference, and must stay that way.
        Certifying a mirror here at one reference while :func:`_mirror`
        completes it at another passes a structure the completion then gets
        wrong. A mirror has one Z0, and two different measured values are two
        estimates of it.
        """
        if self.ports != 2:
            raise ResultError(f"mirror symmetry compares two ports; this result has {self.ports}")
        if self.derived:
            raise ResultError(
                "this matrix was completed from the symmetry it is being asked "
                "to check, so the answer would be zero by construction. Drive "
                "both ports to test the claim"
            )
        self._require_complete("check mirror symmetry")

        # Measured over the points that hold numbers rather than refused. The
        # declaration is about the structure, so a band with a hole in it still
        # answers the question everywhere else. This measurement makes the claim
        # falsifiable, so it survives as much as it honestly can.
        keep = self.kept
        if not keep.any():
            raise ResultError(
                "no frequency point holds numbers, so there is nothing to compare. "
                + ". ".join(self.blank_causes())
            )

        a, b = self.port_numbers
        s = self.s[keep]
        scale = float(np.max(np.abs(s)))
        if scale == 0.0:
            return 0.0
        i, j = self.index_of(a), self.index_of(b)
        gaps = [
            np.abs(s[:, j, j] - s[:, i, i]),
            np.abs(s[:, i, j] - s[:, j, i]),
        ]
        return float(np.max(gaps) / scale)

    def index_of(self, port: int) -> int:
        """Row/column for a document port number."""
        if port not in self.port_numbers:
            raise ResultError(f"no port {port} in this result; it has {list(self.port_numbers)}")
        return self.port_numbers.index(port)

    def parameter(self, receiving: int, driving: int) -> np.ndarray:
        """One S-parameter by *document port number*, against frequency."""
        return self.s[:, self.index_of(receiving), self.index_of(driving)]

    def impedance(self, port: int) -> np.ndarray:
        """The reference impedance the solver used at this port.

        A microstrip port measures it, as ``sqrt(Et*dEt / (Ht*dHt))``, genuinely
        complex and frequency-dependent. A lumped port is set to the resistance
        the user typed. A waveguide port is analytic (``k*Z0/beta``). Calling
        all three "measured" would invite comparing a user input against a
        closed form and believing the agreement.

        The task panel should show this number. It is not the reference
        impedance the matrix is normalised to.
        """
        return self.measured_impedance[:, self.index_of(port)]

    def network(self) -> Any:
        """An ``skrf.Network``: Touchstone, cascading, de-embedding, plots.

        Refuses on an incomplete matrix. A ``Network`` whose columns are ``nan``
        propagates them through every operation it has, cascading, de-embedding
        and renormalising among them, and arrives at the far end as a plot of
        nothing with no indication of where it went wrong.
        """
        self._require_complete("build a Network")
        self._require_usable("build a Network")
        skrf = _skrf.module()
        return skrf.Network(
            frequency=skrf.Frequency.from_f(self.frequency, unit="hz"),
            s=self.s,
            z0=self.reference,
            s_def="power",
            name=str(self.provenance.get("title", "")) or None,
        )

    def touchstone_path(self, path: str | Path) -> Path:
        """Where :meth:`write_touchstone` would put a file asked for at ``path``.

        Public so that the caller can ask. A save dialog checks for an existing
        file under the name the user typed; if this returns a different one,
        that check was answered about the wrong file.

        The suffix is appended unless it is already exactly right, and never
        substituted. ``Path.with_suffix`` replaces whatever it finds after the
        last dot, and a version or a frequency in a filename is
        indistinguishable from a suffix. ``lowpass_2.4GHz`` comes out as
        ``lowpass_2.s2p``, so ``lowpass_2.4GHz`` and ``lowpass_2.9GHz`` write to
        one file, and any existing ``lowpass_2.s2p`` is overwritten without a
        prompt, because the dialog asked about a name that does not exist.

        A wrong Touchstone suffix is appended to rather than corrected:
        ``device.s3p`` from a two-port becomes ``device.s3p.s2p``. The result is
        deliberately ugly. Correcting it silently would hand back a file whose
        name says something the user did not ask for, and the ugly name reports
        what happened.
        """
        path = Path(path)
        wanted = f".s{self.ports}p"
        if path.suffix.lower() == wanted:
            return path
        return path.with_name(path.name + wanted)

    def write_touchstone(self, path: str | Path, per_point: bool = False) -> Path:
        """Write a Touchstone file and return where it went.

        A matrix at one real reference is written with it on the option line,
        whatever ``per_point`` says. Any other real reference is written at each
        point only when ``per_point`` asks for it, because a reader that ignores
        that form reads the file with no reference. :meth:`_require_statable`
        says why.

        Refuses on an incomplete matrix. A ``.sNp`` has a column for every term,
        so writing one from a one-path measurement means inventing the
        unmeasured terms. That is why Touchstone files exported from one-path
        VNAs are hazardous downstream: an assumed S12 and a fabricated S22 look
        identical to measured ones, and the file carries no way to tell. The
        workbench will grow explicit export modes for this, and silently
        guessing is not one of them.

        The name comes back from :meth:`touchstone_path`, which appends and
        never replaces. See its note for why replacing a suffix cannot be done
        safely.
        """
        self._require_complete("write a Touchstone file")
        self._require_usable("write a Touchstone file")
        self._require_statable("write a Touchstone file", per_point)
        written = self.touchstone_path(path)
        at_each_point = not self.one_reference

        # The filename is passed although nothing writes to it. With
        # ``return_string`` scikit-rf builds a StringIO and never opens a file.
        # It still demands a filename, and falls back to the Network's name,
        # which is ``provenance["title"] or None``, and a Problem's title
        # defaults to the empty string. Without the filename, a study whose
        # title was never set fails here on ``No filename given. Network must
        # have a name``, raised inside a vendored library after the solve and
        # naming nothing about the model.
        body = self.network().write_touchstone(
            filename=str(written),
            form="ri",
            return_string=True,
            skrf_comment=False,
            write_z0=at_each_point,
        )
        written.write_text(self._header() + body, encoding="utf-8")
        return written

    def _identification(self) -> list[str]:
        """What instrument, of what, when, over what band.

        This is the shape a network analyser writes. The file leaves with
        nothing beside it, so everything needed to know what it is has to be in
        it. A bare matrix of numbers is unusable six months later, and every
        commercial analyser answers that the same way, with producer, source,
        date, network size, span and reference.

        Both versions are written, and they answer different questions. The
        simulator produced the numbers, and this workbench arranged them.
        Reproducing a result wants the first the way reproducing a bench
        measurement wants the firmware revision, and a bug report has to name
        the second.

        Every value interpolated here goes through :func:`_comment_text`, and
        the study's name is labelled rather than left to start its line. A
        comment is not inert in this format. The reader dispatches on the word
        after the ``!``, and a study called "Gamma sweep" or "Port impedance
        study" would otherwise be read as an HFSS extension that swallows the
        option line beneath it.
        """
        rule = "!" + "=" * 74
        title = _comment_text(self.provenance.get("title") or "")
        solver = _comment_text(self.provenance.get("solver") or "")
        solver_version = _comment_text(self.provenance.get("solver_version") or "")
        simulator = " ".join(
            part for part in (solver, solver_version) if part and part != "unknown"
        )

        lines = [rule]
        if title:
            lines.append(f"! {'Study':<14}{title}")
        lines.append(f"! {'Produced by':<14}FreeCAD Microwave {__version__}")
        if simulator:
            lines.append(f"! {'Simulator':<14}{simulator}")
        lines.append(f"! {'Date':<14}{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"! {'Network':<14}{self.ports}-port")
        lines.append(f"! {'Frequency':<14}{_band(self.frequency)}")
        for index, piece in enumerate(self._reference_lines()):
            lines.append(f"! {'Reference' if not index else '':<14}{piece}")
        # Wrapped rather than capped: each line is one sentence, and a caption
        # cut short loses the half that says what the loss was held as.
        for record in self.provenance.get("modelled") or ():
            (said,) = modelled.said([record])
            caption = (
                "Boundary"
                if modelled.outside(record)
                else "Medium"
                if modelled.medium(record)
                else "Loss"
            )
            for index, piece in enumerate(textwrap.wrap(_comment_text_whole(said), _CAPTION_LIMIT)):
                lines.append(f"! {caption if not index else '':<14}{piece}")
        lines.append(rule)
        return lines

    def _reference_lines(self) -> list[str]:
        """What the file is referenced to, wrapped to the caption's width.

        One real number is the option line's, and says so in a word. Any other
        is stated at each point, and the file says where, what a reader that
        does not look there has instead, and which impedance of a guide each of
        its ports referenced to its own mode states. A port at a number the user
        typed states that number whatever its kind. Every line after the first
        sits under the label, where no word a Touchstone reader dispatches on
        can start it.
        """
        if self.one_reference:
            return [_reference(self.reference)]
        said = [
            f"{self.reference_description()}.",
            "Each port's reference at each frequency is on the comment after that "
            "point's data, in the HFSS form scikit-rf writes and reads. The option "
            "line states no number, which scikit-rf reads as 50 ohm and another "
            "reader may refuse.",
        ]
        stated = {
            int(number): meaning
            for number, meaning in (self.provenance.get(IMPEDANCE_STATED) or {}).items()
            if int(number) in self.self_referenced
        }
        for number, meaning in sorted(stated.items()):
            said.append(f"At port {number} it is {meaning}.")
        if any(meaning != LUMPED_RESISTANCE for meaning in stated.values()):
            said.append(
                "A guide's S-parameters referenced to its own mode are the same whichever "
                "of its impedances is stated, and renormalising them is not."
            )
        if LUMPED_RESISTANCE in stated.values():
            said.append(
                "A lumped port's S-parameters are referenced to its resistance, which does "
                "not change with frequency."
            )
        return [
            piece
            for sentence in said
            for piece in textwrap.wrap(_comment_text_whole(sentence), _CAPTION_LIMIT)
        ]

    def _header(self) -> str:
        """The comment block: what the file is, then whatever was not measured.

        The refusal above exists because a ``.sNp`` has a column for every term
        and no way to say which were invented, which makes files exported from
        one-path VNAs hazardous downstream. A matrix completed from a declared
        symmetry is complete, so it passes that refusal, and unannotated it
        would carry the same anonymity. This block states the assumption in the
        file, in Touchstone's own comment syntax, so the file can be handed on.
        """
        lines: list[str] = self._identification()
        # A band with a hole in it, written by usable(). The frequency list in
        # the file is shorter, and nothing downstream can distinguish that from
        # a sweep that was never asked for those points. So the gap is stated
        # here, for the same reason the derived columns are.
        dropped = self.provenance.get("discarded_points") or []
        if dropped:
            lines.append(
                f"! {len(dropped)} FREQUENCY POINT(S) ARE MISSING "
                f"({_span(np.asarray(dropped, dtype=float))})."
            )
            # Under a label, as the reference is, because a sentence can start
            # with a word a Touchstone reader dispatches on: "Ports 1 and 2".
            for why in self.provenance.get(DROPPED_BECAUSE) or ():
                pieces = textwrap.wrap(_comment_text_whole(f"{why}."), _CAPTION_LIMIT)
                for index, piece in enumerate(pieces):
                    lines.append(f"! {'Because' if not index else '':<14}{piece}")
        if not self.derived:
            return "\n".join(lines) + "\n"
        symmetry = self.provenance.get("symmetry", "a declared symmetry")
        mismatch = self.provenance.get("symmetry_impedance_mismatch")
        lines += [
            "! NOT ALL TERMS WERE MEASURED.",
            f"! Column(s) {list(self.derived)} were derived from {symmetry} "
            f"symmetry declared by the user, from the solve(s) that drove "
            f"port(s) {list(self.drivers)}.",
            "! The run driving port j is taken to be the solved run with the two ports exchanged.",
        ]
        if mismatch is not None:
            lines.append(
                f"! The two ports' measured reference impedances differ by "
                f"{float(mismatch):.3e}. Two ports of a true mirror have the "
                f"same one, so that is how far the declaration is from what was "
                f"measured - not an error bar on the numbers below."
            )
        return "\n".join(lines) + "\n"

    @classmethod
    def from_runs(
        cls,
        runs: Sequence[Any],
        reference: float | None | Sequence[float | None] = 50.0,
        symmetry: str | None = None,
    ) -> SParameters:
        """Assemble one matrix from one solve per driven port.

        FDTD drives one port per run, so column *j* comes from the run that
        excited port *j*. The runs must describe the same structure.
        ``document.sweep()`` guarantees that by re-exciting a single
        translation, and the cell-count check below refuses a list assembled by
        hand from two different solves.

        Fewer runs than ports is allowed, and produces a partial matrix rather
        than a refusal. Driving one port of a two-port measures S11 and S21 as a
        one-path VNA does, with port 2 terminated as the run left it. The
        undriven columns come back ``nan``, and :attr:`driven` says which those
        are. Where every column is known, every port's incident wave is counted
        and the matrix is the device between the ports' planes, whatever
        terminates them - :func:`_volts`.

        ``symmetry`` is the user's declaration about the structure.
        :data:`MIRROR` on a two-port with one run fills the missing column at
        half the solve time. :func:`_mirror` and :func:`_waves` do it, and
        docs/internals/s-matrix-from-runs.md says what it rests on.

        A frequency point where the runs disagree about a port's reference
        impedance by more than :data:`IMPEDANCE_TOLERANCE` comes back ``nan``
        and is named in :attr:`discarded`. The point fails, and the sweep does
        not. So does a point where a port's impedance has no real part, as a
        waveguide port's has below its mode's cutoff: the port carries no power
        there, and :attr:`unpowered` names the point.

        ``reference`` may be a scalar or one value per port, and applies to the
        ports whose columns are known. ``None`` references that port to its own
        impedance, which is the only expressible answer for a dispersive guide.
        A bench measures one by TRL against standards cut from the same guide,
        and no single number is entered anywhere. A column that is neither
        driven nor derived is treated the same way whatever was asked for,
        because moving it needs numbers no run measured.
        """
        if not runs:
            raise ResultError("no runs to assemble")

        frequency = np.asarray(runs[0].frequency, dtype=float)
        driven = [int(run.excited_port) for run in runs]
        if len(set(driven)) != len(driven):
            raise ResultError(
                f"two runs excited the same port ({sorted(driven)}); each column "
                "of the matrix needs its own solve"
            )

        numbers = tuple(sorted(runs[0].ports))
        _check_runs_can_form_one_matrix(runs, frequency, numbers)

        measured, per_run = _measured_impedance(runs, numbers, frequency)
        discarded, spread = _disagreement(per_run, measured, frequency)
        unpowered = tuple(int(i) for i in np.flatnonzero(~_powered(measured)))
        blank = tuple(sorted(set(discarded) | set(unpowered)))
        _require_a_point_to_keep(blank, unpowered, measured, numbers, frequency)
        derived, mismatch, own = _mirror(measured, numbers, sorted(driven), symmetry)
        incident, reflected = _waves(runs, numbers, own, derived)
        volts = _volts(incident, reflected, numbers, sorted(driven), derived, blank)

        # Zero where a port carries no power, and every term there is then
        # 0/0: blanked below, and never handed to the renormalisation.
        factor = _normalisation(own)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = volts * (factor[:, :, None] / factor[:, None, :])
        known = set(driven) | set(derived)
        wanted, at_own = _wanted_reference(reference, numbers, known, own)

        # Blank what was never measured. This runs after the renormalisation, so
        # that the zeros _volts left there could do their work, and before
        # anything else sees the matrix.
        renormalised = _renormalised(frequency, s, own, wanted)
        for column, number in enumerate(numbers):
            if number not in known:
                renormalised[:, :, column] = np.nan + 1j * np.nan
        # And blank the frequency points that had no reference impedance to be
        # normalised to. This blanks the whole N x N rather than one port's row
        # and column, because the renormalisation above mixes every term at a
        # given frequency and one bad Z0 spoils that point entirely. The
        # neighbouring points stay sound because renormalisation is per
        # frequency: s2z and z2s invert one N x N block at a time. They are not
        # left untouched. scikit-rf decides whether to renormalise at all with a
        # whole-band ``np.any(z0 != z_new)``, so one differing point moves the
        # entire band off its exact path and onto the Z-parameter one. The
        # points where a port carries no power are blanked with them.
        if blank:
            renormalised[list(blank), :, :] = np.nan + 1j * np.nan

        provenance = _provenance(runs, driven, spread)
        if derived:
            provenance["symmetry"] = symmetry
            provenance["derived_columns"] = list(derived)
            # How far the derivation's own precondition was missed. It is always
            # recorded, so a result cannot be trusted more than the basis it was
            # built on. The panel reports it when it is large enough to matter.
            provenance["symmetry_impedance_mismatch"] = mismatch
        # A complete matrix at its ports' own impedance has nothing to bound.
        if known != set(numbers) or not np.array_equal(wanted, own):
            columns = [numbers.index(number) for number in sorted(known)]
            derivative = _derivative(frequency, s, own, wanted, columns)
            if known != set(numbers):
                provenance[SENT_BACK] = _sent_back(
                    incident, _reach(derivative), own, numbers, driven, frequency, blank
                )
            magnified = _magnified(derivative, own, wanted, numbers, columns, frequency, blank)
            if magnified:
                provenance[MAGNIFIED] = magnified

        return cls(
            frequency=frequency,
            s=renormalised,
            port_numbers=numbers,
            reference=wanted,
            measured_impedance=measured,
            driven=tuple(sorted(driven)),
            derived=derived,
            discarded=discarded,
            self_referenced=tuple(at_own),
            provenance=provenance,
        )


def _check_runs_can_form_one_matrix(
    runs: Sequence[Any], frequency: np.ndarray, numbers: tuple[int, ...]
) -> None:
    """Refuse runs that cannot be columns of one matrix, or that measured nothing.

    The second question is not about the set. The runs must describe one
    structure sampled the same way, and each must also have measured something,
    since a run that excited nothing comes back looking complete.

    The first check is defence in depth. The real guarantee is upstream, where
    ``document.sweep()`` re-excites a single translation so every column
    describes the same structure by construction. This catches a list assembled
    by hand from separate solves.

    Sameness is judged on the cell count rather than on the envelope digest.
    Every run in a sweep drives a different port and so has a different envelope
    by construction, so a digest would reject the input this exists to accept.
    The cell count is a weaker structural invariant, and two different
    geometries essentially never agree on it.
    """
    for run in runs:
        if not np.array_equal(np.asarray(run.frequency, dtype=float), frequency):
            raise ResultError(
                "the runs were sampled at different frequencies, so their "
                "columns cannot form one matrix"
            )
        if tuple(sorted(run.ports)) != numbers:
            raise ResultError(
                f"run exciting port {run.excited_port} sees ports "
                f"{sorted(run.ports)}, not {list(numbers)}"
            )
        # A run that excited nothing still takes its full wall time and comes
        # back looking complete. openEMS reports a finite port impedance beside
        # a matrix of nan, because every S-parameter it forms is 0/0. A lumped
        # port whose box lies outside the grid does this, and the engine clips
        # such a box away without comment. The check reads the incident wave
        # rather than the answer, because the incident wave is where the cause
        # is visible.
        incident = np.asarray(run.port(int(run.excited_port)).incident)
        if incident.size and not np.any(incident):
            raise ResultError(
                f"the run driven by port {int(run.excited_port)} recorded "
                "no incident wave at any frequency, so it measured nothing "
                "and every S-parameter from it is 0/0. Check that the "
                "port's box lies inside the grid - openEMS clips one that "
                "does not away without comment - and that the face it was "
                "built from spans the gap it is meant to drive"
            )

        if run.provenance.get("cells") != runs[0].provenance.get("cells"):
            raise ResultError(
                "the runs were meshed differently "
                f"({run.provenance.get('cells')} cells against "
                f"{runs[0].provenance.get('cells')}), so they describe "
                "different structures and cannot form one matrix"
            )
        # The provenance keeps run 0's record of the loss for the whole matrix,
        # so a column solved against another model would be filed under it.
        if run.provenance.get("modelled") != runs[0].provenance.get("modelled"):
            raise ResultError(
                f"the run driven by port {int(run.excited_port)} was given its lossy "
                f"materials otherwise than the run driven by port "
                f"{int(runs[0].excited_port)}, so they describe different models "
                "and cannot form one matrix"
            )


def _measured_impedance(
    runs: Sequence[Any], numbers: tuple[int, ...], frequency: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each port's reference impedance per frequency, and the runs behind it.

    The value is averaged across the runs rather than taken from the first.
    Taking ``runs[0]``'s assumes Z_ref describes the port's cross-section rather
    than the run. That holds for a lumped port, where openEMS sets Z_ref to the
    resistance, and for a waveguide, where it is analytic. It fails for the one
    kind where it matters: an MSLPort measures ``sqrt(Et*dEt / (Ht*dHt))`` from
    that run's own fields, so the N runs disagree slightly and column *j*'s
    volts were decomposed with run *j*'s value. Run 0's value is then the wrong
    number for every column but its own, so the estimate that uses all N is the
    one to take.
    """
    per_run = np.stack(
        [
            np.column_stack([np.asarray(run.port(number).z0, dtype=complex) for number in numbers])
            for run in runs
        ]
    )
    measured = per_run.mean(axis=0)

    if not np.all(np.isfinite(measured)):
        column = int(np.argmax(~np.isfinite(measured).all(axis=0)))
        bad = numbers[column]
        missing = ~np.isfinite(measured[:, column])
        # The message names the run as well as the port. What leaves a port
        # without an impedance can be the run's excitation rather than the
        # port, so naming only the port can send the reader to the wrong object.
        drivers = ", ".join(
            f"port {int(run.excited_port)}"
            for run, values in zip(runs, per_run)
            if not np.all(np.isfinite(values[:, column]))
        )
        raise ResultError(
            f"port {bad} reports a non-finite reference impedance at "
            f"{int(missing.sum())} of {missing.size} frequency points "
            f"({_span(frequency[missing])}) in the run driven by {drivers}, so no "
            "S-matrix can be normalised"
        )
    return measured, per_run


def _disagreement(
    per_run: np.ndarray, measured: np.ndarray, frequency: np.ndarray
) -> tuple[tuple[int, ...], float]:
    """Frequency points the runs cannot agree a reference impedance at, and the worst gap.

    How far the runs disagree says whether averaging them papered over a real
    problem.

    The disagreement is judged per frequency point rather than over the whole
    band, because it is local. A microstrip port's measured Z0 goes
    indeterminate only near a standing-wave null, and where the nulls fall
    depends on which end is driven (see :data:`IMPEDANCE_TOLERANCE`). A
    band-wide maximum would throw away a whole sweep because a resonance broke a
    few of its points, most of which are sound.

    One run needs no special case. The mean of one is that one, the differences
    are exactly zero, and nothing is ever discarded. That is also the limit of
    the check; see :data:`IMPEDANCE_TOLERANCE` again.
    """
    disagreement = np.max(
        np.abs(per_run - measured) / np.maximum(np.abs(measured), 1e-30),
        axis=(0, 2),
    )
    discarded = tuple(int(i) for i in np.flatnonzero(disagreement > IMPEDANCE_TOLERANCE))
    spread = float(np.max(disagreement)) if disagreement.size else 0.0
    if len(discarded) == frequency.size and frequency.size:
        raise ResultError(
            f"the runs disagree about port impedance by up to {spread:.0%} "
            "at every frequency, so no point of this sweep can be "
            "normalised. Either the runs describe different structures, or "
            "a port's measurement plane is not on clean transmission line"
        )
    return discarded, spread


def _require_a_point_to_keep(
    blank: tuple[int, ...],
    unpowered: tuple[int, ...],
    measured: np.ndarray,
    numbers: tuple[int, ...],
    frequency: np.ndarray,
) -> None:
    """Refuse a sweep where no point can be normalised.

    :func:`_disagreement` refuses one where the runs disagree everywhere. This
    refuses one where at every point a port carries no power or the runs
    disagree, which a band lying below a waveguide port's cutoff throughout is.
    """
    if not frequency.size or len(blank) < frequency.size:
        return
    ports = [n for i, n in enumerate(numbers) if np.any(measured[list(unpowered), i].real <= 0.0)]
    one = len(ports) == 1
    rest = (
        "" if len(unpowered) == frequency.size else ", and the runs disagree about it at the rest"
    )
    raise ResultError(
        f"no frequency point of this sweep can be normalised: {_ports(ports)} "
        f"carr{'ies' if one else 'y'} no power at {len(unpowered)} of the "
        f"{frequency.size}, where {'its' if one else 'their'} impedance has no real part, "
        f"as a waveguide port's has below its mode's cutoff{rest}. "
        + _start_past(frequency, ~_powered(measured), _powered(measured))
    )


def _waves(
    runs: Sequence[Any],
    numbers: tuple[int, ...],
    own: np.ndarray,
    derived: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Every port's incident and reflected wave in every column, split at ``own``.

    Both come back indexed ``[frequency, port, column]``, as voltage waves, with
    column *j* the run that drove port *j*. A column nobody drove is zero, and a
    column a declared mirror derives is the driven one with the two ports
    exchanged.

    openEMS splits a port's voltage and current with that run's own reference
    impedance, and a microstrip port measures its impedance anew in each run, so
    two runs hand over one port's waves on two slightly different bases. The
    matrix is normalised at one impedance per port, so each run's waves are
    split again at that one. ``u = inc + ref`` and ``i = (inc - ref) / z`` are
    the voltage and current the waves were split from, and splitting them again
    is a change of reference whichever reading produced them. Wherever a run's
    impedance is already the one asked for this returns its waves unchanged.
    """
    size = own.shape[0]
    incident = np.zeros((size, len(numbers), len(numbers)), dtype=complex)
    reflected = np.zeros_like(incident)
    by_excitation = {int(run.excited_port): run for run in runs}
    for column, driving in enumerate(numbers):
        run = by_excitation.get(driving)
        if run is None:
            continue
        for row, number in enumerate(numbers):
            port = run.port(number)
            inc = np.asarray(port.incident, dtype=complex)
            ref = np.asarray(port.reflected, dtype=complex)
            z = np.asarray(port.z0, dtype=complex)
            ratio = own[:, row] / z
            # Where the impedance does not move the waves are kept as they
            # are, so a wave nobody read does not spoil the other one through
            # 0 * nan. Both are asked: numpy's complex division of a number by
            # itself misses one in the last bit for some values, and divides
            # two numbers a bit apart to exactly one for others.
            moved = (own[:, row] != z) & (ratio != 1)
            incident[:, row, column] = np.where(
                moved, 0.5 * ((1 + ratio) * inc + (1 - ratio) * ref), inc
            )
            reflected[:, row, column] = np.where(
                moved, 0.5 * ((1 - ratio) * inc + (1 + ratio) * ref), ref
            )
    # The mirror exchanges the two ports, so the run it stands for sees at each
    # port what the run that was solved saw at the other.
    for other in derived:
        (driving,) = (number for number in numbers if number != other)
        a, b = numbers.index(driving), numbers.index(other)
        incident[:, :, b] = incident[:, ::-1, a]
        reflected[:, :, b] = reflected[:, ::-1, a]
    return incident, reflected


def _volts(
    incident: np.ndarray,
    reflected: np.ndarray,
    numbers: tuple[int, ...],
    driven: Sequence[int],
    derived: tuple[int, ...],
    blanked: tuple[int, ...],
) -> np.ndarray:
    """The matrix in voltage waves, every wave counted wherever every column is known.

    A run measures ``b = S a`` at every port at once. Taking column *j* as
    ``b / a_j`` treats every other port's incident wave as zero, which is what
    a port terminated in its own reference impedance would send back. A port
    sends back whatever terminates it behind its plane instead - an absorber, a
    guide that goes on changing past the port, a lumped port's own reactance -
    and each term of the column then carries ``S_ik a_k / a_j`` of it. Where
    every column is known, as the runs of every port or one run and the mirror
    of it, the matrix is ``B A^-1`` and nothing is carried: that is the device
    between the ports' planes whatever terminates them. Where it is not, the
    column is ``b / a_j``.

    ``B A^-1`` is taken as ``(B D^-1) (A D^-1)^-1``, ``D`` being the driven
    waves, so that ``A D^-1`` is the identity plus what the other ports sent
    back. It has no inverse where the runs' incident waves are in proportion:
    in a two-port, where what each undriven port sends back multiplies to one,
    which takes a termination behind each port returning all of what reaches
    it. No column is separable there, and the assembly is refused by how many
    points it spoiled, as a nan is. The points ``blanked`` names, those
    :func:`_disagreement` discarded and those where a port carries no power,
    keep ``b / a_j``, which the blanking after the renormalisation takes out.

    A column nobody drove is zero, and the caller blanks it again after the
    renormalisation. The zero is safe because the driven columns come out
    identical whatever sits in the undriven ones. See
    :func:`_wanted_reference`. A nan among the known columns, or a driven wave
    of nothing, is a real fault. This refuses it by name, and by how many points
    it spoiled, rather than letting it reach the inverse inside ``s2z``.
    """
    known = [numbers.index(number) for number in sorted(set(driven) | set(derived))]
    unknown = [column for column in range(len(numbers)) if column not in known]
    driving = np.diagonal(incident, axis1=1, axis2=2)[:, None, :]
    # A column read as b / a_j uses its own driven wave and none of the others.
    used = incident if not unknown else driving
    blank = ~np.isfinite(reflected) | ~np.isfinite(used) | (driving == 0)
    blank[:, :, unknown] = False
    if blank.any():
        worst_row, worst_column = np.unravel_index(
            int(np.argmax(blank.any(axis=0))), blank.shape[1:]
        )
        row, column = int(worst_row), int(worst_column)
        points = int(blank[:, row, column].sum())
        raise ResultError(
            f"{_term(numbers[row], numbers[column])} is not a number at {points} "
            f"of {blank.shape[0]} frequency points, so no S-matrix can be "
            "normalised from these runs. The run driven by port "
            f"{numbers[column]} returned no usable field there"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        volts = reflected / driving
        scaled = incident / driving
    volts[:, :, unknown] = 0
    if unknown:
        return volts

    kept = np.ones(volts.shape[0], dtype=bool)
    kept[list(blanked)] = False
    # Singular to working precision: past this no digit of the inverse stands.
    inseparable = kept & ~(np.linalg.cond(scaled) < 1 / np.finfo(float).eps)
    if inseparable.any():
        raise ResultError(
            f"the runs' incident waves are in proportion at {int(inseparable.sum())} "
            f"of {inseparable.size} frequency points, so no column can be told "
            "apart from the others there: what the undriven ports send back "
            "returns all of what reaches them. Something behind a port's plane "
            "reflects it whole - draw each guide or line running on unchanged "
            "from its port into the absorber"
        )
    volts[kept] = volts[kept] @ np.linalg.inv(scaled[kept])
    return volts


def _renormalised(
    frequency: np.ndarray, s: np.ndarray, own: np.ndarray, wanted: np.ndarray
) -> np.ndarray:
    """``s``, pseudo-waves at ``own``, as power waves at ``wanted``.

    A point where a port's own impedance carries no power comes back ``nan``
    and is left out of the renormalisation. Every term there is 0/0, and
    scikit-rf renormalises through Z-parameters, whose inverse refuses a
    ``nan`` anywhere in the band.
    """
    skrf = _skrf.module()
    powered = _powered(own)
    renormalised = np.full(s.shape, np.nan + 1j * np.nan, dtype=complex)
    if not powered.any():
        return renormalised
    network = skrf.Network(
        frequency=skrf.Frequency.from_f(frequency[powered], unit="hz"),
        s=s[powered].copy(),
        z0=own[powered],
        s_def="pseudo",
    )
    network.renormalize(wanted[powered], s_def="power")
    renormalised[powered] = np.asarray(network.s, dtype=complex)
    return renormalised


#: The finite step :func:`_derivative` moves one term of the matrix by. The
#: renormalisation is a rational function of terms of order one, so a step this
#: size leaves the derivative right to about the same figure. scikit-rf
#: renormalises through Z-parameters, which do not exist where ``I - S`` is
#: singular - a series element, a matched line a whole number of half
#: wavelengths long - and it moves such a matrix off the singularity by far more
#: than rounding, so there the derivative is only approximate, and mostly too
#: large.
_STEP = 1e-6


def _derivative(
    frequency: np.ndarray,
    s: np.ndarray,
    own: np.ndarray,
    wanted: np.ndarray,
    columns: Sequence[int],
) -> np.ndarray:
    """How far each term of the reported matrix moves per unit of error in each
    term of the known columns of ``s``.

    ``derivative[f, i, c, r, j]`` is the derivative of the reported term
    ``(i, c)`` with respect to the term ``(r, j)`` of ``s``, for ``j`` in
    ``columns``. Where every port is reported at its own impedance it is taken
    as the identity: at a real impedance the renormalisation is the identity,
    and at a complex one it turns pseudo-waves into power waves, which scales a
    term by ``Re Z / |Z|`` at most, so the identity errs toward complaining.
    Otherwise it is taken of :func:`_renormalised` itself, by a finite step in
    each term, so that it is of the arithmetic the matrix was made with. The
    renormalisation is a rational function of the terms and of no conjugate of
    them, so a real step gives the complex derivative.
    """
    size, count = s.shape[0], s.shape[1]
    derivative = np.zeros((size, count, count, count, count), dtype=complex)
    if np.array_equal(wanted, own):
        for column in columns:
            for row in range(count):
                derivative[:, row, column, row, column] = 1.0
        return derivative
    base = _renormalised(frequency, s, own, wanted)
    for column in columns:
        for row in range(count):
            stepped = s.copy()
            stepped[:, row, column] += _STEP
            moved = _renormalised(frequency, stepped, own, wanted) - base
            derivative[:, :, :, row, column] = moved / _STEP
    return derivative


def _reach(derivative: np.ndarray) -> np.ndarray:
    """How far each term of the reported matrix moves per unit of error in each
    known column of ``s``, whatever the error's direction within the column.

    ``reach[f, i, c, j]`` is the norm over ``r`` of ``derivative[f, i, c, r, j]``
    from :func:`_derivative`.
    """
    return np.sqrt(np.sum(np.abs(derivative) ** 2, axis=3))


def _magnified(
    derivative: np.ndarray,
    own: np.ndarray,
    wanted: np.ndarray,
    numbers: tuple[int, ...],
    columns: Sequence[int],
    frequency: np.ndarray,
    blank: tuple[int, ...],
) -> dict[str, Any] | None:
    """How many times the renormalisation to ``wanted`` can magnify an error in
    the known columns of ``s``, or ``None`` where every port is at its own.

    The factor at a frequency is the norm of :func:`_derivative` as a linear map
    from the known columns' error to the reported matrix's, both in the vector
    norm of their terms. The record keeps its worst over the points not
    blanked where some port moved, the frequency, and for each port moved
    there its reference, its own impedance and the VSWR of the one against the
    other, which is what the panel names a port by. Between real impedances the
    largest of those VSWRs, ``(1 + |G|) / (1 - |G|)``, bounds the factor of a
    passive network, and a matched line between ports of one impedance reaches
    it:
    docs/internals/s-matrix-from-runs.md#a-reference-far-from-the-port-magnifies-the-solves-error.

    A point where no port moved has nothing to magnify, and is left out rather
    than measured: a finite step there reads scikit-rf's way round a singular
    ``I - S``, which is not the identity the renormalisation is.
    """
    if np.array_equal(wanted, own):
        return None
    size, count = derivative.shape[0], derivative.shape[1]
    known = derivative[:, :, columns][:, :, :, :, columns]
    width = count * len(columns)
    flat = known.reshape(size, width, width)
    kept = np.isfinite(flat).all(axis=(1, 2))
    kept &= np.any(wanted[:, columns] != own[:, columns], axis=1)
    kept[list(blank)] = False
    if not kept.any():
        return None
    factor = np.zeros(size)
    factor[kept] = np.linalg.norm(flat[kept], ord=2, axis=(1, 2))
    at = int(np.argmax(factor))
    ports = {}
    for column in columns:
        z, reference = complex(own[at, column]), complex(wanted[at, column])
        if z == reference:
            continue
        reflection = abs((reference - z) / (reference + z))
        ports[str(numbers[column])] = {
            "reference": [reference.real, reference.imag],
            "own": [z.real, z.imag],
            "vswr": float((1 + reflection) / (1 - reflection)),
        }
    return {"factor": float(factor[at]), "frequency": float(frequency[at]), "ports": ports}


def _sent_back(
    incident: np.ndarray,
    reach: np.ndarray,
    own: np.ndarray,
    numbers: tuple[int, ...],
    driven: Sequence[int],
    frequency: np.ndarray,
    blank: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    """How far each driven column of a matrix read a column at a time can be
    off S, from what the ports not driven in its run sent back.

    Read alone, column *j* is ``S[:, j] + S[:, k] x`` with ``x = a_k / a_j``
    over the other ports ``k``, in power waves at each port's own impedance. A
    passive network's S is a contraction, ``|S x| <= |x|``, so column *j* is off
    by at most ``share_j = sqrt(sum_k |a_k|^2) / |a_j|`` in the vector norm, and
    by as much where the network loses nothing. The renormalisation to the
    reference asked for is a function of every driven column at once, so it
    changes each column's error, enlarging or shrinking it, and carries it into
    the others: term ``(i, c)`` of the matrix reported is off by at most
    ``sum_j share_j * reach[i, c, j]`` to first order, ``reach`` from
    :func:`_reach`. At each port's own impedance that is ``share_c``.

    Each driven column records its worst bound over its terms and the points not
    blanked, the frequency, and the port that sent back most there into the
    run that contributes most. A wave nobody read leaves its port out.
    """
    size, count = frequency.size, len(numbers)
    columns = [numbers.index(number) for number in sorted(driven)]
    power = np.abs(incident) * _normalisation(own)[:, :, None]
    share = np.zeros((size, count))
    loudest = np.zeros((size, count), dtype=int)
    for column in columns:
        others = [row for row in range(count) if row != column]
        back = np.nan_to_num(power[:, others, column])
        # The driven wave is finite and not zero, or _volts refused the runs,
        # and it carries power at every point not blanked.
        with np.errstate(divide="ignore", invalid="ignore"):
            share[:, column] = np.sqrt(np.sum(back**2, axis=1)) / power[:, column, column]
        loudest[:, column] = np.asarray(others)[np.argmax(back, axis=1)]

    contribution = reach * share[:, None, None, :]
    contribution[list(blank)] = 0.0
    bound = contribution.sum(axis=3)
    record: dict[str, dict[str, Any]] = {}
    for column in columns:
        at, row = np.unravel_index(int(np.argmax(bound[:, :, column])), (size, count))
        source = int(np.argmax(contribution[at, row, column]))
        record[str(numbers[column])] = {
            "bound": float(bound[at, row, column]),
            "frequency": float(frequency[at]),
            "port": numbers[int(loudest[at, source])],
        }
    return record


def _magnitude(value: float, digits: int = 2) -> str:
    """A magnitude in S, linear and in decibels: ``0.0043 (-47.3 dB)``."""
    return f"{value:.{digits}g} ({20 * np.log10(value):.1f} dB)"


def _term(row: int, column: int) -> str:
    """``S21``, and ``S1,12`` where a port's number has more than one digit."""
    return f"S{row}{column}" if row < 10 and column < 10 else f"S{row},{column}"


def _ports(numbers: Sequence[int]) -> str:
    """``port 2``, ``ports 2 and 3``, ``ports 2, 3 and 4``."""
    return ("port " if len(numbers) == 1 else "ports ") + _listed([str(n) for n in numbers])


def _listed(words: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + f" and {words[-1]}"


def _wanted_reference(
    reference: float | None | Sequence[float | None],
    numbers: tuple[int, ...],
    known: set[int],
    own: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """What each port is to be referenced to, and which ports that leaves at their own.

    One rule covers this: a port is referenced to the number the caller asked
    for, and to its own impedance when the caller asked for nothing. That rule
    covers a port the user pointed at itself, an undriven column, and both ports
    of a derived mirror, which share one impedance by the user's declaration.

    The function takes ``own`` rather than each port's own measurement, for the
    mirror's sake. :func:`_mirror` has already put both ports' waves on one
    common basis, and asking for the measurements here would renormalise the two
    diagonal terms by different amounts and undo the symmetry it just made
    exact.

    An undriven port keeps its own impedance whatever was asked for. That makes
    its renormalisation the identity and leaves the driven columns exactly
    referenced to what the caller asked for. The algebra, and how the
    independence was confirmed against scikit-rf rather than against it, are in
    docs/internals/s-matrix-from-runs.md#an-undriven-port-keeps-its-own-impedance.
    """
    # ``dtype=object`` because ``None`` is one of the values and a complex array
    # cannot hold it. A scalar broadcasts across the ports, which is what a
    # "50-ohm system" is.
    asked = np.broadcast_to(np.asarray(reference, dtype=object), (len(numbers),))
    wanted = own.copy()
    at_own = []
    for column, number in enumerate(numbers):
        if number in known and asked[column] is not None:
            wanted[:, column] = complex(asked[column])
        else:
            at_own.append(number)
    return wanted, at_own


def _provenance(runs: Sequence[Any], driven: Sequence[int], spread: float) -> dict[str, Any]:
    """Where the matrix came from, merged across the runs that made it.

    Fields that describe the structure survive from any run. Fields that
    describe one solve must not, or a matrix assembled from several runs reports
    run 0's wall time as the whole sweep's.
    """
    per_solve = (
        "wall_seconds",
        "recorded_samples",
        "tail_share",
        "near_field",
        "power_balance",
        "max_timesteps",
        "end_criteria",
    )
    provenance = {key: value for key, value in runs[0].provenance.items() if key not in per_solve}
    # Which impedance a port states is kept where every run states the same.
    # The reference is taken over every run, so a port stated two ways is
    # referenced to a mean of two definitions, which no phrase names.
    stated = [run.provenance.get(IMPEDANCE_STATED) or {} for run in runs]
    agreed = {
        number: meaning
        for number, meaning in stated[0].items()
        if all(other.get(number) == meaning for other in stated[1:])
    }
    provenance.pop(IMPEDANCE_STATED, None)
    if agreed:
        provenance[IMPEDANCE_STATED] = agreed
    for key in per_solve:
        values = {
            int(run.excited_port): run.provenance.get(key) for run in runs if key in run.provenance
        }
        if values:
            provenance[key] = values
    provenance["result_library"] = _skrf.description()
    provenance["excitations"] = sorted(driven)
    provenance["reproducible"] = all(run.reproducible for run in runs)
    # The worst disagreement anywhere in the band, including the points that
    # were discarded. It describes the solve rather than the matrix that came
    # out of it, and ``discarded`` says which points survived. The frequencies
    # are not recorded as well: they are already the indices in ``discarded``
    # against the frequency axis, so a second copy could drift from it.
    provenance["impedance_spread"] = spread
    # One digest per run, keyed by the port it drove. A single field cannot
    # describe a matrix built from N envelopes, and dropping them would leave
    # the result unfalsifiable.
    provenance["envelope_digest"] = {
        int(run.excited_port): run.provenance.get("envelope_digest") for run in runs
    }
    return provenance


def mirror_filled(
    s: np.ndarray,
    numbers: tuple[int, ...],
    driven: Sequence[int],
    impedance: np.ndarray,
    symmetry: str | None,
) -> tuple[np.ndarray, tuple[int, ...], float | None]:
    """A matrix each column of which is at its own port's impedance, with the
    column a declared mirror determines filled in.

    Returns the matrix, the columns filled and how far the two ports' stated
    impedances disagree, as a share of the driven port's; ``None`` where either
    port states none.

    For a matrix whose every undriven port was terminated in its own reference
    while the driven one was measured, as a frequency-domain solve terminates a
    port it does not drive. The run nobody solved is then the solved one with
    the two ports exchanged, term for term: S22 is S11 and S12 is S21. A matrix
    assembled from time-domain runs is not one of these, since an undriven port
    there sends back part of what reaches it; :meth:`SParameters.from_runs`
    fills that one from the waves instead.

    Nothing is filled where :func:`_mirror` fills nothing: no declaration, a
    matrix that is not a two-port, or one whose columns are both or neither
    driven.
    """
    if not symmetry:
        return s, (), None
    if symmetry != MIRROR:
        raise ResultError(f"unknown symmetry {symmetry!r}; this layer knows {MIRROR!r}")
    missing = [n for n in numbers if n not in set(driven)]
    if not missing or len(numbers) != 2 or len(driven) != 1:
        return s, (), None
    (other,) = missing
    a = numbers.index(driven[0])
    b = numbers.index(other)
    filled = s.copy()
    filled[:, b, b] = s[:, a, a]
    filled[:, a, b] = s[:, b, a]
    stated = np.asarray(impedance)
    mismatch = None
    if np.all(np.isfinite(stated[:, [a, b]])):
        scale = np.maximum(np.abs(stated[:, a]), 1e-30)
        mismatch = float(np.max(np.abs(stated[:, a] - stated[:, b]) / scale))
    return filled, (other,), mismatch


def _mirror(
    measured: np.ndarray,
    numbers: tuple[int, ...],
    driven: Sequence[int],
    symmetry: str | None,
) -> tuple[tuple[int, ...], float, np.ndarray]:
    """The column a declared mirror determines, and the one basis it holds in.

    Returns the columns to derive, how far the two ports' measured impedances
    disagree, and what each port's waves are split at. That last value is
    ``measured`` when nothing is derived, and the common basis below when
    something is. It is also what "the port's own impedance" means to the
    caller: a mirror declares the two ports to be the same port, so under it
    they have one impedance and the gap between their two measurements is noise.

    The mirror stands for the run nobody solved: the same structure driven from
    the other end sees at each port what the solved run saw at the other.
    :func:`_waves` builds that run and :func:`_volts` counts both, so the
    derived column and the measured one come from one ``B A^-1``. That needs the
    two ports at one reference, and a microstrip port measures
    ``sqrt(Et*dEt / (Ht*dHt))`` from its own probe fields rather than being
    told, so two of them disagree. The common basis is the mean of the two
    measurements, each being an estimate of the one impedance the declaration
    says they share.

    Why the returned gap is evidence about the declaration rather than an error
    bar on S is in docs/internals/s-matrix-from-runs.md.
    """
    if not symmetry:
        return (), 0.0, measured
    if symmetry != MIRROR:
        raise ResultError(f"unknown symmetry {symmetry!r}; this layer knows {MIRROR!r}")

    missing = [n for n in numbers if n not in set(driven)]
    # Nothing to fill, or nothing this symmetry knows how to fill. Deriving
    # nothing is the right outcome, and it is what the pre-flight warning
    # promises ("nothing will be derived from it here"). Raising would throw
    # away a good partial matrix, minutes after the solve, because of a
    # declaration that does not apply to it.
    if not missing or len(numbers) != 2 or len(driven) != 1:
        return (), 0.0, measured

    (other,) = missing
    a = numbers.index(driven[0])
    b = numbers.index(other)

    scale = np.maximum(np.abs(measured[:, a]), 1e-30)
    mismatch = float(np.max(np.abs(measured[:, a] - measured[:, b]) / scale))

    common = measured.copy()
    common[:, a] = common[:, b] = 0.5 * (measured[:, a] + measured[:, b])
    return (other,), mismatch, common
