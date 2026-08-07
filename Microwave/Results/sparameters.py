# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""S-parameters, solver-neutral.

One N×N matrix against frequency, referenced to the impedance the user asked
for, with the provenance to say where it came from. An adapter reads its own
output into :meth:`SParameters.from_runs` and everything above - plots,
Touchstone, extracted scalars - works the same whichever backend ran.

Imports numpy and the standard library at module scope. scikit-rf is reached
through :mod:`._skrf`, and only when a matrix is actually assembled or written.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from . import _skrf


class ResultError(ValueError):
    """The runs handed in cannot be assembled into one S-matrix."""


#: The structure is its own mirror image about the plane between its two ports,
#: and the ports are identical. Then S22 = S11 and, by reciprocity, S12 = S21 -
#: so one solve determines the whole matrix and the second is redundant.
#:
#: A declaration by the user, never inferred. Whether a drawing is symmetric is
#: a question about intent as much as geometry: a board that is symmetric to
#: within a via's placement is symmetric for this purpose and no geometric test
#: will say so.
MIRROR = "mirror"

#: How far the runs of one sweep may disagree about a port's reference
#: impedance, per frequency point, before that point is unusable.
#:
#: They disagree at all because a microstrip port *measures* its own Z0 from
#: ``sqrt(Et*dEt / (Ht*dHt))``, evaluated as finite differences across three
#: grid lines. That form is exact in exact arithmetic however strong the
#: reflection - but near a standing-wave voltage null ``V*dV`` and ``I*dI``
#: both collapse toward zero at once, and the ratio becomes indeterminate.
#: Where the nulls fall depends on which end is driven, so the runs of one
#: sweep disagree exactly where the extraction stopped working.
#:
#: On a resonant open stub: a uniform two-port line agrees exactly at every
#: point; the same line with the stub clear of the probes disagrees at the two
#: points nearest resonance; and with the stub *under* the probes the extraction
#: returns a wildly reactive impedance for a real line.
#:
#: **It catches disagreement, not error, and the difference matters.** Two runs
#: whose extraction fails the same way agree perfectly and pass - which is
#: exactly the third case above: both ports' probes inside the stub junction,
#: both runs equally wrong, and no spread between them. A single run has nothing
#: to disagree with at all, and one-port-driven is the common case. So this
#: refuses a class of bad results rather than certifying the rest: a measurement
#: plane on clean feed line is what makes an extraction right, and no check
#: downstream of the solve can supply one.
IMPEDANCE_TOLERANCE = 0.10


def _span(hz: np.ndarray) -> str:
    """Where a set of frequencies lies, without implying it is contiguous.

    "Lowest ... highest" and not "between ... and ...": the points one resonance
    spoils are contiguous, but nothing makes them so - a structure with two
    resonances puts holes at both ends of a sweep, and "between 1 and 10 GHz"
    would report nine sound gigahertz as dead.
    """
    hz = np.asarray(hz, dtype=float)
    if hz.size == 1:
        return f"at {hz[0] / 1e9:.4g} GHz"
    return f"lowest {hz.min() / 1e9:.4g} GHz, highest {hz.max() / 1e9:.4g} GHz"


def _engineering_hz(hz: float) -> str:
    """One frequency, in the unit an engineer would have said it in.

    A Touchstone body is in hertz because the format says so. A header is read
    by a person, and 1e+10 is not how anyone describes 10 GHz.
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

    One real number, always: :meth:`SParameters._require_one_reference` refuses
    a file whose ports differ or whose impedance is complex, and it does so
    before this is called. So this reports the number rather than surveying for
    one, and if that ever stops being true the refusal will fire before reaching
    here rather than this quietly printing the first port's.
    """
    # `.real` first: a reference impedance is complex in general, and casting
    # the array whole would warn and discard rather than take what was wanted.
    return f"{float(np.asarray(z).real.flat[0]):.6g} ohm"


#: Longest a caption may be. A header line is read at a glance beside seven
#: others, and a FreeCAD ``Label`` has no length limit at all.
_CAPTION_LIMIT = 60


def _comment_text(value: object) -> str:
    """One line of somebody else's text, made safe to put in a comment.

    A Touchstone comment is not inert. The reader dispatches on what follows
    the ``!`` - ``! gamma`` and ``! port impedance`` are HFSS extensions that
    consume the *following* lines as floats - so a study named "Gamma sweep"
    written bare would eat the option line and the file would no longer parse.
    The caller keeps user text off the front of the line; this makes the text
    itself a single printable ASCII line.

    Each for its own reason. Whitespace collapses because an embedded newline
    splits one comment into a second line with no ``!`` on it, which the reader
    reads as data. Non-ASCII goes because the format predates any encoding
    assumption and the parsers are old - a degree sign in a label should not
    decide whether a file opens. And the length is capped because a caption is
    read at a glance.
    """
    text = " ".join(str(value).split())
    text = text.encode("ascii", "replace").decode("ascii")
    if len(text) > _CAPTION_LIMIT:
        text = text[: _CAPTION_LIMIT - 3].rstrip() + "..."
    return text


#: What one port's reference reads as when there is no number to give it. Both
#: are phrased to follow "referenced to" and to sit after "port 3:" equally
#: well, because a label is used in either position depending on what the other
#: ports turn out to say.
_OWN = "its own impedance"
_DISPERSIVE = "an impedance that varies with frequency"


def describe_reference(reference: np.ndarray, port_numbers, at_own=()) -> str:
    """The reference impedance as a line to read, e.g. ``"50 ohm"``.

    A string and not a float, because the reference is per port per frequency
    and a single number could only be right by luck. One value across the whole
    array reads as one value; ports that differ are named; a reference that
    varies with frequency says so rather than quietly reporting its first bin.

    ``at_own`` names the ports referenced to their own impedance: a port the
    user pointed at itself, and a column nobody drove. It decides only what to
    say where there is **no** number - see :func:`_own_or`. A port at its own
    50 ohm reads as 50 ohm; a guide at its own dispersive impedance reads as
    its own impedance, because folding that together with a requested 50 would
    read as "varies with frequency" and say nothing about either.

    Told rather than inferred, because the array cannot answer it for a derived
    mirror whose common impedance disperses: each port then reads as varying
    against its own measurement, and the pair sits at one impedance that is
    neither port's. :meth:`SParameters.from_runs` is the one place that knows.

    Every shape has to read as English after the words "referenced to", which
    the chart's footnote, both Touchstone refusals and the task panel's log
    line all prepend.
    """
    reference = np.atleast_2d(np.asarray(reference, dtype=complex))
    if reference.size == 0:
        return ""

    at_own = set(at_own)
    labels = [
        _own_or(_one_impedance(reference[:, column]), number in at_own)
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
    """``its own impedance`` only where that is the *only* thing there is to say.

    A port at its own impedance still reads as the number when it has one, and
    a lumped port always has one: its own impedance is the resistance that was
    typed. Reporting the reason instead captions the commonest study in the
    workbench - two lumped ports at 50 ohm, one of them undriven - as though its
    two curves sat on different bases, when the undriven column's basis makes
    no difference to any term drawn from the driven one.

    Judging by the number rather than by what was declared is the principle
    :attr:`SParameters.one_reference` already applies to decide what Touchstone
    can hold - the format's limit is on the *numbers*, so a guide and a 50-ohm
    line that agree across the band are one reference. Not the same predicate:
    that one compares exactly and this one to a tolerance.

    **It answers what the terms are against, and not whether that is what was
    asked for.** An undriven port keeps its own impedance whatever was
    requested, so where the two differ this reports the one that was got.
    Nothing says so out loud today.

    What survives is the case the phrase was written for. A guide's own
    impedance disperses, so there is no number, and saying it varies with
    frequency would answer a different question than the one asked.
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


def _normalisation(z: np.ndarray) -> np.ndarray:
    """Pseudo-wave normalisation ``sqrt(Re z) / |z|``, per port per frequency.

    openEMS reports wave amplitudes in **volts** - ``ports.py`` computes
    ``uf_inc = (uf_tot + if_tot * Z_ref) / 2`` and ``uf_ref = uf_tot - uf_inc``,
    with no impedance normalisation at all. Every S-parameter definition in the
    literature normalises; the ratio ``uf_ref_i / uf_inc_j`` that the driver
    writes is therefore not S_ij unless the two ports happen to share a
    reference impedance.

    This is *not* something scikit-rf will fix on the way in. Its ``s_def``
    conversion returns early when every port impedance is real - "results are
    the same for real-valued characteristic impedances", says its own
    docstring - because all three of its definitions already carry the
    normalisation. Handing it raw volts would simply mislabel them.

    Between two ports of differing impedance the uncorrected S21 is wrong by
    tens of percent; corrected, it agrees with the closed form to rounding.
    ``test_sparameters`` holds both against a series resistor, where the closed
    form is exact.

    The pseudo-wave form (Marks & Williams) is the right one to match, because
    openEMS decomposes with ``Z_ref`` rather than its conjugate - and a
    microstrip's ``Z_ref`` is genuinely complex, so the choice is not academic.
    For real impedances it collapses to ``1 / sqrt(z)`` and the correction to
    ``sqrt(z_j / z_i)``.
    """
    return np.sqrt(z.real) / np.abs(z)


@dataclass(frozen=True)
class SParameters:
    """One S-matrix against frequency, at a stated reference impedance.

    ``s`` is indexed ``[frequency, receiving, driving]`` and ``port_numbers``
    gives the document's port number for each index, in the same order - the
    two are read together, because a document whose ports are numbered 2 and 5
    still produces a 2×2 matrix.

    **It may be incomplete, and says so.** FDTD drives one port per run, so
    column *j* exists only if port *j* was driven; ``driven`` names the ones
    that were, and every other column is ``nan``. That is not a degraded
    result - it is what a time-domain solve produces, and it is the same
    thing a one-path VNA (a LiteVNA has one source and two receivers) gives
    you: S11 and S21, exactly, with S12 and S22 simply not measured. Marking
    the gap as ``nan`` rather than zero is the whole point: a zero is a number
    somebody will plot.
    """

    frequency: np.ndarray
    s: np.ndarray
    port_numbers: tuple[int, ...]
    reference: np.ndarray
    measured_impedance: np.ndarray
    #: Document numbers of the ports that were driven - the columns that
    #: exist. ``None`` means all of them, which is what a full sweep produces
    #: and what any caller building a complete matrix by hand means.
    driven: tuple[int, ...] | None = None
    #: Columns nobody drove that were filled from a symmetry the *user*
    #: declared. Present in the matrix, never confusable with measured: the
    #: plot draws them dashed and labels them, a Touchstone header names them
    #: and quotes the error the copy carries, the document stores them apart,
    #: and provenance records which symmetry was claimed.
    derived: tuple[int, ...] = ()
    #: Indices into :attr:`frequency` where every term is ``nan`` because the
    #: runs disagreed about port impedance by more than
    #: :data:`IMPEDANCE_TOLERANCE`. A hole in the *band*, where
    #: :attr:`unmeasured` is a hole in the *matrix*: those columns were never
    #: solved for, these points were solved for and came back unnormalisable.
    #:
    #: They are blanked rather than dropped so a plot breaks its line where the
    #: numbers stop. Removing the points instead would join the two sides of the
    #: gap into one smooth curve, which is the same failure as writing a zero.
    discarded: tuple[int, ...] = ()
    #: Ports referenced to their own impedance rather than to a number: the ones
    #: the caller asked for nothing at, and the columns nobody drove.
    #:
    #: Recorded rather than re-derived. :attr:`reference` and
    #: :attr:`measured_impedance` agree at such a port, but they also agree at a
    #: 50 ohm lumped port *asked* for 50, and they disagree for the derived half
    #: of a mirror, which sits at the pair's one common impedance. Only
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
        # A column nobody drove is at its own impedance whatever the caller
        # asked for, so it belongs here however this object was built. That is
        # not the inference this field exists to avoid - it is a rule
        # :meth:`from_runs` states and enforces, restated where it cannot be
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
    def known(self) -> frozenset:
        """Ports whose columns hold numbers, measured or derived."""
        return frozenset(self.driven) | frozenset(self.derived)

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
    def _kept(self) -> np.ndarray:
        """Boolean mask over :attr:`frequency`: points whose terms hold numbers.

        Asks the **array**, not only the bookkeeping. :attr:`discarded` records
        the points the impedance comparison blanked, which is not the whole test
        - a run can come back all ``nan`` with nothing having disagreed. A
        waveguide port whose plane misses the grid excites nothing, ``|uf_inc|``
        is zero, every term is ``nan``, and a single run has nobody to disagree
        with, so ``discarded`` stays empty. Such a result passes every refusal
        here and writes a Touchstone file of ``nan`` tokens.

        Only the columns that are supposed to hold numbers are asked: an
        undriven column is legitimately ``nan``, and :meth:`_require_complete`
        is what covers that.
        """
        mask = np.ones(np.asarray(self.frequency).size, dtype=bool)
        if self.discarded:
            mask[list(self.discarded)] = False
        columns = [i for i, n in enumerate(self.port_numbers) if n in self.known]
        if columns:
            mask &= np.isfinite(np.asarray(self.s)[:, :, columns]).all(axis=(1, 2))
        return mask

    def blank_span(self) -> str:
        """Where the points that hold no numbers are, as a phrase for a message."""
        blank = np.flatnonzero(~self._kept)
        if blank.size == 0:
            return ""
        return _span(np.asarray(self.frequency, dtype=float)[blank])

    def _require_usable(self, action: str) -> None:
        keep = self._kept
        if keep.all():
            return
        blank = int((~keep).sum())
        why = (
            "because the runs disagreed there about port impedance"
            if self.discarded
            else "and nothing disagreed about them - the solve itself returned "
            "no field, so check the port planes landed on grid lines"
        )
        # Only offered when there is something to keep. On an all-blank solve
        # there is not, and sending the user to a method that can only hand
        # back the same object is worse than saying nothing.
        way_out = (
            "Call usable() for the points that do, or move"
            if keep.any()
            else "Not one point holds a number, so there is nothing to keep: move"
        )
        raise ResultError(
            f"cannot {action}: {blank} of {self.frequency.size} frequency "
            f"points hold no numbers ({self.blank_span()}), {why}. {way_out} "
            "the measurement planes onto clean feed line so the extraction "
            "works across the whole band"
        )

    def reference_description(self) -> str:
        """What this matrix is referenced to, as a line to read."""
        return describe_reference(self.reference, self.port_numbers, self.self_referenced)

    @property
    def one_reference(self) -> bool:
        """True when one real impedance covers every port at every frequency.

        Which is all a Touchstone file can state. Public so a caller can ask
        before opening a save dialog it would have to abandon - the same
        division :attr:`unmeasured` and :meth:`_require_complete` already have.
        """
        z = np.asarray(self.reference, dtype=complex)
        return not z.size or bool(np.all(z == z.flat[0]) and z.flat[0].imag == 0.0)

    def _require_one_reference(self, action: str) -> None:
        """A Touchstone file states one reference impedance, for everything.

        One real number in the option line, shared by every port and every
        frequency - the format has nowhere else to put one. So a matrix
        referenced to a port's own impedance cannot be written at all, and
        neither can the ordinary 30/75 study, which is the case that reached
        the user first: scikit-rf raises *"Network has unequal port impedances
        but reference impedance for renormalization 'r_ref' is not specified"*
        from inside a vendored library, after the save dialog, naming nothing
        about the model.

        Asked of the **array** rather than of what was declared, because the
        format's limit is on the numbers: a guide and a 50-ohm line that happen
        to agree across the band are writable, and two ports declared
        identically whose measurements differ in the sixth digit are not. The
        message says what was asked for - that is the part a user can act on.
        """
        if self.one_reference:
            return
        raise ResultError(
            f"cannot {action}: this matrix is referenced to "
            f"{self.reference_description()}, "
            "and a Touchstone file states one real reference impedance for every "
            "port at every frequency. Reference every port to the same fixed "
            "impedance and assemble again, or keep the result in the document, "
            "where the reference it was measured against is preserved"
        )

    def usable(self) -> SParameters:
        """This result restricted to the frequency points that hold numbers.

        The band comes back shorter and with a gap in it - which is honest, and
        is why nothing does this implicitly. Touchstone and ``skrf.Network`` both
        need it, because a ``nan`` in either propagates through every later
        operation and arrives at the far end as a plot of nothing.

        Asks :attr:`_kept`, for the same reason :meth:`_require_usable` does.
        Guarding on :attr:`discarded` alone meant an all-``nan`` solve with
        nothing discarded - the waveguide port whose plane missed the grid -
        was refused with "call ``usable()`` for the points that do" by a
        refusal that then handed back the identical object.
        """
        keep = self._kept
        if keep.all():
            return self
        provenance = dict(self.provenance)
        provenance["discarded_points"] = [float(value) for value in self.frequency[~keep]]
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

    def mirror_disagreement(self) -> float:
        """How far a fully measured two-port misses being mirror-symmetric.

        The largest of ``|S22 - S11|`` and ``|S12 - S21|``, relative to the
        largest term in the matrix. Zero for a perfectly symmetric structure on
        a perfectly symmetric grid.

        This is what turns a declared symmetry into something falsifiable: solve
        both ports once, read this, and if it is small the declaration is safe
        and every later run can be half the cost. It is meaningless on a matrix
        that was itself derived from the declaration, which is why it refuses
        one.

        It answers **one** of the two things the declaration needs: is the
        *network* its own mirror image, at a common reference? The other - that
        the two ports are mirror images of each other, so their measured
        impedances agree - is not visible here, because renormalising both
        ports to 50 ohm removes it. A series element is its own mirror however
        unequal the impedances feeding it, so this metric reads zero for a
        two-port fed at 25 and 100 ohm. That check costs no solve and belongs in
        pre-flight.

        Both questions are asked at a common reference, and must stay that way:
        certifying a mirror here at one reference while :func:`_derive_mirror`
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

        # Over the points that hold numbers, rather than a refusal. The
        # declaration is about the *structure*, so a band with a hole in it
        # still answers the question everywhere else - and this is the one
        # thing that makes the claim falsifiable, so it should survive as much
        # as it honestly can.
        keep = self._kept
        if not keep.any():
            raise ResultError(
                "no frequency point holds numbers, so there is nothing to "
                "compare"
                + (
                    ": the runs disagreed about port impedance everywhere"
                    if self.discarded
                    else ": the solve returned no field at all, so check the "
                    "port planes landed on grid lines"
                )
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

        **Measured** for a microstrip - ``sqrt(Et*dEt / (Ht*dHt))``, genuinely
        complex and frequency-dependent - but *set* for a lumped port, where it
        is the resistance the user typed, and analytic for a waveguide
        (``k*Z0/beta``). Calling all three "measured" would invite comparing a
        user input against a closed form and believing the agreement.

        This is the number the task panel should show, and it is *not* the
        reference impedance the matrix is normalised to.
        """
        return self.measured_impedance[:, self.index_of(port)]

    def network(self):
        """An ``skrf.Network``: Touchstone, cascading, de-embedding, plots.

        Refuses on an incomplete matrix. A ``Network`` whose columns are ``nan``
        propagates them through every operation it has - cascading,
        de-embedding, renormalising - and arrives at the far end as a plot of
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

        Public because the caller has to be able to *ask*. A save dialog checks
        for an existing file under the name the user typed; if this returns a
        different one, that check was answered about the wrong file.

        The suffix is **appended** unless it is already exactly right. It used
        to be substituted, via ``Path.with_suffix``, and that quietly destroyed
        data: ``with_suffix`` replaces whatever it finds after the last dot, and
        a version or a frequency in a filename is indistinguishable from a
        suffix: ``lowpass_2.4GHz`` comes out as ``lowpass_2.s2p``, so
        ``lowpass_2.4GHz`` and ``lowpass_2.9GHz`` write to one file, and any
        existing ``lowpass_2.s2p`` is overwritten without a prompt, the dialog
        having asked about a name that does not exist.

        A *wrong* Touchstone suffix is appended to rather than corrected:
        ``device.s3p`` from a two-port becomes ``device.s3p.s2p``. Ugly on
        purpose. Correcting it silently would hand back a file whose name says
        something the user did not ask for, and the ugliness is the report.
        """
        path = Path(path)
        wanted = f".s{self.ports}p"
        if path.suffix.lower() == wanted:
            return path
        return path.with_name(path.name + wanted)

    def write_touchstone(self, path: str | Path) -> Path:
        """Write a Touchstone file and return where it went.

        Refuses on an incomplete matrix, and that is deliberate. A ``.sNp`` has
        a column for every term, so writing one from a one-path measurement
        means inventing the terms nobody measured - which is exactly why
        Touchstone files exported from one-path VNAs are hazardous downstream:
        an assumed S12 and a fabricated S22 look identical to measured ones, and
        the file carries no way to tell. The workbench will grow explicit export
        modes for this; silently guessing is not one of them.

        The name comes back from :meth:`touchstone_path`, which **appends** and
        never replaces. See its note: replacing was the earlier behaviour and it
        destroyed files.
        """
        self._require_complete("write a Touchstone file")
        self._require_usable("write a Touchstone file")
        self._require_one_reference("write a Touchstone file")
        written = self.touchstone_path(path)

        # The filename is passed although nothing writes to it: with
        # ``return_string`` scikit-rf builds a StringIO and never opens a file.
        # It still *demands* one - and falls back to the Network's name, which
        # is ``provenance["title"] or None``, and a Problem's title defaults to
        # the empty string. So a study whose title was never set died here on
        # ``No filename given. Network must have a name``, from inside a
        # vendored library, naming nothing about the model, after the solve.
        body = self.network().write_touchstone(
            filename=str(written), form="ri", return_string=True, skrf_comment=False
        )
        written.write_text(self._header() + body, encoding="utf-8")
        return written

    def _identification(self) -> list[str]:
        """What instrument, of what, when, over what band.

        The shape a network analyser writes: the file leaves with nothing beside
        it, so everything needed to know what it is has to be in it. A bare
        matrix of numbers is unusable six months later, and every commercial
        analyser answers that the same way - producer, source, date, network
        size, span, reference.

        Both versions, and they answer different questions: the simulator's is
        what produced the numbers, ours is what arranged them. Reproducing a
        result wants the first the way reproducing a bench measurement wants
        the firmware revision, and the second is what a bug report has to name.

        Every value interpolated here goes through :func:`_comment_text`, and
        the study's name is **labelled rather than left to start its line**.
        A comment is not inert in this format: the reader dispatches on the
        word after the ``!``, and a study called "Gamma sweep" or "Port
        impedance study" would otherwise be read as an HFSS extension that
        swallows the option line beneath it.
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
        lines.append(f"! {'Reference':<14}{_reference(self.reference)}")
        lines.append(rule)
        return lines

    def _header(self) -> str:
        """The comment block: what the file is, then whatever was not measured.

        The refusal above exists because a ``.sNp`` has a column for every term
        and no way to say which were invented - which is what makes files
        exported from one-path VNAs hazardous downstream. A matrix completed
        from a declared symmetry is *complete*, so it passes the refusal, and
        it would carry exactly that anonymity if it went out unannotated. It
        does not: the assumption is stated in the file, in Touchstone's own
        comment syntax, so the file can be handed to anyone.
        """
        lines: list[str] = self._identification()
        # A band with a hole in it, written by usable(). The frequency list in
        # the file is simply shorter, which nothing downstream can distinguish
        # from a sweep that was never asked for those points - so it is said
        # here, for the same reason the derived columns are.
        dropped = self.provenance.get("discarded_points") or []
        if dropped:
            lines += [
                f"! {len(dropped)} FREQUENCY POINT(S) ARE MISSING "
                f"({_span(np.asarray(dropped, dtype=float))}).",
                "! The solves disagreed about port impedance there, so no "
                "reference impedance could be established and those points "
                "could not be normalised.",
            ]
        if not self.derived:
            return "\n".join(lines) + "\n"
        symmetry = self.provenance.get("symmetry", "a declared symmetry")
        mismatch = self.provenance.get("symmetry_impedance_mismatch")
        lines += [
            "! NOT ALL TERMS WERE MEASURED.",
            f"! Column(s) {list(self.derived)} were derived from {symmetry} "
            f"symmetry declared by the user, from the solve(s) that drove "
            f"port(s) {list(self.driven)}.",
            "! S(j,j) is a copy of S(i,i); S(i,j) is a copy of S(j,i) by reciprocity.",
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
        excited port *j*. They must describe the same structure:
        ``document.sweep()`` guarantees that by re-exciting a single
        translation, and the cell-count check below is what refuses a list
        assembled by hand from two different solves.

        **Fewer runs than ports is allowed**, and produces an honest partial
        matrix rather than a refusal - driving one port of a two-port measures
        S11 and S21 exactly, as a one-path VNA does. The undriven columns come
        back ``nan`` and :attr:`driven` says which those are.

        ``symmetry`` is the user's declaration about the structure.
        :data:`MIRROR` on a two-port with one run fills the missing column -
        S22 from S11, S12 from S21 - at half the solve time. The identities
        need both ports at **one** reference, so the driven port is
        renormalised to the other port's measured impedance, the copy is made
        there, and the result is renormalised to what the caller asked for. See
        :func:`_derive_mirror`.

        A frequency point where the runs disagree about a port's reference
        impedance by more than :data:`IMPEDANCE_TOLERANCE` comes back ``nan``
        and is named in :attr:`discarded`. That is a *point* failing and not
        the sweep: the extraction goes indeterminate near a standing-wave null
        and nowhere else.

        ``reference`` may be a scalar or one value per port, and applies to the
        ports whose columns are known. ``None`` references that port to **its
        own** impedance, which is the only expressible answer for a dispersive
        guide - a bench measures one by TRL against standards cut from the same
        guide, and no single number is entered anywhere. A column that is
        neither driven nor derived gets the same treatment whatever was asked
        for, because moving it needs numbers nobody has.
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

        measured, per_run = _measured_impedance(runs, numbers)
        discarded, spread = _disagreement(per_run, measured, frequency)
        volts = _volts(runs, numbers, frequency)

        factor = _normalisation(measured)
        s = volts * (factor[:, :, None] / factor[:, None, :])

        skrf = _skrf.module()
        network = skrf.Network(
            frequency=skrf.Frequency.from_f(frequency, unit="hz"),
            s=s,
            z0=measured,
            s_def="pseudo",
        )

        derived, mismatch, own = _derive_mirror(
            network, measured, numbers, sorted(driven), symmetry
        )
        known = set(driven) | set(derived)
        wanted, at_own = _wanted_reference(reference, numbers, known, own)
        network.renormalize(wanted, s_def="power")

        # Blank what was never measured. After the renormalisation, so the zeros
        # _volts left there could do their work, and before anything else sees
        # the matrix.
        renormalised = np.asarray(network.s, dtype=complex)
        for column, number in enumerate(numbers):
            if number not in known:
                renormalised[:, :, column] = np.nan + 1j * np.nan
        # And blank the frequency points that had no reference impedance to be
        # normalised *to*. The whole N x N, not one port's row and column: the
        # renormalisation above mixes every term at a given frequency, so one
        # bad Z0 spoils that point entirely. What makes the neighbouring points
        # sound is that renormalisation is per frequency - s2z and z2s invert
        # one N x N block at a time - and not that it leaves them untouched:
        # scikit-rf decides whether to renormalise at all with a whole-band
        # ``np.any(z0 != z_new)``, so one differing point moves the entire band
        # off its exact path and onto the Z-parameter one.
        if discarded:
            renormalised[list(discarded), :, :] = np.nan + 1j * np.nan

        provenance = _provenance(runs, driven, spread)
        if derived:
            provenance["symmetry"] = symmetry
            provenance["derived_columns"] = list(derived)
            # How far the derivation's own precondition was missed. Recorded
            # always, so a result cannot be trusted more than the basis it was
            # built on; the panel reports it when it is large enough to matter.
            provenance["symmetry_impedance_mismatch"] = mismatch

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

    The second question is not about the set: the runs must describe one
    structure sampled the same way, *and* each must have measured something,
    since a run that excited nothing comes back looking complete.

    Defence in depth on the first: the real guarantee is upstream, where
    ``document.sweep()`` re-excites a single translation so every column
    describes the same structure by construction. What this catches is a list
    assembled by hand from separate solves.

    Sameness is judged on the *cell count* and not on the envelope digest. Every
    run in a sweep drives a different port and so has a different envelope by
    construction, so a digest would reject precisely the input this exists to
    accept. The cell count is a weaker but honest structural invariant - two
    different geometries essentially never agree on it.
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
        # back looking complete - openEMS reports a finite port impedance
        # beside a matrix of nan, because every S-parameter it forms is 0/0. A
        # lumped port whose box lies outside the grid does this: the engine clips
        # such a box away without comment. Asked of the incident wave rather than
        # of the answer, because this is the one place the *cause* is visible.
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


def _measured_impedance(
    runs: Sequence[Any], numbers: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Each port's reference impedance per frequency, and the runs behind it.

    Averaged across the runs, not taken from the first. Taking ``runs[0]``'s
    assumes Z_ref describes the port's cross-section rather than the run - true
    for a lumped port, where openEMS sets Z_ref to the resistance, and for a
    waveguide, where it is analytic. It is false for the one kind where it
    matters: an MSLPort *measures* ``sqrt(Et*dEt / (Ht*dHt))`` from that run's
    own fields, so the N runs disagree slightly and column *j*'s volts were
    decomposed with run *j*'s value. Run 0's is then simply the wrong number for
    every column but its own, which is the whole argument - no figure is needed
    to prefer the estimate that uses all N.
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
        # Which run, not just which port. Both known causes are properties of the
        # *excitation* rather than of the port being measured, so naming only the
        # port sends the reader to the wrong object - and the second cause is
        # invisible from the port entirely.
        drivers = ", ".join(
            f"port {int(run.excited_port)}"
            for run, values in zip(runs, per_run)
            if not np.all(np.isfinite(values[:, column]))
        )
        raise ResultError(
            f"port {bad} reports a non-finite reference impedance in the "
            f"run driven by {drivers}, so no S-matrix can be normalised. "
            "A waveguide port below its cutoff does this: check "
            "the band against the mode's cutoff frequency. Or a microstrip "
            "port measured in a run that a lumped port excites, which this "
            "engine does at every frequency point - a mixed "
            "microstrip/lumped two-port cannot be assembled at all, and "
            "the way round it is to measure both ports the same way"
        )
    return measured, per_run


def _disagreement(
    per_run: np.ndarray, measured: np.ndarray, frequency: np.ndarray
) -> tuple[tuple[int, ...], float]:
    """Frequency points the runs cannot agree a reference impedance at, and the worst gap.

    How far the runs disagree is the one number that says whether averaging them
    papered over a real problem.

    Judged per frequency point, and *not* over the whole band. The disagreement
    is local by nature: a microstrip port's measured Z0 goes indeterminate only
    near a standing-wave null, and where the nulls fall depends on which end is
    driven (see :data:`IMPEDANCE_TOLERANCE`). A band-wide maximum would throw
    away a whole sweep because a resonance broke a handful of its points, most of
    which are sound.

    One run needs no special case: the mean of one is that one, the differences
    are exactly zero, and nothing is ever discarded. Which is also the honest
    limit of the check - again, :data:`IMPEDANCE_TOLERANCE`.
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


def _volts(runs: Sequence[Any], numbers: tuple[int, ...], frequency: np.ndarray) -> np.ndarray:
    """The raw wave-amplitude ratios, column *j* from the run that drove port *j*.

    Columns nobody drove are filled with **zero, not nan**; the caller blanks
    them again after the renormalisation. A nan does not stay where it is put: it
    reaches
    the inverse inside ``s2z`` and comes back as ``LinAlgError: Array must not
    contain infs or NaNs`` - minutes after the solve, from inside a library,
    naming nothing about the model. So a nan among the columns somebody *did*
    drive is refused here instead, by name and by how many points it spoiled.

    What licenses the zero is that the driven columns come out identical whatever
    sits in the undriven ones - see :func:`_wanted_reference`.
    """
    by_excitation = {int(run.excited_port): run for run in runs}
    volts = np.zeros((frequency.size, len(numbers), len(numbers)), dtype=complex)
    for column, driving in enumerate(numbers):
        run = by_excitation.get(driving)
        if run is None:
            continue
        for row, receiving in enumerate(numbers):
            volts[:, row, column] = run.s(receiving, driving)

    blank = ~np.isfinite(volts)
    if blank.any():
        row, column = np.unravel_index(int(np.argmax(blank.any(axis=0))), blank.shape[1:])
        points = int(blank[:, row, column].sum())
        raise ResultError(
            f"S{numbers[row]}{numbers[column]} is not a number at {points} "
            f"of {frequency.size} frequency points, so no S-matrix can be "
            "normalised from these runs. The run driven by port "
            f"{numbers[column]} returned no usable field there"
        )
    return volts


def _wanted_reference(
    reference: float | None | Sequence[float | None],
    numbers: tuple[int, ...],
    known: set[int],
    own: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """What each port is to be referenced to, and which ports that leaves at their own.

    One rule: a port is referenced to the number the caller asked for, and to its
    own impedance when the caller asked for nothing. Collapsed into it are a port
    the user pointed at itself, a column nobody drove, and both ports of a
    derived mirror, which share one impedance by the user's declaration.

    ``own`` rather than each port's own measurement, for the mirror's sake:
    :func:`_derive_mirror` has already moved the network into a single common
    basis, and asking for the measurements here would renormalise the two
    diagonal terms by different amounts and undo the copy it just made exact.

    **A port nobody drove keeps its own impedance whatever was asked for**, which
    makes its renormalisation the identity. That is not a convenience, it is the
    only honest option, and it is exact: with ``G = diag(g1, 0)``, column 1 of
    ``A^-1 (S - G*)(I - GS)^-1 A*`` needs only column 1 of ``S``, so the driven
    columns of an incomplete matrix are exactly referenced to what the caller
    asked for and nothing is faked. Renormalising the undriven port as well would
    move them and would genuinely need S22.

    That is the derivation, not the implementation - scikit-rf renormalises
    through Z-parameters (``z2s(s2z(...))``), reaching the same answer by another
    route, so the independence was confirmed against *it* rather than against the
    algebra, by filling the unmeasured column with zeros, with random values and
    with 1e3.
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

    Fields that describe the *structure* survive from any run; fields that
    describe one solve must not, or a two-run matrix that took four minutes
    reports whichever eleven seconds run 0 happened to take.
    """
    per_solve = (
        "wall_seconds",
        "recorded_samples",
        "tail_share",
        "max_timesteps",
        "end_criteria",
    )
    provenance = {key: value for key, value in runs[0].provenance.items() if key not in per_solve}
    for key in per_solve:
        values = {
            int(run.excited_port): run.provenance.get(key) for run in runs if key in run.provenance
        }
        if values:
            provenance[key] = values
    provenance["result_library"] = _skrf.description()
    provenance["excitations"] = sorted(driven)
    provenance["reproducible"] = all(run.reproducible for run in runs)
    # The worst disagreement anywhere in the band, including the points that were
    # discarded - it describes the *solve*, not the matrix that came out of it,
    # and which points survived is what ``discarded`` says. Not the frequencies
    # as well: they are already the indices in ``discarded`` against the
    # frequency axis, and a second copy is the one that eventually disagrees.
    provenance["impedance_spread"] = spread
    # One digest per run, keyed by the port it drove. A single field cannot
    # describe a matrix built from N envelopes, and dropping them would leave the
    # result unfalsifiable.
    provenance["envelope_digest"] = {
        int(run.excited_port): run.provenance.get("envelope_digest") for run in runs
    }
    return provenance


def _derive_mirror(
    network: Any,
    measured: np.ndarray,
    numbers: tuple[int, ...],
    driven: Sequence[int],
    symmetry: str | None,
) -> tuple[tuple[int, ...], float, np.ndarray]:
    """Fill the column a declared mirror determines, in a basis it holds in.

    Returns the columns filled, how far the two ports' measured impedances
    disagree, and what each port is referenced to on the way out - which is
    ``measured`` when nothing was derived and the common basis below when
    something was. Modifies ``network`` in place: it comes in referenced to what
    each port measured and goes out referenced to the *undriven* port's
    impedance at both ports, with the missing column filled.

    That third value is what "the port's own impedance" means to the caller: a
    mirror declares the two ports to be the same port, so under it they have one
    impedance and the gap between their two measurements is noise.

    Reciprocity does the off-diagonal (S12 = S21) and the mirror does the
    diagonal (S22 = S11). It is the mirror that is the user's claim -
    reciprocity alone leaves S22 unknown and completes nothing.

    **Both identities need the two ports at one reference.** A lumped port's
    Z_ref is the resistance the user typed and two of them agree exactly; a
    microstrip port *measures* ``sqrt(Et*dEt / (Ht*dHt))`` from its own probe
    fields, so the two ends of a genuinely symmetric line come back slightly
    different. Copy in that measured basis and the renormalisation afterwards
    moves the two diagonal terms by different amounts and undoes the copy - a
    small absolute perturbation, which hides in the passband and dominates in a
    reflection null.

    So a common basis is *made* rather than assumed. The driven port is
    renormalised to the undriven port's measured impedance, which is the one
    move the missing column cannot affect: with ``G = diag(g, 0)`` the driven
    column of ``A^-1 (S - G*)(I - GS)^-1 A*`` needs only the driven column, and
    the caller's renormalisation afterwards moves both columns together.

    The gap between the two measurements is still returned, as evidence about
    whether the *declaration* is credible. It is not an error bar on S: two
    ports of a true mirror have one Z0, and the gap is two noisy estimates of
    it rather than a real asymmetry.
    """
    if not symmetry:
        return (), 0.0, measured
    if symmetry != MIRROR:
        raise ResultError(f"unknown symmetry {symmetry!r}; this layer knows {MIRROR!r}")

    missing = [n for n in numbers if n not in set(driven)]
    # Nothing to fill, or nothing this symmetry knows how to fill. Deriving
    # nothing is the honest outcome and it is what the pre-flight warning
    # promises ("nothing will be derived from it here"); raising would throw
    # away a partial matrix that is perfectly good, minutes after the solve,
    # because of a declaration that simply does not apply to it.
    if not missing or len(numbers) != 2 or len(driven) != 1:
        return (), 0.0, measured

    (other,) = missing
    a = numbers.index(driven[0])
    b = numbers.index(other)

    scale = np.maximum(np.abs(measured[:, a]), 1e-30)
    mismatch = float(np.max(np.abs(measured[:, a] - measured[:, b]) / scale))

    # Into a common basis, and it has to be the *undriven* port's: port a is the
    # only one whose reference can be moved without the column nobody measured.
    # For a true mirror the answer does not depend on which - a symmetric
    # matrix renormalised by a common factor stays symmetric - so nothing is
    # chosen here that the physics does not already fix.
    common = measured.copy()
    common[:, a] = measured[:, b]
    network.renormalize(common, s_def="power")

    filled = np.asarray(network.s, dtype=complex)
    filled[:, b, b] = filled[:, a, a]  # mirror:      S22 = S11
    filled[:, a, b] = filled[:, b, a]  # reciprocity: S12 = S21
    network.s = filled
    return (other,), mismatch, common
