# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What meshing a drawing spent, counted in quantities the code decides.

A demand and its delivery are different claims, and what it took to arrive at
the delivery is a third. The first two are the grid; this is the run that laid
it.

Every count here is a property of the code rather than of the machine it ran
on. Seconds and bytes are the machine's, they move with the load and the model
of processor, and a budget written in them is a runtime written down - which
this project keeps nowhere, because nothing re-derives it. A comparison the
pruning scan makes is the same number on every machine and on every run, so it
can be asserted.

The tally is carried down rather than handed back up, because it is filled
deep in each phase and the values on the way up are already the answer.
:func:`~.document.mesh` fills one and the plan it returns carries it, so a
reader of a finished grid has what it cost without asking for it. The envelope
route builds no plan, so nothing carries the laying half up from it: provenance
is not solver input, and what a grid cost is provenance. The measuring half is
still counted, on the translation itself - see
:attr:`~.document._Translated.spend`.
 The argument is optional throughout for a caller that
reaches one phase on its own and has nothing to add it to.

The point of counting at all is that a phase can be right and unaffordable, and
the suite has no other way to notice. Each count is paired with what the phase
undertakes to keep it under, and that pairing is the assertion - not the count
on its own, which is a figure nobody can fail.

:attr:`Spend.refused` is the exception and is not a cost. Everything above is
what the measurement read; that field is what it could not read, and its
expected value is nothing at all. It rides on this tally rather than on one of
its own. Both are filled while the lengths are read, and both reach a reader of
a finished grid on the plan; a carrier of its own would be a second optional
argument down the same functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Refused", "Spend"]


@dataclass
class Refused:
    """Stations put to the CAD kernel, by the demand they were asked for.

    A station is one question a measurement puts to the CAD kernel at one place.
    The kernel can decline - a pole, a seam, a degenerate edge - and the site
    that asked is right to carry on, because one station is not the face. What
    it cannot do on its own is say so afterwards, and a face whose every station
    was declined raises nothing and looks exactly like a face with nothing to
    ask for.

    Keyed by the source the demand would have carried, because a count on its
    own says some face somewhere could not be asked and the reader's next act is
    to open one. The chord aside, the strings are the ones the features
    themselves are labelled with, so a row here and a row about a delivered
    demand name the same thing. The chord is walked for two callers that label
    their demands differently, so it names the walk instead.
    """

    #: Per source: stations offered, and how many of those were declined.
    stations: dict[str, tuple[int, int]] = field(default_factory=dict)

    def answered(self, source: str) -> None:
        """One more station at ``source``, and the kernel answered it."""
        offered, declined = self.stations.get(source, (0, 0))
        self.stations[source] = (offered + 1, declined)

    def declined(self, source: str) -> None:
        """One more station at ``source``, and the kernel would not answer."""
        offered, declined = self.stations.get(source, (0, 0))
        self.stations[source] = (offered + 1, declined + 1)

    @property
    def lost(self) -> tuple[tuple[str, int], ...]:
        """Sources whose every station was declined, and how many that was.

        Nothing was measured there. The grid is sized by whatever else reaches
        it, and without this the run says the same thing it says for a face that
        genuinely asked for nothing.

        Every entry arrives through one of the two methods above and each counts
        a station, so there is no source here that was never asked and nothing
        to guard against one.
        """
        return tuple(
            (source, offered)
            for source, (offered, declined) in self.stations.items()
            if declined == offered
        )

    @property
    def short(self) -> tuple[tuple[str, int, int], ...]:
        """Sources read at some stations and declined at others, answered of offered.

        The measurement stands: what those stations raised is the same demand it
        would have raised anyway, over fewer places. It is stated because the
        kernel declining at all is an exception, and because the reader deciding
        whether a face is followed closely enough needs the count it was actually
        followed at.
        """
        return tuple(
            (source, offered - declined, offered)
            for source, (offered, declined) in self.stations.items()
            if declined and declined != offered
        )


@dataclass
class Spend:
    """A tally, filled in as the work is done.

    Mutable and shared, so one of these covers a whole mesh however many
    bodies, axes and passes went into it. Nothing here is reset: a caller
    wanting two runs apart uses two of these.
    """

    #: Lengths the measurement read off the drawing, counted before the exact
    #: repeats among them are removed. Nothing else is dropped there, so this is
    #: what the drawing carries rather than what binds, and it is the
    #: denominator of everything the pruning scan spends.
    raised: int = 0

    #: Demands that survived the pruning scan, added up over the scans. One
    #: length is up to three demands, one per axis, and each axis prunes its
    #: own - so this is not a count of lengths and is not comparable with
    #: :attr:`raised` except as a ratio.
    kept: int = 0

    #: Distinct sizes each pruning scan had to compare against, added up over
    #: the scans. Its second dimension: demands all asking one size escape the
    #: scan altogether, and a machined part is a spread of them. A sum of
    #: distinct counts and not a distinct count, which is the only form
    #: available to a tally that never looks back at what it added.
    sizes: int = 0

    #: Demands the pruning scan compared, over every call. Each is one demand
    #: measured against one kept demand finer than it, on one axis. So this is
    #: the scan's own work and not the size of what it was given, and it is
    #: paid once per axis and again for every pass that lays a grid.
    compared: int = 0

    #: Stations the gap walk was given. The walk samples one body's surface and
    #: marches at the neighbour from each station, so this is what
    #: :attr:`probed` is read against. A station the kernel will not answer for
    #: is not one of these, having never reached the walk. One whose face
    #: answers a normal of no length is, and it costs nothing, the walk having
    #: no direction to march in.
    #:
    #: :attr:`Refused.stations` records the same station, keyed by the source it
    #: was asked for. That field is a diagnostic rather than a cost and its key
    #: is a message; a denominator is neither, so the count is kept here as
    #: well.
    walked: int = 0

    #: Points the gap walk asked the CAD kernel about. It marches outward from a
    #: station until a point lands in the neighbour and then halves, so this is
    #: what the walk spends on containment - the dearest thing counted here, and
    #: the one a drawing carrying no gap does not pay at all. A point is charged
    #: once however many solids answer it, and what the walk spends placing the
    #: station itself is not here.
    #:
    #: :attr:`walked` is its denominator: the two divided are what one station
    #: cost, where this count alone says only that a drawing had a neighbour.
    probed: int = 0

    #: Rays cast at a triangulation. A sample can cost several: the cut is
    #: opened to the whole body where it came back empty, the run is looked at
    #: backwards where the sample stands inside the material, it is read again
    #: where the first cast fell short of the crossing that ends it, and the
    #: other way off the surface is asked where the first way found nothing. So
    #: this counts the casts and not the samples, which is what the triangles
    #: below divide by.
    cast: int = 0

    #: Triangles those rays were tested against. The index exists to keep this
    #: far below the surface, so it is the number that says whether the index
    #: is doing its job.
    tested: int = 0

    #: Triangles those same rays would have been tested against with no index
    #: at all, which is the surface once per cast. The denominator of
    #: :attr:`tested`: what the index saves is the two divided, and neither
    #: number says it alone.
    reachable: int = 0

    #: Settling passes taken, added up over every call. A grid settles every
    #: axis, the absorber's own pitch settles each absorbing axis before that,
    #: and every pass of the absorber loop settles them all again - so this is
    #: the whole of what a mesh spent settling and not one axis' count. A pass
    #: carries a published seam size one gap further, and an axis that reaches
    #: its own budget is refused rather than answered.
    settling: int = 0

    #: What the kernel would not answer for, which is the one field here that is
    #: not a cost. See the module docstring.
    refused: Refused = field(default_factory=Refused)

    #: Passes of the absorber loop taken. Each lays a whole grid and every
    #: pass but the last throws it away, so this multiplies most of what a mesh
    #: costs. A model whose interior grades into the block it was
    #: given takes one; one the block has to be laid again for takes another
    #: apiece; and a model that never reconciles is refused at the loop's own
    #: limit rather than handed its closest pass.
    absorbing: int = 0
