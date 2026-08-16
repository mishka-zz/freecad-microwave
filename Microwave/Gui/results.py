# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Turning a sweep's output into the document's result object.

Solver-aware glue, exactly like :mod:`.mesh_preview`: it knows both the openEMS
adapter and the neutral document objects, which is what ``Gui/`` is for. It
imports **no Qt**, so all of it is testable without a display and the task panel
is a few lines of wiring on top.

The sweep's shape
-----------------

FDTD drives one port per run, so an N-port matrix is N solves and column *j*
comes from the run that excited port *j*. Each gets its own directory, named
after the port it drives - ``<simdir>/port2/`` - so every run keeps its
envelope, its digest file and its results beside each other. One directory per
solve, always: the one-port case is a sweep of length one.
"""

import os
import re
from dataclasses import dataclass

from ..Objects.analysis import MIRROR_SYMMETRY, NO_SYMMETRY, find_analysis, members
from ..Objects.kinds import kind_of
from ..Objects.results import createEMSParameters, load, store
from ..Results.sparameters import MIRROR, ResultError, SParameters
from ..Solvers.openems import read as _read
from ..undo import transaction

#: What the matrix is referenced to, in ohms, unless a caller says otherwise.
#: 50 is the convention every microwave instrument and Touchstone file assumes,
#: and it is *not* what the ports measured - that is kept separately, per port
#: per frequency, in ``EMSParameters.PortImpedance*``.
REFERENCE = 50.0


def directory_for(base, port: int) -> str:
    """Where the run driving ``port`` writes its envelope and results."""
    return os.path.join(str(base), f"port{int(port)}")


class NoResult(Exception):
    """Nothing to export, with the reason in the message."""


def results_of(analysis):
    """Every stored matrix in this analysis, in group order.

    Usually one - :func:`record` overwrites. Several is reachable by hand and
    will be reachable by feature the day a result can be kept, and the export
    has to know the difference.
    """
    return [member for member in members(analysis) if kind_of(member) == "EMSParameters"]


def find_results(analysis):
    """The analysis's result object, or ``None``. One per analysis.

    Per analysis and not per document, for the same reason the mesh preview is:
    two studies over one board produce two matrices, and a document-wide lookup
    would overwrite the first study's answer with the second's.

    The *first*, deliberately: this answers where a study's matrix lives for
    writing and reading back, and both want the same one every time. Choosing
    which of several to hand a *user* is :func:`result_in_hand`'s job.
    """
    found = results_of(analysis)
    return found[0] if found else None


def result_in_hand(document, selection=()):
    """The stored matrix a toolbar command was aimed at - to draw, or to write.

    The selection decides whenever it can, including for a result dragged out
    of its group, which has no analysis to be found through.

    Otherwise the study's own - nothing selected, one analysis, one matrix.
    Where that is ambiguous this raises :class:`NoResult` rather than taking
    the first in group order, which would act on the wrong matrix in silence.
    """
    for chosen in selection:
        if kind_of(chosen) == "EMSParameters":
            return chosen

    analysis = find_analysis(document, selection)
    found = results_of(analysis)
    label = getattr(analysis, "Label", None) or analysis.Name
    if not found:
        raise NoResult(f"{label!r} has no S-parameters yet. Run the analysis first")
    if len(found) > 1:
        names = ", ".join(getattr(obj, "Label", None) or obj.Name for obj in found)
        raise NoResult(
            f"{label!r} holds {len(found)} sets of S-parameters ({names}). "
            "Select the one you want first"
        )
    return found[0]


def load_runs(locations):
    """One adapter ``Results`` per location, in the order given.

    A location is either a results file or the directory holding one -
    ``read.read`` accepts both, so callers hand it whatever they have.

    Separate from :func:`assemble` so a caller can look at the runs before they
    are folded into a matrix: checking each against the envelope digest it came
    from is worth reporting per run rather than as one verdict about the set.
    """
    return [_read.read(location) for location in locations]


def declared_symmetry(analysis):
    """The study's symmetry declaration, in the result layer's vocabulary.

    ``None`` when the user has not claimed one, which is the default and the
    honest state: a result then holds only what was measured.
    """
    if str(getattr(analysis, "Symmetry", NO_SYMMETRY)) == MIRROR_SYMMETRY:
        return MIRROR
    return None


def reference_for(analysis) -> float | list[float | None]:
    """What each port's S-parameters are reported against, in matrix order.

    ``EMPort.ReferenceImpedance`` is the "50" in "50-ohm system". It has to be
    passed to ``assemble``, whose default is 50: leave it out and a 75-ohm
    study comes back renormalised to 50 and *labelled* 50, in the panel and in
    every Touchstone file.

    ``None`` for a port referenced to its own impedance, which ``from_runs``
    reads as "resolve this one against what the runs measured". The editor
    hides the number in that mode, so whatever it holds is stale.

    Ordered by port number, which is how :meth:`SParameters.from_runs` indexes
    the matrix. Falls back to the scalar :data:`REFERENCE` when the analysis
    declares no ports.
    """
    from ..Objects.ports import PORT_IMPEDANCE, is_port

    # Sorted by number alone. Two ports carrying one number is the
    # translation's refusal to make, and ordering by the pair would meet it
    # here first as a TypeError comparing None against a float.
    declared = sorted(
        (
            (
                int(obj.Number),
                None if str(obj.ReferencedTo) == PORT_IMPEDANCE else float(obj.ReferenceImpedance),
            )
            for obj in members(analysis)
            if is_port(obj)
        ),
        key=lambda declaration: declaration[0],
    )
    return [impedance for _, impedance in declared] if declared else REFERENCE


def assemble(runs, reference: float | list[float | None] = REFERENCE, symmetry=None) -> SParameters:
    """Build the matrix a set of runs describes.

    Every consistency question - same frequencies, same ports, same grid, no
    port driven twice - is :meth:`SParameters.from_runs`'s to refuse. Nothing
    is re-checked here: a second opinion eventually disagrees with the first.

    ``symmetry`` is the study's declaration, which fills the columns a single
    solve could not measure.
    """
    return SParameters.from_runs(runs, reference=reference, symmetry=symmetry)


def record(analysis, result: SParameters):
    """Put ``result`` into the analysis's result object. Returns the object.

    Creates it on the first run, so the first solve is also what puts it in the
    tree - as the first Update Mesh is what creates the preview.

    It **overwrites**: a study holds the answer to the question it currently
    asks. Exporting a Touchstone file is how a matrix is kept.
    """
    # One undo step, as Update Mesh is, so that Ctrl-Z after a run takes the
    # S-parameters back out rather than reversing something earlier. Named for
    # what disappears, not for the button. See ``Microwave/undo.py``.
    with transaction(analysis.Document, "Store Results"):
        found = find_results(analysis)
        if found is None:
            found = createEMSParameters(analysis.Document)
            analysis.addObject(found)
        store(found, result)
        # Writing a property touches the object, and the touched mark reads as
        # "this is stale" on a matrix measured a second ago. Purged rather than
        # recomputed: ``execute`` has nothing to do.
        found.purgeTouched()
    return found


def stored(analysis):
    """The matrix this analysis last produced, or ``None`` if it has none.

    ``None`` means *absent* and nothing else. A damaged result raises, because
    reading it as "no result yet" would offer Run as the fix for a problem
    running does not fix.
    """
    found = find_results(analysis)
    if found is None or not list(found.PortNumbers):
        return None
    return load(found)


@dataclass(frozen=True)
class TouchstoneExport:
    """What exporting one result would write, and what to say before writing it.

    Qt-free, like everything else here, so the decision is testable and the
    command on the toolbar is wiring. The outcomes are different conversations
    with the user:

    * ``refusal`` set - nothing can be written, and the message says what to
      do about it. The command shows it and stops.
    * ``caveat`` set - something can be written, but not what the user
      probably assumes. They confirm, or they do not.
    * neither - write it.
    """

    result: SParameters | None = None
    refusal: str = ""
    caveat: str = ""
    #: A filename without a suffix. ``write_touchstone`` appends ``.sNp`` from
    #: the port count, which is the only thing that knows how many there are.
    stem: str = ""

    @property
    def suffix(self) -> str:
        """What the written file will be called, for the dialog's filter."""
        return f".s{self.result.ports}p" if self.result is not None else ""


def touchstone_export(holder) -> TouchstoneExport:
    """Decide what a Touchstone export of ``holder`` would do.

    The ways a matrix can fall short have different fixes:

    * **Undriven columns.** No file: a ``.sNp`` has a column for every term
      and no way to mark one as invented. Another solve, or a symmetry.
    * **A reference the format cannot hold.** No file: Touchstone states one
      real impedance per port per frequency, and a port reported against its
      own has none to state. The fix is in the ports.
    * **Frequency points that hold no numbers.** A file of the points that do,
      once the user has said so, with the reason in its own header.

    This asks ahead so the command does not open a file dialog it will have to
    abandon. :meth:`SParameters.write_touchstone` refuses all three on its own,
    and would still refuse if this were wrong.
    """
    try:
        result = load(holder)
    except ResultError as error:
        return TouchstoneExport(refusal=str(error))

    if result.unmeasured:
        return TouchstoneExport(
            refusal=(
                f"{_label(holder)} holds a partial matrix: port(s) "
                f"{list(result.unmeasured)} were never driven, so "
                f"{len(result.unmeasured)} of its {result.ports} columns are "
                "not measured. A Touchstone file has a column for every term "
                "and no way to mark one as invented, so there is nothing "
                "honest to write. Mark those ports as excitation sources and "
                "run again, or declare the study symmetric."
            )
        )

    if not result.one_reference:
        return TouchstoneExport(
            refusal=(
                f"{_label(holder)} is referenced to "
                f"{result.reference_description()}, and a Touchstone file "
                "states one real reference impedance for every port at every "
                "frequency. Give every port the same fixed ReferenceImpedance "
                "and run again, or keep the result in the document, where what "
                "it was measured against is preserved."
            )
        )

    caveat = ""
    if result.discarded and len(result.discarded) >= result.frequency.size:
        # Every point. from_runs refuses this outright, so it arrives only from
        # a document written by something else - but without it the caveat
        # offers to write zero points, and saying yes reaches an IndexError
        # inside scikit-rf.
        return TouchstoneExport(
            refusal=(
                f"{_label(holder)} holds no usable numbers: all "
                f"{result.frequency.size} of its frequency points were "
                "discarded because the solves disagreed there about port "
                "impedance. There is nothing to write. Check that each port's "
                "measurement plane sits on clean transmission line, and run "
                "again."
            )
        )
    if result.discarded:
        caveat = (
            f"{len(result.discarded)} of {result.frequency.size} frequency "
            f"points hold no numbers ({result.blank_span()}) - the solves "
            "disagreed there about port impedance.\n\n"
            f"Write the other {result.frequency.size - len(result.discarded)}? "
            "The file will say which points are missing, and why, in its header."
        )
        result = result.usable()

    return TouchstoneExport(result=result, caveat=caveat, stem=_stem(holder))


def _label(obj) -> str:
    return str(getattr(obj, "Label", None) or getattr(obj, "Name", "the result"))


def _stem(holder) -> str:
    """A default filename: the document and the result, both as the user named them.

    Sanitised down to word characters and the hyphen, because a FreeCAD label
    is free text - ``"50 Ω line / rev B"`` is an ordinary one. The dot goes
    too: ``write_touchstone`` appends the suffix itself, and a label of ``".."``
    would otherwise name the directory above.
    """
    document = getattr(getattr(holder, "Document", None), "Name", "")
    parts = [part for part in (document, _label(holder)) if part]
    return re.sub(r"[^\w-]+", "_", "-".join(parts)).strip("_-") or "sparameters"


def impedance_lines(result: SParameters):
    """One line per port: the impedance the solver measured, at band centre.

    One frequency, named, because a microstrip's Z0 is dispersive and the whole
    curve is in the result object anyway.

    The centre of the points that *hold* numbers. ``measured_impedance`` is
    kept at a discarded frequency, but it is exactly the number that went
    wrong, and a half-wave resonance puts its null at band centre by
    construction.
    """
    kept = [i for i in range(result.frequency.size) if i not in set(result.discarded)]
    if not kept:
        return
    middle = kept[len(kept) // 2]
    hz = float(result.frequency[middle])
    for number in result.port_numbers:
        z = complex(result.impedance(number)[middle])
        sign = "+" if z.imag >= 0 else "-"
        yield (f"Port {number}: {z.real:.2f} {sign} {abs(z.imag):.2f}j ohm at {hz / 1e9:.4g} GHz")
