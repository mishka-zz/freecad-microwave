# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Turning what a backend wrote into the document's result object.

This module is solver-aware glue, like :mod:`.openems_mesh_preview`. It knows
the adapters' readers and the neutral document objects, which is what ``Gui/``
is for. It imports no Qt, so all of it is testable without a display and each
task panel is a few lines of wiring on top.

A study keeps one result per backend. :func:`record` files a matrix under the
backend its provenance names, and a run replaces that backend's earlier answer
and nothing else.

Palace writes a whole matrix from one run, and :func:`from_palace` turns it
into the result layer's form. What follows is openEMS, which does not.

The sweep's shape
-----------------

FDTD drives one port per run, so an N-port matrix is N solves and column *j*
comes from the run that excited port *j*. Each run gets its own directory,
named after the port it drives, such as ``<simdir>/port2/``, so every run keeps
its envelope, its digest file and its results beside each other. There is one
directory per solve in every case. The one-port case is a sweep of length one.
"""

import os
import re
from dataclasses import dataclass, replace

import numpy as np

from ..Objects.analysis import MIRROR_SYMMETRY, NO_SYMMETRY, find_analysis, members
from ..Objects.kinds import kind_of
from ..Objects.results import APART, backend, createEMSParameters, load, provenance, store
from ..Objects.results import label as result_label
from ..Results import compared, modelled
from ..Results.sparameters import (
    IMPEDANCE_STATED,
    LUMPED_RESISTANCE,
    MIRROR,
    ResultError,
    SParameters,
    mirror_filled,
)
from ..Solvers.openems import read as _read
from ..Solvers.palace import balance
from ..Solvers.palace import read as _palace_read
from ..Solvers.palace.capabilities import capabilities as _palace
from ..undo import transaction
from .described import digest, properties

#: What the matrix is referenced to, in ohms, unless a caller says otherwise.
#: 50 is the convention every microwave instrument and Touchstone file assumes.
#: It is not what the ports measured. That is kept separately, per port per
#: frequency, in ``EMSParameters.PortImpedance*``.
REFERENCE = 50.0


def directory_for(base, port: int) -> str:
    """Where the run driving ``port`` writes its envelope and results."""
    return os.path.join(str(base), f"port{int(port)}")


class NoResult(Exception):
    """Nothing to export, with the reason in the message."""


def results_of(analysis):
    """Every stored matrix in this analysis, in group order.

    One per backend that has run, because :func:`record` overwrites only the
    answer the same backend gave. More is reachable by hand, and the export has
    to know the difference.
    """
    return [member for member in members(analysis) if kind_of(member) == "EMSParameters"]


def find_results(analysis, solver: str):
    """The matrix ``solver`` last left in this analysis, or ``None``.

    Per analysis rather than per document, for the reason the mesh preview is:
    two studies over one board produce two matrices. Per backend within it,
    because one study answered by two backends is how a staircase is compared
    with a conforming mesh on the user's own drawing, and an answer that
    overwrote the other would leave nothing to compare.

    :param solver: the backend's name, as its declaration and its provenance
        both state it.
    """
    for found in results_of(analysis):
        if backend(found) == solver:
            return found
    return None


def result_in_hand(document, selection=()):
    """The stored matrix a toolbar command was aimed at, to draw or to write.

    The selection decides whenever it can, including for a result dragged out
    of its group, which has no analysis to be found through.

    Otherwise this takes the study's own matrix: nothing selected, one
    analysis, one matrix. Where that is ambiguous this raises :class:`NoResult`
    rather than taking the first in group order, which would act on the wrong
    matrix without reporting it.
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

    A location is either a results file or the directory holding one.
    ``read.read`` accepts both, so callers hand it whatever they have.

    This is separate from :func:`assemble` so that a caller can look at the
    runs before they are folded into a matrix. Checking each run against the
    envelope digest it came from is worth reporting per run rather than as one
    verdict about the set.
    """
    return [_read.read(location) for location in locations]


def declared_symmetry(analysis):
    """The study's symmetry declaration, in the result layer's vocabulary.

    ``None`` when the user has not claimed one. That is the default and the
    honest state, because a result then holds only what was measured.
    """
    if str(getattr(analysis, "Symmetry", NO_SYMMETRY)) == MIRROR_SYMMETRY:
        return MIRROR
    return None


def reference_for(analysis) -> float | list[float | None]:
    """What each port's S-parameters are reported against, in matrix order.

    ``EMPort.ReferenceImpedance`` is the "50" in "50-ohm system". It has to be
    passed to ``assemble``, whose default is 50. Leave it out and a 75-ohm
    study comes back renormalised to 50 and labelled 50, in the panel and in
    every Touchstone file.

    The entry is ``None`` for a port referenced to its own impedance, which
    ``from_runs`` reads as a request to resolve that port against what the runs
    measured. The editor hides the number in that mode, so whatever it holds is
    stale.

    The list is ordered by port number, which is how
    :meth:`SParameters.from_runs` indexes the matrix. It falls back to the
    scalar :data:`REFERENCE` when the analysis declares no ports.
    """
    from ..Objects.ports import PORT_IMPEDANCE, is_port

    # The sort key is the number alone. Two ports carrying one number is the
    # translation's refusal to make, and ordering by the pair would meet that
    # case here first, as a TypeError comparing None against a float.
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

    :meth:`SParameters.from_runs` refuses every consistency question: same
    frequencies, same ports, same grid, no port driven twice. Nothing is
    re-checked here; :meth:`SParameters.from_runs` owns those refusals.

    ``symmetry`` is the study's declaration, which fills the columns a single
    solve could not measure.
    """
    return SParameters.from_runs(runs, reference=reference, symmetry=symmetry)


def from_palace(answer, title: str = "", symmetry=None) -> SParameters:
    """The matrix a Palace run measured, in the result layer's own form.

    Palace writes every excitation of a run into one table, so there is one run
    and nothing to assemble across runs. A column no port drove holds nan, as an
    undriven column does on every backend.

    Every port is referenced to its own impedance: the table is normalised to
    a wave port's own mode and to a lumped port's resistance, and a matrix
    referenced that way does not depend on what that impedance is called. Where
    the run read a wave port's voltage, Palace states its power-voltage
    impedance; a lumped port states its resistance. That is the number both
    impedances carry, and the provenance names which of the two it is by the
    kind of port. Elsewhere Palace states none, and both are nan rather than a
    figure this workbench would compute on Palace's behalf and store as though
    Palace had said it.

    :param answer: what ``Solvers/palace/read.py`` read back - the band, the
        port each row was measured at, the port each column was driven from,
        the matrix, indexed by sample, row, then column, the power through each
        port's face indexed the same way, and whether the model dissipates.
    :param title: what the result is of, which the chart and a Touchstone
        header name it by.
    :param symmetry: the study's declaration, which fills the column a run
        driving one port of a two-port did not measure. Palace terminates a
        wave port it does not drive in that port's own mode and a lumped port in
        its resistance, which is each port's reference, so the column is the
        measured one with the two ports exchanged.

    What the matrix leaves unaccounted for from each driven port goes into the
    provenance, because the matrix cannot show it and the log scrolls away.
    """
    frequency = np.asarray(answer.frequency, dtype=float)
    ports = tuple(int(number) for number in answer.out)
    s = np.full((frequency.size, len(ports), len(ports)), np.nan + 1j * np.nan)
    for column, driving in enumerate(answer.driven):
        s[:, :, ports.index(int(driving))] = np.asarray(answer.matrix)[:, :, column]
    stated = np.full((frequency.size, len(ports)), np.nan + 1j * np.nan)
    if getattr(answer, "impedance", None) is not None:
        stated[:] = np.asarray(answer.impedance, dtype=float)
    kinds = getattr(answer, "stated", None) or {}
    named = {
        str(number): LUMPED_RESISTANCE
        if kinds.get(number) == _palace_read.RESISTANCE
        else _palace_read.POWER_VOLTAGE
        for column, number in enumerate(ports)
        if np.all(np.isfinite(stated[:, column]))
    }
    driven = sorted(int(number) for number in answer.driven)
    s, derived, mismatch = mirror_filled(s, ports, driven, stated, symmetry)
    return SParameters(
        frequency=frequency,
        s=s,
        port_numbers=ports,
        reference=stated,
        measured_impedance=stated.copy(),
        driven=tuple(driven),
        derived=derived,
        self_referenced=ports,
        provenance={
            **({IMPEDANCE_STATED: named} if named else {}),
            **(
                {
                    "symmetry": symmetry,
                    "derived_columns": list(derived),
                    **({"symmetry_impedance_mismatch": mismatch} if mismatch is not None else {}),
                }
                if derived
                else {}
            ),
            **(
                {compared.SIGN_BY_RULE: list(answer.sign_by_rule)}
                if getattr(answer, "sign_by_rule", ())
                else {}
            ),
            "title": title,
            "solver": _palace().solver,
            "excitations": driven,
            "modelled": [dict(record) for record in answer.modelled],
            "unaccounted": {
                str(shortfall.excitation): {
                    "share": shortfall.share,
                    "frequency": shortfall.frequency,
                    "port": shortfall.port,
                }
                for shortfall in balance.shortfalls(answer)
            },
        },
    )


#: What undoing a stored result is offered as.
RECORDED = "Store Results"


def record(analysis, result: SParameters):
    """Put ``result`` into the analysis's result for its backend. Returns it.

    The backend is the one ``result``'s provenance names, so the object and the
    run that filled it cannot be told apart. The object is created on that
    backend's first run, which also puts it in the tree, as the first Update
    Mesh creates the preview, and it is labelled with the backend so that two
    of them read as two answers to one question.

    It overwrites the same backend's earlier matrix and nothing else. To keep an
    earlier one, export it to a Touchstone file.
    """
    # This is one undo step, as Update Mesh is, so that Ctrl-Z after a run
    # takes the S-parameters back out rather than reversing something earlier.
    # The transaction is named for what disappears rather than for the button.
    # See ``Microwave/undo.py``.
    with transaction(analysis.Document, RECORDED):
        return put(analysis, result)


def put(analysis, result: SParameters):
    """:func:`record` without its undo step, for a caller filing something else
    beside the matrix in one.

    A result that does not say what drawing it was solved from is stamped with
    the drawing the study holds now. :func:`stamped` is how a caller that read
    the drawing when the run started says so instead.
    """
    if not result.provenance.get(compared.DRAWING):
        result = stamped(result, drawing(analysis))
    solver = str(result.provenance.get("solver") or "")
    found = find_results(analysis, solver)
    if found is None:
        found = createEMSParameters(analysis.Document)
        if solver:
            found.Label = result_label(solver)
        analysis.addObject(found)
    store(found, result)
    # Writing a property touches the object, and the touched mark reads as
    # "this is stale" on a matrix measured a second ago. The mark is purged
    # rather than recomputed, because ``execute`` has nothing to do.
    found.purgeTouched()
    return found


def beside(analysis, result: SParameters) -> list[str]:
    """What ``result`` modelled otherwise than another backend's matrix in this
    study, and how far the two matrices stand apart term by term.

    Asked as a run is filed and said in its log, and stored on neither matrix:
    a verdict written onto one goes stale the moment the other backend runs
    again, and a matrix records one solve rather than a comparison. Two matrices
    that cannot be compared get a line saying why.
    """
    solver = str(result.provenance.get("solver") or "")
    mine = result.provenance.get("modelled") or ()
    lines = []
    for other in results_of(analysis):
        by = backend(other)
        if by and by != solver:
            lines += modelled.apart(mine, solver, provenance(other).get("modelled") or (), by)
            try:
                lines += compared.compare(load(other), by, result, solver).lines()
            except (compared.Incomparable, ResultError) as refusal:
                lines.append(f"Not compared with what {by} solved: {refusal}")
    return lines


def comparison(document, selection=()):
    """The two matrices of one study from two backends, compared.

    Two results in the selection are taken as they are. Otherwise the study the
    selection names, or the document's one study, must hold exactly two.

    :raises NoResult: there are not two results to compare.
    :raises compared.Incomparable: the two are not ratios of the same waves.
    """
    chosen = [obj for obj in selection if kind_of(obj) == "EMSParameters"]
    if len(chosen) != 2:
        analysis = find_analysis(document, selection)
        chosen = results_of(analysis)
        if len(chosen) != 2:
            label = getattr(analysis, "Label", None) or analysis.Name
            held = "one set" if len(chosen) == 1 else f"{len(chosen)} sets"
            raise NoResult(
                f"{label!r} holds {held} of S-parameters, and a comparison takes two. Run "
                "the study on a second solver, or select the two to compare"
            )
    first, second = chosen
    return compared.compare(load(first), backend(first), load(second), backend(second))


def stamped(result: SParameters, drawn: str) -> SParameters:
    """``result``, recording that it was solved from the drawing ``drawn``."""
    return replace(result, provenance={**result.provenance, compared.DRAWING: drawn})


def drawing(analysis) -> str:
    """A digest of what the study's answer depends on in the drawing.

    Every material binding and every port the study holds, with each property
    this workbench gave them and each material's, each shape they name described
    by :func:`~.described.measures`, and what the mesh policy says lies beyond the
    structure. Two runs whose digests differ were solved from different
    drawings. Nothing that is a backend's own - a solver, a mesh recipe, a
    refinement region, the policy's demands on a mesh - is in it, since two
    backends are compared on exactly those. A colour is left out, since it
    changes no answer, and so is what :data:`~.described.MEASURED` names.
    """
    from ..Objects.ports import is_port

    held = sorted(
        (obj for obj in members(analysis) if kind_of(obj) == "EMMaterialBinding" or is_port(obj)),
        key=lambda obj: obj.Name,
    )
    beyond = [
        properties(obj, lambda name, obj=obj: obj.getGroupOfProperty(name) == DOMAIN)
        for obj in members(analysis)
        if kind_of(obj) == "EMMeshPolicy"
    ]
    return digest([*map(properties, held), *beyond])


#: Where the mesh policy keeps what lies beyond the structure: the medium, how
#: far it reaches and what stands on each face of the domain.
DOMAIN = "Domain"


def stored(analysis, solver: str):
    """The matrix ``solver`` last produced for this analysis, or ``None``.

    ``None`` means absent and nothing else. A damaged result raises, because
    reading it as "no result yet" would offer Run as the fix for a problem
    running does not fix.
    """
    found = find_results(analysis, solver)
    return None if found is None else load(found)


@dataclass(frozen=True)
class TouchstoneExport:
    """What exporting one result would write, and what to say before writing it.

    This is Qt-free, like everything else here, so the decision is testable and
    the command on the toolbar is wiring. Each outcome asks the user for
    something different:

    * ``refusal`` set - nothing can be written, and the message says what to
      do about it. The command shows it and stops.
    * ``caveat`` set - something can be written, but not what the user
      probably assumes. They confirm, or they do not.
    * neither - write it.
    """

    result: SParameters | None = None
    refusal: str = ""
    caveat: str = ""
    #: A filename without a suffix. ``write_touchstone`` appends ``.sNp``,
    #: taking N from the matrix's port count.
    stem: str = ""
    #: Whether the file states the reference at each point, which the command
    #: passes to ``write_touchstone`` once the user has confirmed the caveat.
    per_point: bool = False

    @property
    def suffix(self) -> str:
        """What the written file will be called, for the dialog's filter."""
        return f".s{self.result.ports}p" if self.result is not None else ""


def touchstone_export(holder) -> TouchstoneExport:
    """Decide what a Touchstone export of ``holder`` would do.

    The ways a matrix can fall short have different fixes:

    * **Undriven columns.** No file is written. A ``.sNp`` has a column for
      every term and no way to mark one as invented. The fix is another solve,
      or a symmetry.
    * **A complex reference.** No file is written. Touchstone states a
      reference as a real impedance, and the wave definitions differ at a
      complex one. The fix is in the ports.
    * **A reference with no number.** No file is written. The solver stated no
      impedance for the port, so there is none to state or to renormalise to,
      and the result stays in the document.
    * **A real reference that is not one number.** A file stating each port's
      reference at each point, once the user has said so. A guide referenced
      to its own mode is the ordinary case, and a reader that ignores that
      form reads the file with no reference.
    * **Frequency points that hold no numbers.** A file of the points that do
      hold numbers, once the user has said so, with the reason in its own
      header.

    This asks ahead so the command does not open a file dialog it will have to
    abandon. :meth:`SParameters.write_touchstone` refuses each of them on its
    own, and would still refuse if this were wrong.
    """
    try:
        result = load(holder)
    except ResultError as error:
        return TouchstoneExport(refusal=str(error))

    # First, because nothing the user can do changes it, where the partial
    # matrix below advises another solve.
    if not np.isfinite(np.asarray(result.reference)).all():
        return TouchstoneExport(
            refusal=(
                f"{_label(holder)} is referenced to each port's own impedance, and "
                "the solver that produced it states no number for that impedance. "
                "A Touchstone file states a real reference impedance, once or at "
                "each frequency point, and there is none here to state or to "
                "renormalise to. Keep the result in the document, where what it was "
                "measured against is preserved."
            )
        )

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

    kept = int(result.kept.sum())
    causes = " ".join(f"{cause}." for cause in result.blank_causes())
    if not kept:
        # from_runs refuses a sweep no point of which holds numbers, so this
        # arrives from a document written by something else. Without this
        # branch the caveat offers to write zero points, and accepting that
        # reaches an IndexError inside scikit-rf.
        return TouchstoneExport(
            refusal=(
                f"{_label(holder)} holds no usable numbers: none of its "
                f"{result.frequency.size} frequency points holds a number. {causes} "
                "There is nothing to write."
            )
        )
    said = []
    question = "Write it?"
    if kept < result.frequency.size:
        said.append(
            f"{result.frequency.size - kept} of {result.frequency.size} frequency "
            f"points hold no numbers ({result.blank_span()}). {causes} The file will "
            "say which points are missing, and why, in its header."
        )
        question = f"Write the other {kept}?"
        result = result.usable()

    # Asked of the points the file will hold.
    per_point = not result.one_reference
    if per_point and not result.real_reference:
        return TouchstoneExport(
            refusal=(
                f"{_label(holder)} is referenced to "
                f"{result.reference_description()}, and a Touchstone file states a "
                "reference as a real impedance, once or at each frequency point. Set "
                "every port's ReferencedTo to Fixed impedance with the same "
                "ReferenceImpedance and run again, or keep the result in the "
                "document, where what it was measured against is preserved."
            )
        )
    if per_point:
        spread = float(np.ptp(np.asarray(result.reference).real))
        meanings = set((result.provenance.get(IMPEDANCE_STATED) or {}).values())
        guides = bool(meanings - {LUMPED_RESISTANCE})
        on_palace = backend(holder) == _palace().solver
        said.append(
            f"{_label(holder)} is referenced to {result.reference_description()}, "
            f"which spans {spread:.3g} ohm over its ports and band, and a Touchstone "
            "option line states one impedance. The file will state each port's at "
            "each frequency point, in the HFSS form scikit-rf writes and reads back. "
            "A reader that ignores that form has only the option line, which states "
            "no number: it may take every port at 50 ohm, as scikit-rf does with "
            "such a line, or refuse the file. "
            + (
                "The header names which impedance each guide port states: openEMS "
                "states a guide's wave impedance and Palace its power-voltage impedance, "
                "so their files of one guide do not cascade into each other. "
                if guides
                else ""
            )
            + (
                "Palace reports a wave port against its own mode alone, so this is the "
                "file a Palace study writes."
                if on_palace and guides
                else "Palace reports a lumped port against its own Resistance. For a file "
                "every reader takes alike, give every lumped port the same Resistance and "
                "run again."
                if on_palace
                else "For a file every reader takes alike, set every port's ReferencedTo "
                "to Fixed impedance with the same ReferenceImpedance and run again."
            )
        )

    caveat = "\n\n".join([*said, question]) if said else ""
    return TouchstoneExport(result=result, caveat=caveat, stem=_stem(holder), per_point=per_point)


def _label(obj) -> str:
    return str(getattr(obj, "Label", None) or getattr(obj, "Name", "the result"))


def _stem(holder) -> str:
    """A default filename: the document and the result, both as the user named them.

    The name is sanitised down to word characters and the hyphen, because a
    FreeCAD label is free text. ``"50 Ω line / rev B"`` is an ordinary one. The
    dot goes too. ``write_touchstone`` appends the suffix itself, and a label
    of ``".."`` would otherwise name the directory above.
    """
    document = getattr(getattr(holder, "Document", None), "Name", "")
    # Not what the mark on the label says about the mesh beside it, which is
    # not the matrix's.
    named = _label(holder)
    bare = named.rstrip("0123456789")
    if bare.endswith(APART):
        named = bare.removesuffix(APART)
    parts = [part for part in (document, named) if part]
    return re.sub(r"[^\w-]+", "_", "-".join(parts)).strip("_-") or "sparameters"


def impedance_lines(result: SParameters):
    """One line per port: the impedance the solver measured, at band centre.

    The line quotes one frequency and names it, because a microstrip's Z0 is
    dispersive and the whole curve is in the result object anyway.

    The frequency is the centre of the points whose impedance is sound.
    ``measured_impedance`` is kept at a discarded frequency, but it is exactly
    the number that went wrong there, and a half-wave resonance puts its null
    at band centre by construction. At an unpowered one it is a port's that
    carries no power.
    """
    wrong = set(result.discarded) | set(result.unpowered)
    kept = [i for i in range(result.frequency.size) if i not in wrong]
    if not kept:
        return
    middle = kept[len(kept) // 2]
    hz = float(result.frequency[middle])
    for number in result.port_numbers:
        z = complex(result.impedance(number)[middle])
        sign = "+" if z.imag >= 0 else "-"
        yield (f"Port {number}: {z.real:.2f} {sign} {abs(z.imag):.2f}j ohm at {hz / 1e9:.4g} GHz")
