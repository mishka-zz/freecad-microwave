# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The acceptance gates, run together and reported with what each one claims.

``pytest -m slow -s | grep GATE`` already prints every figure this workbench
has measured. What it does not print is which of them are *worth* something: the
lines arrive in whatever order the files ran, and nothing beside a number says
whether it was scored against something exact or against a formula with an
error bar of its own - nor whether the interval it was held to was computed from
the run or written down by hand. Those two distinctions are the whole of what a
reader wants and neither is in the output.

So this runs the same gates and puts each figure under the claim it answers.

**Nothing here is a figure.** The table below is the standing part - which gate,
what it is scored against, which activity it belongs to, what decides it, and
what it isolates that the others cannot - and every number in the report comes
out of the run that produced it. A report with the numbers written into it would
be a document nothing re-measures, and it would go on reading as current long
after the meshes moved.

**One file at a time**, because these solve. Running the whole selection at once
puts several openEMS processes on one machine and each of them wants the memory
of a grid; the slowest file here is an order longer than the quickest, and how
long each took is reported rather than remembered.

Run it as ``python3 -m tests.gate_report``, under the interpreter that owns the
openEMS bindings. What in a gate needs no engine passes without one, and
everything that would solve skips itself, so a machine that cannot solve reports
that it measured nothing rather than that it failed.

It exits on what it found, so that the page does not have to be read to know
whether it is evidence. ``0`` where every gate solved and passed; ``3`` where
none of them measured anything, which is a fault in the machine rather than in
the workbench; ``2`` where the command line named a gate that is not here; ``1``
for everything else - a gate that failed, one the suite refused, one that passed
without printing a figure, one that ran in part, one that never reported.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: Which of the two activities a gate is doing, which is the distinction the raw
#: output cannot make and the one that decides what its figure means.
#:
#: The split is the standard one. **Verification** asks whether the equations are
#: being solved right: the reference is exact, so a difference is the solver's
#: and the response to a bad one is a finer mesh. **Validation** asks whether they
#: are the right equations: the reference is a fit or a fabricated board, so a
#: difference is the solver's error and the reference's added together, and past
#: the reference's own accuracy a finer mesh buys nothing anybody can see.
#: **Identity** is neither - reciprocity, passivity, or one problem solved two
#: ways, which need no reference at all.
#:
#: The practical consequence is that only a verification gate supports an
#: accuracy claim, and only a validation gate can be passed by having a bad
#: reference.
VERIFICATION = "verification - the reference is exact"
VALIDATION = "validation - the reference has an error bar of its own"
IDENTITY = "an identity - there is no reference"

#: How a gate decides, named so that a reader can tell a computed interval from a
#: figure somebody chose.
#:
#: ``STUDIED`` is the strong form: several grids, an interval computed from them
#: by the procedure in :mod:`tests.convergence`, and the reference required to be
#: inside it. It gets *harder* to pass as the study improves, which is what a
#: written-down percentage can never do. ``EXTRAPOLATED`` runs the same study but
#: scores where refinement is *heading* rather than where one mesh landed: the
#: limit against the closed form, to a bound the file states, and the observed
#: order against what the mechanism predicts *where the sequence can carry one*.
#: A limit carries a band of its own, and it is not the one computed for the
#: finest grid: those are different quantities about different numbers, and
#: which of them is the wider is a property of the sequence rather than a rule.
#: An extrapolated gate here does not score a limit *against* that band. The
#: cavity bounds its leftover and prints the band beside it, saying on its own
#: line why the two are not folded together; the stripline holds two limits to
#: each other and bounds their disagreement by a band computed for that
#: difference, one grid dropped from both sequences at once.
#: Whether it can is a property of the drawing rather than of the procedure - a
#: boundary the grid samples answers to where the grid fell as well as to how big
#: its cells are - so each file says which of the two it asserts. That says more
#: about the sequence than ``DECLARED`` and less about the answer than
#: ``STUDIED``. ``DECLARED`` is a
#: bound argued from the mechanism and written into the file - still reasoned, but
#: not re-derived by the run, and used where a gate has only one mesh to its name.
#: ``COMPOSED`` is the validation form: the three uncertainties added in
#: quadrature - which is the strongest thing a validation gate can do, and is
#: still only as sharp as its reference. Where every term can be established the
#: error is held against the total. Where one cannot, the gate names it and the
#: total becomes a lower bound: an error inside a lower bound is inside the true
#: uncertainty too, so agreement still means something, while a disagreement
#: establishes nothing and is reported rather than failed. Which of the two a
#: gate is doing is on its own output. ``GRADED`` is for
#: a gate whose answer is a *curve* rather than a number: the whole sweep is
#: scored against the whole reference by the method the CEM standard defines, and
#: the bar is one of that standard's six words rather than a figure anybody here
#: chose.
STUDIED = "a refinement study, and the interval it computes"
EXTRAPOLATED = "a refinement study, and the limit it reaches"
DECLARED = "a bound argued from the mechanism, on a single mesh"
COMPOSED = "the validation uncertainty, composed from the terms that can be established"
GRADED = "the whole curve, on the published six-category scale"

#: Where the acceptance files live, relative to the repository root.
TESTS = Path(__file__).resolve().parent

#: What a gate prints when it has measured something.
MARKER = "GATE "

#: What pytest writes for one test under ``-q``, with no newline after it - so
#: whatever is printed next lands on that same line, and a figure arrives with a
#: run of these in front of it. A print opening with a newline of its own starts
#: a clean line and carries none, so the reader takes them off rather than the
#: gates putting newlines on.
PROGRESS = ".sFExX"

#: What pytest counted, read off its own summary line. ``errors`` and ``error``
#: are the same outcome spelled for one or several.
#:
#: ``xfailed`` and ``xpassed`` carry the names of two of these counts and are the
#: opposite outcome. What keeps them out is the space: a count and its word are
#: separated by one, and ``1 xfailed`` puts an ``x`` where the pattern wants the
#: word to start. The trailing boundary guards the other end only.
COUNT = re.compile(r"\b(\d+) (passed|failed|errors?|skipped)\b")

#: What one gate's run is worth, which is the distinction the page and the exit
#: status both turn on.
#:
#: Only :data:`SOLVED` is evidence. The rest are named apart because they call
#: for different things: a :data:`FAILED` gate is a defect in the workbench,
#: :data:`MEASURED_NOTHING` is a defect in the machine the run was made on, and
#: the rest are the run itself having gone wrong in a way that leaves the page
#: looking finished.
#:
#: :data:`MEASURED_NOTHING` is asked as no figure and something skipped, rather
#: than as every test having skipped. A gate holds tests that need no engine and
#: they pass without one, so a machine that cannot solve reports a run of passes
#: beside its skips, and a verdict keyed on the skips alone would call that a
#: partial run and lay it at the workbench's door.
#:
#: Every gate here prints at least one figure, so :data:`SILENT` - a file that
#: passed and printed nothing - is a gate that stopped measuring while still
#: reading as coverage.
SOLVED = "solved"
FAILED = "failed"
REFUSED = "refused"
MEASURED_NOTHING = "measured nothing"
SILENT = "printed nothing"
PART_RAN = "part of it did not run"
NOTHING_RAN = "nothing ran"

#: What each verdict but :data:`SOLVED` puts under the gate's own heading, so
#: that a section is never read by what it does not say.
#:
#: A gate prints its figure and asserts afterwards, so a failed run has figures,
#: and they have to be labelled as what they are before the page quotes them
#: under a claim.
SAID = {
    FAILED: (
        "**This gate failed.** What is quoted below is what it printed on its "
        "way to failing. None of it is a measurement, and the claim above is "
        "not evidenced by this run."
    ),
    MEASURED_NOTHING: (
        "**This gate measured nothing.** What in it needs no engine passed; "
        "everything that would have solved skipped itself, and what each one "
        "wanted is on its own skip."
    ),
    REFUSED: (
        "**Every assertion passed and pytest still exited non-zero**, so "
        "something in the suite refused this run rather than any gate failing. "
        "The figures below stand; the run does not."
    ),
    SILENT: (
        "**This gate passed and printed nothing**, and every gate here prints. "
        "It has stopped measuring while still reading as coverage."
    ),
    PART_RAN: (
        "**Part of this gate did not run.** Whatever is quoted below is a "
        "subset of what the gate claims, and which subset is not on this page."
    ),
    NOTHING_RAN: (
        "**This run reported no outcome at all**, so it did not finish. Nothing here is a result."
    ),
}


@dataclass(frozen=True)
class Gate:
    """One acceptance file, and what a reader needs to judge its figures by."""

    #: The part of ``test_acceptance_<name>.py`` that names it.
    name: str
    #: What the answer is compared against.
    reference: str
    #: One of :data:`VERIFICATION`, :data:`VALIDATION`, :data:`IDENTITY`.
    activity: str
    #: One of :data:`STUDIED`, :data:`EXTRAPOLATED`, :data:`DECLARED`,
    #: :data:`COMPOSED`, :data:`GRADED`.
    criterion: str
    #: What this gate can say that no other one here can.
    isolates: str

    @property
    def path(self) -> Path:
        return TESTS / f"test_acceptance_{self.name}.py"


#: Every acceptance gate, in the order a reader should meet them: the exact
#: references first, then the identities, then the two scored against a
#: reference that carries an error of its own.
#:
#: ``test_gate_report`` holds this against the files on disk in both
#: directions, so a gate cannot be added without appearing here and a row
#: cannot outlive the file it describes.
GATES = (
    Gate(
        name="openems_waveguide",
        reference="exact - a rectangular guide's phase constant from its own dimensions",
        activity=VERIFICATION,
        criterion=DECLARED,
        isolates=(
            "a propagating mode rather than a static cross-section, and the same "
            "guide drawn on two axes, which catches a closed form computed from "
            "the two numbers the solver got wrong"
        ),
    ),
    Gate(
        name="palace_waveguide",
        reference="exact - the same guide's phase constant, answered by the other backend",
        activity=VERIFICATION,
        criterion=DECLARED,
        isolates=(
            "the second backend answering the closed form the first one is scored "
            "against, on one guide - so what agrees is the document rather than "
            "either engine; a conforming tetrahedral mesh, where the wall is where "
            "it was drawn instead of on the nearest grid line; a port standing on "
            "the guide's own end face, which the other backend refuses by name; a "
            "region numbered above its ports in one drawing and below them in the "
            "other, which an older Palace refuses to read a port beside; and "
            "the whole route from a marked-up document to a matrix in one call, "
            "which every stage of that adapter answers into and nothing else runs"
        ),
    ),
    Gate(
        name="palace_septum",
        reference="exact - the attenuation of a half-width guide below its cutoff",
        activity=VERIFICATION,
        criterion=DECLARED,
        isolates=(
            "metal drawn inside the region the field is in, as a sheet the region "
            "runs round on both sides - which is how every board draws its metal - "
            "reaching the second backend as a condition on a face inside the model; "
            "read as the fall of the transmission between two lengths of septum, so "
            "what each junction does to the wave crossing once cancels and no "
            "fitted constant is left"
        ),
    ),
    Gate(
        name="palace_lossy_guide",
        reference="exact to first order - the surface resistance on each face of the guide",
        activity=VERIFICATION,
        criterion=DECLARED,
        isolates=(
            "a catalog's conducting sheet reaching the second backend as the surface "
            "impedance of its metal, drawn as the guide's walls where the model ends "
            "and as a plane inside it, each face of which carries it; read as the "
            "power lost over the difference of two lengths, so what the ports and the "
            "plane's ends do cancels; and the run made on the processes asked, with "
            "the loss tangent that keeps Palace from hanging where that metal meets a "
            "wave port"
        ),
    ),
    Gate(
        name="palace_stripline_lumped",
        reference="exact - conformal mapping on the shipped stripline's cross-section",
        activity=VERIFICATION,
        criterion=STUDIED,
        isolates=(
            "the lumped port every board drives its trace through, on the second "
            "backend: elements laid across the gap from the picks, driven in the "
            "direction drawn, with a magnetic wall on the rest of their face, and "
            "a matrix referenced to each port's own resistance, shown by the line at "
            "unequal resistances de-embedding to the impedance it does at equal ones; "
            "read through two lengths of line, so the network a port adds is fitted "
            "and taken off before the impedance is compared, and the phase across the "
            "lengths is read with no model of the port in it"
        ),
    ),
    Gate(
        name="palace_dipole",
        reference="conservation of power in a model where nothing dissipates",
        activity=IDENTITY,
        criterion=DECLARED,
        isolates=(
            "a study open to free space on the second backend: the air the adapter "
            "reserves round a radiator drawn as metal alone, the absorbing condition "
            "on its skin, and the power Palace measures through it read back into "
            "the run, so what the port reflects and what left through the open "
            "surface add to what went in. The resonance and the run's own estimate "
            "of what the surface moved are printed and not held: no closed form for a "
            "strip dipole states its accuracy, and the estimate is too loose to hold "
            "a comparison of two clearances to. Before a release, the same dipole in a "
            "medium it is scaled to, held to the vacuum one within what moving the "
            "open surface moves it: the medium fills the reserved air, is meshed at "
            "its wavelength, and the absorbing condition takes it"
        ),
    ),
    Gate(
        name="openems_cavity",
        reference="exact - the wall condition on a spherical Bessel function",
        activity=VERIFICATION,
        criterion=EXTRAPOLATED,
        isolates=(
            "a surface curved in two directions and closed, so the answer depends "
            "on the radius its staircased triangulation actually has; scored on the "
            "limit the study reaches, with where the lattice fell measured beside it "
            "and no rate asserted, the registration being free in three dimensions; "
            "and beside that the same cavity solved with the surface handed over as "
            "drawn, which prices the correction against not making it"
        ),
    ),
    Gate(
        name="openems_pillbox",
        reference="exact - Bessel roots, the dominant one independent of the height",
        activity=VERIFICATION,
        criterion=STUDIED,
        isolates=(
            "a flat conductor face beside a curved one, which asks the translation "
            "opposite things; four modes off one solve, where a mode carrying "
            "transverse field meets a different wall from one that does not; and "
            "the same cavity solved with its flat faces handed over on the grid "
            "line rather than displaced off it, which prices whether such a face "
            "is a conducting boundary at all"
        ),
    ),
    Gate(
        name="openems_coax",
        reference="exact - Laplace's equation in one variable",
        activity=VERIFICATION,
        criterion=STUDIED,
        isolates=(
            "two round walls, neither on a grid line, with the answer dividing by "
            "the smaller radius; scored on the band the study computed, with no "
            "rate asserted, and beside it what the same mesh does when it is slid "
            "under the drawing at each end of the sequence"
        ),
    ),
    Gate(
        name="openems_stripline",
        reference="exact - conformal mapping on a TEM cross-section",
        activity=VERIFICATION,
        criterion=EXTRAPOLATED,
        isolates=(
            "an impedance with nothing curved in the drawing, so the leading error "
            "is the strip's edge rather than a sampled boundary - and it is scored "
            "as a length of metal rather than as a percentage. It also walks the "
            "measurement plane in toward the feed, which is where the distance a "
            "port needs from its source is read rather than assumed"
        ),
    ),
    Gate(
        name="openems_tdr",
        reference="exact - conformal mapping, on each of three stripline sections",
        activity=VERIFICATION,
        criterion=GRADED,
        isolates=(
            "an impedance as a function of *position*, read in the middle of a "
            "section the solver was never asked about directly, on a distance axis "
            "that carries no measurement"
        ),
    ),
    Gate(
        name="openems_two_port",
        reference="reciprocity and passivity",
        activity=IDENTITY,
        criterion=DECLARED,
        isolates=(
            "what a matrix assembled from separate solves has to obey whatever the "
            "structure is, which no closed form is needed to state, with lumped ports "
            "that send back part of what reaches them"
        ),
    ),
    Gate(
        name="openems_medium",
        reference="the scaling of Maxwell's equations with a uniform permittivity",
        activity=IDENTITY,
        criterion=DECLARED,
        isolates=(
            "the study's medium as the driver lays it, under every solid and through "
            "the absorber: a strip dipole in it, over the band and at the resistance "
            "it scales to, held to the dipole in vacuum on one grid within what "
            "moving the absorber moves it"
        ),
    ),
    Gate(
        name="openems_stepped_guide",
        reference="reciprocity, passivity, and the waves a uniform guide's planes predict",
        activity=IDENTITY,
        criterion=DECLARED,
        isolates=(
            "two waveguide ports whose walls end differently, one on the domain and "
            "one on drawn metal, which the engine reads at different scales; every "
            "other gate has its ports alike, where the scales cancel whether or not "
            "anything undoes them. Each port is also read again inside its box, where "
            "a port far from the step reads its mode alone and one near it is warned of. And "
            "the guide changes behind one port, which sends back what reaches it and "
            "which counting every port's waves takes out of the matrix"
        ),
    ),
    Gate(
        name="openems_microstrip",
        reference="Hammerstad, an empirical fit good to about a percent",
        activity=VALIDATION,
        criterion=COMPOSED,
        isolates=(
            "the whole adapter end to end - capabilities, pre-flight, mesh, "
            "envelope, subprocess, parsing - on the line a user is most likely to "
            "draw first. It is the integration path, and not a measurement of this "
            "workbench. It also walks a ladder of measurement planes in toward the "
            "feed on an *open* line, which is a different kind of reading and "
            "carries no reference at all: every rung is scored against the "
            "outermost, so what it says about the distance a port needs from its "
            "source owes nothing to Hammerstad"
        ),
    ),
    Gate(
        name="openems_lowpass",
        reference="a board somebody else fabricated and measured",
        activity=VALIDATION,
        criterion=COMPOSED,
        isolates=(
            "steps, radiation and a finite ground plane at once, which every exact "
            "reference here is chosen to avoid. Loose on purpose, and it sees what "
            "an idealisation cannot - and it is the one gate whose reference is an "
            "instrument, so the term it cannot establish is a real laminate nobody "
            "characterised rather than an omission"
        ),
    ),
)


@dataclass(frozen=True)
class Reading:
    """What one gate's run came back with.

    It holds what the run wrote and what it returned, and derives every reading
    from those, so that no two readings can disagree - figures on a run recorded
    as skipped, an outcome that says failed under a section quoting the numbers
    as measurements.
    """

    gate: Gate
    #: What the run wrote to standard output, which is where pytest reports. Not
    #: standard error, and not the two together: what a library or a loader writes
    #: on the way past arrives after pytest's summary, and the summary is found by
    #: walking backwards, so one line of chatter counting anything would answer
    #: for the run instead of pytest.
    output: str
    #: What pytest exited with. Consulted, because a summary line counts
    #: assertions and this suite's own guard on how much a tier may solve fails a
    #: session in ``pytest_sessionfinish`` while every assertion passed.
    code: int
    #: Wall time, in seconds. Measured, because how long these take is the
    #: reason nobody runs them and a remembered figure would be the wrong one.
    seconds: float

    @property
    def lines(self) -> tuple[str, ...]:
        """Its ``GATE`` lines, marker stripped, in the order they were printed."""
        return readings_from(self.output)

    @property
    def outcome(self) -> str:
        """pytest's own last line, as the page quotes it."""
        return outcome_from(self.output)

    @property
    def tally(self) -> dict[str, int]:
        """What pytest counted, empty where it reported nothing at all."""
        return tally_from(self.output)

    @property
    def verdict(self) -> str:
        """Which of the verdicts this run was.

        Asked most specific first, so a run in more than one of these states is
        named by the one that says the most about it. A run that never reported
        comes first of all: every count below is zero for it, and each question
        after would answer the wrong thing about it.
        """
        counted = self.tally
        if not counted:
            return NOTHING_RAN
        if counted["failed"] or counted["error"]:
            return FAILED
        if counted["skipped"] and not self.lines:
            return MEASURED_NOTHING
        if not counted["passed"]:
            return NOTHING_RAN
        if counted["skipped"]:
            return PART_RAN
        if self.code:
            return REFUSED
        return SILENT if not self.lines else SOLVED


def readings_from(text: str) -> tuple[str, ...]:
    """Every figure a gate printed, in the order it printed them.

    The progress characters come off the front before the marker is looked for,
    and off the front only: what follows the marker is the figure as it was
    printed, down to the full stop some of them end in. Prose that mentions the
    marker begins with a word, which is not among these characters, so it is
    left where it is.
    """
    found = []
    for line in text.splitlines():
        bare = line.strip().lstrip(PROGRESS)
        if bare.startswith(MARKER):
            found.append(bare[len(MARKER) :])
    return tuple(found)


def summary_line(text: str) -> str | None:
    """pytest's summary line, which is the last thing it prints that has one."""
    for line in reversed(text.splitlines()):
        if COUNT.search(line):
            return line.strip().strip("=").strip()
    return None


def outcome_from(text: str) -> str:
    """The summary line as the page quotes it, or that there was none."""
    return summary_line(text) or "no outcome reported"


def tally_from(text: str) -> dict[str, int]:
    """What the summary line counted, keyed by outcome.

    Empty where pytest printed no summary at all, which is the state that has to
    stay apart from a summary counting zero of everything: the first means the
    run did not finish, and the second cannot be printed.
    """
    line = summary_line(text)
    if line is None:
        return {}
    counted = dict.fromkeys(("passed", "failed", "error", "skipped"), 0)
    for number, outcome in COUNT.findall(line):
        counted[outcome.rstrip("s") if outcome.startswith("error") else outcome] += int(number)
    return counted


def solve(gate: Gate, interpreter: str = sys.executable) -> Reading:
    """Run one gate and read its figures back."""
    started = time.monotonic()
    finished = subprocess.run(
        [interpreter, "-m", "pytest", str(gate.path), "-q", "-s"],
        capture_output=True,
        text=True,
        cwd=TESTS.parent,
        check=False,
    )
    return Reading(
        gate=gate,
        output=finished.stdout,
        code=finished.returncode,
        seconds=time.monotonic() - started,
    )


def sentence(text: str) -> str:
    """``text`` as a sentence, since the table holds clauses.

    A clause reads as a clause under a bold lead-in and as a mistake at the
    start of a line, and the same string is wanted both ways.
    """
    text = text.strip()
    return text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")


def headline(readings: Sequence[Reading]) -> str:
    """Whether this page is evidence, in one sentence at the top of it.

    The table below carries the same verdicts; this says the one thing a reader
    wants of a release form without scrolling for it, in the same words.

    A page with no gate on it is answered explicitly, because "nothing is
    unsolved" is true of nothing at all and would otherwise report success.
    """
    if not readings:
        return "No gate ran."
    unsolved = [reading for reading in readings if reading.verdict != SOLVED]
    if not unsolved:
        return "Every gate solved and passed."
    return "Not every gate solved and passed: " + ", ".join(
        f"{reading.gate.name} - {reading.verdict}" for reading in unsolved
    )


def render(readings: Sequence[Reading]) -> str:
    """The report, as markdown.

    Takes a sequence rather than an iterable: the readings are walked for the
    headline, for the table and for the sections, and one that could be walked
    once would put a headline over a page with nothing under it.
    """
    readings = list(readings)
    out = [
        "# What this workbench has measured",
        "",
        "Produced by `python3 -m tests.gate_report`. Every figure below came out",
        "of the run that produced this page; nothing in it was typed by hand.",
        "",
        f"**{headline(readings)}**",
        "",
        "**Read the activity before the number.** A verification gate is scored",
        "against an exact reference, so its figure is the solver's own error and an",
        "accuracy claim can be made from it. A validation gate is scored against a",
        "fit or a fabricated board, so its figure is the solver's error and the",
        "reference's together - and it can be passed by having a poor reference,",
        "which is why the width of the comparison is reported beside the error.",
        "",
        "**And read the criterion before believing the bar bites.** Where a gate",
        "is held to the interval a refinement study computes, that bar comes out of",
        "the run and narrows as the study improves. Where the study is there but",
        "the bar is on the limit and the rate it reaches, the sequence is measured",
        "and the bound on it is still one the file states. Where there is a single",
        "mesh, the bound is argued from the mechanism and the run does not derive",
        "it.",
        "",
        "| gate | verdict | activity | criterion | reference | took |",
        "|---|---|---|---|---|---|",
    ]
    for reading in readings:
        out.append(
            f"| {reading.gate.name} | {reading.verdict} | {reading.gate.activity} | "
            f"{reading.gate.criterion} | {reading.gate.reference} | "
            f"{reading.seconds:.0f} s |"
        )
    out.append("")

    for reading in readings:
        out += [
            f"## {reading.gate.name}",
            "",
            f"**Scored against** {reading.gate.reference}.",
            "",
            f"**This is {reading.gate.activity}**, decided by {reading.gate.criterion}.",
            "",
            f"**What it isolates.** {sentence(reading.gate.isolates)}",
            "",
            f"*{reading.outcome}.*",
            "",
        ]
        if said := SAID.get(reading.verdict):
            out += [said, ""]
        if reading.lines:
            out += ["```"] + list(reading.lines) + ["```", ""]
    return "\n".join(out)


def status_of(readings: Sequence[Reading]) -> int:
    """What the run is worth, as an exit status.

    Success is every gate having solved and passed, and nothing weaker. A run
    with no gate in it needs no case of its own: an empty set of verdicts is
    neither of the two below. What measured nothing is told apart from what
    measured something wrong, because they are answered by different people:
    the first by whoever set up the machine, the second by whoever changed the
    workbench.
    """
    verdicts = {reading.verdict for reading in readings}
    if verdicts == {SOLVED}:
        return 0
    if verdicts == {MEASURED_NOTHING}:
        return 3
    return 1


def main(argv=None) -> int:
    parsed = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parsed.add_argument(
        "--only",
        action="append",
        help="run just this gate, by name; repeatable",
    )
    parsed.add_argument(
        "--out",
        type=Path,
        help="write the report here instead of to standard output",
    )
    arguments = parsed.parse_args(argv)

    asked = set(arguments.only or ())
    if unknown := sorted(asked - {gate.name for gate in GATES}):
        # Named and not here, rather than "none of them is here": asking for one
        # gate that exists and one that does not would otherwise run the first
        # and report a whole run of everything asked for.
        print(f"no gate named {', '.join(unknown)}", file=sys.stderr)
        return 2
    wanted = [gate for gate in GATES if not asked or gate.name in asked]

    readings = []
    for gate in wanted:
        print(f"solving {gate.name} ...", file=sys.stderr, flush=True)
        reading = solve(gate)
        print(
            f"  {reading.verdict}: {reading.outcome} in {reading.seconds:.0f} s",
            file=sys.stderr,
            flush=True,
        )
        readings.append(reading)

    report = render(readings)
    if arguments.out:
        arguments.out.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)

    status = status_of(readings)
    if status:
        print(headline(readings), file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
