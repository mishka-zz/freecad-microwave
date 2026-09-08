# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The pillbox cavity a gate would measure, as numbers both sides share.

Here rather than in either file because the cavity is drawn under ``freecadcmd``
and solved under the interpreter that owns the openEMS bindings, and the two
have to agree about what was built. Importing this needs neither.

What this module and ``pillbox_probe`` carry is the drawing and the arithmetic
both sides share. ``test_acceptance_pillbox`` is what scores the line.

What this cavity is for, beyond a second reference, is the pair of surfaces it
puts next to each other. Its wall curves and its ends do not, and the two ask
opposite things of the translation - one has to be grown before openEMS samples
it, the other has to stay where the grid holds it. A sphere cannot pose that
question, having no flat face, and a shape that is all flat faces cannot pose it
either.

Why a cylinder when a sphere is already gated
---------------------------------------------

Because it takes the measurement apart. A sphere's resonance answers to its
whole surface at once, and that surface is curved in two directions and closed,
so its staircase and its polygonisation arrive together and no reading separates
them.

The mode read here is the one with no variation along the axis. Its frequency is
the first zero of ``J0`` over the radius and **nothing else** - not the height,
not the ends. So:

- **The flat ends are out of the answer.** A rectilinear grid holds them exactly
  in any case, and the mode would not notice if it did not: what is left for the
  discretisation to get wrong is one curved wall, and the resonance prices it in
  frequency units.
- **The same cavity at two heights has one resonance.** No reference, no error
  bar, and it is a different question from any closed form - a formula computed
  from the radius cannot notice the height leaking into the answer.
- **A cylinder is ruled**, so a triangulation of it is exact along the axis and
  approximates only around the circumference. Drawing it coarsely and finely
  moves one term of the error and no other.

The height is the dimension to be careful with
-----------------------------------------------

``TE111`` falls as the cavity is stretched and the flat mode does not, so past a
height of about two radii the two cross and the lowest line is no longer the one
this reads. Both heights here stand well clear of that, and
``tests/test_pillbox_fixture.py`` holds them there against the spectrum itself
rather than against a ratio written down.

What is still in the answer besides the wall
---------------------------------------------

The probe. openEMS builds a lumped port as two conducting caps joined by a
resistance, so it is a metal object standing in the field rather than a circuit
element beside it: it damps what it measures and it pulls the line.

*How much* metal is a question about the grid rather than about the drawing,
because openEMS builds the element from its box moved onto the nearest lines. The
adapter pins those faces, so here it is the box that was drawn at every cell -
which is what lets the sequence be about the wall. :data:`LONG_PROBE` is what
prices what remains: the same cavity solved with the element doubled says how
much of the answer it is worth at all, on the same scale as everything else here
- what a cell of radial displacement is worth.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import jv

from Microwave.Solvers.openems.materials import VACUUM_PERMITTIVITY
from Microwave.Solvers.openems.staircase import PINNED_CLEARANCE
from tests.analytic import reference

#: The cavity's radius, in mm. It alone sets the mode read here, which is what
#: makes a pillbox worth gating on: one number to get wrong.
RADIUS = 15.0

#: The two heights, in mm. Their ratio is the whole point - the mode is the same
#: at both - so what they have to be is far enough apart to be a real change and
#: both well under the height at which ``TE111`` drops below the mode being read.
TALL = 15.0
SHORT = 9.0

#: How thick the conducting can is, in mm. It only has to be metal and it only
#: has to seal: a resonance is entirely inside the bore.
SHELL_WALL = 2.0

#: Vacuum, so the resonance stays a pure function of the radius and the fill
#: contributes nothing but the damping below.
EPS_R = 1.0

#: What damps the ring, as the conductivity openEMS is handed rather than as a
#: loss tangent - the two are the same statement at one frequency, and
#: :data:`LOSS_MEASURED_AT` is which. A perfectly conducting cavity holding a
#: lossless fill rings for ever, so an undamped run would neither finish nor
#: resolve a line.
KAPPA = 0.0016

#: The sweep, in Hz. Wide enough to hold the modes above the one being read, so
#: a fit that has locked onto a neighbour is visible rather than plausible.
BAND = (4e9, 16e9)
POINTS = 1201

#: How far either side of the closed form a line would be looked for, as a share
#: of it. It has to be wide enough to hold what a coarse mesh displaces and far
#: short of the next mode, and ``test_pillbox_fixture`` holds the second against
#: the spectrum this cavity actually carries.
WINDOW = 0.12

#: The probe, in mm. Short: it is metal standing inside the resonator, so it
#: damps what it is measuring and pulls the line as it grows.
PROBE_LENGTH = 1.0
PROBE_WIDTH = 1.0

#: The probe the same cavity is solved with once more, to price what the first
#: one is worth. Doubled rather than nudged, so that what it moves stands above
#: everything else that moves an answer and can be read as a bound on it.
LONG_PROBE = 2.0 * PROBE_LENGTH

#: The resistance the probe is, and the impedance the reflection is reported
#: against - one number, because for a lumped port they are the same thing.
PORT_IMPEDANCE = 50.0

#: How many cells to a wavelength at the top of the band, at each cell size the
#: cavity is solved at.
#:
#: **The coarsest is as coarse as this fixture goes**, and the constraint is the
#: probe rather than the wall: ``test_the_probe_still_spans_a_cell_at_the_coarsest``
#: keeps the cell under the element's own gap, so the element is an integral over
#: more than a single edge. So the lever arm is spent at the fine end.
#:
#: **How far it has to reach** is arithmetic on what the band asks - the lever
#: arm raised to the observed order has to clear one plus the safety factor, and
#: ``test_the_closed_form_is_inside_the_uncertainty_the_study_computed`` states
#: that. How much further than *that* is a measurement: these answers carry a
#: floor this fixture does not remove, and the reach here is one where moving any
#: single point by that floor leaves the verdict where it was.
#:
#: **How many** is set by the procedure that reads them. The fits in
#: :mod:`tests.convergence` carry three unknowns, so a study of three grids
#: passes through its own points and reports a scatter of nothing whatever the
#: data did.
DIVISORS = (20, 30, 45, 84)

#: How far across the cell that holds it the wall stands, in every case here.
#:
#: A cell size says how big the cells are and nothing about where they fall, and
#: on a cylinder standing on the mesh axis the second of those does not average
#: away: the same circle meets the grid at every height. Measured, sliding the
#: lattice through one cell moves the line as far as refining the cell across
#: this whole sequence does - so a study that let it vary would be fitting an
#: exponent through where the lines fell. Every case is slid until the wall
#: stands here, which leaves the cell as the only thing separating the sequence,
#: and :data:`LATTICE_PHASES` is what prices the choice.
#:
#: Halfway, because the two ends of the cell are each a case of their own: a
#: line lying exactly on a conductor's face conducts, so a wall at zero is
#: sampled by a line the metal owns. It is still *a* choice. The sequence solved
#: at another phase is a different sequence with the same limit, not the same
#: one, and what it costs to have chosen is the lattice band.
WALL_PHASE = 0.5

#: How many times the slide that puts it there is repeated. The mesh is graded,
#: so the cell the wall lands in after a slide need not be the one the slide was
#: measured from; a second pass is measured against the cell it reached.
#: ``test_pillbox_drawing`` holds the result rather than trusting this.
WALL_PHASE_PASSES = 2

#: Where the lattice sits against the drawing, as offsets of the grid across the
#: axis, in cells, *beyond* :data:`WALL_PHASE`. The sequence's own alignment is
#: the zero offset and is not repeated here.
#:
#: **The grid moves and the drawing stays.** What openEMS samples is the drawing
#: at the lines, so only the two together mean anything - and moving the drawing
#: is the way round that changes nothing, the mesher planning its lines from the
#: geometry and so carrying them along with it.
#:
#: Across the axis only. A square lattice and a circle centred on it make most of
#: the transverse plane a copy of the rest, so every offset reduces to the
#: triangle from no offset out to half a cell along an axis and half a cell along
#: the diagonal; its corners are here, and a point on the side between them.
#: Along the axis there is nothing to slide - the mode does not use that
#: direction and the ends are flat.
#:
#: Taken from :data:`WALL_PHASE`, a half-cell offset puts the wall *on* a line,
#: which that constant calls a case of its own - metal owning the line that
#: samples it. That is deliberate here rather than awkward: it is an alignment a
#: user's mesh can land on, and what these cases bound is the whole range, so the
#: end of the range belongs in it.
LATTICE_PHASES = ((0.5, 0.0), (0.25, 0.25), (0.5, 0.5))

#: Where the spectrum case stands its probe, as a share of the radius out from
#: the axis and a share of the height along it from the middle. Which way along
#: it does not matter, the two ends being alike.
#:
#: **The arrangement every other case uses is deliberately blind to most of the
#: spectrum, and that is what this moves.** An element on the axis at mid-height
#: sits on a node of every mode with azimuthal variation, the axial field of one
#: going as ``J_m`` of the radius, and on a node again of every mode with an odd
#: number of half-waves along the axis. That is what keeps the dominant line
#: clean and alone; it also leaves nothing else in the band to read.
#:
#: Out along one axis rather than on a diagonal, so that of each pair a circular
#: cavity is degenerate in, the one the probe stands on the crest of is the one
#: it reads.
SPECTRUM_AT = (0.4, 0.25)

#: The mode every case but the spectrum is read for: no azimuthal variation, no
#: variation along the axis, and so a frequency that is the radius and nothing
#: else.
DOMINANT = ("TM", 0, 1, 0)

#: The modes the spectrum case is scored against, as ``kind, order, root,
#: axial``, lowest first. Every one of them has an axial electric field, which is
#: what an element lying along the axis couples to and the whole of what it
#: couples to. A ``TE`` mode has none at all, so reading those wants an element
#: across the axis instead and is a solve of its own.
SPECTRUM_MODES = (DOMINANT, ("TM", 1, 1, 0), ("TM", 0, 1, 1), ("TM", 1, 1, 1))

#: How far either side of each of those to look for its line, as a share of it.
#: Far short of half the gap to the nearest neighbour, which is a property of the
#: cavity rather than of this number and is held in ``test_pillbox_fixture``.
SPECTRUM_WINDOW = 0.012

#: Which cell the spectrum is read at - the middle one, where everything else
#: outside the sequence uses the coarsest.
#:
#: A mode with a half-wave along the axis is displaced further at the coarsest
#: cell than the gap between it and its neighbour, so no window centred on the
#: closed form holds one line and excludes the other, and a fit given both
#: answers with whichever it reached first. The middle cell has room for both.
SPECTRUM_DIVISOR = sorted(DIVISORS)[1]

#: The case that carries it.
SPECTRUM_CASE = f"spectrum-{SPECTRUM_DIVISOR}"

#: The coarsest case with the clearance set to nothing, and its partner.
#:
#: A flat conductor face square to an axis gets a grid line of its own, and the
#: line lands *on* the face - where openEMS' containment segment has nothing to
#: be sure about, so the face can read as air and never zero the tangential field
#: standing on it. The clearance displaces the face into the void by enough for
#: that line to fall in metal, and what that is worth is what this pair reads.
#:
#: **At the coarsest cell**, where a share of one is worth most and the solve is
#: cheapest, and where both of this can's caps turn out to be ambiguous rather
#: than one - which is a property of the drawing and is why the pair is read
#: beside the census rather than instead of it.
#:
#: Read on the dominant line like every other case here, and for the same reason:
#: it is the line with a window wide enough to hold wherever the answer goes. The
#: rest of the spectrum moves too, and by amounts a fit cannot follow once each
#: line has left the window its neighbours leave it.
ON_THE_FACE_CASE = f"on-the-face-{min(DIVISORS)}"
ON_THE_FACE = (ON_THE_FACE_CASE, f"tall-{min(DIVISORS)}")

#: Pinned, not energy-terminated: openEMS re-checks its energy criterion on a
#: wall-clock timer, so an energy-terminated run stops at a step that depends on
#: what else the machine was doing.
#:
#: Scaled with the divisor, which is *not* the same as a fixed length of ring in
#: seconds however much it looks like one: openEMS takes its timestep from the
#: smallest cell in the grid, and that is set by whatever the mesher put its
#: finest lines around rather than by the policy. Measured across these meshes
#: the ring is not constant, and what that is worth to the line was measured too
#: - a threefold ring moves it far less than the sequence's own floor. So the
#: figure stands and the claim does not.
TIMESTEPS_PER_DIVISOR = 1600


@dataclass(frozen=True)
class Case:
    """One solve: a cavity, a mesh, and where that mesh falls against it."""

    divisor: int
    height: float
    probe: float
    phase: tuple[float, float]
    #: What share of its cell a flat face square to an axis is displaced by, so
    #: the line pinned to it falls in metal rather than on the surface. Zero
    #: hands the face over as drawn, which is the member of the pair that prices
    #: the displacement.
    clearance: float = PINNED_CLEARANCE
    #: Where the probe stands, as a share of the radius out from the axis and a
    #: share of the height along it from the middle. The centre is where a mode
    #: with no azimuthal variation and none along the axis is strongest, and
    #: where every other mode has a node.
    probe_at: tuple[float, float] = (0.0, 0.0)


def _cases() -> dict[str, Case]:
    """Every case, as ``name -> Case``.

    The sequence a rate would be read from is the tall cavity at each cell size,
    on the drawing's own lattice. Everything else is that sequence with exactly
    one thing changed, so what each says is about that one thing:

    - the *short* cavity, at the coarsest cell, where the staircase is largest;
    - the *long probe*, at the same cell, which prices the one part of the
      arrangement that is not geometry;
    - the *lattice*, again at the coarsest cell - how far sampling can displace a
      boundary is bounded by the cell it is sampled on, so a figure measured
      there bounds every finer point rather than being carried to it;
    - the *spectrum*, whose probe stands where the rest of the band is rather
      than where the dominant mode is, and which is read against several exact
      roots instead of one;
    - *on the face*, the coarsest case with its flat caps handed over with no
      clearance, which is the one thing it differs in and the pair
      :data:`ON_THE_FACE` reads.

    There is no coarse *triangulation* among them. FreeCAD treats a deflection
    as a hint and refines past it until the volume the triangles enclose is
    close enough to the shape's own, so asking for a coarser surface returns the
    same one; and what that surface loses is far inside a cell here.
    :func:`enclosed_radius` is what prices it, off the triangulation an envelope
    already carries.
    """
    # The coarsest cell is the fewest cells to a wavelength, so it is the
    # smallest divisor and not the largest.
    coarsest = min(DIVISORS)
    plain = Case(coarsest, TALL, PROBE_LENGTH, (0.0, 0.0))
    found = {f"tall-{divisor}": replace(plain, divisor=divisor) for divisor in DIVISORS}
    found[f"short-{coarsest}"] = replace(plain, height=SHORT)
    found[f"probe-{coarsest}"] = replace(plain, probe=LONG_PROBE)
    found[SPECTRUM_CASE] = replace(plain, divisor=SPECTRUM_DIVISOR, probe_at=SPECTRUM_AT)
    found[ON_THE_FACE_CASE] = replace(plain, clearance=0.0)
    for phase in LATTICE_PHASES:
        found[phase_case(phase)] = replace(plain, phase=phase)
    # A case is its own directory, so two names that collided would be one solve
    # reported twice - and a name rounds the offset it is built from.
    assert len(found) == len(DIVISORS) + 4 + len(LATTICE_PHASES), sorted(found)
    return found


def phase_case(phase) -> str:
    """The case name for a lattice offset, in hundredths of a cell."""
    return "phase-{:.0f}-{:.0f}".format(*(100.0 * value for value in phase))


def cell_size(divisor: float) -> float:
    """The bulk cell the mesh policy asks for at this divisor, in mm."""
    return reference.SPEED_OF_LIGHT / BAND[1] * 1e3 / divisor


def timesteps(divisor: float) -> int:
    return int(TIMESTEPS_PER_DIVISOR * divisor)


def wall_cell(lines) -> float:
    """The cell the curved wall is sampled on, in mm, off a finished grid.

    Measured rather than requested. The mesher answers to the drawing as well as
    to the policy - it puts lines of its own where a surface asks for them - so
    what a case was solved on is a property of its grid, and anything scored
    against the cell that was *asked for* would be scored against the wrong
    length the moment the two parted.

    A line falling exactly on the wall belongs to the cell *above* it, which is
    the case the mesher produces whenever it pins a line to the surface. It is a
    choice and not a derivation - the two neighbours differ only where the
    grading changes, and by less than the grading ratio allows.
    """
    lines = np.asarray(lines, dtype=float)
    widths = np.diff(lines)
    inside = np.clip(np.searchsorted(lines, RADIUS, side="right") - 1, 0, len(widths) - 1)
    return float(widths[inside])


def wall_phase(lines) -> float:
    """How far across the cell that holds it the wall stands, as a share of it.

    The other half of what a mesh does to this surface, and the half a cell size
    says nothing about. :data:`WALL_PHASE` is where every case here puts it and
    says why.
    """
    lines = np.asarray(lines, dtype=float)
    above = int(np.searchsorted(lines, RADIUS, side="right"))
    return float((RADIUS - lines[above - 1]) / (lines[above] - lines[above - 1]))


def enclosed_radius(vertices, faces, height: float) -> float:
    """The radius the triangulated bore encloses, in mm.

    A triangulation of a cylinder is a prism on an inscribed polygon: its
    vertices lie on the drawn circle and only its chords cut inside, so what
    reaches the solver is slightly the smaller cavity. Read as the radius of the
    circle holding the same area, which is the form a resonance is sensitive to
    and is directly comparable with a cell.

    The volume is the divergence theorem over the triangles, which needs the
    surface to be closed and says nothing about which way it is wound - so the
    magnitude is taken, a kernel being free to hand back either winding.
    """
    points = np.asarray(vertices, dtype=float)
    triangles = points[np.asarray(faces, dtype=int)]
    volume = abs(
        np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
        / 6.0
    )
    return float(np.sqrt(volume / (np.pi * height)))


def mode_name(mode) -> str:
    """What a mode is called, from the indices the closed form takes."""
    return "{}{}{}{}".format(*mode)


def mode_frequency(mode, height: float = TALL, radius: float = RADIUS) -> float:
    """What one mode of this cavity resonates at, from the closed form.

    The tall cavity unless another is named, because a height has to be given
    and only the modes that use it care which. The drawn radius unless another is
    named, so that a case can be scored against the polygon it was given rather
    than against the circle it was drawn as.
    """
    kind, order, root, axial = mode
    return reference.circular_cavity_frequency(
        radius * 1e-3, height * 1e-3, order=order, root=root, axial=axial, kind=kind, eps_r=EPS_R
    )


def frequency(radius: float = RADIUS) -> float:
    """What the drawing resonates at on the mode every case but the spectrum is
    read for - the one with no variation along the axis, and so the one whose
    frequency is the radius and nothing else."""
    return mode_frequency(DOMINANT, radius=radius)


def radius_of(volume: float, height: float) -> float:
    """The radius of a cylinder of ``height`` enclosing ``volume``, in mm.

    What a triangulated cylinder *is*, as against what it was drawn as. Its ends
    are planar and reach the engine exactly; only the wall is approximated, by a
    prism whose faces are chords of the circle - so this comes back smaller than
    :data:`RADIUS`, by an amount the triangulation sets and no cell size moves.
    """
    return float(np.sqrt(volume / (np.pi * height)))


def axial_field(mode, at=SPECTRUM_AT) -> float:
    """How much of a mode's axial electric field stands where a probe does, as a
    share of that mode's own largest.

    A coupling rather than a volt, and it is what says whether a line will be
    there to fit: an element lying along the axis drives the axial electric
    field and nothing else, so a mode this answers zero for is one the probe
    cannot see however long the run.

    A ``TM`` mode's axial field goes as ``J_m`` of the radius, as the cosine of
    the angle, and as the cosine of the distance along the axis. The probe stands
    on the angle where the first of those is largest, so what is left is the
    other two. A ``TE`` mode has no axial electric field at all.
    """
    kind, order, root, axial = mode
    if kind == "TE":
        return 0.0
    radial, down = at
    across = reference.circular_cavity_root(order=order, root=root, kind=kind) * radial
    # Measured from the middle, where the closed form counts half-waves from one
    # flat end.
    return float(jv(order, across) * np.cos(axial * np.pi * (0.5 - down)))


def effective_radius(measured: float) -> float:
    """The radius a cavity resonating here would have, in mm.

    The whole geometry dependence of the mode being read is one Bessel root over
    the radius, so a measured frequency is a measured radius - which is the
    quantity the staircase displaces, and the one a displacement is legible in.
    """
    root = reference.circular_cavity_root()
    return float(reference.SPEED_OF_LIGHT * root / (2 * np.pi * measured) * 1e3 / np.sqrt(EPS_R))


def displacement_worth(distance: float) -> float:
    """What a wall displaced by ``distance`` mm is worth here, as a share of the
    frequency.

    The mode's frequency is inversely proportional to the radius and depends on
    no other dimension, so the two are the same fraction.
    """
    return distance / RADIUS


def loss_tangent(at: float | None = None) -> float:
    """What :data:`KAPPA` is as a loss tangent, which is what sets Q."""
    at = frequency() if at is None else at
    return float(KAPPA / (2 * np.pi * at * VACUUM_PERMITTIVITY * EPS_R))


#: The dominant mode, which is what the fill's loss is stated at.
LOSS_MEASURED_AT = frequency()

CASES = _cases()

#: The cases a rate is read from, coarsest cell first - the fewest cells to a
#: wavelength being the largest cell.
SEQUENCE = tuple(f"tall-{divisor}" for divisor in sorted(DIVISORS))

#: The case a run that is not doing the study solves: the operating point.
#:
#: The coarsest of the sequence and its own first point. The three cases each
#: paired against it - the short can, the doubled probe and the caps handed over
#: on the grid line - are drawn on that same cell, so nothing solved outside the
#: study is meshed finer than it.
NOMINAL = SEQUENCE[0]
