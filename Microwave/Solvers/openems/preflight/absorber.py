# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checks about the absorbing boundary, and where it reaches.

A PML is made of grid cells rather than of empty space: it takes cells from the
ends of every axis it is declared on. These checks ask whether it stands on
something it can absorb, whether it leaves a model behind after taking its
share, whether each boundary is a word openEMS knows, and whether the depth a
``PML_<n>`` asks for is the ``pml_cells`` the mesher set aside.

The helpers here answer where the absorber reaches on one axis, and whether it
has left an interior at all. :mod:`.ports` asks them too: a port inside the PML
is measured in a medium other than the guide.
"""

from __future__ import annotations

from ..model import AXIS_NAMES, DIMENSIONS, MeshGrid, Problem
from ..plan import structure_bounds, through_faces_past_the_structure
from .finding import REFUSE, WARN, Finding

#: openEMS' boundary vocabulary. ``PML_n`` is matched separately.
_BOUNDARY_WORDS = frozenset({"PEC", "PMC", "MUR"})


#: Report a drawn axis when less than this share of it is left outside the
#: absorber. At one half the absorber has taken more of the model than it left,
#: and past that point most of the model is not being solved.
#:
#: A half is a choice rather than a guess, and the cases either side of it fix
#: the bracket. Both move with the mesh rather than with the drawing. A healthy
#: model keeps most of itself outside the absorber even where the absorber is
#: deliberately deep, and a model coarsened until the block covers the structure
#: is refused outright while meshing, before this could describe it. This check
#: speaks for the range between them.
#:
#: The share does not move in step with the mesh policy across that range. The
#: block is the finest cell the sizing field asks for near the wall, and the
#: band that is read over reaches in proportionally to the cell, so coarsening
#: widens the band, and a feature the wider band swallows collapses the block
#: rather than deepening it. The share therefore tracks the mesh policy over the
#: range as a whole rather than step by step. Read this as a warning about the
#: grid in hand rather than as a threshold to tune against.
_INTERIOR_SHARE_LIMIT = 0.5


def _check_the_absorber_stands_on_the_structure(problem: Problem) -> list[Finding]:
    """The overrun half of the THROUGH question, on an unmeshed envelope.

    ``plan`` refuses this while meshing, which covers every route through the
    workbench. It does not cover the driver. ``python -m ...driver`` reads an
    envelope, checks it and builds, so a grid that came from anywhere else - a
    bug report replayed by hand, or a file edited to reproduce something - would
    be solved without the comparison ever running. The driver re-runs pre-flight
    so that no guard is opt-in.

    :func:`~.grid._check_grid_covers_the_model` is the other half, and reports
    geometry the grid ends short of. Between them the grid's outer edge is
    judged against the structure in both directions.
    """
    findings = through_faces_past_the_structure(
        problem.grid, problem.solids, problem.ports, problem.grid.params.get("padding")
    )
    return [Finding(REFUSE, subject, message) for subject, message in findings]


def _check_the_absorber_leaves_a_model(problem: Problem) -> list[Finding]:
    """A THROUGH face puts the absorber on the structure. Report when it covers
    nearly all of it.

    The block is ``pml_cells`` of the cell the mesher lays at that wall, and a
    cell is a wavelength over ``ElementsPerWavelength``, so the block deepens as
    the band falls or the mesh is coarsened, while the structure the user drew
    does not change. Coarsening the mesh therefore leaves less of the model
    outside the absorber, which is backwards, and the cell count does not show
    it: the count is the same either way, whether those cells cover a millimetre
    of the board or most of it.

    This is a trend rather than a law. What the mesher lays at the wall is the
    finest cell the sizing field asks for nearby, so a coarser policy can widen
    the reach far enough to find a feature and hand back a finer block. This
    check therefore reads the finished grid instead of the policy that produced
    it.

    It warns rather than refusing. A small interior is not by itself wrong: a
    long line measured only in the middle is legitimate. Whether everything
    being measured still fits is asked by
    :func:`~.ports._check_port_clear_of_absorber` and
    :func:`~.ports._check_port_inside_the_grid`, which refuse and name the
    absorber's depth themselves. What is left here is the case with no port in
    the absorber at all, where the only symptom is a model most of which is not
    being solved.

    Only faces the structure runs out through are described. An outward-padded
    face has a domain larger than the structure by design, and a share above one
    describes nothing there.
    """
    grid = problem.grid
    lower, upper = structure_bounds(problem.solids, problem.ports)

    findings = []
    for dim in range(DIMENSIONS):
        # Nothing here for an axis that absorbs nowhere, and nothing for one
        # the absorber covers either. A share of the model is the wrong
        # description of the second: the two blocks together reach at least as
        # far as the model does, so the figures below would account for all of
        # it or more, and `_check_the_absorber_fits_the_axis` refuses such an
        # axis by name.
        interior = _absorber_bounds(grid, dim)
        if interior is None:
            continue
        drawn = float(upper[dim]) - float(lower[dim])
        if drawn <= 0:
            continue
        share = (interior[1] - interior[0]) / drawn
        if share >= _INTERIOR_SHARE_LIMIT:
            continue

        # Measured against the structure the user drew, which is also where the
        # grid now ends, so each end of this subtraction is the block at one
        # face and nothing else. The ends are described one at a time: an axis
        # can absorb on one end and be padded outward on the other, and a padded
        # end covers none of the model.
        taken = [
            f"{gap:.4g} mm at {AXIS_NAMES[dim]}={end}"
            for end, gap in (
                ("min", interior[0] - float(lower[dim])),
                ("max", float(upper[dim]) - interior[1]),
            )
            if gap > 0
        ]
        per_end = (
            f" The absorber is {_absorber_cells(grid, dim)} cells deep and "
            f"covers {' and '.join(taken)} of a {drawn:.4g} mm model."
            if taken
            else ""
        )
        findings.append(
            Finding(
                WARN,
                f"{AXIS_NAMES[dim]} domain",
                f"the absorber leaves {share:.0%} of the structure outside it."
                f"{per_end} Everything measured has to fit in that. Draw "
                f"more structure, raise ElementsPerWavelength, or lower "
                f"PMLCells - note that *coarsening* the mesh deepens the "
                f"absorber rather than leaving more room",
            )
        )
    return findings


def _absorber_cells(grid: MeshGrid, dim: int) -> int:
    """How many cells of absorber this axis declares at each end, or zero.

    ``pml_cells`` is per axis. A closed structure absorbs on some axes and is
    walled by a conductor on the others.

    This answers what the axis declares rather than what it can hold. An axis
    with too few cells to carry two blocks and something between them is
    refused by :func:`_check_the_absorber_fits_the_axis` and reported by
    :func:`_absorber_covers_the_axis`. Answering zero here would instead say
    the axis absorbs nowhere, which silences every check that measures against
    an absorber.

    A declaration of nothing, and one below nothing, are both no absorber. The
    envelope's ``params`` are provenance and nothing validates them, so a
    negative arrives here as readily as a positive.
    """
    cells = grid.params.get("pml_cells")
    if isinstance(cells, (list, tuple)):
        cells = cells[dim] if len(cells) > dim else 0
    if not cells or int(cells) < 0:
        return 0
    return int(cells)


def _absorber_covers_the_axis(grid: MeshGrid, dim: int) -> bool:
    """Whether the two blocks leave no cell between them.

    Asked before the interior, because such an axis has none to ask about. Every
    cell absorbs, so a point anywhere on the axis is in the attenuating region.
    That covers the line the blocks meet on, which carries an absorbing cell on
    each side of it and would otherwise read as an interior of no width.
    """
    cells = _absorber_cells(grid, dim)
    return bool(cells) and len(grid[dim]) - 1 <= 2 * cells


def _absorber_bounds(grid: MeshGrid, dim: int) -> tuple[float, float] | None:
    """Where the interior starts and ends on one axis, absorber excluded.

    ``None`` where there is no interior: the axis declares no absorber, or it
    declares one that leaves no cell between the blocks.
    :func:`_absorber_covers_the_axis` is what tells those apart, and a caller
    that has to describe the axis asks it first.
    """
    cells = _absorber_cells(grid, dim)
    if not cells or _absorber_covers_the_axis(grid, dim):
        return None
    lines = grid[dim]
    return float(lines[cells]), float(lines[-1 - cells])


def _absorber_depth(grid: MeshGrid, dim: int) -> tuple[float, float]:
    """How deep the absorber reaches at each end of an axis, in mm, as laid.

    Measured off the grid rather than recomputed from what the mesher was asked
    for. Nothing predicts this depth any more: the block is the cell the mesh
    laid at that wall, offered back until the interior accepted it. Reading the
    grid is also the right way round for an envelope that reaches pre-flight
    from a bug report or a hand edit, which carries lines rather than the policy
    that produced them.

    This reports how far in the absorber reaches, which is what a point inside
    it raises. How much of the model that covers is the same number seen from
    the other side, since the grid now ends where the structure does.

    Asked only of an axis that has an interior. Where the blocks leave none
    there is no depth as laid to report - the two together reach at least the
    whole axis - and :func:`_absorber_covers_the_axis` is what a caller asks
    instead.
    """
    cells = _absorber_cells(grid, dim)
    if not cells:
        return 0.0, 0.0
    lines = grid[dim]
    return (
        float(lines[cells]) - float(lines[0]),
        float(lines[-1]) - float(lines[-1 - cells]),
    )


def _check_the_absorber_fits_the_axis(problem: Problem) -> list[Finding]:
    """An axis has to hold two blocks of absorber and something between them.

    :func:`~..absorber._validate_absorber` refuses this while meshing, on the
    same comparison, so no grid the mesher laid reaches here. An envelope that
    arrives some other way does, and this is the last comparison before the
    build: the depth goes to the engine as a number of cells and the lines go
    as positions, and nothing downstream holds the two against each other.
    """
    findings = []
    for dim in range(DIMENSIONS):
        cells = _absorber_cells(problem.grid, dim)
        lines = len(problem.grid[dim])
        if not cells or lines >= 2 * cells + 2:
            continue
        findings.append(
            Finding(
                REFUSE,
                f"{AXIS_NAMES[dim]} domain",
                f"the axis has {lines} lines and declares {cells} absorber cells "
                f"at each end, which needs {2 * cells + 2}. The two blocks "
                f"overlap, so every cell on this axis absorbs and nothing on it "
                f"is being solved. Lower PMLCells, or mesh this axis more finely",
            )
        )
    return findings


def _check_boundary(problem: Problem) -> list[Finding]:
    findings = []
    for index, word in enumerate(problem.boundary):
        face = f"{AXIS_NAMES[index // 2]}{'min' if index % 2 == 0 else 'max'}"
        if word in _BOUNDARY_WORDS:
            continue
        if word.startswith("PML_") and word[4:].isdigit() and int(word[4:]) > 0:
            findings += _check_absorber_agrees(problem, index, face, int(word[4:]))
            continue
        findings.append(
            Finding(
                REFUSE,
                f"boundary {face}",
                f"{word!r} is not an openEMS boundary condition; expected "
                f"PML_<n> or one of {sorted(_BOUNDARY_WORDS)}",
            )
        )
    return findings


def _check_a_mur_wall_carries_no_excitation(problem: Problem) -> list[Finding]:
    """A Mur wall a driven port stands on never switches on, so it reflects.

    openEMS delays a Mur boundary until the source on it has finished, and takes
    "finished" from the excitation signal's own length. For the drive this
    adapter builds that length is the whole run, because a custom signal is
    allocated one sample per timestep. The start step then lands past the last
    step and the wall stays a conductor for the entire solve. openEMS reports
    this once, among its startup lines.

    The condition here is wider than openEMS', which asks for an excitation
    element sitting on the wall's own line and pointing along it. A port whose
    box does not reach the wall cannot have put an element there, and one that
    does reach it need not have. This therefore warns rather than refusing. The
    warning is the safe direction for a wall that would otherwise reflect
    everything without a word, and a refusal is the wrong severity to block a
    model on.
    """
    findings = []
    for index, word in enumerate(problem.boundary):
        if word != "MUR":
            continue
        dim, side = index // 2, index % 2
        wall = float(problem.grid[dim][0 if side == 0 else -1])
        for port in problem.ports:
            if not port.excite:
                continue
            low, high = sorted((port.start[dim], port.stop[dim]))
            if not low <= wall <= high:
                continue
            face = f"{AXIS_NAMES[dim]}{'min' if side == 0 else 'max'}"
            findings.append(
                Finding(
                    WARN,
                    f"boundary {face}",
                    f"it is Mur and port {port.number} reaches it, so the "
                    "source may sit on the wall itself. openEMS then holds "
                    "that wall shut until the drive is over, and this "
                    "adapter's drive lasts the whole run - the wall would "
                    "reflect for the entire solve rather than absorb. Put PML "
                    f"on {face}, or move the port clear of it",
                )
            )
    return findings


def _check_absorber_agrees(problem: Problem, index: int, face: str, asked: int) -> list[Finding]:
    """The boundary's PML depth against the cells the mesher set aside for it.

    Two numbers describe one depth, and they are arrived at independently.
    ``PML_<n>`` is what openEMS is told to absorb in. ``pml_cells`` is what the
    mesher laid the block in, and what every clearance check in this package
    measures against. Left uncompared, a mesh reserving one depth against a
    boundary absorbing in another passes in silence.

    openEMS absorbs in the cells it was told to, whatever the mesher intended.
    If the two differ, the interior is not where this package places it, and the
    clearance findings, which are the ones that name objects, are then measured
    against the wrong edge. A deeper absorber can make the answer better, so the
    disagreement produces no symptom of its own.

    A reservation of nothing against a boundary that absorbs is the quietest
    case rather than an exempt one: every clearance check here then has no
    absorber to measure against and says nothing at all, while openEMS
    attenuates the field at that wall in the cells it was told to use.
    """
    dim = index // 2
    reserved = _absorber_cells(problem.grid, dim)
    if reserved == asked:
        return []

    axis = AXIS_NAMES[dim]
    if not reserved:
        consequence = "every clearance checked here is measured as though that wall did not absorb"
    else:
        direction = "deeper into the model than" if asked > reserved else "short of"
        consequence = (
            f"the absorber reaches {direction} the band set aside for it, and "
            "every clearance checked here is measured against the other edge"
        )
    return [
        Finding(
            WARN,
            f"boundary {face}",
            f"it asks for {asked} cells of PML and the mesh set aside "
            f"{reserved} on {axis}. openEMS absorbs in the {asked} it was told "
            f"to, so {consequence}. Set PMLCells and the boundary from one "
            "number",
        )
    ]
