# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Checks about the absorbing boundary, and where it reaches.

A PML is grid, not empty space: it eats cells from the ends of every axis it is
declared on. These ask whether it stands on something it can absorb, whether it
leaves a model behind after taking its share, whether each boundary is a word
openEMS knows, and whether the depth a ``PML_<n>`` asks for is the ``pml_cells``
the mesher actually set aside.

The three helpers here answer "how deep does the absorber reach on this axis",
which :mod:`.ports` needs too - a port inside the PML is measured in a medium
that is not the guide.
"""

from __future__ import annotations

from ..model import AXIS_NAMES, DIMENSIONS, MeshGrid, Problem
from ..write import structure_bounds, through_faces_past_the_structure
from .finding import REFUSE, WARN, Finding

#: openEMS' boundary vocabulary. ``PML_n`` is matched separately.
_BOUNDARY_WORDS = frozenset({"PEC", "PMC", "MUR"})


#: Below this share of a drawn axis left as interior, say so. Not a tuned
#: constant: at one half the absorber has taken more of the model than it left,
#: which is the point past which "the domain is smaller than you drew" stops
#: being a detail. The microstrip example reads 0.76 and is silent; the same
#: board over 2.4-2.5 GHz reads 0.04.
_INTERIOR_SHARE_LIMIT = 0.5


def _check_the_absorber_stands_on_the_structure(problem: Problem) -> list[Finding]:
    """The overrun half of the THROUGH question, on an envelope nobody meshed.

    ``write`` refuses this while meshing, which covers every route through the
    workbench. It does not cover the driver: ``python -m ...driver`` reads an
    envelope, checks it, and builds - so a grid that came from anywhere else,
    a bug report replayed by hand or a file edited to reproduce something, is
    solved without the comparison ever running. The driver re-runs pre-flight
    precisely so no guard is opt-in, and this one was.

    The other half is :func:`~.grid._check_grid_covers_the_model`, which reports
    geometry the grid ends short of. Between them the grid's outer edge is
    judged against the structure in both directions.
    """
    findings = through_faces_past_the_structure(
        problem.grid, problem.solids, problem.ports, problem.grid.params.get("padding")
    )
    return [Finding(REFUSE, subject, message) for subject, message in findings]


def _check_the_absorber_leaves_a_model(problem: Problem) -> list[Finding]:
    """A THROUGH face pulls the domain in. Say when it pulls in nearly all of it.

    ``write.domain`` reserves ``pml_cells`` cells at each such face, and a cell
    is a wavelength over ``ElementsPerWavelength`` - so the reservation grows
    as the band falls or the mesh is *coarsened*, while the structure the user
    drew does not. Coarsening the mesh therefore shrinks the domain, which is
    backwards, and the cell count is blind to it - it is unchanged whether the
    domain holds most of the board or a sliver of it.

    A warning and not a refusal, because a small interior is not by itself
    wrong - a long line measured only in the middle is legitimate. The question
    that decides it, whether everything being measured still fits, is asked by
    :func:`~.ports._check_port_clear_of_absorber` and
    :func:`~.ports._check_port_inside_the_grid`, which refuse and name the
    absorber's depth themselves. What is left here is the case with no port in
    the absorber at all, where the only symptom is a model most of which is not
    being solved.

    Only faces that pull in are described: an outward-padded face has a domain
    larger than the structure by design, and a share above one describes
    nothing there.

    The real fix is to stop measuring the domain in cells whose size is decided
    afterwards.
    """
    grid = problem.grid
    lower, upper = structure_bounds(problem.solids, problem.ports)

    findings = []
    for dim in range(DIMENSIONS):
        interior = _absorber_bounds(grid, dim)
        if interior is None:
            continue
        drawn = float(upper[dim]) - float(lower[dim])
        if drawn <= 0:
            continue
        share = (interior[1] - interior[0]) / drawn
        if share >= _INTERIOR_SHARE_LIMIT:
            continue

        # The interior against the structure, not the absorber as laid: what
        # removes model is the *reservation*, and on a collapsed domain the two
        # differ by threefold. Ends are described one at a time because an axis
        # can be pulled in at one end and padded outward at the other, and a
        # padded end gives up nothing.
        taken = [
            f"{gap:.4g} mm at {AXIS_NAMES[dim]}={end}"
            for end, gap in (
                ("min", interior[0] - float(lower[dim])),
                ("max", float(upper[dim]) - interior[1]),
            )
            if gap > 0
        ]
        per_end = (
            f" The absorber reserves {_absorber_cells(grid, dim)} cells and "
            f"takes {' and '.join(taken)} off a {drawn:.4g} mm model."
            if taken
            else ""
        )
        findings.append(
            Finding(
                WARN,
                f"{AXIS_NAMES[dim]} domain",
                f"the absorber leaves {share:.0%} of the structure as interior."
                f"{per_end} Everything measured has to fit inside that. Draw "
                f"more structure, raise ElementsPerWavelength, or lower "
                f"PMLCells - note that *coarsening* the mesh shrinks the "
                f"domain rather than growing it",
            )
        )
    return findings


def _absorber_cells(grid: MeshGrid, dim: int) -> int:
    """How many cells of absorber this axis carries at each end, or zero.

    ``pml_cells`` is per axis: a closed structure absorbs on some axes and is
    walled by a conductor on the others.
    """
    cells = grid.params.get("pml_cells")
    if isinstance(cells, (list, tuple)):
        cells = cells[dim] if len(cells) > dim else 0
    if not cells:
        return 0
    if 2 * cells >= len(grid[dim]) - 1:
        return 0
    return int(cells)


def _absorber_bounds(grid: MeshGrid, dim: int) -> tuple[float, float] | None:
    """Where the interior starts and ends on one axis, absorber excluded."""
    cells = _absorber_cells(grid, dim)
    if not cells:
        return None
    lines = grid[dim]
    return float(lines[cells]), float(lines[-1 - cells])


def _absorber_depth(grid: MeshGrid, dim: int) -> tuple[float, float]:
    """How deep the absorber reaches at each end of an axis, in mm, as laid.

    Measured off the grid, not recomputed from what ``write.domain`` reserved.
    The two are close on a healthy model and are *not* the same quantity: the
    reservation is counted in a cell size chosen before the mesh exists, while
    the absorber is laid at the interior's own edge pitch. Where the interior
    has collapsed the gap is wide - 48 mm reserved against 16 mm laid, on the
    fixture in ``TestTheAbsorberMustLeaveAModel``.

    So this answers "how far in does the absorber reach", which is the question
    a point *inside* it raises, and it is the wrong number for "how much model
    did the domain give up", which is the interior against the structure. Only
    the reservation removes structure.
    """
    cells = _absorber_cells(grid, dim)
    if not cells:
        return 0.0, 0.0
    lines = grid[dim]
    return (
        float(lines[cells]) - float(lines[0]),
        float(lines[-1]) - float(lines[-1 - cells]),
    )


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


def _check_absorber_agrees(problem: Problem, index: int, face: str, asked: int) -> list[Finding]:
    """The boundary's PML depth against the cells the mesher set aside for it.

    Two numbers for one thing, arrived at independently: ``PML_<n>`` is what
    openEMS is told to absorb in, and ``pml_cells`` is what ``write.domain``
    added outside the structure and what every clearance check in this package
    measures against. Left uncompared, a mesh reserving 8 against a boundary
    asking for 24 goes through in silence.

    openEMS absorbs in the cells it was told to, whatever the mesher intended,
    so the two differing means the interior is not where this package thinks it
    is - and the clearance findings, which are the ones that name objects,
    are then measured against the wrong edge. On that particular model the
    deeper absorber made the answer *better*, which is exactly why it needed
    saying rather than leaving to be noticed.
    """
    dim = index // 2
    reserved = _absorber_cells(problem.grid, dim)
    if not reserved or reserved == asked:
        return []

    direction = "deeper into the model than" if asked > reserved else "short of"
    return [
        Finding(
            WARN,
            f"boundary {face}",
            f"it asks for {asked} cells of PML and the mesh reserved "
            f"{reserved} on {AXIS_NAMES[dim]}. openEMS absorbs in the "
            f"{asked} it was told to, so the absorber reaches {direction} the "
            "band set aside for it, and every clearance checked here is "
            "measured against the other edge. Set PMLCells and the boundary "
            "from one number",
        )
    ]
