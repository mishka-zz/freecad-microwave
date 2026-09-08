# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The absorbing block at each face the structure runs out through.

openEMS terminates a domain with a PML of a whole number of cells, and it
reflects unless every one of them is the same size. So the block is not meshed
like the rest of the domain: its cells are laid at one pitch, and the interior
stops short of the declared wall by however deep the block is.

The arithmetic here is what that costs on an axis: which faces the structure
runs out through, where the interior ends, laying the block's own cells back
onto the interior's lines, how deep what was laid came out, whether the cell
meeting it grades into it, and whether what was laid is a block at all. How
deep one cell should be is a different question, because it is read off the
sizing field the structure asks for; :func:`~.mesh.absorber_pitches` asks it.

What a function here is given is line positions, or the bounds and the policy
they came from. :func:`_add_absorber` lays the block's own lines, which is the
only placement here, and where the interior's lines go is settled before it
runs. No region is read and no demand is raised, so nothing here reaches back
into the mesher.

docs/internals/domain-and-absorber.md works out why the block comes out of the
domain rather than being added beyond it, and what that costs the user who
drew the domain.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .grid import MeshLines
from .regions import _DIM_NAMES, DIMENSIONS, MeshError, MeshParams


def laid_pitches(
    lines: MeshLines,
    params: MeshParams,
    pitches: Sequence[tuple[float | None, float | None]],
) -> tuple[tuple[float | None, float | None], ...]:
    """The cell the interior actually laid beside each block, ready to lay again.

    A block sized from the sizing field and the interior beside it do not come
    out the same size. The field is not the cause. A gap holds a whole number of
    cells, so a segment wanting a fraction over ``n`` lays ``n+1``, and every
    cell in it comes out proportionally finer than asked; and a cell laid from
    the seam spans a field already relaxing away from the demand there, so it
    comes out coarser than the demand. Neither factor is the block's to remove.
    :func:`rough_seams` therefore holds the two to the grading budget rather
    than to each other, and this is what a block is laid again at where they do
    not meet.

    Faces whose absorber goes outside the domain are left alone. There the block
    already copies the interior's edge pitch, and there is nothing to agree
    about.
    """
    out: list[tuple[float | None, float | None]] = []
    for dim in range(DIMENSIONS):
        cells = params.absorber[dim]
        axis = np.asarray(lines[dim], dtype=float)
        spacing = np.diff(axis)
        out.append(
            (
                None if pitches[dim][0] is None else float(spacing[cells]),
                None if pitches[dim][1] is None else float(spacing[-1 - cells]),
            )
        )
    return tuple(out)


def rough_seams(
    laid: Sequence[tuple[float | None, float | None]],
    pitches: Sequence[tuple[float | None, float | None]],
    params: MeshParams,
) -> list[str]:
    """Which blocks the interior does not grade into, named face by face.

    The block and the interior beside it are held to ``max_ratio``, the same
    budget every other pair of adjacent cells is held to, rather than to being
    the same size. They cannot be made the same size: a gap holds a whole number
    of cells, so every cell in it comes out scaled by the arclength that number
    leaves it, and laying the interior's cell back as the block's pitch carries
    that factor into the next pass along with a fresh one.

    The interior misses the block either way about, and the loop moves the block
    in both directions. It comes out finer where a feature just inside the seam
    sets the field there instead of the block's own demand. It comes out coarser
    where the field relaxes away from that demand across the first cell: the
    pitch is demanded at the seam, so the field there is at most the pitch and
    rises at ``slope``, and a cell carrying one unit of arclength across such a
    field measures at most ``pitch * (exp(slope) - 1) / slope`` - which is
    inside ``max_ratio`` for every ratio a policy admits, since
    ``(r - 1) / ln r < r`` above one.

    A cell carries more than one unit of arclength where :func:`~.sizing_field._cell_count` had
    to round a gap's count down to keep its cells off the floor, and there the
    interior can be coarser than that bound. A cell floor comparable to what the
    mesh wants beside the block is what reaches it.
    """
    rough: list[str] = []
    for dim in range(DIMENSIONS):
        limit = params.max_ratio[dim] * (1.0 + 1e-6)
        for cell, pitch, face in zip(laid[dim], pitches[dim], ("lower", "upper")):
            if cell is None or pitch is None:
                continue
            if max(cell, pitch) / min(cell, pitch) > limit:
                rough.append(
                    f"the {_DIM_NAMES[dim]} axis {face} face lays {cell:.6g} against "
                    f"a block of {pitch:.6g}"
                )
    return rough


def absorber_walls(
    lower: float, upper: float, absorbed: tuple[bool, bool]
) -> tuple[tuple[float, float], ...]:
    """Where the structure runs out through the absorber, and which way is in.

    The direction is carried because a wall decides whether a demand lies at the
    cut or reaches past it into the model, and the two ends of an axis answer
    that with opposite inequalities.
    """
    return tuple(
        (bound, inward)
        for bound, inward, taken in ((lower, 1.0, absorbed[0]), (upper, -1.0, absorbed[1]))
        if taken
    )


def inside_the_absorber(
    domain: tuple[tuple[float, float, float], tuple[float, float, float]],
    params: MeshParams,
    pitches: Sequence[tuple[float | None, float | None]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """``domain`` with each absorbed face's block taken off the inside.

    The grid then ends on the structure by arithmetic rather than by prediction.
    The interior starts one block in, and :func:`_add_absorber` lays that same
    block back at the same pitch.
    """
    lower, upper = list(domain[0]), list(domain[1])
    for dim in range(DIMENSIONS):
        cells = params.absorber[dim]
        low, high = pitches[dim]
        if low is not None:
            lower[dim] += cells * low
        if high is not None:
            upper[dim] -= cells * high
        if upper[dim] <= lower[dim]:
            raise MeshError(
                f"the absorber would consume the whole {_DIM_NAMES[dim]} axis: the "
                f"structure spans {domain[0][dim]:.4g} to {domain[1][dim]:.4g} and "
                f"{cells} absorber cells at each end take all of it. Coarsen the "
                "mesh there, ask for fewer absorber cells, or give the face air"
            )
    return (tuple(lower), tuple(upper))  # type: ignore[return-value]


def _add_absorber(
    interior: Sequence[float],
    pml_cells: int,
    pitches: tuple[float | None, float | None] = (None, None),
) -> list[float]:
    """Extend the axis with uniformly spaced absorber cells at both ends.

    Uniform by construction, and it has to be. A graded block reflects, and the
    reflection off it is indistinguishable from the one being measured.

    A face given ``None`` takes the interior's own edge pitch, which is the
    absorber standing in the air beyond the model. A face given a pitch was
    sized by :func:`~.mesh._absorber_cell` before the interior was meshed, and the
    interior was cut short by exactly ``pml_cells`` of it. Laying it back here
    returns the axis to the extent the structure was drawn at, whatever the
    mesher did in between.
    """
    if pml_cells == 0:
        return list(interior)
    if len(interior) < 2:
        raise MeshError("cannot add an absorber to an axis with fewer than two lines")

    lines = list(interior)
    low_pitch = lines[1] - lines[0] if pitches[0] is None else pitches[0]
    high_pitch = lines[-1] - lines[-2] if pitches[1] is None else pitches[1]
    below = [lines[0] - low_pitch * i for i in range(pml_cells, 0, -1)]
    above = [lines[-1] + high_pitch * i for i in range(1, pml_cells + 1)]
    return below + lines + above


def _validate_absorber(array: np.ndarray, spacings: np.ndarray, name: str, pml_cells: int) -> None:
    if pml_cells == 0:
        return
    if array.size < 2 * pml_cells + 2:
        raise MeshError(
            f"{name} axis has {array.size} lines, too few for {pml_cells} "
            "absorber cells at each end plus an interior"
        )
    for label, block in (
        ("lower", spacings[:pml_cells]),
        ("upper", spacings[-pml_cells:]),
    ):
        if not np.allclose(block, block[0], rtol=1e-9, atol=0.0):
            raise MeshError(f"{name} axis {label} absorber is not uniformly spaced: {block}")
