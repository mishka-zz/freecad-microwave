# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The grid every mesh test is built on, and the assertions made about it.

``test_mesh_core``, ``test_mesh_report`` and ``test_mesh_preview`` all mesh the
same microstrip stackup - a 16 mm square of FR4 on a ground plane, inside a
20x20x10 mm domain - because it is the smallest structure with all three of a
dielectric, a zero-thickness conductor and an air gap. One copy, because
separate ones drift: spell ``max_ratio`` out in one and take the default in
another, and the two agree until the day the default moves, then mesh different
grids and say nothing.

Not named ``test_*``, so pytest collects nothing from it.
"""

from __future__ import annotations

import numpy as np

from Microwave.Solvers.openems.mesh import (
    MaterialClass,
    MeshParams,
    Region,
)

#: 20 x 20 x 10 mm, comfortably larger than the stackup so the absorber has
#: somewhere to sit without touching the board.
DOMAIN = ((-10.0, -10.0, -5.0), (10.0, 10.0, 5.0))


def params(**overrides) -> MeshParams:
    """The default policy, spelled out. Every value here is ``MeshParams``' own
    default; they are written rather than inherited so a test reading this file
    sees the grid it is going to get."""
    settings = dict(
        metal_res=0.2,
        dielectric_res=1.0,
        max_ratio=(1.3, 1.3, 1.3),
        min_lines=4,
        pml_cells=8,
    )
    settings.update(overrides)
    return MeshParams(**settings)


def substrate(**overrides) -> Region:
    settings = dict(
        lower=(-8.0, -8.0, 0.0),
        upper=(8.0, 8.0, 1.6),
        material=MaterialClass.DIELECTRIC,
        label="Substrate",
    )
    settings.update(overrides)
    return Region(**settings)


def ground_sheet() -> Region:
    return Region(
        lower=(-8.0, -8.0, 0.0),
        upper=(8.0, 8.0, 0.0),
        material=MaterialClass.METAL,
        label="GroundPlane",
    )


def stackup() -> list[Region]:
    """Substrate over ground: the pair, in the order the mesher receives them."""
    return [substrate(), ground_sheet()]


def has_line(lines: np.ndarray, position: float, tol: float = 1e-9) -> bool:
    return bool(np.any(np.isclose(lines, position, rtol=0.0, atol=tol)))


def cell_at(lines: np.ndarray, position: float) -> float:
    """The width of the cell containing ``position``.

    For anything that is not pinned to the grid - a refinement box, which
    contributes constraints and no lines - this is the honest measurement.
    Counting lines between two unpinned faces is off by up to one cell at each
    end, which is the whole feature on a box only a few cells thick.
    """
    index = int(np.searchsorted(lines, position))
    assert 0 < index < len(lines), f"{position} is outside the grid"
    return float(lines[index] - lines[index - 1])


def pin_at(pins, position: float, tol: float = 1e-9):
    """The one pinned line at ``position``. Asserts there is exactly one.

    Exactness is the point: two pins at one coordinate means a preference was
    reported alongside the anchor that displaced it.
    """
    found = [pin for pin in pins if abs(pin.position - position) <= tol]
    assert len(found) == 1, (
        f"expected exactly one pinned line at {position}, found "
        f"{[(p.position, p.source) for p in found]}"
    )
    return found[0]


def assert_graded_within(lines, ratio: float) -> None:
    """No two adjacent cells on any axis differ by more than ``ratio``."""
    for dim in range(3):
        spacings = np.diff(lines[dim])
        observed = np.maximum(spacings[1:] / spacings[:-1], spacings[:-1] / spacings[1:])
        assert np.max(observed) <= ratio * (1 + 1e-9)
