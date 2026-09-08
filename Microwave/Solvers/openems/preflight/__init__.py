# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Check a problem against what this adapter can do, before running it.

A check refuses, warns, or substitutes. It never passes silently over an
unsupported object: that object produces a refusal naming it, rather than an
empty section in the output file.

These modules run on the FreeCAD side and import neither FreeCAD nor openEMS,
so the whole check is unit-testable without either.

The checks are grouped by subject - :mod:`.materials`, :mod:`.grid`,
:mod:`.absorber`, :mod:`.ports`, :mod:`.precision`, :mod:`.probes`,
:mod:`.solve`. Each is a plain function from a :class:`~..model.Problem` to a
list of :class:`~.finding.Finding`. No check listed in :func:`check` calls
another one listed there, and none of them holds state, so the order below sets
only the order findings are collected in. To add a check, write a function in the module
for its subject and add one line to :func:`check`.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..capabilities import Capabilities, capabilities
from ..model import Problem
from . import absorber, grid, materials, ports, precision, probes, solve

# ``finding`` is left out of the line above: it is the loop variable in most of
# the functions here, so binding the module under that name shadows it. The
# submodule stays reachable as ``preflight.finding``, which is how it is
# imported.
from .finding import REFUSE, SUBSTITUTE, WARN, Finding, _grouped, hz
from .materials import FAR

__all__ = [
    "FAR",
    "REFUSE",
    "SUBSTITUTE",
    "WARN",
    "Finding",
    "UnsupportedModel",
    "check",
    "hz",
    "object_count",
    "refusals",
    "refuse_if_blocked",
]


def check(problem: Problem, caps: Capabilities | None = None) -> list[Finding]:
    """Everything wrong with ``problem``, most severe first.

    This returns findings rather than raising, so the task panel can show the
    user all of them at once instead of one per attempt. The driver and the task
    panel both read :func:`refusals` and act on the list;
    :func:`refuse_if_blocked` is there for a caller that wants the refusal as an
    exception instead.

    Findings are grouped here rather than by each caller that displays them.
    Every route passes through this function, and the driver re-runs it, so a
    rule placed here is not opt-in. The panel, the driver's ``CHECK`` markers
    and the refusal exception all render ``str(finding)``, so each gets the
    grouping without implementing it.
    """
    caps = caps or capabilities()
    findings: list[Finding] = []

    findings += materials._check_materials(problem, caps)
    findings += materials._check_the_loss_was_measured_in_this_band(problem)
    findings += materials._check_the_band_fits_one_conductivity(problem)
    findings += materials._check_coincident_solids(problem)
    findings += materials._check_a_thickness_was_invented(problem)
    findings += materials._check_a_conducting_sheet_spans_a_surface(problem)
    findings += materials._check_sheet_thickness(problem)
    findings += ports._check_lumped_excitation_beside_a_measured_line(problem)
    findings += ports._check_the_launch_direction_was_read(problem)
    findings += materials._check_sheet_fits_the_surface_impedance_model(problem)
    findings += grid._check_cells_per_wavelength(problem)
    findings += grid._check_the_grid_is_a_size_somebody_meant(problem)
    findings += grid._check_grid_covers_the_model(problem)
    findings += grid._check_conductors_are_resolved_across(problem)
    findings += absorber._check_the_absorber_fits_the_axis(problem)
    findings += absorber._check_the_absorber_stands_on_the_structure(problem)
    findings += absorber._check_the_absorber_leaves_a_model(problem)
    findings += ports._check_ports(problem, caps)
    findings += absorber._check_boundary(problem)
    findings += absorber._check_a_mur_wall_carries_no_excitation(problem)
    findings += solve._check_reproducibility(problem)
    findings += solve._check_timestep_factor(problem)
    findings += solve._check_the_excitation_fits_the_run(problem)
    findings += probes._check_probes(problem)
    findings += probes._check_probes_clear_of_the_feed(problem)
    findings += precision._check_vertices_survive_single_precision(problem)

    order = {REFUSE: 0, WARN: 1, SUBSTITUTE: 2}
    return sorted(_grouped(findings), key=lambda f: order.get(f.severity, 3))


def refusals(findings: Sequence[Finding]) -> list[Finding]:
    return [finding for finding in findings if finding.severity == REFUSE]


def object_count(findings: Sequence[Finding]) -> int:
    """How many objects ``findings`` are about.

    A reader who says "three warnings" means three objects. After grouping, one
    finding stands for every object that shared its sentence. Counting the
    findings themselves would report one warning for a wall that five solids
    run through, which understates it by four.
    """
    return sum(len(finding.subjects) for finding in findings)


def refuse_if_blocked(findings: Sequence[Finding]) -> None:
    """Raise if any finding refuses.

    For a caller that wants the refusals as one exception rather than a list to
    render. Every blocking finding goes into the message. The driver and the
    task panel read :func:`refusals` themselves, having reported every finding
    first.
    """
    blocking = refusals(findings)
    if blocking:
        detail = "\n  ".join(str(finding) for finding in blocking)
        raise UnsupportedModel(f"openEMS cannot run this model:\n  {detail}")


class UnsupportedModel(Exception):
    """The model asks for something this adapter has refused."""
