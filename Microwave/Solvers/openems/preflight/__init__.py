# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Check a problem against what this adapter can actually do, before running.

Refuse, warn, or substitute. A silent no-op is never one of them - an
unsupported object produces a refusal naming the object, not an empty section
in the output file.

Runs on the FreeCAD side, and imports neither FreeCAD nor openEMS, so the whole
check is unit-testable without either.

The checks are grouped by what they ask about - :mod:`.materials`,
:mod:`.grid`, :mod:`.absorber`, :mod:`.ports`, :mod:`.probes`, :mod:`.solve` -
and each is a plain function from a :class:`~..model.Problem` to a list of
:class:`~.finding.Finding`. No check :func:`check` dispatches to calls another,
and none of them holds state, so the order below is the order findings are
collected in and nothing else. A new check is a function in the module that
matches its subject, plus one line in :func:`check`.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..capabilities import Capabilities, capabilities
from ..model import Problem
from . import absorber, grid, materials, ports, probes, solve

# Not in the line above: ``finding`` is the loop variable in half the functions
# here, so binding the module under that name shadows it. The submodule is
# reachable as ``preflight.finding`` regardless, which is how it is imported.
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

    Returns findings rather than raising, so the task panel can show the user
    all of them at once instead of one per attempt. :func:`refuse_if_blocked`
    turns them into an exception when a run is actually being started.

    Grouped here rather than by whoever displays them, for the reason the
    driver re-runs this whole function: a step every route already passes
    through is the only place a rule is not opt-in. The panel, the driver's
    ``CHECK`` markers and the refusal exception all render ``str(finding)``,
    so they get it without knowing about it.
    """
    caps = caps or capabilities()
    findings: list[Finding] = []

    findings += materials._check_materials(problem, caps)
    findings += materials._check_the_loss_was_measured_in_this_band(problem)
    findings += materials._check_coincident_solids(problem)
    findings += ports._check_lumped_excitation_beside_a_measured_line(problem)
    findings += materials._check_sheet_thickness(problem)
    findings += materials._check_sheet_fits_the_surface_impedance_model(problem)
    findings += grid._check_cells_per_wavelength(problem)
    findings += grid._check_the_grid_is_a_size_somebody_meant(problem)
    findings += grid._check_grid_covers_the_model(problem)
    findings += absorber._check_the_absorber_stands_on_the_structure(problem)
    findings += absorber._check_the_absorber_leaves_a_model(problem)
    findings += ports._check_ports(problem, caps)
    findings += absorber._check_boundary(problem)
    findings += solve._check_reproducibility(problem)
    findings += solve._check_timestep_factor(problem)
    findings += solve._check_the_excitation_fits_the_run(problem)
    findings += probes._check_probes(problem)
    findings += probes._check_probes_clear_of_the_feed(problem)

    order = {REFUSE: 0, WARN: 1, SUBSTITUTE: 2}
    return sorted(_grouped(findings), key=lambda f: order.get(f.severity, 3))


def refusals(findings: Sequence[Finding]) -> list[Finding]:
    return [finding for finding in findings if finding.severity == REFUSE]


def object_count(findings: Sequence[Finding]) -> int:
    """How many objects ``findings`` are about.

    What a reader means by "three warnings" is three objects, and after
    grouping a finding stands for as many as shared its sentence. Counting the
    findings themselves would report one warning for a wall five solids run
    through, which understates it by four.
    """
    return sum(len(finding.subjects) for finding in findings)


def refuse_if_blocked(findings: Sequence[Finding]) -> None:
    """Raise if anything refuses. Called at the start of a run, not of a write."""
    blocking = refusals(findings)
    if blocking:
        detail = "\n  ".join(str(finding) for finding in blocking)
        raise UnsupportedModel(f"openEMS cannot run this model:\n  {detail}")


class UnsupportedModel(Exception):
    """The model asks for something this adapter has refused."""
