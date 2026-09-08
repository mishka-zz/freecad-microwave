# Workbench Internals

This directory documents the mathematical, physical, and algorithmic derivations
underlying the workbench, intended for contributors and maintainers modifying the
codebase. User-facing documentation resides in the parent `docs/` directory.

The documents here explain design decisions that cannot be reconstructed solely
from the source code, detailing why specific numerical methods and boundary
approximations were chosen over alternatives.

| Page | What it covers |
|---|---|
| [Cell allocation across three axes](cell-allocation.md) | Resolving geometric lengths across independently sized Cartesian grid axes |
| [Deciding where the grid lines go](sizing-field.md) | Grid anchors, sizing field formulation, arclength placement, and symmetry |
| [Measuring what the grid has to resolve](feature-size.md) | Extracting local feature sizes and gap widths from CAD models |
| [Conductor width discretization and warning thresholds](conductor-width.md) | Conductor width discretization, edge treatment, and face sizing |
| [Where the domain ends, and where the absorber goes](domain-and-absorber.md) | Boundary padding, absorbing boundary layer (PML) depth, and domain sizing |
| [Building one S-matrix out of several runs](s-matrix-from-runs.md) | Assembling multi-port S-parameters from sequential solves and applying symmetry |
| [Extracting phase velocity from transmission phase](velocity-from-phase.md) | Deriving phase velocity and distance axes from broadband phase data |

## Purpose

The workbench translates FreeCAD geometry into numerical models for
electromagnetic solvers. In particular, openEMS discretizes structures onto a
rectilinear Yee grid. Because openEMS evaluates fields without alerting the user
to subtle discretization artifacts (such as artificial gaps, mode splitting, or
detuning), the grid generation rules must guarantee correct physical behavior.

These documents record:
- The derivations, benchmark findings, and trade-offs that established each rule.
- Alternative approaches that were evaluated and rejected, along with the reasons
  for their failure.
- Cross-cutting architectural constraints that span multiple modules.

## Scope: Internal docs vs. Source code

**Documented in `docs/internals/`:**
- Mathematical derivations, asymptotic error models, and design trade-offs.
- Multi-module problem formulations.

**Documented in source code:**
- Function arguments, return types, and exceptions (Python docstrings).
- Implementation details and specific line rationale (inline comments).
- Upstream engine and CAD kernel quirks, cited with relevant upstream file and
  line references.

## Maintenance and verification

To prevent documentation from drifting out of date with the codebase:
- Automated tests verify that all `docs/internals/...` citations and section
  anchors referenced in code and comments remain valid (`test_doc_references.py`).
- Numeric benchmarks and measured values are maintained in automated test suites
  and regression gates rather than hardcoded in prose.
- Derivations are kept concise, following standard RF engineering and mathematical
  notation.
