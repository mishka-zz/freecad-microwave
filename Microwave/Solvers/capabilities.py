# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The form an adapter declares itself in.

Each adapter states what it can express, and the workbench reads those
declarations to say which solvers could answer a model. So the form is shared
and the declarations are not: this module holds the shape, and each adapter's
own ``capabilities`` module holds its instance of it.

It sits here rather than under an adapter because an adapter never imports
another. Pure data, and the standard library only, so a declaration can be read
on a machine where no engine is installed.

The names in a declaration describe modelling concepts rather than any solver's
own surface. A microstrip port is a thing a user draws, and what each backend
builds it out of is that backend's business.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Capabilities:
    """One adapter's declaration of what it can express."""

    solver: str
    port_types: frozenset[str]
    materials: frozenset[str]
    domains: frozenset[str]
    outputs: frozenset[str]
    excitations: frozenset[str]
    geometry: frozenset[str]
    notes: dict[str, str] = field(default_factory=dict)

    def supports_port(self, port_type: str) -> bool:
        return port_type in self.port_types

    def supports_material(self, kind: str) -> bool:
        return kind in self.materials

    def supports_output(self, output: str) -> bool:
        return output in self.outputs
