# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Solve a gate's cases when something asks for them, and once each.

A gate holds several arrangements of one problem: a refinement sequence, the
same mesh slid under the drawing, the line turned onto another axis. Most of
them are there for a comparison rather than for the answer, and a fixture that
loops over the whole set before handing any of it back makes every assertion in
the file cost what all of them cost - so a selection asking for one arrangement
pays for the whole set, and no marker can change that.

Here the cost follows the demand. What a run solves is what the tests it
selected named, and a test names its cases by asking for them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

import pytest

__all__ = ["Cases", "parameters"]

Case = TypeVar("Case")


def parameters(names: Iterable[str], nominal: str) -> list:
    """Every case as a test parameter, with all but ``nominal`` held back.

    A property asserted of one solved case is written once and run at every case
    a gate holds. What separates the tiers is which of those a run solves, and
    this is where that is decided: the operating point on every run, the rest
    behind the release marker.

    Derived from the one name rather than listed, so a case added to a gate lands
    in the tier its cost belongs to without anything else being edited.
    """
    named = sorted(names)
    assert nominal in named, f"{nominal!r} is not one of the cases: {named}"
    return [
        pytest.param(name, marks=() if name == nominal else pytest.mark.release) for name in named
    ]


class Cases(Generic[Case]):
    """Each case, measured on first use and kept for the rest of the module."""

    def __init__(self, measure: Callable[[str], Case]):
        self._measure = measure
        self._known: dict[str, Case] = {}

    def __call__(self, name: str) -> Case:
        if name not in self._known:
            self._known[name] = self._measure(name)
        return self._known[name]
