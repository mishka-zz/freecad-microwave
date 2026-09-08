# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What every check shares: how it reports, and when two coordinates are one place.

A check returns :class:`Finding` objects. It never prints, refuses or raises.
:func:`~.check` collects them, and the callers above render ``str(finding)``.

:data:`_ON_THE_GRID` is defined here because several checks have to agree on it.
Two modules that define one tolerance separately will eventually disagree about
it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

REFUSE = "refuse"


WARN = "warn"


SUBSTITUTE = "substitute"


#: How far off a grid line a coordinate may sit and still count as on it, in mm.
#: By the same argument, two coordinates this far apart are one place. The value
#: is one picometre, far below any length this workbench meshes.
#:
#: What it has to cover is the envelope's rounding rather than the mesher's. A
#: pinned position is written back literally while an axis is meshed, and
#: ``mesh._validate`` asks for it exactly. A stored envelope has been through
#: :func:`~..model.canonical`, which keeps ``CANONICAL_DIGITS`` significant
#: digits, so a coordinate is on disk to a part in 1e11 of itself and comes back
#: half of that away from where it was meshed. That is under a picometre while
#: the coordinate is under a metre, and over it above.
_ON_THE_GRID = 1e-9


@dataclass(frozen=True, init=False)
class Finding:
    """One thing wrong, or worth saying, about a problem.

    ``subjects`` holds one entry per object. It is a tuple because one sentence
    is routinely true of several objects. A check whose sentence is about one
    object passes its name and gets a one-tuple; a check whose sentence is about
    two passes both, and :func:`_grouped` builds the longer ones. Passing a bare
    string is a convenience for the call sites, and it is the only coercion
    done. Nothing else is validated, since every caller is in
    this package.

    A subject is a label to show the user rather than a key to look an object up
    by. After merging, the subjects name every object the sentence was said of.

    Duplicates are kept. ``Solid.name`` falls back to the material when a solid
    has no label, so two objects can share a name, and dropping the repeat here
    would make :func:`object_count` undercount the case it exists for. The
    repeat is collapsed for display instead, in :attr:`subject`.
    """

    severity: str
    subjects: tuple[str, ...]
    message: str

    # Written out rather than generated. A lone name is stored as a tuple, and a
    # dataclass states one type for both the argument and the attribute. A
    # generated signature would have to declare either the tuple, which is wrong
    # at every call site passing a name, or the union, which then travels to
    # every reader of `subjects`.
    def __init__(self, severity: str, subjects: tuple[str, ...] | str, message: str) -> None:
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "subjects", (subjects,) if isinstance(subjects, str) else subjects)
        object.__setattr__(self, "message", message)

    @property
    def subject(self) -> str:
        """The subjects as a reader wants them: ``'A'``, ``'A and B'``, ``'A, B and C'``.

        Each distinct name appears once. Two objects under one name are two
        objects and one word, since "Copper and Copper" says no more than the
        singular.
        """
        names = list(dict.fromkeys(self.subjects))
        if len(names) == 1:
            return names[0]
        return f"{', '.join(names[:-1])} and {names[-1]}"

    def __str__(self) -> str:
        return f"[{self.severity}] {self.subject}: {self.message}"


def _grouped(findings: Sequence[Finding]) -> list[Finding]:
    """One finding per distinct thing said, naming every object it was said of.

    A check that walks the model repeats its sentence once per object it holds
    for, so the number of lines follows the size of the model rather than the
    number of things wrong with it. A reader who meets every copy learns to skip
    the section, and the section is there to be read.

    Merging is on the message, so two objects are named together only when the
    sentence is identical down to the coordinates in it. Two solids overhanging
    the same wall at different distances have different sentences and stay
    apart, because one line quoting one position for both would be false. A
    message must therefore carry no numeral counting its own subjects: "both" is
    true when it is written and false after two more objects say it.

    Nothing is dropped. Every subject that went in comes out, repeats included,
    so ``len(finding.subjects)`` is the number of objects a line stands for.
    """
    merged: dict[tuple[str, str], list[str]] = {}
    for finding in findings:
        merged.setdefault((finding.severity, finding.message), []).extend(finding.subjects)
    return [
        Finding(severity, tuple(subjects), message)
        for (severity, message), subjects in merged.items()
    ]


def hz(value: float) -> str:
    """A frequency in the unit a person would quote it in."""
    for limit, suffix in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if value >= limit:
            return f"{value / limit:g} {suffix}"
    return f"{value:g} Hz"
