# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What every check shares: how it reports, and when two coordinates are one place.

A check decides and returns :class:`Finding` objects; it never prints, refuses
or raises. :func:`~.check` collects them and the callers above render
``str(finding)``.

:data:`_ON_THE_GRID` is here for the same reason - it is the one tolerance
several checks have to agree on, and a tolerance two modules define separately
is one they will eventually disagree about.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

REFUSE = "refuse"


WARN = "warn"


SUBSTITUTE = "substitute"


#: How far off a grid line a coordinate may be and still count as on it, in mm,
#: and by the same argument how far two coordinates may sit apart and still be
#: one place. One picometre: far below any length this workbench meshes, and
#: above the rounding a domain wall picks up passing through the mesher's
#: arithmetic.
_ON_THE_GRID = 1e-9


@dataclass(frozen=True)
class Finding:
    """One thing wrong, or worth saying, about a problem.

    ``subjects`` is **one entry per object**, and it is a tuple because one
    sentence is routinely true of several. A check passes the single name it is
    talking about and gets a one-tuple; only :func:`_grouped` builds a longer
    one. Passing a bare string is a convenience for the thirty-odd call sites
    and is the only coercion done - nothing else is validated, because every
    caller is in this package.

    A subject is a label to show the user, never a key to look an object up by:
    after merging it names all of them.

    Duplicates are kept. ``Solid.name`` falls back to the material when a solid
    has no label, so two objects can share a name, and dropping the repeat here
    would make :func:`object_count` undercount the very case it exists for. The
    repeat is collapsed for *display* instead, in :attr:`subject`.
    """

    severity: str
    subjects: tuple[str, ...]
    message: str

    def __post_init__(self) -> None:
        if isinstance(self.subjects, str):
            object.__setattr__(self, "subjects", (self.subjects,))

    @property
    def subject(self) -> str:
        """The subjects as a reader wants them: ``'A'``, ``'A and B'``, ``'A, B and C'``.

        Each distinct name once. Two objects under one name are two objects and
        one word, and "Copper and Copper" says nothing the singular does not.
        """
        names = list(dict.fromkeys(self.subjects))
        if len(names) == 1:
            return names[0]
        return f"{', '.join(names[:-1])} and {names[-1]}"

    def __str__(self) -> str:
        return f"[{self.severity}] {self.subject}: {self.message}"


def _grouped(findings: Sequence[Finding]) -> list[Finding]:
    """One finding per distinct thing said, naming every object it was said of.

    A check that walks the model says its sentence once per object it holds
    for, so the count of lines follows the size of the model rather than the
    number of things wrong with it. Printing every copy is what trains a reader
    to skip the section, and the section exists to be read.

    Merging is on the **message**, so two objects are named together only when
    the sentence is identical down to the coordinates in it. Two solids
    overhanging the same wall at different distances have different sentences
    and stay apart, because one line quoting one position for both would be
    false. It follows that a message must carry no numeral counting its own
    subjects - "both" is true when it is written and false after two more
    objects say it.

    Nothing is dropped: every subject that went in comes out, repeats included,
    so ``len(finding.subjects)`` is how many objects a line stands for.
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
