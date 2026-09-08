# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Put an envelope on disk, and read one back.

The envelope is openEMS' input as this adapter states it, and :mod:`~.model`
declares its shape. This module puts it on disk, writes what goes beside it, and
loads it again.

The grid an envelope carries is planned in :mod:`~.plan`, which this does not
reach.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from .model import Problem

ENVELOPE_NAME = "openems.json"


#: Where what the drawing lost on the way to the envelope is written, beside the
#: envelope itself. It is a separate file rather than a field. The envelope is
#: hashed as the run's provenance, nothing on the driver's side reads this file,
#: and a figure that moves a digest without changing what is solved makes every
#: earlier result look as though it came from a different input.
REPORT_NAME = "geometry.txt"


def write(problem: Problem, directory: str | Path, report: Sequence[str] = ()) -> Path:
    """Serialise the envelope into ``directory``. Returns the file written.

    Writing is a separate stage from running. Writing the input without
    solving is the debugging and bug-report path, and the file it produces is
    self-contained: attach it to an issue and the run is reproducible.

    :param report: Lines saying how the shapes solved differ from the shapes
        drawn. See :func:`~.document.geometry_report`. They are written beside
        the envelope rather than into it, and omitted when there is nothing to
        say. A driver started from an envelope has never seen a drawing, so this
        is the only point on a headless route where both are in hand.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / ENVELOPE_NAME
    path.write_text(problem.to_json(), encoding="utf-8")

    (directory / "envelope.sha256").write_text(problem.digest() + "\n", encoding="utf-8")
    if report:
        (directory / REPORT_NAME).write_text("\n".join(report) + "\n", encoding="utf-8")
    return path


def read_envelope(path: str | Path) -> Problem:
    """Load an envelope. The driver's entry into the model."""
    return Problem.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
