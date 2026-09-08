# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Read a directory a run left behind, the way a gate reads one of its cases.

``gate_report`` runs the gates and reports what they claim. This is the other
half: a solve that has already happened, sitting in a directory, with nobody
asserting anything about it - a case built by hand, a case whose envelope was
edited, an alignment nobody has written a gate for yet.

What it prints is what every gate here computes before it asserts: the port's
impedance averaged over the bins that hold a whole cycle of the record, how far
that varies across the band, how much came back, and what was still in the port
when the record stopped.

No reference and no bar: which closed form a directory should be compared
against is a property of what was drawn, so give ``--against`` a number if there
is one.

    python3 -m tests.read_solve <dir> [<dir> ...] [--against 51.83] [--record 3e-9]

Under the interpreter that owns the openEMS bindings, since it reads through the
adapter's own result layer.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from Microwave.Solvers.openems import read
from Microwave.Solvers.openems.model import Problem


def measured(directory: pathlib.Path, record_seconds: float) -> dict:
    """One solve, as the numbers a gate would look at.

    ``record_seconds`` sets which bins are read: below one cycle in the record a
    transform is reporting a fraction of a period, so those bins are left out.
    """
    result = read.read(str(directory))
    in_band = np.asarray(result.frequency) >= 1.0 / record_seconds
    if not in_band.any():
        raise SystemExit(
            f"{directory}: nothing in the sweep holds a whole cycle of a "
            f"{record_seconds:g} s record, so there is no bin to average over"
        )

    port = min(result.ports)
    impedance = np.asarray(result.port(port).impedance)[in_band]
    line = {
        "port": port,
        "impedance": float(np.mean(impedance)),
        "flatness": float(np.ptp(impedance) / np.mean(impedance)),
        "reflection": float(np.max(np.abs(result.s(port, port)[in_band]))),
        "tail": max(result.tail_share.values()),
        "bins": int(in_band.sum()),
    }

    envelope = directory / "openems.json"
    if envelope.exists():
        problem = Problem.from_dict(json.loads(envelope.read_text()))
        line["cells"] = problem.grid.cell_count
        line["title"] = problem.title
    return line


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directories", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "--against",
        type=float,
        default=None,
        help="a reference impedance in ohms, if the drawing has one",
    )
    parser.add_argument(
        "--record",
        type=float,
        default=3.0e-9,
        help="how long the run recorded, in seconds; sets the lowest bin read",
    )
    arguments = parser.parse_args(argv)

    found = []
    for directory in arguments.directories:
        line = measured(directory, arguments.record)
        found.append((directory, line))
        against = ""
        if arguments.against is not None:
            off = (line["impedance"] - arguments.against) / arguments.against
            against = f" ({100 * off:+.3f} % of {arguments.against:.4f})"
        print(
            f"{directory.name:>20}  port {line['port']}  "
            f"Z {line['impedance']:.4f} ohm{against}, "
            f"flat to {100 * line['flatness']:.3f} %, "
            f"|S11| below {line['reflection']:.4f}, "
            f"tail {line['tail']:.2e}, {line['bins']} bins"
        )

    if len(found) > 1:
        impedances = [line["impedance"] for _, line in found]
        span = float(np.ptp(impedances))
        scale = arguments.against if arguments.against is not None else float(np.mean(impedances))
        print(f"\nspan across these: {span:.6f} ohm, {1e6 * span / scale:.2f} ppm of {scale:.4f}")


if __name__ == "__main__":
    main()
