# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What is still left of the response at the end of a port's record.

A run ends at ``max_timesteps`` whether or not the field has gone, and what gets
cut off is the tail of the very series openEMS transforms. A transform missing
its tail is the response plus leakage - ripple that is not in the device, a
notch at the wrong depth, a magnitude above one - and the results file looks
exactly as it does after a finished run.

Measured as the transform of the record's **last tenth**, against the wave that
drove the run: an absolute error in S, in the units the answer is read in.
Nothing is assumed about the record's shape, and in particular not that its
largest sample belongs to the response - at a driven port the largest sample
is the excitation, so a device ringing far below its drive passes anything
measured against it.

Not the residual openEMS prints beside its progress line, which is a
whole-domain energy sampled on a wall-clock timer.

Pure numpy: no FreeCAD, no openEMS, so the driver and the panel say it the same
way.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

#: The share of the record the measurement drops, which makes what it reports
#: the difference between transforming the whole record and stopping that much
#: early.
TAIL = 0.1

#: How much of this run's S-parameters may still come from that last tenth.
#:
#: An absolute error in S rather than a relative one, so a matched line reading
#: |S11| = 0.002 and a filter reading |S21| = 0.9 are held to one bar - and it
#: is the bar the acceptance gates are set to.
#:
#: What was never recorded stands in a known ratio to what this measures: for a
#: mode of time constant ``tau`` over a record of ``T``, ``1/(exp(T/(10 tau)) -
#: 1)``. That is one where the record covers about seven time constants and
#: climbs without bound below it - so a run that fails this bar is worse than
#: it says, and only a run that passes it comfortably is finished.
WANTED = 0.01


def tail_share(times, signal, frequencies, incident) -> float:
    """What the end of one port's record is worth in this run's S-parameters.

    The transform is openEMS' own - ``2 * dt * sum(u exp(-i 2 pi f t))``, from
    ``utilities.DFT_time2freq`` - taken over the tail alone, which is exactly
    what transforming the whole record adds over stopping early. Divided by the
    driven port's incident wave it is an absolute error in every S-parameter
    this port contributes to.

    The worst frequency, not the mean. Leakage matters where the response is
    smallest, which is where a notch is read.
    """
    times = np.asarray(times, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if times.size < 2:
        # No record, so no transform: none of the answer is accounted for.
        return 1.0
    cut = times.size - max(1, int(times.size * TAIL))
    kernel = np.exp(-2j * np.pi * np.outer(frequencies, times[cut:]))
    contribution = 2 * (times[1] - times[0]) * (kernel * signal[cut:]).sum(axis=1)
    return float(np.max(np.abs(contribution) / np.abs(incident)))


def unfinished(shares: Mapping[int, float]) -> str | None:
    """What to say about a run that stopped before its response did.

    ``None`` when every record was finished with. Ports are named one at a time:
    each watches a different part of the device, so which one is still ringing
    is where to look.
    """
    still = sorted((number, value) for number, value in shares.items() if value > WANTED)
    if not still:
        return None
    named = ", ".join(f"{value:.1%} at port {number}" for number, value in still)
    return (
        "the run stopped before the response did: dropping the last tenth of "
        f"the record moves this run's S-parameters by {named}, against the "
        f"{WANTED:.0%} a finished record moves. Raise max_timesteps"
    )
