# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Solve a study of a saved document without the GUI, and save the runs in it.

Run by ``freecadcmd``, which passes a script the arguments after ``--`` and
refuses an option it does not know anywhere else. ``docs/running.md`` gives the
command line, and ``Microwave/batch.py`` says what each argument does and what
the exit status means.
"""

import sys

from Microwave import batch

sys.exit(batch.main(sys.argv))
