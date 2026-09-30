# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The fault an adapter raises when a document does not describe a run it can make.

Shared because the layer above catches it. A task panel showing a run has to
tell the model's own problem from a defect in the workbench, and it does that by
the type: this one is shown as the model's problem, and anything else prints a
traceback and reads as a bug. That distinction cannot be per adapter, since the
panel does not know which one it started.

It sits here rather than under an adapter for the same reason a declaration
does: an adapter never imports another.
"""

from __future__ import annotations

__all__ = ["TranslationError"]


class TranslationError(Exception):
    """The document does not describe something this adapter can run."""
