# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Reading a property off a document object, and the fault when it cannot be.

A FreeCAD document object is an attribute bag, and this adapter imports no
FreeCAD (see :mod:`~.document`), so every property it reads is reached by duck
typing. The primitives for doing that live here rather than being spelled again
in each module that translates a subject.

The primitives themselves are shared - :mod:`~...properties` holds them, since
every adapter reads a document the same way. They are re-exported here, with the
fault and with the document's spelling of an axis, so that a module translating
a subject reaches all of it in one import.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

# ``X``, ``Y``, ``Z`` - the document's spelling, from portbox, rather than the
# envelope's lowercase one from model. Everything the adapter names an axis for
# is document-facing: a message about PropagationAxis, whose value the user sees
# as "X", or a property name like PaddingXMin. With model's copy the messages
# would spell an axis differently from the property they name. The lowercase
# copy stays in model, where it spells envelope keys. It is re-exported here so
# that the modules translating a subject share one spelling.
from ...portbox import AXIS_NAMES as AXIS_NAMES
from ..errors import TranslationError
from ..properties import clearance, out_of_step
from ..properties import kind as _kind
from ..properties import label as _label
from ..properties import value as _value
from .model import EnvelopeError

__all__ = [
    "AXIS_NAMES",
    "TranslationError",
    "_axis",
    "_kind",
    "_label",
    "_model_fault",
    "_value",
    "clearance",
    "out_of_step",
]

_AXES = {
    "X": (0, 1),
    "-X": (0, -1),
    "Y": (1, 1),
    "-Y": (1, -1),
    "Z": (2, 1),
    "-Z": (2, -1),
}


@contextmanager
def _model_fault(label: str = "") -> Iterator[None]:
    """Re-raise the envelope's own validation as the document's problem.

    Everything the envelope validates arrived from a property field, so an
    :class:`EnvelopeError` raised while translating is something the user typed
    rather than a crash. The panel splits on that: it shows a
    :class:`TranslationError` as the model's problem and routes anything else
    through a catch-all that prints "Internal error" and a traceback. A
    traceback reads as a bug in the workbench, and a message about something the
    user typed must not look like one.

    Use it only around a constructor whose every field is a property, which
    means ``Material``, ``Termination`` and ``Frequency``. ``Problem`` is
    excluded, because it spans ``plan_grid`` and every structural invariant, so
    wrapping it would report a meshing fault as something the user typed.
    ``Solid`` is excluded, because what it checks about a drawing is checked in
    this module first, where the object can be named as the user knows it, and
    what is left is arithmetic done here. ``Port`` is excluded, because its
    ``__post_init__`` validates geometry that ``portbox`` computed. The values a
    user can put into a port are read
    through :func:`_reference_impedance`, :func:`_resistance` and
    :func:`~.model.check_mode` instead, and :func:`_threads` does the same for
    what lands in ``Problem``.

    The catch stays narrow. Widening it to ``Exception`` would turn every
    ``AttributeError`` inside a builder into "cannot translate this model", with
    the traceback thrown away.

    ``label`` is omitted where the envelope's own message already names the
    subject, as :class:`~.model.Material`'s messages do. Prefixing those gives
    "'FR4': material 'FR4': relative permittivity 0.5 is below 1".
    """
    try:
        yield
    except EnvelopeError as error:
        prefix = f"{label!r}: " if label else ""
        raise TranslationError(f"{prefix}{error}") from error


# ---------------------------------------------------------------------------
# Reading properties off document objects
# ---------------------------------------------------------------------------


def _axis(name: str, subject: str) -> tuple[int, int]:
    """``'-Z'`` to ``(2, -1)``: an axis index and which way along it."""
    try:
        return _AXES[name]
    except KeyError:
        raise TranslationError(
            f"{subject}: {name!r} is not an axis; expected one of {sorted(_AXES)}"
        ) from None
