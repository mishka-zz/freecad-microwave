# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Reading a property off a document object, and the fault when it cannot be.

A FreeCAD document object is an attribute bag, and this adapter imports no
FreeCAD - see :mod:`~.document` - so every property it reads is reached by duck
typing. The primitives for doing that are here rather than spelled again in each
module that translates a subject.

:class:`TranslationError` is here for the same reason. Every module below raises
it, and none of them may import another to get it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

# ``X``, ``Y``, ``Z`` - the *document's* spelling, from portbox, not the
# envelope's lowercase one from model. Everything the adapter names an axis for
# is document-facing: a message about PropagationAxis, whose value the user sees
# as "X", or a property name like PaddingXMin. Take model's copy instead and the
# messages spell an axis differently from the property they name. The lowercase
# copy stays in model, where it spells envelope keys. Re-exported here so the
# modules that translate a subject share one spelling of this reasoning.
from ...portbox import AXIS_NAMES as AXIS_NAMES
from .model import EnvelopeError

_AXES = {
    "X": (0, 1),
    "-X": (0, -1),
    "Y": (1, 1),
    "-Y": (1, -1),
    "Z": (2, 1),
    "-Z": (2, -1),
}


class TranslationError(Exception):
    """The document does not describe something this adapter can run."""


@contextmanager
def _model_fault(label: str = "") -> Iterator[None]:
    """Re-raise the envelope's own validation as the document's problem.

    Everything the envelope validates arrived from a property field, so an
    :class:`EnvelopeError` raised while translating is something the user
    typed, not a crash - and the panel splits on exactly that, showing a
    :class:`TranslationError` as the model's problem and routing anything else
    through a catch-all that prints "Internal error" and a traceback. A
    traceback reads as a bug in the workbench, and a message about something
    the user typed must not look like one.

    **Only around a constructor whose every field is a property**, which is
    ``Material``, ``Termination`` and ``Frequency``. Not ``Problem``, which
    spans ``plan_grid`` and every structural invariant, so wrapping it would
    report a meshing fault as something the user typed. Not ``Solid``: what it
    checks about a drawing is checked in this module first, where the object can
    be named as the user knows it, and what is left is arithmetic done here.
    Not ``Port`` either:
    its ``__post_init__`` validates geometry that ``portbox`` computed. The
    values a user can put into a port are read through
    :func:`_reference_impedance`, :func:`_resistance` and
    :func:`~.model.check_mode` instead, and :func:`_threads` does the same for
    what lands in ``Problem``.

    The catch stays narrow: widening it to ``Exception`` turns every
    ``AttributeError`` inside a builder into "cannot translate this model",
    with the traceback thrown away.

    ``label`` is omitted where the envelope's own message already names the
    subject, which :class:`~.model.Material`'s do. Prefixing those gives
    *"'FR4': material 'FR4': relative permittivity 0.5 is below 1"*.
    """
    try:
        yield
    except EnvelopeError as error:
        prefix = f"{label!r}: " if label else ""
        raise TranslationError(f"{prefix}{error}") from error


# ---------------------------------------------------------------------------
# Reading properties off document objects
# ---------------------------------------------------------------------------


def _kind(obj: Any) -> str:
    """The document object's class, by its Python proxy.

    FreeCAD's ``App::FeaturePython`` gives every workbench object the same
    ``TypeId``, so the proxy class is the only thing that distinguishes an
    ``EMPortMicrostrip`` from an ``EMSolverOpenEMS``.

    A copy of ``Objects.kinds.kind_of``, and the only one. It cannot import that
    one, because ``Objects/__init__.py`` imports FreeCAD and this module's whole
    claim is that it does not.
    """
    proxy = getattr(obj, "Proxy", None)
    return type(proxy).__name__ if proxy is not None else ""


def _label(obj: Any) -> str:
    return str(getattr(obj, "Label", None) or getattr(obj, "Name", "?"))


def stale_document(error: AttributeError, objects: Iterable[Any]) -> TranslationError | None:
    """The refusal for a document saved before a property existed, or ``None``.

    FreeCAD stores the properties an object *had* and does not reconcile a
    restored one against the class it belongs to, so a document written by an
    earlier build comes back without whatever has been added since. Every
    property here is reached by duck typing, so the first read of one is a bare
    ``AttributeError`` from inside the adapter - and the panel shows anything
    that is not a :class:`TranslationError` as an internal error with a
    traceback, which reads as a bug in the workbench rather than as a document
    out of step with it.

    Old documents stay unsupported, which is settled. What this changes is that
    being unsupported arrives as a sentence naming the object and the property.

    Three things must hold before an error is rewritten, and together they are
    what keeps this from blaming the user for a fault of ours. The object must
    be one of *ours*, by its proxy. It must really lack the attribute. And the
    attribute must be spelled the way FreeCAD spells a property - CapWords -
    which every property this adapter reads is, and which none of its own
    internals are.

    That still cannot separate a document missing a property from this adapter
    misspelling one, because the two are the same event seen from inside. So the
    message does not claim to know which, and says what to do in either case.
    """
    name = str(getattr(error, "name", "") or "")
    if not name[:1].isupper() or "_" in name:
        return None
    lacking = [obj for obj in objects if _kind(obj) and not hasattr(obj, name)]
    if not lacking:
        return None
    return TranslationError(
        f"{_label(lacking[0])!r} has no {name}. A document saved by an earlier "
        f"build is missing whatever has been added since, and restoring one "
        f"does not fill it in - re-create that object, or rebuild the document "
        f"from the script beside it. If it was made by this build, this is a "
        f"fault in the workbench: please report it with the version"
    )


def _value(quantity: Any) -> float:
    """A property as a plain float, whether or not it carries a unit."""
    return float(getattr(quantity, "Value", quantity))


def _axis(name: str, subject: str) -> tuple[int, int]:
    """``'-Z'`` to ``(2, -1)``: an axis index and which way along it."""
    try:
        return _AXES[name]
    except KeyError:
        raise TranslationError(
            f"{subject}: {name!r} is not an axis; expected one of {sorted(_AXES)}"
        ) from None
