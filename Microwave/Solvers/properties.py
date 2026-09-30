# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Reading a property off a document object, which every adapter does the same way.

A FreeCAD document object is an attribute bag, and no adapter imports FreeCAD,
so every property one reads is reached by duck typing. The primitives for doing
that carry no backend's vocabulary and are shared rather than spelled again in
each adapter.

:func:`kind` is the copy of the document layer's own answer for what an object
is. It cannot import that function: the module holding it imports FreeCAD and
these do not.

:func:`clearance` is a rule rather than a primitive, and it is here because it
is the problem's: every backend that builds free space pads by the same length,
so the length is derived once. :func:`smallest_response` and :data:`WANTED` are
here for the same reason: every backend holds the shortcut its method takes to
one bar, set by the smallest response the study reads.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from .. import units
from .errors import TranslationError

__all__ = [
    "WANTED",
    "clearance",
    "kind",
    "label",
    "out_of_step",
    "reading_down_to",
    "smallest_response",
    "value",
]


def kind(obj: Any) -> str:
    """The document object's class, by its Python proxy.

    FreeCAD's ``App::FeaturePython`` gives every workbench object the same
    ``TypeId``, so the proxy class is the only thing that distinguishes one of
    this workbench's objects from another.
    """
    proxy = getattr(obj, "Proxy", None)
    return type(proxy).__name__ if proxy is not None else ""


def label(obj: Any) -> str:
    """What to call the object in a message: what the user named it."""
    return str(getattr(obj, "Label", None) or getattr(obj, "Name", "?"))


def value(quantity: Any) -> float:
    """A property as a plain float, whether or not it carries a unit."""
    return float(getattr(quantity, "Value", quantity))


def out_of_step(objects: Iterable[Any]) -> None:
    """Refuse a study an object of which departs from what its class declares.

    FreeCAD restores the properties an object was saved with and reconciles
    nothing against its class, so a document saved by an earlier build comes
    back missing what has been added since and offering the choices an
    enumeration had then. No build carries a document across from an earlier
    one. Left alone, a run would read a property that is not there, or one
    whose choices are not the ones the adapter knows.

    A property an object carries and its class does not declare is read by no
    run, and it may be the user's own, added in the property editor. It does not
    stop a run by itself. Where the run is stopped it is named with its value,
    because a property a build has retired holds what the user typed and the
    object made anew does not.

    Every departure of every object is named in one refusal, rather than the
    first one a translation happens to reach. ``objects`` are the study and the
    members the adapter reads, and the objects they link to are searched with
    them: a material lives at the document's root and is reached through a
    binding or the policy's medium. Each object answers for itself through its
    proxy, because what a class declares is known where the classes are and no
    adapter imports them.

    FreeCAD's property editor deletes a property added to an object - right
    click, Delete Property - so a property a build no longer declares is the
    user's to remove there. A property put back by hand has no tooltip, and an
    enumeration put back has no choices, so a missing or changed property is
    mended by making the object anew, which declares each as the class does.
    """
    found = []
    for obj in _with_what_they_link(objects):
        ask = getattr(getattr(obj, "Proxy", None), "departures", None)
        if ask is not None:
            found.append((obj, ask(obj)))
    if not any(departed.stops for _, departed in found):
        return
    said = ". ".join(
        f"{label(obj)!r} {'; '.join(departed.phrases)}"
        for obj, departed in found
        if departed.phrases
    )
    raise TranslationError(
        "This study does not match what this build of the workbench declares, which is "
        "how a document an earlier build saved comes back, and no build carries one "
        f"across: {said}. "
        "Make each object that lacks a property, holds one as another type or offers other "
        "choices anew: Add openEMS Solver and Add Palace Solver make whatever solver, "
        "recipe and mesh policy a study lacks, and a port, a material or a study is made "
        "again as it was made first. Copy what the old object held first: a property this "
        "build does not declare is read by no run, and the property editor deletes it "
        "(right-click it, Delete Property)"
    )


def _with_what_they_link(objects: Iterable[Any]) -> list[Any]:
    """``objects`` and every object one of them links to through a property its
    class declares, each once.

    Each object names its own links through its proxy, so a link the user added
    and an expression are not followed: no run reads through either.
    """
    found: dict[int, Any] = {}
    for obj in objects:
        follow = getattr(getattr(obj, "Proxy", None), "linked", None)
        for reached in (obj, *(follow(obj) if follow is not None else ())):
            found.setdefault(id(reached), reached)
    return list(found.values())


#: How far the open surface stands from the structure when ``Clearance`` is 0,
#: as a share of the wavelength in the study's medium at the top of the band.
#:
#: Calibrated against openEMS' own tutorials, which state it as a length or as a
#: share of the free-space wavelength at the top of their band:
#:
#: =========================  =========================  ==============
#: tutorial                   air to the boundary        in wavelengths
#: =========================  =========================  ==============
#: ``Simple_Patch_Antenna``   70 mm, where lambda_0/20   0.7
#:                            is 4.997 mm
#: ``MSL_Losses``             0.5 lambda_0 at f_stop     0.5
#: ``MSL_NotchFilter``        none laterally: the        0
#:                            substrate meets MUR
#: =========================  =========================  ==============
#:
#: A default for an unknown structure belongs between a radiator and a line,
#: and nearer the line. Giving a line air where ``Through`` was needed costs a
#: reflection; giving it ``Through`` where air was needed costs a little domain.
#:
#: Taken at the top of the band, the surface stands the same number of
#: wavelengths off at the top of every band, so what it does to a radiated wave
#: there is fixed whatever the band. Taken at the bottom, a band starting near
#: DC would put metres of air round a board meshed for its top frequency.
CLEARANCE_WAVELENGTHS = 0.4


def clearance(stated: Any, stop: float, settings: Any, slowing: float = 1.0) -> float:
    """How far the open surface stands from the structure, in millimetres.

    ``stated`` is the mesh policy's ``Clearance`` as the adapter read it,
    ``stop`` the top of the band in hertz, ``settings`` the policy, which a
    refusal names, and ``slowing`` the product a wave slows by the root of in
    the study's medium. The distance is a fact about the problem rather than
    about any method's discretisation, so it is a length on the policy. Zero
    means derived: :data:`CLEARANCE_WAVELENGTHS` of the wavelength in the medium
    at ``stop``, since that is the wavelength the field beyond the structure
    falls off and radiates at. A clearance of nothing on a face open to free
    space is the face ``Ends`` states, so zero is free to mean "derive it".

    Each adapter reads the property itself and hands it here, so what it reads
    is in its own source, and the rule for what the value means is written
    once.

    A negative or non-finite length is refused. FreeCAD's length property holds
    no negative value, so this is reached from a file edited by hand or from a
    script.
    """
    length = value(stated)
    if not math.isfinite(length) or length < 0.0:
        raise TranslationError(
            f"{label(settings)!r}: Clearance is {length:g} mm. It is how far the medium "
            "reaches past the structure on each Air face, so it is a length of zero or "
            "more, and zero derives it from the band"
        )
    if length > 0.0:
        return length
    return CLEARANCE_WAVELENGTHS * units.SPEED_OF_LIGHT / stop / math.sqrt(slowing) * units.MM_PER_M


#: How far a run's own shortcut may move its S-parameters, as a share of the
#: smallest response the study says it reads. A time-domain run stops its
#: record before the field has died away, and an adaptive sweep answers most
#: points from a reduced model; each backend measures what its shortcut moved
#: and holds it to this.
#:
#: A study that says nothing reads down to full scale, and there this is an
#: absolute error in S: a matched line reading |S11| = 0.002 and a filter
#: reading |S21| = 0.9 are held to one bar, which a relative measure could not
#: do. It is also the bar the acceptance gates are set to.
#:
#: It is a share rather than an absolute figure because an error is only
#: harmless beside the term it lands on. A stopband of -40 dB is |S21| = 0.01,
#: so a run wrong by an absolute bar of that size carries the whole of the only
#: quantity the device exists to deliver. Scaling the bar with the declared
#: floor keeps one promise wherever that floor sits: the smallest term the study
#: reads is right to about a tenth of a dB.
WANTED = 0.01


def smallest_response(analysis: Any) -> float:
    """The study's reading floor, converted from decibels to a magnitude in S.

    The conversion is in amplitude rather than power, because the floor is
    weighed against an error in S: twenty decibels a decade, the same decibels
    the response is plotted in.

    Zero is full scale, and it is what a study that has not considered the
    question says, so the ordinary run keeps the absolute bar. A value above
    zero is refused rather than clamped. A passive device answers no more than
    one, so a positive value is a typing slip, and a slip quietly taken as "full
    scale" would leave the property reading as though it had been honoured.
    """
    declared = float(analysis.SmallestResponse)
    if not math.isfinite(declared) or declared > 0.0:
        raise TranslationError(
            f"{label(analysis)!r}: SmallestResponse is {declared:g} dB. It is "
            "how far down the response is read, so it is zero for full scale "
            "and negative below it"
        )
    return 10.0 ** (declared / 20.0)


def reading_down_to(smallest: float) -> str:
    """What a bar of :data:`WANTED` of ``smallest`` is, as the end of a sentence
    naming it: the study reading down to a level, or full scale.

    Three figures on the level. Rounded to whole decibels a shallow declaration
    reads as "-0 dB", and a study that asked for a fraction of one is then told
    the bar of a study that asked for nothing.
    """
    if smallest >= 1.0:
        return "a study read at full scale can carry"
    return f"a study reading down to {20 * math.log10(smallest):.3g} dB can carry"
