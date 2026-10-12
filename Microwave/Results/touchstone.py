# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A Touchstone file read into a matrix. Solver-neutral.

The file is read here, line by line, from the bytes as they came in, so the
matrix, the digest and the copy kept in the document are of one file. Every
line is either read for what the format says it holds or refused with its
number: a comment is never read as data, and data is never left out without a
word. The matrix comes back with the file's own port numbers and the reference
the file states: the option line's ``R``, a Touchstone 2 ``[Reference]`` per
port, or an HFSS ``Port Impedance`` comment after each point.

What is read:

- S parameters in ``RI``, ``MA`` or ``DB``, in ``Hz``, ``kHz``, ``MHz`` or
  ``GHz``, from a Touchstone 1 file or a Touchstone 2.0 file;
- Z parameters in a Touchstone 1 file at one ``R``, which the format normalises
  to that ``R``, turned into S at it;
- the network block of a two-port that also carries noise parameters, with the
  noise left out and said to be.

Y, H and G parameters are refused: the format states their normalisation only
as "normalized". So are mixed-mode data, Z parameters in a Touchstone 2 file or
beside HFSS port impedances, and any line this does not read.

A comment after a point that names the port impedance and holds numbers is read
as HFSS writes one: a real and an imaginary part for each port, on one line,
after every point. It states the reference in place of the option line's ``R``,
and one in any other form is refused rather than taken for prose. HFSS states
its matrix in a wave definition of its own unless the file names another, and a
matrix here is of power waves; the two differ only at a complex reference, where
scikit-rf converts one into the other.

The user declares what each port's numbers are ratios against, because a file's
option line can state a number its calibration did not use: ``R 50`` over a
waveguide calibrated by TRL to the line's own impedance. :func:`declared`
applies the declaration, and never moves a number of the matrix.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field, replace

import numpy as np

from . import _skrf
from .sparameters import SParameters, _engineering_hz

__all__ = [
    "DECLARATIONS",
    "OWN",
    "PER_POINT",
    "STATED",
    "Read",
    "Unreadable",
    "declared",
    "digest",
    "read",
    "text_of",
]

#: What an imported table's ports are referenced to, as the property offers it.
STATED = "As the file states"
OWN = "Port impedance"
DECLARATIONS = (STATED, OWN)

#: The provenance key saying the file stated each port's reference at each
#: point, where the numbers are the port's impedance rather than a nominal one.
PER_POINT = "reference_at_each_point"

#: The byte order mark some editors put ahead of UTF-8 text.
_BOM = b"\xef\xbb\xbf"

#: Where a line ends: CR, LF or both. ``str.splitlines`` ends one at a form
#: feed and at others besides, which a comment may hold.
_LINE = re.compile(r"\r\n|\r|\n")

#: A file's port count is in its name.
_NAMED = re.compile(r"\.s([0-9]{1,4})p$", re.IGNORECASE)

#: Where one word of a line ends and the next begins. ``str.split`` also
#: splits at a no-break space and at others the format does not name.
_SPACE = re.compile(r"[ \t]+")

#: A number as the format writes one: digits, a point, an exponent. Python's
#: ``float`` also takes ``nan``, ``inf``, ``1_0`` and digits of other scripts.
_NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")

_UNITS = {"hz": ("Hz", 1.0), "khz": ("kHz", 1e3), "mhz": ("MHz", 1e6), "ghz": ("GHz", 1e9)}
_PARAMETERS = ("s", "y", "z", "h", "g")
_FORMATS = ("ri", "ma", "db")

#: The largest condition number of ``z + 1`` a Z is turned into S at: the
#: figures lost are its logarithm, and a double holds about sixteen, which
#: leaves six.
_CONDITION = 1e10

#: The comment HFSS writes after a point's data, with each port's impedance.
_IMPEDANCE = re.compile(r"\s*port\s*impedance(.*)", re.IGNORECASE)

#: The comment scikit-rf writes naming the wave definition, and HFSS's own.
_DEFINITION = re.compile(r"S-parameter uses the (power|pseudo|traveling) definition")
_HFSS_DEFINITION = "traveling"

#: Touchstone 2 keywords this reads, by the lower case of what is in brackets.
_KEYWORDS = (
    "version",
    "number of ports",
    "two-port data order",
    "number of frequencies",
    "number of noise frequencies",
    "reference",
    "matrix format",
    "mixed-mode order",
    "begin information",
    "end information",
    "network data",
    "noise data",
    "end",
)


#: Touchstone 2 keywords that stand alone on their line.
_BARE = ("begin information", "end information", "network data", "noise data", "end")


class Unreadable(ValueError):
    """A file this does not read, with the reason and the line where there is one."""


@dataclass(frozen=True)
class Read:
    """A file read: its matrix, what its header said, and how it was read."""

    result: SParameters
    #: The comment lines ahead of the option line, without their ``!``.
    comments: tuple[str, ...]
    #: How the file was read, one statement each: its version, parameters,
    #: format and unit, and anything left out or converted.
    said: tuple[str, ...]
    #: Which of :data:`DECLARATIONS` the file's own reference suggests.
    suggested: str


def digest(data: bytes) -> str:
    """The file's identity: its SHA-256, named."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def text_of(data: bytes) -> tuple[str, str]:
    """The file as text, and the encoding that gives back its bytes.

    UTF-8 where the bytes are UTF-8, and Latin-1 otherwise, which maps every byte
    to one character and back.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise AssertionError("Latin-1 decodes every byte")


@dataclass
class _File:
    """What the lines of one file said, as they are read."""

    name: str
    ports: int
    version: str = "1"
    option: int = 0
    unit: str = "ghz"
    parameter: str = "s"
    format: str = "ma"
    resistance: float | None = 50.0
    stated: list[float] | None = None
    count: int | None = None
    order: str = ""
    matrix: str = "full"
    comments: list[str] = field(default_factory=list)
    definition: str = ""
    #: Each number of the network data, with its line and whether it opens it.
    numbers: list[tuple[float, int, bool]] = field(default_factory=list)
    #: The HFSS port impedances, as the number of points read when each came.
    impedances: list[tuple[int, list[complex], int]] = field(default_factory=list)
    noise: bool = False
    section: str = ""

    def refuse(self, line: int, why: str) -> Unreadable:
        return Unreadable(f"{self.name} line {line}: {why}")


def read(name: str, data: bytes) -> Read:
    """Read ``data``, the bytes of a file called ``name``.

    :raises Unreadable: the file is not one this reads, with the reason.
    """
    named = _NAMED.search(name)
    if named is None or int(named.group(1)) < 1:
        raise Unreadable(
            f"{name}: the file name does not end in .sNp, where N is the number of ports. "
            "Rename the file to end in .sNp"
        )
    if b"\x00" in data:
        raise Unreadable(
            f"{name} holds a NUL byte, as UTF-16 text does. Save the file as ASCII text"
        )
    text, _ = text_of(data.removeprefix(_BOM))
    found = _File(name=name, ports=int(named.group(1)))
    for number, line in enumerate(_LINE.split(text), start=1):
        _line(found, number, line)
    if not found.option:
        raise Unreadable(f"{name} holds no option line, the line that starts with #")
    if found.section == "information":
        raise Unreadable(f"{name} opens [Begin Information] and holds no [End Information]")
    if found.version == "2" and found.section != "end":
        raise Unreadable(f"{name} is a Touchstone 2 file and does not close with [End]")
    if found.parameter != "s" and (found.parameter != "z" or found.version != "1"):
        raise Unreadable(
            f"{name} holds {found.parameter.upper()} parameters in a Touchstone "
            f"{_version(found)} file. Only S parameters, and Z parameters in a Touchstone 1 "
            "file, can be imported. Export S parameters from the instrument or the simulator"
        )

    frequency, s = _points(found)
    reference = _reference(found, frequency.size)
    said = [
        f"Touchstone {_version(found)}, {found.parameter.upper()} parameters in "
        f"{found.format.upper()}, frequency in {_UNITS[found.unit][0]}"
    ]
    if found.parameter == "z":
        if found.impedances:
            raise Unreadable(
                f"{name} holds Z parameters beside HFSS port impedances. Z parameters are "
                "converted to S only at the option line's R"
            )
        s = _from_z(found, frequency, s)
        said.append(f"Z parameters turned into S parameters at {found.resistance:g} ohm")
    definition = found.definition or (_HFSS_DEFINITION if found.impedances else "power")
    if definition != "power" and np.any(reference.imag != 0.0):
        skrf = _skrf.module()
        s = np.asarray(skrf.network.s2s(s, reference, "power", definition), dtype=complex)
        said.append(f"S parameters turned from the {definition} wave definition into power waves")
    if found.noise:
        said.append("noise parameters left out")
    result = SParameters(
        frequency=frequency,
        s=s,
        port_numbers=tuple(range(1, found.ports + 1)),
        reference=reference,
        measured_impedance=reference.copy(),
        provenance={PER_POINT: True} if found.impedances else {},
    )
    return Read(
        result=result,
        comments=tuple(found.comments),
        said=tuple(said),
        suggested=OWN if _moves(reference) else STATED,
    )


def declared(result: SParameters, declaration: str) -> SParameters:
    """``result`` with its ports referenced as ``declaration`` says.

    :data:`STATED` leaves it as the file states. :data:`OWN` references each port
    to its own impedance. A file that states each port's reference at each point
    states the impedance each port had, and the ports keep those numbers. A file
    that states a reference once states a nominal one, and the ports keep none.
    """
    if declaration == STATED:
        return result
    if declaration != OWN:
        raise ValueError(f"no such declaration: {declaration!r}")
    reference = np.array(result.reference, dtype=complex)
    if not result.provenance.get(PER_POINT):
        reference[:] = complex(np.nan, np.nan)
    return replace(
        result,
        reference=reference,
        measured_impedance=reference.copy(),
        self_referenced=tuple(result.port_numbers),
    )


def _moves(reference: np.ndarray) -> bool:
    """Whether any port's stated reference changes across the band."""
    return bool(np.any(reference != reference[:1]))


def _version(found: _File) -> str:
    return "1" if found.version == "1" else "2.0"


def _line(found: _File, number: int, line: str) -> None:
    """Read one line into ``found``, or refuse it."""
    body, _, comment = line.partition("!")
    body = body.strip(" \t")
    if found.section == "information":
        if body.lower().startswith("[end information]"):
            found.section = ""
        return
    if not body:
        if "!" in line:
            _comment(found, number, comment)
        return
    if found.section == "end":
        raise found.refuse(number, "holds data after [End]")
    if body.startswith("["):
        _keyword(found, number, body)
    elif body.startswith("#"):
        if found.option:
            return  # the format reads the first option line and ignores the rest
        found.option = number
        _option(found, number, _words(body[1:]))
    elif found.section == "noise":
        return
    elif found.section == "reference":
        _stated(found, number, body)
    else:
        _data(found, number, body)


def _comment(found: _File, number: int, comment: str) -> None:
    """A line that is a comment: a header line, an HFSS port impedance, or nothing."""
    said = _DEFINITION.search(comment)
    if said:
        found.definition = said.group(1)
    if not found.option:
        found.comments.append(comment.rstrip())
        return
    impedance = _IMPEDANCE.match(comment)
    if impedance is None or found.version != "1" or not found.numbers:
        return
    words = _words(impedance.group(1).partition("!")[0])
    if not words or not all(_NUMBER.fullmatch(word) for word in words):
        return  # a comment that names port impedance and states none
    if len(words) != 2 * found.ports:
        raise found.refuse(
            number,
            f"states {len(words)} numbers for the port impedances of a {found.ports}-port, "
            f"which needs {2 * found.ports} on one line",
        )
    values = [float(word) for word in words]
    pairs = [complex(values[k], values[k + 1]) for k in range(0, len(values), 2)]
    if not all(math.isfinite(abs(z)) for z in pairs):
        raise found.refuse(number, "states a port impedance too large to hold")
    if not all(z.real > 0.0 for z in pairs):
        # A power wave is defined only against an impedance with a positive real
        # part.
        raise found.refuse(
            number,
            "states a port impedance whose real part is not positive, so no power wave "
            "can be referenced to it",
        )
    found.impedances.append((len(found.numbers), pairs, number))


def _option(found: _File, number: int, words: list[str]) -> None:
    """The option line's words, in any order, ``R`` and its value together, each
    kind once."""
    index = 0
    named: set[tuple[str, ...]] = set()
    while index < len(words):
        word = words[index].lower()
        index += 1
        kind = next(
            (kind for kind in (_UNITS, _PARAMETERS, _FORMATS, ("r",)) if word in kind), None
        )
        if kind is not None:
            if tuple(kind) in named:
                raise found.refuse(
                    number,
                    f"the option line names {words[index - 1]!r} and another value of the "
                    "same kind",
                )
            named.add(tuple(kind))
        if word in _UNITS:
            found.unit = word
        elif word in _PARAMETERS:
            found.parameter = word
        elif word in _FORMATS:
            found.format = word
        elif word == "r":
            if index == len(words):
                found.resistance = None  # stated at each point instead
                continue
            found.resistance = _positive(found, number, words[index], "R")
            index += 1
        else:
            raise found.refuse(
                number,
                f"the option line holds {words[index - 1]!r}, which is not a frequency unit, "
                "a parameter, a format or R",
            )


def _keyword(found: _File, number: int, body: str) -> None:
    """A Touchstone 2 keyword, and what follows it on its line."""
    head, _, rest = body[1:].partition("]")
    keyword = head.strip().lower()
    value = rest.strip(" \t")
    if keyword not in _KEYWORDS:
        raise found.refuse(
            number, f"holds [{head.strip()}], which is not a keyword the import reads"
        )
    if value and keyword in _BARE:
        raise found.refuse(number, f"holds {value!r} after [{head.strip()}], which takes no value")
    if keyword == "version":
        if value != "2.0":
            raise found.refuse(
                number, f"is Touchstone {value}. Only Touchstone 1 and 2.0 can be imported"
            )
        if found.option or found.numbers:
            raise found.refuse(number, "states [Version] after the option line")
        found.version = "2"
        return
    if found.version != "2":
        raise found.refuse(number, f"holds [{head.strip()}] in a Touchstone 1 file")
    if not found.option:
        raise found.refuse(number, "holds a keyword before the option line")
    found.section = ""
    if keyword == "number of ports":
        if _count(found, number, value) != found.ports:
            raise found.refuse(
                number, f"states {value} ports, and the file name says {found.ports}"
            )
    elif keyword == "two-port data order":
        if value not in ("12_21", "21_12"):
            raise found.refuse(
                number, f"states the two-port data order {value!r}, which is not 12_21 or 21_12"
            )
        found.order = value
    elif keyword == "number of frequencies":
        found.count = _count(found, number, value)
    elif keyword == "reference":
        found.stated = []
        found.section = "reference"
        if value:
            _stated(found, number, value)
    elif keyword == "matrix format":
        if value.lower() not in ("full", "lower", "upper"):
            raise found.refuse(
                number, f"states the matrix format {value!r}, which is not Full, Lower or Upper"
            )
        found.matrix = value.lower()
    elif keyword == "mixed-mode order":
        raise Unreadable(
            f"{found.name} holds mixed-mode data. Only single-ended S parameters can be "
            "imported. Export single-ended S parameters from the instrument or the simulator"
        )
    elif keyword == "begin information":
        found.section = "information"
    elif keyword == "network data":
        found.section = "network"
    elif keyword == "noise data":
        found.section = "noise"
        found.noise = True
    elif keyword == "end":
        found.section = "end"


def _stated(found: _File, number: int, body: str) -> None:
    """Values of ``[Reference]``, one per port, over as many lines as it takes."""
    assert found.stated is not None
    for word in _words(body):
        found.stated.append(_positive(found, number, word, "a reference"))
    if len(found.stated) > found.ports:
        raise found.refuse(number, f"states a reference for more than {found.ports} ports")
    if len(found.stated) == found.ports:
        found.section = ""


def _data(found: _File, number: int, body: str) -> None:
    if not found.option:
        raise found.refuse(number, "holds data before the option line")
    if found.version == "2" and found.section != "network":
        raise found.refuse(number, "holds data outside [Network Data]")
    for index, word in enumerate(_words(body)):
        if not _NUMBER.fullmatch(word):
            raise found.refuse(number, f"holds {word!r} where a number belongs")
        found.numbers.append((float(word), number, index == 0))


def _points(found: _File) -> tuple[np.ndarray, np.ndarray]:
    """The frequencies, in hertz, and the matrix at each."""
    n = found.ports
    if found.version == "2" and n == 2 and found.matrix == "full" and not found.order:
        raise Unreadable(
            f"{found.name} is a Touchstone 2 two-port file and states no [Two-Port Data Order]"
        )
    terms = n * n if found.matrix == "full" else n * (n + 1) // 2
    width = 1 + 2 * terms
    numbers = found.numbers
    points: list[list[float]] = []
    lines: list[int] = []
    at = 0
    while at < len(numbers):
        value, line, opens = numbers[at]
        if found.version == "1" and not opens:
            raise found.refuse(
                line,
                f"does not start a frequency point where the data of a {n}-port file does, "
                f"and the file name says {n} ports",
            )
        if points and value <= points[-1][0]:
            if found.version == "1" and n == 2:
                _noise(found, at, width, points[-1][0])
                break
            raise found.refuse(line, f"lists {value:g} after {points[-1][0]:g}: frequencies rise")
        if value < 0.0:
            raise found.refuse(line, "lists a negative frequency")
        if at + width > len(numbers):
            raise found.refuse(
                line,
                f"ends partway through a frequency point: a {n}-port point holds {width} numbers",
            )
        if found.version == "1":
            _laid_out(found, numbers[at : at + width])
        if found.format == "ma" and any(
            number < 0.0 for number, _, _ in numbers[at + 1 : at + width : 2]
        ):
            raise found.refuse(line, "opens a point holding a negative magnitude")
        points.append([number for number, _, _ in numbers[at : at + width]])
        lines.append(line)
        at += width
    if not points:
        raise Unreadable(f"{found.name} holds no frequency point")
    if found.count is not None and found.count != len(points):
        raise Unreadable(
            f"{found.name} states [Number of Frequencies] {found.count} and holds {len(points)}"
        )
    table = np.asarray(points, dtype=float)
    first, second = table[:, 1::2], table[:, 2::2]
    # What overflows is refused below, by the point it came from.
    with np.errstate(over="ignore", invalid="ignore"):
        frequency = table[:, 0] * _UNITS[found.unit][1]
        if found.format == "ri":
            pairs = first + 1j * second
        else:
            magnitude = 10.0 ** (first / 20.0) if found.format == "db" else first
            phase = np.deg2rad(second)
            pairs = magnitude * (np.cos(phase) + 1j * np.sin(phase))
    held = np.isfinite(frequency) & np.isfinite(pairs).all(axis=1)
    if not held.all():
        raise found.refuse(
            lines[int(np.flatnonzero(~held)[0])],
            "holds a value too large to hold, as a frequency in hertz or as a term",
        )
    return frequency, _arranged(found, pairs)


def _laid_out(found: _File, point: list[tuple[float, int, bool]]) -> None:
    """Refuse a Touchstone 1 point not laid out as the format lays one out.

    A one-port or a two-port point is one line. A larger one is a line for each
    row of its matrix, the first opening with the frequency.
    """
    n = found.ports
    if point[1][2]:
        raise found.refuse(
            point[0][1],
            "holds a frequency alone, and the first values of its point belong on its line",
        )
    if n <= 2:
        if point[-1][1] != point[0][1]:
            raise found.refuse(
                point[0][1],
                "starts a point that runs over more than one line, and a "
                f"{n}-port point is one line",
            )
        return
    for row in range(1, n):
        _, line, opens = point[1 + 2 * n * row]
        if not opens:
            raise found.refuse(
                line, f"does not start row {row + 1} of a {n}-port point on a new line"
            )


def _noise(found: _File, at: int, width: int, last: float) -> None:
    """What follows a two-port's network data: noise lines of five numbers.

    A line of a network point where noise would begin is a sweep whose
    frequency falls, and is refused as one.
    """
    rest = found.numbers[at:]
    starts = [k for k, (_, _, opens) in enumerate(rest) if opens] + [len(rest)]
    for begin, end in zip(starts, starts[1:], strict=False):
        value, line, _ = rest[begin]
        if end - begin == width:
            raise found.refuse(line, f"lists {value:g} after {last:g}: frequencies rise")
        if end - begin != 5:
            raise found.refuse(line, "is a line of noise parameters, which needs five numbers")
    found.noise = True
    found.numbers = found.numbers[:at]


def _arranged(found: _File, pairs: np.ndarray) -> np.ndarray:
    """Each point's values as a matrix, in the order the file lists them."""
    n = found.ports
    s = np.zeros((pairs.shape[0], n, n), dtype=complex)
    if found.matrix != "full":
        k = 0
        for i in range(n):
            for j in range(i + 1) if found.matrix == "lower" else range(i, n):
                s[:, i, j] = s[:, j, i] = pairs[:, k]
                k += 1
        return s
    s[:] = pairs.reshape(-1, n, n)
    if n == 2 and (found.version == "1" or found.order == "21_12"):
        s = s.transpose(0, 2, 1)
    return s


def _reference(found: _File, points: int) -> np.ndarray:
    """Each port's reference at each point, as the file states it."""
    n = found.ports
    if found.impedances:
        at = [taken for taken, _, _ in found.impedances]
        width = 1 + 2 * n * n
        if len(at) != points or at != [width * (k + 1) for k in range(points)]:
            raise Unreadable(
                f"{found.name} states an HFSS port impedance after {len(at)} of its "
                f"{points} points, and needs one after each point"
            )
        return np.asarray([pairs for _, pairs, _ in found.impedances], dtype=complex)
    if found.stated is not None:
        if len(found.stated) != n:
            raise Unreadable(
                f"{found.name} states [Reference] for {len(found.stated)} of {n} ports"
            )
        return np.tile(np.asarray(found.stated, dtype=complex), (points, 1))
    if found.resistance is None:
        raise Unreadable(
            f"{found.name} states R with no value, and no port impedance after each point"
        )
    return np.full((points, n), complex(found.resistance))


def _from_z(found: _File, frequency: np.ndarray, z: np.ndarray) -> np.ndarray:
    """S of a Z normalised to one real reference: ``(z - 1)(z + 1)^-1``.

    Where ``z + 1`` is singular, as at a Z of ``-R``, there is no S, and where it
    is nearly so the solve loses as many figures as its condition number has.
    """
    unit = np.eye(z.shape[1])
    # (z - 1)(z + 1)^-1 is 1 - 2 (z + 1)^-1, and z + 1 is scaled to its largest
    # term before it is inverted, so a Z far from R neither overflows nor
    # underflows on the way.
    scale = np.abs(z + unit).max(axis=(1, 2))[:, None, None]
    with np.errstate(all="ignore"):
        scaled = (z + unit) / scale
        held = np.array([_conditioned(point) for point in scaled], dtype=bool)
        s = np.zeros_like(z)
        if held.all():
            s = unit - 2.0 * np.linalg.solve(scaled, np.broadcast_to(unit, z.shape)) / scale
    if not held.all():
        at = float(frequency[int(np.flatnonzero(~held)[0])])
        raise Unreadable(
            f"{found.name} states Z parameters at {_engineering_hz(at)} that cannot be "
            "converted to S at the option line's R to six figures, because z + 1 is "
            "singular or nearly so there"
        )
    return np.asarray(s, dtype=complex)


def _conditioned(matrix: np.ndarray) -> bool:
    """Whether ``matrix`` inverts to six figures. One of ``nan``, as a zero
    scaled by itself is, does not."""
    try:
        return bool(np.linalg.cond(matrix) < _CONDITION)
    except np.linalg.LinAlgError:
        return False


def _words(text: str) -> list[str]:
    return [word for word in _SPACE.split(text) if word]


def _count(found: _File, number: int, value: str) -> int:
    if not re.fullmatch(r"[0-9]{1,9}", value):
        raise found.refuse(number, f"states {value!r} where a count belongs")
    return int(value)


def _positive(found: _File, number: int, word: str, what: str) -> float:
    if not _NUMBER.fullmatch(word) or not 0.0 < float(word) < math.inf:
        raise found.refuse(number, f"states {word!r} for {what}, which must be a positive number")
    return float(word)
