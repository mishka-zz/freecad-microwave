# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How a backend modelled each lossy material and its outside, in words.
Solver-neutral.

Two backends solving one document do not solve one model. A loss tangent is held
across the band by one and folded into one conductivity at one frequency by
the other, and a conducting sheet inside the region is two faces carrying a
surface impedance to one and a plane carrying the net current to the other.
Neither difference shrinks with a finer mesh, and neither shows in the matrix.

So each adapter records what it modelled, one record to a material, in its
run's provenance under ``modelled``. A record is a plain mapping, because the
provenance is written as JSON and read back from a document:

``{"material": name, "held": "loss tangent", "loss_tangent": t}``
    a loss tangent held across the band.
``{"material": name, "held": "conductivity", "conductivity": s}``
    a conductivity in S/m held across the band. With ``"folded": k`` and
    ``"at": f``, the conductivity ``k`` of it is a loss tangent folded in at
    ``f`` Hz, so the loss tangent the run models falls as one over frequency.
``{"material": name, "sheet": "net current", "conductivity": s, "thickness": d}``
    a sheet driven by the current through it as a whole, ``d`` in millimetres.
``{"material": name, "sheet": "surface impedance", "conductivity": s,
"thickness": d, "faces": [...]}``
    the metal's surface impedance on a face, passing nothing across it.
    ``faces`` holds ``"inside"`` where a sheet stands inside the region, and so
    has two faces, and ``"boundary"`` where it stands where the model ends.

``{"boundary": "absorbing", "order": n, "clearance": d, "faces": [...],
"magnetic": [...]}``
    the outside of the model is an absorbing condition of order ``n`` on
    ``faces``, as ``{axis}{side}``, standing ``d`` millimetres from the
    structure. The air beside a port's face on each side in ``magnetic`` is a
    magnetic wall.

``{"medium": name}``
    every space no bound body fills is that material rather than vacuum. A run
    in vacuum records none, open study or closed.
``{"boundary": "absorbing layer", "cells": [...], "clearance": d,
"faces": [...], "through": [...], "ends": [...]}``
    the outside of the model is an absorbing layer, ``cells`` deep on each face
    in the order ``{axis}{side}``, standing ``d`` millimetres clear of the
    structure on ``faces``. The structure runs out through the layer on each side
    in ``through``, where the layer stands nowhere clear of it. On each side in
    ``ends`` the domain ends on the structure and the layer stands beyond the
    waveguide ports that cover that side.

A run whose outside is a wall on every face records no boundary, so a result from
a study open to free space and one from the same drawing closed are told apart
after the run.

A lossless material and a perfect conductor are not recorded: every backend here
models them alike. Nothing below branches on which backend wrote a record. The
words follow from the record, and two records are compared field by field.

The two outsides are **not** compared field by field, because no field of one
means what the same field of the other means: a condition on a surface and a
layer of cells absorb differently, and one's order is not the other's depth.
What both state is which faces they held open and how far the boundary stood
from the structure, and :func:`apart` compares those. It compares them only where
both runs recorded an outside: a run that recorded none says nothing here about
its own. The medium is compared on its own record, whatever the outside.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = ["apart", "brief", "outside", "said"]

#: A record, as the provenance holds one.
Record = Mapping[str, Any]

#: What an outside record calls an absorbing layer of cells, as against a
#: condition written on a surface.
LAYER = "absorbing layer"


def outside(record: Record) -> bool:
    """Whether a record is of the model's outside rather than of a material."""
    return "boundary" in record


def _listed(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def medium(record: Record) -> bool:
    """Whether a record is of the study's medium."""
    return "medium" in record and not outside(record)


def _loss(record: Record) -> bool:
    """Whether a record is of one material's loss."""
    return not outside(record) and not medium(record)


def _boundary(record: Record) -> str:
    """What the run modelled its outside as."""
    faces = list(record.get("faces") or ())
    if record["boundary"] == LAYER:
        deep = "/".join(str(int(count)) for count in record.get("cells") or ())
        clear = (
            f", {float(record['clearance']):.4g} mm clear of the structure on {_listed(faces)}"
            if faces
            else ""
        )
        through = list(record.get("through") or ())
        also = f"; the structure runs out through it on {_listed(through)}" if through else ""
        ends = list(record.get("ends") or ())
        if ends:
            also += (
                f"; it stands beyond the waveguide ports covering {_listed(ends)}, where the "
                "domain ends on the structure"
            )
        return f"an absorbing layer {deep} cells deep{clear}{also}"
    magnetic = list(record.get("magnetic") or ())
    also = (
        f"; the air beside a port's face on {_listed(magnetic)} is a magnetic wall"
        if magnetic
        else ""
    )
    return (
        f"an absorbing condition of order {int(record['order'])} on {_listed(faces)}, "
        f"{float(record['clearance']):.4g} mm from the structure{also}"
    )


def _thickness(record: Record) -> str:
    thickness = float(record.get("thickness") or 0.0)
    return f", {thickness:g} mm thick" if thickness else ""


def _how(record: Record) -> str:
    """What the run modelled, without the material's name."""
    if record.get("sheet") == "net current":
        return (
            f"a conducting sheet of {record['conductivity']:.3g} S/m{_thickness(record)}, "
            "carrying the net current through it"
        )
    if record.get("sheet") == "surface impedance":
        where = set(record.get("faces") or ())
        on = {
            frozenset({"inside"}): "on each of its two faces, passing nothing between them",
            frozenset({"boundary"}): "on the one face where the model ends",
        }.get(
            frozenset(where),
            "on each of its two faces where it stands inside the model, passing nothing "
            "between them, and on its one face where the model ends",
        )
        return f"the surface impedance of {record['conductivity']:.3g} S/m{_thickness(record)} {on}"
    if record.get("held") == "loss tangent":
        return f"a loss tangent of {record['loss_tangent']:g}, held across the band"
    conductivity = float(record["conductivity"])
    folded = float(record.get("folded") or 0.0)
    if not folded:
        return f"a conductivity of {conductivity:.3g} S/m, held across the band"
    own = conductivity - folded
    added = f", added to its own {own:.3g} S/m" if own else ""
    return (
        f"a loss tangent folded at {float(record['at']) / 1e9:.4g} GHz into a conductivity "
        f"of {folded:.3g} S/m{added} and held across the band, so the loss tangent "
        "modelled falls as one over frequency"
    )


def said(records: Iterable[Record]) -> list[str]:
    """One line to a record: the material, and what the run modelled, or the
    outside and what the run modelled it as."""
    return [
        f"The outside: {_boundary(record)}"
        if outside(record)
        else f"The medium: {record['medium']!r} fills every space no bound body fills"
        if medium(record)
        else f"{record['material']!r}: {_how(record)}"
        for record in records
    ]


def _short(record: Record) -> str:
    if record.get("sheet") == "net current":
        return "sheet by its net current"
    if record.get("sheet") == "surface impedance":
        return {
            frozenset({"inside"}): "sheet by its two faces",
            frozenset({"boundary"}): "sheet where the model ends",
        }.get(
            frozenset(record.get("faces") or ()),
            "sheet by its faces inside and where the model ends",
        )
    if record.get("held") == "loss tangent":
        return "loss tangent held across the band"
    folded = float(record.get("folded") or 0.0)
    if folded:
        at = f"loss tangent folded at {float(record['at']) / 1e9:.4g} GHz"
        return f"{at}, added to its conductivity" if float(record["conductivity"]) != folded else at
    return "conductivity held across the band"


def brief(records: Iterable[Record]) -> str:
    """Every material's record in one clause, for the line under a chart that
    says how the loss was held. Empty for none."""
    return "; ".join(
        f"{record['material']!r} {_short(record)}" for record in records if _loss(record)
    )


def _model(record: Record) -> tuple[Any, ...]:
    """What decides the model a record states, and none of its values.

    A value differs between two runs where one backend computed it from another
    - a folded conductivity is a loss tangent times a frequency - so comparing
    values would call two runs of one model different. What the loss is held as,
    and what a sheet passes, is the model. A conductivity a loss tangent was
    folded into is held across the band as one stated as a conductivity is, so
    the two are one model.
    """
    if "sheet" in record:
        return ("sheet", record["sheet"], tuple(sorted(record.get("faces") or ())))
    return ("held", record.get("held"))


def apart(
    mine: Sequence[Record], mine_by: str, theirs: Sequence[Record], theirs_by: str
) -> list[str]:
    """One line for each material two runs modelled differently.

    :param mine_by: and ``theirs_by``: what solved each, as its provenance names
        it, which is how the line tells the reader which is which.

    A material is compared by every record either run holds of it, since
    FreeCAD can be set to let two materials share a label. A material only one
    of the two recorded gets a line as well: two runs of one drawing record one
    set of materials, so the two answer different drawings, or one of them
    modelled as lossless what the other did not.

    The outside gets a line where the two runs held different faces open, or
    stood their boundaries different distances from the structure. Nothing else
    about it is compared: see this module's own description.
    """
    ours = _by_material([record for record in mine if _loss(record)])
    others = _by_material([record for record in theirs if _loss(record)])
    lines = _outsides(mine, mine_by, theirs, theirs_by)
    stood_in, other_stood_in = _medium(mine), _medium(theirs)
    if stood_in != other_stood_in:
        lines.append(
            f"the two runs stood in different media: {_named(stood_in)} in what {mine_by} "
            f"solved, and {_named(other_stood_in)} in what {theirs_by} solved"
        )
    for name in dict.fromkeys([*ours, *others]):
        held, other = ours.get(name, []), others.get(name, [])
        if sorted(map(_model, held)) == sorted(map(_model, other)):
            continue
        if not other or not held:
            lossy, solved, lossless = (
                (mine_by, held, theirs_by) if held else (theirs_by, other, mine_by)
            )
            lines.append(
                f"{name!r} is lossy in what {lossy} solved and not in what {lossless} "
                f"solved: {lossy} solved {'; '.join(map(_how, solved))}"
            )
            continue
        lines.append(
            f"{name!r} is not one model in the two: {mine_by} solved "
            f"{'; '.join(map(_how, held))}, and {theirs_by} {'; '.join(map(_how, other))}"
        )
    return lines


#: How far apart two runs' boundaries stand and still stand in the same place, as
#: a share of the smaller distance. A clearance is derived from the band where the
#: document leaves it at zero, and the two backends derive it from one expression,
#: so equal distances are equal to the last bit. This tolerance is for the two
#: having been rounded on their way through a provenance file.
APART = 1e-9


def _outsides(
    mine: Sequence[Record], mine_by: str, theirs: Sequence[Record], theirs_by: str
) -> list[str]:
    """A line where two runs held different faces open, and one where their
    boundaries stood different distances from the structure.

    Asked only where both runs recorded an outside. A run whose outside is a wall
    on every face records none, and one from a backend that records none says
    nothing about its own - neither is a claim that the two models differ.
    """
    ours = [record for record in mine if outside(record)]
    others = [record for record in theirs if outside(record)]
    if not ours or not others:
        return []
    lines = []
    open_faces, other_faces = _faces(ours), _faces(others)
    if open_faces != other_faces:
        lines.append(
            f"the two runs held different faces open: {mine_by} held "
            f"{_held(open_faces)}, and {theirs_by} held {_held(other_faces)}"
        )
    stood, other_stood = _stood(ours), _stood(others)
    if abs(stood - other_stood) > APART * min(stood, other_stood):
        lines.append(
            f"the two runs stood their boundaries different distances from the structure: "
            f"{stood:.4g} mm in what {mine_by} solved, and {other_stood:.4g} mm in what "
            f"{theirs_by} solved"
        )
    return lines


def _faces(records: Sequence[Record]) -> tuple[str, ...]:
    """Every face a run held open, in the order a face name sorts."""
    return tuple(sorted({face for record in records for face in record.get("faces") or ()}))


def _held(faces: Sequence[str]) -> str:
    return _listed(list(faces)) if faces else "none"


def _medium(records: Sequence[Record]) -> str:
    """The medium a run stood in, and empty for vacuum."""
    return next((str(record["medium"]) for record in records if medium(record)), "")


def _named(medium: str) -> str:
    return repr(medium) if medium else "vacuum"


def _stood(records: Sequence[Record]) -> float:
    """The nearest a run's boundary stood to the structure, in millimetres."""
    return min(float(record.get("clearance") or 0.0) for record in records)


def _by_material(records: Sequence[Record]) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = {}
    for record in records:
        grouped.setdefault(str(record["material"]), []).append(record)
    return grouped
