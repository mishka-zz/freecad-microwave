# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The configuration Palace is run on, as a value.

This is Palace's own input format rather than a description of ours. There is no
process boundary here for a private envelope to cross: the mesher is asked for a
mesh, the configuration is written beside it, and Palace reads both. So what this
module holds is the vocabulary of that file - attributes, conditions, a sweep -
and the translation from what a user drew into those numbers is somebody else's.

Nothing here opens a file or starts a process, and Palace need not be installed
to build one of these or to read it back.

Frequencies arrive in Hertz and are written in gigahertz. Lengths arrive in
millimetres and are written unchanged, with ``Model.L0`` saying what one of them
is in metres.

What is a constant here and what is a field
-------------------------------------------

A field is something a drawing or a user decides. The linear solver's own
tolerances are neither: nothing has varied them, and a setting nobody has moved
is one nobody can defend. They stay constants until something measures them.
The cap on its iterations is a constant as well, and there are two: one for a
closed study and one for an open one, which takes further to invert. Each is set
far enough above what its own kind of problem takes that it refuses only a solve
that is not converging.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .capabilities import HZ_PER_GHZ, METRES_PER_UNIT

__all__ = [
    "ABSORBING_ORDER",
    "CONVERGENCE_MEMORY",
    "ESTIMATOR_ITERATIONS",
    "FEWEST_SOLVES",
    "RANK_LOSS_TANGENT",
    "Adaptive",
    "DIRECTIONS",
    "Driven",
    "Filling",
    "LumpedElement",
    "LumpedPort",
    "Material",
    "SurfaceConductivity",
    "Sweep",
    "WavePort",
]

#: How much Palace says about what it is doing. The runs this adapter reads are
#: judged on what they printed as much as on what they wrote, and the two
#: mistakes that leave a plausible answer behind are only ever announced in the
#: log.
VERBOSE = 2

#: The linear solver. ``Default`` lets Palace choose the preconditioner for the
#: problem type and the element order it was given.
LINEAR_SOLVER = "Default"
KRYLOV_SOLVER = "GMRES"
LINEAR_TOLERANCE = 1.0e-10

#: The most iterations the linear solver takes before it stops unconverged, and
#: the run is refused: one for a closed study, and one for a study open to free
#: space, whose operator GMRES takes further to invert.
#:
#: An absorbing surface makes the operator non-Hermitian and slows GMRES, and the
#: farther the surface stands the more iterations a frequency takes. So the closed
#: cap refuses an open solve that is converging, only slowly, and the open cap
#: stands above what an open solve costs. It is raised only where the
#: study is open, so a closed run that is not converging is still refused at the
#: iteration it was refused at before.
#:
#: Raising it changes no run that converged below it. Palace stops at the
#: tolerance before it tests the cap (``palace/linalg/iterative.cpp:625``), and
#: the Krylov space's size, which is the cap where nothing else sets it
#: (``palace/utils/iodata.cpp:503-505``), is allocated ten vectors at a time as
#: the iterations reach it (``palace/linalg/iterative.cpp:484-514``), so the same
#: iterations reach the same answer and hold the same memory.
LINEAR_ITERATIONS = 100
OPEN_LINEAR_ITERATIONS = 1000

#: The most iterations the conjugate gradient solve behind Palace's error
#: estimate takes, which is Palace's own default
#: (``palace/utils/configfile.hpp``, ``estimator_max_it``). It is written because
#: the log is judged by it. An estimate that stops short and the projection of a
#: lumped port's excitation that stops short print the same words, and the count
#: of iterations each stopped at is what separates them: the projection's is
#: fixed at 200 (``palace/models/spaceoperator.cpp``,
#: ``ProjectBdrCoefficientViaMassSolve``).
ESTIMATOR_ITERATIONS = 10000


#: The order of the absorbing condition on the open sides of the reserved air.
#: Palace 0.18.1 adds the second-order condition's curl-curl term with the
#: sign opposite to the weak form of the reference it cites
#: (``palace/models/farfieldboundaryoperator.cpp:108-145``), so a wave meeting
#: the surface at a slant is reflected more at the second order than at the
#: first. The first order absorbs a wave meeting the surface squarely exactly.
#: See ``docs/internals/palace-open-boundary.md``.
ABSORBING_ORDER = 1

#: The loss tangent each lossless material is written with where Palace's ranks
#: could disagree about a port's assembly - :attr:`Driven.ranks_can_disagree`.
#: The imaginary permittivity it gives is this fraction of the real one, far
#: below the relative precision of a double, so the run answers as it does
#: without it to within the solver's tolerance.
RANK_LOSS_TANGENT = 1e-30


#: How many full solves in a row an adaptive sweep takes inside its tolerance
#: before it stops sampling. Palace's own default, stated so that
#: :data:`FEWEST_SOLVES` follows from it.
CONVERGENCE_MEMORY = 2

#: The fewest full solves an adaptive sweep can converge in. It solves both ends
#: of the band first and counts neither as inside the tolerance
#: (``palace/drivers/drivensolver.cpp``, ``SweepAdaptive``), so a sweep capped
#: below this stops at its cap unconverged every time.
FEWEST_SOLVES = 2 + CONVERGENCE_MEMORY


def _ghz(hertz: float) -> float:
    return float(hertz) / HZ_PER_GHZ


@dataclass(frozen=True)
class Filling:
    """What fills a region, in the terms this solver states a material in.

    Loss is one model or the other. Palace stops on a material carrying both,
    and a loss tangent and a conductivity are two ways of saying one thing, so
    the pair cannot be built and the caller says which object it came from.
    """

    permittivity: float = 1.0
    permeability: float = 1.0
    loss_tangent: float = 0.0
    conductivity: float = 0.0

    def __post_init__(self) -> None:
        for name, stated in (
            ("permittivity", self.permittivity),
            ("permeability", self.permeability),
        ):
            if not (math.isfinite(stated) and stated > 0):
                raise ValueError(
                    f"a relative {name} of {stated:g} carries no wave: it is a positive number"
                )
        for name, loss, unit in (
            ("loss tangent", self.loss_tangent, ""),
            ("conductivity", self.conductivity, " S/m"),
        ):
            if not math.isfinite(loss):
                raise ValueError(f"a {name} of {loss:g}{unit} is not a number a loss can be")
            if loss < 0:
                raise ValueError(
                    f"a {name} of {loss:g}{unit} is not a loss: a negative one would be a "
                    "material that supplies energy"
                )
        if self.loss_tangent and self.conductivity:
            raise ValueError(
                f"a loss tangent of {self.loss_tangent} and a conductivity of "
                f"{self.conductivity} are two models of one loss, and this solver "
                "takes one or the other"
            )

    @property
    def slowing(self) -> float:
        """The product whose square root a wave's speed is divided by in here.

        Permeability belongs in it. Leaving it out meshes a ferrite as though it
        were not magnetic, by a factor of the root of its permeability.
        """
        return self.permittivity * self.permeability

    def to_dict(self) -> dict[str, Any]:
        written: dict[str, Any] = {
            "Permittivity": self.permittivity,
            "Permeability": self.permeability,
        }
        if self.loss_tangent:
            written["LossTan"] = self.loss_tangent
        if self.conductivity:
            written["Conductivity"] = self.conductivity
        return written


@dataclass(frozen=True)
class Material:
    """What fills a set of domain attributes.

    A domain attribute no material names is removed from the mesh with one line
    of print, so a region left out here is a region that quietly stops existing.
    """

    attributes: tuple[int, ...]
    filling: Filling = Filling()

    def __post_init__(self) -> None:
        if not self.attributes:
            raise ValueError("a material fills at least one attribute")

    def to_dict(self) -> dict[str, Any]:
        return {"Attributes": list(self.attributes), **self.filling.to_dict()}


@dataclass(frozen=True)
class SurfaceConductivity:
    """Metal of finite conductivity, as a condition on a set of boundary attributes.

    Palace imposes the surface impedance of a conductor at each frequency, from
    the skin depth its conductivity and permeability give there. A thickness
    multiplies that by a factor that reaches the resistance of the whole layer
    where the layer is thin against the skin depth, and zero leaves the factor
    out, which is a conductor thick against it.

    :param conductivity: in siemens per metre.
    :param permeability: the metal's own, relative. It sets the skin depth.
    :param thickness: in the mesh's own length units, and zero for none.
    :param external: whether the attributes stand where the model ends. Palace
        takes the thickness twice over there, since the condition stands on one
        side of the layer (``palace/models/surfaceconductivityoperator.cpp:91-95``).
    """

    attributes: tuple[int, ...]
    conductivity: float
    permeability: float = 1.0
    thickness: float = 0.0
    external: bool = False

    def __post_init__(self) -> None:
        if not self.attributes:
            raise ValueError("a conductivity condition stands on at least one attribute")
        if not (math.isfinite(self.conductivity) and self.conductivity > 0):
            raise ValueError(
                f"a conductivity condition carries {self.conductivity} S/m, and a metal conducts"
            )
        if not (math.isfinite(self.permeability) and self.permeability > 0):
            raise ValueError(
                f"a conductivity condition carries a permeability of {self.permeability}, "
                "and a relative permeability is positive"
            )
        if not (math.isfinite(self.thickness) and self.thickness >= 0):
            raise ValueError(
                f"a conductivity condition is {self.thickness} thick, and a thickness is "
                "zero or positive"
            )

    def to_dict(self) -> dict[str, Any]:
        written: dict[str, Any] = {
            "Attributes": list(self.attributes),
            "Conductivity": self.conductivity,
            "Permeability": self.permeability,
            "External": self.external,
        }
        if self.thickness:
            written["Thickness"] = self.thickness
        return written


@dataclass(frozen=True)
class WavePort:
    """One port on the outside of the model, solved as a cross-section.

    :param index: the port's number, which is the document's. It names the port
        in every output table and is unique across every port and source in the
        run.
    :param attributes: the mesh attributes the port face carries.
    :param behind: a point on the model's side of the face, in the mesh's own
        length units. Palace measures the power through the face in the
        direction leading away from it, so what it reports is what leaves the
        model there. A point in the plane of the face leaves that direction to
        rounding, and that is where Palace puts one when none is given
        (``palace/models/surfacepostoperator.cpp:105-120``).
    :param mode: which mode of the cross-section, ranked by decreasing wave
        number. It is an ordinal and not a name.
    :param offset: how far the reference plane stands from the port face, in the
        mesh's own length units. Palace de-embeds the result by this distance.
    :param excited: whether this port is driven. Which excitation it lands in is
        not stated here: every port marked ``true`` in one configuration would
        share one, and a run whose excitations do not drive exactly one port
        each writes no scattering matrix and says nothing about it. So the index
        is derived below, and the mistake cannot be written.
    :param voltage: the two ends of the line across the face the mode's voltage
        is read along, in the mesh's own length units, or ``None``. Palace then
        states the port's power-voltage impedance, ``|V|^2 / 2P`` with ``P`` the
        power the mode carries, and turns the mode so that the voltage is positive
        (``palace/models/waveportoperator.cpp:963-979``), which takes the place
        of the sign it would otherwise fix by a rule of its own.
    """

    index: int
    attributes: tuple[int, ...]
    behind: tuple[float, float, float]
    mode: int = 1
    offset: float = 0.0
    excited: bool = False
    voltage: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    def __post_init__(self) -> None:
        if self.index <= 0:
            raise ValueError(f"a port number is positive, and this one is {self.index}")
        if not self.attributes:
            raise ValueError(f"port {self.index} stands on at least one attribute")
        if len(self.behind) != 3 or not all(math.isfinite(value) for value in self.behind):
            raise ValueError(
                f"port {self.index} is measured from {self.behind}, and a point is three "
                "finite coordinates"
            )
        if self.mode <= 0:
            raise ValueError(f"port {self.index} asks for mode {self.mode}, and a mode is ranked")
        if self.offset < 0:
            raise ValueError(
                f"port {self.index} de-embeds by {self.offset}, and the reference plane "
                "stands inside the model rather than outside it"
            )

    def to_dict(self, excitation: int) -> dict[str, Any]:
        """The port, driven under ``excitation`` or not driven where it is zero."""
        written: dict[str, Any] = {
            "Index": self.index,
            "Attributes": list(self.attributes),
            "Mode": self.mode,
            "Offset": self.offset,
        }
        if excitation:
            written["Excitation"] = excitation
        if self.voltage is not None:
            written["VoltagePath"] = [list(end) for end in self.voltage]
        return written

    def to_flux(self) -> dict[str, Any]:
        """The power through the port's face, measured leaving the model.

        Indexed by the port's own number, which names it in this table as it
        does in every other: Palace numbers each kind of postprocessing apart
        from the ports (``palace/utils/configfile.cpp:186-204``).
        """
        return {
            "Index": self.index,
            "Attributes": list(self.attributes),
            "Type": "Power",
            "Center": list(self.behind),
        }


#: How a lumped element's direction is written: along an axis, with its sign.
DIRECTIONS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


@dataclass(frozen=True)
class LumpedElement:
    """One face of a lumped port, and the axis it is driven along.

    :param attributes: the mesh attributes the face carries.
    :param direction: one of :data:`DIRECTIONS`. Palace reads the port's voltage
        from the field along it (``LumpedPortData::GetVoltage`` in
        ``palace/models/lumpedportoperator.cpp``), so the sign says which end of
        the element is the positive terminal. It takes a direction within a
        degree of an axis of the box round the element's faces, and the
        element's length along that axis (``UniformElementData`` in
        ``palace/fem/lumpedelement.cpp``).
    """

    attributes: tuple[int, ...]
    direction: str

    def __post_init__(self) -> None:
        if not self.attributes:
            raise ValueError("a lumped element stands on at least one attribute")
        if self.direction not in DIRECTIONS:
            raise ValueError(
                f"a lumped element is driven along {self.direction!r}, and a direction is "
                f"one of {', '.join(DIRECTIONS)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"Attributes": list(self.attributes), "Direction": self.direction}


@dataclass(frozen=True)
class LumpedPort:
    """A resistor across a gap, on elements that stand in parallel.

    :param index: the port's number, as a wave port's.
    :param elements: the faces the resistor is laid on. They stand in parallel:
        Palace gives each element a sheet resistance of the port's resistance
        times its own width over its length times the number of elements
        (``LumpedPortData::GetToSquare`` in
        ``palace/models/lumpedportoperator.hpp``).
    :param resistance: in ohms. Palace references a lumped port's scattering
        parameters to it (``LumpedPortData::GetExcitationRefResistance`` in
        ``palace/models/lumpedportoperator.hpp``), and stops on a port carrying
        no resistance, inductance or capacitance (the ``LumpedPortData``
        constructor in ``palace/models/lumpedportoperator.cpp``). A port here is
        a resistor, so it is positive.
    :param excited: as a wave port's.
    """

    index: int
    elements: tuple[LumpedElement, ...]
    resistance: float
    excited: bool = False

    def __post_init__(self) -> None:
        if self.index <= 0:
            raise ValueError(f"a port number is positive, and this one is {self.index}")
        if not self.elements:
            raise ValueError(f"lumped port {self.index} stands on at least one element")
        if not (math.isfinite(self.resistance) and self.resistance > 0):
            raise ValueError(
                f"lumped port {self.index} carries {self.resistance} ohm, and a resistance "
                "is positive"
            )

    @property
    def attributes(self) -> tuple[int, ...]:
        """Every attribute an element of the port carries."""
        return tuple(sorted({a for element in self.elements for a in element.attributes}))

    def to_dict(self, excitation: int) -> dict[str, Any]:
        """The port, driven under ``excitation`` or not driven where it is zero.

        ``Active`` is not written. Palace defaults it on, and a port written
        inactive loses its resistor as well as its place in the matrix
        (``LumpedPortOperator::GetRsAttrList`` in
        ``palace/models/lumpedportoperator.cpp``).
        """
        written: dict[str, Any] = {
            "Index": self.index,
            "R": self.resistance,
            "Elements": [element.to_dict() for element in self.elements],
        }
        if excitation:
            written["Excitation"] = excitation
        return written

    def to_flux(self) -> dict[str, Any]:
        """The power the port's elements give the model, summed over both sides.

        Palace does not duplicate a lumped port's faces where the model stands
        on both sides of them (``AddInterfaceBdrElements`` in
        ``palace/utils/geodata.cpp``), and a two-sided flux over such a face is
        the flux on one side less the flux on the other
        (``SurfaceFunctional::AssembleLocal`` in
        ``palace/fem/output_functionals.cpp``): what the face puts into the model
        on both sides. On a face where the model ends there is one side,
        and it is the power going in. So one rule measures an element inside the
        model and one where it ends, and the power leaving the model through the
        port is this with the sign turned, which :mod:`.read` applies.
        """
        return {
            "Index": self.index,
            "Attributes": list(self.attributes),
            "Type": "Power",
            "TwoSided": True,
        }


@dataclass(frozen=True)
class Adaptive:
    """A sweep that solves a few frequencies in full and answers every point
    from a reduced model built of them.

    Palace picks each next frequency where the model's error estimate is
    largest, and stops once :data:`CONVERGENCE_MEMORY` solves in a row came
    within ``tolerance`` of the model. The tolerance is on the field, relative
    to its norm, and not on any scattering parameter.

    :param tolerance: the relative field error the model is built to.
    :param solves: the most full solves for each driven port. A sweep that
        takes them all before as many in a row as it needs come within the
        tolerance has not converged, and Palace answers from the model
        regardless.
    """

    tolerance: float
    solves: int

    def __post_init__(self) -> None:
        if not 0 < self.tolerance < 1:
            raise ValueError(
                f"the sweep tolerance is {self.tolerance}, and a relative error is "
                "above zero and below one"
            )
        if self.solves < FEWEST_SOLVES:
            raise ValueError(
                f"an adaptive sweep is allowed {self.solves} full solves, and it solves "
                f"both ends of the band and then needs {CONVERGENCE_MEMORY} more inside "
                f"its tolerance, so it can stop no sooner than {FEWEST_SOLVES}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "AdaptiveTol": self.tolerance,
            "AdaptiveMaxSamples": self.solves,
            "AdaptiveConvergenceMemory": CONVERGENCE_MEMORY,
        }


@dataclass(frozen=True)
class Sweep:
    """The band, and how many points across it.

    A band of some width is written as one linearly spaced block. The older way
    of saying the same thing - a pair of bounds and a step - is deprecated in
    Palace's own schema, and a step derived from a count is a division that has
    to come out exactly right at the far end of the band.

    A band that is one frequency is written as that frequency, listed. Palace
    spaces a linear block by dividing the width by one less than the count, so a
    block of one sample divides by zero and the run solves at a frequency that
    is not a number, writes a table of them and exits clean. It also drops
    repeated samples after expanding a block, so a start and a stop that are
    equal come back as one row however many were asked for.

    Every point is solved in full unless the sweep is ``adaptive``, which is
    allowed only where the points outnumber the full solves it may take:
    otherwise solving each point costs no more than the most the adaptive sweep
    may, and is exact at each.
    """

    start: float
    stop: float
    points: int
    adaptive: Adaptive | None = None

    def __post_init__(self) -> None:
        if self.start <= 0 or self.stop <= 0:
            raise ValueError(
                f"the band runs from {self.start} Hz to {self.stop} Hz, and a "
                "frequency-domain solve has nothing to answer at zero"
            )
        if self.stop < self.start:
            raise ValueError(f"the band ends at {self.stop} Hz and starts at {self.start} Hz")
        if self.points <= 0:
            raise ValueError(f"the band is asked for at {self.points} points")
        if self.adaptive is not None and self.samples <= self.adaptive.solves:
            raise ValueError(
                f"the band is asked for at {self.samples} points and an adaptive sweep is "
                f"allowed {self.adaptive.solves} full solves, as many as solving each point "
                "may take"
            )

    @property
    def highest(self) -> float:
        """The highest frequency the run solves at, in Hz: the start, where the
        band is written as one frequency."""
        return self.start if self._is_one_frequency else self.stop

    @property
    def samples(self) -> int:
        """How many rows the run writes, which is not always what was asked for."""
        return 1 if self._is_one_frequency else self.points

    @property
    def _is_one_frequency(self) -> bool:
        return self.points == 1 or self.start == self.stop

    def to_dict(self) -> dict[str, Any]:
        if self._is_one_frequency:
            return {"Type": "Point", "Freq": [_ghz(self.start)]}
        return {
            "Type": "Linear",
            "MinFreq": _ghz(self.start),
            "MaxFreq": _ghz(self.stop),
            "NSample": self.points,
        }


@dataclass(frozen=True)
class Driven:
    """A frequency-domain driven run, whole.

    :param mesh: the mesh file, by path, in millimetres.
    :param output: where Palace writes its tables.
    :param materials: what fills the domain attributes.
    :param perfect_conductor: the attributes carrying a perfect electric
        conductor.
    :param ports: the ports, of which at least one is driven. A wave port and a
        lumped port are not written into one run.
    :param sweep: the band and the points across it.
    :param order: the solver's polynomial order, which is Palace's own setting
        and decides the answer where the element count has stopped mattering.
        Stated rather than defaulted. The figure this workbench ships is on the
        solver object the user edits, and a default here would be a second copy
        of it that no edit reaches.
    :param conducting: the metal of finite conductivity. Between these and
        ``perfect_conductor`` there is at least one attribute carrying metal: an
        exterior attribute nobody names becomes a perfect magnetic conductor
        with a warning, so what is left out here is not left out of the problem.
    :param magnetic_conductor: the attributes carrying a perfect magnetic
        conductor: the rest of a face where the model ends that a lumped element
        lies in.
    :param absorbing: the attributes carrying the absorbing condition: the open
        sides of the air an open study reserves. They name the outside of the
        model as metal does, so a run holding them needs no metal.
    """

    mesh: str
    output: str
    materials: tuple[Material, ...]
    perfect_conductor: tuple[int, ...]
    ports: tuple[WavePort | LumpedPort, ...]
    sweep: Sweep
    order: int
    conducting: tuple[SurfaceConductivity, ...] = ()
    magnetic_conductor: tuple[int, ...] = ()
    absorbing: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.order <= 0:
            raise ValueError(f"the solver order is {self.order}, and an order is positive")
        if not self.materials:
            raise ValueError("no material fills the mesh, and Palace deletes what none names")
        if not self.perfect_conductor and not self.conducting and not self.absorbing:
            raise ValueError(
                "no attribute carries metal or an absorbing condition, and an exterior "
                "attribute nobody names becomes a perfect magnetic conductor"
            )
        numbers = [port.index for port in self.ports]
        repeated = sorted({number for number in numbers if numbers.count(number) > 1})
        if repeated:
            raise ValueError(
                f"more than one port carries the number {repeated}, and a number names "
                "one port in every table the run writes"
            )
        if not any(port.excited for port in self.ports):
            raise ValueError("no port is driven, so the run has nothing to answer")
        if {type(port) for port in self.ports} == {WavePort, LumpedPort}:
            raise ValueError(
                "the run holds a wave port and a lumped port, and a port's mode run is "
                "written for a run of wave ports alone"
            )
        named = [
            *self.perfect_conductor,
            *self.magnetic_conductor,
            *self.absorbing,
            *(a for port in self.ports for a in port.attributes),
        ]
        twice = sorted({a for a in named if named.count(a) > 1})
        if twice:
            raise ValueError(
                f"the attributes {twice} each carry more than one condition, and a face carries one"
            )

    @property
    def dissipates(self) -> bool:
        """Whether anything in the model turns power into heat: a material
        with a loss tangent or a conductivity, or metal of finite conductivity.
        """
        return bool(self.conducting) or self.lossy_filling

    @property
    def lossy_filling(self) -> bool:
        """Whether a material filling part of the model has a loss tangent or a
        conductivity."""
        return any(
            material.filling.loss_tangent or material.filling.conductivity
            for material in self.materials
        )

    @property
    def ranks_can_disagree(self) -> bool:
        """Whether each lossless material is written with :data:`RANK_LOSS_TANGENT`.

        A wave port's mode solve assembles an imaginary matrix, which is a
        collective, where a material on the port's face carries a loss tangent
        or a conductivity, or where a boundary term lies on the face's edges
        (``palace/models/modeeigensolver.cpp:143-145`` and ``:319-321``). The
        solve holds only the materials on the face
        (``palace/models/waveportoperator.cpp:575-576``) and reduces whether
        they are lossy across the ranks
        (``palace/models/materialoperator.cpp:405-409``). Each rank decides the
        boundary term from the elements it holds. So where metal of finite
        conductivity meets a port and no material on its face is lossy, a rank
        holding none of the metal's edges on the face skips the assembly the
        others wait on, and the run never ends. A loss tangent on every material
        makes every rank assemble. The adapter does not know which metal meets
        which face, so any metal and any wave port are enough.
        """
        return bool(self.conducting) and any(isinstance(port, WavePort) for port in self.ports)

    @property
    def radiated(self) -> int | None:
        """The index the power through the absorbing sides is measured under, or
        ``None`` where the run has none.

        Past every port's number, since each port's own flux is indexed by its
        number and Palace keeps one set of indices for the whole table.
        """
        if not self.absorbing:
            return None
        return max(port.index for port in self.ports) + 1

    def to_radiated_flux(self) -> dict[str, Any]:
        """The power through the absorbing sides, summed over both sides of each face.

        Written two-sided for the rule :meth:`LumpedPort.to_flux` states: on a
        face where the model ends there is one side, and the figure is the power
        going in, whatever way the face is turned. The power leaving is this with
        the sign turned, which :mod:`.read` applies. A reference point would
        decide each face's outward direction by which side of the point it lies
        on, which is outward everywhere only on a surface convex about the point.
        """
        return {
            "Index": self.radiated,
            "Attributes": sorted(self.absorbing),
            "Type": "Power",
            "TwoSided": True,
        }

    @property
    def excitations(self) -> tuple[int, ...]:
        """The port numbers driven, which are also their excitation indices.

        One excitation per driven port, because Palace forms a scattering matrix
        only where every excitation drives exactly one port and produces no
        table at all otherwise. And the index is the port's own number rather
        than a count over the driven ports: past one excitation Palace requires
        every index to equal the index of the port it drives, and stops before
        the first solve where one does not.
        """
        return tuple(sorted(port.index for port in self.ports if port.excited))

    def to_dict(self) -> dict[str, Any]:
        driven = self.excitations
        ordered = sorted(self.ports, key=lambda port: port.index)
        boundaries: dict[str, Any] = {}
        for key, kind in (("WavePort", WavePort), ("LumpedPort", LumpedPort)):
            written = [
                port.to_dict(port.index if port.index in driven else 0)
                for port in ordered
                if isinstance(port, kind)
            ]
            if written:
                boundaries[key] = written
        if self.perfect_conductor:
            boundaries["PEC"] = {"Attributes": sorted(self.perfect_conductor)}
        if self.magnetic_conductor:
            boundaries["PMC"] = {"Attributes": sorted(self.magnetic_conductor)}
        if self.conducting:
            boundaries["Conductivity"] = [metal.to_dict() for metal in self.conducting]
        if self.absorbing:
            boundaries["Absorbing"] = {
                "Attributes": sorted(self.absorbing),
                "Order": ABSORBING_ORDER,
            }
        # What leaves through each port's face, beside what the port reports as
        # its mode. A port absorbs whatever field reaches its face, and the
        # difference is the power it took that the matrix does not show. What
        # leaves through the open surface is measured beside them.
        fluxes = [port.to_flux() for port in ordered]
        if self.absorbing:
            fluxes.append(self.to_radiated_flux())
        boundaries["Postprocessing"] = {"SurfaceFlux": fluxes}
        materials = [material.to_dict() for material in self.materials]
        if self.ranks_can_disagree:
            for filled in materials:
                # A lossy material keeps its own loss: Palace stops on one carrying both.
                if "LossTan" not in filled and "Conductivity" not in filled:
                    filled["LossTan"] = RANK_LOSS_TANGENT
        return {
            "Problem": {"Type": "Driven", "Verbose": VERBOSE, "Output": self.output},
            "Model": {"Mesh": self.mesh, "L0": METRES_PER_UNIT},
            "Domains": {"Materials": materials},
            "Boundaries": boundaries,
            "Solver": {
                "Order": self.order,
                "Driven": {
                    "Samples": [self.sweep.to_dict()],
                    **(self.sweep.adaptive.to_dict() if self.sweep.adaptive else {}),
                },
                "Linear": {
                    "Type": LINEAR_SOLVER,
                    "KSPType": KRYLOV_SOLVER,
                    "Tol": LINEAR_TOLERANCE,
                    "MaxIts": OPEN_LINEAR_ITERATIONS if self.absorbing else LINEAR_ITERATIONS,
                    "EstimatorMaxIts": ESTIMATOR_ITERATIONS,
                },
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def port_modes(
        self, port: int, frequency: float, modes: int, output: str, meeting: frozenset[int]
    ) -> dict[str, Any]:
        """The same model asked for the modes of one port's face at ``frequency``.

        Palace's boundary mode problem cuts the port's faces out of the mesh
        and solves ``modes`` of them, those of the largest propagation constant,
        with the boundaries the driven run carries. Every other port is written
        as a perfect conductor, which is what the port's own mode solve takes it
        for (``palace/models/waveportoperator.cpp:1555-1602``); left out, its
        faces would be named by no condition, and Palace takes such a face for
        a perfect magnetic conductor with a warning.

        :param meeting: the attributes whose faces meet the port's along a
            curve. Metal of finite conductivity on any other attribute stands
            nowhere on the face the problem is solved on, and Palace warns of
            that condition on an attribute the face does not carry. So it is
            written a perfect conductor instead, as the other ports are: an
            outer face named by no condition is the magnetic wall above, and
            Palace passes over a perfect conductor on a face that shares no
            edge with the port's (``palace/models/boundarymodeoperator.cpp:108-113``).

        The power measured through the ports is left out too. It is a question
        about the driven field, and Palace builds what a configuration asks it
        to measure for every problem type (``palace/models/postoperator.cpp:130``).
        So is the line a port's voltage is read along, which asks the driven
        run's mode for its impedance and its sign; the mode problem asks which
        modes the face carries.
        """
        written = self.to_dict()
        written["Problem"] = {"Type": "BoundaryMode", "Verbose": VERBOSE, "Output": output}
        boundaries = written["Boundaries"]
        del boundaries["Postprocessing"]
        (mine,) = [entry for entry in boundaries["WavePort"] if entry["Index"] == port]
        mine.pop("VoltagePath", None)
        others = {
            attribute
            for entry in boundaries["WavePort"]
            if entry["Index"] != port
            for attribute in entry["Attributes"]
        }
        boundaries["WavePort"] = [mine]
        metals = boundaries.pop("Conductivity", [])
        kept = [metal for metal in metals if set(metal["Attributes"]) & meeting]
        if kept:
            boundaries["Conductivity"] = kept
        apart = {
            attribute
            for metal in metals
            if not set(metal["Attributes"]) & meeting
            for attribute in metal["Attributes"]
        }
        if others | apart:
            walls = set(boundaries.get("PEC", {}).get("Attributes", ())) | others | apart
            boundaries["PEC"] = {"Attributes": sorted(walls)}
        written["Solver"]["BoundaryMode"] = {
            "Freq": _ghz(frequency),
            "N": modes,
            "Attributes": mine["Attributes"],
        }
        return written
