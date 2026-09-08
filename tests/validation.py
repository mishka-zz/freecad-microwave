# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Comparing an answer against something that is not exact, and saying how well.

The other half of :mod:`tests.convergence`. That one asks whether the equations
are being solved right, which is a question about arithmetic and needs no
experiment; this one asks whether they are the right equations, which is a
question about the world and cannot be answered without one.

The distinction decides what a disagreement *means*. Against a closed form, a
difference is the solver's error and refining the mesh is the response. Against a
fitted formula or a fabricated board, a difference is the solver's error and the
reference's error added together, and refining the mesh past the reference's own
accuracy buys nothing that can be seen.

The procedure is ASME V&V 20-2009, *Standard for Verification and Validation in
Computational Fluid Dynamics and Heat Transfer*. The comparison error ``E`` is
the simulation minus the reference [eq. (1-5-1)] and carries every error in
both [eq. (1-5-6)]; the validation uncertainty ``u_val`` is what the discretisation, the
inputs and the reference are each worth, added in quadrature [eq. (1-5-10)].

**Quadrature is conditional on those three being independent**, which the
standard states as the condition on that equation rather than as an aside. A
quantity shared between the simulation and the reference makes them correlated,
and correlated terms combine by the longer forms of subsection 5-3 instead - so a
reference evaluated at the same declared numbers the solver was given is only
safe here while the term for those numbers is zero.

**Only an unknown error is budgeted.** The approach the standard outlines
assumes that every error whose sign and magnitude is known has already been
removed by correction, leaving ``u`` to characterise the range containing what
remains [subsection 1-3]. An offset with a size does not become
an uncertainty by being placed in one.

**Not every error the mesh causes is a discretisation term.** An approximation
whose error does not vanish as the cell shrinks - how far away a boundary is
placed, what a curved solid is inscribed to - is what the standard calls a
nonordered approximation, and it classes those as modelling errors rather than
discretisation ones. They do not belong in ``numerical``, and what settles
whether one is small enough is a sensitivity test rather than a refinement study
[subsection 2-4, and Nonmandatory Appendix C, *Far-Field Boundary Errors*].

**What a pass means is narrower than it looks**, and it is the thing most often
got wrong. ``|E| <= u_val`` does not say the model is right. It says the model's
error is *below the resolution of the comparison* - that the instrument being
used to look for a modelling error is not sharp enough to see one. The way to
make a model look good by this test is to have a bad reference, which is why the
size of ``u_val`` is reported here beside every verdict rather than folded away
into a boolean.

**The terms are intervals rather than standard deviations.** ``u_val`` is a
standard uncertainty - the standard deviation of the combined errors - and
turning one into an interval of stated coverage means choosing a coverage factor
for an assumed distribution of those errors [subsection 6-3.2, which lists
factors for several distributions and settles on none]. Nothing here chooses
one, because what goes into ``u_val`` is already an interval: a refinement
study's uncertainty is built to contain the exact answer 95 times in a hundred,
and a reference's quoted accuracy is a bound rather than a spread. The standard
allows exactly that - base everything on the expanded level and no assumption
about a distribution is required [para. 2-4.1] - so the comparison is
made at face value.

**A comparison against a formula is not a validation in the standard's own
sense**: "there can be no validation without experimental data with which to
compare the result of the simulation" [subsection 1-1]. What such a comparison borrows here is
the composition, which prices any comparison whose reference is inexact, and
the wider sense is the one this word carries in this tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Comparison:
    """One quantity, computed and referenced, with the uncertainty of each.

    Every field is in the same units, and they may be absolute or relative as
    long as they are all the one thing - nothing here divides by a value, so a
    comparison stated in ohms and one stated in shares behave alike.
    """

    #: What the solver answered, and what it is being compared against.
    simulated: float
    reference: float
    #: What the discretisation is worth, from a refinement study. This is the
    #: only one of the three that is *ours*, and the only one a mesh can change.
    #: An error the mesh causes that refinement does not remove is not this
    #: term. One correlated with it is, and enters by simple addition rather
    #: than by a slot of its own in the quadrature - the standard's case is an
    #: iteration error, which moves the discretisation error it would otherwise
    #: be squared alongside [eq. (2-4-2)].
    numerical: float = 0.0
    #: What the uncertainty in the inputs is worth, propagated to the answer by
    #: sensitivity coefficients [eq. (3-2-1)] or by sampling [subsection 3-3].
    #: Zero where every input is treated as exact, which is what a formula
    #: evaluated at the same declared numbers the solver was given amounts to.
    #: The standard asks a thorough study to consider this term and then allows
    #: every parameter to be hard-wired instead, which is the limit it calls a
    #: strong model [Nonmandatory Appendix C, *Parametric and Model Form
    #: Uncertainties*]. The zero is
    #: not free: an error kept out of this term has not gone anywhere, it lands
    #: in the modelling error the comparison exists to estimate [subsection 1-5].
    inputs: float = 0.0
    #: What the reference is worth: a fit's quoted accuracy, or an experiment's
    #: measurement uncertainty. Where the reference is itself a model, its own
    #: numerical and input terms enter here [subsection 5-4].
    reference_uncertainty: float = 0.0

    @property
    def error(self) -> float:
        """The comparison error, simulation minus reference. It carries a sign,
        because which way a model is wrong is usually the interesting part."""
        return self.simulated - self.reference

    @property
    def validation_uncertainty(self) -> float:
        """How sharply the comparison can see a modelling error at all."""
        return math.sqrt(self.numerical**2 + self.inputs**2 + self.reference_uncertainty**2)

    @property
    def resolved(self) -> bool:
        """Whether a modelling error is visible above the comparison's own noise.

        The two readings the standard gives are not each other's complement
        [subsection 6-2]. Where ``|E|`` is *much greater* than the validation
        uncertainty, the modelling error is approximately :attr:`error`. Where
        ``|E|`` is at or below it, what has been established is that the
        modelling error is of the same order as, or smaller than, the errors
        that were not the model's - which is weaker than a bound, and weaker
        than :attr:`validation_uncertainty` itself.

        This splits at one validation uncertainty, so False is exactly the
        second reading while True is looser than the first, and the width
        reported beside the verdict is what says how much looser.

        True is not a failure and False is not a pass - they are the two things
        the comparison can say.
        """
        return abs(self.error) > self.validation_uncertainty

    @property
    def dominated_by(self) -> str:
        """Which of the three terms sets the resolution, which says what to fix.

        A comparison limited by its reference cannot be improved by meshing, and
        one limited by its mesh says nothing about the reference yet.
        """
        terms = {
            "the discretisation": self.numerical,
            "the inputs": self.inputs,
            "the reference": self.reference_uncertainty,
        }
        return max(terms, key=lambda name: terms[name])
