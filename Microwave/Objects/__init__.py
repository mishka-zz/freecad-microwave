# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The solver-neutral document objects, and how they are made.

Every class here is prefixed ``EM`` - not ``EMS``, which reads as *openEMS*,
one backend of the three this architecture is built for.

The proxy class name identifies an object in a restored document, so renaming
one of these classes invalidates every saved file that holds it.
"""

from . import port_setup
from ._vp_hook import register_view_provider_injector
from .analysis import (
    EMAnalysis,
    NoAnalysis,
    analyses,
    analysis_of,
    createEMAnalysis,
    find_analysis,
    members,
    solver_of,
)
from .materials import (
    EMMaterial,
    EMMaterialBinding,
    apply_entry,
    create_from_entry,
    createEMMaterial,
    createEMMaterialBinding,
    fill_binding,
    label_for,
    sourced_from,
)
from .mesh import (
    EMMeshPolicy,
    EMMeshRegion,
    createEMMeshPolicy,
    createEMMeshRegion,
    references_from,
)
from .ports import (
    EMPortBase,
    EMPortCoaxial,
    EMPortLumped,
    EMPortMicrostrip,
    EMPortRectWaveguide,
    createEMPortCoaxial,
    createEMPortLumped,
    createEMPortMicrostrip,
    createEMPortRectWaveguide,
    next_port_number,
)
from .preview import EMMeshPreview, createEMMeshPreview
from .results import EMSParameters, createEMSParameters
from .solver import EMSolverOpenEMS, createEMSolverOpenEMS
