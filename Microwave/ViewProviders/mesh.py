# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import HasDisplayMode


class EMMeshPolicyViewProvider(HasDisplayMode):
    ICON = "MeshPolicy.svg"


class EMMeshRegionViewProvider(HasDisplayMode):
    ICON = "MeshRegion.svg"


class EMYeeGridViewProvider(HasDisplayMode):
    ICON = "YeeGrid.svg"


class EMGmshMeshViewProvider(HasDisplayMode):
    ICON = "GmshMesh.svg"
