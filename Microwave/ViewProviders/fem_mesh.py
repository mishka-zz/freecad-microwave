# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""How a mesh FreeCAD draws looks in the tree.

FEM's own view provider draws the mesh. This one gives it this workbench's
icon and the out-of-date overlay, and defines no display mode, leaving FEM's
to draw it.
"""

from . import Provider, out_of_date


class EMFemMeshViewProvider(Provider):
    ICON = "MeshPalace.svg"

    def attach(self, vobj):
        self.Object = vobj.Object

    def getOverlayIcons(self):
        return out_of_date(getattr(self, "Object", None))


def attach(mesh):
    """Give ``mesh`` this view provider, where its view takes a proxy and holds
    none of ours yet, and return whether it was given one. A plain
    ``Fem::FemMeshObject`` has a view that takes no proxy, and keeps FEM's icon
    until the next mesh replaces it."""
    view = getattr(mesh, "ViewObject", None)
    if view is None or not hasattr(view, "Proxy"):
        return False
    if isinstance(view.Proxy, EMFemMeshViewProvider):
        return False
    provider = EMFemMeshViewProvider(view)
    if getattr(provider, "Object", None) is None:
        provider.attach(view)
    view.signalChangeIcon()
    return True
