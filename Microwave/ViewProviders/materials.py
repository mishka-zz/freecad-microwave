# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a material makes the solids bound to it look like.

The colour belongs to the material, and it reaches a solid only through a
binding, so both providers here push the same way: material to binding to solid.
A solid bound twice can still only be one colour, and the last binding in
document order is the one that shows.
"""

from . import HasDisplayMode, add_display_mode


class EMMaterialViewProvider(HasDisplayMode):
    ICON = "Material.svg"

    def updateData(self, obj, prop):
        if prop != "Color":
            return
        # This walks ``obj.Document`` rather than the active document. A
        # material recoloured in a document the user is not looking at must
        # repaint that document's solids rather than whichever document is in
        # front.
        for other in obj.Document.Objects:
            if getattr(other, "Material", None) is obj:
                provider = getattr(other.ViewObject, "Proxy", None)
                if isinstance(provider, EMMaterialBindingViewProvider):
                    provider.update_colors(other.ViewObject)


class EMMaterialBindingViewProvider(HasDisplayMode):
    ICON = "MaterialBinding.svg"

    def attach(self, vobj):
        self.Object = vobj.Object
        add_display_mode(vobj, self.DISPLAY_MODE)
        # Here rather than in __init__ because a restored provider gets attach
        # and not __init__.
        self.update_colors(vobj)

    def onChanged(self, vobj, prop):
        if prop in ["References", "Material"]:
            self.update_colors(vobj)

    def updateData(self, obj, prop):
        if prop in ["References", "Material"]:
            self.update_colors(obj.ViewObject)

    def update_colors(self, vobj):
        obj = vobj.Object
        material = getattr(obj, "Material", None)
        references = getattr(obj, "References", None)
        if material is None or not references:
            return

        # This takes RGB and drops the alpha. FreeCAD's colour tuple carries a
        # fourth component, but a solid's transparency is a property of its own
        # that the user sets the ordinary way. Honouring both would give one
        # appearance two controls, and whichever ran last would win without
        # reporting it. A catalog has no alpha to honour either - its ``color``
        # is ``#RRGGBB``. Assigning three components leaves the solid's alpha at
        # 1.0, which is what this intends.
        color = tuple(getattr(material, "Color", ())[:3])
        if len(color) < 3:
            return

        for ref, _sub in references:
            view = getattr(ref, "ViewObject", None)
            if view is not None:
                view.ShapeColor = color
