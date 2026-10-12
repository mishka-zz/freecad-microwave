# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import HasDisplayMode


class EMMeshPolicyViewProvider(HasDisplayMode):
    ICON = "MeshPolicy.svg"


class EMMeshRegionViewProvider(HasDisplayMode):
    ICON = "MeshRegion.svg"


class _RecipeViewProvider(HasDisplayMode):
    """A backend's mesh settings, holding in the tree the refinement regions
    that amend them.

    A region in the study's own group links the settings it amends in
    ``MeshSettings``, and those settings are where the tree shows it. The study
    leaves out of its own children what a member holds, so the region appears
    once - see ``ViewProviders/analysis.py``. A region the user filed in a
    group inside the study stays under that group, which holds it in the tree.
    """

    def claimChildren(self):
        from ..Objects.analysis import analysis_of

        recipe = getattr(self, "Object", None)
        study = analysis_of(recipe) if recipe is not None else None
        if study is None:
            return []
        return [
            member
            for member in getattr(study, "Group", ()) or ()
            if getattr(member, "MeshSettings", None) is recipe
        ]


class EMYeeGridViewProvider(_RecipeViewProvider):
    ICON = "MeshSettingsOpenEMS.svg"


class EMGmshMeshViewProvider(_RecipeViewProvider):
    ICON = "MeshSettingsPalace.svg"
