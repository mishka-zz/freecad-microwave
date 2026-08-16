# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Where catalogs are looked for.

In this order: the ones that ship with the workbench, the ones the user
dropped in their FreeCAD data directory, and anything named in the environment
or the parameter store.

``search_paths`` is a pure function of its arguments so the ordering - the part
worth testing - can be tested without FreeCAD. The two functions that do touch
FreeCAD import it inside their bodies and return ``None`` when it is not there,
the same crossing-point idiom as ``Objects/port_setup.from_shape``.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Mapping, Sequence

#: Same shape as ``$MICROWAVE_OPENEMS_PYTHON``: an escape hatch that needs no
#: GUI, which is what makes headless tests and CI able to drive this at all.
ENV_VAR = "MICROWAVE_MATERIAL_PATH"

PARAMETER_GROUP = "User parameter:BaseApp/Preferences/Mod/Microwave"
PARAMETER_KEY = "MaterialCatalogPaths"

#: Ships with the workbench. First in the order, so a catalog dropped in a
#: downloads folder can never quietly take over the name 'generic'.
BUNDLED = pathlib.Path(__file__).resolve().parent.parent / "data" / "materials"

#: Under FreeCAD's own user data directory, beside Mod/ and Macro/, because that
#: is where a FreeCAD user already looks for things they installed themselves.
USER_SUBDIRECTORY = ("Microwave", "materials")


def search_paths(
    *,
    bundled: pathlib.Path | None = BUNDLED,
    user_dir: pathlib.Path | None = None,
    configured: Sequence[pathlib.Path] = (),
    env: Mapping[str, str] | None = None,
) -> tuple[pathlib.Path, ...]:
    """Every place to look, in precedence order, without touching the disk."""
    paths: list[pathlib.Path] = []
    if bundled is not None:
        paths.append(pathlib.Path(bundled))
    if user_dir is not None:
        paths.append(pathlib.Path(user_dir))
    paths.extend(pathlib.Path(path) for path in configured)
    raw = (env if env is not None else os.environ).get(ENV_VAR, "")
    paths.extend(pathlib.Path(part) for part in raw.split(os.pathsep) if part)

    # First mention wins, because the order is precedence. Undeduplicated, a
    # path named twice - $MICROWAVE_MATERIAL_PATH pointing at the user
    # directory is the easy way in - loads its catalogs twice and reports the
    # second as colliding with the first, a failure whose two halves are the
    # same file.
    #
    # Lexical, so this still touches no disk: two spellings of one directory
    # are a different problem and resolving them would need one.
    seen = set()
    unique = []
    for path in paths:
        key = os.path.normpath(str(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return tuple(unique)


def freecad_user_dir() -> pathlib.Path | None:
    """``<FreeCAD user data>/Microwave/materials``, or ``None`` outside FreeCAD.

    Version-scoped, as everything under that directory is, so catalogs put there
    are invisible after an upgrade exactly as ``Mod/`` is. Consistency with the
    rest of FreeCAD counts for more than an unscoped directory no one would think
    to look in, but it is why the empty-library message prints the absolute path
    it searched.
    """
    try:
        import FreeCAD
    except ImportError:
        return None
    directory = getattr(FreeCAD, "getUserAppDataDir", None)
    if directory is None:
        return None
    return pathlib.Path(directory()).joinpath(*USER_SUBDIRECTORY)


def configured_paths() -> tuple[pathlib.Path, ...]:
    """Extra folders or files from the parameter store.

    A machine setting, so it lives in FreeCAD's parameters and never in the
    document. A path to a user's downloads folder inside a ``.FCStd`` is worse
    than useless on the machine it is opened on: it is a dangling reference that
    changes what the materials in that document mean.
    """
    try:
        import FreeCAD
    except ImportError:
        return ()
    try:
        raw = FreeCAD.ParamGet(PARAMETER_GROUP).GetString(PARAMETER_KEY, "")
    except Exception:  # a parameter store that will not answer
        return ()
    return tuple(pathlib.Path(part) for part in str(raw).split(os.pathsep) if part)


def installed():
    """The library this machine has, loaded fresh.

    Paths the user chose are passed as ``required`` so a typo in one is
    reported. The bundled and user directories are not: the first is always
    present and the second is absent until a catalog is put in it.
    """
    from .library import load_library

    configured = configured_paths()
    return load_library(
        search_paths(user_dir=freecad_user_dir(), configured=configured),
        bundled=BUNDLED,
        required=configured + _from_environment(),
    )


def _from_environment() -> tuple[pathlib.Path, ...]:
    raw = os.environ.get(ENV_VAR, "")
    return tuple(pathlib.Path(part) for part in raw.split(os.pathsep) if part)
