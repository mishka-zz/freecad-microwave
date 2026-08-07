# Vendored third-party packages

This directory is **not a Python package**. It is a directory placed on
`sys.path` by `Microwave/__init__.py`, so what sits here is importable under
its own name: `import skrf`.

It is **appended**, never inserted. Anything the interpreter already provides
wins, and a user who installed their own copy keeps it. We are the fallback,
not the override.

## scikit-rf 1.13.0

| | |
|---|---|
| Upstream | https://github.com/scikit-rf/scikit-rf |
| Version | 1.13.0 (sdist `scikit_rf-1.13.0.tar.gz` from PyPI) |
| Licence | BSD-3-Clause — `LICENSE.skrf.txt` |
| Modified | **No.** Byte-for-byte upstream, less its eight `tests/` directories |

The eight `tests/` directories --- `skrf/tests` and one each under
`calibration`, `io`, `media`, `vi`, and `vi/vna/{hp,keysight,rohde_schwarz}` ---
are dropped because the library never imports them. Nothing else is touched, so
re-vendoring a later release is a copy rather than a merge.

### Why vendored, and why this version

The workbench needs scikit-rf inside *FreeCAD's* interpreter, and FreeCAD's
interpreter is not ours to install into. Two routes were measured and rejected
on 2026-07-28:

- **FreeCAD's Addon Manager.** `ALLOWED_PYTHON_PACKAGES.txt` is an allow-list of
  93 entries carrying `scikit-image`, `scikit-learn` and `scikit-sparse` but not
  `scikit-rf`, so the dependency would be stripped and the user told to install
  it by hand. Worse, `AddonManager/Addon.py:415` keeps only `dep.package` and
  discards the version constraint — so even once allow-listed, **it cannot be
  pinned**, and would install the version that does not work. See below.
- **pip into FreeCAD's bundle.** Works, needs no sudo, but it is per-machine and
  does not survive reinstalling FreeCAD.

**1.13.0 rather than the newest release.** scikit-rf 2.0.x fails to import on
numpy 1.x, and FreeCAD 1.1.1 ships numpy 1.26.4:

    AttributeError: module 'numpy' has no attribute 'typing'

`skrf/calibration/calibration.py` evaluates `np.typing.NDArray[complex]` at
module scope. `numpy.typing` is a submodule, not an attribute — numpy 2.x
resolves it through a lazy `__getattr__`, numpy 1.x does not. The line is
present in 1.13 too, so 1.x only ever worked *by accident*: 1.13 imported
`from scipy.optimize import ...` at module scope, and importing any scipy
submodule pulls `numpy.typing` in as a side effect. 2.0.0's headline change
moved those imports into functions to cut import time, leaving a bare
`import scipy` — which does not. A latent bug, unmasked.

Measured under FreeCAD 1.1.1's own interpreter (Python 3.11.14, numpy 1.26.4):

| 1.9.0 | 1.10.0 | 1.11.0 | 1.12.0 | 1.13.0 | 2.0.0 | 2.0.1 |
|---|---|---|---|---|---|---|
| ok | ok | ok | ok | **ok** | AttributeError | AttributeError |

1.13.0 is the newest that works, and it also works on numpy 2.4.4 under Python
3.13 — so this one copy spans the official FreeCAD build and any package-manager
build. That is the property to preserve when bumping: choose by *support range*,
not by recency.

Upstream fix: https://github.com/scikit-rf/scikit-rf/pull/1410 (merged
2026-07-22, unreleased as of 2026-07-28). Once a release carrying it exists,
re-run the matrix above against the oldest numpy the workbench supports before
bumping.

### The cost of this version

1.13.0 imports scipy and pandas eagerly, and matplotlib too where it is
installed — about **1 s**, against
0.04 s for 2.0.1. That is why `Microwave/__init__.py` only extends `sys.path`
and never imports skrf: the cost is paid on first use of the result layer, not
on every FreeCAD start.
