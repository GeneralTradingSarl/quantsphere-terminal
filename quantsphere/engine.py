"""Engine dispatch: prefer the compiled C++ core, fall back to NumPy.

Both expose the exact same API:
    simulate_paths, mc_price, pde_price, kalman, bs_price, implied_vol,
    version, hardware_threads
"""

from __future__ import annotations

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

try:
    import qsengine as core  # compiled .pyd placed next to this file

    ENGINE_NATIVE = True
    ENGINE_LABEL = f"C++ NATIVE v{core.version()} · {core.hardware_threads()} threads"
except ImportError:  # pragma: no cover - environment dependent
    from quantsphere import _fallback as core

    ENGINE_NATIVE = False
    ENGINE_LABEL = f"NUMPY FALLBACK v{core.version()}"

__all__ = ["core", "ENGINE_NATIVE", "ENGINE_LABEL"]
