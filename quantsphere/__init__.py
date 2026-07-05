"""QuantSphere Terminal — institutional quantitative pipeline.

C++20 native engine (qsengine, pybind11) with a bit-compatible-API NumPy
fallback, so the platform runs everywhere and runs *fast* where it can.
"""

from quantsphere.engine import core, ENGINE_NATIVE, ENGINE_LABEL

__version__ = "1.0.0"
__all__ = ["core", "ENGINE_NATIVE", "ENGINE_LABEL", "__version__"]
