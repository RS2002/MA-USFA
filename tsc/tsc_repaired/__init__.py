"""Manifest-driven server package for the repaired Manhattan/SUMO method.

This package is intentionally a preparation layer.  It contains the
protocol and strict-entry interfaces needed by the server runner, but it does
not start SUMO or train a model until a filled manifest reaches the explicit
pilot/formal state.
"""

__version__ = "0.2.0-server-adapter"

# Keep package import dependency-free.  ``numpy`` and SUMO are server-side
# dependencies; importing ``tsc_repaired.config`` or running the manifest
# preflight on the laptop must not require either package.

