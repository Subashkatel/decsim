"""The compiled libraries this process's loaders name, for the manifest.

A library is built from tracked C source and is not tracked itself, so
a run's commit does not name the bytes it loaded, and a loader may take
its file from outside the package. A loader registers, when it is
imported, the function it calls itself to name its file; a worker
process imports the same loader under the same environment, so it loads
the file the function names here. The run folder hashes each named file
that exists. This module sits at the lowest level so a loader at any
level can reach it.
"""

import pathlib
from collections.abc import Callable

_NAMERS: list = []


def register(library_path: Callable[[], pathlib.Path]) -> None:
    """Name a loader's library by the function the loader itself calls."""
    _NAMERS.append(library_path)


def paths() -> list:
    """Every named library file that exists, absolute, once each, sorted."""
    found = set()
    for library_path in _NAMERS:
        path = library_path()
        if path.exists():
            absolute = path.resolve()
            found.add(absolute)
    return sorted(found)
