"""Every Python module of the package imports.

An optional third-party backend may be absent from the environment; every
other import failure is this package's own. The compiled Union-Find
libraries beside their C sources are loaded through ctypes, never
imported: Python lists a bare .so as an extension module, and importing
the sanitized build starts the sanitizer runtime, which ends the process.
"""

import importlib
import importlib.util
import pkgutil

import decsim


def import_failure(module_name):
    """The module's failure, or None when it imports or its extra is absent."""
    try:
        importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None
    except Exception as error:
        return (module_name, repr(error))
    return None


def is_python_module(module_name):
    """Whether the module comes from a Python source file."""
    specification = importlib.util.find_spec(module_name)
    origin = specification.origin
    return origin.endswith(".py")


def test_every_module_imports():
    failures = []
    walked = pkgutil.walk_packages(decsim.__path__, prefix="decsim.")
    for info in walked:
        if not is_python_module(info.name):
            continue
        failure = import_failure(info.name)
        if failure is not None:
            failures.append(failure)
    assert failures == []
