"""Every module of the package imports.

An optional third-party backend may be absent from the environment; every
other import failure is this package's own.
"""

import importlib
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


def test_every_module_imports():
    failures = []
    walked = pkgutil.walk_packages(decsim.__path__, prefix="decsim.")
    for info in walked:
        failure = import_failure(info.name)
        if failure is not None:
            failures.append(failure)
    assert failures == []
