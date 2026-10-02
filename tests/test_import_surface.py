"""Every Python module of the package imports.

An optional third-party backend may be absent from the environment; every
other import failure is this package's own. The compiled Union-Find
libraries beside their C sources are loaded through ctypes, never
imported: Python lists a bare .so as an extension module, and importing
the sanitized build starts the sanitizer runtime, which ends the process.
"""

import importlib
import importlib.util
import os
import pathlib
import pkgutil
import subprocess
import sys

import pytest

import decsim

# Each name the root exports, and the module a run file reads it from.
ROOT_EXPORTS = (
    ("Experiment", "decsim.experiments.experiment"),
    ("Point", "decsim.experiments.experiment"),
    ("grid", "decsim.experiments.experiment"),
    ("CollectionSettings", "decsim.experiments.collection"),
    ("MachineSettings", "decsim.settings"),
    ("Machine", "decsim.machine"),
)


def import_failure(module_name):
    """The module's failure, or None when it imports or its extra is absent.

    An absent extra is a third-party module that is not installed; a
    decsim module that cannot be found, or a name one cannot import, is
    this package's own failure.
    """
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        missing = error.name or ""
        missing_parts = missing.split(".")
        if missing_parts[0] != "decsim":
            return None
        return (module_name, repr(error))
    except Exception as error:
        return (module_name, repr(error))
    return None


def is_python_module(module_name):
    """Whether the module comes from a Python source file."""
    specification = importlib.util.find_spec(module_name)
    origin = specification.origin
    return origin.endswith(".py")


def import_failures() -> list:
    """(module, error) for every decsim source module that fails to import."""
    failures = []
    walked = pkgutil.walk_packages(decsim.__path__, prefix="decsim.")
    for info in walked:
        if not is_python_module(info.name):
            continue
        failure = import_failure(info.name)
        if failure is not None:
            failures.append(failure)
    return failures


def test_every_module_imports():
    failures = import_failures()
    assert failures == []


@pytest.mark.parametrize("name, module_name", ROOT_EXPORTS)
def test_the_root_exports_the_object_its_module_defines(name, module_name):
    module = importlib.import_module(module_name)
    exported = getattr(decsim, name)

    assert exported is getattr(module, name)


def test_a_name_the_root_does_not_export_is_refused():
    unexported = "Engine"

    with pytest.raises(AttributeError, match="no attribute 'Engine'"):
        getattr(decsim, unexported)


def test_importing_the_root_leaves_the_machine_unimported():
    """The command line starts light: an export loads on first use."""
    package_file = pathlib.Path(decsim.__file__)
    tree = package_file.parent.parent
    environment = dict(os.environ, PYTHONPATH=str(tree))
    code = "import sys, decsim; print('decsim.machine' in sys.modules)"
    command = [sys.executable, "-c", code]

    completed = subprocess.run(
        command, capture_output=True, text=True, check=True, env=environment
    )

    assert completed.stdout.strip() == "False"
