"""Every module of the package imports by its component path, and no module
of the old flat layout is left behind."""

import importlib
import pkgutil
from pathlib import Path

import decsim


def test_every_module_imports():
    failures = []
    for info in pkgutil.walk_packages(decsim.__path__, prefix="decsim."):
        try:
            importlib.import_module(info.name)
        except Exception as error:      # optional third-party backends may be absent
            if type(error).__name__ not in ("ImportError", "ModuleNotFoundError"):
                failures.append((info.name, repr(error)))
    assert failures == []


def test_no_module_of_the_old_flat_layout_remains():
    root = Path(decsim.__file__).parent
    moved = ["qpu.py", "devices.py", "codes.py", "layouts.py", "planner.py",
             "orchestrators.py", "execution_runtime.py", "rounds.py",
             "factories.py", "controller.py", "policies.py",
             "syndrome_ingress.py", "syndrome_buffer.py"]
    assert [name for name in moved if (root / name).exists()] == []
    assert not (root / "frontends").exists()
    assert not (root / "stimcircuits").exists()
