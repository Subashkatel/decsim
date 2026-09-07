"""Every module imports by its component path, and the flat layout is gone.

Rule 5: a rename is a rename, so no module of the pre-rewrite flat
layout may sit beside the component package that replaced it. An
optional third-party backend may be absent from the environment; every
other import failure is this package's own.
"""

import importlib
import pkgutil
from pathlib import Path

import decsim

# the modules the rewrite moved into component packages
MOVED_MODULES = (
    "qpu.py",
    "devices.py",
    "codes.py",
    "layouts.py",
    "planner.py",
    "orchestrators.py",
    "execution_runtime.py",
    "rounds.py",
    "factories.py",
    "controller.py",
    "policies.py",
    "syndrome_packing.py",
    "syndrome_buffer.py",
    "decoder_manager.py",
    "decoder_memory.py",
    "decoder_memory_transfer.py",
    "decoder_engine.py",
    "decoders.py",
    "schedulers.py",
    "switching.py",
    "window_manager.py",
    "schemes.py",
    "window_interactions.py",
    "dynamic_windows.py",
    "speculative_recovery.py",
    "links.py",
    "link_profiles.py",
    "pauli_frame.py",
    "metrics.py",
    "views.py",
)
# the packages the rewrite renamed or dissolved
MOVED_PACKAGES = (
    "program",
    "orchestrator",
    "stimcircuits",
    "adapters",
    "soft_output",
    "mwpm_decoder",
    "union_find_decoder",
    "tesseract_decoder",
    "relay_bp_decoder",
    "belief_matching_decoder",
    "bposd_decoder",
)


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


def test_no_module_of_the_old_flat_layout_remains():
    package_file = Path(decsim.__file__)
    root = package_file.parent
    left_behind = []
    for name in MOVED_MODULES:
        module_path = root / name
        if module_path.exists():
            left_behind.append(name)
    for name in MOVED_PACKAGES:
        package_path = root / name
        if package_path.exists():
            left_behind.append(name)
    assert left_behind == []
