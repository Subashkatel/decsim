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
             "syndrome_ingress.py", "syndrome_buffer.py", "decoder_manager.py",
             "decoder_memory.py", "decoder_memory_transfer.py", "decoder_engine.py",
             "decoders.py", "schedulers.py", "switching.py", "window_manager.py",
             "schemes.py", "window_interactions.py", "dynamic_windows.py",
             "speculative_recovery.py", "links.py", "link_profiles.py",
             "pauli_frame.py", "metrics.py", "views.py"]
    assert [name for name in moved if (root / name).exists()] == []
    assert not (root / "program").exists()
    assert not (root / "orchestrator").exists()
    assert not (root / "stimcircuits").exists()
    for old_package in ("adapters", "soft_output", "mwpm_decoder", "union_find_decoder",
                        "tesseract_decoder", "relay_bp_decoder",
                        "belief_matching_decoder", "bposd_decoder"):
        assert not (root / old_package).exists()
