import importlib


def test_core_modules_import_from_package_context():
    modules = [
        "decsim.policies",
        "decsim.codes",
        "decsim.decoders",
        "decsim.engine",
        "decsim.factories",
        "decsim.orchestrators",
        "decsim.layouts",
        "decsim.metrics",
        "decsim.message",
        "decsim.pauli_frame",
        "decsim.links",
        "decsim.controllers",
        "decsim.payload_store",
        "decsim.decoder_manager",
        "decsim.protocols",
        "decsim.chip",
        "decsim.planner",
        "decsim.schedulers",
        "decsim.schemes",
        "decsim.devices",
        "decsim.run_spec",
        "decsim.switching",
        "decsim.dynamic_windows",
        "decsim.config",
        "decsim.views",
        "decsim.window_manager",
    ]

    for module in modules:
        importlib.import_module(module)
