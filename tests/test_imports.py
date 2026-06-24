import importlib


def test_core_modules_import_from_package_context():
    modules = [
        "decsim.cluster",
        "decsim.config",
        "decsim.controllers",
        "decsim.codes",
        "decsim.decoders",
        "decsim.devices",
        "decsim.engine",
        "decsim.message",
        "decsim.metrics",
        "decsim.orchestrators",
        "decsim.planner",
        "decsim.protocols",
        "decsim.schedulers",
        "decsim.schemes",
    ]

    for module in modules:
        importlib.import_module(module)
