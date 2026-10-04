"""decsim, a discrete-event simulator of the QEC reaction path.

A readout leaves the QPU, crosses the controller, the buffers and the
decoder, and comes back as an instruction; every hop is priced. The
root is decsim.machine.Machine, built from a MachineSettings; the
experiments layer over it is decsim.experiments, whose command is
`decsim`, and decsim.collect, which runs the shots of a sweep's tasks.

A run file reaches the names it builds an experiment from at the root:
decsim.Experiment, Point, grid, CollectionSettings, MachineSettings and
Machine. Each loads its module on first use (PEP 562 module
__getattr__), so `import decsim.config` and the command line start
without the machine's imports.
"""

import importlib

_MODULE_OF = {
    "Experiment": "decsim.experiments.experiment",
    "Point": "decsim.experiments.experiment",
    "grid": "decsim.experiments.experiment",
    "CollectionSettings": "decsim.experiments.collection",
    "MachineSettings": "decsim.settings",
    "Machine": "decsim.machine",
}
__all__ = tuple(_MODULE_OF)


def __getattr__(name: str) -> object:
    module_name = _MODULE_OF.get(name)
    if module_name is None:
        raise AttributeError(f"module 'decsim' has no attribute {name!r}")
    module = importlib.import_module(module_name)
    return getattr(module, name)
