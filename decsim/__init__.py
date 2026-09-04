"""decsim, a discrete-event simulator of the QEC reaction path.

A readout leaves the QPU, crosses the controller, the buffers and the
decoder, and comes back as an instruction; every hop is priced. The
root is decsim.machine.Machine, built from a MachineSettings; the front
over it is the experiments package.
"""
