"""A machine written as a graph: named parts, the ports they declare, wires.

    part_kinds.py  the kinds of part, found by the MachineSettings slot
                   that takes a part's record, and each kind's ports
    graph.py       Part, Wire and Machine: parts by name, wires checked
                   as they are made, required ports left unwired

A part is a decsim settings record; its parameters sit at its name
(uf.unit_count), so `--set` reaches a graph as it reaches MachineSettings
(decsim/parameters/tree.py).
"""
