"""A machine written as a graph: named parts, the ports they declare, wires.

    part_kinds.py  the kinds of part, each with the record type it
                   takes, its ports and, held once, its default
    graph.py       Part, Wire and Machine: parts by name, wires checked
                   as they are made, required ports left unwired

A part is a decsim settings record; its parameters sit at its name
(uf.unit_count), which `--set` reaches (decsim/parameters/tree.py).
"""
