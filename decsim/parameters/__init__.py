"""Every parameter of a machine, set by its path without editing code.

A machine's settings records are a tree of parts; each field is a
parameter at a dotted path (qpu.distance, weak.unit_count,
weak.engine.fetch_per_round), and each part's record can be
swapped by name (weak.type=union_find). An assignment
`path=value` sets one; `*` and `**` set many, as OMNeT++'s ini files do.

    units.py        values with units: "157ns", "250MHz", "3 cycles"
    fields.py       a record's fields as parameters, the unit in the value
    kinds.py        the records a part's slot takes, found by name
    tree.py         paths over the settings, assignments applied along them
    assignments.py  `path=value` texts from --set and DECSIM_SET
    in_force.py     the assignments every task a run file builds takes

`decsim run run.py --set PATH=VALUE` applies them to every task;
`decsim params run.py` lists the paths and `decsim info NAME` a record's
parameters (docs/how-to/set_parameters.md).
"""
