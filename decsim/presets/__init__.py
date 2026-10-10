"""Machines ready to run, written as graphs, and the link cards they use.

    clocks.py          the chip's and the host's clocks
    link_cards.py      one-cycle hops, the bases' fabrics, the strong
                       side, measured round trips (RoCE v2, NVQLink), and
                       a card provisioned from a run's own geometry
    machines.py        the weak and strong decoder bases, a memory
                       workload, and a strong tier added to a machine
    experiment_one.py  Experiment 1's machine, every value sourced

A preset is a graph (decsim/topology/graph.py): change a part by name
(machine.with_settings("weak", unit_count=2)), or by path with --set,
and give it to decsim.Task.
"""
