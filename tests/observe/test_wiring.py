"""The wiring: every listener connected once, and no source heard twice.

decsim/observe/wiring.py is the one place that connects a listener to
the sources it hears, so the census below is the whole picture: walk the
built machine, find every trace source, and read who hears it. A source
with two listeners of one class is a connection made twice, which would
double a count without failing anything.
"""

import decsim.machine as machine_module
import decsim.observe.trace_source as trace_source
import tests.observe.gate_point as gate_point

EVERY_KNOB = {
    "record_switching_windows": True,
    "round_store_occupancy": True,
    "backlog_trace": True,
    "decoder_utilization": True,
    "decoder_memory_occupancy": True,
    "data_movement": True,
    "trace": "chrome",
}


def _listener_class(listener):
    """The class that hears, through a partial or a bound method."""
    bound = getattr(listener, "func", listener)
    owner = getattr(bound, "__self__", None)
    if owner is None:
        return type(bound)
    return type(owner)


def _walk(root) -> list:
    """(owner, attribute, source) for every trace source the machine holds."""
    found = []
    seen = set()
    pending = [root]
    while pending:
        owner = pending.pop()
        identity = id(owner)
        if identity in seen:
            continue
        seen.add(identity)
        children = _children(owner)
        pending.extend(children)
        sources = _sources_on(owner)
        found.extend(sources)
    return found


def _children(owner) -> list:
    """The decsim values one object holds, directly or in a plain container."""
    state = getattr(owner, "__dict__", None)
    if state is None:
        return []
    children = []
    for value in state.values():
        held = _decsim_values(value)
        children.extend(held)
    return children


def _decsim_values(value) -> list:
    """The value itself when it is a decsim object, or the ones it holds.

    A container may hold containers (the pool keeps its units in a list
    per pool name), so the walk goes through them until it reaches an
    object of this package or something it does not follow.
    """
    if isinstance(value, (list, tuple, set)):
        return _from_items(value)
    if isinstance(value, dict):
        items = value.values()
        return _from_items(items)
    if _is_decsim(value):
        return [value]
    return []


def _from_items(items) -> list:
    """The decsim objects a container holds, however deeply."""
    found = []
    for item in items:
        held = _decsim_values(item)
        found.extend(held)
    return found


def _is_decsim(value) -> bool:
    """Whether the value is one of this package's own objects."""
    kind = type(value)
    module = kind.__module__
    return module.startswith("decsim.")


def _sources_on(owner) -> list:
    """The trace sources one object declares, by attribute name."""
    state = getattr(owner, "__dict__", None)
    if state is None:
        return []
    sources = []
    for name, value in state.items():
        if isinstance(value, trace_source.TraceSource):
            kind = type(owner)
            sources.append((kind.__name__, name, value))
    return sources


def test_no_source_is_heard_twice_by_one_listener_class():
    """One connection per (source, listener class), with every knob on."""
    point = gate_point.settings(**EVERY_KNOB)
    machine = machine_module.Machine.build(point, gate_point.SEED)

    census = _walk(machine)
    doubled = []
    for owner_name, source_name, source in census:
        classes = []
        for listener in source.listeners:
            heard_by = _listener_class(listener)
            classes.append(heard_by)
        if len(classes) != len(set(classes)):
            doubled.append((owner_name, source_name, classes))

    assert doubled == []
    assert len(census) > 30


def test_every_source_has_a_listener_when_every_knob_is_on():
    """A source nothing hears is a fire into an empty list for every run.

    With every knob on, each source a component declares is wired here
    or it is dead (rule 5); a new source added to a component without a
    connection in wiring.py fails this test by name.
    """
    point = gate_point.settings(**EVERY_KNOB)
    machine = machine_module.Machine.build(point, gate_point.SEED)

    census = _walk(machine)
    unheard = []
    for owner_name, source_name, source in census:
        if not source.has_listeners:
            unheard.append((owner_name, source_name))

    assert unheard == []


def test_every_listener_the_section_asks_for_is_built_and_heard():
    """A knob on builds its listener; a knob off leaves the field None."""
    asked_point = gate_point.settings(**EVERY_KNOB)
    silent_point = gate_point.settings()
    asked = machine_module.Machine.build(asked_point, gate_point.SEED)
    silent = machine_module.Machine.build(silent_point, gate_point.SEED)

    for machine in (asked, silent):
        assert machine.observation.log is not None
        assert machine.observation.round_events is not None
        assert machine.observation.stages is not None
    assert asked.observation.trace_writer is not None
    assert asked.observation.data_movement is not None
    assert asked.observation.decode_records is not None
    assert asked.observation.round_store_occupancy is not None
    assert asked.observation.decode_backlog is not None
    assert asked.observation.decoder_utilization is not None
    assert asked.observation.decoder_memory_occupancy is not None
    assert silent.observation.trace_writer is None
    assert silent.observation.data_movement is None
    assert silent.observation.decode_records is None
    assert silent.observation.round_store_occupancy is None
    assert silent.observation.decode_backlog is None
    assert silent.observation.decoder_utilization is None
    assert silent.observation.decoder_memory_occupancy is None
