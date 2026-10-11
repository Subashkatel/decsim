"""A graph as today's MachineSettings, and today's settings as a graph.

Today's builder takes MachineSettings, whose slots hold one decoder or
a weak one escalating to a strong one, and a task's id is the hash of
that record. So a graph runs as the MachineSettings it resolves to, and
keeps the id of the record it resolves to. Each part goes to the slot
its kind and its wiring give it:

    a pool that escalates     weak_decoder; the pool its switching
                              part escalates to, strong_decoder
    a pool alone              strong_decoder placed on the host, else
                              weak_decoder
    a store                   the slot of the pool it feeds; a store
                              no pool reads, the slot left free, weak
                              first
    every other kind          its one slot

Every wire is one of today's paths, and a part's site and a wire's link
price the path (fabrics.py); the two wires through the switching part
are the escalation's one path. A graph of more pools or edges than the
slots hold is a new shape (holds), laid out by graph_layout.py; any
other graph the slots cannot hold, a store feeding a store, is refused.
"""

import decsim.settings as machine_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer
import decsim.topology.fabrics as fabrics
import decsim.topology.graph as graph

POOL_SLOTS = ("weak_decoder", "strong_decoder")
BUFFER_SLOTS = ("weak_syndrome_buffer", "strong_syndrome_buffer")
# the store slot beside each pool slot
BUFFER_BY_POOL = dict(zip(POOL_SLOTS, BUFFER_SLOTS, strict=True))
# each path of today's card by the slots at its ends; the boundary's
# path joins a pool to itself, no wire
PATH_BY_ENDS = {ends: path for path, ends in fabrics.LEGACY_ENDS.items()}
FRIDGE = fabrics.FRIDGE
HOST = fabrics.HOST
# graph_of's pools, by name and site, and each one's read and corrections
WEAK_POOL = ("weak", FRIDGE)
STRONG_POOL = ("strong", HOST)
WEAK_POOL_WIRES = (
    graph.Wire("weak_buffer.reads", "weak.rounds"),
    graph.Wire("weak.corrections", "frame.corrections"),
)
STRONG_POOL_WIRES = (
    graph.Wire("strong_buffer.reads", "strong.rounds"),
    graph.Wire("strong.corrections", "frame.corrections"),
)
STRONG_STORE_WRITE = graph.Wire("controller.rounds", "strong_buffer.rounds")
# the weak pool's escalations reach the switching part, whose re-decodes
# reach the strong pool
ESCALATION_WIRES = (
    graph.Wire("weak.escalate", "switching.results"),
    graph.Wire("switching.redecode", "strong.escalated"),
)
_NOT_BUILT = "not implemented yet (stage 5)"
_ESCALATE = "escalate"
_READS = "reads"
_REDECODE = "redecode"
_SWITCHING_SLOT = "switching"
_DEFAULT_BUFFER = syndrome_buffer.SyndromeBufferSettings()


class ShapeError(graph.GraphError):
    """A graph today's builder cannot build, named by what it holds."""


def holds(machine: graph.Machine) -> bool:
    """Whether today's slots hold the graph: two pools and one edge at most.

    A graph they hold runs as, and keeps the id of, the MachineSettings
    it resolves to; any other is a new shape.
    """
    by_kind = _parts_by_kind(machine)
    pools = by_kind.get("decoder_pool", [])
    edges = by_kind.get(_SWITCHING_SLOT, [])
    return len(pools) <= 2 and len(edges) <= 1


def resolved(machine: graph.Machine) -> machine_settings.MachineSettings:
    """The MachineSettings a graph runs as, and is identified by."""
    _refuse_unbound_ports(machine)
    parts_by_slot = _slots(machine)
    slot_by_part = {part.name: slot for slot, part in parts_by_slot.items()}
    sites = {slot: part.placed for slot, part in parts_by_slot.items()}
    routes = fabrics.legacy_routes(sites)
    wire_links = _wire_links(machine, slot_by_part)
    links = fabrics.card(machine.fabric, routes, wire_links, machine.clock)
    records = {slot: part.settings for slot, part in parts_by_slot.items()}
    return machine_settings.MachineSettings(
        clock=machine.clock,
        links=links,
        observation=machine.observation,
        **records,
    )


def graph_of(settings: machine_settings.MachineSettings) -> graph.Machine:
    """Today's settings as a graph, which resolves to them again.

    Each slot's record is the part of the slot's name, the weak tier in
    the fridge and the strong tier on the host, and the fabric is the
    settings' own card.
    """
    tier_parts, tier_wires = _tier_parts(settings)
    single_parts = _single_parts(settings)
    shared_wires = _shared_wires(settings)
    parts = (*single_parts, *tier_parts)
    wires = (*shared_wires, *tier_wires)
    fabric = fabrics.CardFabric(settings.links)
    return graph.Machine(
        clock=settings.clock,
        parts=parts,
        wires=wires,
        fabric=fabric,
        observation=settings.observation,
    )


def redecode_target(machine: graph.Machine, switching_name: str) -> graph.Part:
    """The pool a switching part's re-decodes reach; refused when none."""
    written = f"{switching_name}.{_REDECODE}"
    for wire in machine.wires:
        if wire.source == written:
            return _part_at(machine, wire.target)
    raise graph.GraphError(
        f"{written} reaches no pool: an escalate edge re-decodes on the "
        "pool its redecode port is wired to"
    )


def _refuse_unbound_ports(machine: graph.Machine) -> None:
    unbound = machine.unbound_ports()
    if unbound:
        raise graph.GraphError(f"required ports with no wire: {unbound}")


def _slots(machine: graph.Machine) -> dict:
    """Each slot's part: one-slot kinds, then the pools, then the stores."""
    by_kind = _parts_by_kind(machine)
    parts_by_slot = {}
    for kind_name, parts in by_kind.items():
        if kind_name in ("decoder_pool", "syndrome_buffer"):
            continue
        _refuse_more_than(parts, 1, kind_name)
        part = parts[0]
        kind = part.kind
        parts_by_slot[kind.slots[0]] = part
    pools = by_kind.get("decoder_pool", [])
    pool_slots = _pool_slots(machine, pools)
    buffers = by_kind.get("syndrome_buffer", [])
    buffer_slots = _buffer_slots(machine, buffers, pool_slots)
    return {**parts_by_slot, **pool_slots, **buffer_slots}


def _parts_by_kind(machine: graph.Machine) -> dict:
    """The parts of each kind, in the graph's order; a kindless part refused."""
    by_kind = {}
    for part in machine.parts:
        kind = part.kind
        if kind is None:
            record_class = type(part.settings)
            raise ShapeError(
                f"{part.name} is a {record_class.__name__}, a part of no "
                f"kind today's machine holds; {_NOT_BUILT}"
            )
        parts = by_kind.setdefault(kind.name, [])
        parts.append(part)
    return by_kind


def _pool_slots(machine: graph.Machine, pools: list) -> dict:
    """The weak and strong pools, from the escalation between them."""
    _refuse_more_than(pools, 2, "decoder_pool")
    escalations = _wires_from(machine, _ESCALATE)
    if len(pools) == 2:
        return _escalating_pools(machine, escalations)
    if escalations:
        raise ShapeError(f"a pool escalates to itself; {_NOT_BUILT}")
    if not pools:
        return {}
    pool = pools[0]
    if pool.placed == fabrics.HOST:
        return {"strong_decoder": pool}
    return {"weak_decoder": pool}


def _escalating_pools(machine: graph.Machine, escalations: list) -> dict:
    if len(escalations) != 1:
        raise ShapeError(
            "two pools run together only with one escalating to the "
            f"other; {_NOT_BUILT}"
        )
    escalation = escalations[0]
    weak = _part_at(machine, escalation.source)
    switching = _part_at(machine, escalation.target)
    strong = redecode_target(machine, switching.name)
    return {"weak_decoder": weak, "strong_decoder": strong}


def _buffer_slots(
    machine: graph.Machine, buffers: list, pool_slots: dict
) -> dict:
    """Each store at the slot of the pool it feeds; the rest, weak first."""
    _refuse_more_than(buffers, 2, "syndrome_buffer")
    fed = _fed_stores(machine, pool_slots)
    unread = [store for store in buffers if store not in fed.values()]
    free_slots = [slot for slot in BUFFER_SLOTS if slot not in fed]
    for slot, store in zip(free_slots, unread, strict=False):
        fed[slot] = store
    return fed


def _fed_stores(machine: graph.Machine, pool_slots: dict) -> dict:
    """The store each pool reads, at the store slot beside the pool's."""
    slot_by_pool = {part.name: slot for slot, part in pool_slots.items()}
    fed = {}
    for read in _wires_from(machine, _READS):
        store = _part_at(machine, read.source)
        reader = _part_at(machine, read.target)
        pool_slot = slot_by_pool.get(reader.name)
        if pool_slot is None:
            raise ShapeError(
                f"{read.source} -> {read.target}: a store feeds a store; "
                f"{_NOT_BUILT}"
            )
        fed[BUFFER_BY_POOL[pool_slot]] = store
    return fed


def _wire_links(machine: graph.Machine, slot_by_part: dict) -> dict:
    """Each wire's own link by its path; a wire on no path refused.

    A wire to or from the switching part is the escalation's path, from
    the weak pool to the strong one.
    """
    links = {}
    for wire in machine.wires:
        path = _path_of(wire, slot_by_part)
        if wire.link is not None:
            links[path] = wire.link
    return links


def _path_of(wire: graph.Wire, slot_by_part: dict) -> str:
    """The path of today's card a wire is, by the slots at its ends."""
    source_name, _, _ = wire.source.partition(".")
    target_name, _, _ = wire.target.partition(".")
    source_slot = slot_by_part[source_name]
    target_slot = slot_by_part[target_name]
    if source_slot == _SWITCHING_SLOT:
        source_slot = POOL_SLOTS[0]
    if target_slot == _SWITCHING_SLOT:
        target_slot = POOL_SLOTS[1]
    ends = (source_slot, target_slot)
    path = PATH_BY_ENDS.get(ends)
    if path is None:
        raise ShapeError(
            f"{wire.source} -> {wire.target} joins {ends[0]} to {ends[1]}, "
            f"no path of today's machine; {_NOT_BUILT}"
        )
    return path


def _wires_from(machine: graph.Machine, port_name: str) -> list:
    """The wires leaving a port of this name, on any part."""
    suffix = f".{port_name}"
    return [wire for wire in machine.wires if wire.source.endswith(suffix)]


def _part_at(machine: graph.Machine, written: str) -> graph.Part:
    part_name, _, _ = written.partition(".")
    return machine.part(part_name)


def _refuse_more_than(parts: list, count: int, kind_name: str) -> None:
    if len(parts) <= count:
        return
    names = [part.name for part in parts]
    raise ShapeError(
        f"{len(parts)} parts of kind {kind_name}, {names}; today's machine "
        f"holds at most {count}; {_NOT_BUILT}"
    )


def _single_parts(settings: machine_settings.MachineSettings) -> tuple:
    """The parts every machine holds once, and the frame when it has one."""
    parts = [
        graph.Part("qpu", settings.qpu),
        graph.Part("controller", settings.controller),
        graph.Part("idle_policy", settings.idle_policy),
        graph.Part("detection_events", settings.detection_events),
        graph.Part("windows", settings.windows),
        graph.Part("decoder_manager", settings.decoder_manager),
        graph.Part("workload", settings.workload),
        graph.Part("magic_state_factory", settings.magic_state_factory),
    ]
    if settings.pauli_frame is not None:
        frame = graph.Part("frame", settings.pauli_frame)
        parts.append(frame)
    if settings.switching is not None:
        switching = graph.Part("switching", settings.switching)
        parts.append(switching)
    return tuple(parts)


def _shared_wires(settings: machine_settings.MachineSettings) -> tuple:
    """The QPU's loop through the controller, and the frame's decision."""
    wires = [
        graph.Wire("qpu.readout", "controller.readout"),
        graph.Wire("controller.pulses", "qpu.pulses"),
    ]
    if settings.pauli_frame is not None:
        decision = graph.Wire("frame.decisions", "controller.decisions")
        wires.append(decision)
    return tuple(wires)


def _tier_parts(settings: machine_settings.MachineSettings) -> tuple:
    """The stores and pools of today's settings as parts, and their wires.

    The weak store is always written; the strong one is a part when a
    strong pool reads it or it differs from the default, the record a
    task's id holds. Each pool reads its tier's store and corrects the
    frame, and the weak pool escalates to the strong one.
    """
    parts = [graph.Part("weak_buffer", settings.weak_syndrome_buffer, FRIDGE)]
    wires = [graph.Wire("controller.rounds", "weak_buffer.rounds")]
    if _has_strong_store(settings):
        strong_store = graph.Part(
            "strong_buffer", settings.strong_syndrome_buffer, HOST
        )
        parts.append(strong_store)
        wires.append(STRONG_STORE_WRITE)
    tiers = (
        (settings.weak_decoder, WEAK_POOL, WEAK_POOL_WIRES),
        (settings.strong_decoder, STRONG_POOL, STRONG_POOL_WIRES),
    )
    for pool_settings, (name, placed), pool_wires in tiers:
        if pool_settings is None:
            continue
        pool = graph.Part(name, pool_settings, placed)
        parts.append(pool)
        framed = _with_frame(pool_wires, settings)
        wires.extend(framed)
    if _has_two_pools(settings):
        wires.extend(ESCALATION_WIRES)
    return tuple(parts), tuple(wires)


def _has_strong_store(settings: machine_settings.MachineSettings) -> bool:
    has_strong_pool = settings.strong_decoder is not None
    return has_strong_pool or settings.strong_syndrome_buffer != _DEFAULT_BUFFER


def _has_two_pools(settings: machine_settings.MachineSettings) -> bool:
    has_weak_pool = settings.weak_decoder is not None
    return has_weak_pool and settings.strong_decoder is not None


def _with_frame(
    pool_wires: tuple, settings: machine_settings.MachineSettings
) -> tuple:
    """A pool's wires; its corrections reach the frame where there is one."""
    read, corrections = pool_wires
    if settings.pauli_frame is None:
        return (read,)
    return (read, corrections)
