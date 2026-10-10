"""A machine as a graph: named parts, each a settings record, and wires.

    machine = graph.Machine(
        clock=FRIDGE_CLOCK,
        parts=(
            graph.Part("qpu", qpu),
            graph.Part("controller", controller),
            graph.Part("chip_buffer", buffer, placed="fridge"),
            graph.Part("uf", union_find_pool, placed="fridge"),
            graph.Part("frame", frame),
        ),
        wires=(
            graph.Wire("qpu.readout", "controller.readout"),
            graph.Wire("controller.rounds", "chip_buffer.rounds"),
            ...
        ),
    )

A part's parameters sit at its name (uf.unit_count, uf.algorithm.type,
uf.placed), parameters/tree.py walking the parts by name. A wire is
checked as it is made, as gem5's PortRef.connect refuses an unknown
role pair and a port already bound (gem5 src/python/m5/params/
port_params.py:106-135); a required port left unwired is found before
the build, as SST checks the whole graph for dangling links before it
runs (sst-core src/sst/core/model/configGraph.cc:238-262).
"""

import collections
import dataclasses
from typing import Any, ClassVar, Optional

import decsim.config as config
import decsim.observe.settings as observe_settings
import decsim.ports as ports
import decsim.topology.fabrics as fabrics
import decsim.topology.part_kinds as part_kinds

# a port written part.port
_PORT_SEPARATOR = "."


class GraphError(ValueError):
    """A part or a wire the graph cannot hold, named by its path."""


@dataclasses.dataclass(frozen=True)
class Part:
    """One named part: a decsim settings record, and where it sits.

    placed names the site, the fridge or the host, from which a fabric
    prices the wires that reach the part; None for a part whose site no
    price reads.
    """

    name: str
    settings: Any
    placed: Optional[str] = None

    @property
    def kind(self) -> Optional[part_kinds.PartKind]:
        """The part's kind, from the slot that takes its record."""
        record_class = type(self.settings)
        return part_kinds.kind_of(record_class)

    def settings_hint(self) -> Any:
        """The type a record swapped into this part must meet.

        A part of no kind takes its own record's class and its subclasses.
        """
        kind = self.kind
        if kind is None:
            return type(self.settings)
        return part_kinds.slot_hint(kind)

    def run_file_fields(self) -> tuple:
        """The record's fields the run file built the workload's circuit from.

        A part of no kind has none; a QPU has its distance.
        """
        kind = self.kind
        if kind is None:
            return ()
        return part_kinds.run_file_fields(kind)


@dataclasses.dataclass(frozen=True)
class Wire:
    """One wire, from an output port to an input port, each part.port.

    link prices the wire; None leaves it to the machine's fabric.
    """

    source: str
    target: str
    link: Optional[ports.LinkCard] = None


@dataclasses.dataclass(frozen=True)
class Machine:
    """A machine's parts by name, the wires between them, its clock.

    clock is the clock every part that names none of its own counts its
    cycles on; fabric prices every route no wire's own link prices;
    observation is what the run records beside its results.
    """

    # the fields whose parts parameters/tree.py addresses by name
    NAMED_PARTS: ClassVar[tuple] = ("parts",)

    clock: Optional[config.Clock] = None
    parts: tuple = ()
    wires: tuple = ()
    fabric: ports.FabricRule = fabrics.ReferenceFabric()
    observation: observe_settings.ObservationSettings = (
        observe_settings.ObservationSettings()
    )

    def __post_init__(self) -> None:
        _refuse_a_repeated_name(self.parts)
        bound = collections.Counter()
        ports = {}
        for wire in self.wires:
            source = self._end(wire.source)
            target = self._end(wire.target)
            _refuse_unpaired_roles(wire, source, target)
            ports[wire.source] = source
            ports[wire.target] = target
            bound.update((wire.source, wire.target))
        _refuse_a_port_bound_twice(bound, ports)

    def part(self, name: str) -> Part:
        """The part of this name."""
        for part in self.parts:
            if part.name == name:
                return part
        names = [part.name for part in self.parts]
        raise GraphError(f"no part {name!r}; the parts are {names}")

    def unbound_ports(self) -> list:
        """Every required port no wire reaches, as part.port."""
        wired = set()
        for wire in self.wires:
            wired.add(wire.source)
            wired.add(wire.target)
        unbound = []
        for part in self.parts:
            unwired = _unwired_required(part, wired)
            unbound.extend(unwired)
        return unbound

    def with_part(self, part: Part) -> "Machine":
        """The machine with this part in place of its namesake, or added."""
        parts = [kept for kept in self.parts if kept.name != part.name]
        if len(parts) == len(self.parts):
            return dataclasses.replace(self, parts=(*self.parts, part))
        replaced = [_chosen(kept, part) for kept in self.parts]
        return dataclasses.replace(self, parts=tuple(replaced))

    def with_settings(self, name: str, **changes: Any) -> "Machine":
        """The machine with fields of one part's record changed."""
        part = self.part(name)
        settings = dataclasses.replace(part.settings, **changes)
        changed = dataclasses.replace(part, settings=settings)
        return self.with_part(changed)

    def with_wires(self, *wires: Wire) -> "Machine":
        """The machine with these wires added."""
        return dataclasses.replace(self, wires=(*self.wires, *wires))

    def without_part(self, name: str) -> "Machine":
        """The machine with a part and every wire that reaches it gone."""
        self.part(name)
        parts = [part for part in self.parts if part.name != name]
        wires = [wire for wire in self.wires if not _reaches(wire, name)]
        return dataclasses.replace(self, parts=tuple(parts), wires=tuple(wires))

    def _end(self, written: str) -> tuple:
        """The (part, port) a wire's end names."""
        part_name, separator, port_name = written.partition(_PORT_SEPARATOR)
        if not separator:
            raise GraphError(f"{written!r}: a wire's end is part.port")
        part = self.part(part_name)
        kind = part.kind
        if kind is None:
            record_class = type(part.settings)
            raise GraphError(
                f"{written}: {part_name} is a {record_class.__name__}, "
                "whose ports are not implemented yet (stage 4)"
            )
        port = kind.port(port_name)
        if port is None:
            ports = kind.port_names()
            raise GraphError(
                f"{written}: a {kind.name} has no port {port_name!r}; "
                f"its ports are {ports}"
            )
        return (part, port)


def _chosen(kept: Part, part: Part) -> Part:
    """The new part where it shares the kept one's name, else the kept."""
    if kept.name == part.name:
        return part
    return kept


def _reaches(wire: Wire, name: str) -> bool:
    """Whether either end of a wire is a port of the named part."""
    source_name, _, _ = wire.source.partition(_PORT_SEPARATOR)
    target_name, _, _ = wire.target.partition(_PORT_SEPARATOR)
    return name in (source_name, target_name)


def _refuse_a_repeated_name(parts: tuple) -> None:
    names = [part.name for part in parts]
    counts = collections.Counter(names)
    repeated = [name for name, count in counts.items() if count > 1]
    if repeated:
        raise GraphError(f"two parts are named {repeated[0]!r}")


def _refuse_unpaired_roles(wire: Wire, source: tuple, target: tuple) -> None:
    """A wire runs from an output to an input of the same signal."""
    _, source_port = source
    _, target_port = target
    is_output = source_port.direction == part_kinds.OUTPUT
    is_input = target_port.direction == part_kinds.INPUT
    is_same_signal = source_port.signal == target_port.signal
    if all((is_output, is_input, is_same_signal)):
        return
    raise GraphError(
        f"{wire.source} -> {wire.target}: ports with roles "
        f"{source_port.role!r} and {target_port.role!r} do not pair; a "
        "wire runs from an output to an input of the same signal"
    )


def _refuse_a_port_bound_twice(bound: collections.Counter, ports: dict) -> None:
    """A port takes one wire unless it takes many."""
    for written, count in bound.items():
        _, port = ports[written]
        if count > 1 and not port.is_many:
            raise GraphError(f"{written} takes one wire and has {count}")


def _unwired_required(part: Part, wired: set) -> list:
    kind = part.kind
    if kind is None:
        return []
    unwired = []
    for port in kind.ports:
        written = f"{part.name}{_PORT_SEPARATOR}{port.name}"
        if port.is_required and written not in wired:
            unwired.append(written)
    return unwired
