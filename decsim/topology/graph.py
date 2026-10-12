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
import decsim.records.program as program_records
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
        """The part's kind, the one whose type takes its record."""
        record_class = type(self.settings)
        return part_kinds.kind_of(record_class)

    def settings_hint(self) -> Any:
        """The type a record swapped into this part must meet.

        A part of no kind takes its own record's class and its subclasses.
        """
        kind = self.kind
        if kind is None:
            return type(self.settings)
        return kind.hint

    def run_file_fields(self, workload: object) -> tuple:
        """The record's fields the machine's workload fixes.

        A part of no kind has none; a QPU has its distance when the
        workload was made at one.
        """
        kind = self.kind
        if kind is None:
            return ()
        return part_kinds.run_file_fields(kind, workload)


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
    readout_routes give a readout of a whole patch footprint its own path
    (links/settings.py ReadoutRoute); observation is what the run records
    beside its results.

    A kind of part held once (part_kinds.KINDS) runs on its default when
    the machine holds no part of it. The workload is a workload lowered
    in Python or a maker row (decsim/producers.py), which at_task lowers
    against the QPU, so the task's id and every shot see the lowered
    workload.
    """

    # the fields whose parts parameters/tree.py addresses by name
    NAMED_PARTS: ClassVar[tuple] = ("parts",)

    clock: Optional[config.Clock] = None
    parts: tuple = ()
    wires: tuple = ()
    fabric: ports.FabricRule = fabrics.ReferenceFabric()
    readout_routes: tuple = ()
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

    def parts_of(self, kind_name: str) -> list:
        """The parts of one kind, in the order the graph writes them."""
        found = []
        for part in self.parts:
            kind = part.kind
            if kind is not None and kind.name == kind_name:
                found.append(part)
        return found

    def held(self, kind_name: str) -> Any:
        """The record of the one part of a kind held once, else its default."""
        found = self.parts_of(kind_name)
        if len(found) > 1:
            names = [part.name for part in found]
            raise GraphError(
                f"the parts {names} are each a {kind_name}, which a machine "
                "holds once"
            )
        if found:
            return found[0].settings
        kind = part_kinds.kind_named(kind_name)
        return kind.default

    def task_facts(self) -> dict:
        """The task's facts a threshold reads, None where none is stated.

        The names are threshold_sources.TASK_FACTS; a window size of
        None is the distance.
        """
        qpu = self.held(part_kinds.QPU)
        workload = self.held(part_kinds.WORKLOAD)
        workload_record = workload.workload_record
        probability = None
        if workload_record is not None:
            probability = workload_record.physical_error_probability
        windows = self.held("windows")
        scheme = windows.scheme
        return {
            "distance": qpu.distance,
            "physical_error_probability": probability,
            "round_period_microseconds": qpu.round_period_microseconds,
            "commit_rounds": scheme.commit_rounds,
            "buffer_rounds": scheme.buffer_rounds,
        }

    def at_task(self) -> "Machine":
        """The machine with its workload lowered and its thresholds read.

        A maker row is lowered against the QPU, then each switching
        part's calibration table row is read at the task's facts, so a
        task's id covers the number it gives; a machine already at its
        task is returned unchanged.
        """
        workload = self.held(part_kinds.WORKLOAD)
        qpu = self.held(part_kinds.QPU)
        lowered = workload.lowered_for(qpu)
        machine = self
        if lowered is not workload:
            machine = self.with_held(part_kinds.WORKLOAD, lowered)
        _refuse_a_route_on_a_patch_no_operation_has(machine)
        facts = machine.task_facts()
        for part in machine.parts_of(part_kinds.SWITCHING):
            threshold = part.settings.threshold
            read = threshold.at_task(facts)
            if read is threshold:
                continue
            settings = dataclasses.replace(part.settings, threshold=read)
            changed = dataclasses.replace(part, settings=settings)
            machine = machine.with_part(changed)
        return machine

    def set_in_the_run_file(self) -> tuple:
        """The paths the graph's workload fixes, each at its part's name."""
        workload = self.held(part_kinds.WORKLOAD)
        paths = []
        for part in self.parts:
            run_file_fields = part.run_file_fields(workload)
            for field in run_file_fields:
                paths.append(f"{part.name}.{field}")
        return tuple(paths)

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

    def with_held(self, kind_name: str, settings: Any) -> "Machine":
        """The machine with its part of a kind held once given this record.

        A machine with no such part gets one named for the kind.
        """
        found = self.parts_of(kind_name)
        if not found:
            part = Part(kind_name, settings)
            return self.with_part(part)
        return self.with_record(found[0].name, settings)

    def with_record(self, name: str, settings: Any) -> "Machine":
        """The machine with one part's record replaced, its site kept."""
        part = self.part(name)
        changed = dataclasses.replace(part, settings=settings)
        return self.with_part(changed)

    def with_settings(self, name: str, **changes: Any) -> "Machine":
        """The machine with fields of one part's record changed."""
        part = self.part(name)
        settings = dataclasses.replace(part.settings, **changes)
        return self.with_record(name, settings)

    def with_wires(self, *wires: Wire) -> "Machine":
        """The machine with these wires added."""
        return dataclasses.replace(self, wires=(*self.wires, *wires))

    def without_wire(self, source: str, target: str) -> "Machine":
        """The machine with the wire from source to target gone."""
        wires = []
        for wire in self.wires:
            if wire.source != source or wire.target != target:
                wires.append(wire)
        if len(wires) == len(self.wires):
            raise GraphError(f"no wire from {source} to {target}")
        return dataclasses.replace(self, wires=tuple(wires))

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
                "whose ports are not implemented yet"
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


def _refuse_a_route_on_a_patch_no_operation_has(machine: Machine) -> None:
    """A readout route on a patch no operation reads out is never taken.

    The fabric takes a route by a readout's whole footprint and
    qpu_to_controller for every other (links/fabric.py), so such a route
    would be ignored in silence. A footprint is an operation's patches,
    one of them, or its qubits (qpu/syndrome_devices.py), so only a
    patch no operation has is certain to match nothing.
    """
    routes = machine.readout_routes
    workload = machine.held(part_kinds.WORKLOAD)
    workload_record = workload.workload_record
    if not routes or workload_record is None:
        return
    occupied = _patches_read_out(workload_record)
    for route in routes:
        _refuse_a_route_off_the_patches(route, occupied)


def _refuse_a_route_off_the_patches(route: Any, occupied: set) -> None:
    """A route that names a patch outside occupied, refused."""
    unknown = [patch for patch in route.patch_ids if patch not in occupied]
    if not unknown:
        return
    raise ValueError(
        f"the readout route for patches {list(route.patch_ids)} "
        f"names {unknown}, which no operation reads out, so it would "
        "never be taken; the operations read out "
        f"{sorted(occupied, key=repr)}"
    )


def _patches_read_out(workload_record: object) -> set:
    """Every patch a readout's footprint can hold."""
    patches = set()
    for operation in workload_record.operations:
        footprint = program_records.patches_of(operation)
        patches.update(footprint)
        patches.update(operation.qubits)
    return patches
