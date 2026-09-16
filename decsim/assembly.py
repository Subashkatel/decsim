"""What the machine is made of, and what is wired to what.

Three tables. SEATS names every component of a run, in the order the
root builds them, beside the function that builds that one seat from the
run's settings. WIRES names every edge between two seats as the port it
fills and the seat that fills it, which is gem5's script assigning one
component's port to another's
(tmp/resources/gem5/configs/learning_gem5/part1/simple.py:68). The root
builds, then binds, then starts: nothing is scheduled while the graph is
still being assembled, so the order the rows sit in cannot move a tick
(tmp/resources/gem5/src/sim/sim_object.hh lines 194 and 280).

SEED_ROOTS names every owner of randomness as the segment the run seed
hashes beside the seat that owns it. A component's seed is derived from
its framed path, so the segment a row names is part of that component's
result and nothing else's; a row whose seat this run does not build
takes None and binds nothing.

A run the machine has no use for a seat in has no SEATS row for it and
no WIRES row either, so no port is ever bound to None; seats_for reads
the five conditions that decide that, once.

Two rows name a seat another row built, because the class takes that
neighbour at construction and cannot take it as a port: the magic state
factory, whose card refuses an ambiguous decode service as it reads it,
and the primary store output, which is one of the two store ends rather
than a third one. Both rows therefore sit after the rows they read.

The member readers come before the tables because a tuple is built when
the module loads and every row names its builder.
"""

import dataclasses
from typing import Any

import decsim.build.controller_side as controller_side
import decsim.build.decoders as decoder_build
import decsim.build.listeners as listener_build
import decsim.build.plan as plan_build
import decsim.build.stores as store_build
import decsim.build.window_side as window_side
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.settings as machine_settings


@dataclasses.dataclass(frozen=True)
class Parts:
    """What a seat's builder reads: the run's fixtures, and the seats so far.

    The settings and the engine are every seat's input; the plan, the
    decoder pool and the escalation policy are the records the root
    compiles from them; the detection event placement is the one seat the
    pool is compiled from, so the root builds it with them. seats holds
    the rows built so far, and only the two rows above read it.
    """

    settings: machine_settings.MachineSettings
    engine: engine_module.Engine
    plan: plan_build.Plan
    escalation_policy: Any
    pool: decoder_build.DecoderPool
    detection_events: ports.DetectionEventPlacement
    seats: dict = dataclasses.field(default_factory=dict)


def _escalation_policy(parts):
    return parts.escalation_policy


def _scheme(parts):
    return parts.plan.scheme


def _window_interaction(parts):
    return parts.plan.window_interaction


def _boundary_policy(parts):
    return parts.plan.boundary_policy


def _error_model_provider(parts):
    return parts.plan.error_model_provider


def _router(parts):
    return parts.pool.router


def _detection_events(parts):
    return parts.detection_events


SEATS = (
    ("escalation_policy", _escalation_policy),
    ("scheme", _scheme),
    ("window_interaction", _window_interaction),
    ("boundary_policy", _boundary_policy),
    ("error_model_provider", _error_model_provider),
    ("router", _router),
    ("detection_events", _detection_events),
    ("conditional_release", controller_side.build_conditional_release),
    ("links", store_build.build_links),
    ("held_rounds", store_build.build_held_rounds),
    ("weak_syndrome_buffer", store_build.build_weak_syndrome_buffer),
    ("strong_syndrome_buffer", store_build.build_strong_syndrome_buffer),
    ("store_transfers", store_build.build_store_transfers),
    ("weak_output", store_build.build_weak_output),
    ("strong_output", store_build.build_strong_output),
    ("primary_output", store_build.build_primary_output),
    ("pauli_frame", store_build.build_pauli_frame),
    ("decoder_manager", controller_side.build_decoder_manager),
    ("factory", controller_side.build_factory),
    ("models", window_side.build_models),
    ("planner", window_side.build_planner),
    ("tracker", window_side.build_tracker),
    ("retention", window_side.build_retention),
    ("window_transfers", window_side.build_window_transfers),
    ("decoder_output", window_side.build_decoder_output),
    ("gate", window_side.build_gate),
    ("builder", window_side.build_request_builder),
    ("ledger", window_side.build_ledger),
    ("results", window_side.build_results),
    ("courier", window_side.build_courier),
    ("committer", window_side.build_committer),
    ("verdict", window_side.build_verdict),
    ("confidence_signal", window_side.build_confidence_signal),
    ("gap_join", window_side.build_gap_join),
    ("requester", window_side.build_requester),
    ("regions", window_side.build_regions),
    ("shape", window_side.build_strong_window_shape),
    ("strong_redecode", window_side.build_strong_redecode),
    ("window_manager", window_side.build_window_manager),
    (
        "strong_syndrome_round_receiver",
        store_build.build_strong_syndrome_round_receiver,
    ),
    (
        "weak_syndrome_round_receiver",
        store_build.build_weak_syndrome_round_receiver,
    ),
    ("memory_arrivals", decoder_build.build_memory_round_arrivals),
    ("transmitter", controller_side.build_transmitter),
    ("syndrome_round_sender", controller_side.build_syndrome_round_sender),
    ("rounds_in_flight", controller_side.build_rounds_in_flight),
    ("assembler", controller_side.build_assembler),
    ("qpu", controller_side.build_qpu),
    ("instruction_output", controller_side.build_instruction_output),
    ("streams", controller_side.build_feedback_streams),
    ("idle_rounds", controller_side.build_idle_rounds),
    ("issuer", controller_side.build_issuer),
    ("controller", controller_side.build_controller),
    ("execution_runtime", controller_side.build_execution_runtime),
    ("decision_dispatch", controller_side.build_decision_dispatch),
)

WIRES = (
    # the fabric every hop rides
    ("store_transfers.link", "links"),
    ("window_transfers.link", "links"),
    ("transmitter.link", "links"),
    ("syndrome_round_sender.link", "links"),
    ("controller.link", "links"),
    ("instruction_output.link", "links"),
    ("decision_dispatch.link", "links"),
    # the stores
    ("weak_syndrome_buffer.held_rounds", "held_rounds"),
    ("strong_syndrome_buffer.held_rounds", "held_rounds"),
    ("weak_output.transfers", "store_transfers"),
    ("weak_output.store", "weak_syndrome_buffer"),
    ("strong_output.transfers", "store_transfers"),
    ("strong_output.store", "strong_syndrome_buffer"),
    ("weak_syndrome_round_receiver.store", "weak_syndrome_buffer"),
    ("weak_syndrome_round_receiver.output", "weak_output"),
    ("weak_syndrome_round_receiver.windows", "window_manager"),
    ("strong_syndrome_round_receiver.store", "strong_syndrome_buffer"),
    ("strong_syndrome_round_receiver.windows", "window_manager"),
    # the window side
    ("models.provider", "error_model_provider"),
    ("models.router", "router"),
    ("planner.scheme", "scheme"),
    ("planner.models", "models"),
    ("tracker.scheme", "scheme"),
    ("tracker.planner", "planner"),
    ("retention.weak_store", "weak_syndrome_buffer"),
    ("retention.strong_store", "strong_syndrome_buffer"),
    ("retention.planner", "planner"),
    ("retention.tracker", "tracker"),
    ("decoder_output.transfers", "window_transfers"),
    ("decoder_output.frame", "pauli_frame"),
    ("gate.planner", "planner"),
    ("gate.interaction", "window_interaction"),
    ("gate.input_fold", "decoder_manager.input_fold()"),
    ("builder.planner", "planner"),
    ("builder.tracker", "tracker"),
    ("builder.interaction", "window_interaction"),
    ("builder.gate", "gate"),
    ("results.planner", "planner"),
    ("results.tracker", "tracker"),
    ("results.retention", "retention"),
    ("results.ledger", "ledger"),
    ("results.conditional_release", "conditional_release"),
    ("results.factory", "factory"),
    ("courier.planner", "planner"),
    ("courier.transfers", "window_transfers"),
    ("courier.interaction", "window_interaction"),
    ("courier.boundary_policy", "boundary_policy"),
    ("courier.windows", "window_manager"),
    ("committer.courier", "courier"),
    ("committer.decoder_output", "decoder_output"),
    ("committer.results", "results"),
    ("committer.strong_redecode", "strong_redecode"),
    ("verdict.planner", "planner"),
    ("verdict.tracker", "tracker"),
    ("verdict.escalation_policy", "escalation_policy"),
    ("verdict.decode_queue", "decoder_manager"),
    ("verdict.committer", "committer"),
    ("verdict.strong_redecode", "strong_redecode"),
    ("gap_join.signal", "confidence_signal"),
    ("gap_join.verdict", "verdict"),
    ("gap_join.decode_queue", "decoder_manager"),
    ("requester.tracker", "tracker"),
    ("requester.retention", "retention"),
    ("requester.builder", "builder"),
    ("requester.decode_queue", "decoder_manager"),
    ("requester.escalation_policy", "escalation_policy"),
    ("requester.verdict", "verdict"),
    ("requester.store_output", "primary_output"),
    ("requester.gap_join", "gap_join"),
    ("regions.planner", "planner"),
    ("regions.tracker", "tracker"),
    ("regions.retention", "retention"),
    ("regions.interaction", "window_interaction"),
    ("shape.regions", "regions"),
    ("shape.planner", "planner"),
    ("shape.retention", "retention"),
    ("shape.builder", "builder"),
    ("shape.requester", "requester"),
    ("shape.ledger", "ledger"),
    ("shape.courier", "courier"),
    ("strong_redecode.shape", "shape"),
    ("strong_redecode.decoder_output", "decoder_output"),
    ("strong_redecode.strong_output", "strong_output"),
    ("strong_redecode.decode_queue", "decoder_manager"),
    ("strong_redecode.verdict", "verdict"),
    ("window_manager.planner", "planner"),
    ("window_manager.tracker", "tracker"),
    ("window_manager.retention", "retention"),
    ("window_manager.requester", "requester"),
    ("window_manager.courier", "courier"),
    ("window_manager.results", "results"),
    ("window_manager.strong_redecode", "strong_redecode"),
    ("window_manager.window_interaction", "window_interaction"),
    # the readout path back through the controller
    ("memory_arrivals.windows", "window_manager"),
    ("transmitter.memory_arrivals", "memory_arrivals"),
    ("transmitter.weak_receiver", "weak_syndrome_round_receiver"),
    ("syndrome_round_sender.weak_receiver", "weak_syndrome_round_receiver"),
    ("syndrome_round_sender.weak_store", "weak_syndrome_buffer"),
    ("syndrome_round_sender.strong_receiver", "strong_syndrome_round_receiver"),
    ("syndrome_round_sender.held_rounds", "held_rounds"),
    ("syndrome_round_sender.transmitter", "transmitter"),
    ("syndrome_round_sender.windows", "window_manager"),
    ("rounds_in_flight.held_rounds", "held_rounds"),
    ("rounds_in_flight.transmitter", "transmitter"),
    ("assembler.detection_events", "detection_events"),
    ("assembler.rounds_in_flight", "rounds_in_flight"),
    ("assembler.syndrome_round_sender", "syndrome_round_sender"),
    ("controller.assembler", "assembler"),
    # the QPU and the control loop
    ("qpu.readout_receiver", "controller"),
    ("qpu.runtime", "execution_runtime"),
    ("qpu.idle_rounds", "idle_rounds"),
    ("instruction_output.qpu", "qpu"),
    ("streams.qpu", "qpu"),
    ("streams.windows", "window_manager"),
    ("streams.runtime", "execution_runtime"),
    ("idle_rounds.decode_queue", "decoder_manager"),
    ("idle_rounds.streams", "streams"),
    ("idle_rounds.qpu", "qpu"),
    ("issuer.streams", "streams"),
    ("issuer.idle_rounds", "idle_rounds"),
    ("issuer.windows", "window_manager"),
    ("issuer.output", "instruction_output"),
    ("execution_runtime.issuer", "issuer"),
    ("execution_runtime.factory", "factory"),
    ("conditional_release.dispatch", "decision_dispatch"),
    ("conditional_release.runtime", "execution_runtime"),
    ("decision_dispatch.instruction_output", "instruction_output"),
)

# The seats whose first work needs a port and happens once the graph is
# wired, which is gem5's init; the factory and the execution runtime
# queue the run's first events instead and start when the run does,
# which is gem5's startup (sim_object.hh lines 194 and 280).
STARTS_WHEN_WIRED = ("planner", "window_manager", "syndrome_round_sender")

# The seed path of every stochastic owner, in the order the run seed
# hashes them: a seat by name, what one of a seat's readers answers, or
# a fixture the root compiled the seats from.
SEED_ROOTS = (
    ("code", "plan.code"),
    ("scheme", "scheme"),
    ("device", "plan.device"),
    ("error_model_provider", "error_model_provider"),
    ("decoder_router", "router"),
    ("factory", "factory"),
    ("escalation_policy", "escalation_policy"),
    ("scheduler", "pool.scheduler"),
    ("decoder_memory_transfer", "decoder_manager.input_transport()"),
    ("boundary_policy", "boundary_policy"),
    ("window_interaction", "window_interaction"),
    ("idle_policy", "plan.idle_policy"),
    ("conditional_release", "conditional_release"),
    ("controller", "controller"),
    ("qpu", "qpu"),
    ("execution_runtime", "execution_runtime"),
    ("pauli_frame", "pauli_frame"),
)


def build_seats(parts: Parts) -> dict:
    """Every seat of this run, built from its settings, in SEATS order."""
    for name, build in seats_for(parts):
        parts.seats[name] = build(parts)
    return parts.seats


def seats_for(parts: Parts) -> tuple:
    """The rows this run builds; a seat it has no use for has none."""
    absent = _absent_seats(parts)
    rows = []
    for name, build in SEATS:
        if name not in absent:
            rows.append((name, build))
    return tuple(rows)


def wires_for(parts: Parts) -> tuple:
    """The rows whose two ends this run builds."""
    built = set()
    for name, _build in seats_for(parts):
        built.add(name)
    rows = []
    for source, target in WIRES:
        if _both_ends_are_built(source, target, built):
            rows.append((source, target))
    return tuple(rows)


def bind(wires: tuple, seats: dict) -> None:
    """Fill every named port with the seat that answers it.

    The assignment runs the port's own refusal of a second bind
    (decsim/ports.py, Port), gem5's PortRef.connect
    (tmp/resources/gem5/src/python/m5/params/port_params.py:109-114). The
    table is this file's input, so a row that names a seat the run did
    not build, a port a class does not declare, or a peer that does not
    answer the port's protocol is refused here, with the row printed.
    """
    for source, target in wires:
        seat_name, port_name = source.split(".")
        seat = _seat(seat_name, seats, source)
        port = _port(seat, port_name, source)
        peer = _peer(target, seats, source)
        _check_answers(peer, port, source, target)
        setattr(seat, port_name, peer)


def start_wired_seats(seats: dict) -> None:
    """Let every seat whose first work needs its ports do that work."""
    for name in STARTS_WHEN_WIRED:
        seats[name].start()


def seed_roots(parts: Parts, seats: dict) -> tuple:
    """The seed path of every stochastic owner; the segments are results."""
    owners = {}
    for name, target in SEED_ROOTS:
        owners[name] = _seed_owner(target, parts, seats)
    return listener_build.build_seed_roots(**owners)


def _seed_owner(target, parts: Parts, seats: dict):
    """What one seed root names, or None when this run has no such owner."""
    if target.endswith("()"):
        return _peer(target, seats, target)
    if "." not in target:
        return seats.get(target)
    fixture_name, member_name = target.split(".")
    fixture = getattr(parts, fixture_name)
    return getattr(fixture, member_name)


def _absent_seats(parts: Parts) -> frozenset:
    """The rows this run has no use for, by the condition that drops them."""
    absent = _absent_strong_seats(parts.escalation_policy)
    absent |= _absent_named_seats(parts)
    return frozenset(absent)


def _absent_strong_seats(escalation_policy) -> set:
    """The room-side rows, which a run that never reads there does without."""
    absent = set()
    if not store_build.uses_strong_store(escalation_policy):
        absent.update(
            (
                "strong_syndrome_buffer",
                "strong_output",
                "strong_syndrome_round_receiver",
            )
        )
    if not escalation_policy.requires_strong_context:
        absent.update(("regions", "shape", "strong_redecode"))
    return absent


def _absent_named_seats(parts: Parts) -> set:
    """The rows a run has only when its yaml or its workload names them."""
    absent = set()
    if parts.settings.pauli_frame is None:
        absent.add("pauli_frame")
    if not window_side.decides_on_a_confidence(parts.settings):
        absent.update(("confidence_signal", "gap_join"))
    if parts.plan.error_model_provider is None:
        absent.add("error_model_provider")
    return absent


def _both_ends_are_built(source: str, target: str, built: set) -> bool:
    """Whether this run built the seat each end of the wire names."""
    source_parts = source.split(".")
    target_parts = target.split(".")
    return source_parts[0] in built and target_parts[0] in built


def _seat(name: str, seats: dict, row: str):
    """The seat a row names; a name off the table is the file's mistake."""
    if name not in seats:
        raise ValueError(
            f"the wire {row!r} names {name!r}, which this run has no seat for"
        )
    return seats[name]


def _port(seat, port_name: str, row: str) -> ports.Port:
    """The port a row fills; a name the class does not declare is refused."""
    owner = type(seat)
    port = getattr(owner, port_name, None)
    if isinstance(port, ports.Port):
        return port
    raise ValueError(f"the wire {row!r} names no port of {owner.__name__}")


def _peer(target: str, seats: dict, row: str):
    """The seat a wire points at, or what one of its readers answers."""
    if not target.endswith("()"):
        return _seat(target, seats, row)
    peer_name, reader_name = target[:-2].split(".")
    peer = _seat(peer_name, seats, row)
    reader = getattr(peer, reader_name)
    return reader()


def _check_answers(peer, port: ports.Port, source: str, target: str) -> None:
    """The peer answers what the port declares, or the row is refused."""
    if isinstance(peer, port.protocol):
        return
    offered = type(peer)
    protocol = port.protocol
    raise ValueError(
        f"the wire {source!r} to {target!r} offers a {offered.__name__}, "
        f"which does not answer {protocol.__name__}"
    )
