"""The controller's stream bookkeeping: the protected region's refusals.

A protected region keeps one live stream on a patch between a start and
an end operation. The program declares the regions and this component
indexes them at load, which is where decsim checks a yaml-shaped input
once and loudly (STYLE.md rule 4). The refusals pinned here are all
raised by that one index pass: a region whose stream nobody owns, two
regions on one stream, an endpoint that is not an operation or does not
hold the region's patch, and a feedback source that is not itself
executed.

NoFeedbackStreams is the row a run without streams gets, so its surface
is checked against the real one: a caller must not learn which it holds.
The protected cycle's one outward call is pinned here too: the boundary
and the seal are cadence changes, and the runtime that holds operations
for them hears each one.
"""

import dataclasses
import functools
from typing import Any, Optional

import pytest

import decsim.controller.feedback_streams as feedback_streams
import decsim.engine as engine_module
import decsim.records.program as program_records


def test_a_protected_group_emits_and_releases_once_per_shared_cycle() -> None:
    program = _group_program()
    engine = engine_module.Engine()
    qpu = _RoundLogQpu(engine)
    windows = _Windows()
    streams = _streams(
        program,
        regions=program.protected_regions,
        engine=engine,
        qpu=qpu,
        window_manager=windows,
    )
    streams.runtime = _Runtime()
    first, final = program.operations
    streams.begin(first)
    assert streams.is_live_protected_patch("A")
    assert streams.is_live_protected_patch("B")
    close = functools.partial(streams.request_closes, final)
    engine.schedule(2000, close)
    engine.run()
    assert qpu.emissions == [(7, 1, 1000, False), (7, 2, 2000, True)]
    assert windows.seals == [(7, 2)]
    assert not streams.is_live_protected_patch("A")
    assert not streams.is_live_protected_patch("B")


def test_a_region_begun_between_edges_emits_on_the_qpu_cycle() -> None:
    """A released operation starts a region between two QPU cycle edges."""
    program = _group_program()
    engine = engine_module.Engine()
    qpu = _RoundLogQpu(engine)
    windows = _Windows()
    streams = _streams(
        program,
        regions=program.protected_regions,
        engine=engine,
        qpu=qpu,
        window_manager=windows,
    )
    streams.runtime = _Runtime()
    first, final = program.operations
    begin = functools.partial(streams.begin, first)
    close = functools.partial(streams.request_closes, final)
    engine.schedule(400, begin)
    engine.schedule(2000, close)
    engine.run()
    assert qpu.emissions == [(7, 1, 1000, False), (7, 2, 2000, True)]
    assert windows.seals == [(7, 2)]


def test_a_seal_the_qpu_refuses_never_reaches_the_windows() -> None:
    """Only the QPU's source saw the rounds, so it attests before the seal."""
    program = _group_program()
    engine = engine_module.Engine()
    qpu = _RoundLogQpu(engine)
    qpu.refused_length = 2
    windows = _Windows()
    streams = _streams(
        program,
        regions=program.protected_regions,
        engine=engine,
        qpu=qpu,
        window_manager=windows,
    )
    streams.runtime = _Runtime()
    first, final = program.operations
    streams.begin(first)
    close = functools.partial(streams.request_closes, final)
    engine.schedule(2000, close)
    with pytest.raises(RuntimeError, match="sealed length differs"):
        engine.run()
    assert qpu.attested_lengths == [(7, 2)]
    assert windows.seals == []


def test_a_group_endpoint_must_hold_every_owner_patch() -> None:
    program = _group_program()
    first, final = program.operations
    partial = dataclasses.replace(final, patches=("B",))
    program = dataclasses.replace(program, operations=(first, partial))
    sentence = "protected stream 7's end operation does not hold every patch"
    with pytest.raises(ValueError, match=sentence):
        _streams(program, regions=program.protected_regions)


def test_a_protected_group_requires_one_common_cadence() -> None:
    program = _group_program()
    first_patch = _resolved_patch("A")
    second_patch = _resolved_patch("B")
    second_patch = dataclasses.replace(second_patch, round_ticks=2000)
    patches = (first_patch, second_patch)
    with pytest.raises(ValueError, match="patches require a common cadence"):
        _streams(program, regions=program.protected_regions, patches=patches)


@pytest.mark.parametrize("patches", [(), ("A", "A")])
def test_a_protected_owner_requires_nonempty_unique_patches(
    patches: tuple,
) -> None:
    program = _group_program()
    owner = dataclasses.replace(program.dynamic_streams[0], patches=patches)
    program = dataclasses.replace(program, dynamic_streams=(owner,))
    with pytest.raises(ValueError, match="requires nonempty unique patches"):
        _streams(program, regions=program.protected_regions)


def test_a_new_group_waits_while_any_member_is_protected() -> None:
    program = _overlapping_group_program()
    engine = engine_module.Engine()
    qpu = _Qpu(engine)
    streams = _streams(
        program, regions=program.protected_regions, engine=engine, qpu=qpu
    )
    streams.begin(program.operations[0])
    assert streams.blocks_start(program.operations[2])


def test_per_patch_idle_callbacks_advance_a_joint_stream_only_once() -> None:
    program = _unprotected_group_program()
    operation = program.operations[0]
    engine = engine_module.Engine()
    qpu = _RoundLogQpu(engine, idle_group=(1, ("A", "B")))
    windows = _Windows()
    streams = _streams(
        program,
        regions=(),
        engine=engine,
        qpu=qpu,
        window_manager=windows,
    )
    streams.begin(operation)
    assert streams.extend_live_stream(operation, "A")
    assert streams.extend_live_stream(operation, "B")
    extend = functools.partial(streams.extend_live_stream, operation, "A")
    engine.schedule(1000, extend)
    engine.run()
    assert qpu.emissions == [(7, 7, 0, False), (7, 8, 1000, False)]
    assert qpu.emission_owner_ids == [7, 7]


@pytest.mark.parametrize(
    ("emits_detector_data", "is_held"), [(False, True), (True, False)]
)
def test_an_idle_group_continues_the_stream_its_last_segment_left(
    emits_detector_data: bool, is_held: bool
) -> None:
    follower = _operation(
        2, patches=("A", "B"), emits_detector_data=emits_detector_data
    )
    program = _unprotected_group_program()
    operations = program.operations + (follower,)
    program = dataclasses.replace(program, operations=operations)
    engine = engine_module.Engine()
    qpu = _RoundLogQpu(engine, idle_group=(2, ("A", "B")))
    windows = _Windows()
    streams = _streams(
        program, regions=(), engine=engine, qpu=qpu, window_manager=windows
    )
    streams.begin(program.operations[0])
    streams.begin(follower)

    assert streams.extend_live_stream(follower, "A") is is_held


def test_a_sealed_stream_is_continued_by_no_idle_patch() -> None:
    program = _unprotected_group_program()
    operation = program.operations[0]
    engine = engine_module.Engine()
    qpu = _RoundLogQpu(engine, idle_group=(1, ("A", "B")))
    windows = _Windows()
    streams = _streams(
        program, regions=(), engine=engine, qpu=qpu, window_manager=windows
    )
    streams.begin(operation)
    streams.seal_finished_streams()

    assert windows.seals == [(7, 6)]
    assert streams.extend_live_stream(operation, "A") is False
    assert qpu.emissions == []


def test_a_partial_idle_group_cannot_advance_its_joint_physical_history() -> (
    None
):
    program = _unprotected_group_program()
    operation = program.operations[0]
    engine = engine_module.Engine()
    qpu = _RoundLogQpu(engine, idle_group=(1, ("A",)))
    windows = _Windows()
    streams = _streams(
        program,
        regions=(),
        engine=engine,
        qpu=qpu,
        window_manager=windows,
    )
    streams.begin(operation)
    with pytest.raises(
        RuntimeError, match="cannot extend a partially idle patch group"
    ):
        streams.extend_live_stream(operation, "A")
    assert qpu.emissions == []


def _operation(
    operation_id: int, *, patches, blocked_by=None, emits_detector_data=True
):
    """One executable operation on the given patches."""
    return program_records.Operation(
        id=operation_id,
        name=f"op{operation_id}",
        qubits=(),
        clifford=True,
        circuit=None,
        consumes_magic_state=False,
        patches=tuple(patches),
        blocked_by=blocked_by,
        emits_detector_data=emits_detector_data,
    )


def _streams(
    program, *, regions, engine=None, qpu=None, window_manager=None, patches=()
):
    """A FeedbackStreams over the program, already loaded."""
    if not patches:
        patches = _owner_patch_records(program)
    resolved_operations = []
    for operation in program.operations:
        resolved = _resolved_operation(operation.id)
        resolved_operations.append(resolved)
    streams = feedback_streams.FeedbackStreams(
        engine,
        regions=regions,
        resolved_operations=tuple(resolved_operations),
        resolved_patches=patches,
    )
    if qpu is not None:
        streams.qpu = qpu
    if window_manager is not None:
        streams.windows = window_manager
    streams.load(program)
    return streams


def _resolved_operation(operation_id: int):
    """The planning facts the table reads for one operation."""
    geometry = program_records.ResolvedCodeGeometry(
        code_name="rotated surface code (d=3)",
        distance=3,
        commit_round_count=3,
        buffer_round_count=3,
        one_patch_spatial_node_count=9,
    )
    return program_records.ResolvedOperationPlanning(
        operation_id=operation_id,
        code_geometry=geometry,
        round_count=6,
        round_ticks=1000,
        spatial_node_count=9,
    )


def _region(stream_id: int, start: int, end: int):
    return program_records.ProtectedRegion(
        stream_id=stream_id,
        start_operation_id=start,
        end_operation_id=end,
    )


def test_a_region_whose_stream_no_dynamic_stream_owns_is_refused():
    first = _operation(1, patches=("p0",))
    program = program_records.ExecutionProgram(operations=(first,))
    regions = (_region(7, 1, 1),)

    with pytest.raises(ValueError) as refusal:
        _streams(program, regions=regions)

    assert "protected stream 7 is none of the program's dynamic streams" in (
        str(refusal.value)
    )


def test_an_endpoint_that_omits_its_owner_patch_is_refused() -> None:
    """The owning stream defines the protected footprint."""
    owner = _operation(7, patches=("p1",))
    first = _operation(1, patches=("p0",))
    program = program_records.ExecutionProgram(
        operations=(first,), dynamic_streams=(owner,)
    )
    regions = (_region(7, 1, 1),)

    with pytest.raises(ValueError) as refusal:
        _streams(program, regions=regions)

    assert (
        "protected stream 7's start operation does not hold every patch "
        "of the stream ('p1',)"
    ) in str(refusal.value)


def test_two_regions_on_one_stream_are_refused():
    owner = _operation(7, patches=("p0",))
    first = _operation(1, patches=("p0",))
    program = program_records.ExecutionProgram(
        operations=(first,), dynamic_streams=(owner,)
    )
    two_on_one_stream = (
        _region(7, 1, 1),
        _region(7, 1, 1),
    )

    with pytest.raises(ValueError) as refusal:
        _streams(program, regions=two_on_one_stream)

    assert "protected stream 7 has two protected regions" in str(refusal.value)


def test_an_endpoint_that_is_no_operation_is_refused():
    owner = _operation(7, patches=("p0",))
    first = _operation(1, patches=("p0",))
    program = program_records.ExecutionProgram(
        operations=(first,), dynamic_streams=(owner,)
    )
    regions = (_region(7, 1, 99),)

    with pytest.raises(ValueError) as refusal:
        _streams(program, regions=regions)

    assert (
        "protected stream 7's end operation is not an operation of the program"
    ) in str(refusal.value)


def test_an_endpoint_that_does_not_hold_the_regions_patch_is_refused():
    owner = _operation(7, patches=("p0",))
    first = _operation(1, patches=("p0",))
    other_patch = _operation(2, patches=("p1",))
    program = program_records.ExecutionProgram(
        operations=(first, other_patch), dynamic_streams=(owner,)
    )
    regions = (_region(7, 1, 2),)

    with pytest.raises(ValueError) as refusal:
        _streams(program, regions=regions)

    assert (
        "protected stream 7's end operation does not hold every patch"
    ) in str(refusal.value)


def test_a_dynamic_stream_that_feeds_a_protected_patch_is_refused():
    """A feedback source that is not itself executed cannot be waited on."""
    owner = _operation(7, patches=("p0",))
    blocked = _operation(1, patches=("p0",), blocked_by=7)
    program = program_records.ExecutionProgram(
        operations=(blocked,), dynamic_streams=(owner,)
    )
    regions = (_region(7, 1, 1),)

    with pytest.raises(ValueError) as refusal:
        _streams(program, regions=regions)

    sentence = str(refusal.value)
    assert "external source 7 from dynamic_streams" in sentence
    assert "protected streams (7,)" in sentence


def test_a_decode_operation_that_feeds_a_protected_patch_is_refused():
    owner = _operation(7, patches=("p0",))
    decode_only = _operation(5, patches=("p0",))
    blocked = _operation(1, patches=("p0",), blocked_by=5)
    program = program_records.ExecutionProgram(
        operations=(blocked,),
        decode_operations=(decode_only,),
        dynamic_streams=(owner,),
    )
    regions = (_region(7, 1, 1),)

    with pytest.raises(ValueError) as refusal:
        _streams(program, regions=regions)

    assert "external source 5 from decode_ops" in str(refusal.value)


def test_a_well_formed_region_is_indexed_at_both_of_its_endpoints():
    owner = _operation(7, patches=("p0",))
    first = _operation(1, patches=("p0",))
    second = _operation(2, patches=("p0",))
    program = program_records.ExecutionProgram(
        operations=(first, second), dynamic_streams=(owner,)
    )
    region = _region(7, 1, 2)

    streams = _streams(program, regions=(region,))

    assert streams.table.regions_starting_at(1) == [region]
    assert streams.table.regions_ending_at(2) == [region]
    assert streams.table.is_protected(7) is True
    assert streams.table.owner_of(7) is owner


def test_every_cadence_change_of_a_protected_region_reaches_the_runtime():
    """The boundary and the seal each retry the operations held for them."""
    owner = _operation(7, patches=("p0",))
    # the region's own stream emits the patch's rounds, so its endpoints
    # emit none of their own
    first = _operation(1, patches=("p0",), emits_detector_data=False)
    second = _operation(2, patches=("p0",), emits_detector_data=False)
    program = program_records.ExecutionProgram(
        operations=(first, second), dynamic_streams=(owner,)
    )
    region = _region(7, 1, 2)
    engine = engine_module.Engine()
    patch = _resolved_patch("p0")
    qpu = _Qpu(engine)
    windows = _Windows()
    streams = _streams(
        program,
        regions=(region,),
        engine=engine,
        qpu=qpu,
        window_manager=windows,
        patches=(patch,),
    )
    runtime = _Runtime()
    streams.runtime = runtime

    streams.begin(first)
    close = functools.partial(streams.request_closes, second)
    engine.schedule(patch.round_ticks, close)
    engine.run()

    assert runtime.retries == 2


def test_region_release_waits_for_other_same_tick_protected_rounds() -> None:
    """Keep the crossing regions together to expose their release ordering."""
    first_begin = _operation(1, patches=(0,), emits_detector_data=False)
    first_end = _operation(2, patches=(0,), emits_detector_data=False)
    second_begin = _operation(3, patches=(1,), emits_detector_data=False)
    joint = _operation(4, patches=(0, 1), emits_detector_data=False)
    third_end = _operation(5, patches=(0,), emits_detector_data=False)
    first_owner = _operation(100, patches=(0,))
    second_owner = _operation(200, patches=(1,))
    third_owner = _operation(300, patches=(0,))
    regions = (
        _region(100, 1, 2),
        _region(200, 3, 4),
        _region(300, 4, 5),
    )
    program = program_records.ExecutionProgram(
        operations=(first_begin, first_end, second_begin, joint, third_end),
        dynamic_streams=(first_owner, second_owner, third_owner),
        protected_regions=regions,
    )
    engine = engine_module.Engine()
    qpu = _RoundLogQpu(engine)
    windows = _Windows()
    first_patch = _resolved_patch(0)
    second_patch = _resolved_patch(1)
    patches = (first_patch, second_patch)
    streams = _streams(
        program,
        regions=regions,
        engine=engine,
        qpu=qpu,
        window_manager=windows,
        patches=patches,
    )
    runtime = _CrossingRegionRuntime(engine, streams, joint)
    streams.runtime = runtime
    streams.begin(first_begin)
    streams.begin(second_begin)
    close_first = functools.partial(streams.request_closes, first_end)
    close_third = functools.partial(streams.request_closes, third_end)
    engine.schedule(1000, close_first)
    engine.schedule(4000, close_third)
    engine.run()

    assert runtime.started_ticks == [2000]
    assert qpu.emissions == [
        (100, 1, 1000, True),
        (200, 1, 1000, False),
        (200, 2, 2000, True),
        (300, 1, 3000, False),
        (300, 2, 4000, True),
    ]


def test_the_empty_row_answers_every_call_the_real_one_does():
    """A run with no streams must not tell a caller which row it holds."""
    real_names = _public_names(feedback_streams.FeedbackStreams)
    empty_names = _public_names(feedback_streams.NoFeedbackStreams)

    assert real_names <= empty_names


def test_the_empty_row_holds_no_operation_and_seals_nothing():
    empty = feedback_streams.NoFeedbackStreams()
    operation = _operation(1, patches=("p0",))

    empty.load(None)
    empty.begin(operation)
    empty.seal_finished_streams()

    assert empty.binding_for(1) is None
    assert empty.blocks_start(operation) is False
    assert empty.is_live_protected_patch("p0") is False
    assert empty.extend_live_stream(operation, "p0") is False


def _resolved_patch(patch_identity):
    """The cadence facts the protected region reads for one patch."""
    geometry = program_records.ResolvedCodeGeometry(
        code_name="rotated surface code (d=3)",
        distance=3,
        commit_round_count=3,
        buffer_round_count=3,
        one_patch_spatial_node_count=9,
    )
    return program_records.ResolvedPatchPlanning(
        patch_identity=patch_identity,
        code_geometry=geometry,
        round_ticks=1000,
        spatial_node_count=9,
    )


class _Runtime:
    """The runtime as the streams reach it: it counts every retry."""

    def __init__(self) -> None:
        self.retries = 0

    def retry_ready_operations(self) -> None:
        """One cadence change."""
        self.retries += 1


class _Qpu:
    """The protected-round QPU, with opaque stream and patch identities."""

    def __init__(self, engine: engine_module.Engine) -> None:
        self.engine = engine

    def next_boundary(self) -> int:
        """The cycle edge at or after now, on the fixture's 1000-tick cycle."""
        whole_cycles = -(-self.engine.now // 1000)
        return whole_cycles * 1000

    def are_patches_idle(self, operation_id: Any, patches: tuple) -> bool:
        """The protected fixture owns no ordinary idle group."""
        del operation_id
        del patches
        return False

    def emit_idle_stream_round(
        self,
        operation: program_records.Operation,
        stream_id: Any,
        stream_round: int,
        *,
        is_final: bool,
    ) -> None:
        """One protected round."""

    def validate_stream_length(
        self, stream_operation: program_records.Operation, round_count: int
    ) -> None:
        """Every length is the one executed."""


class _Windows:
    """The window side the seal reaches."""

    def __init__(self) -> None:
        self.seals: list[tuple] = []

    def bind_stream_operation(
        self, operation_id: Any, stream_id: Any, stream_offset: int
    ) -> None:
        """Accept the controller's public binding declaration."""
        del operation_id
        del stream_id
        del stream_offset

    def has_dynamic_stream(self, stream_id: Any) -> bool:
        """The fixture declares stream seven."""
        return stream_id == 7

    def seal_stream(self, stream_id, stream_round_count: int) -> None:
        """One sealed stream."""
        self.seals.append((stream_id, stream_round_count))


class _RoundLogQpu:
    """Observe emissions with opaque stream and patch identities."""

    def __init__(
        self, engine: engine_module.Engine, idle_group: Optional[tuple] = None
    ) -> None:
        self.engine = engine
        self.emissions: list[tuple] = []
        self.emission_owner_ids: list[int] = []
        self.idle_group = idle_group
        self.attested_lengths: list[tuple] = []
        # a seal at this length differs from the rounds executed
        self.refused_length: Optional[int] = None

    def next_boundary(self) -> int:
        """The cycle edge at or after now, on the fixture's 1000-tick cycle."""
        whole_cycles = -(-self.engine.now // 1000)
        return whole_cycles * 1000

    def are_patches_idle(self, operation_id: Any, patches: tuple) -> bool:
        """Only the explicitly declared group is idle after that operation."""
        return (operation_id, patches) == self.idle_group

    def emit_idle_stream_round(
        self,
        operation: program_records.Operation,
        stream_id: Any,
        stream_round: int,
        *,
        is_final: bool,
    ) -> None:
        """The physical round exposes its ordering through the QPU port."""
        self.emission_owner_ids.append(operation.id)
        emission = (stream_id, stream_round, self.engine.now, is_final)
        self.emissions.append(emission)

    def validate_stream_length(
        self, stream_operation: program_records.Operation, round_count: int
    ) -> None:
        """Record the attestation; refuse the one length the test names."""
        self.attested_lengths.append((stream_operation.id, round_count))
        if round_count == self.refused_length:
            raise RuntimeError("sealed length differs from executed rounds")


class _CrossingRegionRuntime:
    """Start the joint endpoint when both protected patches permit it."""

    def __init__(
        self,
        engine: engine_module.Engine,
        streams: feedback_streams.FeedbackStreams,
        joint: program_records.Operation,
    ) -> None:
        self.engine = engine
        self.streams = streams
        self.joint = joint
        self.started_ticks: list[int] = []

    def retry_ready_operations(self) -> None:
        """A released patch can start a region only at the shared boundary."""
        if self.started_ticks:
            return
        if self.streams.blocks_start(self.joint):
            return
        self.started_ticks.append(self.engine.now)
        self.streams.begin(self.joint)
        self.streams.request_closes(self.joint)


def _public_names(row) -> set:
    """Every method a caller may name on a stream row."""
    names = set()
    for name in dir(row):
        if not name.startswith("_"):
            names.add(name)
    return names


def _owner_patch_records(program: program_records.ExecutionProgram) -> tuple:
    records_by_patch = {}
    for owner in program.dynamic_streams:
        for patch in owner.patches:
            records_by_patch[patch] = _resolved_patch(patch)
    records = records_by_patch.values()
    return tuple(records)


def _group_program() -> program_records.ExecutionProgram:
    owner = _operation(7, patches=("A", "B"))
    first = _operation(1, patches=("A", "B"), emits_detector_data=False)
    final = _operation(2, patches=("A", "B"), emits_detector_data=False)
    region = _region(7, 1, 2)
    return program_records.ExecutionProgram(
        operations=(first, final),
        dynamic_streams=(owner,),
        protected_regions=(region,),
    )


def _overlapping_group_program() -> program_records.ExecutionProgram:
    first = _group_program()
    second_owner = _operation(8, patches=("B", "C"))
    second_begin = _operation(3, patches=("B", "C"), emits_detector_data=False)
    second_end = _operation(4, patches=("B", "C"), emits_detector_data=False)
    second_region = _region(8, 3, 4)
    operations = first.operations + (second_begin, second_end)
    owners = first.dynamic_streams + (second_owner,)
    regions = first.protected_regions + (second_region,)
    return program_records.ExecutionProgram(
        operations=operations, dynamic_streams=owners, protected_regions=regions
    )


def _unprotected_group_program() -> program_records.ExecutionProgram:
    owner = _operation(7, patches=("A", "B"))
    operation = _operation(1, patches=("A", "B"))
    operation = dataclasses.replace(operation, stream_id=7, stream_offset=0)
    return program_records.ExecutionProgram(
        operations=(operation,), dynamic_streams=(owner,)
    )
