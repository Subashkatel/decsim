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
"""

import pytest

import decsim.controller.feedback_streams as feedback_streams
import decsim.records.program as program_records


def _operation(operation_id: int, *, patches, blocked_by=None):
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
    )


def _streams(program, *, regions):
    """A FeedbackStreams over the program, already loaded."""
    resolved_operations = []
    for operation in program.operations:
        resolved = _resolved_operation(operation.id)
        resolved_operations.append(resolved)
    streams = feedback_streams.FeedbackStreams(
        engine=None,
        qpu=None,
        window_manager=None,
        regions=regions,
        resolved_operations=tuple(resolved_operations),
        resolved_patches=(),
        retry_ready_operations=None,
    )
    streams.load(program)
    return streams


def _resolved_operation(operation_id: int):
    """The planning facts the table reads for one operation."""
    geometry = program_records.ResolvedCodeGeometry(
        code_name="rotated surface code (d=3)",
        distance=3,
        commit_round_count=3,
        buffer_round_count=3,
        minimum_leading_buffer_round_count=3,
        minimum_trailing_buffer_round_count=3,
        one_patch_spatial_node_count=9,
        window_floor_justification=None,
    )
    return program_records.ResolvedOperationPlanning(
        operation_id=operation_id,
        code_geometry=geometry,
        round_count=6,
        round_ticks=1000,
        spatial_node_count=9,
    )


def _region(stream_id: int, patch_id, start: int, end: int):
    return program_records.ProtectedRegion(
        patch_id=patch_id,
        stream_id=stream_id,
        start_operation_id=start,
        end_operation_id=end,
    )


def test_a_region_whose_stream_no_dynamic_stream_owns_is_refused():
    first = _operation(1, patches=("p0",))
    program = program_records.ExecutionProgram(operations=(first,))
    regions = (_region(7, "p0", 1, 1),)

    with pytest.raises(ValueError) as refusal:
        _streams(program, regions=regions)

    assert "protected stream 7 owner/patch mismatch" in str(refusal.value)


def test_an_owner_that_holds_another_patch_is_refused():
    """The region's patch is the one its owning stream runs on."""
    owner = _operation(7, patches=("p1",))
    first = _operation(1, patches=("p0",))
    program = program_records.ExecutionProgram(
        operations=(first,), dynamic_streams=(owner,)
    )
    regions = (_region(7, "p0", 1, 1),)

    with pytest.raises(ValueError) as refusal:
        _streams(program, regions=regions)

    assert "protected stream 7 owner/patch mismatch" in str(refusal.value)


def test_two_regions_on_one_stream_are_refused():
    owner = _operation(7, patches=("p0",))
    first = _operation(1, patches=("p0",))
    program = program_records.ExecutionProgram(
        operations=(first,), dynamic_streams=(owner,)
    )
    two_on_one_stream = (
        _region(7, "p0", 1, 1),
        _region(7, "p0", 1, 1),
    )

    with pytest.raises(ValueError) as refusal:
        _streams(program, regions=two_on_one_stream)

    assert "duplicate protected stream 7" in str(refusal.value)


def test_an_endpoint_that_is_no_operation_is_refused():
    owner = _operation(7, patches=("p0",))
    first = _operation(1, patches=("p0",))
    program = program_records.ExecutionProgram(
        operations=(first,), dynamic_streams=(owner,)
    )
    regions = (_region(7, "p0", 1, 99),)

    with pytest.raises(ValueError) as refusal:
        _streams(program, regions=regions)

    assert "protected stream 7 invalid end" in str(refusal.value)


def test_an_endpoint_that_does_not_hold_the_regions_patch_is_refused():
    owner = _operation(7, patches=("p0",))
    first = _operation(1, patches=("p0",))
    other_patch = _operation(2, patches=("p1",))
    program = program_records.ExecutionProgram(
        operations=(first, other_patch), dynamic_streams=(owner,)
    )
    regions = (_region(7, "p0", 1, 2),)

    with pytest.raises(ValueError) as refusal:
        _streams(program, regions=regions)

    assert "protected stream 7 invalid end" in str(refusal.value)


def test_a_dynamic_stream_that_feeds_a_protected_patch_is_refused():
    """A feedback source that is not itself executed cannot be waited on."""
    owner = _operation(7, patches=("p0",))
    blocked = _operation(1, patches=("p0",), blocked_by=7)
    program = program_records.ExecutionProgram(
        operations=(blocked,), dynamic_streams=(owner,)
    )
    regions = (_region(7, "p0", 1, 1),)

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
    regions = (_region(7, "p0", 1, 1),)

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
    region = _region(7, "p0", 1, 2)

    streams = _streams(program, regions=(region,))

    assert streams.table.regions_starting_at(1) == [region]
    assert streams.table.regions_ending_at(2) == [region]
    assert streams.table.is_protected(7) is True
    assert streams.table.owner_of(7) is owner


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


def _public_names(row) -> set:
    """Every method a caller may name on a stream row."""
    names = set()
    for name in dir(row):
        if not name.startswith("_"):
            names.add(name)
    return names
