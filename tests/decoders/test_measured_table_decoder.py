"""The measured_table row against its two referents on the same region.

The time is the measured line, intercept plus slope times iterations,
at the iteration count relay-bp's RelayDecoderF32 reports for the
region (decode_detailed's `iterations`), with the numbers of the cell
measurements.py ships. The answer is the relay_bp row's under the same
run seed. The regions are Stim's rotated memory circuits of 3d rounds,
the measured regions' own shape: at d = 5 and 15 rounds, 360 detectors.
"""

import pytest

import decsim.config as config
import decsim.decoders.measured_table.decoder as measured_table
import decsim.decoders.measured_table.measurements as measurements
import decsim.decoders.relay_belief_propagation.decoder as relay
import decsim.decoders.strong_backend as strong_backend
import decsim.detector_error_model.basis_split as basis_split
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.engine as engine_module
import decsim.records.seeds as seed_records
import decsim.seeding as seeding
from tests.decoders import windows

PHYSICAL = fault_models.FaultRepresentation.PHYSICAL
REQUIREMENT = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED


def _decode_ticks(table, ticket) -> int:
    """The ticks of the one step a measured decode is: the decode."""
    steps = table.steps(ticket)
    (decode,) = steps
    return decode.ticks


def _bind_seed(component) -> None:
    """Bind run seed 7 to the component at one path for both rows."""
    segment = seed_records.RunSeedPathSegment("field", "strong_decoder")
    root = ((segment,), component)
    seeding.bind_run_seed(7, [root])


def _cells_on(partition: str) -> list:
    cells = []
    for cell in measurements.RELAY_BP_TIMES:
        if cell.partition == partition:
            cells.append(cell)
    return cells


def test_the_time_is_the_measured_line_at_relay_bps_own_iterations():
    """a100, whole, 360 detectors: 78.957 us plus 9.762 us an iteration."""
    pytest.importorskip("relay_bp")
    circuit = windows.memory_circuit(5, 15, 0.003)
    model = windows.whole_circuit_window(circuit, 15, REQUIREMENT)
    detection_events, _ = windows.sampled_shots(circuit, 1, 11)
    job = windows.job_for(model, detection_events[0])
    settings = measured_table.MeasuredTableSettings("a100", "whole")
    table = measured_table.MeasuredTable(settings)
    physical = model.require_faults(PHYSICAL)
    compiled = table.decoder.window_decoder._compiled_model(physical)
    syndrome = windows.row_syndrome(model, detection_events[0])
    detailed = compiled.backend.decode_detailed(syndrome)
    ticket = table.submit(job, 0)
    microseconds = 78.957 + 9.762 * detailed.iterations
    assert physical.check.shape[0] == 360
    assert _decode_ticks(table, ticket) == config.microseconds_to_ticks(
        microseconds
    )


@pytest.mark.parametrize(
    "partition, intercept_microseconds, microseconds_per_iteration",
    [("3g.40gb", 72.377, 9.527), ("1g.10gb", 68.534, 15.951)],
)
def test_a_multi_instance_gpu_slice_is_priced_by_its_own_measured_line(
    partition, intercept_microseconds, microseconds_per_iteration
):
    """a100, one slice, 360 detectors: the slice's line, not the card's."""
    pytest.importorskip("relay_bp")
    circuit = windows.memory_circuit(5, 15, 0.003)
    model = windows.whole_circuit_window(circuit, 15, REQUIREMENT)
    detection_events, _ = windows.sampled_shots(circuit, 1, 11)
    job = windows.job_for(model, detection_events[0])
    section = {"device": "a100", "partition": partition}
    settings = measured_table.MeasuredTableSettings.from_yaml(
        section, None, "strong_decoder"
    )
    table = measured_table.MeasuredTable(settings)
    ticket = table.submit(job, 0)
    result = table.result(ticket)
    per_iteration = microseconds_per_iteration * result.iterations
    microseconds = intercept_microseconds + per_iteration
    assert _decode_ticks(table, ticket) == config.microseconds_to_ticks(
        microseconds
    )


def test_the_answer_is_the_relay_bp_rows_under_the_same_run_seed():
    pytest.importorskip("relay_bp")
    circuit = windows.memory_circuit(5, 15, 0.003)
    model = windows.whole_circuit_window(circuit, 15, REQUIREMENT)
    detection_events, _ = windows.sampled_shots(circuit, 1, 11)
    job = windows.job_for(model, detection_events[0])
    row = measured_table.MeasuredTableDecoder()
    reference = relay.RelayBeliefPropagationDecoder()
    _bind_seed(row)
    _bind_seed(reference)
    answer = row.decode(job)
    expected = reference.decode(job)
    assert answer.correction.tolist() == expected.correction.tolist()
    assert answer.logical_observables == expected.logical_observables
    assert answer.iterations == expected.iterations


def test_a_region_is_priced_by_the_measured_region_nearest_in_size():
    """40 rounds at d = 5 hold 960 detectors, nearer 1,008 than 360."""
    pytest.importorskip("relay_bp")
    circuit = windows.memory_circuit(5, 40, 0.003)
    model = windows.whole_circuit_window(circuit, 40, REQUIREMENT)
    detection_events, _ = windows.sampled_shots(circuit, 1, 11)
    job = windows.job_for(model, detection_events[0])
    settings = measured_table.MeasuredTableSettings("gh200", "whole")
    table = measured_table.MeasuredTable(settings)
    ticket = table.submit(job, 0)
    result = table.result(ticket)
    microseconds = 95.243 + 11.044 * result.iterations
    physical = model.require_faults(PHYSICAL)
    assert physical.check.shape[0] == 960
    assert _decode_ticks(table, ticket) == config.microseconds_to_ticks(
        microseconds
    )


def test_a_region_with_no_faults_costs_the_cells_fastest_decode():
    """The line reads 78.957 us at none; no decode took under 85.872."""
    pytest.importorskip("relay_bp")
    circuit = windows.memory_circuit(5, 15, 0.0)
    model = windows.whole_circuit_window(circuit, 15, REQUIREMENT)
    detection_events, _ = windows.sampled_shots(circuit, 1, 11)
    job = windows.job_for(model, detection_events[0])
    settings = measured_table.MeasuredTableSettings("a100", "whole")
    table = measured_table.MeasuredTable(settings)
    ticket = table.submit(job, 0)
    assert _decode_ticks(table, ticket) == config.microseconds_to_ticks(85.872)


@pytest.mark.parametrize("iterations", [1, 3])
def test_a_decode_is_never_priced_under_its_cells_fastest_decode(iterations):
    """1g.10gb, d = 13: the line reads 522 us at 3 iterations, below 0 at 1.

    The fastest of the cell's 2,000 decodes took 1,405.683 us (the
    minimum of its time_ns column), so both price that.
    """
    slice_cells = _cells_on("1g.10gb")
    largest_region = slice_cells[-1]
    microseconds = largest_region.decode_microseconds(iterations)
    assert largest_region.detectors == 6552
    assert microseconds == 1405.683


def test_the_device_runs_one_decode_at_a_time():
    settings = measured_table.MeasuredTableSettings("a100", "mps")
    table = measured_table.MeasuredTable(settings)
    assert table.capacities() == {strong_backend.DISPATCHER: 1}


def test_a_device_and_partition_never_measured_are_refused():
    section = {"device": "gh200", "partition": "mps"}
    with pytest.raises(ValueError) as refusal:
        measured_table.MeasuredTableSettings.from_yaml(
            section, None, "strong_decoder"
        )
    assert str(refusal.value) == (
        "strong_decoder.device 'gh200' with partition 'mps' and bases "
        "'together' has no measurement in measured_table; the measured "
        "ones are [('a100', 'whole', 'together'), "
        "('a100', 'whole', 'apart'), "
        "('a100', 'mps', 'together'), ('a100', '3g.40gb', 'together'), "
        "('a100', '1g.10gb', 'together'), ('gh200', 'whole', 'together'), "
        "('gh200', 'whole', 'apart')]"
    )


def test_a_bases_row_off_the_table_is_refused():
    section = {"device": "gh200", "bases": "xz"}
    with pytest.raises(ValueError) as refusal:
        measured_table.MeasuredTableSettings.from_yaml(
            section, None, "strong_decoder"
        )
    assert str(refusal.value) == (
        "strong_decoder.bases 'xz' is not a row of its table; the rows are "
        "['apart', 'together']"
    )


@pytest.mark.parametrize(
    "device, x_line, z_line",
    [
        ("gh200", (54.440, 6.678), (55.712, 6.664)),
        ("a100", (59.437, 7.752), (59.673, 7.346)),
    ],
)
def test_a_region_decoded_apart_is_its_two_parts_on_their_own_lines(
    device, x_line, z_line
):
    """At d = 5 and 15 rounds: the X part's 168 detectors, then the Z's 192.

    One dispatcher runs the two in series, so the region ends when the
    Z part's line at its own iterations has run after the X part's.
    """
    pytest.importorskip("relay_bp")
    circuit = windows.memory_circuit(5, 15, 0.003)
    requirement = REQUIREMENT.joined(fault_models.DETECTOR_BASES_REQUIRED)
    model = windows.whole_circuit_window(circuit, 15, requirement)
    detection_events, _ = windows.sampled_shots(circuit, 1, 11)
    job = windows.job_for(model, detection_events[0])
    settings = measured_table.MeasuredTableSettings(device, "whole", "apart")
    row = measured_table.MeasuredTableDecoder(settings)
    reference = relay.RelayBeliefPropagationDecoder()
    _bind_seed(row)
    _bind_seed(reference)
    parts = basis_split.split_by_basis(model)
    x_job = windows.job_for(parts["X"], detection_events[0])
    z_job = windows.job_for(parts["Z"], detection_events[0])
    x_answer = reference.decode(x_job)
    z_answer = reference.decode(z_job)
    x_intercept, x_slope = x_line
    z_intercept, z_slope = z_line
    x_microseconds = x_intercept + x_slope * x_answer.iterations
    z_microseconds = z_intercept + z_slope * z_answer.iterations
    x_ticks = config.microseconds_to_ticks(x_microseconds)
    z_ticks = config.microseconds_to_ticks(z_microseconds)
    engine = engine_module.Engine()
    ended = []

    def record_end(result) -> None:
        del result
        ended.append(engine.now)

    row.start(job, engine, record_end)
    engine.run()
    assert ended == [x_ticks + z_ticks]
