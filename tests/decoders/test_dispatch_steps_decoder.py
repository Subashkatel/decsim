"""The dispatch_steps row against the traced kernels and the round trips.

The decode step is the traced kernel line at the iteration count
relay-bp's RelayDecoderF32 reports for the region, the answer is the
relay_bp row's, and the steps around it follow the two CUDA-Q
dispatcher paths (cuda-quantum dispatch_kernel.cu v0.15.2, host_api.md
lines 1065-1113). The region is Stim's rotated memory circuit at d = 5
and 15 rounds, 360 detectors, the traced region's own shape.
"""

import pytest

import decsim.config as config
import decsim.decoders.dispatch_steps.decoder as dispatch_steps
import decsim.decoders.dispatch_steps.measurements as measurements
import decsim.decoders.relay_belief_propagation.decoder as relay
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.engine as engine_module
import decsim.links.channel as channel_module
import decsim.links.link_profiles as link_profiles
import decsim.records.seeds as seed_records
import decsim.seeding as seeding
from tests.decoders import windows

REQUIREMENT = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
DISPATCHER = dispatch_steps.DISPATCHER
WORKER = dispatch_steps.WORKER
ECHO = dispatch_steps.ECHO
ROCE_V2_GPU_CARD = link_profiles.RoceV2GpuFabric.base_card()
NVQLINK_GPU_CARD = link_profiles.NvqlinkGpuFabric.base_card()


def _bind_seed(component) -> None:
    """Bind run seed 7 to the component at one path for both rows."""
    segment = seed_records.RunSeedPathSegment("field", "strong_decoder")
    root = ((segment,), component)
    seeding.bind_run_seed(7, [root])


def _region_job():
    circuit = windows.memory_circuit(5, 15, 0.003)
    model = windows.whole_circuit_window(circuit, 15, REQUIREMENT)
    detection_events, _ = windows.sampled_shots(circuit, 1, 11)
    return windows.job_for(model, detection_events[0])


def _laid_out(steps) -> list:
    """Each step's name, resource and the card it is priced on."""
    layout = []
    for step in steps:
        layout.append((step.name, step.resource, step.priced_on))
    return layout


def test_a_device_path_decode_ends_after_its_fire_and_its_kernel_line():
    """gh200: fire 5.536 us, then 11.840 + 7.243 us an iteration.

    The kernel line is floored at the fastest traced kernel, 19.008 us.
    """
    pytest.importorskip("relay_bp")
    job = _region_job()
    row = dispatch_steps.DispatchStepsDecoder()
    reference = relay.RelayBeliefPropagationDecoder()
    _bind_seed(row)
    _bind_seed(reference)
    reference_answer = reference.decode(job)
    iterations = reference_answer.iterations
    kernel_line = 11.840 + 7.243 * iterations
    kernel_microseconds = max(kernel_line, 19.008)
    fire_ticks = config.microseconds_to_ticks(5.536)
    kernel_ticks = config.microseconds_to_ticks(kernel_microseconds)
    engine = engine_module.Engine()
    ended = []

    def record_end(result) -> None:
        del result
        ended.append(engine.now)

    row.start(job, engine, record_end)
    engine.run()
    assert ended == [fire_ticks + kernel_ticks]


def test_the_device_path_holds_the_dispatcher_for_every_step():
    """Notice, check, handle and respond are the echo's, at zero ticks."""
    pytest.importorskip("relay_bp")
    settings = dispatch_steps.DispatchStepsSettings("gh200", "device")
    backend = dispatch_steps.DispatchSteps(settings)
    job = _region_job()
    ticket = backend.submit(job, 0)
    steps = backend.steps(ticket)
    assert _laid_out(steps) == [
        ("notice", DISPATCHER, ECHO),
        ("check", DISPATCHER, ECHO),
        ("handle", DISPATCHER, ECHO),
        ("fire", DISPATCHER, None),
        ("decode", DISPATCHER, None),
        ("respond", DISPATCHER, ECHO),
    ]
    echo_steps = (steps[0], steps[1], steps[2], steps[5])
    assert [step.ticks for step in echo_steps] == [0, 0, 0, 0]


def test_the_host_path_hands_the_dispatcher_to_a_worker_at_launch():
    """The monitor notices; one worker launches, copies and decodes."""
    pytest.importorskip("relay_bp")
    settings = dispatch_steps.DispatchStepsSettings("a100", "host", 2)
    backend = dispatch_steps.DispatchSteps(settings)
    job = _region_job()
    ticket = backend.submit(job, 0)
    steps = backend.steps(ticket)
    assert backend.capacities() == {DISPATCHER: 1, WORKER: 2}
    assert _laid_out(steps) == [
        ("notice", DISPATCHER, ECHO),
        ("check", DISPATCHER, ECHO),
        ("launch", WORKER, None),
        ("copy_in", WORKER, None),
        ("decode", WORKER, None),
        ("copy_out", WORKER, None),
        ("respond", None, ECHO),
    ]


def _escalation_round_trip(profile, echo_bits: int) -> int:
    """The legs an echo of `echo_bits` crosses out and back, in ticks."""
    legs = (
        profile.weak_decoder_to_strong_decoder,
        profile.strong_buffer_to_strong_decoder,
        profile.strong_decoder_to_frame,
    )
    ticks = 0
    for leg in legs:
        ticks += leg.channel.propagation_latency_ticks
        ticks += _wire_ticks(leg.channel, echo_bits)
    return ticks


def _wire_ticks(channel, bits: int) -> int:
    """The bits' time on a bounded wire; an unbounded one takes none."""
    if channel.capacity is None:
        return 0
    return channel_module.serialization_ticks(bits, channel.capacity)


@pytest.mark.parametrize(
    "profile, echo_bits, echo_microseconds",
    [
        (ROCE_V2_GPU_CARD, 128, 4.5),
        (NVQLINK_GPU_CARD, 256, 3.839),
    ],
)
def test_an_echo_on_a_published_card_costs_its_measured_round_trip(
    profile, echo_bits, echo_microseconds
):
    """Backline 2609.09270 Table III, 4.5 us; NVQLink 2510.25213, 3.839 us.

    Each echoed a 16- and a 32-byte payload (2609.09270 line 1611,
    2510.25213 lines 402-403). The echo's own steps add no ticks, so the
    link legs with that payload's wire time are the measured round trip:
    nothing is counted twice.
    """
    pytest.importorskip("relay_bp")
    settings = dispatch_steps.DispatchStepsSettings("gh200", "device")
    backend = dispatch_steps.DispatchSteps(settings)
    job = _region_job()
    ticket = backend.submit(job, 0)
    steps = backend.steps(ticket)
    echo_steps = (steps[0], steps[1], steps[2], steps[5])
    echo_ticks = sum(step.ticks for step in echo_steps)
    round_trip = _escalation_round_trip(profile, echo_bits) + echo_ticks
    assert round_trip == config.microseconds_to_ticks(echo_microseconds)


def test_the_fire_is_the_graph_round_trip_less_the_echo_it_holds():
    """GH200, mapped 16-byte slot: graph 10.112 us, call (echo) 4.576 us."""
    card = measurements.CARDS[("gh200", "device")]
    assert card.launch_microseconds + 4.576 == pytest.approx(10.112)


def test_a_worker_count_of_true_is_refused_on_the_host_path():
    """A bool is refused though Python counts it an int."""
    fields = {"device": "gh200", "path": "host", "workers": True}
    with pytest.raises(ValueError) as refusal:
        dispatch_steps.DispatchStepsSettings(**fields)
    assert str(refusal.value) == (
        "workers must be a whole number of graph workers, at least 1 (got True)"
    )


def test_workers_on_the_device_path_are_refused():
    fields = {"device": "gh200", "path": "device", "workers": 4}
    with pytest.raises(ValueError) as refusal:
        dispatch_steps.DispatchStepsSettings(**fields)
    assert str(refusal.value) == (
        "workers is the host path's; the device path "
        "decodes on its one dispatcher"
    )


def test_a_record_with_no_measured_card_still_stops_at_build():
    """No card prices the a100's device path, so the build cannot."""
    settings = dispatch_steps.DispatchStepsSettings(
        device="a100", path="device"
    )

    with pytest.raises(KeyError):
        settings.build()
