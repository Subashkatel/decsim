"""A trace source fires to its listeners in connection order, or to none.

The shape is ns-3's TracedCallback (point-to-point-net-device.h:309,
.cc:530, on disk under tmp/resources/l5_buffers): a source with no sink
fires into an empty list.
"""

import decsim.observe.trace_source as trace_source


def test_a_source_with_no_listener_fires_and_nothing_happens():
    source = trace_source.TraceSource()
    source.fire(1, "round")
    assert not source.has_listeners


def test_every_listener_hears_every_fire_once_in_connection_order():
    source = trace_source.TraceSource()
    heard = []

    def first(tick):
        heard.append(("first", tick))

    def second(tick):
        heard.append(("second", tick))

    source.connect(first)
    source.connect(second)
    source.fire(3)
    source.fire(7)
    assert heard == [("first", 3), ("second", 3), ("first", 7), ("second", 7)]
    assert source.has_listeners


def test_a_run_with_every_listener_connected_has_the_same_ticks_as_a_bare_one():
    import dataclasses
    import pathlib

    import decsim.machine as machine_module
    from experiments.experiment_config import load_experiment

    here = pathlib.Path(__file__)
    repository = here.resolve()
    repository = repository.parents[2]
    config_path = repository / "experiments/configs/weak_decoder_baseline.yaml"
    config = load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.003, distance=3, round_period_us=1.0
    )
    bare = machine_module.Machine.build(settings, 0)
    bare.run()
    every_listener = dataclasses.replace(
        settings.observation,
        round_store_occupancy=True,
        backlog_trace=True,
        decoder_utilization=True,
        decoder_memory_occupancy=True,
    )
    heard_settings = dataclasses.replace(settings, observation=every_listener)
    heard = machine_module.Machine.build(heard_settings, 0)
    heard.run()
    bare_frame = bare.pauli_frame.snapshot()
    heard_frame = heard.pauli_frame.snapshot()
    assert heard_frame.records == bare_frame.records
    assert heard.observation.log.lines == bare.observation.log.lines
    assert heard.observation.round_store_occupancy.arrivals == 30
    assert heard.observation.decode_backlog.peak > 0
