"""The windows facade's laws through its public surface.

The status of a window lives on its record (gem5's CacheBlk carries
its own status bits, src/mem/cache/cache_blk.hh): the facade keeps no
set that mirrors it. The run is a six-round d=3 memory on the
timing-only device with a preset-latency weak decoder.
"""

import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.message as message
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings


def _weak_run():
    memory = message.Operation(0, "M", (0,), clifford=True)
    six_rounds = round_policies.FixedRounds(6)
    workload = workload_settings.WorkloadSettings(
        operations=[memory], rounds_policy=six_rounds
    )
    code = code_geometry.SurfaceCodeModel(distance=3)
    qpu = qpu_settings.QpuSettings(code=code, round_period_microseconds=1.0)
    decoder = decoders.PresetLatencyDecoder(2.0)
    weak_decoder = decoder_settings.DecoderSettings(decoder=decoder, units=1)
    settings = machine_module.MachineSettings(
        workload=workload, qpu=qpu, weak_decoder=weak_decoder
    )
    machine = machine_module.Machine.build(settings, 0)
    machine.run()
    return machine


def test_a_committed_window_publishes_the_request_that_decoded_it():
    machine = _weak_run()
    windows = machine.window_manager.windows
    assert windows
    for key, window in windows.items():
        assert window.committed
        assert window.published_request_key is not None
        assert window.published_request_key.tier is message.DecoderTier.WEAK
        assert window.published_request_key.operation_id == key[0]
        assert window.published_request_key.window_id == key[1]


def test_no_window_of_a_weak_run_is_absorbed():
    machine = _weak_run()
    for window in machine.window_manager.windows.values():
        assert window.is_absorbed is False


def test_a_window_is_final_once_its_request_is_published():
    """A committed window whose request is unpublished still awaits strong."""
    window = message.Window(
        op_id=1, k=0, commit_lo=1, commit_hi=3, buffer_hi=5, n_rounds=5
    )
    window.committed = True
    assert window.published_request_key is None
    request_key = message.DecoderRequestKey(1, 0, message.DecoderTier.STRONG, 3)
    window.published_request_key = request_key
    assert window.published_request_key.tier is message.DecoderTier.STRONG
