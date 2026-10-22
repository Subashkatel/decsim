"""A lattice-surgery run's decode waits, against SWIPER-SIM's own.

SWIPER-SIM (2412.05115, swiper/simulator.py DecodingSimulator) runs
toffoli and msd_15to1 at distance 7 with sliding windows, no
speculation, as many decoders as it asks for and every window decoded
in a fixed number of rounds; tools/swiper_reaction.py wrote the rounds
each conditioned instruction waited for a decode
(DeviceData.conditioned_decode_wait_times). decsim runs the same
schedules on SWIPER's machine: a round a microsecond, every hop free,
a unit for every window, the same fixed latency, SWIPER's sliding
regions, and SWIPER's release rule (windows/release_rules.py
DecodedAncestors). Its stall_ticks (records/results.py GateStall) is
SWIPER's wait.

The referent runs are the ones that start each instruction as soon as
it may, as decsim's controller does; SWIPER's own starts an instruction
on fresh patches as late as its successors allow, a device choice that
moves toffoli's last waits. SWIPER cuts a region at a merge's edge only
once a round from beyond it arrives (window_builder.py build_windows),
one round after decsim, which knows the merge when it binds; msd_15to1's
half-distance merges cut regions mid-stride, so there a wait is SWIPER's
or one round less.
"""

import json
import pathlib

import pytest

import decsim.config as config
import decsim.decoders.decoders as decoders
import decsim.decoders.settings as decoder_settings
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.presets.machines as machines
import decsim.qpu.settings as qpu_settings
import decsim.windows.release_rules as release_rules
import decsim.windows.schemes.swiper_sliding as swiper_sliding
import decsim.windows.settings as window_settings
import tests.declared_run as declared_run
import tests.frontends.test_patch_lives as patch_lives_test

HERE = pathlib.Path(__file__)
DATA = HERE.parents[1] / "data"
DISTANCE = 7
ROUND_MICROSECONDS = 1.0
# more units than windows ever wait at once, as SWIPER-SIM's pool grows
UNIT_COUNT = 200
LATENCY_ROUNDS = (1, 5, 10)


def swiper_run(name: str, latency_rounds: int) -> dict:
    """SWIPER-SIM's run at the latency, every instruction started ASAP."""
    path = DATA / f"swiper_{name}_d{DISTANCE}_reaction.json"
    text = path.read_text()
    reading = json.loads(text)
    for run in reading["runs"]:
        is_asap = run["start_rule"] == "asap"
        if is_asap and run["latency_rounds"] == latency_rounds:
            return run
    raise KeyError(latency_rounds)


def decsim_run(name: str, latency_rounds: int):
    """The schedule run by decsim on SWIPER's machine."""
    workload = patch_lives_test.seed_zero_workload(name)
    settings = workload_settings.WorkloadSettings.running(workload)
    scheme = swiper_sliding.SwiperSlidingScheme.Settings()
    rule = release_rules.DecodedAncestors.Settings()
    windows = window_settings.WindowSettings(scheme=scheme, release_rule=rule)
    qpu = qpu_settings.QpuSettings(
        distance=DISTANCE, round_period_microseconds=ROUND_MICROSECONDS
    )
    free_hops = dict.fromkeys(declared_run.DECLARED_HOP_MICROSECONDS, 0.0)
    fabric = declared_run.declared_fabric(**free_hops)
    latency = latency_rounds * ROUND_MICROSECONDS
    algorithm = decoders.PresetLatencyDecoder.Settings(latency)
    pool = decoder_settings.DecoderPoolSettings(
        algorithm=algorithm,
        engine=declared_run.DECLARED_ENGINE,
        unit_count=UNIT_COUNT,
    )
    base = machines.base(settings, windows, qpu, fabric=fabric)
    graph = machines.with_pool(base, pool)
    machine = machine_module.Machine.build(graph, 0)
    return machine.run()


def waits(name: str, latency_rounds: int) -> tuple:
    """The rounds each instruction waited: SWIPER-SIM's, then decsim's.

    SWIPER-SIM names only the instructions that waited; decsim names
    every feedback-blocked one.
    """
    swiper = swiper_run(name, latency_rounds)
    swiper_waits = {
        int(key): rounds for key, rounds in swiper["decode_waits"].items()
    }
    result = decsim_run(name, latency_rounds)
    period = config.microseconds_to_ticks(ROUND_MICROSECONDS)
    decsim_waits = {
        stall.operation_id: stall.stall_ticks / period
        for stall in result.gate_stalls
    }
    return swiper_waits, decsim_waits


@pytest.mark.parametrize("latency_rounds", LATENCY_ROUNDS)
def test_toffolis_waits_are_swiper_sims(latency_rounds):
    swiper_waits, decsim_waits = waits("toffoli", latency_rounds)

    waited = {key: rounds for key, rounds in decsim_waits.items() if rounds}
    assert waited == swiper_waits


@pytest.mark.parametrize("latency_rounds", LATENCY_ROUNDS)
def test_msd_waits_are_swiper_sims_or_a_cut_round_less(latency_rounds):
    swiper_waits, decsim_waits = waits("msd_15to1", latency_rounds)

    gaps = set()
    for key, rounds in decsim_waits.items():
        gap = swiper_waits.get(key, 0) - rounds
        gaps.add(gap)
    assert swiper_waits.keys() <= decsim_waits.keys()
    assert gaps <= {0, 1}
