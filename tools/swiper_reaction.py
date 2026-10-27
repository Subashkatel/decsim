"""SWIPER-SIM's own runs of a schedule, the referent of decsim's gate stalls.

    python tools/swiper_reaction.py SWIPER_CHECKOUT SCHEDULE DISTANCE OUT

Runs in SWIPER's environment with its checkout (github.com/jviszlai/swiper)
as SWIPER_CHECKOUT. DecodingSimulator.run (swiper/simulator.py) decodes
the schedule at the distance with sliding windows, no speculation, as
many decoders as it asks for and every window taking the same rounds,
the device at seed 0. It runs once per latency and per start rule:
"alap" is SWIPER's own, which starts an instruction that initializes
all its patches as late as its successors allow
(device_manager.py _predict_instruction_start_time); "asap" starts
every instruction once its predecessors finish, as decsim's controller
does. Writes OUT as json: per run, the latency, the start rule, the
device's rounds, each instruction's start round, and the rounds each
instruction conditioned on a decode waited for it
(DeviceData.conditioned_decode_wait_times), by instruction index.
"""

import json
import pathlib
import sys

SEED = 0
LATENCY_ROUNDS = (1, 5, 10)
START_RULES = ("alap", "asap")


def main(
    checkout: pathlib.Path,
    schedule_path: pathlib.Path,
    distance: int,
    out: pathlib.Path,
) -> None:
    """SWIPER-SIM's start rounds and decode waits for the schedule."""
    folder = str(checkout)
    sys.path.insert(0, folder)
    import swiper.lattice_surgery_schedule as lattice_surgery_schedule
    import swiper.simulator as simulator

    text = schedule_path.read_text()
    lines = text.strip()
    schedule_class = lattice_surgery_schedule.LatticeSurgerySchedule
    stock_device = simulator.DeviceManager
    asap_device = _asap_device(stock_device)
    device_by_rule = {"alap": stock_device, "asap": asap_device}
    runs = []
    for start_rule in START_RULES:
        simulator.DeviceManager = device_by_rule[start_rule]
        for latency_rounds in LATENCY_ROUNDS:
            schedule = schedule_class.from_str(lines)
            run = _run(simulator, schedule, distance, latency_rounds)
            run["start_rule"] = start_rule
            runs.append(run)
    simulator.DeviceManager = stock_device
    written = json.dumps({"runs": runs})
    out.write_text(written)


def _run(simulator, schedule, distance: int, latency_rounds: int) -> dict:
    """One run: every window decodes in the latency's rounds."""
    run = simulator.DecodingSimulator()
    success, _, device, _, decoding = run.run(
        schedule=schedule,
        distance=distance,
        scheduling_method="sliding",
        decoding_latency_fn=lambda _volume: latency_rounds,
        speculation_mode=None,
        rng=SEED,
    )
    assert success, "SWIPER-SIM did not finish the schedule"
    waits = device.conditioned_decode_wait_times
    return {
        "latency_rounds": latency_rounds,
        "device_rounds": device.num_rounds,
        "decoder_rounds": decoding.num_rounds,
        "start_rounds": device.instruction_start_times,
        "decode_waits": {str(index): rounds for index, rounds in waits.items()},
    }


def _asap_device(stock_device):
    """SWIPER's device, starting every instruction as soon as it may."""

    class AsapDevice(stock_device):
        """No instruction is a startup one, so none waits for successors."""

        def _calc_is_startup_instruction(self, instruction_idx: int) -> bool:
            del instruction_idx
            return False

        def _predict_instruction_start_time(self, instruction_idx, first_round):
            # SWIPER predicts a root only as a startup instruction; as an
            # ordinary one it starts now
            is_placed = (
                instruction_idx in first_round
                or instruction_idx in self._active_instructions
            )
            instruction = self.schedule_instructions[instruction_idx]
            is_started = is_placed or instruction.end_round != -1
            predecessors = self.schedule_dag.predecessors(instruction_idx)
            if is_started or any(True for _ in predecessors):
                predict = stock_device._predict_instruction_start_time
                return predict(self, instruction_idx, first_round)
            first_round[instruction_idx] = self.current_round
            return first_round, []

    return AsapDevice


if __name__ == "__main__":
    swiper_checkout = pathlib.Path(sys.argv[1])
    schedule_file = pathlib.Path(sys.argv[2])
    code_distance = int(sys.argv[3])
    out_file = pathlib.Path(sys.argv[4])
    main(swiper_checkout, schedule_file, code_distance, out_file)
