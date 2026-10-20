"""SWIPER's own reading of a schedule, the referent of lattice_surgery.py.

    python tools/swiper_referent.py SWIPER_CHECKOUT SCHEDULE DISTANCE OUT

Runs in SWIPER's environment (its pyproject's Python 3.10 to 3.12 and
networkx), with its checkout (github.com/jviszlai/swiper) as
SWIPER_CHECKOUT. Writes OUT as json: each instruction's rounds at the
distance (Duration.get_true_duration), the schedule's dependency edges
(LatticeSurgerySchedule.to_dag), the instructions DeviceManager sets
to 0 rounds at seeds 0, 1 and 2, the conditional S it skips, each
instruction's merge faces (Instruction.merge_faces), whether
WindowBuilder.build_windows decodes its rounds, and the patch lives of
the DeviceManager run at seed 0: each life's patch, the instructions
that measured rounds in it, and its rounds by kind (an instruction's,
a T injection's, an idle one), a life starting at a round that
initializes its patch (SyndromeRound.initialized_patch), with its
rounds as runs of one instruction or of idling. The same run feeds
SWIPER's sliding windows (window_builder.py WindowBuilder,
window_manager.py SlidingWindowManager), as simulator.py step_experiment
does: each constructed window's commit region and buffer regions, and
the window dependency edges, each window named by its commit region.
"""

import json
import pathlib
import sys

SEEDS = (0, 1, 2)
CENSUS_SEED = 0


def main(
    checkout: pathlib.Path,
    schedule_path: pathlib.Path,
    distance: int,
    out: pathlib.Path,
) -> None:
    """SWIPER's rounds, edges, skips, faces, decodes and patch lives."""
    folder = str(checkout)
    sys.path.insert(0, folder)
    import swiper.device_manager as device_manager
    import swiper.lattice_surgery_schedule as lattice_surgery_schedule
    import swiper.window_builder as window_builder
    import swiper.window_manager as window_manager

    text = schedule_path.read_text()
    lines = text.strip()
    schedule_class = lattice_surgery_schedule.LatticeSurgerySchedule
    schedule = schedule_class.from_str(lines)
    dag = schedule.to_dag(d=distance)
    rounds = [dag.nodes[node]["duration"] for node in sorted(dag.nodes)]
    edges = sorted(dag.edges)
    skipped = _zeroed_by_seed(device_manager, schedule, distance)
    faces = [_faces(instruction) for instruction in schedule.instructions]
    decoded = [_is_decoded(each) for each in schedule.instructions]
    device = device_manager.DeviceManager(distance, schedule, rng=CENSUS_SEED)
    builder = window_builder.WindowBuilder(distance)
    windows = window_manager.SlidingWindowManager(builder)
    lives = _patch_lives(device, windows)
    document = {
        "rounds": rounds,
        "edges": edges,
        "zeroed_by_seed": skipped,
        "merge_faces": faces,
        "decoded": decoded,
        "patch_lives": lives,
        "sliding_windows": _windows(windows),
        "sliding_window_edges": _window_edges(windows),
    }
    written = json.dumps(document)
    out.write_text(written)


def _zeroed_by_seed(device_manager, schedule, distance: int) -> dict:
    """The instructions DeviceManager sets to 0 rounds, at each seed."""
    skipped = {}
    for seed in SEEDS:
        device = device_manager.DeviceManager(distance, schedule, rng=seed)
        durations = device._instruction_durations
        zeroed = [index for index, rounds in enumerate(durations) if not rounds]
        skipped[str(seed)] = zeroed
    return skipped


def _patch_lives(device, windows) -> list:
    """Each patch life the device runs, every decode reported done.

    The windows read each cycle's rounds, and the empty cycle after the
    last flushes them, as simulator.py step_experiment drives them.
    """
    open_by_patch = {}
    lives = []
    while not device.is_done():
        rounds = device.get_next_round(set())
        windows.process_round(rounds)
        for syndrome_round in rounds:
            life = _life_of(syndrome_round, open_by_patch, lives)
            _count(life, syndrome_round)
            _extend_runs(life, syndrome_round)
    windows.process_round([])
    for life in lives:
        life["instructions"].sort()
    return lives


def _life_of(syndrome_round, open_by_patch: dict, lives: list) -> dict:
    """The round's life: a new one when it initializes its patch."""
    patch = syndrome_round.patch
    is_new = syndrome_round.initialized_patch or patch not in open_by_patch
    if is_new:
        life = {
            "patch": list(patch),
            "instructions": [],
            "rounds": {},
            "runs": [],
        }
        open_by_patch[patch] = life
        lives.append(life)
    return open_by_patch[patch]


def _count(life: dict, syndrome_round) -> None:
    """One round of the life, by its kind."""
    kind = "instruction"
    if syndrome_round.is_unwanted_idle:
        kind = "idle"
    elif syndrome_round.instruction.name == "INJECT_T":
        kind = "injection"
    rounds = life["rounds"]
    rounds[kind] = rounds.get(kind, 0) + 1
    index = syndrome_round.instruction_idx
    if index >= 0 and index not in life["instructions"]:
        life["instructions"].append(index)


def _extend_runs(life: dict, syndrome_round) -> None:
    """The round joins the life's last run, or starts one.

    A run is [first round, rounds, instruction index], the index -1 for
    idle rounds (device_manager.py UNWANTED_IDLE).
    """
    index = syndrome_round.instruction_idx
    runs = life["runs"]
    if runs:
        first, count, last_index = runs[-1]
        if last_index == index and first + count == syndrome_round.round:
            runs[-1][1] += 1
            return
    runs.append([syndrome_round.round, 1, index])


def _windows(windows) -> list:
    """Each constructed window: its commit regions and buffer regions."""
    built = []
    for window_index in windows.all_constructed_windows:
        window = windows.all_windows[window_index]
        commits = [_region(region) for region in window.commit_region]
        buffers = [_region(region) for region in window.buffer_regions]
        buffers.sort()
        built.append({"commit": commits, "buffers": buffers})
    return built


def _window_edges(windows) -> list:
    """Each dependency edge, its windows named by their first commit region."""
    edges = []
    for before, after in windows.window_dag.edges:
        first = windows.all_windows[before].commit_region[0]
        second = windows.all_windows[after].commit_region[0]
        earlier = _region(first)
        later = _region(second)
        edges.append([earlier, later])
    edges.sort()
    return edges


def _region(region) -> list:
    """A region as [patch, first round, rounds]."""
    return [list(region.patch), region.round_start, region.duration]


def _faces(instruction) -> list:
    """The merge faces, each pair and the pairs sorted, as json lists."""
    pairs = [sorted(face) for face in instruction.merge_faces]
    pairs.sort()
    return pairs


def _is_decoded(instruction) -> bool:
    """Whether the window builder decodes its rounds: not a T injection."""
    return instruction.name != "INJECT_T"


if __name__ == "__main__":
    swiper_checkout = pathlib.Path(sys.argv[1])
    schedule_file = pathlib.Path(sys.argv[2])
    code_distance = int(sys.argv[3])
    out_file = pathlib.Path(sys.argv[4])
    main(swiper_checkout, schedule_file, code_distance, out_file)
