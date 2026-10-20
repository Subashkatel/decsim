"""SWIPER's own reading of a schedule, the referent of lattice_surgery.py.

    python tools/swiper_referent.py SWIPER_CHECKOUT SCHEDULE DISTANCE OUT

Runs in SWIPER's environment (its pyproject's Python 3.10 to 3.12 and
networkx), with its checkout (github.com/jviszlai/swiper) as
SWIPER_CHECKOUT. Writes OUT as json: each instruction's rounds at the
distance (Duration.get_true_duration), the schedule's dependency edges
(LatticeSurgerySchedule.to_dag), and the instructions DeviceManager sets
to 0 rounds at seeds 0, 1 and 2, the conditional S it skips.
"""

import json
import pathlib
import sys

SEEDS = (0, 1, 2)


def main(
    checkout: pathlib.Path,
    schedule_path: pathlib.Path,
    distance: int,
    out: pathlib.Path,
) -> None:
    """SWIPER's rounds, edges and skipped instructions, as json."""
    folder = str(checkout)
    sys.path.insert(0, folder)
    import swiper.device_manager as device_manager
    import swiper.lattice_surgery_schedule as lattice_surgery_schedule

    text = schedule_path.read_text()
    lines = text.strip()
    schedule_class = lattice_surgery_schedule.LatticeSurgerySchedule
    schedule = schedule_class.from_str(lines)
    dag = schedule.to_dag(d=distance)
    rounds = [dag.nodes[node]["duration"] for node in sorted(dag.nodes)]
    edges = sorted(dag.edges)
    skipped = {}
    for seed in SEEDS:
        device = device_manager.DeviceManager(distance, schedule, rng=seed)
        durations = device._instruction_durations
        zeroed = [index for index, rounds in enumerate(durations) if not rounds]
        skipped[str(seed)] = zeroed
    document = {"rounds": rounds, "edges": edges, "zeroed_by_seed": skipped}
    written = json.dumps(document)
    out.write_text(written)


if __name__ == "__main__":
    swiper_checkout = pathlib.Path(sys.argv[1])
    schedule_file = pathlib.Path(sys.argv[2])
    code_distance = int(sys.argv[3])
    out_file = pathlib.Path(sys.argv[4])
    main(swiper_checkout, schedule_file, code_distance, out_file)
