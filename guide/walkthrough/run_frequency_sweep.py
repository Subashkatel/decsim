"""Run the walkthrough's frequency sweep and save every result.

Uses the experiment runner exactly as any experiment does: load the yaml,
run every shot of every point, summarize each point into one row. Writes
the runner's usual report (sweep.csv, sweep.md) plus rows.json, which keeps
the backlog trajectories so the figures can be redrawn without rerunning:

    PYTHONPATH=. python guide/walkthrough/run_frequency_sweep.py [config.yaml]
    PYTHONPATH=. python guide/walkthrough/frequency_plots.py [results_dir]

With no argument it runs this folder's frequency_sweep.yaml.
"""

import json
import sys
from pathlib import Path

from experiments.baseline.baseline_closed_loop import (load_config, run_sweep, summarize,
                                                       write_report)

CONFIG = Path(__file__).parent / "frequency_sweep.yaml"


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG
    config = load_config(config_path)
    rows = summarize(run_sweep(config))

    report_dir = Path(config["report_dir"])
    write_report(rows, report_dir)

    commit_rounds = config["windowing"]["commit_rounds"] or config["distance"]
    payload = {"commit_rounds": commit_rounds, "rows": rows}
    (report_dir / "rows.json").write_text(json.dumps(payload, indent=1))
    print(f"{len(rows)} sweep points -> {report_dir}/sweep.csv, sweep.md, rows.json")


if __name__ == "__main__":
    main()
