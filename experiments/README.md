# Experiments

One yaml file is one experiment. `configs/<name>.yaml` holds every knob:
the decode path (`mode: weak_baseline | strong_only`), code and rounds,
windowing, sweep blocks, link cards, decoder cards, staging, boundary
application, frame commit. Nothing about an experiment lives in code.

Run one:

    PYTHONPATH=. python -m experiments.run experiments/configs/strong_decoder_baseline.yaml

Results land in `experiments/results/<name>/`: `sweep.csv` (every column),
`links.csv` (the per-link ledger totals), and the figures. Rerunning the
same config reproduces the same rows; seeds are 0..shots-1 per sweep
point, and only wall-clock derived columns vary (a named real algorithm
is timed on this host). To make a new experiment, copy a yaml and change
numbers; a new yaml key belongs in `experiment_config.py` first.
`configs/reference.yaml` lists every yaml key in one runnable file;
`tests/test_decoder_units.py` fails when it and the loader drift apart,
so update both in the same change as any config-surface edit.

The pipeline is four small modules, one job each, in call order:
`run.py` (orchestrates) -> `experiment_config.py` (the only yaml reader)
-> `build_run.py` (one sweep point -> one wired RunSpec)
-> `measure_shot.py` (one completed run -> one shot's numbers)
-> `sweep_report.py` / `plots.py` (rows -> csv files, figures).

Also here:

- `offline/`: the sampled offline accuracy harness (surface + BB codes,
  slurm array); it owns the list-ordered windowed reference decode.
- `tests/`: the experiment layer's own tests.

Everything this layout replaced (the baseline_closed_loop program, the
external validation gates, guide/walkthrough, prior result folders) is
preserved at the sandbox root in `archive/2026-08-cleanup/`, with the
same relative paths.
