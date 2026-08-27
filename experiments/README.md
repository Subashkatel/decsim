# Experiments

One yaml file is one experiment. `configs/<name>.yaml` holds every knob:
the decode path (`mode: weak_baseline | strong_only`), code and rounds,
windowing, sweep blocks, link cards, decoder cards, staging, boundary
application, frame commit. Nothing about an experiment lives in code.

Run one:

    PYTHONPATH=. python -m experiments.run experiments/configs/strong_decoder_baseline.yaml

Each run is one folder, `experiments/results/<UTC timestamp>-<name>/`,
never overwritten, holding everything needed to interpret or reproduce
it: `manifest.json` (resolved config, git commit, slurm job id, package
versions, host, times), `config/` (the yaml chain, copied verbatim),
`code_state.patch` (uncommitted code, when any), `shots.csv` (one row
per shot), `sweep.csv` (one row per point), `links.csv`, and the
figures. Sampling is deterministic from stim version +
task + d + rounds + p + seed, so manifest plus seeds are the raw data.
Seeds are 0..shots-1 per sweep point; only wall-clock derived columns
vary between reruns (a named real algorithm is timed on this host).
Long runs go through slurm: `sbatch experiments/slurm_run.sh
configs/<name>.yaml`. To make a new experiment, copy a yaml and change
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

- `slurm_run.sh`: one experiment as a cluster job.
- `tests/`: the experiment layer's own tests.

The offline decode-only harness (bulk statistics without the timing
simulator) was removed 2026-08-27 while unused; git history has it.

Everything this layout replaced (the baseline_closed_loop program, the
external validation gates, guide/walkthrough, prior result folders) is
preserved at the sandbox root in `archive/2026-08-cleanup/`, with the
same relative paths.
