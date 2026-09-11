[decsim docs](../README.md) › How-to guides

# How-to guides

A how-to guide gets one task done for a reader who already knows what
decsim is. Each page is the steps and nothing else; the reasons are in
the [explanation](../explanation/README.md) pages and the facts in the
[reference](../reference/README.md) pages.

## Plug something in

decsim is built so that a new part is one class filling a port and one
row in a table, and nothing else changes. These four are that recipe
and its variants.

- [How to add a row to a table](add_a_table_row.md): the general recipe
  for any of the seventeen tables, the refusal a typo gets, and the
  worked examples in the tests.
- [How to add a decoder backend](add_a_decoder_backend.md): the decoder
  case, with the fault model contract and the check against PyMatching.
- [How to plug a component in without a table row](plug_in_without_a_table_row.md):
  hand the machine your own object while the class is still changing.
- [How to add a yaml key](add_a_yaml_key.md): a knob a config file can
  set, checked once at the boundary and documented in the reference
  file.

## Run and read an experiment

A real sweep is millions of shots, and what it writes is meant to be
read back.

- [How to run a sweep on Slurm](run_a_sweep_on_slurm.md): the array
  job, the two knobs that size a task, and the fold back into one
  report.
- [How to read a trace and follow one round or one window](read_a_trace.md):
  a viewer for the whole shot, or one path printed by `decsim trace
  follow`.
- [How to compare two runs](compare_two_runs.md): add shards, read two
  rows side by side, or plot both folders, and what to check before
  believing a difference.
- [How to run a timing study whose numbers do not depend on your computer](run_a_timing_only_study.md):
  price the decoders with cards so the ticks are a function of the
  configuration and the seed.
- [How to run a workload whose next operation waits on a decision](run_a_feedback_workload.md):
  the smallest two-operation workload that sends a decision back to the
  QPU, and the two links it lights up.
