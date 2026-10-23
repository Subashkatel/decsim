[decsim docs](../README.md) › How-to guides

# How-to guides

A how-to guide gets one task done for a reader who has done the
tutorials. Each page is the steps and nothing else; the facts are in
the [reference](../reference/README.md) pages.

- [How to add a decoder backend](add_a_decoder_backend.md): a class
  that fills the decoder port, its settings record, and the check
  against PyMatching.
- [How to add a store, a link card, a signal, a policy or a scheme](add_a_store_card_signal_policy_or_scheme.md):
  a record of your own in a field the machine already has, with the
  component it builds.
- [How to add a component to a part](add_a_component_to_a_part.md): a
  new kind of component, built and wired in the part that holds its
  stage of the loop.
- [How to plug in a workload maker](plug_in_a_workload_maker.md): a
  function that returns an operation list and its physical circuit,
  handed to the machine.
- [How to use your own noise model](use_your_own_noise_model.md): a
  memory's noise row, Stim's channels, SD6 or your own.
- [How to set any parameter without editing code](set_parameters.md):
  `decsim params`, `decsim info` and `--set path=value`, for one task
  or every one, on Slurm too.
- [How to record statistics of your own part](record_statistics.md):
  declare them on the class, update them where the event happens, and
  read them in the run folder.
- [How to write a machine as a graph](write_a_machine_as_a_graph.md):
  named parts, wires and a fabric, in Python, set and described by the
  parts' names.
- [How to build a chain of decoders](build_a_chain_of_decoders.md): a
  pool added behind another, the verdict its edge decides on, and what
  the run reports for it.
- [How to route windows to several pools](route_windows_to_several_pools.md):
  a router in front of the entry pool, a rule that names each window's
  pool, pools escalating to one shared pool, and pools on one device.
- [How to share a pool among patches](share_a_pool_among_patches.md):
  VQD's critical-first schedulers (MLS, RR, MFD) for a pool with fewer
  units than patches.
- [How to decode a window with an ensemble](decode_a_window_with_an_ensemble.md):
  Harmony's and VibeLSD's poolings in one pool, and escalating the
  windows its members disagree on.
- [How to batch a pool's decodes](batch_a_pools_decodes.md): the
  re-decodes waiting for a unit decoded as one batch, priced by the sum
  or the slowest of its members.
- [How to use your own decoder and verdict](use_your_own_decoder_and_verdict.md):
  a decoder and a rule in your own package, with their own evidence,
  seeds and statistics, named by import path.
- [How to sweep a machine over a grid](sweep_a_machine_over_a_grid.md):
  constants.yaml for the numbers, config.py for the machine, run.py for
  the grid, and every number set from the command line.
- [How to build each switching pattern](build_each_switching_pattern.md):
  every pattern's setting, source and guide on one page, and what is
  still refused.
- [How to run a memory as a stream](run_a_memory_as_a_stream.md):
  `decsim stream`, one memory task as a stream of a million rounds in
  flat memory, measured as it goes.
