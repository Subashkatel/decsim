"""Observation: every listener of a run, and the files a run leaves behind.

A component fires a trace source and knows no listener; this package
holds the listeners and the root wires them (wiring.py, from the yaml's
observation section), so a run with none of them gives the same result
record as a run with all of them. observation.py names the listeners of
one run and run_views freezes what they hold.

The ledgers: result_ledger the logical observables per operation,
window_ledger every window's record, stage_records the decoders' stage
rows, referee_audit what a checking decoder re-decoded, decode_records
the switching study's per-request and per-service rows, link_traffic
what each link carried, data_movement how often bits were copied or
referenced, round_store_occupancy a store's occupancy and residence,
queue_depth the decode queue over time, controller_counters the idle
rounds, metrics the integrated step functions, command_events and
runtime_stamps the ticks of each operation's life, round_events the
readout path's flight recorder, sampled_shots what the syndrome source
sampled, and flight_recorder one causal row per hardware transition.
The writers: log_writers for the engine's narrator, trace_writer for a
shot's Chrome trace. settings.py is the section that turns them on.
"""
