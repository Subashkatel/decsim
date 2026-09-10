"""The decoder side: a window's rounds become a correction, at a price.

decoder_manager.py is the facade the window side and the escalation
speak to. A job arrives on the DecodeQueue port; decode_queue.py holds
it in its pool's ready queue under the discipline schedulers.py names;
decode_dispatch.py offers it the next free unit of decoder_pool.py;
decoder_memory_transfer.py moves its rounds into that unit's memory
(decoder_memory.py, decoder_unit.py); and decode_service.py runs the
decode once the input has landed and the window side's gate allows it.
staged_decoder.py prices the unit's own stages around the algorithm.
decode_outcomes.py says what a finished decode means, and
decoder_output.py sends it on: the correction to the Pauli frame, the
committed boundary to the next window, and a selection to the strong
tier that strong_requests.py is waiting on.

The algorithm is one row of DECODERS in settings.py, one folder per
row: minimum_weight_perfect_matching/, belief_matching/, union_find/,
relay_belief_propagation/, tesseract/ and belief_propagation_osd/.
decoder.py holds the defaults every row shares, decoders.py the
timing-only rows and the routers, detection_events.py the formation a
tier does for itself, and verify_windows.py the referee that decodes
every window a second time and compares.
"""
