"""The syndrome buffers: a finished round kept until its last reader is done.

syndrome_buffer.py is the store and ported_syndrome_buffer.py the same
store behind memory ports. The two round receivers are the stores'
receiving ends, and round_output.py their sending end: every round
leaves through it, so the link a decode's input rides is the store's own
fact.
"""
