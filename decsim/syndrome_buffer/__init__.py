"""The syndrome buffers: a finished round kept until its last reader is done.

syndrome_buffer.py is the store itself, the one row of SYNDROME_BUFFERS: rounds
by key, bounded to the count settings.py gives it or unbounded, each
round kept while any hold on it is live (round_holds.py). A machine
always builds the weak syndrome buffer, which the tier that decodes as the
rounds arrive reads; it builds the strong syndrome buffer as well when a tier
reads from the room side (decsim/build/stores.py), and
strong_syndrome_round_receiver.py is that store's receiving end, its room
and its landing.
round_output.py is a store's face on the data path: every round that
leaves leaves through it, so the link a decode's
input rides is the store's own fact and not its reader's.
"""
