"""The syndrome buffers: a finished round kept until its last reader is done.

syndrome_buffer.py is the store itself, built from SyndromeBufferSettings:
rounds by key, bounded to its bits or unbounded, each round kept while
any hold on it is live (round_holds.py), each access priced by a flat
cost. ported_syndrome_buffer.py is the same store behind memory ports,
built from PortedSyndromeBufferSettings, whose accesses take words in
arrival order. A machine builds a store only when a decoder reads it
(decsim/build/readout.py): the weak syndrome buffer when the weak tier
decodes the plan's windows, with weak_syndrome_round_receiver.py as its
receiving end, its room and its landing, and the strong syndrome buffer
when a tier reads from the room side, with
strong_syndrome_round_receiver.py as its receiving end.
round_output.py is a store's face on the data path: every round that
leaves leaves through it, so the link a decode's
input rides is the store's own fact and not its reader's.
"""
