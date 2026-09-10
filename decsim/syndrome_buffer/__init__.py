"""The round stores: a finished round kept until its last reader is done.

round_store.py is the store itself, the one row of ROUND_STORES: rounds
by key, bounded to the count settings.py gives it or unbounded, each
round kept while any hold on it is live (round_holds.py). A machine
always builds Buffer 0, which the tier that decodes as the rounds
arrive reads; it builds Buffer 1 as well when a tier reads from the
room side (decsim/build/stores.py), and strong_round_writer.py is that
store's priced crossing. round_output.py is a store's face on the data
path: every round that leaves leaves through it, so the link a decode's
input rides is the store's own fact and not its reader's.
"""
