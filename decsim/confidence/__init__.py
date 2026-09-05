"""The confidence signals: the soft output a weak decode reports.

The complementary gap of an MWPM decode is the ConfidenceSignal row
(complementary.py); the wrappers attach a signal to a weak decoder on
one core, two cores or two units (decoder.py); the cluster gap of a
Union-Find decode is a Decoder row that reports its own soft output
(cluster.py). Toshio et al. 2510.25222 Sec. III A.
"""
