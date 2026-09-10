"""The confidence signals: the soft output a weak decode reports.

signals.py is the CONFIDENCE_SIGNALS table, one row per soft output a
switching run can ask for, each saying what evidence it needs from the
decode and which logical classes the window must be decoded in.
complementary.py is the complementary gap of an MWPM decode, the
absolute difference of two forced-class minimum weights; cluster.py is
the cluster gap of a weighted Union-Find decode, which is a Decoder row
that reports its own soft output from the growth it already did.
gap_join.py holds a window's solves until the last one arrives, asks
the signal for the confidence, and hands the answering solve to the
window verdict. Toshio et al. 2510.25222 Sec. III A.
"""
