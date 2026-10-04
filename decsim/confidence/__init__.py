"""The confidence signals: the soft output a weak decode reports.

Each row says what evidence it needs from the decode and which logical
classes the window is decoded in. complementary.py is the complementary
gap of a matching decode, cluster.py the cluster gap of a weighted
Union-Find decode, extra_cluster.py that decode's clusters grown on
(Kishi et al. 2602.03336), and gap_join.py joins a window's solves into
its confidence (Toshio et al. 2510.25222 Sec. III A).
"""
