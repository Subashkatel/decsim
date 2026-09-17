/* The weighted Union-Find decoder of one graphlike decoding window.
 *
 * One call grows the clusters of one residual syndrome, takes the
 * minimum weight forest of the edges that made contact, and peels that
 * forest into a correction. The caller owns every buffer and passes the
 * graph as flat arrays.
 *
 * Delfosse and Nickerson arXiv:1709.06218 give the two algorithms:
 * every odd cluster grows by one half edge per round and clusters that
 * meet fuse (Algorithm 1), and the correction is read off a spanning
 * forest of the grown erasure by peeling leaf to root (Algorithm 2).
 * Huang, Newman and Brown arXiv:2004.04693 make the growth weighted: an
 * edge carries an integer length and a front covers one half tick per
 * tick, so a likelier fault is crossed sooner.
 */

#ifndef DECSIM_DECODERS_UNION_FIND_UNION_FIND_H
#define DECSIM_DECODERS_UNION_FIND_UNION_FIND_H

#include <stdint.h>

/* What one decode returns. Anything but union_find_ok leaves the output
 * buffers half written and is a fault of the graph, not of the
 * syndrome. */
enum union_find_status {
  union_find_ok = 0,
  union_find_out_of_memory = 1,
  union_find_interval_order_lost = 2,
  union_find_no_positive_growth = 3,
  union_find_growth_bound_exceeded = 4
};

/* Decode one residual syndrome on one graph.
 *
 * A detector index is a row of the syndrome; -1 is the boundary, which
 * is the node numbered detector_count. Edges arrive in fault order, so
 * an edge index orders exactly as its fault index does and every place
 * that needs fault order uses the edge index. A length is in half
 * ticks, and an edge of length zero starts closed.
 *
 * The caller sizes residual_syndrome at detector_count, selected_edges,
 * interval_is_closed, interval_lower_tick, interval_upper_tick,
 * contact_edges, forest_edges, step_edge_counts, step_hop_counts,
 * step_growth_ticks and step_odd_fusions at edge_count, and
 * contact_count, forest_count, step_count and forest_depth at one.
 * selected_edges carries one flag per edge; the three interval arrays
 * carry the open bounds of every edge the growth did not close;
 * contact_edges carries the edges that fused two clusters, in the
 * order they closed; forest_edges carries the forest, in the order
 * Kruskal took it.
 *
 * The four step arrays are the growth's cycle count per step, one
 * entry per growth step: the boundary edges the step advanced, which is
 * its work; the deepest flood over closed edges from the root of any
 * cluster the step fused, which is its critical path in hops (Helios
 * 2301.08419 lines 623-629: propagating a cluster identifier and its
 * parity takes as many stages as the cluster is deep); the ticks the
 * step spanned, which is how many one-unit growth iterations a unit
 * that grows one unit of weight at a time spends on it (Helios lines
 * 1053-1063); and whether any fusion of the step joined two odd
 * clusters, the fusion whose parity has to cross the fused cluster. A
 * step that fused nothing has zero hops. The growth takes at most one
 * step per edge, so edge_count entries always suffice.
 *
 * forest_depth is the deepest parent chain of the trees the peel walks,
 * a root at zero, which is how many levels the peel's flags and
 * completions cross (Helios processing_unit_single_FPGA_v2.v lines
 * 236-266).
 */
int32_t union_find_decode(
    int32_t detector_count, int32_t edge_count, const int32_t *endpoint_a,
    const int32_t *endpoint_b, const int64_t *length_half_ticks,
    const uint8_t *residual_syndrome, uint8_t *selected_edges,
    uint8_t *interval_is_closed, int64_t *interval_lower_tick,
    int64_t *interval_upper_tick, int32_t *contact_edges,
    int32_t *contact_count, int32_t *forest_edges, int32_t *forest_count,
    int32_t *step_edge_counts, int32_t *step_hop_counts,
    int64_t *step_growth_ticks, uint8_t *step_odd_fusions,
    int32_t *step_count, int32_t *forest_depth);

#endif
