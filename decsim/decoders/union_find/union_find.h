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
 *
 * A second call grows on from where a decode stopped, every cluster
 * and the boundary, until the two boundaries join or a growth limit is
 * reached: Kishi et al. arXiv:2602.03336 Algorithm 1, the extra-cluster
 * gap, with the join read as an edge closing a walk of odd logical
 * parity, the quotient reading of Meister et al. arXiv:2405.07433
 * Definition 9 that cluster_gap.h also takes.
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
 * step_growth_ticks and step_fusion_kinds at edge_count, and
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
 * 1053-1063); and what the strongest of the step's fusions changed, 0
 * when its closing edges united no two clusters, 1 when clusters united
 * and only roots and the touching flag moved, 2 when the survivor's
 * parity took an odd absorbed root's. A step that fused nothing has
 * zero hops. An edge of length zero is closed before the first step and
 * is fused there, so it belongs to no step and is charged in none. The
 * growth takes at most one step per edge, so edge_count entries always
 * suffice.
 *
 * forest_depth is the deepest parent chain of the trees the peel walks,
 * a root at zero, which is how many levels the peel's flags and
 * completions cross (Helios processing_unit_single_FPGA_v2.v lines
 * 236-266). The boundary is a flag an element carries rather than an
 * element, so a tree the peel roots at the boundary node counts from
 * the elements below it.
 */
int32_t union_find_decode(
    int32_t detector_count, int32_t edge_count, const int32_t *endpoint_a,
    const int32_t *endpoint_b, const int64_t *length_half_ticks,
    const uint8_t *residual_syndrome, uint8_t *selected_edges,
    uint8_t *interval_is_closed, int64_t *interval_lower_tick,
    int64_t *interval_upper_tick, int32_t *contact_edges,
    int32_t *contact_count, int32_t *forest_edges, int32_t *forest_count,
    int32_t *step_edge_counts, int32_t *step_hop_counts,
    int64_t *step_growth_ticks, uint8_t *step_fusion_kinds,
    int32_t *step_count, int32_t *forest_depth);

/* The tick the extra growth reports when the boundaries never join. */
enum { union_find_not_joined = -1 };

/* Grow one decode's clusters on and report when the boundaries join.
 *
 * The graph and the residual syndrome are the decode's; logical_parity
 * is one bit per edge, the row the walk parities are summed over. The
 * three interval arrays arrive as the decode left them and are
 * advanced in place, every end growing one half tick per tick, so an
 * edge between two clusters loses two half ticks a tick. The growth
 * stops at growth_limit_ticks. joined_at_tick receives the ticks grown
 * when an edge first closed a walk of odd logical parity, zero when
 * the decode's own closed edges already held one, and
 * union_find_not_joined when the limit came first, which is the
 * confident case.
 *
 * The four step arrays and step_count are the extra growth's cycle
 * count per event, laid out as union_find_decode's; the caller sizes
 * them at edge_count + 1, since the last event may stop at the limit
 * and close nothing. The boundary edges an event advanced count every
 * open edge of every cluster, because every cluster grows; the hops
 * are the deepest flood among the clusters the event fused, and an
 * edge that closed inside one cluster fused nothing.
 */
int32_t union_find_extra_growth(
    int32_t detector_count, int32_t edge_count, const int32_t *endpoint_a,
    const int32_t *endpoint_b, const int64_t *length_half_ticks,
    const uint8_t *logical_parity, const uint8_t *residual_syndrome,
    int64_t growth_limit_ticks, uint8_t *interval_is_closed,
    int64_t *interval_lower_tick, int64_t *interval_upper_tick,
    int32_t *step_edge_counts, int32_t *step_hop_counts,
    int64_t *step_growth_ticks, uint8_t *step_fusion_kinds,
    int32_t *step_count, int64_t *joined_at_tick);

#endif
