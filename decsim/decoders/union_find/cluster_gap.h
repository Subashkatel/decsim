/* The cluster gap of one weighted Union-Find growth.
 *
 * One call quotients the decoding graph by the clusters a decode grew
 * and returns the length of the shortest closed walk of odd logical
 * parity, in the growth's half ticks. Meister et al. arXiv:2405.07433
 * Definition 9 identifies each cluster with a point and takes "the
 * length of the shortest path that covers a logical operator" in that
 * quotient; a covered stretch of an edge therefore costs nothing, and
 * covering a logical operator is a closed walk of odd parity.
 *
 * The caller owns every buffer and passes the growth as flat arrays,
 * the same layout the decode returned it in.
 */

#ifndef DECSIM_DECODERS_UNION_FIND_CLUSTER_GAP_H
#define DECSIM_DECODERS_UNION_FIND_CLUSTER_GAP_H

#include <stdint.h>

#include "union_find.h"

/* The gap when no odd closed walk exists at all. */
enum { union_find_cluster_gap_unreachable = -1 };

/* Walk one growth's quotient graph.
 *
 * The graph is the decode's: a detector index is a row of the
 * syndrome, -1 is the boundary, a length is in half ticks, and a
 * logical parity is zero or one. Per edge the interval is the one the
 * decode left behind: interval_is_closed where the growth covered the
 * whole edge, and otherwise the uncovered span between the two fronts.
 *
 * The caller sizes endpoint_a, endpoint_b, length_half_ticks,
 * logical_parity, interval_is_closed, interval_lower_tick and
 * interval_upper_tick at edge_count, and gap_half_ticks at one.
 * gap_half_ticks receives the gap, or union_find_cluster_gap_unreachable
 * when the growth admits no odd closed walk. The return value is an
 * enum union_find_status, and on anything but union_find_ok
 * gap_half_ticks holds the unreachable sentinel.
 */
int32_t union_find_cluster_gap(
    int32_t detector_count, int32_t edge_count, const int32_t *endpoint_a,
    const int32_t *endpoint_b, const int64_t *length_half_ticks,
    const uint8_t *logical_parity, const uint8_t *interval_is_closed,
    const int64_t *interval_lower_tick, const int64_t *interval_upper_tick,
    int64_t *gap_half_ticks);

#endif
