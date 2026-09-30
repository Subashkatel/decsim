/* What union_find.c and cluster_gap.c share about a graph's nodes and
 * workspace.
 *
 * Both number the boundary as the node after the last detector, and
 * both take their workspace one zeroed array at a time, so a failed
 * take is released by the same call that releases the whole workspace.
 * The helpers are static inline, so the one library the two files
 * link into still exports only the entry points their headers declare.
 */

#ifndef DECSIM_DECODERS_UNION_FIND_GRAPH_WORKSPACE_H
#define DECSIM_DECODERS_UNION_FIND_GRAPH_WORKSPACE_H

#include <stdint.h>
#include <stdlib.h>

/* The detector index an edge names for the boundary. */
enum { boundary_detector = -1 };

/* A zeroed array of count elements, at least one so that a null result
 * always means the allocation failed; a failure clears taken. */
static inline void *allocate_array(int32_t count, size_t size,
                                   int32_t *taken) {
  size_t wanted = (size_t)count;
  if (wanted == 0) {
    wanted = 1;
  }
  void *array = calloc(wanted, size);
  if (array == NULL) {
    *taken = 0;
  }
  return array;
}

/* The node of an edge's endpoint: the boundary is node detector_count. */
static inline int32_t endpoint_node(int32_t detector,
                                    int32_t detector_count) {
  if (detector == boundary_detector) {
    return detector_count;
  }
  return detector;
}

#endif
