/* The cluster gap: the shortest odd closed walk of one growth.
 *
 * The contract is in cluster_gap.h. Every edge becomes a chain of
 * segments split at its interval's ticks; a segment the growth covered
 * costs nothing, an uncovered one costs its length, and the first
 * segment of an edge carries the edge's logical parity. The answer is
 * the minimum over nodes v of the distance from state (v, 0) to state
 * (v, 1) on the parity-doubled node set, which is the odd closed walk
 * Meister et al. arXiv:2405.07433 Definition 9 asks for.
 *
 * Each source is a Dijkstra with a binary heap over those states,
 * cut off at the best walk found so far. A minimum over sources does
 * not depend on the order the sources are visited nor on the cutoff,
 * so the nodes are walked in index order.
 *
 * Nothing here recurses and nothing allocates once the quotient graph
 * stands: one workspace is taken at the top and released on every path
 * out, and a generation stamp rather than a pass over every state
 * separates one source's distances from the next's.
 */

#include "cluster_gap.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

enum { boundary_detector = -1 };

/* An open edge splits at zero, both interval bounds and its length. */
enum { maximum_coordinates = 4 };

static const int64_t unreachable_distance = INT64_MAX;

/* One growth, as the caller laid it out. */
struct gap_inputs {
  int32_t detector_count;
  int32_t edge_count;
  const int32_t *endpoint_a;
  const int32_t *endpoint_b;
  const int64_t *length_half_ticks;
  const uint8_t *logical_parity;
  const uint8_t *interval_is_closed;
  const int64_t *interval_lower_tick;
  const int64_t *interval_upper_tick;
};

/* Everything one walk needs beyond the caller's buffers.
 *
 * A detector is its own node, the boundary is the node numbered
 * detector_count, and an edge's interior splits are the nodes from
 * edge_split_base upward, one per coordinate between the first and the
 * last. A segment holds two adjacency slots, one at each of its ends.
 * A search state is 2 * node + parity.
 */
struct workspace {
  int64_t *edge_coordinates;
  int32_t *edge_coordinate_count;
  int32_t *edge_split_base;
  int32_t *adjacency_start;
  int32_t *adjacency_cursor;
  int32_t *adjacency_neighbor;
  int64_t *adjacency_weight;
  uint8_t *adjacency_parity;
  int64_t *distance;
  int32_t *distance_stamp;
  int64_t *heap_distance;
  int32_t *heap_state;
};

/* One source's search over the workspace's arrays. */
struct search {
  int64_t *heap_distance;
  int32_t *heap_state;
  int32_t heap_size;
  int32_t heap_capacity;
  int64_t *distance;
  int32_t *distance_stamp;
  int32_t generation;
};

static void *allocate_array(int32_t count, size_t size, int32_t *taken) {
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

static void release_workspace(struct workspace *workspace) {
  free(workspace->edge_coordinates);
  free(workspace->edge_coordinate_count);
  free(workspace->edge_split_base);
  free(workspace->adjacency_start);
  free(workspace->adjacency_cursor);
  free(workspace->adjacency_neighbor);
  free(workspace->adjacency_weight);
  free(workspace->adjacency_parity);
  free(workspace->distance);
  free(workspace->distance_stamp);
  free(workspace->heap_distance);
  free(workspace->heap_state);
  memset(workspace, 0, sizeof(*workspace));
}

/* Every pointer is null until it is taken, so a failed take is released
 * by the same call that releases a whole workspace. */
static int32_t take_edge_layout(struct workspace *workspace,
                                int32_t edge_count) {
  int32_t coordinate_count = maximum_coordinates * edge_count;
  int32_t taken = 1;
  memset(workspace, 0, sizeof(*workspace));
  workspace->edge_coordinates =
      allocate_array(coordinate_count, sizeof(int64_t), &taken);
  workspace->edge_coordinate_count =
      allocate_array(edge_count, sizeof(int32_t), &taken);
  workspace->edge_split_base =
      allocate_array(edge_count, sizeof(int32_t), &taken);
  return taken;
}

/* The node count and the slot count are known only once every edge has
 * been split, so the rest of the workspace is taken after the layout. */
static int32_t take_quotient_graph(struct workspace *workspace,
                                   int32_t node_count, int32_t slot_count) {
  int32_t state_count = 2 * node_count;
  int32_t heap_capacity = 2 * slot_count + 2;
  int32_t taken = 1;
  workspace->adjacency_start =
      allocate_array(node_count + 1, sizeof(int32_t), &taken);
  workspace->adjacency_cursor =
      allocate_array(node_count, sizeof(int32_t), &taken);
  workspace->adjacency_neighbor =
      allocate_array(slot_count, sizeof(int32_t), &taken);
  workspace->adjacency_weight =
      allocate_array(slot_count, sizeof(int64_t), &taken);
  workspace->adjacency_parity =
      allocate_array(slot_count, sizeof(uint8_t), &taken);
  workspace->distance = allocate_array(state_count, sizeof(int64_t), &taken);
  workspace->distance_stamp =
      allocate_array(state_count, sizeof(int32_t), &taken);
  workspace->heap_distance =
      allocate_array(heap_capacity, sizeof(int64_t), &taken);
  workspace->heap_state =
      allocate_array(heap_capacity, sizeof(int32_t), &taken);
  return taken;
}

static int32_t endpoint_node(int32_t detector, int32_t detector_count) {
  if (detector == boundary_detector) {
    return detector_count;
  }
  return detector;
}

static void sort_coordinates(int64_t *values, int32_t count) {
  for (int32_t index = 1; index < count; ++index) {
    int64_t value = values[index];
    int32_t place = index;
    while (place > 0 && values[place - 1] > value) {
      values[place] = values[place - 1];
      place -= 1;
    }
    values[place] = value;
  }
}

/* Ascending values with the repeats dropped, compacted in place. */
static int32_t unique_coordinates(int64_t *values, int32_t count) {
  int32_t kept = 0;
  for (int32_t index = 0; index < count; ++index) {
    if (kept > 0 && values[index] == values[kept - 1]) {
      continue;
    }
    values[kept] = values[index];
    kept += 1;
  }
  return kept;
}

/* The tick coordinates where one edge's segments meet, ascending.
 *
 * A closed edge is one segment from zero to its length, and stays one
 * segment when that length is zero; an open edge splits at its
 * interval's bounds as well, and a bound that repeats another
 * coordinate adds no segment. */
static int32_t split_coordinates(const struct gap_inputs *inputs,
                                 int32_t edge_index, int64_t *coordinates) {
  int64_t length = inputs->length_half_ticks[edge_index];
  coordinates[0] = 0;
  if (inputs->interval_is_closed[edge_index]) {
    coordinates[1] = length;
    return 2;
  }
  coordinates[1] = inputs->interval_lower_tick[edge_index];
  coordinates[2] = inputs->interval_upper_tick[edge_index];
  coordinates[3] = length;
  sort_coordinates(coordinates, maximum_coordinates);
  return unique_coordinates(coordinates, maximum_coordinates);
}

/* The segment's cost: zero where the growth covered it. */
static int64_t segment_weight(const struct gap_inputs *inputs,
                              int32_t edge_index, int64_t lower,
                              int64_t upper) {
  if (inputs->interval_is_closed[edge_index]) {
    return 0;
  }
  if (upper <= inputs->interval_lower_tick[edge_index]) {
    return 0;
  }
  if (lower >= inputs->interval_upper_tick[edge_index]) {
    return 0;
  }
  return upper - lower;
}

/* The node at one position of an edge's chain: its two endpoints at the
 * ends and an interior split between them. */
static int32_t path_node(const struct gap_inputs *inputs,
                         const struct workspace *workspace,
                         int32_t edge_index, int32_t position,
                         int32_t segment_count) {
  if (position == 0) {
    int32_t detector = inputs->endpoint_a[edge_index];
    return endpoint_node(detector, inputs->detector_count);
  }
  if (position == segment_count) {
    int32_t detector = inputs->endpoint_b[edge_index];
    return endpoint_node(detector, inputs->detector_count);
  }
  int32_t base = workspace->edge_split_base[edge_index];
  return base + position - 1;
}

/* Split every edge and number its interior nodes, and count the slots
 * the segments will need. Returns the quotient graph's node count. */
static int32_t lay_out_edges(const struct gap_inputs *inputs,
                             struct workspace *workspace,
                             int32_t *slot_count) {
  int32_t node_count = inputs->detector_count + 1;
  int32_t slots = 0;
  for (int32_t edge = 0; edge < inputs->edge_count; ++edge) {
    int32_t first = maximum_coordinates * edge;
    int64_t *coordinates = &workspace->edge_coordinates[first];
    int32_t count = split_coordinates(inputs, edge, coordinates);
    workspace->edge_coordinate_count[edge] = count;
    workspace->edge_split_base[edge] = node_count;
    int32_t interior = count - 2;
    if (interior > 0) {
      node_count += interior;
    }
    slots += 2 * (count - 1);
  }
  *slot_count = slots;
  return node_count;
}

/* Each node's segment count, left one place so the offsets follow. */
static void count_node_degrees(const struct gap_inputs *inputs,
                               struct workspace *workspace) {
  for (int32_t edge = 0; edge < inputs->edge_count; ++edge) {
    int32_t segment_count = workspace->edge_coordinate_count[edge] - 1;
    for (int32_t segment = 0; segment < segment_count; ++segment) {
      int32_t next = segment + 1;
      int32_t left = path_node(inputs, workspace, edge, segment,
                               segment_count);
      int32_t right = path_node(inputs, workspace, edge, next, segment_count);
      workspace->adjacency_start[left + 1] += 1;
      workspace->adjacency_start[right + 1] += 1;
    }
  }
}

static void offsets_from_degrees(struct workspace *workspace,
                                 int32_t node_count) {
  int32_t running = 0;
  for (int32_t node = 0; node <= node_count; ++node) {
    int32_t degree = workspace->adjacency_start[node];
    running += degree;
    workspace->adjacency_start[node] = running;
  }
}

static void place_slot(struct workspace *workspace, int32_t node,
                       int32_t neighbor, int64_t weight, uint8_t parity) {
  int32_t slot = workspace->adjacency_cursor[node];
  workspace->adjacency_cursor[node] = slot + 1;
  workspace->adjacency_neighbor[slot] = neighbor;
  workspace->adjacency_weight[slot] = weight;
  workspace->adjacency_parity[slot] = parity;
}

/* One edge's chain: the logical parity rides the first segment, so a
 * walk that crosses the whole edge crosses the parity exactly once. */
static void place_edge_segments(const struct gap_inputs *inputs,
                                struct workspace *workspace, int32_t edge) {
  int32_t first = maximum_coordinates * edge;
  const int64_t *coordinates = &workspace->edge_coordinates[first];
  int32_t segment_count = workspace->edge_coordinate_count[edge] - 1;
  for (int32_t segment = 0; segment < segment_count; ++segment) {
    int32_t next = segment + 1;
    int64_t lower = coordinates[segment];
    int64_t upper = coordinates[next];
    int64_t weight = segment_weight(inputs, edge, lower, upper);
    uint8_t parity = 0;
    if (segment == 0) {
      parity = inputs->logical_parity[edge];
    }
    int32_t left = path_node(inputs, workspace, edge, segment, segment_count);
    int32_t right = path_node(inputs, workspace, edge, next, segment_count);
    place_slot(workspace, left, right, weight, parity);
    place_slot(workspace, right, left, weight, parity);
  }
}

static void place_segments(const struct gap_inputs *inputs,
                           struct workspace *workspace, int32_t node_count) {
  for (int32_t node = 0; node < node_count; ++node) {
    workspace->adjacency_cursor[node] = workspace->adjacency_start[node];
  }
  for (int32_t edge = 0; edge < inputs->edge_count; ++edge) {
    place_edge_segments(inputs, workspace, edge);
  }
}

static void search_over(struct search *search, struct workspace *workspace,
                        int32_t slot_count) {
  search->heap_distance = workspace->heap_distance;
  search->heap_state = workspace->heap_state;
  search->heap_size = 0;
  search->heap_capacity = 2 * slot_count + 2;
  search->distance = workspace->distance;
  search->distance_stamp = workspace->distance_stamp;
  search->generation = 0;
}

/* A state is pushed only when its distance falls, and a state is
 * processed once, so the pushes of one source are bounded by the
 * relaxations its states can make: two slots per slot, and the source
 * itself. */
static void heap_push(struct search *search, int64_t distance,
                      int32_t state) {
  assert(search->heap_size < search->heap_capacity);
  int32_t place = search->heap_size;
  search->heap_size += 1;
  while (place > 0) {
    int32_t parent = (place - 1) / 2;
    if (search->heap_distance[parent] <= distance) {
      break;
    }
    search->heap_distance[place] = search->heap_distance[parent];
    search->heap_state[place] = search->heap_state[parent];
    place = parent;
  }
  search->heap_distance[place] = distance;
  search->heap_state[place] = state;
}

static int64_t heap_pop(struct search *search, int32_t *state) {
  int64_t smallest = search->heap_distance[0];
  *state = search->heap_state[0];
  search->heap_size -= 1;
  int64_t distance = search->heap_distance[search->heap_size];
  int32_t moved = search->heap_state[search->heap_size];
  int32_t place = 0;
  int32_t child = 1;
  while (child < search->heap_size) {
    int32_t right = child + 1;
    int64_t left_distance = search->heap_distance[child];
    if (right < search->heap_size) {
      if (search->heap_distance[right] < left_distance) {
        child = right;
      }
    }
    if (search->heap_distance[child] >= distance) {
      break;
    }
    search->heap_distance[place] = search->heap_distance[child];
    search->heap_state[place] = search->heap_state[child];
    place = child;
    child = 2 * place + 1;
  }
  search->heap_distance[place] = distance;
  search->heap_state[place] = moved;
  return smallest;
}

/* A stamp from an earlier source stands for a distance this source has
 * not reached yet. */
static int64_t state_distance(const struct search *search, int32_t state) {
  if (search->distance_stamp[state] != search->generation) {
    return unreachable_distance;
  }
  return search->distance[state];
}

static void set_state_distance(struct search *search, int32_t state,
                               int64_t distance) {
  search->distance_stamp[state] = search->generation;
  search->distance[state] = distance;
}

static void relax_neighbors(struct search *search,
                            const struct workspace *workspace, int32_t state,
                            int64_t distance) {
  int32_t node = state / 2;
  int32_t parity = state % 2;
  int32_t first = workspace->adjacency_start[node];
  int32_t last = workspace->adjacency_start[node + 1];
  for (int32_t slot = first; slot < last; ++slot) {
    int32_t neighbor = workspace->adjacency_neighbor[slot];
    int32_t crossed = parity ^ workspace->adjacency_parity[slot];
    int32_t neighbor_state = 2 * neighbor + crossed;
    int64_t candidate = distance + workspace->adjacency_weight[slot];
    int64_t known = state_distance(search, neighbor_state);
    if (candidate >= known) {
      continue;
    }
    set_state_distance(search, neighbor_state, candidate);
    heap_push(search, candidate, neighbor_state);
  }
}

/* The shortest closed odd walk from one node while it is under the
 * running bound, and the bound itself otherwise. */
static int64_t shortest_odd_walk_from(struct search *search,
                                      const struct workspace *workspace,
                                      int32_t reference_node,
                                      int64_t shortest) {
  int32_t source_state = 2 * reference_node;
  int32_t target_state = source_state + 1;
  search->generation += 1;
  search->heap_size = 0;
  set_state_distance(search, source_state, 0);
  heap_push(search, 0, source_state);
  while (search->heap_size > 0) {
    int32_t state = 0;
    int64_t distance = heap_pop(search, &state);
    int64_t known = state_distance(search, state);
    if (distance != known) {
      continue;
    }
    if (distance >= shortest) {
      return shortest;
    }
    if (state == target_state) {
      return distance;
    }
    relax_neighbors(search, workspace, state, distance);
  }
  return shortest;
}

/* A node the growth left with no segment at all reaches nothing, so
 * walking every node costs the same minimum as walking the ones the
 * segments touch. */
static int64_t walk_every_node(struct search *search,
                               const struct workspace *workspace,
                               int32_t node_count) {
  int64_t shortest = unreachable_distance;
  for (int32_t node = 0; node < node_count; ++node) {
    shortest = shortest_odd_walk_from(search, workspace, node, shortest);
  }
  return shortest;
}

int32_t union_find_cluster_gap(
    int32_t detector_count, int32_t edge_count, const int32_t *endpoint_a,
    const int32_t *endpoint_b, const int64_t *length_half_ticks,
    const uint8_t *logical_parity, const uint8_t *interval_is_closed,
    const int64_t *interval_lower_tick, const int64_t *interval_upper_tick,
    int64_t *gap_half_ticks) {
  struct gap_inputs inputs;
  inputs.detector_count = detector_count;
  inputs.edge_count = edge_count;
  inputs.endpoint_a = endpoint_a;
  inputs.endpoint_b = endpoint_b;
  inputs.length_half_ticks = length_half_ticks;
  inputs.logical_parity = logical_parity;
  inputs.interval_is_closed = interval_is_closed;
  inputs.interval_lower_tick = interval_lower_tick;
  inputs.interval_upper_tick = interval_upper_tick;
  *gap_half_ticks = union_find_cluster_gap_unreachable;
  struct workspace workspace;
  if (!take_edge_layout(&workspace, edge_count)) {
    release_workspace(&workspace);
    return union_find_out_of_memory;
  }
  int32_t slot_count = 0;
  int32_t node_count = lay_out_edges(&inputs, &workspace, &slot_count);
  if (!take_quotient_graph(&workspace, node_count, slot_count)) {
    release_workspace(&workspace);
    return union_find_out_of_memory;
  }
  count_node_degrees(&inputs, &workspace);
  offsets_from_degrees(&workspace, node_count);
  place_segments(&inputs, &workspace, node_count);
  struct search search;
  search_over(&search, &workspace, slot_count);
  int64_t shortest = walk_every_node(&search, &workspace, node_count);
  release_workspace(&workspace);
  if (shortest != unreachable_distance) {
    *gap_half_ticks = shortest;
  }
  return union_find_ok;
}
