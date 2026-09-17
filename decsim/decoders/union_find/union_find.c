/* The weighted Union-Find decoder: growth, contact forest and peeling.
 *
 * The contract is in union_find.h. Growth is event driven: each event
 * jumps to the next tick at which some front closes an edge, and a
 * front covers one half tick per tick, so an edge between two growing
 * clusters loses two half ticks per tick and an edge with one growing
 * end loses one.
 *
 * An event touches only the edges of the clusters that still grow.
 * Each cluster keeps the list of edges incident to it, the boundary
 * list of Delfosse and Nickerson arXiv:1709.06218 section 3: to grow a
 * cluster "we must then simply iterate over this list and grow the
 * incident edges", fusion appends one cluster's list to the other's,
 * and a last pass over the list drops what is no longer on the
 * boundary. An edge with no growing end cannot move, so leaving it out
 * of the event changes nothing it would have done.
 *
 * Nothing here recurses and nothing allocates once a decode has begun:
 * one workspace is taken at the top and released on every path out.
 */

#include "union_find.h"

#include <stdlib.h>
#include <string.h>

enum { boundary_detector = -1 };

/* One graph, as the caller laid it out. */
struct graph_arrays {
  int32_t detector_count;
  int32_t edge_count;
  const int32_t *endpoint_a;
  const int32_t *endpoint_b;
  const int64_t *length_half_ticks;
};

/* Everything one decode needs beyond the caller's buffers.
 *
 * A boundary list is a chain of entries through cluster_edge_next.
 * Entry 2 * e stands for edge e at its first endpoint and 2 * e + 1 for
 * the same edge at its second, so an entry names its edge by its own
 * index and a cluster holding both ends of an edge holds two entries
 * for it. An entry stays in the list of the cluster its endpoint is in.
 */
struct workspace {
  int32_t *parent;
  uint8_t *parity;
  uint8_t *touches_boundary;
  int32_t *cluster_edge_head;
  int32_t *cluster_edge_tail;
  int32_t *cluster_edge_next;
  int32_t *active_roots;
  int32_t *next_active_roots;
  int32_t *active_stamp;
  int32_t *working_edges;
  int32_t *working_scratch;
  int32_t *event_stamp;
  int32_t *frozen_root_a;
  int32_t *frozen_root_b;
  int64_t *ticks_until_close;
  int32_t *sorted_contacts;
  int32_t *merge_scratch;
  uint8_t *is_forest_edge;
  int32_t *adjacency_start;
  int32_t *adjacency_cursor;
  int32_t *adjacency_neighbor;
  int32_t *adjacency_edge;
  int32_t *node_stack;
  int32_t *tree_order;
  int32_t *parent_node;
  int32_t *parent_edge;
  int32_t *node_level;
  uint8_t *has_parent;
  uint8_t *residual_defect;
  uint8_t *is_visited;
  int32_t *component_nodes;
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
  free(workspace->parent);
  free(workspace->parity);
  free(workspace->touches_boundary);
  free(workspace->cluster_edge_head);
  free(workspace->cluster_edge_tail);
  free(workspace->cluster_edge_next);
  free(workspace->active_roots);
  free(workspace->next_active_roots);
  free(workspace->active_stamp);
  free(workspace->working_edges);
  free(workspace->working_scratch);
  free(workspace->event_stamp);
  free(workspace->frozen_root_a);
  free(workspace->frozen_root_b);
  free(workspace->ticks_until_close);
  free(workspace->sorted_contacts);
  free(workspace->merge_scratch);
  free(workspace->is_forest_edge);
  free(workspace->adjacency_start);
  free(workspace->adjacency_cursor);
  free(workspace->adjacency_neighbor);
  free(workspace->adjacency_edge);
  free(workspace->node_stack);
  free(workspace->tree_order);
  free(workspace->parent_node);
  free(workspace->parent_edge);
  free(workspace->node_level);
  free(workspace->has_parent);
  free(workspace->residual_defect);
  free(workspace->is_visited);
  free(workspace->component_nodes);
  memset(workspace, 0, sizeof(*workspace));
}

/* Every pointer is null until it is taken, so a failed take is released
 * by the same call that releases a whole workspace. */
static int32_t take_workspace(struct workspace *workspace, int32_t node_count,
                              int32_t edge_count) {
  int32_t entry_count = 2 * edge_count;
  int32_t taken = 1;
  memset(workspace, 0, sizeof(*workspace));
  workspace->parent = allocate_array(node_count, sizeof(int32_t), &taken);
  workspace->parity = allocate_array(node_count, sizeof(uint8_t), &taken);
  workspace->touches_boundary =
      allocate_array(node_count, sizeof(uint8_t), &taken);
  workspace->cluster_edge_head =
      allocate_array(node_count, sizeof(int32_t), &taken);
  workspace->cluster_edge_tail =
      allocate_array(node_count, sizeof(int32_t), &taken);
  workspace->cluster_edge_next =
      allocate_array(entry_count, sizeof(int32_t), &taken);
  workspace->active_roots =
      allocate_array(node_count, sizeof(int32_t), &taken);
  workspace->next_active_roots =
      allocate_array(node_count, sizeof(int32_t), &taken);
  workspace->active_stamp =
      allocate_array(node_count, sizeof(int32_t), &taken);
  workspace->working_edges =
      allocate_array(edge_count, sizeof(int32_t), &taken);
  workspace->working_scratch =
      allocate_array(edge_count, sizeof(int32_t), &taken);
  workspace->event_stamp = allocate_array(edge_count, sizeof(int32_t), &taken);
  workspace->frozen_root_a =
      allocate_array(edge_count, sizeof(int32_t), &taken);
  workspace->frozen_root_b =
      allocate_array(edge_count, sizeof(int32_t), &taken);
  workspace->ticks_until_close =
      allocate_array(edge_count, sizeof(int64_t), &taken);
  workspace->sorted_contacts =
      allocate_array(edge_count, sizeof(int32_t), &taken);
  workspace->merge_scratch =
      allocate_array(edge_count, sizeof(int32_t), &taken);
  workspace->is_forest_edge =
      allocate_array(edge_count, sizeof(uint8_t), &taken);
  workspace->adjacency_start =
      allocate_array(node_count + 1, sizeof(int32_t), &taken);
  workspace->adjacency_cursor =
      allocate_array(node_count, sizeof(int32_t), &taken);
  workspace->adjacency_neighbor =
      allocate_array(entry_count, sizeof(int32_t), &taken);
  workspace->adjacency_edge =
      allocate_array(entry_count, sizeof(int32_t), &taken);
  workspace->node_stack = allocate_array(node_count, sizeof(int32_t), &taken);
  workspace->tree_order = allocate_array(node_count, sizeof(int32_t), &taken);
  workspace->parent_node = allocate_array(node_count, sizeof(int32_t), &taken);
  workspace->parent_edge = allocate_array(node_count, sizeof(int32_t), &taken);
  workspace->node_level = allocate_array(node_count, sizeof(int32_t), &taken);
  workspace->has_parent = allocate_array(node_count, sizeof(uint8_t), &taken);
  workspace->residual_defect =
      allocate_array(node_count, sizeof(uint8_t), &taken);
  workspace->is_visited = allocate_array(node_count, sizeof(uint8_t), &taken);
  workspace->component_nodes =
      allocate_array(node_count, sizeof(int32_t), &taken);
  return taken;
}

static int32_t endpoint_node(int32_t detector, int32_t detector_count) {
  if (detector == boundary_detector) {
    return detector_count;
  }
  return detector;
}

static int32_t find_root(int32_t *parent, int32_t node) {
  int32_t root = node;
  while (parent[root] != root) {
    root = parent[root];
  }
  while (parent[node] != root) {
    int32_t next = parent[node];
    parent[node] = root;
    node = next;
  }
  return root;
}

/* An odd cluster that does not touch the boundary keeps growing. */
static int32_t cluster_grows(const struct workspace *workspace, int32_t root) {
  if (workspace->touches_boundary[root]) {
    return 0;
  }
  return workspace->parity[root] == 1;
}

/* The smaller root index survives, and parity and the shared boundary
 * pass to it. The survivor's root is returned, and -1 when the two
 * nodes already stood in one cluster. */
static int32_t union_nodes(struct workspace *workspace, int32_t left,
                           int32_t right) {
  int32_t left_root = find_root(workspace->parent, left);
  int32_t right_root = find_root(workspace->parent, right);
  if (left_root == right_root) {
    return -1;
  }
  int32_t survivor = left_root;
  int32_t absorbed = right_root;
  if (right_root < left_root) {
    survivor = right_root;
    absorbed = left_root;
  }
  workspace->parent[absorbed] = survivor;
  workspace->parity[survivor] ^= workspace->parity[absorbed];
  if (workspace->touches_boundary[absorbed]) {
    workspace->touches_boundary[survivor] = 1;
  }
  return absorbed;
}

static void append_entry(struct workspace *workspace, int32_t node,
                         int32_t entry) {
  int32_t tail = workspace->cluster_edge_tail[node];
  workspace->cluster_edge_next[entry] = -1;
  if (tail < 0) {
    workspace->cluster_edge_head[node] = entry;
  } else {
    workspace->cluster_edge_next[tail] = entry;
  }
  workspace->cluster_edge_tail[node] = entry;
}

static void unlink_entry(struct workspace *workspace, int32_t root,
                         int32_t previous, int32_t next) {
  if (previous < 0) {
    workspace->cluster_edge_head[root] = next;
  } else {
    workspace->cluster_edge_next[previous] = next;
  }
  if (next < 0) {
    workspace->cluster_edge_tail[root] = previous;
  }
}

/* The absorbed cluster's boundary list joins the survivor's. */
static void splice_edge_list(struct workspace *workspace, int32_t survivor,
                             int32_t absorbed) {
  int32_t head = workspace->cluster_edge_head[absorbed];
  if (head < 0) {
    return;
  }
  int32_t tail = workspace->cluster_edge_tail[survivor];
  if (tail < 0) {
    workspace->cluster_edge_head[survivor] = head;
  } else {
    workspace->cluster_edge_next[tail] = head;
  }
  workspace->cluster_edge_tail[survivor] =
      workspace->cluster_edge_tail[absorbed];
  workspace->cluster_edge_head[absorbed] = -1;
  workspace->cluster_edge_tail[absorbed] = -1;
}

/* One is returned when the two clusters were both odd, which is the
 * fusion whose parity has to cross the fused cluster. */
static int32_t fuse_clusters(struct workspace *workspace, int32_t left,
                             int32_t right) {
  int32_t left_root = find_root(workspace->parent, left);
  int32_t right_root = find_root(workspace->parent, right);
  int32_t joins_two_odd = workspace->parity[left_root];
  joins_two_odd &= workspace->parity[right_root];
  int32_t absorbed = union_nodes(workspace, left, right);
  if (absorbed < 0) {
    return 0;
  }
  int32_t survivor = find_root(workspace->parent, absorbed);
  splice_edge_list(workspace, survivor, absorbed);
  return joins_two_odd;
}

/* Every node alone in its own cluster; the boundary node carries no
 * defect and touches the boundary. A null defects array is the forest's
 * empty syndrome. */
static void reset_clusters(struct workspace *workspace, int32_t detector_count,
                           const uint8_t *defects) {
  for (int32_t node = 0; node < detector_count; ++node) {
    workspace->parent[node] = node;
    workspace->parity[node] = 0;
    workspace->touches_boundary[node] = 0;
  }
  workspace->parent[detector_count] = detector_count;
  workspace->parity[detector_count] = 0;
  workspace->touches_boundary[detector_count] = 1;
  if (defects == NULL) {
    return;
  }
  for (int32_t node = 0; node < detector_count; ++node) {
    workspace->parity[node] = defects[node];
  }
}

/* One entry per endpoint, in edge order, before any cluster fuses. */
static void link_edge_entries(struct workspace *workspace,
                              const struct graph_arrays *graph) {
  int32_t node_count = graph->detector_count + 1;
  for (int32_t node = 0; node < node_count; ++node) {
    workspace->cluster_edge_head[node] = -1;
    workspace->cluster_edge_tail[node] = -1;
  }
  for (int32_t edge = 0; edge < graph->edge_count; ++edge) {
    int32_t node_a =
        endpoint_node(graph->endpoint_a[edge], graph->detector_count);
    int32_t node_b =
        endpoint_node(graph->endpoint_b[edge], graph->detector_count);
    append_entry(workspace, node_a, 2 * edge);
    append_entry(workspace, node_b, 2 * edge + 1);
  }
}

static void initial_intervals(const struct graph_arrays *graph,
                              uint8_t *interval_is_closed,
                              int64_t *interval_lower_tick,
                              int64_t *interval_upper_tick) {
  for (int32_t edge = 0; edge < graph->edge_count; ++edge) {
    int64_t length = graph->length_half_ticks[edge];
    interval_is_closed[edge] = 0;
    if (length == 0) {
      interval_is_closed[edge] = 1;
    }
    interval_lower_tick[edge] = 0;
    interval_upper_tick[edge] = length;
  }
}

/* Every cluster that still grows, each once, under its current root. */
static int32_t initial_active_roots(struct workspace *workspace,
                                    int32_t detector_count, int32_t stamp) {
  int32_t node_count = detector_count + 1;
  int32_t active_count = 0;
  for (int32_t node = 0; node < node_count; ++node) {
    int32_t root = find_root(workspace->parent, node);
    if (workspace->active_stamp[root] == stamp) {
      continue;
    }
    if (!cluster_grows(workspace, root)) {
      continue;
    }
    workspace->active_stamp[root] = stamp;
    workspace->active_roots[active_count] = root;
    active_count += 1;
  }
  return active_count;
}

/* The clusters of the last event's list that still grow. A cluster that
 * only became odd this event fused with one that was already growing,
 * because parity is exclusive-or and the boundary is or, so its root is
 * reached from this list and no cluster is lost. */
static int32_t refresh_active_roots(struct workspace *workspace,
                                    int32_t active_count, int32_t stamp) {
  int32_t next_count = 0;
  for (int32_t position = 0; position < active_count; ++position) {
    int32_t root =
        find_root(workspace->parent, workspace->active_roots[position]);
    if (workspace->active_stamp[root] == stamp) {
      continue;
    }
    if (!cluster_grows(workspace, root)) {
      continue;
    }
    workspace->active_stamp[root] = stamp;
    workspace->next_active_roots[next_count] = root;
    next_count += 1;
  }
  memcpy(workspace->active_roots, workspace->next_active_roots,
         (size_t)next_count * sizeof(int32_t));
  return next_count;
}

/* The edges of the clusters that still grow, each once, with the ones
 * the growth has already closed dropped from the lists on the way. */
static int32_t collect_working_edges(struct workspace *workspace,
                                     const uint8_t *interval_is_closed,
                                     int32_t active_count, int32_t stamp) {
  int32_t working_count = 0;
  for (int32_t position = 0; position < active_count; ++position) {
    int32_t root = workspace->active_roots[position];
    int32_t previous = -1;
    int32_t entry = workspace->cluster_edge_head[root];
    while (entry >= 0) {
      int32_t next = workspace->cluster_edge_next[entry];
      int32_t edge = entry / 2;
      if (interval_is_closed[edge]) {
        unlink_entry(workspace, root, previous, next);
        entry = next;
        continue;
      }
      if (workspace->event_stamp[edge] != stamp) {
        workspace->event_stamp[edge] = stamp;
        workspace->working_edges[working_count] = edge;
        working_count += 1;
      }
      previous = entry;
      entry = next;
    }
  }
  return working_count;
}

static void freeze_roots(struct workspace *workspace,
                         const struct graph_arrays *graph,
                         int32_t working_count) {
  for (int32_t position = 0; position < working_count; ++position) {
    int32_t edge = workspace->working_edges[position];
    int32_t node_a =
        endpoint_node(graph->endpoint_a[edge], graph->detector_count);
    int32_t node_b =
        endpoint_node(graph->endpoint_b[edge], graph->detector_count);
    workspace->frozen_root_a[edge] = find_root(workspace->parent, node_a);
    workspace->frozen_root_b[edge] = find_root(workspace->parent, node_b);
  }
}

/* How many edges can still close, and in how few ticks the first of
 * them does. An edge inside one cluster is no candidate: it closes when
 * the cluster's own front meets itself and fuses nothing. */
static int32_t closing_candidates(struct workspace *workspace,
                                  int32_t working_count,
                                  const int64_t *interval_lower_tick,
                                  const int64_t *interval_upper_tick,
                                  int64_t *elapsed_ticks) {
  int32_t candidate_count = 0;
  *elapsed_ticks = 0;
  for (int32_t position = 0; position < working_count; ++position) {
    int32_t edge = workspace->working_edges[position];
    workspace->ticks_until_close[edge] = -1;
    int32_t left = workspace->frozen_root_a[edge];
    int32_t right = workspace->frozen_root_b[edge];
    if (left == right) {
      continue;
    }
    int64_t rate = cluster_grows(workspace, left);
    rate += cluster_grows(workspace, right);
    if (rate == 0) {
      continue;
    }
    int64_t remaining = interval_upper_tick[edge] - interval_lower_tick[edge];
    int64_t ticks = (remaining + rate - 1) / rate;
    workspace->ticks_until_close[edge] = ticks;
    if (candidate_count == 0 || ticks < *elapsed_ticks) {
      *elapsed_ticks = ticks;
    }
    candidate_count += 1;
  }
  return candidate_count;
}

static int32_t checked_interval(int64_t lower, int64_t upper, int64_t length) {
  if (lower < 0 || lower >= upper || upper > length) {
    return union_find_interval_order_lost;
  }
  return union_find_ok;
}

/* One edge whose ends stand in two clusters: each growing end covers
 * its own share of what is left. */
static int32_t advance_across_clusters(int64_t elapsed_ticks,
                                       int64_t closes_after, int32_t left_grows,
                                       int32_t right_grows, int64_t length,
                                       uint8_t *is_closed, int64_t *lower,
                                       int64_t *upper) {
  if (closes_after == elapsed_ticks) {
    *is_closed = 1;
    return union_find_ok;
  }
  if (left_grows) {
    *lower += elapsed_ticks;
  }
  if (right_grows) {
    *upper -= elapsed_ticks;
  }
  return checked_interval(*lower, *upper, length);
}

/* One edge with both ends in one cluster: its front grows inward from
 * both sides, so it covers two half ticks per tick. */
static int32_t advance_inside_cluster(int64_t elapsed_ticks, int32_t grows,
                                      int64_t length, uint8_t *is_closed,
                                      int64_t *lower, int64_t *upper) {
  if (!grows) {
    return union_find_ok;
  }
  int64_t remaining = *upper - *lower;
  if (2 * elapsed_ticks >= remaining) {
    *is_closed = 1;
    return union_find_ok;
  }
  *lower += elapsed_ticks;
  *upper -= elapsed_ticks;
  return checked_interval(*lower, *upper, length);
}

static int32_t advance_intervals(struct workspace *workspace,
                                 const struct graph_arrays *graph,
                                 int32_t working_count, int64_t elapsed_ticks,
                                 uint8_t *interval_is_closed,
                                 int64_t *interval_lower_tick,
                                 int64_t *interval_upper_tick) {
  for (int32_t position = 0; position < working_count; ++position) {
    int32_t edge = workspace->working_edges[position];
    int32_t left = workspace->frozen_root_a[edge];
    int32_t right = workspace->frozen_root_b[edge];
    int32_t left_grows = cluster_grows(workspace, left);
    int32_t right_grows = cluster_grows(workspace, right);
    int64_t length = graph->length_half_ticks[edge];
    int32_t status = union_find_ok;
    if (left == right) {
      status = advance_inside_cluster(
          elapsed_ticks, left_grows, length, &interval_is_closed[edge],
          &interval_lower_tick[edge], &interval_upper_tick[edge]);
    } else {
      status = advance_across_clusters(
          elapsed_ticks, workspace->ticks_until_close[edge], left_grows,
          right_grows, length, &interval_is_closed[edge],
          &interval_lower_tick[edge], &interval_upper_tick[edge]);
    }
    if (status != union_find_ok) {
      return status;
    }
  }
  return union_find_ok;
}

/* The edges that were already closed when the growth began. Every node
 * is still its own root here, so two ends differ exactly when their
 * nodes differ. */
static int32_t collect_closed_contacts(const struct graph_arrays *graph,
                                       const uint8_t *interval_is_closed,
                                       int32_t *contact_edges) {
  int32_t contact_count = 0;
  for (int32_t edge = 0; edge < graph->edge_count; ++edge) {
    if (!interval_is_closed[edge]) {
      continue;
    }
    int32_t node_a =
        endpoint_node(graph->endpoint_a[edge], graph->detector_count);
    int32_t node_b =
        endpoint_node(graph->endpoint_b[edge], graph->detector_count);
    if (node_a == node_b) {
      continue;
    }
    contact_edges[contact_count] = edge;
    contact_count += 1;
  }
  return contact_count;
}

static int32_t collect_closing_batch(const struct workspace *workspace,
                                     int32_t working_count,
                                     int64_t elapsed_ticks,
                                     int32_t *contact_edges,
                                     int32_t contact_count) {
  for (int32_t position = 0; position < working_count; ++position) {
    int32_t edge = workspace->working_edges[position];
    if (workspace->ticks_until_close[edge] != elapsed_ticks) {
      continue;
    }
    contact_edges[contact_count] = edge;
    contact_count += 1;
  }
  return contact_count;
}

/* One is returned when any fusion of the batch joined two odd clusters. */
static int32_t fuse_contact_batch(struct workspace *workspace,
                                  const struct graph_arrays *graph,
                                  const int32_t *contact_edges, int32_t start,
                                  int32_t end) {
  int32_t joins_two_odd = 0;
  for (int32_t position = start; position < end; ++position) {
    int32_t edge = contact_edges[position];
    int32_t node_a =
        endpoint_node(graph->endpoint_a[edge], graph->detector_count);
    int32_t node_b =
        endpoint_node(graph->endpoint_b[edge], graph->detector_count);
    joins_two_odd |= fuse_clusters(workspace, node_a, node_b);
  }
  return joins_two_odd;
}

static int32_t orders_by_length(const int64_t *length_half_ticks, int32_t left,
                                int32_t right) {
  if (length_half_ticks[left] != length_half_ticks[right]) {
    return length_half_ticks[left] < length_half_ticks[right];
  }
  return left < right;
}

static int32_t orders_by_edge(const int64_t *length_half_ticks, int32_t left,
                              int32_t right) {
  (void)length_half_ticks;
  return left < right;
}

typedef int32_t (*edge_order)(const int64_t *length_half_ticks, int32_t left,
                              int32_t right);

static void merge_runs(edge_order orders_before,
                       const int64_t *length_half_ticks, const int32_t *values,
                       int32_t *merged, int32_t start, int32_t middle,
                       int32_t end) {
  int32_t left = start;
  int32_t right = middle;
  int32_t out = start;
  while (left < middle && right < end) {
    if (orders_before(length_half_ticks, values[right], values[left])) {
      merged[out] = values[right];
      right += 1;
    } else {
      merged[out] = values[left];
      left += 1;
    }
    out += 1;
  }
  while (left < middle) {
    merged[out] = values[left];
    left += 1;
    out += 1;
  }
  while (right < end) {
    merged[out] = values[right];
    right += 1;
    out += 1;
  }
}

static void sort_edges(edge_order orders_before,
                       const int64_t *length_half_ticks, int32_t *values,
                       int32_t *merged, int32_t count) {
  for (int32_t width = 1; width < count; width *= 2) {
    for (int32_t start = 0; start < count; start += 2 * width) {
      int32_t middle = start + width;
      int32_t end = start + 2 * width;
      if (middle > count) {
        middle = count;
      }
      if (end > count) {
        end = count;
      }
      merge_runs(orders_before, length_half_ticks, values, merged, start,
                 middle, end);
    }
    memcpy(values, merged, (size_t)count * sizeof(int32_t));
  }
}

static void place_neighbor(struct workspace *workspace, int32_t node,
                           int32_t neighbor, int32_t edge) {
  int32_t slot = workspace->adjacency_cursor[node];
  workspace->adjacency_neighbor[slot] = neighbor;
  workspace->adjacency_edge[slot] = edge;
  workspace->adjacency_cursor[node] = slot + 1;
}

/* Each node's closed-edge degree, counted into adjacency_start. */
static void count_closed_neighbors(struct workspace *workspace,
                                   const struct graph_arrays *graph,
                                   const uint8_t *interval_is_closed) {
  for (int32_t edge = 0; edge < graph->edge_count; ++edge) {
    if (!interval_is_closed[edge]) {
      continue;
    }
    int32_t node_a =
        endpoint_node(graph->endpoint_a[edge], graph->detector_count);
    int32_t node_b =
        endpoint_node(graph->endpoint_b[edge], graph->detector_count);
    if (node_a == node_b) {
      continue;
    }
    workspace->adjacency_start[node_a + 1] += 1;
    workspace->adjacency_start[node_b + 1] += 1;
  }
}

static void place_closed_neighbors(struct workspace *workspace,
                                   const struct graph_arrays *graph,
                                   const uint8_t *interval_is_closed) {
  for (int32_t edge = 0; edge < graph->edge_count; ++edge) {
    if (!interval_is_closed[edge]) {
      continue;
    }
    int32_t node_a =
        endpoint_node(graph->endpoint_a[edge], graph->detector_count);
    int32_t node_b =
        endpoint_node(graph->endpoint_b[edge], graph->detector_count);
    if (node_a == node_b) {
      continue;
    }
    place_neighbor(workspace, node_a, node_b, edge);
    place_neighbor(workspace, node_b, node_a, edge);
  }
}

/* Each node's neighbours across the edges the growth has closed, laid
 * in the forest's adjacency arrays, which nothing reads until the
 * growth ends. A cluster identifier floods across exactly these edges,
 * one per stage (Helios 2301.08419 lines 623-629). */
static void build_closed_adjacency(struct workspace *workspace,
                                   const struct graph_arrays *graph,
                                   const uint8_t *interval_is_closed) {
  int32_t node_count = graph->detector_count + 1;
  for (int32_t node = 0; node <= node_count; ++node) {
    workspace->adjacency_start[node] = 0;
  }
  count_closed_neighbors(workspace, graph, interval_is_closed);
  for (int32_t node = 0; node < node_count; ++node) {
    workspace->adjacency_start[node + 1] += workspace->adjacency_start[node];
    workspace->adjacency_cursor[node] = workspace->adjacency_start[node];
  }
  place_closed_neighbors(workspace, graph, interval_is_closed);
}

/* Queue every unvisited closed-edge neighbour of node behind tail. */
static int32_t queue_closed_neighbors(struct workspace *workspace,
                                      int32_t node, int32_t depth,
                                      int32_t tail) {
  int32_t end = workspace->adjacency_start[node + 1];
  for (int32_t slot = workspace->adjacency_start[node]; slot < end; ++slot) {
    int32_t neighbor = workspace->adjacency_neighbor[slot];
    if (workspace->is_visited[neighbor]) {
      continue;
    }
    workspace->is_visited[neighbor] = 1;
    workspace->tree_order[neighbor] = depth + 1;
    workspace->node_stack[tail] = neighbor;
    tail += 1;
  }
  return tail;
}

/* The component of detectors reachable from start over closed edges,
 * appended to component_nodes from position count, and its lowest
 * node. The boundary node is every cluster's neighbour and no
 * cluster's member, so the walk stops at it. */
static int32_t collect_flood_component(struct workspace *workspace,
                                       int32_t start, int32_t boundary_node,
                                       int32_t *count) {
  int32_t head = 0;
  int32_t tail = 0;
  workspace->node_stack[tail] = start;
  tail += 1;
  workspace->is_visited[start] = 1;
  int32_t lowest = start;
  while (head < tail) {
    int32_t node = workspace->node_stack[head];
    head += 1;
    if (node == boundary_node) {
      continue;
    }
    workspace->component_nodes[*count] = node;
    *count += 1;
    if (node < lowest) {
      lowest = node;
    }
    tail = queue_closed_neighbors(workspace, node, 0, tail);
  }
  return lowest;
}

/* Hops from a cluster's lowest node, the identifier Helios floods from,
 * to its furthest member over closed edges. */
static int32_t flood_hops_from(struct workspace *workspace, int32_t root,
                               int32_t boundary_node) {
  int32_t head = 0;
  int32_t tail = 0;
  workspace->node_stack[tail] = root;
  tail += 1;
  workspace->is_visited[root] = 1;
  workspace->tree_order[root] = 0;
  int32_t deepest = 0;
  while (head < tail) {
    int32_t node = workspace->node_stack[head];
    head += 1;
    if (node == boundary_node) {
      continue;
    }
    int32_t depth = workspace->tree_order[node];
    if (depth > deepest) {
      deepest = depth;
    }
    tail = queue_closed_neighbors(workspace, node, depth, tail);
  }
  for (int32_t position = 0; position < tail; ++position) {
    workspace->is_visited[workspace->node_stack[position]] = 0;
  }
  return deepest;
}

/* Unmark the component members from position first, and the boundary. */
static void clear_visited(struct workspace *workspace, int32_t first,
                          int32_t end, int32_t boundary_node) {
  for (int32_t member = first; member < end; ++member) {
    workspace->is_visited[workspace->component_nodes[member]] = 0;
  }
  workspace->is_visited[boundary_node] = 0;
}

/* The deepest flood among the clusters one step's contacts fused. Each
 * component of detectors is flooded once, from its lowest node; the
 * boundary node joins clusters in the union-find without joining their
 * floods. */
static int32_t fused_flood_hops(struct workspace *workspace,
                                const struct graph_arrays *graph,
                                const uint8_t *interval_is_closed,
                                const int32_t *contact_edges, int32_t start,
                                int32_t end) {
  build_closed_adjacency(workspace, graph, interval_is_closed);
  int32_t boundary_node = graph->detector_count;
  int32_t member_count = 0;
  int32_t deepest = 0;
  for (int32_t position = start; position < end; ++position) {
    int32_t edge = contact_edges[position];
    int32_t node = endpoint_node(graph->endpoint_a[edge], boundary_node);
    if (node == boundary_node) {
      node = endpoint_node(graph->endpoint_b[edge], boundary_node);
    }
    if (node == boundary_node || workspace->is_visited[node]) {
      continue;
    }
    int32_t first_member = member_count;
    int32_t lowest = collect_flood_component(
        workspace, node, boundary_node, &member_count);
    clear_visited(workspace, first_member, member_count, boundary_node);
    int32_t hops = flood_hops_from(workspace, lowest, boundary_node);
    for (int32_t member = 0; member < member_count; ++member) {
      workspace->is_visited[workspace->component_nodes[member]] = 1;
    }
    if (hops > deepest) {
      deepest = hops;
    }
  }
  clear_visited(workspace, 0, member_count, boundary_node);
  return deepest;
}

/* Grow until no cluster is odd or no odd cluster has an edge left to
 * grow along. The second end is best effort: PECOS and ldpc peel what
 * there is rather than refuse. */
static int32_t grow_clusters(struct workspace *workspace,
                             const struct graph_arrays *graph,
                             uint8_t *interval_is_closed,
                             int64_t *interval_lower_tick,
                             int64_t *interval_upper_tick,
                             int32_t *contact_edges, int32_t *contact_count,
                             int32_t *step_edge_counts,
                             int32_t *step_hop_counts,
                             int64_t *step_growth_ticks,
                             uint8_t *step_odd_fusions, int32_t *step_count) {
  link_edge_entries(workspace, graph);
  *contact_count =
      collect_closed_contacts(graph, interval_is_closed, contact_edges);
  fuse_contact_batch(workspace, graph, contact_edges, 0, *contact_count);
  int32_t event_count = 0;
  int32_t active_count =
      initial_active_roots(workspace, graph->detector_count, 1);
  while (active_count > 0) {
    int32_t stamp = event_count + 1;
    int32_t working_count = collect_working_edges(
        workspace, interval_is_closed, active_count, stamp);
    sort_edges(orders_by_edge, graph->length_half_ticks,
               workspace->working_edges, workspace->working_scratch,
               working_count);
    freeze_roots(workspace, graph, working_count);
    int64_t elapsed_ticks = 0;
    int32_t candidate_count =
        closing_candidates(workspace, working_count, interval_lower_tick,
                           interval_upper_tick, &elapsed_ticks);
    if (candidate_count == 0) {
      return union_find_ok;
    }
    if (elapsed_ticks <= 0) {
      return union_find_no_positive_growth;
    }
    int32_t status = advance_intervals(
        workspace, graph, working_count, elapsed_ticks, interval_is_closed,
        interval_lower_tick, interval_upper_tick);
    if (status != union_find_ok) {
      return status;
    }
    int32_t batch_start = *contact_count;
    *contact_count = collect_closing_batch(workspace, working_count,
                                           elapsed_ticks, contact_edges,
                                           batch_start);
    int32_t joins_two_odd = fuse_contact_batch(
        workspace, graph, contact_edges, batch_start, *contact_count);
    event_count += 1;
    if (event_count > graph->edge_count) {
      return union_find_growth_bound_exceeded;
    }
    step_edge_counts[event_count - 1] = working_count;
    step_growth_ticks[event_count - 1] = elapsed_ticks;
    step_odd_fusions[event_count - 1] = (uint8_t)joins_two_odd;
    step_hop_counts[event_count - 1] =
        fused_flood_hops(workspace, graph, interval_is_closed, contact_edges,
                         batch_start, *contact_count);
    *step_count = event_count;
    active_count =
        refresh_active_roots(workspace, active_count, event_count + 1);
  }
  return union_find_ok;
}

/* A spanning forest of the contacts, lightest edges first (Kruskal), on
 * a set that carries no defects. */
static int32_t contact_forest(struct workspace *workspace,
                              const struct graph_arrays *graph,
                              const int32_t *contact_edges,
                              int32_t contact_count, int32_t *forest_edges) {
  reset_clusters(workspace, graph->detector_count, NULL);
  memcpy(workspace->sorted_contacts, contact_edges,
         (size_t)contact_count * sizeof(int32_t));
  sort_edges(orders_by_length, graph->length_half_ticks,
             workspace->sorted_contacts, workspace->merge_scratch,
             contact_count);
  int32_t forest_count = 0;
  for (int32_t position = 0; position < contact_count; ++position) {
    int32_t edge = workspace->sorted_contacts[position];
    int32_t node_a =
        endpoint_node(graph->endpoint_a[edge], graph->detector_count);
    int32_t node_b =
        endpoint_node(graph->endpoint_b[edge], graph->detector_count);
    if (find_root(workspace->parent, node_a) ==
        find_root(workspace->parent, node_b)) {
      continue;
    }
    union_nodes(workspace, node_a, node_b);
    forest_edges[forest_count] = edge;
    forest_count += 1;
  }
  return forest_count;
}

/* Each node's forest neighbours, in edge order, which is fault order. */
static void build_forest_adjacency(struct workspace *workspace,
                                   const struct graph_arrays *graph,
                                   const int32_t *forest_edges,
                                   int32_t forest_count) {
  int32_t node_count = graph->detector_count + 1;
  memset(workspace->is_forest_edge, 0, (size_t)graph->edge_count);
  for (int32_t position = 0; position < forest_count; ++position) {
    workspace->is_forest_edge[forest_edges[position]] = 1;
  }
  for (int32_t node = 0; node <= node_count; ++node) {
    workspace->adjacency_start[node] = 0;
  }
  for (int32_t edge = 0; edge < graph->edge_count; ++edge) {
    if (!workspace->is_forest_edge[edge]) {
      continue;
    }
    int32_t node_a =
        endpoint_node(graph->endpoint_a[edge], graph->detector_count);
    int32_t node_b =
        endpoint_node(graph->endpoint_b[edge], graph->detector_count);
    workspace->adjacency_start[node_a + 1] += 1;
    workspace->adjacency_start[node_b + 1] += 1;
  }
  for (int32_t node = 0; node < node_count; ++node) {
    workspace->adjacency_start[node + 1] += workspace->adjacency_start[node];
    workspace->adjacency_cursor[node] = workspace->adjacency_start[node];
  }
  for (int32_t edge = 0; edge < graph->edge_count; ++edge) {
    if (!workspace->is_forest_edge[edge]) {
      continue;
    }
    int32_t node_a =
        endpoint_node(graph->endpoint_a[edge], graph->detector_count);
    int32_t node_b =
        endpoint_node(graph->endpoint_b[edge], graph->detector_count);
    place_neighbor(workspace, node_a, node_b, edge);
    place_neighbor(workspace, node_b, node_a, edge);
  }
}

static int32_t collect_component(struct workspace *workspace, int32_t start) {
  int32_t component_count = 0;
  int32_t top = 0;
  workspace->is_visited[start] = 1;
  workspace->node_stack[top] = start;
  top += 1;
  while (top > 0) {
    top -= 1;
    int32_t node = workspace->node_stack[top];
    workspace->component_nodes[component_count] = node;
    component_count += 1;
    int32_t end = workspace->adjacency_start[node + 1];
    for (int32_t slot = workspace->adjacency_start[node]; slot < end; ++slot) {
      int32_t neighbor = workspace->adjacency_neighbor[slot];
      if (workspace->is_visited[neighbor]) {
        continue;
      }
      workspace->is_visited[neighbor] = 1;
      workspace->node_stack[top] = neighbor;
      top += 1;
    }
  }
  return component_count;
}

/* The tree's nodes from the root outward; parents filled on the way. A
 * node is adopted once, so the first neighbour in edge order is the
 * first child visited. */
static int32_t walk_tree(struct workspace *workspace, int32_t root) {
  int32_t order_count = 0;
  int32_t top = 0;
  workspace->has_parent[root] = 1;
  workspace->parent_node[root] = -1;
  workspace->node_stack[top] = root;
  top += 1;
  while (top > 0) {
    top -= 1;
    int32_t node = workspace->node_stack[top];
    workspace->tree_order[order_count] = node;
    order_count += 1;
    int32_t first = workspace->adjacency_start[node];
    for (int32_t slot = workspace->adjacency_start[node + 1] - 1; slot >= first;
         --slot) {
      int32_t neighbor = workspace->adjacency_neighbor[slot];
      if (workspace->has_parent[neighbor]) {
        continue;
      }
      workspace->has_parent[neighbor] = 1;
      workspace->parent_node[neighbor] = node;
      workspace->parent_edge[neighbor] = workspace->adjacency_edge[slot];
      workspace->node_stack[top] = neighbor;
      top += 1;
    }
  }
  return order_count;
}

/* Each node's level below the root of its tree, the root at zero, and
 * the deepest of them returned. walk_tree leaves a parent before its
 * children in tree_order, so one pass over that order fills them. */
static int32_t tree_depth(struct workspace *workspace, int32_t order_count) {
  int32_t deepest = 0;
  for (int32_t position = 0; position < order_count; ++position) {
    int32_t node = workspace->tree_order[position];
    int32_t parent = workspace->parent_node[node];
    int32_t level = 0;
    if (parent >= 0) {
      level = workspace->node_level[parent] + 1;
    }
    workspace->node_level[node] = level;
    if (level > deepest) {
      deepest = level;
    }
  }
  return deepest;
}

/* Peel one tree leaf to root: a node that still carries a defect
 * selects its parent edge and flips its parent. An odd component
 * without the boundary leaves its root's defect behind, and the caller
 * reports that detector as unmatched. */
static int32_t peel_tree(struct workspace *workspace,
                         int32_t component_count, int32_t root,
                         const uint8_t *residual_syndrome,
                         int32_t boundary_node, uint8_t *selected_edges) {
  for (int32_t position = 0; position < component_count; ++position) {
    int32_t node = workspace->component_nodes[position];
    workspace->residual_defect[node] = 0;
    if (node != boundary_node) {
      workspace->residual_defect[node] = residual_syndrome[node];
    }
  }
  int32_t order_count = walk_tree(workspace, root);
  for (int32_t position = order_count - 1; position >= 1; --position) {
    int32_t node = workspace->tree_order[position];
    if (!workspace->residual_defect[node]) {
      continue;
    }
    selected_edges[workspace->parent_edge[node]] = 1;
    workspace->residual_defect[workspace->parent_node[node]] ^= 1;
  }
  return tree_depth(workspace, order_count);
}

static int32_t component_carries_a_defect(const struct workspace *workspace,
                                          int32_t component_count,
                                          const uint8_t *residual_syndrome,
                                          int32_t boundary_node) {
  for (int32_t position = 0; position < component_count; ++position) {
    int32_t node = workspace->component_nodes[position];
    if (node == boundary_node) {
      continue;
    }
    if (residual_syndrome[node]) {
      return 1;
    }
  }
  return 0;
}

static int32_t component_holds_the_boundary(const struct workspace *workspace,
                                            int32_t component_count,
                                            int32_t boundary_node) {
  for (int32_t position = 0; position < component_count; ++position) {
    if (workspace->component_nodes[position] == boundary_node) {
      return 1;
    }
  }
  return 0;
}

/* Every component is met from its smallest node, which is its root
 * unless the boundary is in it. A component with no defect is skipped,
 * and the deepest tree the peel walks is returned. */
static int32_t peel_forest(struct workspace *workspace,
                           const struct graph_arrays *graph,
                           const uint8_t *residual_syndrome,
                           const int32_t *forest_edges, int32_t forest_count,
                           uint8_t *selected_edges) {
  build_forest_adjacency(workspace, graph, forest_edges, forest_count);
  int32_t node_count = graph->detector_count + 1;
  int32_t boundary_node = graph->detector_count;
  int32_t deepest = 0;
  for (int32_t start = 0; start < node_count; ++start) {
    if (workspace->is_visited[start]) {
      continue;
    }
    int32_t component_count = collect_component(workspace, start);
    if (!component_carries_a_defect(workspace, component_count,
                                    residual_syndrome, boundary_node)) {
      continue;
    }
    int32_t root = start;
    if (component_holds_the_boundary(workspace, component_count,
                                     boundary_node)) {
      root = boundary_node;
    }
    int32_t depth = peel_tree(workspace, component_count, root,
                              residual_syndrome, boundary_node,
                              selected_edges);
    if (depth > deepest) {
      deepest = depth;
    }
  }
  return deepest;
}

int32_t union_find_decode(
    int32_t detector_count, int32_t edge_count, const int32_t *endpoint_a,
    const int32_t *endpoint_b, const int64_t *length_half_ticks,
    const uint8_t *residual_syndrome, uint8_t *selected_edges,
    uint8_t *interval_is_closed, int64_t *interval_lower_tick,
    int64_t *interval_upper_tick, int32_t *contact_edges,
    int32_t *contact_count, int32_t *forest_edges, int32_t *forest_count,
    int32_t *step_edge_counts, int32_t *step_hop_counts,
    int64_t *step_growth_ticks, uint8_t *step_odd_fusions,
    int32_t *step_count, int32_t *forest_depth) {
  struct graph_arrays graph;
  graph.detector_count = detector_count;
  graph.edge_count = edge_count;
  graph.endpoint_a = endpoint_a;
  graph.endpoint_b = endpoint_b;
  graph.length_half_ticks = length_half_ticks;
  *contact_count = 0;
  *forest_count = 0;
  *step_count = 0;
  *forest_depth = 0;
  struct workspace workspace;
  if (!take_workspace(&workspace, detector_count + 1, edge_count)) {
    release_workspace(&workspace);
    return union_find_out_of_memory;
  }
  memset(selected_edges, 0, (size_t)edge_count);
  initial_intervals(&graph, interval_is_closed, interval_lower_tick,
                    interval_upper_tick);
  reset_clusters(&workspace, detector_count, residual_syndrome);
  int32_t status = grow_clusters(
      &workspace, &graph, interval_is_closed, interval_lower_tick,
      interval_upper_tick, contact_edges, contact_count, step_edge_counts,
      step_hop_counts, step_growth_ticks, step_odd_fusions, step_count);
  if (status != union_find_ok) {
    release_workspace(&workspace);
    return status;
  }
  *forest_count = contact_forest(&workspace, &graph, contact_edges,
                                 *contact_count, forest_edges);
  *forest_depth = peel_forest(&workspace, &graph, residual_syndrome,
                              forest_edges, *forest_count, selected_edges);
  release_workspace(&workspace);
  return union_find_ok;
}
