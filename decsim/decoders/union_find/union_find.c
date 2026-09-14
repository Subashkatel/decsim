/* The weighted Union-Find decoder: growth, contact forest and peeling.
 *
 * The contract is in union_find.h. Growth is event driven: each event
 * jumps to the next tick at which some front closes an edge, and a
 * front covers one half tick per tick, so an edge between two growing
 * clusters loses two half ticks per tick and an edge with one growing
 * end loses one.
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

/* Everything one decode needs beyond the caller's buffers. The cluster
 * arrays are indexed by node, the frozen and candidate arrays by edge,
 * and the peeling arrays by node again. */
struct workspace {
  int32_t *parent;
  uint8_t *parity;
  uint8_t *touches_boundary;
  uint8_t *is_active_root;
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
  uint8_t *has_parent;
  uint8_t *residual_defect;
  uint8_t *is_visited;
  int32_t *component_nodes;
};

static void *allocate_array(int32_t count, size_t size) {
  size_t wanted = (size_t)count;
  if (wanted == 0) {
    wanted = 1;
  }
  return calloc(wanted, size);
}

static void release_workspace(struct workspace *workspace) {
  free(workspace->parent);
  free(workspace->parity);
  free(workspace->touches_boundary);
  free(workspace->is_active_root);
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
  int32_t slot_count = 2 * node_count;
  memset(workspace, 0, sizeof(*workspace));
  workspace->parent = allocate_array(node_count, sizeof(int32_t));
  workspace->parity = allocate_array(node_count, sizeof(uint8_t));
  workspace->touches_boundary = allocate_array(node_count, sizeof(uint8_t));
  workspace->is_active_root = allocate_array(node_count, sizeof(uint8_t));
  workspace->frozen_root_a = allocate_array(edge_count, sizeof(int32_t));
  workspace->frozen_root_b = allocate_array(edge_count, sizeof(int32_t));
  workspace->ticks_until_close = allocate_array(edge_count, sizeof(int64_t));
  workspace->sorted_contacts = allocate_array(edge_count, sizeof(int32_t));
  workspace->merge_scratch = allocate_array(edge_count, sizeof(int32_t));
  workspace->is_forest_edge = allocate_array(edge_count, sizeof(uint8_t));
  workspace->adjacency_start = allocate_array(node_count + 1, sizeof(int32_t));
  workspace->adjacency_cursor = allocate_array(node_count, sizeof(int32_t));
  workspace->adjacency_neighbor = allocate_array(slot_count, sizeof(int32_t));
  workspace->adjacency_edge = allocate_array(slot_count, sizeof(int32_t));
  workspace->node_stack = allocate_array(node_count, sizeof(int32_t));
  workspace->tree_order = allocate_array(node_count, sizeof(int32_t));
  workspace->parent_node = allocate_array(node_count, sizeof(int32_t));
  workspace->parent_edge = allocate_array(node_count, sizeof(int32_t));
  workspace->has_parent = allocate_array(node_count, sizeof(uint8_t));
  workspace->residual_defect = allocate_array(node_count, sizeof(uint8_t));
  workspace->is_visited = allocate_array(node_count, sizeof(uint8_t));
  workspace->component_nodes = allocate_array(node_count, sizeof(int32_t));
  if (workspace->parent == NULL || workspace->parity == NULL ||
      workspace->touches_boundary == NULL ||
      workspace->is_active_root == NULL || workspace->frozen_root_a == NULL ||
      workspace->frozen_root_b == NULL ||
      workspace->ticks_until_close == NULL ||
      workspace->sorted_contacts == NULL || workspace->merge_scratch == NULL ||
      workspace->is_forest_edge == NULL ||
      workspace->adjacency_start == NULL ||
      workspace->adjacency_cursor == NULL ||
      workspace->adjacency_neighbor == NULL ||
      workspace->adjacency_edge == NULL || workspace->node_stack == NULL ||
      workspace->tree_order == NULL || workspace->parent_node == NULL ||
      workspace->parent_edge == NULL || workspace->has_parent == NULL ||
      workspace->residual_defect == NULL || workspace->is_visited == NULL ||
      workspace->component_nodes == NULL) {
    return 0;
  }
  return 1;
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

/* The smaller root index survives, and parity and the shared boundary
 * pass to it. */
static void union_nodes(struct workspace *workspace, int32_t left,
                        int32_t right) {
  int32_t left_root = find_root(workspace->parent, left);
  int32_t right_root = find_root(workspace->parent, right);
  if (left_root == right_root) {
    return;
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

static void freeze_roots(struct workspace *workspace,
                         const struct graph_arrays *graph) {
  for (int32_t edge = 0; edge < graph->edge_count; ++edge) {
    int32_t node_a =
        endpoint_node(graph->endpoint_a[edge], graph->detector_count);
    int32_t node_b =
        endpoint_node(graph->endpoint_b[edge], graph->detector_count);
    workspace->frozen_root_a[edge] = find_root(workspace->parent, node_a);
    workspace->frozen_root_b[edge] = find_root(workspace->parent, node_b);
  }
}

/* An odd cluster that does not touch the boundary keeps growing. */
static int32_t freeze_activity(struct workspace *workspace,
                               int32_t detector_count) {
  int32_t node_count = detector_count + 1;
  int32_t any_active = 0;
  for (int32_t node = 0; node < node_count; ++node) {
    int32_t root = find_root(workspace->parent, node);
    uint8_t active = 0;
    if (!workspace->touches_boundary[root] && workspace->parity[root] == 1) {
      active = 1;
    }
    workspace->is_active_root[root] = active;
    if (active) {
      any_active = 1;
    }
  }
  return any_active;
}

/* How many edges can still close, and in how few ticks the first of
 * them does. An edge inside one cluster is no candidate: it closes when
 * the cluster's own front meets itself and fuses nothing. */
static int32_t closing_candidates(struct workspace *workspace,
                                  const struct graph_arrays *graph,
                                  const uint8_t *interval_is_closed,
                                  const int64_t *interval_lower_tick,
                                  const int64_t *interval_upper_tick,
                                  int64_t *elapsed_ticks) {
  int32_t candidate_count = 0;
  *elapsed_ticks = 0;
  for (int32_t edge = 0; edge < graph->edge_count; ++edge) {
    workspace->ticks_until_close[edge] = -1;
    if (interval_is_closed[edge]) {
      continue;
    }
    int32_t left = workspace->frozen_root_a[edge];
    int32_t right = workspace->frozen_root_b[edge];
    if (left == right) {
      continue;
    }
    int64_t rate = workspace->is_active_root[left];
    rate += workspace->is_active_root[right];
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
                                       int64_t closes_after, uint8_t left_grows,
                                       uint8_t right_grows, int64_t length,
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
static int32_t advance_inside_cluster(int64_t elapsed_ticks, uint8_t grows,
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
                                 int64_t elapsed_ticks,
                                 uint8_t *interval_is_closed,
                                 int64_t *interval_lower_tick,
                                 int64_t *interval_upper_tick) {
  for (int32_t edge = 0; edge < graph->edge_count; ++edge) {
    if (interval_is_closed[edge]) {
      continue;
    }
    int32_t left = workspace->frozen_root_a[edge];
    int32_t right = workspace->frozen_root_b[edge];
    uint8_t left_grows = workspace->is_active_root[left];
    uint8_t right_grows = workspace->is_active_root[right];
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
                                     const struct graph_arrays *graph,
                                     int64_t elapsed_ticks,
                                     int32_t *contact_edges,
                                     int32_t contact_count) {
  for (int32_t edge = 0; edge < graph->edge_count; ++edge) {
    if (workspace->ticks_until_close[edge] != elapsed_ticks) {
      continue;
    }
    contact_edges[contact_count] = edge;
    contact_count += 1;
  }
  return contact_count;
}

static void union_contact_batch(struct workspace *workspace,
                                const struct graph_arrays *graph,
                                const int32_t *contact_edges, int32_t start,
                                int32_t end) {
  for (int32_t position = start; position < end; ++position) {
    int32_t edge = contact_edges[position];
    int32_t node_a =
        endpoint_node(graph->endpoint_a[edge], graph->detector_count);
    int32_t node_b =
        endpoint_node(graph->endpoint_b[edge], graph->detector_count);
    union_nodes(workspace, node_a, node_b);
  }
}

/* Grow until no cluster is odd or no odd cluster has an edge left to
 * grow along. The second end is best effort: PECOS and ldpc peel what
 * there is rather than refuse. */
static int32_t grow_clusters(struct workspace *workspace,
                             const struct graph_arrays *graph,
                             uint8_t *interval_is_closed,
                             int64_t *interval_lower_tick,
                             int64_t *interval_upper_tick,
                             int32_t *contact_edges, int32_t *contact_count) {
  *contact_count =
      collect_closed_contacts(graph, interval_is_closed, contact_edges);
  union_contact_batch(workspace, graph, contact_edges, 0, *contact_count);
  int32_t event_count = 0;
  while (1) {
    freeze_roots(workspace, graph);
    if (!freeze_activity(workspace, graph->detector_count)) {
      return union_find_ok;
    }
    int64_t elapsed_ticks = 0;
    int32_t candidate_count = closing_candidates(
        workspace, graph, interval_is_closed, interval_lower_tick,
        interval_upper_tick, &elapsed_ticks);
    if (candidate_count == 0) {
      return union_find_ok;
    }
    if (elapsed_ticks <= 0) {
      return union_find_no_positive_growth;
    }
    int32_t status =
        advance_intervals(workspace, graph, elapsed_ticks, interval_is_closed,
                          interval_lower_tick, interval_upper_tick);
    if (status != union_find_ok) {
      return status;
    }
    int32_t batch_start = *contact_count;
    *contact_count = collect_closing_batch(workspace, graph, elapsed_ticks,
                                           contact_edges, batch_start);
    union_contact_batch(workspace, graph, contact_edges, batch_start,
                        *contact_count);
    event_count += 1;
    if (event_count > graph->edge_count) {
      return union_find_growth_bound_exceeded;
    }
  }
}

static int32_t orders_before(const int64_t *length_half_ticks, int32_t left,
                             int32_t right) {
  if (length_half_ticks[left] != length_half_ticks[right]) {
    return length_half_ticks[left] < length_half_ticks[right];
  }
  return left < right;
}

static void merge_runs(const int64_t *length_half_ticks, const int32_t *values,
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

/* Lightest edge first, ties broken by the edge index, which is the
 * fault order Kruskal's forest is pinned to. */
static void sort_by_length_then_edge(const int64_t *length_half_ticks,
                                     int32_t *values, int32_t *merged,
                                     int32_t count) {
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
      merge_runs(length_half_ticks, values, merged, start, middle, end);
    }
    memcpy(values, merged, (size_t)count * sizeof(int32_t));
  }
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
  sort_by_length_then_edge(graph->length_half_ticks,
                           workspace->sorted_contacts,
                           workspace->merge_scratch, contact_count);
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

static void place_neighbor(struct workspace *workspace, int32_t node,
                           int32_t neighbor, int32_t edge) {
  int32_t slot = workspace->adjacency_cursor[node];
  workspace->adjacency_neighbor[slot] = neighbor;
  workspace->adjacency_edge[slot] = edge;
  workspace->adjacency_cursor[node] = slot + 1;
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
    for (int32_t slot = workspace->adjacency_start[node + 1] - 1;
         slot >= first; --slot) {
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

/* Peel one tree leaf to root: a node that still carries a defect
 * selects its parent edge and flips its parent. An odd component
 * without the boundary leaves its root's defect behind, and the caller
 * reports that detector as unmatched. */
static void peel_tree(struct workspace *workspace, int32_t component_count,
                      int32_t root, const uint8_t *residual_syndrome,
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
 * unless the boundary is in it. A component with no defect is skipped. */
static void peel_forest(struct workspace *workspace,
                        const struct graph_arrays *graph,
                        const uint8_t *residual_syndrome,
                        const int32_t *forest_edges, int32_t forest_count,
                        uint8_t *selected_edges) {
  build_forest_adjacency(workspace, graph, forest_edges, forest_count);
  int32_t node_count = graph->detector_count + 1;
  int32_t boundary_node = graph->detector_count;
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
    peel_tree(workspace, component_count, root, residual_syndrome,
              boundary_node, selected_edges);
  }
}

int32_t union_find_decode(
    int32_t detector_count, int32_t edge_count, const int32_t *endpoint_a,
    const int32_t *endpoint_b, const int64_t *length_half_ticks,
    const uint8_t *residual_syndrome, uint8_t *selected_edges,
    uint8_t *interval_is_closed, int64_t *interval_lower_tick,
    int64_t *interval_upper_tick, int32_t *contact_edges,
    int32_t *contact_count, int32_t *forest_edges, int32_t *forest_count) {
  struct graph_arrays graph;
  graph.detector_count = detector_count;
  graph.edge_count = edge_count;
  graph.endpoint_a = endpoint_a;
  graph.endpoint_b = endpoint_b;
  graph.length_half_ticks = length_half_ticks;
  *contact_count = 0;
  *forest_count = 0;
  struct workspace workspace;
  if (!take_workspace(&workspace, detector_count + 1, edge_count)) {
    release_workspace(&workspace);
    return union_find_out_of_memory;
  }
  memset(selected_edges, 0, (size_t)edge_count);
  initial_intervals(&graph, interval_is_closed, interval_lower_tick,
                    interval_upper_tick);
  reset_clusters(&workspace, detector_count, residual_syndrome);
  int32_t status =
      grow_clusters(&workspace, &graph, interval_is_closed,
                    interval_lower_tick, interval_upper_tick, contact_edges,
                    contact_count);
  if (status != union_find_ok) {
    release_workspace(&workspace);
    return status;
  }
  *forest_count = contact_forest(&workspace, &graph, contact_edges,
                                 *contact_count, forest_edges);
  peel_forest(&workspace, &graph, residual_syndrome, forest_edges,
              *forest_count, selected_edges);
  release_workspace(&workspace);
  return union_find_ok;
}
