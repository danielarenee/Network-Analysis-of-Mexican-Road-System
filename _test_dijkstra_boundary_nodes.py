"""
Run Dijkstra all-against-all over the boundary/clique nodes, using the
Road_Network object's simplify()+boundary classification, but parallelized
across processes since each source node's Dijkstra run is independent.

We can't just hand a bound Road_Network method to ProcessPoolExecutor: the
object carries geopandas/shapely attributes that aren't cheap (or guaranteed)
to pickle per-task. Instead we pull out the two things Dijkstra actually
needs - the igraph graph and the node_id<->igraph-index mapping - pickle
those ONCE per worker (via the pool initializer), and reimplement the same
heap-Dijkstra loop as a standalone function that runs entirely inside each
worker process.
"""

import heapq
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from road_network import Road_Network

BASE_DIR = Path(__file__).resolve().parent

SOURCE = "inegi"
ENT = "27"
WEIGHT = "length"
N_WORKERS = 8

source_kwargs_inegi = {
    "inegi_graph_path": BASE_DIR / "data" / "processed" / f"road_network_{ENT}.pkl"
}


# ----------------------------------------------------------------------
# Worker-side state + functions (run in each subprocess)
# ----------------------------------------------------------------------
_G = None
_NODE_TO_IG = None


def _init_worker(igraph_obj, node_to_ig):
    """Runs once per worker process when the pool starts."""
    global _G, _NODE_TO_IG
    _G = igraph_obj
    _NODE_TO_IG = node_to_ig


def _dijkstra_single_source(source, target_ig_ids, weight):
    """Same heap-Dijkstra as Road_Network.dijkstra, for one source."""
    g = _G
    source_ig = _NODE_TO_IG[source]

    dist = {source_ig: 0.0}
    prev = {source_ig: None}
    visited = set()
    pending = set(target_ig_ids)
    heap = [(0.0, source_ig)]

    while heap and pending:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        pending.discard(u)

        for v in g.neighbors(u):
            edge_id = g.get_eid(u, v)
            w = g.es[edge_id][weight]
            new_dist = d + w
            if new_dist < dist.get(v, float("inf")):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    distances = {}
    paths = {}
    for target_ig in target_ig_ids:
        if target_ig not in dist:
            continue
        path_ig = []
        node = target_ig
        while node is not None:
            path_ig.append(node)
            node = prev[node]
        path_ig.reverse()

        target_id = g.vs[target_ig]["node_id"]
        distances[target_id] = dist[target_ig]
        paths[target_id] = [g.vs[i]["node_id"] for i in path_ig]

    return source, distances, paths


def _dijkstra_chunk(sources_chunk, target_ig_ids, weight):
    """One task = a batch of sources, to amortize per-task overhead."""
    results = {}
    for source in sources_chunk:
        _, distances, paths = _dijkstra_single_source(source, target_ig_ids, weight)
        results[source] = (distances, paths)
    return results


def _chunk(seq, n_chunks):
    size = max(1, -(-len(seq) // n_chunks))  # ceil division
    return [seq[i:i + size] for i in range(0, len(seq), size)]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":

    # 1. LOAD + SIMPLIFY (protecting boundary nodes)
    print(f"[1/3] Loading and simplifying graph (source={SOURCE!r})...")
    t0 = time.time()

    road = Road_Network(
        source_kwargs_inegi,
        source = SOURCE,
        id_city_label = "CVEGEO",
        length_attr = "length",
        keep_larger_cc = True,
        to_undirected = True,
        to_simple = True
    )
    simplified, num_iterations, simplify_time = road.simplify(protect_boundary_nodes=True)
    print(f"    {simplified.n:,} nodes, {simplified.m:,} edges, "
          f"{num_iterations} iterations ({time.time()-t0:.1f}s)")

    boundary_nodes = list(simplified.all_boundary_nodes)
    print(f"    Boundary (clique) nodes: {len(boundary_nodes):,} "
          f"across {len(simplified.boundary_nodes):,} localities")

    # 2. PARALLEL ALL-AGAINST-ALL DIJKSTRA OVER BOUNDARY NODES
    print(f"[2/3] Running Dijkstra all-against-all over boundary nodes "
          f"with {N_WORKERS} worker processes...")
    t0 = time.time()

    ig_graph = simplified._graph(kind="ig")
    node_to_ig = simplified.node_to_ig
    target_ig_ids = {node_to_ig[n] for n in boundary_nodes}

    chunks = _chunk(boundary_nodes, N_WORKERS)

    distances = {}
    paths = {}
    with ProcessPoolExecutor(
        max_workers = N_WORKERS,
        initializer = _init_worker,
        initargs = (ig_graph, node_to_ig),
    ) as pool:
        futures = [
            pool.submit(_dijkstra_chunk, chunk, target_ig_ids, WEIGHT)
            for chunk in chunks
        ]
        for future in futures:
            chunk_results = future.result()
            for source, (source_distances, source_paths) in chunk_results.items():
                distances[source] = source_distances
                paths[source] = source_paths

    elapsed = time.time() - t0
    total_pairs = sum(len(targets) for targets in distances.values())
    possible_pairs = len(boundary_nodes) ** 2  # targets include the source itself
    print(f"    Computed {total_pairs:,} / {possible_pairs:,} reachable ordered pairs "
          f"({elapsed:.1f}s)")

    # 3. SAMPLE OUTPUT
    print("[3/3] Sample output:")
    first_source = boundary_nodes[0]
    first_targets = distances[first_source]
    print(f"    First 5 distances from boundary node {first_source}:")
    for target, dist in list(first_targets.items())[:5]:
        print(f"        → node {target}: {dist:.2f} m")
