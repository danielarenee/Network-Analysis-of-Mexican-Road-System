"""
simplify the road network before identifying boundary/region nodes

"""

import time
from pathlib import Path

from road_network import Road_Network

BASE_DIR = Path(__file__).resolve().parent

SOURCE = "inegi"
ENT = "27"

source_kwargs_inegi = {
    "inegi_graph_path": BASE_DIR / "data" / "processed" / f"road_network_{ENT}.pkl"
}

# 1. LOAD RAW GRAPH
print(f"[1/4] Loading graph (source={SOURCE!r})...")
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
print(f"    Graph loaded: {road.n:,} nodes, {road.m:,} edges ({time.time()-t0:.1f}s)")

# 2. SIMPLIFY FIRST
print("[2/4] Simplifying graph iteratively (boundaries not protected)...")
t0 = time.time()

simplified_graph, num_iterations = road.simplify(protect_boundary_nodes=False)

print(f"    Converged in {num_iterations} iterations → {simplified_graph.n:,} nodes, {simplified_graph.m:,} edges ({time.time()-t0:.1f}s)")

# 3. CLASSIFY BOUNDARY/REGION NODES ON THE SIMPLIFIED GRAPH
print("[3/4] Boundary/region nodes on the simplified graph:")
print(f"    External nodes: {simplified_graph.n_external:,}")
print(f"    Internal nodes: {simplified_graph.n_internal:,}")
print(f"    Boundary nodes: {simplified_graph.n_boundary:,}")
print(f"    Inner nodes: {simplified_graph.n_inner:,}")
print(f"    Localities: {len(simplified_graph.boundary_nodes):,}")

simplified_graph.plot_labeled_network(
    f"Simplified-first Road Network (ENT {ENT}, {num_iterations} iterations)"
)

# 4. BUILD REDUCED CLIQUE GRAPH 
print("[4/4] Building locality cliques from post-simplification boundaries...")
t0 = time.time()

simplified_graph.reduce_city_subraphs()
print(f"    Reduced graph: {simplified_graph.reduced_graph.order():,} nodes, "
      f"{simplified_graph.reduced_graph.size():,} edges ({time.time()-t0:.1f}s)")

simplified_graph.plot_boundary_nodes_network(
    title="Boundary Node Network (simplified first)"
)
