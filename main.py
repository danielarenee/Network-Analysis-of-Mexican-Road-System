"""
Road Network Analysis

takes two different types of input:

1. "osmnx"  : OpenStreetMap data downloaded on the fly via osmnx.
CVEGEO locality codes are assigned through a spatial join 
with an INEGI shapefile.

2. "inegi"  : Official Red Nacional de Caminos data, loaded from a
pre-built NetworkX pickle. CVEGEO codes are already stored
on the nodes as 'id_polygon'.

operations performed:
  1. load geographic data and road network
  2. Identify boundary nodes between different localities (CVEGEO regions)
  3. Build clique graphs representing locality connections
  4. Perform Delaunay triangulation on locality centroids
  5. Simplify road networks by pruning low-degree nodes
  6. Compute inter-region distance matrix between boundary nodes
"""
#%%

import pickle
import time
import networkx as nx
from pathlib import Path
import src.utils as fc

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent

# CONSTANTS

SOURCE = "inegi"  # "osmnx" or "inegi"
KEEP_LARGER_CC = True
CODE_NAME = "CVEGEO"

#  OSMnx settings 
SHAPEFILE_PATH = BASE_DIR / "data" / "raw" / "shp" / "27l.shp"
CENTER_LAT = 17.930714
CENTER_LON = -93.507545
NETWORK_RADIUS = 7000  # meters

# INEGI settings 
INEGI_GRAPH_PATH = BASE_DIR / "data" / "processed" / "road_network.pkl"


#%%
# DATA LOADING AND PREPROCESSING

print(f"[1/5] Loading graph (source={SOURCE!r})...")
t0 = time.time()

graph, gdf_nodes_labeled, gdf_localities, cvegeo_map, CRS, PLOT_MARGIN = fc.load_and_preprocess_graph(
    source=SOURCE,
    shapefile_path=SHAPEFILE_PATH,
    center_lat=CENTER_LAT,
    center_lon=CENTER_LON,
    network_radius=NETWORK_RADIUS,
    inegi_graph_path=INEGI_GRAPH_PATH,
)
# Keep only the major connected component
if KEEP_LARGER_CC:
    cc = nx.weakly_connected_components(graph)
    larger_cc_nodes = max(cc, key=len)
    graph = graph.subgraph(larger_cc_nodes).copy()
    
print(f"    Graph loaded: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges ({time.time()-t0:.1f}s)")

connected = nx.is_weakly_connected(graph)
print(f"    Is (weakly) connected?: {connected}")

external_nodes = [
    node for node, idx in graph.nodes(data=CODE_NAME) if idx is None
]
print(f"    External nodes: {len(external_nodes):,}, Internal nodes: {graph.number_of_nodes()-len(external_nodes):,}")


# Visualization
fc.plot_labeled_network(graph, gdf_nodes_labeled, gdf_localities, source=SOURCE)

# BOUNDARY NODE IDENTIFICATION

print(f"[2/5] Identifying boundary nodes...")
t0 = time.time()

boundary_nodes_by_locality = fc.identify_boundary_nodes(graph, cvegeo_map)

total_boundary = sum(len(v) for v in boundary_nodes_by_locality.values())
print(f"    {len(boundary_nodes_by_locality):,} localities, {total_boundary:,} boundary nodes total ({time.time()-t0:.1f}s)")


# CLIQUE GRAPH CONSTRUCTION

print(f"[3/5] Building locality cliques (this may take several minutes)...")
t0 = time.time()

reduced_graph = fc.build_reduced_clique_graph(graph, boundary_nodes_by_locality)

print(f"    Reduced graph: {reduced_graph.number_of_nodes():,} nodes, {reduced_graph.number_of_edges():,} edges ({time.time()-t0:.1f}s)")

# --- Visualization ---
fc.plot_boundary_nodes_network(reduced_graph, gdf_localities, PLOT_MARGIN)


# ITERATIVE GRAPH SIMPLIFICATION

print(f"[4/5] Simplifying graph iteratively...")
t0 = time.time()

simplified_graph, num_iterations = fc.simplify_iteratively(graph)

print(f"    Converged in {num_iterations} iterations → {simplified_graph.number_of_nodes():,} nodes, {simplified_graph.number_of_edges():,} edges ({time.time()-t0:.1f}s)")

connected = nx.is_weakly_connected(simplified_graph)
print(f"    Is connected?: {connected}")

external_nodes = [
    node for node, idx in simplified_graph.nodes(data=CODE_NAME) if idx is None
]
print(f"    External nodes: {len(external_nodes):,}, Internal nodes: {simplified_graph.number_of_nodes()-len(external_nodes):,}")

fc.plot_simplified_graph(simplified_graph, gdf_nodes_labeled, gdf_localities, num_iterations)

# SINTER-REGION DISTANCE MATRIX 

"""
print(f"[5/5] Computing inter-region distance matrix (this may take a while)...")
t0 = time.time()

distance_matrix, node_to_region = fc.calculate_border_nodes_distance_matrix(
    graph, boundary_nodes_by_locality
)

print(f"    Done ({time.time()-t0:.1f}s)")

fc.print_distance_matrix_samples(distance_matrix, node_to_region)

# Save the computed distance matrix and node mapping to disk
matrix_save_path = BASE_DIR / "data" / "processed" / "distance_matrix.pkl"
with open(matrix_save_path, "wb") as f:
    pickle.dump({
        "distance_matrix": distance_matrix,
        "node_to_region": node_to_region
    }, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\n[Saved] Distance matrix saved to {matrix_save_path}")
"""