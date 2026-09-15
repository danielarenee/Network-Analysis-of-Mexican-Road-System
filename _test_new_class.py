# Packages
import time
import networkx as nx
from pathlib import Path

from road_network import Road_Network

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent
TESTS_DIR = BASE_DIR / "tests"

# CONSTANTS
SOURCE = "inegi"
POLYGONS_PATH = BASE_DIR / "data" / "raw" / "shp" / "27l.shp"

# INEGI settings 
source_kwargs_inegi = {
    "inegi_graph_path": BASE_DIR / "data" / "processed" / "road_network.pkl"
    }

# DATA LOADING AND PREPROCESSING
print(f"[1/5] Loading graph (source={SOURCE!r})...")
t0 = time.time()

road = Road_Network(
    source_kwargs_inegi,
    source = "inegi",
    id_city_label = "CVEGEO",
    length_attr = "length",
    keep_larger_cc = True,
    to_undirected = True,
    to_simple = True
    )
print(f"    Graph loaded: {road.n:,} nodes, {road.m:,} edges ({time.time()-t0:.1f}s)")
print(f"    External nodes: {road.n_external:,}")
print(f"    Internal nodes: {road.n_internal:,}")
print(f"    Boundary nodes: {road.n_boundary:,}")
print(f"    Localities: {len(road.boundary_nodes):,}")
# Visualization
road.plot_labeled_network()


# CLIQUE GRAPH CONSTRUCTION

print(f"[3/5] Building locality cliques (this may take several minutes)...")
t0 = time.time()

road.reduce_city_subraphs()
print(f"    Graph loaded: {road.n:,} nodes, {road.m:,} edges ({time.time()-t0:.1f}s)")
print(f"    Reduced graph: {road.reduced_graph.order():,} nodes, {road.reduced_graph.size():,} edges ({time.time()-t0:.1f}s)")

# --- Visualization ---
road.plot_boundary_nodes_network()


# ITERATIVE GRAPH SIMPLIFICATION

print(f"[4/5] Simplifying graph iteratively...")
t0 = time.time()

simplified_graph, num_iterations = road.simplify()

print(f"    Converged in {num_iterations} iterations → {simplified_graph.n:,} nodes, {simplified_graph.m:,} edges ({time.time()-t0:.1f}s)")
print(f"    External nodes: {simplified_graph.n_external:,}")
print(f"    Internal nodes: {simplified_graph.n_internal:,}")
print(f"    Boundary nodes: {simplified_graph.n_boundary:,}")

simplified_graph.plot_labeled_network()

road.networkx_to_igraph()
simplified_graph.networkx_to_igraph()


path = "C:\\Users\\Hector Saib\\Documents\\Zoom\\"
nodes_gdf, edges_gdf = road.to_gdf()
nodes_gdf.to_file(path + "road_n.gpkg", driver = "GPKG")
edges_gdf.to_file(path + "road_e.gpkg", driver = "GPKG")

nodes_gdf, edges_gdf = simplified_graph.to_gdf()
nodes_gdf.to_file(path + "simplified_n.gpkg", driver = "GPKG")
edges_gdf.to_file(path + "simplified_e.gpkg", driver = "GPKG")