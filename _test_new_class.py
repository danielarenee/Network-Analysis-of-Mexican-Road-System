# Packages
import time
import networkx as nx
from pathlib import Path

from road_network import Road_Network
import src.utils as fc

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent
TESTS_DIR = BASE_DIR / "tests"

# CONSTANTS
SOURCE = "inegi"

ENT = "31"

# INEGI settings 
source_kwargs_inegi = {
    "inegi_graph_path": BASE_DIR / "data" / "processed" / f"road_network_{ENT}.pkl"
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
print(f"    Inner nodes: {road.n_inner:,}")
print(f"    Localities: {len(road.boundary_nodes):,}")
# Visualization
road.plot_labeled_network("INEGI - Initial Road Network")

# ITERATIVE GRAPH SIMPLIFICATION
print(f"[2/5] Simplifying graph iteratively...")
t0 = time.time()

simplified_graph, num_iterations = road.simplify()

print(f"    Converged in {num_iterations} iterations → {simplified_graph.n:,} nodes, {simplified_graph.m:,} edges ({time.time()-t0:.1f}s)")
print(f"    External nodes: {simplified_graph.n_external:,}")
print(f"    Internal nodes: {simplified_graph.n_internal:,}")
print(f"    Boundary nodes: {simplified_graph.n_boundary:,}")
print(f"    Inner nodes: {simplified_graph.n_inner:,}")

simplified_graph.plot_labeled_network("Fully Simplified Graph")

# SPLIT GRAPH 
print("[3/5] Splitting graph into internal and external subgraphs...")
internal_graph, external_graph = road.split()
internal_graph.plot_labeled_network("Internal subgraphs")
print("Internal graph")
print(f"    External nodes: {internal_graph.n_external:,}")
print(f"    Internal nodes: {internal_graph.n_internal:,}")
print(f"    Boundary nodes: {internal_graph.n_boundary:,}")
print(f"    Inner nodes: {internal_graph.n_inner:,}")

external_graph.plot_labeled_network("External subgraphs")
print("External graph")
print(f"    External nodes: {external_graph.n_external:,}")
print(f"    Internal nodes: {external_graph.n_internal:,}")
print(f"    Boundary nodes: {external_graph.n_boundary:,}")
print(f"    Inner nodes: {external_graph.n_inner:,}")

path = "C:\\Users\\Hector Saib\\Documents\\Zoom\\"
nodes_gdf, edges_gdf = external_graph.to_gdf()
nodes_gdf.to_file(path + f"external_n_{ENT}.gpkg", driver = "GPKG")
edges_gdf.to_file(path + f"external_e_{ENT}.gpkg", driver = "GPKG")
"""
#%%%
simplified_graph.networkx_to_igraph()
#path = "C:\\Users\\Hector Saib\\Documents\\Zoom\\"
path = "C:\\Users\\Saib\\Documents\\Zoom\\"
d, p, R, F, contador, final_time = simplified_graph.voronoi_dijkstra()
nodes_gdf, edges_gdf = simplified_graph.to_gdf(R=R, d=d)
nodes_gdf.to_file(path + f"simplified_n_{ENT}.gpkg", driver = "GPKG")
edges_gdf.to_file(path + f"simplified_e_{ENT}.gpkg", driver = "GPKG")


d, p, R, F, contador, final_time = road.voronoi_dijkstra()
nodes_gdf, edges_gdf = road.to_gdf(R=R, d=d)
nodes_gdf.to_file(path + f"road_n_{ENT}.gpkg", driver = "GPKG", index=False)
edges_gdf.to_file(path + f"road_e_{ENT}.gpkg", driver = "GPKG", index=False)

#%%%

# CLIQUE GRAPH CONSTRUCTION
print(f"[3/5] Building locality cliques (this may take several minutes)...")
t0 = time.time()

road.reduce_city_subraphs()
print(f"    Graph loaded: {road.n:,} nodes, {road.m:,} edges ({time.time()-t0:.1f}s)")
print(f"    Reduced graph: {road.reduced_graph.order():,} nodes, {road.reduced_graph.size():,} edges ({time.time()-t0:.1f}s)")

# --- Visualization ---
road.plot_boundary_nodes_network()
"""