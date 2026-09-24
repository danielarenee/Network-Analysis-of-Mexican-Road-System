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
FILE = f"road_network_{ENT}.pkl"
PATH_SAVE = Path("C:\\Users\\Hector Saib\\Documents\\Zoom\\")

# INEGI settings 
source_kwargs_inegi = {
    "inegi_graph_path": BASE_DIR / "data" / "processed" / FILE
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

road.save(PATH_SAVE, "road")
road2 = Road_Network.load(PATH_SAVE, "road")
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
internal_graph, external_graph = simplified_graph.split()
internal_graph.plot_labeled_network("Internal subgraphs")
print("Internal graph")
print(f"    External nodes: {internal_graph.n_external:,}")
print(f"    Internal nodes: {internal_graph.n_internal:,}")
print(f"    Boundary nodes: {internal_graph.n_boundary:,}")
print(f"    Inner nodes: {internal_graph.n_inner:,}")
print(f"    Total edges: {internal_graph.m:,}")


external_graph.plot_labeled_network("External subgraphs")
print("External graph")
print(f"    External nodes: {external_graph.n_external:,}")
print(f"    Internal nodes: {external_graph.n_internal:,}")
print(f"    Boundary nodes: {external_graph.n_boundary:,}")
print(f"    Inner nodes: {external_graph.n_inner:,}")
print(f"    Total edges: {external_graph.m:,}")


print("[4/5] Compute voronoi network diagram on external graph..")
d, p, R, F, contador, final_time = external_graph.voronoi_network_diagram()

nodes_gdf, edges_gdf = external_graph.to_gdf(R, d)
nodes_gdf.to_file(PATH_SAVE / f"external_n_{ENT}.gpkg", driver = "GPKG")
edges_gdf.to_file(PATH_SAVE /  f"external_e_{ENT}.gpkg", driver = "GPKG")

nodes_gdf, edges_gdf = internal_graph.to_gdf()
nodes_gdf.to_file(PATH_SAVE / f"internal_n_{ENT}.gpkg", driver = "GPKG")
edges_gdf.to_file(PATH_SAVE /  f"internal_e_{ENT}.gpkg", driver = "GPKG")


print("[5/5] Compute voronoi dense graph..")
voronoi_dense_graph, inter_voronoi_graph = external_graph.voronoi_dense_grap(R, F)
voronoi_dense_graph.plot_labeled_network()
print(f"    External nodes: {voronoi_dense_graph.n_external:,}")
print(f"    Internal nodes: {voronoi_dense_graph.n_internal:,}")
print(f"    Boundary nodes: {voronoi_dense_graph.n_boundary:,}")
print(f"    Inner nodes: {voronoi_dense_graph.n_inner:,}")
print(f"    Total edges: {voronoi_dense_graph.m:,}")
nodes_gdf, edges_gdf = voronoi_dense_graph.to_gdf()
nodes_gdf.to_file(PATH_SAVE /  f"vdg_n_{ENT}.gpkg", driver = "GPKG")
edges_gdf.to_file(PATH_SAVE /  f"vdg_e_{ENT}.gpkg", driver = "GPKG")

inter_voronoi_graph.plot_labeled_network()
print(f"    External nodes: {inter_voronoi_graph.n_external:,}")
print(f"    Internal nodes: {inter_voronoi_graph.n_internal:,}")
print(f"    Boundary nodes: {inter_voronoi_graph.n_boundary:,}")
print(f"    Inner nodes: {inter_voronoi_graph.n_inner:,}")
print(f"    Total edges: {inter_voronoi_graph.m:,}")
nodes_gdf, edges_gdf = inter_voronoi_graph.to_gdf()
nodes_gdf.to_file(PATH_SAVE /  f"inter_vor_n_{ENT}.gpkg", driver = "GPKG")
edges_gdf.to_file(PATH_SAVE /  f"inter_vor_e_{ENT}.gpkg", driver = "GPKG")

