# Packages
import time
from pathlib import Path

from road_network import Road_Network
from city_network import City_Network

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent

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

# Visualization
road.plot_labeled_network("INEGI - Initial Road Network")
# ITERATIVE GRAPH SIMPLIFICATION
print("[2/5] Simplifying graph iteratively...")
t0 = time.time()

simplified_graph, num_iterations, t = road.simplify()

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


nodes_gdf, edges_gdf  = external_graph.to_gdf()
nodes_gdf.to_file(PATH_SAVE/f"external_n_{ENT}.gpkg", driver="GPKG")
edges_gdf.to_file(PATH_SAVE/f"external_e_{ENT}.gpkg", driver="GPKG")

print("[4/5] Compute voronoi network diagram on external graph..")
d, p, R, F, contador, final_time = external_graph.voronoi_network_diagram()


road.build_region_graph()


city_network = external_graph.build_region_graph()
city = City_Network(city_network)
city.plot((25, 25))
city.plot((25, 25), False)

print(f"n: {city.n}")
print(f"m: {city.m}")}

print(f"node connectivity: {city.vertex_connectivity}")
print(f"edge connectivity: {city.edge_connectivity}")

#city._City_Network__ig_graph.articulation_points()

city.save(PATH_SAVE, f"city_network_{ENT}")





