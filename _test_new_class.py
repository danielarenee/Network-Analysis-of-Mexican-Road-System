# Packages
import time
from pathlib import Path

from road_network import Road_Network

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent

# CONSTANTS
SOURCE = "inegi"

ENT = "15"
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
print(f"[2/5] Simplifying graph iteratively...")
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


print("[4/5] Compute voronoi network diagram on external graph..")
d, p, R, F, contador, final_time = external_graph.voronoi_network_diagram()

g = external_graph._graph("ig")
node_map = external_graph.node_to_ig
boundary_nodes = external_graph.boundary_nodes
boundary_nodes = external_graph._Road_Network__boundary_nodes
id_city_label = external_graph._Road_Network__id_city_label
external_city_id = None

#%%%%
import geopandas as gpd

city_network = external_graph.build_region_graph()
layout = city_network.layout_fruchterman_reingold(niter=2000)

import igraph as ig
ig.plot(
    city_network,
    layout=layout,
    vertex_size=5,
    bbox=(1200, 900),
    margin=50,
)

#%%%
path = BASE_DIR / "data" / "raw" / "LocalitiesGrouped_2020_data.gpkg"
gdf = gpd.read_file(path)

regions_list = list(city_network.vs["region"])
gdf = gdf[gdf["id_convex"].isin(regions_list)]

centroids = gpd.GeoDataFrame(
    {
        "id_convex": gdf["id_convex"],
        "geometry": gdf.geometry.centroid,
    },
    crs=gdf.crs,
)

nodes_df = (
        city_network.get_vertex_dataframe()
        .rename_axis("vertex_id")
        .reset_index()
    )
centros = centroids.set_index("id_convex").geometry

nodes_gdf = gpd.GeoDataFrame(
       nodes_df,
       geometry=[centros.loc[region] for region in nodes_df["region"]],
       crs=centroids.crs,
   )


from shapely.geometry import LineString
edges_df = (
        city_network.get_edge_dataframe()
        .rename_axis("edge_id")
        .reset_index()
    )
coords = [(point.x, point.y) for point in nodes_gdf.geometry]
lines = [
    LineString([coords[u], coords[v]])
    for u, v in zip(edges_df["source"], edges_df["target"])
]

edges_gdf = gpd.GeoDataFrame(
    edges_df,
    geometry=lines,
    crs=centroids.crs,
)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(25, 25))

edges_gdf.plot(ax=ax, linewidth=0.4, color="gray", alpha=0.6)
nodes_gdf.plot(ax=ax, markersize=4, color="red")

ax.set_axis_off()
