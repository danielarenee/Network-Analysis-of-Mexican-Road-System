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

import math
import pickle

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from collections import defaultdict
from itertools import islice
from scipy.spatial import Delaunay
from shapely.geometry import Point

import time

import redes_viales.src.func as fc

#%%

# CONSTANTS

SOURCE = "inegi"  # "osmnx" or "inegi"

#  OSMnx settings 
SHAPEFILE_PATH = "/Users/danielarenee/Desktop/honores/redes_viales/Data/shp/27l.shp"
CENTER_LAT = 17.930714
CENTER_LON = -93.507545
NETWORK_RADIUS = 7000  # meters

# INEGI settings 
INEGI_GRAPH_PATH = (
    "/Users/danielarenee/PycharmProjects/"
    "Network-Analysis-of-Mexican-Road-System/"
    "road_network/test/road_network.pkl"
)

#%%
# ============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================================

print(f"[1/5] Loading graph (source={SOURCE!r})...")
t0 = time.time()

if SOURCE == "osmnx":
    CRS = "EPSG:4326"
    PLOT_MARGIN = 0.002  # degrees

    # load locality polygons
    gdf_localities = gpd.read_file(SHAPEFILE_PATH).to_crs(CRS)

    # download road network with osmnx
    graph = ox.graph_from_point(
        (CENTER_LAT, CENTER_LON),
        dist=NETWORK_RADIUS,
        network_type="drive",
    )

    # convert nodes to geodataframe 
    gdf_nodes, _ = ox.graph_to_gdfs(graph)
    # reset index so 'osmid' (the node id) becomes a regular column
    gdf_nodes = gdf_nodes.reset_index()

    # spatial join to assign each node its locality (CVEGEO)
    gdf_nodes_labeled = gpd.sjoin(
        gdf_nodes, gdf_localities, how="left", predicate="within"
    )

    # write CVEGEO back into the graph as a node attribute
    cvegeo_map = gdf_nodes_labeled.set_index("osmid")["CVEGEO"].to_dict()
    nx.set_node_attributes(graph, cvegeo_map, name="CVEGEO")

    # visualization
    fig, ax = ox.plot_graph(graph, show=False, close=False)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    gdf_localities.boundary.plot(ax=ax, color="red")
    gdf_nodes_labeled.plot(ax=ax, column="CVEGEO", cmap="Set2")
    ax.set_title("Ernesto Aguirre, Tabasco - Labeled Road Network",
                 fontsize=16, color="white")
    plt.show()

elif SOURCE == "inegi":
    CRS = "EPSG:6372"
    PLOT_MARGIN = 500  # meters
    gdf_localities = None  # no shapefile needed 

    # load pre-built INEGI graph 
    with open(INEGI_GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)

    # rename id_polygon 
    cvegeo_map = {}
    for node_id, data in graph.nodes(data=True):
        val = data.get("id_polygon")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            graph.nodes[node_id]["CVEGEO"] = None
            cvegeo_map[node_id] = None
        else:
            graph.nodes[node_id]["CVEGEO"] = int(val)
            cvegeo_map[node_id] = int(val)

    # build gdf_nodes_labeled 
    nodes_data = [
        {"node_id": node_id, "x": data["x"], "y": data["y"],
         "CVEGEO": data.get("CVEGEO")}
        for node_id, data in graph.nodes(data=True)
    ]
    df_nodes = pd.DataFrame(nodes_data)
    df_nodes["geometry"] = gpd.points_from_xy(df_nodes["x"], df_nodes["y"])
    gdf_nodes_labeled = gpd.GeoDataFrame(df_nodes, geometry="geometry", crs=CRS)

else:
    raise ValueError(f"Unknown SOURCE: {SOURCE!r}. Use 'osmnx' or 'inegi'.")

print(f"    Graph loaded: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges ({time.time()-t0:.1f}s)")

#%%
# ============================================================================
# 2. BOUNDARY NODE IDENTIFICATION
# Purpose: Identify nodes that lie on the boundary between different localities
# Approach: A node is considered a boundary node if it has at least one neighbor
#           belonging to a different locality (CVEGEO code)
# Output: Dictionary mapping each locality code to its set of boundary nodes
# ============================================================================

print(f"[2/5] Identifying boundary nodes...")
t0 = time.time()

boundary_nodes_by_locality: dict[str, set[int]] = defaultdict(set)

for node in graph.nodes:
    node_loc = cvegeo_map.get(node)
    if pd.isna(node_loc):
        continue

    for neighbor in graph.neighbors(node):
        neighbor_loc = cvegeo_map.get(neighbor)
        if pd.isna(neighbor_loc) or neighbor_loc != node_loc:
            boundary_nodes_by_locality[node_loc].add(node)
            break

boundary_nodes_by_locality = dict(boundary_nodes_by_locality)

total_boundary = sum(len(v) for v in boundary_nodes_by_locality.values())
print(f"    {len(boundary_nodes_by_locality):,} localities, {total_boundary:,} boundary nodes total ({time.time()-t0:.1f}s)")

#%%
# ============================================================================
# SECTION 3: CLIQUE GRAPH CONSTRUCTION
# Purpose: Build a reduced graph where each locality is represented by a clique
#          of its boundary nodes
# Approach: For each locality, create a complete graph (clique) connecting all
#           pairs of boundary nodes with weighted edges representing shortest
#           paths within that locality
# Output: A unified graph containing all boundary nodes connected within and
#         across localities
# ============================================================================

print(f"[3/5] Building locality cliques (this may take several minutes)...")
t0 = time.time()

locality_cliques = []
total_locs = len(boundary_nodes_by_locality)
t_last = time.time()
for i, loc in enumerate(boundary_nodes_by_locality, 1):
    locality_cliques.append(fc.construir_clique_localidad(graph, loc, boundary_nodes_by_locality))
    now = time.time()
    if now - t_last >= 5 or i == total_locs:
        n_frontier = len(boundary_nodes_by_locality[loc])
        elapsed = now - t0
        rate = i / elapsed if elapsed > 0 else 0
        eta = (total_locs - i) / rate if rate > 0 else float("inf")
        print(f"    [{i:,}/{total_locs:,}] loc={loc}  boundary_nodes={n_frontier}  "
              f"elapsed={elapsed:.0f}s  rate={rate:.1f} loc/s  ETA={eta:.0f}s", flush=True)
        t_last = now
reduced_graph = nx.compose_all(locality_cliques)

print(f"    Reduced graph: {reduced_graph.number_of_nodes():,} nodes, {reduced_graph.number_of_edges():,} edges ({time.time()-t0:.1f}s)")

# --- Visualization ---
fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

if gdf_localities is not None:
    gdf_localities.boundary.plot(ax=ax, color="gray", linewidth=0.5)

node_positions = {
    nid: (data["x"], data["y"])
    for nid, data in reduced_graph.nodes(data=True)
}

nx.draw_networkx_nodes(
    reduced_graph, pos=node_positions, ax=ax,
    node_size=24, node_color="white",
)
nx.draw_networkx_edges(
    reduced_graph, pos=node_positions, ax=ax,
    width=1, edge_color="orange",
)

ax.set_title("Boundary Node Network by Locality", fontsize=20, color="white")
ax.axis("off")

coords = np.array(list(node_positions.values()))
ax.set_xlim(coords[:, 0].min() - PLOT_MARGIN, coords[:, 0].max() + PLOT_MARGIN)
ax.set_ylim(coords[:, 1].min() - PLOT_MARGIN, coords[:, 1].max() + PLOT_MARGIN)

plt.show()

#%%
"""
# SECTION 4: DELAUNAY TRIANGULATION OF LOCALITY CENTROIDS 
# this creates a triangulation conecting locality centroids and visualizes

centroids_df = (
    gdf_nodes_labeled
    .dropna(subset=["CVEGEO"])
    .groupby("CVEGEO")[["x", "y"]]
    .mean()
)

locality_labels = centroids_df.index.tolist()
centroid_points = centroids_df.values  # numpy array

delaunay_tri = Delaunay(centroid_points)

plt.figure(figsize=(6, 6))
plt.triplot(
    centroid_points[:, 0], centroid_points[:, 1],
    delaunay_tri.simplices, linewidth=0.8,
)
plt.scatter(centroid_points[:, 0], centroid_points[:, 1], color="red", s=30)
for i, label in enumerate(locality_labels):
    plt.text(
        centroid_points[i, 0], centroid_points[i, 1],
        label, fontsize=8, ha="center", va="center",
    )
plt.title("Delaunay Triangulation of Locality Centroids")
plt.xlabel("Longitude" if CRS == "EPSG:4326" else "X (m)")
plt.ylabel("Latitude" if CRS == "EPSG:4326" else "Y (m)")
plt.gca().set_aspect("equal", "box")
plt.show()
"""

#%%

# ============================================================================
# SECTION 5: ITERATIVE GRAPH SIMPLIFICATION
# Purpose: Simplify the road network through iterative application of three
#          operations until a fixed point is reached
# Approach: Repeatedly apply these steps until no more changes occur:
#   1. Simplify multiple edges (keep shortest between each node pair)
#   2. Remove degree-1 nodes (dead ends)
#   3. Remove degree-2 nodes and merge their incident edges
# ============================================================================

print(f"[4/5] Simplifying graph iteratively...")
t0 = time.time()

simplified_graph, num_iterations = fc.simplify_iteratively(graph)

print(f"    Converged in {num_iterations} iterations → {simplified_graph.number_of_nodes():,} nodes, {simplified_graph.number_of_edges():,} edges ({time.time()-t0:.1f}s)")

fig, ax = ox.plot_graph(simplified_graph, show=False, close=False)
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
if gdf_localities is not None:
    gdf_localities.boundary.plot(ax=ax, color="red")
gdf_nodes_labeled.plot(ax=ax, column="CVEGEO", cmap="Set2")
ax.set_title(
    f"Fully Simplified Graph (Converged in {num_iterations} iterations)",
    fontsize=16, color="white",
)
plt.show()

#%%
# ============================================================================
# SECTION 6: INTER-REGION DISTANCE MATRIX
# ============================================================================

print(f"[5/5] Computing inter-region distance matrix (this may take a while)...")
t0 = time.time()

distance_matrix, node_to_region = fc.calculate_border_nodes_distance_matrix(
    graph, boundary_nodes_by_locality
)

print(f"    Done ({time.time()-t0:.1f}s)")

total_connections = sum(len(targets) for targets in distance_matrix.values())
total_boundary_nodes = len(node_to_region)

print(f"\n{'=' * 60}")
print("INTER-REGION DISTANCE")
print(f"{'=' * 60}")
print(f"Total inter-region connections: {total_connections}")
print(f"Total boundary nodes: {total_boundary_nodes}")

print("\nSample inter-region distances (first boundary node):")

first_node = next(iter(distance_matrix))
first_targets = distance_matrix[first_node]
source_region = node_to_region[first_node]

print(f"\n  First 5 targets from node {first_node} (region {source_region}):")

for target_node, distance in islice(first_targets.items(), 5):
    target_region = node_to_region[target_node]
    print(f"    -> node {target_node} (region {target_region}): {distance:.2f} m")
