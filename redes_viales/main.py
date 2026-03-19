"""
Road Network Analysis

This script analyzes road networks in, Mexico using graph theory and
geospatial analysis. It performs the following operations:

1. Load geographic data (shapefiles) and road network data (OSMnx)
2. Identify boundary nodes between different localities (CVEGEO regions)
3. Build clique graphs representing locality connections
4. Perform Delaunay triangulation on locality centroids
5. Simplify road networks by pruning low-degree nodes

The analysis focuses on understanding connectivity between localities and
simplifying complex road network representations.

"""

import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import shapely as shp
from networkx.classes import neighbors

import src.func as fc
import numpy as np

from shapely.geometry import Point
from scipy.spatial import Delaunay

#%%

# =============================================================================
# CONSTANTS
# =============================================================================

SHAPEFILE_PATH = "/Users/danielarenee/Desktop/honores/redes_viales/Data/shp/27l.shp"
CENTER_LAT = 17.930714
CENTER_LON = -93.507545
NETWORK_RADIUS = 7000  # meters
CRS = "EPSG:4326"

#%%
# ============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================================

# --- Load locality polygons ---
gdf_localities = gpd.read_file(SHAPEFILE_PATH).to_crs(CRS)

# --- Download road network ---
graph = ox.graph_from_point(
    (CENTER_LAT, CENTER_LON),
    dist=NETWORK_RADIUS,
    network_type="drive",
)

# --- Convert nodes to GeoDataFrame ---
gdf_nodes, _ = ox.graph_to_gdfs(graph)
# Reset index so 'osmid' (the node id) becomes a regular column
gdf_nodes = gdf_nodes.reset_index()

# --- Spatial join: assign each node its locality (CVEGEO) ---
gdf_nodes_labeled = gpd.sjoin(
    gdf_nodes, gdf_localities, how="left", predicate="within"
)

# --- Write CVEGEO back into the graph as a node attribute ---
cvegeo_map = gdf_nodes_labeled.set_index("osmid")["CVEGEO"].to_dict()
nx.set_node_attributes(graph, cvegeo_map, name="CVEGEO")

# --- Visualization ---
fig, ax = ox.plot_graph(graph, show=False, close=False)
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
gdf_localities.boundary.plot(ax=ax, color="red")
gdf_nodes_labeled.plot(ax=ax, column="CVEGEO", cmap="Set2")
ax.set_title("Ernesto Aguirre, Tabasco - Labeled Road Network",
             fontsize=16, color="white")
plt.show()

#%%
# ============================================================================
# SECTION 2: BOUNDARY NODE IDENTIFICATION
# Purpose: Identify nodes that lie on the boundary between different localities
# Approach: A node is considered a boundary node if it has at least one neighbor
#           belonging to a different locality (CVEGEO code)
# Output: Dictionary mapping each locality code to its set of boundary nodes
# ============================================================================

from collections import defaultdict

boundary_nodes_by_locality: dict[str, set[int]] = defaultdict(set)

for node in graph.nodes:
    node_loc = cvegeo_map.get(node)  # reuse the dict we already built
    if pd.isna(node_loc):
        continue

    for neighbor in graph.neighbors(node):
        neighbor_loc = cvegeo_map.get(neighbor)
        if pd.isna(neighbor_loc) or neighbor_loc != node_loc:
            boundary_nodes_by_locality[node_loc].add(node)
            break

# Convert back to plain dict if fc.construir_clique_localidad expects one
boundary_nodes_by_locality = dict(boundary_nodes_by_locality)

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

locality_cliques = [
    fc.construir_clique_localidad(graph, loc, boundary_nodes_by_locality)
    for loc in boundary_nodes_by_locality
]
reduced_graph = nx.compose_all(locality_cliques)

# --- Visualization ---
fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
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
margin = 0.002
ax.set_xlim(coords[:, 0].min() - margin, coords[:, 0].max() + margin)
ax.set_ylim(coords[:, 1].min() - margin, coords[:, 1].max() + margin)

plt.show()

#%%

# ============================================================================
# SECTION 4: DELAUNAY TRIANGULATION OF LOCALITY CENTROIDS
# Purpose: Create a triangulation connecting locality centroids to understand
#          spatial relationships between localities
# Approach: Calculate the centroid of each locality (mean of all node positions)
#           and perform Delaunay triangulation on these centroids
# Output: Delaunay triangulation visualization showing locality connections
# ============================================================================

centroids_df = (
    gdf_nodes_labeled
    .dropna(subset=["CVEGEO"])
    .groupby("CVEGEO")[["x", "y"]]
    .mean()
)

locality_labels = centroids_df.index.tolist()
centroid_points = centroids_df.values  # already a numpy array

# Delaunay triangulation
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
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.gca().set_aspect("equal", "box")
plt.show()

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

simplified_graph, num_iterations = fc.simplify_iteratively(graph)

fig, ax = ox.plot_graph(simplified_graph, show=False, close=False)
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
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

distance_matrix, node_to_region = fc.calculate_border_nodes_distance_matrix(
    graph, boundary_nodes_by_locality
)

total_connections = sum(len(targets) for targets in distance_matrix.values())
total_boundary_nodes = len(node_to_region)

print(f"\n{'=' * 60}")
print("INTER-REGION DISTANCE")
print(f"{'=' * 60}")
print(f"Total inter-region connections: {total_connections}")
print(f"Total boundary nodes: {total_boundary_nodes}")

# Show sample distances for first boundary node
print("\nSample inter-region distances (first boundary node):")

first_node = next(iter(distance_matrix))
first_targets = distance_matrix[first_node]
source_region = node_to_region[first_node]

print(f"\n  First 5 targets from node {first_node} (region {source_region}):")

from itertools import islice

for target_node, distance in islice(first_targets.items(), 5):
    target_region = node_to_region[target_node]
    print(f"    → node {target_node} (region {target_region}): {distance:.2f} m")