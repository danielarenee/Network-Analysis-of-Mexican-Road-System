import pickle
import math
import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
from collections import defaultdict
from itertools import islice
from shapely.geometry import Point
from scipy.spatial import Delaunay

import src.utils as fc

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

# Cargar desde pkl
pkl_path = BASE_DIR / "data" / "processed" / "road_network.pkl"
with open(pkl_path, "rb") as f:
    graph = pickle.load(f)

# renombrar id_polygon → CVEGEO
cvegeo_map = {}
for node_id, data in graph.nodes(data=True):
    val = data.get("id_polygon")
    if val is None or (isinstance(val, float) and math.isnan(val)):
        graph.nodes[node_id]["CVEGEO"] = None
        cvegeo_map[node_id] = None
    else:
        graph.nodes[node_id]["CVEGEO"] = int(val)
        cvegeo_map[node_id] = int(val)

# Construir gdf_nodes_labeled (se usa en Secciones 4 y 5)
nodes_data = []
for node_id, data in graph.nodes(data=True):
    nodes_data.append({
        "node_id": node_id,
        "x": data["x"],
        "y": data["y"],
        "CVEGEO": data.get("CVEGEO")
    })
df_nodes = pd.DataFrame(nodes_data)
df_nodes["geometry"] = [Point(r["x"], r["y"]) for _, r in df_nodes.iterrows()]
gdf_nodes_labeled = gpd.GeoDataFrame(df_nodes, geometry="geometry", crs="EPSG:6372")


#%%
# ============================================================================
# SECTION 2: BOUNDARY NODE IDENTIFICATION
# Purpose: Identify nodes that lie on the boundary between different localities
# Approach: A node is considered a boundary node if it has at least one neighbor
#           belonging to a different locality (CVEGEO code)
# Output: Dictionary mapping each locality code to its set of boundary nodes
# ============================================================================

boundary_nodes_by_locality: dict[str, set[int]] = defaultdict(set)

for node in graph.nodes:
    node_loc = cvegeo_map.get(node)  # reuse the dict we already built
    if pd.isna(node_loc):
        continue

    for neighbor in graph.neighbors(node):
        neighbor_loc = cvegeo_map.get(neighbor)
        if pd.isna(neighbor_loc):
            continue  # saltar vecinos sin localidad (igual que main.py original)
        if neighbor_loc != node_loc:
            boundary_nodes_by_locality[node_loc].add(node)
            break

# Convert back to plain dict if fc.build_locality_clique expects one
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
    fc.build_locality_clique(graph, loc, boundary_nodes_by_locality)
    for loc in boundary_nodes_by_locality
]
reduced_graph = nx.compose_all(locality_cliques)

# --- Visualization ---
fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

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
margin = 500  # en metros (EPSG:6372)
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
gdf_nodes_labeled.plot(ax=ax, column="CVEGEO", cmap="tab20", markersize=2)
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

for target_node, distance in islice(first_targets.items(), 5):
    target_region = node_to_region[target_node]
    print(f"    → node {target_node} (region {target_region}): {distance:.2f} m")