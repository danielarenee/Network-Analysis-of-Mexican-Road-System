"""
Road Network Analysis

This script analyzes road networks in Tabasco, Mexico using graph theory and
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
# ============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================================
#
# Purpose: Load geographic data and road network, then label nodes with their
#          corresponding locality (CVEGEO code)
#
# Steps:
#   1. Load locality polygons from shapefile
#   2. Download road network from OpenStreetMap
#   3. Convert nodes to GeoDataFrame
#   4. Perform spatial join to assign localities to nodes
#   5. Visualize the labeled network
# ============================================================================

# Load shapefile containing locality polygons (CVEGEO regions)
shapefile_path = "/Users/danielarenee/Desktop/honores/redes_viales/Data/shp/27l.shp"

gdf_localities = gpd.read_file(shapefile_path)
# Convert to EPSG:4326 (WGS84) coordinate reference system
gdf_localities = gdf_localities.to_crs(4326)

# Download road network graph from OpenStreetMap using OSMnx
# Center point coordinates for Ernesto Aguirre, Tabasco
lat = 17.930714
lon = -93.507545
# Alternative center point (commented out):
# lat = 18.067192
# lon = -93.498951
center_point = (lat, lon)
graph = ox.graph_from_point(
    center_point,
    dist=7000,  # 7km radius from center point
    network_type='drive'  # Only drivable roads
)

# Extract node data from graph into a DataFrame
nodes_data = []
for node_id, data in graph.nodes(data=True):
    nodes_data.append({
        "node_id": node_id,
        "x": data["x"],  # Longitude
        "y": data["y"]   # Latitude
    })
df_nodes = pd.DataFrame(nodes_data)

# Convert nodes DataFrame to GeoDataFrame with Point geometries
nodes_geom = []
for _, row in df_nodes.iterrows():
    point = Point(row["x"], row["y"])
    nodes_geom.append(point)
df_nodes["geometry"] = nodes_geom
gdf_nodes = gpd.GeoDataFrame(df_nodes, geometry="geometry", crs="EPSG:4326")

# Perform spatial join to determine which locality each node belongs to
# This assigns CVEGEO codes to nodes based on polygon containment
gdf_nodes_labeled = gpd.sjoin(gdf_nodes, gdf_localities, how="left", predicate="within")

# Label graph nodes with their locality (CVEGEO) as a node attribute
for _, row in gdf_nodes_labeled.iterrows():
    node_id = row["node_id"]
    locality_code = row["CVEGEO"]
    graph.nodes[node_id]["CVEGEO"] = locality_code

# Visualize the labeled road network
fig, ax = ox.plot_graph(graph, show=False, close=False)
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
gdf_localities.boundary.plot(ax=ax, color="red")
gdf_nodes_labeled.plot(ax=ax, column="CVEGEO", cmap="Set2")
ax.set_title("Ernesto Aguirre, Tabasco - Labeled Road Network", fontsize=16, color="white")
plt.show()

#%%
# ============================================================================
# SECTION 2: BOUNDARY NODE IDENTIFICATION
# ============================================================================
#
# Purpose: Identify nodes that lie on the boundary between different localities
#
# Approach: A node is considered a boundary node if it has at least one neighbor
#           belonging to a different locality (CVEGEO code)
#
# Output: Dictionary mapping each locality code to its set of boundary nodes
# ============================================================================

# Dictionary to store boundary nodes for each locality
# Format: {locality_code: set(boundary_node_ids)}
boundary_nodes_by_locality = dict()

for node in graph.nodes:
    node_locality = graph.nodes[node]["CVEGEO"]
    if node_locality is None:
        continue  # Skip nodes not in any locality

    # Check if this node has neighbors in different localities
    for neighbor in graph.neighbors(node):
        neighbor_locality = graph.nodes[neighbor]["CVEGEO"]
        if neighbor_locality is None:
            continue  # Skip neighbors without locality
        if neighbor_locality != node_locality:
            # This node connects to a different locality, so it's a boundary node
            if node_locality not in boundary_nodes_by_locality:
                boundary_nodes_by_locality[node_locality] = set()
            boundary_nodes_by_locality[node_locality].add(node)
            break  # One different neighbor is sufficient

# Optional: Print statistics about boundary nodes
# for locality, nodes in boundary_nodes_by_locality.items():
#     print(f"Locality {locality} has {len(nodes)} boundary nodes")

#%%
# ============================================================================
# SECTION 3: CLIQUE GRAPH CONSTRUCTION
# ============================================================================
#
# Purpose: Build a reduced graph where each locality is represented by a clique
#          of its boundary nodes
#
# Approach: For each locality, create a complete graph (clique) connecting all
#           pairs of boundary nodes with weighted edges representing shortest
#           paths within that locality
#
# Output: A unified graph containing all boundary nodes connected within and
#         across localities
# ============================================================================

# Create empty graph to store the complete reduced network
reduced_graph = nx.Graph()

# Build clique for each locality and add to the reduced graph
for locality_code in boundary_nodes_by_locality.keys():
    # Build clique graph for this locality
    locality_clique = fc.construir_clique_localidad(graph, locality_code, boundary_nodes_by_locality)

    # Add all nodes from locality clique to the reduced graph (with attributes)
    for node_id, node_data in locality_clique.nodes(data=True):
        reduced_graph.add_node(node_id, **node_data)

    # Add all edges from locality clique to the reduced graph (with weight and path)
    for u, v, edge_data in locality_clique.edges(data=True):
        reduced_graph.add_edge(u, v, **edge_data)

# Visualize the reduced boundary node network
fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Plot locality polygons as background
gdf_localities.boundary.plot(ax=ax, color="gray", linewidth=0.5)

# Extract node positions for visualization
node_positions = {}
for node_id, node_data in reduced_graph.nodes(data=True):
    x = node_data["x"]
    y = node_data["y"]
    node_positions[node_id] = (x, y)

# Draw boundary nodes
nx.draw_networkx_nodes(
    reduced_graph,
    pos=node_positions,
    ax=ax,
    node_size=24,
    node_color="white"
)

# Draw edges between boundary nodes
nx.draw_networkx_edges(
    reduced_graph,
    pos=node_positions,
    ax=ax,
    width=1,
    edge_color="orange"
)

ax.set_title("Boundary Node Network by Locality", fontsize=20, color="white")
ax.axis("off")

# Set axis limits with small margin around nodes
x_coords = [coord[0] for coord in node_positions.values()]
y_coords = [coord[1] for coord in node_positions.values()]
margin = 0.002  # Adjust this value to increase/decrease margin
x_min, x_max = min(x_coords) - margin, max(x_coords) + margin
y_min, y_max = min(y_coords) - margin, max(y_coords) + margin
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.show()

#%%
# ============================================================================
# SECTION 4: DELAUNAY TRIANGULATION OF LOCALITY CENTROIDS
# ============================================================================
#
# Purpose: Create a triangulation connecting locality centroids to understand
#          spatial relationships between localities
#
# Approach: Calculate the centroid of each locality (mean of all node positions)
#           and perform Delaunay triangulation on these centroids
#
# Output: Delaunay triangulation visualization showing locality connections
# ============================================================================

# Group nodes by locality
# Format: {locality_code: [list_of_node_ids]}
localities_dict = {}
for node, data in graph.nodes(data=True):
    locality_code = data.get("CVEGEO")
    if locality_code is None:
        continue
    localities_dict.setdefault(locality_code, []).append(node)

# Calculate centroid for each locality (mean of all node coordinates)
centroids = {}
for locality_code, node_list in localities_dict.items():
    coords = []
    for node_id in node_list:
        node_data = graph.nodes[node_id]
        if "x" in node_data and "y" in node_data:
            x = node_data["x"]
            y = node_data["y"]
            coords.append((x, y))
    if not coords:
        continue
    # Calculate mean x and y coordinates
    xs, ys = zip(*coords)
    centroids[locality_code] = (sum(xs) / len(xs), sum(ys) / len(ys))

# Prepare data for Delaunay triangulation
locality_labels = list(centroids.keys())
# Convert centroids to numpy array
centroid_points = np.array([centroids[loc] for loc in locality_labels])

# Compute Delaunay triangulation
delaunay_triangulation = Delaunay(centroid_points)

# Visualize the triangulation
plt.figure(figsize=(6, 6))
plt.triplot(centroid_points[:, 0], centroid_points[:, 1],
            delaunay_triangulation.simplices, linewidth=0.8)
plt.scatter(centroid_points[:, 0], centroid_points[:, 1], color='red', s=30)
for i, locality_code in enumerate(locality_labels):
    plt.text(centroid_points[i, 0], centroid_points[i, 1], locality_code,
             fontsize=8, ha='center', va='center')
plt.title("Delaunay Triangulation of Locality Centroids")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.gca().set_aspect('equal', 'box')
plt.show()

#%%
# ============================================================================
# SECTION 5: GRAPH SIMPLIFICATION BY PRUNING
# ============================================================================
#
# Purpose: Simplify the road network by iteratively removing low-degree nodes
#
# Approach:
#   Step 1: Remove degree-1 nodes (dead ends / cul-de-sacs)
#   Step 2: Remove degree-2 nodes and merge their incident edges
#
# Output: Progressively simplified graphs with fewer nodes and edges
# ============================================================================

# STEP 1: Prune degree-1 nodes (leaf nodes)
pruned_graph_deg1, removed_nodes_deg1 = fc.podar_grado_1(graph)

# Visualize graph after degree-1 pruning
fig, ax = ox.plot_graph(pruned_graph_deg1, show=False, close=False)
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
gdf_localities.boundary.plot(ax=ax, color="red")
gdf_nodes_labeled.plot(ax=ax, column="CVEGEO", cmap="Set2")
ax.set_title("After Degree-1 Pruning",
             fontsize=16, color="white")
plt.show()

# STEP 2: Prune degree-2 nodes (merge straight segments)
pruned_graph_deg2, removed_nodes_deg2 = fc.podar_grado_2(pruned_graph_deg1)

# Visualize graph after degree-2 pruning
fig, ax = ox.plot_graph(pruned_graph_deg2, show=False, close=False)
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
gdf_localities.boundary.plot(ax=ax, color="red")
gdf_nodes_labeled.plot(ax=ax, column="CVEGEO", cmap="Set2")
ax.set_title("After Full Pruning (Deg-1 & Deg-2)",
             fontsize=16, color="white")
plt.show()

print(f"Original graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
print(f"After degree-1 pruning: {pruned_graph_deg1.number_of_nodes()} nodes, {pruned_graph_deg1.number_of_edges()} edges")
print(f"After degree-2 pruning: {pruned_graph_deg2.number_of_nodes()} nodes, {pruned_graph_deg2.number_of_edges()} edges")

#%%
# ============================================================================
# SECTION 6: SIMPLIFY MULTIPLE EDGES
# ============================================================================
#
# Purpose: Remove parallel edges between node pairs, keeping only the shortest
#
# Approach: For each pair of nodes (u, v) with multiple edges in the same
#           direction, keep only the edge with minimum length
#
# Output: Simplified graph with no parallel edges, plus statistics about
#         which edges were removed
# ============================================================================

# Simplify multiple edges (keep shortest between each node pair)
graph_original, graph_simplified, multiple_edges_info = fc.simplificar_aristas_multiples(pruned_graph_deg2)

# Visualize the simplified graph
fig, ax = ox.plot_graph(graph_simplified, show=False, close=False)
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
gdf_localities.boundary.plot(ax=ax, color="red")
gdf_nodes_labeled.plot(ax=ax, column="CVEGEO", cmap="Set2")
ax.set_title("After Simplifying Multiple Edges",
             fontsize=16, color="white")
plt.show()

# Print statistics about edge simplification
print(f"\nMultiple edges simplification:")
print(f"Original graph: {graph_original.number_of_edges()} edges")
print(f"Simplified graph: {graph_simplified.number_of_edges()} edges")
print(f"Number of node pairs with multiple edges: {len(multiple_edges_info)}")
total_removed = sum(info['removed'] for info in multiple_edges_info.values())
print(f"Total edges removed: {total_removed}")
