"""
Shortest Path on Road Networks

finds minimum cost path between two nodes with heapq dijkstra implementation
"""

import math
import pickle
import random

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import pandas as pd

import src.utils as fc

#%%

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

# =============================================================================
# CONSTANTS
# =============================================================================

SOURCE = "inegi"   # "osmnx" | "inegi"
SOURCE_NODE = None  # set to a node ID, or None to pick randomly
TARGET_NODE = None  # set to a node ID, or None to pick randomly
WEIGHT = "length"   # edge attribute used as cost

# OSMnx settings
SHAPEFILE_PATH = BASE_DIR / "data" / "raw" / "shp" / "27l.shp"
CENTER_LAT  = 17.930714
CENTER_LON  = -93.507545
NETWORK_RADIUS = 7000  # meters

# INEGI settings
INEGI_GRAPH_PATH = BASE_DIR / "data" / "processed" / "road_network.pkl"


#%%
# ============================================================================
# 1. LOAD GRAPH
# ============================================================================

if SOURCE == "osmnx":
    CRS = "EPSG:4326"  # coordinate system 

    gdf_localities = gpd.read_file(SHAPEFILE_PATH).to_crs(CRS)  # load locality polygons

    graph = ox.graph_from_point(
        (CENTER_LAT, CENTER_LON),  # center of the area to download
        dist=NETWORK_RADIUS,       # radius in meters
        network_type="drive",     
    )

    gdf_nodes, _ = ox.graph_to_gdfs(graph)  # convert graph nodes to a GeoDataFrame 
    gdf_nodes = gdf_nodes.reset_index()  

    gdf_nodes_labeled = gpd.sjoin(
        gdf_nodes, gdf_localities, how="left", predicate="within"  # tag each node with locality
    )

    cvegeo_map = gdf_nodes_labeled.set_index("osmid")["CVEGEO"].to_dict()  # build a {node_id: CVEGEO_code} dict
    nx.set_node_attributes(graph, cvegeo_map, name="CVEGEO")           

elif SOURCE == "inegi":
    CRS = "EPSG:6372" 

    with open(INEGI_GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)  

    for node_id, data in graph.nodes(data=True): # iterate over every node and its attribute dict
        val = data.get("id_polygon") # read the locality ID 
        if val is None or (isinstance(val, float) and math.isnan(val)): # treat missing or NaN values as no locality
            graph.nodes[node_id]["CVEGEO"] = None
        else:
            graph.nodes[node_id]["CVEGEO"] = int(val) # store as an integer to match osmnx

else:
    raise ValueError(f"Unknown SOURCE: {SOURCE!r}. Use 'osmnx' or 'inegi'.")

#%%
# ============================================================================
# 2. PICK NODES
# ============================================================================

nodes = list(graph.nodes)  # get all node IDs as a plain list

if SOURCE_NODE is None:
    SOURCE_NODE = random.choice(nodes)   # pick random
if TARGET_NODE is None:
    TARGET_NODE = random.choice(nodes)   # pick random...
    while TARGET_NODE == SOURCE_NODE:
        TARGET_NODE = random.choice(nodes)  # ... but keep re-picking until its different from the source

print(f"Source : {SOURCE_NODE}")
print(f"Target : {TARGET_NODE}")

#%%
# ============================================================================
# 3. RUN DIJKSTRA
# ============================================================================

try:
    distance, path = fc.dijkstra_heap(graph, SOURCE_NODE, TARGET_NODE, weight=WEIGHT)  
    print(f"\nShortest path distance : {distance:.2f} m")
    print(f"Number of nodes in path: {len(path)}")
    print(f"Path: {path}")
except nx.NetworkXNoPath:  # raised when no route exists 
    print(f"\nNo path found between {SOURCE_NODE} and {TARGET_NODE}.")
    path = []
    distance = None

#%%
# ============================================================================
# 4. VISUALIZE
# ============================================================================

if path:
    # Use frozensets so both (A,B) and (B,A) match — bidirectional roads share
    # the same physical geometry but exist as two directed edges in the MultiDiGraph.
    # Without this, the reverse-direction edge is drawn gray on top of the gold one.
    path_edge_set = {frozenset([a, b]) for a, b in zip(path[:-1], path[1:])}

    edge_colors = [
        "#FFD700" if frozenset([u, v]) in path_edge_set else "#333333"
        for u, v, _ in graph.edges(keys=True)
    ]
    edge_widths = [
        3.0 if frozenset([u, v]) in path_edge_set else 0.5
        for u, v, _ in graph.edges(keys=True)
    ]

    graph_no_geom = graph.copy()
    for _, _, data in graph_no_geom.edges(data=True):
        data.pop("geometry", None)

    fig, ax = ox.plot_graph(
        graph_no_geom,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        node_size=0,
        show=False,
        close=False,
        bgcolor="black",
    )

    # mark source and target
    src_x = graph.nodes[SOURCE_NODE]["x"]  # longitude of the source node
    src_y = graph.nodes[SOURCE_NODE]["y"]  # latitude of the source node
    tgt_x = graph.nodes[TARGET_NODE]["x"]
    tgt_y = graph.nodes[TARGET_NODE]["y"]

    ax.scatter(src_x, src_y, c="lime",   s=8, zorder=5, label="source")  
    ax.scatter(tgt_x, tgt_y, c="red",    s=8, zorder=5, label="target")
    ax.legend(loc="upper left", fontsize=10, facecolor="#111111", labelcolor="white")

    title = (
        f"Shortest path  {SOURCE_NODE} → {TARGET_NODE}\n"
        f"Distance: {distance:.2f} m  |  Nodes: {len(path)}"
    )
    ax.set_title(title, fontsize=13, color="white")
    plt.show()
