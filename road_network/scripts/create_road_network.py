###########
# LIBRARIES
###########
# Import libraries
import geopandas as gpd
import networkx as nx
import pickle

from tqdm import tqdm
from pathlib import Path

###########
# VARIABLES
###########
# Set relative path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = str(BASE_DIR)


roads = gpd.read_file(DATA_DIR + "\\temp\\roads.gpkg")
unions = gpd.read_file(DATA_DIR + "\\temp\\unions.gpkg")


# Create digraph
G = nx.DiGraph()

# Add nodes from unions (points)
print("Adding nodes...")
G.add_nodes_from(
    tqdm(unions["ID_UNION"],
         total = len(unions))
    )
# Check nodes
print(f"Nodes: {G.order():,}")

# Create edges
edges = list(
    zip(
        roads["UNION_INI"],
        roads["UNION_FIN"],
        [{"idx": i} for i in roads["ID_RED"]]
        )
    )
mask = roads["CIRCULA"] == "Dos sentidos"
edges += list(
    zip(
        roads.loc[mask, "UNION_FIN"],
        roads.loc[mask, "UNION_INI"],
        [{"idx": i} for i in roads.loc[mask, "ID_RED"]]
        )
    )

# Add edges from roads (lines)
print("Adding edges...")
G.add_edges_from(
    tqdm(edges,
         total = len(edges))
    )
# Check edges
print(f"Edges: {G.size():,}")

# Save graph
with open(DATA_DIR + "\\temp\\road_network.pkl", "wb") as f:
    pickle.dump(
        G, f,
        protocol = pickle.HIGHEST_PROTOCOL
        )
