###########
# PACKAGES
###########
# Import packages
import geopandas as gpd
import networkx as nx
import pickle

from tqdm import tqdm
from pathlib import Path

###########
# VARIABLES
###########
# Set absolute path
BASE_DIR = Path(__file__).resolve().parent.parent


###########
# FUNCTIONS
###########
def create_road_network(roads, unions, save_path):
    # Create digraph
    G = nx.MultiDiGraph()
    
    # Set graphs attributes
    crs = unions.crs
    epsg = crs.to_epsg()
    G.graph = {"crs": f"epsg:{epsg}"}
    
    # Create and add nodes in bulk
    print("Creating nodes list...")
    nodes = [
        (node_id, {"x": geom.x, "y": geom.y, "id_polygon": idx})
        for node_id, geom, idx in zip(
            unions["ID_UNION"],
            unions["geometry"],
            unions["id_convex"]
        )
    ]
    print("Adding nodes...")
    G.add_nodes_from(nodes)
    
    # Check nodes
    print(f"Nodes: {G.order():,}")
    
    print("Creating edges list...")
    # Forward edges
    forward_edges = [
        (u, v, {"name": name, "length": length, "geometry": geom})
        for u, v, name, length, geom in zip(
            roads["UNION_INI"],
            roads["UNION_FIN"],
            roads["NOMBRE"],
            roads["LONGITUD"],
            roads["geometry"]
        )
    ]
    
    # Reverse edges for two-way roads
    mask = roads["CIRCULA"] == "Dos sentidos"
    two_way_roads = roads[mask]
    reverse_edges = [
        (v, u, {"name": name, "length": length, "geometry": geom})
        for u, v, name, length, geom in zip(
            two_way_roads["UNION_INI"],
            two_way_roads["UNION_FIN"],
            two_way_roads["NOMBRE"],
            two_way_roads["LONGITUD"],
            two_way_roads["geometry"]
        )
    ]
    
    edges = forward_edges + reverse_edges
    
    # Add edges from roads (lines) in bulk
    print("Adding edges...")
    G.add_edges_from(edges)
    
    # Check edges
    print(f"Edges: {G.size():,}")
    
    # Save graph
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(
            G, f,
            protocol = pickle.HIGHEST_PROTOCOL
            )
    return G


###########
# MAIN
###########

# Paths of roads and unions
unions_path = BASE_DIR / "data" / "processed" / "unions.gpkg"
roads_path = BASE_DIR / "data" / "processed" / "roads.gpkg"
graph_save_path = BASE_DIR / "data" / "processed" / "road_network.pkl"

# Import .gpkg of roads and unions
unions = gpd.read_file(unions_path)
roads = gpd.read_file(roads_path)

# Create road network
create_road_network(roads, unions, save_path = graph_save_path)

