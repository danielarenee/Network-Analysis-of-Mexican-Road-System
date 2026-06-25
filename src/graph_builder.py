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
    
    # Create node attributes
    nodes_attributes = (
        {"x": x, "y": y, "id_polygon": idx}
        for x, y, idx in zip(
                unions["geometry"].x,
                unions["geometry"].y,
                unions["id_convex"]
                )
        )
    
    # Add nodes from unions (points)
    print("Adding nodes...")
    G.add_nodes_from(
        tqdm(
            zip(
                unions["ID_UNION"],
                nodes_attributes
                ),
             total = len(unions)
             )
        )
    # Check nodes
    print(f"Nodes: {G.order():,}")
    
    # Create edge attributes
    edge_attributes = [
        {"name": n,
         "length": l,
         "geometry": g}
        for n, l, g in zip(
                roads["NOMBRE"],
                roads["LONGITUD"],
                roads["geometry"]
                )
        ]
    # Create edges
    edges = list(
        zip(
            roads["UNION_INI"],
            roads["UNION_FIN"],
            edge_attributes
            )
        )
    mask = roads["CIRCULA"] == "Dos sentidos"
    # Create edge attributes
    edge_attributes += [
        {"name": n,
         "length": l,
         "geometry": g}
        for n, l, g in zip(
                roads.loc[mask,"NOMBRE"],
                roads.loc[mask,"LONGITUD"],
                roads.loc[mask,"geometry"]
                )
        ]
    edges += list(
        zip(
            roads.loc[mask, "UNION_FIN"],
            roads.loc[mask, "UNION_INI"],
            edge_attributes
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

