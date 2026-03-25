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
DATA_DIR = str(BASE_DIR)


###########
# FUNCTIONS
###########
def create_road_network(roads, unions, save_folder = "temp"):
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
    with open(DATA_DIR + f"\\{save_folder}\\road_network.pkl", "wb") as f:
        pickle.dump(
            G, f,
            protocol = pickle.HIGHEST_PROTOCOL
            )
    return G


###########
# MAIN
###########

# folder = "temp"
folder = "test"

# Import .gpkg of roads and unions
unions = gpd.read_file(DATA_DIR + f"\\{folder}\\unions.gpkg")
roads = gpd.read_file(DATA_DIR + f"\\{folder}\\roads.gpkg")

# Create road network
create_road_network(roads, unions, save_folder = folder)
