###########
# PACKAGES
###########
# Import packages
from geopandas  import read_file
import networkx as nx
import pickle

from pathlib import Path

###########
# VARIABLES
###########
# Set absolute path
BASE_DIR = Path(__file__).resolve().parent.parent


###########
# FUNCTIONS
###########
def create_road_network(roads, unions):
    # Create digraph
    G = nx.MultiDiGraph()
    
    # Set graphs attributes
    crs = unions.crs
    epsg = crs.to_epsg()
    G.graph = {"crs": f"epsg:{epsg}"}
    
    # Create and add nodes in bulk
    print("Creating nodes list...")
    nodes = [
        (node_id, {"x": geom.x, "y": geom.y, "id_polygon": idx, "ent": ent})
        for node_id, geom, idx, ent in zip(
            unions["ID_UNION"],
            unions["geometry"],
            unions["id_convex"],
            unions["CVE_ENT"]
        )
    ]
    print("Adding nodes...")
    G.add_nodes_from(nodes)
    
    print("Creating edges list...")
    # Forward edges
    forward_edges = [
        (u, v, {"name": name, "length": length, "geometry": geom, "id_net": idx})
        for u, v, name, length, geom, idx in zip(
            roads["UNION_INI"],
            roads["UNION_FIN"],
            roads["NOMBRE"],
            roads["LONGITUD"],
            roads["geometry"],
            roads["ID_RED"]
        )
    ]
    
    # Reverse edges for two-way roads
    mask = roads["CIRCULA"] == "Dos sentidos"
    two_way_roads = roads[mask]
    reverse_edges = [
        (v, u, {"name": name, "length": length, "geometry": geom.reverse(), "id_net": idx})
        for u, v, name, length, geom, idx in zip(
            two_way_roads["UNION_INI"],
            two_way_roads["UNION_FIN"],
            two_way_roads["NOMBRE"],
            two_way_roads["LONGITUD"],
            two_way_roads["geometry"],
            two_way_roads["ID_RED"]
        )
    ]
    
    edges = forward_edges + reverse_edges
    
    # Add edges from roads (lines) in bulk
    print("Adding edges...")
    G.add_edges_from(edges)
    
    # Remove isolated nodes
    print("Removing isolated nodes...")
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    # Check nodes
    print(f"Nodes: {G.order():,}")
    
    # Check edges
    print(f"Edges: {G.size():,}")
    
    return G


###########
# MAIN
###########

# Paths of roads and unions
unions_path = BASE_DIR / "data" / "processed" / "unions.gpkg"
roads_path = BASE_DIR / "data" / "processed" / "roads.gpkg"
graph_save_path = BASE_DIR / "data" / "processed"

# Import .gpkg of roads and unions
print("Reading geospatial data...")
unions = read_file(unions_path)
roads = read_file(roads_path)

# Create road network
print("Building graph...")
G = create_road_network(roads, unions)

# Save graph
save_path = graph_save_path / "road_network.pkl"
save_path.parent.mkdir(parents=True, exist_ok=True)
print("Saving graph...")
with open(save_path, "wb") as f:
    pickle.dump(
        G, f,
        protocol = pickle.HIGHEST_PROTOCOL
        )

for i in range(1, 33):
    print(f"Extracting subraph {i}")
    ent = str(i).zfill(2)

    nodes = [
        node
        for node, attributes in G.nodes(data=True)
        if str(attributes.get("ent")).zfill(2) == ent
    ]

    H = G.subgraph(nodes).copy()

    print(f"Saving subraph {i}")
    save_path = graph_save_path / f"road_network_{ent}.pkl"
    
    with open(save_path, "wb") as f:
        pickle.dump(
            H, f,
            protocol = pickle.HIGHEST_PROTOCOL
            )
