import geopandas as gpd
import osmnx as ox
import pandas as pd
import networkx as nx
import pickle
import math

def load_and_preprocess_osmnx_graph(gdf_localities, id_city_label = "CVEGEO", **source_kwargs):
    """
    Load a road network from OSMnx.
    """
    # download road network with osmnx
    center = (
        source_kwargs["center_lat"],
        source_kwargs["center_lon"]
        )
    graph = ox.graph_from_point(
        center,
        dist = source_kwargs["network_radius"],
        network_type = "drive",
    )
    # convert nodes to geodataframe
    gdf_nodes, _ = ox.graph_to_gdfs(graph)
    gdf_nodes = gdf_nodes.reset_index()
    # spatial join to assign each node its locality (CVEGEO)
    gdf_nodes_labeled = gpd.sjoin(
        gdf_nodes, gdf_localities, how="left", predicate="within"
    )
    # write CVEGEO back into the graph as a node attribute
    cvegeo_map = gdf_nodes_labeled.set_index("osmid")[id_city_label].to_dict()
    cvegeo_map = {k: (None if pd.isna(v) else v) for k, v in cvegeo_map.items()}
    nx.set_node_attributes(graph, cvegeo_map, name=id_city_label)    
    
    return graph, gdf_nodes_labeled

def load_and_preprocess_inegi_graph(id_city_label = "CVEGEO", **source_kwargs):
    if source_kwargs["inegi_graph_path"] is None:
        raise ValueError("inegi_graph_path is required for source='inegi'")

    with open(source_kwargs["inegi_graph_path"], "rb") as f:
        graph = pickle.load(f)

    # rename id_polygon
    cvegeo_map = {}
    for node_id, data in graph.nodes(data=True):
        val = data.get("id_polygon")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            graph.nodes[node_id][id_city_label] = None
            cvegeo_map[node_id] = None
        else:
            graph.nodes[node_id][id_city_label] = int(val)
            cvegeo_map[node_id] = int(val)

    # build gdf_nodes_labeled
    nodes_data = [
        {"node_id": node_id, "x": data["x"], "y": data["y"],
         id_city_label: data.get(id_city_label)}
        for node_id, data in graph.nodes(data=True)
    ]
    df_nodes = pd.DataFrame(nodes_data)
    df_nodes["geometry"] = gpd.points_from_xy(df_nodes["x"], df_nodes["y"])
    gdf_nodes_labeled = gpd.GeoDataFrame(df_nodes, geometry="geometry")
    return graph, gdf_nodes_labeled
        


class Road_Network:
 
    # ------------------------------------------------------
    # Constructor
    # ------------------------------------------------------
    def __init__(
            self,
            source_kwargs,
            polygons_path,
            source = "inegi",
            id_city_label = "CVEGEO",
            keep_larger_cc = True,
    ):
        self.id_city_label = id_city_label
        if source == "osmnx":
            self.crs = "EPSG:4326"
            # load locality polygons
            self.gdf_localities = ( 
                gpd.read_file(polygons_path)
                .to_crs(self.crs)
            )
            
            graph, gdf_nodes_labeled = load_and_preprocess_osmnx_graph(
                self.gdf_localities,
                self.id_city_label,
                **source_kwargs
                )
            self.graph = graph
            self.gdf_nodes_labeled = gdf_nodes_labeled
            
        elif source == "inegi":
            crs = "EPSG:6372"
            # load locality polygons
            self.gdf_localities = ( 
                gpd.read_file(polygons_path)
                .to_crs(self.crs)
            )
            
            graph, gdf_nodes_labeled = load_and_preprocess_inegi_graph(
                self.id_city_label,
                **source_kwargs
                )
            self.graph = graph
            self.gdf_nodes = self.gdf_nodes_labeled.set_crs(crs)
        else:
            raise ValueError(f"Unknown source: {source!r}. Use 'osmnx' or 'inegi'.")