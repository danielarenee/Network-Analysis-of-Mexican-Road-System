import networkx as nx

from src.utils_2 import load_osmnx_graph, load_inegi_graph, to_connected, to_undirected, to_simple_graph
import src.utils as fc

class Road_Network:
 
    # ------------------------------------------------------
    # Constructor
    # ------------------------------------------------------
    def __init__(
            self,
            source_kwargs,
            source = "inegi",
            id_city_label = "CVEGEO",
            length_attr = "length",
            keep_larger_cc = True,
            to_undirected = True,
            to_simple = True
    ):
        self.id_city_label = id_city_label
        self.length_attr = length_attr
        self.source = source
        self.gdf_localities = None
        
        if source == "osmnx":
            self.crs = "EPSG:4326"
            self.graph = load_osmnx_graph(**source_kwargs)
        elif source == "inegi":
            self.crs = "EPSG:6372"
            self.graph = load_inegi_graph(**source_kwargs)
            self.preprocess_inegi_graph()
        else:
            raise ValueError(f"Unknown source: {source!r}. Use 'osmnx' or 'inegi'.")

        # Keep only the major connected component
        if keep_larger_cc:
            self._to_connected()
        if to_undirected:
            self._to_undirected()
        if to_simple:
            self._to_simple_graph()
            
    def plot_labeled_network(self):
        fc.plot_labeled_network(
            graph = self.graph,
            gdf_nodes_labeled = self.gdf_nodes_labeled,
            gdf_localities = self.gdf_localities,
            source  = self.source
        )

    def preprocess_inegi_graph(self):
        import math
        import pandas as pd
        import geopandas as gpd
               
        # rename id_polygon
        region_map = {}
        for node_id, data in self.graph.nodes(data=True):
            val = data.get("id_polygon")
            if val is None or (isinstance(val, float) and math.isnan(val)):
                self.graph.nodes[node_id][self.id_city_label] = None
                region_map[node_id] = None
            else:
                self.graph.nodes[node_id][self.id_city_label] = int(val)
                region_map[node_id] = int(val)

        # build gdf_nodes_labeled
        nodes_data = [
            {"node_id": node_id, "x": data["x"], "y": data["y"],
             self.id_city_label: data.get(self.id_city_label)}
            for node_id, data in self.graph.nodes(data=True)
        ]
        df_nodes = pd.DataFrame(nodes_data)
        df_nodes["geometry"] = gpd.points_from_xy(df_nodes["x"], df_nodes["y"])
        self.gdf_nodes_labeled = gpd.GeoDataFrame(
            df_nodes, 
            geometry="geometry", 
            crs=self.crs
        )
            
    def _to_connected(self):
        self.graph = to_connected(self.graph)
        
    def _to_undirected(self):
        self.graph = to_undirected(self.graph)
        
    def _to_simple_graph(self):
        self.graph = to_simple_graph(self.graph, self.length_attr)
            

