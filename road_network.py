import networkx as nx

from src.utils_2 import load_osmnx_graph, load_inegi_graph
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
            self.graph, 
            self.gdf_nodes_labeled, 
            source="inegi")

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
        if self.graph.is_directed():
            cc = nx.weakly_connected_components(self.graph)
        else:
            cc = nx.connected_components(self.graph)
        larger_cc_nodes = max(cc, key=len)
        self.graph = self.graph.subgraph(larger_cc_nodes).copy()
        
    def _to_undirected(self):
        if not self.graph.is_directed():
            return
                
        graph = nx.MultiGraph()
        
        graph.graph.update(self.graph.graph)
        graph.add_nodes_from(self.graph.nodes(data=True))
                             
        if self.graph.is_multigraph():
            edges = self.graph.edges(keys=True, data=True)
        else:
            edges = (
                (u, v, None, attr) 
                for u, v, attr  in self.graph.edges(data=True)
            )
        for u, v, k, attr in edges:
            new_key = (u, v, k)
            graph.add_edge(u, v, new_key, **attr)
            graph.edges[u, v, new_key].update(attr)
        
        self.graph = graph
        
    def _to_simple_graph(self):
        if not self.graph.is_multigraph():
            return
        
        if self.graph.is_directed():
            simple_graph = nx.DiGraph()
        else:
            simple_graph = nx.Graph()
        
        simple_graph.graph.update(self.graph.graph)
        simple_graph.add_nodes_from(self.graph.nodes(data = True))
        
        for u, v, k, attr in self.graph.edges(keys=True, data=True):
            if self.length_attr not in attr:
                raise KeyError(f"{self.length_attr} is not an attribute")
            candidate_length  = attr[self.length_attr]
            if not simple_graph.has_edge(u, v):
                simple_graph.add_edge(u, v)
                simple_graph[u][v].update(attr)
                continue
            current_length = simple_graph[u][v][self.length_attr]
            if candidate_length < current_length:
                simple_graph[u][v].clear()
                simple_graph[u][v].update(attr)            
        self.graph = simple_graph
            

