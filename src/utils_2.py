import math
import networkx as nx
import pandas as pd
import geopandas as gpd

def load_osmnx_graph(**source_kwargs):
    import osmnx as ox
    
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
    return graph


def load_inegi_graph(**source_kwargs):
    import pickle
    with open(source_kwargs["inegi_graph_path"], "rb") as f:
        graph = pickle.load(f)
    return graph


def preprocess_inegi_graph(graph, id_city_label, crs):           
    # rename id_polygon
    region_map = {}
    for node_id, data in graph.nodes(data=True):
        val = data.get("id_polygon")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            graph.nodes[node_id][id_city_label] = None
            region_map[node_id] = None
        else:
            graph.nodes[node_id][id_city_label] = int(val)
            region_map[node_id] = int(val)

    # build gdf_nodes_labeled
    nodes_data = [
        {"node_id": node_id, "x": data["x"], "y": data["y"],
         id_city_label: data.get(id_city_label)}
        for node_id, data in graph.nodes(data=True)
    ]
    df_nodes = pd.DataFrame(nodes_data)
    df_nodes["geometry"] = gpd.points_from_xy(df_nodes["x"], df_nodes["y"])
    gdf_nodes_labeled = gpd.GeoDataFrame(
        df_nodes, 
        geometry="geometry", 
        crs = crs
    )
    return gdf_nodes_labeled, region_map

def to_connected(graph):
    if graph.is_directed():
        cc = nx.weakly_connected_components(graph)
    else:
        cc = nx.connected_components(graph)
    larger_cc_nodes = max(cc, key=len)
    graph_connected = graph.subgraph(larger_cc_nodes).copy()
    
    return graph_connected


def to_undirected(graph):
    if not graph.is_directed():
        return
            
    graph_undirected = nx.MultiGraph()
    
    graph_undirected.graph.update(graph.graph)
    graph_undirected.add_nodes_from(graph.nodes(data=True))
                         
    if graph.is_multigraph():
        edges = graph.edges(keys=True, data=True)
    else:
        edges = (
            (u, v, None, attr) 
            for u, v, attr  in graph.edges(data=True)
        )
    for u, v, k, attr in edges:
        new_key = (u, v, k)
        graph_undirected.add_edge(u, v, new_key, **attr)
        graph_undirected.edges[u, v, new_key].update(attr)
    
    return graph_undirected


def to_simple_graph(graph, length_attr="length"):
    if not graph.is_multigraph():
        return
    
    if graph.is_directed():
        simple_graph = nx.DiGraph()
    else:
        simple_graph = nx.Graph()
    
    simple_graph.graph.update(graph.graph)
    simple_graph.add_nodes_from(graph.nodes(data = True))
    
    for u, v, k, attr in graph.edges(keys=True, data=True):
        if length_attr not in attr:
            raise KeyError(f"{length_attr} is not an attribute")
        candidate_length  = attr[length_attr]
        if not simple_graph.has_edge(u, v):
            simple_graph.add_edge(u, v)
            simple_graph[u][v].update(attr)
            continue
        current_length = simple_graph[u][v][length_attr]
        if candidate_length < current_length:
            simple_graph[u][v].clear()
            simple_graph[u][v].update(attr)            
    return simple_graph
    