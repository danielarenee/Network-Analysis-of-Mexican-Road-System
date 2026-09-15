import math
import igraph as ig
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
        return graph
            
    if graph.is_multigraph():
        graph_undirected = nx.MultiGraph()
    else:
        graph_undirected = nx.Graph()
    
    graph_undirected.graph.update(graph.graph)
    graph_undirected.add_nodes_from(graph.nodes(data=True))
                         
    if graph.is_multigraph():
        for u, v, k, attr in graph.edges(keys=True, data=True):
            graph_undirected.add_edge(u, v, key=k, **attr)
    else:
        for u, v, attr in graph.edges(data=True):
            graph_undirected.add_edge(u, v, **attr)
    
    return graph_undirected


def to_simple_graph(graph, length_attr="length"):
    if not graph.is_multigraph():
        return graph
    
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


def networkx_to_igraph(
        nx_graph: nx.Graph,
        id_city_label: str,
        node_attributes_labels =  ["id_polygon", "x", "y"],
        edge_attributes_labels = ["name", "length", "geometry"]
        ) -> ig.Graph:
    
    nodes = list(nx_graph.nodes)
    n = len(nodes)
    nodes_dict = {node: index for index, node in enumerate(nodes)}
    
    directed = nx_graph.is_directed()
    
    ig_graph = ig.Graph(
        n = n,
        directed = directed,
    )

    ig_graph.vs["id_nx"] = nodes

    for attribute in node_attributes_labels + [id_city_label]:
        ig_graph.vs[attribute] = [
            nx_graph.nodes[node].get(attribute)
            for node in nodes
        ]

    edges = []
    edges_attributes = []
    
    if nx_graph.is_multigraph():
        shortest_edges = {}
        for source, target, _, data in nx_graph.edges(keys=True, data=True):
            pair = (source, target)
            if (pair not in shortest_edges
                or data["length"] < shortest_edges[pair]["length"]):
                shortest_edges[pair] = data
        edge_iterator = ((source, target, data)
                         for (source, target), data in shortest_edges.items())
    else:
        edge_iterator = nx_graph.edges(data=True)
    
    for source, target, data in edge_iterator:
        edge = (nodes_dict[source], nodes_dict[target])
        edges.append(edge)
        edges_attributes.append(dict(data))

    ig_graph.add_edges(edges)
    
    for attribute in edge_attributes_labels:
        ig_graph.es[attribute] = [
            data.get(attribute)
            for data in edges_attributes
        ]
        
    for attribute, value in nx_graph.graph.items():
        ig_graph[attribute] = value
    
    return ig_graph


def igraph_to_gdf(
        g : ig.Graph,
        crs = "EPSG:6372",
        R = None,
        d = None,
        ):
    
    node_ids = list(range(g.vcount()))

    nodes_df  = pd.DataFrame({
        "node_id": node_ids,
        "x": g.vs["x"],
        "y": g.vs["y"],
        "id_nx": g.vs["id_nx"],
        "id_polygon": g.vs["id_polygon"],
        "R": R,
    })
    if d is not None:
        nodes_df["d"] = d
    if R is not None:
        nodes_df["R"] = R
        
    nodes_gdf  = gpd.GeoDataFrame(
        nodes_df ,
        geometry = gpd.points_from_xy(nodes_df ["x"], nodes_df ["y"]),
        crs = crs
        )
    
    edges_df = (
        g.get_edge_dataframe()
        .rename_axis("edge_id")
        .reset_index()
    )
    edges_gdf = gpd.GeoDataFrame(
        edges_df,
        geometry="geometry",
        crs=crs,
    )
    
    return nodes_gdf, edges_gdf
