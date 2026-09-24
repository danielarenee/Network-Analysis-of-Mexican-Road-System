import math
import igraph as ig
import networkx as nx
import pandas as pd
import geopandas as gpd

def load_osmnx_graph(**source_kwargs):
    import osmnx as ox
    
    # Define network center
    center = (
        source_kwargs["center_lat"],
        source_kwargs["center_lon"]
        )
    # Download driving network
    graph = ox.graph_from_point(
        center,
        dist = source_kwargs["network_radius"],
        network_type = "drive",
    )
    return graph


def load_inegi_graph(**source_kwargs):
    import pickle
    # Load serialized graph
    with open(source_kwargs["inegi_graph_path"], "rb") as f:
        graph = pickle.load(f)
    return graph


def preprocess_inegi_graph(graph, id_city_label, crs):         
    # Standardize region identifiers
    region_map = {}
    for node_id, data in graph.nodes(data=True):
        val = data.get("id_polygon")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            graph.nodes[node_id][id_city_label] = None
            region_map[node_id] = None
        else:
            graph.nodes[node_id][id_city_label] = int(val)
            region_map[node_id] = int(val)

    # Build node table
    nodes_data = [
        {"node_id": node_id, "x": data["x"], "y": data["y"],
         id_city_label: data.get(id_city_label)}
        for node_id, data in graph.nodes(data=True)
    ]
    df_nodes = pd.DataFrame(nodes_data)
    
    # Create node geometries
    df_nodes["geometry"] = gpd.points_from_xy(df_nodes["x"], df_nodes["y"])
    gdf_nodes_labeled = gpd.GeoDataFrame(
        df_nodes, 
        geometry="geometry", 
        crs = crs
    )
    
    return gdf_nodes_labeled, region_map


def to_connected(graph):
    """
    Keep only the largest connected component of the graph.

    Parameters
    ----------
    graph : networkx.Graph
        Input road network.

    Returns
    -------
    networkx.Graph
        Copy of the largest connected component.
    """
    # Select the appropriate connectivity criterion
    if graph.is_directed():
        cc = nx.weakly_connected_components(graph)
    else:
        cc = nx.connected_components(graph)
        
    # Extract largest component
    larger_cc_nodes = max(cc, key=len)
    graph_connected = graph.subgraph(larger_cc_nodes)
    
    return graph_connected.copy()


def to_undirected(graph):
    """
    Convert a directed graph to an undirected graph.


    Parameters
    ----------
    graph : networkx.Graph
        Input graph.

    Returns
    -------
    networkx.Graph
        Undirected graph with the original attributes.
    """
    # Return unchanged if already undirected
    if not graph.is_directed():
        return graph
    
    # Preserve multigraph structure
    if graph.is_multigraph():
        graph_undirected = nx.MultiGraph()
    else:
        graph_undirected = nx.Graph()
    
    # Copy graph and node attributes
    graph_undirected.graph.update(graph.graph)
    graph_undirected.add_nodes_from(graph.nodes(data=True))
    
    # Copy edges and their attributes
    if graph.is_multigraph():
        for u, v, k, attr in graph.edges(keys=True, data=True):
            graph_undirected.add_edge(u, v, key=k, **attr)
    else:
        for u, v, attr in graph.edges(data=True):
            graph_undirected.add_edge(u, v, **attr)
    
    return graph_undirected.copy()


def to_simple_graph(graph, length_attr="length"):
    """
    Convert a multigraph into a simple graph
    Keeping the edge with the minimum length

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.
    length_attr : str, default="length"
        Edge attribute used to select among parallel edges.

    Returns
    -------
    networkx.Graph
        Simple graph containing the shortest parallel edge.
    """
    # Return unchanged if the graph is already simple
    if not graph.is_multigraph():
        return graph
    
    # Preserve graph direction
    if graph.is_directed():
        simple_graph = nx.DiGraph()
    else:
        simple_graph = nx.Graph()
    
    # Copy graph and node attributes
    simple_graph.graph.update(graph.graph)
    simple_graph.add_nodes_from(graph.nodes(data = True))
    
    # Keep the shortest edge between each node pair
    for u, v, k, attr in graph.edges(keys=True, data=True):
        
        if length_attr not in attr:
            raise KeyError(f"{length_attr} is not an attribute")
        candidate_length  = attr[length_attr]
        
        # Add first edge found between u and v
        if not simple_graph.has_edge(u, v):
            simple_graph.add_edge(u, v)
            simple_graph[u][v].update(attr)
            continue
        current_length = simple_graph[u][v][length_attr]
        
        # Replace current edge if the candidate is shorter
        if candidate_length < current_length:
            simple_graph[u][v].clear()
            simple_graph[u][v].update(attr)            
    return simple_graph.copy()

def igraph_to_networkx(
        ig_graph: ig.Graph
        ) -> nx.Graph:
    """Convert an igraph.Graph to NetworkX representation."""
    directed = ig_graph.is_directed()
    multiple = ig_graph.has_multiple()

    if directed:
        nx_graph = nx.MultiDiGraph() if multiple else nx.DiGraph()
    else:
        nx_graph = nx.MultiGraph() if multiple else nx.Graph()

    node_ids = ig_graph.vs["node_id"]

    if len(set(node_ids)) != len(node_ids):
        raise ValueError("'node_id' values ​​must be unique.")

    for vertex, node_id in zip(ig_graph.vs, node_ids):
        attributes = vertex.attributes()
        attributes.pop("node_id")  
        nx_graph.add_node(node_id, **attributes)

    for edge in ig_graph.es:
        source = node_ids[edge.source]
        target = node_ids[edge.target]
        nx_graph.add_edge(source, target, **edge.attributes())

    nx_graph.graph.update({
        attribute: ig_graph[attribute]
        for attribute in ig_graph.attributes()
    })

    return nx_graph

def networkx_to_igraph(
        nx_graph: nx.Graph,
        id_city_label: str,
        node_attributes_labels =  ["id_polygon", "x", "y", "boundary"],
        edge_attributes_labels = ["id_net", "name", "length", "geometry"]
        ) -> ig.Graph:
    """
    Convert a NetworkX graph into an igraph representation.
 
    Parameters
    ----------
    nx_graph : networkx.Graph
        Input NetworkX graph.
    id_city_label : str
        Node attribute containing the region identifier.
    node_attributes_labels : list, optional
        Node attributes to copy to igraph.
    edge_attributes_labels : list, optional
        Edge attributes to copy to igraph.
 
    Returns
    -------
    igraph.Graph
        igraph representation.
    """
    # Map NetworkX node IDs to igraph indices
    nodes = list(nx_graph.nodes)
    n = len(nodes)
    nodes_dict = {node: index for index, node in enumerate(nodes)}
    
    directed = nx_graph.is_directed()
    
    # Initialize igraph structure
    ig_graph = ig.Graph(
        n = n,
        directed = directed,
    )

    # Preserve original node identifiers
    ig_graph.vs["node_id"] = nodes
    # Copy node attributes
    for attribute in node_attributes_labels + [id_city_label]:
        ig_graph.vs[attribute] = [
            nx_graph.nodes[node].get(attribute)
            for node in nodes
        ]

    edges = []
    edges_attributes = []
    
    # Reduce parallel edges to the shortest one
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
    
    # Convert edge endpoints to igraph indices
    for source, target, data in edge_iterator:
        edge = (nodes_dict[source], nodes_dict[target])
        edges.append(edge)
        edges_attributes.append(dict(data))
    
    # Add edges
    ig_graph.add_edges(edges)
    # Copy edge attributes
    for attribute in edge_attributes_labels:
        ig_graph.es[attribute] = [
            data.get(attribute)
            for data in edges_attributes
        ]
    # Copy graph-level attributes
    for attribute, value in nx_graph.graph.items():
        ig_graph[attribute] = value
    
    return ig_graph


def igraph_to_gdf(
        g : ig.Graph,
        crs = "EPSG:6372",
        R = None,
        d = None,
        ):
    """
    Convert an igraph road network into node and edge GeoDataFrames.

    Parameters
    ----------
    g : igraph.Graph
        Input road network.
    crs : str, default="EPSG:6372"
        Coordinate reference system assigned to the GeoDataFrames.
    R : list, optional
        Region associated with each vertex.
    d : list, optional
        Distance associated with each vertex.

    Returns
    -------
    nodes_gdf : geopandas.GeoDataFrame
        Vertex attributes and point geometries.
    edges_gdf : geopandas.GeoDataFrame
        Edge attributes and geometries.
    """
    # Build node attribute table
    node_ids = list(range(g.vcount()))
    nodes_df  = pd.DataFrame({
        "id": node_ids,
        "x": g.vs["x"],
        "y": g.vs["y"],
        "node_id": g.vs["node_id"],
        "id_polygon": g.vs["id_polygon"],
        "R": R,
        "boundary": g.vs["boundary"]
    })
    
    # Add optional algorithm outputs
    if d is not None:
        nodes_df["d"] = d
    if R is not None:
        nodes_df["R"] = R
    
    # Create node geometries
    nodes_gdf  = gpd.GeoDataFrame(
        nodes_df ,
        geometry = gpd.points_from_xy(nodes_df ["x"], nodes_df ["y"]),
        crs = crs
        )
    # Extract edge attributes and geometries
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
