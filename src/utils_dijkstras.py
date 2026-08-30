import igraph as ig
import networkx as nx
import heapq

from numpy import inf
from time import time as t
from pandas import DataFrame
from geopandas import GeoDataFrame, points_from_xy


node_attributes_labels = ["id_polygon", "x", "y"]
edge_attributes_labels = ["name", "length", "geometry"]

def networkx_to_igraph(
        nx_graph: nx.Graph,
        id_city_label: str,
        directed = False
        ) -> ig.Graph:
    
    nodes = list(nx_graph.nodes)
    n = len(nodes)
    nodes_dict = {node: index for index, node in enumerate(nodes)}
    
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

def dijkstra_city_network(
        g: ig.Graph,
        id_city : str,
        id_external = None
        ):
    print("Initializing...")
    # Number of vertices
    n = g.vcount()
    # List of predecessors
    p = [None] * n
    # List of regions
    R = list(g.vs[id_city])
    # List of distances
    d = [inf if R[v] == id_external else 0 for v in range(n)]
    
    F = [False if R[v] == 0 else True for v in range(n)]
    
    # Priority queue
    Q = []
    for u in range(n):
        heapq.heappush(Q, (d[u], u))
    
    print("Running...")
    start = t()
    contador = 0
    while Q:
        dist_u, u = heapq.heappop(Q)
    
        if dist_u > d[u]:
            continue
    
        flag = False
        for v in g.neighbors(u):
            try:
                e_id = g.get_eid(u, v)
            except:
                e_id = g.get_eid(v, u)
            w_uv = g.es[e_id]["length"]
    
            if d[u] + w_uv < d[v]:
                contador += 1
                if p[v] is not None:
                    F[p[v]] = True
                d[v] = d[u] + w_uv
                p[v] = u
                R[v] = R[u]
                F[v] = True
                heapq.heappush(Q, (d[v], v))
            else:
                if R[v] != R[u]:
                  flag = True
        F[u] = flag
    final_time = t()-start
    
    print("Iterations: ", contador)
    print("Time: ", final_time, " s")
    return d, p, R, F

def igraph_to_gdf(
        g : ig.Graph,
        R : list,
        d = None,
        crs = "EPSG:6372",
        ):
    node_ids = list(range(g.vcount()))

    nodes_df  = DataFrame({
        "node_id": node_ids,
        "x": g.vs["x"],
        "y": g.vs["y"],
        "id_nx": g.vs["id_nx"],
        "id_polygon": g.vs["id_polygon"],
        "R": R,
    })
    if d is not None:
        nodes_df["d"] = d
    nodes_gdf  = GeoDataFrame(
        nodes_df ,
        geometry = points_from_xy(nodes_df ["x"], nodes_df ["y"]),
        crs = crs
        )
    
    edges_df = (
        g.get_edge_dataframe()
        .rename_axis("edge_id")
        .reset_index()
    )
    edges_gdf = GeoDataFrame(
        edges_df,
        geometry="geometry",
        crs=crs,
    )
    
    return nodes_gdf, edges_gdf

