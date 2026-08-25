import igraph as ig
import networkx as nx
import heapq

from numpy import inf


node_attributes_labels = ["id_polygon", "x", "y"]
edge_attributes_labels = ["name", "length", "geometry"]

def networkx_to_igraph(
        nx_graph: nx.Graph,
        id_city_label: str
        ) -> ig.Graph:
    
    nodes = list(nx_graph.nodes)
    n = len(nodes)
    nodes_dict = {node: index for index, node in enumerate(nodes)}
    
    ig_graph = ig.Graph(
        n = n,
        directed = nx_graph.is_directed(),
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
        id_external = None,
        draw = False
        ):
    n = g.vcount()
    p = [None] * n
    R = list(g.vs[id_city])
    d = [inf if R[v] == id_external else 0 for v in range(n)]
    
    Q = []
    for u in range(n):
        heapq.heappush(Q, (d[u], u))
     
    contador = 0
    while Q:
        dist_u, u = heapq.heappop(Q)
    
        if dist_u > d[u]:
            continue
    
        for v in g.neighbors(u):
            try:
                e_id = g.get_eid(u, v)
            except:
                e_id = g.get_eid(v, u)
            w_uv = g.es[e_id]["length"]
    
            if d[u] + w_uv < d[v]:
                contador += 1
                d[v] = d[u] + w_uv
                p[v] = u
                R[v] = R[u]
                heapq.heappush(Q, (d[v], v))
    
    print(contador)
    return d, p, R