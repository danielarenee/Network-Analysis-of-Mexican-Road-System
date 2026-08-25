import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt
import heapq

from matplotlib.collections import LineCollection
from numpy import inf, asarray
from time import time as t
from glasbey import create_palette


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
        id_external = None,
        draw = False,
        **draw_kwargs
        ):
    # Number of vertices
    n = g.vcount()
    # List of predecessors
    p = [None] * n
    # List of regions
    R = list(g.vs[id_city])
    # List of distances
    d = [inf if R[v] == id_external else 0 for v in range(n)]
    
    # Priority queue
    Q = []
    for u in range(n):
        heapq.heappush(Q, (d[u], u))
    
    if draw:
        colors = create_colors(R, id_external)
        fig, ax, scatter, edge_collection, edge_colors =  init_draw(g, R, colors)
        
    
    start = t()
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
    final_time = t()-start
    
    print("Iterations: ", contador)
    print("Time: ", final_time, " s")
    return d, p, R

def create_colors(labels,
                  id_external = None,
                  color_external_node = "lightgray"):
    palette = create_palette(palette_size = len(labels))
    colors = {i: color for i, color in zip(labels, palette)}
    colors[id_external] = color_external
    return colors

def init_draw(
        g, 
        R,
        colors, 
        figsize = (10, 10),
        node_size = 15,
        edge_width = 2,
        initial_edge_color = "gray",
        background_color = "darkslategray",
        ):
    
    xs = g.vs["x"]
    ys = g.vs["y"]
    
    fig, ax = plt.subplots(figsize = figsize)
      
    segments = [
        [(xs[e.source], ys[e.source]),
         (xs[e.target], ys[e.target])]
        for e in g.es
    ]
    # Initial colors of edges
    edge_colors = [ initial_edge_color ] * g.ecount()
    edge_collection = LineCollection(
        segments,
        colors = edge_colors,
        linewidths = edge_width,
        zorder=2
    )
    ax.add_collection(edge_collection)
      
    # Límites según la red
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
      
    ax.set_aspect("equal")
    ax.axis("off")
      
    # Capa dinámica: nodos
    node_colors = [colors[r] for r in R]
    scatter = ax.scatter(
        xs,
        ys,
        s = node_size,
        c = node_colors,
        zorder = 3
    )
      
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    return fig, ax, scatter, edge_collection, edge_colors
