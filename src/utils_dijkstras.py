import igraph as ig
import imageio.v2 as imageio
import networkx as nx
import matplotlib.pyplot as plt
import heapq

from matplotlib.collections import LineCollection
from numpy import inf, asarray
from time import time as t
from glasbey import create_palette
from seaborn import color_palette

from IPython.display import display, Video


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
        step = 10
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
    
    if draw:
        print("Creating color palette...")
        colors = create_colors(R, id_external)
        print("Creating initial figure...")
        fig, ax, scatter, edge_collection, edge_colors =  init_draw(g, R, colors)
        frame = capture_frame(fig, scatter, edge_collection, edge_colors, R, F, colors)
        frames = [frame]
        updates = 0
    
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
                if draw:                   
                    updates += 1
                    if updates % step == 0:
                        frame = capture_frame(fig, scatter, edge_collection, edge_colors, 
                                              R, F, colors,
                                              edge_id = e_id, base_vertex = u)
                        frames.append(frame)
        F[u] = flag
    if draw:
        frame = capture_frame(fig, scatter, edge_collection, edge_colors, R, F, colors)
        frames.append(frame)
        imageio.mimsave(
            f"video_test.mp4",
            frames,
            fps=1,
            codec="libx264",
            macro_block_size=None
        )
    final_time = t()-start
    
    print("Iterations: ", contador)
    print("Time: ", final_time, " s")
    return d, p, R, F

def create_colors(
        labels,
        id_external = None,
        color_external_node = "lightgray",
        method="seaborn"
        ):
    labels = list(labels)
    n_colors = len(labels)

    if method == "glasbey":
        palette = create_palette(palette_size=n_colors)

    elif method == "seaborn":
        palette = color_palette("husl", n_colors=n_colors).as_hex()
    colors = dict(zip(labels, palette))
    colors[id_external] = color_external_node
    return colors
    

def init_draw(
        g, 
        R,
        colors, 
        figsize = (25, 25),
        node_size = 1,
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


def capture_frame(
        fig,
        scatter,
        edge_collection,
        edge_colors,
        R,
        F,
        colors,
        edge_id=None,
        base_vertex=None):

    # Actualizar colores de nodos
    node_colors = [colors[r] for r in R]
    scatter.set_color(node_colors)

    # Tamaños según F
    node_sizes = [1 if f else 1 for f in F]
    scatter.set_sizes(node_sizes)

    # Actualizar permanentemente color de arista aceptada
    if edge_id is not None and base_vertex is not None:
        edge_colors[edge_id] = colors[R[base_vertex]]

    # Estilos iniciales de todas las aristas
    edge_styles = ["solid"] * len(edge_colors)

    edge_collection.set_color(edge_colors)
    edge_collection.set_linestyle(edge_styles)

    fig.canvas.draw()
    frame = asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    return frame

