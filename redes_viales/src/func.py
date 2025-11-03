"""
Graph utility functions for road network analysis.

This module provides functions for analyzing and simplifying road networks represented
as NetworkX graphs. It includes functionality for:
- Computing heuristics for pathfinding algorithms
- Building clique graphs from locality boundary nodes
- Pruning graphs by removing low-degree vertices
- Simplifying multiple edges by keeping the shortest one


"""

import networkx as nx
from math import sqrt
from shapely.geometry import LineString
from collections import deque


def euclidean_heuristic(u, v, graph):
    """
    Calculate the Euclidean distance between two nodes in a graph.

    This function is commonly used as a heuristic for A* pathfinding algorithm,
    providing an admissible estimate of the distance between nodes based on
    their geographic coordinates.

    Parameters
    ----------
    u : hashable
        The identifier of the first node.
    v : hashable
        The identifier of the second node.
    graph : networkx.Graph
        The graph containing both nodes. Nodes must have 'x' and 'y' attributes
        representing their coordinates.

    Returns
    -------
    float
        The Euclidean distance between nodes u and v.

    """
    x1, y1 = graph.nodes[u]['x'], graph.nodes[u]['y']
    x2, y2 = graph.nodes[v]['x'], graph.nodes[v]['y']
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)

def construir_clique_localidad(graph, cvgeo_target, nodos_frontera):
    """
    Build a clique graph for a specific locality based on boundary nodes.

    This function creates a complete graph (clique) where nodes represent boundary
    points of a locality (identified by CVEGEO code), and edges represent the
    shortest paths between these boundary nodes within the locality. The shortest
    paths are computed using the A* algorithm with Euclidean distance heuristic.

    Algorithm steps:
    1. Filter all nodes belonging to the target locality (cvgeo_target)
    2. Extract the induced subgraph for that locality
    3. Identify boundary nodes for the locality
    4. Create a new clique graph with these boundary nodes
    5. Connect each pair of boundary nodes with A* shortest path
    6. Store the path length and route in edge attributes

    Parameters
    ----------
    graph : networkx.Graph
        The original road network graph. Nodes must have 'CVEGEO', 'x', and 'y'
        attributes.
    cvgeo_target : str
        The CVEGEO code (geographic identifier) of the target locality.
    nodos_frontera : dict
        Dictionary mapping CVEGEO codes to sets of boundary node IDs.
        Format: {cvgeo: {node_id1, node_id2, ...}}

    Returns
    -------
    networkx.Graph
        A clique graph where:
        - Nodes are boundary nodes of the locality (with original attributes)
        - Edges connect all pairs of boundary nodes that have a path
        - Edge attributes include:
            * 'weight': The total Euclidean distance of the path
            * 'path': List of node IDs representing the shortest path

    Notes
    -----
    - If no path exists between two boundary nodes, they are not connected
    - The function uses A* pathfinding for efficiency
    - Edge weights are computed as sum of Euclidean distances between consecutive
      nodes in the path
    """
    # Step 1: Filter nodes belonging to the target locality
    nodos_localidad = []

    for node_id, data in graph.nodes(data=True):
        cvgeo = data.get("CVEGEO")
        if cvgeo == cvgeo_target:
            nodos_localidad.append(node_id)

    # Step 2: Extract induced subgraph for this locality
    subgrafo = graph.subgraph(nodos_localidad).copy()

    # Step 3: Get boundary nodes for this locality
    # If locality has no boundary nodes, returns empty set
    frontera = nodos_frontera.get(cvgeo_target, set())
    frontera_lista = list(frontera)
    # Ensure boundary nodes are in the subgraph
    frontera_lista = [n for n in frontera_lista if n in subgrafo]

    # Step 4: Create new clique graph with boundary nodes
    grafo_clique = nx.Graph()
    for nodo in frontera:
        # Add boundary nodes with their original attributes
        grafo_clique.add_node(nodo, **graph.nodes[nodo])

    # Step 5: Connect each pair of boundary nodes using A* shortest path
    for i in range(len(frontera_lista)):
        for j in range(i + 1, len(frontera_lista)):
            u = frontera_lista[i]
            v = frontera_lista[j]
            try:
                # Find shortest path using A* with Euclidean heuristic
                camino = nx.astar_path(subgrafo, u, v, heuristic = lambda a, b: euclidean_heuristic(a, b, subgrafo))
                # Calculate total path weight (sum of Euclidean distances)
                peso = sum(
                    euclidean_heuristic(camino[k], camino[k + 1], subgrafo)
                    for k in range(len(camino) - 1)
                )
                grafo_clique.add_edge(u, v, weight=peso, path=camino)
            except nx.NetworkXNoPath:
                # Skip if no path exists between boundary nodes
                continue

    return grafo_clique

def podar_grado_1(graph, min_degree=1):
    """
    Iteratively prune leaf nodes (low-degree vertices) from a graph.

    This function removes nodes with degree <= min_degree in an iterative manner.
    When a node is removed, its neighbors might also become leaf nodes, so they
    are added to the processing queue. This process continues until no more
    nodes can be pruned.

    The pruning algorithm uses a breadth-first approach with a queue to ensure
    efficient removal of all qualifying nodes. This is useful for simplifying
    road networks by removing dead-end streets and cul-de-sacs.

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        The input graph to be pruned. Can be directed or undirected.
    min_degree : int, optional (default=1)
        Maximum degree threshold for pruning. Nodes with degree <= min_degree
        will be removed. Default is 1 (removes leaf nodes only).

    Returns
    -------
    H : networkx.Graph or networkx.DiGraph
        The pruned graph with leaf nodes removed. Same type as input graph.
    removed_nodes : list
        List of node IDs that were removed during pruning, in the order they
        were removed.

    Notes
    -----
    - The function creates a copy of the input graph, so the original is unchanged
    - Uses an undirected view for degree calculations even if input is directed
    - Maintains a queue and set to avoid processing nodes multiple times
    - Time complexity: O(V + E) where V is vertices and E is edges

    """
    H = graph.copy()
    Hu = H.to_undirected(as_view=True)  # Undirected view for degree calculations

    # Initialize queue with all initial leaf nodes
    leaf_nodes = [n for n, d in Hu.degree() if d <= min_degree]
    queue = deque(leaf_nodes)
    in_queue = set(queue)  # Track nodes in queue to avoid duplicates
    removed_nodes = []  # List of removed nodes

    # Iteratively remove leaf nodes
    while queue:
        u = queue.popleft()
        if Hu.degree(u) <= min_degree:
            nbrs = list(Hu.neighbors(u))
            H.remove_node(u)
            removed_nodes.append(u)
            # Check if neighbors became leaf nodes after removal
            for v in nbrs:
                if v in H and Hu.degree(v) <= min_degree and v not in in_queue:
                    queue.append(v)
                    in_queue.add(v)

    return H, removed_nodes

def podar_grado_2(graph):
    """
    Iteratively prune degree-2 nodes and merge their incident edges.

    This function simplifies a graph by removing nodes that have exactly two
    neighbors (degree-2 nodes). When such a node is removed, its two neighbors
    are directly connected with a new edge that combines the length and geometry
    of the two original edges. This is useful for simplifying road networks by
    removing unnecessary intermediate points along straight road segments.

    The function processes nodes iteratively using a queue. When edges are merged,
    the new edge has:
    - Combined length (sum of the two incident edges)
    - Combined geometry (union of the two LineString geometries)

    Algorithm steps:
    1. Identify all nodes with degree exactly 2
    2. For each degree-2 node v with neighbors u1 and u2:
       a. Find the shortest edge between v and u1
       b. Find the shortest edge between v and u2
       c. Create a new edge between u1 and u2 with combined length and geometry
       d. Remove node v
    3. Continue until no more degree-2 nodes remain

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        The input graph to be pruned. Can be directed or undirected.
        Edges should have 'length' and optionally 'geometry' attributes.

    Returns
    -------
    H : networkx.Graph or networkx.DiGraph
        The simplified graph with degree-2 nodes removed. Same type as input.
    removed_nodes : list
        List of node IDs that were removed during pruning (currently returns
        empty list - implementation incomplete).

    Notes
    -----
    - The function creates a copy of the input graph, so the original is unchanged
    - If an edge lacks a 'geometry' attribute, a LineString is constructed from
      node coordinates
    - Uses undirected view for degree calculations even if input is directed
    - The function selects the shortest edge when multiple edges exist between nodes
    - Time complexity: O(V + E) where V is vertices and E is edges
    """
    H = graph.copy()
    Hu = H.to_undirected(as_view=True)  # Undirected view for degree calculations

    # Initialize queue with all degree-2 nodes
    nodes_deg_2 = [n for n, d in Hu.degree() if d == 2]
    queue = deque(nodes_deg_2)
    in_queue = set(queue)  # Track nodes in queue to avoid duplicates
    removed_nodes = []  # List of removed nodes

    # Iteratively remove degree-2 nodes and merge edges
    while queue:
        v = queue.popleft()

        # Skip if node was already removed
        if v not in H.nodes():
            continue
        # Skip if node no longer has degree 2
        if Hu.degree(v) != 2:
            continue

        if Hu.degree(v) == 2:
            nbrs = list(Hu.neighbors(v))  # Should be exactly u1 and u2
            if len(nbrs) != 2:
                continue
            u1, u2 = nbrs

            # Get all incident edges to v (with attributes)
            # Concatenate in/out edges to handle directed graphs
            incident_edges = list(H.out_edges(v, keys=True, data=True)) + \
                             list(H.in_edges(v, keys=True, data=True))

            # Filter edges connecting to each neighbor
            edges_v_u1 = [e_data for (a, b, k, e_data) in incident_edges if (a == u1 or b == u1)]
            edges_v_u2 = [e_data for (a, b, k, e_data) in incident_edges if (a == u2 or b == u2)]

            # Select shortest edge to each neighbor
            edge1 = min(edges_v_u1, key=lambda e_data: e_data.get("length"))
            edge2 = min(edges_v_u2, key=lambda e_data: e_data.get("length"))

            # Extract length and geometry from edges
            len1 = edge1.get("length")
            len2 = edge2.get("length")
            geom1 = edge1.get("geometry")
            geom2 = edge2.get("geometry")

            # If geometry is missing, construct LineString from coordinates
            if geom1 is None:
                geom1 = LineString([
                    (H.nodes[u1]["x"], H.nodes[u1]["y"]),
                    (H.nodes[v]["x"], H.nodes[v]["y"])
                ])

            if geom2 is None:
                geom2 = LineString([
                    (H.nodes[u2]["x"], H.nodes[u2]["y"]),
                    (H.nodes[v]["x"], H.nodes[v]["y"])
                ])

            # Create new merged edge and remove degree-2 node
            new_length = len1 + len2
            new_geom = geom1.union(geom2)
            H.add_edge(u1, u2, length=new_length, geometry=new_geom)
            H.remove_node(v)

    return H, removed_nodes

def simplificar_aristas_multiples(graph, weight_attr='length'):
    """
    Identify multiple edges between node pairs and keep only the smallest one.

    This function processes MultiDiGraph instances to remove parallel edges
    (multiple edges between the same pair of nodes in the same direction).
    For each ordered pair of nodes (u, v) with multiple edges from u to v,
    only the edge with the minimum weight is kept.

    This is useful for simplifying road networks where multiple road segments
    might exist between the same intersections, and we want to keep only the
    shortest/fastest route.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The input directed multigraph that may contain parallel edges.
        Edges should have a weight attribute (default: 'length').
    weight_attr : str, optional (default='length')
        The edge attribute to use for comparing edge weights.
        The edge with the minimum value of this attribute is kept.

    Returns
    -------
    graph_original : networkx.MultiDiGraph
        A copy of the original graph with all multiple edges preserved.
    graph_simplified : networkx.MultiDiGraph
        The simplified graph with only the minimum-weight edge kept between
        each ordered pair of nodes.
    multiple_edges_info : dict
        Dictionary containing information about removed edges:
        {(u, v): {'count': n, 'removed': m, 'kept_weight': w}}
        where n is total edges from u to v, m is number removed,
        w is the weight of the kept edge.

    Notes
    -----
    - This function only works with MultiDiGraph instances
    - Edges (u,v) and (v,u) are treated separately (direction matters)
    - If an edge lacks the specified weight attribute, it defaults to float('inf')
    - Original graph is not modified; copies are returned


    """
    # Create copies of the input graph
    graph_original = graph.copy()
    graph_simplified = graph.copy()

    # Dictionary to store information about which node pairs had multiple edges
    # {(u, v): {'count': total_edges, 'removed': num_removed, 'kept_weight': min_weight}}
    multiple_edges_info = {}

    # Set to track which (u, v) pairs are already processed
    processed_pairs = set()

    # Iterate through edges in the graph
    # where key is the node's identifier and data is a dict of its attributes

    # using list() because the graph will be modified during iteration
    for u, v, key, data in list(graph_simplified.edges(keys=True, data=True)):
        pair = (u, v)

        # skip if already processed
        if pair in processed_pairs:
            continue

        # get all edges from u to v
        if graph_simplified.has_edge(u, v):
            edges = graph_simplified[u][v] # get info

            # only process if there are multiple edges
            if len(edges) > 1:
                # find edge with minimum weight
                min_key = None           # to store the key of the shortest edge
                min_weight = float('inf')  # start with infinity

                # iterate all parallel edges
                for edge_key, edge_data in edges.items():
                    # get weight (default: inf)
                    weight = edge_data.get(weight_attr, float('inf'))

                    if weight < min_weight: # update
                        min_weight = weight
                        min_key = edge_key

                # Store info in dict
                multiple_edges_info[pair] = {
                    'count': len(edges),           # number of parallel edges
                    'removed': len(edges) - 1,     # number of removed edges
                    'kept_weight': min_weight      # weight of the edge we're keeping
                }

                # remove all edges except the one with min weight
                # we use a list since the graph will be modified during iteration
                for edge_key in list(edges.keys()):
                    if edge_key != min_key: # using the key
                        graph_simplified.remove_edge(u, v, key=edge_key)

        # mark pair as processed
        processed_pairs.add(pair)

    return graph_original, graph_simplified, multiple_edges_info