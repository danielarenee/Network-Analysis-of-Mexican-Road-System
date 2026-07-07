"""
Graph utility functions for road network analysis.

This module provides functions for analyzing and simplifying road networks represented
as NetworkX graphs. It includes functionality for:
- Computing heuristics for pathfinding algorithms
- Building clique graphs from locality boundary nodes
- Pruning graphs by removing low-degree vertices
- Simplifying multiple edges by keeping the shortest one
- Dijkstra algorithm heap implementation


"""

import heapq
import time

import networkx as nx
from math import sqrt
from shapely.geometry import LineString
from collections import deque
from tqdm import tqdm


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

def build_locality_clique(graph, cvgeo_target, nodos_frontera, nodos_localidad=None):
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
    nodos_localidad : list, optional
        Pre-filtered list of node IDs belonging to the target locality.

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
    if nodos_localidad is None:
        nodos_localidad = [
            node_id for node_id, data in graph.nodes(data=True)
            if data.get("CVEGEO") == cvgeo_target
        ]

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
    # Add boundary nodes with their original attributes in bulk
    grafo_clique.add_nodes_from((nodo, graph.nodes[nodo]) for nodo in frontera)

    # Step 5: Connect each pair of boundary nodes using single-source shortest path (BFS)
    n = len(frontera_lista)
    total_pairs = n * (n - 1) // 2
    
    # Cache node coordinates to speed up distance calculations
    coords = {node_id: (data["x"], data["y"]) for node_id, data in subgrafo.nodes(data=True)}
    
    edges_to_add = []
    with tqdm(total=total_pairs, desc=f"      loc={cvgeo_target}", leave=False, disable=total_pairs <= 50) as pbar:
        for i in range(n):
            u = frontera_lista[i]
            # Compute shortest paths from u to all reachable nodes in subgrafo using BFS
            try:
                paths = nx.single_source_shortest_path(subgrafo, u)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                paths = {}

            for j in range(i + 1, n):
                v = frontera_lista[j]
                if v in paths:
                    camino = paths[v]
                    peso = 0.0
                    curr_node = camino[0]
                    curr_x, curr_y = coords[curr_node]
                    for next_node in camino[1:]:
                        next_x, next_y = coords[next_node]
                        peso += sqrt((curr_x - next_x)**2 + (curr_y - next_y)**2)
                        curr_node = next_node
                        curr_x, curr_y = next_x, next_y
                    edges_to_add.append((u, v, {'weight': peso, 'path': camino}))
            
            pbar.update(n - 1 - i)

    grafo_clique.add_edges_from(edges_to_add)

    return grafo_clique

def prune_degree_1(graph, min_degree=1):
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

def prune_degree_2(graph):
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

            # Direct dictionary lookups to find incident edges between v and u1 / u2
            edges_v_u1 = []
            if u1 in H[v]:
                edges_v_u1.extend(H[v][u1].values())
            if v in H[u1]:
                edges_v_u1.extend(H[u1][v].values())

            edges_v_u2 = []
            if u2 in H[v]:
                edges_v_u2.extend(H[v][u2].values())
            if v in H[u2]:
                edges_v_u2.extend(H[u2][v].values())

            # Select shortest edge to each neighbor
            edge1 = min(edges_v_u1, key=lambda e_data: e_data.get("length", float('inf')))
            edge2 = min(edges_v_u2, key=lambda e_data: e_data.get("length", float('inf')))

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

def simplify_multiple_edges(graph, weight_attr='length'):
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

    # Iterate through adjacency structure directly to avoid expensive edge iteration & lookups
    for u, neighbors in list(graph_simplified.adj.items()):
        for v, edges in list(neighbors.items()):
            if len(edges) > 1:
                # Find the key of the edge with the minimum weight
                min_key = min(
                    edges.keys(),
                    key=lambda k: edges[k].get(weight_attr, float('inf'))
                )
                min_weight = edges[min_key].get(weight_attr, float('inf'))

                # Store info in dict
                multiple_edges_info[(u, v)] = {
                    'count': len(edges),           # number of parallel edges
                    'removed': len(edges) - 1,     # number of removed edges
                    'kept_weight': min_weight      # weight of the edge we're keeping
                }

                # Remove all edges except the one with min weight
                for edge_key in list(edges.keys()):
                    if edge_key != min_key:
                        graph_simplified.remove_edge(u, v, key=edge_key)

    return graph_original, graph_simplified, multiple_edges_info

def simplify_iteratively(graph):
    """
    Iteratively simplify a graph until no more changes occur.

    This function applies three simplification operations in sequence and repeats
    until the graph no longer changes. This is necessary because each operation
    can create opportunities for the others:
    - Simplifying multiple edges can create low-degree nodes
    - Pruning degree-1 nodes can create degree-2 nodes
    - Pruning degree-2 nodes can create multiple edges between previously
      unconnected or singly-connected nodes

    The algorithm stops when an iteration produces no change in the number of
    nodes and edges, indicating a fixed point has been reached.

    Simplification steps (repeated until convergence):
    1. Simplify multiple edges - keep only shortest edge between each node pair
    2. Prune degree-1 nodes - remove leaf nodes (dead ends)
    3. Prune degree-2 nodes - merge nodes with exactly two neighbors

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiDiGraph
        The input graph to be simplified. Should have 'length' attribute on edges.

    Returns
    -------
    simplified_graph : networkx.Graph or networkx.MultiDiGraph
        The fully simplified graph after reaching a fixed point.
    num_iterations : int
        The number of iterations performed before convergence.

    Notes
    -----
    - The function creates a copy of the input graph, so the original is unchanged

    """
    graph = graph.copy()

    iteration = 0
    while True:
        nodes_before = graph.number_of_nodes()
        edges_before = graph.number_of_edges()

        _, graph, _ = simplify_multiple_edges(graph)
        graph, _ = prune_degree_1(graph)
        graph, _ = prune_degree_2(graph)

        iteration += 1

        # Check for convergence
        nodes_after = graph.number_of_nodes()
        edges_after = graph.number_of_edges()

        if nodes_after == nodes_before and edges_after == edges_before:
            break

    return graph, iteration

def calculate_border_nodes_distance_matrix(graph, boundary_nodes_by_locality):
    """
    Compute distance matrix between border vertices of different regions.

    This function calculates shortest path distances between boundary nodes
    that belong to DIFFERENT regions.

    Algorithm Overview:
    -------------------
    1. Collect all boundary nodes across all regions
    2. For each boundary node in region A:
       a. Find all boundary nodes in OTHER regions (exclude same region)
       b. Calculate shortest path distance to each external boundary node
       c. Store in distance matrix
    3. Result: Complete distance matrix for inter-region connectivity

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiDiGraph
        The road network graph with nodes having 'CVEGEO' attribute indicating
        their region, and edges having 'length' or weight attribute.
    boundary_nodes_by_locality : dict
        Dictionary mapping region codes to sets of boundary node IDs.
        Format: {locality_code: {node_id1, node_id2, ...}}
        A boundary node is one that connects to at least one node in a
        different region.

    Returns
    -------
    distance_matrix : dict
        Nested dictionary where distance_matrix[u][v] is the shortest path
        distance from node u to node v, where u and v are in different regions.
        Format: {node_u: {node_v: distance, ...}, ...}

    """

    # 1. Build a mapping from boundary nodes to their regions
    node_to_region = {}
    for region_code, boundary_nodes in boundary_nodes_by_locality.items():
        for node_id in boundary_nodes:
            node_to_region[node_id] = region_code

    # Hence we get { node_id: "region_code"}

    # 2. Collect all boundary nodes across all regions (for iteration)
    # We turn the original dictionary of sets into a list
    all_boundary_nodes = []
    for boundary_nodes in boundary_nodes_by_locality.values():
        all_boundary_nodes.extend(boundary_nodes) # .extend para desempaquetar en la lista

    # 3. Initialize distance matrix (nested dict)
    # Structure: distance_matrix[source][target] = distance
    distance_matrix = {}

    # 4. Compute shortest paths between boundary nodes of diff regions
    for source_node in tqdm(all_boundary_nodes, desc="Computing distance matrix", unit="node"):
        # Get the region of the source node
        source_region = node_to_region[source_node]
        # Initialize dict for this source node
        distance_matrix[source_node] = {}

        # Compute shortest paths from source_node to all other nodes in the graph
        try:
            lengths = nx.single_source_dijkstra_path_length(graph, source_node, weight='length')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            lengths = {}

        # Filter target nodes that are in different regions and reachable
        for r_code, nodes in boundary_nodes_by_locality.items():
            if r_code == source_region:
                continue
            for target_node in nodes:
                dist = lengths.get(target_node)
                if dist is not None:
                    distance_matrix[source_node][target_node] = dist

    return distance_matrix, node_to_region


def dijkstra_heap(graph, source, target, weight="length"):
    """
    Dijkstra's shortest path algorithm using a binary min-heap (heapq)

    Parameters
    ----------
    graph : networkx.Graph 
        Road network with numeric edge weight attribute.
    source : hashable
        Starting node ID.
    target : hashable
        Destination node ID.
    weight : str, optional
        Edge attribute to use as cost. Default 'length'.

    Returns
    -------
    distance : float
        Total cost of the shortest path.
    path : list
        Ordered list of node IDs from source to target.

    Raises
    ------
    nx.NetworkXNoPath
        If no path exists between source and target.
    KeyError
        If source or target are not in the graph.
    """
    is_multi = graph.is_multigraph()

    dist = {source: 0.0} # dist to source is 0
    prev = {source: None}

    visited = set() # initialize S 
    heap = [(0.0, source)] # initialize Q 

    while heap: # while Q =/ empty 
        d, u = heapq.heappop(heap) # extract node u with min dist until now

        # since heapq has no way to perform "decrease key" it simply inserts the same
        # node with a smaller distance
        if u in visited:
            continue # ignores the versions of (u,d) with longer distance d
        visited.add(u) 

        if u == target:
            break

        for v, edge_data in graph[u].items(): # for each neighbor v of u 
            # get the length u -> v 
            if is_multi:
                # edge_data is {key: attr_dict}
                w = min( # looks for "lenght" key 
                    attrs.get(weight, float("inf")) # inf if doesnt find it
                    for attrs in edge_data.values() 
                )
            else: 
                w = edge_data.get(weight, float("inf"))

            new_dist = d + w # cumulative dist 
            if new_dist < dist.get(v, float("inf")):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v)) # push to queue

    if target not in dist:
        raise nx.NetworkXNoPath(f"No path between {source} and {target}.")

    # reconstruct path by walking predecessors back from target
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    return dist[target], path


def load_and_preprocess_graph(
    source,
    shapefile_path=None,
    center_lat=None,
    center_lon=None,
    network_radius=None,
    inegi_graph_path=None,
):
    """
    Load and preprocess road network graph from OSMnx or INEGI source.

    Parameters
    ----------
    source : str
        The data source, either 'osmnx' or 'inegi'.
    shapefile_path : Path or str, optional
        Path to shapefile (required if source is 'osmnx').
    center_lat : float, optional
        Latitude of network center (required if source is 'osmnx').
    center_lon : float, optional
        Longitude of network center (required if source is 'osmnx').
    network_radius : float, optional
        Radius of network in meters (required if source is 'osmnx').
    inegi_graph_path : Path or str, optional
        Path to INEGI pickled graph (required if source is 'inegi').

    Returns
    -------
    graph : networkx.MultiDiGraph
        The loaded and processed road network graph.
    gdf_nodes_labeled : geopandas.GeoDataFrame
        GeoDataFrame of nodes with spatial/CVEGEO labels.
    gdf_localities : geopandas.GeoDataFrame or None
        GeoDataFrame of locality boundaries (None if source is 'inegi').
    cvegeo_map : dict
        Mapping from node ID to CVEGEO region ID.
    crs : str
        Coordinate Reference System.
    plot_margin : float
        Suggested margin for plotting.
    """
    import math
    import pickle
    import geopandas as gpd
    import osmnx as ox
    import pandas as pd
    import networkx as nx

    if source == "osmnx":
        crs = "EPSG:4326"
        plot_margin = 0.002  # degrees

        if shapefile_path is None:
            raise ValueError("shapefile_path is required for source='osmnx'")

        # load locality polygons
        gdf_localities = gpd.read_file(shapefile_path).to_crs(crs)

        # download road network with osmnx
        graph = ox.graph_from_point(
            (center_lat, center_lon),
            dist=network_radius,
            network_type="drive",
        )

        # convert nodes to geodataframe
        gdf_nodes, _ = ox.graph_to_gdfs(graph)
        gdf_nodes = gdf_nodes.reset_index()

        # spatial join to assign each node its locality (CVEGEO)
        gdf_nodes_labeled = gpd.sjoin(
            gdf_nodes, gdf_localities, how="left", predicate="within"
        )

        # write CVEGEO back into the graph as a node attribute
        cvegeo_map = gdf_nodes_labeled.set_index("osmid")["CVEGEO"].to_dict()
        cvegeo_map = {k: (None if pd.isna(v) else v) for k, v in cvegeo_map.items()}
        nx.set_node_attributes(graph, cvegeo_map, name="CVEGEO")

    elif source == "inegi":
        crs = "EPSG:6372"
        plot_margin = 500  # meters
        gdf_localities = None

        if inegi_graph_path is None:
            raise ValueError("inegi_graph_path is required for source='inegi'")

        with open(inegi_graph_path, "rb") as f:
            graph = pickle.load(f)

        # rename id_polygon
        cvegeo_map = {}
        for node_id, data in graph.nodes(data=True):
            val = data.get("id_polygon")
            if val is None or (isinstance(val, float) and math.isnan(val)):
                graph.nodes[node_id]["CVEGEO"] = None
                cvegeo_map[node_id] = None
            else:
                graph.nodes[node_id]["CVEGEO"] = int(val)
                cvegeo_map[node_id] = int(val)

        # build gdf_nodes_labeled
        nodes_data = [
            {"node_id": node_id, "x": data["x"], "y": data["y"],
             "CVEGEO": data.get("CVEGEO")}
            for node_id, data in graph.nodes(data=True)
        ]
        df_nodes = pd.DataFrame(nodes_data)
        df_nodes["geometry"] = gpd.points_from_xy(df_nodes["x"], df_nodes["y"])
        gdf_nodes_labeled = gpd.GeoDataFrame(df_nodes, geometry="geometry", crs=crs)

    else:
        raise ValueError(f"Unknown source: {source!r}. Use 'osmnx' or 'inegi'.")

    return graph, gdf_nodes_labeled, gdf_localities, cvegeo_map, crs, plot_margin


def plot_labeled_network(graph, gdf_nodes_labeled, gdf_localities=None, source="osmnx"):
    """
    Plot the labeled road network.
    """
    import osmnx as ox
    import matplotlib.pyplot as plt

    fig, ax = ox.plot_graph(graph, show=False, close=False)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    if source == "osmnx":
        if gdf_localities is not None:
            gdf_localities.boundary.plot(ax=ax, color="red")
        gdf_nodes_labeled.plot(ax=ax, column="CVEGEO", cmap="Set2")
        ax.set_title("Ernesto Aguirre, Tabasco - Labeled Road Network",
                     fontsize=16, color="white")
    elif source == "inegi":
        gdf_nodes_labeled.plot(ax=ax, column="CVEGEO", cmap="tab20", markersize=1)
        ax.set_title("INEGI - Initial Road Network", fontsize=16, color="white")
    plt.show()


def identify_boundary_nodes(graph, cvegeo_map):
    """
    Identify nodes that lie on the boundary between different localities.
    """
    from collections import defaultdict

    boundary_nodes_by_locality = defaultdict(set)

    for node in graph.nodes:
        node_loc = cvegeo_map.get(node)
        if node_loc is None:
            continue

        for neighbor in graph.neighbors(node):
            neighbor_loc = cvegeo_map.get(neighbor)
            if neighbor_loc is None or neighbor_loc != node_loc:
                boundary_nodes_by_locality[node_loc].add(node)
                break

    return dict(boundary_nodes_by_locality)


def build_reduced_clique_graph(graph, boundary_nodes_by_locality):
    """
    Build a reduced graph where each locality is represented by a clique of its boundary nodes.
    """
    import networkx as nx
    from collections import defaultdict
    from tqdm import tqdm

    locality_cliques = []

    # Pre-group nodes by locality to optimize clique construction
    nodes_by_locality = defaultdict(list)
    for node_id, data in graph.nodes(data=True):
        loc = data.get("CVEGEO")
        if loc is not None:
            nodes_by_locality[loc].append(node_id)

    pbar = tqdm(boundary_nodes_by_locality.items(), desc="Building locality cliques")
    for loc, frontier_nodes in pbar:
        n_frontier = len(frontier_nodes)
        pbar.set_postfix(loc=loc, boundary_nodes=n_frontier)
        loc_nodes = nodes_by_locality.get(loc, [])
        locality_cliques.append(
            build_locality_clique(
                graph, loc, boundary_nodes_by_locality, nodos_localidad=loc_nodes
            )
        )
    reduced_graph = nx.compose_all(locality_cliques)
    return reduced_graph


def plot_boundary_nodes_network(reduced_graph, gdf_localities=None, plot_margin=0.002, title="Boundary Node Network by Locality"):
    """
    Plot the boundary node network.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    if gdf_localities is not None:
        gdf_localities.boundary.plot(ax=ax, color="gray", linewidth=0.5)

    node_positions = {
        nid: (data["x"], data["y"])
        for nid, data in reduced_graph.nodes(data=True)
    }

    nx.draw_networkx_nodes(
        reduced_graph, pos=node_positions, ax=ax,
        node_size=24, node_color="white",
    )
    nx.draw_networkx_edges(
        reduced_graph, pos=node_positions, ax=ax,
        width=1, edge_color="orange",
    )

    ax.set_title(title, fontsize=20, color="white")
    ax.axis("off")

    coords = np.array(list(node_positions.values()))
    ax.set_xlim(coords[:, 0].min() - plot_margin, coords[:, 0].max() + plot_margin)
    ax.set_ylim(coords[:, 1].min() - plot_margin, coords[:, 1].max() + plot_margin)

    plt.show()


def plot_delaunay_triangulation(gdf_nodes_labeled, crs, title="Delaunay Triangulation of Locality Centroids"):
    """
    Compute and plot Delaunay triangulation of locality centroids.
    """
    import matplotlib.pyplot as plt
    from scipy.spatial import Delaunay

    centroids_df = (
        gdf_nodes_labeled
        .dropna(subset=["CVEGEO"])
        .groupby("CVEGEO")[["x", "y"]]
        .mean()
    )

    locality_labels = centroids_df.index.tolist()
    centroid_points = centroids_df.values  # numpy array

    delaunay_tri = Delaunay(centroid_points)

    plt.figure(figsize=(6, 6))
    plt.triplot(
        centroid_points[:, 0], centroid_points[:, 1],
        delaunay_tri.simplices, linewidth=0.8,
    )
    plt.scatter(centroid_points[:, 0], centroid_points[:, 1], color="red", s=30)
    for i, label in enumerate(locality_labels):
        plt.text(
            centroid_points[i, 0], centroid_points[i, 1],
            label, fontsize=8, ha="center", va="center",
        )
    plt.title(title)
    plt.xlabel("Longitude" if crs == "EPSG:4326" else "X (m)")
    plt.ylabel("Latitude" if crs == "EPSG:4326" else "Y (m)")
    plt.gca().set_aspect("equal", "box")
    plt.show()


def plot_simplified_graph(simplified_graph, gdf_nodes_labeled, gdf_localities=None, num_iterations=0):
    """
    Plot the simplified graph.
    """
    import osmnx as ox
    import matplotlib.pyplot as plt

    fig, ax = ox.plot_graph(simplified_graph, show=False, close=False)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    if gdf_localities is not None:
        gdf_localities.boundary.plot(ax=ax, color="red")
    gdf_nodes_labeled.plot(ax=ax, column="CVEGEO", cmap="Set2")
    ax.set_title(
        f"Fully Simplified Graph (Converged in {num_iterations} iterations)",
        fontsize=16, color="white",
    )
    plt.show()


def print_distance_matrix_samples(distance_matrix, node_to_region):
    """
    Print samples from the inter-region distance matrix.
    """
    from itertools import islice

    total_connections = sum(len(targets) for targets in distance_matrix.values())
    total_boundary_nodes = len(node_to_region)

    print(f"\n{'=' * 60}")
    print("INTER-REGION DISTANCE")
    print(f"{'=' * 60}")
    print(f"Total inter-region connections: {total_connections}")
    print(f"Total boundary nodes: {total_boundary_nodes}")

    print("\nSample inter-region distances (first boundary node):")

    first_node = next(iter(distance_matrix))
    first_targets = distance_matrix[first_node]
    source_region = node_to_region[first_node]

    print(f"\n  First 5 targets from node {first_node} (region {source_region}):")

    for target_node, distance in islice(first_targets.items(), 5):
        target_region = node_to_region[target_node]
        print(f"    -> node {target_node} (region {target_region}): {distance:.2f} m")