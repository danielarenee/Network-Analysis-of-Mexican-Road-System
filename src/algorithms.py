import igraph as ig
import heapq

from numpy import inf
from time import time as t
from collections import defaultdict
from shapely import LineString


def dijkstra(g, node_map, source, targets=None, weight="length"):
    """
    Compute shortest paths from a single source node to a subset of
    target nodes using a binary-heap Dijkstra over the igraph
    representation.

    Parameters
    ----------
    source
        Source node
    targets : optional (Default: None)
        Destination nodes. If None, distances are computed to every node
        reachable from the source.
    weight : str, optional
        Edge attribute used as cost. Default 'length'.

    Returns
    -------
    distances : dict
        Mapping from target node ID to its shortest distance from
        the source.
    paths : dict
        Mapping from target node ID to the ordered list of node IDs
        along the shortest path from the source.
    """

    # first build the igraph graph in case we havent 


    # g.vs["node_id"] has the original NetworkX node ID for every vertex
    # so we can work with igraph indexes and Nx mode_ids
    node_to_ig = node_map

    # translate source node id to igraph index
    if source not in node_to_ig:
        raise KeyError(f"Source node {source!r} is not in the graph.")
    source_ig = node_to_ig[source]

    if targets is None:
        target_ig_ids = None
    else: # same thing for target nodes
        if isinstance(targets, int):
            targets = [targets]
            
        target_ig_ids = set()
        for target in targets:
            if target not in node_to_ig:
                raise KeyError(f"Target node {target!r} is not in the graph.")
            target_ig_ids.add(node_to_ig[target])

    # Dijkstra
    # dist[u] : dict of best known distance from source_ig to u so far
    # prev[u] : predecessor of u on that best-known path (used to walk the path 
    #           back to the source).
    # visited : set of vertices whose shortest distance is finalized
    # heap : min-priority queue of (distance, vertex) pairs, always
    #           popping the currently-closest unvisited vertex next
    # pending : if we were given specific targets, this is the set of
    #           igraph indices we're still waiting to finalize. it lets us 
    #           break out of the loop when empty

    # initialize
    dist = {source_ig: 0.0} 
    prev = {source_ig: None}
    visited = set()
    pending = set(target_ig_ids) if target_ig_ids is not None else None
    heap = [(0.0, source_ig)] # first push

    # keep going while there is something left to explore
    # and we have targets unresolved
    while heap and (pending is None or pending):
        d, u = heapq.heappop(heap) 
        if u in visited:
            continue
        visited.add(u)
        if pending is not None:
            pending.discard(u)

        # relax every edge (u, v) aka. if going through u gives v a shorter
        # distance than what we currently have, record the improvement
        # and push the new candidate distance to the heap
        
        for v in g.neighbors(u):
            edge_id = g.get_eid(u, v)
            w = g.es[edge_id][weight]               

            new_dist = d + w
            if new_dist < dist.get(v, float("inf")):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    # decide which vertices we want results for
    if target_ig_ids is not None:
        result_ig_ids = target_ig_ids
    else:
        result_ig_ids = set(dist) - {source_ig}

    distances = {}
    paths = {}
    for target_ig in result_ig_ids:

        # a target may be unreachable 
        if target_ig not in dist:
            continue

        # reconstruct the path by walking predecessors backwards from
        # target to the source
        path_ig = []
        node = target_ig
        while node is not None:
            path_ig.append(node)
            node = prev[node]
        path_ig.reverse()

        target_id = g.vs[target_ig]["node_id"]
        distances[target_id] = dist[target_ig]
        paths[target_id] = [g.vs[i]["node_id"] for i in path_ig]

    # output is two dicts, one with distances {100: 154.4}
    # and one with paths {100: [150, 189, 200]}
    return distances, paths

def multi_source_dijkstra(g, node_map, sources, targets=None, weight="length"):
    """
    Compute shortest paths from several source nodes to a subset of
    target nodes. This is just a wrapper around `dijkstra`

    Parameters
    ----------
    sources : iterable
        Source nodes, using the original NetworkX node identifiers
    targets : optional (Default: None)
        Destination nodes, passed through to `dijkstra` for every
        source. If None, distances are computed to every node
        reachable from each source.
    weight : str, optional
        Edge attribute used as cost. Default 'length'.

    Returns
    -------
    distances : dict
        Nested dict: distances[source][target] = shortest distance.
    paths : dict
        Nested dict: paths[source][target] = ordered list of node
        IDs from that source to that target.
    """
    from tqdm import tqdm
    # initialize dicts
    distances = {}
    paths = {}

    for source in sources:
        source_distances, source_paths = dijkstra(
            g = g, node_map = node_map,
            source = source, targets=targets, weight=weight
        )
        distances[source] = source_distances
        paths[source] = source_paths

    # outer key will be source and inner key will be target
    # distances {100:{200:1385.4, 300:346.3}, 250:{...}}

    return distances, paths


def build_voronoi_netwkork_diagram(
        g: ig.Graph,
        id_city_label : str,
        external_city_id = None
        ):
    """
    Compute a graph Voronoi partition using multi-source Dijkstra.

    Vertices assigned to internal regions are used as sources with initial
    distance zero. External vertices are progressively assigned to their
    nearest region according to shortest-path distance.

    Parameters
    ----------
    g : igraph.Graph
        Graph containing the road network. Edges must have a ``length``
        attribute.
    id_city_label : str
        Vertex attribute containing the initial region identifier.
    external_city_id : optional (Default: None)
        Identifier used for vertices outside the initial regions.

    Returns
    -------
    d : list
        Shortest distance from each vertex to its assigned region.
    p : list
        Predecessor of each vertex in the shortest-path forest.
    R : list
        Region assigned to each vertex after the Voronoi partition.
    F : list
        Boolean flags identifying vertices on boundaries between regions.
    contador : int
        Number of successful distance relaxations.
    final_time : float
        Algorithm execution time in seconds.
    """
    
    print("Initializing...")
    # Number of vertices
    n = g.vcount()
    # List of predecessors
    prev = [None] * n
    # List of regions
    R = list(g.vs[id_city_label])
    # List of distances
    dist = [inf if R[v] == external_city_id else 0 for v in range(n)]
    # List of frontier flags
    F = [False if R[v] == 0 else True for v in range(n)]
    
    # Initialize Priority queue
    Q = []
    for u in range(n):
        heapq.heappush(Q, (dist[u], u))
    
    print("Running...")
    start = t()
    contador = 0
    
    # Propagate region labels using Dijkstra's algorithm
    while Q:
        # Extract vertex with minimum tentative distance
        dist_u, u = heapq.heappop(Q)
        
        # Skip outdated queue entries
        if dist_u > dist[u]:
            continue
    
        flag = False
        
        # Explore neighbors of u
        for v in g.neighbors(u):
            # Retrieve the edge connecting u and v
            try:
                e_id = g.get_eid(u, v)
            except:
                e_id = g.get_eid(v, u)
            w_uv = g.es[e_id]["length"]
            
            # Relax edge and propagate the region of u
            if dist[u] + w_uv < dist[v]:
                contador += 1
                
                # If v already had a predecessor, its previous predecessor may
                # become part of the frontier after the reassignment
                if prev[v] is not None:
                    F[prev[v]] = True
                    
                # Update shortest-path information for v
                dist[v] = dist[u] + w_uv
                prev[v] = u
                # Propagate the Voronoi region of u to v
                R[v] = R[u]
                # Mark v temporarily as a frontier candidate
                F[v] = True
                
                # Push updated state into the queue
                heapq.heappush(Q, (dist[v], v))
                
            else:
                # Detect adjacency between different regions
                if R[v] != R[u]:
                  flag = True
                  
        # u belongs to the frontier only if at least one adjacent vertex
        # is assigned to a different region
        F[u] = flag
        
    final_time = t()-start
    
    print("Iterations: ", contador)
    print("Time: ", final_time, " s")
    return dist, prev, R, F, contador, final_time


def build_voronoi_dense_graph(
        g : ig.Graph,
        R : dict,
        F : list,
        weight = "length"
        ):
    from tqdm import tqdm
    
    node_ids = g.vs["node_id"]
    
    nodes_by_voronoi = defaultdict(list)
    frontier_by_voronoi = defaultdict(list)
    
    for v, region in enumerate(R):
        nodes_by_voronoi[region].append(v)
        if F[v]:
            frontier_by_voronoi[region].append(v)
    
    ff_distances_by_voronoi  = {}
    ff_path_by_voronoi  = {}
    
    cliques = {}
    for region, nodes in tqdm(nodes_by_voronoi.items()):
        sub_g = g.induced_subgraph(nodes)
        node_map = {node_id: i for i, node_id in enumerate(sub_g.vs["node_id"])}
        
        frontiers = [
            node_ids[v]
            for v in frontier_by_voronoi[region]
        ]
        boundaries = [
           v["node_id"] for v in sub_g.vs() if v["boundary"]
           ]
        
        frontiers = list(set(frontiers + boundaries))

        ff_distance, ff_paths = multi_source_dijkstra(
                g = sub_g,
                node_map = node_map,
                sources = frontiers,
                targets = frontiers,
                weight = weight,
            )
        ff_distances_by_voronoi[region] = ff_distance
        ff_path_by_voronoi[region] = ff_paths
        
        clique = ig.Graph.Full(len(frontiers))
        missing_edges = []
        for attr in sub_g.vs.attributes():
            for v in clique.vs():
                node_id = node_map[frontiers[v.index]]
                v[attr] = sub_g.vs[node_id][attr]
                
        for e in clique.es():
            u, v = e.source, e.target
            u_node_id = clique.vs[u]["node_id"]
            v_node_id = clique.vs[v]["node_id"]
            
            if v_node_id not in ff_distance[u_node_id]:
                missing_edges.append(e.index)
                continue
    
            e[weight] = ff_distances_by_voronoi[region][u_node_id][v_node_id]
            e["geometry"] = LineString([
                (clique.vs[u]["x"], clique.vs[u]["y"]),
                (clique.vs[v]["x"], clique.vs[v]["y"])
            ])
        
        clique.delete_edges(missing_edges)
        cliques[region] = clique
            


"""def build_voronoi_dense_graph(
        g : ig.Graph,
        R : dict,
        F : list,
        weight = "length"
        ):
    from tqdm import tqdm
    
    node_ids = g.vs["node_id"]
    
    nodes_by_voronoi = defaultdict(list)
    frontier_by_voronoi = defaultdict(list)
    
    for v, region in enumerate(R):
        nodes_by_voronoi[region].append(v)
        if F[v]:
            frontier_by_voronoi[region].append(v)
    
    ff_distances_by_voronoi  = {}
    ff_path_by_voronoi  = {}
    
    fb_distances_by_voronoi  = {}
    fb_path_by_voronoi  = {}
    
    for voronoi, nodes in tqdm(nodes_by_voronoi.items()):
        frontiers = [
            node_ids[v]
            for v in frontier_by_voronoi[voronoi]
        ]
       
        sub_g = g.induced_subgraph(nodes)
        node_map = {node_id: i for i, node_id in enumerate(sub_g.vs["node_id"])}
        
        ff_distance, ff_paths = multi_source_dijkstra(
                g = sub_g,
                node_map = node_map,
                sources = frontiers,
                targets = frontiers,
                weight = weight,
            )
        ff_distances_by_voronoi[voronoi] = ff_distance
        ff_path_by_voronoi[voronoi] = ff_paths
        
        boundaries = [
            v["node_id"] for v in sub_g.vs() if v["boundary"]
            ]
        fb_distances, fb_paths = multi_source_dijkstra(
            g = sub_g,
            node_map = node_map,
            sources = frontiers,
            targets = boundaries,
            weight=weight,
        )
        fb_distances_by_voronoi[voronoi] = fb_distances
        fb_path_by_voronoi[voronoi] = fb_paths
  """      