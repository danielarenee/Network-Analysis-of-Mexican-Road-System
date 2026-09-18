import igraph as ig
import heapq

from numpy import inf
from time import time as t
from tqdm import tqdm

def voronoi_dijkstra(
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


def dijkstra_heap_ig(
        graph : ig.Graph,
        source : int,
        target : int,
        weight = "length"):
    """
    Dijkstra's shortest path algorithm using a binary min-heap (heapq)
    and igraph representation (undirected simple graph)

    Parameters
    ----------
    graph : igraph.Graph 
        Road network with non-negative edge weight attribute.
    source : int
        Starting node ID.
    target : int
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
   ValueError
        If no path exists between source and target.
    KeyError
        If source or target are not in the graph.
    """
    n = graph.vcount()
    if source < 0 or source >= n:
        raise KeyError(f"Source vertex {source} is not in the graph.")
    if target < 0 or target >= n:
        raise KeyError(f"Target vertex {target} is not in the graph.")

     # Initialize distances and predecessors
    dist = {source: 0.0} # dist to source is 0
    prev = {source: None}

    visited = set()        # initialize S 
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

        for v in graph.neighbors(u): # for each neighbor v of u 
            # get the length u -> v 
            edge_id = graph.get_eid(u, v)
            w = graph.es[edge_id][weight]

            new_dist = d + w # cumulative dist 
            if new_dist < dist.get(v, float("inf")):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v)) # push to queue

    if target not in dist:
        raise ValueError(f"No path between {source} and {target}.")

    # reconstruct path by walking predecessors back from target
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    return dist[target], path


def dijkstra_subset_ig(
        graph: ig.Graph,
        S,
        weight="length"
        ):
        
    S = list(S)

    distances = {}
    paths = {} 
    
    # Compute shortest path for each pair of vertices in S
    for i in tqdm(range(len(S))):
        source = S[i]

        for j in range(i + 1, len(S)):
            target = S[j]

            # Compute shortest distance and path
            distance, path = dijkstra_heap_ig(
                graph,
                source,
                target,
                weight=weight
            )

            distances[(source, target)] = distance
            paths[(source, target)] = path

    return distances, paths