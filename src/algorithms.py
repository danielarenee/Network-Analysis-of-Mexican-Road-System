import igraph as ig
import heapq

from numpy import inf
from time import time as t


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
    p = [None] * n
    # List of regions
    R = list(g.vs[id_city_label])
    # List of distances
    d = [inf if R[v] == external_city_id else 0 for v in range(n)]
    # List of frontier flags
    F = [False if R[v] == 0 else True for v in range(n)]
    
    # Initialize Priority queue
    Q = []
    for u in range(n):
        heapq.heappush(Q, (d[u], u))
    
    print("Running...")
    start = t()
    contador = 0
    
    # Propagate region labels using Dijkstra's algorithm
    while Q:
        # Extract vertex with minimum tentative distance
        dist_u, u = heapq.heappop(Q)
        
        # Skip outdated queue entries
        if dist_u > d[u]:
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
            if d[u] + w_uv < d[v]:
                contador += 1
                
                # If v already had a predecessor, its previous predecessor may
                # become part of the frontier after the reassignment
                if p[v] is not None:
                    F[p[v]] = True
                    
                # Update shortest-path information for v
                d[v] = d[u] + w_uv
                p[v] = u
                # Propagate the Voronoi region of u to v
                R[v] = R[u]
                # Mark v temporarily as a frontier candidate
                F[v] = True
                
                # Push updated state into the queue
                heapq.heappush(Q, (d[v], v))
                
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
    return d, p, R, F, contador, final_time


