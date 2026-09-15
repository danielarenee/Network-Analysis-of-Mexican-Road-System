import igraph as ig
import networkx as nx
import heapq

from numpy import inf
from time import time as t
from pandas import DataFrame
from geopandas import GeoDataFrame, points_from_xy

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


