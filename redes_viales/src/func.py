import networkx as nx
from math import sqrt
from shapely.geometry import LineString
from collections import deque


"""
heuristica :D
"""

def euclidean_heuristic(u, v, graph): # heuristica
    x1, y1 = graph.nodes[u]['x'], graph.nodes[u]['y']
    x2, y2 = graph.nodes[v]['x'], graph.nodes[v]['y']
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)

"""
Construir clique por localidad
"""

def construir_clique_localidad(graph, cvgeo_target, nodos_frontera): #grafo, localidad, dict frontera
    # 1. filtrar nodos de la localidad
    nodos_localidad = []  # lista para filtrar los nodos de esa localidad

    for node_id, data in graph.nodes(data=True):  # recorremos todos los nodos y sus atributos
        cvgeo = data.get("CVEGEO")  # extraemos el CVEGEO
        if cvgeo == cvgeo_target:  # comparamos con el CVEGEO tarfet
            nodos_localidad.append(node_id)  # si matchea agregamos el nodo a la lista

    # 2. extraer el subgrafo inducido
    subgrafo = graph.subgraph(nodos_localidad).copy()

    # 3. extraer nodos frontera de esa localidad
    # busca los nodos frontera de cada localidad, si no tiene, devuelve un set vacio
    frontera = nodos_frontera.get(cvgeo_target, set())
    frontera_lista = list(frontera)
    #asegurar que los nodos esten en el subgrafo
    frontera_lista = [n for n in frontera_lista if n in subgrafo]

    # 3. crear nuevo grafo clique
    grafo_clique = nx.Graph()
    for nodo in frontera: # agregar los nodos frontera
        grafo_clique.add_node(nodo, **graph.nodes[nodo])  # conservar atributos originales

    # 4. conectar cada par de nodos frontera con A* (si hay camino)
    for i in range(len(frontera_lista)):
        for j in range(i + 1, len(frontera_lista)):
            u = frontera_lista[i] # nodos por conectar
            v = frontera_lista[j]
            try: # A* (lambda para usar euclidean heuristic con 2 args)
                camino = nx.astar_path(subgrafo, u, v, heuristic = lambda a, b: euclidean_heuristic(a, b, subgrafo))
                peso = sum( # calcular pesos del recorrido completo
                    euclidean_heuristic(camino[k], camino[k + 1], subgrafo)
                    for k in range(len(camino) - 1)
                )
                grafo_clique.add_edge(u, v, weight=peso, path=camino)
            except nx.NetworkXNoPath:
                continue  # no conectar si no hay camino

    return grafo_clique

"""
Podar iterativamente los nodos hoja
"""

def podar_grado_1(graph, min_degree=1):

    H = graph.copy()
    Hu = H.to_undirected(as_view=True) # vista de H

    leaf_nodes = [n for n, d in Hu.degree() if d <= min_degree]
    queue = deque(leaf_nodes) # nodos por eliminar
    in_queue = set(queue) # para evitar duplicados
    removed_nodes = [] # output

    # loop para podar
    while queue:
        u = queue.popleft()
        if Hu.degree(u) <= min_degree:
            nbrs = list(Hu.neighbors(u))
            H.remove_node(u)
            removed_nodes.append(u)
            for v in nbrs: #
                if v in H and Hu.degree(v) <= min_degree and v not in in_queue:
                    queue.append(v) # para que entre en el while
                    in_queue.add(v) # para no repetir

    return H, removed_nodes

"""
Podar iterativamente los nodos de grado 2 y unir las aristas 
"""
def podar_grado_2(graph):

    H = graph.copy()
    Hu = H.to_undirected(as_view=True) # vista de H

    # partir de la lista de los que tienen grado 2
    nodes_deg_2 = [n for n, d in Hu.degree() if d == 2]
    queue = deque(nodes_deg_2)  # nodos por eliminar
    in_queue = set(queue) # para evitar duplicados
    removed_nodes = [] #output

    # loop para podar
    while queue:
        v = queue.popleft()
        # sacar a sus vecinos
        if v not in H.nodes():
            continue
        if Hu.degree(v) != 2:
            continue
        if Hu.degree(v) == 2:
            nbrs = list(Hu.neighbors(v)) # = u1 y u2
            if len(nbrs) != 2:
                continue
            u1, u2 = nbrs

            # aristas incidentes a v (con atributos)
            #incident_edges = list(H.edges(v, keys=True, data=True))
            incident_edges = list(H.out_edges(v, keys=True, data=True)) + \
                             list(H.in_edges(v, keys=True, data=True))

            # filtrar las aristas que conectan con u1 y con u2
            edges_v_u1 = [e_data for (a, b, k, e_data) in incident_edges if (a == u1 or b == u1)]
            edges_v_u2 = [e_data for (a, b, k, e_data) in incident_edges if (a == u2 or b == u2)]

            # elegir arista de menor length para cada vecino
            edge1 = min(edges_v_u1, key=lambda e_data: e_data.get("length"))
            edge2 = min(edges_v_u2, key=lambda e_data: e_data.get("length"))

            # conseguir length y geometry
            len1 = edge1.get("length")
            len2 = edge2.get("length")
            geom1 = edge1.get("geometry")
            geom2 = edge2.get("geometry")

            # si falta geometry, construir LineString mínima
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

            # agregar arista con la nueva longitud y geometria
            new_length = len1 + len2
            new_geom = geom1.union(geom2)
            H.add_edge(u1, u2, length=new_length, geometry=new_geom)
            H.remove_node(v)

    return H, removed_nodes