import geopandas as gpd
import osmnx as ox
import networkx as nx
from math import sqrt

def euclidean_heuristic(u, v, graph): # heuristica
    x1, y1 = graph.nodes[u]['x'], graph.nodes[u]['y']
    x2, y2 = graph.nodes[v]['x'], graph.nodes[v]['y']
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)

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