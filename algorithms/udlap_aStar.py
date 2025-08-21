import osmnx as ox
import networkx as nx
import heapq
import numpy as np
import matplotlib.pyplot as plt

# Obtener el grafo de la UDLAP
graph = ox.graph_from_place( 'Universidad de las Américas Puebla, Cholula, Puebla, México', network_type='walk' )

# Definir coordenadas del punto de origen y destino
orig_lat, orig_lon = 19.051946, -98.285089  # Entrada Gaos
dest_lat, dest_lon = 19.053889639058962, -98.28212519342718  # Escuela de Ingeniería
# Encontrar los nodos más cercanos en el grafo
orig_node = ox.distance.nearest_nodes(graph, orig_lon, orig_lat)
dest_node = ox.distance.nearest_nodes(graph, dest_lon, dest_lat)

# --- Definir la heurística: (Distancia Euclidiana) ---
def heuristic(n1, n2):
    #coord de cada nodo
    lat1, lon1 = graph.nodes[n1]['y'], graph.nodes[n1]['x']
    lat2, lon2 = graph.nodes[n2]['y'], graph.nodes[n2]['x']
    return np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) # dist euclidiana

# --- Implementación de A* usando la distancia como peso ---
def a_star_osmnx(graph, start, goal, h):
   
    open_set = [] #nodos por explorar
    heapq.heappush(open_set, (0, start))  # (f-score, nodo)

    came_from = {}  # Para reconstruir el camino

    # gScore almacena el costo más bajo conocido de start -> nodo
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start] = 0 # costo del inicoi al inicio es 0

    # fScore = gScore + heurística
    f_score = {node: float('inf') for node in graph.nodes}
    f_score[start] = h(start, goal) #estimacion del costo start->goal

    while open_set:
        _, current = heapq.heappop(open_set)   # se extrae el nodo en open set con menor fScore

        if current == goal:  # si llegamos al destino
            return reconstruct_path(came_from, current) #reconstruye el camino 

        for neighbor in graph.neighbors(current): #para cada neighbor de current 
            weight = graph[current][neighbor][0].get("length", 1) #sdistancia entre current y neighbor
            tentative_g_score = g_score[current] + weight

            if tentative_g_score < g_score[neighbor]:  # si encontramos un mejor camino
                came_from[neighbor] = current  #actualizamos camefrom, gscore y fscore 
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + h(neighbor, goal)

                 # agregar el nodo a la cola si no está ya
                if neighbor not in [n[1] for n in open_set]: #extrae los nodos sin fscore de openset 
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No hay camino

# --- Función para reconstruir el camino encontrado ---
def reconstruct_path(came_from, current):
  
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.insert(0, current)
    return total_path

# --- Ejecutar A* en el grafo ---
route_astar = a_star_osmnx(graph, orig_node, dest_node, heuristic)

if route_astar:
    print("Ruta encontrada con A*:", route_astar)
else:
    print("No se encontró una ruta.")

# --- Visualizar la ruta ---
fig, ax = ox.plot_graph_route(
    graph, route_astar, route_color="purple", node_size=10, node_color="gray", bgcolor="white")

