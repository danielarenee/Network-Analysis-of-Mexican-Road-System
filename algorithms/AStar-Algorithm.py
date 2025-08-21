#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:26:23 2025

@author: danielarenee
"""

import heapq
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt

# Función para reconstruir el camino encontrado - - - - - - - - - - - - - -

def reconstruct_path(cameFrom, current): #came from es un diccionario
    total_path = [current]  # se inicializa con el nodo final (current)
    while current in cameFrom:  # mientras haya un nodo padre
        current = cameFrom[current]  # se mueve al nodo padre
        total_path.insert(0, current)  # lo agregga al inicio de la lista
    return total_path

# Función A* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def a_star(graph, start, goal, h):
    openSet = []  # cola de prioridad con nodos por explorar (min heap)
    heapq.heappush(openSet, (0, start))  # agrega al start node con fscore de 0

    cameFrom = {}  # Para reconstruir el camino

    # gScore[n] almacena el costo más bajo conocido de start -> n
    gScore = {node: float('inf') for node in range(graph.vcount())} #diccionario con default value de inf
    gScore[start] = 0 #el gscore de start es 0 porque empezamos de ahi

    # fScore[n] = gScore[n] + h(n), nuestra mejor estimación del costo total
    fScore = {node: float('inf') for node in range(graph.vcount())}
    #graph.vcount devuelve el numero de nodos en el grafo
    fScore[start] = h(start, goal) #el fscore de start es la heuristica de start a goal

    while openSet: #mientras haya nodos por explorar (openset not empty)
        _, current = heapq.heappop(openSet) # se extrae el nodo en open set con menor fScore
        # y lo guarda en current (ignora el fscore)
        
        if current == goal:  # si llegamos al destino
            return reconstruct_path(cameFrom, current) #reconstruye el camino 

        for neighbor in graph.neighbors(current): #para cada neighbor de current 
            weight = graph.es[graph.get_eid(current, neighbor)]["weight"]#peso del camino entre current y neighbor
            tentative_gScore = gScore[current] + weight

            if tentative_gScore < gScore[neighbor]:  # si encontramos un mejor camino
                cameFrom[neighbor] = current #actualizamos camefrom, gscore y fscore 
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = tentative_gScore + h(neighbor, goal)

                # agregar el nodo a la cola si no está ya
                if neighbor not in [n[1] for n in openSet]: #extrae los nodos sin fscore de openset 
                    heapq.heappush(openSet, (fScore[neighbor], neighbor))

    return None  # Si no hay camino disponible


# Crear el grafo con conexiones con 50% de probabilidad
n_nodes = 10
g = ig.Graph(directed=False)
g.add_vertices(n_nodes)

edges = []
weights = []

for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        if np.random.choice([0, 1], p=[0.5, 0.5]):  # 50% de probabilidad de conectar nodos
            weight = np.random.randint(1, 11)  # Peso entre 1 y 10
            edges.append((i, j))
            weights.append(weight)

g.add_edges(edges)
g.es["weight"] = weights

# Crear coordenadas aleatorias para cada nodo (para heuristica)
node_positions = {i: (np.random.randint(0, 100), np.random.randint(0, 100)) for i in range(n_nodes)}

# Función heurística (distancia euclidiana)
def heuristic(node1, node2):
    x1, y1 = node_positions[node1]
    x2, y2 = node_positions[node2]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Ejecutar A* 
source_node = 0
goal_node = 4#n_nodes-1
path = a_star(g, source_node, goal_node, heuristic) # crear el path 

if path:
    print(f"Camino encontrado con A* de {source_node} a {goal_node}: {path}")
else:
    print("No hay camino disponible entre los nodos seleccionados.")

# 7 Funcion visualización del grafo 
def plot_graph(graph, path):
    fig, ax = plt.subplots(figsize=(10, 10))

    path_edges = set(zip(path, path[1:]))

    edge_colors = []
    for edge in graph.es:
        source, target = edge.source, edge.target
        if (source, target) in path_edges or (target, source) in path_edges:
            edge_colors.append("blue")  # resaltar en azul las conexiones del camino encontrado
        else:
            edge_colors.append("gray")  # mantener en gris las demás conexiones


    # Dibujar el grafo con nodos numerados
    ig.plot(
        graph,
        layout = node_positions.values(),
        target=ax,
        vertex_label=[str(i) for i in range(graph.vcount())],  # Números en los nodos
        vertex_color="pink",
        edge_label=graph.es["weight"],
        edge_color=edge_colors
    )

    plt.title("Grafo con el camino de A*")
    plt.show()

# mostrar el grafo con el camino encontrado
if path:
    plot_graph(g, path)
    
    