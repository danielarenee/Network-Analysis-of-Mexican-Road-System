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
        total_path.insert(0, current)  # lo agrega al inicio de la lista
    return total_path

# Función A* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

   
def a_star_debug(graph, start, goal, h):
    openSet = []
    heapq.heappush(openSet, (0, start))
    cameFrom = {}

    gScore = {node: float('inf') for node in range(graph.vcount())}
    gScore[start] = 0

    fScore = {node: float('inf') for node in range(graph.vcount())}
    fScore[start] = h(start, goal)

    while openSet:
        _, current = heapq.heappop(openSet)

        print(f"\nExplorando nodo: {current}")
        print(f"gScore actual: {gScore}")
        print(f"fScore actual: {fScore}")
        print(f"openSet: {[n[1] for n in openSet]}")

        if current == goal:
            return reconstruct_path(cameFrom, current)

        for neighbor in graph.neighbors(current):
            weight = graph.es[graph.get_eid(current, neighbor)]["weight"]
            tentative_gScore = gScore[current] + weight  

            if tentative_gScore < gScore[neighbor]:  
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = tentative_gScore + h(neighbor, goal)

                if neighbor not in [n[1] for n in openSet]:  
                    heapq.heappush(openSet, (fScore[neighbor], neighbor))

    return None



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

# Crear coordenadas aleatorias para cada nodo (para heurística)
node_positions = {i: (np.random.randint(0, 100), np.random.randint(0, 100)) for i in range(n_nodes)}

# Función heurística (distancia euclidiana)
def heuristic(node1, node2):
    x1, y1 = node_positions[node1]
    x2, y2 = node_positions[node2]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Ejecutar A* 
source_node = 0
goal_node = 4 # n_nodes-1
path = a_star_debug(g, source_node, goal_node, heuristic) # crear el path 

if path:
    print(f"Camino encontrado con A* de {source_node} a {goal_node}: {path}")

    # mostrar el costo total del camino encontrado
    total_cost = sum(
        g.es[g.get_eid(path[i], path[i + 1])]["weight"]
        for i in range(len(path) - 1)
    )
    print(f"Costo total del camino: {total_cost}")

else:
    print("No hay camino disponible entre los nodos seleccionados.")

# 7 Funcion visualización del grafo 
def plot_graph(graph, path):
    layout = graph.layout("fr")  
    fig, ax = plt.subplots(figsize=(10, 10))

    path_edges = set(zip(path, path[1:]))

    edge_colors = []
    for edge in graph.es:
        source, target = edge.source, edge.target
        if (source, target) in path_edges or (target, source) in path_edges:
            edge_colors.append("red")  # CAMBIO: Ahora resalta en rojo el camino mínimo
        else:
            edge_colors.append("gray")  # Mantener en gris las demás conexiones

    # Dibujar el grafo con nodos numerados
    ig.plot(
        graph,
        layout=layout,
        target=ax,
        vertex_label=[str(i) for i in range(graph.vcount())],  # Números en los nodos
        vertex_color="pink",
        edge_label=graph.es["weight"],  # Mostramos los pesos de las aristas
        edge_color=edge_colors  # Aplicar los colores corregidos
    )

    plt.title("Grafo con el camino de A* minimizando costo")
    plt.show()

# mostrar el grafo con el camino encontrado
if path:
    plot_graph(g, path)
    
  