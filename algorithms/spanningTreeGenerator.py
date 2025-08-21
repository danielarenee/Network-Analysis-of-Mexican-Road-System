#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import random

def generateRandomST(G):

    # empezar con un vertice
    start_node = random.choice(list(G.nodes()))  # elegir un nodo inicial al azar
    visited = set([start_node])  # conjunto de nodos ya visitados
    tree_edges = []  # lista de aristas que forman el árbol
    current_node = start_node
    
    
    while len(visited) < len(G.nodes()): # mientras queden nodos por visitar
        neighbors = list(G.neighbors(current_node)) #vecinos
        next_node = random.choice(neighbors)   # hacer una caminata aleatoria
         # cada que se encuentre un nodo por primera vez...
        if next_node not in visited: 
            tree_edges.append((current_node, next_node))  # agregar la arista del que viene
            visited.add(next_node)
        current_node = next_node  # moverse al siguiente nodo
        # cuando todos los nodos se hayan descubierto... 
    T = nx.Graph()
    T.add_nodes_from(G.nodes())
    T.add_edges_from(tree_edges) #las aristas marcadas son un spanning tree
    return T
