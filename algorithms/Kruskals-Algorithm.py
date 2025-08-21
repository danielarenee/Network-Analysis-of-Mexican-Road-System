import numpy as np  # Para generar los puntos aleatorios
import matplotlib.pyplot as plt  # para graficar
from scipy.spatial.distance import pdist  # para obtener las aristas
from itertools import combinations  # para conseguir todas las combinaciones de aristas

# 1) Generar los puntos aleatorios
n_points = 10
points = np.random.uniform(0, 100, size=(n_points, 2))  # genera puntos en el plano 2D
print("Puntos generados:\n", points)

# 2) Calcular las aristas con sus pesos usando pdist y combinations
edge_weights = pdist(points)  # calcula las distancias entre cada par de nodos
edges = list(combinations(range(n_points), 2))  # genera todas las combinaciones posibles de nodos
weighted_edges = list(zip(edges, edge_weights))  # asigna pesos a las aristas con list y zip
# Ordenar las aristas por peso antes de usar Kruskals
weighted_edges.sort(key=lambda x: x[1])  # ordena por el peso (segundo valor en la tupla)

# 4) Función para graficar
def plot_graph(edges, points, title="Graph", filename=None, show=True):
    plt.figure(figsize=(8, 8))

    # nodos 
    x, y = points[:, 0], points[:, 1]  # extrae las coordenadas x y y de la matriz de puntos
    plt.scatter(x, y, color="pink", zorder=2, s=100)  # dibuja los nodos rositas
    for i, (x_coord, y_coord) in enumerate(points): #recorre los nodos y los etiqueta
        plt.text(x_coord, y_coord, f"{i}", fontsize=12, color="purple", zorder=3) # escribe el indice sobre el nodo
    
    # aristas
    for edge in edges:  # Dibujamos las aristas seleccionadas
        u, v = edge
        plt.plot([points[u, 0], points[v, 0]],
                 [points[u, 1], points[v, 1]],
                 color="gray", linestyle="--", zorder=1)
    
    plt.title(title) # agrega el titulo de la grafica 
    if filename:
        plt.savefig(filename) #si el filename no es none, guarda la imagen con filename
    if show:
        plt.show()
    plt.close()

# 4) Graficar el grafo completo
plot_graph(edges, points, title="Complete Graph", filename="complete_graph.png")

# 5) Algoritmo de Kruskal - - - - - - - - - - - - - - - - - -

mst_edges = []  # lista para almacenar las aristas del MST
components = {i: {i} for i in range(n_points)}  # diccionario de los componentes conectados

for (u, v), weight in weighted_edges: # recorre las aristas ordenadas por peso
    if components[u] is not components[v]: # si u y v pertenecen a componentes diferentes (asegurar que no hayan ciclos)
        mst_edges.append((u, v))  # se agrega la arista al mst
        new_component = components[u] | components[v] # unir los componenetes ("|" sirve para union de conjuntos)
        for node in new_component: #recorre los nodos en el nuevo componente
            components[node] = new_component # actualiza el diccionario 
    
    # Terminamos cuando tenemos n-1 aristas
    if len(mst_edges) == n_points - 1:
        break

# 6) Graficar el MST
print("\n -- Árbol de Expansión Mínima (MST) -- ")
print(mst_edges)
plot_graph(mst_edges, points, title="Minimum Spanning Tree (Kruskal)")