import numpy as np
import matplotlib.pyplot as plt

n_nodes = 9

# 1) Crear la matriz de adyacencia - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
adj_matrix = np.zeros((n_nodes, n_nodes), dtype=int) # se inicializa con ceros la matriz 
for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        # con probabilidad del 50% se crea una conexión entre nodos
        edge = np.random.choice([0, 1], p=[0.5, 0.5])
        if edge == 1: #si estan conectados.. 
            weight = np.random.randint(0, 11)  # asigna un peso aleatorio entre 0 y 10
            adj_matrix[i][j] = weight  # se asigna a ij y ji 
            adj_matrix[j][i] = weight
        else:
            adj_matrix[i][j] = 0
            adj_matrix[j][i] = 0

print("Matriz de adyacencia:\n", adj_matrix) #imprimir la matriz de adjacencia 

# 2) Crear el diccionario de vecinos a partir de la matriz - - - - - - - - - - - - - - - - - -
# ... donde cada nodo tiene una lista de sus vecinos
neighbors = {}
for i in range(n_nodes):  # para cada nodo i (key) se genera una lista de nodos j si hay una conexion 
    neighbors[i] = []      # AKA: la entrada en la matriz no es 0
    for j in range(n_nodes):
        if adj_matrix[i][j] != 0: 
            neighbors[i].append(j)

print("\nDiccionario de vecinos:") #imprime el diccionario de vecinos
for node, nbrs in neighbors.items():
    print(f"{node}: {nbrs}")

# 3) Inicializar las distancias y el diccionario de predecesores  - - - - - - - - - - - - - - -
distances = [float('inf')] * n_nodes # se crea una lista de tamaño nodos y se inicializan los elementos en infinito
distances[2] = 0  # se asigna 0 al nodo fuente

# diccionario de predecesores
predecessors = {}        # se crea un diccionario vacío para almacenar el predecesor de cada nodo
for i in range(n_nodes): # se recorre cada nodo (de 0 a n_nodes - 1)
    predecessors[i] = None  # se asigna None como predecesor inicial para el nodo i

# Conjunto de nodos sin visitar
unvisited = set(range(n_nodes)) #inicialmente todos

# 4) Algoritmo de Dijkstra - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
while unvisited: # mientras hayan nodos sin visitar 
    # seleccionar el nodo con la menor distancia
    current = min(unvisited, key=lambda node: distances[node]) # en cada iteracion se selecciona el nodo current que tenga la menor distancia 
     # en la lista distances entre los nodos en unvisited
    unvisited.remove(current) # se elimina ese nodo del conjunto unvisited
    
    # actualizar distancias para cada vecino
    for neighbor in neighbors[current]:
        if neighbor in unvisited: #para cada vecino que siga en unvisited
            if distances[current] + adj_matrix[current][neighbor] < distances[neighbor]: 
                # si la suma de la distancia minima encontrada hacia current 
                # + el peso de la arista entre current y neighbor 
                # es menor que la distancia minima encontrada hacia neighbor
                distances[neighbor] = distances[current] + adj_matrix[current][neighbor] # se actualiza la distancia de neighbor con la suma 
                predecessors[neighbor] = current # se actualiza el diccionario predecessors 
                #asignando a neighbor el nodo current. AKA indica que para llegar a neighbor de la mejor forma se pasó por current

# 5) Imprimir las distancias más cortas desde el nodo 0 - - - - - - - - - - - - - - - - - - - - 
print("\nDistancias más cortas desde el nodo 0:")
for i, d in enumerate(distances): # Se recorre la lista distances con enumerate para obtener el índice del nodo Y su distancia.
    print(f"Nodo {i}: {d}")

# 6) Función reconstruct_path  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# reconstruir el camino desde el nodo 0 a un nodo objetivo 
def reconstruct_path(predecessors, target): # usando predecessors, reconstruye el camino desde el nodo 0 hasta un nodo target
    path = []
    while target is not None:
        path.insert(0, target) #se inserta el nodo target al inicio de la lista path
        target = predecessors[target] # se actualiza target al nodo predecesor del actual
    return path # Una vez que target es None (llegado al inicio), se retorna la lista path completa,
    # aka el camino desde el nodo 0 hasta el nodo target original

# 7) Imprimir los caminos desde el nodo 0 hacia cada nodo - - - - - - - - - - - - - - - - - - - - 
print("\nCaminos desde el nodo 0:")
for i in range(n_nodes):
    path = reconstruct_path(predecessors, i) # se recorre cada nodo y se llama a la función para obtener el camino hasta el nodo i 
    print(f"Nodo {i}: {path}")

# 8) Función plot graph - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - -
def plot_graph(adj_matrix, points, title="Grafo"):
    plt.figure(figsize=(8,8))
    x, y = points[:, 0], points[:, 1] #se extraen las coordenadas x y y de cada punto del arreglo points
    plt.scatter(x, y, color="pink", zorder=2, s=100)
    for i, (xi, yi) in enumerate(points): #se recorre cada punto  con su índice y se coloca el número del nodo
        plt.text(xi, yi, f"{i}", fontsize=12, color="purple", zorder=3)
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adj_matrix[i][j] != 0:
                plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], #dibuja las aristas
                         color="gray", linestyle="--", zorder=1)
                # agregar etiqueta con el peso de la arista
                xm = (points[i,0] + points[j,0]) / 2 # calcula la coordenada x del punto medio entre los nodos i y j
                ym = (points[i,1] + points[j,1]) / 2 # calcula la coordenada y 
                plt.text(xm, ym, str(adj_matrix[i][j]), color="gray", fontsize=10, zorder=5)
    
    
    plt.title(title)
    plt.show()
    plt.close()

# 9) Generar puntos aleatorios para la visualización del grafo y graficarlo  - - - - - -- - - - - - -
points = np.random.uniform(0, 100, size=(n_nodes, 2))
plot_graph(adj_matrix, points)
