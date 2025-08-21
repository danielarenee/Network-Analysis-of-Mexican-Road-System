import osmnx as ox
import matplotlib.pyplot as plt
from itertools import combinations

# Obtener el grafo de la UDLAP
graph = ox.graph_from_place("Universidad de las Américas Puebla, Cholula, Puebla, México", network_type="walk")

# Obtener lista de nodos y sus coordenadas
nodes = list(graph.nodes) # lista de todos los nodos del grafo
node_positions = {node: (graph.nodes[node]['x'], graph.nodes[node]['y']) for node in nodes}
#diccionario nodos con sus coordenadas 

# Construir la lista de aristas con pesos (distancia real)
edges = [] #lista de aristas
for u, v, data in graph.edges(data=True): # recorre cada arista
    weight = data.get('length', 1) # obtiene la distancia
    edges.append(((u, v), weight)) # y la guarda como ((u,v),peso)

# Ordenar las aristas por peso
edges.sort(key=lambda x: x[1])  # ordenar las listas segun x[1] que es el peso
#lambda : criterio de orden 

# --- Algoritmo de Kruskal ---
mst_edges = []   # lista para almacenar las aristas del MST
components = {node: {node} for node in nodes}  # diccionario de los componentes conectados
# cada nodo es inicialmente su propio componente

for (u, v), weight in edges: # recorre las aristas ordenadas por peso
    if components[u] is not components[v]:  # si están en diferentes componentes, lse unen
        mst_edges.append((u, v))
        new_component = components[u] | components[v]  # unión de conjuntos
        for node in new_component: #recorre los nodos en el nuevo componente
            components[node] = new_component  # actualizar componentes

    if len(mst_edges) == len(nodes) - 1:  # terminar al tener n-1 aristas
        break


# funcion para graficar MST
def plot_mst(graph, edges, node_positions):
    fig, ax = plt.subplots(figsize=(10, 10))
    ox.plot_graph(graph, ax=ax, node_size=10, edge_color="gray", show=False, close=False)

    # dibujar solo las aristas del MST
    for u, v in edges:
        x_values = [node_positions[u][0], node_positions[v][0]] #coordenadas x de la arista
        y_values = [node_positions[u][1], node_positions[v][1]] #coordenas y de la arista
        ax.plot(x_values, y_values, color="purple", linewidth=2) #dibujar la linea

    ax.set_title("MST", fontsize=36)

    plt.show()

# Graficar MST
plot_mst(graph, mst_edges, node_positions)
