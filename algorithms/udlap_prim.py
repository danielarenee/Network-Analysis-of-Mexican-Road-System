import osmnx as ox
import matplotlib.pyplot as plt

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

# --- Algoritmo de Prim ---
visited = set()
unvisited = set(nodes)

# Nodo inicial arbitrario
start_node = nodes[0]
visited.add(start_node) # se marca como visitado
unvisited.remove(start_node)

mst_edges = []

while unvisited:  # mientras haya nodos no visitados
    min_edge = None
    min_weight = float('inf')

    for (u, v), weight in edges:   # recorre todas las aristas y sus pesos
    # verifica si esta arista conecta un nodo visitado con uno no visitado
        if (u in visited and v in unvisited) or (v in visited and u in unvisited):
            # si el peso de esta arista es menor que el mínimo encontrado hasta ahora
            if weight < min_weight:
                min_edge = (u, v) # actualiza la mejor arista 
                min_weight = weight # actualiza el peso minimo

    if min_edge:  # añadir la arista mínima al MST
        mst_edges.append(min_edge)
        u, v = min_edge
        visited.add(v) if u in visited else visited.add(u)
        unvisited.remove(v) if v in unvisited else unvisited.remove(u)

# funcion para graficar MST
def plot_mst(graph, edges, node_positions):
    fig, ax = plt.subplots(figsize=(10, 10))
    ox.plot_graph(graph, ax=ax, node_size=5, edge_color="lightgray", show=False, close=False)
    # dibujar solo las aristas del MST
    for u, v in edges:
        x_values = [node_positions[u][0], node_positions[v][0]]
        y_values = [node_positions[u][1], node_positions[v][1]]
        ax.plot(x_values, y_values, color="purple", linewidth=2)
    ax.set_title("MST usando Prim sobre OSMnx")
    plt.show()

# graficar MST
plot_mst(graph, mst_edges, node_positions)
