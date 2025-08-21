
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

graph_place = ox.graph_from_place( 'Universidad de las Américas Puebla, Cholula, Puebla, México', network_type='walk' )


# Definir coordenadas del punto de origen y destino
orig_lat, orig_lon = 19.051946, -98.285089  # Entrada Gaos
dest_lat, dest_lon = 19.053889639058962, -98.28212519342718  # Escuela de Ingeniería

# Encontrar los nodos más cercanos en el grafo
orig = ox.distance.nearest_nodes(graph_place, orig_lon, orig_lat)
dest = ox.distance.nearest_nodes(graph_place, dest_lon, dest_lat)

# Calcular la ruta más corta entre los nodos encontrados utilizando la distancia como peso
route = ox.shortest_path(graph_place, orig, dest, weight='length')

# Graficar la ruta más corta sobre la red vial
fig, ax = ox.plot_graph_route(graph_place, route, show=False, close=False, route_color="purple",node_color="gray", bgcolor="white")

# Agregar un título descriptivo a la gráfica
ax.set_title("Ruta más corta entre dos nodos en la UDLAP", fontsize=14, color="white")
plt.show()


