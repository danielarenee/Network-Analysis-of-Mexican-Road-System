import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point
from math import sqrt

# Ruta al archivo shapefile
ruta = "/Users/danielarenee/Desktop/honores/redes_viales/Data/shp/27l.shp"

# Leer el archivo
gdf_poly = gpd.read_file(ruta)

# 1) cambiar el crs de los poligonos a 4326
gdf_poly = gdf_poly.to_crs(4326)

# crear el grafo
lat =17.930714
lon = -93.507545

"""
lat= 18.067192
lon = -93.498951
"""

center_point = (lat, lon)
graph = ox.graph_from_point(
    center_point,
    dist=7000,
    network_type='drive'
)

# 2) crear un df de los nodos
nodes_data = [] #set para los nodos
for node_id, data in graph.nodes(data=True):
    nodes_data.append({
        "node_id": node_id, #id, x y y
        "x": data["x"],
        "y": data["y"]
    })

df_nodes = pd.DataFrame(nodes_data) #crear el df

# 3) crear un geodataframe

nodes_geom =[]
for _, row in df_nodes.iterrows():
    punto = Point(row["x"],row["y"]) #punto para la geometria
    nodes_geom.append(punto)
df_nodes["geometry"] = nodes_geom
gdf_nodes = gpd.GeoDataFrame(df_nodes, geometry="geometry", crs="EPSG:4326")

# 4) union espacial
#busca que puntos estan dentro de que poligono
gdf_union = gpd.sjoin(gdf_nodes, gdf_poly, how="left", predicate="within")

# 5) etiquetar los nodos
for _, row in gdf_union.iterrows(): #asignar el cvgeo de cada nodo como atributo
    node_id = row["node_id"]
    label = row["CVEGEO"]
    graph.nodes[node_id]["CVEGEO"] = label #cada nodo en el grafo tiene un atributo nuevo "CVEGEO"

fig, ax = ox.plot_graph(graph, show=False, close=False)
fig.patch.set_facecolor('black')  # fondo fuera del gráfico
ax.set_facecolor('black')
gdf_poly.boundary.plot(ax=ax, color="red")
gdf_union.plot(ax=ax, column="CVEGEO", cmap="Set2")
ax.set_title("Ernersto aguirre, Tabasco", fontsize=16, color="white")
plt.show()

#------- UPDATE --------
# 1) identificar nodos frontera en un diccionario
# (tienen al menos un vecino con un cvgeo diferente)
fronteras_por_localidad = dict()

for node in graph.nodes:
    cvgeo_node = graph.nodes[node]["CVEGEO"]
    if cvgeo_node is None:
        continue # saltar los que no estan en localidades

    for vecino in graph.neighbors(node):
        cvgeo_vecino = graph.nodes[vecino]["CVEGEO"]
        if cvgeo_vecino is None:
            continue # saltar vecinos sin localidad
        if cvgeo_vecino != cvgeo_node: #si es nodo frontera...
            if cvgeo_node not in fronteras_por_localidad:
                fronteras_por_localidad[cvgeo_node] = set() #conjunto de nodos frontera
            fronteras_por_localidad[cvgeo_node].add(node)
            break # basta con un vecino distinto

for loc, nodos in fronteras_por_localidad.items():
    print(f"Localidad {loc} tiene {len(nodos)} nodos frontera")

# 2) crear funcion

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

# 3) ejecutar para cada localidad
grafo_reducido_total = nx.Graph()
for cvgeo in fronteras_por_localidad.keys(): # recorrer cada localidad con nodos frontera
    grafo_local = construir_clique_localidad(graph, cvgeo, fronteras_por_localidad)

    # agregar todos los nodos al grafo total (con atributos)
    for nodo, datos in grafo_local.nodes(data=True):
        grafo_reducido_total.add_node(nodo, **datos)

    # agregar todas las aristas (con peso y camino)
    for u, v, datos in grafo_local.edges(data=True):
        grafo_reducido_total.add_edge(u, v, **datos)

# 4) visualizar
fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('black')  # fondo fuera del gráfico
ax.set_facecolor('black')         # fondo dentro del gráfico
# poligonos de localidades como fondo
gdf_poly.boundary.plot(ax=ax, color="gray", linewidth=0.5)

# dibujar los nodos frontera
pos = {}
# recorremos todos los nodos del grafo reducido, junto con sus atributos
for n, d in grafo_reducido_total.nodes(data=True):
    x = d["x"]
    y = d["y"]
    pos[n] = (x, y)

nx.draw_networkx_nodes(
    grafo_reducido_total,
    pos=pos,
    ax=ax,
    node_size=24,
    node_color="white"
)

# dibujar las aristas entre nodos frontera
nx.draw_networkx_edges(
    grafo_reducido_total,
    pos=pos,
    ax=ax,
    width=1,
    edge_color="orange"
)

ax.set_title("Red de nodos frontera por localidad", fontsize=20, color="white")
ax.axis("off")

# extraer listas de coordenadas x e y
x_coords = [coord[0] for coord in pos.values()]
y_coords = [coord[1] for coord in pos.values()]
margin = 0.002  # puedes aumentar o reducir esto
x_min, x_max = min(x_coords) - margin, max(x_coords) + margin
y_min, y_max = min(y_coords) - margin, max(y_coords) + margin
# aplicar los límites al gráfico
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.show()


print(len(grafo_reducido_total.nodes))
