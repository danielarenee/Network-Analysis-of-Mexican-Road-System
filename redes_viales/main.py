import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import shapely as shp
from networkx.classes import neighbors

import src.func as fc
import numpy as np

from shapely.geometry import Point
from scipy.spatial import Delaunay


#%%

# Ruta al archivo shapefile
ruta = "/Users/danielarenee/Desktop/honores/redes_viales/Data/shp/27l.shp"

gdf_poly = gpd.read_file(ruta)
gdf_poly = gdf_poly.to_crs(4326)

# crear grafo
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

# crear df de los nodos
nodes_data = [] #set para los nodos
for node_id, data in graph.nodes(data=True):
    nodes_data.append({
        "node_id": node_id, #id, x y y
        "x": data["x"],
        "y": data["y"]
    })
df_nodes = pd.DataFrame(nodes_data) #crear el df

# crear gdf
nodes_geom =[]
for _, row in df_nodes.iterrows():
    punto = Point(row["x"],row["y"]) #punto para la geometria
    nodes_geom.append(punto)
df_nodes["geometry"] = nodes_geom
gdf_nodes = gpd.GeoDataFrame(df_nodes, geometry="geometry", crs="EPSG:4326")

# union espacial
#busca que puntos estan dentro de que poligono
gdf_union = gpd.sjoin(gdf_nodes, gdf_poly, how="left", predicate="within")

# etiquetar nodos
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

# 1) identificar nodos frontera en un diccionario
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


#for loc, nodos in fronteras_por_localidad.items():
#    print(f"Localidad {loc} tiene {len(nodos)} nodos frontera")


# 3) ejecutar para cada localidad
grafo_reducido_total = nx.Graph()
for cvgeo in fronteras_por_localidad.keys(): # recorrer cada localidad con nodos frontera
    grafo_local = fc.construir_clique_localidad(graph, cvgeo, fronteras_por_localidad)

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

# filtrar por localidad - crea diccionario CVGEO: [lista de nodos]
localidades_por_cvegeo = {}
for node, data in graph.nodes(data=True):
    cvgeo = data.get("CVEGEO")
    if cvgeo is None:
        continue
    localidades_por_cvegeo.setdefault(cvgeo, []).append(node)

# conseguir centroides- loc : tupla de coords
centroides = {}
for loc, nodes in localidades_por_cvegeo.items():
    coords = []
    for n in nodes:
        data_nodo = graph.nodes[n]
        if "x" in data_nodo and "y" in data_nodo:
            x = data_nodo["x"]
            y = data_nodo["y"]
            coords.append((x, y))
    if not coords:
        continue
    #conseguir una lista de xs y una de ys
    xs, ys = zip(*coords)
    centroides[loc] = (sum(xs) / len(xs), sum(ys) / len(ys))

# lista de CVGEOS de las localidades que tienen centroide
labels = list(centroides.keys())
# covertir a array para input delaunay
points = np.array([centroides[loc] for loc in labels])

# construir triangulacion de delaunay
tri = Delaunay(points)

# graficar
plt.figure(figsize=(6,6))
plt.triplot(points[:,0], points[:,1], tri.simplices, linewidth=0.8)
plt.scatter(points[:,0], points[:,1], color='red', s=30)
for i, loc in enumerate(labels):
    plt.text(points[i,0], points[i,1], loc, fontsize=8, ha='center', va='center')
plt.title("Triangulación de Delaunay de centroides")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.gca().set_aspect('equal', 'box')
plt.show()

#%%
# hacer función para ir eliminando iterativamente los vértices de grado 1



grafo_podado, removed_nodes = fc.podar_grado_1(graph)

fig, ax = ox.plot_graph(grafo_podado, show=False, close=False)
fig.patch.set_facecolor('black')  # fondo fuera del gráfico
ax.set_facecolor('black')
gdf_poly.boundary.plot(ax=ax, color="red")
gdf_union.plot(ax=ax, column="CVEGEO", cmap="Set2")
ax.set_title("Ernersto aguirre, Tabasco", fontsize=16, color="white")
plt.show()

#%%
# ------- UPDATE --------

grafo_podado_2, removed_nodes_2 = fc.podar_grado_2(grafo_podado)

fig, ax = ox.plot_graph(grafo_podado_2, show=False, close=False)
fig.patch.set_facecolor('black')  # fondo fuera del gráfico
ax.set_facecolor('black')
gdf_poly.boundary.plot(ax=ax, color="red")
gdf_union.plot(ax=ax, column="CVEGEO", cmap="Set2")
ax.set_title("Ernersto aguirre, Tabasco", fontsize=16, color="white")
plt.show()



#%%
