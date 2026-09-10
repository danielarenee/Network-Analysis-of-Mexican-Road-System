import time
import src.utils as fc
import networkx as nx
from pathlib import Path

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent
TESTS_DIR = BASE_DIR / "tests"

# CONSTANTS
SOURCE = "inegi"  # "osmnx" or "inegi"
KEEP_LARGER_CC = True

CODE_NAME = "CVEGEO"
#  OSMnx settings 
SHAPEFILE_PATH = BASE_DIR / "data" / "raw" / "shp" / "27l.shp"
CENTER_LAT = 17.930714
CENTER_LON = -93.507545
NETWORK_RADIUS = 7000  # meters
# INEGI settings 
INEGI_GRAPH_PATH = BASE_DIR / "data" / "processed" / "road_network.pkl"

# DATA LOADING AND PREPROCESSING
print(f"[1/5] Loading graph (source={SOURCE!r})...")
t0 = time.time()

graph, gdf_nodes_labeled, gdf_localities, cvegeo_map, CRS, PLOT_MARGIN = fc.load_and_preprocess_graph(
    source=SOURCE,
    shapefile_path=SHAPEFILE_PATH,
    center_lat=CENTER_LAT,
    center_lon=CENTER_LON,
    network_radius=NETWORK_RADIUS,
    inegi_graph_path=INEGI_GRAPH_PATH,
)
# Keep only the major connected component
if KEEP_LARGER_CC:
    cc = nx.weakly_connected_components(graph)
    larger_cc_nodes = max(cc, key=len)
    graph = graph.subgraph(larger_cc_nodes).copy()
print(f"    Graph loaded: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges ({time.time()-t0:.1f}s)")
connected = nx.is_weakly_connected(graph)
print(f"    Is (weakly) connected?: {connected}")
external_nodes = [
    node for node, idx in graph.nodes(data=CODE_NAME) if idx is None
]
print(f"    External nodes: {len(external_nodes):,}, Internal nodes: {graph.number_of_nodes()-len(external_nodes):,}")

# Convertir a grafo simple no dirigido
graph = graph.to_undirected(as_view=False)
planar = nx.is_planar(graph)
print(f"    Is planar: {planar}")

# Visualization
fc.plot_labeled_network(graph, gdf_nodes_labeled, gdf_localities, source=SOURCE)

# BOUNDARY NODE IDENTIFICATION
print(f"[2/5] Identifying boundary nodes...")
t0 = time.time()
boundary_nodes_by_locality = fc.identify_boundary_nodes(graph, cvegeo_map)
total_boundary = sum(len(v) for v in boundary_nodes_by_locality.values())
print(f"    {len(boundary_nodes_by_locality):,} localities, {total_boundary:,} boundary nodes total ({time.time()-t0:.1f}s)")

#%%%
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Unir los vértices de todas las regiones
dictionary_nodes = set().union(*boundary_nodes_by_locality.values())

# Algunos pueden haberse eliminado al tomar la componente principal
highlighted_nodes = dictionary_nodes.intersection(graph.nodes)
regular_nodes = set(graph.nodes).difference(highlighted_nodes)
missing_nodes = dictionary_nodes.difference(graph.nodes)

# Posiciones geográficas
pos = {
    node: (data["x"], data["y"])
    for node, data in graph.nodes(data=True)
}

fig, ax = plt.subplots(figsize=(14, 14))

# Opcional: límites de las localidades
if gdf_localities is not None:
    gdf_localities.boundary.plot(
        ax=ax,
        color="black",
        linewidth=0.5,
        alpha=0.4,
    )

# Aristas
nx.draw_networkx_edges(
    graph,
    pos,
    ax=ax,
    edge_color="#8c8c8c",
    width=0.5,
    alpha=0.6,
)

# Nodos que no aparecen en el diccionario
nx.draw_networkx_nodes(
    graph,
    pos,
    nodelist=list(regular_nodes),
    ax=ax,
    node_color="#bdbdbd",
    node_size=5,
    alpha=0.7,
)

# Nodos que aparecen en el diccionario
nx.draw_networkx_nodes(
    graph,
    pos,
    nodelist=list(highlighted_nodes),
    ax=ax,
    node_color="#e41a1c",
    node_size=35,
    edgecolors="black",
    linewidths=0.5,
    label="Vértices regionales",
)

legend_elements = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor="#e41a1c",
        markeredgecolor="black",
        markersize=8,
        label="En el diccionario",
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor="#bdbdbd",
        markersize=6,
        label="Fuera del diccionario",
    ),
]

ax.legend(handles=legend_elements)
ax.set_aspect("equal")
ax.set_axis_off()
plt.tight_layout()
plt.show()

print(f"Nodos resaltados: {len(highlighted_nodes):,}")
print(f"Nodos normales:   {len(regular_nodes):,}")
print(f"Nodos del diccionario ausentes del grafo: {len(missing_nodes):,}")

if missing_nodes:
    print("Ausentes:", sorted(missing_nodes))
    
    
    #%%%
import pandas as pd
from pathlib import Path


def normalize_cvegeo(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]

    return value


# Número total de nodos por CVEGEO
nodes_by_locality = (
    gdf_nodes_labeled.loc[
        gdf_nodes_labeled["CVEGEO"].notna(),
        ["CVEGEO"],
    ]
    .assign(
        CVEGEO=lambda df: df["CVEGEO"].map(normalize_cvegeo)
    )
    .groupby("CVEGEO")
    .size()
    .rename("numero_nodos")
)

# Número de nodos boundary por CVEGEO
boundary_count = pd.Series(
    {
        normalize_cvegeo(cvegeo): len(set(nodes))
        for cvegeo, nodes in boundary_nodes_by_locality.items()
    },
    name="numero_nodos_boundary",
    dtype="int64",
)

# Combinar resultados
summary = (
    nodes_by_locality
    .to_frame()
    .join(boundary_count, how="outer")
    .fillna(0)
    .astype({
        "numero_nodos": "int64",
        "numero_nodos_boundary": "int64",
    })
    .rename_axis("CVEGEO")
    .reset_index()
    .sort_values("CVEGEO")
    .reset_index(drop=True)
)

# Guardar en la carpeta de trabajo de Spyder
output_path = Path.cwd() / "conteo_nodos_por_cvegeo.csv"

summary.to_csv(
    output_path,
    index=False,
    sep=";",
    encoding="utf-8-sig",
)

print(summary)
print(f"\nArchivo creado en:\n{output_path.resolve()}")