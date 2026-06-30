###########
# PACKAGES
###########
# Import packages
import geopandas as gpd

from pathlib import Path

###########
# VARIABLES
###########
# Set relative path
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Columns to select from the join file
columns_union = ["ID_UNION", "geometry"]
columns_cities = ["id_convex", "geometry"]
columns_roads = ["NOMBRE", "LONGITUD", "geometry", "CIRCULA", "UNION_INI", "UNION_FIN", "COND_PAV", "ESTATUS", "CONDICION", "TIPO_VIAL"]

# Features fo filter (roads)
# COND_PAV: ['N/A', 'Con pavimento', 'Sin pavimento']
road_surface = ["Con pavimento", "N/A"]
# ESTATUS: ['Habilitado', 'Deshabilitado']
status = ["Habilitado"]
# CONDICION: ['En operación', 'En construcción - cerrado', 'En construcción - abierto']
operational = ['En operación', 'En construcción - abierto']
# CIRCULA: ['Un sentido', 'Dos sentidos', 'Cerrada en ambos sentidos', 'N/A']
direction = ['Cerrada en ambos sentidos']
# TIPO_VIAL: [      'Periférico',            'Calle',          'Avenida',
#         'Circuito',         'Viaducto',          'Calzada',
#         'Eje vial',           'Enlace',        'Retorno U',
#        'Boulevard',        'Carretera',          'Privada',
#          'Retorno',     'Prolongación',         'Corredor',
#         'Callejón',         'Glorieta',           'Camino',
#          'Cerrada',             'Otro',          'Andador',
#           'Vereda',         'Diagonal', 'Rampa de frenado',
#         'Peatonal',     'Continuación',       'Ampliación',
#   'Circunvalación',           'Pasaje']
road_type = ["Privada", "Callejón", "Camino", "Cerrada", "Vereda",
             "Rampa de frenado", "Peatonal", "Pasaje"]
# Coordinate Reference System
epsg = 6372

###########
# MAIN
###########

# Define paths
rnc_gpkg_path = BASE_DIR / "data" / "raw" / "rnc2025.gpkg"
localities_gpkg_path = BASE_DIR / "data" / "raw" / "LocalitiesGrouped_2020_data.gpkg"
unions_output_path = BASE_DIR / "data" / "processed" / "unions.gpkg"
roads_output_path = BASE_DIR / "data" / "processed" / "roads.gpkg"

# Read unions (points)
print(f"Reading unions layer from {rnc_gpkg_path}...")
rnc_union = gpd.read_file(rnc_gpkg_path, layer = "union_p", columns = columns_union)
# Set crs
rnc_union = rnc_union.to_crs(epsg=epsg)

# Read city boundaries
print(f"Reading city boundaries from {localities_gpkg_path}...")
cities = gpd.read_file(localities_gpkg_path, columns = columns_cities)
cities["id_convex"] = cities["id_convex"].astype(int)

print("Performing spatial join...")
rnc_union = gpd.sjoin(
    rnc_union,
    cities,
    how="left",
    predicate="within"
)
rnc_union = rnc_union.drop(columns="index_right")

# Save file
print(f"Saving unions to {unions_output_path}...")
unions_output_path.parent.mkdir(parents=True, exist_ok=True)
rnc_union.to_file(unions_output_path,
                  driver = "GPKG",
                  index = False)
# Delete file
del rnc_union

# Read roads (lines)
print(f"Reading red_vial layer from {rnc_gpkg_path}...")
rnc_roads = gpd.read_file(rnc_gpkg_path, layer = "red_vial", columns = columns_roads)
# Filter
print("Filtering road layers...")
rnc_roads = rnc_roads[rnc_roads["COND_PAV"].isin(road_surface)]
rnc_roads = rnc_roads[rnc_roads["ESTATUS"].isin(status)]
rnc_roads = rnc_roads[rnc_roads["CONDICION"].isin(operational)]
rnc_roads = rnc_roads[~rnc_roads["CIRCULA"].isin(direction)]
rnc_roads = rnc_roads[~rnc_roads["TIPO_VIAL"].isin(road_type)]
# Set crs
rnc_roads = rnc_roads.to_crs(epsg=epsg)
# Save file
print(f"Saving roads to {roads_output_path}...")
rnc_roads.to_file(roads_output_path,
                  driver = "GPKG",
                  index = False)
del rnc_roads

