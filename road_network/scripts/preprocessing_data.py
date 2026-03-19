###########
# LIBRARIES
###########
# Import libraries
import geopandas as gpd

from pathlib import Path

###########
# VARIABLES
###########
# Set relative path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = str(BASE_DIR)

# Columns to select from the join file
columns_union = ["ID_UNION", "geometry"]

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
# Read unions (points)
rnc_union = gpd.read_file(DATA_DIR + "\\temp\\rnc2025.gpkg",
                    layer = "union_p")
# Select columns
rnc_union = rnc_union[columns_union]
# Set crs
rnc_union = rnc_union.to_crs(epsg=epsg)
# Save file
rnc_union.to_file(DATA_DIR + "\\temp\\unions.gpkg",
                  driver = "GPKG",
                  index = False)
# Delete file
del rnc_union

# Read roads (lines)
rnc_roads = gpd.read_file(DATA_DIR + "\\temp\\rnc2025.gpkg",
                    layer = "red_vial")
# Filter
rnc_roads = rnc_roads[rnc_roads["COND_PAV"].isin(road_surface)]
rnc_roads = rnc_roads[rnc_roads["ESTATUS"].isin(status)]
rnc_roads = rnc_roads[rnc_roads["CONDICION"].isin(operational)]
rnc_roads = rnc_roads[~rnc_roads["CIRCULA"].isin(direction)]
rnc_roads = rnc_roads[~rnc_roads["TIPO_VIAL"].isin(road_type)]
# Set crs
rnc_roads = rnc_roads.to_crs(epsg=epsg)
# Save file
rnc_roads.to_file(DATA_DIR + "\\temp\\roads.gpkg",
                  driver = "GPKG",
                  index = False)