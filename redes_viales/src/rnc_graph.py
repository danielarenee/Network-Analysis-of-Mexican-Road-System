import geopandas as gpd
import pandas as pd
import networkx as nx
import fiona
from datetime import datetime
import numpy as np

#%% 1. EXPLORAR EL GPKG
ruta_gpkg = '/Users/danielarenee/Desktop/honores 2/rnc_data/conjunto_de_datos/rnc2025.gpkg'

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}") # llevar track del proceso con timestamps

#%% 2. CARGAR LAS CAPAS

log("Paso 1: Cargando las capas...")

# Cargar union y redvial
union = gpd.read_file(ruta_gpkg, layer='union_p')
redvial = gpd.read_file(ruta_gpkg, layer='red_vial')

# mantener solo columnas esenciales
columnas_a_mantener = [
    'ID_RED', 'TIPO_VIAL', 'CIRCULA',
    'UNION_INI', 'UNION_FIN', 'LONGITUD', 'geometry'
]
redvial = redvial[columnas_a_mantener].copy()

log("-> Capas listas!")

#%% 3. EXPLORAR LA COLUMNA "CIRCULA" EN REDVIAL
"""
Valores únicos columna "CIRCULA" en redvial : 
- Un sentido (734,468)
- Dos sentidos (3,490,930)
- Cerrada en ambos sentidos (124,237)
- N/A (56)

Total registros: 4,349,691
"""

#%% 4. ELIMINAR DUPLICADOS

log("Paso 2: Eliminando edges duplicados...")

log(f"Edges originales: {len(redvial):,}")

# extraer solo las columnas que necesitamos
cols_necesarias = ['ID_RED', 'UNION_INI', 'UNION_FIN', 'LONGITUD']
redvial_temp = redvial[cols_necesarias].copy()

log("Ordenando por LONGITUD...")
redvial_temp_sorted = redvial_temp.sort_values('LONGITUD')

log("Eliminando duplicados ...") # index to keep set
index_to_keep = redvial_temp_sorted.drop_duplicates(
    subset=['UNION_INI', 'UNION_FIN'],
    keep='first'
).index # Mantenemos el primero aka. el de menor longitud

log(f"Filtrando DataFrame original...")
redvial_sin_dup = redvial.loc[index_to_keep].copy() # filtra filas con indices de index to keep

log(f"- Edges sin duplicados: {len(redvial_sin_dup):,}")
log(f"- Eliminados: {len(redvial) - len(redvial_sin_dup):,}")
log("-> Paso 2 completado!")

#%% 5. DUPLICAR LAS VIAS DE DOBLE SENTIDO

log("Paso 3: Duplicando vías de doble sentido...")

# separar según tipo de circulación
un_sentido = redvial_sin_dup[redvial_sin_dup['CIRCULA'] == 'Un sentido'].copy()
doble_sentido = redvial_sin_dup[redvial_sin_dup['CIRCULA'] == 'Dos sentidos'].copy()
cerradas = redvial_sin_dup[redvial_sin_dup['CIRCULA'] == 'Cerrada en ambos sentidos'].copy()
na_sentido = redvial_sin_dup[redvial_sin_dup['CIRCULA'] == 'N/A'].copy()

"""
CONTEO:
- Un sentido: 733,475
- Dos sentidos: 3,489,356
- Cerradas: 124,155
- N/A: 56
"""

# crear sentido inverso para doble sentido (u -> v se vuelve v -> u)
log("Creando edges inversos...")
doble_sentido_inverso = doble_sentido.copy()  # nuevo df, intercambiamos columnas
doble_sentido_inverso['UNION_INI'], doble_sentido_inverso['UNION_FIN'] = \
    doble_sentido['UNION_FIN'].values, doble_sentido['UNION_INI'].values

#TODO: si vamos a trabajar como grafo dirigido, tenemos que girar la geometry

# actualizar CIRCULA a "Un sentido" en doble_sentido y su inverso
doble_sentido['CIRCULA'] = 'Un sentido'
doble_sentido_inverso['CIRCULA'] = 'Un sentido'

# Combinar
log("Combinando edges...")
edges_final = pd.concat([
    un_sentido,
    doble_sentido,
    doble_sentido_inverso,
    cerradas,
    na_sentido
], ignore_index=True)

log("-> Paso 3 completado!")

# 5.2 GUARDAR COMO GPKG

log("Guardando edges_final...")
edges_final.to_file(
    'edges_final.gpkg',
    layer='edges',
    driver='GPKG'
)

log("-> edges_final.gpkg guardado")

#%%
