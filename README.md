# Mexican Road Network Analysis

This repository brings together geospatial and graph-based work on road connectivity in Mexico. It contains two related but distinct workflows: a national road-network pipeline based on INEGI data, and a local case study of connections between localities in Tabasco.

## What the project examines

The code explores how road segments can be represented as weighted graphs, how nearby localities connect through roads, and how network structure can be simplified or summarized. The two workflows use different inputs and should not be read as one end-to-end analysis.

## Data

### National road-network workflow

- **INEGI Red Nacional de Caminos (RNC) 2025:** the scripts expect the GeoPackage layers `red_vial` (road segments) and `union_p` (road junctions). The full RNC GeoPackage is not included in this repository. [INEGI source package](https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/productos/geografia/caminos/2025/794551163030_gpk.zip)
- **Grouped locality data:** [`LocalitiesGrouped_2020_data.gpkg`](road_network/data/LocalitiesGrouped_2020_data.gpkg) supplies locality polygons used to associate road junctions with areas.
- **Prepared test inputs:** `road_network/test/` contains GeoPackages and a serialized graph for development and testing.

### Tabasco locality case study

- **Locality boundaries:** [`27l.shp`](redes_viales/Data/shp/27l.shp) and its companion shapefile files are included in the repository.
- **Road network:** `redes_viales/main.py` requests a driving network from OpenStreetMap through OSMnx, centered near Ernesto Aguirre, Tabasco, with a 7 km radius. The returned network depends on the OpenStreetMap data available when the script runs.

## Methods and outputs

### National workflow

The preprocessing script filters RNC roads by surface, status, operating condition, direction, and road type, projects the layers, and saves prepared road and junction GeoPackages. The graph builder creates a directed NetworkX multigraph: road segments carry attributes such as name, length, and geometry, and two-way roads are represented in both directions. The graph is saved as `road_network.pkl`.

A separate exploratory RNC script also removes duplicate junction pairs, retains the shortest segment among duplicates, and adds reverse edges for two-way roads before writing `edges_final.gpkg`.

### Tabasco workflow

The local script joins OSMnx road nodes to the locality boundaries using `CVEGEO` codes. It identifies nodes at locality boundaries, builds within-locality clique graphs using shortest paths, triangulates locality centroids with Delaunay triangulation, simplifies the road graph by removing dead ends and merging degree-2 nodes, and calculates inter-region distances. Its current outputs are plots displayed during execution and a printed distance summary; it does not save a complete set of final figures or results.

## Related conference presentation

The work is related to the presentation *Un algoritmo para la construcción de la red de conectividad carretera entre ciudades* (“An algorithm for constructing an intercity road-connectivity network”), by Héctor S. Maravillo G. and Daniela R. Ramírez Gutiérrez, presented at the XIV Congreso de la Sociedad Mexicana de Investigación de Operaciones (UDEM, October 7–9, 2026). The presentation describes an intercity connectivity algorithm and reports its own network results. Those figures are not presented here as outputs reproduced by this repository.

## Running the code

These are research scripts rather than a packaged command-line tool, and a fresh checkout is not currently reproducible without local setup:

1. Install the Python packages used by the relevant workflow, including GeoPandas, NetworkX, OSMnx, pandas, NumPy, Shapely, SciPy, Matplotlib, and (for the national scripts) tqdm.
2. For the national workflow, obtain the full INEGI RNC 2025 GeoPackage and place it where the scripts expect it. Then prepare the road and junction layers before building the graph. The checked-in `road_network/test/` files are prepared inputs, not the full source GeoPackage.
3. For the Tabasco workflow, update `SHAPEFILE_PATH` in `redes_viales/main.py` to point to the included `redes_viales/Data/shp/27l.shp`, then run `python redes_viales/main.py`. An internet connection is needed for the OSMnx network request.

Several scripts contain machine-specific paths or Windows-style path construction. Adjust those paths locally before running the national pipeline as well.
