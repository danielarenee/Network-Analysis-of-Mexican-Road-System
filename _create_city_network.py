# ------------------------------------------------------
# PACKAGES
# ------------------------------------------------------ 
from pathlib import Path

from road_network import Road_Network
from city_network import City_Network

# ------------------------------------------------------
# GLOBAL VARIABLES
# ------------------------------------------------------ 
# Get project root directory
BASE_DIR = Path(__file__).resolve().parent

PATH = BASE_DIR / "data" / "results"
FILE = "road_network_external_parallel"
#FILE = "external_29"
#PATH = Path("C:\\Users\\Saib\\Documents\\Zoom\\")


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------ 

# Import national road network
national_network = Road_Network.load(PATH, FILE)
print("National road network")
print(f"    External nodes: {national_network.n_external:,}")
print(f"    Internal nodes: {national_network.n_internal:,}")
print(f"    Boundary nodes: {national_network.n_boundary:,}")
print(f"    Inner nodes: {national_network.n_inner:,}")
print(f"    Total edges: {national_network.m:,}")
print(f"    Localities: {len(national_network.boundary_nodes):,}")
print(f"    Mean degree: {national_network.mean_degree:.2f}")
print(f"    Density: {national_network.density:.5f}")


# Build city network
region_graph, t = national_network.build_region_graph()

# Create a instance of City_Network class
city_network = City_Network(region_graph)

print("National city network")
print(f"  Execution time: {t:.5f}")
print(f"    Order: {city_network.n:,}")
print(f"    Size: {city_network.m:,}")
print(f"    Vertex connectivity: {city_network.vertex_connectivity:,}")
print(f"    Edge connectivity: {city_network.edge_connectivity:,}")

print(f"    Mean degree: {city_network.mean_degree:.2f}")
print(f"    Density: {city_network.density:.5f}")


# Draw city_network
city_network.plot((25, 25))
city_network.plot((25, 25), False)

