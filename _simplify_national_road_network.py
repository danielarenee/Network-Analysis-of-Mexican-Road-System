import time

from pathlib import Path

from road_network import Road_Network

# ------------------------------------------------------
# GLOBAL VARIABLES
# ------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

FILE = "road_network_31.pkl"

INPUT_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "data" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# INEGI settings 
source_kwargs_inegi = {
    "inegi_graph_path": BASE_DIR / "data" / "processed" / FILE
    }

# ------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------
def print_graph_info(graph, title):
    print(title)
    print(f"    Total edges (m): {graph.m:,}")
    print(f"    Total nodes (n): {graph.n:,}")
    print(f"    External nodes: {graph.n_external:,}")
    print(f"    Internal nodes: {graph.n_internal:,}")
    print(f"    Boundary nodes: {graph.n_boundary:,}")
    print(f"    Inner nodes: {graph.n_inner:,}")
    
# ------------------------------------------------------
# LOAD NATIONAL ROAD NETWORK
# ------------------------------------------------------

print("[1/3] Loading national road network...")
t0 = time.time()
road = Road_Network(
    source_kwargs_inegi,
    source = "inegi",
    id_city_label = "CVEGEO",
    length_attr = "length",
    keep_larger_cc = True,
    to_undirected = True,
    to_simple = True
    )
print(f"    Loaded in {time.time() - t0:.1f}s")

# ------------------------------------------------------
# SIMPLIFY
# ------------------------------------------------------
print("\n[2/3] Simplifying national road network...")
simplified_graph, num_iterations, t0 = road.simplify()
print(
    f"    Converged in {num_iterations} iterations "
    f"({time.time() - t0:.1f}s)"
)
print_graph_info(simplified_graph, "Simplified graph")
# Save graph
simplified_file = "road_network_simplified.pkl"
simplified_graph.save(
    path=RESULTS_DIR,
    file=simplified_file,
)
print(f"    Simplified graph saved correctly: {RESULTS_DIR / simplified_file}")

# ------------------------------------------------------
# SPLIT
# ------------------------------------------------------
print("\n[3/3] Splitting simplified graph...")
t0 = time.time()

internal_graph, external_graph = simplified_graph.split()

print(f"    Split completed in {time.time() - t0:.1f}s")

print_graph_info(internal_graph, "Internal graph")
internal_file = "road_network_internal.pkl"
internal_graph.save(
    path=RESULTS_DIR,
    file=internal_file,
)
print(f"    Internal graph saved correctly: {RESULTS_DIR / internal_file}")

print_graph_info(external_graph, "External graph")
external_file = "road_network_external.pkl"
external_graph.save(
    path=RESULTS_DIR,
    file=external_file,
)
print(f"    External graph saved correctly: {RESULTS_DIR / external_file}")