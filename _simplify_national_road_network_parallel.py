import os
import sys
import time

from datetime import datetime
from pathlib import Path

from road_network import Road_Network

# ------------------------------------------------------
# GLOBAL VARIABLES
# ------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

FILE = "road_network.pkl"
# Loading the national graph alone takes ~12 GB; this assumes a 32 GB+ machine
N_WORKERS = min(8, os.cpu_count() or 1)

INPUT_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "data" / "results"
LOG_FILE = RESULTS_DIR / f"simplify_national_parallel_{datetime.now():%Y%m%d_%H%M%S}.log"

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


class Tee:
    """Write to the terminal and to the log file at the same time."""
    def __init__(self, stream, log):
        self.stream = stream
        self.log = log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()


# Workers are started with "spawn" on macOS/Windows, which re-imports this
# file, so everything that does work must live under the main guard.
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log = open(LOG_FILE, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    sys.stderr = Tee(sys.__stderr__, log)

    total_start = time.time()
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Workers: {N_WORKERS} (CPU cores: {os.cpu_count()})")
    print(f"Log file: {LOG_FILE}\n")

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
    print_graph_info(road, "Original graph")

    # ------------------------------------------------------
    # SIMPLIFY (PARALLEL)
    # ------------------------------------------------------
    print(f"\n[2/3] Simplifying national road network ({N_WORKERS} workers)...")
    simplified_graph, num_iterations, t = road.simplify(n_workers = N_WORKERS)
    print(
        f"    Done in {t:.1f}s "
        f"(final pass converged in {num_iterations} iterations)"
    )
    print_graph_info(simplified_graph, "Simplified graph")

    simplified_file = "road_network_simplified_parallel.pkl"
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
    internal_file = "road_network_internal_parallel.pkl"
    internal_graph.save(
        path=RESULTS_DIR,
        file=internal_file,
    )
    print(f"    Internal graph saved correctly: {RESULTS_DIR / internal_file}")

    print_graph_info(external_graph, "External graph")
    external_file = "road_network_external_parallel.pkl"
    external_graph.save(
        path=RESULTS_DIR,
        file=external_file,
    )
    print(f"    External graph saved correctly: {RESULTS_DIR / external_file}")

    total = time.time() - total_start
    print(f"\nFinished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"TOTAL TIME: {total:.1f}s ({total / 60:.1f} min)")
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    log.close()
