# Packages
import time
import networkx as nx
from pathlib import Path

from road_network import Road_Network

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent
TESTS_DIR = BASE_DIR / "tests"

# CONSTANTS
SOURCE = "inegi"
POLYGONS_PATH = BASE_DIR / "data" / "raw" / "shp" / "27l.shp"

# INEGI settings 
source_kwargs_inegi = {
    "inegi_graph_path": BASE_DIR / "data" / "processed" / "road_network.pkl"
    }


# DATA LOADING AND PREPROCESSING
print(f"[1/5] Loading graph (source={SOURCE!r})...")
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
road.plot_labeled_network()