"""
este solo es un minitest de la función que visualiza dijkstra entre 2 nodos
"""

import random
from pathlib import Path
from road_network import Road_Network

BASE_DIR = Path(__file__).resolve().parent
ENT = "27"  # any of the 32 state codes in data/processed/road_network_XX.pkl

road = Road_Network(
    {"inegi_graph_path": BASE_DIR / "data" / "processed" / f"road_network_{ENT}.pkl"},
    source="inegi",
)

nodes = list(road.graph.nodes)
source, target = random.sample(nodes, 2)

distance, path = road.plot_shortest_path(source, target)
print(f"Distance: {distance:.2f} m, nodes in path: {len(path)}")

"""
note 2 self:
conda activate base
python minitest_viz.py
"""