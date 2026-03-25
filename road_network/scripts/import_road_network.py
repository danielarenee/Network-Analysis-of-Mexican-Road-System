###########
# PACKAGES
###########
# Import packages
import pickle

from pathlib import Path


###########
# VARIABLES
############ 
# Set absolute path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = str(BASE_DIR)
folder = "test"

###########
# MAIN
############ 

#Import road network
with open(DATA_DIR + f"\\{folder}\\road_network.pkl", "rb") as f:
    G = pickle.load(f)
    
# Print network features
print("Order: ", G.order())
print("Size: ", G.size())
print("Graph type: ", type(G))
print("Directed:", G.is_directed())
print("Node attributes: ", list(list(G.nodes(data=True))[0][1].keys()))
print("Edge attributes: ", list(list(G.edges(data=True))[0][2].keys()))
