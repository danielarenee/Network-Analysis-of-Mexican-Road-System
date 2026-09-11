def load_osmnx_graph(**source_kwargs):
    import osmnx as ox
    
    # download road network with osmnx
    center = (
        source_kwargs["center_lat"],
        source_kwargs["center_lon"]
        )
    graph = ox.graph_from_point(
        center,
        dist = source_kwargs["network_radius"],
        network_type = "drive",
    )
    return graph

def load_inegi_graph(**source_kwargs):
    import pickle
    with open(source_kwargs["inegi_graph_path"], "rb") as f:
        graph = pickle.load(f)
    return graph