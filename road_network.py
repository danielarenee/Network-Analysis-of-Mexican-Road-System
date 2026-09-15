from copy import deepcopy

from src.utils_2 import load_osmnx_graph, load_inegi_graph, to_connected, to_undirected, to_simple_graph, preprocess_inegi_graph
import src.utils as fc

class Road_Network:
    
    # ------------------------------------------------------
    # Atrributes
    # ------------------------------------------------------
    @property
    def graph(self):
        return self.__graph.copy()

    @property
    def boundary_nodes(self):
        if self.__boundary_nodes is None:
            self.__boundary_nodes = self.__compute_boundary_nodes()
        return self.__boundary_nodes
    
    @property
    def external_nodes(self):
        if self.__external_nodes is None:
            self.__external_nodes = self.__compute_external_nodes()
        return self.__external_nodes
    
    @property
    def reduced_graph(self):
        return self.__reduced_graph.copy()
    
    
    @property
    def n(self):
        return self.__graph.order()
    
    @property
    def m(self):
        return self.__graph.size()
    
    @property
    def n_external(self):
        return len(self.external_nodes)
    
    @property
    def n_internal(self):
        return self.n - self.n_external
    
    @property
    def n_boundary(self):
        return sum(len(v) for v in self.__boundary_nodes.values())
    
    
    # ------------------------------------------------------
    # Constructor
    # ------------------------------------------------------
    def __init__(
            self,
            source_kwargs,
            source = "inegi",
            id_city_label = "CVEGEO",
            external_city_id = None,
            length_attr = "length",
            keep_larger_cc = True,
            to_undirected = True,
            to_simple = True
    ):
        self.__source = source
        self.__id_city_label = id_city_label
        self.__external_city_id = external_city_id
        self.__length_attr = length_attr

        self.__gdf_localities = None
        self.__boundary_nodes = None
        self.__external_nodes = None
        self.__reduced_graph = None
        
        if source == "osmnx":
            self.__crs = "EPSG:4326"
            self.__graph = load_osmnx_graph(**source_kwargs)
            self.__plot_margin = 0.002
        elif source == "inegi":
            self.__crs = "EPSG:6372"
            self.__plot_margin = 500  # meters
            self.__graph = load_inegi_graph(**source_kwargs)
            self.__gdf_nodes_labeled, self.__region_map = preprocess_inegi_graph(
                self.__graph,
                self.__id_city_label,
                self.__crs
            )
        else:
            raise ValueError(f"Unknown source: {source!r}. Use 'osmnx' or 'inegi'.")

        # Keep only the major connected component
        if keep_larger_cc:
            self.__to_connected()
        if to_undirected:
            self.__to_undirected()
        if to_simple:
            self.__to_simple_graph()
        
        self.boundary_nodes
        self.external_nodes
    
    def simplify(self):
        
        new = deepcopy(self)
        
        simplified_graph, num_iterations = fc.simplify_iteratively(self.graph)
        
        new.__graph = simplified_graph.copy()
        new.__boundary_nodes = new.__compute_boundary_nodes()
        new.__external_nodes = new.__compute_external_nodes()
        return new, num_iterations
    
            
    def plot_labeled_network(self):
        fc.plot_labeled_network(
            graph = self.__graph,
            gdf_nodes_labeled = self.__gdf_nodes_labeled,
            gdf_localities = self.__gdf_localities,
            source  = self.__source
        )
        
    def reduce_city_subraphs(self):
        if self.__reduced_graph is None:
            self.__reduced_graph = fc.build_reduced_clique_graph(
                self.__graph,
                self.__boundary_nodes,
                self.__id_city_label
            )
        return self.__reduced_graph
    
    def plot_boundary_nodes_network(self,
                                    title = "Boundary Node Network by Locality"):
        fc.plot_boundary_nodes_network(
            self.__reduced_graph,
            self.__gdf_localities, 
            self.__plot_margin,
            title
            )

    def __to_connected(self):
        self.__graph = to_connected(self.__graph)
        
    def __to_undirected(self):
        self.__graph = to_undirected(self.__graph)
        
    def __to_simple_graph(self):
        self.__graph = to_simple_graph(self.__graph, self.__length_attr)
    
    def __compute_boundary_nodes(self):
        boundary_nodes = self.__boundary_nodes = fc.identify_boundary_nodes(
            self.__graph,
            self.__region_map,
        )
        return boundary_nodes
    
    def __compute_external_nodes(self):
        external_nodes = [
            node for node, idx in self.__graph.nodes(
                data = self.__id_city_label
            ) if idx is self.__external_city_id
        ]
        return external_nodes
