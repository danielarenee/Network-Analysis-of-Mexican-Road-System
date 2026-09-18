from copy import deepcopy
from networkx import set_node_attributes

from src.utils_2 import (load_osmnx_graph, load_inegi_graph, to_connected, 
                         to_undirected, to_simple_graph, preprocess_inegi_graph, 
                         networkx_to_igraph, igraph_to_gdf)
from src.algorithms import voronoi_dijkstra
import src.utils as fc


class Road_Network:
    
    # ------------------------------------------------------
    # ATTRIBUTES
    # ------------------------------------------------------
    @property
    def graph(self):
        """Return a copy of the NetworkX Graph representation."""
        return self._graph()

    @property
    def boundary_nodes(self):
        """Return a dictionary of boundary nodes by region."""
        if self.__boundary_nodes is None:
            self.__boundary_nodes = self.__compute_boundary_nodes()
            
            # Map NetworkX node IDs to igraph vertex indices
            map_node_id_to_ig = {
                node_id: i
                for i, node_id in enumerate(self.__nx_graph.nodes)
            }
            self.__boundary_nodes_ig = {
                region: {
                    map_node_id_to_ig[node_id]
                    for node_id in nodes
                }
                for region, nodes in self.__boundary_nodes.items()
            }
            
        return self.__boundary_nodes
    
    @property
    def external_nodes(self):
        """Return a list of nodes outside regions"""
        if self.__external_nodes is None:
            self.__external_nodes = self.__compute_external_nodes()
        return self.__external_nodes
    
    @property
    def reduced_graph(self):
        """Return a copy of the reduced road network."""
        return self.__reduced_graph.copy()
    
    @property
    def n(self):
        """Return the number of nodes."""
        return self.__nx_graph.order()
    
    @property
    def m(self):
        """Return the number of edges."""
        return self.__nx_graph.size()
    
    @property
    def n_external(self):
        """Return the number of external nodes."""
        return len(self.external_nodes)
    
    @property
    def n_internal(self):
        """Return the number of non-external nodes."""
        return self.n - self.n_external
    
    @property
    def n_boundary(self):
        """Return the total number of boundary nodes."""
        return sum(len(v) for v in self.__boundary_nodes.values())
    
    
    # ------------------------------------------------------
    # CONSTRUCTOR
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
        """
        Load and preprocess the road network.

        The graph can optionally be restricted to its largest connected
        component, converted to undirected form, and simplified.
        """
        
        # Set parameters
        self.__source = source
        self.__id_city_label = id_city_label
        self.__external_city_id = external_city_id
        self.__length_attr = length_attr

        self.__gdf_localities = None
        self.__boundary_nodes = None
        self.__external_nodes = None
        self.__reduced_graph = None
        
        self.__ig_graph = None
        
        # Load graph and define source-specific spatial parameters
        if source == "osmnx":
            self.__crs = "EPSG:4326"
            self.__nx_graph = load_osmnx_graph(**source_kwargs)
            self.__plot_margin = 0.002
        elif source == "inegi":
            self.__crs = "EPSG:6372"
            self.__plot_margin = 500  # meters
            self.__nx_graph = load_inegi_graph(**source_kwargs)
            self.__gdf_nodes_labeled, self.__region_map = preprocess_inegi_graph(
                self.__nx_graph,
                self.__id_city_label,
                self.__crs
            )
        else:
            raise ValueError(f"Unknown source: {source!r}. Use 'osmnx' or 'inegi'.")
        
        # Normalize graph topology
        if keep_larger_cc:
            self.__to_connected()
        if to_undirected:
            self.__to_undirected()
        if to_simple:
            self.__to_simple_graph()
        
        # Identify relevant node classes
        self.boundary_nodes
        self.external_nodes


    # ------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------
    def simplify(self):
        
        """
        Iteratively simplify the road network.
         
        Returns
        -------
        Road_Network
            Independent copy containing the simplified graph.
        int
            Number of simplification iterations performed.
        """
        new = deepcopy(self)
        
        simplified_graph, num_iterations = fc.simplify_iteratively(self.graph)
        
        new.__nx_graph = simplified_graph.copy()
        
        # Recompute node classifications after changing topology
        new.__boundary_nodes = new.__compute_boundary_nodes()
        new.__external_nodes = new.__compute_external_nodes()
        return new, num_iterations
    
            
    def plot_labeled_network(self):
        """Plot the road network colored or labeled by locality."""
        fc.plot_labeled_network(
            graph = self.__nx_graph,
            gdf_nodes_labeled = self.__gdf_nodes_labeled,
            gdf_localities = self.__gdf_localities,
            source  = self.__source
        )
    
    
    def plot_boundary_nodes_network(self,
                                    title = "Boundary Node Network by Locality"):
        """Plot the reduced network and its locality boundary nodes."""
        fc.plot_boundary_nodes_network(
            self.__reduced_graph,
            self.__gdf_localities, 
            self.__plot_margin,
            title
            )
        
        
    def reduce_city_subraphs(self):
        """
       Compute the reduced boundary-node clique graphs.

        Returns
        -------
        networkx.Graph
            Reduced representation of the original road network.
        """
        if self.__reduced_graph is None:
            self.__reduced_graph = fc.build_reduced_clique_graph(
                self.__nx_graph,
                self.__boundary_nodes,
                self.__id_city_label
            )
        return self.__reduced_graph
    
    
    def networkx_to_igraph(self):
        """Convert the NetworkX graph to an igraph representation."""
        if self.__source == "inegi":
            self.__ig_graph = networkx_to_igraph(
                nx_graph = self.__nx_graph,
                id_city_label = self.__id_city_label,
                )
    

    def to_gdf(self,
               R = None,
               d = None):
        """
        Convert the igraph network into node and edge GeoDataFrames.

        Parameters
        ----------
        R : optional (Default: None)
            Region associated with graph nodes.
        d : optional (Default: None)
            Distance associated with graph nodes.

        Returns
        -------
        geopandas.GeoDataFrame
            Node geometries and attributes.
        geopandas.GeoDataFrame
            Edge geometries and attributes.
        """
        if self.__ig_graph is None:
            self.networkx_to_igraph()        
        nodes_gdf, edges_gdf = igraph_to_gdf(
            g = self.__ig_graph,
            crs = self.__crs,
            R = R,
            d = d
        )
        return nodes_gdf, edges_gdf 
    
    
    def voronoi_dijkstra(self):
        """
        Run the Voronoi-Dijkstra algorithm on the igraph network.
    
        Returns
        -------
        d
            Shortest distance computed for each node from its assigned source.
        p
            Predecessor of each node in the shortest-path tree.
        R
            Region assigned to each node by the Voronoi partition.
        F
            Frontier information between neighboring Voronoi regions.
        contador
            Number of iterations performed by the algorithm.
        final_time
            Total execution time of the algorithm.
        """
        d, p, R, F, contador, final_time = voronoi_dijkstra(
            g = self.__ig_graph, 
            id_city_label = self.__id_city_label,
            external_city_id = self.__external_city_id
        )
        return d, p, R, F, contador, final_time
    
    
    # ------------------------------------------------------
    # PROTECTED METHODS
    # ------------------------------------------------------
    def _graph(self, kind = "nx"):
        """Return a copy of the NetworkX or igraph representation."""
        if kind == "nx":
            return self.__nx_graph.copy()
        elif kind == "ig":
            return self.__ig_graph.copy()
        else:
            raise ValueError(f"Unknown kind: {kind!r}. Use 'nx' or 'ig'.")
    
    
    # ------------------------------------------------------
    # PRIVATE METHODS
    # ------------------------------------------------------
    def __to_connected(self):
        """Keep only the main connected component."""
        self.__nx_graph = to_connected(self.__nx_graph)
        
        
    def __to_undirected(self):
        """Convert the network to an undirected graph."""
        self.__nx_graph = to_undirected(self.__nx_graph)
        
        
    def __to_simple_graph(self):
        """Convert the network to a simple graph"""
        self.__nx_graph = to_simple_graph(self.__nx_graph, self.__length_attr)
    
    
    def __compute_boundary_nodes(self):
        """
        Identify the nodes connecting regions with external vertices or other regions..
        """
        boundary_nodes = self.__boundary_nodes = fc.identify_boundary_nodes(
            self.__nx_graph,
            self.__region_map,
        )
        
        # Initialize all nodes as non-boundary
        set_node_attributes(self.__nx_graph, False, "boundary")
        
        # Mark identified boundary nodes
        for nodes in boundary_nodes.values():
            for node in nodes:
                self.__nx_graph.nodes[node]["boundary"] = True
        return boundary_nodes
    
    
    def __compute_external_nodes(self):
        """Identify nodes assigned to the external region."""
        external_nodes = [
            node for node, idx in self.__nx_graph.nodes(
                data = self.__id_city_label
            ) if idx is self.__external_city_id
        ]
        return external_nodes
