import heapq
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
 
    @property
    def node_to_ig(self):
        if self.__node_to_ig is None:
            self.__node_to_ig = {node_id: i for i, node_id in enumerate(self.__ig_graph.vs["node_id"])}
        return self.__node_to_ig
    
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
        self.__node_to_ig = None
               
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
            
        self.networkx_to_igraph()
        
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
        
        all_boundary_nodes = set().union(*self.boundary_nodes.values())
        simplified_graph, num_iterations = fc.simplify_iteratively(
            self.graph,
            all_boundary_nodes
        )
        
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


    def dijkstra(self, source, targets=None, weight="length"):
        """
        Compute shortest paths from a single source node to a subset of
        target nodes using a binary-heap Dijkstra over the igraph
        representation.

        Parameters
        ----------
        source
            Source node
        targets : optional (Default: None)
            Destination nodes. If None, distances are computed to every node
            reachable from the source.
        weight : str, optional
            Edge attribute used as cost. Default 'length'.

        Returns
        -------
        distances : dict
            Mapping from target node ID to its shortest distance from
            the source.
        paths : dict
            Mapping from target node ID to the ordered list of node IDs
            along the shortest path from the source.
        """

        # first build the igraph graph in case we havent 
        if self.__ig_graph is None:
            self.networkx_to_igraph()

        g = self.__ig_graph # g is the igraph copy

        # g.vs["node_id"] has the original NetworkX node ID for every vertex
        # so we can work with igraph indexes and Nx mode_ids
        node_to_ig = self.node_to_ig

        # translate source node id to igraph index
        if source not in node_to_ig:
            raise KeyError(f"Source node {source!r} is not in the graph.")
        source_ig = node_to_ig[source]

        if targets is None:
            target_ig_ids = None
        else: # same thing for target nodes
            if isinstance(targets, int):
                targets = [targets]
                
            target_ig_ids = set()
            for target in targets:
                if target not in node_to_ig:
                    raise KeyError(f"Target node {target!r} is not in the graph.")
                target_ig_ids.add(node_to_ig[target])

        # Dijkstra
        # dist[u] : dict of best known distance from source_ig to u so far
        # prev[u] : predecessor of u on that best-known path (used to walk the path 
        #           back to the source).
        # visited : set of vertices whose shortest distance is finalized
        # heap : min-priority queue of (distance, vertex) pairs, always
        #           popping the currently-closest unvisited vertex next
        # pending : if we were given specific targets, this is the set of
        #           igraph indices we're still waiting to finalize. it lets us 
        #           break out of the loop when empty

        # initialize
        dist = {source_ig: 0.0} 
        prev = {source_ig: None}
        visited = set()
        pending = set(target_ig_ids) if target_ig_ids is not None else None
        heap = [(0.0, source_ig)] # first push

        # keep going while there is something left to explore
        # and we have targets unresolved
        while heap and (pending is None or pending):
            d, u = heapq.heappop(heap) 
            if u in visited:
                continue
            visited.add(u)
            if pending is not None:
                pending.discard(u)

            # relax every edge (u, v) aka. if going through u gives v a shorter
            # distance than what we currently have, record the improvement
            # and push the new candidate distance to the heap
            for v in g.neighbors(u):
                edge_id = g.get_eid(u, v)
                w = g.es[edge_id][weight]

                new_dist = d + w
                if new_dist < dist.get(v, float("inf")):
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(heap, (new_dist, v))

        # decide which vertices we want results for
        if target_ig_ids is not None:
            result_ig_ids = target_ig_ids
        else:
            result_ig_ids = set(dist) - {source_ig}

        distances = {}
        paths = {}
        for target_ig in result_ig_ids:

            # a target may be unreachable 
            if target_ig not in dist:
                continue

            # reconstruct the path by walking predecessors backwards from
            # target to the source
            path_ig = []
            node = target_ig
            while node is not None:
                path_ig.append(node)
                node = prev[node]
            path_ig.reverse()

            target_id = g.vs[target_ig]["node_id"]
            distances[target_id] = dist[target_ig]
            paths[target_id] = [g.vs[i]["node_id"] for i in path_ig]

        # output is two dicts, one with distances {100: 154.4}
        # and one with paths {100: [150, 189, 200]}
        return distances, paths


    def multi_source_dijkstra(self, sources, targets=None, weight="length"):
        """
        Compute shortest paths from several source nodes to a subset of
        target nodes. This is just a wrapper around `dijkstra`

        Parameters
        ----------
        sources : iterable
            Source nodes, using the original NetworkX node identifiers
        targets : optional (Default: None)
            Destination nodes, passed through to `dijkstra` for every
            source. If None, distances are computed to every node
            reachable from each source.
        weight : str, optional
            Edge attribute used as cost. Default 'length'.

        Returns
        -------
        distances : dict
            Nested dict: distances[source][target] = shortest distance.
        paths : dict
            Nested dict: paths[source][target] = ordered list of node
            IDs from that source to that target.
        """

        # initialize dicts
        distances = {}
        paths = {}

        for source in sources:
            source_distances, source_paths = self.dijkstra(
                source, targets=targets, weight=weight
            )
            distances[source] = source_distances
            paths[source] = source_paths

        # outer key will be source and inner key will be target
        # distances {100:{200:1385.4, 300:346.3}, 250:{...}}

        return distances, paths


    def boundary_distance_matrix(self, weight="length"):
        """
        Compute shortest-path distances between boundary nodes that
        belong to different regions.

        For each region, its boundary nodes are used as sources and the
        boundary nodes of every other region are used as targets

        Parameters
        ----------
        weight : str, optional
            Default 'length'.

        Returns
        -------
        distances : dict
            Nested dict: distances[source][target] = shortest distance,
            for source/target boundary nodes belonging to different
            regions.
        paths : dict
            Nested dict: paths[source][target] = ordered list of node
            IDs, for the same source/target pairs.
        """
        boundary_nodes_by_region = self.boundary_nodes

        distances = {}
        paths = {}

        # process one region at a time so we can exclude that region's
        # own boundary nodes from its targets
        for region, nodes in boundary_nodes_by_region.items():

            # boundary nodes of every other region
            other_targets = [
                node_id
                for other_region, other_nodes in boundary_nodes_by_region.items()
                if other_region != region
                for node_id in other_nodes
            ]

            region_distances, region_paths = self.multi_source_dijkstra(
                sources=nodes,
                targets=other_targets,
                weight=weight
            )

            distances.update(region_distances)
            paths.update(region_paths)

        return distances, paths


    # ------------------------------------------------------
    # PROTECTED METHODS
    # ------------------------------------------------------
    def _graph(self, kind = "nx"):
        """Return a copy of the NetworkX or igraph representation."""
        if kind == "nx":
            return self.__nx_graph.copy()
        elif kind == "ig":
            if self.__ig_graph is None:
                self.networkx_to_igraph()
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
