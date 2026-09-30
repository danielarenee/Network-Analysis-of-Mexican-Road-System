import matplotlib.pyplot as plt
import igraph as ig
import pickle

from pathlib import Path
from geopandas import GeoDataFrame, read_file
from shapely.geometry import LineString

REGION_PATH = Path(__file__).resolve().parent / "data" / "raw" / "LocalitiesGrouped_2020_data.gpkg"
        
class City_Network:
    
    # ------------------------------------------------------
    # ATTRIBUTES
    # ------------------------------------------------------
    @property
    def node_gdf(self):
        return self.__node_gdf.copy()
    
    @property
    def edge_gdf(self):
        return self.__edge_gdf.copy()
    
    @property
    def n(self):
        """Return the number of nodes."""
        return self.__ig_graph.vcount()
    
    @property
    def m(self):
        """Return the number of edges."""
        return self.__ig_graph.ecount()
    
    @property
    def vertex_connectivity(self):
        return self.__ig_graph.vertex_connectivity()
    
    @property
    def edge_connectivity(self):
        return self.__ig_graph.edge_connectivity()
    
    @property
    def mean_degree(self):
        degrees = self.__ig_graph.degree()
        return sum(degrees) / len(degrees)
    
    @property
    def density(self):
        return self.__ig_graph.density()

    # ------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------
    def __init__(
            self, 
            g: ig.Graph()
        ):
        self.__ig_graph = g
        self.__crs = g["crs"]
        
        regions_gdf = read_file(REGION_PATH)

        self.__nodes_gdf = self.__create_nodes_gdf(regions_gdf)
        self.__edges_gdf = self.__create_edges_gdf()
      
    
    # ------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------   
    def plot(
            self,
            figsize = (10, 10),
            geometry = True,
            node_size = 5,
            edge_width = 0.5
            ):
        fig, ax = plt.subplots(figsize = figsize)

        if geometry:
            self.__edges_gdf.plot(ax=ax, linewidth=edge_width, color="gray", alpha=0.8)
            self.__nodes_gdf.plot(ax=ax, markersize=node_size, color="red")
        else:
            #layout = self.__ig_graph.layout_fruchterman_reingold(niter=2000)
            layout = self.__ig_graph.layout_drl()
            ig.plot(
                self.__ig_graph,
                target=ax,
                layout=layout,
                vertex_size=node_size,
                bbox=(1200, 900),
                margin=50,
                edge_width=edge_width,
                edge_color="#80808066",
            )
        ax.set_axis_off()
        plt.show()
        
    def save(self, path, file_name, gdf = False):   
        """Save the city network and the associated GeoDataFrames."""
        
        graph_file = Path(path) / f"{file_name}.pkl"
        nodes_file = Path(path) / f"{file_name}_nodes.gpkg"
        edges_file = Path(path) / f"{file_name}_edges.gpkg"
        with graph_file.open("wb") as handle:
           pickle.dump(
               self.__ig_graph,
               handle,
               protocol=pickle.HIGHEST_PROTOCOL
           )
        
        if gdf:
            self.__nodes_gdf.to_file(nodes_file, driver="GPKG")
            self.__edges_gdf.to_file(edges_file, driver="GPKG")
    
    # ------------------------------------------------------
    # PRIVATE METHODS
    # ------------------------------------------------------   
    def __create_nodes_gdf(
            self,
            regions_gdf: GeoDataFrame
        ) -> GeoDataFrame:
        
        regions = self.__ig_graph.vs["region"]
        selected = regions_gdf.loc[regions_gdf["id_convex"].isin(regions)].copy()
        selected.to_crs(self.__crs)

        centroids = selected.geometry.centroid
        centroid_by_region = dict(zip(selected["id_convex"], centroids))
        nodes_df = (
            self.__ig_graph.get_vertex_dataframe()
            .rename_axis("vertex_id")
            .reset_index()
        )
        centroids = [
            centroid_by_region[region] 
            for region in nodes_df["region"]
        ]
        
        nodes_gdf = GeoDataFrame(
            nodes_df,
            geometry=centroids,
            crs=self.__crs,
        )
        
        return nodes_gdf
    
    def __create_edges_gdf(self) -> GeoDataFrame:
        edges_df = (
            self.__ig_graph.get_edge_dataframe()
            .rename_axis("edge_id")
            .reset_index()
        )
        coords = [(point.x, point.y) for point in self.__nodes_gdf.geometry]
        lines = [
            LineString([coords[source], coords[target]])
            for source, target in zip(edges_df["source"], edges_df["target"])
        ]
        edges_gdf = GeoDataFrame(
            edges_df,
            geometry=lines,
            crs=self.__crs,
        )
        
        return edges_gdf
