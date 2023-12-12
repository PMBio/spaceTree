import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
import scvi
from scvi.model.utils import mde
import matplotlib as mpl
from .plotting import plot_xenium
import os
from sklearn.neighbors import NearestNeighbors

def flatten(l):
    return [item for sublist in l for item in sublist]
def rna_seq_prep(adata_seq):
    adata_seq = adata_seq.raw.to_adata()
    adata_seq.X.data = np.exp(adata_seq.X.data) - 1
    adata_seq.X = adata_seq.X.multiply(adata_seq.obs.nCount_RNA.to_numpy()[:, np.newaxis]).tocsr()
    adata_seq.X = np.round(adata_seq.X / 1e4)
    adata_seq.var_names_make_unique()
    return(adata_seq)

def run_scvi(adata, outdir = "data/interim/res_scvi.csv", highly_variable_genes = False, plot_extra = []):
    """
    Runs scVI on the input AnnData object and returns a DataFrame with the cell embeddings and source labels.

    Parameters:
    adata (anndata.AnnData): Input AnnData object with gene expression data.
    outdir (str): Output directory to save the resulting DataFrame.

    Returns:
    pandas.DataFrame: DataFrame with cell embeddings and source labels.
    """
    sc.pp.filter_cells(adata, min_genes=3)

    # Copy the gene expression data to a new layer called "counts"
    adata.layers["counts"] = adata.X.copy()

    # Normalize the data to have a total count of 1e4
    sc.pp.normalize_total(adata, target_sum=1e4)

    # Log-transform the data
    sc.pp.log1p(adata)

    # Keep the full dimension of the data safe
    adata.raw = adata
    if highly_variable_genes:
        # Identify highly variable genes
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=2000,
            layer="counts",
            batch_key="source",
            subset=True
        )

    # Set up the AnnData object for scVI
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="source")

    # Train the scVI model
    vae = scvi.model.SCVI(adata, n_layers=2, n_latent=30)
    vae.train()

    # Get the latent representation of the cells
    adata.obsm["X_scVI"] = vae.get_latent_representation()

    # Apply MDE to the latent representation
    adata.obsm["X_mde"] = mde(adata.obsm["X_scVI"])

    # Visualize the cell embeddings
    sc.pl.embedding(
        adata,
        basis="X_mde",
        color=["source"] + plot_extra,
        frameon=False,
        ncols=1,
    )

    # Create a DataFrame with the cell embeddings and source labels
    cell_source = pd.DataFrame(adata.obsm["X_mde"], index = adata.obs.index)
    cell_source["source"] = adata.obs.source

    # Save the DataFrame to a CSV file
    cell_source.to_csv(outdir)

    # Return the DataFrame
    return cell_source

def record_edges(emb_rna, emb_spatial, n_neigb, edge_type):
    # Prepare the data and fit the nearest neighbors model on emb_spatial
    if edge_type == "sc2xen" or edge_type == "sc2vis":
        nbrs = NearestNeighbors(n_neighbors=n_neigb, algorithm='auto').fit(emb_spatial.values)
        # Retrieve distances and indices of the n_neigb nearest neighbors in emb_spatial for each point in emb_rna
        distances, indices = nbrs.kneighbors(emb_rna.values)
    elif edge_type == "sc2sc":
        data = emb_rna.values
        nbrs = NearestNeighbors(n_neighbors=n_neigb, algorithm='auto').fit(data)    
        distances, indices = nbrs.kneighbors(data)
    else:
        raise ValueError("edge_type must be either 'sc2xen' or 'sc2sc' or 'sc2vis'")
    
    # Extract node names from the DataFrames' indices
    rna_names = emb_rna.index.values
    spatial_names = emb_spatial.index.values
    
    # Create edges using the distances and indices from kneighbors
    node1 = spatial_names[indices].reshape(-1, 1)
    node2 = np.repeat(rna_names, n_neigb).reshape(-1, 1)
    weights = distances.reshape(-1, 1)

    edges = np.hstack((node1, node2, weights))
    edges_df = pd.DataFrame(edges, columns=["node1", "node2", "weight"])
    edges_df["type"] = edge_type

    return edges_df
def show_weights_distribution(edges, spatial, spatial_type = "visium", library_id = None):
    top_match = edges[["node2","weight"]].groupby("node2").min()
    spatial.obs = spatial.obs.join(top_match)
    if spatial_type == "visium":
        spatial.obs["weight"] = spatial.obs["weight"].astype(float)
        if library_id is not None:
            with mpl.rc_context({'axes.facecolor':  'black',
                            'figure.figsize': [10, 10]}):
                sc.pl.spatial(spatial, cmap='magma',
                            # show first 8 cell types
                            color="weight",
                            img_key='lowres', alpha_img = 0.5,
                            library_id = library_id,
                            )
        else:
            with mpl.rc_context({'axes.facecolor':  'black',
                        'figure.figsize': [10, 10]}):
                sc.pl.spatial(spatial, cmap='magma',
                        # show first 8 cell types
                        color="weight",
                        img_key='lowres', alpha_img = 0.5
                        )

    elif spatial_type == "xenium":
        plot_xenium(spatial.obs.x_centroid, spatial.obs.y_centroid, 
            spatial.obs["weight"], palette = "magma")




def normalize_edge_weights(edges):
    edges.weight = (edges.weight - edges.weight.min())/(edges.weight.max() - edges.weight.min())
    edges.weight  = 1 - edges.weight
    return edges

def convert_array_row_col_to_int(visium):
    visium.obs["array_row"] = visium.obs["array_row"].astype(int)
    visium.obs["array_col"] = visium.obs["array_col"].astype(int)
    coor_int = [[int(x[0]),int(x[1])] for x in visium.obsm["spatial"]]
    visium.obsm["spatial"] = np.array(coor_int)
    return visium

def create_edges_for_visium_nodes(visium):
    vis_nodes = visium.obs.index
    edges_vis = []
    for node in tqdm(vis_nodes):
        x= visium.obs.loc[node].array_row
        y= visium.obs.loc[node].array_col
        tmp = visium.obs[(visium.obs.array_row >=x-2)&(visium.obs.array_row <=x+2)&(visium.obs.array_col <=y+2)&(visium.obs.array_col >=y-2)].copy()
        tmp.loc[:,"degree"] = 2
        tmp.loc[(tmp.array_row >=x-1)&(tmp.array_row <=x+1)&(tmp.array_col <=y+1)&(tmp.array_col >=y-1),"degree"] = 1
        nodes1 = tmp[tmp.degree == 1].index
        for n in nodes1:
            if n != node:
                edges_vis.append([node,n, 1])
        nodes2 = tmp[tmp.degree == 2].index
        for n in nodes2:
            edges_vis.append([node,n, 0.5])
    edges_vis = pd.DataFrame(edges_vis, columns=["node1", "node2", "weight"])
    edges_vis["type"] = "vis2grid"
    return edges_vis
def create_edges_for_xenium_nodes(xenium, d=20):
    # Extract centroids
    centroids = xenium.obs[["x_centroid", "y_centroid"]].values
    node_list = xenium.obs.index.values
    
    # NearestNeighbors search
    nbrs = NearestNeighbors(n_neighbors=centroids.shape[0], algorithm='auto').fit(centroids)
    distances, indices = nbrs.kneighbors(centroids)
    
    # Find the min and max distances for normalization
    min_distances = distances[:, 0].reshape(-1, 1)
    max_distances = distances[:, -1].reshape(-1, 1)
    
    # Extract the top d distances and indices
    top_d_distances = distances[:, :d+1]
    top_d_indices = indices[:, :d+1]

    # Normalize the top d distances
    norm_values = 1 - (top_d_distances - min_distances) / (max_distances - min_distances)

    # Create edges
    node1 = np.repeat(node_list, d+1).reshape(-1, 1)
    node2 = node_list[top_d_indices].reshape(-1, 1)
    weights = norm_values.reshape(-1, 1)
    distances_array = np.full(weights.shape, d)
    
    edges_xen = np.hstack((node1, node2, weights, distances_array))
    edges_xen_df = pd.DataFrame(edges_xen, columns=["node1", "node2", "weight", "distance"])
    edges_xen_df["type"] = "xen2grid"

    return edges_xen_df

def create_edges_for_xenium_nodes_fast(xenium, d=20):
    # Extract centroids
    centroids = xenium.obs[["x_centroid", "y_centroid"]].values
    node_list = xenium.obs.index.values
    
    # NearestNeighbors search
    nbrs = NearestNeighbors(n_neighbors=d+1, algorithm='auto').fit(centroids)  # +1 for the self-loop
    distances, indices = nbrs.kneighbors(centroids)
    
    # Normalize distances
    # No need to drop the first column for self-loops
    min_distances = distances[:, 0].reshape(-1, 1)  # This is always 0 for self-loops
    max_distances = distances[:, -1].reshape(-1, 1)
    norm_values = 1 - (distances - min_distances) / (max_distances - min_distances)

    # Create edges
    node1 = np.repeat(node_list, d+1).reshape(-1, 1)  # +1 for the self-loop
    node2 = node_list[indices].reshape(-1, 1)
    weights = norm_values.reshape(-1, 1)
    distances_array = np.full(weights.shape, d+1)  # +1 for the self-loop
    
    edges_xen = np.hstack((node1, node2, weights, distances_array))
    edges_xen_df = pd.DataFrame(edges_xen, columns=["node1", "node2", "weight", "distance"])
    edges_xen_df["type"] = "xen2grid"

    return edges_xen_df
def create_edges_for_xenium_nodes_global(xenium, percentile=1, sample_size=1000):
    # Extract centroids
    centroids = xenium.obs[["x_centroid", "y_centroid"]].values
    node_list = xenium.obs.index.values

    # Initialize NearestNeighbors
    nbrs = NearestNeighbors().fit(centroids)

    # Use a sample of the data to estimate the 1% distance threshold
    if centroids.shape[0] > sample_size:
        sample_indices = np.random.choice(centroids.shape[0], sample_size, replace=False)
        sample_centroids = centroids[sample_indices]
    else:
        sample_centroids = centroids

    # Find the distances in the sample
    sample_distances, _ = nbrs.kneighbors(sample_centroids)

    # Exclude the self-distances (which are always 0 and at column index 0)
    sample_distances_without_self = sample_distances[:, 1:]

    # Flatten the array to get a single list of distances and then find the 1% threshold
    threshold_distance = np.percentile(sample_distances_without_self.flatten(), percentile)

    # Find all points within the threshold distance using radius_neighbors
    # This will be more memory efficient than kneighbors for the full distance matrix
    radius_distances, radius_indices = nbrs.radius_neighbors(centroids, radius=threshold_distance, sort_results=True)

    # Create the edges, excluding self-loops
    edges = []
    for i, (indices, dists) in enumerate(zip(radius_indices, radius_distances)):
        for index, distance in zip(indices, dists):
            # Check if it's a self-loop and set the normalized distance accordingly
            normalized_distance = 1.0 if i == index else 1 - (distance / threshold_distance)
            edges.append((node_list[i], node_list[index], normalized_distance))


    # Create DataFrame from edges
    edges_xen_df = pd.DataFrame(edges, columns=["node1", "node2", "weight"])
    edges_xen_df["distance"] = threshold_distance
    edges_xen_df["type"] = "xen2grid"

    return edges_xen_df

def save_edges_and_embeddings(edges, emb_spatial, emb_rna, outdir = "data/interim/",suffix = ""):
    edges.to_csv(f"{outdir}edges_{suffix}.csv")
    emb_spatial.to_csv(f"{outdir}embedding_spatial_{suffix}.csv")
    emb_rna.to_csv(f"{outdir}embedding_rna_{suffix}.csv")

