# %%
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm
from scipy.spatial import distance
import scvi
from scvi.model.utils import mde
import os
import spaceTree.preprocessing as pp
from spaceTree.dataset import *
os.chdir("/home/o313a/clonal_GNN/")
import re
#%% Paths
path_xen= "data/raw/xenium/outs/cell_feature_matrix.h5"
path_scvi = "data/interim/res_scvi_xenium.csv" # leave emty string of needed to be recomputed
#%% Reding the data
xenium = sc.read_10x_h5(filename=path_xen)
df = pd.read_csv('data/raw/xenium/outs/cells.csv')
df.set_index(xenium.obs_names, inplace=True)
xenium.obs = df.copy()
xenium.obsm["spatial"] = xenium.obs[["x_centroid", "y_centroid"]].copy().to_numpy()
xenium.var_names_make_unique()

#%%
if len(path_scvi) == 0:
    path_sc = "data/interim/scrna.h5ad"
    adata_seq = sc.read_h5ad(path_sc)
    adata_seq.obs.drop(columns = ["celltype_major","celltype_minor"], inplace = True)
    adata_seq.obs.index = adata_seq.obs.index.str.replace(".", "-")
    adata_seq = pp.rna_seq_prep(adata_seq)
    xenium.obs["source"] = "spatial"
    adata_seq.obs["source"] = "scRNA"
    adata = xenium.concatenate(adata_seq)
    cell_source = pp.run_scvi(adata, "data/interim/res_scvi_xenium.csv")
    cell_source.index = [re.sub(r'-(\d+).*', r'-\1', s) for s in cell_source.index]
    emb_spatial =cell_source[cell_source.source == "spatial"][[0,1]]
    emb_spatial.index = [x.split("-")[0] for x in emb_spatial.index]
    emb_rna = cell_source[cell_source.source == "scRNA"][[0,1]]#%%

else:
    cell_source = pd.read_csv(path_scvi, index_col = 0)
    cell_source.index = [re.sub(r'-(\d+).*', r'-\1', s) for s in cell_source.index]
    emb_spatial =cell_source[cell_source.source == "spatial"][['0', '1']]
    emb_spatial.index = [x.split("-")[0] for x in emb_spatial.index]

    emb_rna = cell_source[cell_source.source == "scRNA"][['0', '1']]#%%
#%%
print("1. Recording edges between RNA and spatial embeddings...")
edges_sc2xen = pp.record_edges(emb_spatial, emb_rna,10, "sc2xen")
pp.show_weights_distribution(edges_sc2xen,xenium, 'xenium')
edges_sc2xen = pp.normalize_edge_weights(edges_sc2xen)
#%%
print("2. Recording edges between RNA embeddings...")
edges_sc2sc = pp.record_edges(emb_rna,emb_rna, 10, "sc2sc")
edges_sc2sc = pp.normalize_edge_weights(edges_sc2sc)

#%%
print("3. Creating edges for Xenium nodes...")
edges_xen2grid_g1 = pp.create_edges_for_xenium_nodes_global(xenium,1)
edges_xen2grid_g10 = pp.create_edges_for_xenium_nodes_global(xenium,10)

#%%
edges_files = [edges_xen2grid_g1,edges_xen2grid_g10]
edge_types = ["1g","10g"]
print("4. Saving edges and embeddings...")
for edge_file, edge_type in zip(edges_files, edge_types):
    edges = pd.concat([edges_sc2xen, edges_sc2sc, edge_file])
    edges.node1 = edges.node1.astype(str)
    edges.node2 = edges.node2.astype(str)
    pp.save_edges_and_embeddings(edges, emb_spatial, emb_rna, suffix = edge_type)



    print("5. Loading cell type annotations...")

    overcl = pd.read_csv("data/interim/clones_over.csv")
    overcl.columns = ["node1","clone"]
    overcl.node1 = [x[:-2] for x in overcl.node1]
    cell_type = pd.read_excel("data/raw/Requested_Cell_Barcode_Type_Matrices.xlsx", sheet_name="scFFPE-Seq")
    cell_type.columns = ["node1","cell_type"]
    overcl = cell_type.merge(overcl, on = "node1", how = "left")
    print("6. Constructing dataset...")

    edges, overcl = preprocess_data(edges, overcl,"sc2xen","xen2grid")
    embedding_paths = {"spatial":f"data/interim/embedding_spatial_{edge_type}.csv",
                    "rna":f"data/interim/embedding_rna_{edge_type}.csv"}
    emb_vis_nodes, emb_rna_nodes, edges, node_encoder = read_and_merge_embeddings(embedding_paths, edges)
    edges.weight = edges.weight.astype(float)
    data, encoding_dict = create_data_object(edges, emb_vis_nodes, emb_rna_nodes, node_encoder)
    torch.save(data, f"data/processed/data_xen_{edge_type}.pt")
    with open(f'data/processed/full_encoding_xen_{edge_type}.pkl', 'wb') as fp:
        pickle.dump(encoding_dict, fp)
    print("7. Sanity checks...")
    data.edge_attr = data.edge_attr.reshape((-1,1))
    hold_out_indices = np.where(data.y_clone == -1)[0]
    hold_out = torch.tensor(hold_out_indices, dtype=torch.long)
    hold_in_indices = np.arange(data.x.shape[0])
    hold_in = [index for index in hold_in_indices if index not in hold_out]
    del data.edge_type

    from torch_geometric.loader import NeighborLoader

    loader = NeighborLoader(
        data,
        num_neighbors=[10] * 3,
        batch_size=128,input_nodes = hold_in
    )



    for dat in loader:
        assert -1 not in dat.y_clone.unique()

    # %%
